"""Light recommender model, without the heavy Turi Create dependency."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from board_game_recommender.abc import BaseGamesRecommender
from board_game_recommender.baseline import dataframe_from_scores

LOGGER = logging.getLogger(__name__)

# float32 precision exceeds the model's training noise, halving memory for free.
_FLOAT32_FIELDS = (
    "users_linear_terms",
    "users_factors",
    "items_linear_terms",
    "items_factors",
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Set as AbstractSet
    from typing import Any, Self

    import polars as pl


class SortedLabelIndex:
    """Maps labels to their original position via a sorted permutation.

    An `np.searchsorted()` lookup over a permutation array avoids the
    per-item Python object overhead of a `dict` built from the same labels.
    """

    def __init__(self, labels: np.ndarray) -> None:
        # `sorter=` looks up positions without materializing a sorted copy of
        # `labels`, which -- unlike `self._order` -- is exactly as big as the
        # duplicate array this class exists to avoid.
        self._labels = labels
        self._order = np.argsort(labels)

    def __getitem__(self, queries: Iterable[Any]) -> np.ndarray:
        # a bare str/int would otherwise iterate into garbage instead of erroring
        if isinstance(queries, (str, int, np.str_, np.integer)):
            msg = f"expected a batch of labels, got a single one: {queries!r}"
            raise TypeError(msg)
        queries_array = np.asarray(list(queries))
        positions = np.searchsorted(
            self._labels,
            queries_array,
            sorter=self._order,
        ).clip(max=len(self._labels) - 1)
        candidate_indexes = self._order[positions]
        found = self._labels[candidate_indexes] == queries_array
        return cast("np.ndarray", np.where(found, candidate_indexes, -1))


@dataclass(frozen=True)
class CollaborativeFilteringData:
    """Labels, vectors and matrices for linear collaborative filtering models."""

    intercept: float
    users_labels: np.ndarray  # (num_users,)
    users_linear_terms: np.ndarray  # (num_users,)
    users_factors: np.ndarray  # (num_users, num_factors)
    items_labels: np.ndarray  # (num_items,)
    items_linear_terms: np.ndarray  # (num_items,)
    items_factors: np.ndarray  # (num_factors, num_items)

    def to_npz(self, file_path: Path | str) -> None:
        """Save data into an .npz file."""

        file_path = Path(file_path).resolve()
        LOGGER.info("Saving data as .npz to <%s>", file_path)
        with file_path.open(mode="wb") as file:
            np.savez(file=file, **asdict(self))
        LOGGER.info("Done saving <%s>", file_path)

    @classmethod
    def from_npz(cls, file_path: Path | str) -> Self:
        """Load data from an .npz file."""

        file_path = Path(file_path).resolve()
        LOGGER.info("Loading data as .npz from <%s>", file_path)
        with file_path.open(mode="rb") as file:
            files = np.load(file=file)
            files_dict = {
                key: (
                    float(files[key])
                    if key == "intercept"
                    else files[key].astype(np.float32)
                    if key in _FLOAT32_FIELDS
                    else files[key]
                )
                for key in files.files
            }
            assert all(
                isinstance(key, str) and isinstance(value, (np.ndarray, float))
                for key, value in files_dict.items()
            ), "All keys must be strings and all values must be numpy arrays or floats"
            return cls(**files_dict)  # type: ignore[arg-type]


class LightGamesRecommender(BaseGamesRecommender[int, str]):
    """Light recommender without Turi Create dependency."""

    _known_games: frozenset[int] | None = None
    _known_users: frozenset[str] | None = None

    def __init__(self, data: CollaborativeFilteringData) -> None:
        assert data.users_factors.shape[-1] == data.items_factors.shape[0]
        num_factors = data.items_factors.shape[0]
        # TODO check other dimensions as well (num_users and num_items)

        self.intercept: float = data.intercept

        self.users_labels: np.ndarray = data.users_labels
        self.users_indexes = SortedLabelIndex(data.users_labels)
        self.users_linear_terms = np.concatenate(
            (data.users_linear_terms, np.zeros(1, dtype=data.users_linear_terms.dtype)),
        )
        self.users_factors = np.concatenate(
            (
                data.users_factors,
                np.zeros((1, num_factors), dtype=data.users_factors.dtype),
            ),
            axis=0,
        )

        self.items_labels: np.ndarray = data.items_labels
        self.items_indexes = SortedLabelIndex(data.items_labels)
        self.items_linear_terms = np.concatenate(
            (data.items_linear_terms, np.zeros(1, dtype=data.items_linear_terms.dtype)),
        )
        self.items_factors = np.concatenate(
            (
                data.items_factors,
                np.zeros((num_factors, 1), dtype=data.items_factors.dtype),
            ),
            axis=1,
        )

        LOGGER.info(
            "Loaded light recommender with %d users and %d items",
            len(self.users_labels),
            len(self.items_labels),
        )

    def to_npz(self, file_path: Path | str) -> None:
        """Save data into an .npz file."""
        # Undo __init__'s padding instead of keeping a second copy just for this.
        CollaborativeFilteringData(
            intercept=self.intercept,
            users_labels=self.users_labels,
            users_linear_terms=self.users_linear_terms[:-1],
            users_factors=self.users_factors[:-1, :],
            items_labels=self.items_labels,
            items_linear_terms=self.items_linear_terms[:-1],
            items_factors=self.items_factors[:, :-1],
        ).to_npz(file_path)

    @classmethod
    def from_npz(
        cls,
        file_path: Path | str,
    ) -> Self:
        """Load data from an .npz file."""
        data = CollaborativeFilteringData.from_npz(file_path)
        return cls(data)

    @property
    def known_games(self) -> AbstractSet[int]:
        if self._known_games is not None:
            return self._known_games
        self._known_games = frozenset(self.items_labels.tolist())
        return self._known_games

    @property
    def rated_games(self) -> AbstractSet[int]:
        return self.known_games

    @property
    def num_games(self) -> int:
        return len(self.items_labels)

    @property
    def known_users(self) -> AbstractSet[str]:
        if self._known_users is not None:
            return self._known_users
        self._known_users = frozenset(self.users_labels.tolist())
        return self._known_users

    @property
    def num_users(self) -> int:
        return len(self.users_labels)

    def _recommendation_scores(
        self,
        *,
        users: list[str] | None = None,
        games: list[int] | None = None,
        avg_users: bool = False,
    ) -> np.ndarray:
        """Calculate recommendations scores for certain users and games."""

        if users:
            user_ids = self.users_indexes[users]
            users_factors = self.users_factors[user_ids]
            users_linear_terms = self.users_linear_terms[user_ids].reshape(-1, 1)
        else:
            users_factors = self.users_factors[:-1, :]
            users_linear_terms = self.users_linear_terms[:-1].reshape(-1, 1)

        if avg_users:
            users_factors = users_factors.mean(axis=0).reshape(1, -1)
            users_linear_terms = users_linear_terms.mean(axis=0).reshape(1, 1)

        if games:
            game_ids = self.items_indexes[games]
            items_factors = self.items_factors[:, game_ids]
            items_linear_terms = self.items_linear_terms[game_ids].reshape(1, -1)
        else:
            items_factors = self.items_factors[:, :-1]
            items_linear_terms = self.items_linear_terms[:-1].reshape(1, -1)

        return cast(
            "np.ndarray",
            users_factors @ items_factors  # (num_users, num_items)
            + users_linear_terms  # (num_users, 1)
            + items_linear_terms  # (1, num_items)
            + self.intercept,  # (1,)
        )

    def _game_scores(
        self,
        games: list[int] | None = None,
    ) -> np.ndarray:
        """Calculate average game scores from bias terms."""

        if games:
            game_ids = self.items_indexes[games]
            items_linear_terms = self.items_linear_terms[game_ids]
        else:
            items_linear_terms = self.items_linear_terms[:-1]

        return cast("np.ndarray", items_linear_terms + self.intercept)

    def recommend(
        self,
        users: Iterable[str],
        **kwargs: Any,  # noqa: ARG002
    ) -> pl.DataFrame:
        """Calculate recommendations for certain users."""

        users = list(users)
        scores = self._recommendation_scores(users=users)
        return dataframe_from_scores(
            users=users,
            games=self.items_labels,
            scores=scores,
        )

    def recommend_as_numpy(
        self,
        users: Iterable[str],
        games: Iterable[int],
    ) -> np.ndarray:
        """Calculate recommendations for certain users and games as a numpy array."""

        users = list(users)
        games = list(games)

        return self._recommendation_scores(users=users, games=games)

    def recommend_group(
        self,
        users: Iterable[str],
        **kwargs: Any,  # noqa: ARG002
    ) -> pl.DataFrame:
        """Calculate recommendations for a group of users."""

        users = list(users)
        scores = (
            self._recommendation_scores(users=users, avg_users=True)
            if users
            else self._game_scores()
        )
        return dataframe_from_scores(
            users=["_all"],
            games=self.items_labels,
            scores=scores,
        )

    def recommend_group_as_numpy(
        self,
        users: Iterable[str],
        games: Iterable[int],
    ) -> np.ndarray:
        """
        Calculate recommendations for a group of users and games as a numpy array.
        """

        users = list(users)
        games = list(games)
        return (
            self._recommendation_scores(users=users, games=games, avg_users=True)
            if users
            else self._game_scores(games).reshape(1, -1)
        )

    def _similarity_scores(self, games: list[int]) -> np.ndarray:
        """
        Cosine similarity between the given games and every known game.

        The result has shape (len(games), num_games). Games unknown to the model
        have no latent factors, so they score 0 against everything.
        """

        game_ids = self.items_indexes[games]
        game_factors = self.items_factors[:, game_ids]
        return cosine_similarity(game_factors, self.items_factors[:, :-1])

    def recommend_similar(
        self,
        games: Iterable[int],
        **kwargs: Any,  # noqa: ARG002
    ) -> pl.DataFrame:
        """
        Recommend games similar to the given games based on cosine similarity
        of latent factors, averaged over all the given games.
        """

        games = list(games)
        scores = (
            self._similarity_scores(games).mean(axis=0).reshape(1, -1)
            if games
            else np.zeros((1, self.num_games))
        )
        return dataframe_from_scores(
            users=["_all"],
            games=self.items_labels,
            scores=scores,
        )

    def similar_games(
        self,
        games: Iterable[int],
        **kwargs: Any,  # noqa: ARG002
    ) -> pl.DataFrame:
        """
        Find games similar to the given games based on
        cosine similarity of latent factors.
        """

        games = list(games)
        return dataframe_from_scores(
            users=games,  # type: ignore[arg-type]
            games=self.items_labels,
            scores=self._similarity_scores(games),
        )


def cosine_similarity(matrix_1: np.ndarray, matrix_2: np.ndarray) -> np.ndarray:
    """
    Calculates the cosine similarity between two matrices.

    The input matrices need to be of shape (m,n) and (m,l);
    the result shape will be (n,l). Columns with zero norm score 0 throughout.
    """

    dot_product = matrix_1.T @ matrix_2  # (n,l)
    matrix_1_norm = np.linalg.norm(matrix_1, axis=0)  # (n,)
    matrix_2_norm = np.linalg.norm(matrix_2, axis=0)  # (l,)
    outer_prod_norm = np.outer(matrix_1_norm, matrix_2_norm)  # (n,l)

    # Zero vectors are orthogonal to everything by convention, rather than NaN:
    # a single unknown game would otherwise poison an entire recommendation.
    return cast(
        "np.ndarray",
        np.divide(
            dot_product,
            outer_prod_norm,
            out=np.zeros_like(dot_product),
            where=outer_prod_norm != 0,
        ),
    )  # (n,l)
