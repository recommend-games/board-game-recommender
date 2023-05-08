"""Light recommender model, without the heavy Turi Create dependency."""

import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, FrozenSet, Iterable, List, Optional, Type, Union

import numpy as np
import pandas as pd

from board_game_recommender.base import BaseGamesRecommender
from board_game_recommender.baseline import dataframe_from_scores

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    import turicreate

RecommenderModel = Union[
    "turicreate.recommender.factorization_recommender.FactorizationRecommender",
    "turicreate.recommender.ranking_factorization_recommender.RankingFactorizationRecommender",
]


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

    def to_npz(self: "CollaborativeFilteringData", file_path: Union[Path, str]) -> None:
        """Save data into an .npz file."""

        file_path = Path(file_path).resolve()
        LOGGER.info("Saving data as .npz to <%s>", file_path)
        with file_path.open(mode="wb") as file:
            np.savez(file=file, **asdict(self))
        LOGGER.info("Done saving <%s>", file_path)

    @classmethod
    def from_npz(
        cls: Type["CollaborativeFilteringData"],
        file_path: Union[Path, str],
    ) -> "CollaborativeFilteringData":
        """Load data from an .npz file."""

        file_path = Path(file_path).resolve()
        LOGGER.info("Loading data as .npz from <%s>", file_path)
        with file_path.open(mode="rb") as file:
            files = np.load(file=file)
            files_dict = {
                key: float(files[key]) if key == "intercept" else files[key]
                for key in files.files
            }
            return cls(**files_dict)


class LightGamesRecommender(BaseGamesRecommender):
    """Light recommender without Turi Create dependency."""

    _known_games: Optional[FrozenSet[int]] = None
    _known_users: Optional[FrozenSet[str]] = None

    def __init__(
        self: "LightGamesRecommender",
        data: CollaborativeFilteringData,
    ) -> None:
        self.data = data

        assert data.users_factors.shape[-1] == data.items_factors.shape[0]
        num_factors = data.items_factors.shape[0]
        # TODO check other dimensions as well (num_users and num_items)

        self.intercept: float = data.intercept

        num_users = len(data.users_labels)
        self.users_labels: List[str] = list(data.users_labels)
        self.users_indexes = defaultdict(
            lambda: -1,
            zip(data.users_labels, range(num_users)),
        )
        self.users_linear_terms = np.concatenate((data.users_linear_terms, np.zeros(1)))
        self.users_factors = np.concatenate(
            (data.users_factors, np.zeros((1, num_factors))),
            axis=0,
        )

        num_items = len(data.items_labels)
        self.items_labels: List[int] = list(data.items_labels)
        self.items_indexes = defaultdict(
            lambda: -1,
            zip(data.items_labels, range(num_items)),
        )
        self.items_linear_terms = np.concatenate((data.items_linear_terms, np.zeros(1)))
        self.items_factors = np.concatenate(
            (data.items_factors, np.zeros((num_factors, 1))),
            axis=1,
        )

        LOGGER.info(
            "Loaded light recommender with %d users and %d items",
            len(self.users_labels),
            len(self.items_labels),
        )

    @classmethod
    def from_turi_create(
        cls: Type["LightGamesRecommender"],
        model: RecommenderModel,
        *,
        user_id: str = "bgg_user_name",
        item_id: str = "bgg_id",
    ) -> "LightGamesRecommender":
        """Create a LightGamesRecommender from a Turi Create model."""
        data = turi_create_to_numpy(model=model, user_id=user_id, item_id=item_id)
        return cls(data=data)

    def to_npz(self: "LightGamesRecommender", file_path: Union[Path, str]) -> None:
        """Save data into an .npz file."""
        self.data.to_npz(file_path)

    @classmethod
    def from_npz(
        cls: Type["LightGamesRecommender"],
        file_path: Union[Path, str],
    ) -> "LightGamesRecommender":
        """Load data from an .npz file."""
        data = CollaborativeFilteringData.from_npz(file_path)
        return cls(data)

    @property
    def known_games(self: "LightGamesRecommender") -> FrozenSet[int]:
        if self._known_games is not None:
            return self._known_games
        self._known_games = frozenset(self.items_labels)
        return self._known_games

    @property
    def rated_games(self: "LightGamesRecommender") -> FrozenSet[int]:
        return self.known_games

    @property
    def num_games(self: "LightGamesRecommender") -> int:
        return len(self.items_labels)

    @property
    def known_users(self: "LightGamesRecommender") -> FrozenSet[str]:
        if self._known_users is not None:
            return self._known_users
        self._known_users = frozenset(self.users_labels)
        return self._known_users

    @property
    def num_users(self: "LightGamesRecommender") -> int:
        return len(self.users_labels)

    def _recommendation_scores(
        self: "LightGamesRecommender",
        users: Optional[List[str]] = None,
        games: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Calculate recommendations scores for certain users and games."""

        if users:
            user_ids = np.array([self.users_indexes[user] for user in users])
            users_factors = self.users_factors[user_ids]
            users_linear_terms = self.users_linear_terms[user_ids].reshape(-1, 1)
        else:
            users_factors = self.users_factors[:-1, :]
            users_linear_terms = self.users_linear_terms[:-1].reshape(-1, 1)

        if games:
            game_ids = np.array([self.items_indexes[game] for game in games])
            items_factors = self.items_factors[:, game_ids]
            items_linear_terms = self.items_linear_terms[game_ids].reshape(1, -1)
        else:
            items_factors = self.items_factors[:, :-1]
            items_linear_terms = self.items_linear_terms[:-1].reshape(1, -1)

        return (
            users_factors @ items_factors  # (num_users, num_items)
            + users_linear_terms  # (num_users, 1)
            + items_linear_terms  # (1, num_items)
            + self.intercept  # (1,)
        )

    def recommend(
        self: "LightGamesRecommender",
        users: Iterable[str],
        **kwargs,
    ) -> pd.DataFrame:
        """Calculate recommendations for certain users."""

        users = list(users)
        scores = self._recommendation_scores(users=users)
        return dataframe_from_scores(users, self.items_labels, scores)

    def recommend_as_numpy(
        self: "LightGamesRecommender",
        users: Iterable[str],
        games: Iterable[int],
    ) -> np.ndarray:
        """Calculate recommendations for certain users and games as a numpy array."""

        users = list(users)
        games = list(games)
        return self._recommendation_scores(users=users, games=games)

    def recommend_similar(
        self: "LightGamesRecommender",
        games: Iterable[int],
        **kwargs,
    ) -> pd.DataFrame:
        """
        Recommend games similar to the given games based on cosine similarity of latent factors.
        """

        games = list(games)
        game_ids = np.array([self.items_indexes[game] for game in games])
        game_factors = self.items_factors[:, game_ids]

        scores = cosine_similarity(game_factors, self.items_factors[:, :-1]).mean(
            axis=0
        )

        result = pd.DataFrame(index=self.items_labels, data={"score": scores})
        result["rank"] = result["score"].rank(method="min", ascending=False).astype(int)
        result.sort_values("rank", inplace=True)

        return result

    def similar_games(
        self: "LightGamesRecommender",
        games: Iterable[int],
        **kwargs,
    ) -> pd.DataFrame:
        """Find games similar to the given games based on cosine similarity of latent factors."""

        games = list(games)
        game_ids = np.array([self.items_indexes[game] for game in games])
        game_factors = self.items_factors[:, game_ids]

        scores = cosine_similarity(game_factors, self.items_factors[:, :-1])
        return dataframe_from_scores(games, self.items_labels, scores)


def cosine_similarity(matrix_1: np.ndarray, matrix_2: np.ndarray) -> np.ndarray:
    """
    Calculates the cosine similarity between two matrices.

    The input matrices need to be of shape (m,n) and (m,l); the result shape will be (n,l).
    """

    dot_product = matrix_1.T @ matrix_2  # (n,l)
    matrix_1_norm = np.linalg.norm(matrix_1, axis=0)  # (n,)
    matrix_2_norm = np.linalg.norm(matrix_2, axis=0)  # (l,)
    outer_prod_norm = np.outer(matrix_1_norm, matrix_2_norm)  # (n,l)

    return dot_product / outer_prod_norm  # (n,l)


def turi_create_to_numpy(
    model: RecommenderModel,
    *,
    user_id: str = "bgg_user_name",
    item_id: str = "bgg_id",
) -> CollaborativeFilteringData:
    """Convert a Turi Create model into NumPy arrays."""

    intercept = model.coefficients["intercept"]
    users_labels = model.coefficients[user_id][user_id].to_numpy()
    users_linear_terms = model.coefficients[user_id]["linear_terms"].to_numpy()
    LOGGER.info("Loaded %d user linear terms", len(users_linear_terms))
    users_factors = model.coefficients[user_id]["factors"].to_numpy()
    LOGGER.info("Loaded user factors with shape %dx%d", *users_factors.shape)

    items_labels = model.coefficients[item_id][item_id].to_numpy()
    items_linear_terms = model.coefficients[item_id]["linear_terms"].to_numpy()
    LOGGER.info("Loaded %d item linear terms", len(items_linear_terms))
    items_factors = model.coefficients[item_id]["factors"].to_numpy().T
    LOGGER.info("Loaded item factors with shape %dx%d", *items_factors.shape)

    return CollaborativeFilteringData(
        intercept=intercept,
        users_labels=users_labels,
        users_linear_terms=users_linear_terms,
        users_factors=users_factors,
        items_labels=items_labels,
        items_linear_terms=items_linear_terms,
        items_factors=items_factors,
    )
