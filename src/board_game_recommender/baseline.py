"""Baseline recommender models."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, cast

import numpy as np
import polars as pl

from board_game_recommender.abc import BaseGamesRecommender

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path
    from typing import Any, Self


def dataframe_from_scores(
    *,
    users: Iterable[str],
    games: Iterable[Any],
    scores: np.ndarray,  # shape: (users, games)
) -> pl.DataFrame:
    """Creates a Pandas DataFrame out of raw recommendation scores."""

    users = tuple(users)
    games = tuple(games)

    rank_cols = (
        pl.col(f"{column}_score")
        .rank(method="min", descending=True)
        .alias(f"{column}_rank")
        for column in users
    )

    result = (
        pl.DataFrame(
            data=scores.T,
            schema=[f"{column}_score" for column in users],
        )
        .lazy()
        .with_columns(
            pl.Series("index", games),
            *rank_cols,
        )
    )

    if len(users) == 1:
        result = result.sort(f"{users[0]}_rank")

    return result.collect()


class RandomGamesRecommender(BaseGamesRecommender):
    """Random recommender."""

    def __init__(self) -> None:
        self.rng = np.random.default_rng()

    @property
    def known_games(self) -> frozenset[int]:
        return frozenset()

    @property
    def rated_games(self) -> frozenset[int]:
        return frozenset()

    @property
    def known_users(self) -> frozenset[str]:
        return frozenset()

    def _recommendation_scores(self, users: int, games: int) -> np.ndarray:
        """Random scores."""
        return self.rng.random((users, games))

    def recommend(  # type: ignore[override]
        self,
        users: Iterable[str],
        games: Iterable[int],
        **kwargs,  # noqa: ARG002
    ) -> pl.DataFrame:
        """Random recommendations for certain users."""

        users = list(users)
        games = list(games)
        scores = self._recommendation_scores(users=len(users), games=len(games))

        return dataframe_from_scores(users=users, games=games, scores=scores)

    def recommend_as_numpy(
        self,
        users: Iterable[str],
        games: Iterable[int],
    ) -> np.ndarray:
        """Random recommendations for certain users and games as a numpy array."""
        users = list(users)
        games = list(games)
        return self._recommendation_scores(users=len(users), games=len(games))

    def recommend_group(  # type: ignore[override]
        self,
        users: Iterable[str],  # noqa: ARG002
        games: Iterable[int],
        **kwargs,  # noqa: ARG002
    ) -> pl.DataFrame:
        """Random recommendations for a group of users."""

        games = list(games)
        scores = self._recommendation_scores(users=1, games=len(games))

        return dataframe_from_scores(users=["_all"], games=games, scores=scores)

    def recommend_group_as_numpy(
        self,
        users: Iterable[str],  # noqa: ARG002
        games: Iterable[int],
    ) -> np.ndarray:
        """Random recommendations for a group of users and games as a numpy array."""
        games = list(games)
        return self._recommendation_scores(users=1, games=len(games))

    def recommend_similar(self, games: Iterable[int], **kwargs) -> pl.DataFrame:
        raise NotImplementedError

    def similar_games(self, games: Iterable[int], **kwargs) -> pl.DataFrame:
        raise NotImplementedError


class PopularGamesRecommender(BaseGamesRecommender):
    """Popular games recommender."""

    id_field: str = "bgg_id"
    user_id_field: str = "bgg_user_name"
    rating_id_field: str = "bgg_user_rating"

    scores: dict[int, float]
    raw_scores: np.ndarray
    default_value: float
    game_ids: tuple[int, ...]

    _known_games: frozenset[int] | None = None

    def __init__(
        self,
        game_ids: Iterable[int],
        scores: np.ndarray,
        default_value: float | None = None,
    ) -> None:
        self.default_value = (
            default_value if default_value is not None else scores.mean()
        )
        self.raw_scores = scores
        self.game_ids = tuple(game_ids)
        self.scores = defaultdict(
            self.default_factory,
            zip(self.game_ids, self.raw_scores),
        )

    @classmethod
    def train(cls, ratings: pl.DataFrame) -> Self:
        """Train the recommender from ratings data."""
        raise NotImplementedError

    @classmethod
    def train_from_csv(cls, ratings_file: Path | str) -> Self:
        """Train the recommender from a ratings file in CSV format."""
        ratings = pl.read_csv(ratings_file)
        return cls.train(
            ratings.select(cls.id_field, cls.user_id_field, cls.rating_id_field),
        )

    @classmethod
    def train_from_json_lines(cls, ratings_file: Path | str) -> Self:
        """Train the recommender from a ratings file in JSON lines format."""
        ratings = pl.read_ndjson(ratings_file)
        return cls.train(
            ratings.select(cls.id_field, cls.user_id_field, cls.rating_id_field),
        )

    @property
    def known_games(self) -> frozenset[int]:
        if self._known_games is not None:
            return self._known_games
        self._known_games = frozenset(self.game_ids)
        return self._known_games

    @property
    def rated_games(self) -> frozenset[int]:
        return self.known_games

    @property
    def num_games(self) -> int:
        return len(self.game_ids)

    @property
    def known_users(self) -> frozenset[str]:
        return frozenset()

    def default_factory(self) -> float:
        """Default value for unknown games."""
        return self.default_value

    def _recommendation_scores(
        self,
        users: int,
        games: list[int] | None = None,
    ) -> np.ndarray:
        """Popularity scores."""
        scores = (
            np.array([self.scores[game_id] for game_id in games])
            if games
            else self.raw_scores
        )
        return np.tile(scores, [users, 1])

    def recommend(
        self,
        users: Iterable[str],
        **kwargs,  # noqa: ARG002
    ) -> pl.DataFrame:
        """Popular recommendations for certain users."""
        users = list(users)
        scores = self._recommendation_scores(users=len(users))
        return dataframe_from_scores(users=users, games=self.game_ids, scores=scores)

    def recommend_as_numpy(
        self,
        users: Iterable[str],
        games: Iterable[int],
    ) -> np.ndarray:
        """Popular recommendations for certain users and games as a numpy array."""
        users = list(users)
        games = list(games)
        return self._recommendation_scores(users=len(users), games=games)

    def recommend_group(
        self,
        users: Iterable[str],  # noqa: ARG002
        **kwargs,  # noqa: ARG002
    ) -> pl.DataFrame:
        """Popular recommendations for a group of users."""
        scores = self._recommendation_scores(users=1)
        return dataframe_from_scores(users=["_all"], games=self.game_ids, scores=scores)

    def recommend_group_as_numpy(
        self,
        users: Iterable[str],  # noqa: ARG002
        games: Iterable[int],
    ) -> np.ndarray:
        """Popular recommendations for a group of users and games as a numpy array."""
        games = list(games)
        return self._recommendation_scores(users=1, games=games)

    def recommend_similar(
        self,
        games: Iterable[int],
        **kwargs,
    ) -> pl.DataFrame:
        raise NotImplementedError

    def similar_games(
        self,
        games: Iterable[int],
        **kwargs,
    ) -> pl.DataFrame:
        raise NotImplementedError


class PopularMeanGamesRecommender(PopularGamesRecommender):
    """Recommend games by their mean rating score."""

    @classmethod
    def train(cls, ratings: pl.DataFrame) -> Self:
        data = ratings.group_by(cls.id_field).agg(
            mean=pl.col(cls.rating_id_field).mean(),
        )
        return cls(
            game_ids=data[cls.id_field],
            scores=data["mean"].to_numpy(),
            default_value=cast(float, ratings[cls.rating_id_field].mean()),
        )


class PopularBayesianGamesRecommender(PopularGamesRecommender):
    """Recommend games by their Bayesian average rating score."""

    ratings_per_dummy: float = 10_000
    dummy_rating: float | None = 5.5

    @classmethod
    def train(cls, ratings: pl.DataFrame) -> Self:
        num_dummies = len(ratings) / cls.ratings_per_dummy
        dummy_rating = cast(
            float,
            ratings[cls.rating_id_field].mean()
            if cls.dummy_rating is None
            else cls.dummy_rating,
        )

        stats = (
            ratings.group_by(cls.id_field)
            .agg(
                mean=pl.col(cls.rating_id_field).mean(),
                size=pl.len(),
            )
            .with_columns(
                bayes=pl.col("mean") * pl.col("size")
                + dummy_rating * num_dummies / (pl.col("size") + num_dummies),
            )
        )

        return cls(
            game_ids=stats[cls.id_field],
            scores=stats["bayes"].to_numpy(),
            default_value=dummy_rating,
        )


class PopularNumRatingsGamesRecommender(PopularGamesRecommender):
    """Recommend games by their number of ratings."""

    @classmethod
    def train(cls, ratings: pl.DataFrame) -> Self:
        data = ratings.group_by(cls.id_field).agg(count=pl.len())
        return cls(
            game_ids=data[cls.id_field],
            scores=data["count"].to_numpy(),
        )
