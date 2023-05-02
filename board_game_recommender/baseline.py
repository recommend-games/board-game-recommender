"""Baseline recommender models."""

import logging
import os
from typing import Any, FrozenSet, Iterable, List, Optional, Union

import numpy as np
import pandas as pd

from board_game_recommender.base import BaseGamesRecommender

LOGGER = logging.getLogger(__name__)
PATH = Union[str, os.PathLike]


def dataframe_from_scores(
    columns: List[Any],
    index: Iterable[Any],
    scores: np.ndarray,
) -> pd.DataFrame:
    """TODO."""

    result = pd.DataFrame(
        index=list(index),
        columns=pd.MultiIndex.from_product([columns, ["score"]]),
        data=scores.T,
    )
    result[pd.MultiIndex.from_product([columns, ["rank"]])] = result.rank(
        method="min",
        ascending=False,
    ).astype(int)

    if len(columns) == 1:
        result.sort_values((columns[0], "rank"), inplace=True)

    return result[pd.MultiIndex.from_product([columns, ["score", "rank"]])]


class RandomGamesRecommender(BaseGamesRecommender):
    """Random recommender."""

    def __init__(self) -> None:
        self.rng = np.random.default_rng()

    @property
    def known_games(self) -> FrozenSet[int]:
        return frozenset()

    @property
    def rated_games(self) -> FrozenSet[int]:
        return frozenset()

    @property
    def known_users(self) -> FrozenSet[str]:
        return frozenset()

    def _recommendation_scores(self, users: int, games: int) -> np.ndarray:
        """Random scores."""
        return self.rng.random((users, games))

    def recommend(
        self,
        users: Iterable[str],
        games: Iterable[int],
        **kwargs,
    ) -> pd.DataFrame:
        """Random recommendations for certain users."""

        users = list(users)
        games = list(games)
        scores = self._recommendation_scores(users=len(users), games=len(games))

        return dataframe_from_scores(users, games, scores)

    def recommend_as_numpy(
        self,
        users: Iterable[str],
        games: Iterable[int],
    ) -> np.ndarray:
        """Random recommendations for certain users and games as a numpy array."""
        users = list(users)
        games = list(games)
        return self._recommendation_scores(users=len(users), games=len(games))

    def recommend_similar(self, games: Iterable[int], **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    def similar_games(self, games: Iterable[int], **kwargs) -> pd.DataFrame:
        raise NotImplementedError


class PopularGamesRecommender(BaseGamesRecommender):
    """Popular games recommender."""

    id_field: str = "bgg_id"
    user_id_field: str = "bgg_user_name"
    rating_id_field: str = "bgg_user_rating"

    _known_games: Optional[FrozenSet[int]] = None

    def __init__(self, data: pd.Series) -> None:
        self.data = data

    @classmethod
    def train(cls, ratings: pd.DataFrame) -> "PopularGamesRecommender":
        """TODO."""
        raise NotImplementedError

    @classmethod
    def train_from_csv(cls, ratings_file: PATH) -> "PopularGamesRecommender":
        """TODO."""
        ratings = pd.read_csv(ratings_file)
        return cls.train(
            ratings[
                [
                    cls.id_field,
                    cls.user_id_field,
                    cls.rating_id_field,
                ]
            ]
        )

    @classmethod
    def train_from_json_lines(cls, ratings_file: PATH) -> "PopularGamesRecommender":
        """TODO."""
        ratings = pd.read_json(ratings_file, orient="records", lines=True)
        return cls.train(
            ratings[
                [
                    cls.id_field,
                    cls.user_id_field,
                    cls.rating_id_field,
                ]
            ]
        )

    @property
    def known_games(self) -> FrozenSet[int]:
        if self._known_games is not None:
            return self._known_games
        self._known_games = frozenset(self.data.index)
        return self._known_games

    @property
    def rated_games(self) -> FrozenSet[int]:
        return self.known_games

    @property
    def num_games(self) -> int:
        return len(self.data)

    @property
    def known_users(self) -> FrozenSet[str]:
        return frozenset()

    def _recommendation_scores(
        self,
        users: int,
        games: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Popularity scores."""
        scores = self.data.loc[games] if games else self.data
        return np.tile(scores.to_numpy(), [users, 1])

    def recommend(
        self,
        users: Iterable[str],
        **kwargs,
    ) -> pd.DataFrame:
        """Popular recommendations for certain users."""
        users = list(users)
        scores = self._recommendation_scores(users=len(users))
        return dataframe_from_scores(users, self.data.index, scores)

    def recommend_as_numpy(
        self,
        users: Iterable[str],
        games: Iterable[int],
    ) -> np.ndarray:
        """Random recommendations for certain users and games as a numpy array."""
        users = list(users)
        games = list(games)
        return self._recommendation_scores(users=len(users), games=games)

    def recommend_similar(self, games: Iterable[int], **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    def similar_games(self, games: Iterable[int], **kwargs) -> pd.DataFrame:
        raise NotImplementedError


class PopularMeanGamesRecommender(PopularGamesRecommender):
    """TODO."""

    @classmethod
    def train(cls, ratings: pd.DataFrame) -> "PopularMeanGamesRecommender":
        """TODO."""
        data = ratings.groupby(cls.id_field, sort=False)[cls.rating_id_field].mean()
        return cls(data=data)


class PopularBayesianGamesRecommender(PopularGamesRecommender):
    """TODO."""

    ratings_per_dummy: float = 10_000
    dummy_rating: Optional[float] = 5.5

    @classmethod
    def train(cls, ratings: pd.DataFrame) -> "PopularBayesianGamesRecommender":
        """TODO."""

        num_dummies = len(ratings) / cls.ratings_per_dummy
        dummy_rating = (
            ratings[cls.rating_id_field].mean()
            if cls.dummy_rating is None
            else cls.dummy_rating
        )

        stats = ratings.groupby(
            cls.id_field,
            sort=False,
        )[
            cls.rating_id_field
        ].agg(["size", "mean"])

        data = (stats["mean"] * stats["size"] + dummy_rating * num_dummies) / (
            stats["size"] + num_dummies
        )

        return cls(data=data)


class PopularNumRatingsGamesRecommender(PopularGamesRecommender):
    """TODO."""

    @classmethod
    def train(cls, ratings: pd.DataFrame) -> "PopularNumRatingsGamesRecommender":
        """TODO."""
        data = ratings.groupby(cls.id_field, sort=False).size()
        return cls(data=data)
