"""Baseline recommender models."""

import logging
import os
from typing import FrozenSet, Iterable, List, Optional, Union

import numpy as np
import pandas as pd

from board_game_recommender.base import BaseGamesRecommender

LOGGER = logging.getLogger(__name__)
PATH = Union[str, os.PathLike]


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
    def num_games(self) -> int:
        return 0

    @property
    def known_users(self) -> FrozenSet[str]:
        return frozenset()

    @property
    def num_users(self) -> int:
        return 0

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

        result = pd.DataFrame(
            index=games,
            columns=pd.MultiIndex.from_product([users, ["score"]]),
            data=scores.T,
        )
        result[pd.MultiIndex.from_product([users, ["rank"]])] = result.rank(
            method="min",
            ascending=False,
        ).astype(int)

        if len(users) == 1:
            result.sort_values((users[0], "rank"), inplace=True)

        return result[pd.MultiIndex.from_product([users, ["score", "rank"]])]

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

    @property
    def num_users(self) -> int:
        return 0

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

        result = pd.DataFrame(
            index=self.data.index,
            columns=pd.MultiIndex.from_product([users, ["score"]]),
            data=scores.T,
        )
        result[pd.MultiIndex.from_product([users, ["rank"]])] = result.rank(
            method="min",
            ascending=False,
        ).astype(int)

        if len(users) == 1:
            result.sort_values((users[0], "rank"), inplace=True)

        return result[pd.MultiIndex.from_product([users, ["score", "rank"]])]

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
