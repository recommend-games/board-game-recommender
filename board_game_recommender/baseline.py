"""Baseline recommender models."""

import logging
from typing import FrozenSet, Iterable

import numpy as np
import pandas as pd

from board_game_recommender.base import BaseGamesRecommender

LOGGER = logging.getLogger(__name__)


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
