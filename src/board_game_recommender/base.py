"""Abstract base recommender class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Set as AbstractSet

    import polars as pl

GameKeyType = TypeVar("GameKeyType")
UserKeyType = TypeVar("UserKeyType")


class BaseGamesRecommender(ABC, Generic[GameKeyType, UserKeyType]):
    """Abstract base games recommender class."""

    @property
    @abstractmethod
    def known_games(self) -> AbstractSet[GameKeyType]:
        """IDs of all known games."""

    @property
    def num_games(self) -> int:
        """Number of known games."""
        return len(self.known_games)

    @property
    @abstractmethod
    def rated_games(self) -> AbstractSet[GameKeyType]:
        """IDs of all rated games."""

    @property
    @abstractmethod
    def known_users(self) -> AbstractSet[UserKeyType]:
        """IDs of all known users."""

    @property
    def num_users(self) -> int:
        """Number of known users."""
        return len(self.known_users)

    @abstractmethod
    def recommend(
        self,
        users: Iterable[UserKeyType],
        **kwargs,
    ) -> pl.DataFrame:
        """Recommend games for given users."""

    @abstractmethod
    def recommend_as_numpy(
        self,
        users: Iterable[UserKeyType],
        games: Iterable[GameKeyType],
    ) -> np.ndarray:
        """Recommend games for given users and games as a numpy array."""

    @abstractmethod
    def recommend_group(
        self,
        users: Iterable[UserKeyType],
        **kwargs,
    ) -> pl.DataFrame:
        """Recommend games for given group of users."""

    @abstractmethod
    def recommend_group_as_numpy(
        self,
        users: Iterable[UserKeyType],
        games: Iterable[GameKeyType],
    ) -> np.ndarray:
        """Recommend games for given group of users and games as a numpy array."""

    @abstractmethod
    def recommend_similar(
        self,
        games: Iterable[GameKeyType],
        **kwargs,
    ) -> pl.DataFrame:
        """Recommend games similar to the given ones."""

    @abstractmethod
    def similar_games(
        self,
        games: Iterable[GameKeyType],
        **kwargs,
    ) -> pl.DataFrame:
        """Find games similar to the given ones."""

    def recommend_random_games_as_numpy(
        self,
        users: Iterable[UserKeyType],
        games: Iterable[GameKeyType],
        *,
        num_games: int = 1,
        random_seed: int | None = None,
    ) -> np.ndarray:
        """
        Select random games based on recommendations for a given group of
        users and games as a numpy array.
        """

        users = list(users)
        games = list(games)

        if not games:
            return np.array([])

        # TODO: allow for different weighting schemes
        weights = self.recommend_group_as_numpy(users, games).reshape(-1)
        weights = np.exp(weights)

        rng = np.random.default_rng(seed=random_seed)
        return rng.choice(
            a=games,  # type: ignore[arg-type]
            size=min(num_games, len(games)),
            replace=False,
            p=weights / weights.sum(),
        )
