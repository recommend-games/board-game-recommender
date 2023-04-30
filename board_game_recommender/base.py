"""Abstract base recommender class."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, AbstractSet, Generic, Iterable, TypeVar, Union

import numpy as np

GameKeyType = TypeVar("GameKeyType")
UserKeyType = TypeVar("UserKeyType")

if TYPE_CHECKING:
    import pandas
    import turicreate

DataFrame = Union["pandas.DataFrame", "turicreate.SFrame"]


class BaseGamesRecommender(ABC, Generic[GameKeyType, UserKeyType]):
    """Abstract base games recommender class."""

    @property
    @abstractmethod
    def known_games(self: "BaseGamesRecommender") -> AbstractSet[GameKeyType]:
        """IDs of all known games."""

    @property
    def num_games(self: "BaseGamesRecommender") -> int:
        """Number of known games."""
        return len(self.known_games)

    @property
    @abstractmethod
    def rated_games(self: "BaseGamesRecommender") -> AbstractSet[GameKeyType]:
        """IDs of all rated games."""

    @property
    @abstractmethod
    def known_users(self: "BaseGamesRecommender") -> AbstractSet[UserKeyType]:
        """IDs of all known users."""

    @property
    def num_users(self: "BaseGamesRecommender") -> int:
        """Number of known users."""
        return len(self.known_users)

    @abstractmethod
    def recommend(
        self: "BaseGamesRecommender",
        users: Iterable[UserKeyType],
        **kwargs,
    ) -> DataFrame:
        """Recommend games for given users."""

    @abstractmethod
    def recommend_as_numpy(
        self: "BaseGamesRecommender",
        users: Iterable[UserKeyType],
        games: Iterable[GameKeyType],
    ) -> np.ndarray:
        """Recommend games for given users and games as a numpy array."""

    @abstractmethod
    def recommend_similar(
        self: "BaseGamesRecommender",
        games: Iterable[GameKeyType],
        **kwargs,
    ) -> DataFrame:
        """Recommend games similar to the given ones."""

    @abstractmethod
    def similar_games(
        self: "BaseGamesRecommender",
        games: Iterable[GameKeyType],
        **kwargs,
    ) -> DataFrame:
        """Find games similar to the given ones."""
