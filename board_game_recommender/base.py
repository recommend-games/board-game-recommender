"""Abstract base recommender class."""

from abc import ABC, abstractmethod
from typing import AbstractSet, Generic, Iterable, TypeVar

GameKeyType = TypeVar("GameKeyType")
UserKeyType = TypeVar("UserKeyType")


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
    def recommend(self: "BaseGamesRecommender", users: Iterable[UserKeyType]):
        """Recommend games for given users."""

    @abstractmethod
    def recommend_similar(self: "BaseGamesRecommender", games: Iterable[GameKeyType]):
        """Recommend games similar to the given ones."""

    @abstractmethod
    def similar_games(self: "BaseGamesRecommender", games: Iterable[GameKeyType]):
        """Find games similar to the given ones."""
