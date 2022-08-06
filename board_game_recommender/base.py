"""Abstract base recommender class."""

from abc import ABC, abstractmethod
from typing import AbstractSet, Generic, Iterable, TypeVar

GameKeyType = TypeVar("GameKeyType")
UserKeyType = TypeVar("UserKeyType")


class BaseGamesRecommender(ABC, Generic[GameKeyType, UserKeyType]):
    """Abstract base recommender class."""

    @property
    @abstractmethod
    def known_games(self: "BaseGamesRecommender") -> AbstractSet[GameKeyType]:
        pass

    @property
    def num_games(self: "BaseGamesRecommender") -> int:
        return len(self.known_games)

    @property
    @abstractmethod
    def rated_games(self: "BaseGamesRecommender") -> AbstractSet[GameKeyType]:
        pass

    @property
    @abstractmethod
    def known_users(self: "BaseGamesRecommender") -> AbstractSet[UserKeyType]:
        pass

    @property
    def num_users(self: "BaseGamesRecommender") -> int:
        return len(self.known_users)

    @abstractmethod
    def recommend(self: "BaseGamesRecommender", users: Iterable[UserKeyType]):
        pass

    @abstractmethod
    def recommend_similar(self: "BaseGamesRecommender", games: Iterable[GameKeyType]):
        pass

    @abstractmethod
    def similar_games(self: "BaseGamesRecommender", games: Iterable[GameKeyType]):
        pass
