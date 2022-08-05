"""Abstract base recommender class."""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Set
from typing import Generic, TypeVar

GameKeyType = TypeVar("GameKeyType")
UserKeyType = TypeVar("UserKeyType")


class BaseGamesRecommender(ABC, Generic[GameKeyType, UserKeyType]):
    """Abstract base recommender class."""

    @property
    @abstractmethod
    def known_games(self: "BaseGamesRecommender") -> Set[GameKeyType]:
        pass

    @property
    @abstractmethod
    def rated_games(self: "BaseGamesRecommender") -> Set[GameKeyType]:
        pass

    @property
    @abstractmethod
    def num_games(self: "BaseGamesRecommender") -> int:
        pass

    @property
    @abstractmethod
    def known_users(self: "BaseGamesRecommender") -> Set[UserKeyType]:
        pass

    @property
    @abstractmethod
    def num_users(self: "BaseGamesRecommender") -> int:
        pass

    @abstractmethod
    def recommend(self: "BaseGamesRecommender", users: Iterable[UserKeyType]):
        pass

    @abstractmethod
    def recommend_similar(self: "BaseGamesRecommender", games: Iterable[GameKeyType]):
        pass

    @abstractmethod
    def similar_games(self: "BaseGamesRecommender", games: Iterable[GameKeyType]):
        pass
