"""Abstract base recommender class."""

from abc import ABC, abstractmethod


class BaseGamesRecommender(ABC):
    """Abstract base recommender class."""

    @property
    @abstractmethod
    def rated_games(self):
        pass

    @property
    @abstractmethod
    def known_games(self):
        pass

    @property
    @abstractmethod
    def known_users(self):
        pass

    @property
    @abstractmethod
    def num_games(self):
        pass

    @abstractmethod
    def recommend(
        self,
        users=None,
        similarity_model=False,
        games=None,
        games_filters=None,
        exclude=None,
        exclude_known=True,
        exclude_clusters=True,
        exclude_compilations=True,
        num_games=None,
        ascending=True,
        columns=None,
        star_percentiles=None,
        **kwargs,
    ):
        pass

    @abstractmethod
    def recommend_similar(
        self,
        games=None,
        items=None,
        games_filters=None,
        threshold=0.001,
        num_games=None,
        columns=None,
        **kwargs,
    ):
        pass

    @abstractmethod
    def similar_games(self, games, num_games=10, columns=None):
        pass
