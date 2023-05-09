"""TODO."""

import logging
import os
from typing import Dict, FrozenSet, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
from lightfm import LightFM
from scipy.sparse import coo_matrix

from board_game_recommender.base import BaseGamesRecommender
from board_game_recommender.baseline import dataframe_from_scores

LOGGER = logging.getLogger(__name__)
PATH = Union[str, os.PathLike]


def train(ratings_file: PATH) -> tuple:
    """TODO."""

    ratings = pd.read_csv(ratings_file)
    ratings.dropna(inplace=True)
    LOGGER.info("Loaded %d ratings from <%s>", len(ratings), ratings_file)

    ratings["bgg_id"] = pd.Categorical(ratings["bgg_id"])
    ratings["bgg_user_name"] = pd.Categorical(ratings["bgg_user_name"])
    ratings["user_id"] = ratings["bgg_user_name"].cat.codes
    ratings["item_id"] = ratings["bgg_id"].cat.codes

    users_labels = list(ratings["bgg_user_name"].cat.categories)
    num_users = len(users_labels)
    users_indexes = dict(zip(users_labels, range(num_users)))

    items_labels = list(ratings["bgg_id"].cat.categories)
    num_items = len(items_labels)
    items_indexes = dict(zip(items_labels, range(num_items)))

    ratings_matrix = coo_matrix(
        (ratings["bgg_user_rating"], (ratings["user_id"], ratings["item_id"])),
        shape=(num_users, num_items),
    )

    model = LightFM()
    model.fit(ratings_matrix)

    return model, users_labels, users_indexes, items_labels, items_indexes


class LightFMGamesRecommender(BaseGamesRecommender):
    """TODO."""

    _known_games: Optional[FrozenSet[int]] = None
    _known_users: Optional[FrozenSet[str]] = None

    def __init__(
        self,
        model: LightFM,
        users_labels: Iterable[str],
        items_labels: Iterable[int],
    ) -> None:
        self.model = model

        self.users_labels: List[str] = list(users_labels)
        self.users_indexes: Dict[str, int] = dict(
            zip(
                self.users_labels,
                range(self.num_users),
            )
        )

        self.items_labels: List[int] = list(items_labels)
        self.items_indexes: Dict[int, int] = dict(
            zip(
                self.items_labels,
                range(self.num_games),
            )
        )

        LOGGER.info(
            "Loaded LightFM recommender with %d users and %d items",
            self.num_users,
            self.num_games,
        )

    @property
    def known_games(self) -> FrozenSet[int]:
        if self._known_games is not None:
            return self._known_games
        self._known_games = frozenset(self.items_labels)
        return self._known_games

    @property
    def rated_games(self) -> FrozenSet[int]:
        return self.known_games

    @property
    def num_games(self) -> int:
        return len(self.items_labels)

    @property
    def known_users(self) -> FrozenSet[str]:
        if self._known_users is not None:
            return self._known_users
        self._known_users = frozenset(self.users_labels)
        return self._known_users

    @property
    def num_users(self) -> int:
        return len(self.users_labels)

    def _recommendation_scores(
        self,
        users: Optional[List[str]] = None,
        games: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Calculate recommendations scores for certain users and games."""

        user_ids = (
            np.array([self.users_indexes[user] for user in users])
            if users
            else np.arange(self.num_users)
        )
        game_ids = (
            np.array([self.items_indexes[game] for game in games])
            if games
            else np.arange(self.num_games)
        )

        user_ids_rep = np.repeat(user_ids, len(game_ids))
        game_ids_rep = np.tile(game_ids, len(user_ids))

        predictions = self.model.predict(user_ids_rep, game_ids_rep)

        return predictions.reshape((len(user_ids), len(game_ids)))

    def recommend(self, users: Iterable[str], **kwargs) -> pd.DataFrame:
        """Calculate recommendations for certain users."""

        users = list(users)
        scores = self._recommendation_scores(users=users)
        return dataframe_from_scores(users, self.items_labels, scores)

    def recommend_as_numpy(
        self,
        users: Iterable[str],
        games: Iterable[int],
    ) -> np.ndarray:
        """Calculate recommendations for certain users and games as a numpy array."""

        users = list(users)
        games = list(games)
        return self._recommendation_scores(users=users, games=games)

    def recommend_similar(self, games: Iterable[int], **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    def similar_games(self, games: Iterable[int], **kwargs) -> pd.DataFrame:
        raise NotImplementedError
