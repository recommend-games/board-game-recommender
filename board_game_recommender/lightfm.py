"""TODO."""

import logging
import os
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from lightfm import LightFM
from scipy.sparse import coo_matrix

from board_game_recommender.base import BaseGamesRecommender
from board_game_recommender.baseline import dataframe_from_scores

LOGGER = logging.getLogger(__name__)
PATH = Union[str, os.PathLike]


def dataframe_to_matrix(
    data: pd.DataFrame,
    *,
    user_col: str,
    item_col: str,
    rating_col: str,
) -> Tuple[coo_matrix, List[str], List[int]]:
    """TODO."""

    user_cat = pd.Categorical(data[user_col])
    user_labels = list(user_cat.categories)
    user_ids = user_cat.codes

    item_cat = pd.Categorical(data[item_col])
    item_labels = list(item_cat.categories)
    item_ids = item_cat.codes

    matrix = coo_matrix(
        (data[rating_col], (user_ids, item_ids)),
        shape=(len(user_labels), len(item_labels)),
    )

    return matrix, user_labels, item_labels


class LightFMGamesRecommender(BaseGamesRecommender):
    """TODO."""

    id_field: str = "bgg_id"
    user_id_field: str = "bgg_user_name"
    rating_id_field: str = "bgg_user_rating"

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

    @classmethod
    def train(
        cls,
        ratings: pd.DataFrame,
        *,
        num_factors=32,
        max_iterations=100,
        verbose=False,
    ) -> "LightFMGamesRecommender":
        """TODO."""

        ratings_matrix, user_labels, item_labels = dataframe_to_matrix(
            data=ratings,
            user_col=cls.user_id_field,
            item_col=cls.id_field,
            rating_col=cls.rating_id_field,
        )

        # FIXME more model params
        model = LightFM()
        # FIXME more training params
        model.fit(ratings_matrix)

        return cls(model=model, users_labels=user_labels, items_labels=item_labels)

    @classmethod
    def train_from_csv(cls, ratings_file: PATH) -> "LightFMGamesRecommender":
        """Train the recommender from a ratings file in CSV format."""
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
    def train_from_json_lines(cls, ratings_file: PATH) -> "LightFMGamesRecommender":
        """Train the recommender from a ratings file in JSON lines format."""
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
