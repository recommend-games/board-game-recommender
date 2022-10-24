"""Light recommender model, without the heavy Turi Create dependency."""

import logging
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, FrozenSet, Iterable, List, Optional, Union

import numpy as np
import pandas as pd

from board_game_recommender.base import BaseGamesRecommender

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    import turicreate

RecommenderModel = Union[
    "turicreate.recommender.factorization_recommender.FactorizationRecommender",
    "turicreate.recommender.ranking_factorization_recommender.RankingFactorizationRecommender",
]


@dataclass(frozen=True)
class CollaborativeFilteringData:
    """Labels, vectors and matrices for linear collaborative filtering models."""

    intercept: float
    users_labels: np.ndarray
    users_linear_terms: np.ndarray
    users_factors: np.ndarray
    items_labels: np.ndarray
    items_linear_terms: np.ndarray
    items_factors: np.ndarray


class LightGamesRecommender(BaseGamesRecommender):
    """Light recommender without Turi Create dependency."""

    _known_games: Optional[FrozenSet[int]] = None
    _known_users: Optional[FrozenSet[str]] = None

    def __init__(
        self: "LightGamesRecommender",
        model: RecommenderModel,
        *,
        user_id: str = "bgg_user_name",
        item_id: str = "bgg_id",
    ):
        data = turi_create_to_numpy(model=model, user_id=user_id, item_id=item_id)

        self.intercept: float = data.intercept
        self.users_labels: List[str] = list(data.users_labels)
        self.users_indexes = dict(zip(data.users_labels, range(len(data.users_labels))))
        self.users_linear_terms = data.users_linear_terms
        self.users_factors = data.users_factors
        self.items_labels: List[int] = list(data.items_labels)
        self.items_indexes = dict(zip(data.items_labels, range(len(data.items_labels))))
        self.items_linear_terms = data.items_linear_terms
        self.items_factors = data.items_factors

        LOGGER.info(
            "Loaded light recommender with %d users and %d items",
            len(self.users_labels),
            len(self.items_labels),
        )

    @property
    def known_games(self: "LightGamesRecommender") -> FrozenSet[int]:
        if self._known_games is not None:
            return self._known_games
        self._known_games = frozenset(self.items_labels)
        return self._known_games

    @property
    def rated_games(self: "LightGamesRecommender") -> FrozenSet[int]:
        return self.known_games

    @property
    def num_games(self: "LightGamesRecommender") -> int:
        return len(self.items_labels)

    @property
    def known_users(self: "LightGamesRecommender") -> FrozenSet[str]:
        if self._known_users is not None:
            return self._known_users
        self._known_users = frozenset(self.users_labels)
        return self._known_users

    @property
    def num_users(self: "LightGamesRecommender") -> int:
        return len(self.users_labels)

    def recommend(
        self: "LightGamesRecommender",
        users: Iterable[str],
        **kwargs,
    ) -> pd.DataFrame:
        """Calculate recommendations for certain users."""

        users = list(users)
        user_ids = np.array([self.users_indexes[user] for user in users])

        scores = (
            self.users_factors[user_ids] @ self.items_factors
            + self.users_linear_terms[user_ids].reshape(len(user_ids), 1)
            + self.items_linear_terms
            + self.intercept
        )

        result = pd.DataFrame(
            index=self.items_labels,
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

    def recommend_similar(
        self: "LightGamesRecommender",
        games: Iterable[int],
        **kwargs,
    ):
        raise NotImplementedError

    def similar_games(
        self: "LightGamesRecommender",
        games: Iterable[int],
        **kwargs,
    ):
        raise NotImplementedError


def turi_create_to_numpy(
    model: RecommenderModel,
    *,
    user_id: str = "bgg_user_name",
    item_id: str = "bgg_id",
) -> CollaborativeFilteringData:
    """Convert a Turi Create model into NumPy arrays."""

    intercept = model.coefficients["intercept"]
    users_labels = model.coefficients[user_id][user_id].to_numpy()
    users_linear_terms = model.coefficients[user_id]["linear_terms"].to_numpy()
    LOGGER.info("Loaded %d user linear terms", len(users_linear_terms))
    users_factors = model.coefficients[user_id]["factors"].to_numpy()
    LOGGER.info("Loaded user factors with shape %dx%d", *users_factors.shape)

    items_labels = model.coefficients[item_id][item_id].to_numpy()
    items_linear_terms = model.coefficients[item_id]["linear_terms"].to_numpy()
    LOGGER.info("Loaded %d item linear terms", len(items_linear_terms))
    items_factors = model.coefficients[item_id]["factors"].to_numpy().T
    LOGGER.info("Loaded item factors with shape %dx%d", *items_factors.shape)

    return CollaborativeFilteringData(
        intercept=intercept,
        users_labels=users_labels,
        users_linear_terms=users_linear_terms,
        users_factors=users_factors,
        items_labels=items_labels,
        items_linear_terms=items_linear_terms,
        items_factors=items_factors,
    )


def _main():
    from board_game_recommender.recommend import BGGRecommender

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8.8s [%(name)s:%(lineno)s] %(message)s",
    )

    user = "markus shepherd"
    num_games = 10

    for model_path in sys.argv[1:]:
        LOGGER.info("Loading model from <%s>…", model_path)
        recommender = BGGRecommender.load(model_path)
        LOGGER.info("Loaded model: %r", recommender)

        light = LightGamesRecommender(recommender.model)
        recommendations = light.recommend([user])
        print(recommendations.head(num_games))

        recommendations = recommender.model.recommend(
            users=[user],
            exclude_known=False,
            k=num_games,
        )
        recommendations.print_rows(num_games)


if __name__ == "__main__":
    _main()
