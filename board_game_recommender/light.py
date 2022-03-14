"""Light recommender model, without the heavy Turi Create dependency."""

import logging
import sys

from typing import Iterable

import numpy as np
import pandas as pd

from board_game_recommender.recommend import BGGRecommender

LOGGER = logging.getLogger(__name__)


class LightRecommender:
    """Light recommender without Turi Create dependency."""

    def __init__(self, model, *, user_id="bgg_user_name", item_id="bgg_id"):
        (
            intercept,
            users_labels,
            users_linear_terms,
            users_factors,
            items_labels,
            items_linear_terms,
            items_factors,
        ) = turi_create_to_numpy(model=model, user_id=user_id, item_id=item_id)

        self.intercept = intercept
        self.users_labels = users_labels
        self.users_indexes = dict(zip(users_labels, range(len(users_labels))))
        self.users_linear_terms = users_linear_terms
        self.users_factors = users_factors
        self.items_labels = items_labels
        self.items_indexes = dict(zip(items_labels, range(len(items_labels))))
        self.items_linear_terms = items_linear_terms
        self.items_factors = items_factors

        LOGGER.info(
            "Loaded light recommender with %d users and %d items",
            len(self.users_labels),
            len(self.items_labels),
        )

    def recommend(self: "LightRecommender", users: Iterable[str]) -> pd.DataFrame:
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


def turi_create_to_numpy(model, *, user_id="bgg_user_name", item_id="bgg_id"):
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

    return (
        intercept,
        users_labels,
        users_linear_terms,
        users_factors,
        items_labels,
        items_linear_terms,
        items_factors,
    )


def _main():
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

        light = LightRecommender(recommender.model)
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
