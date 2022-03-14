"""Light recommender model, without the heavy Turi Create dependency."""

import logging
import sys

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


# from games.utils import load_recommender
# r = load_recommender("data.bk/recommender_bgg/")
# users_map = dict(zip(users, range(len(users))))
# users_map["markus shepherd"]
# games_map = dict(zip(games, range(len(games))))
# g = users_factors[405817].reshape(1, 32) @ f_games.T
# g.reshape(79113) + w_games + mu + users_linear_terms[405817]
# rec = g.reshape(79113) + w_games + mu + users_linear_terms[405817]
# games[(-rec).argsort()]
# games.to_numpy()[(-rec).argsort()]
# recommendations = games.to_numpy()[(-rec).argsort()]
# r.recommend("markus shepherd", exclude_known=False)
# recommendations[10]
# recommendations[:10]
# sorted(rec, reverse=True)[:10]


def _main():
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8.8s [%(name)s:%(lineno)s] %(message)s",
    )

    for model_path in sys.argv[1:]:
        LOGGER.info("Loading model from <%s>…", model_path)
        recommender = BGGRecommender.load(model_path)
        LOGGER.info("Loaded model: %r", recommender)
        light = LightRecommender(recommender.model)


if __name__ == "__main__":
    _main()
