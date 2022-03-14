"""Light recommender model, without the heavy Turi Create dependency."""

import logging

LOGGER = logging.getLogger(__name__)


def turi_create_to_numpy(model, *, user_id="bgg_user_name", item_id="bgg_id"):
    """Convert a Turi Create model into NumPy arrays."""

    intercept = model.coefficients["intercept"]
    l_users = model.coefficients[user_id][user_id].to_numpy()
    w_users = model.coefficients[user_id]["linear_terms"].to_numpy()
    LOGGER.info("Loaded %d user linear terms", len(w_users))
    f_users = model.coefficients[user_id]["factors"].to_numpy()
    LOGGER.info("Loaded user factors with shape %dx%d", *f_users.shape)

    l_items = model.coefficients[item_id][item_id].to_numpy()
    w_items = model.coefficients[item_id]["linear_terms"].to_numpy()
    LOGGER.info("Loaded %d item linear terms", len(w_items))
    f_items = model.coefficients[item_id]["factors"].to_numpy().T
    LOGGER.info("Loaded item factors with shape %dx%d", *f_items.shape)

    return intercept, l_users, w_users, f_users, l_items, w_items, f_items


# from games.utils import load_recommender
# r = load_recommender("data.bk/recommender_bgg/")
# users_map = dict(zip(users, range(len(users))))
# users_map["markus shepherd"]
# games_map = dict(zip(games, range(len(games))))
# g = f_users[405817].reshape(1, 32) @ f_games.T
# g.reshape(79113) + w_games + mu + w_users[405817]
# rec = g.reshape(79113) + w_games + mu + w_users[405817]
# games[(-rec).argsort()]
# games.to_numpy()[(-rec).argsort()]
# recommendations = games.to_numpy()[(-rec).argsort()]
# r.recommend("markus shepherd", exclude_known=False)
# recommendations[10]
# recommendations[:10]
# sorted(rec, reverse=True)[:10]
