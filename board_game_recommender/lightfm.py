"""TODO."""

import logging
import os
from typing import Union

import pandas as pd
from lightfm import LightFM
from scipy.sparse import coo_matrix

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
