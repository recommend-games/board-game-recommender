import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import polars as pl

from board_game_recommender.base import BaseGamesRecommender

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecommenderTestData:
    """TODO."""

    user_ids: Tuple[str, ...]
    game_ids: np.ndarray
    ratings: np.ndarray


def load_test_data(
    path: Union[str, Path],
    ratings_per_user: int,
    user_id_key: str = "bgg_user_name",
    game_id_key: str = "bgg_id",
    ratings_key: str = "bgg_user_rating",
) -> RecommenderTestData:
    """TODO."""

    path = Path(path).resolve()
    LOGGER.info("Loading test data from <%s>…", path)

    data = pl.read_csv(path)
    LOGGER.info("Read %d rows", len(data))

    if len(data) % ratings_per_user != 0:
        raise ValueError(
            f"The number of rows ({len(data)}) is not divisible by "
            + f"the number of ratings per user ({ratings_per_user})"
        )

    user_ids = tuple(data[user_id_key][::ratings_per_user])
    game_ids = data[game_id_key].view().reshape((-1, ratings_per_user))
    ratings = data[ratings_key].view().reshape((-1, ratings_per_user))

    return RecommenderTestData(user_ids=user_ids, game_ids=game_ids, ratings=ratings)


@dataclass(frozen=True)
class RecommenderMetrics:
    """TODO."""

    ndcg: Dict[int, float]
    ndcg_exp: Dict[int, float]


def calculate_metrics(
    recommender: BaseGamesRecommender,
    test_data: RecommenderTestData,
) -> RecommenderMetrics:
    """TODO."""

    raise NotImplementedError
