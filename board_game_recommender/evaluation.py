"""Evaluate recommender models."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple, Union

import numpy as np
import polars as pl
from sklearn.metrics import ndcg_score

from board_game_recommender.base import BaseGamesRecommender

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecommenderTestData:
    """Test data for recommender model evaluation."""

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
    """Load RecommenderTestData from CSV."""

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
    """Recommender model evaluation metrics."""

    ndcg: Dict[int, float]
    ndcg_exp: Dict[int, float]
    rmse: float


def calculate_metrics(
    recommender: BaseGamesRecommender,
    test_data: RecommenderTestData,
    *,
    k_values: Union[None, int, Iterable[int]],
) -> RecommenderMetrics:
    """Calculate RecommenderMetrics for given recommender model and RecommenderTestData."""

    y_true = test_data.ratings
    y_pred = np.array(
        [
            recommender.recommend_as_numpy(users=(user,), games=games)[0, :]
            for user, games in zip(test_data.user_ids, test_data.game_ids)
        ]
    )

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape of ratings ({y_true.shape}) does not match "
            + f"shape of predictions ({y_pred.shape})"
        )

    rmse = float(np.sqrt(np.square(y_true - y_pred).mean()))

    if k_values is None:
        k_values = frozenset()
    elif isinstance(k_values, int):
        k_values = frozenset({k_values})
    else:
        k_values = frozenset(k_values)

    k_values = sorted(k_values | {y_true.shape[-1]})
    ndcg = {}

    for k in k_values:
        ndcg[k] = ndcg_score(
            y_true=y_true,
            y_score=y_pred,
            k=k,
        )

    y_true = np.exp2(y_true) - 1
    ndcg_exp = {}

    for k in k_values:
        ndcg_exp[k] = ndcg_score(
            y_true=y_true,
            y_score=y_pred,
            k=k,
        )

    return RecommenderMetrics(ndcg=ndcg, ndcg_exp=ndcg_exp, rmse=rmse)
