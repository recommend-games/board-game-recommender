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
    effective_catalog_size: Dict[int, float]


def prediction_scores(
    recommender: BaseGamesRecommender,
    test_data: RecommenderTestData,
) -> np.ndarray:
    """Calculate the predicted scores from the recommender for the given test data."""
    return np.array(
        [
            recommender.recommend_as_numpy(users=(user,), games=games)[0, :]
            for user, games in zip(test_data.user_ids, test_data.game_ids)
        ]
    )


def effective_catalog_size(
    test_data: RecommenderTestData,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Calculate the effective catalog size (ECS)."""

    assert len(test_data.user_ids) == len(y_pred)
    assert test_data.game_ids.shape == y_pred.shape

    pos_counts = (
        pl.LazyFrame(
            data={
                "user_id": np.repeat(test_data.user_ids, y_pred.shape[-1]),
                "game_id": test_data.game_ids.reshape(-1),
                "prediction": y_pred.reshape(-1),
            }
        )
        .with_columns(
            prediction_rank=pl.col("prediction")
            .rank(method="random", descending=True)
            .over("user_id")
        )
        .groupby("game_id", "prediction_rank")
        .count()
        .sort("game_id", "prediction_rank")
        .select(
            "game_id",
            "prediction_rank",
            pl.col("count").cumsum().over("game_id"),
        )
        .collect()
        .pivot(
            columns="game_id",
            index="prediction_rank",
            values="count",
            aggregate_function=None,
        )
        .lazy()
        .sort("prediction_rank")
        .fill_null(strategy="forward")
        .fill_null(0)
        .drop("prediction_rank")
        .collect()
        .to_numpy()
    )

    probs = pos_counts / pos_counts.sum(axis=1).reshape((-1, 1))
    ranks = np.argsort(-1 * pos_counts).argsort() + 1

    return 2 * np.sum(probs * ranks, axis=1) + 1


def calculate_metrics(
    recommender: BaseGamesRecommender,
    test_data: RecommenderTestData,
    *,
    k_values: Union[None, int, Iterable[int]],
) -> RecommenderMetrics:
    """Calculate RecommenderMetrics for given recommender model and RecommenderTestData."""

    y_true = test_data.ratings
    y_pred = prediction_scores(recommender, test_data)

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

    ecs_all = effective_catalog_size(test_data, y_pred)
    ecs = {k: ecs_all[k - 1] for k in k_values}

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

    return RecommenderMetrics(
        ndcg=ndcg,
        ndcg_exp=ndcg_exp,
        rmse=rmse,
        effective_catalog_size=ecs,
    )
