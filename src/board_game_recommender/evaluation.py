"""Evaluate recommender models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    import os
    from collections.abc import Iterable

    from board_game_recommender.abc import BaseGamesRecommender

LOGGER = logging.getLogger(__name__)

DEFAULT_USER_ID_KEY = "bgg_user_name"
DEFAULT_GAME_ID_KEY = "bgg_id"
DEFAULT_RATINGS_KEY = "bgg_user_rating"


@dataclass(frozen=True)
class RecommenderTestData[GameKeyType, UserKeyType]:
    """Test data for recommender model evaluation."""

    user_ids: tuple[UserKeyType, ...]  # (num_users,)
    game_ids: np.ndarray  # (num_users, ratings_per_user)
    ratings: np.ndarray  # (num_users, ratings_per_user)


@dataclass(frozen=True)
class RecommenderMetrics:
    """Recommender model evaluation metrics."""

    ndcg: dict[int, float]
    ndcg_exp: dict[int, float]
    rmse: float
    effective_catalog_size: dict[int, float]


def ratings_train_test_split(  # noqa: PLR0913
    *,
    path_in: str | os.PathLike[str],
    path_out_train: str | os.PathLike[str] | None = None,
    path_out_test: str | os.PathLike[str] | None = None,
    threshold_power_users: int = 200,
    num_test_rows: int = 100,
    user_id_key: str = DEFAULT_USER_ID_KEY,
    game_id_key: str = DEFAULT_GAME_ID_KEY,
    ratings_key: str = DEFAULT_RATINGS_KEY,
    seed: int | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Split the ratings in the given JSON lines file into train and test data.

    Test rows are sampled from "power users", i.e., users with at least
    `threshold_power_users` ratings; exactly `num_test_rows` of their ratings are
    held out. All other ratings end up in the training data.
    """

    if num_test_rows > threshold_power_users:
        msg = (
            f"Cannot hold out {num_test_rows} rows per user "
            f"from users with as few as {threshold_power_users} ratings"
        )
        raise ValueError(msg)

    path_in = Path(path_in).resolve()
    path_out_train = Path(path_out_train).resolve() if path_out_train else None
    path_out_test = Path(path_out_test).resolve() if path_out_test else None

    LOGGER.info(
        "Reading ratings from <%s>, sampling %d rows "
        "from users with at least %d ratings",
        path_in,
        num_test_rows,
        threshold_power_users,
    )

    is_power_user = pl.len().over(user_id_key) >= threshold_power_users
    is_sampled = (
        pl.int_range(pl.len()).shuffle(seed=seed).over(user_id_key) < num_test_rows
    )

    ratings = (
        pl.scan_ndjson(path_in)
        .drop_nulls(subset=[ratings_key, game_id_key, user_id_key])
        .select(
            game_id_key,
            user_id_key,
            ratings_key,
            (is_power_user & is_sampled).alias("is_test_row"),
        )
        .collect()
    )
    LOGGER.info("Done reading %d ratings from <%s>", len(ratings), path_in)

    data_train = (
        ratings.filter(~pl.col("is_test_row"))
        .drop("is_test_row")
        .sort(user_id_key, game_id_key)
    )
    data_test = (
        ratings.filter("is_test_row").drop("is_test_row").sort(user_id_key, game_id_key)
    )
    del ratings

    LOGGER.info(
        "Split into %d training and %d test rows",
        len(data_train),
        len(data_test),
    )

    if path_out_train:
        LOGGER.info("Writing training data to <%s>", path_out_train)
        data_train.write_csv(path_out_train)

    if path_out_test:
        LOGGER.info("Writing test data to <%s>", path_out_test)
        data_test.write_csv(path_out_test)

    return data_train, data_test


def load_test_data(
    path: str | os.PathLike[str],
    ratings_per_user: int,
    user_id_key: str = DEFAULT_USER_ID_KEY,
    game_id_key: str = DEFAULT_GAME_ID_KEY,
    ratings_key: str = DEFAULT_RATINGS_KEY,
) -> RecommenderTestData:
    """Load RecommenderTestData from CSV."""

    path = Path(path).resolve()
    LOGGER.info("Loading test data from <%s>…", path)

    data = pl.read_csv(path)
    LOGGER.info("Read %d rows", len(data))

    if len(data) % ratings_per_user != 0:
        msg = (
            f"The number of rows ({len(data)}) is not divisible by "
            f"the number of ratings per user ({ratings_per_user})"
        )
        raise ValueError(msg)

    user_ids = data[user_id_key].to_numpy().reshape((-1, ratings_per_user))
    if not (user_ids == user_ids[:, :1]).all():
        msg = (
            f"Test data is not grouped into blocks of {ratings_per_user} rows "
            "per user; sort it by user before loading"
        )
        raise ValueError(msg)

    game_ids = data[game_id_key].to_numpy().reshape((-1, ratings_per_user))
    ratings = data[ratings_key].to_numpy().reshape((-1, ratings_per_user))

    return RecommenderTestData(
        user_ids=tuple(user_ids[:, 0]),
        game_ids=game_ids,
        ratings=ratings,
    )


def prediction_scores[GameKeyType, UserKeyType](
    recommender: BaseGamesRecommender[GameKeyType, UserKeyType],
    test_data: RecommenderTestData[GameKeyType, UserKeyType],
) -> np.ndarray:
    """Calculate the predicted scores from the recommender for the given test data."""
    return np.array(
        [
            recommender.recommend_as_numpy(users=(user,), games=games)[0, :]
            for user, games in zip(test_data.user_ids, test_data.game_ids, strict=True)
        ],
    )


def _dcg(gains: np.ndarray, k: int) -> np.ndarray:
    """Discounted cumulative gain of the top k columns of each row."""
    discounts = 1 / np.log2(np.arange(2, k + 2))
    return (gains[:, :k] * discounts).sum(axis=-1)


def ndcg_score(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """
    Normalised discounted cumulative gain, averaged over all rows.

    Equivalent to `sklearn.metrics.ndcg_score(..., ignore_ties=True)`: tied scores
    are broken by column order rather than by averaging gains across the tie. This
    makes the metric deterministic, and is exact for the continuous scores that
    factorisation models produce.
    """

    ranked_gains = np.take_along_axis(y_true, np.argsort(-y_score, axis=-1), axis=-1)
    ideal_gains = -np.sort(-y_true, axis=-1)

    dcg = _dcg(ranked_gains, k)
    idcg = _dcg(ideal_gains, k)

    scores = np.divide(dcg, idcg, out=np.zeros_like(dcg), where=idcg != 0)
    return float(scores.mean())


def effective_catalog_size[GameKeyType, UserKeyType](
    test_data: RecommenderTestData[GameKeyType, UserKeyType],
    y_pred: np.ndarray,
) -> np.ndarray:
    """
    Calculate the effective catalog size (ECS) for every top-k cutoff.

    ECS measures how much of the catalog a recommender actually uses: it is 1 if
    every user is recommended the same single game, and N if all N games are
    recommended equally often. See Gomez-Uribe & Hunt, "The Netflix Recommender
    System: Algorithms, Business Value, and Innovation" (2015).

    The returned array is indexed by k - 1, i.e., element k - 1 is the ECS over
    the top k recommendations.
    """

    if len(test_data.user_ids) != len(y_pred):
        msg = (
            f"Number of users ({len(test_data.user_ids)}) does not match "
            f"number of prediction rows ({len(y_pred)})"
        )
        raise ValueError(msg)
    if test_data.game_ids.shape != y_pred.shape:
        msg = (
            f"Shape of game IDs ({test_data.game_ids.shape}) does not match "
            f"shape of predictions ({y_pred.shape})"
        )
        raise ValueError(msg)

    num_ranks = y_pred.shape[-1]

    # The games each user would be recommended, in descending order of prediction
    ranked_games = np.take_along_axis(
        test_data.game_ids,
        np.argsort(-y_pred, axis=-1),
        axis=-1,
    )
    game_indexes = np.reshape(
        np.unique(ranked_games, return_inverse=True)[1],
        ranked_games.shape,
    )
    num_games = game_indexes.max() + 1

    # counts[k - 1, g]: how often game g appears in a user's top k recommendations
    counts = np.array(
        [
            np.bincount(game_indexes[:, rank], minlength=num_games)
            for rank in range(num_ranks)
        ],
    ).cumsum(axis=0)

    probs = counts / counts.sum(axis=-1, keepdims=True)
    ranks = np.argsort(-counts, axis=-1).argsort(axis=-1) + 1

    return 2 * (probs * ranks).sum(axis=-1) - 1


def calculate_metrics[GameKeyType, UserKeyType](
    recommender: BaseGamesRecommender[GameKeyType, UserKeyType],
    test_data: RecommenderTestData[GameKeyType, UserKeyType],
    *,
    k_values: int | Iterable[int] | None = None,
) -> RecommenderMetrics:
    """Calculate RecommenderMetrics for given recommender model and test data."""

    y_true = test_data.ratings
    y_pred = prediction_scores(recommender, test_data)

    if y_true.shape != y_pred.shape:
        msg = (
            f"Shape of ratings ({y_true.shape}) does not match "
            f"shape of predictions ({y_pred.shape})"
        )
        raise ValueError(msg)

    rmse = float(np.sqrt(np.square(y_true - y_pred).mean()))

    if k_values is None:
        k_values = frozenset()
    elif isinstance(k_values, int):
        k_values = frozenset({k_values})
    else:
        k_values = frozenset(k_values)

    ks = sorted(k_values | {y_true.shape[-1]})

    ecs_all = effective_catalog_size(test_data, y_pred)
    # At k == the full candidate width, every game is "recommended" to every
    # user, so ECS collapses to a property of the test split, not the model.
    # Only report it for k's the caller actually asked for.
    ecs_ks = sorted(k_values)

    y_true_exp = np.exp2(y_true) - 1

    return RecommenderMetrics(
        ndcg={k: ndcg_score(y_true=y_true, y_score=y_pred, k=k) for k in ks},
        ndcg_exp={k: ndcg_score(y_true=y_true_exp, y_score=y_pred, k=k) for k in ks},
        rmse=rmse,
        effective_catalog_size={k: float(ecs_all[k - 1]) for k in ecs_ks},
    )
