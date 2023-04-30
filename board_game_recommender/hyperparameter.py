"""Methods for hyperparameter tuning."""

import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import polars as pl
import turicreate as tc

from board_game_recommender.evaluation import calculate_metrics, load_test_data
from board_game_recommender.recommend import BGGRecommender

LOGGER = logging.getLogger(__name__)
PATH = Union[str, os.PathLike]


def ratings_train_test_split(
    *,
    path_in: PATH,
    path_out_train: Optional[PATH],
    path_out_test: Optional[PATH],
    threshold_power_users: int = 200,
    num_test_rows: int = 100,
    user_id_key: str = "bgg_user_name",
    game_id_key: str = "bgg_id",
    ratings_key: str = "bgg_user_rating",
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Split the given ratings into train and test data."""

    path_in = Path(path_in).resolve()
    path_out_train = Path(path_out_train).resolve() if path_out_train else None
    path_out_test = Path(path_out_test).resolve() if path_out_test else None

    LOGGER.info(
        "Reading ratings from <%s>, sampling %d rows from users with at least %d ratings",
        path_in,
        num_test_rows,
        threshold_power_users,
    )
    ratings = (
        pl.scan_ndjson(path_in)
        .filter(pl.col(ratings_key).is_not_null())
        .select(
            game_id_key,
            user_id_key,
            ratings_key,
            (
                (pl.col(game_id_key).count().over(user_id_key) >= threshold_power_users)
                & (pl.arange(0, pl.count()).shuffle().over(user_id_key) < num_test_rows)
            ).alias("is_test_row"),
        )
        .collect()
    )
    LOGGER.info("Done reading %d ratings from <%s>", len(ratings), path_in)

    train_test: Dict[bool, pl.DataFrame] = ratings.partition_by(
        "is_test_row",
        as_dict=True,
    )
    del ratings

    data_train = train_test[False]
    data_train.drop_in_place("is_test_row")
    data_train = data_train.sort("bgg_user_name", "bgg_id")

    data_test = train_test[True]
    data_test.drop_in_place("is_test_row")
    data_test = data_test.sort("bgg_user_name", "bgg_id")
    del train_test

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


def find_best_num_factors(
    *,
    path_train: PATH,
    path_test: PATH,
    num_factors_list: Iterable[int],
    metric_name: str = "ndcg_exp",
    k_value: int = 10,
    ratings_per_user: int = 100,
    user_id_key: str = "bgg_user_name",
    game_id_key: str = "bgg_id",
    ratings_key: str = "bgg_user_rating",
    max_iterations: int = 25,
    verbose: bool = False,
) -> int:
    """Hyperparameter tuning."""

    path_train = Path(path_train).resolve()
    path_test = Path(path_test).resolve()
    num_factors_list = sorted(num_factors_list)

    LOGGER.info(
        "Reading training data from <%s> and test data from <%s>",
        path_train,
        path_test,
    )

    train = tc.SFrame.read_csv(path_train)
    test = load_test_data(
        path=path_test,
        ratings_per_user=ratings_per_user,
        user_id_key=user_id_key,
        game_id_key=game_id_key,
        ratings_key=ratings_key,
    )

    LOGGER.info(
        "Hyperparameter tuning on %d training and %d test rows with the following factors: %s",
        len(train),
        test.ratings.shape[0] * test.ratings.shape[1],
        num_factors_list,
    )

    all_metrics = {}

    for num_factors in num_factors_list:
        LOGGER.info(
            "Training model with %d latent factors on training data",
            num_factors,
        )
        model = tc.ranking_factorization_recommender.create(
            observation_data=train,
            user_id=user_id_key,
            item_id=game_id_key,
            target=ratings_key,
            num_factors=num_factors,
            max_iterations=max_iterations,
            verbose=verbose,
        )
        LOGGER.info("Done training.")

        recommender = BGGRecommender(model=model)
        metrics = calculate_metrics(
            recommender=recommender,
            test_data=test,
            k_values=k_value,
        )
        all_metrics[num_factors] = metrics
        LOGGER.info("Metrics: %r", metrics)

    scores = {
        num_factors: asdict(metrics)[metric_name][k_value]
        for num_factors, metrics in all_metrics.items()
    }
    best = max(scores.items(), key=lambda x: x[1])
    LOGGER.info(
        "The best <%s> of %.5f was achieved with %d latent factors",
        metric_name,
        best[1],
        best[0],
    )
    return best[0]
