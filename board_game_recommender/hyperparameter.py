"""Methods for hyperparameter tuning."""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import polars as pl

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

    data_train = train_test[False]
    data_train.drop_in_place("is_test_row")
    data_train = data_train.sort("bgg_user_name", "bgg_id")

    data_test = train_test[True]
    data_test.drop_in_place("is_test_row")
    data_test = data_test.sort("bgg_user_name", "bgg_id")

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
