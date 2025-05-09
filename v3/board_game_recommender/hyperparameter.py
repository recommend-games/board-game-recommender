"""Methods for hyperparameter tuning."""

import argparse
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
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
    user_id_key: str = BGGRecommender.user_id_field,
    game_id_key: str = BGGRecommender.id_field,
    ratings_key: str = BGGRecommender.rating_id_field,
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
    user_id_key: str = BGGRecommender.user_id_field,
    game_id_key: str = BGGRecommender.id_field,
    ratings_key: str = BGGRecommender.rating_id_field,
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

    train = tc.SFrame.read_csv(str(path_train))
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


def hyperparameter_tuning(
    *,
    ratings_path: PATH,
    num_factors_list: Iterable[int],
    metric_name: str = "ndcg_exp",
    k_value: int = 10,
    threshold_power_users: int = 200,
    num_test_rows: int = 100,
    user_id_key: str = BGGRecommender.user_id_field,
    game_id_key: str = BGGRecommender.id_field,
    ratings_key: str = BGGRecommender.rating_id_field,
    max_iterations: int = 25,
    verbose: bool = False,
) -> int:
    """Run hyperparameter tuning."""

    num_factors_list = sorted(num_factors_list)

    with TemporaryDirectory() as dir_out:
        path_out = Path(dir_out).resolve()
        LOGGER.info("Using temporary dir <%s>", path_out)

        path_out_train = path_out / "train.csv"
        path_out_test = path_out / "test.csv"

        ratings_train_test_split(
            path_in=ratings_path,
            path_out_train=path_out_train,
            path_out_test=path_out_test,
            threshold_power_users=threshold_power_users,
            num_test_rows=num_test_rows,
            user_id_key=user_id_key,
            game_id_key=game_id_key,
            ratings_key=ratings_key,
        )

        return find_best_num_factors(
            path_train=path_out_train,
            path_test=path_out_test,
            num_factors_list=num_factors_list,
            metric_name=metric_name,
            k_value=k_value,
            ratings_per_user=num_test_rows,
            user_id_key=user_id_key,
            game_id_key=game_id_key,
            ratings_key=ratings_key,
            max_iterations=max_iterations,
            verbose=verbose,
        )


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Find the best number of latent factors for a collaborative filtering model",
    )
    parser.add_argument("ratings", help="path to ratings in JSON lines format")
    parser.add_argument(
        "--num-factors",
        "-n",
        nargs="+",
        type=int,
        default=(32,),
        help="number of latent factors to try out",
    )
    parser.add_argument(
        "--metric",
        "-m",
        default="ndcg_exp",
        choices=("ndcg", "ndcg_exp"),
        help="target metric to choose the best model",
    )
    parser.add_argument(
        "--k-value",
        "-k",
        type=int,
        default=10,
        help="use the top k recommendations for evaluation",
    )
    parser.add_argument(
        "--power-users",
        "-p",
        type=int,
        default=200,
        help="users with at least this many ratings will be included in the test set",
    )
    parser.add_argument(
        "--test-rows",
        "-t",
        type=int,
        default=100,
        help="number of test rows per (power) user",
    )
    parser.add_argument(
        "--user-id-key",
        "-u",
        default=BGGRecommender.user_id_field,
        help="User ID field in the ratings file",
    )
    parser.add_argument(
        "--game-id-key",
        "-g",
        default=BGGRecommender.id_field,
        help="Game ID field in the ratings file",
    )
    parser.add_argument(
        "--ratings-key",
        "-r",
        default=BGGRecommender.rating_id_field,
        help="Ratings field in the ratings file",
    )
    parser.add_argument(
        "--max-iterations",
        "-M",
        type=int,
        default=100,
        help="maximal number of training steps",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="log level (repeat for more verbosity)",
    )

    return parser.parse_args()


def _main():
    args = _parse_args()

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG if args.verbose > 0 else logging.INFO,
        format="%(asctime)s %(levelname)-8.8s [%(name)s:%(lineno)s] %(message)s",
    )

    LOGGER.info(args)

    best_num_factors = hyperparameter_tuning(
        ratings_path=args.ratings,
        num_factors_list=args.num_factors,
        metric_name=args.metric,
        k_value=args.k_value,
        threshold_power_users=args.power_users,
        num_test_rows=args.test_rows,
        user_id_key=args.user_id_key,
        game_id_key=args.game_id_key,
        ratings_key=args.ratings_key,
        max_iterations=args.max_iterations,
        verbose=bool(args.verbose),
    )

    # TODO log results to MLflow

    LOGGER.info("The best number of latent factors is: %d", best_num_factors)


if __name__ == "__main__":
    _main()
