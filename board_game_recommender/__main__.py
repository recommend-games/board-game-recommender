#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Main script & CLI entry point."""

import argparse
import logging
import os
import sys

from .recommend import BGARecommender, BGGRecommender

LOGGER = logging.getLogger(__name__)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _parse_args():
    parser = argparse.ArgumentParser(description="train board game recommender model")
    parser.add_argument("users", nargs="*", help="users to be recommended games")
    parser.add_argument(
        "--bga", "-b", action="store_true", help="use Board Game Atlas data"
    )
    parser.add_argument("--model", "-m", help="model directory")
    parser.add_argument("--train", "-t", action="store_true", help="train a new model")
    parser.add_argument(
        "--similarity", "-s", action="store_true", help="train a new similarity model"
    )
    parser.add_argument("--games-file", "-G", help="games file")
    parser.add_argument("--ratings-file", "-R", help="ratings file")
    parser.add_argument(
        "--side-data-columns",
        "-S",
        nargs="+",
        help="game features to use in recommender model",
    )
    parser.add_argument(
        "--num-rec", "-n", type=int, default=10, help="number of games to recommend"
    )
    parser.add_argument(
        "--max-iterations",
        "-M",
        type=int,
        default=100,
        help="maximal number of training steps",
    )
    parser.add_argument(
        "--diversity", "-d", type=float, default=0, help="diversity in recommendations"
    )
    parser.add_argument(
        "--cooperative", "-c", action="store_true", help="recommend cooperative games"
    )
    parser.add_argument(
        "--games", "-g", type=int, nargs="+", help="restrict to these games"
    )
    parser.add_argument("--players", "-p", type=int, help="player count")
    parser.add_argument("--complexity", "-C", type=float, nargs="+", help="complexity")
    parser.add_argument("--time", "-T", type=float, help="max playing time")
    parser.add_argument("--worst", "-w", action="store_true", help="show worst games")
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

    if args.bga:
        model_cls = BGARecommender
        model_dir = args.model or os.path.join(BASE_DIR, ".bga")
        games_file = args.games_file or os.path.join(
            BASE_DIR, "results", "bga_GameItem.jl"
        )
        ratings_file = args.ratings_file or os.path.join(
            BASE_DIR, "results", "bga_RatingItem.jl"
        )
    else:
        model_cls = BGGRecommender
        model_dir = args.model or os.path.join(BASE_DIR, ".bgg")
        games_file = args.games_file or os.path.join(
            BASE_DIR, "results", "bgg_GameItem.jl"
        )
        ratings_file = args.ratings_file or os.path.join(
            BASE_DIR, "results", "bgg_RatingItem.jl"
        )

    games_filters = {}

    if args.cooperative:
        games_filters["cooperative"] = True

    if args.players:
        # TODO min_players, min_players_rec, or min_players_best?
        games_filters["min_players__lte"] = args.players
        games_filters["max_players__gte"] = args.players

    if args.complexity:
        if len(args.complexity) == 1:
            games_filters["complexity__lte"] = args.complexity[0]
        else:
            games_filters["complexity__gte"] = args.complexity[0]
            games_filters["complexity__lte"] = args.complexity[1]

    if args.time:
        games_filters["min_time__lte"] = args.time * 1.1

    if args.train:
        recommender = model_cls.train_from_files(
            games_file=games_file,
            ratings_file=ratings_file,
            side_data_columns=args.side_data_columns,
            similarity_model=args.similarity,
            max_iterations=args.max_iterations,
            verbose=bool(args.verbose),
        )
        recommender.save(model_dir)
    else:
        recommender = model_cls.load(model_dir)

    for user in [None] + args.users:
        LOGGER.info("#" * 100)

        recommendations = recommender.recommend(
            users=user,
            games=args.games,
            games_filters=games_filters,
            similarity_model=args.similarity,
            num_games=None if args.worst else args.num_rec,
            diversity=0 if args.worst else args.diversity,
        )

        LOGGER.info("best games for <%s>", user or "everyone")
        recommendations.print_rows(num_rows=args.num_rec)

        if args.worst:
            LOGGER.info("worst games for <%s>", user or "everyone")
            recommendations.sort("rank", False).print_rows(num_rows=args.num_rec)


if __name__ == "__main__":
    _main()
