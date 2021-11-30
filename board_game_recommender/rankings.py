"""Calcuate game rankings."""

import logging

from pathlib import Path
from typing import Union

import turicreate as tc

from .recommend import BGGRecommender
from .trust import user_trust

LOGGER = logging.getLogger(__name__)


def calculate_rankings(
    recommender: Union[BGGRecommender, Path, str],
    path_ratings: Union[Path, str],
    top: int = 100,
    min_ratings: int = 10,
) -> tc.SFrame:
    """Calcuate game rankings."""

    recommender = (
        BGGRecommender.load(recommender)
        if isinstance(recommender, (Path, str))
        else recommender
    )

    LOGGER.info("Using <%s> for recommendations", recommender)

    columns = BGGRecommender.columns_ratings
    columns["updated_at"] = str

    LOGGER.info("Loading ratings from <%s>", path_ratings)
    ratings = BGGRecommender.load_ratings_json(
        ratings_json=path_ratings,
        columns=columns,
    )
    LOGGER.info("Loaded %d ratings", len(ratings))

    users = ratings.groupby(
        key_column_names="bgg_user_name",
        operations={"ratings_count": tc.aggregate.COUNT()},
    )
    LOGGER.info("Found %d users in total", len(users))

    trust = user_trust(ratings=ratings, min_ratings=min_ratings)
    LOGGER.info("Calculated trust scores for %d users", len(trust))
    del ratings

    users = users.join(
        right=trust,
        on="bgg_user_name",
        how="inner",
    )

    heavy_users = users[(users["ratings_count"] >= min_ratings) & (users["trust"] > 0)]
    LOGGER.info(
        "Got user info for %d users, %d of which are heavy users",
        len(users),
        len(heavy_users),
    )
    del users

    # TODO exclude compilations
    recommendations = recommender.model.recommend(
        users=heavy_users["bgg_user_name"],
        k=top,
        exclude_known=False,
    )
    LOGGER.info("Calculated a total of %d recommendations", len(recommendations))
    del recommender

    recommendations = recommendations.join(heavy_users, on="bgg_user_name", how="inner")
    recommendations["rev_rank"] = top + 1 - recommendations["rank"]
    recommendations["rev_rank_weighted"] = (
        recommendations["rev_rank"] * recommendations["trust"]
    )

    scores = recommendations.groupby(
        key_column_names="bgg_id",
        operations={
            "score": tc.aggregate.SUM("rev_rank"),
            "score_weighted": tc.aggregate.SUM(
                "rev_rank_weighted",
            ),
        },
    )
    del recommendations

    total_weight = heavy_users["trust"].sum()
    scores["score"] = scores["score"] / len(heavy_users)
    scores["score_weighted"] = scores["score_weighted"] / total_weight
    LOGGER.info("Calculated ranking scores for %d games", len(scores))

    # TODO what to do with ties?
    scores = scores.sort(["score_weighted", "score"], ascending=False)
    scores["rank_weighted"] = range(1, len(scores) + 1)
    scores = scores.sort(["score", "score_weighted"], ascending=False)
    scores["rank"] = range(1, len(scores) + 1)
    # scores.print_rows(100)

    return scores
