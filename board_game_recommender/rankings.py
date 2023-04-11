"""Calcuate game rankings."""

import logging
from pathlib import Path
from typing import Union

import turicreate as tc

from board_game_recommender.recommend import BGGRecommender
from board_game_recommender.trust import user_trust

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

    games = ratings.groupby(
        key_column_names="bgg_id",
        operations={
            "avg_rating": tc.aggregate.MEAN("bgg_user_rating"),
            "num_votes": tc.aggregate.COUNT(),
        },
    )
    LOGGER.info("Found %d games in total", len(games))
    del ratings

    if recommender.compilations is not None:
        games = games.filter_by(
            values=recommender.compilations,
            column_name="bgg_id",
            exclude=True,
        )
        LOGGER.info(
            "Restrict recommendations to %d games after removing compilations",
            len(games),
        )

    users = users.join(
        right=trust,
        on="bgg_user_name",
        how="inner",
    )
    del trust

    heavy_users = users[(users["ratings_count"] >= min_ratings) & (users["trust"] > 0)]
    LOGGER.info(
        "Got user info for %d users, %d of which are heavy users",
        len(users),
        len(heavy_users),
    )
    del users

    recommendations = recommender.model.recommend(
        users=heavy_users["bgg_user_name"],
        items=games["bgg_id"],
        exclude_known=False,
        k=top,
    )
    standard_recommendations = recommender.model.recommend(
        users=[None],
        items=games["bgg_id"],
        exclude_known=False,
        k=len(games),
    )["bgg_id", "score"]
    standard_recommendations.rename({"score": "score_standard"}, inplace=True)
    LOGGER.info(
        "Calculated a total of %d recommendations",
        len(recommendations) + len(standard_recommendations),
    )
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
    del heavy_users

    scores = scores.join(
        right=standard_recommendations,
        on="bgg_id",
        how="outer",
    ).join(
        right=games,
        on="bgg_id",
        how="outer",
    )
    LOGGER.info("Calculated different scores for a total of %d games", len(scores))
    del games, standard_recommendations

    for col in ("score", "score_weighted", "score_standard", "avg_rating", "num_votes"):
        scores = scores.fillna(col, 0)

    # TODO what to do with ties?
    scores = scores.sort(
        key_column_names=[
            "score_weighted",
            "score",
            "score_standard",
            "avg_rating",
            "num_votes",
        ],
        ascending=False,
    )
    scores["rank_weighted"] = range(1, len(scores) + 1)

    scores = scores.sort(
        key_column_names=[
            "score",
            "score_weighted",
            "score_standard",
            "avg_rating",
            "num_votes",
        ],
        ascending=False,
    )
    scores["rank"] = range(1, len(scores) + 1)

    return scores
