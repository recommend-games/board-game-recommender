"""Calcuate game rankings."""

from pathlib import Path
from typing import Union

import turicreate as tc

from .recommend import BGGRecommender
from .trust import user_trust


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

    columns = BGGRecommender.columns_ratings
    columns["updated_at"] = str

    ratings = BGGRecommender.load_ratings_json(
        ratings_json=path_ratings,
        columns=columns,
    )

    users = ratings.groupby(
        key_column_names="bgg_user_name",
        operations={"ratings_count": tc.aggregate.COUNT()},
    )

    trust = user_trust(ratings=ratings, min_ratings=min_ratings)
    del ratings

    users = users.join(
        tc.SFrame(data={"bgg_user_name": trust.index, "trust": trust.values}),
        on="bgg_user_name",
        how="inner",
    )

    heavy_users = users[(users["ratings_count"] >= min_ratings) & (users["trust"] > 0)]
    del users

    recommendations = recommender.model.recommend(
        users=heavy_users["bgg_user_name"],
        exclude_known=False,
        k=top,
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

    scores = scores.sort(["score_weighted", "score"], ascending=False)
    scores["rank_weighted"] = range(1, len(scores) + 1)
    scores = scores.sort(["score", "score_weighted"], ascending=False)
    scores["rank"] = range(1, len(scores) + 1)
    # scores.print_rows(100)

    return scores
