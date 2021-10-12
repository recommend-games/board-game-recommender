import turicreate as tc
from board_game_recommender import BGGRecommender

recommender = BGGRecommender.load("../recommend-games-server/data/recommender_bgg")
columns = BGGRecommender.columns_ratings
columns["updated_at"] = str
ratings = BGGRecommender.load_ratings_json(
    "../board-game-data/scraped/bgg_RatingItem.jl", columns=columns
)
ratings["month"] = ratings["updated_at"].apply(
    lambda x: x[:7] if isinstance(x, str) else None
)

user_rating_count = ratings.groupby(
    key_column_names="bgg_user_name", operations={"count": tc.aggregate.COUNT()}
)
heavy_users = user_rating_count[user_rating_count["count"] >= 100]["bgg_user_name"]
# len(heavy_users)

game_rating_count = ratings.groupby(
    key_column_names="bgg_id", operations={"count": tc.aggregate.COUNT()}
)
big_games = game_rating_count[game_rating_count["count"] >= 250]["bgg_id"]
# len(big_games)

rec = recommender.model.recommend(
    items=big_games, users=heavy_users, k=len(heavy_users), exclude_known=False
)
# rec.shape

scores = rec.groupby(
    key_column_names="bgg_id", operations={"score": tc.aggregate.MEAN("score")}
).sort("score", ascending=False)
scores.print_rows(100)

heavy_users = user_rating_count[user_rating_count["count"] >= 10]["bgg_user_name"]
rec = recommender.model.recommend(users=heavy_users, exclude_known=False, k=100)
rec["rev_rank"] = 101 - rec["rank"]
scores = rec.groupby(
    key_column_names="bgg_id", operations={"score": tc.aggregate.SUM("rev_rank")}
)
scores["score"] = scores["score"] / len(heavy_users)
scores = scores.sort("score", ascending=False)
scores.print_rows(100)
