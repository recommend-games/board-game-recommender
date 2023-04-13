# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import turicreate as tc
from pytility import clear_list
from sklearn.metrics import ndcg_score

# %%
from board_game_recommender import BGGRecommender
from board_game_recommender.utils import filter_sframe

# %load_ext nb_black
# %load_ext lab_black

# %%
games = BGGRecommender.load_games_json("../../board-game-data/scraped/bgg_GameItem.jl")
games.shape

# %%
filters = {
    f"{BGGRecommender.id_field}__apply": bool,
    "num_votes__gte": 1,
}
columns = clear_list(k.split("__")[0] for k in filters)
games = filter_sframe(games[columns].dropna(), **filters)
games.shape

# %%
ratings = BGGRecommender.load_ratings_json(
    "../../board-game-data/scraped/bgg_RatingItem.jl"
)
ratings.shape

# %%
ratings = ratings.filter_by(games[BGGRecommender.id_field], BGGRecommender.id_field)
ratings.shape

# %%
train, test = tc.recommender.util.random_split_by_user(
    ratings,
    user_id=BGGRecommender.user_id_field,
    item_id=BGGRecommender.id_field,
    max_num_users=100_000,
    item_test_proportion=0.5,
)
train.shape, test.shape

# %%
(
    len(train[BGGRecommender.user_id_field].unique()),
    len(test[BGGRecommender.user_id_field].unique()),
)

# %%
recommender = BGGRecommender.train(
    games=games,
    ratings=train,
    max_iterations=10,
    verbose=True,
    defaults=False,
)
recommender.model

# %%
recommender.model.recommend(
    users=["markus shepherd"],
    exclude_known=False,
)


# %%
def tc_to_pd(data, id_col, score_col):
    data = data.to_dataframe()
    return data.set_index(id_col)[score_col].sort_index()


# %%
def calculate_ndcg(data, recommender=recommender):
    grouped = data.groupby(
        key_column_names=recommender.user_id_field,
        operations={"rated_games": tc.aggregate.CONCAT(recommender.id_field)},
    )
    for row in grouped:
        user = row[recommender.user_id_field]
        rated = row["rated_games"]

        if len(rated) <= 1:
            continue

        y_pred = recommender.model.recommend(
            users=[user],
            items=rated,
            k=len(rated),
            exclude_known=False,
        )[recommender.id_field, "score"]
        y_pred = tc_to_pd(y_pred, recommender.id_field, "score")
        y_test = data[data[recommender.user_id_field] == user][
            recommender.id_field,
            recommender.rating_id_field,
        ]
        y_test = tc_to_pd(y_test, recommender.id_field, recommender.rating_id_field)
        assert (y_pred.index == y_test.index).all()
        yield ndcg_score(y_true=[y_test], y_score=[y_pred])


# %%
scores = tuple(calculate_ndcg(data=test, recommender=recommender))
