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

# %%
import polars as pl
from sklearn.metrics import ndcg_score
import turicreate as tc

# %load_ext nb_black
# %load_ext lab_black

# %%
ratings = (
    pl.read_ndjson("../../board-game-data/scraped/bgg_RatingItem.jl")
    .lazy()
    .filter(pl.col("bgg_user_rating").is_not_null())
    .select(
        "bgg_id",
        "bgg_user_name",
        "bgg_user_rating",
        (
            (pl.col("bgg_id").count().over("bgg_user_name") >= 100)
            & (pl.arange(0, pl.count()).shuffle().over("bgg_user_name") < 25)
        ).alias("is_test_row"),
    )
    .collect()
)

# %%
train_test = ratings.partition_by(
    "is_test_row",
    as_dict=True,
)
data_train = train_test[False]
data_train.drop_in_place("is_test_row")
data_train = data_train.sort("bgg_user_name", "bgg_id")
data_test = train_test[True]
data_test.drop_in_place("is_test_row")
data_test = data_test.sort("bgg_user_name", "bgg_id")
data_train.shape, data_test.shape

# %%
data_train.write_csv("ratings_train.csv")
data_test.write_csv("ratings_test.csv")
del train_test, data_train

# %%
ratings = tc.SFrame.read_csv("ratings_train.csv")
ratings

# %%
model = tc.ranking_factorization_recommender.create(
    observation_data=ratings,
    user_id="bgg_user_name",
    item_id="bgg_id",
    target="bgg_user_rating",
    num_factors=32,
    max_iterations=10,
    verbose=True,
)

# %%
y_true = data_test["bgg_user_rating"].to_numpy().reshape((-1, 25))
y_true.shape


# %%
def recommend_from_pl(data):
    user = data["bgg_user_name"][0]
    sa = model.recommend(
        users=[user],
        items=data["bgg_id"].to_numpy(),
        k=len(data),
        exclude_known=False,
    ).sort("bgg_id")["score"]
    assert len(data) == len(sa)
    return pl.DataFrame(data={"score": sa.to_numpy()})


# %%
recommendations = data_test.groupby("bgg_user_name").apply(recommend_from_pl)
y_score = recommendations.to_numpy().reshape((-1, 25))

# %%
ndcg_score(y_true, y_score)
