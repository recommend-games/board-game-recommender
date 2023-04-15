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
from functools import partial
import numpy as np
import polars as pl
from sklearn.metrics import ndcg_score
import turicreate as tc

# %load_ext nb_black
# %load_ext lab_black

# %%
THRESHOLD_POWER_USERS = 100
NUM_LABELS = 25
TOP_K = 10

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
            (pl.col("bgg_id").count().over("bgg_user_name") >= THRESHOLD_POWER_USERS)
            & (pl.arange(0, pl.count()).shuffle().over("bgg_user_name") < NUM_LABELS)
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
del ratings, train_test, data_train


# %%
def recommend_from_pl(data, model):
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
def calculate_ndcg(data, model, n_labels=NUM_LABELS, k=None):
    y_true = data["bgg_user_rating"].to_numpy().reshape((-1, n_labels))
    recommendations = data.groupby("bgg_user_name").apply(
        partial(recommend_from_pl, model=model)
    )
    y_score = recommendations.to_numpy().reshape((-1, n_labels))
    return ndcg_score(y_true, y_score, k=k)


# %%
ratings = tc.SFrame.read_csv("ratings_train.csv")
ratings.shape

# %%
results = {}
for num_factors in (4, 8, 16, 32, 64, 128):
    print(f"{num_factors=}")
    tc_model = tc.ranking_factorization_recommender.create(
        observation_data=ratings,
        user_id="bgg_user_name",
        item_id="bgg_id",
        target="bgg_user_rating",
        num_factors=num_factors,
        max_iterations=10,
        verbose=False,
    )
    ndcg = calculate_ndcg(data=data_test, model=tc_model, n_labels=NUM_LABELS, k=TOP_K)
    print(ndcg)
    results[num_factors] = {"num_factors": num_factors, "model": model, "ndcg": ndcg}

# %%
{k: v["ndcg"] for k, v in results.items()}

# %%
y_true = data_test["bgg_user_rating"].to_numpy().reshape((-1, NUM_LABELS))
y_rand = np.random.random(y_true.shape)
ndcg_score(y_true, y_rand, k=TOP_K)
