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
from sklearn.preprocessing import FunctionTransformer, QuantileTransformer
import turicreate as tc

# %load_ext nb_black
# %load_ext lab_black

# %%
NUM_LABELS = 25
TOP_K = 10

# %%
data_train = tc.SFrame.read_csv("ratings_train.csv")
data_train.shape

# %%
data_test = pl.read_csv("ratings_test.csv")
data_test.shape

# %%
quantile_transformer = QuantileTransformer()
quantile_transformer.fit(data_train["bgg_user_rating"].to_numpy().reshape(-1, 1))

# %%
exp_transformer = FunctionTransformer(lambda x: np.exp2(x) - 1)


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
def calculate_ndcg(data, model, *, transformer=None, n_labels=NUM_LABELS, k=None):
    y_true = data["bgg_user_rating"].view()
    if transformer is not None:
        y_true = transformer.transform(y_true.reshape(-1, 1))
    recommendations = data.groupby("bgg_user_name").apply(
        partial(recommend_from_pl, model=model)
    )
    y_score = recommendations.to_numpy()
    return ndcg_score(
        y_true=y_true.reshape((-1, n_labels)),
        y_score=y_score.reshape((-1, n_labels)),
        k=k,
    )


# %%
results = {}
for num_factors in (8,):  # (4, 8, 16, 32, 64, 128)
    print(f"{num_factors=}")
    tc_model = tc.ranking_factorization_recommender.create(
        observation_data=data_train,
        user_id="bgg_user_name",
        item_id="bgg_id",
        target="bgg_user_rating",
        num_factors=num_factors,
        max_iterations=10,
        verbose=False,
    )
    ndcg = calculate_ndcg(
        data=data_test,
        model=tc_model,
        transformer=None,
        n_labels=NUM_LABELS,
        k=TOP_K,
    )
    print(f"{ndcg=:.5f}")
    ndcg_quantile = calculate_ndcg(
        data=data_test,
        model=tc_model,
        transformer=quantile_transformer,
        n_labels=NUM_LABELS,
        k=TOP_K,
    )
    print(f"{ndcg_quantile=:.5f}")
    ndcg_exp = calculate_ndcg(
        data=data_test,
        model=tc_model,
        transformer=exp_transformer,
        n_labels=NUM_LABELS,
        k=TOP_K,
    )
    print(f"{ndcg_exp=:.5f}")
    results[num_factors] = {
        "num_factors": num_factors,
        "model": tc_model,
        "ndcg": ndcg,
        "ndcg_quantile": ndcg_quantile,
        "ndcg_exp": ndcg_exp,
    }
    print()

# %%
y_true = data_test["bgg_user_rating"].to_numpy().reshape((-1, NUM_LABELS))
y_rand = np.random.random(y_true.shape)
ndcg_score(y_true, y_rand, k=TOP_K)

# %%
y_true = quantile_transformer.transform(
    data_test["bgg_user_rating"].view().reshape(-1, 1)
).reshape((-1, NUM_LABELS))
y_rand = np.random.random(y_true.shape)
ndcg_score(y_true, y_rand, k=TOP_K)

# %%
y_true = exp_transformer.transform(
    data_test["bgg_user_rating"].view().reshape(-1, 1)
).reshape((-1, NUM_LABELS))
y_rand = np.random.random(y_true.shape)
ndcg_score(y_true, y_rand, k=TOP_K)
