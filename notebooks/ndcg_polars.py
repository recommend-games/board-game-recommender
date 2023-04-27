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
NUM_LABELS = 100
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
def true_scores(data, *, transformer=None, n_labels=NUM_LABELS):
    y_true = data["bgg_user_rating"].view()
    if transformer is not None:
        y_true = transformer.transform(y_true.reshape(-1, 1))
    return y_true.reshape((-1, n_labels))


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


def recommendation_scores(data, model, *, n_labels=NUM_LABELS):
    recommendations = pl.concat(
        recommend_from_pl(data[start : start + n_labels], model)
        for start in range(0, len(data), n_labels)
    )
    return recommendations.to_numpy().reshape((-1, n_labels))


# %%
results = {}
for num_factors in (4, 8, 16, 32, 64, 128):
    print(f"{num_factors=}")
    tc_model = tc.ranking_factorization_recommender.create(
        observation_data=data_train,
        user_id="bgg_user_name",
        item_id="bgg_id",
        target="bgg_user_rating",
        num_factors=num_factors,
        max_iterations=100,
        verbose=False,
    )
    print("Done training.")
    y_score = recommendation_scores(data=data_test, model=tc_model, n_labels=NUM_LABELS)

    ndcg = ndcg_score(
        y_true=true_scores(
            data_test,
            transformer=None,
            n_labels=NUM_LABELS,
        ),
        y_score=y_score,
        k=TOP_K,
    )
    print(f"{ndcg=:.5f}")

    ndcg_quantile = ndcg_score(
        y_true=true_scores(
            data_test,
            transformer=quantile_transformer,
            n_labels=NUM_LABELS,
        ),
        y_score=y_score,
        k=TOP_K,
    )
    print(f"{ndcg_quantile=:.5f}")

    ndcg_exp = ndcg_score(
        y_true=true_scores(
            data_test,
            transformer=exp_transformer,
            n_labels=NUM_LABELS,
        ),
        y_score=y_score,
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
print("random scores")
y_true = true_scores(
    data_test,
    transformer=None,
    n_labels=NUM_LABELS,
)
y_rand = np.random.random(y_true.shape)

ndcg = ndcg_score(
    y_true=y_true,
    y_score=y_rand,
    k=TOP_K,
)
print(f"{ndcg=:.5f}")

ndcg_quantile = ndcg_score(
    y_true=true_scores(
        data_test,
        transformer=quantile_transformer,
        n_labels=NUM_LABELS,
    ),
    y_score=y_rand,
    k=TOP_K,
)
print(f"{ndcg_quantile=:.5f}")

ndcg_exp = ndcg_score(
    y_true=true_scores(
        data_test,
        transformer=exp_transformer,
        n_labels=NUM_LABELS,
    ),
    y_score=y_rand,
    k=TOP_K,
)
print(f"{ndcg_exp=:.5f}")
