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
import numpy as np
import polars as pl
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import FunctionTransformer
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
def true_scores(data, *, transformer=None, n_labels=NUM_LABELS):
    y_true = data["bgg_user_rating"].view()
    if transformer is not None:
        y_true = transformer.transform(y_true.reshape(-1, 1))
    return y_true.reshape((-1, n_labels))


def recommend_from_pl(data, model):
    user = data["bgg_user_name"][0]
    rec = model.recommend(
        users=[user],
        items=data["bgg_id"].to_numpy(),
        k=len(data),
        exclude_known=False,
    ).sort("bgg_id")["score"]
    assert len(data) == len(rec)
    return rec.to_numpy()


def recommendation_scores(data, model, *, n_labels=NUM_LABELS):
    return np.array(
        [
            recommend_from_pl(data[start : start + n_labels], model)
            for start in range(0, len(data), n_labels)
        ]
    )


def print_scores(data, y_score, *, k=TOP_K, n_labels=NUM_LABELS):
    ndcg = ndcg_score(
        y_true=true_scores(
            data,
            transformer=None,
            n_labels=n_labels,
        ),
        y_score=y_score,
        k=k,
    )
    print(f"{ndcg=:.5f}")

    ndcg_exp = ndcg_score(
        y_true=true_scores(
            data,
            transformer=FunctionTransformer(lambda x: np.exp2(x) - 1),
            n_labels=n_labels,
        ),
        y_score=y_score,
        k=k,
    )
    print(f"{ndcg_exp=:.5f}")

    return {
        "ndcg": ndcg,
        "ndcg_exp": ndcg_exp,
    }


# %%
for num_factors in (8, 16, 32):
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
    y_rec = recommendation_scores(data=data_test, model=tc_model, n_labels=NUM_LABELS)
    print_scores(data=data_test, y_score=y_rec, k=TOP_K, n_labels=NUM_LABELS)
    print()

# %%
results = {}
for num_factors in (8, 16, 32):
    print(f"{num_factors=}")
    tc_model = tc.factorization_recommender.create(
        observation_data=data_train,
        user_id="bgg_user_name",
        item_id="bgg_id",
        target="bgg_user_rating",
        num_factors=num_factors,
        max_iterations=100,
        verbose=False,
    )
    print("Done training.")
    y_rec = recommendation_scores(data=data_test, model=tc_model, n_labels=NUM_LABELS)
    print_scores(data=data_test, y_score=y_rec, k=TOP_K, n_labels=NUM_LABELS)
    print()

# %%
print("random scores")
y_rand = np.random.random(y_rec.shape)
print_scores(data=data_test, y_score=y_rand, k=TOP_K, n_labels=NUM_LABELS)
