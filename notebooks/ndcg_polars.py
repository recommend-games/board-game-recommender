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
)

# %%
train_test = ratings.collect().partition_by("is_test_row", as_dict=True)
train_test[False].shape, train_test[True].shape
