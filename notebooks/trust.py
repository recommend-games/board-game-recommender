# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import turicreate as tc

from board_game_recommender.trust import user_trust

# %load_ext nb_black
# %load_ext lab_black

# %%
trust = user_trust("../../board-game-data/scraped/bgg_RatingItem.jl")
trust.shape

# %%
trust.sort("trust", ascending=False)

# %%
trust[
    trust["bgg_user_name"].is_in(
        [
            "markus shepherd",
            "tomvasel",
            "quinns",
            "nickolaskola",
            "jonpurkis",
        ]
    )
].sort("trust", ascending=False)

# %%
s = trust["trust"]
tc.visualization.histogram(s[s > 0])
