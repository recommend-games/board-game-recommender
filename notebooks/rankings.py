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
import logging
import sys

from board_game_recommender import BGGRecommender
from board_game_recommender.rankings import calculate_rankings

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8.8s [%(name)s:%(lineno)s] %(message)s",
)

# %load_ext nb_black
# %load_ext lab_black

# %%
recommender = BGGRecommender.load("../../recommend-games-server/data/recommender_bgg/")
recommender

# %%
rankings = calculate_rankings(
    recommender=recommender,
    path_ratings="../../board-game-data/scraped/bgg_RatingItem.jl",
    top=100,
    min_ratings=10,
)
rankings.shape

# %%
rankings.print_rows(100)

# %%
rankings.export_csv("rankings.csv")
