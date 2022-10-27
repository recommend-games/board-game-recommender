"""Board game recommenders."""

from board_game_recommender.__version__ import VERSION, __version__
from board_game_recommender.base import BaseGamesRecommender

try:
    from board_game_recommender.light import LightGamesRecommender
except ImportError:
    pass

try:
    from board_game_recommender.recommend import (
        BGARecommender,
        BGGRecommender,
        GamesRecommender,
    )
except ImportError:
    pass
