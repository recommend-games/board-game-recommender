# board-game-recommender

Board game recommendation engine. View the recommendations live at
[Recommend.Games](https://recommend.games/)! Install via

```bash
pip install board-game-recommender
```

## Training new recommender models

### Environment

[Requires Python 3](https://pythonclock.org/). Make sure
[Pipenv](https://docs.pipenv.org/) is installed and create the virtual environment:

```bash
python3 -m pip install --upgrade pipenv
pipenv install --dev
pipenv shell
```

### Datasets

In order to train the models you will need appropriate game and rating data.
You can either scrape your own using the [board-game-scraper](https://gitlab.com/recommend.games/board-game-scraper)
project or take a look at the [BoardGameGeek guild](https://boardgamegeek.com/thread/2287371/boardgamegeek-games-and-ratings-datasets)
to obtain existing datasets.

At the moment there are [recommender implementations](board_game_recommender/recommend.py)
for two sources: [BoardGameGeek](https://boardgamegeek.com/) and [Board Game Atlas](https://www.boardgameatlas.com/).

### Models

We use the recommender implementation by [Turi Create](https://github.com/apple/turicreate).
Two recommender models are supported out of the box:

* [`RankingFactorizationRecommender`](https://apple.github.io/turicreate/docs/api/generated/turicreate.recommender.ranking_factorization_recommender.RankingFactorizationRecommender.html)
 (default): Learns latent factors for each user and game, generally yielding
 very interesting recommendations.
* [`ItemSimilarityRecommender`](https://apple.github.io/turicreate/docs/api/generated/turicreate.recommender.item_similarity_recommender.ItemSimilarityRecommender.html):
 Ranks a game according to its similarity to other ratings by a user, often
 resulting in less interesting recommendations. However, this model is also
 able to find games similar to a given game.

### Run the training

Run the training via the [main script](board_game_recommender/__main__.py):

```bash
python -m board_game_recommender --help
```

E.g., train the default BGG mode like so:

```bash
python -m board_game_recommender \
    --train \
    --games-file bgg_GameItem.jl \
    --ratings-file bgg_RatingItem.jl \
    --model model/output/dir
```

## Links

* [board-game-recommender](https://gitlab.com/recommend.games/board-game-recommender):
 This repository
* [Recommend.Games](https://recommend.games/): board game recommender website
* [recommend-games-server](https://gitlab.com/recommend.games/recommend-games-server):
 Server code for [Recommend.Games](https://recommend.games/)
* [board-game-scraper](https://gitlab.com/recommend.games/board-game-scraper):
 Board game data scraper
