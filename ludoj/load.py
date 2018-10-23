#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' loading ranking '''

import logging
import os
import sys

import requests

from .recommend import GamesRecommender

LOGGER = logging.getLogger(__name__)


def _upload(games, url, id_field='bgg_id'):
    LOGGER.info('uploading recommendations to <%s>...', url)

    count = -1

    for count, game in enumerate(games):
        if count and count % 1000 == 0:
            LOGGER.info('updated %d items so far', count)

        id_ = game.get(id_field)
        if id_field is None:
            continue

        data = {
            'rec_rank': game.get('rank'),
            'rec_rating': game.get('score'),
        }
        response = requests.patch(url=url.format(id_), data=data)

        if not response.ok:
            LOGGER.warning(
                'there was a problem with the request for %r; reason: %s', game, response.reason)

    return count + 1


def _main():
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8.8s [%(name)s:%(lineno)s] %(message)s'
    )

    path = os.path.join(os.path.dirname(__file__), '..', '.tc', '')
    LOGGER.info('loading recommender from <%s>...', path)

    recommender = GamesRecommender.load('.tc/')
    games = recommender.recommend()
    count = _upload(games, 'http://127.0.0.1:8000/games/{}/')

    LOGGER.info('done updating %d items', count)


if __name__ == '__main__':
    _main()
