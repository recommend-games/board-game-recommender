#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' loading ranking '''

import argparse
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
        response = requests.patch(url=os.path.join(url, str(id_), ''), data=data)

        if not response.ok:
            LOGGER.warning(
                'there was a problem with the request for %r; reason: %s', game, response.reason)

    return count + 1


def _parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--model', '-m',
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.tc')),
        help='model directory')
    parser.add_argument(
        '--url', '-u', default='http://127.0.0.1:8000/games/', help='upload URL')
    parser.add_argument('--id-field', '-i', default='bgg_id', help='ID field')
    parser.add_argument(
        '--verbose', '-v', action='count', default=0, help='log level (repeat for more verbosity)')

    return parser.parse_args()


def _main():
    args = _parse_args()

    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if args.verbose > 0 else logging.INFO,
        format='%(asctime)s %(levelname)-8.8s [%(name)s:%(lineno)s] %(message)s'
    )

    LOGGER.info(args)

    LOGGER.info('loading recommender from <%s>...', args.model)

    recommender = GamesRecommender.load(args.model)
    games = recommender.recommend()
    count = _upload(games=games, url=args.url, id_field=args.id_field)

    LOGGER.info('done updating %d items', count)


if __name__ == '__main__':
    _main()
