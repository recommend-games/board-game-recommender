"""Tests for the package's public API."""

from __future__ import annotations

import importlib
import pkgutil

import board_game_recommender


def test_public_api_is_importable() -> None:
    """Every name in __all__ can actually be imported from the package."""
    for name in board_game_recommender.__all__:
        assert hasattr(board_game_recommender, name), f"{name} is not importable"


def test_public_api_is_sorted_and_unique() -> None:
    assert board_game_recommender.__all__ == sorted(set(board_game_recommender.__all__))


def test_recommenders_are_exported() -> None:
    """
    Every concrete recommender is reachable from the top level.

    v3 supported `from board_game_recommender import LightGamesRecommender`, and
    downstream consumers rely on it, so new recommenders should be exported too
    rather than only living in their own module.
    """

    exported = set(board_game_recommender.__all__)
    base = board_game_recommender.BaseGamesRecommender

    for module_info in pkgutil.iter_modules(board_game_recommender.__path__):
        try:
            module = importlib.import_module(
                f"{board_game_recommender.__name__}.{module_info.name}",
            )
        except ImportError:
            # Modules behind an optional extra cannot be inspected when that
            # extra is not installed, which is the case in CI.
            continue

        for name in dir(module):
            obj = getattr(module, name)
            if (
                isinstance(obj, type)
                and issubclass(obj, base)
                and obj.__module__ == module.__name__
            ):
                assert name in exported, f"{name} is not exported from the package"
