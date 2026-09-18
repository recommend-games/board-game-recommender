"""Tests for the light recommender's similarity methods."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from board_game_recommender.light import (
    CollaborativeFilteringData,
    LightGamesRecommender,
    cosine_similarity,
)

if TYPE_CHECKING:
    from pathlib import Path

    import polars as pl

UNKNOWN_GAME = 9999
# Game 3 sits exactly between the orthogonal games 1 and 2
MIDDLE_GAME = 3
SQRT_HALF = 1 / math.sqrt(2)


@pytest.fixture(name="recommender")
def fixture_recommender() -> LightGamesRecommender:
    """
    A model whose item factors have known angles.

    Games 1 and 2 are orthogonal; game 3 sits exactly between them, so it is at
    45 degrees to each and cosine similarity 1/sqrt(2).
    """
    return LightGamesRecommender(
        CollaborativeFilteringData(
            intercept=7.0,
            users_labels=np.array(["alice"]),
            users_linear_terms=np.array([0.5]),
            users_factors=np.array([[1.0, 0.0]]),
            items_labels=np.array([1, 2, 3]),
            items_linear_terms=np.array([0.1, 0.2, 0.3]),
            items_factors=np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]),
        ),
    )


def _scores(frame: pl.DataFrame, column: str) -> dict[int, float]:
    """Map game ID to score, so assertions do not depend on row order."""
    return dict(zip(frame["index"].to_list(), frame[column].to_list(), strict=True))


def test_cosine_similarity() -> None:
    matrix = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    similarity = cosine_similarity(matrix, matrix)

    np.testing.assert_allclose(
        similarity,
        [
            [1.0, 0.0, SQRT_HALF],
            [0.0, 1.0, SQRT_HALF],
            [SQRT_HALF, SQRT_HALF, 1.0],
        ],
        atol=1e-12,
    )


def test_cosine_similarity_treats_zero_vectors_as_orthogonal() -> None:
    matrix_1 = np.array([[1.0, 0.0], [0.0, 0.0]])  # second column is a zero vector
    matrix_2 = np.array([[1.0], [0.0]])

    with np.errstate(invalid="raise", divide="raise"):
        similarity = cosine_similarity(matrix_1, matrix_2)

    # A zero vector scores 0 rather than NaN
    np.testing.assert_allclose(similarity, [[1.0], [0.0]])


def test_similar_games(recommender: LightGamesRecommender) -> None:
    result = recommender.similar_games([1, 3])

    assert set(result.columns) == {"index", "1_score", "3_score", "1_rank", "3_rank"}

    np.testing.assert_allclose(
        [_scores(result, "1_score")[game] for game in (1, 2, 3)],
        [1.0, 0.0, SQRT_HALF],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        [_scores(result, "3_score")[game] for game in (1, 2, 3)],
        [SQRT_HALF, SQRT_HALF, 1.0],
        atol=1e-12,
    )
    # Game 3 is most similar to itself, the other two tie behind it
    assert _scores(result, "3_rank")[MIDDLE_GAME] == 1


def test_similar_games_with_unknown_game(recommender: LightGamesRecommender) -> None:
    with np.errstate(invalid="raise", divide="raise"):
        result = recommender.similar_games([UNKNOWN_GAME])

    # An unknown game has no factors, so it is similar to nothing
    np.testing.assert_allclose(result[f"{UNKNOWN_GAME}_score"].to_numpy(), 0.0)


def test_similar_games_without_games(recommender: LightGamesRecommender) -> None:
    result = recommender.similar_games([])

    assert result.columns == ["index"]
    assert result["index"].to_list() == [1, 2, 3]


def test_recommend_similar(recommender: LightGamesRecommender) -> None:
    result = recommender.recommend_similar([1])

    assert set(result.columns) == {"index", "_all_score", "_all_rank"}
    # Results come back ranked best first
    assert result["index"].to_list() == [1, 3, 2]
    assert result["_all_rank"].to_list() == [1, 2, 3]

    np.testing.assert_allclose(
        [_scores(result, "_all_score")[game] for game in (1, 2, 3)],
        [1.0, 0.0, SQRT_HALF],
        atol=1e-12,
    )


def test_recommend_similar_averages_over_games(
    recommender: LightGamesRecommender,
) -> None:
    result = recommender.recommend_similar([1, 2])

    # Game 3 lies between games 1 and 2, so it beats both of them on average
    assert result["index"][0] == MIDDLE_GAME
    np.testing.assert_allclose(
        [_scores(result, "_all_score")[game] for game in (1, 2, 3)],
        [0.5, 0.5, SQRT_HALF],
        atol=1e-12,
    )


def test_recommend_similar_with_unknown_game(
    recommender: LightGamesRecommender,
) -> None:
    with np.errstate(invalid="raise", divide="raise"):
        result = recommender.recommend_similar([1, UNKNOWN_GAME])

    scores = _scores(result, "_all_score")
    # The unknown game contributes 0 to the average instead of poisoning it with NaN
    assert not np.isnan(list(scores.values())).any()
    np.testing.assert_allclose(
        [scores[game] for game in (1, 2, 3)],
        [0.5, 0.0, SQRT_HALF / 2],
        atol=1e-12,
    )


def test_from_npz_casts_factor_arrays_to_float32(tmp_path: Path) -> None:
    data = CollaborativeFilteringData(
        intercept=7.0,
        users_labels=np.array(["alice"]),
        users_linear_terms=np.array([0.5]),
        users_factors=np.array([[1.0, 0.0]]),
        items_labels=np.array([1, 2, 3]),
        items_linear_terms=np.array([0.1, 0.2, 0.3]),
        items_factors=np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]),
    )
    file_path = tmp_path / "model.npz"
    data.to_npz(file_path)

    loaded = CollaborativeFilteringData.from_npz(file_path)

    assert loaded.users_linear_terms.dtype == np.float32
    assert loaded.users_factors.dtype == np.float32
    assert loaded.items_linear_terms.dtype == np.float32
    assert loaded.items_factors.dtype == np.float32
    # Labels are untouched
    assert loaded.items_labels.dtype == data.items_labels.dtype

    recommender = LightGamesRecommender(loaded)

    # The padding row/column added in __init__ must not upcast the arrays back
    assert recommender.users_linear_terms.dtype == np.float32
    assert recommender.users_factors.dtype == np.float32
    assert recommender.items_linear_terms.dtype == np.float32
    assert recommender.items_factors.dtype == np.float32


def test_light_recommender_does_not_retain_original_numeric_arrays() -> None:
    # regression test: `.data` used to keep every numeric array duplicated
    data = CollaborativeFilteringData(
        intercept=7.0,
        users_labels=np.array(["alice"]),
        users_linear_terms=np.array([0.5]),
        users_factors=np.array([[1.0, 0.0]]),
        items_labels=np.array([1, 2, 3]),
        items_linear_terms=np.array([0.1, 0.2, 0.3]),
        items_factors=np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]),
    )
    original_arrays = (
        data.users_linear_terms,
        data.users_factors,
        data.items_linear_terms,
        data.items_factors,
    )

    recommender = LightGamesRecommender(data)

    assert not any(
        attr is array
        for attr in vars(recommender).values()
        for array in original_arrays
    )


def test_recommender_to_npz_round_trips(tmp_path: Path) -> None:
    data = CollaborativeFilteringData(
        intercept=7.0,
        users_labels=np.array(["alice", "bob"]),
        users_linear_terms=np.array([0.5, -0.2]),
        users_factors=np.array([[1.0, 0.0], [0.5, 0.5]]),
        items_labels=np.array([1, 2, 3]),
        items_linear_terms=np.array([0.1, 0.2, 0.3]),
        items_factors=np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]),
    )
    recommender = LightGamesRecommender(data)

    file_path = tmp_path / "model.npz"
    recommender.to_npz(file_path)
    reloaded = LightGamesRecommender.from_npz(file_path)

    assert reloaded.users_labels == recommender.users_labels
    assert reloaded.items_labels == recommender.items_labels
    assert reloaded.intercept == recommender.intercept
    np.testing.assert_allclose(reloaded.users_factors, recommender.users_factors)
    np.testing.assert_allclose(reloaded.items_factors, recommender.items_factors)
    np.testing.assert_allclose(
        reloaded.users_linear_terms,
        recommender.users_linear_terms,
    )
    np.testing.assert_allclose(
        reloaded.items_linear_terms,
        recommender.items_linear_terms,
    )


def test_recommender_to_npz_preserves_label_dtype(tmp_path: Path) -> None:
    # rebuilding from a Python list would narrow the dtype to fit the content
    data = CollaborativeFilteringData(
        intercept=7.0,
        users_labels=np.array(["alice", "bob"], dtype="<U40"),
        users_linear_terms=np.array([0.5, -0.2]),
        users_factors=np.array([[1.0, 0.0], [0.5, 0.5]]),
        items_labels=np.array([1, 2, 3]),
        items_linear_terms=np.array([0.1, 0.2, 0.3]),
        items_factors=np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]),
    )
    recommender = LightGamesRecommender(data)

    file_path = tmp_path / "model.npz"
    recommender.to_npz(file_path)

    with file_path.open(mode="rb") as file:
        files = np.load(file=file)
        assert files["users_labels"].dtype == data.users_labels.dtype


def test_recommend_similar_without_games(recommender: LightGamesRecommender) -> None:
    with np.errstate(invalid="raise", divide="raise"):
        result = recommender.recommend_similar([])

    np.testing.assert_allclose(result["_all_score"].to_numpy(), 0.0)
    assert result["_all_rank"].to_list() == [1, 1, 1]
