"""Tests for the evaluation harness."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

from board_game_recommender.evaluation import (
    RecommenderTestData,
    calculate_metrics,
    effective_catalog_size,
    load_test_data,
    ndcg_score,
    prediction_scores,
    ratings_train_test_split,
)
from board_game_recommender.light import (
    CollaborativeFilteringData,
    LightGamesRecommender,
)

if TYPE_CHECKING:
    from pathlib import Path

RATINGS_PER_USER = 3

USER_RATING_COUNTS = {"alice": 10, "bob": 10, "carol": 3}
NUM_RATED_ROWS = sum(USER_RATING_COUNTS.values())
UNRATED_GAME_ID = 99
POWER_USER_THRESHOLD = 5
NUM_POWER_USERS = 2
NUM_TEST_ROWS = 2
NUM_SMALL_ROWS = 4


@pytest.fixture(name="recommender")
def fixture_recommender() -> LightGamesRecommender:
    """A tiny two-factor collaborative filtering model."""
    return LightGamesRecommender(
        CollaborativeFilteringData(
            intercept=7.0,
            users_labels=np.array(["alice", "bob"]),
            users_linear_terms=np.array([0.5, -0.5]),
            users_factors=np.array([[1.0, 0.0], [0.0, 1.0]]),
            items_labels=np.array([1, 2, 3]),
            items_linear_terms=np.array([0.1, 0.2, 0.3]),
            items_factors=np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]]),
        ),
    )


@pytest.fixture(name="test_data")
def fixture_test_data() -> RecommenderTestData[int, str]:
    """Test data covering two users and three games each."""
    return RecommenderTestData(
        user_ids=("alice", "bob"),
        game_ids=np.array([[1, 2, 3], [1, 2, 3]]),
        ratings=np.array([[9.0, 6.0, 7.0], [5.0, 10.0, 8.0]]),
    )


def _write_ndjson(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def test_ratings_train_test_split(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for user, count in USER_RATING_COUNTS.items():
        rows += [
            {"bgg_user_name": user, "bgg_id": i, "bgg_user_rating": 5.0 + i % 5}
            for i in range(count)
        ]
    # Rows without a rating must be dropped entirely
    rows.append(
        {"bgg_user_name": "alice", "bgg_id": UNRATED_GAME_ID, "bgg_user_rating": None},
    )
    path_in = _write_ndjson(tmp_path / "ratings.jl", rows)

    path_train = tmp_path / "train.csv"
    path_test = tmp_path / "test.csv"

    train, test = ratings_train_test_split(
        path_in=path_in,
        path_out_train=path_train,
        path_out_test=path_test,
        threshold_power_users=POWER_USER_THRESHOLD,
        num_test_rows=NUM_TEST_ROWS,
        seed=13,
    )

    # Two power users contribute two test rows each; carol is below the threshold
    assert len(test) == NUM_POWER_USERS * NUM_TEST_ROWS
    assert sorted(test["bgg_user_name"].unique()) == ["alice", "bob"]
    assert len(train) == NUM_RATED_ROWS - len(test)
    assert UNRATED_GAME_ID not in train["bgg_id"]

    # Train and test are disjoint, and together they are the full rated data
    combined = pl.concat([train, test]).sort("bgg_user_name", "bgg_id")
    assert len(combined) == NUM_RATED_ROWS
    assert len(combined.unique()) == NUM_RATED_ROWS

    # Both outputs were written and round-trip
    assert pl.read_csv(path_train).equals(train)
    assert pl.read_csv(path_test).equals(test)

    # Test rows are sorted by user, so load_test_data can block them up
    assert test["bgg_user_name"].to_list() == ["alice", "alice", "bob", "bob"]


def test_ratings_train_test_split_drops_rows_with_missing_ids(
    tmp_path: Path,
) -> None:
    # A null game or user id survives as NaN once the label array round-trips
    # through numpy, which then breaks the label -> index lookup in train().
    rows: list[dict[str, object]] = [
        {"bgg_user_name": "alice", "bgg_id": 1, "bgg_user_rating": 7.0},
        {"bgg_user_name": "alice", "bgg_id": None, "bgg_user_rating": 8.0},
        {"bgg_user_name": None, "bgg_id": 2, "bgg_user_rating": 6.0},
    ]
    path_in = _write_ndjson(tmp_path / "ratings.jl", rows)

    train, test = ratings_train_test_split(
        path_in=path_in,
        threshold_power_users=1,
        num_test_rows=0,
    )

    assert len(train) == 1
    assert len(test) == 0
    assert train["bgg_id"].to_list() == [1]


def test_ratings_train_test_split_without_output_paths(tmp_path: Path) -> None:
    rows = [
        {"bgg_user_name": "alice", "bgg_id": i, "bgg_user_rating": 7.0}
        for i in range(NUM_SMALL_ROWS)
    ]
    path_in = _write_ndjson(tmp_path / "ratings.jl", rows)

    train, test = ratings_train_test_split(
        path_in=path_in,
        threshold_power_users=2,
        num_test_rows=1,
    )

    assert len(train) == NUM_SMALL_ROWS - 1
    assert len(test) == 1
    assert not list(tmp_path.glob("*.csv"))


def test_ratings_train_test_split_rejects_impossible_holdout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Cannot hold out"):
        ratings_train_test_split(
            path_in=tmp_path / "ratings.jl",
            threshold_power_users=10,
            num_test_rows=100,
        )


def test_load_test_data(tmp_path: Path) -> None:
    path = tmp_path / "test.csv"
    pl.DataFrame(
        {
            "bgg_user_name": ["alice"] * 3 + ["bob"] * 3,
            "bgg_id": [1, 2, 3, 1, 2, 3],
            "bgg_user_rating": [9.0, 6.0, 7.0, 5.0, 10.0, 8.0],
        },
    ).write_csv(path)

    data = load_test_data(path, ratings_per_user=RATINGS_PER_USER)

    assert data.user_ids == ("alice", "bob")
    np.testing.assert_array_equal(data.game_ids, [[1, 2, 3], [1, 2, 3]])
    np.testing.assert_allclose(data.ratings, [[9.0, 6.0, 7.0], [5.0, 10.0, 8.0]])


def test_load_test_data_rejects_indivisible_row_count(tmp_path: Path) -> None:
    path = tmp_path / "test.csv"
    pl.DataFrame(
        {
            "bgg_user_name": ["alice"] * 2,
            "bgg_id": [1, 2],
            "bgg_user_rating": [9.0, 6.0],
        },
    ).write_csv(path)

    with pytest.raises(ValueError, match="not divisible"):
        load_test_data(path, ratings_per_user=RATINGS_PER_USER)


def test_load_test_data_rejects_ungrouped_users(tmp_path: Path) -> None:
    path = tmp_path / "test.csv"
    pl.DataFrame(
        {
            "bgg_user_name": ["alice", "bob", "alice", "bob", "alice", "bob"],
            "bgg_id": [1, 1, 2, 2, 3, 3],
            "bgg_user_rating": [9.0, 5.0, 6.0, 10.0, 7.0, 8.0],
        },
    ).write_csv(path)

    with pytest.raises(ValueError, match="not grouped into blocks"):
        load_test_data(path, ratings_per_user=RATINGS_PER_USER)


def test_prediction_scores(
    recommender: LightGamesRecommender,
    test_data: RecommenderTestData[int, str],
) -> None:
    scores = prediction_scores(recommender, test_data)

    assert scores.shape == (2, 3)
    # alice: intercept 7 + user bias 0.5 + item bias + factors (1,0)·q_i
    np.testing.assert_allclose(scores[0], [8.6, 7.7, 8.3])
    # bob: intercept 7 + user bias -0.5 + item bias + factors (0,1)·q_i
    np.testing.assert_allclose(scores[1], [6.6, 7.7, 7.3])


def test_ndcg_score_perfect_and_reversed_ranking() -> None:
    y_true = np.array([[3.0, 2.0, 1.0]])

    assert ndcg_score(y_true, np.array([[3.0, 2.0, 1.0]]), k=3) == pytest.approx(1.0)

    reversed_score = ndcg_score(y_true, np.array([[1.0, 2.0, 3.0]]), k=3)
    assert 0.0 < reversed_score < 1.0

    # DCG of the reversed ranking is 1/log2(2) + 2/log2(3) + 3/log2(4)
    dcg = 1 + 2 / math.log2(3) + 3 / 2
    idcg = 3 + 2 / math.log2(3) + 1 / 2
    assert reversed_score == pytest.approx(dcg / idcg)


def test_ndcg_score_ignores_rows_without_relevance() -> None:
    y_true = np.array([[0.0, 0.0], [1.0, 0.0]])
    y_score = np.array([[0.2, 0.1], [0.2, 0.1]])
    # The all-zero row scores 0, the perfectly ranked row scores 1
    assert ndcg_score(y_true, y_score, k=2) == pytest.approx(0.5)


def test_ndcg_score_at_k_only_considers_top_k() -> None:
    y_true = np.array([[1.0, 1.0, 10.0]])
    # Swapping the two irrelevant items below k must not change the score
    first = ndcg_score(y_true, np.array([[3.0, 2.0, 1.0]]), k=1)
    second = ndcg_score(y_true, np.array([[3.0, 1.0, 2.0]]), k=1)
    assert first == pytest.approx(second)


def test_effective_catalog_size_identical_rankings() -> None:
    num_users = 5
    test_data: RecommenderTestData[int, str] = RecommenderTestData(
        user_ids=tuple(f"u{i}" for i in range(num_users)),
        game_ids=np.tile([10, 20, 30], (num_users, 1)),
        ratings=np.zeros((num_users, 3)),
    )
    y_pred = np.tile([3.0, 2.0, 1.0], (num_users, 1))

    # Every user sees the same ranking, so the top k uses exactly k games
    np.testing.assert_allclose(effective_catalog_size(test_data, y_pred), [1, 2, 3])


def test_effective_catalog_size_spread_rankings() -> None:
    num_users = 3
    test_data: RecommenderTestData[int, str] = RecommenderTestData(
        user_ids=tuple(f"u{i}" for i in range(num_users)),
        game_ids=np.tile([10, 20, 30], (num_users, 1)),
        ratings=np.zeros((num_users, 3)),
    )
    # Cyclic rotations: at every k all three games are used equally often
    y_pred = np.array([np.roll([3.0, 2.0, 1.0], i) for i in range(num_users)])

    np.testing.assert_allclose(effective_catalog_size(test_data, y_pred), [3, 3, 3])


def test_effective_catalog_size_rejects_user_count_mismatch(
    test_data: RecommenderTestData[int, str],
) -> None:
    with pytest.raises(ValueError, match="Number of users"):
        effective_catalog_size(test_data, np.zeros((3, 3)))


def test_effective_catalog_size_rejects_shape_mismatch(
    test_data: RecommenderTestData[int, str],
) -> None:
    with pytest.raises(ValueError, match="Shape of game IDs"):
        effective_catalog_size(test_data, np.zeros((2, 4)))


def test_calculate_metrics(
    recommender: LightGamesRecommender,
    test_data: RecommenderTestData[int, str],
) -> None:
    metrics = calculate_metrics(recommender, test_data, k_values=1)

    # The full width is always included alongside the requested k values,
    # but not for ECS: at k == full width it is degenerate (see below).
    assert sorted(metrics.ndcg) == [1, 3]
    assert sorted(metrics.ndcg_exp) == [1, 3]
    assert sorted(metrics.effective_catalog_size) == [1]

    y_pred = prediction_scores(recommender, test_data)
    expected_rmse = float(np.sqrt(np.square(test_data.ratings - y_pred).mean()))
    assert metrics.rmse == pytest.approx(expected_rmse)

    assert metrics.ndcg[3] == pytest.approx(
        ndcg_score(test_data.ratings, y_pred, k=3),
    )
    assert metrics.ndcg_exp[3] == pytest.approx(
        ndcg_score(np.exp2(test_data.ratings) - 1, y_pred, k=3),
    )
    assert all(0.0 <= value <= 1.0 for value in metrics.ndcg.values())


def test_calculate_metrics_k_values_variants(
    recommender: LightGamesRecommender,
    test_data: RecommenderTestData[int, str],
) -> None:
    assert sorted(calculate_metrics(recommender, test_data).ndcg) == [3]
    assert sorted(calculate_metrics(recommender, test_data, k_values=None).ndcg) == [3]
    assert sorted(
        calculate_metrics(recommender, test_data, k_values=(1, 2)).ndcg,
    ) == [1, 2, 3]


def test_calculate_metrics_ecs_excludes_auto_added_full_width(
    recommender: LightGamesRecommender,
    test_data: RecommenderTestData[int, str],
) -> None:
    # Full width is auto-added to ndcg/rmse's k's but must not leak into ECS,
    # where k == full width is always degenerate (see calculate_metrics).
    assert calculate_metrics(recommender, test_data).effective_catalog_size == {}
    assert sorted(
        calculate_metrics(
            recommender,
            test_data,
            k_values=(1, 2),
        ).effective_catalog_size,
    ) == [1, 2]


def test_calculate_metrics_rejects_shape_mismatch(
    recommender: LightGamesRecommender,
) -> None:
    test_data: RecommenderTestData[int, str] = RecommenderTestData(
        user_ids=("alice", "bob"),
        game_ids=np.array([[1, 2, 3], [1, 2, 3]]),
        ratings=np.array([[9.0, 6.0], [5.0, 10.0]]),
    )
    with pytest.raises(ValueError, match="Shape of ratings"):
        calculate_metrics(recommender, test_data)
