"""Tests for the PyTorch collaborative filtering model."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

pytest.importorskip("torch", reason="the torch extra is not installed")

import torch  # type: ignore[import-not-found]

from board_game_recommender.dnn import (
    CollaborativeFilteringModel,
    TrainingResult,
    _main,
    _parse_args,
    train,
)
from board_game_recommender.light import (
    CollaborativeFilteringData,
    LightGamesRecommender,
)

if TYPE_CHECKING:
    from pathlib import Path

NUM_USERS = 5
NUM_ITEMS = 7
NUM_FACTORS = 3
INTERCEPT = 7.5
SEED = 23

# Turi's own defaults, matched by train()'s and the CLI's.
DEFAULT_NUM_FACTORS = 32
DEFAULT_RANKING_REGULARIZATION = 0.25


@pytest.fixture(name="model")
def fixture_model() -> CollaborativeFilteringModel:
    """A small model with every parameter moved off its initial value."""

    torch.manual_seed(SEED)
    model = CollaborativeFilteringModel(
        num_users=NUM_USERS,
        num_items=NUM_ITEMS,
        num_factors=NUM_FACTORS,
    )
    with torch.no_grad():
        model.user_biases.weight.normal_()
        model.item_biases.weight.normal_()
        model.intercept.fill_(INTERCEPT)
    return model


def _all_pairs() -> tuple[torch.Tensor, torch.Tensor]:
    """Every (user, item) combination, flattened."""
    return (
        torch.arange(NUM_USERS).repeat_interleave(NUM_ITEMS),
        torch.arange(NUM_ITEMS).repeat(NUM_USERS),
    )


def _score_matrix(model: CollaborativeFilteringModel) -> np.ndarray:
    users, items = _all_pairs()
    with torch.no_grad():
        return model(users, items).reshape(NUM_USERS, NUM_ITEMS).numpy()


def _parameters(
    model: CollaborativeFilteringModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """User factors, item factors, user biases and item biases as numpy."""
    return (
        model.user_factors.weight.detach().numpy(),
        model.item_factors.weight.detach().numpy(),
        model.user_biases.weight.detach().numpy().reshape(-1),
        model.item_biases.weight.detach().numpy().reshape(-1),
    )


def test_initialisation() -> None:
    torch.manual_seed(SEED)
    model = CollaborativeFilteringModel(
        num_users=NUM_USERS,
        num_items=NUM_ITEMS,
        num_factors=NUM_FACTORS,
    )

    assert model.user_factors.weight.shape == (NUM_USERS, NUM_FACTORS)
    assert model.item_factors.weight.shape == (NUM_ITEMS, NUM_FACTORS)
    assert model.user_biases.weight.shape == (NUM_USERS, 1)
    assert model.item_biases.weight.shape == (NUM_ITEMS, 1)

    # Factors are random so that users do not all start out identical,
    # while the linear terms start from no opinion at all.
    assert model.user_factors.weight.abs().sum() > 0
    assert model.item_factors.weight.abs().sum() > 0
    assert model.user_biases.weight.abs().sum() == 0
    assert model.item_biases.weight.abs().sum() == 0
    assert float(model.intercept) == 0.0


@pytest.mark.parametrize(
    ("field", "value"),
    [("num_users", 0), ("num_users", -1), ("num_items", 0), ("num_factors", 0)],
)
def test_rejects_non_positive_dimensions(field: str, value: int) -> None:
    kwargs = {"num_users": 2, "num_items": 2, "num_factors": 2, field: value}
    with pytest.raises(ValueError, match="must be positive"):
        CollaborativeFilteringModel(**kwargs)  # type: ignore[arg-type]


def test_forward_shape(model: CollaborativeFilteringModel) -> None:
    users, items = _all_pairs()
    assert model(users, items).shape == (NUM_USERS * NUM_ITEMS,)


def test_forward_broadcasts(model: CollaborativeFilteringModel) -> None:
    """A single user against every item needs no manual repeating."""
    scores = model(torch.tensor([0]), torch.arange(NUM_ITEMS))
    assert scores.shape == (NUM_ITEMS,)
    np.testing.assert_allclose(
        scores.detach().numpy(),
        _score_matrix(model)[0],
        rtol=1e-6,
    )


def test_forward_matches_the_intended_formula(
    model: CollaborativeFilteringModel,
) -> None:
    """score = intercept + user bias + item bias + <user factors, item factors>."""

    user_factors, item_factors, user_biases, item_biases = _parameters(model)
    expected = (
        user_factors @ item_factors.T
        + user_biases.reshape(-1, 1)
        + item_biases.reshape(1, -1)
        + float(model.intercept)
    )

    np.testing.assert_allclose(_score_matrix(model), expected, rtol=1e-6)


def test_scores_match_the_light_recommender(
    model: CollaborativeFilteringModel,
) -> None:
    """
    The model produces what `LightGamesRecommender` serves.

    This is the contract that lets a trained model be exported to an .npz file
    and served without Turi Create, so it is the property most worth pinning
    down. Tolerance is float32 against float64, not an approximation.
    """

    user_factors, item_factors, user_biases, item_biases = _parameters(model)

    recommender = LightGamesRecommender(
        CollaborativeFilteringData(
            intercept=float(model.intercept),
            users_labels=np.array([f"u{index}" for index in range(NUM_USERS)]),
            users_linear_terms=user_biases,
            users_factors=user_factors,
            items_labels=np.arange(NUM_ITEMS),
            items_linear_terms=item_biases,
            items_factors=item_factors.T,
        ),
    )

    served = recommender.recommend_as_numpy(
        users=[f"u{index}" for index in range(NUM_USERS)],
        games=list(range(NUM_ITEMS)),
    )

    np.testing.assert_allclose(_score_matrix(model), served, rtol=1e-5, atol=1e-6)


def test_gradients_reach_every_parameter(
    model: CollaborativeFilteringModel,
) -> None:
    users, items = _all_pairs()
    model(users, items).pow(2).mean().backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"{name} received no gradient"
        assert parameter.grad.abs().sum() > 0, f"{name} received a zero gradient"


def _synthetic_ratings(
    *,
    num_users: int,
    num_items: int,
    seed: int,
) -> pl.DataFrame:
    """
    Ratings generated from a known low-rank structure, plus noise.

    Every user rates every item, so `train()` can be exercised against a
    complete, dense matrix without needing to think about missing entries.
    """

    rng = np.random.default_rng(seed)
    user_factors = rng.normal(size=(num_users, 2))
    item_factors = rng.normal(size=(num_items, 2))
    true_scores = 5.0 + user_factors @ item_factors.T
    noisy_scores = true_scores + rng.normal(scale=0.1, size=true_scores.shape)

    return pl.DataFrame(
        {
            "bgg_user_name": [
                f"user{u}" for u in range(num_users) for _ in range(num_items)
            ],
            "bgg_id": [item for _ in range(num_users) for item in range(num_items)],
            "bgg_user_rating": noisy_scores.reshape(-1).tolist(),
        },
    )


def test_train_initializes_the_intercept_at_the_mean_rating() -> None:
    """`num_epochs=0` isolates the initialisation from any training step."""

    ratings = _synthetic_ratings(num_users=NUM_USERS, num_items=NUM_ITEMS, seed=5)
    expected_mean = float(
        ratings["bgg_user_rating"].cast(pl.Float32).to_numpy().mean(),
    )

    result = train(ratings, num_factors=NUM_FACTORS, num_epochs=0, seed=SEED)

    assert float(result.model.intercept) == pytest.approx(expected_mean, abs=1e-3)


def test_train_initializes_biases_from_rating_means() -> None:
    """Same reasoning as the intercept, per user and per item."""

    ratings = _synthetic_ratings(num_users=NUM_USERS, num_items=NUM_ITEMS, seed=6)
    global_mean = float(ratings["bgg_user_rating"].cast(pl.Float32).to_numpy().mean())

    user_mean_by_label = dict(
        ratings.group_by("bgg_user_name").agg(pl.col("bgg_user_rating").mean()).rows(),
    )
    item_mean_by_label = dict(
        ratings.group_by("bgg_id").agg(pl.col("bgg_user_rating").mean()).rows(),
    )

    result = train(ratings, num_factors=NUM_FACTORS, num_epochs=0, seed=SEED)

    expected_user_bias = (
        np.array([user_mean_by_label[label] for label in result.user_labels])
        - global_mean
    )
    expected_item_bias = (
        np.array([item_mean_by_label[label] for label in result.item_labels])
        - global_mean
    )

    np.testing.assert_allclose(
        result.model.user_biases.weight.detach().numpy().reshape(-1),
        expected_user_bias,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        result.model.item_biases.weight.detach().numpy().reshape(-1),
        expected_item_bias,
        atol=1e-4,
    )


def test_train_result_shapes() -> None:
    ratings = _synthetic_ratings(num_users=NUM_USERS, num_items=NUM_ITEMS, seed=1)

    result = train(ratings, num_factors=NUM_FACTORS, num_epochs=1, seed=SEED)

    assert set(result.user_labels) == {f"user{u}" for u in range(NUM_USERS)}
    assert set(result.item_labels) == set(range(NUM_ITEMS))
    assert result.model.num_users == NUM_USERS
    assert result.model.num_items == NUM_ITEMS
    assert result.model.num_factors == NUM_FACTORS


def test_train_drops_null_ratings() -> None:
    ratings = pl.DataFrame(
        {
            "bgg_user_name": ["a", "a", "b"],
            "bgg_id": [1, 2, 1],
            "bgg_user_rating": [7.0, None, 8.0],
        },
    )

    result = train(ratings, num_factors=2, num_epochs=1, seed=SEED)

    # The null row must not create a phantom item
    assert set(result.item_labels) == {1}


def test_train_drops_rows_with_missing_ids() -> None:
    # A null id round-trips through numpy as NaN, which breaks the label ->
    # index lookup below rather than just being silently ignored.
    ratings = pl.DataFrame(
        {
            "bgg_user_name": ["a", "a", None],
            "bgg_id": [1, None, 2],
            "bgg_user_rating": [7.0, 8.0, 6.0],
        },
    )

    result = train(ratings, num_factors=2, num_epochs=1, seed=SEED)

    assert set(result.user_labels) == {"a"}
    assert set(result.item_labels) == {1}


def _all_pairs_for(
    result: TrainingResult,
    ratings: pl.DataFrame,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map a ratings frame's rows onto the model's user/item indexes."""
    user_index = {label: index for index, label in enumerate(result.user_labels)}
    item_index = {label: index for index, label in enumerate(result.item_labels)}
    users = [user_index[u] for u in ratings["bgg_user_name"].to_list()]
    items = [item_index[i] for i in ratings["bgg_id"].to_list()]
    return torch.tensor(users), torch.tensor(items)


def test_train_overfits_a_small_dataset() -> None:
    """
    With enough capacity and enough epochs, training error should collapse.

    This is the property that actually matters: that the loop as a whole
    (indexing, batching, optimiser step) drives the loss down, not just that
    individual pieces are wired up correctly.
    """

    ratings = _synthetic_ratings(num_users=8, num_items=12, seed=2)
    true_ratings = ratings["bgg_user_rating"].to_numpy()
    baseline_mse = float(np.mean((true_ratings - true_ratings.mean()) ** 2))

    # Every user rates every item here, so ranking regularisation would
    # fight the fit: it pushes scores down for items sampled as "unrated"
    # that are, in this fixture, actually rated. Regularisation is its own
    # concern, tested separately.
    result = train(
        ratings,
        num_factors=8,
        num_epochs=300,
        learning_rate=0.05,
        regularization=0,
        linear_regularization=0,
        ranking_regularization=0,
        seed=SEED,
    )

    users, items = _all_pairs_for(result, ratings)
    with torch.no_grad():
        predictions = result.model(users, items).numpy()

    trained_mse = float(
        np.mean((predictions - ratings["bgg_user_rating"].to_numpy()) ** 2),
    )

    assert trained_mse < baseline_mse / 10


def test_train_is_reproducible_with_a_seed() -> None:
    ratings = _synthetic_ratings(num_users=5, num_items=7, seed=3)

    first = train(ratings, num_factors=4, num_epochs=5, seed=SEED)
    second = train(ratings, num_factors=4, num_epochs=5, seed=SEED)

    np.testing.assert_allclose(
        first.model.user_factors.weight.detach().numpy(),
        second.model.user_factors.weight.detach().numpy(),
    )
    np.testing.assert_allclose(
        first.model.item_factors.weight.detach().numpy(),
        second.model.item_factors.weight.detach().numpy(),
    )


def _model_scores(
    model: CollaborativeFilteringModel,
    user_labels: np.ndarray,
    item_labels: np.ndarray,
) -> np.ndarray:
    users = torch.arange(len(user_labels)).repeat_interleave(len(item_labels))
    items = torch.arange(len(item_labels)).repeat(len(user_labels))
    with torch.no_grad():
        return model(users, items).reshape(len(user_labels), len(item_labels)).numpy()


def test_to_collaborative_filtering_data_feeds_the_light_recommender() -> None:
    """A trained model can be wrapped for serving without any extra plumbing."""

    ratings = _synthetic_ratings(num_users=5, num_items=6, seed=4)
    result = train(ratings, num_factors=3, num_epochs=5, seed=SEED)

    recommender = LightGamesRecommender(result.to_collaborative_filtering_data())
    served = recommender.recommend_as_numpy(
        users=list(result.user_labels),
        games=list(result.item_labels),
    )

    np.testing.assert_allclose(
        served,
        _model_scores(result.model, result.user_labels, result.item_labels),
        rtol=1e-5,
        atol=1e-6,
    )


def test_train_result_labels_are_npz_safe() -> None:
    """Object-dtype arrays, which polars gives string columns, can't be saved
    to `.npz` without `allow_pickle=True`."""

    ratings = _synthetic_ratings(num_users=NUM_USERS, num_items=NUM_ITEMS, seed=8)
    result = train(ratings, num_factors=NUM_FACTORS, num_epochs=0, seed=SEED)

    assert result.user_labels.dtype != object
    assert result.item_labels.dtype != object


def test_to_collaborative_filtering_data_survives_an_npz_round_trip(
    tmp_path: Path,
) -> None:
    ratings = _synthetic_ratings(num_users=5, num_items=6, seed=7)
    result = train(ratings, num_factors=3, num_epochs=5, seed=SEED)

    path = tmp_path / "model.npz"
    result.to_collaborative_filtering_data().to_npz(path)
    reloaded = LightGamesRecommender.from_npz(path)

    served = reloaded.recommend_as_numpy(
        users=list(result.user_labels),
        games=list(result.item_labels),
    )

    np.testing.assert_allclose(
        served,
        _model_scores(result.model, result.user_labels, result.item_labels),
        rtol=1e-5,
        atol=1e-6,
    )


def _sparse_synthetic_ratings(
    *,
    num_users: int,
    num_items: int,
    num_ratings_per_user: int,
    seed: int,
) -> pl.DataFrame:
    """
    Like `_synthetic_ratings()`, but each user rates only a subset of items.

    Ranking regularisation only has something to push down if some items are
    actually unrated, which the dense fixture rules out by construction.
    """

    rng = np.random.default_rng(seed)
    user_factors = rng.normal(size=(num_users, 2))
    item_factors = rng.normal(size=(num_items, 2))
    true_scores = 5.0 + user_factors @ item_factors.T
    noisy_scores = true_scores + rng.normal(scale=0.1, size=true_scores.shape)

    users: list[str] = []
    items: list[int] = []
    values: list[float] = []
    for user in range(num_users):
        rated = rng.choice(num_items, size=num_ratings_per_user, replace=False)
        for item in rated:
            users.append(f"user{user}")
            items.append(int(item))
            values.append(float(noisy_scores[user, item]))

    return pl.DataFrame(
        {"bgg_user_name": users, "bgg_id": items, "bgg_user_rating": values},
    )


def _mean_unrated_score(result: TrainingResult, ratings: pl.DataFrame) -> float:
    """Average predicted score over every user's un-rated items."""

    user_index = {label: index for index, label in enumerate(result.user_labels)}
    item_index = {label: index for index, label in enumerate(result.item_labels)}
    rated_ids_by_user = dict(
        ratings.group_by("bgg_user_name").agg(pl.col("bgg_id")).rows(),
    )

    scores = []
    with torch.no_grad():
        for label, rated_ids in rated_ids_by_user.items():
            rated = set(rated_ids)
            unrated_items = torch.tensor(
                [item_index[item] for item in result.item_labels if item not in rated],
            )
            unrated_users = torch.full_like(unrated_items, user_index[label])
            scores.append(result.model(unrated_users, unrated_items))

    return float(torch.cat(scores).mean())


def test_ranking_regularization_lowers_scores_for_unrated_items() -> None:
    """The point of the ranking term: push scores down for items a user
    never rated, towards `unobserved_rating_value`."""

    ratings = _sparse_synthetic_ratings(
        num_users=20,
        num_items=50,
        num_ratings_per_user=10,
        seed=10,
    )
    without_ranking = train(
        ratings,
        num_factors=8,
        num_epochs=100,
        learning_rate=0.05,
        regularization=0,
        linear_regularization=0,
        ranking_regularization=0,
        seed=SEED,
    )
    with_ranking = train(
        ratings,
        num_factors=8,
        num_epochs=100,
        learning_rate=0.05,
        regularization=0,
        linear_regularization=0,
        ranking_regularization=1.0,
        seed=SEED,
    )

    assert _mean_unrated_score(with_ranking, ratings) < _mean_unrated_score(
        without_ranking,
        ratings,
    )


def test_regularization_shrinks_factor_magnitudes() -> None:
    ratings = _synthetic_ratings(num_users=NUM_USERS, num_items=NUM_ITEMS, seed=11)
    unregularized = train(
        ratings,
        num_factors=NUM_FACTORS,
        num_epochs=50,
        learning_rate=0.05,
        regularization=0,
        linear_regularization=0,
        ranking_regularization=0,
        seed=SEED,
    )
    regularized = train(
        ratings,
        num_factors=NUM_FACTORS,
        num_epochs=50,
        learning_rate=0.05,
        regularization=1.0,
        linear_regularization=0,
        ranking_regularization=0,
        seed=SEED,
    )

    assert regularized.model.user_factors.weight.norm() < (
        unregularized.model.user_factors.weight.norm()
    )


def test_unobserved_rating_value_is_estimated_from_the_data(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Turi's own default: the estimated 5% quantile, mean - 1.96 * std."""

    ratings = _synthetic_ratings(num_users=NUM_USERS, num_items=NUM_ITEMS, seed=12)
    values = ratings["bgg_user_rating"].cast(pl.Float32).to_numpy()
    expected = float(values.mean() - 1.96 * values.std(ddof=1))

    with caplog.at_level("INFO"):
        train(ratings, num_factors=NUM_FACTORS, num_epochs=0, seed=SEED)

    assert f"{expected:.4f}" in caplog.text


def test_parse_args_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["dnn.py", "ratings.jl", "model.npz"])

    args = _parse_args()

    assert args.ratings == "ratings.jl"
    assert args.output == "model.npz"
    assert args.num_factors == DEFAULT_NUM_FACTORS
    assert args.ranking_regularization == DEFAULT_RANKING_REGULARIZATION
    assert args.unobserved_rating_value is None
    assert args.k_values == (10,)
    assert args.seed is None


def test_main_trains_evaluates_and_saves_a_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    An end-to-end run of the actual CLI entry point: parse arguments, split,
    train, evaluate, save. This is the one test covering all of that wiring
    at once, rather than each piece in isolation.
    """

    ratings = _synthetic_ratings(num_users=10, num_items=15, seed=20)
    ratings_path = tmp_path / "ratings.jl"
    ratings.write_ndjson(ratings_path)
    output_path = tmp_path / "model.npz"

    # --test-rows must be at least the default --k-values (10), or
    # calculate_metrics() has too few candidate columns for nDCG@10.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dnn.py",
            str(ratings_path),
            str(output_path),
            "--power-users",
            "10",
            "--test-rows",
            "10",
            "--num-epochs",
            "1",
            "--seed",
            str(SEED),
        ],
    )

    _main()

    assert output_path.exists()

    recommender = LightGamesRecommender.from_npz(output_path)
    scores = recommender.recommend_as_numpy(
        users=list(recommender.known_users),
        games=list(recommender.known_games),
    )
    assert np.isfinite(scores).all()
