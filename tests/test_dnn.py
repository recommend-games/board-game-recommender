"""Tests for the PyTorch collaborative filtering model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

pytest.importorskip("torch", reason="the torch extra is not installed")

import torch  # type: ignore[import-not-found]

from board_game_recommender.dnn import (
    CollaborativeFilteringModel,
    TrainingResult,
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

    result = train(
        ratings,
        num_factors=8,
        num_epochs=300,
        learning_rate=0.05,
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
