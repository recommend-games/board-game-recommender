"""Tests for the PyTorch collaborative filtering model."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch", reason="the torch extra is not installed")

import torch  # type: ignore[import-not-found]

from board_game_recommender.dnn import (
    CollaborativeFilteringModel,
)
from board_game_recommender.light import (
    CollaborativeFilteringData,
    LightGamesRecommender,
)

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
