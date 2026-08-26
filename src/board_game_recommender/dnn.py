"""Collaborative filtering model implemented in PyTorch."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl
import torch  # type: ignore[import-not-found]
from torch import nn, optim  # type: ignore[import-not-found]

from board_game_recommender.evaluation import (
    DEFAULT_GAME_ID_KEY,
    DEFAULT_RATINGS_KEY,
    DEFAULT_USER_ID_KEY,
)
from board_game_recommender.light import CollaborativeFilteringData

if TYPE_CHECKING:
    import numpy as np

LOGGER = logging.getLogger(__name__)


class CollaborativeFilteringModel(nn.Module):
    """
    Linear collaborative filtering model.

    Scores take the same form as Turi Create's factorization recommenders, which
    is what `LightGamesRecommender` serves:

        score(user, item) = intercept
                          + user_bias[user]
                          + item_bias[item]
                          + <user_factors[user], item_factors[item]>

    Users and items are addressed by contiguous indexes, not by their labels;
    mapping BGG user names and game IDs onto those indexes is the caller's job.
    """

    def __init__(
        self,
        *,
        num_users: int,
        num_items: int,
        num_factors: int = 32,
    ) -> None:
        super().__init__()

        if num_users <= 0:
            msg = f"Number of users must be positive, got {num_users}"
            raise ValueError(msg)
        if num_items <= 0:
            msg = f"Number of items must be positive, got {num_items}"
            raise ValueError(msg)
        if num_factors <= 0:
            msg = f"Number of factors must be positive, got {num_factors}"
            raise ValueError(msg)

        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = num_factors

        self.user_factors = nn.Embedding(num_users, num_factors)
        self.item_factors = nn.Embedding(num_items, num_factors)
        self.user_biases = nn.Embedding(num_users, 1)
        self.item_biases = nn.Embedding(num_items, 1)
        self.intercept = nn.Parameter(torch.zeros(()))

        # Small random factors break the symmetry that would otherwise leave
        # every user identical; the linear terms start from no opinion at all.
        nn.init.normal_(self.user_factors.weight, std=0.1)
        nn.init.normal_(self.item_factors.weight, std=0.1)
        nn.init.zeros_(self.user_biases.weight)
        nn.init.zeros_(self.item_biases.weight)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """
        Predict scores for the given pairs of user and item indexes.

        The two tensors must broadcast against each other; the result has their
        broadcast shape.
        """

        dot_product = (self.user_factors(users) * self.item_factors(items)).sum(dim=-1)

        return (
            dot_product
            + self.user_biases(users).squeeze(-1)
            + self.item_biases(items).squeeze(-1)
            + self.intercept
        )


@dataclass(frozen=True)
class TrainingResult:
    """A trained model, together with the labels its indexes refer to."""

    model: CollaborativeFilteringModel
    user_labels: np.ndarray
    item_labels: np.ndarray

    def to_collaborative_filtering_data(self) -> CollaborativeFilteringData:
        """Convert to the format `LightGamesRecommender` serves."""

        user_biases = self.model.user_biases.weight.detach().numpy()
        item_biases = self.model.item_biases.weight.detach().numpy()

        return CollaborativeFilteringData(
            intercept=float(self.model.intercept),
            users_labels=self.user_labels,
            users_linear_terms=user_biases.reshape(-1),
            users_factors=self.model.user_factors.weight.detach().numpy(),
            items_labels=self.item_labels,
            items_linear_terms=item_biases.reshape(-1),
            items_factors=self.model.item_factors.weight.detach().numpy().T,
        )


def _numpy_safe(labels: np.ndarray) -> np.ndarray:
    """
    Give string labels a fixed-width dtype rather than polars' `object`.

    `.npz` files can't store object arrays without `allow_pickle=True`, which
    `CollaborativeFilteringData.from_npz()` deliberately doesn't set.
    """
    return labels.astype(str) if labels.dtype == object else labels


def _ranking_loss(
    model: CollaborativeFilteringModel,
    batch_users: torch.Tensor,
    num_sampled_negative_examples: int,
    unobserved_rating_value: float,
) -> torch.Tensor:
    """
    Turi's ranking term for one batch: for each user, sample random items and
    score them, take the highest-scoring one (the worst ranking violation),
    and measure its distance from `unobserved_rating_value`.
    """

    negative_items = torch.randint(
        0,
        model.num_items,
        (len(batch_users), num_sampled_negative_examples),
    )
    negative_users = batch_users.unsqueeze(1).expand(-1, num_sampled_negative_examples)
    hardest_negative = model(negative_users, negative_items).amax(dim=-1)
    unobserved_target = torch.full_like(hardest_negative, unobserved_rating_value)
    return nn.functional.mse_loss(hardest_negative, unobserved_target)


def train(  # noqa: PLR0913
    ratings: pl.DataFrame,
    *,
    user_id_key: str = DEFAULT_USER_ID_KEY,
    game_id_key: str = DEFAULT_GAME_ID_KEY,
    ratings_key: str = DEFAULT_RATINGS_KEY,
    num_factors: int = 32,
    num_epochs: int = 20,
    batch_size: int = 1 << 16,
    learning_rate: float = 1e-3,
    regularization: float = 1e-9,
    linear_regularization: float = 1e-9,
    ranking_regularization: float = 0.25,
    unobserved_rating_value: float | None = None,
    num_sampled_negative_examples: int = 4,
    seed: int | None = None,
) -> TrainingResult:
    """
    Train a `CollaborativeFilteringModel` on mean squared error plus the
    ranking objective from Turi Create's RankingFactorizationRecommender.

    Dense minibatches, shuffled each epoch, `Adam`. Defaults match Turi's.
    For each training row, `num_sampled_negative_examples` random items are
    scored for that row's user; the highest-scoring one, the worst ranking
    violation, is pushed towards `unobserved_rating_value`. `regularization`
    and `linear_regularization` are standard L2 penalties on the factors and
    biases touched in each batch. Set `ranking_regularization=0` to disable
    the ranking term.
    """

    if seed is not None:
        torch.manual_seed(seed)

    ratings = ratings.filter(pl.col(ratings_key).is_not_null())

    user_labels = _numpy_safe(
        ratings[user_id_key].unique(maintain_order=True).to_numpy(),
    )
    item_labels = _numpy_safe(
        ratings[game_id_key].unique(maintain_order=True).to_numpy(),
    )
    user_index = {label: index for index, label in enumerate(user_labels)}
    item_index = {label: index for index, label in enumerate(item_labels)}

    indexed = ratings.select(
        pl.col(user_id_key)
        .replace_strict(user_index, return_dtype=pl.Int64)
        .alias("user"),
        pl.col(game_id_key)
        .replace_strict(item_index, return_dtype=pl.Int64)
        .alias("item"),
        pl.col(ratings_key).cast(pl.Float32).alias("rating"),
    )

    # polars may hand back a read-only view; torch.from_numpy on one of those is
    # undefined behaviour, so copy explicitly rather than suppress the warning.
    users = torch.from_numpy(indexed["user"].to_numpy().copy())
    items = torch.from_numpy(indexed["item"].to_numpy().copy())
    target = torch.from_numpy(indexed["rating"].to_numpy().copy())

    model = CollaborativeFilteringModel(
        num_users=len(user_labels),
        num_items=len(item_labels),
        num_factors=num_factors,
    )

    # Adam's step size is roughly bounded by the learning rate regardless of
    # gradient magnitude, so leaving these at 0 would take far more epochs
    # than is practical to reach a realistic rating scale. Seed them with the
    # standard baseline predictor instead: global mean, then each user's and
    # item's average deviation from it.
    global_mean = float(target.mean())
    user_means = indexed.group_by("user").agg(pl.col("rating").mean()).sort("user")
    item_means = indexed.group_by("item").agg(pl.col("rating").mean()).sort("item")
    user_bias_init = user_means["rating"].to_numpy() - global_mean
    item_bias_init = item_means["rating"].to_numpy() - global_mean

    model.intercept.data.fill_(global_mean)
    model.user_biases.weight.data.copy_(
        torch.from_numpy(user_bias_init.astype("float32")).reshape(-1, 1),
    )
    model.item_biases.weight.data.copy_(
        torch.from_numpy(item_bias_init.astype("float32")).reshape(-1, 1),
    )

    # Turi's own default: an unrated item should be pushed towards the low
    # end of the rating scale, not towards 0.
    if unobserved_rating_value is None:
        unobserved_rating_value = global_mean - 1.96 * float(target.std())
        LOGGER.info("Estimated unobserved_rating_value: %.4f", unobserved_rating_value)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    num_rows = len(users)
    for epoch in range(num_epochs):
        permutation = torch.randperm(num_rows)
        epoch_loss = 0.0
        for start in range(0, num_rows, batch_size):
            batch = permutation[start : start + batch_size]
            batch_users = users[batch]
            batch_items = items[batch]

            optimizer.zero_grad()
            prediction = model(batch_users, batch_items)
            loss = nn.functional.mse_loss(prediction, target[batch])

            if regularization:
                loss = loss + regularization * (
                    model.user_factors(batch_users).pow(2).sum()
                    + model.item_factors(batch_items).pow(2).sum()
                )
            if linear_regularization:
                loss = loss + linear_regularization * (
                    model.user_biases(batch_users).pow(2).sum()
                    + model.item_biases(batch_items).pow(2).sum()
                )
            if ranking_regularization:
                loss = loss + ranking_regularization * _ranking_loss(
                    model,
                    batch_users,
                    num_sampled_negative_examples,
                    unobserved_rating_value,
                )

            loss.backward()
            optimizer.step()
            epoch_loss += float(loss) * len(batch)
        LOGGER.info(
            "Epoch %d/%d: loss %.4f",
            epoch + 1,
            num_epochs,
            epoch_loss / num_rows,
        )

    return TrainingResult(model=model, user_labels=user_labels, item_labels=item_labels)
