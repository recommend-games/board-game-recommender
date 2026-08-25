"""Collaborative filtering model implemented in PyTorch."""

from __future__ import annotations

import torch  # type: ignore[import-not-found]
from torch import nn  # type: ignore[import-not-found]


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
