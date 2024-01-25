import logging
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Type

import lightning
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

LOGGER = logging.getLogger(__name__)


class CollaborativeFilteringModel(lightning.LightningModule):
    def __init__(
        self,
        *,
        num_users: int,
        num_items: int,
        embedding_dim: int = 32,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.linear = nn.Linear(embedding_dim, 1)
        self.loss_fn = nn.MSELoss()
        self.learning_rate = learning_rate
        self.double()
        self.save_hyperparameters()

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        user_embedded = self.user_embedding(user)
        item_embedded = self.item_embedding(item)
        product = user_embedded * item_embedded
        return self.linear(product).squeeze()

    def training_step(self, batch: torch.Tensor, batch_idx: int = 0) -> torch.Tensor:
        user, item, target = batch
        prediction = self(user, item)
        loss = self.loss_fn(prediction, target)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int = 0) -> torch.Tensor:
        user, item, target = batch
        prediction = self(user, item)
        loss = self.loss_fn(prediction, target)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)


def load_jl(path: os.PathLike, schema: Dict[str, Type[pl.DataType]]) -> pl.DataFrame:
    path = Path(path).resolve()
    LOGGER.info("Loading %s", path)
    return pl.read_ndjson(path, schema=schema)


def load_data(
    ratings_path: os.PathLike,
) -> Tuple[pl.DataFrame, pl.Series, Dict[str, int], pl.Series, Dict[int, int]]:
    ratings = load_jl(
        path=ratings_path,
        schema={
            "bgg_user_name": pl.Utf8,
            "bgg_id": pl.Int64,
            "bgg_user_rating": pl.Float64,
        },
    )
    ratings = ratings.drop_nulls()

    users = ratings["bgg_user_name"].unique()
    user_ids = {user: i for i, user in enumerate(users)}

    games = ratings["bgg_id"].unique()
    game_ids = {game: i for i, game in enumerate(games)}

    ratings = ratings.with_columns(
        user_id=ratings["bgg_user_name"].replace(user_ids, return_dtype=pl.Int64),
        game_id=ratings["bgg_id"].replace(game_ids, return_dtype=pl.Int64),
    )

    return ratings, users, user_ids, games, game_ids


def train_model(
    *,
    ratings_path: os.PathLike,
    max_epochs: int = 10,
    batch_size: int = 1024,
) -> CollaborativeFilteringModel:
    ratings, users, user_ids, games, game_ids = load_data(ratings_path)

    model = CollaborativeFilteringModel(
        num_users=len(users),
        num_items=len(games),
        embedding_dim=32,
        learning_rate=1e-3,
    )

    user_ids_array = ratings["user_id"].to_numpy()
    user_ids_tensor = torch.from_numpy(user_ids_array)
    game_ids_array = ratings["game_id"].to_numpy()
    game_ids_tensor = torch.from_numpy(game_ids_array)
    ratings_array = ratings["bgg_user_rating"].to_numpy()
    ratings_tensor = torch.from_numpy(ratings_array)

    dataset = TensorDataset(user_ids_tensor, game_ids_tensor, ratings_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    trainer = lightning.Trainer(max_epochs=max_epochs)
    trainer.fit(model, train_loader, val_loader)

    return model


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )

    data_dir = Path(__file__).parent.parent.parent / "board-game-data"
    ratings_path = data_dir / "scraped" / "bgg_RatingItem.jl"

    train_model(ratings_path=ratings_path)
