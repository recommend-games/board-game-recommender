import logging
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple, Type

import lightning
import numpy as np
import polars as pl
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

LOGGER = logging.getLogger(__name__)


class CollaborativeFilteringModel(lightning.LightningModule):
    @classmethod
    def load_from_dir(
        cls,
        save_dir: os.PathLike,
        checkpoint_file: os.PathLike,
        items_file: os.PathLike = "items.npz",
    ) -> "CollaborativeFilteringModel":
        save_dir = Path(save_dir).resolve()
        LOGGER.info("Loading model from <%s>", save_dir)

        items_path = save_dir / items_file
        LOGGER.info("Loading items from <%s>", items_path)
        with np.load(items_path, allow_pickle=True) as items:
            users = items["users"]
            games = items["games"]

        checkpoint_path = save_dir / checkpoint_file
        LOGGER.info("Loading checkpoint from <%s>", checkpoint_path)

        return cls.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            users=users,
            games=games,
        )

    def __init__(
        self,
        *,
        users: Iterable[str],
        games: Iterable[int],
        embedding_dim: int = 32,
        learning_rate: float = 1e-3,
    ):
        super().__init__()

        self.users = np.array(list(users), dtype=np.str_)
        self.user_ids = {user: i for i, user in enumerate(self.users)}
        self.games = np.array(list(games), dtype=np.int32)
        self.game_ids = {game: i for i, game in enumerate(self.games)}

        self.user_embedding = nn.Embedding(len(self.users), embedding_dim)
        self.user_biases = nn.Parameter(torch.rand(len(self.users)))
        self.game_embedding = nn.Embedding(len(self.games), embedding_dim)
        self.game_biases = nn.Parameter(torch.rand(len(self.games)))
        self.intercept = nn.Parameter(torch.rand(1))
        self.loss_fn = nn.MSELoss()

        self.learning_rate = learning_rate

        self.save_hyperparameters(ignore=("users", "user_ids", "games", "game_ids"))

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        assert user.shape == item.shape
        user_embedded = self.user_embedding(user)  # (num_input, embedding_dim)
        user_bias = self.user_biases[user]  # (num_input,)
        game_embedded = self.game_embedding(item)  # (num_input, embedding_dim)
        game_bias = self.game_biases[item]  # (num_input,)
        dot_product = torch.sum(user_embedded * game_embedded, dim=-1)  # (num_input,)
        return dot_product + user_bias + game_bias + self.intercept  # (num_input,)

    def recommend(self, user: str, n: int = 10) -> np.ndarray:
        user_id = self.user_ids[user]
        user_tensor = torch.tensor([user_id])
        game_tensor = torch.arange(len(self.games))
        predictions = self(user_tensor, game_tensor)
        top_n = torch.topk(predictions, n)
        return self.games[top_n.indices.numpy()]

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
) -> Tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    ratings = load_jl(
        path=ratings_path,
        schema={
            "bgg_user_name": pl.Utf8,
            "bgg_id": pl.Int32,
            "bgg_user_rating": pl.Float32,
        },
    )
    ratings = ratings.drop_nulls()

    users = ratings["bgg_user_name"].unique()
    user_ids = {user: i for i, user in enumerate(users)}

    games = ratings["bgg_id"].unique()
    game_ids = {game: i for i, game in enumerate(games)}

    ratings = ratings.with_columns(
        user_id=ratings["bgg_user_name"].replace(user_ids, return_dtype=pl.Int32),
        game_id=ratings["bgg_id"].replace(game_ids, return_dtype=pl.Int32),
    )

    return ratings, users.to_numpy(), games.to_numpy()


def train_model(
    *,
    ratings_path: os.PathLike,
    max_epochs: int = 10,
    batch_size: int = 1024,
    save_dir: os.PathLike = ".",
    fast_dev_run: bool = False,
) -> CollaborativeFilteringModel:
    ratings, users, games = load_data(ratings_path)

    model = CollaborativeFilteringModel(
        users=users,
        games=games,
        embedding_dim=32,
        learning_rate=1e-3,
    )

    user_ids_array = ratings["user_id"].to_numpy(writable=True)
    user_ids_tensor = torch.from_numpy(user_ids_array)
    game_ids_array = ratings["game_id"].to_numpy(writable=True)
    game_ids_tensor = torch.from_numpy(game_ids_array)
    ratings_array = ratings["bgg_user_rating"].to_numpy(writable=True)
    ratings_tensor = torch.from_numpy(ratings_array)

    num_cpus = os.cpu_count() or 1
    dataset = TensorDataset(user_ids_tensor, game_ids_tensor, ratings_tensor)
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_cpus - 1,
        persistent_workers=num_cpus > 1,
        shuffle=True,
    )
    val_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_cpus - 1,
        persistent_workers=num_cpus > 1,
        shuffle=False,
    )

    save_dir = Path(save_dir).resolve()
    LOGGER.info("Saving items to <%s>", save_dir)

    # TODO: Add early stopping and increase max_epochs

    checkpoint_callback = lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    csv_logger = lightning.pytorch.loggers.csv_logs.CSVLogger(
        save_dir=save_dir,
    )

    trainer = lightning.Trainer(
        max_epochs=max_epochs,
        logger=[csv_logger],
        callbacks=[checkpoint_callback],
        default_root_dir=save_dir,
        fast_dev_run=fast_dev_run,
    )

    trainer.fit(model, train_loader, val_loader)

    # TODO: Save best model (or create `best.ckpt` symlink to best model)
    best_model_path = Path(checkpoint_callback.best_model_path).resolve()
    items_path = best_model_path.parent / "items.npz"
    LOGGER.info("Saving items to <%s>", items_path)
    np.savez(
        file=items_path,
        users=users,
        games=games,
    )

    return model


def _main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )

    data_dir = Path(__file__).parent.parent.parent / "board-game-data"
    ratings_path = data_dir / "scraped" / "bgg_RatingItem.jl"

    train_model(
        ratings_path=ratings_path,
        max_epochs=10,
        batch_size=1024,
        save_dir=".",
        fast_dev_run=False,
    )


if __name__ == "__main__":
    _main()
