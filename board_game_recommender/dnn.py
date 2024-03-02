import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Type, Union

import lightning
import numpy as np
import polars as pl
import torch
import torchmetrics
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

LOGGER = logging.getLogger(__name__)

PATH_OR_STR = Union[os.PathLike, str]

BASE_DIR = Path(__file__).parent.parent.resolve()


class CollaborativeFilteringModel(lightning.LightningModule):
    @classmethod
    def load_from_dir(
        cls,
        save_dir: PATH_OR_STR,
        checkpoint_file: PATH_OR_STR,
        items_file: PATH_OR_STR = "items.npz",
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
        regularization: Optional[float] = None,  # 1e-8
        linear_regularization: Optional[float] = None,  # 1e-10
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

        self.regularization = regularization
        self.linear_regularization = linear_regularization

        self.learning_rate = learning_rate

        self.train_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.val_rmse = torchmetrics.MeanSquaredError(squared=False)

        self.save_hyperparameters(ignore=("users", "user_ids", "games", "game_ids"))

    # Regularized loss function from Turicreate's FactorizationRecommender:
    # https://apple.github.io/turicreate/docs/api/generated/turicreate.recommender.factorization_recommender.FactorizationRecommender.html
    def loss_fn(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.regularization and not self.linear_regularization:
            return nn.functional.mse_loss(prediction, target)

        loss = (prediction - target) ** 2

        if self.regularization:
            user_embedding = self.user_embedding.weight
            game_embedding = self.game_embedding.weight
            loss += self.regularization * (
                torch.sum(user_embedding**2) + torch.sum(game_embedding**2)
            )

        if self.linear_regularization:
            user_bias = self.user_biases
            game_bias = self.game_biases
            loss += self.linear_regularization * (
                torch.sum(user_bias**2) + torch.sum(game_bias**2)
            )

        return torch.mean(loss)

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
        self.train_rmse(prediction, target)
        self.log("train_rmse", self.train_rmse, prog_bar=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int = 0) -> torch.Tensor:
        user, item, target = batch
        prediction = self(user, item)
        loss = self.loss_fn(prediction, target)
        self.log("val_loss", loss)
        self.val_rmse(prediction, target)
        self.log("val_rmse", self.val_rmse)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)


def load_jl(path: PATH_OR_STR, schema: Dict[str, Type[pl.DataType]]) -> pl.DataFrame:
    path = Path(path).resolve()
    LOGGER.info("Loading %s", path)
    return pl.read_ndjson(path, schema=schema)


def load_data(
    ratings_path: PATH_OR_STR,
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
    ratings_path: PATH_OR_STR,
    max_epochs: int = 100,
    batch_size: int = 1024,
    save_dir: PATH_OR_STR = ".",
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

    num_cpus = 1  # os.cpu_count() or 1
    # TODO: Train/test/val split
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

    checkpoint_callback = lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    early_stopping_callback = lightning.pytorch.callbacks.early_stopping.EarlyStopping(
        monitor="val_loss",
        mode="min",
        min_delta=0.0,
        patience=3,
        verbose=True,
    )

    csv_logger = lightning.pytorch.loggers.csv_logs.CSVLogger(
        save_dir=save_dir,
    )

    trainer = lightning.Trainer(
        max_epochs=max_epochs,
        logger=[csv_logger],
        callbacks=[checkpoint_callback, early_stopping_callback],
        default_root_dir=save_dir,
        fast_dev_run=fast_dev_run,
    )

    trainer.fit(model, train_loader, val_loader)

    best_model_path = Path(checkpoint_callback.best_model_path).resolve()
    best_model_alias = best_model_path.parent / "best.ckpt"
    LOGGER.info("Linking best model <%s> to <%s>", best_model_path, best_model_alias)
    best_model_alias.symlink_to(best_model_path)

    items_path = best_model_path.parent / "items.npz"
    LOGGER.info("Saving items to <%s>", items_path)
    np.savez(
        file=items_path,
        users=users,
        games=games,
    )

    return model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a collaborative filtering model",
    )

    parser.add_argument(
        "--ratings-path",
        type=Path,
        default=BASE_DIR.parent / "board-game-data" / "scraped" / "bgg_RatingItem.jl",
        help="Path to the ratings data",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=100,
        help="Maximum number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=BASE_DIR,
        help="Directory to save the model",
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run a fast development run",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase verbosity",
    )

    return parser.parse_args()


def _main():
    args = _parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose > 0 else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )

    LOGGER.info(args)

    train_model(
        ratings_path=args.ratings_path,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        fast_dev_run=args.fast_dev_run,
    )


if __name__ == "__main__":
    _main()
