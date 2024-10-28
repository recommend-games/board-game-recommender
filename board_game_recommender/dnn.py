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
        ratings_mean: Optional[float] = None,
        user_ratings_mean: Optional[Iterable[float]] = None,
        game_ratings_mean: Optional[Iterable[float]] = None,
        regularization: Optional[float] = None,  # 1e-8
        linear_regularization: Optional[float] = None,  # 1e-10
        ranking_regularization: Optional[float] = None,  # 0.25
        unobserved_rating_value: Optional[float] = None,  # mean - 1.96*std_dev
        num_sampled_negative_examples: Optional[int] = None,  # 4
        learning_rate: float = 1e-3,
    ):
        super().__init__()

        self.automatic_optimization = False

        self.users = np.array(list(users), dtype=np.str_)
        self.user_ids = {user: i for i, user in enumerate(self.users)}

        self.user_ratings_mean = (
            np.array(list(user_ratings_mean), dtype=np.float32)
            if user_ratings_mean is not None
            else None
        )
        assert self.user_ratings_mean is None or len(self.user_ratings_mean) == len(
            self.users
        ), "User ratings mean must have the same length as the number of users"

        self.games = np.array(list(games), dtype=np.int32)
        self.game_ids = {game: i for i, game in enumerate(self.games)}

        self.game_ratings_mean = (
            np.array(list(game_ratings_mean), dtype=np.float32)
            if game_ratings_mean is not None
            else None
        )
        assert self.game_ratings_mean is None or len(self.game_ratings_mean) == len(
            self.games
        ), "Game ratings mean must have the same length as the number of games"

        self.ratings_mean = ratings_mean

        assert embedding_dim > 0, "Embedding dimension must be positive"
        assert (
            regularization is None or regularization > 0
        ), "Regularization must be positive"
        assert (
            linear_regularization is None or linear_regularization > 0
        ), "Linear regularization must be positive"
        assert (
            ranking_regularization is None or ranking_regularization > 0
        ), "Ranking regularization must be positive"
        assert (
            unobserved_rating_value is None or unobserved_rating_value > 0
        ), "Unobserved rating value must be positive"
        assert (
            num_sampled_negative_examples is None or num_sampled_negative_examples > 0
        ), "Number of sampled negative examples must be positive"
        assert learning_rate > 0, "Learning rate must be positive"

        self.embedding_dim = embedding_dim
        self.regularization = regularization
        self.linear_regularization = linear_regularization
        self.ranking_regularization = ranking_regularization
        self.unobserved_rating_value = unobserved_rating_value
        self.num_sampled_negative_examples = num_sampled_negative_examples
        self.learning_rate = learning_rate

        self.user_embedding = nn.Embedding(len(self.users), embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)

        if self.user_ratings_mean is None:
            self.user_biases = nn.Parameter(torch.zeros(len(self.users)))
            nn.init.normal_(self.user_biases, std=0.01)
        else:
            self.user_biases = nn.Parameter(torch.tensor(self.user_ratings_mean))

        self.game_embedding = nn.Embedding(len(self.games), embedding_dim)
        nn.init.normal_(self.game_embedding.weight, std=0.1)

        if self.game_ratings_mean is None:
            self.game_biases = nn.Parameter(torch.zeros(len(self.games)))
            nn.init.normal_(self.game_biases, std=0.01)
        else:
            self.game_biases = nn.Parameter(torch.tensor(self.game_ratings_mean))

        intercept = self.ratings_mean if self.ratings_mean is not None else 0.0
        self.intercept = nn.Parameter(torch.tensor(intercept))

        self.train_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.val_rmse = torchmetrics.MeanSquaredError(squared=False)

        self.save_hyperparameters(
            ignore=(
                "users",
                "user_ratings_mean",
                "games",
                "game_ratings_mean",
            ),
        )

    def loss_fn(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = nn.functional.mse_loss(prediction, target)

        # Regularized loss function from Turicreate's FactorizationRecommender:
        # https://apple.github.io/turicreate/docs/api/generated/turicreate.recommender.factorization_recommender.FactorizationRecommender.html
        if self.regularization:
            # TODO: Only regularize the embeddings of the observed users and items
            user_embedding = self.user_embedding.weight
            game_embedding = self.game_embedding.weight
            loss += self.regularization * (
                torch.sum(user_embedding**2) + torch.sum(game_embedding**2)
            )

        if self.linear_regularization:
            # TODO: Only regularize the biases of the observed users and items
            user_bias = self.user_biases
            game_bias = self.game_biases
            loss += self.linear_regularization * (
                torch.sum(user_bias**2) + torch.sum(game_bias**2)
            )

        # Ranking regularization from Turicreate's RankingFactorizationRecommender:
        # https://apple.github.io/turicreate/docs/api/generated/turicreate.recommender.ranking_factorization_recommender.RankingFactorizationRecommender.html
        if (
            self.ranking_regularization
            and self.unobserved_rating_value
            and self.num_sampled_negative_examples
        ):
            # For each user–item pair in the training data, we sample a number of negative examples
            # TODO: Make sure those pairs are not in the training data
            users_sample = torch.randint(
                low=0,
                high=len(self.users),
                size=(len(target), 1),
                device=self.device,
            ).expand(-1, self.num_sampled_negative_examples)
            games_sample = torch.randint(
                low=0,
                high=len(self.games),
                size=(len(target), self.num_sampled_negative_examples),
                device=self.device,
            )
            unobserved_predictions = self(users_sample, games_sample)
            unobserved_predictions_max, _ = torch.max(unobserved_predictions, dim=-1)
            unobserved_targets = (
                torch.ones_like(unobserved_predictions_max, device=self.device)
                * self.unobserved_rating_value
            )
            loss += self.ranking_regularization * nn.functional.mse_loss(
                unobserved_predictions_max,
                unobserved_targets,
            )

        return loss

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        assert user.shape == item.shape
        user_embedded = self.user_embedding(user)  # (num_input, embedding_dim)
        user_bias = self.user_biases[user]  # (num_input,)
        game_embedded = self.game_embedding(item)  # (num_input, embedding_dim)
        game_bias = self.game_biases[item]  # (num_input,)
        dot_product = torch.sum(user_embedded * game_embedded, dim=-1)  # (num_input,)
        return dot_product + user_bias + game_bias + self.intercept  # (num_input,)

    @torch.no_grad()
    def recommend(self, user: str, n: int = 10) -> np.ndarray:
        user_id = self.user_ids[user]
        num_games = len(self.games)
        user_tensor = torch.tensor([user_id] * num_games, dtype=torch.int64).to(
            self.device
        )
        game_tensor = torch.arange(num_games).to(self.device)
        predictions = self(user_tensor, game_tensor)
        top_n = torch.topk(predictions, n)
        return self.games[top_n.indices.cpu().numpy()]

    def training_step(self, batch: torch.Tensor, batch_idx: int = 0) -> torch.Tensor:
        user, item, target = batch
        prediction = self(user, item)

        loss = self.loss_fn(prediction, target)
        self.log("train_loss", loss, prog_bar=True)

        self.train_rmse(prediction, target)
        self.log("train_rmse", self.train_rmse, prog_bar=True)

        user_optimizer, item_optimizer = self.optimizers()

        # Alternate between optimizing the user and item embeddings
        if batch_idx % 2 == 0:
            assert isinstance(user_optimizer, optim.Optimizer)
            user_optimizer.zero_grad()
            self.manual_backward(loss)
            user_optimizer.step()
        else:
            assert isinstance(item_optimizer, optim.Optimizer)
            item_optimizer.zero_grad()
            self.manual_backward(loss)
            item_optimizer.step()

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int = 0) -> torch.Tensor:
        user, item, target = batch
        prediction = self(user, item)
        loss = self.loss_fn(prediction, target)
        self.log("val_loss", loss)
        self.val_rmse(prediction, target)
        self.log("val_rmse", self.val_rmse)
        return loss

    def configure_optimizers(self) -> Tuple[optim.Optimizer, optim.Optimizer]:
        user_optimizer = optim.Adam(
            params=[
                self.user_embedding.weight,
                self.user_biases,
                self.intercept,
            ],
            lr=self.learning_rate,
        )
        game_optimizer = optim.Adam(
            params=[
                self.game_embedding.weight,
                self.game_biases,
                self.intercept,
            ],
            lr=self.learning_rate,
        )
        return user_optimizer, game_optimizer


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
        # user_ratings_mean=TODO,
        games=games,
        # game_ratings_mean=TODO,
        ratings_mean=ratings["bgg_user_rating"].mean(),
        embedding_dim=32,
        learning_rate=1e-3,
        # regularization=1e-8,
        # linear_regularization=1e-10,
        # ranking_regularization=0.25,
        # unobserved_rating_value=ratings["bgg_user_rating"].quantile(0.05),
        # num_sampled_negative_examples=4,
    )

    user_ids_array = ratings["user_id"].to_numpy(writable=True)
    user_ids_tensor = torch.from_numpy(user_ids_array)
    game_ids_array = ratings["game_id"].to_numpy(writable=True)
    game_ids_tensor = torch.from_numpy(game_ids_array)
    ratings_array = ratings["bgg_user_rating"].to_numpy(writable=True)
    ratings_tensor = torch.from_numpy(ratings_array)

    num_cpus = os.cpu_count() or 1
    # TODO: Train/test/val split
    dataset = TensorDataset(user_ids_tensor, game_ids_tensor, ratings_tensor)
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=4 if num_cpus > 1 else 0,
        persistent_workers=num_cpus > 1,
        shuffle=True,
    )
    val_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=4 if num_cpus > 1 else 0,
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
