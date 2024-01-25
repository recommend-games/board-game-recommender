import lightning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


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
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)


def train_model(
    *,
    model: CollaborativeFilteringModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    max_epochs: int = 10,
) -> CollaborativeFilteringModel:
    trainer = lightning.Trainer(max_epochs=max_epochs)
    trainer.fit(model, train_loader, val_loader)
    return model
