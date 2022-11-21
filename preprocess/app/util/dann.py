import torch
import torch.nn as nn

from .gradient_reversal import GradientReversal
from .mlp import get_mlp


class DANN(nn.Module):
    def __init__(
        self,
        encoder1: nn.Module,
        encoder2: nn.Module,
        predictor: nn.Module,
        discriminator: nn.Module,
        scale: float,
        discriminator_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.predictor = predictor
        self.discriminator = discriminator
        self.g_reversal = GradientReversal(scale)
        self.discriminator_loss_weight = discriminator_loss_weight

    def encode(self, X: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        e1 = self.encoder1(X)
        e2 = self.encoder2(X)
        e = z[:, None] * e1 + (1 - z[:, None]) * e2
        return e

    def __call__(
        self, X: torch.Tensor, z: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        e = self.encode(X, z)
        p_pred = self.predictor(e)
        loss_supervised = nn.CrossEntropyLoss()(p_pred, y)
        e_g_reversed = self.g_reversal(e)
        z_pred = self.discriminator(e_g_reversed)
        loss_unsupervised = nn.BCEWithLogitsLoss()(z_pred.squeeze(), z)
        self.loss_supervised = loss_supervised
        self.loss_unsupervised = loss_unsupervised
        loss = loss_supervised + self.discriminator_loss_weight * loss_unsupervised
        return loss

    def count_correct(
        self, X: torch.Tensor, z: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        y_pred = self.predict(X, z)
        return torch.sum(y_pred == y)

    def predict(self, X: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        self.eval()
        e = self.encode(X, z)
        y_pred = self.predictor(e)
        return torch.argmax(y_pred, dim=1)

    def predict_proba(self, X: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        self.eval()
        e = self.encode(X, z)
        y_pred = self.predictor(e)
        return torch.softmax(self(y_pred), dim=1)


def get_dann(
    D: int,
    K: int,
    d_hidden: int,
    d_latent: int,
    n_layers: int,
    discriminator_loss_weight: float,
    scale: float = 0.1,
) -> nn.Module:
    encoder1 = get_mlp(D, d_hidden, d_latent, n_layers)
    encoder2 = get_mlp(D, d_hidden, d_latent, n_layers)
    discriminator = get_mlp(d_latent, d_hidden, 1, n_layers)
    predictor = get_mlp(d_latent, d_hidden, K, n_layers)
    model = DANN(
        encoder1, encoder2, predictor, discriminator, scale, discriminator_loss_weight
    )
    return model
