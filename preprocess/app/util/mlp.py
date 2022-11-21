import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
        self, n_features: int, dropout: bool = True, bias: bool = True
    ) -> None:
        assert len(n_features) >= 2
        super(MLP, self).__init__()
        self.linears = nn.ModuleList(
            [
                nn.Linear(n_features[i], n_features[i + 1], bias=bias)
                for i in range(len(n_features) - 1)
            ]
        )
        self.dropout = dropout
        self.fine_tune = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for linear in self.linears[:-1]:
            h = F.relu(linear(h))
            if self.dropout:
                h = F.dropout(h)
        if self.fine_tune:
            h = h.detach()
        h = self.linears[-1](h)
        return h


def get_mlp(
    in_features: int,
    n_units: int,
    out_features: int,
    n_layers: int,
    dropout: bool = True,
    bias: bool = True,
) -> nn.Module:
    units = [in_features] + [n_units] * n_layers + [out_features]
    return MLP(units, dropout, bias)
