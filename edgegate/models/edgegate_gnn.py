from __future__ import annotations
import torch
import torch.nn as nn
from torch_geometric.data import Data


class EdgeGateGNN(nn.Module):
    """Stacked EdgeTypeAwareConv layers + per-edge MLP confidence head."""

    def __init__(self, num_layers: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        raise NotImplementedError

    def forward(self, data: Data) -> torch.Tensor:
        """Returns per-edge confidence scores in [0, 1], shape (E,)."""
        raise NotImplementedError
