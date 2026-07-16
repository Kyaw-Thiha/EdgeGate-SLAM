from __future__ import annotations
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class EdgeTypeAwareConv(MessagePassing):
    """Edge-type-aware message passing layer."""

    def __init__(self, in_channels: int, out_channels: int, num_edge_types: int = 2):
        super().__init__(aggr="add")
        raise NotImplementedError

    def forward(self, x, edge_index, edge_attr, edge_type):
        raise NotImplementedError

    def message(self, x_j, edge_attr, edge_type):
        raise NotImplementedError
