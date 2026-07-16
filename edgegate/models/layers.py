from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class EdgeTypeAwareConv(MessagePassing):
    """Single message-passing layer for SE(2) pose graphs with type-specific projections.

    Design rationale:
        Odometry edges and loop-closure edges have structurally different roles —
        odometry is locally reliable but accumulates drift; loop-closures correct
        drift but can be catastrophically wrong (perceptual aliasing). Giving each
        type its own independent weight matrix (W_odom, W_loop) makes this
        distinction architecturally explicit rather than asking a shared MLP to
        learn it from a type embedding.

    Message function:
        m_{i→j} = W_{edge_type} · concat(x_j, edge_attr)
        where x_j are the SOURCE node features (in PyG convention, x_j = features
        of the node sending the message) and edge_attr = [dx, dy, dθ, Ixx, Iyy, Iθθ].

    Aggregation: sum (aggr="add") — standard MPNN choice, Phase 0 baseline.
    Output: ReLU(sum of messages), shape (N, out_channels).

    Edge features are consumed at every layer but never updated (fixed across
    layers per GNN.md §2.4 — "full MPNN with mutable edge state" is a Phase 1+
    ablation).

    Args:
        in_channels: Dimensionality of input node features.
        out_channels: Dimensionality of output node features (= hidden_dim in EdgeGateGNN).
        edge_attr_dim: Dimensionality of edge features (default 6: [dx,dy,dθ,Ixx,Iyy,Iθθ]).
        num_edge_types: Number of distinct edge types (default 2: odom / loop-closure).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_attr_dim: int = 6,
        num_edge_types: int = 2,
    ) -> None:
        super().__init__(aggr="add")
        # One independent linear per edge type: W_t maps concat(x_j, edge_attr) → message.
        # Separate parameters enforce the odom/LC architectural split at the weight level.
        self.type_linears = nn.ModuleList(
            [
                nn.Linear(in_channels + edge_attr_dim, out_channels)
                for _ in range(num_edge_types)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """Run one round of typed message passing.

        Args:
            x: Node features, shape (N, in_channels).
            edge_index: Graph connectivity [src; dst], shape (2, E).
            edge_attr: Edge features [dx,dy,dθ,Ixx,Iyy,Iθθ], shape (E, edge_attr_dim).
            edge_type: Integer edge-type label per edge, shape (E,).
                       0 = odometry, 1 = loop-closure.

        Returns:
            Updated node features after one round of aggregation, shape (N, out_channels).
            ReLU is applied to the aggregated sum before returning.
        """
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_type=edge_type)
        return F.relu(out)

    def message(
        self,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-edge messages using type-specific linear projections.

        PyG passes x_j = source-node features for each edge, aligned with edge_attr
        and edge_type (all shape (E, *)).

        Implementation note — vectorised type dispatch:
            We compute ALL type projections for ALL edges (stack of T linear maps),
            then index-select the correct projection per edge. This avoids an
            in-place scatter (which breaks autograd) while remaining fully vectorised.
            At T=2 this doubles the FLOPs for the linear maps, but pose graphs are
            small and the alternative (masked loop with in-place assignment) is
            gradient-unsafe.

        Args:
            x_j: Source node features for each edge, shape (E, in_channels).
            edge_attr: Edge features, shape (E, edge_attr_dim).
            edge_type: Integer type label per edge, shape (E,).

        Returns:
            Per-edge messages before aggregation, shape (E, out_channels).
        """
        inp = torch.cat([x_j, edge_attr], dim=-1)  # (E, in_channels + edge_attr_dim)

        # Compute all T projections simultaneously, then select by type.
        # msgs[e, t, :] = type_linears[t](inp[e])
        msgs = torch.stack(
            [lin(inp) for lin in self.type_linears], dim=1
        )  # (E, T, out_channels)

        # Pick the projection corresponding to each edge's type.
        idx = torch.arange(msgs.size(0), device=msgs.device)
        return msgs[idx, edge_type]  # (E, out_channels)
