from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GraphNorm
from edgegate.models.layers import EdgeTypeAwareConv

# ── Constants matching graph_builder.py's to_pyg() output ────────────────────
_NODE_FEAT_DIM = 3  # [x, y, θ] initial pose guess
_EDGE_ATTR_DIM = 6  # [dx, dy, dθ, Ixx, Iyy, Iθθ]
_NUM_EDGE_TYPES = 2  # 0 = odometry, 1 = loop-closure


class EdgeGateGNN(nn.Module):
    """GNN confidence scorer for SE(2) pose-graph outlier rejection.

    Computes a per-edge inlier confidence score in [0, 1] for every edge in a
    pose graph. These scores are passed to a solver via scale_information() in
    solvers/base.py, which scales each edge's information matrix by w², making
    low-confidence edges contribute weakly to the optimisation objective.

    Architecture:

        1. Input projection
           Linear(3, hidden_dim) + ReLU
           Maps initial pose guesses [x, y, θ] into the hidden embedding space.

        2. L × EdgeTypeAwareConv layers  (message passing)
           Each layer: m_{i→j} = W_{type} · concat(x_j, edge_attr), aggregated by sum.
           Layers 1..L-1 wrap the conv in a residual block with dropout + GraphNorm:
               h = GraphNorm(h + Dropout(Conv(h)))
           Layer 0 has no residual (dimensions may differ if in_channels ≠ hidden_dim,
           though here they always match after the input projection).

           GraphNorm rather than BatchNorm is used because we normalise per graph, not
           per mini-batch element — needed when batching pose graphs of different sizes.
           LayerNorm is an acceptable fallback if GraphNorm proves unstable (GNN.md §3).

        3. Confidence head  (per-edge scoring)
           head_input = concat(h_i, h_j, edge_attr_embed, onehot(edge_type))
           confidence = sigmoid(MLP(head_input))   # MLP: Linear → ReLU → Dropout → Linear

           Why concat(h_i, h_j) in directed order, not symmetric (h_i + h_j)?
               The measurement itself is directional (i→j), so h_i and h_j are NOT
               interchangeable. Symmetric pooling would lose that convention.

           Why re-inject edge_attr_embed after L rounds of MP?
               Node aggregation across L layers can blur the raw measurement signal
               (e.g. information-matrix magnitude). Re-injecting it gives the head a
               direct, unblurred path to the local edge features.

           Why re-inject onehot(edge_type)?
               After L rounds of propagation, edge_type has acted only as a routing
               signal (which W_t was applied). It isn't reliably recoverable as a
               feature value from h_i, h_j alone. Making it explicit costs nothing
               and removes a potential silent failure mode.

    Planned ablations (do NOT modify this class — add new classes in layers.py):
        A. Replace EdgeTypeAwareConv with ECC/NNConv-style continuous edge-conditioning.
        B. Swap aggr="add" for attention-based aggregation (TransformerConv).
        C. GRU over the odometry chain with loop-closures as cross-links.
        D. Mutable edge-state MPNN across layers.
    See docs/GNN.md §4 for ordering and rationale.

    Args:
        num_layers: Number of EdgeTypeAwareConv layers (default 3 per GNN.md §3).
        hidden_dim: Width of node/edge embeddings throughout (default 64).
        dropout: Dropout probability, applied in residual blocks and the head MLP
                 (default 0.1; valid range 0.1–0.2 per GNN.md §3).
    """

    def __init__(
        self,
        num_layers: int = 3,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # ── Input projections ─────────────────────────────────────────────────
        # Project initial pose guesses into the hidden embedding space.
        self.input_proj = nn.Linear(_NODE_FEAT_DIM, hidden_dim)

        # Separate projection for edge_attr used in the confidence head.
        # Kept distinct from the message-function path so the head can learn a
        # different representation of the same edge features.
        self.edge_proj = nn.Linear(_EDGE_ATTR_DIM, hidden_dim)

        # ── Message-passing stack ─────────────────────────────────────────────
        self.convs = nn.ModuleList(
            [EdgeTypeAwareConv(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

        # One norm per residual block (layers 1..L-1); none for layer 0.
        self.norms = nn.ModuleList(
            [GraphNorm(hidden_dim) for _ in range(max(num_layers - 1, 0))]
        )

        self.dropout = nn.Dropout(dropout)

        # ── Confidence head ───────────────────────────────────────────────────
        # Input dim breakdown: h_i (hidden_dim) + h_j (hidden_dim)
        #                    + ea_emb (hidden_dim) + onehot(_NUM_EDGE_TYPES)
        head_in_dim = 3 * hidden_dim + _NUM_EDGE_TYPES
        self.head = nn.Sequential(
            nn.Linear(head_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # output in [0, 1] — project-wide confidence convention
        )

    def forward(self, data: Data) -> torch.Tensor:
        """Compute per-edge confidence scores.

        Args:
            data: PyG Data object produced by graph_builder.to_pyg(). Expected
                  attributes:
                    - x          (N, 3)  initial pose guesses [x, y, θ]
                    - edge_index (2, E)  directed edges [src; dst]
                    - edge_attr  (E, 6)  [dx, dy, dθ, Ixx, Iyy, Iθθ]
                    - edge_type  (E,)    integer 0=odom / 1=loop-closure
                    - batch      (N,)    optional batch vector (set by PyG Batch)

        Returns:
            Confidence scores, shape (E,), dtype float32, values in [0, 1].
            A score near 1 means the GNN believes the edge is an inlier;
            near 0 means likely an outlier.
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        edge_type = data.edge_type
        # batch is None for single-graph inputs; GraphNorm handles both cases.
        batch = getattr(data, "batch", None)

        # ── 1. Initial node projection ────────────────────────────────────────
        h = F.relu(self.input_proj(x))  # (N, hidden_dim)

        # ── 2. Message-passing layers ─────────────────────────────────────────
        for i, conv in enumerate(self.convs):
            h_new = conv(h, edge_index, edge_attr, edge_type)  # (N, hidden_dim)
            if i == 0:
                # First layer: no residual (graph_builder guarantees hidden_dim
                # matches, but conceptually this is the "warm-up" pass).
                h = h_new
            else:
                # Residual block: h = GraphNorm(h + Dropout(Conv(h)))
                # Dropout on the conv output (not the aggregated raw signal) per
                # GNN.md §3 — "never on raw aggregation."
                h = self.norms[i - 1](h + self.dropout(h_new), batch)

        # ── 3. Per-edge confidence head — loop-closure edges only ────────────────
        # Odometry edges are always inliers by construction; running the head on
        # them would silently scale the most reliable edges with an unsupervised,
        # near-random sigmoid output. Hardcode w_odom = 1.0 end-to-end (train and
        # eval) — see GNN.md §2.2 and implementation_details.md §edge_bce.
        E = edge_type.size(0)
        lc_mask = edge_type == 1                              # (E,) bool
        lc_idx = lc_mask.nonzero(as_tuple=True)[0]           # LC edge indices

        # Start from all-ones; index_put fills LC positions with head output.
        # index_put (non-in-place) returns a new tensor — autograd-safe.
        scores = torch.ones(E, dtype=h.dtype, device=h.device)

        if lc_idx.numel() > 0:
            h_i = h[edge_index[0][lc_idx]]                   # (E_lc, hidden_dim)
            h_j = h[edge_index[1][lc_idx]]                   # (E_lc, hidden_dim)
            # Directed concat: h_i before h_j — measurement convention is i→j.
            ea_emb = F.relu(self.edge_proj(edge_attr[lc_idx]))        # (E_lc, hidden_dim)
            type_oh = F.one_hot(edge_type[lc_idx], _NUM_EDGE_TYPES).float()  # (E_lc, 2)

            head_input = torch.cat([h_i, h_j, ea_emb, type_oh], dim=-1)
            lc_scores = self.head(head_input).squeeze(-1)             # (E_lc,)
            scores = scores.index_put((lc_idx,), lc_scores)

        return scores  # (E,): odom positions = 1.0, LC positions ∈ [0, 1]
