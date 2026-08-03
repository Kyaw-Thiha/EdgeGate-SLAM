"""Tests for edgegate/models/layers.py and edgegate/models/edgegate_gnn.py.

Coverage strategy:
  - EdgeTypeAwareConv: output shape, type-routing correctness (each W_t is
    actually used for the right edges), gradient flow.
  - EdgeGateGNN: output shape/dtype/range, gradient flow through the full
    forward pass, robustness to num_layers=1 (no residual blocks), and
    correctness on a batched PyG graph (the training-time code path).

All tests use small synthetic graphs so they run fast on CPU.
"""
import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

from edgegate.data.synthetic_generator import generate
from edgegate.data.graph_builder import to_pyg
from edgegate.models.layers import EdgeTypeAwareConv
from edgegate.models.edgegate_gnn import EdgeGateGNN


# ── Shared fixtures ────────────────────────────────────────────────────────────

def _small_pyg_graph(seed: int = 0) -> Data:
    """A small 20-node, ~25-edge synthetic graph as a PyG Data object."""
    g = generate(
        num_poses=20,
        num_loop_closures=4,
        outlier_rate=33,
        outlier_structure="random",
        seed=seed,
        proximity_threshold=5.0,
    )
    return to_pyg(g)


def _minimal_typed_graph() -> Data:
    """Minimal hand-crafted graph with exactly one odom and one LC edge.

    Topology: 4 nodes, 2 disconnected edges.
        edge 0: node 0 → node 1, type=0 (odometry)
        edge 1: node 2 → node 3, type=1 (loop-closure)

    Used by type-routing tests to verify W_odom and W_loop are applied to the
    correct edges without confounding from shared neighbourhood context.
    """
    return Data(
        x=torch.ones(4, 3),
        edge_index=torch.tensor([[0, 2], [1, 3]], dtype=torch.long),
        edge_attr=torch.ones(2, 6),
        edge_type=torch.tensor([0, 1], dtype=torch.long),
    )


# ── EdgeTypeAwareConv ─────────────────────────────────────────────────────────

class TestEdgeTypeAwareConv:

    def test_output_shape(self):
        """Conv output should be (N, out_channels) regardless of edge count."""
        data = _small_pyg_graph()
        N = data.x.size(0)
        conv = EdgeTypeAwareConv(in_channels=3, out_channels=16)

        out = conv(data.x, data.edge_index, data.edge_attr, data.edge_type)

        assert out.shape == (N, 16)

    def test_output_dtype(self):
        data = _small_pyg_graph()
        conv = EdgeTypeAwareConv(in_channels=3, out_channels=16)
        out = conv(data.x, data.edge_index, data.edge_attr, data.edge_type)
        assert out.dtype == torch.float32

    def test_type_routing_odom_zeroed(self):
        """Zeroing W_odom should produce zero output for nodes that only receive
        odometry messages, while nodes receiving LC messages remain nonzero.

        Graph: edge (0→1, type=0) and edge (2→3, type=1).
        After W_odom ← 0: node 1's only message is zero; node 3's is nonzero.
        """
        data = _minimal_typed_graph()
        conv = EdgeTypeAwareConv(in_channels=3, out_channels=8)

        # Force W_odom to zero (both weight and bias)
        with torch.no_grad():
            conv.type_linears[0].weight.zero_()
            conv.type_linears[0].bias.zero_()
            # Make W_loop produce a known nonzero output
            conv.type_linears[1].weight.fill_(0.1)
            conv.type_linears[1].bias.fill_(0.1)

        out = conv(data.x, data.edge_index, data.edge_attr, data.edge_type)

        # Node 1 only receives from odom edge (zeroed) → output must be zero
        # (ReLU(0) = 0; sum aggregation of a single zero message)
        assert out[1].abs().max().item() == 0.0, "node 1 should have zero output"

        # Node 3 only receives from LC edge (nonzero weights) → output nonzero
        assert out[3].abs().max().item() > 0.0, "node 3 should have nonzero output"

    def test_type_routing_loop_closure_zeroed(self):
        """Symmetric check: zeroing W_loop leaves odom output intact."""
        data = _minimal_typed_graph()
        conv = EdgeTypeAwareConv(in_channels=3, out_channels=8)

        with torch.no_grad():
            conv.type_linears[1].weight.zero_()
            conv.type_linears[1].bias.zero_()
            conv.type_linears[0].weight.fill_(0.1)
            conv.type_linears[0].bias.fill_(0.1)

        out = conv(data.x, data.edge_index, data.edge_attr, data.edge_type)

        assert out[3].abs().max().item() == 0.0, "node 3 should have zero output"
        assert out[1].abs().max().item() > 0.0, "node 1 should have nonzero output"

    def test_gradients_flow(self):
        """Autograd must be able to differentiate through the conv."""
        data = _small_pyg_graph()
        x = data.x.requires_grad_(True)
        conv = EdgeTypeAwareConv(in_channels=3, out_channels=16)

        out = conv(x, data.edge_index, data.edge_attr, data.edge_type)
        out.sum().backward()

        assert x.grad is not None
        assert not x.grad.isnan().any(), "NaN gradients detected"


# ── EdgeGateGNN ───────────────────────────────────────────────────────────────

class TestEdgeGateGNN:

    def test_output_shape(self):
        """GNN output must be (E,) — one scalar per edge."""
        data = _small_pyg_graph()
        E = data.edge_index.size(1)
        model = EdgeGateGNN()

        scores = model(data)

        assert scores.shape == (E,), f"expected ({E},), got {scores.shape}"

    def test_output_dtype(self):
        data = _small_pyg_graph()
        model = EdgeGateGNN()
        scores = model(data)
        assert scores.dtype == torch.float32

    def test_output_range(self):
        """All confidence scores must be in [0, 1] (sigmoid output)."""
        data = _small_pyg_graph()
        model = EdgeGateGNN()

        scores = model(data)

        assert scores.min().item() >= 0.0, "confidence scores must be ≥ 0"
        assert scores.max().item() <= 1.0, "confidence scores must be ≤ 1"

    def test_gradients_flow(self):
        """Loss must be differentiable back to the GNN's parameters."""
        data = _small_pyg_graph()
        model = EdgeGateGNN()

        scores = model(data)
        # Simulate BCE loss against all-inlier labels
        target = torch.ones_like(scores)
        loss = F.binary_cross_entropy(scores, target)
        loss.backward()

        # At least one parameter should have received a gradient
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "no gradients reached model parameters"
        for g in grads:
            assert not g.isnan().any(), "NaN gradient in model parameters"

    def test_single_layer_no_residual(self):
        """num_layers=1 has no residual blocks; must still produce correct shapes."""
        data = _small_pyg_graph()
        E = data.edge_index.size(1)
        model = EdgeGateGNN(num_layers=1, hidden_dim=32)

        scores = model(data)

        assert scores.shape == (E,)
        assert scores.min().item() >= 0.0
        assert scores.max().item() <= 1.0

    def test_hidden_dim_configurable(self):
        """Different hidden_dims must produce the same output shape."""
        data = _small_pyg_graph()
        E = data.edge_index.size(1)

        for hdim in (16, 64, 128):
            scores = EdgeGateGNN(hidden_dim=hdim)(data)
            assert scores.shape == (E,), f"hidden_dim={hdim} broke output shape"

    def test_batched_graphs(self):
        """GNN must work on a PyG Batch of multiple graphs (training code path).

        During edge_bce training, graphs are batched via Batch.from_data_list.
        The batch vector is forwarded to GraphNorm so it normalises per graph,
        not over the concatenated multi-graph.
        """
        data0 = _small_pyg_graph(seed=0)
        data1 = _small_pyg_graph(seed=1)
        batch = Batch.from_data_list([data0, data1])

        E_total = data0.edge_index.size(1) + data1.edge_index.size(1)
        model = EdgeGateGNN()

        scores = model(batch)

        assert scores.shape == (E_total,), (
            f"batched output shape: expected ({E_total},), got {scores.shape}"
        )
        assert scores.min().item() >= 0.0
        assert scores.max().item() <= 1.0

    def test_batched_graphs_gradients(self):
        """Gradients must flow correctly when processing a batched graph."""
        data0 = _small_pyg_graph(seed=0)
        data1 = _small_pyg_graph(seed=1)
        batch = Batch.from_data_list([data0, data1])

        model = EdgeGateGNN()
        scores = model(batch)
        target = torch.ones_like(scores)
        F.binary_cross_entropy(scores, target).backward()

        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0
        for g in grads:
            assert not g.isnan().any()

    def test_eval_mode_deterministic(self):
        """In eval mode, two identical forward passes must produce identical output.

        Dropout is active during training and disabled during eval — this test
        verifies the model honours nn.Module's train/eval distinction.
        """
        data = _small_pyg_graph()
        model = EdgeGateGNN(dropout=0.5)
        model.eval()

        with torch.no_grad():
            out1 = model(data)
            out2 = model(data)

        torch.testing.assert_close(out1, out2)

    def test_odom_scores_in_range(self):
        """Odometry edges produce valid sigmoid outputs in [0, 1].

        The GNN now runs the confidence head on all edges. Odometry edges
        receive learned weights supervised via trajectory loss. This test
        verifies scores are in the valid sigmoid range — not exactly 1.0.
        """
        data = _minimal_typed_graph()
        model = EdgeGateGNN()
        model.eval()

        with torch.no_grad():
            scores = model(data)

        odom_mask = data.edge_type == 0
        lc_mask   = data.edge_type == 1

        odom_vals = scores[odom_mask]
        assert (odom_vals >= 0.0).all() and (odom_vals <= 1.0).all(), (
            f"odom scores should be in [0, 1], got {odom_vals.tolist()}"
        )
        lc_vals = scores[lc_mask]
        assert (lc_vals >= 0.0).all() and (lc_vals <= 1.0).all(), (
            f"LC scores should be in [0, 1], got {lc_vals.tolist()}"
        )
