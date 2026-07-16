"""Tests for edgegate/losses/edge_bce.py and edgegate/losses/trajectory_loss.py.

Coverage strategy:
  - EdgeBCELoss: correct LC-only masking, reduction modes, ValueError on no-LC
    input, gradient flow through the BCE loss.
  - TrajectoryLoss: solver is called with correct arguments, position MSE is
    computed correctly, gradients flow back through the solver to confidence.
    A lightweight mock solver is used so these tests run without PyPose.
"""
import pytest
import torch
import torch.nn.functional as F
import numpy as np

from edgegate.losses.edge_bce import EdgeBCELoss
from edgegate.losses.trajectory_loss import TrajectoryLoss
from edgegate.data.types import PoseGraph
from edgegate.solvers.base import Solver


# ── EdgeBCELoss ───────────────────────────────────────────────────────────────

class TestEdgeBCELoss:

    def _make_inputs(self, n_odom: int = 4, n_lc: int = 6, seed: int = 0):
        """Build a minimal confidence / labels / edge_type triple.

        Returns:
            confidence: (n_odom + n_lc,) float tensor with requires_grad
            labels:     (n_odom + n_lc,) float tensor, alternating 0/1 over LC
            edge_type:  (n_odom + n_lc,) long tensor, 0=odom 1=LC
        """
        torch.manual_seed(seed)
        E = n_odom + n_lc
        confidence = torch.rand(E, requires_grad=True)
        labels = torch.zeros(E)
        labels[n_odom::2] = 1.0          # every other LC edge is an inlier
        edge_type = torch.zeros(E, dtype=torch.long)
        edge_type[n_odom:] = 1           # last n_lc edges are LC
        return confidence, labels, edge_type

    def test_loss_is_scalar(self):
        """forward() must return a zero-dimensional tensor."""
        loss_fn = EdgeBCELoss()
        conf, labels, et = self._make_inputs()
        loss = loss_fn(conf, labels, et)
        assert loss.ndim == 0

    def test_loss_only_over_lc_edges(self):
        """Setting all odom confidences to 0.0 must not change the loss value.

        If odom edges were included, flipping their confidence from random
        values to 0.0 would produce different BCE. Equality here proves that
        only the LC portion participates.
        """
        loss_fn = EdgeBCELoss()
        n_odom, n_lc = 4, 6
        torch.manual_seed(42)
        conf_rand = torch.rand(n_odom + n_lc)
        conf_zero_odom = conf_rand.clone()
        conf_zero_odom[:n_odom] = 0.0

        labels = torch.zeros(n_odom + n_lc)
        labels[n_odom::2] = 1.0
        edge_type = torch.zeros(n_odom + n_lc, dtype=torch.long)
        edge_type[n_odom:] = 1

        loss_rand = loss_fn(conf_rand, labels, edge_type)
        loss_zero = loss_fn(conf_zero_odom, labels, edge_type)

        torch.testing.assert_close(loss_rand, loss_zero)

    def test_reduction_mean_vs_sum(self):
        """'sum' loss should equal 'mean' loss scaled by number of LC edges."""
        n_odom, n_lc = 4, 6
        torch.manual_seed(7)
        conf = torch.rand(n_odom + n_lc)
        labels = torch.zeros(n_odom + n_lc)
        labels[n_odom:] = 1.0
        edge_type = torch.zeros(n_odom + n_lc, dtype=torch.long)
        edge_type[n_odom:] = 1

        loss_mean = EdgeBCELoss(reduction="mean")(conf, labels, edge_type)
        loss_sum = EdgeBCELoss(reduction="sum")(conf, labels, edge_type)

        torch.testing.assert_close(loss_sum, loss_mean * n_lc)

    def test_no_lc_edges_raises(self):
        """A graph with no LC edges should raise ValueError, not silently return 0."""
        loss_fn = EdgeBCELoss()
        E = 5
        conf = torch.rand(E)
        labels = torch.zeros(E)
        edge_type = torch.zeros(E, dtype=torch.long)  # all odom

        with pytest.raises(ValueError, match="no loop-closure edges"):
            loss_fn(conf, labels, edge_type)

    def test_perfect_predictions_low_loss(self):
        """Predictions that exactly match labels should give near-zero loss."""
        loss_fn = EdgeBCELoss()
        n_odom, n_lc = 3, 4
        E = n_odom + n_lc
        labels = torch.zeros(E)
        labels[n_odom:] = torch.tensor([1.0, 0.0, 1.0, 0.0])

        # Confidences very close to labels for LC edges
        conf = torch.full((E,), 0.5)
        conf[n_odom:] = torch.tensor([1.0 - 1e-6, 1e-6, 1.0 - 1e-6, 1e-6])

        edge_type = torch.zeros(E, dtype=torch.long)
        edge_type[n_odom:] = 1

        loss = loss_fn(conf, labels, edge_type)
        assert loss.item() < 0.01, f"perfect predictions gave high loss: {loss.item()}"

    def test_gradients_flow(self):
        """Loss must be differentiable w.r.t. confidence."""
        loss_fn = EdgeBCELoss()
        conf, labels, et = self._make_inputs()

        loss = loss_fn(conf, labels, et)
        loss.backward()

        assert conf.grad is not None
        # Gradients should only be nonzero for LC edges (the odom slice was masked out)
        n_odom = (et == 0).sum().item()
        assert conf.grad[:n_odom].abs().sum().item() == 0.0, (
            "odom edges should have zero gradient — they are excluded from the loss"
        )
        assert conf.grad[n_odom:].abs().sum().item() > 0.0, (
            "LC edges should receive gradient through BCE"
        )

    def test_invalid_reduction_raises(self):
        """Unsupported reduction string should raise AssertionError at init time."""
        with pytest.raises(AssertionError):
            EdgeBCELoss(reduction="none")


# ── TrajectoryLoss ────────────────────────────────────────────────────────────

class _IdentitySolver(Solver):
    """Minimal mock solver: returns confidence-weighted average of gt positions.

    This is not a real PGO solver — it exists to test that TrajectoryLoss
    calls the solver correctly and that gradients flow through it back to
    confidence. The solve output is a differentiable function of edge_weights
    so backward() can be tested without PyPose.
    """

    def __init__(self, gt_poses: torch.Tensor) -> None:
        self._gt = gt_poses  # (N, 3) ground-truth used to produce a fake estimate

    def solve(
        self,
        graph: PoseGraph,
        edge_weights: torch.Tensor,
        max_iterations: int | None = None,
    ):
        # Fake estimate: gt + small offset weighted by mean(edge_weights)
        # This is differentiable w.r.t. edge_weights.
        offset = (1.0 - edge_weights.mean()) * 0.1
        poses_est = self._gt + offset
        return poses_est, True, max_iterations or 1, 0.0


def _make_pose_graph(n_nodes: int = 8) -> PoseGraph:
    """Build a tiny PoseGraph with synthetic data (no real edges needed)."""
    rng = np.random.default_rng(0)
    node_init = rng.standard_normal((n_nodes, 3)).astype(np.float32)
    # Two odom edges + one LC edge — minimal to satisfy the data contract
    edge_index = np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int64)
    edge_measurement = rng.standard_normal((3, 3)).astype(np.float32)
    edge_info = np.tile(np.eye(3, dtype=np.float32)[None], (3, 1, 1))
    edge_type = np.array([0, 0, 1], dtype=np.int64)
    gt_poses = rng.standard_normal((n_nodes, 3)).astype(np.float32)
    return PoseGraph(
        node_init=node_init,
        edge_index=edge_index,
        edge_measurement=edge_measurement,
        edge_info=edge_info,
        edge_type=edge_type,
        gt_node_poses=gt_poses,
    )


class TestTrajectoryLoss:

    def test_loss_is_scalar(self):
        """forward() must return a zero-dimensional tensor."""
        graph = _make_pose_graph()
        gt = torch.from_numpy(graph.gt_node_poses).float()
        solver = _IdentitySolver(gt)
        loss_fn = TrajectoryLoss(solver=solver, train_iterations=3)

        conf = torch.ones(graph.edge_index.shape[1])
        loss = loss_fn(graph, conf, gt)

        assert loss.ndim == 0

    def test_loss_zero_on_perfect_estimate(self):
        """When solver returns exactly gt_poses, position MSE must be 0."""
        graph = _make_pose_graph()
        gt = torch.from_numpy(graph.gt_node_poses).float()

        # Solver that ignores confidence and returns gt exactly
        class PerfectSolver(Solver):
            def solve(self, g, w, max_iterations=None):
                return gt.clone(), True, 1, 0.0

        loss_fn = TrajectoryLoss(solver=PerfectSolver(), train_iterations=5)
        conf = torch.ones(graph.edge_index.shape[1])
        loss = loss_fn(graph, conf, gt)

        assert loss.item() < 1e-10, f"expected ~0 loss, got {loss.item()}"

    def test_loss_uses_xy_only(self):
        """TrajectoryLoss should compute MSE on columns 0,1 (x,y), not theta.

        Verify: modifying only the theta column of gt_poses does not change the
        returned loss.
        """
        graph = _make_pose_graph()
        gt = torch.from_numpy(graph.gt_node_poses).float()

        class FixedSolver(Solver):
            def __init__(self, poses):
                self._poses = poses
            def solve(self, g, w, max_iterations=None):
                return self._poses, True, 1, 0.0

        fixed_est = gt + 0.5   # known offset on all 3 columns
        solver = FixedSolver(fixed_est)
        loss_fn = TrajectoryLoss(solver=solver, train_iterations=1)
        conf = torch.ones(graph.edge_index.shape[1])

        gt_modified_theta = gt.clone()
        gt_modified_theta[:, 2] += 100.0  # huge change in theta only

        loss_orig = loss_fn(graph, conf, gt)
        loss_theta = loss_fn(graph, conf, gt_modified_theta)

        # theta column is excluded, so the loss should change (offset on x,y still applies)
        # More precisely: loss uses gt[:, :2], so same change in theta → same x,y loss
        torch.testing.assert_close(loss_orig, loss_theta)

    def test_train_iterations_passed_to_solver(self):
        """solver.solve() must receive max_iterations=train_iterations."""
        graph = _make_pose_graph()
        gt = torch.from_numpy(graph.gt_node_poses).float()
        received = {}

        class RecordingSolver(Solver):
            def solve(self, g, w, max_iterations=None):
                received["max_iterations"] = max_iterations
                return gt.clone(), True, max_iterations or 1, 0.0

        loss_fn = TrajectoryLoss(solver=RecordingSolver(), train_iterations=7)
        conf = torch.ones(graph.edge_index.shape[1])
        loss_fn(graph, conf, gt)

        assert received.get("max_iterations") == 7, (
            f"expected max_iterations=7, got {received.get('max_iterations')}"
        )

    def test_gradients_flow_to_confidence(self):
        """Loss must be differentiable back to confidence (the GNN training signal).

        The mock solver is designed to return a function of mean(edge_weights) so
        that gradients can flow back through to confidence. Use conf=0.5 so the
        solver offset is (1 - 0.5) * 0.1 = 0.05 (nonzero loss, nonzero gradient).
        """
        graph = _make_pose_graph()
        gt = torch.from_numpy(graph.gt_node_poses).float()
        solver = _IdentitySolver(gt)
        loss_fn = TrajectoryLoss(solver=solver, train_iterations=3)

        conf = torch.full((graph.edge_index.shape[1],), 0.5, requires_grad=True)
        loss = loss_fn(graph, conf, gt)
        loss.backward()

        assert conf.grad is not None
        assert not conf.grad.isnan().any(), "NaN gradient in confidence"
        assert conf.grad.abs().sum().item() > 0.0, "zero gradient — loss not differentiable"
