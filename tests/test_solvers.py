import math
import numpy as np
import pytest
import torch

from edgegate.data.synthetic_generator import generate
from edgegate.data.types import PoseGraph
from edgegate.solvers.base import scale_information
from edgegate.solvers.pypose_solver import PyPoseSolver
from edgegate.solvers.gtsam_solver import GTSAMSolver


_NUM_POSES = 50
_NUM_LC = 4


def _clean_graph(seed: int = 0) -> tuple[PoseGraph, torch.Tensor]:
    """Outlier-free graph with uniform edge weights."""
    g = generate(
        num_poses=_NUM_POSES,
        num_loop_closures=_NUM_LC,
        outlier_rate=0,
        outlier_structure="random",
        seed=seed,
        proximity_threshold=5.0,
    )
    E = g.edge_index.shape[1]
    return g, torch.ones(E)


def _angle_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Wrap-aware absolute angular difference, element-wise."""
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return d.abs()


# ── scale_information ─────────────────────────────────────────────────────────

def test_scale_information_uniform_weights():
    info = torch.tensor([[500.0, 0.0, 0.0, 500.0, 0.0, 100.0]])
    result = scale_information(info, torch.ones(1))
    torch.testing.assert_close(result, info)


def test_scale_information_zero_weights():
    info = torch.ones(2, 6)
    result = scale_information(info, torch.zeros(2))
    torch.testing.assert_close(result, torch.zeros(2, 6))


def test_scale_information_half_weights():
    info = torch.tensor([[4.0, 0.0, 0.0, 4.0, 0.0, 4.0]])
    result = scale_information(info, torch.tensor([0.5]))
    torch.testing.assert_close(result, torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 1.0]]))


def test_scale_information_batch():
    info = torch.ones(3, 6)
    w = torch.tensor([1.0, 2.0, 3.0])
    result = scale_information(info, w)
    expected = torch.tensor([[1.0] * 6, [4.0] * 6, [9.0] * 6])
    torch.testing.assert_close(result, expected)


# ── PyPoseSolver ──────────────────────────────────────────────────────────────

def test_pypose_solver_shapes():
    g, w = _clean_graph()
    poses, converged, n_iters, cost = PyPoseSolver().solve(g, w)
    assert poses.shape == (_NUM_POSES, 3)
    assert isinstance(converged, bool)
    assert isinstance(n_iters, int)
    assert isinstance(cost, float)


def test_pypose_solver_output_dtype():
    g, w = _clean_graph()
    poses, *_ = PyPoseSolver().solve(g, w)
    assert isinstance(poses, torch.Tensor)
    assert poses.dtype == torch.float32


def test_pypose_solver_converges_on_clean_graph():
    g, w = _clean_graph()
    _, converged, _, _ = PyPoseSolver().solve(g, w)
    assert converged


def test_pypose_solver_pose0_is_fixed():
    """Pose 0 must remain at the initial anchor (gauge freedom fixed)."""
    g, w = _clean_graph()
    poses, *_ = PyPoseSolver().solve(g, w)
    init0 = torch.from_numpy(g.node_init[0]).float()
    torch.testing.assert_close(poses[0], init0, atol=1e-4, rtol=0)


def test_pypose_solver_cost_decreases():
    """Final cost must be lower than the initial squared-residual sum."""
    g, w = _clean_graph()
    from edgegate.solvers.pypose_solver import _PGOModel

    edge_info = torch.from_numpy(g.edge_info).float()
    info_sqrt = scale_information(edge_info, w)[:, [0, 3, 5]].clamp(min=0).sqrt()
    model = _PGOModel(g, info_sqrt)

    with torch.no_grad():
        initial_cost = (model() ** 2).sum().item()

    _, _, _, final_cost = PyPoseSolver().solve(g, w)
    assert final_cost < initial_cost


# ── GTSAMSolver ───────────────────────────────────────────────────────────────

def test_gtsam_solver_shapes():
    g, w = _clean_graph()
    poses, converged, n_iters, cost = GTSAMSolver(kernel="none").solve(g, w)
    assert poses.shape == (_NUM_POSES, 3)
    assert isinstance(converged, bool)
    assert isinstance(n_iters, int)
    assert isinstance(cost, float)


def test_gtsam_solver_output_dtype():
    g, w = _clean_graph()
    poses, *_ = GTSAMSolver(kernel="none").solve(g, w)
    assert isinstance(poses, torch.Tensor)
    assert poses.dtype == torch.float32


def test_gtsam_solver_converges_on_clean_graph():
    g, w = _clean_graph()
    _, converged, _, _ = GTSAMSolver(kernel="none").solve(g, w)
    assert converged


def test_gtsam_gnc_rejects_non_unit_weights():
    """kernel='gnc' must assert when edge_weights != 1 (double-robustification guard)."""
    g, w = _clean_graph()
    with pytest.raises(AssertionError):
        GTSAMSolver(kernel="gnc").solve(g, w * 0.5)


def test_gtsam_gnc_accepts_unit_weights():
    g, w = _clean_graph()
    poses, converged, _, _ = GTSAMSolver(kernel="gnc").solve(g, w)
    assert poses.shape == (_NUM_POSES, 3)
    assert converged


def test_gtsam_invalid_kernel_raises():
    with pytest.raises(AssertionError):
        GTSAMSolver(kernel="switchable")


# ── Solver agreement on outlier-free graphs ───────────────────────────────────

@pytest.mark.parametrize("seed", [0, 1, 7])
def test_pypose_gtsam_agree_on_clean_graph(seed: int):
    """Both solvers must converge and reach equivalent objective values on outlier-free graphs.

    PGO is non-convex: two backends can produce different pose configurations that
    both achieve the same chi² cost (null-space degeneracy with few loop closures).
    Comparing objective values — not poses — is the principled agreement criterion.

    PyPose minimises sum(||r_white||²) while GTSAM uses the ½ convention, so
    pp_cost ≈ 2 × gt_cost. Relative disagreement should stay below 2%.
    """
    g, w = _clean_graph(seed=seed)

    _, pp_converged, _, pp_cost = PyPoseSolver().solve(g, w)
    _, gt_converged, _, gt_cost = GTSAMSolver(kernel="none").solve(g, w)

    assert pp_converged, f"PyPose solver did not converge (seed={seed})"
    assert gt_converged, f"GTSAM solver did not converge (seed={seed})"

    gt_cost_scaled = gt_cost * 2.0  # GTSAM uses ½·chi², normalise to same scale
    rel_diff = abs(pp_cost - gt_cost_scaled) / (gt_cost_scaled + 1e-10)
    assert rel_diff < 0.02, (
        f"seed={seed}: relative cost disagreement {rel_diff:.4f} > 2% "
        f"(pp={pp_cost:.4f}, gt×2={gt_cost_scaled:.4f})"
    )
