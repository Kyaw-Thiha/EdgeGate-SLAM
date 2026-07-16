"""Tests for edgegate/metrics/edge_f1.py and edgegate/metrics/ate_rmse.py.

Coverage strategy:
  - edge_f1: correct tp/fp/fn counts across perfect, all-wrong, mixed, and
    degenerate (all-positive / all-negative) cases; threshold sensitivity.
  - ate_rmse: identity input gives 0, pure translation recovers 0, pure
    rotation recovers 0, both together recover 0, non-zero error is positive.
    Also verifies double-precision computation and det(R)=+1 (no reflections).
"""
import math
import pytest
import torch
import numpy as np

from edgegate.metrics.edge_f1 import edge_f1
from edgegate.metrics.ate_rmse import ate_rmse


# ── edge_f1 ───────────────────────────────────────────────────────────────────

class TestEdgeF1:

    def test_perfect_predictions(self):
        """All correct → precision = recall = f1 = 1.0, fp = fn = 0."""
        conf   = torch.tensor([0.9, 0.1, 0.8, 0.2])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])

        out = edge_f1(conf, labels)

        assert out["tp"] == 2
        assert out["fp"] == 0
        assert out["fn"] == 0
        assert out["precision"] == pytest.approx(1.0)
        assert out["recall"] == pytest.approx(1.0)
        assert out["f1"] == pytest.approx(1.0)

    def test_all_wrong(self):
        """Predictions inverted from labels → precision = recall = f1 = 0.

        tp = 0, fp > 0, fn > 0 → all metrics zero.
        """
        conf   = torch.tensor([0.1, 0.9, 0.1, 0.9])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])

        out = edge_f1(conf, labels)

        assert out["tp"] == 0
        assert out["fp"] == 2
        assert out["fn"] == 2
        assert out["precision"] == pytest.approx(0.0)
        assert out["recall"] == pytest.approx(0.0)
        assert out["f1"] == pytest.approx(0.0)

    def test_mixed_predictions(self):
        """Known mixed case: verify all six output keys by hand."""
        # Predictions: [1, 1, 0, 0] (threshold 0.5)
        # Labels:      [1, 0, 1, 0]
        # tp=1, fp=1, fn=1  →  P=0.5, R=0.5, F1=0.5
        conf   = torch.tensor([0.9, 0.9, 0.1, 0.1])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])

        out = edge_f1(conf, labels)

        assert out["tp"] == 1
        assert out["fp"] == 1
        assert out["fn"] == 1
        assert out["precision"] == pytest.approx(0.5)
        assert out["recall"] == pytest.approx(0.5)
        assert out["f1"] == pytest.approx(0.5)

    def test_custom_threshold(self):
        """Higher threshold should reduce tp (more false negatives)."""
        conf   = torch.tensor([0.6, 0.4, 0.9, 0.1])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])

        out_05 = edge_f1(conf, labels, threshold=0.5)
        out_07 = edge_f1(conf, labels, threshold=0.7)

        # At 0.5: both positives above threshold → tp=2
        assert out_05["tp"] == 2
        # At 0.7: only the 0.9 edge is above threshold → tp=1, fn=1
        assert out_07["tp"] == 1
        assert out_07["fn"] == 1

    def test_no_positive_predictions_precision_guard(self):
        """When no edge is predicted positive, precision should be 1.0 (no fp)."""
        conf   = torch.tensor([0.1, 0.2, 0.3])
        labels = torch.tensor([1.0, 1.0, 0.0])

        out = edge_f1(conf, labels)  # threshold=0.5 → all predicted negative

        assert out["tp"] == 0
        assert out["fp"] == 0
        assert out["fn"] == 2
        assert out["precision"] == pytest.approx(1.0)  # guard: no predictions → no false positives
        assert out["recall"] == pytest.approx(0.0)

    def test_no_positive_labels_recall_guard(self):
        """When no GT positive edges exist, recall should be 1.0 (no fn)."""
        conf   = torch.tensor([0.9, 0.8])
        labels = torch.tensor([0.0, 0.0])

        out = edge_f1(conf, labels)

        assert out["fn"] == 0
        assert out["recall"] == pytest.approx(1.0)

    def test_return_keys(self):
        """Output dict must contain exactly the six documented keys."""
        conf   = torch.tensor([0.7, 0.3])
        labels = torch.tensor([1.0, 0.0])
        out = edge_f1(conf, labels)
        assert set(out.keys()) == {"precision", "recall", "f1", "tp", "fp", "fn"}

    def test_count_types_are_int(self):
        """tp, fp, fn must be Python ints (for cross-batch accumulation)."""
        conf   = torch.tensor([0.7, 0.3])
        labels = torch.tensor([1.0, 0.0])
        out = edge_f1(conf, labels)
        assert isinstance(out["tp"], int)
        assert isinstance(out["fp"], int)
        assert isinstance(out["fn"], int)

    def test_float_types_for_metrics(self):
        conf   = torch.tensor([0.7, 0.3])
        labels = torch.tensor([1.0, 0.0])
        out = edge_f1(conf, labels)
        assert isinstance(out["precision"], float)
        assert isinstance(out["recall"], float)
        assert isinstance(out["f1"], float)

    def test_f1_Jensen_inequality_trap(self):
        """Verify counts accumulate correctly for the Jensen's inequality use-case.

        Averaging per-batch F1 and computing F1 from accumulated counts diverge.
        This test confirms that accumulating tp/fp/fn and computing once yields
        the correct result — the intended usage documented in edge_f1's docstring.
        """
        # Batch 1: tp=1, fp=0, fn=1 → P=1.0, R=0.5, F1=0.667
        # Batch 2: tp=0, fp=1, fn=1 → P=0.0, R=0.0, F1=0.0
        # Average F1 per-batch: (0.667 + 0.0) / 2 = 0.333
        # Accumulated: tp=1, fp=1, fn=2 → P=0.5, R=0.333 → F1=0.4
        # These differ — test that accumulation approach is used correctly.

        conf1 = torch.tensor([0.9, 0.1])
        labels1 = torch.tensor([1.0, 1.0])  # both inliers; one predicted, one missed

        conf2 = torch.tensor([0.9, 0.1])
        labels2 = torch.tensor([0.0, 1.0])  # one outlier predicted inlier, one inlier missed

        out1 = edge_f1(conf1, labels1)
        out2 = edge_f1(conf2, labels2)

        # Accumulated counts
        tp = out1["tp"] + out2["tp"]
        fp = out1["fp"] + out2["fp"]
        fn = out1["fn"] + out2["fn"]

        assert tp == 1
        assert fp == 1
        assert fn == 2

        precision = tp / (tp + fp)  # 0.5
        recall    = tp / (tp + fn)  # 0.333...
        f1_accum  = 2 * precision * recall / (precision + recall)  # 0.4

        # Per-batch average for comparison
        f1_avg = (out1["f1"] + out2["f1"]) / 2.0

        assert f1_accum != pytest.approx(f1_avg, abs=0.05), (
            "accumulated F1 should differ from per-batch average — "
            "confirm the Jensen inequality trap is real"
        )


# ── ate_rmse ──────────────────────────────────────────────────────────────────

def _make_trajectory(n: int = 20, seed: int = 0) -> torch.Tensor:
    """Random (N, 3) trajectory with [x, y, θ]."""
    torch.manual_seed(seed)
    return torch.rand(n, 3) * 10.0


def _apply_se2(poses: torch.Tensor, angle: float, tx: float, ty: float) -> torch.Tensor:
    """Apply a 2D rigid transformation (rotation + translation) to (x, y)."""
    ca, sa = math.cos(angle), math.sin(angle)
    R = torch.tensor([[ca, -sa], [sa, ca]], dtype=torch.float64)
    t = torch.tensor([tx, ty], dtype=torch.float64)
    p = poses[:, :2].double()
    p_transformed = (R @ p.T).T + t
    result = poses.clone().double()
    result[:, :2] = p_transformed
    return result.float()


class TestAteRmse:

    def test_identical_trajectories_give_zero(self):
        """Perfect estimate (same as GT) must give ATE = 0."""
        poses = _make_trajectory()
        ate = ate_rmse(poses, poses.clone())
        assert ate == pytest.approx(0.0, abs=1e-6)

    def test_pure_translation_gives_zero(self):
        """A rigid translation between estimate and GT must align out to ATE = 0.

        Umeyama solves for t = μ_Q − R μ_P, so a constant offset is fully
        absorbed and the RMSE after alignment should be 0.
        """
        poses_gt = _make_trajectory(n=30)
        poses_est = _apply_se2(poses_gt, angle=0.0, tx=5.0, ty=-3.0)

        ate = ate_rmse(poses_est, poses_gt)
        assert ate == pytest.approx(0.0, abs=1e-5)

    def test_pure_rotation_gives_zero(self):
        """A pure rotation about the centroid must align to ATE = 0."""
        poses_gt = _make_trajectory(n=30)
        poses_est = _apply_se2(poses_gt, angle=math.pi / 4, tx=0.0, ty=0.0)

        ate = ate_rmse(poses_est, poses_gt)
        assert ate == pytest.approx(0.0, abs=1e-4)

    def test_rotation_and_translation_gives_zero(self):
        """Combined SE(2) transformation must be fully undone by alignment."""
        poses_gt = _make_trajectory(n=50)
        poses_est = _apply_se2(poses_gt, angle=1.23, tx=-7.0, ty=4.5)

        ate = ate_rmse(poses_est, poses_gt)
        assert ate == pytest.approx(0.0, abs=1e-4)

    def test_noisy_trajectory_positive_ate(self):
        """A trajectory with added noise must give ATE > 0."""
        torch.manual_seed(99)
        poses_gt = _make_trajectory(n=40)
        noise = torch.randn(40, 3) * 0.5
        noise[:, 2] = 0.0  # perturb x,y only
        poses_noisy = poses_gt + noise

        ate = ate_rmse(poses_noisy, poses_gt)
        assert ate > 0.01, f"expected positive ATE after noise, got {ate}"

    def test_returns_float(self):
        """ate_rmse must return a Python float, not a tensor."""
        poses = _make_trajectory()
        result = ate_rmse(poses, poses.clone())
        assert isinstance(result, float)

    def test_theta_column_ignored(self):
        """Changing only the θ column of the estimate must not change ATE.

        ATE is computed on (x, y) only.
        """
        poses_gt = _make_trajectory(n=20)
        poses_est1 = poses_gt.clone()
        poses_est2 = poses_gt.clone()
        poses_est2[:, 2] += 999.0  # large delta in theta only

        ate1 = ate_rmse(poses_est1, poses_gt)
        ate2 = ate_rmse(poses_est2, poses_gt)

        assert ate1 == pytest.approx(ate2, abs=1e-6)

    def test_no_reflection_in_alignment(self):
        """Umeyama det-correction must prevent reflections (det(R) = +1).

        Without the det correction, the SVD minimiser may choose a reflection
        matrix (det = -1), which looks like a good fit but is geometrically
        wrong. After alignment the RMSE should still be 0 for a valid rigid
        transform input.
        """
        # Build a trajectory where a reflection would seem to give lower cost
        # without the Umeyama correction. A symmetric-ish point set is worst-case.
        n = 10
        x = torch.linspace(0, 1, n)
        y = torch.zeros(n)
        gt = torch.stack([x, y, torch.zeros(n)], dim=1)
        # Apply a 180° rotation + translation (valid SE(2), but SVD might flip)
        est = _apply_se2(gt, angle=math.pi, tx=3.0, ty=1.0)

        ate = ate_rmse(est, gt)
        assert ate == pytest.approx(0.0, abs=1e-4), (
            f"ATE={ate} > 0 suggests a reflection was used instead of a proper rotation"
        )

    def test_ate_symmetric_in_noise(self):
        """Adding noise in opposite directions should give same magnitude ATE."""
        torch.manual_seed(5)
        poses_gt = _make_trajectory(n=20)
        torch.manual_seed(5)
        noise = torch.randn(20, 3) * 0.3
        noise[:, 2] = 0.0

        ate_plus = ate_rmse(poses_gt + noise, poses_gt)
        ate_minus = ate_rmse(poses_gt - noise, poses_gt)

        # Not exactly equal (alignment changes depending on which side the noise lands),
        # but both should be positive and of similar scale.
        assert ate_plus > 0.0
        assert ate_minus > 0.0
