import numpy as np
import pytest
from edgegate.data.outlier_injection import inject_labeled_loop_closures


def _make_straight_trajectory(n: int = 30) -> np.ndarray:
    """Simple right-moving line trajectory for deterministic testing."""
    poses = np.zeros((n, 3))
    poses[:, 0] = np.arange(n, dtype=np.float64)
    return poses


def _make_looping_trajectory(n: int = 30) -> np.ndarray:
    """Trajectory that returns near origin, creating proximal revisits."""
    poses = np.zeros((n, 3))
    half = n // 2
    # Move right for first half
    for i in range(1, half):
        poses[i, 0] = poses[i - 1, 0] + 1.0
        poses[i, 2] = 0.0
    # Turn around and move back for second half
    poses[half, 2] = np.pi
    for i in range(half + 1, n):
        poses[i, 0] = poses[i - 1, 0] - 1.0
        poses[i, 2] = np.pi
    return poses


def test_inject_zero_outliers():
    """All LCs should be inliers when outlier_rate=0."""
    rng = np.random.default_rng(0)
    ref = _make_looping_trajectory(30)
    lc_edges, lc_meas, lc_labels = inject_labeled_loop_closures(
        ref, num_loop_closures=5, outlier_rate=0,
        outlier_structure="random", rng=rng,
        proximity_threshold=5.0, min_gap=3,
    )
    assert lc_edges.shape == (2, 5)
    assert lc_meas.shape == (5, 3)
    assert lc_labels.shape == (5,)
    assert np.all(lc_labels == 1.0)
    assert lc_edges.dtype == np.int64


def test_inject_all_outliers():
    """All LCs should be outliers when outlier_rate=100."""
    rng = np.random.default_rng(1)
    ref = _make_looping_trajectory(30)
    lc_edges, lc_meas, lc_labels = inject_labeled_loop_closures(
        ref, num_loop_closures=5, outlier_rate=100,
        outlier_structure="random", rng=rng,
        proximity_threshold=2.0, min_gap=3, outlier_distance_threshold=10.0,
    )
    assert np.all(lc_labels == 0.0)


def test_mixed_labels():
    """At 50% rate, approximately half should be outliers."""
    rng = np.random.default_rng(2)
    ref = _make_looping_trajectory(30)
    lc_edges, lc_meas, lc_labels = inject_labeled_loop_closures(
        ref, num_loop_closures=10, outlier_rate=50,
        outlier_structure="random", rng=rng,
        proximity_threshold=5.0, min_gap=3,
    )
    num_inliers = int(np.sum(lc_labels == 1.0))
    num_outliers = int(np.sum(lc_labels == 0.0))
    assert num_inliers + num_outliers == 10
    assert num_inliers >= 3
    assert num_outliers >= 3


def test_edge_indices_in_range():
    """All injected edge indices must be valid node IDs."""
    rng = np.random.default_rng(3)
    ref = _make_looping_trajectory(30)
    lc_edges, _, _ = inject_labeled_loop_closures(
        ref, num_loop_closures=8, outlier_rate=30,
        outlier_structure="random", rng=rng,
        proximity_threshold=3.0, min_gap=2,
    )
    assert np.all(lc_edges >= 0)
    assert np.all(lc_edges < len(ref))


def test_inlier_pairs_are_proximal():
    """Inlier LCs must be between spatially-close nodes."""
    rng = np.random.default_rng(4)
    ref = _make_looping_trajectory(30)
    lc_edges, _, lc_labels = inject_labeled_loop_closures(
        ref, num_loop_closures=3, outlier_rate=0,
        outlier_structure="random", rng=rng,
        proximity_threshold=2.0, min_gap=3,
    )
    assert np.all(lc_labels == 1.0)
    for k in range(lc_edges.shape[1]):
        i, j = lc_edges[0, k], lc_edges[1, k]
        dist = np.linalg.norm(ref[i, :2] - ref[j, :2])
        assert dist < 2.0


def test_outlier_pairs_are_distant():
    """Outlier LCs must be between spatially-distant nodes."""
    rng = np.random.default_rng(5)
    ref = _make_looping_trajectory(30)
    lc_edges, _, lc_labels = inject_labeled_loop_closures(
        ref, num_loop_closures=5, outlier_rate=100,
        outlier_structure="random", rng=rng,
        outlier_distance_threshold=10.0, min_gap=3,
    )
    assert np.all(lc_labels == 0.0)
    for k in range(lc_edges.shape[1]):
        i, j = lc_edges[0, k], lc_edges[1, k]
        dist = np.linalg.norm(ref[i, :2] - ref[j, :2])
        assert dist > 10.0


def test_min_gap_respected():
    """All LC pairs must satisfy j - i >= min_gap."""
    rng = np.random.default_rng(6)
    ref = _make_looping_trajectory(30)
    min_gap = 5
    lc_edges, _, _ = inject_labeled_loop_closures(
        ref, num_loop_closures=10, outlier_rate=50,
        outlier_structure="random", rng=rng,
        proximity_threshold=5.0, min_gap=min_gap,
    )
    gaps = lc_edges[1] - lc_edges[0]
    assert np.all(gaps >= min_gap)


def test_clustered_structure():
    """Clustered mode should not crash and produce valid output."""
    rng = np.random.default_rng(7)
    ref = _make_looping_trajectory(50)
    lc_edges, lc_meas, lc_labels = inject_labeled_loop_closures(
        ref, num_loop_closures=10, outlier_rate=60,
        outlier_structure="clustered", rng=rng,
        proximity_threshold=5.0, outlier_distance_threshold=15.0,
    )
    assert lc_edges.shape == (2, 10)
    assert lc_meas.shape == (10, 3)
    assert lc_labels.shape == (10,)


def test_uniform_outlier_measurement():
    """Uniform measurement mode should not crash and produce measurements in [-π, π]."""
    rng = np.random.default_rng(8)
    ref = _make_looping_trajectory(30)
    _, lc_meas, _ = inject_labeled_loop_closures(
        ref, num_loop_closures=5, outlier_rate=100,
        outlier_structure="random", rng=rng,
        outlier_measurement="uniform",
    )
    assert np.all(lc_meas >= -np.pi - 1e-6)
    assert np.all(lc_meas <= np.pi + 1e-6)


def test_gaussian_outlier_measurement():
    """Gaussian outlier measurements should differ from true measurements."""
    rng = np.random.default_rng(9)
    ref = _make_looping_trajectory(30)
    _, lc_meas, _ = inject_labeled_loop_closures(
        ref, num_loop_closures=5, outlier_rate=100,
        outlier_structure="random", rng=rng,
        outlier_measurement="gaussian", outlier_offset_std=10.0,
    )
    assert lc_meas.shape == (5, 3)
    assert not np.allclose(lc_meas, 0.0)


def test_reproducible_with_seed():
    """Same seed + same params must produce identical output."""
    params = dict(
        num_loop_closures=5, outlier_rate=40, outlier_structure="random",
        proximity_threshold=3.0, outlier_distance_threshold=10.0,
    )
    ref = _make_looping_trajectory(30)
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    e1, m1, l1 = inject_labeled_loop_closures(ref, rng=rng1, **params)
    e2, m2, l2 = inject_labeled_loop_closures(ref, rng=rng2, **params)
    assert np.array_equal(e1, e2)
    assert np.allclose(m1, m2)
    assert np.array_equal(l1, l2)


def test_insufficient_candidates_raises():
    """When there aren't enough proximal pairs, ValueError should be raised."""
    rng = np.random.default_rng(10)
    ref = _make_straight_trajectory(10)
    with pytest.raises(ValueError, match="candidates"):
        inject_labeled_loop_closures(
            ref, num_loop_closures=100, outlier_rate=0,
            outlier_structure="random", rng=rng,
            proximity_threshold=0.1, min_gap=1,
        )
