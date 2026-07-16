from __future__ import annotations
import warnings
import numpy as np
from edgegate.data.types import PoseGraph
from edgegate.data.se2_utils import compose, inverse_compose, angle_wrap

# Fixed isotropic information matrices, upper-tri ordering [Ixx, Ixy, Ixθ, Iyy, Iyθ, Iθθ].
# Matches standard PGO benchmark convention (e.g. Intel, M3500).
ODOM_INFO = np.array([500.0, 0.0, 0.0, 500.0, 0.0, 100.0])
LC_INFO   = np.array([100.0, 0.0, 0.0, 100.0, 0.0,  50.0])

# Measurement noise standard deviations [σ_x, σ_y, σ_θ]
ODOM_NOISE_STD = np.array([0.02, 0.02, 0.01])
LC_NOISE_STD   = np.array([0.05, 0.05, 0.02])


def _generate_trajectory(
    num_poses: int, segment_length: int, rng: np.random.Generator
) -> np.ndarray:
    """Manhattan-world trajectory: axis-aligned segments with ±90° turns.

    Segment lengths are randomised in [segment_length//2, segment_length] so the
    robot doesn't revisit at perfectly regular intervals.
    """
    heading_values = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])  # E, N, W, S
    heading_idx = 0

    poses = np.zeros((num_poses, 3))
    poses[0, 2] = heading_values[heading_idx]
    steps_since_turn = 0
    next_turn_at = int(rng.integers(max(1, segment_length // 2), segment_length + 1))

    for i in range(1, num_poses):
        theta = heading_values[heading_idx]
        poses[i, 0] = poses[i - 1, 0] + np.cos(theta)
        poses[i, 1] = poses[i - 1, 1] + np.sin(theta)
        poses[i, 2] = theta
        steps_since_turn += 1

        if steps_since_turn >= next_turn_at:
            heading_idx = (heading_idx + int(rng.choice([-1, 1]))) % 4
            steps_since_turn = 0
            next_turn_at = int(rng.integers(max(1, segment_length // 2), segment_length + 1))

    return poses


def _candidate_pairs(
    poses: np.ndarray, min_gap: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Upper-triangular pose pairs with j - i >= min_gap and their Euclidean distances."""
    N = len(poses)
    ii, jj = np.triu_indices(N, k=min_gap)
    dists = np.linalg.norm(poses[ii, :2] - poses[jj, :2], axis=-1)
    return ii, jj, dists


def _sample_n(
    pairs: list[tuple[int, int]], n: int, rng: np.random.Generator
) -> list[tuple[int, int]]:
    if len(pairs) < n:
        raise ValueError(f"Need {n} pairs but only {len(pairs)} candidates available. "
                         "Try increasing num_poses or adjusting proximity/distance thresholds.")
    idx = rng.choice(len(pairs), size=n, replace=False)
    return [pairs[int(k)] for k in idx]


def generate(
    num_poses: int,
    num_loop_closures: int,
    outlier_rate: float,
    outlier_structure: str,
    seed: int,
    segment_length: int = 10,
    proximity_threshold: float = 2.0,
    outlier_distance_threshold: float = 10.0,
    outlier_measurement: str = "gaussian",
    outlier_offset_std: float = 5.0,
    min_gap: int = 5,
) -> PoseGraph:
    """Generate a synthetic SE(2) pose graph with ground-truth inlier labels.

    Args:
        num_poses:                  Number of robot poses (nodes).
        num_loop_closures:          Total loop-closure edges to inject.
        outlier_rate:               Percentage of loop closures that are outliers (0-100).
        outlier_structure:          "random" | "clustered" — whether outlier source
                                    poses are spread or concentrated in a time window.
        seed:                       RNG seed for full reproducibility.
        segment_length:             Mean Manhattan-world segment length before a turn.
        proximity_threshold:        Max Euclidean distance for inlier LC candidates.
        outlier_distance_threshold: Min Euclidean distance for outlier LC candidates
                                    (models perceptual aliasing: spatially-far pairs
                                    that look similar to the perception front-end).
        outlier_measurement:        "gaussian" (true + large offset, primary/hard case) |
                                    "uniform"  (fully random, easy ablation).
        outlier_offset_std:         Std of the Gaussian offset for outlier measurements.
        min_gap:                    Minimum pose-ID gap for loop-closure candidates
                                    (prevents near-consecutive IDs from being treated
                                    as loop closures rather than odometry).

    Odometry edges are always labelled inlier (label=1.0).
    """
    rng = np.random.default_rng(seed)

    # ── 1. Ground-truth trajectory ────────────────────────────────────────────
    gt_poses = _generate_trajectory(num_poses, segment_length, rng)

    # ── 2. Noisy odometry edges ───────────────────────────────────────────────
    odom_meas = np.zeros((num_poses - 1, 3))
    for i in range(num_poses - 1):
        true_meas = inverse_compose(gt_poses[i], gt_poses[i + 1])
        noisy = true_meas + rng.normal(0.0, ODOM_NOISE_STD)
        noisy[2] = angle_wrap(noisy[2])
        odom_meas[i] = noisy

    # ── 3. Initial pose guess: chain-compose noisy odometry from origin ───────
    node_init = np.zeros((num_poses, 3))
    for i in range(num_poses - 1):
        node_init[i + 1] = compose(node_init[i], odom_meas[i])

    # ── 4. Candidate pairs ────────────────────────────────────────────────────
    ii, jj, dists = _candidate_pairs(gt_poses, min_gap)
    proximal_pairs = [(int(ii[k]), int(jj[k])) for k in np.where(dists < proximity_threshold)[0]]
    distant_pairs  = [(int(ii[k]), int(jj[k])) for k in np.where(dists > outlier_distance_threshold)[0]]

    # ── 5. Split counts ───────────────────────────────────────────────────────
    num_outliers = int(round(num_loop_closures * outlier_rate / 100.0))
    num_inliers  = num_loop_closures - num_outliers

    # ── 6. Sample inlier LCs from spatially-close pairs ──────────────────────
    inlier_pairs = _sample_n(proximal_pairs, num_inliers, rng) if num_inliers > 0 else []

    # ── 7. Sample outlier LCs from spatially-distant pairs ───────────────────
    if num_outliers > 0:
        if outlier_structure == "clustered":
            window = max(1, int(num_poses * 0.3))
            t0 = int(rng.integers(0, max(1, num_poses - window)))
            windowed = [(i, j) for (i, j) in distant_pairs if t0 <= i < t0 + window]
            if len(windowed) < num_outliers:
                warnings.warn(
                    f"Clustered outliers: only {len(windowed)} candidates in window "
                    f"[{t0}, {t0 + window}); falling back to global distant pairs."
                )
                windowed = distant_pairs
            outlier_pairs = _sample_n(windowed, num_outliers, rng)
        else:
            outlier_pairs = _sample_n(distant_pairs, num_outliers, rng)
    else:
        outlier_pairs = []

    # ── 8. Build loop-closure edge arrays ─────────────────────────────────────
    lc_n = num_inliers + num_outliers
    lc_edge_index   = np.zeros((2, lc_n), dtype=np.int64)
    lc_measurements = np.zeros((lc_n, 3))
    lc_labels       = np.zeros(lc_n, dtype=np.float32)

    for k, (i, j) in enumerate(inlier_pairs):
        lc_edge_index[:, k] = [i, j]
        true_meas = inverse_compose(gt_poses[i], gt_poses[j])
        meas = true_meas + rng.normal(0.0, LC_NOISE_STD)
        meas[2] = angle_wrap(meas[2])
        lc_measurements[k] = meas
        lc_labels[k] = 1.0

    for k, (i, j) in enumerate(outlier_pairs):
        idx = num_inliers + k
        lc_edge_index[:, idx] = [i, j]
        if outlier_measurement == "gaussian":
            true_meas = inverse_compose(gt_poses[i], gt_poses[j])
            meas = true_meas + rng.normal(0.0, outlier_offset_std, 3)
        else:  # uniform — easy ablation
            meas = rng.uniform(-np.pi, np.pi, 3)
        meas[2] = angle_wrap(meas[2])
        lc_measurements[idx] = meas
        lc_labels[idx] = 0.0

    # ── 9. Assemble full PoseGraph ────────────────────────────────────────────
    E_odom = num_poses - 1
    E = E_odom + lc_n

    edge_index      = np.zeros((2, E), dtype=np.int64)
    edge_measurement = np.zeros((E, 3))
    edge_info       = np.zeros((E, 6))
    edge_type       = np.zeros(E, dtype=np.int64)
    edge_label      = np.ones(E, dtype=np.float32)  # odometry always inlier

    # Odometry block
    edge_index[0, :E_odom] = np.arange(num_poses - 1)
    edge_index[1, :E_odom] = np.arange(1, num_poses)
    edge_measurement[:E_odom] = odom_meas
    edge_info[:E_odom] = ODOM_INFO
    edge_type[:E_odom] = 0

    # Loop-closure block
    edge_index[:, E_odom:] = lc_edge_index
    edge_measurement[E_odom:] = lc_measurements
    edge_info[E_odom:] = LC_INFO
    edge_type[E_odom:] = 1
    edge_label[E_odom:] = lc_labels

    return PoseGraph(
        node_init=node_init,
        edge_index=edge_index,
        edge_measurement=edge_measurement,
        edge_info=edge_info,
        edge_type=edge_type,
        edge_label=edge_label,
    )
