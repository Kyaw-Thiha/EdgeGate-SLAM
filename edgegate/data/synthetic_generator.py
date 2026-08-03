from __future__ import annotations
import numpy as np
from edgegate.data.types import PoseGraph
from edgegate.data.se2_utils import compose, inverse_compose, angle_wrap
from edgegate.data.outlier_injection import inject_labeled_loop_closures

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


def generate(
    num_poses: int,
    outlier_rate: float,
    outlier_structure: str,
    seed: int,
    num_loop_closures: int | None = None,
    segment_length: int = 10,
    proximity_threshold: float = 2.0,
    outlier_distance_threshold: float = 10.0,
    outlier_measurement: str = "gaussian",
    outlier_offset_std: float = 5.0,
    min_gap: int = 5,
    info_scale: float = 1.0,
    lc_ratio: int | None = None,
    include_residuals: bool = False,
    randomize_domain: bool = False,
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
        info_scale:                 Scalar multiplier on the LC_INFO matrix
                                    (fixed odometry info is never scaled).
                                    Default 1.0 = no change.
        lc_ratio:                   Target poses-per-LC ratio. When provided,
                                    num_loop_closures = max(1, num_poses // lc_ratio).
                                    Overrides explicit num_loop_closures.

    Odometry edges are always labelled inlier (label=1.0).
    """
    rng = np.random.default_rng(seed)

    # Domain randomization: override fixed params with per-graph draws.
    # Covers the same ranges as the domain sweep (num_poses × lc_ratio × info_scale)
    # plus segment_length for trajectory-topology variation.
    if randomize_domain:
        num_poses = int(rng.choice([100, 500, 1000, 3500]))
        segment_length = int(rng.choice([3, 5, 10]))
        lc_ratio = int(rng.choice([1, 5, 15, 40]))
        info_scale = float(10 ** rng.uniform(-1, 2))  # log-uniform [0.1, 100]

    # Resolve lc_ratio -> num_loop_closures
    if lc_ratio is not None:
        num_loop_closures = max(1, num_poses // lc_ratio)
    elif num_loop_closures is None:
        raise ValueError(
            "Must specify either num_loop_closures or lc_ratio"
        )

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

    # ── 4. Inject labelled loop-closure edges ────────────────────────────────
    lc_edge_index, lc_measurements, lc_labels = inject_labeled_loop_closures(
        reference_poses=gt_poses,
        num_loop_closures=num_loop_closures,
        outlier_rate=outlier_rate,
        outlier_structure=outlier_structure,
        rng=rng,
        outlier_measurement=outlier_measurement,
        outlier_offset_std=outlier_offset_std,
        proximity_threshold=proximity_threshold,
        outlier_distance_threshold=outlier_distance_threshold,
        min_gap=min_gap,
        inlier_noise_std=tuple(LC_NOISE_STD),
    )

    # ── 5. Assemble full PoseGraph ────────────────────────────────────────────
    E_odom = num_poses - 1
    lc_n = num_loop_closures
    E = E_odom + lc_n

    edge_index       = np.zeros((2, E), dtype=np.int64)
    edge_measurement = np.zeros((E, 3))
    edge_info        = np.zeros((E, 6))
    edge_type        = np.zeros(E, dtype=np.int64)
    edge_label       = np.ones(E, dtype=np.float32)  # odometry always inlier

    # Odometry block
    edge_index[0, :E_odom] = np.arange(num_poses - 1)
    edge_index[1, :E_odom] = np.arange(1, num_poses)
    edge_measurement[:E_odom] = odom_meas
    edge_info[:E_odom] = ODOM_INFO
    edge_type[:E_odom] = 0

    # Loop-closure block
    edge_index[:, E_odom:] = lc_edge_index
    edge_measurement[E_odom:] = lc_measurements
    edge_info[E_odom:] = LC_INFO * info_scale
    edge_type[E_odom:] = 1
    edge_label[E_odom:] = lc_labels

    # ── 6. GT residuals (optional — for residual-guided re-weighting) ──────────
    edge_residual = None
    if include_residuals:
        edge_residual = np.zeros((E, 3))
        for e in range(E):
            src, dst = edge_index[0, e], edge_index[1, e]
            true_rel = inverse_compose(gt_poses[src], gt_poses[dst])
            residual = edge_measurement[e] - true_rel
            residual[2] = angle_wrap(residual[2])
            edge_residual[e] = residual

    return PoseGraph(
        node_init=node_init,
        edge_index=edge_index,
        edge_measurement=edge_measurement,
        edge_info=edge_info,
        edge_type=edge_type,
        edge_label=edge_label,
        gt_node_poses=gt_poses,
        edge_residual=edge_residual,
    )
