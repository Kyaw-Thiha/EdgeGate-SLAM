from __future__ import annotations
import warnings
import numpy as np
from scipy.spatial import cKDTree
from edgegate.data.se2_utils import inverse_compose, angle_wrap


def _proximal_pairs(
    poses: np.ndarray, min_gap: int, proximity_threshold: float
) -> list[tuple[int, int]]:
    """Pose pairs within proximity_threshold with index gap >= min_gap. O(N log N + output)."""
    tree = cKDTree(poses[:, :2])
    result = []
    for i, j in tree.query_pairs(r=proximity_threshold):
        lo, hi = (i, j) if i < j else (j, i)
        if hi - lo >= min_gap:
            result.append((lo, hi))
    result.sort()
    return result


def _distant_pairs(
    poses: np.ndarray,
    min_gap: int,
    outlier_distance_threshold: float,
    n_needed: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Random-sample candidate pairs beyond outlier_distance_threshold. O(n_sample)."""
    if n_needed == 0:
        return []
    N = len(poses)
    n_sample = min(max(n_needed * 20, 2000), N * (N - 1) // 2)
    # Draw i uniformly from [0, N-min_gap); j from [i+min_gap, N-1]
    i_arr = rng.integers(0, N - min_gap, size=n_sample)
    max_offset = (N - min_gap - i_arr).astype(np.int64)  # always >= 1
    offsets = (rng.random(n_sample) * max_offset).astype(np.int64)
    j_arr = i_arr + min_gap + offsets
    dists = np.linalg.norm(poses[i_arr, :2] - poses[j_arr, :2], axis=-1)
    mask = dists > outlier_distance_threshold
    seen: set[tuple[int, int]] = set()
    result = []
    for i, j in zip(i_arr[mask].tolist(), j_arr[mask].tolist()):
        pair = (int(i), int(j))
        if pair not in seen:
            seen.add(pair)
            result.append(pair)
    return result


def _sample_n(
    pairs: list[tuple[int, int]], n: int, rng: np.random.Generator
) -> list[tuple[int, int]]:
    if len(pairs) < n:
        raise ValueError(
            f"Need {n} pairs but only {len(pairs)} candidates available. "
            "Try increasing num_poses or adjusting proximity/distance thresholds."
        )
    idx = rng.choice(len(pairs), size=n, replace=False)
    return [pairs[int(k)] for k in idx]


def inject_labeled_loop_closures(
    reference_poses: np.ndarray,
    num_loop_closures: int,
    outlier_rate: float,
    outlier_structure: str,
    rng: np.random.Generator,
    outlier_measurement: str = "gaussian",
    outlier_offset_std: float = 5.0,
    proximity_threshold: float = 2.0,
    outlier_distance_threshold: float = 10.0,
    min_gap: int = 5,
    inlier_noise_std: tuple[float, float, float] = (0.05, 0.05, 0.02),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inject loop-closure edges with known inlier/outlier labels.

    Inlier LCs are sampled from spatially-close pose pairs; outlier LCs are
    sampled from spatially-distant pairs (modelling perceptual aliasing).
    Returns (lc_edge_index, lc_measurements, lc_labels) — callers append these
    to the existing graph edge arrays.

    Args:
        reference_poses:     (N, 3) poses used for proximity thresholding and
                             true relative measurement computation. For synthetic
                             data this is the noise-free gt_poses; for real
                             benchmarks without ground truth, pass the optimized
                             (clean-solve) trajectory.
        num_loop_closures:   Total loop-closure edges to inject.
        outlier_rate:        Percentage of LCs that are outliers (0--100).
        outlier_structure:   "random" | "clustered".
        rng:                 Seeded numpy Generator instance.
        outlier_measurement: "gaussian" (true + offset) | "uniform" (fully random).
        outlier_offset_std:  Std of Gaussian offset for outlier measurements.
        proximity_threshold: Max Euclidean distance for inlier LC pair candidates.
        outlier_distance_threshold: Min Euclidean distance for outlier candidates.
        min_gap:             Minimum pose-index gap for candidate pairs.
        inlier_noise_std:    (σ_x, σ_y, σ_θ) measurement noise for inlier LCs.
    """
    assert outlier_structure in ("random", "clustered")
    assert outlier_measurement in ("gaussian", "uniform")
    inlier_noise = np.array(inlier_noise_std)
    N = reference_poses.shape[0]

    num_outliers = int(round(num_loop_closures * outlier_rate / 100.0))
    num_inliers = num_loop_closures - num_outliers

    proximal_pairs = _proximal_pairs(reference_poses, min_gap, proximity_threshold)
    distant_pairs = _distant_pairs(
        reference_poses, min_gap, outlier_distance_threshold,
        n_needed=num_outliers, rng=rng,
    )

    inlier_pairs = (
        _sample_n(proximal_pairs, num_inliers, rng) if num_inliers > 0 else []
    )

    if num_outliers > 0:
        if outlier_structure == "clustered":
            window = max(1, int(N * 0.3))
            t0 = int(rng.integers(0, max(1, N - window)))
            windowed = [
                (i, j) for (i, j) in distant_pairs if t0 <= i < t0 + window
            ]
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

    total = num_inliers + num_outliers
    lc_edge_index = np.zeros((2, total), dtype=np.int64)
    lc_measurements = np.zeros((total, 3))
    lc_labels = np.zeros(total, dtype=np.float32)

    for k, (i, j) in enumerate(inlier_pairs):
        lc_edge_index[:, k] = [i, j]
        true_meas = inverse_compose(reference_poses[i], reference_poses[j])
        meas = true_meas + rng.normal(0.0, inlier_noise)
        meas[2] = angle_wrap(meas[2])
        lc_measurements[k] = meas
        lc_labels[k] = 1.0

    for k, (i, j) in enumerate(outlier_pairs):
        idx = num_inliers + k
        lc_edge_index[:, idx] = [i, j]
        if outlier_measurement == "gaussian":
            true_meas = inverse_compose(reference_poses[i], reference_poses[j])
            meas = true_meas + rng.normal(0.0, outlier_offset_std, 3)
        else:
            meas = rng.uniform(-np.pi, np.pi, 3)
        meas[2] = angle_wrap(meas[2])
        lc_measurements[idx] = meas
        lc_labels[idx] = 0.0

    return lc_edge_index, lc_measurements, lc_labels
