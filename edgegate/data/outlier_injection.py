from __future__ import annotations
import warnings
import numpy as np
from edgegate.data.se2_utils import inverse_compose, angle_wrap


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

    ii, jj, dists = _candidate_pairs(reference_poses, min_gap)
    proximal_pairs = [
        (int(ii[k]), int(jj[k]))
        for k in np.where(dists < proximity_threshold)[0]
    ]
    distant_pairs = [
        (int(ii[k]), int(jj[k]))
        for k in np.where(dists > outlier_distance_threshold)[0]
    ]

    num_outliers = int(round(num_loop_closures * outlier_rate / 100.0))
    num_inliers = num_loop_closures - num_outliers

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
