#!/usr/bin/env python
"""Compute per-benchmark domain-shift statistics.

Produces data/domain_shift.json with per-benchmark:
  - Edge-type ratio (odometry : organic loop-closure)
  - Information matrix diagonal stats (Ixx, Iyy, Itheta per edge type)
  - Information matrix off-diagonal stats (Ixy, Ixtheta, Iytheta per edge type)
  - Trajectory properties from reference solve (path length, diameter, avg LC dist)

Run once, commit the output, never recompute.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from edgegate.data.g2o_io import load_g2o
from edgegate.solvers.gtsam_solver import GTSAMSolver


BENCHMARKS = {
    "intel": ROOT / "data" / "raw" / "intel.g2o",
    "m3500": ROOT / "data" / "raw" / "M3500.g2o",
    "mit": ROOT / "data" / "raw" / "MIT.g2o",
    "csail": ROOT / "data" / "raw" / "CSAIL.g2o",
    "city10000": ROOT / "data" / "raw" / "city10000.g2o",
}


def _dist_stats(values: np.ndarray) -> dict:
    return {
        "min": float(np.min(values)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def compute_benchmark_stats(g2o_path: Path) -> dict:
    pg = load_g2o(str(g2o_path))

    odom_mask = pg.edge_type == 0
    lc_mask = pg.edge_type == 1

    num_odom = int(odom_mask.sum())
    num_lc = int(lc_mask.sum())

    diag_cols = {"Ixx": 0, "Iyy": 3, "Itheta": 5}
    info_diag_odom = {}
    info_diag_lc = {}
    for name, col in diag_cols.items():
        if num_odom > 0:
            info_diag_odom[name] = _dist_stats(pg.edge_info[odom_mask, col])
        if num_lc > 0:
            info_diag_lc[name] = _dist_stats(pg.edge_info[lc_mask, col])

    offdiag_cols = {"Ixy": 1, "Ixtheta": 2, "Iytheta": 4}
    info_offdiag_odom = {}
    info_offdiag_lc = {}
    for name, col in offdiag_cols.items():
        if num_odom > 0:
            info_offdiag_odom[name] = _dist_stats(pg.edge_info[odom_mask, col])
        if num_lc > 0:
            info_offdiag_lc[name] = _dist_stats(pg.edge_info[lc_mask, col])

    solver = GTSAMSolver(kernel="none")
    edge_weights = torch.ones(pg.edge_index.shape[1], dtype=torch.float64)
    poses, converged, iters, cost = solver.solve(
        pg, edge_weights, max_iterations=200
    )

    positions = poses[:, :2]
    path_length = float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
    diameter = float(np.max(np.linalg.norm(
        positions[:, None] - positions[None, :], axis=-1
    )))

    if num_lc > 0:
        lc_dists = np.linalg.norm(
            positions[pg.edge_index[1, lc_mask]]
            - positions[pg.edge_index[0, lc_mask]],
            axis=1,
        )
        avg_lc_dist = float(np.mean(lc_dists))
    else:
        avg_lc_dist = 0.0

    return {
        "num_nodes": int(pg.node_init.shape[0]),
        "num_odom_edges": num_odom,
        "num_organic_lc_edges": num_lc,
        "edge_type_ratio_odom_per_lc": round(num_odom / max(num_lc, 1), 2),
        "info_diag_odom": info_diag_odom,
        "info_diag_lc": info_diag_lc,
        "info_offdiag_odom": info_offdiag_odom,
        "info_offdiag_lc": info_offdiag_lc,
        "trajectory_path_length": round(path_length, 2),
        "trajectory_diameter": round(diameter, 2),
        "avg_lc_endpoint_distance": round(avg_lc_dist, 2),
        "reference_solve_converged": converged,
        "reference_solve_iterations": iters,
        "reference_solve_cost": round(cost, 2),
    }


def main() -> None:
    results = {}
    for name, path in BENCHMARKS.items():
        if not path.exists():
            print(f"  SKIP {name}: {path} not found")
            continue
        print(f"  Processing {name} ...")
        try:
            results[name] = compute_benchmark_stats(path)
        except Exception as e:
            print(f"  FAIL {name}: {e}")
            results[name] = {"error": str(e)}

    out_path = ROOT / "data" / "domain_shift.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
