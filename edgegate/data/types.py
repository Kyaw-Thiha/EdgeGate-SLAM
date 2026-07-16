from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class PoseGraph:
    node_init: np.ndarray         # (N, 3) initial pose guess (x, y, θ)
    edge_index: np.ndarray        # (2, E)
    edge_measurement: np.ndarray  # (E, 3) relative (dx, dy, dθ)
    edge_info: np.ndarray         # (E, 6) upper-tri of 3×3 info matrix [xx, xy, xθ, yy, yθ, θθ]
    edge_type: np.ndarray         # (E,) 0=odometry, 1=loop-closure
    edge_label: np.ndarray | None = None      # (E,) ground-truth inlier; synthetic only, None for real data
    gt_node_poses: np.ndarray | None = None  # (N, 3) noise-free trajectory; synthetic only, None for real data
    manifold: str = "SE2"         # "SE2" | "SE3" — marker for future extension, not branched on yet
