from __future__ import annotations
from pathlib import Path
import numpy as np
from edgegate.data.types import PoseGraph


def load_g2o(path: str | Path) -> PoseGraph:
    """Parse a .g2o file (SE(2) only) into a PoseGraph.

    Edge type is inferred from node-ID distance:
        |id2 - id1| == 1  →  odometry (type 0)
        otherwise         →  loop-closure (type 1)

    edge_label is always None — real .g2o files have no ground-truth inlier labels.
    """
    vertices: dict[int, list[float]] = {}
    raw_edges: list[tuple] = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            tag = parts[0]
            if tag == "VERTEX_SE2":
                vid = int(parts[1])
                vertices[vid] = [float(parts[2]), float(parts[3]), float(parts[4])]
            elif tag == "EDGE_SE2":
                id1, id2 = int(parts[1]), int(parts[2])
                meas = [float(parts[3]), float(parts[4]), float(parts[5])]
                info = [float(parts[6 + k]) for k in range(6)]
                raw_edges.append((id1, id2, meas, info))

    sorted_ids = sorted(vertices)
    id_to_idx = {vid: i for i, vid in enumerate(sorted_ids)}

    node_init = np.array(
        [vertices[vid] for vid in sorted_ids], dtype=np.float64
    )  # (N, 3)

    E = len(raw_edges)
    edge_index = np.empty((2, E), dtype=np.int64)
    edge_measurement = np.empty((E, 3), dtype=np.float64)
    edge_info = np.empty((E, 6), dtype=np.float64)
    edge_type = np.empty(E, dtype=np.int64)

    for i, (id1, id2, meas, info) in enumerate(raw_edges):
        edge_index[0, i] = id_to_idx[id1]
        edge_index[1, i] = id_to_idx[id2]
        edge_measurement[i] = meas
        edge_info[i] = info
        edge_type[i] = 0 if abs(id2 - id1) == 1 else 1

    return PoseGraph(
        node_init=node_init,
        edge_index=edge_index,
        edge_measurement=edge_measurement,
        edge_info=edge_info,
        edge_type=edge_type,
        edge_label=None,
    )


def save_g2o(
    graph: PoseGraph,
    path: str | Path,
    poses: np.ndarray | None = None,
) -> None:
    """Write a PoseGraph to .g2o format.

    Args:
        poses: (N, 3) optimised poses to write as VERTEX_SE2 positions.
               Falls back to graph.node_init when None.
    """
    node_poses = poses if poses is not None else graph.node_init
    lines: list[str] = []

    for i, (x, y, theta) in enumerate(node_poses):
        lines.append(f"VERTEX_SE2 {i} {x:.10g} {y:.10g} {theta:.10g}")

    for e in range(graph.edge_index.shape[1]):
        i1 = int(graph.edge_index[0, e])
        i2 = int(graph.edge_index[1, e])
        dx, dy, dtheta = graph.edge_measurement[e]
        info_str = " ".join(f"{v:.10g}" for v in graph.edge_info[e])
        lines.append(
            f"EDGE_SE2 {i1} {i2} {dx:.10g} {dy:.10g} {dtheta:.10g} {info_str}"
        )

    Path(path).write_text("\n".join(lines) + "\n")
