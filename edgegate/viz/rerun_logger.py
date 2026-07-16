from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import rerun as rr
from rerun import RecordingStream


def _conf_to_color(c: float) -> tuple[int, int, int]:
    """Map confidence in [0, 1] to an RGB colour: red=0 → green=1."""
    r = int(220 + (50 - 220) * c)
    g = int(50 + (200 - 50) * c)
    return (r, g, 50)


def _log(rec: Optional[RecordingStream], entity: str, component) -> None:
    if rec is not None:
        rec.log(entity, component)
    else:
        rr.log(entity, component)


def _set_time(rec: Optional[RecordingStream], epoch: int) -> None:
    if rec is not None:
        rec.set_time_sequence("epoch", epoch)
    else:
        rr.set_time_sequence("epoch", epoch)


def log_pose_graph(
    graph,
    poses: np.ndarray,
    confidence: np.ndarray,
    *,
    epoch: Optional[int] = None,
    rec: Optional[RecordingStream] = None,
) -> None:
    """Log a pose graph with edge confidence heatmap to Rerun.

    Args:
        graph: PoseGraph providing topology, optional GT and edge labels.
        poses: (N, 3) solved poses [x, y, theta].
        confidence: (E,) per-edge scores in [0, 1]; odometry edges should be 1.0.
        epoch: if set, advances the "epoch" timeline before logging.
        rec: RecordingStream to log to; uses the global stream if None.
    """
    if epoch is not None:
        _set_time(rec, epoch)

    N = poses.shape[0]
    edge_index = np.asarray(graph.edge_index)   # (2, E)
    edge_type = np.asarray(graph.edge_type)      # (E,)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    node_init = np.asarray(graph.node_init)
    _log(rec, "trajectory/initial", rr.Points2D(
        positions=node_init[:, :2],
        colors=np.full((N, 3), [150, 200, 255], dtype=np.uint8),
        radii=0.08,
    ))
    _log(rec, "trajectory/solved", rr.Points2D(
        positions=poses[:, :2],
        colors=np.full((N, 3), [255, 255, 255], dtype=np.uint8),
        radii=0.12,
    ))
    if graph.gt_node_poses is not None:
        gt = np.asarray(graph.gt_node_poses)
        _log(rec, "trajectory/gt", rr.Points2D(
            positions=gt[:, :2],
            colors=np.full((N, 3), [100, 220, 100], dtype=np.uint8),
            radii=0.10,
        ))

    # ── Edges ─────────────────────────────────────────────────────────────────
    odom_idx = np.where(edge_type == 0)[0]
    lc_idx = np.where(edge_type == 1)[0]

    if len(odom_idx) > 0:
        strips = [
            [poses[edge_index[0, i], :2].tolist(), poses[edge_index[1, i], :2].tolist()]
            for i in odom_idx
        ]
        _log(rec, "edges/odometry", rr.LineStrips2D(
            strips=strips,
            colors=np.full((len(odom_idx), 3), [120, 120, 120], dtype=np.uint8),
            radii=0.02,
        ))

    if len(lc_idx) > 0:
        strips = [
            [poses[edge_index[0, i], :2].tolist(), poses[edge_index[1, i], :2].tolist()]
            for i in lc_idx
        ]
        colors = np.array(
            [_conf_to_color(float(confidence[i])) for i in lc_idx], dtype=np.uint8
        )
        _log(rec, "edges/loop_closure", rr.LineStrips2D(
            strips=strips,
            colors=colors,
            radii=0.03,
        ))

    if graph.edge_label is not None:
        edge_label = np.asarray(graph.edge_label)
        inlier_idx = np.where((edge_type == 1) & (edge_label >= 0.5))[0]
        if len(inlier_idx) > 0:
            strips = [
                [poses[edge_index[0, i], :2].tolist(), poses[edge_index[1, i], :2].tolist()]
                for i in inlier_idx
            ]
            _log(rec, "edges/gt_inliers", rr.LineStrips2D(
                strips=strips,
                colors=np.full((len(inlier_idx), 3), [50, 220, 50], dtype=np.uint8),
                radii=0.04,
            ))


def log_metrics(
    metrics_log: list[dict],
    *,
    rec: Optional[RecordingStream] = None,
) -> None:
    """Batch-log scalar training metrics from metrics.json onto the epoch timeline.

    Args:
        metrics_log: list of per-epoch dicts with keys epoch, train_loss, val_f1, val_ate.
        rec: RecordingStream to log to; uses the global stream if None.
    """
    from rerun.components import ScalarBatch
    from rerun import TimeSequenceColumn

    epochs = np.array([r["epoch"] for r in metrics_log], dtype=np.int64)
    time_col = TimeSequenceColumn("epoch", epochs)
    send = rec.send_columns if rec is not None else rr.send_columns

    losses = np.array([r.get("train_loss", float("nan")) for r in metrics_log])
    send("metrics/train_loss", [time_col], [ScalarBatch(losses)])

    f1_vals = [r.get("val_f1") for r in metrics_log]
    if any(v is not None for v in f1_vals):
        f1_arr = np.array([v if v is not None else float("nan") for v in f1_vals])
        send("metrics/val_f1", [time_col], [ScalarBatch(f1_arr)])

    ate_vals = [r.get("val_ate") for r in metrics_log]
    if any(v is not None for v in ate_vals):
        ate_arr = np.array([v if v is not None else float("nan") for v in ate_vals])
        send("metrics/val_ate", [time_col], [ScalarBatch(ate_arr)])


def log_laser_scans(
    carmen_log_path: str,
    poses: np.ndarray,
    *,
    max_range: float = 8.0,
    rec: Optional[RecordingStream] = None,
) -> None:
    """Parse a Carmen FLASER log and log scan points as a static world-frame point cloud.

    Each FLASER line is associated with the robot pose at the same index.  Polar
    readings are converted to world-frame Cartesian using the robot pose embedded
    in poses (not the odometry pose in the log, which may have accumulated drift).

    Args:
        carmen_log_path: path to an Intel/MIT Carmen .log file.
        poses: (N, 3) robot poses [x, y, theta] in world frame.
        max_range: readings >= this distance (metres) are filtered as invalid.
        rec: RecordingStream to log to; uses the global stream if None.
    """
    N = poses.shape[0]
    all_points: list[list[float]] = []

    scan_idx = 0
    with open(carmen_log_path, "r") as fh:
        for line in fh:
            if not line.startswith("FLASER"):
                continue
            if scan_idx >= N:
                break
            parts = line.split()
            n_readings = int(parts[1])
            px, py, ptheta = poses[scan_idx, 0], poses[scan_idx, 1], poses[scan_idx, 2]

            for k in range(n_readings):
                r = float(parts[2 + k])
                if r >= max_range:
                    continue
                angle_k = -math.pi / 2.0 + k * math.pi / max(n_readings - 1, 1)
                world_angle = ptheta + angle_k
                all_points.append([
                    px + r * math.cos(world_angle),
                    py + r * math.sin(world_angle),
                ])
            scan_idx += 1

    if not all_points:
        return

    _set_time(rec, 0)
    _log(rec, "scans/laser", rr.Points2D(
        positions=np.array(all_points, dtype=np.float32),
        colors=(180, 180, 180),
        radii=0.02,
    ))


def log_eval_comparison(
    method_dirs: dict[str, str],
    *,
    graph_idx: int = 0,
    spawn: bool = True,
    save_rrd: Optional[str] = None,
) -> None:
    """Overlay multiple evaluation methods on the same test graph (paper-figure viz).

    Args:
        method_dirs: mapping of method name → Hydra evaluate.py output directory.
            Each dir must contain per_graph/graph_NNN/poses.npy + confidence.npy
            and a graph_info.json (written when save_poses=true).
        graph_idx: which test graph to visualise.
        spawn: whether to open the Rerun viewer.
        save_rrd: if given, save the recording to this path.
    """
    METHOD_COLORS: dict[str, tuple[int, int, int]] = {
        "learned": (50,  120, 255),
        "gnc":     (255, 140,  50),
        "uniform": (160,  50, 220),
        "dcs":     ( 50, 200, 180),
    }
    DEFAULT_COLOR = (200, 200, 200)

    rec = rr.new_recording(
        application_id="edgegate-eval-compare", spawn=spawn, make_default=True
    )

    first_dir = next(iter(method_dirs.values()))
    graph_info = json.loads((Path(first_dir) / "graph_info.json").read_text())
    edge_index = np.array(graph_info["edge_index"])
    edge_type = np.array(graph_info["edge_type"])
    gt_raw = graph_info.get("gt_node_poses")
    if gt_raw is not None:
        gt = np.array(gt_raw)
        rr.log("trajectory/gt", rr.Points2D(
            positions=gt[:, :2],
            colors=(100, 220, 100),
            radii=0.10,
        ))

    graph_subdir = f"graph_{graph_idx:03d}"
    lc_idx = np.where(edge_type == 1)[0]

    for method_name, method_dir in method_dirs.items():
        poses_path = Path(method_dir) / "per_graph" / graph_subdir / "poses.npy"
        conf_path = Path(method_dir) / "per_graph" / graph_subdir / "confidence.npy"
        if not poses_path.exists() or not conf_path.exists():
            print(f"  [{method_name}] missing artifacts in {method_dir}/per_graph/{graph_subdir}/ — skipping")
            continue
        poses = np.load(str(poses_path))
        confidence = np.load(str(conf_path))
        color = METHOD_COLORS.get(method_name, DEFAULT_COLOR)

        rr.log(f"methods/{method_name}/trajectory", rr.Points2D(
            positions=poses[:, :2],
            colors=color,
            radii=0.12,
        ))
        if len(lc_idx) > 0:
            strips = [
                [poses[edge_index[0, i], :2].tolist(), poses[edge_index[1, i], :2].tolist()]
                for i in lc_idx
            ]
            colors = np.array(
                [_conf_to_color(float(confidence[i])) for i in lc_idx], dtype=np.uint8
            )
            rr.log(f"methods/{method_name}/edges", rr.LineStrips2D(
                strips=strips, colors=colors, radii=0.03,
            ))

    if save_rrd is not None:
        rec.save(save_rrd)


def log_run(
    run_dir: str,
    *,
    carmen_log_path: Optional[str] = None,
    spawn: bool = True,
    save_rrd: Optional[str] = None,
) -> None:
    """Replay a full training run: metrics timeline + per-checkpoint pose graph.

    Args:
        run_dir: Hydra training output directory with graph_info.json, metrics.json,
            and checkpoints/epoch_NNN/{poses.npy, edge_weights.npy}.
        carmen_log_path: optional path to Carmen .log for static laser scan overlay.
        spawn: whether to open the Rerun viewer.
        save_rrd: if given, save the recording to this path.
    """
    from edgegate.data.types import PoseGraph

    run_path = Path(run_dir)
    rec = rr.new_recording(
        application_id="edgegate-slam", spawn=spawn, make_default=True
    )

    graph_info_path = run_path / "graph_info.json"
    if not graph_info_path.exists():
        raise FileNotFoundError(f"graph_info.json not found in {run_dir}")
    graph_info = json.loads(graph_info_path.read_text())

    edge_index_arr = np.array(graph_info["edge_index"], dtype=np.int64)
    E = edge_index_arr.shape[1]
    edge_label_raw = graph_info.get("edge_label")
    graph = PoseGraph(
        node_init=np.array(graph_info["node_init"], dtype=np.float32),
        edge_index=edge_index_arr,
        edge_measurement=np.zeros((E, 3), dtype=np.float32),
        edge_info=np.zeros((E, 6), dtype=np.float32),
        edge_type=np.array(graph_info["edge_type"], dtype=np.int64),
        edge_label=(
            np.array(edge_label_raw, dtype=np.float32) if edge_label_raw is not None else None
        ),
        gt_node_poses=(
            np.array(graph_info["gt_node_poses"], dtype=np.float32)
            if graph_info.get("gt_node_poses") is not None
            else None
        ),
    )

    metrics_path = run_path / "metrics.json"
    if metrics_path.exists():
        metrics_log = json.loads(metrics_path.read_text())
        log_metrics(metrics_log, rec=rec)

    ckpt_root = run_path / "checkpoints"
    ckpt_dirs = (
        sorted(ckpt_root.glob("epoch_*"), key=lambda p: int(p.name.split("_")[1]))
        if ckpt_root.exists()
        else []
    )

    first_poses: Optional[np.ndarray] = None
    for ckpt_dir in ckpt_dirs:
        epoch = int(ckpt_dir.name.split("_")[1])
        poses_path = ckpt_dir / "poses.npy"
        weights_path = ckpt_dir / "edge_weights.npy"
        if not poses_path.exists() or not weights_path.exists():
            continue
        poses = np.load(str(poses_path))
        confidence = np.load(str(weights_path))
        if first_poses is None:
            first_poses = poses
        log_pose_graph(graph, poses, confidence, epoch=epoch, rec=rec)

    if carmen_log_path is not None:
        base_poses = first_poses if first_poses is not None else graph.node_init
        log_laser_scans(carmen_log_path, base_poses, rec=rec)

    if save_rrd is not None:
        rec.save(save_rrd)
