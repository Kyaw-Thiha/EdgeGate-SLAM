from __future__ import annotations

import gzip
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


def _conf_to_method_color(
    c: float, method_color: tuple[int, int, int]
) -> tuple[int, int, int]:
    """Per-panel LC edge color: crimson=rejected, method color=accepted.

    Blends from crimson (conf=0) to the method's pastel hue (conf=1) so each
    panel's accepted edges are instantly recognisable by method color.
    """
    r0, g0, b0 = 200, 40, 60      # crimson for rejected
    r1, g1, b1 = method_color
    return (
        int(r0 + (r1 - r0) * c),
        int(g0 + (g1 - g0) * c),
        int(b0 + (b1 - b0) * c),
    )


def _log(rec: Optional[RecordingStream], entity: str, component) -> None:
    if rec is not None:
        rec.log(entity, component)
    else:
        rr.log(entity, component)


def _set_time(rec: Optional[RecordingStream], epoch: int) -> None:
    rr.set_time("epoch", sequence=epoch, recording=rec)


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
        colors=np.full((N, 3), [120, 160, 210], dtype=np.uint8),
        radii=0.10,
    ))
    _log(rec, "trajectory/solved", rr.Points2D(
        positions=poses[:, :2],
        colors=np.full((N, 3), [230, 230, 240], dtype=np.uint8),
        radii=0.16,
    ))
    if graph.gt_node_poses is not None:
        gt = np.asarray(graph.gt_node_poses)
        _log(rec, "trajectory/gt", rr.Points2D(
            positions=gt[:, :2],
            colors=np.full((N, 3), [140, 240, 140], dtype=np.uint8),
            radii=0.14,
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
        accepted = [i for i in lc_idx if float(confidence[i]) >= 0.5]
        rejected = [i for i in lc_idx if float(confidence[i]) < 0.5]
        if accepted:
            _log(rec, "edges/lc_accepted", rr.LineStrips2D(
                strips=[[poses[edge_index[0, i], :2].tolist(), poses[edge_index[1, i], :2].tolist()] for i in accepted],
                colors=(100, 230, 120),   # green
                radii=0.04,
            ))
        if rejected:
            _log(rec, "edges/lc_rejected", rr.LineStrips2D(
                strips=[[poses[edge_index[0, i], :2].tolist(), poses[edge_index[1, i], :2].tolist()] for i in rejected],
                colors=(210, 50, 65),     # crimson
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
    epochs = np.array([r["epoch"] for r in metrics_log], dtype=np.int64)
    time_col = rr.TimeColumn("epoch", sequence=epochs)
    send = rec.send_columns if rec is not None else rr.send_columns

    losses = np.array([r.get("train_loss", float("nan")) for r in metrics_log])
    send("metrics/train_loss", [time_col], rr.Scalars.columns(scalars=losses))

    f1_vals = [r.get("val_f1") for r in metrics_log]
    if any(v is not None for v in f1_vals):
        f1_arr = np.array([v if v is not None else float("nan") for v in f1_vals])
        send("metrics/val_f1", [time_col], rr.Scalars.columns(scalars=f1_arr))

    ate_vals = [r.get("val_ate") for r in metrics_log]
    if any(v is not None for v in ate_vals):
        ate_arr = np.array([v if v is not None else float("nan") for v in ate_vals])
        send("metrics/val_ate", [time_col], rr.Scalars.columns(scalars=ate_arr))


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
    opener = gzip.open if str(carmen_log_path).endswith(".gz") else open
    with opener(carmen_log_path, "rt") as fh:
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

    _log(rec, "scans/laser", rr.Points2D(
        positions=np.array(all_points, dtype=np.float32),
        colors=(55, 55, 60),
        radii=0.015,
    ))


def log_eval_comparison(
    method_dirs: dict[str, str],
    *,
    graph_idx: int = 0,
    spawn: bool = True,
    save_rrd: Optional[str] = None,
    carmen_log_path: Optional[str] = None,
) -> None:
    """Side-by-side panel comparison of evaluation methods (paper-figure viz).

    Each method gets its own Rerun panel showing the laser scan background,
    GT trajectory, and that method's solved trajectory + LC edge confidence.

    Args:
        method_dirs: mapping of method name → Hydra evaluate.py output directory.
            Each dir must contain per_graph/graph_NNN/poses.npy + confidence.npy
            and a graph_info.json (written when save_poses=true).
        graph_idx: which test graph to visualise.
        spawn: whether to open the Rerun viewer.
        save_rrd: if given, save the recording to this path.
        carmen_log_path: optional Carmen .log / .log.gz for laser scan background.
    """
    import rerun.blueprint as rrb

    # Node colors — medium pastels
    METHOD_COLORS: dict[str, tuple[int, int, int]] = {
        "learned": (100, 160, 240),   # pastel blue
        "gnc":     (240, 175,  85),   # pastel amber
        "uniform": (185, 125, 245),   # pastel violet
        "dcs":     ( 80, 215, 175),   # pastel teal
    }
    # LC accepted colors — warm/complementary hues vs the cool node palette
    # so accepted edges pop clearly against the trajectory dots
    METHOD_LC_COLORS: dict[str, tuple[int, int, int]] = {
        "learned": (255, 190,  80),   # warm amber  — against blue nodes
        "gnc":     (100, 230, 255),   # cool cyan   — against amber nodes
        "uniform": (180, 255,  90),   # lime green  — against violet nodes
        "dcs":     (255, 140, 185),   # warm pink   — against teal nodes
    }
    LC_REJECTED_COLOR: tuple[int, int, int] = (210, 50, 65)   # crimson
    DEFAULT_COLOR = (210, 210, 210)
    CONF_THRESHOLD = 0.5  # confidence boundary between accepted and rejected

    # Build blueprint after entities are logged so Rerun resolves paths correctly.
    # Layout: left column = overview of all methods; right column = one panel per method.
    method_names = list(method_dirs.keys())

    rec = rr.RecordingStream(application_id="edgegate-eval-compare", make_default=True)
    if spawn:
        rec.spawn()

    # Log everything without a timeline — compare mode is a static snapshot.
    # Call reset_time so no stray epoch dimension causes entities to vanish.
    rec.reset_time()

    first_dir = next(iter(method_dirs.values()))
    graph_info = json.loads((Path(first_dir) / "graph_info.json").read_text())
    edge_index = np.array(graph_info["edge_index"])
    edge_type = np.array(graph_info["edge_type"])
    gt_raw = graph_info.get("gt_node_poses")
    if gt_raw is not None:
        gt = np.array(gt_raw)
        rr.log("trajectory/gt", rr.Points2D(
            positions=gt[:, :2],
            colors=(140, 240, 140),
            radii=0.18,
        ))

    graph_subdir = f"graph_{graph_idx:03d}"
    odom_idx = np.where(edge_type == 0)[0]
    lc_idx = np.where(edge_type == 1)[0]

    first_poses: Optional[np.ndarray] = None
    for method_name, method_dir in method_dirs.items():
        poses_path = Path(method_dir) / "per_graph" / graph_subdir / "poses.npy"
        conf_path = Path(method_dir) / "per_graph" / graph_subdir / "confidence.npy"
        if not poses_path.exists() or not conf_path.exists():
            print(f"  [{method_name}] missing artifacts in {method_dir}/per_graph/{graph_subdir}/ — skipping")
            continue
        poses = np.load(str(poses_path))
        confidence = np.load(str(conf_path))
        if first_poses is None:
            first_poses = poses
        color = METHOD_COLORS.get(method_name, DEFAULT_COLOR)
        lc_color = METHOD_LC_COLORS.get(method_name, DEFAULT_COLOR)

        rr.log(f"methods/{method_name}/nodes", rr.Points2D(
            positions=poses[:, :2],
            colors=color,
            radii=0.20,
        ))
        if len(odom_idx) > 0:
            odom_strips = [
                [poses[edge_index[0, i], :2].tolist(), poses[edge_index[1, i], :2].tolist()]
                for i in odom_idx
            ]
            path_color = tuple(max(0, c - 30) for c in color)
            rr.log(f"methods/{method_name}/path", rr.LineStrips2D(
                strips=odom_strips, colors=path_color, radii=0.04,
            ))

        if len(lc_idx) > 0:
            accepted = [i for i in lc_idx if float(confidence[i]) >= CONF_THRESHOLD]
            rejected = [i for i in lc_idx if float(confidence[i]) < CONF_THRESHOLD]

            if accepted:
                strips = [
                    [poses[edge_index[0, i], :2].tolist(), poses[edge_index[1, i], :2].tolist()]
                    for i in accepted
                ]
                rr.log(f"methods/{method_name}/lc_accepted", rr.LineStrips2D(
                    strips=strips, colors=lc_color, radii=0.06,
                ))
            if rejected:
                strips = [
                    [poses[edge_index[0, i], :2].tolist(), poses[edge_index[1, i], :2].tolist()]
                    for i in rejected
                ]
                rr.log(f"methods/{method_name}/lc_rejected", rr.LineStrips2D(
                    strips=strips, colors=LC_REJECTED_COLOR, radii=0.04,
                ))

    if carmen_log_path is not None and first_poses is not None:
        log_laser_scans(carmen_log_path, first_poses, rec=rec)

    # Send blueprint after all entities exist so path resolution is reliable.
    # Use absolute paths (leading /) to avoid origin-relative ambiguity.
    # Horizontal: left = overview (full height), right = stacked per-method panels.
    shared = ["/scans/laser", "/trajectory/gt"]
    method_panels = [
        rrb.Spatial2DView(
            name=name.upper(),
            contents=shared + [f"/methods/{name}/**"],
        )
        for name in method_names
    ]
    overview_panel = rrb.Spatial2DView(
        name="ALL METHODS",
        contents=["/scans/laser", "/trajectory/gt", "/methods/**"],
    )
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            overview_panel,
            rrb.Vertical(*method_panels),
            column_shares=[2, 1],
        ),
        auto_views=False,
        collapse_panels=False,
    )
    rec.send_blueprint(blueprint, make_active=True, make_default=True)

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
    rec = rr.RecordingStream(application_id="edgegate-slam", make_default=True)
    if spawn:
        rec.spawn()

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
