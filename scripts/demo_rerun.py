"""Rerun visualization entry point for EdgeGate-SLAM.

Three modes:
  --live      Fresh synthetic graph; optionally apply a trained GNN checkpoint.
  --replay    Replay a full training run from a Hydra output directory.
  --compare   Overlay multiple eval methods on the same test graph (paper figure).
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from edgegate.viz.rerun_logger import (
    log_eval_comparison,
    log_laser_scans,
    log_pose_graph,
    log_run,
)


def _run_live(args: argparse.Namespace) -> None:
    import rerun as rr
    from edgegate.data.synthetic_generator import generate
    from edgegate.data.graph_builder import to_pyg
    from edgegate.solvers.pypose_solver import PyPoseSolver

    graph = generate(
        num_poses=args.num_poses,
        num_loop_closures=10,
        outlier_rate=args.outlier_rate,
        outlier_structure="random",
        seed=args.seed,
    )
    E = graph.edge_index.shape[1]

    solver = PyPoseSolver()
    if args.checkpoint:
        from edgegate.models.edgegate_gnn import EdgeGateGNN
        model = EdgeGateGNN()
        model.load_state_dict(
            torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        )
        model.eval()
        data = to_pyg(graph)
        with torch.no_grad():
            confidence = model(data).cpu().numpy()
        conf_tensor = torch.from_numpy(confidence)
    else:
        conf_tensor = torch.ones(E)
        confidence = conf_tensor.numpy()

    poses, _, _, _ = solver.solve(graph, conf_tensor, max_iterations=None)
    poses_np = poses.cpu().numpy()

    rec = rr.RecordingStream(application_id="edgegate-slam-live", make_default=True)
    rec.spawn()

    log_pose_graph(graph, poses_np, confidence, epoch=0, rec=rec)

    if args.laser:
        log_laser_scans(args.laser, poses_np, rec=rec)

    if args.save_rrd:
        rec.save(args.save_rrd)


def _run_replay(args: argparse.Namespace) -> None:
    log_run(
        args.replay,
        carmen_log_path=args.laser,
        spawn=True,
        save_rrd=args.save_rrd,
    )


def _run_compare(args: argparse.Namespace) -> None:
    method_dirs: dict[str, str] = {}
    for pair in args.compare:
        name, path = pair.split("=", 1)
        method_dirs[name] = path

    log_eval_comparison(
        method_dirs,
        graph_idx=args.graph_idx,
        spawn=True,
        save_rrd=args.save_rrd,
        carmen_log_path=args.laser,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EdgeGate-SLAM Rerun visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true", help="Live demo on a fresh synthetic graph")
    mode.add_argument("--replay", metavar="RUN_DIR", help="Replay a training run directory")
    mode.add_argument(
        "--compare",
        nargs="+",
        metavar="METHOD=DIR",
        help="Compare methods: learned=outputs/... gnc=outputs/...",
    )

    # live-mode options
    parser.add_argument("--checkpoint", metavar="PATH", help="GNN checkpoint (.pt) for --live")
    parser.add_argument("--num-poses", type=int, default=50)
    parser.add_argument("--outlier-rate", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)

    # compare-mode options
    parser.add_argument("--graph-idx", type=int, default=0)

    # shared options
    parser.add_argument("--laser", metavar="CARMEN_LOG", help="Carmen .log for laser scan overlay")
    parser.add_argument("--save-rrd", metavar="OUT.rrd", help="Save recording to .rrd file")

    args = parser.parse_args()

    if args.live:
        _run_live(args)
    elif args.replay:
        _run_replay(args)
    else:
        _run_compare(args)


if __name__ == "__main__":
    main()
