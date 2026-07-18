from __future__ import annotations
import time
import numpy as np
import torch
from edgegate.data.graph_builder import to_pyg
from edgegate.data.types import PoseGraph
from edgegate.metrics.ate_rmse import ate_rmse
from edgegate.solvers.base import Solver


def evaluate_one_graph(
    model: torch.nn.Module,
    solver: Solver,
    graph: PoseGraph,
) -> dict:
    """Run a single PoseGraph through the GNN + solver pipeline.

    Returns per-graph counts and metrics:
        tp, fp, fn, ate, poses, confidence, lc_confidence, lc_labels,
        final_cost, num_iterations, converged, solve_time_s, solver_failed
    """
    model.eval()
    with torch.no_grad():
        data = to_pyg(graph)
        conf = model(data)

        if hasattr(data, "edge_label") and data.edge_label is not None:
            # Exclude sentinel -1.0 labels (existing real-benchmark edges when
            # outliers are injected on top — those edges have no ground-truth label).
            lc_mask = (data.edge_type == 1) & (data.edge_label >= 0)
            pred = conf[lc_mask] >= 0.5
            label = data.edge_label[lc_mask] >= 0.5
            tp = int((pred & label).sum().item())
            fp = int((pred & ~label).sum().item())
            fn = int((~pred & label).sum().item())
            lc_confidence = conf[lc_mask].cpu().numpy()
            lc_labels = data.edge_label[lc_mask].cpu().numpy()
        else:
            tp = fp = fn = 0
            lc_confidence = np.array([], dtype=np.float32)
            lc_labels = np.array([], dtype=np.float32)

        t0 = time.monotonic()
        try:
            poses, converged, iters, cost = solver.solve(graph, conf, max_iterations=None)
            solve_time_s = time.monotonic() - t0
            solver_failed = False
        except RuntimeError:
            solve_time_s = time.monotonic() - t0
            solver_failed = True
            converged = None
            iters = None
            cost = None
            poses = torch.from_numpy(graph.node_init).float()

        ate = None
        if not solver_failed and graph.gt_node_poses is not None:
            gt = torch.from_numpy(graph.gt_node_poses).float()
            ate = ate_rmse(poses, gt)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "ate": ate,
        "poses": poses.cpu().numpy(),
        "confidence": conf.cpu().numpy(),
        "lc_confidence": lc_confidence,
        "lc_labels": lc_labels,
        "final_cost": float(cost) if cost is not None else None,
        "num_iterations": iters,
        "converged": converged,
        "solve_time_s": solve_time_s,
        "solver_failed": solver_failed,
    }


def evaluate_one_graph_classical(
    solver: Solver,
    graph: PoseGraph,
    edge_weights: torch.Tensor,
) -> dict:
    """Run a classical baseline (no GNN) on a single PoseGraph.

    Returns per-graph metrics:
        ate, poses, confidence, lc_confidence, lc_labels,
        final_cost, num_iterations, converged, solve_time_s, solver_failed
    """
    edge_type = torch.from_numpy(graph.edge_type)
    edge_label = (
        torch.from_numpy(graph.edge_label) if graph.edge_label is not None else None
    )
    if edge_label is not None:
        lc_mask = (edge_type == 1) & (edge_label >= 0)
        lc_confidence = edge_weights[lc_mask].cpu().numpy()
        lc_labels = edge_label[lc_mask].cpu().numpy()
    else:
        lc_confidence = np.array([], dtype=np.float32)
        lc_labels = np.array([], dtype=np.float32)

    t0 = time.monotonic()
    try:
        poses, converged, iters, cost = solver.solve(graph, edge_weights, max_iterations=None)
        solve_time_s = time.monotonic() - t0
        solver_failed = False
    except RuntimeError:
        solve_time_s = time.monotonic() - t0
        solver_failed = True
        converged = None
        iters = None
        cost = None
        poses = torch.from_numpy(graph.node_init).float()

    ate = None
    if not solver_failed and graph.gt_node_poses is not None:
        gt = torch.from_numpy(graph.gt_node_poses).float()
        ate = ate_rmse(poses, gt)

    return {
        "ate": ate,
        "poses": poses.cpu().numpy(),
        "confidence": edge_weights.cpu().numpy(),
        "lc_confidence": lc_confidence,
        "lc_labels": lc_labels,
        "final_cost": float(cost) if cost is not None else None,
        "num_iterations": iters,
        "converged": converged,
        "solve_time_s": solve_time_s,
        "solver_failed": solver_failed,
    }


def accumulate_metrics(results: list[dict]) -> dict:
    """Aggregate per-graph metrics into summary statistics.

    F1 is computed from accumulated tp/fp/fn counts — never averaged from
    per-graph F1 scores (Jensen's inequality trap: averaging F1 across
    variable-size graphs is not the same as computing F1 from total counts).

    Returns:
        precision, recall, f1, tp, fp, fn, ate,
        final_cost, solve_time_s, num_iterations_mean, converged_count, failed_count
    """
    tp = sum(r.get("tp", 0) for r in results)
    fp = sum(r.get("fp", 0) for r in results)
    fn = sum(r.get("fn", 0) for r in results)

    ate_values = [r["ate"] for r in results if r.get("ate") is not None]
    cost_values = [r["final_cost"] for r in results if r.get("final_cost") is not None]
    time_values = [r["solve_time_s"] for r in results if r.get("solve_time_s") is not None]
    iter_values = [
        r["num_iterations"] for r in results
        if r.get("num_iterations") is not None and r["num_iterations"] >= 0
    ]
    converged_count = sum(1 for r in results if r.get("converged") is True)
    failed_count = sum(1 for r in results if r.get("solver_failed") is True)

    if tp + fp + fn > 0:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
    else:
        precision = recall = f1 = None  # no edge labels available

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "ate": sum(ate_values) / len(ate_values) if ate_values else None,
        "final_cost": sum(cost_values) / len(cost_values) if cost_values else None,
        "solve_time_s": sum(time_values) / len(time_values) if time_values else None,
        "num_iterations_mean": sum(iter_values) / len(iter_values) if iter_values else None,
        "converged_count": converged_count,
        "failed_count": failed_count,
    }
