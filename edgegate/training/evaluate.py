from __future__ import annotations
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
        {"tp": int, "fp": int, "fn": int, "ate": float | None}
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
        else:
            tp = fp = fn = 0

        poses, converged, iters, cost = solver.solve(graph, conf, max_iterations=None)

        ate = None
        if graph.gt_node_poses is not None:
            gt = torch.from_numpy(graph.gt_node_poses).float()
            ate = ate_rmse(poses, gt)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "ate": ate,
        "poses": poses.cpu().numpy(),
        "confidence": conf.cpu().numpy(),
    }


def evaluate_one_graph_classical(
    solver: Solver,
    graph: PoseGraph,
    edge_weights: torch.Tensor,
) -> dict:
    """Run a classical baseline (no GNN) on a single PoseGraph.

    Returns per-graph metrics (no F1 counts — classical baselines don't produce
    per-edge confidence scores):
        {"ate": float | None}
    """
    poses, converged, iters, cost = solver.solve(graph, edge_weights, max_iterations=None)

    ate = None
    if graph.gt_node_poses is not None:
        gt = torch.from_numpy(graph.gt_node_poses).float()
        ate = ate_rmse(poses, gt)

    return {
        "ate": ate,
        "poses": poses.cpu().numpy(),
        "confidence": edge_weights.cpu().numpy(),
    }


def accumulate_metrics(results: list[dict]) -> dict:
    """Aggregate per-graph metrics into summary statistics.

    F1 is computed from accumulated tp/fp/fn counts — never averaged from
    per-graph F1 scores (Jensen's inequality trap: averaging F1 across
    variable-size graphs is not the same as computing F1 from total counts).

    Returns:
        {"precision": float, "recall": float, "f1": float | None,
         "tp": int, "fp": int, "fn": int, "ate": float | None}
        f1 is None when tp+fp+fn == 0 (no edge labels available).
        ate is None when no graph had gt_node_poses.
    """
    tp = sum(r.get("tp", 0) for r in results)
    fp = sum(r.get("fp", 0) for r in results)
    fn = sum(r.get("fn", 0) for r in results)

    ate_values = [r["ate"] for r in results if r.get("ate") is not None]

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

    ate = sum(ate_values) / len(ate_values) if ate_values else None

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "ate": ate,
    }
