from __future__ import annotations
import math
import time
import numpy as np
import torch
from edgegate.data.graph_builder import to_pyg
from edgegate.data.types import PoseGraph
from edgegate.metrics.ate_rmse import ate_rmse
from edgegate.solvers.base import Solver


def _compute_edge_residuals(
    poses: torch.Tensor,
    edge_index: torch.Tensor,
    measurements: np.ndarray,
) -> torch.Tensor:
    """Compute per-edge SE(2) residuals between solver output and measurements.

    Args:
        poses:        (N, 3) optimized poses [x, y, θ] from solver.
        edge_index:   (2, E) source/dst indices.
        measurements: (E, 3) raw edge measurements [dx, dy, dθ] (numpy).

    Returns:
        (E, 3) residuals [rx, ry, rθ] — torch tensor on same device as poses.
    """
    E = edge_index.size(1)
    residuals = torch.zeros(E, 3, dtype=poses.dtype, device=poses.device)
    meas_t = torch.from_numpy(measurements).float().to(poses.device)

    src, dst = edge_index[0], edge_index[1]
    dx_w = poses[dst, 0] - poses[src, 0]
    dy_w = poses[dst, 1] - poses[src, 1]
    ci, si = torch.cos(poses[src, 2]), torch.sin(poses[src, 2])

    expected_x = ci * dx_w + si * dy_w
    expected_y = -si * dx_w + ci * dy_w
    expected_theta = poses[dst, 2] - poses[src, 2]
    expected_theta = (expected_theta + math.pi) % (2 * math.pi) - math.pi

    residuals[:, 0] = meas_t[:, 0] - expected_x
    residuals[:, 1] = meas_t[:, 1] - expected_y
    dtheta = meas_t[:, 2] - expected_theta
    residuals[:, 2] = (dtheta + math.pi) % (2 * math.pi) - math.pi
    return residuals


def evaluate_one_graph(
    model: torch.nn.Module,
    solver: Solver,
    graph: PoseGraph,
    residual_iterations: int = 1,
) -> dict:
    """Run a single PoseGraph through the GNN + solver pipeline.

    When residual_iterations > 1, performs residual-guided iterative re-weighting:
    solve → compute per-edge residuals → update edge features → GNN re-predicts
    → repeat. This gives the GNN the same adaptive feedback loop that GNC/DCS
    have built into their kernel-based re-weighting.

    Returns per-graph counts and metrics:
        tp, fp, fn, ate, poses, confidence, lc_confidence, lc_labels,
        final_cost, num_iterations, converged, solve_time_s, solver_failed,
        residual_iterations
    """
    model.eval()
    with torch.no_grad():
        data = to_pyg(graph)
        conf = model(data)

        if hasattr(data, "edge_label") and data.edge_label is not None:
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

        # Ensure edge_attr has room for residuals (3 extra columns)
        if data.edge_attr.size(1) == 6:
            zeros = torch.zeros(data.edge_attr.size(0), 3,
                                dtype=data.edge_attr.dtype,
                                device=data.edge_attr.device)
            data.edge_attr = torch.cat([data.edge_attr, zeros], dim=1)

        total_solve_time = 0.0
        last_poses = None

        for it in range(residual_iterations):
            if it > 0 and last_poses is not None:
                residuals = _compute_edge_residuals(
                    last_poses, data.edge_index, graph.edge_measurement
                )
                data.edge_attr[:, 6:] = residuals
                conf = model(data)

            t0 = time.monotonic()
            try:
                poses, converged, iters, cost = solver.solve(
                    graph, conf, max_iterations=None
                )
                total_solve_time += time.monotonic() - t0
                solver_failed = False
                last_poses = poses
            except RuntimeError:
                total_solve_time += time.monotonic() - t0
                solver_failed = True
                converged = None
                iters = None
                cost = None
                last_poses = torch.from_numpy(graph.node_init).float()
                break

        poses = last_poses

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
        "solve_time_s": total_solve_time,
        "solver_failed": solver_failed,
        "residual_iterations": residual_iterations,
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
    with torch.no_grad():
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


def _build_pruned_posegraph(graph: PoseGraph, keep_mask: np.ndarray) -> PoseGraph:
    """Return a new PoseGraph containing only edges where keep_mask is True."""
    return PoseGraph(
        node_init=graph.node_init,
        edge_index=graph.edge_index[:, keep_mask],
        edge_measurement=graph.edge_measurement[keep_mask],
        edge_info=graph.edge_info[keep_mask],
        edge_type=graph.edge_type[keep_mask],
        edge_label=(
            graph.edge_label[keep_mask]
            if graph.edge_label is not None
            else None
        ),
        gt_node_poses=graph.gt_node_poses,
    )


def evaluate_one_graph_hybrid(
    model: torch.nn.Module,
    solver_dcs: Solver,
    graph: PoseGraph,
    threshold: float = 0.5,
    hybrid_mode: str = "prune",
) -> dict:
    """GNN → DCS hybrid evaluation.

    Mode 'prune':   GNN confidence → remove edges < threshold → DCS on pruned graph.
    Mode 'two_pass': Same, then warm-start DCS on full graph from clean result.

    Returns per-graph metrics plus hybrid-specific logging.
    """
    from edgegate.solvers.gtsam_solver import GTSAMSolver

    model.eval()
    with torch.no_grad():
        data = to_pyg(graph)
        conf = model(data)

    # Classification metrics from GNN
    lc_mask = graph.edge_type == 1
    if graph.edge_label is not None:
        valid_mask = lc_mask & (graph.edge_label >= 0.0)
        if valid_mask.any():
            labels_t = torch.from_numpy(graph.edge_label)
            pred = (conf[valid_mask] >= 0.5).float()
            gt = (labels_t[valid_mask] >= 0.5).float()
            tp = int((pred * gt).sum().item())
            fp = int((pred * (1 - gt)).sum().item())
            fn = int(((1 - pred) * gt).sum().item())
        else:
            tp = fp = fn = 0
    else:
        tp = fp = fn = 0

    # Remove low-confidence loop-closure edges
    conf_np = conf.cpu().numpy()
    remove_mask = lc_mask & (conf_np < threshold)
    keep_mask = ~remove_mask

    n_lc_total = int(lc_mask.sum())
    n_removed = int(remove_mask.sum())
    if n_removed > 0:
        removed_conf = conf_np[remove_mask]
        removed_mean = float(removed_conf.mean())
        removed_min = float(removed_conf.min())
        removed_max = float(removed_conf.max())
    else:
        removed_mean = removed_min = removed_max = 0.0

    pruned_graph = _build_pruned_posegraph(graph, keep_mask)

    pass1_ate = pass1_cost = pass1_iters = None
    pass1_converged = False

    if hybrid_mode == "two_pass":
        solver_clean = GTSAMSolver(kernel="none")
        w1 = torch.ones(pruned_graph.edge_index.shape[1], dtype=torch.float64)
        t0 = time.monotonic()
        pass1_poses, pass1_converged, pass1_iters, pass1_cost = (
            solver_clean.solve(pruned_graph, w1)
        )
        t1 = time.monotonic()

        if graph.gt_node_poses is not None:
            gt = torch.from_numpy(graph.gt_node_poses).float()
            pass1_ate = ate_rmse(pass1_poses, gt)
        else:
            pass1_ate = None

        warm_graph = PoseGraph(
            node_init=pass1_poses.cpu().numpy(),
            edge_index=graph.edge_index,
            edge_measurement=graph.edge_measurement,
            edge_info=graph.edge_info,
            edge_type=graph.edge_type,
            edge_label=graph.edge_label,
            gt_node_poses=graph.gt_node_poses,
        )
        w2 = torch.ones(graph.edge_index.shape[1], dtype=torch.float64)
        t0_2 = time.monotonic()
        poses, converged, iters, cost = solver_dcs.solve(warm_graph, w2)
        solve_time = (t1 - t0) + (time.monotonic() - t0_2)
    else:
        w = torch.ones(pruned_graph.edge_index.shape[1], dtype=torch.float64)
        t0 = time.monotonic()
        poses, converged, iters, cost = solver_dcs.solve(pruned_graph, w)
        solve_time = time.monotonic() - t0

    ate = None
    if graph.gt_node_poses is not None:
        gt = torch.from_numpy(graph.gt_node_poses).float()
        ate = ate_rmse(poses, gt)

    result = {
        "tp": tp, "fp": fp, "fn": fn,
        "ate": ate,
        "poses": poses.cpu().numpy(),
        "confidence": conf.cpu().numpy(),
        "final_cost": float(cost) if cost is not None else None,
        "num_iterations": iters,
        "converged": converged,
        "solve_time_s": solve_time,
        "solver_failed": False,
        "hybrid_mode": hybrid_mode,
        "hybrid_threshold": threshold,
        "hybrid_edges_total_lc": n_lc_total,
        "hybrid_edges_removed": n_removed,
        "hybrid_removed_conf_mean": removed_mean,
        "hybrid_removed_conf_min": removed_min,
        "hybrid_removed_conf_max": removed_max,
    }
    if hybrid_mode == "two_pass":
        result.update({
            "hybrid_pass1_ate": pass1_ate,
            "hybrid_pass1_cost": pass1_cost,
            "hybrid_pass1_iters": pass1_iters,
            "hybrid_pass1_converged": pass1_converged,
        })
    return result
