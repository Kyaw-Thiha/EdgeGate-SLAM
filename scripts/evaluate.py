from __future__ import annotations

import glob
import json
import os
import warnings
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from edgegate.data.g2o_io import load_g2o
from edgegate.data.outlier_injection import inject_labeled_loop_closures
from edgegate.data.synthetic_generator import generate, LC_INFO
from edgegate.training.evaluate import (
    evaluate_one_graph,
    evaluate_one_graph_classical,
    accumulate_metrics,
)


_BENCHMARK_PATHS = {
    "intel": "data/raw/intel.g2o",
    "m3500": "data/raw/M3500.g2o",
    "mit": "data/raw/MIT.g2o",
    "csail": "data/raw/CSAIL.g2o",
    "manhattan": "data/raw/manhattan.g2o",
    "city10000": "data/raw/city10000.g2o",
    "parking-garage": "data/raw/parking-garage.g2o",
}

# sphere2500 and parking-garage from SE-Sync are SE(3) quaternion format — not
# parseable by the SE(2)-only g2o_io.py parser. Deferred to Phase 2 (SE3 extension).
# See docs/implementation_details.md §"Future Work" and the SE3 extension plan.
_SE3_DATASETS = {"sphere2500", "parking-garage"}


def _seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_synthetic(cfg: DictConfig) -> list:
    data_cfg = cfg.get("data", {})
    kwargs = {
        "num_poses": data_cfg.get("num_poses", 50),
        "num_loop_closures": data_cfg.get("num_loop_closures", 20),
        "outlier_rate": data_cfg.get("outlier_rate", 30),
        "outlier_structure": data_cfg.get("outlier_structure", "random"),
        "segment_length": data_cfg.get("segment_length", 5),
        "proximity_threshold": data_cfg.get("proximity_threshold", 2.0),
        "info_scale": data_cfg.get("info_scale", 1.0),
    }
    lc_ratio = data_cfg.get("lc_ratio")
    if lc_ratio is not None:
        kwargs["lc_ratio"] = lc_ratio
    n = cfg.eval_mode.get("num_test_graphs", 20)
    seed = cfg.eval_mode.get("test_seed", 999)
    return [generate(**kwargs, seed=seed + i) for i in range(n)]


def _load_benchmark(cfg: DictConfig) -> list:
    ds_name = cfg.eval_mode.get("dataset", "")
    if ds_name in _SE3_DATASETS:
        warnings.warn(
            f"'{ds_name}' is an SE(3) dataset. The current g2o_io.py parser only "
            f"supports VERTEX_SE2/EDGE_SE2 format — SE(3) QUAT parsing is deferred "
            f"to Phase 2. Skipping evaluation on {ds_name}."
        )
        return []
    rel_path = _BENCHMARK_PATHS.get(ds_name, "")
    if not rel_path:
        warnings.warn(f"No path mapping for dataset '{ds_name}' — skipping.")
        return []
    from hydra.utils import get_original_cwd
    path = str(Path(get_original_cwd()) / rel_path)
    if not Path(path).exists():
        warnings.warn(f"Benchmark file not found: {path} — skipping.")
        return []

    graph = load_g2o(path)

    # Always compute ref_poses for ATE (pseudo-GT = outlier-free clean solve).
    # Real benchmarks have no independently-sourced ground truth; ATE is
    # ATE-against-reference-solve per implementation_details.md.
    if graph.gt_node_poses is not None:
        ref_poses = graph.gt_node_poses
    else:
        from edgegate.solvers.gtsam_solver import GTSAMSolver
        _ref_solver = GTSAMSolver(kernel="none")
        _ref_w = torch.ones(graph.edge_index.shape[1])
        _ref_poses_t, _, _, _ = _ref_solver.solve(graph, _ref_w)
        ref_poses = _ref_poses_t.cpu().numpy()

    if cfg.eval_mode.get("inject_outliers", False):
        rng = np.random.default_rng(cfg.eval_mode.get("test_seed", 999))

        # Resolve LC count: injection_lc_ratio overrides injection_num_lcs
        num_lcs = cfg.eval_mode.get("injection_num_lcs", None)
        lc_ratio = cfg.eval_mode.get("injection_lc_ratio", None)
        if num_lcs is None and lc_ratio is not None:
            num_lcs = max(1, graph.node_init.shape[0] // lc_ratio)
        elif num_lcs is None:
            num_lcs = 20

        try:
            lc_edges, lc_meas, lc_labels = inject_labeled_loop_closures(
                reference_poses=ref_poses,
                num_loop_closures=num_lcs,
                outlier_rate=cfg.eval_mode.get("injection_outlier_rate", 30),
                outlier_structure=cfg.eval_mode.get("injection_outlier_structure", "random"),
                rng=rng,
            )
        except ValueError:
            warnings.warn(
                f"Outlier injection failed for {ds_name} — graph topology may not "
                f"have enough proximal/distant pairs. Returning graph without injected "
                f"labels (F1 will be N/A)."
            )
            from edgegate.data.types import PoseGraph
            return [PoseGraph(
                node_init=graph.node_init,
                edge_index=graph.edge_index,
                edge_measurement=graph.edge_measurement,
                edge_info=graph.edge_info,
                edge_type=graph.edge_type,
                edge_label=graph.edge_label,
                gt_node_poses=ref_poses,
            )]

        E_existing = graph.edge_index.shape[1]
        E_new = lc_edges.shape[1]
        E = E_existing + E_new

        new_idx = np.zeros((2, E), dtype=np.int64)
        new_meas = np.zeros((E, 3))
        new_info = np.zeros((E, 6))
        new_type = np.zeros(E, dtype=np.int64)
        new_label = np.full(E, -1.0, dtype=np.float32)

        new_idx[:, :E_existing] = graph.edge_index
        new_meas[:E_existing] = graph.edge_measurement
        new_info[:E_existing] = graph.edge_info
        new_type[:E_existing] = graph.edge_type

        new_idx[:, E_existing:] = lc_edges
        new_meas[E_existing:] = lc_meas
        new_info[E_existing:] = LC_INFO
        new_type[E_existing:] = 1
        new_label[E_existing:] = lc_labels

        from edgegate.data.types import PoseGraph
        return [PoseGraph(
            node_init=graph.node_init,
            edge_index=new_idx,
            edge_measurement=new_meas,
            edge_info=new_info,
            edge_type=new_type,
            edge_label=new_label,
            gt_node_poses=ref_poses,  # was graph.gt_node_poses (= None for real benchmarks)
        )]

    # No injection — still patch ref_poses in for downstream ATE computation
    from edgegate.data.types import PoseGraph
    return [PoseGraph(
        node_init=graph.node_init,
        edge_index=graph.edge_index,
        edge_measurement=graph.edge_measurement,
        edge_info=graph.edge_info,
        edge_type=graph.edge_type,
        edge_label=graph.edge_label,
        gt_node_poses=ref_poses,
    )]


def _load_data(cfg: DictConfig) -> list:
    ds_name = cfg.eval_mode.get("dataset", "synthetic")
    if ds_name == "synthetic":
        return _load_synthetic(cfg)
    return _load_benchmark(cfg)


def _load_model(cfg: DictConfig) -> torch.nn.Module:
    """Load a GNN model from checkpoint. Shared between learned and hybrid methods."""
    from hydra.utils import get_original_cwd

    model_dir = cfg.eval_mode.get("model_dir")
    ckpt_path = cfg.eval_mode.get("checkpoint_path")
    if model_dir is not None:
        p = Path(model_dir)
        if not p.is_absolute():
            p = Path(get_original_cwd()) / p
        ckpt_path = str(p / "model_best.pt")
    elif ckpt_path is None:
        ckpt_path = str(Path(get_original_cwd()) / "model_best.pt")
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. "
            "Set eval_mode.checkpoint_path, eval_mode.model_dir, "
            "or place model_best.pt in cwd."
        )
    model = instantiate(cfg.model)
    model.load_state_dict(
        torch.load(ckpt_path, map_location="cpu", weights_only=True)
    )
    model.eval()
    return model


def _setup_method(cfg: DictConfig) -> dict:
    method_name = cfg.eval_method.get("method", "learned")
    method: dict = {"type": method_name, "model": None}

    if method_name == "learned":
        model = _load_model(cfg)
        method["model"] = model
        solver = instantiate(cfg.solver)
    elif method_name == "gnc":
        from edgegate.solvers.gtsam_solver import GTSAMSolver
        solver = GTSAMSolver(kernel="gnc")
    elif method_name == "dcs":
        from edgegate.solvers.gtsam_solver import GTSAMSolver
        solver = GTSAMSolver(kernel="dcs")
    elif method_name == "hybrid_gnn_dcs":
        from edgegate.solvers.gtsam_solver import GTSAMSolver
        model = _load_model(cfg)
        method["model"] = model
        method["threshold"] = cfg.eval_method.get("threshold", 0.5)
        method["hybrid_mode"] = cfg.eval_method.get("hybrid_mode", "prune")
        solver = GTSAMSolver(
            kernel="dcs",
            dcs_param=cfg.eval_method.get("dcs_param", 1.0),
        )
    elif method_name == "uniform":
        solver = instantiate(cfg.solver)
    elif method_name == "switchable":
        raise NotImplementedError(
            "Switchable constraints is not yet implemented. "
            "See docs/implementation_details.md §'Future Work'."
        )
    else:
        raise ValueError(f"Unknown method: {method_name}")

    method["solver"] = solver
    return method


def _safe_serialize(obj):
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


def _save_graph_info(graph, out_dir: str) -> None:
    info = {
        "node_init": graph.node_init.tolist(),
        "edge_index": graph.edge_index.tolist(),
        "edge_type": graph.edge_type.tolist(),
        "edge_label": graph.edge_label.tolist() if graph.edge_label is not None else None,
        "gt_node_poses": graph.gt_node_poses.tolist() if graph.gt_node_poses is not None else None,
    }
    Path(out_dir, "graph_info.json").write_text(json.dumps(info))


def _run_evaluate(cfg: DictConfig) -> None:
    from hydra.core.hydra_config import HydraConfig
    try:
        out_dir = HydraConfig.get().runtime.output_dir
    except Exception:
        out_dir = "."

    _seed(cfg.eval_mode.get("test_seed", 999))

    graphs = _load_data(cfg)
    if not graphs:
        print("No graphs to evaluate. Exiting.")
        return

    method = _setup_method(cfg)
    solver = method["solver"]
    model = method["model"]
    method_type = method["type"]

    save_poses = cfg.eval_mode.get("save_poses", True)
    graph_info_saved = False

    results = []
    for i, graph in enumerate(graphs):
        if method_type == "learned":
            r = evaluate_one_graph(
                model, solver, graph,
                residual_iterations=cfg.eval_mode.get("residual_iterations", 1),
            )
        elif method_type == "hybrid_gnn_dcs":
            from edgegate.training.evaluate import evaluate_one_graph_hybrid
            r = evaluate_one_graph_hybrid(
                model, solver, graph,
                threshold=method.get("threshold", 0.5),
                hybrid_mode=method.get("hybrid_mode", "prune"),
            )
        else:
            w = torch.ones(graph.edge_index.shape[1])
            r = evaluate_one_graph_classical(solver, graph, w)
        r["graph_idx"] = i

        if save_poses and "poses" in r and r["poses"] is not None:
            pg_dir = os.path.join(out_dir, "per_graph", f"graph_{i:03d}")
            os.makedirs(pg_dir, exist_ok=True)
            np.save(os.path.join(pg_dir, "poses.npy"), r["poses"])
            np.save(os.path.join(pg_dir, "confidence.npy"), r["confidence"])
            # LC arrays for PR curve generation (Phase 1)
            if r.get("lc_confidence") is not None and len(r["lc_confidence"]) > 0:
                np.save(os.path.join(pg_dir, "lc_confidence.npy"), r["lc_confidence"])
                np.save(os.path.join(pg_dir, "lc_labels.npy"), r["lc_labels"])
            if not graph_info_saved:
                _save_graph_info(graph, out_dir)
                graph_info_saved = True

        results.append(r)

    agg = accumulate_metrics(results)

    if cfg.eval_mode.get("print_table", True):
        print()
        print(f"Evaluation: {method_type} on {len(graphs)} graphs")
        print(f"{'='*50}")
        if agg["f1"] is not None:
            print(f"  Precision : {agg['precision']:.4f}")
            print(f"  Recall    : {agg['recall']:.4f}")
            print(f"  F1        : {agg['f1']:.4f}")
            print(f"  TP/FP/FN  : {agg['tp']} / {agg['fp']} / {agg['fn']}")
        else:
            print("  F1        : N/A (no edge labels)")
        if agg["ate"] is not None:
            print(f"  ATE (RMSE): {agg['ate']:.6f}")
        else:
            print("  ATE       : N/A (no ground truth)")
        if agg.get("final_cost") is not None:
            print(f"  Cost F(x) : {agg['final_cost']:.4f}")
        if agg.get("solve_time_s") is not None:
            print(f"  Solve time: {agg['solve_time_s'] * 1000:.1f} ms (mean)")
        print(f"  Converged : {agg.get('converged_count', 0)} / {len(graphs)}")
        if agg.get("failed_count", 0) > 0:
            print(f"  Failed    : {agg['failed_count']}")
        print()

    if cfg.eval_mode.get("save_per_graph", False):
        per_graph = [{k: _safe_serialize(v) for k, v in r.items()}
                     for r in results]
        Path(out_dir, "per_graph.json").write_text(
            json.dumps(per_graph, indent=2)
        )

    if cfg.eval_mode.get("save_summary", True):
        summary = {
            "method": method_type,
            "num_graphs": len(graphs),
            **{k: _safe_serialize(v) for k, v in agg.items()},
        }
        Path(out_dir, "summary.json").write_text(
            json.dumps(summary, indent=2)
        )

    if cfg.eval_mode.get("log_to_wandb", False):
        import wandb
        wandb.init(
            project=cfg.logging.get("project", "edgegate-slam"),
            entity=cfg.logging.get("entity"),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb_log = {}
        if agg["f1"] is not None:
            wandb_log["eval/f1"] = agg["f1"]
            wandb_log["eval/precision"] = agg["precision"]
            wandb_log["eval/recall"] = agg["recall"]
        if agg["ate"] is not None:
            wandb_log["eval/ate_rmse"] = agg["ate"]
        if agg.get("final_cost") is not None:
            wandb_log["eval/final_cost"] = agg["final_cost"]
        if agg.get("solve_time_s") is not None:
            wandb_log["eval/solve_time_ms"] = agg["solve_time_s"] * 1000
        wandb_log["eval/converged_count"] = agg.get("converged_count", 0)
        wandb_log["eval/failed_count"] = agg.get("failed_count", 0)
        if wandb_log:
            wandb.log(wandb_log)
        wandb.finish()


def _run_aggregate(cfg: DictConfig) -> None:
    pattern = cfg.eval_mode.get("sweep_glob", "runs/sweep_*")
    run_dirs = sorted(glob.glob(pattern))
    if not run_dirs:
        print(f"No run directories found matching: {pattern}")
        return

    rows = []
    for d in run_dirs:
        summary_path = Path(d) / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())

        hydra_config_path = Path(d) / ".hydra" / "config.yaml"
        hydra_cfg = {}
        if hydra_config_path.exists():
            hydra_cfg = OmegaConf.to_container(
                OmegaConf.load(str(hydra_config_path)), resolve=True
            )

        row = {"run_dir": d, **summary}
        if isinstance(hydra_cfg, dict):
            data_cfg = hydra_cfg.get("data", {})
            if isinstance(data_cfg, dict):
                row["outlier_rate"] = data_cfg.get("outlier_rate")
                row["outlier_structure"] = data_cfg.get("outlier_structure")
            eval_mode = hydra_cfg.get("eval_mode", {})
            eval_method = hydra_cfg.get("eval_method", {})
            if isinstance(eval_method, dict):
                row["method"] = eval_method.get("method")
        rows.append(row)

    if not rows:
        print("No summary.json files found in run directories.")
        return

    keys = ["run_dir", "method", "outlier_rate", "outlier_structure",
            "f1", "precision", "recall", "ate", "tp", "fp", "fn"]
    csv_path = cfg.eval_mode.get("output_csv", "sweep_comparison.csv")
    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {csv_path}")


@hydra.main(config_path="../configs", config_name="evaluate", version_base=None)
def main(cfg: DictConfig) -> None:
    mode = cfg.eval_mode.get("mode", "evaluate")
    if mode == "evaluate":
        seeds_str = cfg.eval_mode.get("test_seeds")
        if seeds_str is not None:
            if isinstance(seeds_str, (list, tuple)):
                seeds = [int(s) for s in seeds_str]
            else:
                seeds = [int(s) for s in str(seeds_str).strip("[]").split(",")]
            _run_evaluate_multi_seed(cfg, seeds)
        else:
            _run_evaluate(cfg)
    elif mode == "aggregate":
        _run_aggregate(cfg)
    else:
        raise ValueError(f"Unknown eval mode: {mode}")


def _run_evaluate_multi_seed(cfg: DictConfig, seeds: list[int]) -> None:
    from omegaconf import OmegaConf

    all_summaries = []
    for seed in seeds:
        cfg_copy = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cfg_copy.eval_mode.test_seed = seed
        OmegaConf.update(cfg, "eval_mode.test_seed", seed, force_add=True)
        print(f"\n--- Multi-seed eval: seed={seed} ---")
        try:
            _run_evaluate(cfg)
            # Read the summary.json that _run_evaluate just wrote
            out_dir = _get_output_dir()
            summary_path = Path(out_dir) / "summary.json"
            if summary_path.exists():
                all_summaries.append((seed, json.loads(summary_path.read_text())))
        except Exception as e:
            print(f"  FAILED seed {seed}: {e}")

    if len(all_summaries) < 2:
        return

    _write_multi_seed_summary(all_summaries)


def _get_output_dir() -> str:
    try:
        from hydra.core.hydra_config import HydraConfig
        return HydraConfig.get().runtime.output_dir
    except Exception:
        return "."


def _write_multi_seed_summary(all_summaries: list[tuple[int, dict]]) -> None:
    import numpy as np

    NUMERIC = [
        "f1", "precision", "recall", "ate", "rotation_error",
        "final_cost", "solve_time_s", "gnn_time_s",
        "num_iterations_mean", "num_graphs",
    ]
    ci: dict = {"n_seeds": len(all_summaries), "per_seed": []}

    for seed, s in all_summaries:
        seed_entry = {"seed": seed}
        for key in NUMERIC:
            if key in s and s[key] is not None:
                seed_entry[key] = s[key]
        for key in ["converged_count", "failed_count", "tp", "fp", "fn"]:
            if key in s:
                seed_entry[key] = s[key]
        ci["per_seed"].append(seed_entry)

    for key in NUMERIC:
        vals = [s[1].get(key) for s in all_summaries if s[1].get(key) is not None]
        if vals:
            arr = np.array(vals, dtype=float)
            ci[f"{key}_mean"] = float(arr.mean())
            ci[f"{key}_std"] = float(arr.std())

    out_dir = _get_output_dir()
    with open(Path(out_dir) / "summary.json", "w") as f:
        json.dump(ci, f, indent=2)
    print(f"\nMulti-seed CI ({len(all_summaries)} seeds) → {out_dir}/summary.json")


if __name__ == "__main__":
    main()
