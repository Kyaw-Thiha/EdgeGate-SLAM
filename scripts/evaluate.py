from __future__ import annotations

import glob
import json
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
        "proximity_threshold": data_cfg.get("proximity_threshold", 2.0),
    }
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
    path = _BENCHMARK_PATHS.get(ds_name, "")
    if not path or not Path(path).exists():
        warnings.warn(f"Benchmark file not found: {path} — skipping.")
        return []

    graph = load_g2o(path)

    if cfg.eval_mode.get("inject_outliers", False):
        rng = np.random.default_rng(cfg.eval_mode.get("test_seed", 999))

        if graph.gt_node_poses is not None:
            ref_poses = graph.gt_node_poses
        else:
            from edgegate.solvers.gtsam_solver import GTSAMSolver
            solver = GTSAMSolver(kernel="none")
            w = torch.ones(graph.edge_index.shape[1])
            ref_poses, _, _, _ = solver.solve(graph, w)
            ref_poses = ref_poses.cpu().numpy()

        lc_edges, lc_meas, lc_labels = inject_labeled_loop_closures(
            reference_poses=ref_poses,
            num_loop_closures=cfg.eval_mode.get("injection_num_lcs", 20),
            outlier_rate=cfg.eval_mode.get("injection_outlier_rate", 30),
            outlier_structure=cfg.eval_mode.get("injection_outlier_structure", "random"),
            rng=rng,
        )

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
        graph = PoseGraph(
            node_init=graph.node_init,
            edge_index=new_idx,
            edge_measurement=new_meas,
            edge_info=new_info,
            edge_type=new_type,
            edge_label=new_label,
            gt_node_poses=graph.gt_node_poses,
        )

    return [graph]


def _load_data(cfg: DictConfig) -> list:
    ds_name = cfg.eval_mode.get("dataset", "synthetic")
    if ds_name == "synthetic":
        return _load_synthetic(cfg)
    return _load_benchmark(cfg)


def _setup_method(cfg: DictConfig) -> dict:
    method_name = cfg.eval_method.get("method", "learned")
    method: dict = {"type": method_name, "model": None}

    if method_name == "learned":
        ckpt_path = cfg.eval_mode.get("checkpoint_path")
        if ckpt_path is None:
            ckpt_path = "model_best.pt"
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}. "
                "Set eval_mode.checkpoint_path or place model_best.pt in cwd."
            )
        model = instantiate(cfg.model)
        model.load_state_dict(
            torch.load(ckpt_path, map_location="cpu", weights_only=True)
        )
        model.eval()
        method["model"] = model
        solver = instantiate(cfg.solver)
    elif method_name == "gnc":
        from edgegate.solvers.gtsam_solver import GTSAMSolver
        solver = GTSAMSolver(kernel="gnc")
    elif method_name == "dcs":
        from edgegate.solvers.gtsam_solver import GTSAMSolver
        solver = GTSAMSolver(kernel="dcs")
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

    results = []
    for i, graph in enumerate(graphs):
        if method_type == "learned":
            r = evaluate_one_graph(model, solver, graph)
        else:
            w = torch.ones(graph.edge_index.shape[1])
            r = evaluate_one_graph_classical(solver, graph, w)
        r["graph_idx"] = i
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
        _run_evaluate(cfg)
    elif mode == "aggregate":
        _run_aggregate(cfg)
    else:
        raise ValueError(f"Unknown eval mode: {mode}")


if __name__ == "__main__":
    main()
