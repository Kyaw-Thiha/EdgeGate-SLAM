#!/usr/bin/env python
"""Orchestrate the full Phase 0 training → evaluation → baseline pipeline.

Runs:
  1. Train: outlier-rate × structure sweep (10 models)
  2. Eval learned: each model on synthetic + real benchmarks
  3. Eval baselines: uniform / GNC / DCS on synthetic + real benchmarks
  4. Aggregate: sweep summary CSV

All output lands in runs/<timestamp>/. Each step saves to known subdirectories
so downstream steps can find them without parsing Hydra output structures.

Usage:
    pixi run run-sweep
    pixi run run-sweep -- --skip-train            # eval only, using existing models
    pixi run run-sweep -- --skip-baselines         # train + learned eval only
    pixi run run-sweep -- --epochs 50 --rates 30,50,70   # custom params
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SWEEP_PARAMS = [
    (10, "random"),
    (30, "random"),
    (50, "random"),
    (70, "random"),
    (90, "random"),
    (10, "clustered"),
    (30, "clustered"),
    (50, "clustered"),
    (70, "clustered"),
    (90, "clustered"),
]

SYNTH_DATASETS = ["synthetic"]
REAL_DATASETS = ["intel", "m3500", "mit", "csail", "city10000"]  # dropped manhattan (M3500 duplicate)
ALL_DATASETS = SYNTH_DATASETS + REAL_DATASETS

BASELINES = [
    ("uniform", "eval_method.method=uniform solver=gtsam solver.kernel=none"),
    ("gnc", "eval_method.method=gnc"),
    ("dcs", "eval_method.method=dcs"),
]


def _sep(title: str) -> None:
    print(f"\n{'='*60}\n  {title}\n{'='*60}", flush=True)


def _run(cmd_list: list[str], cwd: Path = ROOT) -> bool:
    print(f"  $ {' '.join(cmd_list)}", flush=True)
    t0 = time.time()
    try:
        result = subprocess.run(cmd_list, cwd=cwd, capture_output=True, text=True)
    except Exception as e:
        print(f"  FAILED: {e}", file=sys.stderr)
        return False
    elapsed = time.time() - t0
    if result.returncode != 0:
        tail = result.stderr[-300:] if result.stderr else "(no output)"
        print(f"  FAILED (exit {result.returncode}, {elapsed:.0f}s): {tail}", file=sys.stderr)
        return False
    print(f"  OK ({elapsed:.0f}s)", flush=True)
    return True


def train_models(
    sweep_dir: Path,
    epochs: int,
    rates: list[int] | None = None,
    seeds: list[int] | None = None,
) -> list[tuple[str, Path]]:
    """Train one model per (outlier_rate, structure, seed) combo. Returns (label, model_dir) pairs."""
    if seeds is None:
        seeds = [0]
    _sep("STEP 1: Training sweep")
    models: list[tuple[str, Path]] = []
    train_dir = sweep_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    for rate, structure in SWEEP_PARAMS:
        if rates and rate not in rates:
            continue
        for seed in seeds:
            label = f"{rate}pct_{structure}_seed{seed}"
            out_dir = train_dir / label
            cmd = [
                "pixi", "run", "train",
                f"train.epochs={epochs}",
                f"data.outlier_rate={rate}",
                f"data.outlier_structure={structure}",
                f"train.seed={seed}",
                f"hydra.run.dir={out_dir}",
            ]
            if _run(cmd):
                models.append((label, out_dir))
            else:
                print(f"  WARNING: training failed for {label}, skipping downstream evals",
                      file=sys.stderr)
            print()

    print(f"Trained {len(models)}/{len(SWEEP_PARAMS) * len(seeds)} models → {train_dir}/")
    return models


def eval_learned(sweep_dir: Path, models: list[tuple[str, Path]]) -> None:
    """Evaluate each trained model on synthetic + real benchmarks."""
    _sep("STEP 2: Evaluate learned models")
    eval_dir = sweep_dir / "eval_learned"

    for label, model_dir in models:
        for ds in ALL_DATASETS:
            out_path = eval_dir / label / ds
            out_path.mkdir(parents=True, exist_ok=True)
            cmd = [
                "pixi", "run", "evaluate",
                f"eval_mode.dataset={ds}",
                f"eval_mode.model_dir={model_dir.resolve()}",
                f"hydra.run.dir={out_path}",
            ]
            _run(cmd)
        print()


def eval_baselines(sweep_dir: Path) -> None:
    """Evaluate classical baselines on synthetic + real benchmarks."""
    _sep("STEP 3: Evaluate classical baselines")
    bl_dir = sweep_dir / "eval_baselines"

    for method_name, extra_args in BASELINES:
        for ds in ALL_DATASETS:
            out_path = bl_dir / method_name / ds
            out_path.mkdir(parents=True, exist_ok=True)
            cmd = [
                "pixi", "run", "evaluate",
                f"eval_mode.dataset={ds}",
                f"hydra.run.dir={out_path}",
            ] + extra_args.split()
            _run(cmd)
        print()


def aggregate(sweep_dir: Path) -> None:
    """Collect all summary.json files into a single CSV."""
    _sep("STEP 4: Aggregate results")
    rows: list[dict] = []
    for sf in sorted(sweep_dir.glob("**/summary.json")):
        d = json.loads(sf.read_text())
        rel = sf.parent.relative_to(sweep_dir)
        parts = str(rel).split(os.sep)
        if parts[0] == "eval_learned" and len(parts) >= 3:
            d["category"] = "learned"
            d["model"] = parts[1]
            d["dataset"] = parts[2]
        elif parts[0] == "eval_baselines" and len(parts) >= 3:
            d["category"] = parts[1]
            d["model"] = "—"
            d["dataset"] = parts[2]
        else:
            d["category"] = "other"
            d["model"] = "—"
            d["dataset"] = str(rel)
        rows.append(d)

    if not rows:
        print("No summary.json files found.")
        return

    csv_path = sweep_dir / "results.csv"
    keys = ["category", "model", "dataset", "method", "num_graphs",
            "f1", "precision", "recall", "ate", "tp", "fp", "fn",
            "final_cost", "solve_time_s", "converged_count", "failed_count",
            "num_iterations_mean"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows → {csv_path}")


def aggregate_ci(sweep_dir: Path, sr_threshold: float = 1.0) -> None:
    """Compute per-(config, dataset) mean ± std and Success Rate across seeds."""
    import re
    import numpy as np
    from collections import defaultdict

    raw_csv = sweep_dir / "results.csv"
    if not raw_csv.exists():
        print("results.csv not found — run aggregate() first.")
        return

    with open(raw_csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    def _config_key(model_label: str) -> str:
        # Strip trailing _seed{N} to get the (outlier_rate, structure) key
        m = re.match(r"(\d+pct_(?:random|clustered))", model_label)
        return m.group(1) if m else model_label

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            _config_key(row.get("model", "")),
            row.get("dataset", ""),
            row.get("category", ""),
        )
        groups[key].append(row)

    NUMERIC = ["f1", "precision", "recall", "ate", "final_cost", "solve_time_s"]
    ci_rows = []
    for (config, dataset, category), group in sorted(groups.items()):
        cr: dict = {
            "config": config,
            "dataset": dataset,
            "category": category,
            "n_seeds": len(group),
        }
        for col in NUMERIC:
            vals = []
            for r in group:
                v = r.get(col)
                if v not in (None, "", "None"):
                    try:
                        vals.append(float(v))
                    except ValueError:
                        pass
            cr[f"mean_{col}"] = float(np.mean(vals)) if vals else None
            cr[f"std_{col}"] = float(np.std(vals)) if vals else None
        ate_vals = [
            float(r["ate"]) for r in group
            if r.get("ate") not in (None, "", "None")
        ]
        cr["success_rate"] = (
            sum(1 for v in ate_vals if v < sr_threshold) / len(ate_vals)
            if ate_vals else None
        )
        ci_rows.append(cr)

    ci_csv = sweep_dir / "results_ci.csv"
    ci_keys = (
        ["config", "dataset", "category", "n_seeds", "success_rate"]
        + [f"{stat}_{col}" for col in NUMERIC for stat in ("mean", "std")]
    )
    with open(ci_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ci_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ci_rows)
    print(f"CI summary ({len(ci_rows)} groups, SR threshold={sr_threshold}) → {ci_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full Phase 0 sweep pipeline")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--rates", type=str, default=None,
                        help="comma-separated outlier rates, e.g. 30,50,70")
    parser.add_argument("--seeds", type=str, default="0",
                        help="comma-separated training seeds, e.g. 0,1,2")
    parser.add_argument("--sr-threshold", type=float, default=1.0,
                        help="ATE threshold for Success Rate in results_ci.csv")
    parser.add_argument("--output", type=str, default=None,
                        help="output directory (default: runs/sweep_<timestamp>)")
    args = parser.parse_args()

    if args.output:
        sweep_dir = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        sweep_dir = ROOT / "runs" / f"sweep_{ts}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    rates = [int(r) for r in args.rates.split(",")] if args.rates else None
    seeds = [int(s) for s in args.seeds.split(",")]

    # ── Training ─────────────────────────────────────────────────────────────
    if args.skip_train:
        train_dir = sweep_dir / "train"
        models: list[tuple[str, Path]] = []
        for rate, structure in SWEEP_PARAMS:
            if rates and rate not in rates:
                continue
            for seed in seeds:
                label = f"{rate}pct_{structure}_seed{seed}"
                model_dir = train_dir / label
                if (model_dir / "model_best.pt").exists():
                    models.append((label, model_dir))
        print(f"Found {len(models)} existing models in {train_dir}/")
        if not models:
            print("No existing models found and --skip-train set. Nothing to do.",
                  file=sys.stderr)
            sys.exit(1)
    else:
        models = train_models(sweep_dir, args.epochs, rates, seeds)
        if not models:
            print("No models trained successfully.", file=sys.stderr)
            sys.exit(1)

    # ── Evaluation ───────────────────────────────────────────────────────────
    eval_learned(sweep_dir, models)

    if not args.skip_baselines:
        eval_baselines(sweep_dir)

    # ── Aggregate ────────────────────────────────────────────────────────────
    aggregate(sweep_dir)
    aggregate_ci(sweep_dir, sr_threshold=args.sr_threshold)
    print(f"\nDone. Results → {sweep_dir}/")


if __name__ == "__main__":
    main()
