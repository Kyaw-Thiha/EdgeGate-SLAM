# Experiment Plan — Sweep-004

> **Status:** Pre-execution. All infrastructure built, ready to run.
> **Pre-flight:** `--train-overrides` added to `run_sweep.py` for custom training params.

---

## Tier 1 — Domain Sweep (256 BCE Runs)

**Purpose:** Find the data config that minimizes the generalization gap.  
**Question answered:** How much of the real-benchmark ATE gap is pure training-data distribution mismatch?

### Training
```bash
pixi run train --multirun \
    data.outlier_rate=50,90 data.outlier_structure=clustered,random \
    data.num_poses=100,500,1000,3500 \
    data.lc_ratio=1,5,15,40 \
    data.info_scale=0.1,1.0,10.0,100.0
```
- **256 runs**, BCE-only (default), ~2h with 4 joblib workers
- **Sweeps:** `[50pct_clustered, 50pct_random, 90pct_clustered, 90pct_random]` × 4 graph sizes × 4 LC ratios × 4 info scales
- Each run: 100 epochs, 80 train / 20 val synthetic graphs, `model_best.pt` saved

### Evaluation
1. **Synthetic triage:** Evaluate ALL 256 models on synthetic test set only (256 × 1 dataset = 256 evals)
2. **Top-5 promotion:** Select top-5 models by synthetic F1 for real-benchmark evaluation
3. **Real benchmark eval:** 5 models × 5 real datasets × 7 injection combos = 175 evals
```bash
# Full injection sweep per real dataset:
# injection_outlier_rate=10,30,50,70 at lc_ratio=15
# injection_outlier_rate=30 at lc_ratio=5,15,40
```

### Key Metric
Synthetic F1 vs real-benchmark ATE. Best domain config = smallest synthetic-to-real ATE degradation.

---

## Tier 2 — Loss Function Comparison (3 Runs)

**Gate:** After Tier 1 analysis identifies the best domain config.  
**Purpose:** Determine whether trajectory loss improves ATE over BCE.

### Training (on best domain config from Tier 1)
```bash
# Run A: BCE baseline
pixi run train \
    data.outlier_rate=<best> data.outlier_structure=<best> \
    data.num_poses=<best> data.lc_ratio=<best> data.info_scale=<best> \
    train.loss_mode=bce

# Run B: Trajectory loss (K=5, ~4-11h)
pixi run train \
    ... \
    train.loss_mode=trajectory train.solver_train_iterations=5

# Run C: Combined BCE + trajectory (K=5, ~4-11h)
pixi run train \
    ... \
    train.loss_mode=combined train.solver_train_iterations=5
```

### Evaluation (all 3 models on all 6 datasets)
- `residual_iterations=1` (standard single-pass)
- `eval_method.method=hybrid_gnn_dcs` (prune mode, threshold=0.5)
- Full injection sweep on real benchmarks

### Questions Answered
- **BCE vs Trajectory:** Does end-to-end ATE optimization beat pure edge classification?
- **Odometry re-weighting effect:** Trajectory loss enables learned odometry weights — does this help?
- **BCE + Trajectory vs either alone:** Does combined training outperform single-objective?

---

## Tier 3 — Residual Features (2 Runs, Gated)

**Gate:** Only if Tier 2 shows trajectory loss meaningfully improves ATE (>10% reduction over BCE).

### Training (on same best domain config)
```bash
# Run D: BCE + residuals
pixi run train \
    ... \
    data.include_residuals=true model.edge_attr_dim=9 \
    train.loss_mode=bce

# Run E: Trajectory + residuals
pixi run train \
    ... \
    data.include_residuals=true model.edge_attr_dim=9 \
    train.loss_mode=trajectory train.solver_train_iterations=5
```

### Evaluation (both models on all 6 datasets)
- `residual_iterations=3` (iteration loop: solve → residuals → re-predict × 3)
- Compare against Tier 2 non-residual models at `residual_iterations=1`

### Question Answered
Does residual-guided iterative re-weighting close the GNN-DCS ATE gap?

---

## Execution Timeline

```
Night 1: Tier 1 training (256 BCE, ~2h) → Tier 1 eval (synthetic triage, ~1h)
Day 2:   Analyze Tier 1 → pick best domain config
Night 2: Tier 2 run B (trajectory, ~4-11h)
Night 3: Tier 2 run C (combined, ~4-11h)
Night 3: Tier 2 eval (3 models × full suite, ~2h)
Day 4:   Analyze Tier 2 → decide Tier 3 gate
Night 4: Tier 3 training if gated (~4-11h) → Tier 3 eval (~1h)
```

**Total: 4-5 overnights**, dominated by trajectory loss training runs.

---

## Tier 2+ Tier 3 Eval Command Reference

```bash
# Evaluate learned model with residual iterations
pixi run evaluate \
    eval_mode.dataset=intel \
    eval_mode.model_dir=<model_dir> \
    eval_mode.residual_iterations=3 \
    hydra.run.dir=<out_dir>

# Evaluate with GNN->DCS hybrid (prune mode)
pixi run evaluate \
    eval_mode.dataset=intel \
    eval_mode.model_dir=<model_dir> \
    eval_method.method=hybrid_gnn_dcs \
    eval_method.hybrid_mode=prune \
    eval_method.threshold=0.5 \
    hydra.run.dir=<out_dir>

# Evaluate with GNN->DCS hybrid (two_pass mode)
pixi run evaluate \
    eval_mode.dataset=intel \
    eval_mode.model_dir=<model_dir> \
    eval_method.method=hybrid_gnn_dcs \
    eval_method.hybrid_mode=two_pass \
    eval_method.threshold=0.5 \
    hydra.run.dir=<out_dir>
```

---

## Domain Sweep Run Reference

Tier 1 uses Hydra multirun directly (not `run_sweep.py`) since it sweeps data params, not outlier rates:

```bash
pixi run train --multirun \
    data.outlier_rate=50,90 data.outlier_structure=clustered,random \
    data.num_poses=100,500,1000,3500 \
    data.lc_ratio=1,5,15,40 \
    data.info_scale=0.1,1.0,10.0,100.0
```

Output lands in `runs/sweep_<ts>/train/...` directories. Each run gets its own `model_best.pt`.

The matching eval uses a one-off script or manual evaluate.py calls (not `run_sweep.py` since Tier 1 doesn't use the standard SWEEP_PARAMS grid). For synthetic triage:
```bash
for model_dir in runs/sweep_*/train/*/; do
    pixi run evaluate \
        eval_mode.dataset=synthetic \
        eval_mode.model_dir="$model_dir" \
        eval_mode.num_test_graphs=20 \
        hydra.run.dir=runs/sweep_*/eval_synthetic/$(basename $model_dir)/
done
```
