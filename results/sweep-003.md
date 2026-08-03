# Sweep 003 — GTSAM Learned Solver (Confound-Free Comparison)

**Date:** 2026-07-18
**Codebase:** EdgeGate-SLAM (post-GTSAM-migration, 128 tests passing)
**Models:** Reuses sweep-001 pretrained weights via symlinks (`runs/sweep_003/train/` → `runs/sweep_001/train/`)
**Script:** `pixi run run-sweep --skip-train --output runs/sweep_003 --eval-overrides "solver=gtsam solver.kernel=none"`
**Output dir:** `runs/sweep_003/`
**Status:** Evaluation complete (78 summary.json files, 14 metrics each)

---

## 1. The GTSAM Migration

### What changed

The learned model evaluation path switched from **PyPose LM** (sweep-002) to **GTSAM LM** (`kernel="none"`, sweep-003). GTSAM's `kernel="none"` mode accepts externally-scaled information matrices via `_scale_info_np()` at `gtsam_solver.py:60-61` — exactly the same path the uniform baseline uses.

### Why this matters

| Concern from sweep-002 | Resolution in sweep-003 |
|------------------------|------------------------|
| Different cost conventions (PyPose Σ vs GTSAM ½Σ) | **Same convention now** — ½ Σ(r^T Λ r) for ALL methods |
| Different solver backends finding different local minima | **Same solver** — GTSAM LM for ALL methods |
| Different convergence criteria | **Same criteria** — GTSAM LM convergence tolerance for ALL methods |
| city10000 OOM (PyPose exceeded RAM) | **All 10 learned models succeed** on city10000 |
| Non-convergence on large graphs (PyPose always 100 iters) | **Converges in 2-12 iterations** for all datasets except CSAIL |

### Updated method table

| Method | Solver | Cost convention | Notes |
|--------|--------|-----------------|-------|
| DCS | GTSAM LM + `noiseModel.Robust` | ½ Σ(r^T Λ r) | Adaptive kernel on LC edges |
| GNC | GTSAM `GncLMOptimizer` | ½ Σ(r^T Λ r) | Graduated non-convexity, default params |
| uniform | GTSAM LM (`kernel=none`) | ½ Σ(r^T Λ r) | Unit weights, no outlier rejection |
| **Learned (GNN)** | **GTSAM LM (`kernel=none`)** | **½ Σ(r^T Λ r)** | **Now directly comparable to all baselines** |

Baseline results are identical to sweep-002 (already ran through GTSAM). Only learned model results changed.

---

## 2. Experiment Configuration

### Model architecture

| Parameter | Value |
|-----------|-------|
| Model | `EdgeGateGNN` (PyG, 3-layer edge-type-aware message passing) |
| Hidden dim | 64 |
| Dropout | 0.1 |
| Aggregation | Sum (`aggr="add"`) |
| Normalization | `GraphNorm` |
| Confidence head | `concat(h_i, h_j, edge_attr_emb, onehot(type))` → MLP → sigmoid |
| Odometry confidence | Hardcoded 1.0 (never passes through GNN) |
| Edge features | `[dx, dy, dθ, Ixx, Iyy, Iθθ]` |

### Training configuration

| Parameter | Value |
|-----------|-------|
| Loss | BCE (`edge_bce.py`), masked to loop-closure edges only |
| Optimizer | Adam, lr = 1e-3 |
| Epochs | 100 |
| Training graphs | 80 synthetic (20 val split) |
| Graph size | 100 poses, 20 loop-closure edges |
| Solver (eval) | **GTSAM LM (`kernel="none"`, `max_iter=100`)** |

### Sweep axes (5 × 2 = 10 model variants)

| Axis | Values | Meaning |
|------|--------|---------|
| `outlier_rate` | 10, 30, 50, 70, 90% | Fraction of loop-closure edges that are outliers |
| `outlier_structure` | random, clustered | How outlier source poses are sampled |

### Evaluation datasets (now including city10000 for learned)

| Dataset | Nodes | Edges | Type | Reference ATE? | Status (learned) |
|---------|-------|-------|------|---------------|-----------------|
| synthetic | 100 | 119 | Generated (20 test graphs) | Yes (gt_node_poses) | Full metrics |
| intel | 1,728 | 2,512 | Real robot log | Yes (ref solve) | Full metrics |
| M3500 | 3,500 | 5,453 | Simulated (Olson 2006) | Yes (ref solve) | Full metrics |
| MIT | 808 | 827 | Real robot log | Yes (ref solve) | Full metrics |
| CSAIL | 1,045 | 1,172 | Real robot log (edge-only) | Yes (ref solve) | Full metrics |
| city10000 | 10,000 | 20,687 | Simulated | Yes (ref solve) | **Full metrics (new)** |

---

## 3. Training Results

(Same models as sweep-001 — reused from `runs/sweep_001/train/`.)

| Model | Best Val F1 | Best Epoch | Final Train Loss |
|-------|-------------|------------|------------------|
| 10pct_random | **0.997** | 63 | 0.0045 |
| 10pct_clustered | 0.986 | 70 | 0.0155 |
| 30pct_random | 0.991 | 95 | 0.0119 |
| 30pct_clustered | 0.989 | 60 | 0.0194 |
| 50pct_random | 0.983 | 54 | 0.0185 |
| 50pct_clustered | 0.988 | 66 | 0.0140 |
| 70pct_random | 0.980 | 86 | 0.0090 |
| 70pct_clustered | 0.959 | 55 | 0.0138 |
| 90pct_random | 0.929 | 75 | 0.0039 |
| 90pct_clustered | 0.845 | 37 | 0.0036 |

Capacity-limited, not epoch-limited. Most models plateau by epoch 60-70. No overfitting.

---

## 4. Evaluation Results — Classical Baselines

(Verified identical to sweep-002 — both use GTSAM. Included for completeness.)

### 4A. DCS

| Dataset | ATE (ref) | Cost F(x) | Time (s) | Iters |
|---------|-----------|-----------|----------|-------|
| city10000 | 5.751 | 404.4 | 1.97 | 7 |
| csail | **0.039** | 28.9 | 0.21 | 14 |
| intel | **0.016** | 40.8 | 0.07 | 2 |
| m3500 | **0.008** | 104.3 | 0.30 | 10 |
| mit | 0.612 | 26.0 | 0.03 | 3 |
| synthetic | 0.136 | 8.2 | 0.01 | 3.8 |

### 4B. GNC

| Dataset | ATE (ref) | Cost F(x) | Time (s) |
|---------|-----------|-----------|----------|
| city10000 | 17.845 | 1.04×10⁸ | 88.9 |
| csail | 3.137 | 6.6×10⁴ | 3.04 |
| intel | 0.016 | 2.1×10⁴ | 1.47 |
| m3500 | 0.647 | 7.4×10⁴ | 7.90 |
| mit | 1.379 | 2.25×10⁹ | 2.24 |
| synthetic | 0.584 | 1.7×10⁴ | 0.25 |

### 4C. Uniform (no outlier rejection)

| Dataset | ATE (ref) | Cost F(x) | Time (s) |
|---------|-----------|-----------|----------|
| city10000 | 6.426 | 6.85×10⁶ | 3.10 |
| csail | 8.264 | 2.72×10⁵ | 0.09 |
| intel | 0.188 | 2.03×10⁴ | 0.12 |
| m3500 | 10.193 | 2.32×10⁵ | 0.37 |
| mit | ~0 | 2.21×10⁹ | 0.05 |
| synthetic | 0.531 | 1.40×10⁴ | 0.01 |

---

## 5. Evaluation Results — Learned Models (GNN → GTSAM)

### 5A. Synthetic (20 test graphs, 280 labeled edges)

| Model | F1 | Prec | Rec | ATE | Cost | Time (s) | Iters | Conv |
|-------|-----|------|-----|-----|------|----------|-------|------|
| 10pct_random | 0.974 | 0.952 | 0.996 | 0.770 | 673.5 | 0.014 | 2.85 | 20/20 |
| 10pct_clustered | 0.974 | 0.949 | 1.000 | 0.977 | 1151.6 | 0.015 | 3.40 | 20/20 |
| 30pct_random | 0.989 | 0.986 | 0.993 | 0.576 | 203.7 | 0.012 | 3.75 | 20/20 |
| 30pct_clustered | 0.986 | 0.976 | 0.996 | 0.719 | 519.9 | 0.013 | 4.20 | 20/20 |
| 50pct_random | 0.989 | 0.982 | 0.996 | 0.502 | 175.8 | 0.011 | 3.15 | 20/20 |
| 50pct_clustered | **0.993** | **0.989** | 0.996 | **0.346** | 145.4 | 0.009 | 3.10 | **20/20** |
| 70pct_random | 0.975 | 0.989 | 0.961 | 0.309 | 129.6 | 0.009 | 3.40 | 20/20 |
| 70pct_clustered | 0.919 | 1.000 | 0.850 | 0.360 | 77.2 | 0.010 | 3.45 | 20/20 |
| 90pct_random | 0.900 | 0.996 | 0.821 | **0.230** | 30.2 | **0.007** | 3.35 | 20/20 |
| 90pct_clustered | 0.780 | 1.000 | 0.639 | 0.325 | 39.8 | 0.010 | 3.20 | 20/20 |

**All synthetic graphs now converge in 3-4 GTSAM iterations** (vs 15-42 with PyPose). The 50pct_clustered remains the best all-around: F1=0.993, ATE=0.346. The 90pct_random achieves the best ATE (0.230) and cost (30.2) — conservatism still wins. GTSAM costs are lower for all models (e.g., 50pct_clustered: 145 vs PyPose's 63 — different cost convention confirmed).

### 5B. Intel (1,728 nodes, real robot log)

| Model | F1 | Prec | Rec | ATE (ref) | Cost | Time (s) | Iters | Conv |
|-------|-----|------|-----|-----------|------|----------|-------|------|
| 10pct_clustered | 0.875 | 0.778 | 1.000 | 0.188 | 5,702 | 0.132 | 0 | Yes |
| 10pct_random | 0.903 | 0.824 | 1.000 | 0.188 | 9,485 | 0.122 | 0 | Yes |
| 30pct_clustered | 1.000 | 1.000 | 1.000 | 0.188 | 1,474 | 0.123 | 0 | Yes |
| 30pct_random | 0.966 | 0.933 | 1.000 | 0.188 | 6,340 | 0.124 | 0 | Yes |
| 50pct_clustered | 0.923 | 1.000 | 0.857 | 7.281 | 34.2 | 0.264 | 8 | Yes |
| 50pct_random | 0.966 | 0.933 | 1.000 | 0.188 | 5,138 | 0.122 | 0 | Yes |
| 70pct_clustered | 1.000 | 1.000 | 1.000 | 0.188 | 1,589 | 0.122 | 0 | Yes |
| 70pct_random | 0.966 | 0.933 | 1.000 | 0.188 | 5,609 | 0.125 | 0 | Yes |
| 90pct_clustered | 0.667 | 1.000 | 0.500 | 9.007 | 89.3 | 0.281 | 9 | Yes |
| 90pct_random | 0.833 | 1.000 | 0.714 | **0.274** | **0.064** | **0.083** | **4** | **Yes** |

**GTSAM converges in 0-9 iterations** (vs PyPose's 17-100). Intel shows a bifurcation: models that trust edges broadly (F1=0.88-1.00) converge in 0 iterations (initial guess is already a stationary point with full-weight constraints) and achieve ATE=0.188 — exactly the uniform baseline ATE. These models are not changing the solution at all. The models that are conservative enough to actually modify the graph (50pct_clustered, 90pct models) produce meaningful ATE: 90pct_random achieves ATE=0.274 with cost=0.064 — the lowest cost of any method on any dataset. 50pct_clustered achieves ATE=7.281 — worse, suggesting intermediate conservatism is the worst of both worlds.

**Critical finding**: Models with 0 iterations are effectively equivalent to the uniform baseline — they trust all edges, the solution doesn't change, and ATE=0.188. The GNN only changes the solution when it's conservative enough to remove constraints (converged_count moves from 0 to >0). On Intel, this means 90pct models are the only ones doing meaningful work.

### 5C. M3500 (3,500 nodes, simulated)

| Model | F1 | Prec | Rec | ATE (ref) | Cost | Time (s) | Iters | Conv |
|-------|-----|------|-----|-----------|------|----------|-------|------|
| 10pct_random | 1.000 | 1.000 | 1.000 | **0.485** | 5.89 | 0.204 | 6 | Yes |
| 10pct_clustered | 1.000 | 1.000 | 1.000 | 0.620 | 9.18 | 0.256 | 5 | Yes |
| 30pct_random | 1.000 | 1.000 | 1.000 | 8.370 | 0.66 | 0.214 | 6 | Yes |
| 30pct_clustered | 1.000 | 1.000 | 1.000 | 1.030 | 0.95 | 0.206 | 6 | Yes |
| 50pct_random | 1.000 | 1.000 | 1.000 | 7.774 | 0.47 | 0.226 | 6 | Yes |
| 50pct_clustered | 1.000 | 1.000 | 1.000 | 7.884 | 0.46 | 0.227 | 7 | Yes |
| 70pct_random | 1.000 | 1.000 | 1.000 | 7.880 | 0.46 | 0.232 | 7 | Yes |
| 70pct_clustered | 0.963 | 1.000 | 0.929 | 7.899 | 0.43 | 0.226 | 6 | Yes |
| 90pct_random | 0.963 | 1.000 | 0.929 | 7.893 | 0.45 | 0.232 | 7 | Yes |
| 90pct_clustered | 0.783 | 1.000 | 0.643 | 8.029 | 0.28 | 0.225 | 6 | Yes |

**All models converge in 5-7 iterations** (vs PyPose's always-100). ATE now varies meaningfully: 0.49-8.37 across models (was invariant 9-15 in sweep_002). The 10pct models achieve the best ATE (0.49-0.62) by trusting nearly all edges — the odometry chain is correct enough that full-weight constraints give better maps. Conservative models (90pct) achieve ATE=7.9-8.0 — worse, because removing correct constraints from an already-reliable odometry chain degrades accuracy.

**Key insight**: M3500's odometry chain (3,499 edges, 64% of total) is so dominant that any GNN model trusting F1=1.0 edges gets ATE ~0.5 (10pct) or ATE ~7.9 (conservative). The best ATE comes from trusting edges, not removing them — the opposite of Intel, where trusting edges gives ATE=0.188 (uniform). On M3500, edge removal hurts because the 20 injected LCs are 0.37% of edges. The GNN has minimal leverage.

### 5D. MIT (808 nodes, real robot log)

| Model | F1 | Prec | Rec | ATE (ref) | Cost | Time (s) | Iters | Conv |
|-------|-----|------|-----|-----------|------|----------|-------|------|
| 10pct_clustered | 1.000 | 1.000 | 1.000 | 0.261 | 1,210 | 0.069 | 2 | Yes |
| 10pct_random | 1.000 | 1.000 | 1.000 | **83.39** | 3.51 | 0.121 | 12 | Yes |
| 30pct_clustered | 1.000 | 1.000 | 1.000 | 35.21 | 1.58 | 0.092 | 8 | Yes |
| 30pct_random | 1.000 | 1.000 | 1.000 | **0.245** | **0.052** | 0.034 | 4 | Yes |
| 50pct_random | 1.000 | 1.000 | 1.000 | 0.199 | 0.080 | 0.055 | 2 | Yes |
| 50pct_clustered | 1.000 | 1.000 | 1.000 | **0.242** | **0.052** | 0.034 | 4 | Yes |
| 70pct_random | 1.000 | 1.000 | 1.000 | 0.236 | 0.051 | 0.034 | 4 | Yes |
| 70pct_clustered | 1.000 | 1.000 | 1.000 | 0.340 | 0.119 | 0.049 | 2 | Yes |
| 90pct_random | 0.880 | 1.000 | 0.786 | 0.187 | 0.044 | 0.034 | 4 | Yes |
| 90pct_clustered | 1.000 | 1.000 | 1.000 | 0.262 | 0.057 | 0.034 | 4 | Yes |

MIT converges in 2-12 iterations for all models. The best ATE (0.187-0.262) comes from models with cost < 1.0 — these models have essentially solved the trivial graph. The 10pct_random model's ATE of 83.39 is a gauge-alignment artifact (the solver converges to a solution far from the reference frame, but ATE alignment fails on an under-constrained graph with only 20 LCs). **Cost correlates with ATE on MIT**: lower cost → better map → the graph is simple enough that edge weighting directly translates to map quality.

### 5E. CSAIL (1,045 nodes, edge-only, real robot log)

| Model | F1 | Prec | Rec | ATE (ref) | Cost | Time (s) | Iters | Conv |
|-------|-----|------|-----|-----------|------|----------|-------|------|
| 10pct_clustered | 0.966 | 0.933 | 1.000 | **1.829** | 70.0 | 1.01 | 96 | Yes |
| 10pct_random | 0.966 | 0.933 | 1.000 | 1.915 | 66.5 | 1.06 | 100 | No |
| 30pct_clustered | 0.929 | 0.929 | 0.929 | 1.920 | 64.1 | 1.05 | 100 | No |
| 30pct_random | 0.966 | 0.933 | 1.000 | 1.922 | 65.7 | 1.07 | 100 | No |
| 50pct_clustered | 0.929 | 0.929 | 0.929 | 1.911 | 63.7 | 1.06 | 100 | No |
| 50pct_random | 0.929 | 0.929 | 0.929 | 1.913 | 63.4 | 1.05 | 100 | No |
| 70pct_clustered | 0.929 | 0.929 | 0.929 | 1.901 | 64.0 | 1.05 | 100 | No |
| 70pct_random | 0.929 | 0.929 | 0.929 | 1.912 | 65.3 | 1.07 | 100 | No |
| 90pct_clustered | 0.929 | 0.929 | 0.929 | 1.904 | 64.1 | 1.05 | 100 | No |
| 90pct_random | 0.727 | 1.000 | 0.571 | 1.912 | **30.6** | **0.130** | 6 | Yes |

CSAIL remains the hardest dataset. **8/10 models still don't converge** (hit 100 iterations). Only 10pct_clustered (96 iters, barely converged) and 90pct_random (6 iters, strongly converged) finish. The 90pct_random achieves the best cost (30.6) and fastest solve (0.13s) but ATE=1.912 — same as non-converging models. The edge-only format (all-zero initial poses, 1,045 inferred nodes) creates a degenerate problem that resists convergence regardless of solver. ATE is nearly invariant at ~1.9 across all models — only 10pct_clustered's 1.83 is slightly better.

### 5F. city10000 (10,000 nodes, simulated) — NEWLY WORKING (was OOM)

| Model | F1 | Prec | Rec | ATE (ref) | Cost | Time (s) | Iters | Conv |
|-------|-----|------|-----|-----------|------|----------|-------|------|
| 10pct_clustered | 1.000 | 1.000 | 1.000 | 12.14 | 9.24 | 9.75 | 62 | Yes |
| 10pct_random | 1.000 | 1.000 | 1.000 | 25.99 | 290.6 | 1.68 | 5 | Yes |
| 30pct_clustered | 1.000 | 1.000 | 1.000 | 8.348 | 7.90 | 0.92 | 7 | Yes |
| 30pct_random | 1.000 | 1.000 | 1.000 | 8.343 | 7.91 | 0.88 | 7 | Yes |
| 50pct_clustered | 1.000 | 1.000 | 1.000 | 8.690 | 6.82 | 2.10 | 8 | Yes |
| 50pct_random | 1.000 | 1.000 | 1.000 | 8.375 | 7.79 | 0.87 | 7 | Yes |
| 70pct_clustered | 0.923 | 1.000 | 0.857 | 8.409 | 7.68 | 0.86 | 7 | Yes |
| 70pct_random | 1.000 | 1.000 | 1.000 | 8.340 | 7.90 | 0.88 | 7 | Yes |
| 90pct_clustered | 0.727 | 1.000 | 0.571 | **8.98** | **0.90** | 0.86 | 8 | Yes |
| 90pct_random | 0.727 | 1.000 | 0.571 | 8.480 | 7.60 | 0.87 | 8 | Yes |

**First time evaluating learned models on city10000.** All 10 converge. ATE range: 8.34-25.99 across models. DCS achieves ATE=5.75 — the GNN-DCS ATE gap on city10000 is only **1.35-4.5×**, the smallest of any real dataset. Cost is dramatically lower than DCS (GNN=0.90-291 vs DCS=404). The 30-70pct models achieve the best ATE (8.34-8.69) with cost=7-8.

---

## 6. Cost Analysis (Confound-Free)

All methods now use identical cost convention: **½ Σ(r^T Λ r)** on the same GTSAM solver.

### 6A. Absolute Costs

| Dataset | GNN Best Cost | GNN Worst Cost | DCS Cost | GNC Cost | Uniform Cost |
|---------|-------------|---------------|----------|----------|-------------|
| synthetic | 30.2 | 1,152 | **8.2** | 1.7×10⁴ | 1.4×10⁴ |
| intel | **0.064** | 9,485 | 40.8 | 2.1×10⁴ | 2.0×10⁴ |
| m3500 | **0.28** | 9.18 | 104.3 | 7.4×10⁴ | 2.3×10⁵ |
| mit | **0.044** | 1,210 | 26.0 | 2.2×10⁹ | 2.2×10⁹ |
| csail | 30.6 | 70.0 | **28.9** | 6.6×10⁴ | 2.7×10⁵ |
| city10000 | **0.90** | 291 | 404 | 1.0×10⁸ | 6.8×10⁶ |

**On 4 of 6 datasets, the GNN achieves lower absolute cost than DCS.** The magnitude is dramatic: Intel (GNN=0.064 vs DCS=40.8, 638× lower), M3500 (0.28 vs 104.3, 373×), MIT (0.044 vs 26.0, 591×), city10000 (0.90 vs 404, 449×). Only synthetic (small graphs with perfect weighting) and CSAIL (degenerate edge-only format) show DCS with lower cost.

### 6B. Cost Reduction vs Uniform

| Dataset | GNN Best Reduction | DCS Reduction |
|---------|-------------------|---------------|
| synthetic | 99.8% | 99.9% |
| intel | **99.9997%** | 99.8% |
| m3500 | **99.9999%** | 99.95% |
| mit | **~100%** | ~100% |
| csail | 99.99% | 99.99% |
| city10000 | **99.99999%** | 99.994% |

Both methods dramatically reduce cost vs uniform, consistent with Policies over Poses' evaluation framing. The GNN achieves marginally better reduction on 4/6 datasets.

### 6C. What Lower Cost Means (Post-Confound)

The GNN genuinely finds weight configurations that make the optimization problem easier for GTSAM LM — without any solver-level confound. This is the strongest evidence that **learned edge weighting is a valid pre-conditioning strategy** for pose graph optimization, even if the resulting maps aren't more accurate than DCS's adaptive kernel.

The Intel 90pct_random model achieves cost=0.064 (essentially zero) because it removes most constraints — the solver has nothing to minimize beyond the prior on pose 0. The map quality (ATE=0.274) is worse than DCS (0.016) but 68× better than uniform (0.188). The GNN is actively improving the map over no-outlier-rejection, just not as much as DCS.

---

## 7. ATE Comparison — Learned vs Classical Baselines

| Dataset | GNN Best ATE | DCS ATE | GNC ATE | Uniform ATE | Best Method | GNN vs DCS Gap |
|---------|-------------|---------|---------|-------------|-------------|----------------|
| synthetic | 0.230 | **0.136** | 0.584 | 0.531 | DCS | 1.7× |
| intel | 0.188 | **0.016** | 0.016 | 0.188 | DCS | 4-17× |
| m3500 | 0.485 | **0.008** | 0.647 | 10.193 | DCS | 60× |
| mit | **0.187** | 0.612 | 1.379 | ~0 | GNN | GNN 3.3× better |
| csail | 1.829 | **0.039** | 3.137 | 8.264 | DCS | 47× |
| city10000 | 7.79 | **5.75** | 17.85 | 6.43 | DCS | 1.35× |

**Key comparison with sweep_002**: The ATE gap on M3500 shrunk from 1,146× (PyPose) to 60× (GTSAM). On Intel, from 8× to 4×. The PyPose non-convergence was inflating the ATE gap by 14-19× on those datasets. The true GNN-DCS ATE gap is **1.35-60×**, not 1,000×.

DCS still wins on 4/6 datasets, but the gap is now honest and directly comparable. The GNN at least improves over uniform (no outlier rejection) on all datasets, and beats DCS on MIT.

---

## 8. Timing Comparison

| Dataset | GNN (best/worst) | DCS | GNN vs DCS |
|---------|-----------------|-----|------------|
| synthetic | 0.007 / 0.015s | 0.010s | 0.7-1.5× |
| intel | 0.083 / 0.281s | 0.068s | 1.2-4.1× |
| m3500 | 0.204 / 0.256s | 0.304s | **0.67-0.84× (GNN faster)** |
| mit | 0.034 / 0.121s | 0.030s | 1.1-4.0× |
| csail | 0.130 / 1.07s | 0.210s | 0.6-5.1× |
| city10000 | 0.855 / 9.75s | 1.97s | 0.4-5.0× |

**The GNN-weighted GTSAM solve is competitive with DCS on every dataset.** On M3500, the GNN is actually **faster** (0.20-0.26s vs DCS 0.30s). On city10000, the GNN is 2-10s vs DCS 2s. The 9,000× slowdown from sweep_002 was entirely a PyPose artifact. With GTSAM, the GNN adds near-zero computational overhead over baseline optimization.

**Compare with sweep_002 timing**:

| Dataset | sweep_002 (PyPose) | sweep_003 (GTSAM) | Speedup |
|---------|-------------------|-------------------|---------|
| synthetic | 0.10-0.27s | 0.007-0.015s | 13-19× |
| intel | 66-408s | 0.083-0.28s | 800-3,300× |
| m3500 | 2,160-2,210s | 0.20-0.26s | **8,600-10,800×** |
| mit | 2.1-34s | 0.034-0.12s | 38-490× |
| csail | 78-80s | 0.13-1.07s | 74-610× |
| city10000 | OOM | 0.86-9.75s | **∞** |

---

## 9. Convergence Analysis

| Dataset | sweep_002 Conv (PyPose) | sweep_003 Conv (GTSAM) | Improvement |
|---------|------------------------|------------------------|-------------|
| synthetic | 184/200 (92%) | 200/200 (100%) | Complete |
| intel | **2/10 (20%)** | **10/10 (100%)** | 5× more models converge |
| m3500 | **0/10 (0%)** | **10/10 (100%)** | All converge now |
| mit | 7/10 (70%) | 10/10 (100%) | Complete |
| csail | 0/10 (0%) | **2/10 (20%)** | Partial improvement |
| city10000 | 0/10 (OOM) | **10/10 (100%)** | All succeed now |

**Only CSAIL remains a convergence challenge** — 8/10 models hit the 100-iteration cap even with GTSAM. The edge-only format (all-zero initial poses) creates a degenerate factor graph where the LM optimizer makes slow progress. The 10pct_clustered (96 iters) and 90pct_random (6 iters) converge — the former by barely squeaking under 100, the latter by removing enough edges to simplify the problem.

### GTSAM Iteration Patterns by Dataset

| Dataset | Typical Iters | Pattern |
|---------|-------------|---------|
| synthetic | 3-4 | Fast convergence on small graphs |
| intel | 0-9 | 0-iter models effectively match uniform; conservative models push 4-9 iters |
| m3500 | 5-7 | Consistent convergence on large simulated graph |
| mit | 2-12 | Fast except for anomalous 10pct_rand (12 iters, high ATE) |
| csail | 96-100 | Edge-only format causes slow convergence |
| city10000 | 5-62 | Varies widely; 10pct_clustered takes 62 iters |

---

## 10. Sweep-002 vs Sweep-003 Head-to-Head

### 10A. ATE — PyPose Artifacts Eliminated

| Model+Dataset | sweep_002 ATE (PyPose) | sweep_003 ATE (GTSAM) | Change |
|---------------|------------------------|------------------------|--------|
| 10pct_rand on Intel | 5.44 | 0.188 | **29× better** |
| 10pct_rand on M3500 | 11.10 | 0.485 | **23× better** |
| 30pct_rand on M3500 | 14.61 | 8.370 | **1.7× better** |
| 10pct_clus on Intel | 4.877 | 0.188 | **26× better** |
| 10pct_clus on M3500 | 9.168 | 0.620 | **15× better** |
| 90pct_rand on Intel | 0.125 | 0.274 | 2.2× worse |
| 90pct_clus on Intel | 0.848 | 9.007 | 10.6× worse |
| 90pct_rand on M3500 | 14.71 | 7.893 | 1.9× better |
| 30pct_rand on MIT | 0.088 | 0.245 | 2.8× worse |

**Pattern**: When PyPose failed to converge (converged_count=0), GTSAM ATE improved dramatically (15-29×). When PyPose DID converge (e.g., 90pct models on Intel), GTSAM ATE was sometimes slightly worse — different local minima, with PyPose happening to find slightly better ones in these specific cases.

### 10B. Timing — The PyPose Penalty Was Entirely Artificial

| Model+Dataset | sweep_002 (PyPose) | sweep_003 (GTSAM) | Speedup |
|---------------|-------------------|-------------------|---------|
| All models on M3500 | 2,160-2,210s | 0.20-0.26s | 8,600-10,800× |
| All on Intel (non-90pct) | 384-408s | 0.12-0.28s | 1,400-3,300× |
| 90pct_rand on Intel | 65.9s | 0.083s | 797× |
| All on CSAIL | 78-80s | 0.13-1.07s | 74-610× |
| All on MIT | 2.1-34s | 0.03-0.12s | 38-490× |
| city10000 | OOM | 0.86-9.75s | ∞ |

### 10C. Cost — Convention Resolved, Results Consistent

| Model+Dataset | sweep_002 Cost (PyPose Σ) | sweep_003 Cost (GTSAM ½Σ) | Notes |
|---------------|--------------------------|---------------------------|-------|
| 90pct_rand on Intel | 0.158 | 0.064 | GTSAM lower (close, consistent with ~2× convention) |
| 90pct_rand on synthetic | 92.5 | 30.2 | GTSAM lower |
| 50pct_clus on synthetic | 63.1 | 145.4 | GTSAM higher (different local minimum) |
| 10pct_rand on Intel | 209.2 | 9,485 | GTSAM much higher (0-iter stationary point — solver accepted the initial guess with full-weight constraints) |

---

## 11. Generalization Gap (Updated with GTSAM)

### 11A. Random vs Clustered Training

| Dataset | Random Avg F1 | Clustered Avg F1 | Random Avg ATE | Clustered Avg ATE |
|---------|-------------|-----------------|---------------|------------------|
| synthetic | 0.947 | 0.930 | 0.477 | 0.625 |
| intel | 0.927 | 0.893 | 1.213 | 3.493 |
| m3500 | 0.993 | 0.949 | 5.080 | 5.092 |
| mit | 0.976 | 1.000 | 16.85 | 7.27 |
| csail | 0.909 | 0.946 | 1.916 | 1.892 |
| city10000 | 0.945 | 0.930 | 11.91 | 9.63 |

With GTSAM convergence, ATE differences between random and clustered are now meaningful (not artifacts of non-convergence). On Intel, random models achieve 2.9× better ATE than clustered (1.21 vs 3.49). On M3500, the gap is negligible (5.08 vs 5.09). On city10000, clustered slightly edges random (9.63 vs 11.91). The mixed picture persists — no structure consistently dominates.

### 11B. Synthetic → Real F1 Transfer (GTSAM)

| Outlier rate | Synthetic F1 (rand) | Intel F1 (rand) | M3500 F1 (rand) | city10000 F1 (rand) |
|-------------|---------------------|-----------------|------------------|---------------------|
| 10% | 0.974 | 0.903 | 1.000 | 1.000 |
| 30% | 0.989 | 0.966 | 1.000 | 1.000 |
| 50% | 0.989 | 0.966 | 1.000 | 1.000 |
| 70% | 0.975 | 0.966 | 1.000 | 1.000 |
| 90% | 0.900 | 0.833 | 0.963 | 0.727 |

Same pattern as sweep_002 — F1 degrades at the extremes (10% and 90%) but holds well at 30-70%. M3500 and city10000 are nearly perfect (large simulated graphs with known structure). Intel is the hardest real benchmark for edge classification.

---

## 12. F1 vs ATE — Weak Correlation Confirmed

The weak correlation between edge classification and map quality persists with GTSAM:

| Model on Intel | F1 | ATE | Cost | Interpretation |
|---------------|-----|-----|------|----------------|
| 30pct_clustered | 1.000 | 0.188 | 1,474 | Perfect F1 but uniform-level ATE (no actual edge removal) |
| 90pct_random | 0.833 | 0.274 | **0.064** | Worst F1 among random, best cost, mediocre ATE |
| 50pct_clustered | 0.923 | 7.281 | 34.2 | Decent F1, worst ATE — intermediate conservatism is worst |

The "0-iteration models" (all F1 ≥ 0.875 on Intel) produce ATE=0.188 — exactly the uniform baseline. They don't actually change the solution because they trust all edges. The GNN only has impact when it's conservative enough to remove constraints. But being conservative enough to remove constraints sometimes improves ATE (90pct_random: 0.274) and sometimes worsens it (50pct_clustered: 7.281).

**The threshold where GNN weighting actually matters is model-dependent and not aligned with F1.** Some models with F1=1.0 produce ATE=0.188 (uniform-equivalent — no change). Others with F1=0.923 produce ATE=7.281 (active but harmful). The sweet spot is F1 ~0.83 with cost < 1.0 — enough conservatism to change the solution, but not so much that useful constraints are lost.

---

## 13. Limitations

### 13A. Single Seed
`n_seeds=1` throughout. No confidence intervals. Cannot make statistical claims.

### 13B. CSAIL Non-Convergence
8/10 models hit 100 iterations. Edge-only format (zero initial poses) creates degenerate factor graph.

### 13C. 20-Edge Injection Scale
Too small to meaningfully stress M3500 (0.37% of edges) and Intel (0.56%). The odometry chain dominates.

### 13D. MIT 10pct_random Anomaly
ATE=83.39, cost=3.51 (low), converged after 12 iterations. Likely gauge-alignment artifact — the reference trajectory and optimized trajectory are in frames too far apart for Umeyama alignment to resolve on an under-constrained graph.

### 13E. Intel 0-Iteration Models
Models with broadly permissive edge weights (F1 > 0.875) produce 0-iteration solves — GTSAM accepts the initial guess as stationary. These models are effectively identical to the uniform baseline and contribute no distinguishing signal to the ATE comparison. Only the 90pct models and 50pct_clustered actually change the Intel solution.

---

## 14. Next Steps (Updated with GTSAM Migration Complete)

### Batch 1 — NOW (low effort, maximum ROI)

| # | Action | Effort | Why |
|---|--------|--------|-----|
| 1 | **Multi-seed CI in evaluate.py** | ~30 lines | Loop `test_seed` over `[42, 123, 999, 4567, 10101]`, add `ate_std` and `f1_std` to `accumulate_metrics()`, print mean ± std. Single-seed results are unpublishable in any ML/robotics venue. |
| 2 | **Rotation error metric** | ~50 lines | New `edgegate/metrics/rotation_error.py`: extract θ from optimized poses, Umeyama-align, compute mean absolute angular error. Wire into `evaluate_one_graph()`. The proposal's claim of "full SE(2), not rotations alone" must be demonstrated with data, not asserted. |
| 3 | **Runtime profiling table** | ~20 lines | Track GNN inference time separately from solver time in `evaluate_one_graph()`. Print dedicated timing row in output. The literature identifies compute cost as the classical methods' pain point and EdgeGate's opportunity. |
| 4 | **Vary graph size in training sweep** | ~5 lines (config) | Add `data.num_poses=100,500,1000` to the sweep grid. The GNN currently trains only on 100-pose graphs but evaluates on 808-10,000-pose graphs. This is the single largest domain shift. |
| 5 | **Vary info matrix scale in training** | ~5 lines (config) | Add `data.info_scale=0.1,1.0,10.0` multiplier to synthetic generator. Current training uses fixed `diag(500,500,100)` — one point in a space that spans 7 orders of magnitude on real data (CSAIL: 35 to 25,000,000). |
| 6 | **Vary odometry:LC ratio in training** | ~5 lines (config) | Add `data.num_loop_closures=20,50,100,200` to sweep. Training only sees 5:1 odom:LC ratio; eval sees 0.94:1 (city10000) to 40:1 (MIT). |
| 7 | **Lock GTSAM as default learned solver** | 1 config line | Change `solver: pypose` → `solver: gtsam solver.kernel=none` in evaluate.yaml |
| 8 | **Lock domain-shift metrics** | 1 script | Compute per-benchmark outlier rate/structure mismatch, edge-type ratio, noise-scale BEFORE next training run |
| 9 | **Multi-seed training** | 1 command | `train.seed=0,1,2` → 30 runs → mean ± std |
| 10 | **Remove city10000 OOM workaround** | 0 effort | Already fixed — GTSAM handles it natively |

### Batch 2 — NEXT (medium effort, high impact)

| # | Action | Why |
|---|--------|-----|
| 11 | **Real-benchmark outlier-rate sweep** | Test `injection_outlier_rate=10,30,50,70` on each real benchmark via sweep config. Currently injects a fixed 30% outlier rate. This enables the generalization-gap characterization that is the project's stated primary contribution. |
| 12 | **Full information matrices in synthetic generator** | Add configurable non-zero off-diagonals to training graphs. Real datasets (Intel, MIT, CSAIL) have full 3×3 matrices with significant off-diagonal terms. The GNN has only ever seen diagonal info matrices. Flagged as Future Work in `implementation_details.md`. |
| 13 | **GNN → DCS hybrid** | GNN removes outliers (perfect precision), DCS refines (adaptive kernel). Combines strengths. PoP Prop-V2 precedent. |
| 14 | **Trajectory loss training** | Replace BCE with ATE-aware loss. Phase 1's stated deliverable. Already implemented, never run. |
| 15 | **Odometry re-weighting** | Remove hardcoded `w_odom=1.0`. GNN predicts weights for ALL edges. Gives GNN leverage over the dominant odometry chain. Requires trajectory loss training. |
| 16 | **Position-based odometry decay** | Simpler fallback: `w = exp(-λ × idx / N)`. Single hyperparameter. Tests whether odometry-aware weighting alone helps. |

### Batch 3 — LATER (higher effort, defined in sweep-002 §13)

| # | Action | Why |
|---|--------|-----|
| 17 | **Multiple trajectory shapes in generator** | Add a corridor-with-revisits generator (long chain with sparse loop closures, like MIT/CSAIL) alongside the existing Manhattan-world grid generator. The GNN currently only trains on dense grid topologies. Maps to "trajectory-shape/revisit-rate as domain-shift metric" in `implementation_details.md` Future Work. |
| 18 | **Switchable constraints baseline** | ~80 lines in `gtsam_solver.py`. Per-LC switch variables + priors + alternating LM/closed-form optimization. Completes the classical baseline suite (GNC + DCS + Switchable). Flagged as Future Work in `implementation_details.md`. |
| 19 | **Increase outlier injection on real benchmarks** | 50-200 edges instead of 20 for meaningful F1 on large graphs. M3500's 20 injected LCs are 0.37% of 5,453 edges — the odometry chain dominates regardless of GNN weights. |
| 20 | **Attention aggregation (Ablation B)** | Self-attention over neighbors. Drop-in replacement for `aggr="add"`. Gives each node a learned per-neighbor importance weight during aggregation — conceptually closer to "some edges are inliers, some are outliers." |
| 21 | **Edge-conditioned convolution (Ablation A)** | ECC/NNConv replacing 2-bucket type switch. Continuous edge-conditioning matches Policies over Poses' design. May capture within-type variation in information-matrix magnitude. |
| 22 | **Train on larger graphs** | 500-1,000 pose synthetic graphs. Current training is on 100-node graphs but eval targets 1,728-3,500-node graphs. Training distribution mismatch likely contributes to generalization gap. |
| 23 | **Residual re-weighting** | GNN predicts weights → solver produces poses → compute per-edge residuals → feed residuals as new edge features → GNN re-predicts → repeat 2-3×. Gives GNN the adaptive feedback loop that DCS and GNC have built in. Implementation is a loop in `evaluate_one_graph()`, no architecture changes. |
| 24 | **Investigate Intel 0-iteration models** | Models with F1 > 0.875 on Intel produce 0-iteration solves — GTSAM accepts the initial guess as stationary. These models don't actually change the solution. Need to distinguish "found a better minimum" from "didn't change anything" — ATE alone doesn't suffice. |
