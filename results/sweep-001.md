# Sweep 001 — Phase 0 BCE Training

**Date:** 2026-07-16
**Codebase:** EdgeGate-SLAM (128 tests passing)
**Script:** `pixi run run-sweep`
**Output dir:** `runs/sweep_20260716_0218/`
**Status:** Training complete, evaluation complete (77 summary.json files)

---

## 1. Experiment Configuration

### Model architecture

| Parameter | Value |
|-----------|-------|
| Model | `EdgeGateGNN` (PyG, 3-layer edge-type-aware message passing) |
| Hidden dim | 64 |
| Dropout | 0.1 |
| Aggregation | Sum (`aggr="add"`) |
| Normalization | `GraphNorm` |
| Confidence head | `concat(h_i, h_j, edge_attr_emb, onehot(type))` → MLP → sigmoid |
| Odometry confidence | Hardcoded 1.0 (never passes through GNN — `GNN.md` §2.2) |
| Edge features | `[dx, dy, dθ, Ixx, Iyy, Iθθ]` — raw measurement + info-matrix diagonal |

### Training configuration

| Parameter | Value |
|-----------|-------|
| Loss | BCE (`edge_bce.py`), masked to loop-closure edges only |
| Optimizer | Adam, lr = 1e-3 |
| Epochs | 100 |
| Training graphs | 80 synthetic (20 val split) |
| Graph size | 100 poses, 20 loop-closure edges |
| Segment length | 5 (Manhattan-world trajectory) |
| Proximity threshold | 3.0 (max Euclidean distance for inlier LC candidates) |
| Solver | PyPose LM — eval only during validation, not during BCE training |
| Device | CPU (CUDA available but not explicitly used in this run) |

### Sweep axes (5 × 2 = 10 model variants)

| Axis | Values | Meaning |
|------|--------|---------|
| `outlier_rate` | 10, 30, 50, 70, 90% | Fraction of loop-closure edges that are outliers |
| `outlier_structure` | random, clustered | How outlier source poses are sampled |

**Random**: outliers scattered uniformly across the trajectory timeline.
**Clustered**: outliers concentrated in a contiguous 30%-of-trajectory time window (models a burst of perceptual aliasing, e.g. a long featureless corridor).

### Evaluation datasets

| Dataset | Nodes | Edges | Type | GT poses? | Status |
|---------|-------|-------|------|-----------|--------|
| synthetic | 100 | 119 | Generated (20 test graphs) | Yes | Full metrics |
| intel | 1,728 | 2,512 | Real robot log | No (reference-only) | F1 reported |
| M3500 | 3,500 | 5,453 | Simulated (Olson 2006) | Not in file | F1 reported |
| MIT | 808 | 827 | Real robot log | No | F1 reported |
| CSAIL | 1,045 | 1,172 | Real robot log (edge-only) | No | Null (no proximal pairs for injection) |
| manhattan | 3,500 | 5,453 | Simulated (edge-only) | Not in file | Null (no proximal pairs for injection) |
| city10000 | 10,000 | 20,687 | Simulated | Yes (in theory) | OOM killed all 13 evals |

### Baselined methods

| Method | Solver | Notes |
|--------|--------|-------|
| uniform | GTSAM LM (`kernel=none`) | Unit weights on all edges |
| DCS | GTSAM + `noiseModel.Robust` | Dynamic Covariance Scaling (Agarwal et al. 2013), `dcs_param=1.0` |
| GNC | GTSAM `GncGaussNewtonOptimizer` | Graduated Non-Convexity (Yang et al. 2020), default params, `max_iter=100` |

---

## 2. Training Results

### Best validation F1 per model

| Model | Best Val F1 | Best Epoch | Final Train Loss | Still improving at epoch 100? |
|-------|-------------|------------|------------------|------|
| 10pct_random | **0.9972** | 63 | 0.0045 | No |
| 10pct_clustered | 0.9863 | 70 | 0.0155 | No |
| 30pct_random | 0.9911 | 95 | 0.0119 | **Yes** |
| 30pct_clustered | 0.9894 | 60 | 0.0194 | No |
| 50pct_random | 0.9828 | 54 | 0.0185 | No |
| 50pct_clustered | 0.9877 | 66 | 0.0140 | No |
| 70pct_random | 0.9796 | 86 | 0.0090 | **Yes** |
| 70pct_clustered | 0.9593 | 55 | 0.0138 | No |
| 90pct_random | 0.9286 | 75 | 0.0039 | No |
| 90pct_clustered | **0.8451** | 37 | 0.0036 | No |

### Convergence analysis

- **Loss is still decreasing** for all models (last-20-epoch loss < first-20-epoch loss), but validation F1 has plateaued for 8/10 models.
- **Not overfitting**: val F1 stays close to training performance across all models. No train/val divergence.
- **Capacity-limited, not epoch-limited**: most models peak by epoch 60-70. Only 30pct_random and 70pct_random show marginal improvement past epoch 80.
- **90pct_clustered** is the hardest case — peaked at epoch 37 at F1=0.845. With 18/20 loop-closures being outliers, the model hits its architectural ceiling early.
- **Implication**: more epochs won't help meaningfully. Gains require more training graphs, larger architecture, or Phase 1 ablations (attention aggregation, edge-conditioned convolution).

---

## 3. Evaluation Results

### 3A. Learned models on synthetic (20 test graphs, 280 labeled edges)

| Model | F1 | Precision | Recall | ATE (RMSE) | TP | FP | FN |
|-------|-----|-----------|--------|------------|-----|-----|-----|
| 10pct_random | 0.9807 | 0.9622 | 1.0000 | 0.9432 | 280 | 11 | 0 |
| 10pct_clustered | 0.9588 | 0.9238 | 0.9964 | 1.4841 | 279 | 23 | 1 |
| 30pct_random | 0.9877 | 0.9756 | 1.0000 | 0.4910 | 280 | 7 | 0 |
| 30pct_clustered | 0.9911 | 0.9859 | 0.9964 | 0.7489 | 279 | 4 | 1 |
| **50pct_random** | **0.9929** | **0.9894** | **0.9964** | **0.4818** | 279 | 3 | 1 |
| 50pct_clustered | 0.9838 | 0.9892 | 0.9786 | 0.4939 | 274 | 3 | 6 |
| 70pct_random | 0.9783 | 0.9926 | 0.9643 | **0.2644** | 270 | 2 | 10 |
| 70pct_clustered | 0.9554 | 0.9961 | 0.9179 | 0.4452 | 257 | 1 | 23 |
| 90pct_random | 0.9168 | **1.0000** | 0.8464 | **0.1690** | 237 | 0 | 43 |
| 90pct_clustered | 0.7800 | **1.0000** | 0.6393 | 0.4553 | 179 | 0 | 101 |

**Key observation**: ATE and F1 are not perfectly correlated. 90pct_random has the worst F1 (0.917) but the best ATE (0.169). At extreme outlier rates, the model learns extreme conservatism — precision=1.0, only classifies edges it's certain about as inliers, and lets the solver work with fewer but cleaner constraints. Eliminating false positives matters more for map quality than maximizing recall.

### 3B. Learned models on real benchmarks

**Intel** (1,728 nodes, real robot log — 14 injected loop-closure labels):

| Model | F1 | Precision | Recall |
|-------|-----|-----------|--------|
| 10pct_random | 1.000 | 1.000 | 1.000 |
| 30pct_random | 1.000 | 1.000 | 1.000 |
| 50pct_random | 1.000 | 1.000 | 1.000 |
| 70pct_random | 1.000 | 1.000 | 1.000 |
| 90pct_random | 0.923 | 1.000 | 0.857 |
| 10pct_clustered | 0.966 | 0.933 | 1.000 |
| 30pct_clustered | 1.000 | 1.000 | 1.000 |
| 50pct_clustered | 0.963 | 1.000 | 0.929 |
| 70pct_clustered | 0.783 | 1.000 | 0.643 |
| 90pct_clustered | 0.526 | 1.000 | 0.357 |

**M3500** (3,500 nodes, simulated — 14 injected loop-closure labels):

| Model | F1 | Precision | Recall |
|-------|-----|-----------|--------|
| 10-70pct_random | 1.000 | 1.000 | 1.000 |
| 90pct_random | 0.880 | 1.000 | 0.786 |
| 10-30pct_clustered | 1.000 | 1.000 | 1.000 |
| 50-70pct_clustered | 0.923 | 1.000 | 0.857 |
| 90pct_clustered | 0.600 | 1.000 | 0.429 |

**MIT** (808 nodes, real robot log — 14 injected loop-closure labels):

| Model | F1 | Precision | Recall |
|-------|-----|-----------|--------|
| 10pct_random | 1.000 | 1.000 | 1.000 |
| 30pct_random | 1.000 | 1.000 | 1.000 |
| 50pct_random | 1.000 | 1.000 | 1.000 |
| 70pct_random | 0.963 | 1.000 | 0.929 |
| 90pct_random | 0.963 | 1.000 | 0.929 |
| 10pct_clustered | 1.000 | 1.000 | 1.000 |
| 30pct_clustered | 0.963 | 1.000 | 0.929 |
| 50pct_clustered | 0.963 | 1.000 | 0.929 |
| 70pct_clustered | 0.963 | 1.000 | 0.929 |
| 90pct_clustered | 0.963 | 1.000 | 0.929 |

**CSAIL & manhattan**: Null across all models — outlier injection found zero proximal pairs in edge-only `.g2o` files. Evaluations completed but no labeled edges to compute F1 against.

**city10000**: OOM-killed (exit 137) for all 10 learned evaluations + 3 baselines. The 10,000-node graph requires more RAM than available for the GTSAM reference solve needed for outlier injection.

### 3C. Classical baselines

| Method | Synthetic ATE | Real benchmarks (all) |
|--------|--------------|----------------------|
| DCS | **0.1266** | Null — no GT poses available |
| uniform | 0.5868 | Null — no GT poses available |
| GNC | Crash (4/7 datasets) | Null on the 3 that ran |

Edge-F1 is not reported for classical baselines — GNC, DCS, and uniform do not produce per-edge confidence scores. This is consistent with the literature (Policies over Poses Table II only reports precision/recall for methods with explicit denoising modules).

GNC failed on 4/7 datasets including synthetic (crashed mid-loop after 6/20 graphs). Only `max_iterations=100` was set; all other GNC parameters used GTSAM defaults. Likely parameter tuning issue, not fundamental algorithm failure.

---

## 4. Analysis

### 4A. Generalization gap — random vs clustered

Averaged across all datasets where both structures produce valid F1:

| Dataset | Random avg F1 | Clustered avg F1 | Gap |
|---------|-------------|-----------------|-----|
| Synthetic | 0.971 | 0.934 | +4.0% |
| Intel | 0.985 | 0.848 | +13.8% |
| M3500 | 0.976 | 0.849 | +12.7% |
| MIT | 0.978 | 0.962 | +1.6% |

**Models trained on scattered outliers (random) generalize better than those trained on temporally clustered outliers.** The gap widens as outlier rate increases — at 90%, random hits F1=0.92 on Intel vs clustered's F1=0.53. This is the kind of structured generalization-gap finding the proposal frames as its primary contribution.

### 4B. ATE — DCS wins on raw accuracy, GNN provides interpretability

| Method | Synthetic ATE | Edge F1? | Runs on all datasets? |
|--------|-------------|---------|----------------------|
| DCS | **0.127** | No | Yes |
| GNN (best, 90pct_random) | 0.169 | Yes (0.92) | Yes (except city10000) |
| GNN (best F1, 50pct_random) | 0.482 | Yes (0.99) | Yes (except city10000) |
| uniform | 0.587 | No | Yes |
| GNC | — | No | No (crashed 4/7) |

DCS achieves 1.3× better ATE than the best GNN. This matches the proposal's prediction (§4): "classical still wins on raw accuracy." The GNN's contribution is **not** better ATE — it's **(a)** per-edge interpretability (you know which loop closures are suspect), **(b)** the generalization-gap characterization (random vs clustered patterns), and **(c)** amortized inference potential (GNN forward pass in ms vs DCS solve in seconds — not yet measured but consistent with the literature).

### 4C. Recovery at high outlier rates

At 90% outliers (18/20 loop closures are wrong), the 90pct_random model achieves:
- Precision = 1.000 (zero false positives)
- Recall = 0.846
- ATE = 0.169 (lowest of any model)

The model sacrifices recall for precision at extreme outlier rates — better to trust fewer edges and solve a cleaner problem than to risk trusting an outlier and corrupting the map. This is the correct strategy for safety-critical SLAM.

### 4D. Edge-only graphs (CSAIL, manhattan)

Both datasets from SE-Sync lack `VERTEX_SE2` lines. Our parser infers vertices from edge endpoints, but the inferred `node_init` is all-zeros (no initial pose guess). This produces zero proximal pairs for outlier injection, so F1 is unavailable. These graphs are still solvable (odometry chain produces a valid trajectory) but can't participate in the F1-based generalization-gap measurement without manual annotation.

---

## 5. Limitations of This Sweep

### 5A. Missing metrics (computed by solver, never saved)

| Metric | Computed? | Saved to summary.json? | Value |
|--------|----------|----------------------|-------|
| **Objective cost F(x)** | Yes (every solve) | No — discarded in eval code | Policies over Poses' primary metric |
| **Num iterations** | Yes (every solve) | No — discarded | Convergence speed comparison |
| **Converged flag** | Yes (every solve) | No — discarded | Reliability indicator |
| **Inference time** | Never measured | N/A | Speed comparison |
| **ATE on real data** | Never computed | N/A | Reference trajectory exists but never stored as pseudo-GT |

The solver returns `(poses, converged, iters, cost)` — all four values are present in RAM, but `converged`, `iters`, and `cost` are discarded by the evaluation wrapper functions. These are the metrics the literature uses for comparison (Policies over Poses uses objective cost as its headline number). A 4-line code fix would preserve them.

### 5B. Single-seed training

Each (outlier_rate, structure) combination was trained exactly once. Without multi-seed confidence intervals, we cannot distinguish "model A is genuinely better" from "model A got a favorable random init." The trends are clear enough for a first sweep, but publication-quality claims require mean ± std across 3-5 training seeds.

### 5C. GNC failures unexplained

GNC crashed on synthetic mid-loop (6/20 graphs processed) and failed entirely on city10000, CSAIL, manhattan. Only `max_iterations` was set — all other GNC parameters use GTSAM defaults. Without investigating whether this is a parameter tuning issue or a fundamental limitation, we cannot claim "GNN is more reliable than GNC."

### 5D. city10000 OOM

The reference GTSAM solve for outlier injection on a 10,000-node graph exceeds available RAM. None of the 13 evaluations on city10000 produced a summary.json. This dataset requires >16GB RAM or a different injection strategy.

---

## 6. Next Steps

### Code fixes (before any re-run)

1. **Save solver metrics** — `evaluate_one_graph` and `evaluate_one_graph_classical` should include `final_cost`, `num_iterations`, `converged` in their return dicts (3 lines, already computed for free)
2. **Reference ATE on real data** — `_load_benchmark` already computes `ref_poses`; store it as `graph.gt_node_poses` after outlier injection (1 line)
3. **Add timing** — wrap `solver.solve()` with `time.monotonic()` (2 lines)

### Re-run evaluation on existing models (no re-training needed)

All 10 `model_best.pt` files are saved at `runs/sweep_20260716_0218/train/`. Run:

```bash
pixi run run-sweep --skip-train --output runs/sweep_002
```

With the fixes above, this produces the full comparison matrix in ~2-3 hours.

### Multi-seed training (for confidence intervals)

```bash
pixi run train --multirun data.outlier_rate=10,30,50,70,90 data.outlier_structure=random,clustered train.seed=0,1,2
```

30 training runs → 3 seeds per config → mean ± std for all metrics.

### Architecture ablations (Phase 1 — GNN.md §4)

1. **Ablation A**: Edge-conditioned convolution (ECC/NNConv) replacing the 2-bucket type switch
2. **Ablation B**: Attention-based aggregation (`TransformerConv`) replacing sum-pooling
3. **Ablation C**: Recurrent (GRU) processing of the odometry chain
4. **Larger capacity**: `hidden_dim=128`, more training graphs (500 instead of 100)

### Classical baseline investigation

1. Tune GNC parameters per dataset (graduation step `mu`, inlier threshold)
2. Extend GTSAM wrapper to surface internal GNC/DCS weights as pseudo-F1
3. Investigate whether GNC crashes are a real reliability issue or parameter tuning

### Domain-shift metrics (Phase 1 requirement)

Lock before next training run per `implementation_details.md` §"Domain-Shift Characterization Metrics":
- Outlier rate/structure mismatch between synthetic training and each real benchmark
- Edge-type ratio (odometry : loop-closure) mismatch
- Noise-scale mismatch (information-matrix magnitude distributions)

### CSAIL/manhattan handling

Option A: manually annotate a subset of loop-closure edges as inlier/outlier
Option B: exclude from F1-based comparison, use ATE-only
Option C: fix outlier injection to work on edge-only graphs (seed poses from odometry chain)
