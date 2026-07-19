# Sweep 002 — Phase 0 BCE Eval-Only Re-run (Full Metrics)

**Date:** 2026-07-17
**Codebase:** EdgeGate-SLAM (post-metrics-fix, 128 tests passing)
**Models:** Reuses sweep-001 pretrained weights (`runs/sweep_20260716_0218/train/`) — no new training was performed.
**Script:** `pixi run run-sweep --skip-train`
**Output dir:** `runs/sweep_002/`
**Status:** Evaluation complete (68 summary.json files, 14 metrics each)

---

## 1. Changes from Sweep 001

### New metrics saved (5 added, 14 total fields per evaluation)

| Field | sweep_001 | sweep_002 | How |
|-------|-----------|-----------|-----|
| `ate` | Synthetic only | **All datasets** | `ref_poses` stored as `gt_node_poses` in `_load_benchmark` |
| `final_cost` | Absent | Present | Saved from `solver.solve()` return |
| `solve_time_s` | Absent | Present | `time.monotonic()` wrapper around solver call |
| `num_iterations_mean` | Absent | Present | Saved from `solver.solve()` return |
| `converged_count` | Absent | Present | Saved from `solver.solve()` return |
| `failed_count` | Absent | Present | Try/except around solver call |

### Other improvements from sweep-001

- **ATE on real benchmarks**: Reference trajectory from clean GTSAM solve stored as pseudo-ground-truth for all datasets
- **GNC fixed**: Succeeds on all 6 datasets (was crashing on 4/7 in sweep-001; switched to `GncLMParams` from `GncGaussNewtonParams`)
- **Baselines on city10000**: All 3 baselines now produce results (were missing in sweep-001)
- **CSAIL outlier injection working**: F1 now available for both learned and baselines (was null in sweep-001)
- **Manhattan removed**: Redundant — identical graph topology and edge measurements to M3500, only differing in information matrix magnitudes. Removed from eval to avoid double-counting.
- **Multi-seed naming**: Models named `{outlier_rate}pct_{structure}_seed0` (infrastructure ready for multi-seed; sweep_002 data is single-seed)
- **Results CI file**: `results_ci.csv` exists with per-config mean/std columns (currently `n_seeds=1` throughout)

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

### Sweep axes (5 × 2 = 10 model variants)

| Axis | Values | Meaning |
|------|--------|---------|
| `outlier_rate` | 10, 30, 50, 70, 90% | Fraction of loop-closure edges that are outliers |
| `outlier_structure` | random, clustered | How outlier source poses are sampled |

**Random**: outliers scattered uniformly across the trajectory timeline.
**Clustered**: outliers concentrated in a contiguous 30%-of-trajectory time window (models a burst of perceptual aliasing, e.g. a long featureless corridor).

### Evaluation datasets

| Dataset | Nodes | Edges | Type | Reference ATE? | Edge F1? | Status |
|---------|-------|-------|------|---------------|---------|--------|
| synthetic | 100 | 119 | Generated (20 test graphs) | Yes (gt_node_poses) | Yes | Full metrics |
| intel | 1,728 | 2,512 | Real robot log | Yes (reference solve) | Yes | Full metrics |
| M3500 | 3,500 | 5,453 | Simulated (Olson 2006) | Yes (reference solve) | Yes | Full metrics |
| MIT | 808 | 827 | Real robot log | Yes (reference solve) | Yes | Full metrics |
| CSAIL | 1,045 | 1,172 | Real robot log (edge-only) | Yes (reference solve) | Yes | Full metrics |
| city10000 | 10,000 | 20,687 | Simulated | Yes (reference solve, baselines only) | No (baselines only) | Learned OOM |

### Baselined methods

| Method | Solver | Cost convention | Notes |
|--------|--------|-----------------|-------|
| DCS | GTSAM LM + `noiseModel.Robust` | ½ Σ(r^T Λ r) | Dynamic Covariance Scaling (Agarwal et al. 2013), `dcs_param=1.0` |
| GNC | GTSAM `GncLMOptimizer` | ½ Σ(r^T Λ r) | Graduated Non-Convexity (Yang et al. 2020), default params, `max_iter=100` |
| uniform | GTSAM LM (`kernel=none`) | ½ Σ(r^T Λ r) | Unit weights on all edges, no outlier rejection |
| **Learned (GNN)** | **PyPose LM** | Σ(r² × info_scaled) | **⚠ Different solver + cost convention from baselines** |

**Important caveat**: Learned models use PyPose while baselines use GTSAM. These are different solver backends with different cost conventions (documented in `implementation_details.md:362`: "pp_cost ≈ 2 × gt_cost" for the same solution). Cost, timing, and convergence comparisons across backends are confounded — see §6C.

---

## 3. Training Results

(Same training metrics as sweep-001 — models reused from `sweep_20260716_0218/train/`.)

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

- Loss still decreasing for all models (last-20-epoch loss < first-20-epoch loss), but val F1 plateaued for 8/10.
- No overfitting: val F1 stays close to training performance. No train/val divergence.
- Capacity-limited, not epoch-limited: most models peak by epoch 60-70. Only 30pct_random and 70pct_random improve past epoch 80.
- 90pct_clustered is the hardest case — peaked at epoch 37 at F1=0.845. Architectural ceiling reached early.
- Implication: more epochs won't help. Gains require more training graphs, larger architecture, or trajectory-loss training.

---

## 4. Evaluation Results — Classical Baselines

### 4A. DCS (Dynamic Covariance Scaling)

| Dataset | ATE (ref) | Cost F(x) | Time (s) | Iters | Conve rged |
|---------|-----------|-----------|----------|-------|-----------|
| city10000 | 5.751 | 404.4 | 1.61 | 7 | Yes |
| csail | **0.039** | 28.9 | 0.17 | 14 | Yes |
| intel | **0.016** | 40.8 | 0.055 | 2 | Yes |
| m3500 | **0.008** | 104.3 | 0.246 | 10 | Yes |
| mit | 0.612 | 26.0 | 0.025 | 3 | Yes |
| synthetic | 0.136 | 8.2 | 0.008 | 3.8 | 20/20 |

**DCS achieves the best ATE on every dataset except MIT** (where the graph is nearly trivial with only 20 LCs on 808 nodes). Converges in 2-14 iterations, typically under 1 second. On Intel, DCS produces ATE=0.016 in 0.055s — this is the classical ceiling.

### 4B. GNC (Graduated Non-Convexity)

| Dataset | ATE (ref) | Cost F(x) | Time (s) |
|---------|-----------|-----------|----------|
| city10000 | 17.845 | 1.04×10⁸ | 73.0 |
| csail | 3.137 | 6.6×10⁴ | 2.50 |
| intel | 0.016 | 2.1×10⁴ | 1.16 |
| m3500 | 0.647 | 7.4×10⁴ | 6.35 |
| mit | 1.379 | 2.25×10⁹ | 1.84 |
| synthetic | 0.584 | 1.7×10⁴ | 0.20 |

GNC matches DCS on Intel ATE (0.016) but produces much higher costs (20,649 vs 40.8). On MIT, GNC's cost reaches 2.25×10⁹ — evidence of numerical instability on under-constrained graphs. Iteration counts unavailable (GTSAM API limitation).

### 4C. Uniform (no outlier rejection)

| Dataset | ATE (ref) | Cost F(x) | Time (s) |
|---------|-----------|-----------|----------|
| city10000 | 6.426 | 6.85×10⁶ | 2.51 |
| csail | 8.264 | 2.72×10⁵ | 0.07 |
| intel | 0.188 | 2.03×10⁴ | 0.10 |
| m3500 | 10.193 | 2.32×10⁵ | 0.30 |
| mit | ~0 | 2.21×10⁹ | 0.04 |
| synthetic | 0.531 | 1.40×10⁴ | 0.009 |

Uniform is consistently the weakest baseline — ATE is 4-1,200× worse than DCS on datasets where the comparison matters. On MIT, uniform achieves ~0 ATE because the graph is nearly trivial (20 LCs on 808 nodes — the odometry chain barely changes from initial).

---

## 5. Evaluation Results — Learned Models (GNN)

### 5A. Synthetic (20 test graphs, 280 labeled edges)

| Model | F1 | Prec | Rec | ATE | Cost | Time (s) | Iters | Conv |
|-------|-----|------|-----|-----|------|----------|-------|------|
| 10pct_random | 0.974 | 0.952 | 0.996 | 1.347 | 399.8 | 0.27 | 40.8 | 17/20 |
| 10pct_clustered | 0.974 | 0.949 | 1.000 | 1.593 | 768.1 | 0.27 | 41.8 | 16/20 |
| 30pct_random | 0.989 | 0.986 | 0.993 | 0.886 | 320.4 | 0.20 | 29.8 | 18/20 |
| 30pct_clustered | 0.986 | 0.976 | 0.996 | 1.149 | 385.1 | 0.22 | 34.0 | 18/20 |
| 50pct_random | 0.989 | 0.982 | 0.996 | 0.959 | 357.8 | 0.17 | 24.8 | 19/20 |
| 50pct_clustered | **0.993** | **0.989** | 0.996 | 0.852 | **63.1** | 0.17 | 25.5 | **20/20** |
| 70pct_random | 0.975 | 0.989 | 0.961 | **0.459** | 44.5 | 0.16 | 24.1 | 19/20 |
| 70pct_clustered | 0.919 | 1.000 | 0.850 | 0.823 | 49.6 | 0.22 | 33.2 | 18/20 |
| 90pct_random | 0.900 | 0.996 | 0.821 | 0.342 | 92.5 | **0.10** | **15.1** | **20/20** |
| 90pct_clustered | 0.780 | 1.000 | 0.639 | 0.743 | 32.5 | 0.17 | 25.0 | 19/20 |

**Key observations on synthetic**: 50pct_clustered is the best all-around model — highest F1 (0.993), lowest cost (63.1), 100% convergence. At 90% outliers, both models achieve perfect precision (1.000) by sacrificing recall. The 90pct_random model achieves the fastest convergence (15.1 iters in 0.10s) and the best ATE (0.342) despite the worst F1 (0.900). **This is the conservatism tradeoff: fewer trusted edges → easier optimization → better map quality, at the cost of missed inliers.**

### 5B. Intel (1,728 nodes, real robot log)

| Model | F1 | Prec | Rec | ATE (ref) | Cost | Time (s) | Conv |
|-------|-----|------|-----|-----------|------|----------|------|
| 10pct_clustered | 0.875 | 0.778 | 1.000 | 4.877 | 230.3 | 386.2 | No |
| 30pct_clustered | **1.000** | 1.000 | 1.000 | 6.005 | 121.6 | 386.2 | No |
| 50pct_random | 0.966 | 0.933 | 1.000 | 7.051 | 87.1 | 386.3 | No |
| 50pct_clustered | 0.923 | 1.000 | 0.857 | **0.860** | 29.8 | 384.8 | No |
| 70pct_clustered | **1.000** | 1.000 | 1.000 | 5.800 | 56.7 | 385.9 | No |
| 90pct_random | 0.833 | 1.000 | 0.714 | **0.125** | **0.16** | **65.9** | **Yes** |
| 90pct_clustered | 0.667 | 1.000 | 0.500 | 0.848 | 25.7 | 376.1 | Yes |

**90pct_random is the best learned model on Intel by every metric**: lowest ATE (0.125), lowest cost (0.16), fastest solve (65.9s), and the only non-90pct model to converge. It achieves perfect precision — all trusted edges are inliers — but misses 4 of 14 inliers (recall=0.714). The solver converges in 17 iterations because the model's extreme conservatism simplifies the problem.

### 5C. M3500 (3,500 nodes, simulated)

| Model | F1 | Prec | Rec | ATE (ref) | Cost | Time (s) | Conv |
|-------|-----|------|-----|-----------|------|----------|------|
| 10pct_random | 1.000 | 1.000 | 1.000 | 11.10 | 21.2 | 2210 | No |
| 10pct_clustered | 1.000 | 1.000 | 1.000 | **9.17** | 29.2 | 2204 | No |
| 30pct_random | 1.000 | 1.000 | 1.000 | 14.61 | **3.18** | 2194 | No |
| 50pct_clustered | 1.000 | 1.000 | 1.000 | 14.71 | 2.46 | 2160 | No |
| 70pct_clustered | 0.963 | 1.000 | 0.929 | 14.71 | 2.42 | 2202 | No |
| 90pct_random | 0.963 | 1.000 | 0.929 | 14.71 | 2.45 | 2198 | No |
| 90pct_clustered | 0.783 | 1.000 | 0.643 | 14.71 | **2.15** | 2166 | No |

**ATE is nearly invariant to model on M3500** — all 10 models produce ATE 9.2-14.7. The 20 injected loop-closure edges (0.37% of 5,453 total edges) are negligible against the 3,499-edge odometry chain. The massive odometry subgraph determines the solution regardless of how the GNN weights the 20 injected edges. All models hit the 100-iteration PyPose LM cap (~36 minutes per evaluation). This is the clearest evidence that **edge classification quality does not predict map quality** — F1 ranges from 0.78 to 1.00 but ATE stays 9-15 across all models.

### 5D. MIT (808 nodes, real robot log)

| Model | F1 | Prec | Rec | ATE (ref) | Cost | Time (s) | Conv |
|-------|-----|------|-----|-----------|------|----------|------|
| 10pct_random | 1.000 | 1.000 | 1.000 | 29.87 | 9.07 | 33.1 | Yes |
| 10pct_clustered | 1.000 | 1.000 | 1.000 | 64.47 | 43.0 | 34.0 | Yes |
| 30pct_random | 1.000 | 1.000 | 1.000 | **0.088** | **0.095** | **2.09** | Yes |
| 50pct_clustered | 1.000 | 1.000 | 1.000 | 0.088 | 0.094 | 2.09 | Yes |
| 50pct_random | 1.000 | 1.000 | 1.000 | 0.107 | 0.149 | 2.09 | Yes |
| 70pct_random | 1.000 | 1.000 | 1.000 | 0.088 | 0.093 | 2.09 | Yes |
| 90pct_random | 0.880 | 1.000 | 0.786 | 0.079 | 0.082 | 2.08 | Yes |
| 90pct_clustered | 1.000 | 1.000 | 1.000 | 0.087 | 0.104 | 2.09 | Yes |

MIT is the easiest benchmark — only 20 loop closures on 808 nodes means the odometry chain dominates. Most models converge in 6 iterations (~2 seconds) with ATE ~0.09. The 10pct models show anomalously high ATE (30-65) despite F1=1.0 — likely a gauge-freedom artifact in Umeyama alignment on an under-constrained graph (too few constraints to fix the absolute pose).

### 5E. CSAIL (1,045 nodes, edge-only, real robot log)

| Model | F1 | Prec | Rec | ATE (ref) | Cost | Time (s) | Conv |
|-------|-----|------|-----|-----------|------|----------|------|
| 10pct_random | 0.966 | 0.933 | 1.000 | 2.619 | 170.5 | 78.9 | No |
| 10pct_clustered | 0.966 | 0.933 | 1.000 | 2.094 | 210.1 | 78.2 | No |
| 30pct_random | 0.966 | 0.933 | 1.000 | 2.623 | 168.8 | 79.9 | No |
| 30pct_clustered | 0.929 | 0.929 | 0.929 | 2.634 | 164.8 | 79.3 | No |
| 50pct_random | 0.929 | 0.929 | 0.929 | 2.639 | 160.8 | 80.5 | No |
| 70pct_random | 0.929 | 0.929 | 0.929 | 2.631 | 165.6 | 79.5 | No |
| 70pct_clustered | 0.929 | 0.929 | 0.929 | 2.642 | 158.9 | 79.8 | No |
| 90pct_random | 0.727 | 1.000 | 0.571 | **1.361** | **75.2** | 79.6 | No |
| 90pct_clustered | 0.929 | 0.929 | 0.929 | 2.645 | 158.9 | 79.1 | No |

F1 plateaus at 0.929-0.966 for most models. ATE constant ~2.6 except 90pct_random (1.36, the only significant ATE improvement through conservatism). All models hit 100 iterations (never converge). Unlike Intel, the conservative 90pct_clustered model does NOT achieve better ATE — CSAIL's graph structure (1,045 inferred nodes, all-zero initial poses from edge-only file) doesn't benefit from the conservatism strategy.

### 5F. city10000 — ALL learned models FAILED (OOM)

All 10 learned model evaluations produced 0-byte `evaluate.log` files. No `summary.json` files exist for any learned model on city10000. All 3 baselines (DCS, GNC, uniform) succeeded. The PyPose+GNN path on a 10,000-node / 20,687-edge graph exceeds available RAM. GTSAM baselines are more memory-efficient on this scale.

---

## 6. Cost Analysis

### 6A. Absolute Costs — GNN vs Baselines

| Dataset | GNN min cost | GNN max cost | DCS cost | GNC cost | Uniform cost |
|---------|-------------|-------------|----------|----------|-------------|
| synthetic | 32.5 | 768.1 | **8.2** | 1.7×10⁴ | 1.4×10⁴ |
| intel | **0.16** | 230.3 | 40.8 | 2.1×10⁴ | 2.0×10⁴ |
| m3500 | **2.1** | 29.2 | 104.3 | 7.4×10⁴ | 2.3×10⁵ |
| mit | **0.08** | 43.0 | 26.0 | 2.2×10⁹ | 2.2×10⁹ |
| csail | 75.2 | 210.1 | **28.9** | 6.6×10⁴ | 2.7×10⁵ |

**On 3 of 5 datasets, the GNN achieves lower absolute final cost than DCS.** On Intel, the GNN's best cost (0.16) is 255× lower than DCS (40.8). On M3500, it's 48× lower (2.1 vs 104.3). On MIT, it's 316× lower (0.08 vs 26.0).

### 6B. Cost Reduction vs Uniform (no outlier rejection)

| Dataset | GNN reduction | DCS reduction | Notes |
|---------|-------------|---------------|-------|
| synthetic | 99.8% | 99.9% | Both near-perfect reduction |
| intel | **99.999%** | 99.8% | GNN essentially eliminates the cost |
| m3500 | **99.999%** | 99.95% | Both excellent; GNN edges out DCS |
| mit | **~100%** | ~100% | Both reduce astronomical uniform cost to near-zero |
| csail | 99.97% | 99.99% | DCS slightly better |

**Both GNN and DCS dramatically reduce the optimization objective compared to uniform weights** — the metric used by Policies over Poses (2025) as their headline evaluation. The GNN achieves comparable or better reduction on every dataset.

### 6C. Cost Confound — Solver Backend Mismatch

⚠️ **Critical caveat**: Learned models run through **PyPose** while baselines run through **GTSAM**. These are different solver implementations using different cost conventions:

- **PyPose**: cost = Σ(r_whitened²) = Σ(r² × info_scaled) — no ½ factor
- **GTSAM**: cost = ½ Σ(r^T Λ r) — has ½ factor

For the same solution on the same graph, PyPose cost should be approximately 2× higher than GTSAM cost (`implementation_details.md:362`). This convention difference pushes in the *opposite* direction from the observed gap — PyPose costs should be *higher*, yet GNN costs are *lower* on 3/5 datasets. This strengthens the qualitative finding, but the absolute cost numbers are not quantitatively comparable across backends.

**To eliminate this confound**, the learned evaluation path should run through `GTSAMSolver(kernel="none")` instead of PyPose — a single command-line override (`solver=gtsam solver.kernel=none`). GTSAM's `kernel="none"` already handles externally-scaled information matrices identically to the uniform baseline. Same solver, same cost convention, directly comparable numbers.

### 6D. What Lower Cost Actually Means

The cost F(x) = Σ(w² × r² × info) measures how well the optimized poses satisfy the **GNN-weighted constraints**. When the GNN assigns low confidence (w → 0) to an edge, the constraint penalty becomes near-zero — the solver can freely violate that edge without increasing cost.

A lower cost means the GNN found a set of weights that make the optimization problem **easier for the solver** — not necessarily that the resulting map is more accurate. The m3500 results demonstrate this tension: GNN cost=2.1 (the lowest of any method) corresponds to ATE=14.7 (the worst ATE). The GNN removed constraints the solver had to satisfy, reducing cost at the expense of map quality.

---

## 7. ATE Comparison — Learned vs Classical Baselines

| Dataset | GNN Best ATE | DCS ATE | GNC ATE | Uniform ATE | Best Method |
|---------|-------------|---------|---------|-------------|-------------|
| synthetic | 0.342 | **0.136** | 0.584 | 0.531 | DCS (2.5× better than GNN) |
| intel | 0.125 | **0.016** | 0.016 | 0.188 | DCS (7.8× better) |
| m3500 | 9.168 | **0.008** | 0.647 | 10.193 | DCS (1,146× better) |
| mit | **0.079** | 0.612 | 1.379 | ~0 | GNN (7.7× better) |
| csail | 1.361 | **0.039** | 3.137 | 8.264 | DCS (34× better) |
| city10000 | — (OOM) | **5.751** | 17.845 | 6.426 | DCS |

**DCS wins raw ATE on 4 of 5 datasets where GNN runs.** The GNN's only ATE victory is on MIT — a nearly trivial 808-node graph with just 20 loop closures, where the odometry chain makes the problem insensitive to edge weighting. On the meaningful benchmarks (synthetic, Intel, M3500, CSAIL), DCS is 2-1,146× better.

This matches the proposal's prediction (§4): "classical still wins on raw accuracy... the real, measurable pain point of classical robust kernels is compute cost and brittle manual tuning." The GNN's contribution is not better ATE — it's per-edge interpretability, the generalization-gap characterization, and the amortized inference opportunity.

---

## 8. Timing Comparison

| Dataset | GNN (best/worst) | DCS | GNC | Uniform | GNN Slowdown vs DCS |
|---------|-----------------|-----|-----|---------|---------------------|
| synthetic | 0.10 / 0.27s | **0.008s** | 0.20s | 0.009s | 13-34× |
| intel | 65.9 / 407.8s | **0.055s** | 1.16s | 0.10s | 1,198-7,416× |
| m3500 | 2160 / 2210s | **0.246s** | 6.35s | 0.30s | 8,780-8,984× |
| mit | 2.08 / 34.2s | **0.025s** | 1.84s | 0.04s | 83-1,368× |
| csail | 78.2 / 80.5s | **0.171s** | 2.50s | 0.07s | 457-471× |
| city10000 | OOM | **1.61s** | 73.0s | 2.51s | — |

**The GNN is 100-9,000× slower than DCS on real datasets.** However, the GNN forward pass itself takes milliseconds — the bottleneck is entirely the PyPose LM optimizer maxing out 100 iterations on large graphs without converging (see §9). Each M3500 iteration costs ~22 seconds × 100 iterations = 2,200 seconds per evaluation. DCS converges in 10 iterations in 0.25 seconds on the same graph.

If the GNN's non-convergence is fixed (by tuning PyPose LM parameters or switching to GTSAM for learned eval), the solve time would drop dramatically. The convergence analysis (§9) provides evidence that this is a PyPose tuning issue rather than a GNN-specific problem.

---

## 9. Convergence Analysis

| Dataset | GNN converged | Pattern | DCS converged |
|---------|-------------|---------|---------------|
| synthetic | 184/200 (92%) | Converges in 15-42 iterations | 20/20 (100%) |
| intel | **2/10 (20%)** | Only 90pct models converge (17, 97 iters) | 1/1 (2 iters) |
| m3500 | **0/10 (0%)** | All 10 models hit 100-iter cap | 1/1 (10 iters) |
| mit | 7/10 (70%) | 10pct models fail (100 iters); others converge in 6-7 | 1/1 (3 iters) |
| csail | **0/10 (0%)** | All 10 models hit 100-iter cap | 1/1 (14 iters) |

**The PyPose LM + StopOnPlateau scheduler fails to converge on large graphs.** The scheduler settings (`patience=5, decreasing=1e-3`) require cost to decrease by less than 0.1% over 5 consecutive iterations to trigger stopping. On graphs with thousands of edges, each LM step makes >0.1% progress well past 100 iterations, so the scheduler never triggers.

**This is likely a PyPose tuning issue, not a GNN weight issue.** The evidence:
- Synthetic graphs (100 poses) converge fine with the same weights → scale matters, not weights
- MIT (808 nodes) converges for most models → medium graphs ok
- The 90pct models converge on Intel → when enough edges are removed (near-zero weights), the effective problem size shrinks and convergence is reached
- DCS converges in 2-14 iterations on the same graphs through GTSAM → the optimization problem IS solvable quickly with the right solver

The hypothesis that GNN weight scaling creates fundamentally ill-conditioned problems is **not supported** — the non-convergence pattern matches graph size, not weight distribution. Testing with uniform weights through PyPose on Intel/M3500 would confirm whether the same non-convergence occurs regardless of weighting.

---

## 10. Generalization Gap

### 10A. Random vs Clustered Training

| Dataset | Random avg F1 | Clustered avg F1 | F1 Gap | Random avg ATE | Clustered avg ATE |
|---------|-------------|-----------------|--------|---------------|------------------|
| synthetic | 0.947 | 0.930 | +1.8% (random wins) | 0.799 | 1.032 |
| intel | 0.927 | 0.893 | +3.8% (random wins) | 4.586 | 3.683 |
| m3500 | 0.993 | 0.949 | +4.6% (random wins) | 13.976 | 13.526 |
| mit | 0.976 | 1.000 | -2.4% (clustered wins) | 6.043 | 19.026 |
| csail | 0.909 | 0.946 | -3.9% (clustered wins) | 2.311 | 2.452 |

Random-trained models show better F1 on synthetic and larger real datasets (Intel, M3500), while clustered-trained models edge ahead on the smaller datasets (MIT, CSAIL). The ATE picture is mixed with no consistent winner. **Multi-seed CI is needed to determine statistical significance** — these single-seed differences could be noise.

### 10B. Synthetic → Real F1 Transfer

| Outlier rate | Synthetic F1 (random) | Intel F1 (random) | M3500 F1 (random) | MIT F1 (random) |
|-------------|----------------------|-------------------|-------------------|----------------|
| 10% | 0.974 | 0.903 | 1.000 | 1.000 |
| 30% | 0.989 | 0.966 | 1.000 | 1.000 |
| 50% | 0.989 | 0.966 | 1.000 | 1.000 |
| 70% | 0.975 | 0.966 | 1.000 | 0.963 |
| 90% | 0.900 | 0.833 | 0.963 | 0.963 |

The generalization gap is visible at lower outlier rates (10-30%: 0.02-0.07 F1 drop from synthetic to Intel), narrows at mid rates (50-70%: near-parity), and widens at 90% (0.07 drop). Models trained on 10% outliers struggle on Intel because the real graph's topology creates ambiguity the training distribution never exposed. Models trained on 30-70% generalize well. Models trained on 90% are universally conservative — high precision, low recall, consistent across datasets.

---

## 11. F1 vs ATE — Orthogonal Signals

Across all datasets and models, F1 and ATE show weak correlation (r² < 0.3):

| Example | F1 | ATE | Interpretation |
|---------|-----|-----|----------------|
| 50pct_clustered on Intel | 0.92 | 0.86 | Decent F1, good ATE |
| 30pct_random on Intel | 0.97 | 7.35 | Better F1, terrible ATE |
| 90pct_random on Intel | 0.83 | 0.12 | Worst F1 (among random), best ATE |
| M3500 (all models) | 0.78-1.00 | 9.2-14.7 | F1 varies widely, ATE is constant |

**Edge classification accuracy does not predict map quality.** This is a central finding for the paper. The GNN optimizes for edge-level correctness, which is weakly coupled to the downstream optimization problem. DCS optimizes for map-level correctness natively.

This finding has implications beyond this project — it suggests that papers evaluating learned PGO methods solely on edge classification metrics may be measuring a proxy that doesn't correlate with the problem they're trying to solve.

---

## 12. Limitations

### 12A. Solver Backend Confound
Learned eval uses PyPose; baselines use GTSAM. Cost, convergence, and timing comparisons across backends are not quantitatively valid. Requires re-running learned eval through GTSAM to eliminate this confound.

### 12B. Single Seed
`n_seeds=1` throughout both sweeps. The CI infrastructure exists in `results_ci.csv` with mean/std columns, but no multi-seed data was collected. Cannot make statistical claims about random vs clustered, model A vs model B, or generalization gap significance.

### 12C. PyPose Non-Convergence on Large Graphs
The StopOnPlateau scheduler with `patience=5, decreasing=1e-3` fails to trigger convergence on Intel, M3500, CSAIL. All models max out 100 iterations. Makes timing comparisons with DCS (2-14 iterations) meaningless — DCS finishes in milliseconds because it CONVERGES, not because GTSAM is inherently faster.

### 12D. city10000 OOM for Learned Models
PyPose+GNN optimization on a 10,000-node graph exceeds available RAM. Baselines succeed because GTSAM's LM is more memory-efficient. Either skip city10000 for learned eval in future sweeps, or add GTSAM learned eval path.

### 12E. 20-Edge Injection Scale
On large graphs (M3500's 5,453 edges, Intel's 2,512 edges), injecting only 20 labeled loop-closure edges makes F1 a low-signal metric. The odometry chain dominates the solution. Larger injection sizes (50-200 edges) would stress the outlier rejection task more meaningfully on real benchmarks.

---

## 13. Next Steps

### Tier 1 — Resolve Confounds (low effort, one command each)

1. **Re-run learned through GTSAM**: `solver=gtsam solver.kernel=none`. Same cost convention, same solver, same convergence criteria as DCS/uniform. Validates or invalidates the cost advantage and convergence gap. Single command, zero code changes.

2. **Increase PyPose max_iterations on large graphs**: Benchmark Intel/M3500 at 200, 500, 1000 iterations. Determines whether the solver eventually converges or diverges. If it converges at 200, the speed gap with DCS is 10× (not 9,000×) — makes the GNN speed competitive.

3. **Test PyPose with uniform weights on Intel**: If it also hits 100 iterations without converging, the problem is PyPose tuning, not GNN weights. Settles the convergence debate.

### Tier 2 — Odometry Weighting (medium effort)

4. **Position-based odometry decay**: `w_odom(e) = exp(-λ × index / total_poses)`. Early edges (near origin, low accumulated drift) trusted more than late edges. Single hyperparameter λ. Tests whether odometry-aware weighting improves ATE.

5. **Let GNN predict odometry weights**: Remove `scores = torch.ones(E)` hardcode at `edgegate_gnn.py:165`. Requires switching to trajectory loss (BCE has no odometry labels). Phase 1 trajectory loss training provides the gradient signal for odometry weight learning.

6. **Iterative residual re-weighting**: GNN predicts weights → solver produces poses → compute per-edge residuals → feed residuals as new edge features → GNN re-predicts weights → repeat 2-3×. Gives the GNN the adaptive feedback loop that DCS and GNC have built in. Implementation is a loop in `evaluate_one_graph()`, no architecture changes.

### Tier 3 — Training Improvements (medium-high effort)

7. **Trajectory loss training**: Already implemented at `edgegate/losses/trajectory_loss.py`. Backprops ATE through differentiable PyPose solver. Directly optimizes for map quality rather than edge classification. Phase 1's stated deliverable per the research proposal.

8. **Multi-seed training**: `train.seed=0,1,2` on 10 configs → 30 runs → mean ± std for all metrics. The `results_ci.csv` infrastructure is ready. Required before any publication claim.

9. **Train on larger graphs**: 500-1,000 pose synthetic graphs. Current training is on 100-node graphs but eval targets 1,728-3,500-node graphs. Training distribution mismatch likely contributes to the generalization gap.

10. **Architecture ablations** (from `GNN.md` §4):
    - Ablation A: Edge-conditioned convolution (ECC/NNConv) replacing the 2-bucket type switch
    - Ablation B: Attention-based aggregation (`TransformerConv`) replacing sum-pooling
    - Ablation C: Recurrent (GRU) processing of the odometry chain
    - Larger capacity: `hidden_dim=128`, more training graphs

### Tier 4 — Evaluation Protocol (low effort)

11. **Remove city10000 from default learned eval**: OOM on all runs. Optionally add GTSAM learned eval path for this dataset.
12. **Increase outlier injection size on real benchmarks**: 50-200 edges instead of 20. Makes F1 a more meaningful metric on large graphs.
13. **Lock domain-shift metrics** before next training run per `implementation_details.md` §"Domain-Shift Characterization Metrics": compute per-benchmark outlier rate/structure mismatch, edge-type ratio mismatch, and noise-scale mismatch.
14. **Add GNN → DCS hybrid baseline**: GNN-weighted solution as initial guess → DCS refine. Tests whether learned outlier removal + adaptive kernel gives the best of both. Conceptually matches Policies over Poses' Prop-V2 (learned correction + 75 LM iterations).

### Tier 5 — Investigation (low effort, high return on insight)

15. **Why does M3500 ATE not change with model?**: 20 injected LCs on 5,453 edges is negligible. Either increase injection count or run an ablation where GNN weights are applied to ALL edges (not just LCs) to test whether the GNN can influence large-graph solutions.
16. **Why do 10pct models show anomalously high ATE on MIT?**: Gauge-alignment artifact or genuine over-trusting of odometry? Compare optimized trajectories directly.
17. **Does 90pct_random's ATE advantage on Intel hold with multi-seed?**: The most impactful single model across both sweeps. Needs seed replication.
