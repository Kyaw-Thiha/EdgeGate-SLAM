# EdgeGate SLAM

GNN-based per-edge inlier confidence for pose-graph SLAM. See `docs/EdgeGate_SLAM_Research_Proposal.md` for full context, and `docs/architecture.md` for structural decisions.

## Setup

Install [Pixi](https://pixi.sh), then:

```bash
# GPU environment (default — CUDA 12.4, full training stack)
pixi install

# CPU environment (unit tests, data work, no CUDA required)
pixi install -e cpu
```

**Verify install** before writing code:

```bash
pixi run python -c "import gtsam, pypose, torch_geometric; print('ok')"
```

If the PyG CUDA extension index URL in `pixi.toml` needs updating (the exact PyTorch version that resolves may differ), adjust the `extra-index-urls` line under `[feature.gpu.pypi-options]` to match.

## Quickstart — Phase 0 (BCE training)

```bash
# 1. Download real benchmark datasets (held out, eval-only)
pixi run download-benchmarks

# 2. Run tests (sanity check)
pixi run -e cpu test

# 3. Preview synthetic data generation (optional)
pixi run gen-synthetic

# 4. Train the GNN with BCE loss on synthetic data
pixi run train

# Or run the entire training→evaluation pipeline in one command:
pixi run run-sweep                         # full sweep: 10 models × 7 datasets
pixi run run-sweep -- --epochs 50          # shorter training
pixi run run-sweep -- --rates 30,50,70     # subset of outlier rates

# 5. Evaluate learned model on synthetic test graphs
pixi run evaluate eval_mode.dataset=synthetic

# 6. Evaluate learned model on real benchmarks (with injected outliers)
pixi run evaluate eval_mode.dataset=intel
pixi run evaluate eval_mode.dataset=m3500

# 7. Evaluate classical baselines (uniform, GNC, DCS) for comparison
pixi run evaluate eval_mode.dataset=synthetic eval_method.method=uniform solver=gtsam solver.kernel=none
pixi run evaluate eval_mode.dataset=synthetic eval_method.method=gnc
pixi run evaluate eval_mode.dataset=synthetic eval_method.method=dcs
```

## Full Evaluation Plan

Per `docs/EdgeGate_SLAM_Research_Proposal.md` §8, the primary research contribution is a
**synthetic-to-real generalization-gap study**, not raw classification accuracy.

| Step | What | Command |
|------|------|---------|
| Train | BCE on synthetic (default: 50 poses, 30% outliers) | `pixi run train` |
| Synth eval | GNN vs. uniform/GNC/DCS on synthetic test graphs | `pixi run evaluate eval_mode.dataset=synthetic eval_method.method=<name>` |
| Outlier sweep | Train across outlier rates 10-90% | `pixi run train --multirun +sweep=outlier_grid hydra/launcher=joblib hydra.launcher.n_jobs=4 data.outlier_rate=10,30,50,70,90 data.outlier_structure=random,clustered` |
| Real eval (once) | Report on held-out benchmarks **exactly once per model version** | `pixi run evaluate eval_mode.dataset=<intel,m3500,mit,csail,manhattan,city10000>` |
| Sweep aggregate | Collect sweep results into comparison CSV | `pixi run evaluate eval_mode.mode=aggregate` |

**Real benchmarks are eval-only.** Never used during training or hyperparameter tuning.
Results are reported once per model version — iterative reporting silently turns them into
a validation set and invalidates the generalization-gap claim.

### Available benchmarks (all in `data/raw/`)

| Dataset | Nodes | Edges | GT poses? | Source |
|---------|-------|-------|-----------|--------|
| intel | 1,728 | 2,512 | No (real robot log) | SE-Sync |
| M3500 | 3,500 | 5,453 | Yes (simulated) | Carlone |
| M3500a | 3,500 | 5,453 | Yes (simulated, +0.1rad noise) | Carlone |
| MIT | 808 | 827 | No (real robot log) | SE-Sync |
| CSAIL | 1,045 | 1,172 | No (real robot log) | SE-Sync |
| manhattan | 3,500 | 5,453 | Yes (simulated) | SE-Sync |
| city10000 | 10,000 | 20,687 | Yes (simulated) | SE-Sync |
| sphere2500 | — | — | SE(3) — deferred to Phase 2 | SE-Sync |
| parking-garage | — | — | SE(3) — deferred to Phase 2 | SE-Sync |

Phase 1 adds domain-shift characterization metrics (outlier-rate/structure mismatch,
edge-type ratio mismatch, noise-scale mismatch) — computed once per benchmark, reported
alongside degradation numbers. See `docs/implementation_details.md` §"Domain-Shift
Characterization Metrics".

## Common Commands

```bash
# Run tests
pixi run -e cpu test

# Generate synthetic training graphs (Hydra-configurable)
pixi run gen-synthetic
pixi run gen-synthetic data.num_poses=100 data.outlier_rate=50

# Download real benchmarks (skips existing files; --extras for SE(3) + M3500a)
pixi run download-benchmarks

# Train
pixi run train
pixi run train train.epochs=200 train.lr=5e-4          # custom LR/epochs
pixi run train train.loss_mode=trajectory                     # trajectory loss (requires GPU)
pixi run train data.outlier_rate=50 data.outlier_structure=clustered

# Evaluate — learned model (default)
pixi run evaluate eval_mode.dataset=synthetic
pixi run evaluate eval_mode.dataset=intel

# Evaluate — classical baselines (no GNN needed)
pixi run evaluate eval_mode.dataset=synthetic eval_method.method=uniform solver=gtsam solver.kernel=none
pixi run evaluate eval_mode.dataset=synthetic eval_method.method=gnc
pixi run evaluate eval_mode.dataset=synthetic eval_method.method=dcs

# Hydra multirun — outlier-rate sweep
pixi run train --multirun +sweep=outlier_grid hydra/launcher=joblib hydra.launcher.n_jobs=8

# Hydra multirun — solver-iterations sweep (trajectory-loss ablation)
pixi run train --multirun +sweep=solver_iterations_grid hydra/launcher=joblib hydra.launcher.n_jobs=4

# Visualize — quick live demo (no trained model needed)
pixi run demo-rerun -- --live

# Visualize — replay a training run (scrub epoch timeline)
pixi run demo-rerun -- --replay outputs/<date>/<time>/

# Visualize — compare methods on the same graph (paper figure)
pixi run demo-rerun -- --compare learned=outputs/eval/learned gnc=outputs/eval/gnc --graph-idx 0

# Save any visualization to .rrd for offline sharing
pixi run demo-rerun -- --live --save-rrd results/demo.rrd
rerun results/demo.rrd
```

## Visualization (Rerun)

See `docs/visualization.md` for full usage — live demo, epoch replay, laser scan overlay, and multi-method comparison.

```bash
pixi run demo-rerun -- --live                          # quick start, no model needed
pixi run demo-rerun -- --replay outputs/<date>/<time>/ # replay a training run
```

## Project layout

```
edgegate-slam/
├── README.md
├── pyproject.toml
├── pixi.toml                          # Pixi manifest: python, cuda-version, deps
├── pixi.lock                          # auto-generated, hash-level, multi-platform lockfile
├── .gitignore
│
├── configs/
│   ├── config.yaml                    # root: defaults list composes the groups below
│   ├── evaluate.yaml                  # root: eval mode, method, dataset, solver, model, logging
│   ├── data/
│   │   ├── synthetic.yaml
│   │   └── benchmark.yaml             # Intel / M3500 / Sphere2500 / parking-garage
│   ├── model/
│   │   └── edgegate_gnn.yaml
│   ├── solver/
│   │   ├── pypose.yaml
│   │   └── gtsam.yaml
│   ├── loss/
│   │   ├── edge_bce.yaml
│   │   └── trajectory.yaml
│   ├── train/
│   │   └── default.yaml
│   ├── logging/
│   │   └── wandb.yaml
│   ├── sweep/
│   │   ├── outlier_grid.yaml
│   │   └── solver_iterations_grid.yaml
│   ├── eval_mode/
│   │   ├── evaluate.yaml              # mode=evaluate
│   │   └── aggregate.yaml             # mode=aggregate (sweep comparison)
│   ├── eval_method/
│   │   ├── learned.yaml               # GNN checkpoint + solver
│   │   ├── uniform.yaml               # unit weights baseline
│   │   ├── gnc.yaml                   # GNC kernel (GTSAM)
│   │   ├── dcs.yaml                   # DCS kernel (GTSAM)
│   │   └── switchable.yaml            # switchable constraints (NOT YET IMPLEMENTED)
│   └── eval_dataset/
│       ├── synthetic.yaml
│       ├── intel.yaml
│       ├── m3500.yaml
│       ├── mit.yaml
│       ├── csail.yaml
│       ├── manhattan.yaml
│       ├── city10000.yaml
│       ├── parking-garage.yaml
│       └── sphere2500.yaml            # SE(3) — deferred to Phase 2
│
├── data/
│   ├── raw/                           # downloaded .g2o benchmarks — held out, eval-only
│   ├── synthetic/                     # generated graphs + labels (gitignored)
│   └── README.md
│
├── docs/
│   ├── EdgeGate_SLAM_Research_Proposal.md
│   ├── architecture.md
│   ├── implementation_details.md
│   ├── GNN.md
│   └── visualization.md
│
├── scripts/
│   ├── download_benchmarks.py
│   ├── generate_synthetic.py
│   ├── train.py                       # @hydra.main entry point
│   ├── evaluate.py
│   └── demo_rerun.py                  # loads persisted logs, renders in Rerun
│
├── edgegate/
│   ├── data/
│   │   ├── types.py                   # PoseGraph dataclass
│   │   ├── g2o_io.py                  # parse/write .g2o format
│   │   ├── se2_utils.py               # SE(2) compose / inverse_compose / angle_wrap
│   │   ├── synthetic_generator.py
│   │   ├── outlier_injection.py       # standalone LC injection with labels
│   │   └── graph_builder.py           # PoseGraph -> PyG Data
│   │
│   ├── models/
│   │   ├── layers.py                  # edge-type-aware message passing layer(s)
│   │   └── edgegate_gnn.py            # stacked layers + per-edge MLP confidence head
│   │
│   ├── solvers/
│   │   ├── base.py                    # Solver interface — the load-bearing abstraction
│   │   ├── pypose_solver.py
│   │   └── gtsam_solver.py
│   │
│   ├── losses/
│   │   ├── edge_bce.py
│   │   └── trajectory_loss.py
│   │
│   ├── metrics/
│   │   ├── edge_f1.py
│   │   └── ate_rmse.py
│   │
│   ├── viz/
│   │   └── rerun_logger.py            # reads persisted logs, writes to Rerun
│   │
│   └── training/
│       ├── trainer.py                 # plain PyTorch loop, Hydra cfg, wandb + JSON logging
│       └── evaluate.py                # shared eval utilities (imported by trainer + evaluate script)
│
├── tests/
│   ├── test_g2o_io.py
│   ├── test_graph_builder.py
│   ├── test_outlier_injection.py
│   ├── test_losses.py
│   ├── test_metrics.py
│   ├── test_models.py
│   ├── test_evaluate_utils.py
│   └── test_solvers.py                # solver agreement + GNC/DCS kernel tests
│
└── runs/                              # gitignored: Hydra output dirs, checkpoints, persisted logs
```
