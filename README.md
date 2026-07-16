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

**Install spike (do this before writing code):** confirm GTSAM + PyPose + PyG resolve together cleanly:

```bash
pixi run python -c "import gtsam, pypose, torch_geometric; print('ok')"
```

If the PyG CUDA extension index URL in `pixi.toml` needs updating (the exact PyTorch version that resolves may differ), adjust the `extra-index-urls` line under `[feature.gpu.pypi-options]` to match.

## Common commands

```bash
# Run tests (cpu env is sufficient)
pixi run -e cpu test

# Generate synthetic training graphs
pixi run gen-synthetic

# Download real benchmarks (Intel, M3500, Sphere2500, parking-garage)
pixi run download-benchmarks

# Train (uses default gpu env)
pixi run train

# Evaluate — GNN vs. classical baselines (GNC, DCS)
pixi run evaluate

# Evaluate with specific method
pixi run evaluate eval_method=learned  eval_mode.checkpoint_path=model_best.pt
pixi run evaluate eval_method=uniform
pixi run evaluate eval_method=gnc
pixi run evaluate eval_method=dcs

# Evaluate on synthetic test graphs (separate seed from training)
pixi run evaluate eval_mode.num_test_graphs=20 eval_mode.test_seed=999

# Evaluate on real benchmark (requires .g2o file in data/raw/)
pixi run evaluate eval_mode.dataset=intel eval_mode.inject_outliers=true

# Aggregate sweep results into comparison table
pixi run evaluate eval_mode=aggregate eval_mode.sweep_glob="runs/sweep_*"

# Visualize a past run with Rerun
pixi run demo-rerun

# Hydra multirun — outlier-rate sweep
pixi run train --multirun data.outlier_rate=10,30,50,70,90 data.outlier_structure=random,clustered

# Hydra multirun — solver-iterations sweep (trajectory-loss ablation)
pixi run train --multirun train.solver_train_iterations=1,3,5,10,20
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
│   └── sweep/
│       ├── outlier_grid.yaml
│       └── solver_iterations_grid.yaml
│
├── data/
│   ├── raw/                           # downloaded .g2o benchmarks — held out, eval-only
│   ├── synthetic/                     # generated graphs + labels (gitignored)
│   └── README.md
│
├── docs/
│   ├── EdgeGate_SLAM_Research_Proposal.md
│   ├── architecture.md
│   └── implementation_details.md
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
│   │   ├── g2o_io.py                  # parse/write .g2o format
│   │   ├── synthetic_generator.py
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
│       └── trainer.py                 # plain PyTorch loop, Hydra cfg, wandb + JSON logging
│
├── tests/
│   ├── test_g2o_io.py
│   ├── test_graph_builder.py
│   └── test_solvers.py                # pypose_solver vs gtsam_solver must agree on outlier-free graphs
│
└── runs/                              # gitignored: Hydra output dirs, checkpoints, persisted logs
```
