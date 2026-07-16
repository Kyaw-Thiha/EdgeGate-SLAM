# EdgeGate SLAM — Architecture

## Overview

EdgeGate SLAM learns per-edge inlier confidence for pose-graph SLAM using a
graph neural network (GNN), conditioning a classical weighted least-squares
solver instead of relying on hand-crafted robust kernels (switchable
constraints, DCS, GNC). The pose graph is edge-typed (odometry vs.
loop-closure), and the GNN's confidence score scales each edge's information
matrix before optimization.

Two research angles are supported by this architecture, not just one:

1. **Supervised edge classification (MVP)** — train the GNN against synthetic
   ground-truth inlier labels (BCE), then hand weights to a solver at eval time.
2. **End-to-end differentiable training (extension)** — backprop the final
   trajectory error directly through a differentiable solver into the GNN,
   with no edge labels required.

The structure below is built so switching between these — and later adding an
RL-based extension — requires new modules, not rewrites of existing ones.

## Execution Phases (decided July 2026, after literature review)

The near-term research contribution is **Angle 1 (supervised edge
classification)**, framed around a **synthetic-to-real generalization-gap
study** rather than a bare classification-accuracy number. Classical robust
kernels (GNC, DCS) already tolerate 70-80% outliers well on the standard
benchmarks, so an isolated accuracy claim on synthetic data is not itself a
strong contribution — a rigorously characterized generalization gap is. This
elevates the Train/Val/Test Split Protocol's held-out real-benchmark
evaluation (`implementation_details.md`) from an ethical footnote to the
primary evaluation axis.

Three phases:

- **Phase 0 (build now)** — data pipeline, GNN, `edge_bce` loss, `Solver`
  interface + GTSAM adapter, and `evaluate.py`'s unified baseline-vs-learned
  harness. Nothing here is new relative to the structure already documented
  below; this is direct execution of the existing plan.
- **Phase 1 (OOD / generalization-gap study)** — uses the existing
  Train/Val/Test protocol unchanged, but requires one new deliverable made
  explicit in `implementation_details.md`: domain-shift characterization
  metrics, locked *before* training begins, so the generalization-gap claim
  isn't fit to results after the fact.
- **Phase 2 (VSA/SSP extension, stretch, not yet architected)** — see §8
  below. Not a blocker for Phase 0/1; only pursued once Phase 1 results
  exist.

MARL-based extensions (à la RL-PGO / Policies over Poses) and a full pivot to
a different SLAM sub-area or robotics area (e.g. object-goal navigation,
agile flight) were both considered and explicitly deprioritized for this
phase: MARL requires actor-critic/ADMM-consensus infrastructure
disproportionate to available time and is already being actively worked by
better-resourced groups; a full pivot away from pose-graph SLAM would trade a
low-compute, small-dataset problem for one requiring large-scale simulation
or real hardware. See `EdgeGate_SLAM_Research_Proposal.md` for the full
literature landscape and reasoning behind these calls.

## Tech Stack

| Concern | Choice | Role |
|---|---|---|
| GNN framework | PyTorch Geometric (PyG) | edge-type-aware message passing |
| Primary solver | PyPose | differentiable, PyTorch-native, enables end-to-end training and future RL |
| Baseline solver | GTSAM | classical kernels (GNC, DCS natively; switchable constraints reimplemented) — used only for baseline comparisons, not primary training |
| Config management | Hydra | composable configs, multirun sweeps |
| Experiment tracking | Weights & Biases | logging, run comparison, sweep aggregation |
| Visualization | Rerun | post-hoc visualization from persisted logs (not live-logged during training) |
| Environment management | Pixi | CUDA + C++-extension + Python dependency resolution and locking |

PyPose is primary because it's PyTorch-native (no numpy/foreign-object
boundary), actively maintained, matches GTSAM/Ceres accuracy on PGO
benchmarks, and — critically — makes the solver differentiable, which both
enables the trajectory-loss research angle and gives a natural path to RL
later (a differentiable solver doubles as a differentiable environment/reward
signal). GTSAM is kept specifically because GNC and DCS are natively
implemented and maintained there, and two of your three named baselines come
for free as a result.

## Project Structure

```
edgegate-slam/
├── README.md
├── pyproject.toml
├── pixi.toml                 # Pixi manifest: python, cuda-version, deps
├── pixi.lock                 # auto-generated, hash-level, multi-platform lockfile
├── .gitignore
│
├── configs/
│   ├── config.yaml            # root: defaults list composes the groups below
│   ├── data/
│   │   ├── synthetic.yaml
│   │   └── benchmark.yaml     # Intel / M3500 / Sphere2500 / parking-garage
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
│   ├── raw/                   # downloaded .g2o benchmarks — held out, eval-only
│   ├── synthetic/              # generated graphs + labels (gitignored)
│   └── README.md
│
├── scripts/
│   ├── download_benchmarks.py
│   ├── generate_synthetic.py
│   ├── train.py                # @hydra.main(config_path="../configs", config_name="config")
│   ├── evaluate.py
│   └── demo_rerun.py           # loads persisted logs, renders in Rerun
│
├── edgegate/
│   ├── data/
│   │   ├── g2o_io.py           # parse/write .g2o format
│   │   ├── synthetic_generator.py
│   │   └── graph_builder.py    # PoseGraph -> PyG Data
│   │
│   ├── models/
│   │   ├── layers.py           # edge-type-aware message passing layer(s)
│   │   └── edgegate_gnn.py     # stacked layers + per-edge MLP confidence head
│   │
│   ├── solvers/
│   │   ├── base.py             # Solver interface — the load-bearing abstraction
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
│   │   └── rerun_logger.py     # reads persisted logs, writes to Rerun
│   │
│   └── training/
│       └── trainer.py          # plain PyTorch loop, built from Hydra cfg, logs to wandb + persisted JSON
│
├── tests/
│   ├── test_g2o_io.py
│   ├── test_graph_builder.py
│   └── test_solvers.py         # pypose_solver vs gtsam_solver must agree on outlier-free graphs
│
└── runs/                        # gitignored: Hydra output dirs, checkpoints, persisted logs
```

No `notebooks/` directory — exploratory work happens outside the tracked
project, nothing gets imported from a notebook.

## Core Architectural Decisions

### 1. The `Solver` interface is the load-bearing abstraction

Everything — training, evaluation, and any future RL environment — talks to a
solver only through:

```python
solve(graph: PoseGraph, edge_weights: Tensor, max_iterations: int | None = None)
    -> (poses, converged: bool, num_iterations: int, final_cost: float)
```

`pypose_solver.py` and `gtsam_solver.py` both implement this signature. This
is what allows swapping solvers via a one-line Hydra override
(`solver=gtsam`), and what will let an RL environment later wrap
`pypose_solver` without touching anything upstream.

- `max_iterations` set low (e.g. single digits) during `trajectory_loss`
  training, for bounded compute/memory per step; left uncapped (full
  convergence) at evaluation time, since the GNN's weights are frozen and
  correctness — not step time — is what matters there.
- `converged` / `final_cost` let `evaluate.py` distinguish "didn't converge"
  from "converged but produced a bad trajectory" — these should never be
  conflated in reported results.

### 2. `PoseGraph` is the framework-agnostic interchange format

A single plain dataclass, produced by `g2o_io.py` / `synthetic_generator.py`,
that nothing downstream mutates directly:

```python
@dataclass
class PoseGraph:
    node_init: np.ndarray        # (N, 3) initial pose guess (x, y, θ)
    edge_index: np.ndarray       # (2, E)
    edge_measurement: np.ndarray # (E, 3) relative (dx, dy, dθ)
    edge_info: np.ndarray        # (E, 6) upper-tri of 3x3 information matrix
    edge_type: np.ndarray        # (E,) 0=odometry, 1=loop-closure
    edge_label: np.ndarray | None # (E,) ground-truth inlier, synthetic only
```

`graph_builder.py` converts this to a PyG `Data` object for the GNN.
`pypose_solver.py` and `gtsam_solver.py` each independently convert
`PoseGraph` + predicted edge weights into their own native representation.
Neither solver adapter needs to know the other exists.

### 3. Loss modules are swappable, not branches in one file

`losses/edge_bce.py` (MVP, no solver call during training) and
`losses/trajectory_loss.py` (solver runs inside the training step, requires
backprop through it) are separate modules selected via
`configs/loss/*.yaml`. Switching between the MVP research question and the
more novel end-to-end angle is a config change, not a code change.

### 4. Model is split into `layers.py` and `edgegate_gnn.py`

`layers.py` holds the actual edge-type-aware message-passing building
block(s) — small, independently testable. `edgegate_gnn.py` stacks them and
applies the confidence head. This split lets the message-passing design be
iterated on without touching checkpointing, training-loop code, or the head.

### 5. `evaluate.py` treats learned and classical methods uniformly

Classical baselines (GNC, DCS, switchable constraints via GTSAM) don't need a
GNN or training — they run directly on a `PoseGraph`. `evaluate.py` has one
evaluation path that both "GNN + solver" and "classical kernel alone" go
through, so comparisons use identical graphs, metrics, and seeds rather than
being produced by two scripts that could drift apart.

### 6. Visualization is decoupled from training

`trainer.py` never calls Rerun directly. It persists structured logs
(poses, edge weights, convergence status, metrics per epoch/checkpoint) as
JSON. `rerun_logger.py` and `demo_rerun.py` read those logs after the fact.
This keeps training fast and dependency-free, and makes any past run
re-visualizable on demand, not just the one currently training.

### 7. No `rl/` package yet

RL is intentionally not scaffolded. When it's added, it will wrap
`pypose_solver` through the same `Solver` interface, as its own package
alongside `training/`, without requiring changes to the modules above.

### 8. VSA/SSP extension (Phase 2 — not yet implemented)

A candidate stretch extension encodes pose-graph edges with Spatial Semantic
Pointers (SSPs) / vector symbolic architecture (VSA) representations — e.g.
scoring loop-closure consistency via algebraic binding/unbinding (a
consistent cycle of edges should compose back to ~identity) as a
training-cheap, structurally interpretable alternative or complement to the
GNN's learned confidence score.

Integration point, when this is built: a new module (tentatively
`edgegate/features/ssp_encoding.py`), wired in as a **standalone baseline arm
through `evaluate.py`** first — the same slot occupied by
GNC/DCS/switchable-constraints — before attempting any fusion into
`graph_builder.py` or `edgegate_gnn.py`. This keeps the working Phase 0/1
pipeline unaffected regardless of whether the SSP baseline pans out.

Open decision, not yet resolved: whether the SSP implementation should be
plain PyTorch/NumPy (fast to iterate, no new dependency) or built on
Nengo/nengo-spa (adds a spiking-simulation dependency, but opens a path to
neuromorphic/Loihi deployment later). Resolve this with the lab before
writing the module — it changes both the dependency footprint and the Phase
2 timeline.
