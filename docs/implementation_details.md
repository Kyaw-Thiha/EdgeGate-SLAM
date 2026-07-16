# EdgeGate SLAM — Implementation Details

This document covers implementation-level decisions that support
`architecture.md` — conventions, protocols, and open questions that are more
likely to be revised as coding progresses. Read `architecture.md` first for
the structural/interface-level picture.

## SLAM Background (quick reference)

A pose graph has nodes = robot poses over time, and edges = relative
transformation measurements between poses, each with an information matrix
expressing measurement confidence.

- **Odometry edges** connect consecutive poses — individually reliable but
  accumulate drift over long runs.
- **Loop-closure edges** connect non-consecutive poses when the robot
  believes it has revisited a location. These correct drift when correct, but
  are catastrophic when wrong (perceptual aliasing) — this is the core
  problem the project addresses.
- **Backend optimization** solves for poses that best satisfy all
  constraints jointly (nonlinear weighted least squares) — this is the
  "solver" (PyPose/GTSAM).
- **Edge-level F1** measures inlier/outlier classification correctness
  directly, without running the solver.
- **ATE (Absolute Trajectory Error)** measures the optimized trajectory's
  distance from ground truth, after solving — the metric that reflects
  end-to-end performance.

## Data Schema

See `architecture.md` §2 for the `PoseGraph` dataclass itself. Open question,
not yet decided: what goes into GNN **node features**. `node_init` (the
odometry-chained initial pose guess) is the natural baseline candidate, since
it's already required for solver initialization regardless of what else is
used.

A live design fork worth deciding deliberately when writing `graph_builder.py`:
whether to also feed edges a **residual-under-initial-guess** feature (how
inconsistent a loop-closure measurement is with what the rest of the graph
implies). This is a strong signal — it's the exact quantity classical kernels
like DCS/GNC threshold on — but including it shifts the model from "pure
topological consistency pattern" (the proposal's original framing) toward a
hybrid geometric+topological model. Not a blocker, but shouldn't be decided
by accident.

## Edge-Weight → Information-Matrix Convention

**Decision: scale the full information matrix by `w²`**, equivalently scale
the residual by `w` (where `w` is the GNN's `[0,1]` confidence output).

This follows the dominant convention in the robust-back-end literature:
switchable constraints defines the weighted residual as `Ψ(φ)·e(x_i, x_j)`
with `Ψ` mapping into `[0,1]`, and since cost is `e^T Λ e`, scaling the
residual by `w` is mathematically equivalent to scaling `Λ` by `w²`.
Max-mixture models do this even more explicitly, scaling the Jacobian,
residual, and information matrix together. This is distinct from DCS's
convention, which scales the information matrix linearly (not squared)
through its own scaling function.

Implement as a single shared function
(`scale_information(Λ, w) -> Λ * w**2`) used identically by both
`pypose_solver.py` and `gtsam_solver.py`, so the two solver adapters can never
silently diverge on what a given confidence score means.

## Solver Convergence / Failure Handling

Pattern borrowed from the differentiable-factor-graph-optimization
literature (which uses fixed-iteration Gauss-Newton for the training-time
surrogate loss, and full LM only at evaluation):

- **During `trajectory_loss` training**: the solver runs a **small, fixed
  number of iterations (K)**, not "until convergence." This bounds
  compute/memory per training step regardless of how poor the GNN's current
  edge weights are, and avoids unbounded solver spin early in training when
  weights are near-random.
- **At evaluation time**: the solver runs to full convergence (no iteration
  cap), since the GNN is frozen and correctness matters more than step time.
- `edge_bce` training never calls the solver at all during training (pure
  classification against synthetic labels) — the solver only runs at eval
  time for that loss too, always to full convergence.

**K is a first-class ablation axis, not a fixed constant.** Exposed as
`train.solver_train_iterations` in `configs/train/default.yaml`, swept via
Hydra multirun (`configs/sweep/solver_iterations_grid.yaml`), e.g.
`K ∈ {1, 3, 5, 10, 20}`. Track, per K value:
- Final ATE / edge-F1 (does the surrogate solve produce a GNN that performs
  as well as one trained with larger K)
- Gradient norm at the GNN's input, as a function of K — the concrete,
  checkable signal for whether a small K meaningfully distorts what the GNN
  learns, rather than just eyeballing final metrics.

Run this K-sweep independently from the outlier-rate sweep initially (see
below) rather than crossing the two grids — K is a training-dynamics
question, outlier rate is a robustness question; crossing them multiplies
compute for a combined question not yet needed. Fix a good K first, then
sweep outlier rate on top of it.

## Batching Strategy

- **`edge_bce` training**: standard PyG batching (disjoint-union graphs)
  works without modification, since the solver is never called during
  training for this loss.
- **`trajectory_loss` training**: both PyPose and GTSAM's differentiable
  competitor libraries (e.g. Theseus) support real batched differentiable
  PGO, but batching requires graphs of **matching size/topology** — a batch
  is a stack of same-sized problems vectorized with a batch dimension, not
  arbitrary variable-size graphs. Since the synthetic generator is under our
  control, generate graphs with a **fixed pose count / fixed loop-closure
  count per batch** (bucket by size, or fix globally per run) to get genuine
  GPU-batched training. Batch size is therefore a real, tunable
  hyperparameter here, not forced to 1.
- **Real benchmark `.g2o` datasets**: held out entirely as eval-only (see
  train/val/test protocol below), evaluated one graph at a time — no
  batching concern since they're never part of a training batch.
- **Enforced in code**: `Solver.solve()` always takes a single, unbatched
  `PoseGraph`. Any looping over a batch happens in `trainer.py` /
  `evaluate.py`, never inside a solver class — keeps both solver
  implementations symmetric and batching complexity in exactly one place.

Open item, not yet resolved: whether PyPose's LM optimizer backpropagates via
full unrolling through iterations or implicit differentiation (Theseus's
paper frames implicit differentiation as its specific advance over
unrolling-only prior work) — needs checking directly against PyPose's solver
internals when `pypose_solver.py` is actually written, since it affects
memory feasibility for larger K values.

## Train / Val / Test Split Protocol

- **Synthetic data** (generator-produced graphs) is the only data used for
  training and validation / hyperparameter selection.
- **Real benchmark `.g2o` datasets** (Intel, M3500, Sphere2500,
  parking-garage) are held out entirely — never touched during training or
  hyperparameter tuning. Report results on them exactly once per model
  version, not iteratively, or they quietly become a validation set and the
  synthetic-to-real generalization gap (the project's own stated ethical
  consideration) is no longer being honestly measured.

### Domain-Shift Characterization Metrics (Phase 1 addition)

As of the OOD-focused framing (see `architecture.md`), the generalization
gap is the primary contribution, not a footnote — which means it must be
characterized, not just reported as a single before/after number. Before
training starts, lock a small set of metrics that quantify how far each real
benchmark's statistics sit from the synthetic training distribution:

- Outlier rate and structure (random vs. clustered) mismatch
- Edge-type ratio (odometry : loop-closure) mismatch
- Noise-scale mismatch (information-matrix magnitude distributions)

Compute these once per real benchmark, store alongside the eval-once
results, and report degradation (edge-F1 / ATE) as a function of these
divergence metrics — not just as a flat synthetic-vs-real comparison. Decide
the exact metric definitions before the first training run; changing them
after seeing results risks fitting the explanation to the outcome rather
than the other way around.

## Baseline Integration in `evaluate.py`

`evaluate.py` must support running a solver directly on a graph with **no
GNN at all** — either uniform edge weights, or a classical robust kernel
(GNC / DCS / switchable-constraints-via-GTSAM) doing its own internal
weighting — as one branch of the same evaluation function used for "GNN +
solver." This guarantees identical graphs, metrics, and seeds across learned
and classical methods, rather than risking drift between separate ad hoc
scripts.

Switchable constraints specifically: its reference implementation (Vertigo)
targets `gtsam 2.0`, an old pre-rewrite version incompatible with current
GTSAM (4.2/4.3) — plan to reimplement the switch-variable logic directly as a
GTSAM custom factor rather than trying to get the original Vertigo code
running.

## Reproducibility / Seeding

Seeds that matter: the **synthetic-generator seed** (controls which
graphs/outlier patterns are produced) and the **model-init seed**, both set
explicitly per run and logged to wandb. Solvers themselves (LM/GN) are
deterministic given an initial guess — no solver-side seed needed.

## Outlier-Rate / Structure Sweep

Run via Hydra multirun, e.g.:

```
python scripts/train.py --multirun \
  data.outlier_rate=10,30,50,70,90 \
  data.outlier_structure=random,clustered
```

Predefine this grid in `configs/sweep/outlier_grid.yaml` so it's a one-line
invocation. Each run gets its own isolated Hydra output directory
automatically; with the resolved config logged to wandb per run,
`outlier_rate` / `outlier_structure` become filterable/groupable dashboard
fields with no manual bookkeeping. `evaluate.py` aggregates across the sweep
(globbing run directories, or querying the wandb API) into the final
comparison table/plot.

## Logging & Visualization Pipeline

- `trainer.py` persists structured logs (poses, edge weights, convergence
  status, metrics) as JSON at each epoch/checkpoint — not on every step, and
  never calls Rerun directly.
- `rerun_logger.py` takes a **log file path** as input and is invoked
  separately (via `scripts/demo_rerun.py`) to render any past run — training
  progression, final test rollouts, cross-sweep comparisons — without paying
  an I/O cost during training and without needing Rerun open while training
  runs.
- wandb handles live scalar tracking (loss curves, metrics) during training;
  Rerun is for spatial/trajectory visualization, used post-hoc.

## Environment Management — Pixi

- `pixi.toml`: human-authored manifest — Python version, CUDA version (via
  the `cuda-version` conda-forge metapackage), top-level dependencies
  (PyTorch, PyG, PyPose, GTSAM, Hydra, wandb, Rerun).
- `pixi.lock`: auto-generated, hash-level, multi-platform lockfile, produced
  automatically whenever dependencies change — no separate lock-generation
  step needed.
- Use Pixi's **feature/environment composition** deliberately: a `gpu`
  feature (CUDA-enabled torch/PyG build) and a `cpu` feature (fast local
  iteration for `g2o_io` / `graph_builder` / test work that doesn't need
  CUDA), composed rather than forcing every contributor through the full GPU
  stack just to run unit tests.
- **Do a real install spike early**: confirm GTSAM + PyPose + PyG resolve
  together under Pixi before writing much code. If any of the three lacks a
  conda-forge package and needs a pip fallback inside the Pixi environment,
  better to find out in a 20-minute spike than three weeks into
  implementation.
- Fallback if Pixi resolution proves difficult in practice: a Dockerfile
  pinned to a confirmed-working combination, as an escalation path — not the
  default.

## Training Loop Framework

Plain PyTorch loop in `trainer.py` — no PyTorch Lightning or Ignite. Reasons:
- Both frameworks are opinionated about loop shape, and this project's loop
  has real irregularities neither anticipates cleanly: a solver call
  mid-step with its own convergence behavior, fixed-size-graph batch
  construction, and an RL loop coming later that doesn't fit the
  epoch/batch paradigm either framework assumes.
- What these frameworks typically provide — checkpointing, multi-GPU, mixed
  precision, logging integration — is either unnecessary here (pose graphs
  are small; multi-GPU/mixed-precision aren't the bottleneck) or already
  covered (wandb handles logging, Hydra handles config).
- A hand-written loop is more transparent for debugging gradient flow
  through a custom differentiable solver — worth more here than boilerplate
  reduction.

If reconsidered later, Ignite (thinner, event-handler-based) is the less
disruptive option versus Lightning's fuller framework — but only worth
revisiting if `trainer.py` accumulates enough repetitive boilerplate despite
Hydra/wandb already covering config and logging.

## VSA/SSP Extension — Implementation Notes (Phase 2, stretch)

See `architecture.md` §8 for the integration point and rationale. No
implementation decisions are locked yet beyond: build as a standalone
`evaluate.py` baseline arm first, and resolve the plain-PyTorch vs.
Nengo/nengo-spa dependency question before starting the module, since it
determines whether Pixi's `pixi.toml` needs a new dependency group.
