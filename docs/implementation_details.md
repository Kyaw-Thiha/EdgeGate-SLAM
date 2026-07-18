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

**Decision (locked, July 2026): edge features = raw measurement
`(dx, dy, dθ)` + `edge_type` + information-matrix diagonal/norm.** No
residual-under-initial-guess feature in Phase 0/1. Reasoning:

- This matches the proposal's actual claim — that the inlier/outlier
  distinction is a topological consistency pattern extractable by message
  passing. Including the residual would quietly turn the model into a
  hybrid geometric+topological one, and the proposal's framing would need to
  be rewritten to still be accurate.
- The residual-under-initial-guess is *literally* the quantity DCS/GNC
  threshold on. If the GNN's input already contains it, an F1 win over
  those baselines stops being an interesting result — of course a model
  handed the classical kernel's own signal matches or beats it. The
  scientifically clean question is whether pure topology, without that
  signal, gets competitive performance.
- Computing it correctly requires composing the full odometry chain in
  SE(2) (not naive coordinate subtraction) — a real chance for a subtle bug.
  Deferring it keeps Phase 0 from depending on getting that right first.

The information-matrix diagonal/norm is included from the start (not
deferred alongside the residual) because it isn't a *derived geometric*
signal — it requires no solving, it's metadata already present on every
real `.g2o` edge, and excluding it doesn't buy the same clean-ablation
story that excluding the residual does.

See §"Future Work" for revisiting the residual feature as a Phase 2+
addition once a pure-topology baseline is established.

**Decision (locked, July 2026): add `gt_node_poses` as a new `PoseGraph`
field** — `(N, 3)` clean/noise-free trajectory, synthetic-only (`None` for
real `.g2o` data), same optionality pattern as `edge_label`. Not stubbed with
`NotImplementedError`: `synthetic_generator.py` already computes this
trajectory internally, before Gaussian noise is applied, in order to produce
`edge_label` for free (per `EdgeGate_SLAM_Research_Proposal.md` §4) — the
same generator pass can expose it. Required by both `trajectory_loss.py`
(training target) and `ate_rmse.py` (eval-time ground truth).

**Ground-truth availability — corrected (July 2026):** No current benchmark
`.g2o` file — including M3500, Sphere2500, or any dataset from Carlone's
dataset page — ships with an independently-verifiable ground-truth trajectory.
Every "ground truth" in the PGO literature, including this project, is
**pseudo-GT: the optimized trajectory of the outlier-free data** (explicitly
stated in Carlone et al. IROS 2014 for Intel/M3500: *"we take as ground truth
the optimized trajectory of the outlier-free data"*). The `gt_node_poses` field
is `None` for all real `.g2o` files as loaded from disk; `scripts/evaluate.py`
computes a pseudo-GT reference solve and stores it as `gt_node_poses` at eval
time so ATE is reportable on all benchmarks. Any ATE reported on real data in
this project is ATE-against-reference-solve, never true ground truth — this
distinction must be stated explicitly wherever results are reported.

## GNN Architecture

Component-level design for `edgegate/models/layers.py` and
`edgegate_gnn.py` — message function, confidence head, residuals/
normalization, locked hyperparameters (with reasoning), and the planned
Phase 1+ architecture ablations (edge-conditioned convolution, attention
aggregation, recurrent chain processing) — is documented separately in
**`GNN.md`**, not duplicated here. This section stays the model's
integration surface (its inputs/outputs at the data-schema and `Solver`
boundary); `GNN.md` is the internals. Update `GNN.md`, not this section,
when the message-passing layer, confidence head, or GNN hyperparameters
change.

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

**Scope (locked, July 2026): this scaling applies to loop-closure edges
only.** Odometry edges always get `w = 1.0` — passed as a fixed constant, not
as the GNN's output. See "Loss Function Design" below for why the confidence
head is never trained or evaluated on odometry edges to begin with; scaling
by an untrained value would be the same bug at solve time that masking
avoids at train time. `scale_information` itself doesn't need to know
about edge type — callers simply always pass `w=1.0` for odometry rows of the
weight tensor.

## Loss Function Design (`edgegate/losses/`)

### `edge_bce.py` — masked to loop-closure edges, odometry excluded end-to-end

**Decision: compute BCE only over loop-closure edges** (`forward(confidence,
labels, edge_type)`, masking to `edge_type == 1`), not all edges, and not
`pos_weight`-scaled BCE over the full edge set.

Reasoning: odometry labels are always `1` by construction (the synthetic
generator never corrupts an odometry edge — see `EdgeGate_SLAM_Research_
Proposal.md` §3's own framing that odometry is "individually reliable" and
only loop closures are the outlier target). Including odometry in the loss,
weighted or not, only adds trivially-easy background examples to a task
that's actually about loop-closure classification; it doesn't sharpen the
signal, it dilutes it. `pos_weight` would be the right tool if odometry
edges were a real (if rare) negative class — they aren't, so there's no
class-imbalance problem to correct for once odometry is out of the loss at
all.

**This decision has a load-bearing consequence for the confidence head,
not just the loss:** if the loss never touches odometry edges, the head's
output for them is never supervised — running it anyway and using that
unsupervised value to scale `Λ` at solve time would silently corrupt the
most reliable edges in the graph with an arbitrary (near-init) weight. So
the confidence head is **only ever invoked on loop-closure edges**, at both
train and eval time; odometry edges get a hardcoded `confidence = 1.0` that
never touches the network. This keeps the masked-loss decision and the
solve-time behavior telling the same story, rather than masking the loss
while leaving an untrained side-channel live in the solver. See `GNN.md`
§2.2 for the confidence-head-level version of this decision, and note the
corresponding one-word correction to `architecture.md`'s Figure 1 caption:
confidence scales each *loop-closure* edge's information matrix, not "each
edge's."

### `trajectory_loss.py` — position-only error, `gt_node_poses` as target

**Decision: mean squared Euclidean distance on `(x, y)` positions only**,
not full SE(2) pose error (translation + wrapped angle residual), for
Phase 0/1.

Reasoning: positions are what ATE measures anyway (see `ate_rmse.py` below),
so training-time surrogate loss and eval-time metric stay aligned — not
training against one notion of error and reporting another. Avoiding the
angle-wrapping component also keeps the gradient path simpler during
early-K-iteration training, when poses may still be far from ground truth
and a poorly-conditioned angle residual is more likely to produce unstable
gradients than a clean quadratic position term. Full SE(2) residual error
(requiring `se2_utils.py`'s angle-wrapping) is a legitimate Phase 1+
ablation once the K-sweep/training pipeline is otherwise stable — this is
the direction differentiable-factor-graph-optimization papers in the
literature review (Yi et al. 2021) tend to go, backpropagating the full
pose error rather than position alone.

`gt_poses` is `PoseGraph.gt_node_poses` (see Data Schema above) — computed
by the synthetic generator, not stubbed. Since `gt_node_poses` is `None` for
real `.g2o` data by construction, `trajectory_loss` is a synthetic-only
training loss, consistent with the Train/Val/Test protocol below (real data
is eval-only regardless of which training loss produced the model).

## Synthetic Generator Design

Decisions locked July 2026, after checking what the closest sibling papers
(Policies over Poses, and the standard PGO benchmark lineage — Manhattan,
Sphere2500, Grid/City10000, Torus, Cubicle, Rim, all descending from Olson's
2006 "Manhattan world" generator and Carlone/Rosen's SE-Sync suite) actually
do. None of them use a free random walk — this is worth internalizing before
writing `synthetic_generator.py`, since it changes the trajectory-shape
decision below from a free design choice to "match established practice."

### Trajectory shape

**Decision: Manhattan-world-style structured generation (grid-constrained
motion, axis-aligned segments with turns, periodic revisits at a
controllable rate) — not a free 2D random walk.**

Reasoning: loop closures require geometric revisit structure to occur
meaningfully at all. A true random walk revisits earlier positions only by
chance, at a rate that depends on step size and drift — this produces a
trajectory topology that doesn't resemble the real benchmarks (Intel, M3500)
being evaluated against for domain shift, both of which are corridor/room
traversals with *deliberate* repeated visits, not undirected wandering. Every
synthetic dataset in the standard PGO benchmark suite is structured for
exactly this reason. This is treated as borrowed infrastructure, not a
research contribution surface: none of the surrounding literature claims
novelty for its trajectory generator, and design effort spent here has
near-zero payoff for this project's actual contribution (edge-type-aware
confidence + generalization-gap characterization) while a low-quality
generator carries real risk of silently corrupting the Phase 1 measurement.

Explicitly rejected: a full physics/sensor simulator (Gazebo, etc.) — out of
scope, no evidence any comparable paper needs one for pose-graph-level
evaluation.

### Loop-closure placement

**Decision: inlier and outlier loop closures are sampled from different
populations, not the same proximity-agnostic pool.**

- **Inlier (true) loop closures**: sampled between pose pairs that are
  actually spatially close (within a proximity threshold) — this is what a
  real place-recognition front-end does, and falls out naturally from the
  Manhattan-world generator's revisit structure above (the robot is
  genuinely back at a previously-visited grid cell).
- **Outlier loop closures**: sampled between pose pairs that are spatially
  **distant** — modeling perceptual aliasing, where the front-end proposes a
  closure between two poses that *look* similar but aren't actually
  nearby. This is what makes them wrong, and what makes them the hard,
  realistic case to catch geometrically.

Rejected: sampling outliers uniformly at random over all pairs regardless of
proximity. Most such pairs would already be trivially far apart in the
trajectory and easy to reject, producing an artificially easy synthetic
outlier distribution that inflates F1 relative to real perceptual aliasing —
exactly the kind of thing that shows up later as an unexplained sim-to-real
gap with no clear cause.

### Outlier measurement type

**Decision: treat this as a locked, documented sweep axis (like outlier rate
and structure), with the Gaussian-shifted-offset case as primary and
uniform-random as the explicit "easy ablation," not the default.**

- **Uniform random replacement**: wildly geometrically inconsistent, trivial
  for any topological method to flag. The easy case.
- **Gaussian shift by a large offset**: a plausible-looking but wrong
  measurement — closer to what actual perceptual aliasing produces, since a
  misidentified place usually does look somewhat geometrically similar to
  the true one. The hard, realistic case.

Given that Phase 1's whole framing is an honest generalization-gap
characterization, headline synthetic numbers should be reported primarily
against the hard case; using uniform-random as the default would risk
inflating synthetic performance by construction.

### Information matrices

**Decision: fixed isotropic values to start** (matches standard benchmark
convention, e.g. `diag(500, 500, 100)` for odometry, `diag(100, 100, 50)`
for loop-closures), **not sampled per-edge, for Phase 0/1.** Removes a
confound while the core pipeline is being built. See §"Future Work" for
sampled information matrices as a Phase 1 dependency once the noise-scale
domain-shift metric needs a range to correlate against.

### Synthetic outlier realism — cross-cutting note

Loop-closure placement, outlier measurement type, and information-matrix
sampling all bear directly on how "easy" synthetic outliers are relative to
real ones — precisely the axis the Phase 1 generalization-gap study is
trying to measure. These are locked-before-training decisions, not generator
implementation details to be revisited casually after seeing results; if
they're loosened or changed after Phase 1 numbers come in, the
generalization-gap claim is no longer honestly measured, for the same reason
the real-benchmark eval-once rule exists.

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

**Resolved (July 2026):** PyPose's LM (`pypose.optim.LM`) backpropagates via
**full unrolling** — it uses `modjac` (module Jacobian via standard autograd)
to compute `∂residuals/∂parameters` at each iteration step, and the gradient
chain runs through every step back to the GNN outputs. This is the "prior
work" approach Theseus's paper contrasts against its implicit-differentiation
advance. Memory cost scales with K (number of unrolled steps), which is why
`train.solver_train_iterations` must stay small for feasibility. For K ≤ 5
this is manageable; for K > 20 it likely requires gradient checkpointing.
No solver-side change is needed for Phase 0 (`edge_bce` never calls the
solver during training); revisit memory profiling when `trajectory_loss` is
wired in Phase 1.

**PyPose has no SE(2) Lie type** (only SE3/SO3/RxSO3/Sim3 are available in
v0.9.5). `pypose_solver.py` stores poses as plain `(N, 3)` `nn.Parameter`
tensors and implements the SE(2) residual manually in `torch` — the LM
optimizer treats all parameters as Euclidean-space, which is correct for
small-angle steps (no manifold-aware retraction needed). Pose 0 is stored
as a fixed `register_buffer` (gauge anchor); poses 1..N-1 are optimized.

## Solver Adapter Implementation Notes

### Cost convention mismatch
PyPose and GTSAM use different cost conventions:
- **PyPose** minimises `sum(||r_white||²)` where `r_white = r * sqrt(info_diag)` — the full squared whitened-residual sum.
- **GTSAM** minimises `½ · sum(r^T Λ r)` — the standard ½-chi² convention.

For diagonal information matrices (which is all Phase 0/1 uses), `pp_cost ≈ 2 × gt_cost` at convergence. This is a pure bookkeeping difference; the optimum found is the same. Confirmed empirically: relative cost disagreement is < 0.3% across all test seeds.

### GNC non-composability (locked)
`GTSAMSolver(kernel="gnc")` must **never** receive GNN-predicted `edge_weights`. GNC computes its own adaptive inlier weights internally from raw residuals — passing pre-scaled information matrices (from `scale_information(Λ, w²)`) would double-robustify the problem and confound any GNN-vs-classical comparison. A runtime `AssertionError` is raised if non-unit weights are passed to `kernel="gnc"`. This kernel should only ever be invoked in `evaluate.py`'s classical-baseline branch, on raw `edge_info`, with uniform weights.

### Solver agreement criterion
The `test_solvers.py` agreement test checks **cost equivalence**, not pose equivalence. With few loop closures (~4 in a 50-node graph), the PGO problem has near-degenerate directions: two correct backends can find different pose configurations with the same objective value (null-space non-uniqueness). Empirically confirmed on seed=0: max position discrepancy 0.17 units with < 0.3% cost difference. Pose-distance comparison would produce spurious test failures for a mathematically correct solver pair. The 2% cost threshold in `test_pypose_gtsam_agree_on_clean_graph` is conservative; tighten only if a specific rounding-error bound becomes relevant.

### GTSAM GNC iteration count
`GncGaussNewtonOptimizer` does not expose an `iterations()` method. `GTSAMSolver(kernel="gnc")` returns `num_iterations=-1` as a sentinel. Callers should treat `-1` as "unknown" and not compare it against `kernel="none"` iteration counts.

## Train / Val / Test Split Protocol

- **Synthetic data** (generator-produced graphs) is the only data used for
  training and validation / hyperparameter selection.
- **Real benchmark `.g2o` datasets** (Intel, M3500, Sphere2500,
  parking-garage) are held out entirely — never touched during training or
  hyperparameter tuning. Report results on them exactly once per model
  version, not iteratively, or they quietly become a validation set and the
  synthetic-to-real generalization gap (the project's own stated ethical
  consideration) is no longer being honestly measured.

**Note on `manhattan.g2o`:** Confirmed to be the same underlying graph as
M3500 (Olson et al. 2006) with different information-matrix scaling — a
different noise model on the same trajectory, not an independent benchmark.
It is excluded from the evaluation sweep (`REAL_DATASETS` in
`scripts/run_sweep.py`). If noise-scale variation becomes a sweep axis
(see "Sampled information matrices" in Future Work below), the
M3500/manhattan pair can serve as a locked noise-scale ablation rather than
two independent datasets.

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

### Ground-truth availability splits which metric applies where (locked, July 2026)

Not every real benchmark supports every metric, and this needs to be locked
into the protocol now rather than discovered mid-Phase-1:

- **All real benchmarks (M3500, Sphere2500, Intel, MIT/CSAIL)** use
  pseudo-GT (see corrected caveat above). Both **edge-F1** and
  **pseudo-GT-aligned ATE** are reportable on all benchmarks, but ATE must
  always be labeled "ATE-against-reference-solve" wherever reported —
  never presented as true ground truth.
- **Intel and MIT/CSAIL** are real sensor logs; **M3500 and Sphere2500**
  are simulator-generated trajectories. All four lack independently-sourced
  external ground truth. The distinction that matters for result labeling is
  not simulator vs. sensor, but pseudo-GT vs. true GT — and none of these
  datasets have true GT.
- **Outlier injection on real data**: as originally released, Intel/MIT/
  CSAIL contain no known false-positive loop closures — this is standard
  across the literature (e.g. AEROS notes that these datasets do not provide
  false-positive loop closures, so outliers are randomly injected using the
  same procedure as the synthetic datasets). This project follows the same
  convention: edge-F1 evaluation on real data means real trajectory geometry
  and real odometry/measurement noise, with synthetically-injected
  loop-closure outliers layered on top via the same
  `synthetic_generator.py` outlier-injection logic used for the fully
  synthetic sets. This must be stated explicitly alongside the eval-once
  results — "real" in this study's generalization-gap claim covers
  trajectory/noise statistics, not the outlier labels themselves, and
  stating this upfront (rather than letting a reader assume otherwise)
  is part of the honest-measurement discipline the eval-once rule already
  commits to.

## Metrics Implementation Notes (`edgegate/metrics/`)

### `edge_f1.py` — return counts, not just the three scalars

**Decision: return `{"precision", "recall", "f1", "tp", "fp", "fn"}`**, not
just precision/recall/F1.

Reasoning: F1 is a nonlinear function of `tp`/`fp`/`fn`, so averaging
per-batch F1 scores and averaging per-batch counts-then-computing-F1-once
diverge whenever batch size or class balance varies across batches — which
it will, across the outlier-rate/structure sweep. `trainer.py`/`evaluate.py`
must accumulate counts across batches/graphs and compute F1 exactly once at
the point of reporting, never average an average.

### `ate_rmse.py` — Umeyama alignment, not zero-anchoring

**Decision: full Umeyama alignment** (SE(2) rigid-body alignment, no scale,
closed-form via SVD) between estimated and ground-truth trajectories before
computing RMSE, not simple zero-anchoring (subtract `poses[0]`).

Reasoning: this is the field's actual convention, not just the more-correct
option — it's the same alignment step used across the standard SLAM
evaluation tooling lineage (Sturm et al. 2012's TUM RGB-D benchmark
convention, and the `evo` library built on it). Zero-anchoring is only
correct when both trajectories share the same origin/gauge by construction,
which holds for synthetic data (our solver anchors pose 0, and
`gt_node_poses` is generated in the same frame) but is not guaranteed on
real data, and gauge freedom in PGO is exactly the failure mode Umeyama
alignment exists to remove — two trajectories with identical internal
geometry but different global placement should register as ATE ≈ 0, and
zero-anchoring doesn't guarantee that in general.

**Real-benchmark caveat, cross-referencing the Data Schema section above:**
Umeyama alignment requires a `gt_node_poses` target to align against. This
is only genuinely available for the simulated real benchmarks (M3500
family, Sphere2500). For Intel and MIT/CSAIL, there is no independently
sourced ground truth to align to — see the Train/Val/Test protocol below
for how ATE reporting is scoped accordingly.

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
running. **Deferred to follow-up (July 2026):** the `configs/eval/method/switchable.yaml`
config exists as a stub raising `NotImplementedError` with a pointer here.
GNC and DCS are the classical baselines available in Phase 0. Reimplementing
switchable constraints requires constructing a custom factor graph with one
switch variable per loop-closure edge plus a prior on each switch, and
running an alternating optimisation schedule (LM on poses / closed-form
switch update). See §"Future Work" for the deferred item.

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

## Future Work

Items discussed and deliberately deferred — not blockers for Phase 0/1, but
recorded here so they're picked up on purpose rather than forgotten or
re-decided by accident later.

- **Residual-under-initial-guess as an edge feature.** Deferred out of
  Phase 0/1 (see Data Schema section above) so the MVP stays a clean
  pure-topology ablation against DCS/GNC. Once that baseline exists,
  revisit as an explicit second config (`graph_builder.py` feature flag)
  and compare — this is the natural way to measure how much a hybrid
  geometric+topological model buys over pure topology, rather than
  assuming it helps. Requires correct SE(2) chain composition, not naive
  coordinate subtraction.
- **Trajectory-shape / revisit-rate as a domain-shift metric.** Flagged
  during the synthetic-generator discussion: even with Manhattan-world-style
  generation replacing random walk, the synthetic revisit rate/structure may
  still not match a given real benchmark's (Intel vs. M3500 likely differ
  from each other, too). Candidate fourth domain-shift metric alongside the
  three already locked in §"Domain-Shift Characterization Metrics" —
  definition needs to be worked out and locked before it can be used, same
  rule as the existing three.
- **Sampled (non-fixed) information matrices.** Fixed isotropic values are
  the Phase 0/1 default (see Synthetic Generator Design above). Revisit once
  Phase 1's noise-scale-mismatch metric needs synthetic data spanning a
  *range* of noise scales to correlate degradation against — fixed isotropic
  can't produce that variation by construction.
- **Crossing the K-sweep (solver iterations) with the outlier-rate/structure
  sweep.** Currently run independently on purpose (K is a training-dynamics
  question, outlier rate is a robustness question — crossing them multiplies
  compute for a combined question not yet needed). Revisit only if a
  specific hypothesis emerges that needs the interaction term.
- **Full physics/sensor simulator (e.g. Gazebo).** Explicitly considered and
  rejected for trajectory generation (see Synthetic Generator Design above)
  — no evidence any comparable paper needs one at the pose-graph level.
  Would only become relevant if the project ever moved upstream into
  front-end/sensor-level work, which is out of scope for this project's
  current framing.
- **Outlier-type sweep beyond the two locked cases.** Uniform-random and
  Gaussian-shifted-offset are the two cases currently planned (Synthetic
  Generator Design above). Other structured outlier models (e.g.
  systematically biased along a specific axis, or correlated/clustered
  offset magnitudes rather than i.i.d.) are a possible future addition if
  the Gaussian case alone proves insufficient to explain a real-benchmark
  degradation pattern.
- **Full SE(2) trajectory-loss error (translation + wrapped angle
  residual) vs. position-only MSE.** Position-only is the Phase 0/1 default
  (see "Loss Function Design" above) to keep early-training gradients clean
  and to match what `ate_rmse.py` measures. Revisit once `trajectory_loss`
  training is stable — the differentiable-factor-graph-optimization
  literature (Yi et al. 2021) typically backprops the full pose error, and
  it's an open question whether the rotation component matters enough here
  to justify the added angle-wrapping complexity.
- **GNN architecture ablations (full detail in `GNN.md` §4).** Four
  candidates, ordered by priority, none blocking Phase 0: (A) swapping the
  type-specific linear message function for a continuous edge-conditioned
  convolution (ECC/NNConv-style, matching *Policies over Poses*'s design);
  (B) swapping sum-aggregation for attention-based aggregation
  (`TransformerConv`); (C) a GRU/recurrent pass over the odometry chain,
  with loop-closures as cross-links, again echoing *Policies over Poses*'s
  hybrid GRU + edge-conditioned-GNN shape; (D) edge-state updates across
  layers (full mutable-edge-state MPNN). All four require only
  `layers.py`/`edgegate_gnn.py` changes and go through the existing
  `evaluate.py` harness unchanged.
- **Switchable constraints (Vertigo reimplementation).** Deferred out of
  Phase 0 (see §"Baseline Integration in `evaluate.py`"). The reference
  Vertigo implementation targets GTSAM 2.0 (pre-rewrite, incompatible
  with GTSAM 4.x). Reimplementing it requires constructing a custom factor
  graph with per-loop-closure switch variables + priors and an alternating
  optimisation schedule (LM on poses, closed-form switch update) —
  estimating ~60–80 lines of GTSAM factor-graph construction. The
  `configs/eval/method/switchable.yaml` stub raises `NotImplementedError`
  pointing here. GNC and DCS are the classical baselines available in
  Phase 0/1.
- **SE(3) extension (sphere2500, parking-garage).** The SE-Sync repo ships
  sphere2500 and parking-garage as `VERTEX_SE3:QUAT` / `EDGE_SE3:QUAT` format
  (3D quaternion poses with 6-DOF measurements and 6×6 information matrices),
  while the current `g2o_io.py` parser, `PoseGraph` fixed array shapes,
  `graph_builder.py`/GNN feature dimensions, both solver adapters, and
  `ate_rmse.py` (2D Umeyama only) are all SE(2)-only. The `PoseGraph.manifold`
  field already exists as a marker. Full SE(3) support requires extending every
  layer of the stack: parser (recognise SE3:QUAT), `PoseGraph` (dynamic or
  separate array shapes), `graph_builder.py` (dispatch on manifold for feature
  dims), GNN model (configurable `node_feat_dim`/`edge_attr_dim`), GTSAM
  solver (swap `Pose2`→`Pose3`), PyPose solver (manual SE(3) residual or SO(3)
  + R³), and ATE metric (3D Umeyama). `evaluate.py` skips SE3 datasets with a
  warning. Not blocking Phase 0/1.
- **KITTI 00/05, TUM FR1-DESK (and optionally EuRoC MAV) as Phase 1+
  real-GT evaluation datasets.** These are the field-standard datasets with
  genuine external ground truth (RTK-GPS <10 cm for KITTI; Vicon
  sub-millimeter for EuRoC; 8-camera motion capture for TUM RGB-D) — unlike
  the current `.g2o` benchmarks, which have only pseudo-GT. The adaptation
  recipe from TACO (Olivastri et al. 2026, arXiv:2606.29851): run ORB-SLAM3
  on the raw sensor data to extract a pose graph with odometry + loop-closure
  edges, use SE-Sync on the outlier-free version as pseudo-GT (same convention
  as current benchmarks), inject outliers via the same
  `outlier_injection.py` random-constraint policy. All three datasets are SE(3);
  the SE(3) stack extension (see above) is a prerequisite. Suitable for Phase 1
  only after the SE(3) extension lands. Add one entry to `_BENCHMARK_PATHS` in
  `scripts/evaluate.py` and one dataset config in `configs/eval_dataset/` per
  dataset.
