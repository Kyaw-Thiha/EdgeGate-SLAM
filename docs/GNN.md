# EdgeGate SLAM — GNN Architecture (`edgegate/models/`)

This document is the single source of truth for the GNN's internal design —
the component-level decisions inside `layers.py` and `edgegate_gnn.py`. Read
`architecture.md` first for how the GNN fits into the overall system (its
inputs/outputs at the `Solver` boundary); this document is one level deeper.

Decisions below are locked for **Phase 0** (supervised edge classification,
`edge_bce` loss). Anything marked as an ablation is explicitly deferred to
Phase 1, tracked here so it's picked up on purpose rather than re-litigated
or forgotten.

---

## 1. Inputs (from `graph_builder.py`, unchanged by this document)

- `x` — `(N, 3)` node features: initial pose guess `[x, y, θ]`.
- `edge_attr` — `(E, 6)`: `[dx, dy, dθ, Ixx, Iyy, Iθθ]` (measurement +
  information-matrix diagonal, per `implementation_details.md`'s locked Data
  Schema decision — no residual-under-initial-guess feature in Phase 0/1).
- `edge_type` — `(E,)`: `0 = odometry, 1 = loop-closure`.
- `edge_index` — `(2, E)`, directed i→j, matching the measurement
  convention (measurement is "pose j relative to pose i").

---

## 2. Component-by-component design

### 2.1 Message function — **type-specific linear projection**

Each edge's message is `W_{type} · concat(x_j, edge_attr)`, where
`W_odom` and `W_loop` are two independent `nn.Linear` layers selected by
`edge_type`. Implemented as a `MessagePassing` subclass, `aggr="add"`.

**Why this and not the alternatives:**

- **Rejected: shared MLP + learned type embedding.** Shares parameters
  across odom/loop-closure edges, which is the opposite of what the
  project's core hypothesis argues for (odom and loop-closure edges have
  structurally different roles and reliability — see
  `EdgeGate_SLAM_Research_Proposal.md` §3). A shared MLP has to *learn* this
  distinction from a type embedding; a type-switched linear map makes it
  architecturally explicit instead.
- **Rejected (for Phase 0, kept as Phase 1 ablation): full edge-conditioned
  convolution (ECC/NNConv).** In this family, a small hypernetwork MLP maps
  `edge_attr` directly to a continuous weight matrix (Simonovsky &
  Komodakis, 2017), rather than switching between two discrete matrices.
  This is notably the design used by the closest sibling paper, *Policies
  over Poses* (Ghanta & Parasuraman, 2025,
  [arXiv:2510.22740](https://arxiv.org/abs/2510.22740)), whose GNN encoder
  applies edge-conditioned convolution with adaptive edge-gating to denoise
  noisy edges. It's a strictly more expressive superset of the
  type-switched design (type-switching is the degenerate 2-bucket case of
  edge-conditioning). Not used in Phase 0 because it introduces a second
  source of uncertainty (hypernetwork capacity/init) while the rest of the
  pipeline is still being validated — see §4, Ablation A.

### 2.2 Confidence head — **node embeddings + re-injected edge features + edge-type one-hot, loop-closure edges only**

After `L` rounds of message passing, edge `(i→j)`'s confidence score is:

```
head_input = concat(h_i, h_j, edge_attr_embed, onehot(edge_type))
confidence = sigmoid(MLP(head_input))   # MLP: Linear -> ReLU -> Linear -> 1
```

**Why:**

- Re-injecting `edge_attr` (not just `h_i, h_j`) gives the head a direct
  path to the raw measurement and information-matrix magnitude, which node
  aggregation across `L` layers can blur or dilute.
- The extra explicit `onehot(edge_type)` (beyond what already went into the
  message function) exists because by the time the head sees `h_i, h_j`,
  `edge_type` has only acted as a *routing* signal (which linear map was
  applied) — it isn't necessarily still recoverable as a feature value after
  several rounds of nonlinear aggregation. Making it explicit again costs
  nothing and removes a guess.
- `h_i` and `h_j` are concatenated in directed order, not pooled
  symmetrically (`h_i + h_j`), because the measurement itself is directional
  (i→j) — this is an intentional match to that convention, not an oversight.

**Decision (locked, July 2026): the confidence head is only ever invoked on
loop-closure edges.** Odometry edges receive a hardcoded `confidence = 1.0`
that never passes through the network, at both train and eval time. This
follows directly from `edge_bce`'s loss being masked to loop-closure edges
(see `implementation_details.md`'s "Loss Function Design" section): the
synthetic generator never corrupts an odometry edge, so there is no
outlier-detection task on odometry to supervise in the first place, and
running the head on them anyway would mean an unsupervised, effectively-
arbitrary value silently scaling the information matrix of the graph's most
reliable edges. Masking the loss but still computing (and using) the head's
output on odometry would be an inconsistency between training and
deployment; hardcoding `w_odom = 1.0` end-to-end removes it. Node embeddings
(`h_i`, `h_j`) for odometry-adjacent nodes are still computed and still flow
through message passing as normal — only the *confidence-head application*
is restricted to loop-closure edges, not the representation learning
upstream of it.

### 2.3 Residual connections — **yes, with normalization**

`h = h + Dropout(Conv(h))`, followed by a normalization layer
(`GraphNorm` or `LayerNorm`) applied after the residual add, at every layer
beyond the first.

**Why:** Standard practice for training stability in stacked GNNs, and
specifically relevant here because a `≥3`-layer GNN without residuals is
prone to oversmoothing on graphs where all loop-closure edges are
downstream of the same long odometry chain. Normalization is added on top of
residuals — not just the residuals alone — because the two edge types have
structurally different information-matrix magnitudes by construction
(fixed isotropic values `diag(500,500,100)` for odometry vs.
`diag(100,100,50)` for loop-closure, per `implementation_details.md`'s
Synthetic Generator Design decision). That ~5x scale mismatch compounds
across residual layers without normalization, and becomes a live risk once
the K-sweep makes depth a variable rather than a fixed constant.

### 2.4 Edge features across layers — **fixed, not updated**

Edge representations (`edge_attr`) are treated as fixed inputs to the
message function at every layer; there is no per-layer edge-state update
(no "full MPNN with mutable edge state").

**Why:** Matches the original Message Passing Neural Network formulation
(Gilmer et al., 2017), which the ECC/NNConv family itself builds on — edge
features are consumed, not mutated, in that lineage. Also keeps `layers.py`
an independently testable building block (`architecture.md` §4's stated
purpose for that module): an edge-state-updating layer requires a second
set of per-layer parameters and is harder to unit-test in isolation.
Deferred as a Phase 1+ ablation, not rejected outright.

---

## 3. Locked parameters (Phase 0)

| Parameter | Value | Why |
|---|---|---|
| `num_layers` | 3 | 2 is the minimum depth for a loop-closure edge to receive any signal from its surrounding odometry context beyond its immediate endpoints; 3 gives a small margin without inviting oversmoothing on graphs this small (synthetic graphs are tens to low hundreds of nodes). |
| `hidden_dim` | 64 | Small graphs, synthetic-data-first regime — no evidence yet that more capacity is needed. Sweep candidate `{32, 64, 128}` if Phase 1 budget allows, but not a Phase 0 blocker. |
| `dropout` | 0.1–0.2 | Applied inside the message MLP and the confidence head only — never on raw aggregation, since dropping aggregated signal (as opposed to learned-weight signal) would directly corrupt the very information the solver depends on. |
| `aggr` | `"add"` (sum) | Standard MPNN choice; matches the `MessagePassing` base class default already committed to in the stub. Kept as-is for Phase 0 — see §4, Ablation B for why sum-aggregation itself is a live candidate for revision. |
| Confidence head shape | `Linear(3·hidden_dim + edge_attr_embed_dim + 2 → hidden_dim) → ReLU → Linear(hidden_dim → 1) → sigmoid`, **invoked on loop-closure edges only** | `3·hidden_dim` from `concat(h_i, h_j, edge_attr_embed)`; `+2` from the `onehot(edge_type)` re-injection (§2.2). Sigmoid because the confidence score is defined project-wide as `[0,1]` (see `architecture.md` §"Core Architectural Decisions" #1 and `implementation_details.md`'s edge-weight → information-matrix convention). Odometry edges get a hardcoded `w=1.0` instead — see §2.2. |
| Normalization | `GraphNorm` (or `LayerNorm` if `GraphNorm` proves unstable in practice) | Counteracts the ~5x fixed information-matrix scale mismatch between edge types (§2.3). Exact choice between the two left to empirical check during implementation — not a design-level decision worth locking prematurely. |
| Weight init | PyTorch default (Kaiming/He, via `nn.Linear` defaults) | No project-specific reason yet to deviate from standard practice. |

---

## 4. Planned ablations (Phase 1+, not blockers for Phase 0)

Ordered by priority — highest first.

### Ablation A — Continuous edge-conditioning (ECC/NNConv) vs. discrete type-switch
Swap §2.1's two-`Linear` type switch for a single hypernetwork MLP mapping
`edge_attr` (incl. one-hot type) to a continuous weight matrix, matching the
ECC/NNConv family and the design used by *Policies over Poses*. Tests
whether continuous edge-conditioning captures anything the discrete
2-bucket split misses (e.g. within-type variation in information-matrix
magnitude that a single shared-per-type matrix can't differentially
weight). Directly comparable against the Phase 0 baseline through the same
`evaluate.py` harness — no other pipeline changes required.

### Ablation B — Attention-based aggregation (`TransformerConv`) vs. sum-aggregation
Swap `aggr="add"` sum-pooling for graph attention (PyG's `TransformerConv`,
which natively supports edge features). Highest-priority *aggregation*-level
ablation because it's the most direct architectural answer to the project's
actual problem: sum-aggregation has no mechanism to down-weight a bad
neighbor's message *before* the confidence head sees it — a corrupted
loop-closure edge contributes to the sum unconditionally. Attention gives
each node a learned per-neighbor importance weight during aggregation
itself, which is conceptually closer to "some incoming edges are inliers,
some are outliers" than post-hoc scoring after uniform pooling. Cheap to
implement (drop-in `MessagePassing` layer swap).

### Ablation C — Recurrent (GRU) processing of the odometry chain
The odometry sub-graph is literally a path graph (sequential poses in time
order). A GRU/LSTM walking that chain, with loop-closure edges injected as
cross-links or a separate attention term, is a legitimate inductive bias —
and matches the hybrid shape used by *Policies over Poses* (GRU memory
layered on top of its edge-conditioned GNN encoder). Deferred below A and B
because it's a larger architectural departure (a genuinely different
processing paradigm, not a drop-in layer swap) and the project's stated
premise is that the inlier/outlier signal is a *topological* pattern
extractable by message passing alone — a recurrent-chain model would need
its own justification for why sequential order helps beyond what topology
already encodes.

### Ablation D — Edge-state updating across layers (full MPNN with mutable edge state)
See §2.4. Lowest priority: no literature precedent in this project's direct
lineage suggests it's needed, and it adds a second set of per-layer
parameters plus testing complexity for a currently-hypothetical gain. Only
worth revisiting if A and B fail to close a specific, diagnosed gap that
edge-state mutation would plausibly explain.

### Not planned as ablations (and why)
- **CNN-based architecture**: pose graphs have no grid structure — a CNN
  has no natural way to consume irregular topology. The only way a CNN
  re-enters this project is via an occupancy-grid representation, which is
  a different sub-problem entirely (that's what the VSA-OGM baseline
  variant would cover, per `EdgeGate_SLAM_Research_Proposal.md` §7 — not
  this GNN).

---

## 5. Open questions (not yet locked)

- Exact choice between `GraphNorm` and `LayerNorm` (§3) — resolve
  empirically once training is running, not a design-level blocker.
- Whether Ablation A and Ablation B should ever be *combined* (attention
  aggregation over a continuously edge-conditioned message) — deferred
  until each is evaluated independently, same reasoning as
  `implementation_details.md`'s decision not to cross the K-sweep with the
  outlier-rate sweep prematurely.
