# EdgeGate SLAM — Research Proposal & Project Brief

*Living document. Last updated July 2026. Written so a future agent or
collaborator picking up this project can get full context — motivation,
literature grounding, decisions made and why, and current plan — without
re-reading the full project history.*

Author: Kyaw Thiha (Kevin), University of Toronto —
kyaw.thiha@mail.utoronto.ca — https://github.com/Kyaw-Thiha/EdgeGate-SLAM

---

## 1. One-paragraph summary

EdgeGate SLAM learns a graph neural network (GNN) that scores per-edge
inlier confidence for pose-graph SLAM, conditioning a classical weighted
least-squares solver instead of relying on hand-crafted robust kernels
(switchable constraints, DCS, GNC). The pose graph is edge-typed (odometry
vs. loop-closure), and the GNN's confidence score scales each edge's
information matrix before optimization. The project's current framing is
**not** "GNN beats classical kernels on synthetic accuracy" — the literature
review below shows that claim would be weak — but rather a rigorously
characterized **synthetic-to-real generalization-gap study**, with a
vector-symbolic-architecture (VSA) extension as a lower-priority stretch
addition.

## 2. Problem statement (from original proposal)

Pose-graph SLAM's back-end is solved via nonlinear least squares, so a
single incorrect loop-closure constraint (e.g. from perceptual aliasing) can
catastrophically corrupt the resulting map. Robust loop-closure handling is
essential for long-term deployment in repetitive environments and
multi-robot fleets. Prior robust back-ends rely on hand-designed kernels —
switchable constraints, dynamic covariance scaling (DCS), graduated
non-convexity (GNC) — or, more recently, learned GNN approaches (NeuRoRA,
RL-PGO), but the learned approaches restrict their signal to rotations alone
and treat odometry and loop-closure edges uniformly, ignoring their
differing reliability.

## 3. Core research idea

Learn a GNN that scores per-edge inlier confidence from full SE(2)/SE(3)
pose-graph topology (not rotations alone), with an explicit
odometry-vs-loop-closure edge-type prior, conditioning a classical weighted
least-squares solver. Two research angles, supported by the same
architecture (see `architecture.md`):

1. **Supervised edge classification (MVP)** — train against synthetic
   ground-truth inlier labels via BCE, hand weights to a solver at eval time.
2. **End-to-end differentiable training (extension)** — backprop trajectory
   error through a differentiable solver into the GNN, no edge labels
   required.

## 4. Literature landscape and where the actual gap is

A structured literature review (alphaXiv + web search, July 2026) found:

**No exact prior work combines** edge-type-aware GNN confidence +
differentiable solver + trajectory-loss training for pose-graph outlier
rejection. But the surrounding space is filling in fast:

- [Policies over Poses (Oct 2025)](https://arxiv.org/abs/2510.22740) —
  closest sibling. Edge-conditioned GNN with adaptive gating denoiser, full
  SE(2), multi-robot, tested on Intel/M3500/MIT/CSAIL. Trained via MARL
  (Graph-Aware SAC), not differentiable-solver backprop.
- [DeepCORD (Jul 2026)](https://arxiv.org/abs/2607.08735) — unfolds a
  distributed solver into differentiable iterations, trained via
  self-supervised unrolled cost. Learns solver hyperparameters (mass,
  damping, step size), not edge outlier confidence.
- [Learning to Filter Outlier Edges in Global SfM (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Damblon_Learning_to_Filter_Outlier_Edges_in_Global_SfM_CVPR_2025_paper.pdf) —
  GNN edge classifier via plain BCE (the MVP path, essentially), but for SfM
  translation averaging, not SLAM pose graphs — no odometry/loop-closure
  distinction exists in that setting.
- [Differentiable Factor Graph Optimization for Learning Smoothers (2021)](https://arxiv.org/abs/2105.08257) —
  conceptual ancestor of "backprop through the solver," predates GNN-based
  edge weighting.
- RL-PGO (2023) and PoGO-Net (2021) — confirmed rotation-only, no denoising;
  both cited by Policies over Poses as the gap it fills.

**Why classical still wins on raw accuracy**: GNC tolerates 70-80% outliers
and is more accurate than specialized local solvers on standard benchmarks.
[Why does Deep Learning Improve Visual SLAM? (Cioffi & Scaramuzza, 2026)](https://arxiv.org/abs/2607.06023)
found — via controlled ablation on ORB-SLAM3 + DROID-SLAM's optical flow —
that learned gains in visual SLAM come from **frontend data association and
uncertainty estimation**, not from replacing the geometric backend itself;
a classical feed-forward backend fed by learned frontend signals matched or
beat fully learned systems, including out-of-distribution. The real,
measurable pain point of classical robust kernels is **compute cost**
(GNC requires several seconds vs. milliseconds for standard optimizers —
an amortized-inference / learning-to-optimize opportunity, not an accuracy
one) and **brittle manual tuning in distributed/multi-robot settings**
(the explicit motivation for both Policies over Poses and DeepCORD).

**Conclusion drawn from this**: the strongest, least-contested contribution
available right now is not "beat GNC on synthetic accuracy," but a
rigorous, honestly-reported synthetic-to-real generalization-gap
characterization — an angle none of the above papers make their central
contribution, and one that already matches this project's own stated
ethical-considerations section.

## 5. Research directions considered and rejected (with reasoning)

- **MARL-based extension** (matching Policies over Poses / DeepCORD) —
  deprioritized. Requires actor-critic/ADMM-consensus infrastructure
  disproportionate to available time, and is being actively worked by
  better-resourced groups (UGA, Michigan) right now — real risk of
  redundant effort.
- **Full pivot to a different robotics sub-area** (object-goal navigation,
  agile drone flight — inspired by RPG Zurich-adjacent papers the user
  shared: "What Matters in RL-Based Methods for Object-Goal Navigation?"
  (ECCV 2026), "Learning Agile Quadrotor Flight in the Real World" (RSS
  2026, [project page](https://rpg.ifi.uzh.ch/lafr/)), and "Dream to Fly"
  (ICRA 2026)) — deprioritized.
  Not a topic-fit issue; a resource one. These require large-scale
  simulation (240+ GPU-hours), real hardware (quadrotors, motion capture),
  or both — years of accumulated lab infrastructure this project doesn't
  have. The transferable lesson kept from these papers: rigorous empirical
  ablation studies (no new SOTA method, just clarifying what matters) are a
  legitimate, citable contribution shape, and one this project should
  emulate regardless of topic.
- **VSA/SSP as the primary vehicle** — deprioritized to a stretch addition
  (Phase 2), not because it's a bad idea, but because it is unplanned,
  unfamiliar tooling, and shouldn't block getting the core MVP + OOD study
  working first. See §7.

## 6. Execution plan

- **Phase 0 (build now)** — data pipeline (`g2o_io`, `synthetic_generator`,
  `graph_builder`), GNN (`layers.py`, `edgegate_gnn.py`), `edge_bce` loss,
  `Solver` interface + GTSAM adapter, `evaluate.py`'s unified
  baseline-vs-learned harness. No new design decisions needed — this is
  direct execution of `architecture.md` / `implementation_details.md` as
  already written.
- **Phase 1 (OOD / generalization-gap study)** — uses the existing
  Train/Val/Test protocol (real benchmarks held out, reported exactly once)
  unchanged, plus one new deliverable: domain-shift characterization
  metrics (outlier-rate/structure mismatch, edge-type ratio mismatch,
  noise-scale mismatch), locked before training begins, reported as what
  the gap correlates with — not just a flat before/after number.
- **Phase 2 (VSA/SSP extension, stretch)** — see §7. Only pursued once
  Phase 1 results exist.

## 7. VSA/SSP extension (Phase 2 — genuinely new territory)

Motivated by the user's exposure to Spatial Semantic Pointers (SSPs) /
vector symbolic architectures (VSAs) via Michael Furlong (CNRG lab,
Waterloo). Literature check found:

- [SSP-SLAM (Dumont, Furlong, Orchard, Eliasmith — Frontiers in
  Neuroscience, 2023)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1190515/full) —
  spiking-neural SLAM using SSPs for path integration + semantic
  object-location binding. A different SLAM formulation entirely
  (continuous attractor dynamics, front-end/mapping-focused) — no
  loop-closure outlier rejection, no classical/differentiable solver.
- [VSA-OGM (Snyder et al., arXiv Aug 2024 / npj Unconventional Computing
  2026)](https://arxiv.org/abs/2408.09066) — VSA-based occupancy grid
  mapping, explicitly framed around the efficiency/interpretability
  tradeoff. Different sub-problem (occupancy grids, not pose graphs).
- [HyperSpace (Apr 2026)](https://arxiv.org/html/2604.15113v1) — recent
  general framework for spatial encoding in VSAs; useful survey if this
  phase proceeds.

**The gap**: nobody has combined VSA/SSP representations with pose-graph GNN
outlier rejection specifically. Candidate approach: encode relative pose
measurements as SSPs so that composing a cycle of edges algebraically
should return the identity when the loop is consistent; route
deviation-from-identity as a cheap, structurally interpretable outlier
signal — either standalone or as a GNN edge feature.

**Why it could suit this project specifically**: VSA binding/unbinding is
largely algebraic, not gradient-trained — lower sample complexity than the
mainstream GNN route (a real answer to limited compute/data), and supports
training-free online updates (maps onto continual-adaptation interest).
Direct mentorship access via Furlong meaningfully de-risks execution on
unfamiliar tooling.

**Honest tradeoff**: smaller community (VSA-in-robotics is still mostly
DARPA/Army-funded academic work, per Snyder's funding line) — easier bar to
clear for "future researchers build on this" than for "industry adopts
this," at least currently.

**Integration point when built**: new module (`edgegate/features/ssp_encoding.py`),
wired in as a standalone baseline arm through `evaluate.py` first — same
slot as GNC/DCS/switchable-constraints — before any fusion into
`graph_builder.py` / `edgegate_gnn.py`.

**Open decision**: plain PyTorch/NumPy SSP implementation (fast, no new
dependency) vs. Nengo/nengo-spa (adds spiking-simulation dependency, opens a
neuromorphic/Loihi deployment story). Resolve with Furlong before writing
the module.

## 8. Evaluation plan

- **Metrics**: edge-level F1 (outlier detection), ATE/RMSE (full optimized
  map) — both already scoped.
- **Baselines**: uniform edge weights, GNC, DCS, switchable-constraints
  (via GTSAM custom factor, since Vertigo targets an incompatible old GTSAM
  version) — all run through the same `evaluate.py` path as the GNN, per
  architecture decision §5.
- **New for Phase 1**: domain-shift metrics per real benchmark (§6), and
  degradation reported as a function of those metrics, not just a flat
  synthetic-vs-real number.
- **Reporting discipline**: real benchmarks touched exactly once per model
  version — never used for iterative tuning, or the generalization-gap
  claim is no longer honestly measured.

## 9. Open questions / decisions pending

- Domain-shift characterization metric definitions (must be locked before
  the first training run — see `implementation_details.md`).
- VSA implementation library (plain PyTorch vs. Nengo/nengo-spa) — pending
  conversation with Furlong.
- Whether node features should include a residual-under-initial-guess
  signal (shifts the model from "pure topological consistency" toward a
  hybrid geometric+topological model) — flagged in `implementation_details.md`,
  not yet decided.
- PyPose's backprop mechanism (full unrolling vs. implicit differentiation)
  — affects memory feasibility if/when the trajectory-loss extension is
  attempted; needs checking against PyPose internals directly when that
  phase starts.

## 10. References

### From the original proposal

1. G. Grisetti, R. Kümmerle, C. Stachniss, W. Burgard, "A tutorial on
   graph-based SLAM," *IEEE Intelligent Transportation Systems Magazine*,
   2010.
2. C. Cadena et al., "Past, present, and future of SLAM: Toward the
   robust-perception age," *IEEE T-RO*, 2016.
3. R. Kümmerle, G. Grisetti, H. Strasdat, K. Konolige, W. Burgard, "g2o: A
   general framework for graph optimization," *ICRA*, 2011.
4. N. Sünderhauf, P. Protzel, "Switchable constraints for robust pose graph
   SLAM," *IROS*, 2012.
5. P. Agarwal et al., "Robust map optimization using dynamic covariance
   scaling," *ICRA*, 2013.
6. H. Yang, P. Antonante, V. Tzoumas, L. Carlone, "Graduated non-convexity
   for robust spatial perception," *IEEE RA-L*, 2020.
   ([arXiv:1909.08605](https://arxiv.org/abs/1909.08605))
7. P. Purkait, T.-J. Chin, I. Reid, "NeuRoRA: Neural robust rotation
   averaging," *ECCV*, 2020.
8. N. Kourtzanidis, S. Saeedi, "RL-PGO: Reinforcement learning-based planar
   pose-graph optimization," *IEEE CSL*, 2023.
9. L. Carlone, R. Tron, K. Daniilidis, F. Dellaert, "Initialization
   techniques for 3D SLAM," *ICRA*, 2015.
10. M. Fey, J. E. Lenssen, "Fast graph representation learning with PyTorch
    Geometric," *ICLR Workshop*, 2019.

### Literature review additions (July 2026)

- Sai Krishna Ghanta, Ramviyas Parasuraman, "Policies over Poses: RL-based
  Distributed Pose-Graph Optimization for Multi-Robot SLAM," 2025.
  [arXiv:2510.22740](https://arxiv.org/abs/2510.22740)
- Jaeho Shin, Maani Ghaffari, Yulun Tian, "Learning Adaptive Solvers for
  Distributed Factor Graph Optimization on Matrix Lie Groups (DeepCORD),"
  2026. [arXiv:2607.08735](https://arxiv.org/abs/2607.08735)
- Nicole Damblon, Marc Pollefeys, Dániel Baráth, "Learning to Filter Outlier
  Edges in Global SfM," *CVPR*, 2025.
  [PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Damblon_Learning_to_Filter_Outlier_Edges_in_Global_SfM_CVPR_2025_paper.pdf)
- B. Yi, M. A. Lee, A. Kloss, R. Martín-Martín, J. Bohg, "Differentiable
  Factor Graph Optimization for Learning Smoothers," *IROS*, 2021.
  [arXiv:2105.08257](https://arxiv.org/abs/2105.08257)
- Tong Wei, Giorgos Tolias, Jiří Matas, Daniel Barath, "Global-Aware Edge
  Prioritization for Pose Graph Initialization," 2026.
  [arXiv:2602.21963](https://arxiv.org/abs/2602.21963)
- Giovanni Cioffi, Davide Scaramuzza, "Why does Deep Learning Improve Visual
  SLAM?," 2026. [arXiv:2607.06023](https://arxiv.org/abs/2607.06023)
- Nicole Sandra-Yaffa Dumont, P. Michael Furlong, Jeff Orchard, Chris
  Eliasmith, "Exploiting semantic information in a spiking neural SLAM
  system (SSP-SLAM)," *Frontiers in Neuroscience*, 2023.
  [DOI](https://doi.org/10.3389/fnins.2023.1190515)
- Shay Snyder, Andrew Capodieci, David Gorsich, Maryam Parsa, "Brain
  Inspired Probabilistic Occupancy Grid Mapping with Vector Symbolic
  Architectures (VSA-OGM)," *npj Unconventional Computing*, 2026.
  [arXiv:2408.09066](https://arxiv.org/abs/2408.09066)
- "HyperSpace: A Generalized Framework for Spatial Encoding in
  Hyperdimensional Representations," 2026.
  [arXiv:2604.15113](https://arxiv.org/html/2604.15113v1)
- H. Wang, B. Sun, J. Xing, F. Yang, M. Hutter, D. Shah, D. Scaramuzza, M.
  Pollefeys, "What Matters in RL-Based Methods for Object-Goal Navigation?,"
  *ECCV*, 2026. [project page](https://honwang0054.github.io/What-matters-in-RL-ObjNav-web/)
  (methodological precedent for rigorous ablation-style contributions —
  shared by the user, not independently retrieved via search)
- Y. Ren, Z. Zhu, J. Xing, D. Scaramuzza, "Learning Agile Quadrotor Flight
  in the Real World," *RSS*, 2026.
  [project page](https://rpg.ifi.uzh.ch/lafr/) (continual/online-adaptation
  precedent; considered as a pivot target and deprioritized on resource
  grounds — see §5)
- A. Romero, A. Shenai, I. Geles, E. Aljalbout, D. Scaramuzza, "Dream to
  Fly: Model-Based Reinforcement Learning for Vision-Based Drone Flight,"
  *ICRA*, 2026. (shared by the user, not independently retrieved via
  search; no public preprint link identified in the provided document)
