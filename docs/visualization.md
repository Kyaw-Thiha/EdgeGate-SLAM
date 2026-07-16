# EdgeGate SLAM — Visualization

Visualization runs post-hoc via `scripts/demo_rerun.py`, which reads artifacts written by `train.py` and `evaluate.py` and renders them in the [Rerun](https://rerun.io) viewer. Training never calls Rerun directly — any past run can be replayed on demand.

All commands use `pixi run demo-rerun -- <args>`.

---

## Prerequisites

```bash
# Install the full stack (includes rerun-sdk >= 0.18)
pixi install

# (Optional) Download the Intel raw Carmen log for laser scan overlay
pixi run download-benchmarks -- -e    # saves data/raw/intel.log.gz
```

---

## Mode 1 — Live demo (pose-graph confidence heatmap)

Generates a fresh synthetic graph and shows edges coloured by GNN confidence: red = likely outlier, green = likely inlier.

```bash
# Uniform confidence baseline — no trained model needed
pixi run demo-rerun -- --live

# With a trained GNN checkpoint
pixi run demo-rerun -- --live --checkpoint outputs/<date>/<time>/model_best.pt

# Custom graph parameters
pixi run demo-rerun -- --live --num-poses 80 --outlier-rate 50 --seed 7
```

---

## Mode 2 — Replay a training run (epoch timeline)

Loads `graph_info.json`, `metrics.json`, and all `checkpoints/epoch_NNN/` artifacts from a Hydra output directory. The Rerun viewer's epoch timeline lets you scrub through training to watch edge confidence evolve.

```bash
pixi run demo-rerun -- --replay outputs/<date>/<time>/
```

The viewer shows:
- **Epoch timeline** — scrub to any saved checkpoint
- **Edges** — coloured by GNN confidence at that epoch
- **Scalar plots** — `train_loss`, `val_f1`, `val_ate` over epochs

Checkpoints are saved every `train.checkpoint_every` epochs (default: 5). More checkpoints = smoother scrubbing.

---

## Mode 3 — Eval comparison (multi-method overlay)

Overlays multiple evaluation methods on the same test graph — the paper-figure visualization showing where GNN confidence diverges from classical baselines.

**Step 1:** Run `evaluate.py` for each method. Each run writes `per_graph/graph_NNN/poses.npy` and `confidence.npy` to its Hydra output dir.

```bash
pixi run evaluate -- eval_method=learned eval_mode.model_dir=outputs/<date>/<time>
pixi run evaluate -- eval_method=gnc
pixi run evaluate -- eval_method=uniform
```

**Step 2:** Compare them:

```bash
pixi run demo-rerun -- \
  --compare learned=outputs/eval/learned gnc=outputs/eval/gnc uniform=outputs/eval/uniform \
  --graph-idx 0
```

`--graph-idx` selects which test graph to display (0-indexed, default: 0).

---

## Laser scan overlay (Intel dataset)

Adds a static grey point cloud by reprojecting raw Carmen FLASER readings onto the optimised poses. Works with `--live`, `--replay`, and `--compare`.

```bash
# Live
pixi run demo-rerun -- --live --laser data/raw/intel.log.gz

# Replay
pixi run demo-rerun -- --replay outputs/<date>/<time>/ --laser data/raw/intel.log.gz

# Compare
pixi run demo-rerun -- \
  --compare learned=outputs/eval/learned gnc=outputs/eval/gnc \
  --laser data/raw/intel.log.gz
```

Only the Intel dataset has a matching Carmen log. The scan index is aligned to pose index (scan `i` → pose `i`), which is valid for the Intel log's sequential structure.

---

## Saving and sharing

Add `--save-rrd <path>` to any mode to save the recording as a `.rrd` file. The file can be opened later with `rerun` or shared with collaborators — no active recording needed.

```bash
pixi run demo-rerun -- --replay outputs/<date>/<time>/ --save-rrd results/training.rrd
pixi run demo-rerun -- --compare ... --save-rrd results/comparison.rrd

# Open a saved recording
rerun results/comparison.rrd
```

---

## What gets logged

| Entity | Content |
|--------|---------|
| `trajectory/initial` | Node initial poses (light blue) |
| `trajectory/solved` | Solver output poses (white) |
| `trajectory/gt` | Ground-truth poses, if available (green) |
| `edges/odometry` | Odometry edges (grey) |
| `edges/loop_closure` | LC edges coloured by confidence — red→green |
| `edges/gt_inliers` | True inlier LC edges, if labels available (bright green) |
| `scans/laser` | Carmen laser points in world frame (grey, static) |
| `metrics/train_loss` | Training loss over epochs |
| `metrics/val_f1` | Validation F1 over epochs |
| `metrics/val_ate` | Validation ATE over epochs |
| `methods/<name>/trajectory` | Per-method solved poses (compare mode) |
| `methods/<name>/edges` | Per-method LC edge confidence (compare mode) |
