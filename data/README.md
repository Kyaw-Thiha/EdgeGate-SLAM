# data/

`raw/` and `synthetic/` are gitignored — they are created at runtime.

## raw/

Downloaded `.g2o` benchmark datasets. **Held out entirely — never used during training or hyperparameter tuning.** Results on these are reported exactly once per model version.

Benchmarks: Intel, M3500, Sphere2500, parking-garage.

```bash
pixi run download-benchmarks
```

## synthetic/

Generated pose graphs with ground-truth inlier labels. Used for training and validation only.

```bash
pixi run gen-synthetic
```
