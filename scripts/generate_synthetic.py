from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from edgegate.data.g2o_io import save_g2o
from edgegate.data.synthetic_generator import generate


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    data_cfg = cfg.data
    kwargs = {
        "num_poses": data_cfg.num_poses,
        "num_loop_closures": data_cfg.num_loop_closures,
        "outlier_rate": data_cfg.outlier_rate,
        "outlier_structure": data_cfg.outlier_structure,
        "segment_length": data_cfg.get("segment_length", 5),
        "proximity_threshold": data_cfg.proximity_threshold,
        "seed": data_cfg.seed,
    }
    graph = generate(**kwargs)

    out_dir = Path("data/synthetic")
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / "graph.g2o"
    save_g2o(graph, path)
    print(f"Saved synthetic pose graph to {path}")

    E = graph.edge_index.shape[1]
    N = graph.node_init.shape[0]
    lc = int((graph.edge_type == 1).sum())
    lc_outliers = int((graph.edge_label[graph.edge_type == 1] == 0).sum())
    print(f"  {N} nodes, {E} edges ({E - lc} odometry, {lc} loop-closures)")
    print(f"  Loop-closure outliers: {lc_outliers}/{lc} ({lc_outliers / lc * 100:.0f}%)" if lc > 0
          else "  No loop-closure edges.")

    if cfg.get("data").get("num_graphs", 1) > 1:
        for i in range(1, cfg.data.num_graphs):
            kwargs["seed"] = data_cfg.seed + i
            graph = generate(**kwargs)
            path = out_dir / f"graph_{i}.g2o"
            save_g2o(graph, path)
            print(f"Saved synthetic pose graph to {path}")


if __name__ == "__main__":
    main()
