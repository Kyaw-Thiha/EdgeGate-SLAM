from __future__ import annotations

import json
import os
import random

import numpy as np
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch_geometric.loader import DataLoader as PyGDataLoader

from edgegate.data.graph_builder import to_pyg
from edgegate.data.synthetic_generator import generate
from edgegate.losses.edge_bce import EdgeBCELoss
from edgegate.losses.trajectory_loss import TrajectoryLoss
from edgegate.metrics.ate_rmse import ate_rmse
from edgegate.solvers.base import Solver


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _init_wandb(cfg: DictConfig) -> None:
    group = f"{cfg.train.loss_mode}_synth_{cfg.data.outlier_structure}"
    tags = [
        f"olr:{cfg.data.outlier_rate}",
        f"structure:{cfg.data.outlier_structure}",
        f"loss:{cfg.train.loss_mode}",
        f"seed:{cfg.train.seed}",
    ]
    wandb.init(
        project=cfg.logging.project,
        entity=cfg.logging.entity,
        group=group,
        tags=tags,
        config=OmegaConf.to_container(cfg, resolve=True),
    )


def _generate_data(cfg: DictConfig) -> tuple[list, list]:
    num_graphs = cfg.train.get("num_graphs", 100)
    num_val = max(1, int(num_graphs * cfg.train.val_split))
    num_train = num_graphs - num_val

    data_kwargs = {
        "num_poses": cfg.data.num_poses,
        "num_loop_closures": cfg.data.num_loop_closures,
        "outlier_rate": cfg.data.outlier_rate,
        "outlier_structure": cfg.data.outlier_structure,
        "proximity_threshold": cfg.data.get("proximity_threshold", 2.0),
    }

    graphs = []
    for i in range(num_graphs):
        g = generate(**data_kwargs, seed=cfg.data.seed + i)
        graphs.append(g)

    rng = np.random.default_rng(cfg.train.seed + 1)
    indices = rng.permutation(num_graphs)
    train_idx = indices[:num_train]
    val_idx = indices[num_train:]

    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    return train_graphs, val_graphs


def _build_losses(cfg: DictConfig, solver: Solver) -> dict:
    mode = cfg.train.loss_mode
    losses: dict = {}
    if mode in ("bce", "combined"):
        losses["bce"] = EdgeBCELoss()
    if mode in ("trajectory", "combined"):
        losses["trajectory"] = TrajectoryLoss(solver, cfg.train.solver_train_iterations)
    if not losses:
        raise ValueError(f"Unknown loss_mode: {mode}")
    return losses


def _train_epoch_bce(model, bce_loss, optimizer, train_graphs, cfg):
    model.train()
    pyg_graphs = [to_pyg(g) for g in train_graphs]
    loader = PyGDataLoader(pyg_graphs, batch_size=cfg.train.batch_size, shuffle=True)

    total_loss = 0.0
    n_graphs = 0
    for batch in loader:
        optimizer.zero_grad()
        conf = model(batch)
        loss = bce_loss(conf, batch.edge_label, batch.edge_type)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        n_graphs += batch.num_graphs
    return total_loss / max(n_graphs, 1)


def _train_epoch_per_graph(model, losses, optimizer, train_graphs, cfg):
    model.train()
    total_loss = 0.0
    weight = cfg.train.trajectory_loss_weight

    for graph in train_graphs:
        optimizer.zero_grad()
        data = to_pyg(graph)
        conf = model(data)

        components = []
        if "bce" in losses:
            components.append(losses["bce"](conf, data.edge_label, data.edge_type))
        if "trajectory" in losses:
            gt = torch.from_numpy(graph.gt_node_poses).float()
            components.append(weight * losses["trajectory"](graph, conf, gt))

        loss = sum(components)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(train_graphs), 1)


def _train_epoch(model, losses, optimizer, train_graphs, cfg):
    if cfg.train.loss_mode == "bce":
        return _train_epoch_bce(model, losses["bce"], optimizer, train_graphs, cfg)
    return _train_epoch_per_graph(model, losses, optimizer, train_graphs, cfg)


def _validate(model, solver, val_graphs, cfg):
    model.eval()
    tp = fp = fn = 0
    ate_sum = 0.0
    n_ate = 0

    with torch.no_grad():
        for graph in val_graphs:
            data = to_pyg(graph)
            conf = model(data)

            lc_mask = data.edge_type == 1
            pred = conf[lc_mask] >= 0.5
            label = data.edge_label[lc_mask] >= 0.5
            tp += int((pred & label).sum().item())
            fp += int((pred & ~label).sum().item())
            fn += int((~pred & label).sum().item())

            if graph.gt_node_poses is not None:
                gt = torch.from_numpy(graph.gt_node_poses).float()
                poses, converged, iters, cost = solver.solve(
                    graph, conf, max_iterations=None
                )
                ate_val = ate_rmse(poses, gt)
                ate_sum += ate_val
                n_ate += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    metrics = {"val_f1": f1, "tp": tp, "fp": fp, "fn": fn}
    if n_ate > 0:
        metrics["val_ate"] = ate_sum / n_ate
    return metrics


def _save_graph_info(viz_graph) -> None:
    info = {
        "edge_index": viz_graph.edge_index.tolist(),
        "edge_type": viz_graph.edge_type.tolist(),
        "edge_label": (
            viz_graph.edge_label.tolist()
            if viz_graph.edge_label is not None
            else None
        ),
        "gt_node_poses": (
            viz_graph.gt_node_poses.tolist()
            if viz_graph.gt_node_poses is not None
            else None
        ),
        "node_init": viz_graph.node_init.tolist(),
    }
    with open("graph_info.json", "w") as f:
        json.dump(info, f)


def _save_checkpoint(model, solver, viz_graph, epoch) -> None:
    ckpt_dir = f"checkpoints/epoch_{epoch:03d}"
    os.makedirs(ckpt_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        data = to_pyg(viz_graph)
        conf = model(data)
        poses, converged, iters, cost = solver.solve(
            viz_graph, conf, max_iterations=None
        )

    np.save(os.path.join(ckpt_dir, "poses.npy"), poses.cpu().numpy())
    np.save(os.path.join(ckpt_dir, "edge_weights.npy"), conf.cpu().numpy())


def train(cfg: DictConfig) -> None:
    _set_seed(cfg.train.seed)

    train_graphs, val_graphs = _generate_data(cfg)

    model = instantiate(cfg.model)
    solver = instantiate(cfg.solver)
    losses = _build_losses(cfg, solver)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    _init_wandb(cfg)

    viz_graph = val_graphs[0]
    _save_graph_info(viz_graph)

    best_f1 = -1.0
    metrics_log: list[dict] = []

    for epoch in range(1, cfg.train.epochs + 1):
        train_loss = _train_epoch(model, losses, optimizer, train_graphs, cfg)
        val_metrics = _validate(model, solver, val_graphs, cfg)

        entry = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        metrics_log.append(entry)
        wandb.log(entry, step=epoch)

        if epoch % cfg.train.checkpoint_every == 0:
            _save_checkpoint(model, solver, viz_graph, epoch)

        f1 = val_metrics.get("val_f1", -1.0)
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "model_best.pt")

    torch.save(model.state_dict(), "model_last.pt")
    with open("metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=2)
    wandb.finish()
