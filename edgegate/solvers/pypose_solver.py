from __future__ import annotations
import math
import torch
import torch.nn as nn
import numpy as np
import pypose.optim as ppopt
from edgegate.data.types import PoseGraph
from edgegate.solvers.base import Solver, scale_information


def _angle_wrap(theta: torch.Tensor) -> torch.Tensor:
    return (theta + math.pi) % (2 * math.pi) - math.pi


class _PGOModel(nn.Module):
    """SE(2) pose graph as a residual model for PyPose's second-order optimizer.

    Residuals are pre-whitened by sqrt(scaled_info_diag) so the optimizer minimises
    the standard weighted chi² objective without needing an external weight matrix.
    Pose 0 is a fixed buffer (gauge anchor); poses 1..N-1 are nn.Parameters.
    """

    def __init__(self, graph: PoseGraph, info_sqrt: torch.Tensor) -> None:
        super().__init__()
        poses = torch.from_numpy(graph.node_init).float()
        self.register_buffer("pose0", poses[0:1].clone())           # (1, 3) fixed anchor
        self.poses = nn.Parameter(poses[1:].clone())                 # (N-1, 3) optimised
        self.register_buffer("src", torch.from_numpy(graph.edge_index[0]).long())
        self.register_buffer("dst", torch.from_numpy(graph.edge_index[1]).long())
        self.register_buffer("meas", torch.from_numpy(graph.edge_measurement).float())
        self.register_buffer("info_sqrt", info_sqrt)                 # (E, 3)

    def forward(self, _=None) -> torch.Tensor:
        poses = torch.cat([self.pose0, self.poses], dim=0)           # (N, 3)
        pi = poses[self.src]                                         # (E, 3)
        pj = poses[self.dst]                                         # (E, 3)

        dx_w = pj[:, 0] - pi[:, 0]
        dy_w = pj[:, 1] - pi[:, 1]
        ci = torch.cos(pi[:, 2])
        si = torch.sin(pi[:, 2])

        residuals = torch.stack([
            ci * dx_w + si * dy_w - self.meas[:, 0],
            -si * dx_w + ci * dy_w - self.meas[:, 1],
            _angle_wrap(pj[:, 2] - pi[:, 2] - self.meas[:, 2]),
        ], dim=1)                                                    # (E, 3)

        return (residuals * self.info_sqrt).reshape(-1)              # (E*3,)


class PyPoseSolver(Solver):
    """SE(2) pose-graph optimizer backed by PyPose's Levenberg-Marquardt solver.

    Keeps inputs and outputs as torch.Tensors throughout so the call-site can
    choose whether to wrap in torch.no_grad() (Phase 0) or let gradients flow
    through the unrolled optimisation steps (Phase 1 trajectory_loss path).
    """

    def __init__(
        self,
        damping: float = 1e-4,
        patience: int = 5,
        decreasing: float = 1e-3,
    ) -> None:
        self.damping = damping
        self.patience = patience
        self.decreasing = decreasing

    def solve(
        self,
        graph: PoseGraph,
        edge_weights: torch.Tensor,
        max_iterations: int | None = None,
    ) -> tuple[torch.Tensor, bool, int, float]:
        max_iter = max_iterations if max_iterations is not None else 100

        edge_info = torch.from_numpy(graph.edge_info).float()
        info_scaled = scale_information(edge_info, edge_weights)             # (E, 6)
        info_sqrt = info_scaled[:, [0, 3, 5]].clamp(min=0).sqrt()           # (E, 3)

        model = _PGOModel(graph, info_sqrt)
        strategy = ppopt.strategy.Constant(damping=self.damping)
        optimizer = ppopt.LM(model, strategy=strategy)
        scheduler = ppopt.scheduler.StopOnPlateau(
            optimizer,
            steps=max_iter,
            patience=self.patience,
            decreasing=self.decreasing,
        )

        while scheduler.continual():
            loss = optimizer.step(None)
            scheduler.step(loss)

        converged = int(scheduler.steps) < int(scheduler.max_steps)
        num_iterations = int(scheduler.steps)
        final_cost = float(optimizer.loss) if optimizer.loss is not None else float("inf")

        with torch.no_grad():
            poses = torch.cat([model.pose0, model.poses], dim=0).clone()

        return poses, converged, num_iterations, final_cost
