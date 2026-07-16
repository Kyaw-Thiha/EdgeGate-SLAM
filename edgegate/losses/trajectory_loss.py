from __future__ import annotations
import torch
import torch.nn as nn
from edgegate.solvers.base import Solver


class TrajectoryLoss(nn.Module):
    """Trajectory error backpropped through the differentiable solver."""

    def __init__(self, solver: Solver, train_iterations: int):
        super().__init__()
        self.solver = solver
        self.train_iterations = train_iterations

    def forward(self, graph, confidence: torch.Tensor, gt_poses: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
