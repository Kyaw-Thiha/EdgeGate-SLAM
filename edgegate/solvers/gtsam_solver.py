from __future__ import annotations
import torch
from edgegate.data.graph_builder import PoseGraph
from edgegate.solvers.base import Solver


class GTSAMSolver(Solver):
    def __init__(self, kernel: str = "gnc"):
        # kernel: "gnc" | "dcs" | "switchable" | "none"
        self.kernel = kernel

    def solve(
        self,
        graph: PoseGraph,
        edge_weights: torch.Tensor,
        max_iterations: int | None = None,
    ) -> tuple[torch.Tensor, bool, int, float]:
        raise NotImplementedError
