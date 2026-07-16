from __future__ import annotations
from abc import ABC, abstractmethod
import torch
from edgegate.data.graph_builder import PoseGraph


def scale_information(info: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Scale information matrix by w² — shared by all solver adapters."""
    return info * w.unsqueeze(-1) ** 2


class Solver(ABC):
    @abstractmethod
    def solve(
        self,
        graph: PoseGraph,
        edge_weights: torch.Tensor,
        max_iterations: int | None = None,
    ) -> tuple[torch.Tensor, bool, int, float]:
        """Returns (poses, converged, num_iterations, final_cost)."""
