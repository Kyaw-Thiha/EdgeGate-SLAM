from __future__ import annotations
from edgegate.data.graph_builder import PoseGraph


def generate(
    num_poses: int,
    num_loop_closures: int,
    outlier_rate: float,
    outlier_structure: str,  # "random" | "clustered"
    seed: int,
) -> PoseGraph:
    raise NotImplementedError
