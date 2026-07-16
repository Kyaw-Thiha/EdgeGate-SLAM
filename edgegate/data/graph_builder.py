from __future__ import annotations
from torch_geometric.data import Data
from edgegate.data.types import PoseGraph


def to_pyg(graph: PoseGraph) -> Data:
    """Convert a PoseGraph to a PyG Data object for GNN input."""
    raise NotImplementedError
