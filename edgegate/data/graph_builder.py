from __future__ import annotations
import torch
from torch_geometric.data import Data
from edgegate.data.types import PoseGraph

# edge_info upper-tri ordering: [Ixx, Ixy, Ixθ, Iyy, Iyθ, Iθθ]
# Diagonal (Ixx, Iyy, Iθθ) lives at indices 0, 3, 5.
_INFO_DIAG = [0, 3, 5]


def to_pyg(graph: PoseGraph) -> Data:
    """Convert a PoseGraph to a PyG Data object for GNN input.

    Attributes on the returned Data:
        x          (N, 3)  node features: initial pose guess [x, y, θ]
        edge_index (2, E)
        edge_attr  (E, 6)  [dx, dy, dθ, Ixx, Iyy, Iθθ]
        edge_type  (E,)    0=odometry, 1=loop-closure  (separate from edge_attr
                           so EdgeTypeAwareConv can use type-specific projections
                           without slicing it back out)
        edge_label (E,)    ground-truth inlier flag; only present for synthetic graphs
    """
    x = torch.from_numpy(graph.node_init).float()
    edge_index = torch.from_numpy(graph.edge_index).long()

    meas = torch.from_numpy(graph.edge_measurement).float()            # (E, 3)
    info_diag = torch.from_numpy(graph.edge_info[:, _INFO_DIAG]).float()  # (E, 3)
    edge_attr = torch.cat([meas, info_diag], dim=1)                    # (E, 6)

    edge_type = torch.from_numpy(graph.edge_type).long()

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_type=edge_type)
    if graph.edge_label is not None:
        data.edge_label = torch.from_numpy(graph.edge_label).float()
    return data
