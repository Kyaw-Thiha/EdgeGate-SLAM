from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeBCELoss(nn.Module):
    """Binary cross-entropy loss for loop-closure inlier/outlier classification.

    Computes BCE only over loop-closure edges (edge_type == 1). Odometry edges
    are excluded entirely — the synthetic generator never corrupts an odometry
    edge, so they carry no discriminative signal and including them (weighted or
    not) would dilute the loss with trivially-easy background examples.

    This exclusion is intentionally consistent end-to-end: EdgeGateGNN also
    hardcodes w_odom = 1.0 and never runs the confidence head on odometry edges.
    Loss and inference therefore tell the same story at both train and eval time
    (see implementation_details.md §edge_bce and GNN.md §2.2).

    Args:
        reduction: 'mean' (default) or 'sum', passed to F.binary_cross_entropy.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        assert reduction in ("mean", "sum"), (
            f"reduction must be 'mean' or 'sum', got {reduction!r}"
        )
        self.reduction = reduction

    def forward(
        self,
        confidence: torch.Tensor,
        labels: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """Compute BCE loss over loop-closure edges.

        Args:
            confidence: Per-edge confidence scores in [0, 1], shape (E,).
                        Produced by EdgeGateGNN.forward() — odometry positions
                        are hardcoded to 1.0 by the GNN, but this loss masks
                        them out regardless for consistency and safety.
            labels:     Ground-truth inlier labels in {0.0, 1.0}, shape (E,).
                        Typically data.edge_label from a synthetic PoseGraph.
            edge_type:  Integer edge types, shape (E,).
                        0 = odometry (excluded), 1 = loop-closure (loss target).

        Returns:
            Scalar BCE loss over loop-closure edges only.

        Raises:
            ValueError: If no loop-closure edges are present — a graph with no
                        LCs is a data-pipeline bug, not a recoverable condition.
        """
        lc_mask = edge_type == 1
        if not lc_mask.any():
            raise ValueError(
                "EdgeBCELoss: no loop-closure edges (edge_type == 1) found. "
                "Ensure each graph in the batch contains at least one LC edge."
            )

        lc_conf   = confidence[lc_mask]   # (E_lc,) ∈ [0, 1]
        lc_labels = labels[lc_mask]       # (E_lc,) ∈ {0.0, 1.0}

        return F.binary_cross_entropy(lc_conf, lc_labels, reduction=self.reduction)
