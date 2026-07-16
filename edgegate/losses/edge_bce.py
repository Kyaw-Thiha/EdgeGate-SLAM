from __future__ import annotations
import torch
import torch.nn as nn


class EdgeBCELoss(nn.Module):
    """BCE loss against synthetic ground-truth inlier labels. No solver call."""

    def forward(self, confidence: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
