from __future__ import annotations
import torch


def edge_f1(confidence: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> dict:
    """Returns dict with precision, recall, f1."""
    raise NotImplementedError
