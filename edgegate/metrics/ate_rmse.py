from __future__ import annotations
import torch


def ate_rmse(poses_est: torch.Tensor, poses_gt: torch.Tensor) -> float:
    """Absolute Trajectory Error RMSE after optimal alignment."""
    raise NotImplementedError
