from __future__ import annotations
import torch


def rotation_error(poses_est: torch.Tensor, poses_gt: torch.Tensor) -> float:
    """Mean absolute angular error (radians) after Umeyama SE(2) alignment.

    Uses the same Umeyama rigid-body alignment as ate_rmse() — computes the
    rotation matrix R from the SVD-based cross-covariance, applies it to align
    the estimated trajectory to ground truth, then measures the per-pose
    angular discrepancy.

    Args:
        poses_est: Estimated poses, shape (N, 3), columns [x, y, θ].
        poses_gt:  Ground-truth poses, shape (N, 3), columns [x, y, θ].

    Returns:
        Mean absolute angular error in radians.
    """
    P = poses_est[:, :2].double()
    Q = poses_gt[:, :2].double()
    theta_est = poses_est[:, 2].double()
    theta_gt = poses_gt[:, 2].double()

    mu_P = P.mean(dim=0)
    mu_Q = Q.mean(dim=0)
    P_c = P - mu_P
    Q_c = Q - mu_Q

    H = P_c.T @ Q_c
    U, _S, Vt = torch.linalg.svd(H)
    V = Vt.T
    d = torch.det(V @ U.T)
    D = torch.diag(torch.tensor([1.0, d.item()], dtype=torch.float64, device=P.device))
    R = V @ D @ U.T

    rot_angle = torch.atan2(R[1, 0], R[0, 0])
    theta_aligned = theta_est - rot_angle

    abs_error = torch.abs(theta_aligned - theta_gt)
    return float(abs_error.mean())
