from __future__ import annotations
import torch


def ate_rmse(poses_est: torch.Tensor, poses_gt: torch.Tensor) -> float:
    """Absolute Trajectory Error RMSE after Umeyama SE(2) rigid-body alignment.

    Aligns the estimated trajectory to the ground-truth via a SE(2) rigid-body
    transformation (rotation + translation, no scale) solved in closed form via
    SVD, then computes RMSE on (x, y) positions.

    Why Umeyama alignment and not simple zero-anchoring (subtract poses[0])?
        Zero-anchoring only holds when both trajectories share exactly the same
        gauge/origin by construction. Our solver anchors pose 0 to fix gauge
        freedom, but this is not guaranteed on real benchmark data, and PGO
        gauge freedom is exactly the failure mode Umeyama alignment exists to
        remove — two trajectories with identical internal geometry but different
        global placement should give ATE ≈ 0, and zero-anchoring does not
        guarantee that in general. Umeyama is also the field's actual convention:
        it is the alignment step used by the evo evaluation library and the TUM
        RGB-D benchmark protocol (Sturm et al. 2012).

    Note on benchmark availability:
        No current .g2o benchmark file (including M3500, Sphere2500, Intel, or
        MIT/CSAIL) has independently-verifiable ground truth — all "GT" in the
        PGO literature is pseudo-GT: the optimized trajectory of the
        outlier-free data. For real benchmarks, scripts/evaluate.py computes
        a clean-solve reference trajectory and stores it as gt_node_poses so
        this function can report ATE on all datasets. Any ATE on real data is
        ATE-against-reference-solve, not true ground truth.

    Implementation — Umeyama (1991) for 2D positions (SE(2) rigid body):
        1. Centre both trajectories: P_c = P - μ_P,  Q_c = Q - μ_Q
        2. Cross-covariance:  H = P_c^T Q_c  (2×2)
        3. SVD:  H = U S V^T
        4. Correct for reflection:  D = diag(1, det(V U^T))
        5. Rotation:  R = V D U^T  (det(R) = +1)
        6. Translation:  t = μ_Q − R μ_P
        7. RMSE of aligned P against Q

    Args:
        poses_est: Estimated poses, shape (N, 3), columns [x, y, θ].
                   Produced by Solver.solve(), already a torch.Tensor.
        poses_gt:  Ground-truth poses, shape (N, 3), columns [x, y, θ].
                   Typically torch.from_numpy(graph.gt_node_poses).float().

    Returns:
        ATE RMSE as a Python float, in the same position units as the input
        (simulator units for synthetic data; metres for real benchmarks).
    """
    # Work in float64 throughout to avoid SVD precision loss on small poses.
    P = poses_est[:, :2].double()   # (N, 2) estimated (x, y)
    Q = poses_gt[:, :2].double()    # (N, 2) ground-truth (x, y)

    mu_P = P.mean(dim=0)            # (2,)
    mu_Q = Q.mean(dim=0)            # (2,)
    P_c  = P - mu_P                 # (N, 2) centred
    Q_c  = Q - mu_Q                 # (N, 2) centred

    # Cross-covariance matrix H = P_c^T Q_c  →  (2, 2)
    H = P_c.T @ Q_c

    # SVD:  H = U S V^T
    U, _S, Vt = torch.linalg.svd(H)
    V = Vt.T

    # Reflection correction: ensure det(R) = +1 (proper rotation, not a flip)
    d = torch.det(V @ U.T)
    D = torch.diag(torch.tensor([1.0, d.item()], dtype=torch.float64, device=P.device))
    R = V @ D @ U.T                 # (2, 2) rotation matrix

    # Translation: t = μ_Q − R μ_P
    t = mu_Q - R @ mu_P             # (2,)

    # Apply alignment: P_aligned = (R P_c^T)^T + μ_Q
    P_aligned = (R @ P_c.T).T + mu_Q   # (N, 2)

    rmse = torch.sqrt(((P_aligned - Q) ** 2).sum(dim=-1).mean())
    return float(rmse)
