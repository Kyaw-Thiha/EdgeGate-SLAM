from __future__ import annotations
import numpy as np
import torch
import gtsam
from edgegate.data.types import PoseGraph
from edgegate.solvers.base import Solver


def _upper_tri_to_full(ut: np.ndarray) -> np.ndarray:
    """(6,) upper-tri edge_info → (3, 3) symmetric information matrix."""
    return np.array([
        [ut[0], ut[1], ut[2]],
        [ut[1], ut[3], ut[4]],
        [ut[2], ut[4], ut[5]],
    ])


def _scale_info_np(edge_info: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Scale information matrices by w²: Λ_scaled = Λ * w²."""
    return edge_info * (weights ** 2)[:, np.newaxis]


class GTSAMSolver(Solver):
    """SE(2) pose-graph optimizer backed by GTSAM.

    kernel="none": plain LM with information matrices scaled by edge_weights².
    kernel="gnc":  GTSAM's built-in GNC (Graduated Non-Convexity).
                   GNC computes its own robust weights internally — do NOT pass
                   GNN-predicted edge_weights (would double-robustify and confound
                   any GNN-vs-classical comparison). Asserted at runtime.
    kernel="dcs":  Dynamic Covariance Scaling (Agarwal et al., ICRA 2013).
                   Wraps each loop-closure factor's noise model with a DCS robust
                   kernel, then runs plain LM. DCS computes its own scaling
                   internally — same non-unit-weights guard as GNC.
    """

    _PRIOR_SIGMA = 1e-6  # tight prior on pose 0 to fix gauge freedom

    def __init__(self, kernel: str = "none", dcs_param: float = 1.0) -> None:
        assert kernel in ("none", "gnc", "dcs"), (
            f"kernel must be 'none', 'gnc', or 'dcs', got {kernel!r}"
        )
        self.kernel = kernel
        self.dcs_param = dcs_param

    def solve(
        self,
        graph: PoseGraph,
        edge_weights: torch.Tensor,
        max_iterations: int | None = None,
    ) -> tuple[torch.Tensor, bool, int, float]:
        if self.kernel in ("gnc", "dcs"):
            assert torch.allclose(edge_weights, torch.ones_like(edge_weights)), (
                f"kernel='{self.kernel}' computes its own weights internally; "
                "do not pass GNN-predicted edge_weights (would double-robustify)."
            )

        max_iter = max_iterations if max_iterations is not None else 100

        weights_np = edge_weights.detach().cpu().numpy()
        info_scaled = _scale_info_np(graph.edge_info, weights_np)   # (E, 6)

        N = graph.node_init.shape[0]
        E = graph.edge_index.shape[1]

        # ── Factor graph ──────────────────────────────────────────────────────
        fg = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()

        for i in range(N):
            x, y, theta = graph.node_init[i]
            initial.insert(i, gtsam.Pose2(float(x), float(y), float(theta)))

        p0 = graph.node_init[0]
        prior_noise = gtsam.noiseModel.Isotropic.Sigma(3, self._PRIOR_SIGMA)
        fg.add(gtsam.PriorFactorPose2(
            0, gtsam.Pose2(float(p0[0]), float(p0[1]), float(p0[2])), prior_noise
        ))

        for e in range(E):
            i = int(graph.edge_index[0, e])
            j = int(graph.edge_index[1, e])
            dx, dy, dtheta = graph.edge_measurement[e]
            info_mat = _upper_tri_to_full(info_scaled[e])
            noise = gtsam.noiseModel.Gaussian.Information(info_mat)
            if self.kernel == "dcs" and graph.edge_type[e] == 1:
                noise = gtsam.noiseModel.Robust.Create(
                    gtsam.noiseModel.mEstimator.DCS.Create(self.dcs_param),
                    noise,
                )
            fg.add(gtsam.BetweenFactorPose2(
                i, j, gtsam.Pose2(float(dx), float(dy), float(dtheta)), noise
            ))

        # ── Optimize ──────────────────────────────────────────────────────────
        if self.kernel in ("none", "dcs"):
            params = gtsam.LevenbergMarquardtParams()
            params.setMaxIterations(max_iter)
            optimizer = gtsam.LevenbergMarquardtOptimizer(fg, initial, params)
            result = optimizer.optimize()
            num_iterations = int(optimizer.iterations())
            final_cost = float(fg.error(result))
            converged = num_iterations < max_iter
        else:  # gnc
            params = gtsam.GncLMParams()
            params.setMaxIterations(max_iter)
            optimizer = gtsam.GncLMOptimizer(fg, initial, params)
            result = optimizer.optimize()
            num_iterations = -1   # GNC doesn't expose iteration count
            final_cost = float(fg.error(result))
            converged = True      # GNC always produces a result

        # ── Extract poses as torch.Tensor ─────────────────────────────────────
        poses = torch.zeros(N, 3)
        for i in range(N):
            p = result.atPose2(i)
            poses[i] = torch.tensor([p.x(), p.y(), p.theta()])

        return poses, converged, num_iterations, final_cost
