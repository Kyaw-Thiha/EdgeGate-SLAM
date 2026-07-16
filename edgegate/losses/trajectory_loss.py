from __future__ import annotations
import torch
import torch.nn as nn
from edgegate.solvers.base import Solver


class TrajectoryLoss(nn.Module):
    """Trajectory position error backpropagated through a differentiable solver.

    Runs the solver for a fixed K iterations (NOT to convergence) and measures
    the mean squared Euclidean distance on (x, y) positions between the K-step
    optimised trajectory and the ground-truth noise-free trajectory.

    Design decisions (see implementation_details.md §trajectory_loss):

    Position-only MSE, not full SE(2) error:
        ATE is also a position metric, so training and eval measure the same
        quantity. Avoiding the angle residual keeps the gradient path well-
        conditioned early in training when poses are far from ground truth —
        angle-wrapping discontinuities are more likely to produce unstable
        gradients during the K-sweep than a clean quadratic position term.

    Fixed K iterations, not convergence:
        Bounding compute/memory per step is essential because PyPoseSolver
        backpropagates via full unrolling (modjac at each step) — memory scales
        linearly with K. K is a first-class ablation axis exposed via Hydra as
        train.solver_train_iterations. See implementation_details.md for sweep
        guidance.

    Synthetic-only:
        PoseGraph.gt_node_poses is only populated by synthetic_generator.py;
        it is None for real .g2o data by construction. Real data is eval-only
        and never enters a training batch, so this constraint is structural.

    Args:
        solver:           Solver instance. Must be PyPoseSolver for
                          backpropagation — only PyPoseSolver keeps the solve
                          as a differentiable torch computation graph.
        train_iterations: Number of solver iterations K per training step.
    """

    def __init__(self, solver: Solver, train_iterations: int) -> None:
        super().__init__()
        self.solver = solver
        self.train_iterations = train_iterations

    def forward(
        self,
        graph,
        confidence: torch.Tensor,
        gt_poses: torch.Tensor,
    ) -> torch.Tensor:
        """Run K-step solver and compute position MSE against ground truth.

        Args:
            graph:      PoseGraph (single, unbatched). gt_node_poses must be
                        set — do not call this loss on real .g2o PoseGraphs.
            confidence: Per-edge confidence scores from EdgeGateGNN.forward(),
                        shape (E,), values in [0, 1]. Odometry edges should
                        carry value 1.0 (GNN already hardcodes this). Gradients
                        flow from the loss back through the unrolled solver
                        steps to confidence, and from there into the GNN.
            gt_poses:   Ground-truth node poses, shape (N, 3), columns [x,y,θ].
                        Typically torch.from_numpy(graph.gt_node_poses).float().

        Returns:
            Scalar mean squared Euclidean distance on (x, y) positions,
            differentiable w.r.t. confidence through the unrolled K-step solve.
        """
        poses_est, _, _, _ = self.solver.solve(
            graph, confidence, max_iterations=self.train_iterations
        )
        # Position-only: column indices 0 (x) and 1 (y), no angle term.
        return ((poses_est[:, :2] - gt_poses[:, :2]) ** 2).mean()
