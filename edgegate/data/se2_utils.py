from __future__ import annotations
import numpy as np


def angle_wrap(theta: float | np.ndarray) -> float | np.ndarray:
    return (theta + np.pi) % (2 * np.pi) - np.pi


def compose(pose_i: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """T_i · delta → next global pose.

    Apply delta (expressed in pose_i's local frame) to get the next world pose.
    pose_i: (3,) [x, y, θ]
    delta:  (3,) [dx, dy, dθ] in pose_i's frame
    """
    ci, si = np.cos(pose_i[2]), np.sin(pose_i[2])
    return np.array([
        pose_i[0] + ci * delta[0] - si * delta[1],
        pose_i[1] + si * delta[0] + ci * delta[1],
        angle_wrap(pose_i[2] + delta[2]),
    ])


def inverse_compose(pose_i: np.ndarray, pose_j: np.ndarray) -> np.ndarray:
    """T_i^{-1} · T_j → relative transform from i to j in i's local frame.

    Returns (3,) [dx, dy, dθ]: the measurement an edge from i to j would carry.
    Not naive coordinate subtraction — displaces into i's local frame via rotation.
    """
    dx_w = pose_j[0] - pose_i[0]
    dy_w = pose_j[1] - pose_i[1]
    ci, si = np.cos(pose_i[2]), np.sin(pose_i[2])
    return np.array([
        ci * dx_w + si * dy_w,
        -si * dx_w + ci * dy_w,
        angle_wrap(pose_j[2] - pose_i[2]),
    ])
