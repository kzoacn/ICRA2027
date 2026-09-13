"""Public Panda geometry for selecting equivalent Cartesian grasp frames.

Link transforms and joint limits are transcribed from robosuite's public Panda
robot.xml. The reported TCP is grip_site (97 mm past right_hand), while its
orientation is right_hand. No simulator or scene asset is read here. The world
base transform is recovered from the current joint and TCP proprioception.
This bounded local IK check is a reachability heuristic, not collision checking.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


JOINT_LIMITS = np.array([
    [-2.8973, 2.8973], [-1.7628, 1.7628], [-2.8973, 2.8973],
    [-3.0718, -0.0698], [-2.8973, 2.8973], [-0.0175, 3.7525],
    [-2.8973, 2.8973],
])
_POSITIONS = np.array([
    [0, 0, .333], [0, 0, 0], [0, -.316, 0], [.0825, 0, 0],
    [-.0825, .384, 0], [0, 0, 0], [.088, 0, 0],
])
_FIXED_ROTATIONS = Rotation.from_rotvec(
    np.array([0, -1, 1, 1, -1, 1, 1])[:, None]
    * np.array([np.pi / 2, 0, 0])[None, :]
).as_matrix()
_HAND_ROTATION = Rotation.from_quat([0, 0, -.383, .924]).as_matrix()


def panda_tcp_pose(joints: np.ndarray) -> np.ndarray:
    """TCP pose in the public robot base frame; metres and radians."""
    q = np.asarray(joints, dtype=np.float64)
    if q.shape != (7,) or not np.all(np.isfinite(q)):
        raise ValueError("Panda kinematics needs seven finite joint positions")
    rotation = np.eye(3)
    position = np.zeros(3)
    for offset, fixed, angle in zip(_POSITIONS, _FIXED_ROTATIONS, q):
        position += rotation @ offset
        c, s = np.cos(angle), np.sin(angle)
        rotation = rotation @ fixed @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    position += rotation[:, 2] * .1065
    rotation = rotation @ _HAND_ROTATION
    position += rotation[:, 2] * .097
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotation, position
    return pose


def path_joint_margin(joints, world_tcp, positions, rotation):
    """Minimum joint-limit clearance over a local IK path, or -inf on failure."""
    q = np.asarray(joints, dtype=np.float64)
    if (q.shape != (7,) or not np.all(np.isfinite(q))
            or np.any(q < JOINT_LIMITS[:, 0]) or np.any(q > JOINT_LIMITS[:, 1])):
        return -np.inf
    world_base = np.asarray(world_tcp) @ np.linalg.inv(panda_tcp_pose(q))
    lower, upper = JOINT_LIMITS[:, 0] + .001, JOINT_LIMITS[:, 1] - .001
    seed = np.clip(q, lower, upper)
    margin = np.inf
    for position in positions:
        def residual(candidate):
            pose = world_base @ panda_tcp_pose(candidate)
            return np.r_[5 * (pose[:3, 3] - position),
                         Rotation.from_matrix(rotation @ pose[:3, :3].T).as_rotvec()]

        solution = least_squares(residual, seed, bounds=(lower, upper), max_nfev=160)
        error = residual(solution.x)
        if np.linalg.norm(error[:3]) > .025 or np.linalg.norm(error[3:]) > .06:
            return -np.inf
        seed = solution.x
        margin = min(margin, float(np.min(np.minimum(seed-JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1]-seed))))
    return margin


def reachable_equivalent_frame(joints, world_tcp, positions, preferred, alternate):
    """Keep the preferred roll unless only its equivalent passes local IK.

    Every waypoint must fit within 5 mm / .06 rad. These are geometric residual
    thresholds, independent of task identity or environment outcomes.
    """
    if path_joint_margin(joints, world_tcp, positions, preferred) >= 0:
        return preferred.copy()
    if path_joint_margin(joints, world_tcp, positions, alternate) >= 0:
        return alternate.copy()
    return preferred.copy()
