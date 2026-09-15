"""Small, dependency-free RGB-D geometry helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .observation import CameraFrame


def metric_depth_from_normalized(depth: ArrayLike, near_m: float, far_m: float) -> NDArray[np.float32]:
    """Convert MuJoCo's nonlinear [0, 1] depth buffer to metres."""
    normalized = np.asarray(depth, dtype=np.float64)
    if near_m <= 0 or far_m <= near_m:
        raise ValueError("expected 0 < near_m < far_m")
    if np.any(normalized < 0.0) or np.any(normalized > 1.0):
        raise ValueError("normalized depth must lie in [0, 1]")
    depth_m = near_m / (1.0 - normalized * (1.0 - near_m / far_m))
    return depth_m.astype(np.float32)


def transform_points(points: ArrayLike, transform: ArrayLike) -> NDArray[np.float64]:
    points_array = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(transform, dtype=np.float64)
    if points_array.shape[-1] != 3 or matrix.shape != (4, 4):
        raise ValueError("expected points (..., 3) and transform (4, 4)")
    homogeneous = np.concatenate([points_array, np.ones(points_array.shape[:-1] + (1,))], axis=-1)
    return (homogeneous @ matrix.T)[..., :3]


def backproject_depth(
    frame: CameraFrame,
    *,
    mask: ArrayLike | None = None,
    stride: int = 1,
    world: bool = True,
) -> NDArray[np.float64]:
    """Backproject upright depth into camera or world coordinates."""
    if stride <= 0:
        raise ValueError("stride must be positive")
    depth = frame.depth_m[::stride, ::stride].astype(np.float64)
    rows, cols = np.indices(depth.shape)
    rows = rows * stride
    cols = cols * stride
    valid = np.isfinite(depth) & (depth > 0)
    if mask is not None:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape != frame.depth_m.shape:
            raise ValueError("mask must match the depth image")
        valid &= mask_array[::stride, ::stride]
    z = depth[valid]
    x = (cols[valid] - frame.calibration.cx) * z / frame.calibration.fx
    projection_rows = rows[valid]
    if frame.calibration.observation_v_flipped:
        projection_rows = frame.calibration.height - 1 - projection_rows
    y = (projection_rows - frame.calibration.cy) * z / frame.calibration.fy
    points = np.stack([x, y, z], axis=-1)
    return transform_points(points, frame.calibration.T_world_camera) if world else points


def project_world_points(points_world: ArrayLike, frame: CameraFrame) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Project world points to upright ``(u, v)`` pixels and return camera z."""
    camera = transform_points(points_world, frame.calibration.T_camera_world)
    z = camera[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = frame.calibration.fx * camera[..., 0] / z + frame.calibration.cx
        v = frame.calibration.fy * camera[..., 1] / z + frame.calibration.cy
    if frame.calibration.observation_v_flipped:
        v = frame.calibration.height - 1 - v
    return np.stack([u, v], axis=-1), z


def quaternion_xyzw_to_matrix(quaternion: ArrayLike) -> NDArray[np.float64]:
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError("quaternion must have shape (4,)")
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("zero quaternion")
    x, y, z, w = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
