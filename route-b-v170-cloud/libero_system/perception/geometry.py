"""RGB-D projection, horizontal support estimation, and 3-D connectivity."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from .schema import ObjectInstance, ObservedGeometry, PointCloud, RGBDFrame


FloatArray = NDArray[np.float64]


class GeometryError(RuntimeError):
    pass


@dataclass(frozen=True)
class ComponentConfig:
    voxel_size_m: float = 0.006
    connectivity_radius_m: float = 0.015
    min_points: int = 36
    min_voxels: int = 4

    def __post_init__(self) -> None:
        if self.voxel_size_m <= 0 or self.connectivity_radius_m < self.voxel_size_m:
            raise ValueError("connectivity radius must be at least one positive voxel")
        if self.min_points <= 0 or self.min_voxels <= 0:
            raise ValueError("component minima must be positive")


def backproject_frame(
    frame: RGBDFrame,
    *,
    camera_index: int = 0,
    mask: NDArray[np.bool_] | None = None,
    stride: int = 1,
    min_depth_m: float = 0.05,
    max_depth_m: float = 3.0,
) -> PointCloud:
    """Back-project an upright metric RGB-D image into the world frame."""

    if camera_index < 0:
        raise ValueError("camera_index cannot be negative")
    if stride <= 0 or not 0 < min_depth_m < max_depth_m:
        raise ValueError("invalid projection sampling or depth interval")
    valid = np.isfinite(frame.depth_m) & (frame.depth_m >= min_depth_m) & (frame.depth_m <= max_depth_m)
    if mask is not None:
        supplied = np.asarray(mask, dtype=np.bool_)
        if supplied.shape != frame.depth_m.shape:
            raise ValueError("projection mask must match the frame")
        valid &= supplied
    if stride > 1:
        sample = np.zeros_like(valid)
        sample[::stride, ::stride] = True
        valid &= sample
    rows, cols = np.nonzero(valid)
    if len(rows) == 0:
        return PointCloud(
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.uint8),
            np.empty((0,), dtype=np.int32),
            np.empty((0, 2), dtype=np.float64),
        )
    z = frame.depth_m[rows, cols]
    fx, fy = frame.intrinsics[0, 0], frame.intrinsics[1, 1]
    cx, cy = frame.intrinsics[0, 2], frame.intrinsics[1, 2]
    projection_rows = frame.height - 1 - rows if frame.observation_v_flipped else rows
    camera_points = np.column_stack(
        ((cols - cx) * z / fx, (projection_rows - cy) * z / fy, z)
    )
    rotation = frame.world_from_camera[:3, :3]
    translation = frame.world_from_camera[:3, 3]
    world_points = camera_points @ rotation.T + translation
    return PointCloud(
        world_points,
        frame.rgb[rows, cols],
        np.full(len(rows), camera_index, dtype=np.int32),
        np.column_stack((cols, rows)).astype(np.float64),
    )


def fuse_rgbd_frames(
    frames: Sequence[RGBDFrame],
    *,
    stride: int = 2,
    min_depth_m: float = 0.05,
    max_depth_m: float = 3.0,
) -> PointCloud:
    if not frames:
        raise ValueError("at least one RGB-D frame is required")
    clouds = [
        backproject_frame(
            frame,
            camera_index=index,
            stride=stride,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
        )
        for index, frame in enumerate(frames)
    ]
    return PointCloud(
        np.concatenate([cloud.points_world for cloud in clouds]),
        np.concatenate([cloud.colors_rgb for cloud in clouds]),
        np.concatenate([cloud.camera_indices for cloud in clouds]),
        np.concatenate([cloud.pixels_uv for cloud in clouds]),
    )


def estimate_horizontal_support(
    points_world: FloatArray,
    *,
    z_bounds: tuple[float, float] = (0.75, 1.05),
    bin_size_m: float = 0.003,
    refine_tolerance_m: float = 0.0045,
    min_inliers: int = 100,
) -> float:
    """Estimate the dominant horizontal plane without normals or scene state.

    A smoothed z-mode is robust for LIBERO's large horizontal work surface.
    The returned height is the median of points around the mode, rather than a
    histogram-bin centre.
    """

    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_world must have shape (N, 3)")
    lower, upper = map(float, z_bounds)
    if not lower < upper or bin_size_m <= 0 or refine_tolerance_m <= 0:
        raise ValueError("invalid table-estimation parameters")
    z = points[:, 2]
    z = z[np.isfinite(z) & (z >= lower) & (z <= upper)]
    if len(z) < min_inliers:
        raise GeometryError(f"only {len(z)} points lie inside the support-height interval")
    edges = np.arange(lower, upper + bin_size_m, bin_size_m)
    counts, _ = np.histogram(z, bins=edges)
    if len(counts) < 3:
        raise GeometryError("support-height interval is too narrow")
    smoothed = np.convolve(counts.astype(np.float64), np.array([0.25, 0.5, 0.25]), mode="same")
    best = int(np.argmax(smoothed))
    mode = (edges[best] + edges[best + 1]) / 2.0
    inliers = z[np.abs(z - mode) <= refine_tolerance_m]
    if len(inliers) < min_inliers:
        raise GeometryError(f"dominant support plane has only {len(inliers)} inliers")
    return float(np.median(inliers))


def _voxel_centroids(points: FloatArray, voxel_size_m: float) -> tuple[FloatArray, NDArray[np.int64]]:
    keys = np.floor(points / voxel_size_m).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    count = int(inverse.max()) + 1 if len(inverse) else 0
    centroids = np.zeros((count, 3), dtype=np.float64)
    weights = np.bincount(inverse, minlength=count).astype(np.float64)
    np.add.at(centroids, inverse, points)
    centroids /= weights[:, None]
    return centroids, inverse


def connected_components_3d(
    points_world: FloatArray,
    config: ComponentConfig = ComponentConfig(),
) -> list[NDArray[np.int64]]:
    """Return point indices grouped by cKDTree connectivity of occupied voxels."""

    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("points_world must be a finite (N, 3) array")
    if len(points) < config.min_points:
        return []
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover - scipy is part of environment.yml
        raise RuntimeError("scipy is required for sensor-only 3-D connectivity") from exc

    voxels, point_to_voxel = _voxel_centroids(points, config.voxel_size_m)
    parent = np.arange(len(voxels), dtype=np.int64)
    size = np.ones(len(voxels), dtype=np.int64)

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = int(parent[item])
        return item

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if size[left_root] < size[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        size[left_root] += size[right_root]

    pairs = cKDTree(voxels).query_pairs(config.connectivity_radius_m, output_type="ndarray")
    for left, right in pairs:
        union(int(left), int(right))
    roots = np.fromiter((find(index) for index in range(len(voxels))), dtype=np.int64)
    point_roots = roots[point_to_voxel]
    components: list[NDArray[np.int64]] = []
    for root in np.unique(point_roots):
        voxel_count = int(np.count_nonzero(roots == root))
        indices = np.flatnonzero(point_roots == root)
        if voxel_count >= config.min_voxels and len(indices) >= config.min_points:
            components.append(indices.astype(np.int64))
    components.sort(key=len, reverse=True)
    return components


def fit_observed_geometry(
    points_world: FloatArray,
    *,
    support_height_m: float | None = None,
    force_support_plane: bool = False,
) -> ObservedGeometry:
    """Fit an upright OBB while retaining robust visible-surface AABB bounds."""

    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 8:
        raise GeometryError("at least eight 3-D points are required")
    lower, upper = np.quantile(points, [0.01, 0.99], axis=0)
    xy_center = np.median(points[:, :2], axis=0)
    centered_xy = points[:, :2] - xy_center
    covariance = centered_xy.T @ centered_xy / max(len(points) - 1, 1)
    _, eigenvectors = np.linalg.eigh(covariance)
    major = eigenvectors[:, -1]
    if major[0] < 0 or (abs(major[0]) < 1e-9 and major[1] < 0):
        major *= -1
    minor = np.array([-major[1], major[0]])
    axes = np.array(
        [[major[0], minor[0], 0.0], [major[1], minor[1], 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    local_xy = centered_xy @ axes[:2, :2]
    local_lower, local_upper = np.quantile(local_xy, [0.01, 0.99], axis=0)
    local_mid = (local_lower + local_upper) / 2.0
    center_xy = xy_center + axes[:2, :2] @ local_mid
    bottom = lower[2]
    if (
        support_height_m is not None
        and (
            force_support_plane
            or support_height_m - 0.015 <= lower[2] <= support_height_m + 0.020
        )
    ):
        bottom = float(support_height_m)
    center_z = (bottom + upper[2]) / 2.0
    extents = np.array(
        [local_upper[0] - local_lower[0], local_upper[1] - local_lower[1], upper[2] - bottom],
        dtype=np.float64,
    )
    extents = np.maximum(extents, 0.002)
    robust_lower = lower.copy()
    robust_lower[2] = bottom
    return ObservedGeometry(
        center_world=np.array([center_xy[0], center_xy[1], center_z]),
        axes_world=axes,
        extents_m=extents,
        bounds_min_world=robust_lower,
        bounds_max_world=upper,
    )


def center_ray_intersection(
    component: PointCloud,
    frames: Sequence[RGBDFrame],
    plane_height_m: float,
    *,
    fallback_xy: FloatArray,
) -> FloatArray:
    """Intersect the best component 2-D bbox centre ray with a horizontal plane."""

    fallback = np.asarray(fallback_xy, dtype=np.float64)
    if fallback.shape != (2,):
        raise ValueError("fallback_xy must have shape (2,)")
    choices: list[tuple[float, int, FloatArray]] = []
    for camera_index, frame in enumerate(frames):
        selected = np.flatnonzero(component.camera_indices == camera_index)
        if len(selected) < 4:
            continue
        pixels = component.pixels_uv[selected]
        lower, upper = np.quantile(pixels, [0.01, 0.99], axis=0)
        centre_uv = (lower + upper) / 2.0
        optical_world = frame.world_from_camera[:3, 2]
        verticality = abs(float(optical_world[2]))
        score = verticality + 0.015 * math.log1p(len(selected))
        choices.append((score, camera_index, centre_uv))
    if not choices:
        return np.array([fallback[0], fallback[1], plane_height_m], dtype=np.float64)
    _, camera_index, centre_uv = max(choices, key=lambda item: item[0])
    frame = frames[camera_index]
    projection_v = (
        frame.height - 1 - centre_uv[1]
        if frame.observation_v_flipped
        else centre_uv[1]
    )
    camera_ray = np.array(
        [
            (centre_uv[0] - frame.intrinsics[0, 2]) / frame.intrinsics[0, 0],
            (projection_v - frame.intrinsics[1, 2]) / frame.intrinsics[1, 1],
            1.0,
        ],
        dtype=np.float64,
    )
    origin = frame.world_from_camera[:3, 3]
    direction = frame.world_from_camera[:3, :3] @ camera_ray
    if abs(direction[2]) < 1e-6:
        return np.array([fallback[0], fallback[1], plane_height_m], dtype=np.float64)
    distance = (plane_height_m - origin[2]) / direction[2]
    if distance <= 0:
        return np.array([fallback[0], fallback[1], plane_height_m], dtype=np.float64)
    result = origin + distance * direction
    return np.array([result[0], result[1], plane_height_m], dtype=np.float64)


def provisional_instance(
    instance_id: str,
    component: PointCloud,
    frames: Sequence[RGBDFrame],
    *,
    support_height_m: float,
    color_histogram: FloatArray | None = None,
) -> ObjectInstance:
    observed = fit_observed_geometry(
        component.points_world,
        support_height_m=support_height_m,
        force_support_plane=True,
    )
    height = max(float(observed.extents_m[2]), 0.006)
    centre_height = support_height_m + height / 2.0
    center = center_ray_intersection(
        component,
        frames,
        centre_height,
        fallback_xy=observed.center_world[:2],
    )
    extents = observed.extents_m.copy()
    grasp = center.copy()
    grasp[2] = support_height_m + height
    return ObjectInstance(
        instance_id=instance_id,
        center_world=center,
        grasp_point_world=grasp,
        axes_world=observed.axes_world,
        extents_m=extents,
        observed=observed,
        confidence=0.35,
        point_count=len(component.points_world),
        color_histogram=color_histogram,
    )
