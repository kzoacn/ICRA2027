"""Measure fixture support surfaces from calibrated depth and visible handles."""
from types import SimpleNamespace

import numpy as np

from .models import SceneObject, SensorObservation


def shelf_region_from_points(
    points: np.ndarray, relation: str, approach_from: np.ndarray,
) -> SceneObject:
    """Recover an open shelf bay from three observed horizontal depth modes."""
    points = np.asarray(points, dtype=np.float64)
    if relation not in {"upper_shelf", "lower_shelf"}:
        raise ValueError("unknown shelf level")
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 300:
        raise ValueError("shelf cloud is too small")
    if not np.all(np.isfinite(points)):
        raise ValueError("non-finite shelf cloud")
    edges = np.arange(float(points[:, 2].min()), float(points[:, 2].max()) + 0.003, 0.003)
    if len(edges) < 3:
        raise ValueError("shelf cloud has no height structure")
    counts, edges = np.histogram(points[:, 2], edges)
    planes = []
    for index in np.argsort(-counts):
        if counts[index] < max(60, 0.01 * len(points)):
            break
        height = float((edges[index] + edges[index + 1]) / 2.0)
        if any(abs(height - previous[0]) < 0.035 for previous in planes):
            continue
        samples = points[np.abs(points[:, 2] - height) < 0.003]
        lo, hi = np.quantile(samples, (0.02, 0.98), axis=0)
        if min((hi - lo)[:2]) < 0.065 or hi[2] - lo[2] > 0.005:
            continue
        planes.append((float(np.median(samples[:, 2])), samples))
        if len(planes) == 3:
            break
    if len(planes) != 3:
        raise ValueError("three broad shelf support planes were not observed")
    planes.sort(key=lambda item: item[0])
    gaps = np.diff([item[0] for item in planes])
    if np.any(gaps < 0.040) or np.any(gaps > 0.220):
        raise ValueError("observed shelf clearances are inconsistent")
    roof = planes[-1][1]
    _, vectors = np.linalg.eigh(np.cov(roof[:, :2], rowvar=False))
    width = vectors[:, -1]
    outward = np.array((-width[1], width[0]))
    if float(outward @ (np.asarray(approach_from)[:2] - np.median(roof[:, :2], axis=0))) < 0.0:
        outward *= -1.0
    width = np.array((outward[1], -outward[0]))
    axes = np.column_stack((np.r_[width, 0.0], np.r_[outward, 0.0], (0.0, 0.0, 1.0)))
    projected = roof @ axes
    lower, upper = np.quantile(projected, (0.02, 0.98), axis=0)
    lower[:2] += 0.015
    upper[:2] -= 0.015
    level = 1 if relation == "upper_shelf" else 0
    lower[2] = planes[level][0]
    upper[2] = planes[level + 1][0] - 0.005
    extents = upper - lower
    if min(extents[:2]) < 0.070 or max(extents[:2]) > 0.45:
        raise ValueError("shelf bay footprint is not sufficiently observed")
    center = axes @ ((lower + upper) / 2.0)
    half = np.abs(axes) @ (extents / 2.0)
    return SceneObject("cabinet shelf", center, axes, extents,
                       center - half, center + half, 0.8, len(points),
                       surface_points_world=points)


def cabinet_top_surface(observation: SensorObservation) -> SceneObject:
    """Locate the fixed cabinet roof behind its upper drawer handle.

    A whole-cabinet crop includes extended drawers and can also select distant
    kitchen cabinetry. Handle geometry binds the local fixture; a horizontal
    depth mode measures the actual roof independently of the moving drawers.
    """
    from ..common import CameraCalibration, CameraFrame
    from ..common.camera_geometry import backproject_depth
    from ..contact.detectors import DrawerHandleDetector

    cameras = {}
    for name, frame in observation.cameras.items():
        intr = frame.intrinsics
        cameras[name] = CameraFrame(frame.rgb, frame.depth_m, CameraCalibration(
            name, intr.width, intr.height,
            np.array(((intr.fx, 0.0, intr.cx), (0.0, intr.fy, intr.cy), (0.0, 0.0, 1.0))),
            frame.world_from_camera.matrix, frame.observation_v_flipped,
        ))
    # DrawerHandleDetector.detect consumes only the two camera fields.
    handle = DrawerHandleDetector().detect(SimpleNamespace(cameras=cameras), "top")
    points = np.concatenate([
        backproject_depth(frame, stride=1, world=True) for frame in cameras.values()
    ])
    delta = points - handle.point_world
    normal_coordinate = delta @ handle.outward_world
    local = points[
        (np.abs(delta @ handle.feature_axis_world) < 0.18)
        & (normal_coordinate < 0.02) & (normal_coordinate > -0.30)
        & (delta[:, 2] > -0.020) & (delta[:, 2] < 0.070)
    ]
    if len(local) < 120:
        raise LookupError("too little roof support near the observed cabinet handle")
    bins = np.arange(float(local[:, 2].min()), float(local[:, 2].max()) + 0.003, 0.003)
    if len(bins) < 2:
        raise LookupError("cabinet roof depth interval is degenerate")
    counts, edges = np.histogram(local[:, 2], bins)
    mode = int(np.argmax(counts))
    height = float((edges[mode] + edges[mode + 1]) / 2.0)
    plane = local[np.abs(local[:, 2] - height) < 0.003]
    if len(plane) < 120:
        raise LookupError("cabinet roof plane has insufficient support")
    lower, upper = np.quantile(plane, (0.02, 0.98), axis=0)
    span = upper - lower
    if (min(span[:2]) < 0.060 or max(span[:2]) > 0.40 or span[2] > 0.004):
        raise LookupError("cabinet roof patch is not a broad horizontal support")
    # Give the observed plane a finite numerical thickness without adding any
    # hidden fixture dimensions or moving its measured top surface.
    lower[2] = min(lower[2], upper[2] - 0.001)
    return SceneObject(
        "cabinet", (lower + upper) / 2.0, np.eye(3), upper - lower,
        lower, upper, handle.confidence, len(plane), surface_points_world=plane,
    )
