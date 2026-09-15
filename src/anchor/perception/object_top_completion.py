"""Complete an identity-supported flat package below its visible upper face."""
import numpy as np


def complete_visible_flat_top(instance, points):
    """Correct an upward extrusion only when raw RGB-D supports a broad top.

    A shallow detector crop can contain only the top face. Its lowest visible
    points then do not describe the object's supporting surface. This helper
    changes Z using the measured face and public thickness, keeping the crop's
    identity, XY, orientation and dimensions.
    """
    dims = instance.extents_m
    long_size, short_size = sorted(dims[:2], reverse=True)
    height = float(dims[2])
    observed = instance.observed
    if (height > .040 or long_size < .055 or short_size > .055
            or long_size < 1.4 * short_size
            or abs(instance.axes_world[2, 2]) < .98
            or observed.bounds_max_world[2] - observed.bounds_min_world[2] > min(.004, .25 * height)
            or instance.center_world[2] <= observed.bounds_max_world[2]
            or points is None):
        return instance, None
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        return instance, None
    points = points[np.all(np.isfinite(points), axis=1)]
    local = (points - instance.center_world) @ instance.axes_world
    inside = np.all(np.abs(local[:, :2]) <= dims[:2] / 2 + .003, axis=1)
    points, local = points[inside], local[inside]
    if len(points) < 36:
        return instance, None
    median_z = float(np.median(points[:, 2]))
    inliers = np.abs(points[:, 2] - median_z) <= .002
    if np.count_nonzero(inliers) < 36 or np.mean(inliers) < .80:
        return instance, None
    points, local = points[inliers], local[inliers]
    spans = np.diff(np.quantile(local[:, :2], [.05, .95], axis=0), axis=0)[0]
    if np.any(spans < .50 * dims[:2]):
        return instance, None
    xy = points[:, :2] - instance.center_world[:2]
    design = np.column_stack((xy, np.ones(len(points))))
    plane, _, rank, _ = np.linalg.lstsq(design, points[:, 2], rcond=None)
    if rank < 3 or np.linalg.norm(plane[:2]) > np.tan(np.deg2rad(10.)):
        return instance, None
    residual = np.abs(design @ plane - points[:, 2])
    top = float(plane[2])
    if (np.quantile(residual, .95) > .002
            or not observed.bounds_min_world[2] - .003 <= top <= observed.bounds_max_world[2] + .003
            or instance.center_world[2] <= top):
        return instance, None
    center = instance.center_world.copy()
    center[2] = top - height / 2
    grasp = instance.grasp_point_world.copy()
    grasp[2] += center[2] - instance.center_world[2]
    completed = instance.with_semantics(instance.label_scores, center_world=center, grasp_point_world=grasp)
    return completed, dict(kind='object_visible_top_completion', instance_id=instance.instance_id,
        measured_top_z_m=top, public_height_m=height, points=len(points), visible_spans_m=spans.tolist(),
        plane_slope_xy=plane[:2].tolist(), plane_residual95_m=float(np.quantile(residual, .95)),
        old_center_world=instance.center_world.tolist(), center_world=center.tolist())
