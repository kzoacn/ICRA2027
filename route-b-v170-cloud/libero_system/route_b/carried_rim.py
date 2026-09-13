"""Measure a carried circular rim after excluding the visible robot hand."""
import numpy as np

from .anchored_rim import anchored_rim_from_points
from .models import SceneObject


def fit_carried_rim(points, initial, pose, aperture, expected_offset):
    """Use current arc geometry near the independently held bowl identity.

    The public Panda hand envelopes remove points belonging to the fingers
    and palm. A retained initial height completes the current visible rim;
    nearby curved body points must not be mistaken for a supporting table.
    This refines load geometry and does not verify a grasp or placement.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("carried rim requires a metric point cloud")
    local = (points - pose.position) @ pose.rotation
    boxes = [
        (np.array([-.032, -.104, -.123]), np.array([.032, .101, -.031])),
        (np.array([-.011, aperture / 2 - .001, -.045]),
         np.array([.011, aperture / 2 + .027, .010])),
        (np.array([-.011, -aperture / 2 - .027, -.045]),
         np.array([.011, -aperture / 2 + .001, .010])),
    ]
    hand = np.zeros(len(points), dtype=bool)
    for lower, upper in boxes:
        hand |= np.all((local > lower - .004) & (local < upper + .004), axis=1)
    center = pose.position + np.asarray(expected_offset)
    half = np.abs(initial.axes_world) @ (initial.extents_m / 2)
    anchor = SceneObject(initial.name, center, initial.axes_world, initial.extents_m,
                         center - half, center + half, initial.confidence,
                         initial.point_count)
    fitted = anchored_rim_from_points(points[~hand], anchor)
    height = initial.height_m
    center = fitted.centroid_world.copy()
    center[2] = fitted.bounds_max_world[2] - height / 2
    extents = fitted.extents_m.copy()
    extents[2] = height
    half = extents / 2
    return SceneObject(initial.name, center, np.eye(3), extents,
                       center - half, center + half, fitted.confidence,
                       fitted.point_count, surface_points_world=fitted.surface_points_world)
