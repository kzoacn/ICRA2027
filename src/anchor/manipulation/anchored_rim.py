"""Recover an initially identified vessel from a fresh, partially visible rim."""
import numpy as np

from .models import SceneObject
from .rim_geometry import fit_visible_upper_rim


def anchored_rim_from_points(points, anchor):
    """Confirm identity through local current geometry after fixture occlusion.

    The old observation only bounds the search and vessel size. The new centre
    and top come from current RGB-D arc samples; a hidden or non-circular patch
    cannot authorize using the old grasp point.
    """
    vertical = int(np.argmax(np.abs(anchor.axes_world[2])))
    horizontal = [i for i in range(3) if i != vertical]
    if abs(anchor.axes_world[2, vertical]) < .95:
        raise ValueError('anchor is not an upright vessel')
    size = anchor.extents_m[horizontal]
    if max(size) > 1.2*min(size):
        raise ValueError('anchor footprint is not approximately circular')
    radius = min(size)/2
    height = anchor.height_m
    points = np.asarray(points, dtype=float)
    valid = (np.all(np.isfinite(points), axis=1)
             & (np.linalg.norm(points[:, :2]-anchor.centroid_world[:2], axis=1) < 1.5*radius)
             & (points[:, 2] > anchor.bounds_min_world[2]+.3*height)
             & (points[:, 2] < anchor.bounds_max_world[2]+.3*height))
    local = points[valid]
    center_xy, measured_radius, top = fit_visible_upper_rim(local)
    if not .8*radius <= measured_radius <= 1.2*radius:
        raise ValueError('fresh rim diameter disagrees with the identified vessel')
    if np.linalg.norm(center_xy-anchor.centroid_world[:2]) > radius:
        raise ValueError('fresh rim is outside the identity neighbourhood')
    if abs(top-anchor.bounds_max_world[2]) > .5*height:
        raise ValueError('fresh rim height disagrees with the identified vessel')
    # A precontact crop may fuse the tabletop into the vessel's lower bound.
    # A broad current support just outside the rim can bound that height;
    # do not propagate a body bottom below a newly observed tabletop.
    distance = np.linalg.norm(points[:, :2]-center_xy, axis=1)
    support = points[(distance > 1.05*measured_radius) & (distance < 1.5*measured_radius)
                     & (points[:, 2] >= anchor.bounds_min_world[2]-.005)
                     & (points[:, 2] <= top-.3*measured_radius)]
    if len(support) >= 60:
        bins, counts = np.unique(np.rint(support[:, 2]/.002).astype(int), return_counts=True)
        mode = bins[np.argmax(counts)]*.002
        plane = support[abs(support[:, 2]-mode) < .002]
        if len(plane) >= 60 and min(np.ptp(plane[:, :2], axis=0)) > measured_radius:
            observed_height = top-float(np.median(plane[:, 2]))
            if .3*measured_radius <= observed_height <= 2*measured_radius:
                height = min(height, observed_height)
    center = np.r_[center_xy, top-height/2]
    half = np.array([measured_radius, measured_radius, height/2])
    return SceneObject(anchor.name, center, np.eye(3), 2*half, center-half, center+half,
                       anchor.confidence, len(local), surface_points_world=local)
