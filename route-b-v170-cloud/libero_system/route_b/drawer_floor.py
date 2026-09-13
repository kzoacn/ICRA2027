"""Locate the visible floor of an open drawer for placement."""
import cv2
import numpy as np

from .models import SceneObject


def drawer_floor_region(points, fixture):
    """Use a broad connected horizontal support, excluding the front panel.

    A language-grounded drawer box can be dominated by its front and handle.
    Its low height band anchors the floor search; RGB-D determines the floor
    footprint and centre. No fixture pose or region from the simulator is used.
    """
    points = np.asarray(points, dtype=np.float64)
    vertical = int(np.argmax(np.abs(fixture.axes_world[2])))
    horizontal = [i for i in range(3) if i != vertical]
    axis = fixture.axes_world[:, horizontal[0]].copy()
    axis[2] = 0.
    if np.linalg.norm(axis) < .9:
        raise ValueError('drawer has no stable horizontal frame')
    axis /= np.linalg.norm(axis)
    axes = np.column_stack((axis, np.cross([0., 0., 1.], axis), [0., 0., 1.]))
    local = (points - fixture.centroid_world) @ axes
    half = fixture.extents_m[horizontal] / 2 + .060
    low = fixture.bounds_min_world[2]
    high = low + min(.030, .4 * fixture.height_m)
    selected = (np.all(np.abs(local[:, :2]) <= half, axis=1)
                & (points[:, 2] >= low - .008) & (points[:, 2] <= high))
    points, local = points[selected], local[selected]
    if len(points) < 100:
        raise ValueError('drawer floor is not sufficiently observed')
    bins, counts = np.unique(np.rint(points[:, 2] / .002).astype(int), return_counts=True)
    height = bins[np.argmax(counts)] * .002
    band = np.abs(points[:, 2] - height) <= .003
    points, local = points[band], local[band]
    cell = .005
    origin = local[:, :2].min(axis=0)
    pixels = np.floor((local[:, :2] - origin) / cell).astype(int)
    shape = pixels.max(axis=0) + 1
    occupied = np.zeros(tuple(shape), dtype=np.uint8)
    occupied[pixels[:, 0], pixels[:, 1]] = 1
    joined = cv2.morphologyEx(occupied, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    if count < 2:
        raise ValueError('drawer floor has no connected support')
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    band = labels[pixels[:, 0], pixels[:, 1]] == component
    points, local = points[band], local[band]
    lower, upper = np.quantile(local[:, :2], [.02, .98], axis=0)
    size = upper - lower
    if len(points) < 100 or min(size) < .060 or max(size) > .40:
        raise ValueError('drawer floor footprint is too narrow or unbounded')
    coverage = stats[component, cv2.CC_STAT_AREA] * cell**2 / float(np.prod(size))
    if coverage < .35:
        raise ValueError('drawer floor is a sparse edge rather than a support')
    center = fixture.centroid_world + axes[:, :2] @ ((lower + upper) / 2)
    floor = float(np.median(points[:, 2]))
    top = max(fixture.bounds_max_world[2], floor + .020)
    center[2] = (floor + top) / 2
    extents = np.r_[size, top - floor]
    half_world = np.abs(axes) @ (extents / 2)
    return SceneObject(name=fixture.name, centroid_world=center, axes_world=axes,
                       extents_m=extents, bounds_min_world=center-half_world,
                       bounds_max_world=center+half_world,
                       confidence=fixture.confidence, point_count=len(points),
                       surface_points_world=points)
