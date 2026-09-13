"""Separate a cluttered flat-object transfer from its final descent."""
import numpy as np

from .models import Pose


def flat_pregrasp_transfer(points, current, pregrasp, *, aperture=.08):
    """Freeze clearance waypoints before independent Cartesian axes descend.

    Nearby visible tall surfaces identify a cluttered traverse. The initial
    measured height is retained, or raised above current clutter, while XY
    and grasp orientation settle. No later observation can ratchet this
    height or re-enable a completed waypoint.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 30:
        return []
    local = (points - current.position) @ current.rotation
    boxes = [
        (np.array([-.032, -.104, -.123]), np.array([.032, .101, -.031])),
        (np.array([-.011, aperture / 2 - .001, -.045]), np.array([.011, aperture / 2 + .027, .010])),
        (np.array([-.011, -aperture / 2 - .027, -.045]), np.array([.011, -aperture / 2 + .001, .010])),
    ]
    hand = np.zeros(len(points), dtype=bool)
    for lower, upper in boxes:
        hand |= np.all((local > lower - .004) & (local < upper + .004), axis=1)
    points = points[~hand]
    delta = pregrasp.position[:2] - current.position[:2]
    length = float(np.linalg.norm(delta))
    if length < .04:
        return []
    along = ((points[:, :2] - current.position[:2]) @ delta) / (length * length)
    closest = current.position[:2] + np.clip(along, 0., 1.)[:, None] * delta
    nearby = ((along >= 0.) & (along <= 1.)
              & (np.linalg.norm(points[:, :2] - closest, axis=1) < .10)
              & (points[:, 2] > pregrasp.position[2] - .025))
    cloud = points[nearby]
    if len(cloud) < 30:
        return []
    safe_height = max(current.position[2], float(np.quantile(cloud[:, 2], .99)) + .060)
    if safe_height <= pregrasp.position[2] + .025:
        return []
    waypoints = []
    if safe_height > current.position[2] + .005:
        high = current.position.copy()
        high[2] = safe_height
        waypoints.append(Pose(high, current.rotation))
    across = pregrasp.position.copy()
    across[2] = safe_height
    waypoints.append(Pose(across, pregrasp.rotation))
    return waypoints
