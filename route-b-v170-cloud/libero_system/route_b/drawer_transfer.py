"""Plan the wrist transition and bowl clearance after opening a drawer."""
import numpy as np

from .models import Pose


def drawer_pregrasp_frame(current, joints, positions, preferred):
    """Prefer an equivalent jaw frame with more clearance along the IK path.

    Both choices retain the rim point and vertical approach. Checking only
    the endpoints misses the nearly straight elbow encountered during a
    simultaneous wrist turn and translation after releasing the handle.
    """
    from .rim_transfer_frame import _path_margin

    alternate = preferred @ np.diag((-1.0, -1.0, 1.0))
    margins = [
        _path_margin(current.matrix, np.asarray(joints),
                     [(position, rotation) for position in positions])
        for rotation in (preferred, alternate)
    ]
    use_alternate = (margins[1] is not None and margins[1] >= .15
                     and (margins[0] is None or margins[1] > margins[0] + .10))
    selected = alternate if use_alternate else preferred
    return selected.copy(), {
        "strategy": "continuous_public_ik_joint_clearance",
        "preferred_path_margin_rad": margins[0],
        "alternate_path_margin_rad": margins[1],
        "equivalent_selected": use_alternate,
        "selected_jaw_axis_world": selected[:, 1].tolist(),
    }


def drawer_pre_lift_clearance(current, source, drawer, held_offset, outward):
    """Keep the wrist and height fixed while clearing the visible front edge."""
    normal = np.asarray(outward, dtype=float).copy()
    normal[2] = 0.0
    length = np.linalg.norm(normal)
    if length < .9:
        return None
    normal /= length
    center = current.position + np.asarray(held_offset)
    half_width = float(np.abs(normal @ source.axes_world) @ (source.extents_m / 2))
    front = float(drawer.centroid_world @ normal
                  + np.abs(normal @ drawer.axes_world) @ (drawer.extents_m / 2))
    # The observed floor ends inside the front panel; leave room for its
    # physical thickness and for a slight tilt of the rim-grasped bowl.
    needed = front + half_width + .025 - float(center @ normal)
    if (not .005 < needed < .12
            or float(center @ normal) < float(drawer.centroid_world @ normal)
            or source.bounds_min_world[2] > drawer.bounds_max_world[2] + .015):
        return None
    point = current.position + needed * normal
    return Pose(point, current.rotation)
