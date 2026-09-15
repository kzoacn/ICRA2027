"""Transport a grasped rigid object's original RGB-D shape with the measured wrist."""
import numpy as np

from .models import SceneObject


def propagate_grasped_shape(initial, measured, grasp_rotation, current_rotation):
    """Retain the fresh centre while avoiding shape-axis swaps in an occluded crop."""
    delta = np.asarray(current_rotation) @ np.asarray(grasp_rotation).T
    axes = delta @ initial.axes_world
    half = np.abs(axes) @ (initial.extents_m / 2.0)
    return SceneObject(
        measured.name, measured.centroid_world, axes, initial.extents_m,
        measured.centroid_world - half, measured.centroid_world + half,
        measured.confidence, measured.point_count,
    )


def compact_center_pinch_offset(initial, grasp_pose, current_pose, width):
    """Rigid centre offset for an upright compact body held across its width.

    During a close-fitting centre pinch, the visible front surface can move
    toward the camera as the fingers occlude the body. Use the pre-contact
    full-body centre when the grasp geometry and measured aperture agree.
    Edge grasps, handles, flat packages and large wrist rotations are excluded.
    This estimates load geometry after separate grasp verification; it does
    not establish that a grasp succeeded.
    """
    from scipy.spatial.transform import Rotation

    vertical = int(np.argmax(np.abs(initial.axes_world[2])))
    planar = [i for i in range(3) if i != vertical]
    small, large = sorted(initial.extents_m[planar])
    height = float(initial.extents_m[vertical])
    if (small <= 0. or large / small > 1.12 or not 1.05 * large <= height <= 2.0 * small
            or abs(initial.axes_world[2, vertical]) < .95
            or not .80 * small <= width <= 1.08 * large
            or grasp_pose.rotation[2, 2] > -.90 or current_pose.rotation[2, 2] > -.90):
        return None
    offset = initial.centroid_world - grasp_pose.position
    if np.linalg.norm(offset[:2]) > .10 * small or not .10 * height <= -offset[2] <= .55 * height:
        return None
    delta = current_pose.rotation @ grasp_pose.rotation.T
    if Rotation.from_matrix(delta).magnitude() > .18:
        return None
    return delta @ offset
