"""Use visible solid-body clearance to centre open jaws before withdrawing."""
import numpy as np

from .models import Pose


def open_jaw_peel_pose(source, robot):
    """Tilt the open palm away from a still-visible tall body after withdrawal.

    A bounded rotation about the jaw axis moves the palm out of the body's
    neighbourhood while leaving the aperture and TCP position unchanged.
    """
    if source is None or robot.gripper_width_m < .070:
        return None
    pose=robot.ee_pose
    if pose.rotation[2,2] > -.95:
        return None
    local=(source.centroid_world-pose.position)@pose.rotation
    size=np.abs(pose.rotation.T@source.axes_world)@source.extents_m
    if (source.height_m < 1.6*max(size[:2]) or not .020<=size[0]<=.080
            or np.linalg.norm(local)>.14 or not .005<=local[2]<=.10
            or abs(local[0])<.005 or size[1] > robot.gripper_width_m-.008):
        return None
    angle=np.sign(local[0])*np.deg2rad(35.)
    c,s=np.cos(angle),np.sin(angle)
    return Pose(pose.position,pose.rotation@np.array([[c,0,s],[0,1,0],[-s,0,c]]))


def open_jaw_recenter_pose(source, robot):
    """A bounded lateral motion for a narrow body still beside an open finger.

    This is a recovery proposal after a failed placement observation, not a
    grasp/release certificate. Cavities too wide for the jaws, a fallen body,
    side-entry wrists, and objects outside the hand neighbourhood are omitted.
    """
    if source is None or robot.gripper_width_m < .070:
        return None
    pose=robot.ee_pose
    if pose.rotation[2,2] > -.95:
        return None
    axis=pose.rotation[:,1].copy();axis[2]=0.
    if np.linalg.norm(axis)<.95:
        return None
    axis/=np.linalg.norm(axis)
    offset=source.centroid_world-pose.position
    if not .005 <= -offset[2] <= .090 or np.linalg.norm(offset)>.075:
        return None
    width=float(np.abs(axis@source.axes_world)@source.extents_m)
    free_half=(float(robot.gripper_width_m)-width)/2.
    if width<.020 or free_half<.005:
        return None
    lateral=float(offset@axis)
    if not max(.004,free_half-.004)<abs(lateral)<=.025:
        return None
    position=pose.position+lateral*axis
    # A tall body's cap can remain supported on the front edge of a finger
    # after the jaws open. Centre the aperture and withdraw sideways from the
    # visible body, keeping height. The whole observed width plus finger-pad
    # clearance avoids stopping beneath the opposite side of the cap.
    tangent=pose.rotation[:,0].copy();tangent[2]=0.
    tangent_norm=np.linalg.norm(tangent)
    if tangent_norm>=.95:
        tangent/=tangent_norm
        tangent_offset=float(offset@tangent)
        tangent_width=float(np.abs(tangent@source.axes_world)@source.extents_m)
        if (source.height_m>=1.6*max(width,tangent_width)
                and .020<=tangent_width<=.070 and abs(tangent_offset)>=.005):
            stroke=min(.065,tangent_width+.012)
            position-=np.sign(tangent_offset)*stroke*tangent
    return Pose(position,pose.rotation)
