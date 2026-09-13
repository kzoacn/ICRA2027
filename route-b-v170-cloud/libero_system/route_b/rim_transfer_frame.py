"""Choose a clear carry frame while preserving a circular bowl landing centre."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation,Slerp
from ..common.panda_kinematics import JOINT_LIMITS,panda_tcp_pose
from ..perception.flat_mesh import public_array
from ..perception.panda_visual_mask import visual_robot_arm_mask
def _hand_mask(points, position, rotation, aperture, inflation=0.):
    local = (points - position) @ rotation
    occupied = np.zeros(len(points), dtype=bool)
    for shifted, label, low, high in (
        (local + [0., 0., .097], 'PALM', [-.032, -.104, -.026], [.032, .101, .066]),
        (local - [0., aperture / 2, .0524 - .097], 'FINGER', [-.011, -.001, 0.], [.011, .027, .054]),
        ((local - [0., -aperture / 2, .0524 - .097]) @ np.diag([-1., -1., 1.]),
         'FINGER', [-.011, -.001, 0.], [.011, .027, .054]),
    ):
        # The bounding box only accelerates exact public collision hull tests.
        near = np.all((shifted >= np.array(low) - max(0., inflation)) &
                      (shifted <= np.array(high) + max(0., inflation)), axis=1)
        if inflation > 0.:
            # Offset planes can extend a sharp corner beyond an equally padded box.
            near[:] = True
        equations = public_array(label)
        occupied[near] |= np.all(shifted[near] @ equations[:, :3].T + equations[:, 3]
                                 < inflation, axis=1)
    return occupied

def _path_margin(tcp, joints, poses):
    """Follow intermediate Cartesian poses with continuous public-model IK."""
    initial = panda_tcp_pose(joints)
    inverse_base = np.linalg.inv(tcp @ np.linalg.inv(initial))
    # Solve in the robot base frame. World-coordinate finite differences can
    # otherwise choose different redundant IK branches after a rigid scene
    # transform. Picometre rounding removes only frame-conversion roundoff.
    poses = [
        (np.round(inverse_base[:3, :3] @ point + inverse_base[:3, 3], 12),
         Rotation.from_matrix(np.round(inverse_base[:3, :3] @ frame, 12)).as_matrix())
        for point, frame in poses
    ]
    q = joints.copy(); position = initial[:3, 3]; rotation = initial[:3, :3]
    margin = float('inf')
    for destination, target_rotation in poses:
        distance = float(np.linalg.norm(destination - position))
        angle = float(np.linalg.norm(Rotation.from_matrix(target_rotation @ rotation.T).as_rotvec()))
        samples = max(1, int(np.ceil(distance / .025)), int(np.ceil(angle / .20)))
        interpolator = Slerp([0., 1.], Rotation.from_matrix([rotation, target_rotation]))
        for fraction in np.linspace(0., 1., samples + 1)[1:]:
            point = position + fraction * (destination - position)
            target = interpolator(fraction).as_matrix()
            def residual(values):
                pose = panda_tcp_pose(values)
                return np.r_[5 * (pose[:3, 3] - point),
                             Rotation.from_matrix(target @ pose[:3, :3].T).as_rotvec(),
                             .01 * (values - q)]
            fit = least_squares(residual, np.clip(q, JOINT_LIMITS[:, 0] + .001,
                               JOINT_LIMITS[:, 1] - .001),
                               bounds=(JOINT_LIMITS[:, 0] + .001, JOINT_LIMITS[:, 1] - .001),
                               max_nfev=180)
            error = residual(fit.x)
            if np.linalg.norm(error[:3]) > .025 or np.linalg.norm(error[3:6]) > .06:
                return None
            q = fit.x
            margin = min(margin, float(np.minimum(q - JOINT_LIMITS[:, 0],
                                                 JOINT_LIMITS[:, 1] - q).min()))
            if margin < .05:
                return None
        position = destination; rotation = target_rotation
    return margin

def _payload_mask(points, position, rotation, initial_rotation, offset, extents, padding):
    relative = rotation @ initial_rotation.T
    center = position + relative @ offset
    local = (points - center) @ relative
    radius = max(extents[:2]) / 2 + padding
    half_height = extents[2] / 2 + padding
    return ((np.linalg.norm(local[:, :2], axis=1) < radius)
            & (abs(local[:, 2]) < half_height))

def _carry_occupancy(points, tcp, poses, aperture, offset, extents):
    position, rotation = tcp[:3, 3], tcp[:3, :3]
    counts = []
    integral = 0.
    for destination, target in poses:
        interpolate = Slerp([0., 1.], Rotation.from_matrix([rotation, target]))
        distance = float(np.linalg.norm(destination - position))
        angle = float(np.linalg.norm(Rotation.from_matrix(target @ rotation.T).as_rotvec()))
        # Integrate over swept distance so a shorter segment does not receive
        # a higher score merely because fixed-count samples become denser.
        swept_distance = distance + .15 * angle
        samples = max(1, int(np.ceil(swept_distance / .010)))
        for fraction in np.linspace(0., 1., samples + 1)[1:]:
            point = position + fraction * (destination - position)
            frame = interpolate(fraction).as_matrix()
            hit = _hand_mask(points, point, frame, aperture, -.002)
            hit |= _payload_mask(points, point, frame, tcp[:3, :3], offset, extents, -.002)
            counts.append(int(hit.sum()))
            integral += int(hit.sum()) * swept_distance / samples
        position, rotation = destination, target
    return integral, max(counts, default=0)


def choose_rim_transfer_frame(points, tcp, joints, aperture, release, clearance_z,
                              held_offset, extents):
    """Keep a clear original frame; otherwise try small upright wrist yaw changes.

    The bowl centre, release height and carry height are preserved. The wrist
    target shifts by the rotated held offset, so rotating a rim grip does not
    silently move the intended object landing point. Measured surface clearance
    and continuous public IK are required for the complete path.
    """
    points,tcp,joints,release,offset,extents = [np.asarray(v,dtype=float) for v in
        (points,tcp,joints,release,held_offset,extents)]
    if (points.ndim!=2 or points.shape[1]!=3 or tcp.shape!=(4,4) or joints.shape!=(7,)
            or release.shape!=(3,) or offset.shape!=(3,) or extents.shape!=(3,)
            or not all(np.isfinite(v).all() for v in (tcp,joints,release,offset,extents))
            or not np.isfinite(aperture) or not 0<=aperture<=.08
            or not np.isfinite(clearance_z) or clearance_z<max(tcp[2,3],release[2])
            or np.any(extents<=0) or max(extents[:2])/min(extents[:2])>1.3
            or tcp[2,2]>-.95 or np.linalg.norm(offset[:2])<.025
            or np.any(joints<JOINT_LIMITS[:,0]) or np.any(joints>JOINT_LIMITS[:,1])):
        return None
    points=points[np.isfinite(points).all(1)]
    points=points[(np.linalg.norm(points-release,axis=1)<.7)
                  | (np.linalg.norm(points-tcp[:3,3],axis=1)<.35)]
    own=visual_robot_arm_mask(points,tcp,joints)
    own|=_hand_mask(points,tcp[:3,3],tcp[:3,:3],aperture,.004)
    own|=_payload_mask(points,tcp[:3,3],tcp[:3,:3],tcp[:3,:3],offset,extents,.015)
    points=points[~own]
    if len(points)<100:return None
    high=tcp[:3,3].copy();high[2]=clearance_z
    def path(delta):
        position=release+offset-delta@offset
        rotation=delta@tcp[:3,:3]
        pre=position.copy();pre[2]=clearance_z
        return [(high,tcp[:3,:3]),(pre,rotation),(position,rotation)]
    original=path(np.eye(3))
    old_total,old_peak=_carry_occupancy(points,tcp,original,aperture,offset,extents)
    if old_total<.5 or old_peak<10:return None
    for magnitude in (30.,45.,60.,90.):
        choices=[]
        for sign in (-1.,1.):
            yaw=sign*magnitude
            delta=Rotation.from_euler('z',yaw,degrees=True).as_matrix()
            poses=path(delta)
            total,peak=_carry_occupancy(points,tcp,poses,aperture,offset,extents)
            if total>.10 or peak>3:continue
            margin=_path_margin(tcp,joints,poses)
            if margin is None:continue
            choices.append((total,peak,-margin,yaw,delta,poses))
        if choices:
            total,peak,negative_margin,yaw,delta,poses=min(choices,key=lambda c:c[:4])
            return delta,dict(strategy='observed_clear_rim_carry_frame',yaw_degrees=yaw,
                old_observed_occupancy=old_total,old_peak_occupancy=old_peak,
                selected_observed_occupancy=total,selected_peak_occupancy=peak,
                minimum_joint_margin_rad=-negative_margin,
                original_release_world_m=release.tolist(),release_world_m=poses[-1][0].tolist(),
                preserved_landing_center_world_m=(release+offset).tolist(),
                original_rotation_world=tcp[:3,:3].tolist(),release_rotation_world=poses[-1][1].tolist(),
                clearance_z_m=float(clearance_z))
    return None


def observed_rim_transfer_frame(observation, source, release_pose, clearance_z, held_offset):
    from ..perception.adapters import coerce_rgbd_frame
    from ..perception.geometry import backproject_frame
    if not {'agentview','wrist'}<=set(observation.cameras):return None
    joints=getattr(observation.robot,'joint_position',None)
    if joints is None:return None
    points=np.concatenate([backproject_frame(coerce_rgbd_frame(f,name=n),stride=2).points_world
                           for n,f in observation.cameras.items()])
    return choose_rim_transfer_frame(points,observation.robot.ee_pose.matrix,joints,
        observation.robot.gripper_width_m,release_pose.position,clearance_z,held_offset,source.extents_m)
