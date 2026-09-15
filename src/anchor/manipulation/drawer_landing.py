"""Choose a contained drawer landing with room for the carried Panda hand."""
from types import SimpleNamespace
import numpy as np
from .models import Pose, SceneObject
from .drawer_interior import drawer_interior_from_points
from .gripper_clearance import panda_approach_occupancy
from ..common.panda_kinematics import path_joint_margin


def choose_drawer_landing(points, handle, source, destination, release, robot):
    """Translate a landing within measured physical walls; keep height and wrist.

    The object footprint must fit the existing public drawer geometry. Only a
    large decrease in observed hand obstruction can replace the old landing;
    the payload's swept obstruction must not increase. This does not treat
    missing depth as proof of collision-free motion.
    """
    if robot.joint_position is None:
        return destination, None
    vertical = int(np.argmax(abs(source.axes_world[2])))
    planar = [i for i in range(3) if i != vertical]
    if ('bowl' not in source.name.split() or abs(source.axes_world[2, vertical]) < .95
            or max(source.extents_m[planar]) / min(source.extents_m[planar]) >= 1.20):
        return destination, None
    points = np.asarray(points, dtype=float)
    points = points[np.all(np.isfinite(points), axis=1)]
    floor = drawer_interior_from_points(points, handle, destination, source)
    width = float(robot.gripper_width_m)
    local = (points - robot.ee_pose.position) @ robot.ee_pose.rotation
    own = np.zeros(len(points), dtype=bool)
    for lower, upper in [([-.032, -.104, -.123], [.032, .101, -.031]),
                         ([-.011, width / 2 - .001, -.045], [.011, width / 2 + .027, .010]),
                         ([-.011, -width / 2 - .027, -.045], [.011, -width / 2 + .001, .010])]:
        own |= np.all((local > np.array(lower) - .004) & (local < np.array(upper) + .004), axis=1)
    own |= np.all(abs((points - source.centroid_world) @ source.axes_world)
                  <= source.extents_m / 2 + .004, axis=1)
    nearby = np.linalg.norm(points[:, :2] - destination.centroid_world[:2], axis=1) < .35
    cloud = points[nearby & ~own]
    baseline = panda_approach_occupancy(cloud, release, width)
    if baseline < 30:
        return destination, None
    vertical = int(np.argmax(abs(source.axes_world[2])))
    planar = [i for i in range(3) if i != vertical]
    radius = float(min(source.extents_m[planar]) / 2)
    height = float(source.extents_m[vertical])

    def payload_occupancy(center):
        selected = (np.linalg.norm(cloud[:, :2] - center[:2], axis=1) < radius + .003)
        selected &= cloud[:, 2] > destination.bounds_min_world[2] + .004
        selected &= cloud[:, 2] < destination.bounds_min_world[2] + height + .10
        return int(selected.sum())

    payload_baseline = payload_occupancy(destination.centroid_world)
    options = []
    seen = set()
    for lateral in (0., -.01, .01, -.02, .02, -.03, .03):
        for inward in (0., -.01, .01, -.02, .02, -.03, .03, -.04, .04):
            shift = floor.axes_world[:, :2] @ np.array([lateral, inward])
            request = SceneObject(destination.name, destination.centroid_world + shift,
                                  destination.axes_world, destination.extents_m,
                                  destination.bounds_min_world + shift, destination.bounds_max_world + shift,
                                  destination.confidence, destination.point_count)
            bounded = drawer_interior_from_points(points, handle, request, source)
            shift = bounded.centroid_world - destination.centroid_world
            shift[2] = 0.
            key = tuple(np.round(shift, 6))
            if key in seen or np.linalg.norm(shift) < .003:
                continue
            seen.add(key)
            pose = Pose(release.position + shift, release.rotation)
            occupied = panda_approach_occupancy(cloud, pose, width)
            payload = payload_occupancy(destination.centroid_world + shift)
            if occupied > .25 * baseline or occupied >= 20 or payload > payload_baseline:
                continue
            options.append((occupied, float(np.linalg.norm(shift)), payload, bounded, pose, shift))
    for occupied, distance, payload, bounded, pose, shift in sorted(options, key=lambda item: item[:3]):
        margin = path_joint_margin(robot.joint_position, robot.ee_pose.matrix,
                                   [pose.position + [0., 0., .10], pose.position], pose.rotation)
        if not np.isfinite(margin) or margin < .05:
            continue
        center = destination.centroid_world + shift
        # Preserve the already grounded floor height; the fresh floor only
        # establishes containment and must agree with it within 10 mm.
        half = abs(bounded.axes_world) @ (bounded.extents_m / 2)
        selected = SceneObject(destination.name, center, bounded.axes_world, bounded.extents_m,
                               center - half, center + half, bounded.confidence, bounded.point_count,
                               surface_points_world=bounded.surface_points_world)
        return selected, dict(strategy='contained_drawer_landing_with_observed_hand_clearance',
                              shift_world_m=shift.tolist(), baseline_occupied=baseline,
                              selected_occupied=occupied, baseline_payload_occupied=payload_baseline,
                              selected_payload_occupied=payload, joint_margin=float(margin))
    return destination, None


def observed_drawer_landing(observation, source, destination, release):
    from ..common import CameraCalibration, CameraFrame
    from ..contact.detectors import DrawerHandleDetector
    from ..perception.adapters import coerce_rgbd_frame
    from ..perception.geometry import backproject_frame

    frames = [coerce_rgbd_frame(frame, name=name) for name, frame in observation.cameras.items()]
    if len(frames) != 2:
        raise LookupError('drawer landing needs both public views')
    cameras = {f.name: CameraFrame(f.rgb, f.depth_m, CameraCalibration(
        f.name, f.rgb.shape[1], f.rgb.shape[0], f.intrinsics, f.world_from_camera,
        f.observation_v_flipped)) for f in frames}
    points = np.concatenate([backproject_frame(f, stride=2).points_world for f in frames])
    detector = DrawerHandleDetector()
    proposals = []
    for rank in ('top', 'middle', 'bottom'):
        try:
            handle = detector.detect(SimpleNamespace(cameras=cameras), rank)
            above = handle.point_world[2] - destination.bounds_min_world[2]
            if not .012 < above < .050:
                continue
            drawer_interior_from_points(points, handle, destination, source)
        except (LookupError, ValueError):
            continue
        proposals.append((abs(above - .029), handle))
    if not proposals:
        raise LookupError('no visible handle supports the grounded drawer floor')
    handle = min(proposals, key=lambda proposal: proposal[0])[1]
    return choose_drawer_landing(points, handle, source, destination, release, observation.robot)
