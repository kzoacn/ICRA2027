"""Check drawer clearance and equivalent-frame geometry without simulator truth."""
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "route-b-v170-cloud"))
from libero_system.route_b.drawer_transfer import drawer_pre_lift_clearance, drawer_pregrasp_frame
from libero_system.route_b.models import Pose, SceneObject
from libero_system.common.panda_kinematics import panda_tcp_pose


def box(name, center, extent, rotation=np.eye(3)):
    center, extent = np.asarray(center), np.asarray(extent)
    radius = np.abs(rotation) @ (extent / 2)
    return SceneObject(name, center, rotation, extent, center-radius, center+radius, .9, 200)


current = Pose(np.array([-.14, 0., 1.045]), np.diag([1., -1., -1.]))
offset = np.array([.044, 0., -.012])
bowl = box("bowl", current.position+offset, [.104, .104, .050])
drawer = box("drawer", [0., -.085, 1.092], [.184, .112, .055])
outward = np.array([0., 1., 0.])
clear = drawer_pre_lift_clearance(current, bowl, drawer, offset, outward)
assert clear is not None
assert clear.position[2] == current.position[2]
assert np.array_equal(clear.rotation, current.rotation)
assert (clear.position+offset)[1] - bowl.extents_m[1]/2 > drawer.bounds_max_world[1] + .020
# A bowl already outside the lip or above the panel needs no extra waypoint.
assert drawer_pre_lift_clearance(Pose(current.position+[0., .2, 0.], current.rotation),
                                bowl, drawer, offset, outward) is None
high_bowl = box("bowl", bowl.centroid_world+[0., 0., .3], bowl.extents_m)
assert drawer_pre_lift_clearance(Pose(current.position+[0., 0., .3], current.rotation),
                                high_bowl, drawer, offset, outward) is None

# Public robot geometry supplies a post-handle pose and a rim approach.
joints = np.array([.44, 1.06, 0., -.95, -1.37, 1.12, 1.27])
tcp = panda_tcp_pose(joints)
robot_pose = Pose(tcp[:3, 3], tcp[:3, :3])
positions = [tcp[:3, 3]+[-.18, -.107, -.156], tcp[:3, 3]+[-.18, -.107, -.256]]
preferred = np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., -1.]])
selected, trace = drawer_pregrasp_frame(robot_pose, joints, positions, preferred)
assert trace["equivalent_selected"]
assert trace["alternate_path_margin_rad"] > trace["preferred_path_margin_rad"] + .1
assert np.allclose(selected[:, 2], preferred[:, 2])
assert abs(selected[:, 1] @ preferred[:, 1]) > .999999
for yaw in [.3, 1.2, -2.0]:
    rotation = Rotation.from_euler("z", yaw).as_matrix()
    shift = np.array([.2, -.3, .1])
    moved_bowl = box("bowl", rotation@bowl.centroid_world+shift,
                     bowl.extents_m, rotation@bowl.axes_world)
    moved_drawer = box("drawer", rotation@drawer.centroid_world+shift,
                       drawer.extents_m, rotation@drawer.axes_world)
    moved = drawer_pre_lift_clearance(
        Pose(rotation@current.position+shift, rotation@current.rotation),
        moved_bowl, moved_drawer, rotation@offset, rotation@outward)
    assert moved is not None
    assert np.allclose(moved.position, rotation@clear.position+shift)
    turned, moved_trace = drawer_pregrasp_frame(
        Pose(rotation@robot_pose.position+shift, rotation@robot_pose.rotation),
        joints, [rotation@p+shift for p in positions], rotation@preferred)
    assert np.allclose(turned, rotation@selected)
    assert moved_trace["equivalent_selected"] == trace["equivalent_selected"]

print("Drawer geometry checks passed: payload clearance, unchanged grasp geometry, and yaw/translation invariance.")
