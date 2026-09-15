"""Geometric invariants and false-positive guards for new sensor localizers."""
from pathlib import Path
from types import SimpleNamespace
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from anchor.manipulation.models import SceneObject
from anchor.manipulation.microwave_interior import microwave_cavity_from_walls
from anchor.contact.detectors import MicrowaveDoorHandleDetector
from anchor.manipulation.stove_support import visible_burner_support


def rejected(function, *args):
    try:
        function(*args)
    except LookupError:
        return
    raise AssertionError('ambiguous or absent sensor support was accepted')


capture = np.load(Path(__file__).parent / 'fixtures/microwave_sensor_surfaces.npz')
def captured_frame(index):
    return MicrowaveDoorHandleDetector._fixture_frame(
        capture['reference_position'], capture[f'candidate_{index}_center'],
        capture[f'candidate_{index}_axes'], capture[f'candidate_{index}_extents']/2)
rejected(microwave_cavity_from_walls, capture['candidate_0_points'], captured_frame(0))
base_frame = captured_frame(1)
base_points = capture['candidate_1_points']
base, _ = microwave_cavity_from_walls(base_points, base_frame)
for yaw in [0., .7, 2.1]:
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array(((c, -s, 0.), (s, c, 0.), (0., 0., 1.)))
    translation = np.array((.31, -.12, .13))
    frame = SimpleNamespace(center=rotation@base_frame.center+translation,
                            outward=rotation@base_frame.outward,
                            handle_side=rotation@base_frame.handle_side)
    points = base_points@rotation.T+translation
    cavity, trace = microwave_cavity_from_walls(points, frame)
    expected = rotation@base.centroid_world+translation
    assert np.linalg.norm(cavity.centroid_world-expected) < .004
    assert cavity.axes_world[:, 1]@(rotation@base.axes_world[:, 1]) > .999
    rejected(microwave_cavity_from_walls, points[points[:, 2] > points[:, 2].max()-.005], frame)

size = 160
row, col = np.indices((size, size))
mask = (row-80)**2 + (col-80)**2 <= 62**2
rgb = np.full((size, size, 3), 255, dtype=np.uint8)
rgb[mask] = (200, 20, 20)
frame = SimpleNamespace(rgb=rgb, depth_m=np.ones((size, size)),
                        intrinsics=np.array(((1000., 0., 80.), (0., 1000., 80.), (0., 0., 1.))),
                        world_from_camera=np.eye(4), observation_v_flipped=False,
                        height=size, width=size)
center = np.array((0., 0., .990)); extent = np.array((.22, .20, .060))
fixture = SceneObject('stove', center, np.eye(3), extent, center-extent/2, center+extent/2, .8, 100)
stove, trace = visible_burner_support([frame], fixture)
assert np.linalg.norm(stove.centroid_world[:2]) < .002
assert np.isclose(stove.bounds_max_world[2], 1.0035)
frame.rgb = rgb.copy(); frame.rgb[:, :80] = 255
rejected(visible_burner_support, [frame], fixture)
print('Sensor geometry checks passed: yaw/translation, physical heights, table/roof-only/partial-disk rejection.')
