"""Separate the observed flat-stove support from its taller adjacent control."""
from types import SimpleNamespace
import numpy as np
from .models import SceneObject


def visible_burner_support(frames, preferred):
    """Localize the visible red burner inside the grounded stove crop.

    The circle is a rendered RGB feature. Its depth supplies the physical
    metal/burner height; no evaluator region or scene object pose is read.
    """
    import cv2
    from ..perception.geometry import backproject_frame

    candidates = []
    for frame in frames:
        hsv = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2HSV)
        red = ((hsv[:, :, 0] <= 12) & (hsv[:, :, 1] >= 80)
               & (hsv[:, :, 2] >= 65))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(red.astype(np.uint8), 8)
        for index in range(1, count):
            if stats[index, cv2.CC_STAT_AREA] < 45:
                continue
            points = backproject_frame(frame, mask=labels == index).points_world
            if len(points) < 45:
                continue
            low, high = np.quantile(points, [.02, .98], axis=0)
            span = high - low
            center = (low + high) / 2.0
            if (not np.all((span[:2] >= .090) & (span[:2] <= .140))
                    or span[2] > .005
                    or np.linalg.norm(center[:2] - preferred.centroid_world[:2]) > .090
                    or not preferred.bounds_min_world[2] < center[2] < preferred.bounds_max_world[2] + .025):
                continue
            candidates.append((len(points), center, points))
    if not candidates:
        raise LookupError('no complete visible burner disk inside the observed stove')
    _, disk, points = max(candidates, key=lambda item: item[0])
    # The public square metal plate is 190 mm wide; its collision surface is
    # 3.5 mm above the visible red disk. Fit the horizontal frame from the
    # grounded stove, but remove the knob from both centre and footprint.
    vertical = int(np.argmax(abs(preferred.axes_world[2])))
    planar = [i for i in range(3) if i != vertical]
    axis = preferred.axes_world[:, planar[0]].copy(); axis[2] = 0
    axis /= np.linalg.norm(axis)
    axes = np.column_stack((axis, np.cross([0., 0., 1.], axis), [0., 0., 1.]))
    top = float(disk[2] + .0035)
    center = disk.copy(); center[2] = top - .02275
    extents = np.array((.190, .190, .0455))
    half = abs(axes) @ (extents / 2)
    result = SceneObject(preferred.name, center, axes, extents, center-half, center+half,
                         preferred.confidence, len(points), surface_points_world=points)
    return result, dict(strategy='rgbd_visible_burner_disk_plus_public_plate_geometry',
                        disk_center_world_m=disk.tolist(), old_center_world_m=preferred.centroid_world.tolist(),
                        support_z_m=top, disk_points=len(points))


def bounded_stove_height(points, preferred):
    """Correct a high crop only when a broad metal plane independently supports it.

    The public flat-stove physical base top is at local 20 mm; its load-bearing
    burner collision plate top is at 25.5 mm. Thus the observed metal plane
    implies a support 5.5 mm above it, without any scene pose or task region.
    Preserve the old horizontal shape, slots, floor and orientation.
    """
    points = np.asarray(points, dtype=float)
    points = points[np.all(np.isfinite(points), axis=1)]
    vertical = int(np.argmax(abs(preferred.axes_world[2])))
    if len(points) < 300 or abs(preferred.axes_world[2, vertical]) < .999:
        return preferred, None
    bins, counts = np.unique(np.rint(points[:, 2] / .002).astype(int), return_counts=True)
    plane = points[abs(points[:, 2] - bins[np.argmax(counts)] * .002) < .002]
    if len(plane) < 300:
        return preferred, None
    lower, upper = np.quantile(plane[:, :2], [.01, .99], axis=0)
    spans = np.sort(upper - lower)
    if not (.08 <= spans[0] <= .30 and .11 <= spans[1] <= .35):
        return preferred, None
    metal = float(np.median(plane[:, 2]))
    bottom = float(preferred.bounds_min_world[2])
    if not .015 < metal - bottom < .060:
        return preferred, None
    top = metal + .0055
    excess = float(preferred.bounds_max_world[2] - top)
    if not .010 < excess < .050:
        return preferred, None
    center = preferred.centroid_world.copy(); center[2] = (bottom + top) / 2
    extents = preferred.extents_m.copy(); extents[vertical] = top - bottom
    maximum = preferred.bounds_max_world.copy(); maximum[2] = top
    result = SceneObject(preferred.name, center, preferred.axes_world, extents,
                         preferred.bounds_min_world, maximum, preferred.confidence,
                         preferred.point_count, surface_points_world=plane)
    return result, dict(strategy='visible_metal_plane_plus_public_burner_thickness',
                        metal_plane_z_m=metal, old_top_z_m=float(preferred.bounds_max_world[2]),
                        support_z_m=top, removed_height_m=excess, plane_points=len(plane))


def observed_stove_support(observation, preferred):
    from ..common import CameraCalibration, CameraFrame
    from ..goal_skills.detectors import StoveKnobDetector, PlateFrontDetector
    from ..perception.adapters import coerce_rgbd_frame

    frames = [coerce_rgbd_frame(frame, name=name) for name, frame in observation.cameras.items()]
    try:
        return visible_burner_support(frames, preferred)
    except LookupError:
        pass
    cameras = {f.name: CameraFrame(f.rgb, f.depth_m, CameraCalibration(
        f.name, f.rgb.shape[1], f.rgb.shape[0], f.intrinsics, f.world_from_camera,
        f.observation_v_flipped)) for f in frames}
    if not {'agentview', 'wrist'} <= set(cameras):
        raise LookupError('stove support needs calibrated agent and wrist views')
    knob = StoveKnobDetector().detect(SimpleNamespace(cameras=cameras))
    groups = [PlateFrontDetector._stove_surface_in_frame(frame, knob) for frame in cameras.values()]
    groups = [group for group in groups if group is not None]
    if not groups:
        raise LookupError('no metal support next to the observed stove knob')
    return bounded_stove_height(np.concatenate(groups), preferred)
