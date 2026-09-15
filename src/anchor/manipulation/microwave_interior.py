"""Ground a microwave cavity from its visible roof and physical wall sizes."""
import numpy as np
from .models import SceneObject


def microwave_cavity_from_walls(points, frame):
    """Recover the control-panel corner when the roof is partly occluded.

    The tall front panel identifies the control side independently of the
    robot's viewing side. Two vertical planes and a roof height anchor the
    fixed physical wall offsets, so a triangular roof crop cannot move the
    bay centre or rotate its entry direction.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 300 or np.ptp(points[:, 2]) < .12:
        raise LookupError('microwave needs visible vertical body structure')
    bins, counts = np.unique(np.rint(points[:, 2] / .003).astype(int), return_counts=True)
    heights = [(counts[i], bins[i]*.003) for i in range(len(bins))
               if bins[i]*.003 > frame.center[2]+.020]
    if not heights:
        raise LookupError('microwave lacks a supported roof height')
    _, height = max(heights)
    roof = points[abs(points[:, 2] - height) < .003]
    if len(roof) < 150:
        raise LookupError('microwave roof has too few metric samples')
    roof_z = float(np.median(roof[:, 2]))
    walls = points[(points[:, 2] < roof_z-.020) & (points[:, 2] > roof_z-.182)]
    best = None
    heading = np.arctan2(frame.outward[1], frame.outward[0])
    for angle in heading + np.linspace(-.45, .45, 91):
        normal = np.array((np.cos(angle), np.sin(angle), 0.))
        coordinate = walls @ normal
        bins, counts = np.unique(np.rint(coordinate/.003).astype(int), return_counts=True)
        if not len(bins):
            continue
        for index in np.argsort(-counts)[:3]:
            selected = walls[abs(coordinate-bins[index]*.003) < .002]
            if len(selected) < 100 or np.ptp(selected[:, 2]) < .10:
                continue
            width = np.cross(normal, [0., 0., 1.])
            if np.ptp(selected @ width) < .040:
                continue
            if best is None or len(selected) > best[0]:
                best = (len(selected), normal, selected)
    if best is None:
        raise LookupError('no broad vertical control-panel face')
    _, outward, panel = best
    _, vectors = np.linalg.eigh(np.cov(panel[:, :2], rowvar=False))
    refined = np.r_[vectors[:, 0], 0.]
    if refined @ outward < 0:
        refined *= -1
    if refined @ outward > .98:
        outward = refined
    width = np.cross(outward, [0., 0., 1.])
    side = float((np.median(panel, axis=0) - frame.center) @ width)
    if abs(side) < .035:
        raise LookupError('front panel does not identify the control side')
    control_side = width * np.sign(side)
    outer = float(np.quantile(walls @ control_side, .99))
    side_wall = walls[abs(walls @ control_side - outer) < .002]
    if len(side_wall) < 100 or np.ptp(side_wall[:, 2]) < .10:
        raise LookupError('control-side outer wall is not sufficiently observed')
    front = float(np.median(panel @ outward))
    # Visible mesh outer corner: X=172.292 mm, Y=-110.422 mm;
    # physical inner floor centre: X=-40 mm, Y=-4 mm.
    center = control_side*(outer-.212292) + outward*(front-.106422)
    floor = roof_z - .186911 + .024
    ceiling = roof_z - .186911 + .168
    center[2] = (floor+ceiling)/2
    axes = np.column_stack((width, outward, [0., 0., 1.]))
    extents = np.array((.197, .158, ceiling-floor))
    half = abs(axes) @ (extents/2)
    result = SceneObject('microwave', center, axes, extents, center-half, center+half,
                         .8, len(points), surface_points_world=points)
    return result, dict(strategy='rgbd_control_corner_and_roof_plus_public_wall_geometry',
                        roof_z_m=roof_z, floor_z_m=floor, front_plane_coordinate_m=front,
                        cavity_center_world_m=center.tolist(), outward_world=outward.tolist(),
                        control_side_world=control_side.tolist(), panel_points=len(panel),
                        side_wall_points=len(side_wall))


def observed_microwave_cavity(perception, observation, reference_position):
    from ..integration.adapters import AnchorMicrowaveDoorDetector
    from ..contact.detectors import MicrowaveDoorHandleDetector

    detector = AnchorMicrowaveDoorDetector(perception)
    center, axes, half = detector._select_fixture_geometry(observation)
    frame = MicrowaveDoorHandleDetector._fixture_frame(reference_position, center, axes, half)
    points = detector._fixture_surface_points_world
    if points is None:
        raise LookupError('microwave body lacks a fresh RGB-D surface')
    return microwave_cavity_from_walls(points, frame)


def observed_door_panel(observation, hinge, control_side, outward, *, reference_angle=None, open_only=False):
    """Fit the visible vertical panel in a measured appliance hinge frame.

    A swept physical door slab excludes the static control panel and the
    cavity back wall. Only calibrated dark, neutral RGB-D surfaces enter the
    fit; the angle is measured anew, never read from a simulator joint.
    """
    from ..common import backproject_depth

    clouds = []
    for camera in observation.cameras.values():
        rgb = camera.rgb.astype(float)
        mask = (rgb.max(axis=2) < 150) & (np.ptp(rgb, axis=2) < 40)
        cloud = backproject_depth(camera, mask=mask, world=True)
        delta = cloud - hinge
        cloud = cloud[(abs(delta[:, 2]) < .087)
                      & (np.linalg.norm(delta[:, :2], axis=1) < .275)
                      & (np.linalg.norm(delta[:, :2], axis=1) > .035)
                      & (delta @ outward > (.040 if open_only else -.025))
                      & (np.linalg.norm(cloud-observation.proprio.ee_position_world, axis=1) > .045)]
        if len(cloud):
            clouds.append(cloud)
    if not clouds:
        raise LookupError('no visible microwave panel samples')
    points = np.concatenate(clouds)
    _, unique = np.unique(np.rint(points/.004).astype(int), axis=0, return_index=True)
    points = points[unique]
    delta = points-hinge
    candidates = []
    for angle in np.linspace(-2.12, .12, 225):
        if open_only and angle > -.20:
            continue
        if reference_angle is not None and abs(angle-reference_angle) > .35:
            continue
        radial = np.cos(angle)*control_side-np.sin(angle)*outward
        normal = np.sin(angle)*control_side+np.cos(angle)*outward
        along, depth = delta @ radial, delta @ normal
        selected = (along > .040) & (along < .245) & (abs(depth-.013) < .017)
        fit = points[selected]
        if len(fit) < 35:
            continue
        span = np.ptp(fit @ radial)
        height = np.ptp(fit[:, 2])
        if span < .095 or height < .085:
            continue
        residual = float(np.median(abs(depth[selected]-.013)))
        candidates.append((len(fit), -residual, float(angle), radial, normal, span, height))
    if not candidates:
        raise LookupError('no supported vertical microwave panel within the physical sweep')
    count, residual, angle, radial, normal, span, height = max(candidates, key=lambda row: row[:2])
    return angle, radial, normal, dict(
        strategy='rgbd_vertical_panel_in_frozen_physical_hinge_sweep',
        angle_rad=angle, inlier_voxels=count, cloud_voxels=len(points),
        radial_span_m=float(span), vertical_span_m=float(height),
        median_plane_error_m=-residual,
        leading_candidates=[{'angle_rad':row[2], 'inlier_voxels':row[0]}
                            for row in sorted(candidates,key=lambda row:row[:2],reverse=True)[:8]],
    )
