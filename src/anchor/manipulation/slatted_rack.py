"""Recover the two parallel tilted shelves of a slatted rack from RGB-D."""
import numpy as np
from scipy.spatial import cKDTree


def _tilted_planes(points):
    points = np.asarray(points, dtype=float)
    if len(points) > 12000:
        points = points[np.linspace(0, len(points)-1, 12000).astype(int)]
    tree = cKDTree(points)
    indices = np.linspace(0, len(points)-1, min(1500, len(points))).astype(int)
    distance, neighbors = tree.query(points, k=16)
    neighborhoods = points[neighbors]
    centers = neighborhoods.mean(axis=1)
    centered = neighborhoods - centers[:, None, :]
    values, vectors = np.linalg.eigh(np.einsum('nki,nkj->nij', centered, centered))
    normals = vectors[:, :, 0]
    normals[normals[:, 2] < 0] *= -1
    eligible = ((normals[:, 2] > .65) & (normals[:, 2] < .95)
                & (distance[:, -1] < .045) & (values[:, 0] < values[:, 1] * .1))
    eligible_indices = indices[eligible[indices]]
    ranked = []
    for center, normal in zip(centers[eligible_indices], normals[eligible_indices]):
        delta = points - center
        for _ in range(2):
            members = ((np.abs(delta @ normal) < .003) & (np.linalg.norm(delta, axis=1) < .22)
                       & (np.abs(normals @ normal) > .97))
            if members.sum() < 100:
                break
            cloud = points[members]
            center = cloud.mean(axis=0)
            _, vectors = np.linalg.eigh(np.cov(cloud, rowvar=False))
            normal = vectors[:, 0]
            if normal[2] < 0:normal *= -1
            delta = points-center
        if not .65 < normal[2] < .95:
            continue
        members = ((np.abs(delta @ normal) < .0025) & (np.linalg.norm(delta, axis=1) < .22)
                   & (np.abs(normals @ normal) > .97))
        if members.sum() < 120:
            continue
        cloud = points[members]
        horizontal = np.cross(normal, (0., 0., 1.));horizontal /= np.linalg.norm(horizontal)
        upslope = np.cross(normal, horizontal)
        axes = np.column_stack((horizontal, upslope, normal))
        lo, hi = np.quantile(cloud @ axes, (.02, .98), axis=0)
        span = hi-lo
        if not (.15 < span[0] < .35 and .065 < span[1] < .22 and span[2] < .005):
            continue
        midpoint = axes @ ((lo+hi)/2)
        ranked.append((len(cloud), midpoint, axes, span, cloud))
    ranked.sort(key=lambda row: -row[0])
    selected=[]
    for row in ranked:
        if any(abs(row[2][:,2] @ old[2][:,2]) > .98
               and abs((row[1]-old[1]) @ old[2][:,2]) < .012 for old in selected):
            continue
        selected.append(row)
    return selected



def slatted_rack_from_points(points, approach_from):
    from .models import SceneObject
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 300 or not np.all(np.isfinite(points)):
        raise ValueError("rack requires a finite RGB-D cloud")
    candidates = _tilted_planes(points)
    pairs = []
    for i, first in enumerate(candidates):
        for second in candidates[i + 1:]:
            lower, upper = sorted((first, second), key=lambda item: item[1][2])
            delta = upper[1] - lower[1]
            normal = upper[2][:, 2]
            if (normal @ lower[2][:, 2] > .97 and .070 < delta @ normal < .135
                    and .070 < delta[2] < .165
                    and abs(delta @ upper[2][:, 0]) < .065):
                pairs.append((first[0] + second[0], upper))
    if not pairs:
        raise LookupError("two parallel tilted rack surfaces were not observed")
    _, (count, center, axes, spans, cloud) = max(pairs, key=lambda item: item[0])
    axes = axes.copy()
    # Point the horizontal width axis toward the current hand, so its palm
    # approaches from the near end when placing an elongated bottle sideways.
    if axes[:, 0] @ (np.asarray(approach_from) - center) < 0:
        axes[:, :2] *= -1
    spans = spans.copy();spans[2] = max(.001, spans[2])
    half = np.abs(axes) @ (spans / 2)
    return SceneObject("wine rack", center, axes, spans, center-half, center+half,
                       .8, count, surface_points_world=cloud)


def slatted_rack_surface(observation):
    from ..perception.adapters import coerce_rgbd_frame
    from ..perception.geometry import backproject_frame
    frame = observation.cameras["agentview"]
    points = backproject_frame(coerce_rgbd_frame(frame, name="agentview"), stride=2).points_world
    delta = points - observation.robot.ee_pose.position
    local = (np.linalg.norm(delta[:, :2], axis=1) < .8) & (np.abs(delta[:, 2]) < .5)
    return slatted_rack_from_points(points[local], observation.robot.ee_pose.position)


def slatted_bottle_rotation(source, destination, robot, held_offset, preplace_height):
    """Choose an equivalent bottle roll with room to the public joint limits."""
    tool_z = -destination.axes_world[:, 0]
    jaw = destination.axes_world[:, 1]
    preferred = np.column_stack((np.cross(jaw, tool_z), jaw, tool_z))
    joints = getattr(robot, "joint_position", None)
    if joints is None:
        return preferred
    from ..common.panda_kinematics import path_joint_margin

    def margin(rotation):
        delta = rotation @ robot.ee_pose.rotation.T
        normal = destination.axes_world[:, 2]
        half_height = float(np.abs(normal @ delta @ source.axes_world) @ source.extents_m) / 2
        center = destination.centroid_world + normal * (half_height + .008)
        release = center - delta @ held_offset
        high = release + np.array((0., 0., preplace_height))
        return path_joint_margin(joints, robot.ee_pose.matrix, (high, release), rotation)

    preferred_margin = margin(preferred)
    if preferred_margin >= .10:
        return preferred
    alternate = preferred @ np.diag((-1., -1., 1.))
    alternate_margin = margin(alternate)
    if alternate_margin >= .10 and alternate_margin > preferred_margin + .10:
        return alternate
    return preferred
