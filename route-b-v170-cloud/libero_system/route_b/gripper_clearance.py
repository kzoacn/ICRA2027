"""Public Panda hand envelopes for comparing RGB-D approach candidates."""
import numpy as np

from .models import Pose


def panda_approach_occupancy(points: np.ndarray, pose: Pose, aperture: float) -> int:
    """Count observed points in conservative palm/finger envelopes along descent.

    Dimensions come from the public Panda hand/finger meshes and 97 mm grip
    site offset. This is a candidate ranking score, not a collision-free proof.
    """
    boxes = [
        (np.array((-.032, -.104, -.123)), np.array((.032, .101, -.031))),
        (np.array((-.011, aperture / 2 - .001, -.045)),
         np.array((.011, aperture / 2 + .027, .010))),
        (np.array((-.011, -aperture / 2 - .027, -.045)),
         np.array((.011, -aperture / 2 + .001, .010))),
    ]
    count = 0
    for height in np.linspace(0.0, 0.10, 8):
        local = (points - pose.position - np.array((0.0, 0.0, height))) @ pose.rotation
        occupied = np.zeros(len(points), dtype=bool)
        for lower, upper in boxes:
            occupied |= np.all((local > lower) & (local < upper), axis=1)
        count += int(np.count_nonzero(occupied))
    return count


def choose_flat_pinch_pose(points, position, rotation, long_axis, short_axis, extents):
    """Keep the old thin-package grasp unless a clearer nearby pose is observed."""
    points = np.asarray(points, dtype=np.float64)
    long_extent, short_extent = map(float, extents)
    aperture = max(.030, short_extent + .014)
    long_offset = max(0.0, min(.020, long_extent / 2.0 - .018))
    short_offset = max(0.0, min(.010, short_extent / 2.0 - .010))
    candidates = []
    for degrees in (-20.0, 0.0, 20.0, -40.0, 40.0):
        angle = np.deg2rad(degrees)
        cosine, sine = np.cos(angle), np.sin(angle)
        tilted = rotation @ np.array(((cosine, 0., sine), (0., 1., 0.), (-sine, 0., cosine)))
        for along in (0.0, -long_offset, long_offset):
            for across in (0.0, -short_offset, short_offset):
                offset = along * long_axis + across * short_axis
                pose = Pose(position + offset, tilted)
                score = panda_approach_occupancy(points, pose, aperture)
                candidates.append((score, float(np.linalg.norm(offset)), abs(degrees + 20.0),
                                   degrees, pose))
    baseline = candidates[0]
    selected = min(candidates, key=lambda item: item[:3])
    if baseline[0] < 10 or selected[0] > 0.50 * baseline[0]:
        selected = baseline
    return selected[4], {
        "flat_pinch_pitch_rad": float(np.deg2rad(selected[3])),
        "flat_pinch_palm_baseline_occupancy": baseline[0],
        "flat_pinch_palm_selected_occupancy": selected[0],
        "flat_pinch_clearance_offset_world_m": (selected[4].position - position).tolist(),
    }
