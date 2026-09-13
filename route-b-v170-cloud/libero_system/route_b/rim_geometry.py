"""Fit a visible upright vessel rim from its metric surface cloud."""
from __future__ import annotations

import numpy as np


def fit_visible_upper_rim(points: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return XY centre, radius and height of a sufficiently observed top arc.

    A narrow upper band excludes the handle and lower body of an upright mug.
    The residual and angular-support gates reject an edge or a tiny fragment
    whose circle centre would otherwise be poorly constrained.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 40:
        raise ValueError("too few metric surface points for an upper rim")
    if not np.all(np.isfinite(points)):
        raise ValueError("non-finite upper-rim cloud")
    top = float(np.quantile(points[:, 2], 0.99))
    band = points[points[:, 2] >= top - 0.012, :2]
    if len(band) < 24:
        raise ValueError("too few upper-arc samples")
    origin = np.mean(band, axis=0)
    local = band - origin
    design = np.column_stack((2.0 * local, np.ones(len(local))))
    solution, _, rank, _ = np.linalg.lstsq(
        design, np.sum(local * local, axis=1), rcond=None
    )
    squared_radius = float(solution[2] + solution[:2] @ solution[:2])
    if rank != 3 or squared_radius <= 0.0:
        raise ValueError("upper arc does not constrain a circle")
    center = solution[:2] + origin
    radius = float(np.sqrt(squared_radius))
    if not 0.025 <= radius <= 0.065:
        raise ValueError("upper-rim radius outside vessel range")
    residual = np.abs(np.linalg.norm(band - center, axis=1) - radius)
    if float(np.quantile(residual, 0.90)) > 0.004:
        raise ValueError("upper surface is not a circular rim")
    angles = np.sort(np.arctan2(band[:, 1] - center[1], band[:, 0] - center[0]))
    gaps = np.diff(np.r_[angles, angles[0] + 2.0 * np.pi])
    if 2.0 * np.pi - float(np.max(gaps)) < np.deg2rad(60.0):
        raise ValueError("upper arc has insufficient angular coverage")
    return center, radius, top
