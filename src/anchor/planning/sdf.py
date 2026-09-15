"""Small signed-distance primitives used by the Planning optimiser."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def _points(value: object) -> tuple[FloatArray, bool]:
    result = np.asarray(value, dtype=np.float64)
    one = result.ndim == 1
    if one:
        result = result[None, :]
    if result.ndim != 2 or result.shape[1] != 3 or not np.all(np.isfinite(result)):
        raise ValueError("points must be finite Nx3")
    return result, one


@dataclass(frozen=True)
class NearestFieldDiagnostic:
    """Identity of the field and query sample producing a minimum SDF value."""

    raw_distance_m: float
    field_index: int
    sample_index: int
    nearest_point_world_m: tuple[float, float, float]
    source_instance_id: str | None = None
    source_label: str | None = None
    field_center_world_m: tuple[float, float, float] | None = None
    field_half_extents_m: tuple[float, float, float] | None = None
    field_axes_world: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] | None = None


@dataclass(frozen=True)
class EmptySDF:
    distance_value: float = 1_000.0

    def distance(self, points_world: FloatArray) -> FloatArray:
        points, one = _points(points_world)
        result = np.full(len(points), float(self.distance_value), dtype=np.float64)
        return result[0] if one else result


@dataclass(frozen=True)
class SphereSDF:
    center: FloatArray
    radius: float

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=np.float64)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError("sphere center must be finite xyz")
        if not np.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("sphere radius must be positive")
        object.__setattr__(self, "center", center.copy())

    def distance(self, points_world: FloatArray) -> FloatArray:
        points, one = _points(points_world)
        result = np.linalg.norm(points - self.center, axis=1) - self.radius
        return result[0] if one else result


@dataclass(frozen=True)
class BoxSDF:
    center: FloatArray
    half_extents: FloatArray
    axes: FloatArray = field(default_factory=lambda: np.eye(3))
    source_instance_id: str | None = None
    source_label: str | None = None
    # Optional current-frame sensor evidence.  These samples are never used
    # by the distance function; they only let higher layers distinguish a
    # world-static surface from a camera/tool-relative reconstruction without
    # inventing points from the OBB.  Array fields deliberately do not
    # participate in dataclass equality.
    surface_points_world: FloatArray | None = field(
        default=None, repr=False, compare=False
    )
    surface_points_by_camera: tuple[tuple[str, FloatArray], ...] = field(
        default=(), repr=False, compare=False
    )

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=np.float64)
        extent = np.asarray(self.half_extents, dtype=np.float64)
        axes = np.asarray(self.axes, dtype=np.float64)
        if center.shape != (3,) or extent.shape != (3,) or axes.shape != (3, 3):
            raise ValueError("box center/half_extents/axes have invalid shapes")
        if np.any(extent <= 0) or not np.all(np.isfinite(center)) or not np.all(np.isfinite(extent)):
            raise ValueError("box values must be finite and extents positive")
        if not np.allclose(axes.T @ axes, np.eye(3), atol=2e-3):
            raise ValueError("box axes must be orthonormal")
        for name in ("source_instance_id", "source_label"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a non-empty string or None")
        points = None
        if self.surface_points_world is not None:
            points = np.asarray(self.surface_points_world, dtype=np.float64)
            if (
                points.ndim != 2
                or points.shape[1:] != (3,)
                or len(points) < 8
                or not np.all(np.isfinite(points))
            ):
                raise ValueError("surface_points_world must be finite Nx3 sensor data")
            points = points.copy()
        cameras: list[tuple[str, FloatArray]] = []
        seen: set[str] = set()
        for camera_name, camera_points in self.surface_points_by_camera:
            values = np.asarray(camera_points, dtype=np.float64)
            if (
                not isinstance(camera_name, str)
                or not camera_name.strip()
                or camera_name in seen
                or values.ndim != 2
                or values.shape[1:] != (3,)
                or len(values) < 8
                or not np.all(np.isfinite(values))
            ):
                raise ValueError(
                    "surface_points_by_camera requires unique names and finite Nx3 data"
                )
            seen.add(camera_name)
            cameras.append((camera_name, values.copy()))
        object.__setattr__(self, "center", center.copy())
        object.__setattr__(self, "half_extents", extent.copy())
        object.__setattr__(self, "axes", axes.copy())
        object.__setattr__(self, "surface_points_world", points)
        object.__setattr__(self, "surface_points_by_camera", tuple(cameras))

    def distance(self, points_world: FloatArray) -> FloatArray:
        points, one = _points(points_world)
        local = (points - self.center) @ self.axes
        q = np.abs(local) - self.half_extents
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(np.max(q, axis=1), 0.0)
        result = outside + inside
        return result[0] if one else result


class PointCloudSDF:
    """Unsigned surface samples inflated into a conservative obstacle field."""

    def __init__(self, points_world: FloatArray, inflation_radius: float = 0.01) -> None:
        points, _ = _points(points_world)
        if len(points) == 0:
            raise ValueError("point cloud cannot be empty")
        if inflation_radius < 0 or not np.isfinite(inflation_radius):
            raise ValueError("inflation_radius must be finite and non-negative")
        self.points = points.copy()
        self.inflation_radius = float(inflation_radius)
        try:
            from scipy.spatial import cKDTree

            self._tree = cKDTree(self.points)
        except ImportError:  # pragma: no cover - the supported environment has SciPy
            self._tree = None

    def distance(self, points_world: FloatArray) -> FloatArray:
        points, one = _points(points_world)
        if self._tree is not None:
            nearest = self._tree.query(points, k=1, workers=1)[0]
        else:  # bounded fallback for minimal installations
            nearest = np.min(
                np.linalg.norm(points[:, None, :] - self.points[None, :, :], axis=-1), axis=1
            )
        result = nearest - self.inflation_radius
        return result[0] if one else result


class CompositeSDF:
    def __init__(self, fields: Sequence[object]) -> None:
        if not fields:
            raise ValueError("CompositeSDF requires at least one field")
        if any(not callable(getattr(field, "distance", None)) for field in fields):
            raise TypeError("each field must implement distance(points)")
        self.fields = tuple(fields)

    def distance(self, points_world: FloatArray) -> FloatArray:
        values = np.stack([field.distance(points_world) for field in self.fields], axis=0)
        return np.min(values, axis=0)

    def nearest_field_diagnostic(
        self, points_world: FloatArray
    ) -> NearestFieldDiagnostic:
        """Return deterministic provenance for the minimum over fields and samples."""

        points, _ = _points(points_world)
        values = np.stack(
            [np.asarray(field.distance(points), dtype=np.float64) for field in self.fields],
            axis=0,
        )
        flat_index = int(np.argmin(values))
        field_index, sample_index = np.unravel_index(flat_index, values.shape)
        nearest = self.fields[int(field_index)]
        center = getattr(nearest, "center", None)
        half_extents = getattr(nearest, "half_extents", None)
        axes = getattr(nearest, "axes", None)
        return NearestFieldDiagnostic(
            raw_distance_m=float(values[field_index, sample_index]),
            field_index=int(field_index),
            sample_index=int(sample_index),
            nearest_point_world_m=tuple(
                float(value) for value in points[sample_index]
            ),
            source_instance_id=getattr(nearest, "source_instance_id", None),
            source_label=getattr(nearest, "source_label", None),
            field_center_world_m=(
                tuple(float(value) for value in np.asarray(center, dtype=np.float64))
                if center is not None
                else None
            ),
            field_half_extents_m=(
                tuple(
                    float(value)
                    for value in np.asarray(half_extents, dtype=np.float64)
                )
                if half_extents is not None
                else None
            ),
            field_axes_world=(
                tuple(
                    tuple(float(value) for value in row)
                    for row in np.asarray(axes, dtype=np.float64)
                )
                if axes is not None
                else None
            ),
        )
