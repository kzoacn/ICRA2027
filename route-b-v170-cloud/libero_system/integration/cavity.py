"""Sensor-only geometry helpers for top-open fixture cavities.

The RGB-D geometry builder estimates oriented boxes with PCA.  PCA axes are
unsigned: an otherwise identical observation may return either ``a`` or
``-a``.  The helpers in this module deliberately treat planar axes as lines,
not directed vectors, so a sign flip cannot rotate a Panda wrist by 180
degrees or switch which rim is approached.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from libero_system.route_c.perception import SceneEntity


FloatArray = NDArray[np.float64]


class CavityGeometryError(ValueError):
    """Raised when visible geometry cannot define a safe top-open cavity."""


def _finite_vector(value: object, name: str) -> FloatArray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise CavityGeometryError(f"{name} must be a finite xyz vector")
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise CavityGeometryError(f"{name} must be non-zero")
    return vector / norm


def _canonical_unsigned_axis(axis: FloatArray) -> FloatArray:
    """Choose one deterministic representative of the line ``{a, -a}``."""

    result = np.asarray(axis, dtype=np.float64).copy()
    pivot = int(np.argmax(np.abs(result)))
    if result[pivot] < 0.0:
        result *= -1.0
    return result


def _validated_pose(value: object) -> FloatArray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise CavityGeometryError("world_from_ee must be a finite 4x4 pose")
    if not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
        raise CavityGeometryError("world_from_ee has an invalid homogeneous row")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
        raise CavityGeometryError("world_from_ee rotation must be orthonormal")
    if float(np.linalg.det(rotation)) < 0.0:
        raise CavityGeometryError("world_from_ee rotation must be right handed")
    return pose.copy()


@dataclass(frozen=True)
class CavityFrame:
    """Right-handed top-open cavity frame inferred only from visible RGB-D.

    The frame columns are ``(lateral, depth, up)``.  ``lateral`` follows one
    of the two planar OBB lines.  Callers may disambiguate those lines with a
    sensor/proprioception-derived approach hint; otherwise the longer stable
    direction is used.  The direction *sign* has no physical meaning and is
    canonicalised solely for deterministic output.
    """

    world_from_cavity: FloatArray
    half_extents_m: FloatArray

    def __post_init__(self) -> None:
        pose = _validated_pose(self.world_from_cavity)
        extents = np.asarray(self.half_extents_m, dtype=np.float64)
        if (
            extents.shape != (3,)
            or not np.all(np.isfinite(extents))
            or np.any(extents <= 0.0)
        ):
            raise CavityGeometryError(
                "cavity half_extents_m must contain three finite positive values"
            )
        object.__setattr__(self, "world_from_cavity", pose)
        object.__setattr__(self, "half_extents_m", extents.copy())

    @property
    def center_world(self) -> FloatArray:
        return self.world_from_cavity[:3, 3].copy()

    @property
    def lateral_axis_world(self) -> FloatArray:
        return self.world_from_cavity[:3, 0].copy()

    @property
    def depth_axis_world(self) -> FloatArray:
        return self.world_from_cavity[:3, 1].copy()

    @property
    def up_axis_world(self) -> FloatArray:
        return self.world_from_cavity[:3, 2].copy()


def infer_cavity_frame(
    reference: SceneEntity,
    *,
    world_up: object = (0.0, 0.0, 1.0),
    lateral_hint_world: object | None = None,
    min_up_alignment: float = 0.70,
    min_planar_half_extent_m: float = 0.012,
    min_stable_planar_aspect_ratio: float = 1.10,
) -> CavityFrame:
    """Infer a stable cavity frame from a sensor-derived entity OBB.

    When ``lateral_hint_world`` is supplied, the planar OBB line with the
    greatest absolute alignment to that hint becomes ``lateral``.  This is
    useful for a drawer rim grasp: the initial horizontal EE-to-object vector
    is public proprioception plus RGB-D geometry and distinguishes the
    reachable left/right opening direction from the drawer's front/back
    depth, even when depth happens to be the longer visible component.

    Without a hint, the longest sufficiently horizontal OBB axis becomes
    ``lateral``.  For a nearly square footprint PCA cannot determine a stable
    principal axis, so aspect ratios below
    ``min_stable_planar_aspect_ratio`` use the projected world-X line as a
    deterministic fallback.  Output half extents are recomputed with the OBB
    support function and therefore remain correct after PCA sign flips, axis
    selection, or the square-footprint fallback.
    """

    if not isinstance(reference, SceneEntity):
        raise CavityGeometryError("reference must be a SceneEntity")
    if reference.region is None:
        raise CavityGeometryError(
            f"visual reference {reference.instance_id!r} has no target region"
        )
    if (
        not math.isfinite(float(min_up_alignment))
        or not 0.0 < float(min_up_alignment) <= 1.0
    ):
        raise CavityGeometryError("min_up_alignment must be in (0, 1]")
    if (
        not math.isfinite(float(min_planar_half_extent_m))
        or float(min_planar_half_extent_m) <= 0.0
    ):
        raise CavityGeometryError("min_planar_half_extent_m must be positive")
    if (
        not math.isfinite(float(min_stable_planar_aspect_ratio))
        or float(min_stable_planar_aspect_ratio) <= 1.0
    ):
        raise CavityGeometryError(
            "min_stable_planar_aspect_ratio must be greater than 1"
        )

    region = reference.region
    up_hint = _finite_vector(world_up, "world_up")
    surface_up = _finite_vector(region.surface_normal, "region.surface_normal")
    if float(np.dot(surface_up, up_hint)) < 0.0:
        surface_up *= -1.0
    if float(np.dot(surface_up, up_hint)) < float(min_up_alignment):
        raise CavityGeometryError("visible cavity surface is not sufficiently top-open")

    axes = np.asarray(region.axes, dtype=np.float64)
    half_extents = np.asarray(region.half_extents, dtype=np.float64)
    if (
        axes.shape != (3, 3)
        or not np.all(np.isfinite(axes))
        or not np.allclose(axes.T @ axes, np.eye(3), atol=2e-3)
    ):
        raise CavityGeometryError("reference region axes must be finite and orthonormal")
    if (
        half_extents.shape != (3,)
        or not np.all(np.isfinite(half_extents))
        or np.any(half_extents <= 0.0)
    ):
        raise CavityGeometryError("reference region half extents must be positive")

    vertical_index = int(np.argmax(np.abs(axes.T @ surface_up)))
    planar_indices = [index for index in range(3) if index != vertical_index]
    projected: list[tuple[int, FloatArray]] = []
    for index in planar_indices:
        axis = axes[:, index] - surface_up * float(np.dot(axes[:, index], surface_up))
        norm = float(np.linalg.norm(axis))
        if norm > 0.25:
            projected.append((index, axis / norm))
    if len(projected) != 2:
        raise CavityGeometryError("reference OBB has no stable planar cavity axes")

    projected.sort(key=lambda item: float(half_extents[item[0]]), reverse=True)
    long_index, lateral = projected[0]
    short_index, _ = projected[1]
    long_extent = float(half_extents[long_index])
    short_extent = float(half_extents[short_index])
    if min(long_extent, short_extent) < float(min_planar_half_extent_m):
        raise CavityGeometryError("visible cavity footprint is too small")

    if lateral_hint_world is not None:
        hint = _finite_vector(lateral_hint_world, "lateral_hint_world")
        hint -= surface_up * float(np.dot(hint, surface_up))
        hint_norm = float(np.linalg.norm(hint))
        if hint_norm < 0.25:
            raise CavityGeometryError(
                "lateral_hint_world is not sufficiently parallel to the cavity surface"
            )
        hint /= hint_norm

        # OBB/PCA directions are unsigned, hence absolute dot products.  The
        # extent and canonical-axis terms make the exactly bisecting case
        # deterministic without injecting a task or fixture identity.
        def hinted_axis_key(item: tuple[int, FloatArray]) -> tuple[float, float, float, float, float]:
            index, axis = item
            canonical = _canonical_unsigned_axis(axis)
            return (
                abs(float(np.dot(axis, hint))),
                float(half_extents[index]),
                abs(float(canonical[0])),
                abs(float(canonical[1])),
                abs(float(canonical[2])),
            )

        _, lateral = max(projected, key=hinted_axis_key)
    # In the repeated-eigenvalue case PCA may arbitrarily swap or rotate its
    # two planar vectors.  A world-fixed line is more stable and equally valid
    # for a square opening when no measured approach hint is available.
    elif long_extent / short_extent < float(min_stable_planar_aspect_ratio):
        world_x = np.array((1.0, 0.0, 0.0), dtype=np.float64)
        lateral = world_x - surface_up * float(np.dot(world_x, surface_up))
        if float(np.linalg.norm(lateral)) < 0.25:
            world_y = np.array((0.0, 1.0, 0.0), dtype=np.float64)
            lateral = world_y - surface_up * float(np.dot(world_y, surface_up))
        lateral /= float(np.linalg.norm(lateral))

    lateral = _canonical_unsigned_axis(lateral)
    depth = np.cross(surface_up, lateral)
    depth /= float(np.linalg.norm(depth))
    lateral = np.cross(depth, surface_up)
    lateral /= float(np.linalg.norm(lateral))
    lateral = _canonical_unsigned_axis(lateral)
    depth = np.cross(surface_up, lateral)
    depth /= float(np.linalg.norm(depth))

    rotation = np.column_stack((lateral, depth, surface_up))
    if float(np.linalg.det(rotation)) <= 0.0:
        raise CavityGeometryError("failed to construct a right-handed cavity frame")
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = region.center

    # Projection of an OBB onto a unit direction has support radius
    # sum_i h_i |a_i dot d|.  This avoids assuming PCA's vertical component is
    # always stored at a fixed column.
    output_half_extents = np.array(
        [
            float(np.sum(half_extents * np.abs(axes.T @ direction)))
            for direction in (lateral, depth, surface_up)
        ],
        dtype=np.float64,
    )
    return CavityFrame(pose, output_half_extents)


def align_panda_finger_axis(
    world_from_ee: object,
    desired_axis_world: object,
) -> FloatArray:
    """Align Panda local-Y with an unsigned cavity axis, preserving tool-Z.

    The sign closest to the current local-Y direction is chosen, preventing an
    unnecessary 180-degree wrist turn.  Canonicalising before that comparison
    also makes the exact orthogonal/tie case invariant to PCA sign flips.
    """

    pose = _validated_pose(world_from_ee)
    desired = _canonical_unsigned_axis(
        _finite_vector(desired_axis_world, "desired_axis_world")
    )
    tool_z = pose[:3, 2].copy()
    desired -= tool_z * float(np.dot(desired, tool_z))
    norm = float(np.linalg.norm(desired))
    if norm < 0.25:
        raise CavityGeometryError(
            "desired finger axis is too parallel to Panda tool-Z"
        )
    desired /= norm
    if float(np.dot(desired, pose[:3, 1])) < 0.0:
        desired *= -1.0

    tool_x = np.cross(desired, tool_z)
    tool_x /= float(np.linalg.norm(tool_x))
    tool_y = np.cross(tool_z, tool_x)
    tool_y /= float(np.linalg.norm(tool_y))
    aligned = pose.copy()
    aligned[:3, :3] = np.column_stack((tool_x, tool_y, tool_z))
    return aligned


__all__ = [
    "CavityFrame",
    "CavityGeometryError",
    "align_panda_finger_axis",
    "infer_cavity_frame",
]
