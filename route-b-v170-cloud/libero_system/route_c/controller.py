"""Executable phased Route C controller with MPC replanning and recovery."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
import math
from typing import Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from ..common.grasp_journal import (
    GraspAttemptEvent,
    GraspAttemptJournal,
    GraspEvidenceCategory,
    GraspReason,
    JawBehavior,
    PendingGraspEngagement,
)
from .compiler import TemplateConstraintCompiler
from .grasp import (
    BoundConstraintGraph,
    GraspBinder,
    GraspBindingError,
    GraspCandidate,
    GraspProvider,
)
from .optimizer import (
    CostWeights,
    MotionRequest,
    OptimisationError,
    RecedingHorizonOptimizer,
)
from .perception import (
    EntityResolver,
    PerceptionError,
    SceneEntity,
    SceneEstimate,
    SceneEstimator,
    TargetRegion,
)
from .schema import ConstraintGraph, ConstraintKind, Phase, Relation


FloatArray = NDArray[np.float64]


_RELATIVE_PLACEMENT_GAP_M = 0.012
_LOW_PROFILE_LANDMARK_MAX_HALF_HEIGHT_M = 0.030
_LOW_PROFILE_LANDMARK_MAX_ASPECT_RATIO = 0.35
_PLANAR_PLACEMENT_RELATIONS = frozenset(
    {
        Relation.LEFT_OF,
        Relation.RIGHT_OF,
        Relation.FRONT_OF,
        Relation.BEHIND,
    }
)


# Normalized cavity geometry from the public
# ``wooden_two_layer_shelf.xml`` asset.  Values are expressed relative to the
# collision AABB measured by RGB-D/static asset completion, so they follow the
# sensed fixture pose and scale and never consult the active simulator.  The
# two sites are mechanically asymmetric: the upper bay occupies most of the
# top half, while the lower bay lies below the middle shelf.
_SHELF_CAVITY_PRIORS: Mapping[str, tuple[float, float, float]] = {
    # (centre offset / fixture half-height,
    #  cavity half-height / fixture half-height,
    #  horizontal half-extent scale)
    "upper_shelf": (0.307, 0.646, 0.86),
    "lower_shelf": (-0.867, 0.338, 0.86),
}


class ExecutionError(RuntimeError):
    pass


class TerminalExecutionError(RuntimeError):
    """Stop a Route C run immediately while preserving its attempt journal."""


def _semantic_planar_axes(target: SceneEntity) -> tuple[FloatArray, FloatArray]:
    """Return instruction-space front/right axes from public camera geometry."""

    world_up = (
        np.asarray(target.region.surface_normal, dtype=np.float64)
        if target.region is not None
        else np.array((0.0, 0.0, 1.0), dtype=np.float64)
    )
    world_up /= float(np.linalg.norm(world_up))

    raw_right = target.keypoints.get("view_right_axis")
    raw_forward = target.keypoints.get("view_forward_axis")
    right = None
    front = None
    if raw_right is not None:
        candidate = np.asarray(raw_right, dtype=np.float64).copy()
        if candidate.shape == (3,) and np.all(np.isfinite(candidate)):
            candidate -= world_up * float(np.dot(candidate, world_up))
            norm = float(np.linalg.norm(candidate))
            if norm >= 1e-6:
                right = candidate / norm
    if raw_forward is not None:
        candidate = -np.asarray(raw_forward, dtype=np.float64).copy()
        if candidate.shape == (3,) and np.all(np.isfinite(candidate)):
            candidate -= world_up * float(np.dot(candidate, world_up))
            norm = float(np.linalg.norm(candidate))
            if norm >= 1e-6:
                front = candidate / norm

    if front is None:
        view_origin = target.keypoints.get("view_origin")
        if view_origin is not None:
            candidate = np.asarray(view_origin, dtype=np.float64) - target.position
            candidate -= world_up * float(np.dot(candidate, world_up))
            norm = float(np.linalg.norm(candidate))
            if norm >= 1e-6:
                front = candidate / norm
    if front is None and right is not None:
        front = np.cross(right, world_up)
        front /= float(np.linalg.norm(front))
    if right is None and front is not None:
        right = np.cross(world_up, front)
        right /= float(np.linalg.norm(right))
    if front is None or right is None:
        # Public LIBERO world-frame fallback: front is +X and right is +Y.
        return (
            np.array((1.0, 0.0, 0.0), dtype=np.float64),
            np.array((0.0, 1.0, 0.0), dtype=np.float64),
        )

    # Camera pitch can make independently projected optical and image axes
    # slightly non-orthogonal.  Keep optical front, then Gram-Schmidt right
    # while preserving its calibrated sign.
    right_hint = right.copy()
    right -= front * float(np.dot(right, front))
    norm = float(np.linalg.norm(right))
    if norm < 1e-6:
        right = np.cross(world_up, front)
    else:
        right /= norm
    if float(np.dot(right, right_hint)) < 0.0:
        right *= -1.0
    return front, right


def _semantic_planar_direction(
    target: SceneEntity, relation: Relation
) -> FloatArray:
    front, right = _semantic_planar_axes(target)
    directions = {
        Relation.LEFT_OF: -right,
        Relation.RIGHT_OF: right,
        Relation.FRONT_OF: front,
        Relation.BEHIND: -front,
    }
    try:
        return directions[relation]
    except KeyError as exc:
        raise ExecutionError(
            f"relation {relation.value!r} has no planar language direction"
        ) from exc


@dataclass(frozen=True)
class _RelativePlacementGeometry:
    """Sensor-derived clearance geometry for a planar language relation."""

    direction: FloatArray
    source_radius_m: float
    target_radius_m: float
    gap_m: float
    required_separation_m: float
    orthogonal_limit_m: float
    low_profile_landmark: bool


def _obb_support_radius(
    axes: FloatArray,
    half_extents: FloatArray,
    direction: FloatArray,
) -> float:
    """Return an OBB support radius along a normalized world direction."""

    world_direction = np.array(direction, dtype=np.float64, copy=True)
    norm = float(np.linalg.norm(world_direction))
    if not math.isfinite(norm) or norm < 1e-8:
        raise ExecutionError("relative-placement direction is degenerate")
    world_direction /= norm
    return float(
        np.sum(
            np.asarray(half_extents, dtype=np.float64)
            * np.abs(np.asarray(axes, dtype=np.float64).T @ world_direction)
        )
    )


def _relative_placement_geometry(
    source: SceneEntity,
    target: SceneEntity,
    target_region: TargetRegion,
    relation: Relation,
    params: Mapping[str, float | str],
) -> _RelativePlacementGeometry:
    """Compute a stable centre slot for a planar language relation.

    The compiler's historical ``relative_offset`` remains a backward-compatible
    minimum for externally supplied graphs.  Tall landmarks use conservative
    OBB clearance.  A low, plate-like landmark is different: its single-view
    RGB-D OBB radius is noisy and using that radius as a centre offset can move
    the object farther every time the plate rim is completed.  For such a
    landmark the language offset plus the requested margin defines a stable
    tabletop slot.  MuJoCo / SDF collision checking still protects the actual
    motion; the semantic endpoint no longer follows a swollen one-frame OBB.
    """

    if relation not in _PLANAR_PLACEMENT_RELATIONS:
        raise ExecutionError(
            f"relation {relation.value!r} has no planar placement geometry"
        )
    direction = _semantic_planar_direction(target, relation)
    source_radius = _obb_support_radius(
        source.pose[:3, :3], 0.5 * source.extent, direction
    )
    target_radius = _obb_support_radius(
        target_region.axes, target_region.half_extents, direction
    )
    gap = float(params.get("xy_margin", _RELATIVE_PLACEMENT_GAP_M))
    legacy_minimum = float(params.get("relative_offset", 0.0))
    if not math.isfinite(gap) or gap < 0.0:
        raise ExecutionError("relative-placement xy_margin must be finite and non-negative")
    if not math.isfinite(legacy_minimum) or legacy_minimum < 0.0:
        raise ExecutionError(
            "relative-placement relative_offset must be finite and non-negative"
        )

    world_up = np.asarray(target_region.surface_normal, dtype=np.float64).copy()
    world_up_norm = float(np.linalg.norm(world_up))
    if not math.isfinite(world_up_norm) or world_up_norm < 1e-8:
        raise ExecutionError("relative-placement target surface normal is degenerate")
    world_up /= world_up_norm
    target_vertical_radius = _obb_support_radius(
        target.pose[:3, :3], 0.5 * target.extent, world_up
    )
    target_planar_radius = max(target_radius, 1e-6)
    low_profile_landmark = bool(
        legacy_minimum > 0.0
        and target_vertical_radius <= _LOW_PROFILE_LANDMARK_MAX_HALF_HEIGHT_M
        and target_vertical_radius
        <= _LOW_PROFILE_LANDMARK_MAX_ASPECT_RATIO * target_planar_radius
        and target_radius <= legacy_minimum + 2.0 * gap
        and source_radius + gap <= legacy_minimum
    )

    footprint_separation = source_radius + target_radius
    if low_profile_landmark:
        required = max(legacy_minimum + gap, source_radius + gap)
    else:
        required = max(footprint_separation + gap, legacy_minimum)

    orthogonal_direction = np.cross(world_up, direction)
    orthogonal_norm = float(np.linalg.norm(orthogonal_direction))
    if not math.isfinite(orthogonal_norm) or orthogonal_norm < 1e-8:
        raise ExecutionError("relative-placement planar axes are degenerate")
    # The same explicit margin is the cross-axis slot width.  The synthesizer
    # targets zero orthogonal displacement, so accepting a source-width band
    # here would hide target drift and reproduce the unbounded-half-plane bug.
    orthogonal_limit = gap
    return _RelativePlacementGeometry(
        direction=direction,
        source_radius_m=source_radius,
        target_radius_m=target_radius,
        gap_m=gap,
        required_separation_m=required,
        orthogonal_limit_m=orthogonal_limit,
        low_profile_landmark=low_profile_landmark,
    )


def _target_subregion_offset(
    target: SceneEntity,
    params: Mapping[str, float | str],
) -> FloatArray:
    """Resolve a semantic subregion in a visible target-local frame."""

    if target.region is None:
        raise ExecutionError("placement target has no RGB-D target region")
    name = params.get("target_subregion")
    if name is None:
        # Backward-compatible numeric graph fields; new planner output uses
        # the calibrated visual-local subregion below.
        offset_x = float(params.get("world_offset_x_fraction", 0.0))
        offset_y = float(params.get("world_offset_y_fraction", 0.0))
        axes = target.region.axes
        half_extents = target.region.half_extents

        def support(direction: FloatArray) -> float:
            return float(np.sum(half_extents * np.abs(axes.T @ direction)))

        world_x = np.array([1.0, 0.0, 0.0])
        world_y = np.array([0.0, 1.0, 0.0])
        return (
            world_x * offset_x * support(world_x)
            + world_y * offset_y * support(world_y)
        )
    if name in _SHELF_CAVITY_PRIORS:
        cavity = _shelf_cavity_region(target, params)
        return cavity.center - target.region.center
    if name not in {"left", "right", "front", "back"}:
        raise ExecutionError(f"unsupported target subregion: {name!r}")

    world_up = np.asarray(target.region.surface_normal, dtype=np.float64)
    world_up /= float(np.linalg.norm(world_up))
    view_origin = target.keypoints.get("view_origin")
    if view_origin is None:
        if target.label == "caddy":
            raise ExecutionError("caddy subregion requires a calibrated visual origin")
        view = np.array([0.0, -1.0, 0.0])
    else:
        view = np.asarray(view_origin, dtype=np.float64) - target.region.center
        view -= world_up * float(np.dot(view, world_up))
        if float(np.linalg.norm(view)) < 1e-6:
            raise ExecutionError("target visual bearing is degenerate")
        view /= float(np.linalg.norm(view))

    axes = target.region.axes
    half_extents = target.region.half_extents
    if target.label == "caddy":
        horizontal: list[tuple[float, FloatArray]] = []
        for index in range(3):
            axis = axes[:, index].copy()
            axis -= world_up * float(np.dot(axis, world_up))
            norm = float(np.linalg.norm(axis))
            if norm < 0.65:
                continue
            horizontal.append((float(half_extents[index] * norm), axis / norm))
        if len(horizontal) < 2:
            raise ExecutionError("caddy RGB-D OBB has no stable horizontal frame")
        horizontal.sort(key=lambda item: item[0])
        front = horizontal[0][1].copy()
        if float(np.dot(front, view)) < 0.0:
            front *= -1.0
        long_axis = horizontal[-1][1].copy()
        left_hint = np.cross(front, world_up)
        if float(np.dot(long_axis, left_hint)) < 0.0:
            long_axis *= -1.0
        left = long_axis
        fraction = 0.67 if name in {"left", "right"} else 0.43
    else:
        front = view
        left = np.cross(front, world_up)
        left /= float(np.linalg.norm(left))
        fraction = 0.52

    direction = {
        "front": front,
        "back": -front,
        "left": left,
        "right": -left,
    }[str(name)]
    support = float(
        np.sum(half_extents * np.abs(axes.T @ direction))
    )
    offset = direction * fraction * support
    if target.label == "caddy" and name == "front":
        # The public desk-caddy geometry has a low front pocket: its centre is
        # 45 mm below the three tall compartments, about 0.53 of the visible
        # caddy half-height.  Scale that semantic asset-family prior by the
        # measured RGB-D OBB so it remains valid under pose and size error.
        vertical_support = float(
            np.sum(half_extents * np.abs(axes.T @ world_up))
        )
        offset -= world_up * 0.53 * vertical_support
    return offset


def _shelf_cavity_region(
    target: SceneEntity,
    params: Mapping[str, float | str],
) -> TargetRegion:
    """Construct one shelf cavity from a sensor-derived fixture OBB.

    The instruction selects upper versus lower; position, yaw, and scale come
    from the current RGB-D entity.  Public asset ratios only describe the
    fixture's fixed internal construction.  Ambiguous or implausibly small
    shelf geometry fails closed instead of silently falling back to the whole
    fixture box.
    """

    name = str(params.get("target_subregion", ""))
    try:
        centre_fraction, height_fraction, planar_fraction = _SHELF_CAVITY_PRIORS[
            name
        ]
    except KeyError as exc:
        raise ExecutionError(f"unsupported shelf cavity: {name!r}") from exc
    if target.region is None:
        raise ExecutionError("shelf cavity target has no RGB-D target region")
    if "shelf" not in " ".join(target.label.lower().replace("_", " ").split()):
        raise ExecutionError("shelf cavity semantics require a visually grounded shelf")

    region = target.region
    world_up = np.asarray(region.surface_normal, dtype=np.float64)
    world_up /= float(np.linalg.norm(world_up))
    alignment = np.abs(region.axes.T @ world_up)
    vertical_axis = int(np.argmax(alignment))
    if float(alignment[vertical_axis]) < 0.90:
        raise ExecutionError("shelf RGB-D OBB has no stable gravity-aligned axis")
    vertical_support = float(
        np.sum(region.half_extents * np.abs(region.axes.T @ world_up))
    )
    horizontal_supports = [
        float(region.half_extents[index])
        for index in range(3)
        if index != vertical_axis
    ]
    if vertical_support < 0.060 or min(horizontal_supports) < 0.055:
        raise ExecutionError("shelf RGB-D OBB is too small to localize a cavity")

    half_extents = region.half_extents.copy()
    for index in range(3):
        if index == vertical_axis:
            half_extents[index] = max(
                0.010,
                height_fraction * vertical_support / float(alignment[index]),
            )
        else:
            half_extents[index] *= planar_fraction
    center = region.center + world_up * centre_fraction * vertical_support
    return TargetRegion(center, region.axes, half_extents, world_up)


def _placement_target_region(
    target: SceneEntity,
    params: Mapping[str, float | str],
) -> TargetRegion:
    """Return the exact geometry named by a placement goal."""

    name = params.get("target_subregion")
    if name in _SHELF_CAVITY_PRIORS:
        return _shelf_cavity_region(target, params)
    if target.region is None:
        raise ExecutionError("placement target has no RGB-D target region")
    return target.region


class _SelectorViewExhausted(PerceptionError):
    """Terminal, high-clearance failure after bounded visual reacquisition."""


GRIPPER_OPEN = -1.0
GRIPPER_CLOSE = 1.0


class GraspMode(StrEnum):
    """Mechanical retention mode under LIBERO's native gripper convention."""

    PINCH = "pinch"
    RIM_PINCH = "rim_pinch"
    EXPAND = "expand"


@runtime_checkable
class GraspModeSelector(Protocol):
    def select(self, source_label: str) -> GraspMode: ...


class EntityGraspModeSelector:
    """Use a closing rim pinch for bowls and centre pinch otherwise.

    ``EXPAND`` remains available only through explicit dependency injection
    for non-bowl mechanisms.  Exact, normalised bowl labels are protected so
    no configuration can re-enable an internal expansion grasp for them.
    """

    def __init__(
        self,
        expansion_entities: Sequence[str] = (),
        *,
        rim_pinch_entities: Sequence[str] = ("black bowl", "bowl", "white bowl"),
    ) -> None:
        protected_rim_entities = frozenset({"black bowl", "bowl"})
        self.expansion_entities = frozenset(
            self._normalise(name) for name in expansion_entities
        )
        self.rim_pinch_entities = protected_rim_entities | frozenset(
            self._normalise(name) for name in rim_pinch_entities
        )
        if "" in self.expansion_entities or "" in self.rim_pinch_entities:
            raise ValueError("grasp-mode entity names must be non-empty")
        protected_overlap = self.expansion_entities & protected_rim_entities
        if protected_overlap:
            names = ", ".join(sorted(protected_overlap))
            raise ValueError(
                "closed-finger rim entities cannot use internal expansion: "
                f"{names}"
            )
        overlap = self.expansion_entities & self.rim_pinch_entities
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(f"grasp-mode entity sets overlap: {names}")

    def select(self, source_label: str) -> GraspMode:
        normalised = self._normalise(source_label)
        if normalised in self.expansion_entities:
            return GraspMode.EXPAND
        if normalised in self.rim_pinch_entities:
            return GraspMode.RIM_PINCH
        return GraspMode.PINCH

    @staticmethod
    def _normalise(label: str) -> str:
        return " ".join(label.lower().replace("_", " ").split())


@dataclass(frozen=True)
class ControllerFeedback:
    accepted: bool
    detail: str = ""
    position_error_m: float | None = None
    rotation_error_rad: float | None = None


@runtime_checkable
class RobotController(Protocol):
    """Minimal adapter implemented by a robosuite/OSC or real robot driver."""

    def current_ee_pose(self) -> FloatArray: ...

    def execute_waypoints(
        self, poses_world: FloatArray, phase: Phase, gripper_command: float
    ) -> ControllerFeedback: ...

    def set_gripper(self, gripper_command: float) -> ControllerFeedback: ...

    def grasp_confirmed(self, mode: GraspMode) -> bool: ...


@dataclass(frozen=True)
class Verification:
    success: bool
    relation: Relation
    position_error_m: float
    detail: str


@runtime_checkable
class RelationVerifier(Protocol):
    def verify(self, bound: BoundConstraintGraph, scene: SceneEstimate) -> Verification: ...


class GeometryRelationVerifier:
    """Verify visible 3-D geometry, without benchmark predicates or simulator state."""

    def __init__(self, tolerance_m: float = 0.025) -> None:
        self.tolerance_m = float(tolerance_m)

    def verify(self, bound: BoundConstraintGraph, scene: SceneEstimate) -> Verification:
        source = self._tracked_or_label(scene, bound.source_id, bound.source_label)
        target = self._tracked_or_label(scene, bound.target_id, bound.target_label)
        relation = bound.graph.goal_relation
        params = next(
            x.parameters for x in bound.graph.constraints if x.kind == ConstraintKind.GOAL_RELATION
        )
        try:
            target_region = _placement_target_region(target, params)
        except ExecutionError as exc:
            return Verification(False, relation, math.inf, str(exc))

        if relation == Relation.ON:
            local = target_region.local_coordinates(source.position[None, :])[0]
            horizontal_limit = target_region.half_extents[:2] + float(params.get("xy_margin", 0.0))
            horizontal_error = float(np.linalg.norm(np.maximum(np.abs(local[:2]) - horizontal_limit, 0.0)))
            source_bottom = source.position[2] - source.extent[2] / 2.0
            target_top = target.position[2] + target.extent[2] / 2.0
            vertical_error = abs(float(source_bottom - target_top))
            region_error = self._target_region_error(source, target, params)
            error = max(horizontal_error, vertical_error, region_error)
            return Verification(
                error <= self.tolerance_m,
                relation,
                error,
                f"horizontal={horizontal_error:.4f}, vertical={vertical_error:.4f}, "
                f"subregion={region_error:.4f}",
            )

        if relation == Relation.IN:
            margin = float(params.get("inside_margin", 0.0))
            limits = np.maximum(target_region.half_extents - margin, 1e-6)
            shelf_cavity = params.get("target_subregion") in _SHELF_CAVITY_PRIORS
            # LIBERO's public shelf regions contain an object's reference point;
            # elongated handles (notably the frying pan) are allowed to extend
            # through the open front.  Ordinary containers retain the stronger
            # all-corners containment gate.
            containment_points = (
                source.position[None, :]
                if shelf_cavity
                else self._corners(source)
            )
            local = np.abs(target_region.local_coordinates(containment_points))
            overflow = np.maximum(local - limits, 0.0)
            containment_error = float(np.max(np.linalg.norm(overflow, axis=1)))
            region_error = self._target_region_error(source, target, params)
            error = max(containment_error, region_error)
            containment_name = "cavity-centre" if shelf_cavity else "containment"
            return Verification(
                error <= self.tolerance_m,
                relation,
                error,
                f"{containment_name} overflow={containment_error:.4f}, "
                f"subregion={region_error:.4f}",
            )

        if relation == Relation.UNDER:
            source_top = source.position[2] + source.extent[2] / 2.0
            target_bottom = target.position[2] - target.extent[2] / 2.0
            vertical_error = max(float(source_top - target_bottom), 0.0)
            planar_error = float(np.linalg.norm(source.position[:2] - target.position[:2]))
            planar_allowance = float(np.linalg.norm(target.extent[:2]) / 2.0)
            error = max(vertical_error, planar_error - planar_allowance, 0.0)
            return Verification(
                error <= self.tolerance_m,
                relation,
                error,
                f"under vertical={vertical_error:.4f}, planar={planar_error:.4f}",
            )

        if relation not in _PLANAR_PLACEMENT_RELATIONS:
            return Verification(
                False,
                relation,
                math.inf,
                "relation is not geometrically implemented",
            )
        try:
            relative = _relative_placement_geometry(
                source, target, target_region, relation, params
            )
        except ExecutionError as exc:
            return Verification(False, relation, math.inf, str(exc))
        delta = source.position - target_region.center
        signed = float(np.dot(delta, relative.direction))
        signed_error = abs(signed - relative.required_separation_m)
        footprint_separation = relative.source_radius_m + relative.target_radius_m
        overlap = (
            0.0
            if relative.low_profile_landmark
            else max(footprint_separation - signed, 0.0)
        )
        world_up = np.asarray(target_region.surface_normal, dtype=np.float64)
        world_up /= float(np.linalg.norm(world_up))
        planar_delta = delta - world_up * float(np.dot(delta, world_up))
        orthogonal = float(
            np.linalg.norm(planar_delta - signed * relative.direction)
        )
        orthogonal_error = max(orthogonal - relative.orthogonal_limit_m, 0.0)
        # Directional language denotes a bounded placement slot, not an
        # unbounded half-plane.  The same desired centre used by synthesis is
        # therefore checked on both sides.  Tall landmarks additionally retain
        # the strict no-footprint-overlap gate; low plate-like landmarks rely
        # on the executed SDF / physical contact while avoiding noisy rim OBBs.
        signed_tolerance = min(
            max(self.tolerance_m, 0.0),
            relative.gap_m,
        )
        error = max(signed_error, orthogonal_error, overlap)
        success = bool(
            signed_error <= signed_tolerance + 1e-9
            and orthogonal_error <= 1e-9
            and overlap <= 1e-9
        )
        return Verification(
            success,
            relation,
            error,
            f"relative separation={signed:.4f}, required="
            f"{relative.required_separation_m:.4f}, deviation={signed_error:.4f}, "
            f"orthogonal={orthogonal:.4f}, orthogonal_limit="
            f"{relative.orthogonal_limit_m:.4f}, overlap={overlap:.4f}, "
            f"low_profile={relative.low_profile_landmark}, "
            f"signed_tolerance={signed_tolerance:.4f}",
        )

    @staticmethod
    def _target_region_error(
        source: SceneEntity,
        target: SceneEntity,
        params: Mapping[str, float | str],
    ) -> float:
        has_subregion = "target_subregion" in params
        if params.get("target_subregion") in _SHELF_CAVITY_PRIORS:
            # Shelf membership was already checked against the derived cavity
            # OBB in ``verify``.  A distance-to-centre penalty would incorrectly
            # collapse a valid 3-D cavity to a 25-mm ball.
            return 0.0
        offset_x = float(params.get("world_offset_x_fraction", 0.0))
        offset_y = float(params.get("world_offset_y_fraction", 0.0))
        if not has_subregion and not offset_x and not offset_y:
            return 0.0
        assert target.region is not None
        desired = target.region.center.copy()
        desired += _target_subregion_offset(target, params)
        distance = float(np.linalg.norm(source.position - desired))
        # The verifier's configured tolerance supplies the finite acceptance
        # radius.  Returning the raw error guarantees that the target centre
        # or opposite half cannot satisfy a requested compartment.
        return distance

    @staticmethod
    def _tracked_or_label(scene: SceneEstimate, instance_id: str, label: str) -> SceneEntity:
        try:
            return scene.by_id(instance_id)
        except PerceptionError:
            candidates = [x for x in scene.entities if x.label == label]
            if not candidates:
                raise
            return max(candidates, key=lambda x: x.confidence)

    @staticmethod
    def _corners(entity: SceneEntity) -> FloatArray:
        signs = np.array(
            [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
        )
        local = signs * (entity.extent / 2.0)
        return local @ entity.pose[:3, :3].T + entity.position


@dataclass(frozen=True)
class PhaseAttempt:
    task_attempt: int
    phase: Phase
    attempt: int
    success: bool
    detail: str


@dataclass(frozen=True, slots=True)
class _PendingGraspEngagement:
    source_text: str
    source_class: str
    grasp_mode: str
    jaw_behavior: JawBehavior


@dataclass(frozen=True, slots=True)
class _GraspTerminalHint:
    reason: GraspReason
    evidence_source: GraspEvidenceCategory


@dataclass(frozen=True)
class RouteCResult:
    success: bool
    task_text: str
    attempts: tuple[PhaseAttempt, ...]
    verification: Verification | None
    failure: str | None = None
    source_id: str | None = None
    target_id: str | None = None
    grasp_attempt_events: tuple[GraspAttemptEvent, ...] = ()


@dataclass(frozen=True)
class RouteCControllerConfig:
    max_task_attempts: int = 2
    max_phase_attempts: int = 2
    max_mpc_replans: int = 12
    # A fixture-constrained cavity descent gets at most two complete OSC
    # chunks without reaching the already configured pose/contact gate.  A
    # third blind descent was observed to consume another full chunk while
    # making no useful progress, delaying a sensor-distinct rim candidate.
    # Ordinary grasps and any chunk that reaches a typed contact gate are
    # unaffected.
    max_cavity_grasp_incomplete_chunks: int = 2
    position_tolerance_m: float = 0.008
    orientation_tolerance_rad: float = 0.10
    lift_distance_m: float = 0.11
    retreat_distance_m: float = 0.10
    placement_clearance_ceiling_z_m: float = 1.38
    # Static Panda/OSC deployment calibration.  This is intentionally
    # independent of the RGB-D perception crop and benchmark suite.
    recovery_motion_ceiling_z_m: float = 1.33
    expand_grasp_z_offset_m: float = 0.025
    observation_max_age_s: float = 1.0
    # A strict nested 2-D selector match may identify the source while the
    # fixture's partial RGB-D OBB remains unsafe to bind.  Route C may move an
    # empty hand to at most two high wrist-camera views, but only a subsequent
    # ordinary 3-D source/reference pair is allowed to reach grasp proposal.
    max_selector_active_views: int = 2
    selector_view_clearance_m: float = 0.180
    selector_view_second_retreat_m: float = 0.060
    selector_view_workspace_margin_m: float = 0.010

    def __post_init__(self) -> None:
        if min(
            self.max_task_attempts,
            self.max_phase_attempts,
            self.max_mpc_replans,
            self.max_cavity_grasp_incomplete_chunks,
        ) < 1:
            raise ValueError("attempt and replan limits must be positive")
        if self.expand_grasp_z_offset_m <= 0:
            raise ValueError("expand_grasp_z_offset_m must be positive")
        if not np.isfinite(self.recovery_motion_ceiling_z_m):
            raise ValueError("recovery motion ceiling must be finite")
        if not np.isfinite(self.placement_clearance_ceiling_z_m):
            raise ValueError("placement clearance ceiling must be finite")
        if not 0 <= self.max_selector_active_views <= 2:
            raise ValueError("selector active views must be between zero and two")
        if min(
            self.selector_view_clearance_m,
            self.selector_view_second_retreat_m,
            self.selector_view_workspace_margin_m,
        ) <= 0:
            raise ValueError("selector active-view distances must be positive")


@dataclass(frozen=True)
class _PhaseMotionAnchor:
    """Absolute phase target frozen at the first sensor/proprio binding."""

    goal_pose: FloatArray
    min_height_m: float | None


class GoalSynthesizer:
    """Turn a bound graph into per-phase SE(3) goals from fresh visual state."""

    def __init__(self, resolver: EntityResolver | None = None) -> None:
        self.resolver = resolver or EntityResolver()

    def motion_request(
        self,
        phase: Phase,
        bound: BoundConstraintGraph,
        scene: SceneEstimate,
        current_pose: FloatArray,
        config: RouteCControllerConfig,
        grasp_mode: GraspMode,
    ) -> MotionRequest:
        source = self._entity(scene, bound.source_id, bound.source_label)
        target = self._entity(scene, bound.target_id, bound.target_label)
        object_from_ee = self._effective_object_from_ee(bound, source, grasp_mode, config)
        grasp = source.pose @ object_from_ee
        goal = np.array(current_pose, dtype=np.float64, copy=True)
        pregrasp = self._parameter(bound, ConstraintKind.GRASP_BINDING, phase, "pregrasp_distance", 0.09)

        if phase == Phase.APPROACH:
            goal = grasp.copy()
            if grasp_mode == GraspMode.EXPAND:
                goal[2, 3] += pregrasp
            else:
                goal[:3, 3] -= bound.grasp.candidate.approach_world * pregrasp
        elif phase == Phase.GRASP:
            goal = grasp
        elif phase == Phase.LIFT:
            goal[:3, 3] += np.array([0.0, 0.0, config.lift_distance_m])
        elif phase == Phase.TRANSFER:
            goal = self._placement_ee_pose(bound, source, target, scene, object_from_ee)
            # Tall objects on high supports need less extra vertical travel.
            # Keep a positive clearance above the release pose while avoiding
            # a fixed 110-mm overshoot beyond the Panda's usable reach.
            goal[2, 3] += min(config.lift_distance_m, max(
                0.025, config.placement_clearance_ceiling_z_m - goal[2, 3]
            ))
        elif phase == Phase.PLACE:
            goal = self._placement_ee_pose(bound, source, target, scene, object_from_ee)
        elif phase == Phase.RETREAT:
            goal[2, 3] += min(config.retreat_distance_m, max(
                0.025, config.placement_clearance_ceiling_z_m - goal[2, 3]
            ))
        else:
            raise ExecutionError(f"phase {phase.value!r} has no motion goal")

        clearance = self._parameter(
            bound, ConstraintKind.COLLISION_CLEARANCE, phase, "clearance", 0.025
        )
        radius = self._parameter(bound, ConstraintKind.COLLISION_CLEARANCE, phase, "tool_radius", 0.025)
        margin = self._parameter(bound, ConstraintKind.REACHABILITY, phase, "margin", 0.01)
        min_height = None
        if phase == Phase.TRANSFER:
            safe_margin = self._parameter(bound, ConstraintKind.ABOVE, phase, "margin", 0.08)
            min_height = scene.scene_floor_z + safe_margin
        weights = CostWeights(
            obstacle=self._weight(bound, ConstraintKind.COLLISION_CLEARANCE, phase, 100.0),
            smoothness=self._weight(bound, ConstraintKind.SMOOTHNESS, phase, 2.0),
            reachability=self._weight(bound, ConstraintKind.REACHABILITY, phase, 120.0),
            safe_height=self._weight(bound, ConstraintKind.ABOVE, phase, 25.0),
        )
        return MotionRequest(
            phase=phase,
            start_pose=current_pose,
            goal_pose=goal,
            clearance_m=clearance,
            tool_radius_m=radius,
            workspace_margin_m=margin,
            min_height_m=min_height,
            weights=weights,
        )

    @staticmethod
    def _entity(scene: SceneEstimate, instance_id: str, label: str) -> SceneEntity:
        try:
            return scene.by_id(instance_id)
        except PerceptionError:
            matches = [x for x in scene.entities if x.label == label]
            if not matches:
                raise
            return max(matches, key=lambda x: x.confidence)

    def _placement_ee_pose(
        self,
        bound: BoundConstraintGraph,
        source: SceneEntity,
        target: SceneEntity,
        scene: SceneEstimate,
        object_from_ee: FloatArray,
    ) -> FloatArray:
        params = next(
            x.parameters for x in bound.graph.constraints if x.kind == ConstraintKind.GOAL_RELATION
        )
        target_region = _placement_target_region(target, params)
        relation = bound.graph.goal_relation
        object_goal = source.pose.copy()
        vertical = float(params.get("vertical_offset", 0.006))
        if relation == Relation.ON:
            position = target.position.copy()
            position[2] = target.position[2] + target.extent[2] / 2.0 + source.extent[2] / 2.0 + vertical
        elif relation == Relation.IN:
            shelf_cavity = params.get("target_subregion") in _SHELF_CAVITY_PRIORS
            if shelf_cavity:
                world_up = np.asarray(target_region.surface_normal, dtype=np.float64)
                source_vertical_half = float(
                    np.sum(
                        0.5
                        * source.extent
                        * np.abs(source.pose[:3, :3].T @ world_up)
                    )
                )
                cavity_vertical_half = float(
                    np.sum(
                        target_region.half_extents
                        * np.abs(target_region.axes.T @ world_up)
                    )
                )
                if 2.0 * source_vertical_half > 2.0 * cavity_vertical_half + 0.020:
                    raise ExecutionError(
                        "source is taller than the sensor-localized shelf cavity"
                    )
                # Place on the cavity floor rather than suspending the object at
                # its volume centre.  The open front deliberately permits a pan
                # handle to protrude, matching the public region's reference-
                # point containment while MPC still guards observed collisions.
                position = (
                    target_region.center
                    - world_up * cavity_vertical_half
                    + world_up * (source_vertical_half + vertical)
                )
            else:
                available = 2.0 * target_region.half_extents - 2.0 * float(
                    params.get("inside_margin", 0.015)
                )
                if np.any(source.extent > available + 0.02):
                    raise ExecutionError(
                        "source is larger than the visually estimated target interior"
                    )
                position = target_region.center.copy()
        elif relation == Relation.UNDER:
            position = target.position.copy()
            position[2] = scene.scene_floor_z + source.extent[2] / 2.0 + vertical
        else:
            relative = _relative_placement_geometry(
                source, target, target_region, relation, params
            )
            position = (
                target_region.center
                + relative.direction * relative.required_separation_m
            )
            source_vertical_half = _obb_support_radius(
                source.pose[:3, :3],
                0.5 * source.extent,
                np.array((0.0, 0.0, 1.0), dtype=np.float64),
            )
            position[2] = scene.scene_floor_z + source_vertical_half + vertical
        if relation in {Relation.ON, Relation.IN} and (
            "target_subregion" in params
            or float(params.get("world_offset_x_fraction", 0.0))
            or float(params.get("world_offset_y_fraction", 0.0))
        ) and params.get("target_subregion") not in _SHELF_CAVITY_PRIORS:
            position += _target_subregion_offset(target, params)
        object_goal[:3, 3] = position
        return object_goal @ object_from_ee

    @staticmethod
    def _effective_object_from_ee(
        bound: BoundConstraintGraph,
        source: SceneEntity,
        grasp_mode: GraspMode,
        config: RouteCControllerConfig,
    ) -> FloatArray:
        if grasp_mode != GraspMode.EXPAND:
            return bound.grasp.object_from_ee
        # The bowl is approached closed, inserted top-down at bowl pose +25 mm,
        # then retained by opening against the inner wall.  Preserve the grasp
        # candidate orientation but bind the measured bowl centre explicitly.
        expand_grasp = bound.grasp.candidate.world_from_ee.copy()
        expand_grasp[0, 3] = source.position[0]
        expand_grasp[1, 3] = source.position[1]
        expand_grasp[2, 3] = source.position[2] + config.expand_grasp_z_offset_m
        return np.linalg.inv(source.pose) @ expand_grasp

    @staticmethod
    def _weight(
        bound: BoundConstraintGraph, kind: ConstraintKind, phase: Phase, default: float
    ) -> float:
        for constraint in bound.graph.constraints_for(phase):
            if constraint.kind == kind:
                return constraint.weight
        return default

    @staticmethod
    def _parameter(
        bound: BoundConstraintGraph,
        kind: ConstraintKind,
        phase: Phase,
        name: str,
        default: float,
    ) -> float:
        for constraint in bound.graph.constraints_for(phase):
            if constraint.kind == kind and name in constraint.parameters:
                return float(constraint.parameters[name])
        return default


class RouteCController:
    """Compile, observe, bind, optimise, execute, verify, and recover."""

    _MOTION_PHASES = {
        Phase.APPROACH,
        Phase.GRASP,
        Phase.LIFT,
        Phase.TRANSFER,
        Phase.PLACE,
        Phase.RETREAT,
    }

    def __init__(
        self,
        compiler: TemplateConstraintCompiler,
        observer: SceneEstimator,
        grasp_provider: GraspProvider,
        robot: RobotController,
        mpc: RecedingHorizonOptimizer,
        binder: GraspBinder | None = None,
        verifier: RelationVerifier | None = None,
        goals: GoalSynthesizer | None = None,
        grasp_mode_selector: GraspModeSelector | None = None,
        config: RouteCControllerConfig | None = None,
    ) -> None:
        self.compiler = compiler
        self.observer = observer
        self.grasp_provider = grasp_provider
        self.robot = robot
        self.mpc = mpc
        self.binder = binder or GraspBinder()
        self.verifier = verifier or GeometryRelationVerifier()
        self.goals = goals or GoalSynthesizer()
        self.grasp_mode_selector = grasp_mode_selector or EntityGraspModeSelector()
        self.config = config or RouteCControllerConfig()
        self._active_grasp_mode = GraspMode.PINCH
        self._phase_motion_anchors: dict[Phase, _PhaseMotionAnchor] = {}
        self._selector_active_view_count = 0
        self._selector_initial_bearing_xy: FloatArray | None = None
        # A run-local journal is also retained on the controller so an
        # integration-owned terminal interruption (for example, the OSC step
        # budget) can return an honest partial result without changing the
        # controller's exception or recovery behavior.
        self._attempt_journal: list[PhaseAttempt] = []
        self._last_run_verification: Verification | None = None
        self._last_run_source_id: str | None = None
        self._last_run_target_id: str | None = None
        self._grasp_attempt_journal = GraspAttemptJournal()
        self._pending_grasp_engagement: _PendingGraspEngagement | None = None
        self._grasp_terminal_hint: _GraspTerminalHint | None = None

    @property
    def grasp_attempt_events(self) -> tuple[GraspAttemptEvent, ...]:
        """Completed physical jaw engagements in the current run."""

        return self._grasp_attempt_journal.records

    @property
    def pending_grasp_engagement(self) -> PendingGraspEngagement | None:
        """Allowed-field snapshot of an issued, unfinished jaw engagement."""

        pending = self._pending_grasp_engagement
        if pending is None:
            return None
        return PendingGraspEngagement(
            attempt_index=len(self._grasp_attempt_journal.records) + 1,
            source_text=pending.source_text,
            source_class=pending.source_class,
            grasp_mode=pending.grasp_mode,
            jaw_behavior=pending.jaw_behavior,
        )

    def run(self, task_text: str) -> RouteCResult:
        self._attempt_journal = []
        self._last_run_verification = None
        self._last_run_source_id = None
        self._last_run_target_id = None
        self._grasp_attempt_journal.reset()
        self._pending_grasp_engagement = None
        self._grasp_terminal_hint = None
        graph = self.compiler.compile(task_text)
        labels = self._requested_labels(graph)
        attempts = self._attempt_journal
        last_verification: Verification | None = None
        last_failure: str | None = None
        self._selector_active_view_count = 0
        self._selector_initial_bearing_xy = None

        for task_attempt in range(1, self.config.max_task_attempts + 1):
            self._phase_motion_anchors.clear()
            holding_object = False
            try:
                scene, source, fresh_resolution_trace = (
                    self._observe_and_resolve_source(graph, labels)
                )
                # Select and validate the mechanical mode before asking a
                # grasp provider to construct any candidate geometry.  This
                # keeps even an injected selector from turning a black-bowl
                # approach into an interior/expansion trajectory.
                grasp_mode = self._select_grasp_mode(graph.source.label)
                candidates = self._propose_grasps(
                    graph,
                    scene,
                    source,
                    fresh_resolution_trace,
                )
                bound = self.binder.bind(graph, scene, candidates)
                self._last_run_source_id = bound.source_id
                self._last_run_target_id = bound.target_id
                self._require_black_bowl_rim_pinch(
                    bound.source_label, grasp_mode
                )
                self._active_grasp_mode = grasp_mode
                for phase in graph.phases:
                    if phase == Phase.VERIFY:
                        scene = self.observer.observe(labels)
                        last_verification = self.verifier.verify(bound, scene)
                        self._last_run_verification = last_verification
                        attempts.append(
                            PhaseAttempt(
                                task_attempt,
                                phase,
                                1,
                                last_verification.success,
                                last_verification.detail,
                            )
                        )
                        if not last_verification.success:
                            raise ExecutionError(f"visual verification failed: {last_verification.detail}")
                        self._finish_grasp_engagement(
                            accepted=True,
                            reason=GraspReason.ACCEPTED,
                            evidence_source=GraspEvidenceCategory.PROPRIOCEPTION,
                        )
                        return RouteCResult(
                            True,
                            task_text,
                            tuple(attempts),
                            last_verification,
                            source_id=bound.source_id,
                            target_id=bound.target_id,
                            grasp_attempt_events=self.grasp_attempt_events,
                        )

                    success = False
                    for phase_attempt in range(1, self.config.max_phase_attempts + 1):
                        try:
                            bound = self._execute_phase(phase, bound, labels, grasp_mode)
                            attempts.append(PhaseAttempt(task_attempt, phase, phase_attempt, True, "ok"))
                            if phase == Phase.GRASP:
                                holding_object = True
                            elif phase == Phase.LIFT:
                                # Route C's successful LIFT is the first
                                # retention proof after the jaw command.  Keep
                                # the event pending until here so a load/proof
                                # failure rejects the correct engagement.
                                self._finish_grasp_engagement(
                                    accepted=True,
                                    reason=GraspReason.ACCEPTED,
                                    evidence_source=(
                                        GraspEvidenceCategory.PROPRIOCEPTION
                                    ),
                                )
                            elif phase == Phase.RELEASE:
                                holding_object = False
                            success = True
                            break
                        except TerminalExecutionError as exc:
                            self._reject_pending_grasp(exc)
                            last_failure = str(exc)
                            attempts.append(
                                PhaseAttempt(
                                    task_attempt,
                                    phase,
                                    phase_attempt,
                                    False,
                                    last_failure,
                                )
                            )
                            return self._interrupted_result(task_text, exc)
                        except (ExecutionError, OptimisationError, PerceptionError) as exc:
                            # A generic first LIFT failure is retryable in
                            # place with the same closed/open jaw engagement.
                            # Do not turn temporary OSC/optimisation failure
                            # into a permanent rejected grasp before that
                            # retry gets its retention proof.  GRASP failures
                            # terminate their own engagement; a final LIFT
                            # failure is abandoned by task-level recovery.
                            if (
                                phase == Phase.GRASP
                                or phase_attempt >= self.config.max_phase_attempts
                            ):
                                self._reject_pending_grasp(exc)
                            last_failure = str(exc)
                            attempts.append(
                                PhaseAttempt(task_attempt, phase, phase_attempt, False, last_failure)
                            )
                            # APPROACH already replans from fresh RGB-D and the
                            # measured current EE pose.  If its first attempt
                            # made useful progress but missed convergence,
                            # retry the same grasp anchor in place; switching
                            # candidates and retreating wastes the progress.
                            # Only a true GRASP failure is evidence against the
                            # active grasp candidate.
                            if phase == Phase.GRASP:
                                candidate_id = bound.grasp.candidate.candidate_id
                                requires_reacquire = getattr(
                                    self.grasp_provider,
                                    "requires_full_reacquire",
                                    None,
                                )
                                if callable(requires_reacquire) and bool(
                                    requires_reacquire(candidate_id)
                                ):
                                    # Never sweep from one drawer wall to the
                                    # other at grasp height.  Let the outer
                                    # recovery retreat, capture fresh RGB-D,
                                    # bind the next ranked candidate, and run
                                    # APPROACH again from above.
                                    raise GraspBindingError(
                                        "cavity grasp failed; full pregrasp "
                                        "reacquisition required"
                                    ) from exc
                                try:
                                    bound = bound.next_grasp()
                                    self._phase_motion_anchors.pop(Phase.GRASP, None)
                                except GraspBindingError:
                                    pass
                            # The final failed phase attempt is handed to the
                            # task-level recovery below, which performs exactly
                            # one recovery.  APPROACH keeps its physical
                            # progress; held LIFT/TRANSFER/PLACE phases retry
                            # the same absolute target without moving upward.
                            # GRASP/RELEASE/RETREAT remain free-space events and
                            # may use one best-effort retreat before retry.
                            if phase_attempt < self.config.max_phase_attempts:
                                if phase not in {
                                    Phase.APPROACH,
                                    Phase.LIFT,
                                    Phase.TRANSFER,
                                    Phase.PLACE,
                                }:
                                    self._safe_retreat()
                        except GraspBindingError as exc:
                            self._reject_pending_grasp(exc)
                            # A typed request for task-level visual/grasp
                            # reacquisition is still a real failed phase
                            # attempt.  Preserve it before the existing outer
                            # recovery path runs unchanged.
                            last_failure = str(exc)
                            attempts.append(
                                PhaseAttempt(
                                    task_attempt,
                                    phase,
                                    phase_attempt,
                                    False,
                                    last_failure,
                                )
                            )
                            raise
                    if not success:
                        raise ExecutionError(last_failure or f"{phase.value} failed")
            except TerminalExecutionError as exc:
                return self._interrupted_result(task_text, exc)
            except (ExecutionError, OptimisationError, PerceptionError, GraspBindingError) as exc:
                last_failure = str(exc)
                if isinstance(exc, _SelectorViewExhausted):
                    # The empty hand is already at the last high observation
                    # pose.  Do not descend or start another task attempt when
                    # no fresh 3-D anchor was recovered.
                    released = self.robot.set_gripper(
                        self._released_gripper_command(
                            self._select_grasp_mode(graph.source.label)
                        )
                    )
                    if not released.accepted:
                        last_failure = (
                            f"{last_failure}; recovery release failed: "
                            f"{released.detail or 'robot rejected release'}"
                        )
                    break
                if holding_object:
                    # Never move upward merely because a held phase failed.
                    # Release first; only a confirmed gripper command permits a
                    # subsequent free-space recovery motion.
                    released = self.robot.set_gripper(
                        self._released_gripper_command(self._active_grasp_mode)
                    )
                    holding_object = False
                    if released.accepted:
                        discard_released = getattr(
                            self.goals,
                            "discard_recovery_released_binding",
                            None,
                        )
                        if callable(discard_released):
                            # A failed held phase ends with an observed gripper
                            # release, so the last proprio-propagated source at
                            # the raised EE is no longer a valid next-attempt
                            # track.  Sensor adapters may drop exactly that
                            # dynamic id and force fresh RGB-D reacquisition.
                            discard_released(bound.source_id)
                        self._safe_retreat()
                    else:
                        last_failure = (
                            f"{last_failure}; recovery release failed: "
                            f"{released.detail or 'robot rejected release'}"
                        )
                else:
                    self._safe_retreat()
                    self.robot.set_gripper(
                        self._released_gripper_command(self._active_grasp_mode)
                    )
                continue

        self._reject_pending_grasp(last_failure or "route C run failed")
        return RouteCResult(
            False,
            task_text,
            tuple(attempts),
            last_verification,
            last_failure,
            grasp_attempt_events=self.grasp_attempt_events,
        )

    def _interrupted_result(
        self, task_text: str, failure: BaseException | str
    ) -> RouteCResult:
        """Snapshot completed and active phase attempts after an interruption."""

        self._reject_pending_grasp(failure)
        return RouteCResult(
            False,
            task_text,
            tuple(self._attempt_journal),
            self._last_run_verification,
            str(failure),
            self._last_run_source_id,
            self._last_run_target_id,
            self.grasp_attempt_events,
        )

    def _observe_and_resolve_source(
        self,
        graph: ConstraintGraph,
        labels: Sequence[str],
    ) -> tuple[SceneEstimate, SceneEntity, object]:
        """Resolve from one explicitly scoped source observation when needed.

        An ON selector may request a same-call typed support measurement.  An
        IN selector keeps the ordinary observation path; its optional
        ``last_selector_view_hint`` can only move an empty hand to a bounded
        observation pose.  Neither mechanism supplies a fixture OBB or grasp
        context.  The loop returns solely after the resolver accepts a fresh
        measured source/reference pair or a strict same-frame, source-only
        support proof, which never manufactures reference geometry or an
        obstacle field.
        """

        source_selector = graph.source.selector
        use_source_selector_observation = bool(
            graph.source.role == "source"
            and source_selector is not None
            and source_selector.relation is Relation.ON
            and len(source_selector.references) == 1
        )
        while True:
            observe_source_selector = getattr(
                self.observer, "observe_source_selector", None
            )
            if (
                use_source_selector_observation
                and callable(observe_source_selector)
            ):
                # Selector context is scoped to this exact observation.  Pass
                # it again on every task-resolution retry; an observer must
                # never infer it from mutable pending state left by a prior
                # frame.
                scene = observe_source_selector(labels, graph.source)
            else:
                scene = self.observer.observe(labels)
            previous_resolution_trace = getattr(
                self.binder.resolver, "last_resolution_trace", None
            )
            try:
                source = self.binder.resolver.resolve(graph.source, scene)
            except PerceptionError as exc:
                hint = self._matching_selector_view_hint(graph, scene)
                if hint is None:
                    if self._selector_active_view_count:
                        raise _SelectorViewExhausted(
                            f"{exc}; strict selector evidence disappeared after "
                            f"{self._selector_active_view_count} active view(s)"
                        ) from exc
                    raise
                if (
                    self._selector_active_view_count
                    >= self.config.max_selector_active_views
                ):
                    raise _SelectorViewExhausted(
                        f"{exc}; no fresh 3-D selector pair after "
                        f"{self._selector_active_view_count} active view(s)"
                    ) from exc
                self._execute_selector_active_view(graph, scene, hint)
                self._invalidate_for_selector_reacquisition()
                self._selector_active_view_count += 1
                continue
            current_resolution_trace = getattr(
                self.binder.resolver, "last_resolution_trace", None
            )
            # The production selector replaces this dictionary for every
            # support-relation resolution.  Object identity is therefore a
            # freshness token: a resolver that emitted nothing for this
            # observation must not lend stale context to a grasp provider.
            fresh_resolution_trace = (
                current_resolution_trace
                if current_resolution_trace is not previous_resolution_trace
                else None
            )
            return scene, source, fresh_resolution_trace

    def _matching_selector_view_hint(
        self,
        graph: ConstraintGraph,
        scene: SceneEstimate,
    ) -> Mapping[str, object] | None:
        selector = graph.source.selector
        if (
            selector is None
            or selector.relation != Relation.IN
            or len(selector.references) != 1
        ):
            return None
        hint = getattr(self.observer, "last_selector_view_hint", None)
        if not isinstance(hint, Mapping):
            return None

        def normalise(value: object) -> str:
            return " ".join(str(value).lower().replace("_", " ").split())

        if (
            normalise(hint.get("relation", "")) != Relation.IN.value
            or normalise(hint.get("source_label", "")) != graph.source.label
            or normalise(hint.get("reference_label", ""))
            != selector.references[0]
        ):
            return None
        try:
            center = np.asarray(hint["source_center_world"], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            return None
        capture_matches = False
        if scene.capture_id:
            hint_capture = hint.get("capture_id")
            capture_matches = (
                type(hint_capture) is str
                and hint_capture == scene.capture_id
            )
        else:
            # Legacy/non-formal fixtures may not yet carry a sensor-content
            # commitment.  Their numeric timestamp remains a compatibility
            # fallback only; formal Route C requires a populated capture_id.
            try:
                timestamp_s = float(hint["timestamp_s"])
            except (KeyError, TypeError, ValueError):
                return None
            capture_matches = bool(
                np.isfinite(timestamp_s)
                and abs(timestamp_s - float(scene.timestamp_s)) <= 1e-6
            )
        if (
            center.shape != (3,)
            or not np.all(np.isfinite(center))
            or not capture_matches
            or np.any(center < scene.workspace_min)
            or np.any(center > scene.workspace_max)
        ):
            return None
        return hint

    def _execute_selector_active_view(
        self,
        graph: ConstraintGraph,
        scene: SceneEstimate,
        hint: Mapping[str, object],
    ) -> None:
        center = np.asarray(hint["source_center_world"], dtype=np.float64)
        source_top_z = self._selector_source_top_z(scene, graph.source.label, center)
        requested_height = source_top_z + self.config.selector_view_clearance_m
        ceiling = min(
            self.config.recovery_motion_ceiling_z_m,
            float(scene.workspace_max[2]),
        )
        if requested_height > ceiling + 1e-9:
            raise _SelectorViewExhausted(
                "strict selector source has no calibrated high-clearance view"
            )

        current = np.asarray(self.robot.current_ee_pose(), dtype=np.float64)
        if current.shape != (4, 4) or not np.all(np.isfinite(current)):
            raise _SelectorViewExhausted("current EE pose is unavailable for active view")
        if self._selector_initial_bearing_xy is None:
            initial_offset = current[:2, 3] - center[:2]
            length = float(np.linalg.norm(initial_offset))
            if length > 1e-6:
                self._selector_initial_bearing_xy = initial_offset / length

        safe_height = max(float(current[2, 3]), requested_height)
        if safe_height > ceiling + 1e-9:
            raise _SelectorViewExhausted(
                "current EE pose exceeds calibrated active-view ceiling"
            )
        command = self._released_gripper_command(
            self._select_grasp_mode(graph.source.label)
        )
        raised = current.copy()
        raised[2, 3] = safe_height
        if safe_height > float(current[2, 3]) + 1e-4:
            feedback = self.robot.execute_waypoints(
                np.stack((current, raised)), Phase.RETREAT, command
            )
            if not feedback.accepted:
                raise _SelectorViewExhausted(
                    feedback.detail or "vertical selector-view clearance failed"
                )

        view = raised.copy()
        if self._selector_active_view_count == 0:
            view[:2, 3] = center[:2]
        else:
            bearing = self._selector_initial_bearing_xy
            if bearing is None:
                raise _SelectorViewExhausted(
                    "initial EE-to-source bearing is unavailable for second view"
                )
            view[:2, 3] = (
                center[:2]
                + self.config.selector_view_second_retreat_m * bearing
            )
        lower = scene.workspace_min[:2] + self.config.selector_view_workspace_margin_m
        upper = scene.workspace_max[:2] - self.config.selector_view_workspace_margin_m
        if np.any(view[:2, 3] < lower) or np.any(view[:2, 3] > upper):
            raise _SelectorViewExhausted(
                "selector active-view XY target is outside calibrated workspace"
            )
        if float(np.linalg.norm(view[:2, 3] - raised[:2, 3])) > 1e-4:
            feedback = self.robot.execute_waypoints(
                np.stack((raised, view)), Phase.RETREAT, command
            )
            if not feedback.accepted:
                raise _SelectorViewExhausted(
                    feedback.detail or "constant-height selector view failed"
                )

    @staticmethod
    def _selector_source_top_z(
        scene: SceneEstimate,
        source_label: str,
        source_center: FloatArray,
    ) -> float:
        matches = [entity for entity in scene.entities if entity.label == source_label]
        if not matches:
            raise _SelectorViewExhausted(
                "strict selector hint has no measured source geometry"
            )
        source = min(
            matches,
            key=lambda entity: float(
                np.linalg.norm(entity.position - source_center)
            ),
        )
        if float(np.linalg.norm(source.position - source_center)) > 0.060:
            raise _SelectorViewExhausted(
                "strict selector hint does not match measured source geometry"
            )
        top = source.keypoints.get("top")
        if top is not None:
            top_z = float(np.asarray(top, dtype=np.float64)[2])
        else:
            world_span = np.abs(source.pose[:3, :3]) @ source.extent
            top_z = float(source.position[2] + 0.5 * world_span[2])
        if not np.isfinite(top_z) or top_z < float(source.position[2]):
            raise _SelectorViewExhausted(
                "strict selector source top is not a valid sensor measurement"
            )
        return top_z

    def _invalidate_for_selector_reacquisition(self) -> None:
        invalidate = getattr(
            self.observer, "invalidate_selector_reacquisition", None
        )
        if not callable(invalidate):
            raise _SelectorViewExhausted(
                "observer cannot guarantee fresh selector reacquisition"
            )
        invalidate()

    def _propose_grasps(
        self,
        graph: ConstraintGraph,
        scene: SceneEstimate,
        source: SceneEntity,
        resolution_trace: object,
    ) -> Sequence[GraspCandidate]:
        """Pass a freshly resolved IN anchor to providers that support it.

        This path is deliberately narrow.  The optional provider receives no
        benchmark task id or simulator state, only the RGB-D-derived entity
        used by the semantic resolver to validate the source relation.
        """

        fallback = self.grasp_provider.propose
        selector = graph.source.selector
        if selector is None or selector.relation != Relation.IN:
            return fallback(scene, source.instance_id)
        if not isinstance(resolution_trace, Mapping):
            return fallback(scene, source.instance_id)
        if resolution_trace.get("relation") != Relation.IN.value:
            return fallback(scene, source.instance_id)
        if resolution_trace.get("selected_source_id") != source.instance_id:
            return fallback(scene, source.instance_id)
        anchor_id = resolution_trace.get("selected_anchor_id")
        if not isinstance(anchor_id, str) or not anchor_id:
            return fallback(scene, source.instance_id)
        try:
            reference = scene.by_id(anchor_id)
        except PerceptionError:
            return fallback(scene, source.instance_id)

        # Reject a trace from a different IN selector even if a frame-local id
        # was recycled.  This substring rule mirrors EntityResolver's
        # open-vocabulary label fallback.
        expected_labels = selector.references
        if not expected_labels or not any(
            expected == reference.label
            or expected in reference.label
            or reference.label in expected
            for expected in expected_labels
        ):
            return fallback(scene, source.instance_id)
        contextual = getattr(self.grasp_provider, "propose_with_reference", None)
        if not callable(contextual):
            return fallback(scene, source.instance_id)
        return contextual(scene, source.instance_id, reference)

    def _execute_phase(
        self,
        phase: Phase,
        bound: BoundConstraintGraph,
        labels: Sequence[str],
        grasp_mode: GraspMode,
    ) -> BoundConstraintGraph:
        source_label = str(
            getattr(bound, "source_label", labels[0] if labels else "")
        )
        self._require_black_bowl_rim_pinch(source_label, grasp_mode)
        if phase == Phase.RELEASE:
            feedback = self.robot.set_gripper(self._released_gripper_command(grasp_mode))
            if not feedback.accepted:
                raise ExecutionError(feedback.detail or "gripper failed to open")
            return bound

        if phase not in self._MOTION_PHASES:
            raise ExecutionError(f"unhandled phase: {phase.value}")
        self.mpc.reset()
        anchor = self._phase_motion_anchors.get(phase)
        cavity_grasp_incomplete_chunks = 0
        requires_full_reacquire = getattr(
            self.grasp_provider, "requires_full_reacquire", None
        )
        cavity_grasp = bool(
            phase == Phase.GRASP
            and callable(requires_full_reacquire)
            and requires_full_reacquire(bound.grasp.candidate.candidate_id)
        )
        for _ in range(self.config.max_mpc_replans):
            scene = self.observer.observe(labels)
            current = np.asarray(self.robot.current_ee_pose(), dtype=np.float64)
            request = self.goals.motion_request(
                phase, bound, scene, current, self.config, grasp_mode
            )
            if anchor is None:
                anchor = _PhaseMotionAnchor(
                    request.goal_pose.copy(), request.min_height_m
                )
                self._phase_motion_anchors[phase] = anchor
            fixed_goal = anchor.goal_pose.copy()
            if phase == Phase.LIFT:
                # A retry may start above the original absolute lift target due
                # to OSC overshoot.  Never descend, but equally never add a new
                # lift_distance to the already lifted current pose.
                fixed_goal[2, 3] = max(fixed_goal[2, 3], current[2, 3])
            request = replace(
                request,
                goal_pose=fixed_goal,
                min_height_m=anchor.min_height_m,
            )
            if self._at_goal(current, request.goal_pose):
                if phase == Phase.GRASP:
                    self._engage_gripper(bound, grasp_mode, source_text=labels[0])
                return bound
            chunk = self.mpc.replan(request, scene)
            terminal_goal_chunk = bool(
                len(chunk.poses) >= 1
                and np.allclose(
                    np.asarray(chunk.poses[-1], dtype=np.float64),
                    request.goal_pose,
                    rtol=1e-9,
                    atol=1e-9,
                )
            )
            feedback = self.robot.execute_waypoints(
                chunk.poses, phase, self._motion_gripper_command(phase, grasp_mode)
            )
            if not feedback.accepted:
                # The low-level tracker uses a slightly tighter waypoint
                # tolerance.  If real OSC execution has nevertheless reached
                # the controller's typed phase tolerance, complete normally
                # instead of discarding the physical progress.
                current_after = np.asarray(
                    self.robot.current_ee_pose(), dtype=np.float64
                )
                if self._at_goal(current_after, request.goal_pose):
                    if phase == Phase.GRASP:
                        self._engage_gripper(bound, grasp_mode, source_text=labels[0])
                    return bound
                raise ExecutionError(feedback.detail or "robot rejected trajectory chunk")

            # Do not spend a third complete low-level chunk repeatedly
            # descending beside the same fixture wall.  This test uses only
            # the frozen sensor goal and the existing typed _at_goal gates;
            # it never treats partial progress as grasp/contact evidence.
            if cavity_grasp and terminal_goal_chunk:
                current_after = np.asarray(
                    self.robot.current_ee_pose(), dtype=np.float64
                )
                if self._at_goal(current_after, request.goal_pose):
                    continue
                cavity_grasp_incomplete_chunks += 1
                if (
                    cavity_grasp_incomplete_chunks
                    >= self.config.max_cavity_grasp_incomplete_chunks
                ):
                    raise ExecutionError(
                        "cavity grasp remained incomplete after "
                        f"{cavity_grasp_incomplete_chunks} OSC chunks"
                    )

        current = np.asarray(self.robot.current_ee_pose(), dtype=np.float64)
        if anchor is None:
            raise ExecutionError(f"{phase.value} produced no motion goal")
        goal = anchor.goal_pose.copy()
        if phase == Phase.LIFT:
            goal[2, 3] = max(goal[2, 3], current[2, 3])
        if not self._at_goal(current, goal):
            raise ExecutionError(f"{phase.value} did not converge within MPC replan budget")
        if phase == Phase.GRASP:
            self._engage_gripper(bound, grasp_mode, source_text=labels[0])
        return bound

    def _select_grasp_mode(self, source_label: str) -> GraspMode:
        mode = self.grasp_mode_selector.select(source_label)
        self._require_black_bowl_rim_pinch(source_label, mode)
        return mode

    @staticmethod
    def _require_black_bowl_rim_pinch(
        source_label: str, grasp_mode: GraspMode
    ) -> None:
        normalized = " ".join(
            str(source_label).lower().replace("_", " ").replace("-", " ").split()
        )
        words = normalized.split()
        black_bowl = any(
            words[index : index + 2] == ["black", "bowl"]
            for index in range(max(0, len(words) - 1))
        )
        if black_bowl and grasp_mode is not GraspMode.RIM_PINCH:
            raise ExecutionError(
                "black bowl requires closed-finger rim pinch before planning"
            )

    def _at_goal(self, current: FloatArray, goal: FloatArray) -> bool:
        position = float(np.linalg.norm(current[:3, 3] - goal[:3, 3]))
        if self._active_grasp_mode == GraspMode.EXPAND:
            # A round bowl and an internal expansion grasp are axisymmetric:
            # the tool z-axis must stay aligned, while yaw about that axis has
            # no mechanical or task meaning.
            cosine = float(
                np.clip(
                    np.dot(current[:3, 2], goal[:3, 2]),
                    -1.0,
                    1.0,
                )
            )
            orientation = math.acos(cosine)
        else:
            delta = Rotation.from_matrix(current[:3, :3]).inv() * Rotation.from_matrix(
                goal[:3, :3]
            )
            orientation = float(delta.magnitude())
        return (
            position <= self.config.position_tolerance_m
            and orientation <= self.config.orientation_tolerance_rad
        )

    def _safe_retreat(self) -> None:
        try:
            current = np.asarray(self.robot.current_ee_pose(), dtype=np.float64)
            goal = current.copy()
            goal[2, 3] = min(
                current[2, 3] + min(self.config.retreat_distance_m, 0.05),
                self.config.recovery_motion_ceiling_z_m,
            )
            if goal[2, 3] <= current[2, 3] + 1e-4:
                return
            self.robot.execute_waypoints(
                np.stack((current, goal)),
                Phase.RETREAT,
                self._released_gripper_command(self._active_grasp_mode),
            )
        except (ExecutionError, OptimisationError):
            # Recovery is best-effort; the original failure remains authoritative.
            # External evaluator termination is a RuntimeError on purpose and
            # must propagate immediately instead of being swallowed here.
            pass

    @staticmethod
    def _requested_labels(graph: object) -> tuple[str, ...]:
        labels = [graph.source.label, graph.target.label]
        if graph.source.selector is not None:
            labels.extend(graph.source.selector.references)
        if graph.target.selector is not None:
            labels.extend(graph.target.selector.references)
        return tuple(dict.fromkeys(labels))

    def _engage_gripper(
        self,
        bound: BoundConstraintGraph,
        grasp_mode: GraspMode,
        *,
        source_text: str,
    ) -> None:
        """Issue exactly one jaw event and retain it until post-grasp proof."""

        if self._pending_grasp_engagement is not None:
            raise ExecutionError("previous grasp engagement has no terminal outcome")
        source_text = " ".join(
            str(getattr(bound, "source_label", source_text))
            .lower()
            .replace("_", " ")
            .split()
        )
        if source_text == "black bowl" and grasp_mode is not GraspMode.RIM_PINCH:
            # Fail before commanding the hand: black bowls may never become an
            # expansion/interior-brace event, even under a custom selector.
            raise ExecutionError("black bowl requires closed-finger rim pinch")
        jaw_behavior = (
            JawBehavior.OPEN_FINGERS_INTERIOR_BRACE
            if grasp_mode is GraspMode.EXPAND
            else JawBehavior.CLOSE_FINGERS
        )
        self._pending_grasp_engagement = _PendingGraspEngagement(
            source_text=source_text,
            source_class=source_text,
            grasp_mode=grasp_mode.value,
            jaw_behavior=jaw_behavior,
        )
        self._grasp_terminal_hint = None
        feedback = self.robot.set_gripper(
            self._engaged_gripper_command(grasp_mode)
        )
        if not feedback.accepted:
            self._grasp_terminal_hint = _GraspTerminalHint(
                GraspReason.JAW_COMMAND_REJECTED,
                GraspEvidenceCategory.CONTROLLER_EXECUTION,
            )
            raise ExecutionError(feedback.detail or "robot rejected grasp command")
        if not self.robot.grasp_confirmed(grasp_mode):
            self._grasp_terminal_hint = _GraspTerminalHint(
                GraspReason.RETENTION_REJECTED,
                GraspEvidenceCategory.PROPRIOCEPTION,
            )
            raise ExecutionError(
                feedback.detail or "grasp was not confirmed by proprioception"
            )

    def _finish_grasp_engagement(
        self,
        *,
        accepted: bool,
        reason: GraspReason,
        evidence_source: GraspEvidenceCategory,
    ) -> None:
        pending = self._pending_grasp_engagement
        if pending is None:
            return
        self._grasp_attempt_journal.append(
            source_text=pending.source_text,
            source_class=pending.source_class,
            grasp_mode=pending.grasp_mode,
            jaw_behavior=pending.jaw_behavior,
            accepted=accepted,
            reason=reason,
            evidence_source=evidence_source,
        )
        self._pending_grasp_engagement = None
        self._grasp_terminal_hint = None

    def _reject_pending_grasp(self, failure: BaseException | str) -> None:
        if self._pending_grasp_engagement is None:
            return
        hint = self._grasp_terminal_hint
        if hint is not None:
            reason = hint.reason
            evidence_source = hint.evidence_source
        elif "budget" in str(failure).lower():
            reason = GraspReason.STEP_BUDGET_EXHAUSTED
            evidence_source = GraspEvidenceCategory.CONTROLLER_EXECUTION
        elif isinstance(failure, GraspBindingError):
            reason = GraspReason.RETENTION_REJECTED
            evidence_source = GraspEvidenceCategory.PROPRIOCEPTION
        elif isinstance(failure, OptimisationError):
            reason = GraspReason.TRAJECTORY_REJECTED
            evidence_source = GraspEvidenceCategory.CONTROLLER_EXECUTION
        else:
            reason = GraspReason.EXECUTION_STALLED
            evidence_source = GraspEvidenceCategory.CONTROLLER_EXECUTION
        self._finish_grasp_engagement(
            accepted=False,
            reason=reason,
            evidence_source=evidence_source,
        )

    @staticmethod
    def _engaged_gripper_command(mode: GraspMode) -> float:
        return GRIPPER_OPEN if mode == GraspMode.EXPAND else GRIPPER_CLOSE

    @staticmethod
    def _released_gripper_command(mode: GraspMode) -> float:
        return GRIPPER_CLOSE if mode == GraspMode.EXPAND else GRIPPER_OPEN

    @classmethod
    def _motion_gripper_command(cls, phase: Phase, mode: GraspMode) -> float:
        if phase in {Phase.LIFT, Phase.TRANSFER, Phase.PLACE}:
            return cls._engaged_gripper_command(mode)
        return cls._released_gripper_command(mode)
