"""Instruction-plan execution for Route C.

This module is deliberately downstream of :mod:`task_planner`: language is
parsed once, every atomic goal is dispatched in order, and unsupported goal
types fail closed.  Pick/place goals retain Route C's constraint graph and
continuous MPC controller.  Fixture goals use the same RGB-D scene/SDF and
MPC interface through a small sensor-target provider protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from ..common.grasp_journal import GraspAttemptEvent
from .compiler import TemplateConstraintCompiler
from .controller import (
    ControllerFeedback,
    ExecutionError,
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    GraspMode,
    PhaseAttempt,
    RobotController,
    RouteCResult,
    Verification,
)
from .optimizer import (
    InfeasibleClearanceDiagnostic,
    MotionRequest,
    OptimisationError,
    RecedingHorizonOptimizer,
)
from .perception import SceneEstimate, SceneEstimator
from .schema import ConstraintGraph, ConstraintKind, EntityRef, Phase, Relation, SpatialSelector
from .sdf import CompositeSDF
from .task_planner import (
    AtomicGoal,
    AtomicGoalKind,
    EntitySelector,
    PlacementRegion,
    PlannerRelation,
    RouteCTaskPlanner,
    SelectorKind,
    TaskEntityRef,
    TaskPlan,
)


FloatArray = NDArray[np.float64]


class GoalLoweringError(ValueError):
    """A typed high-level goal cannot be represented by the motion graph."""


class ContactExecutionError(RuntimeError):
    """A sensor-derived contact operation could not be completed safely."""


_GOAL_RELATIONS = {
    PlannerRelation.ON: Relation.ON,
    PlannerRelation.IN: Relation.IN,
    PlannerRelation.LEFT_OF: Relation.LEFT_OF,
    PlannerRelation.RIGHT_OF: Relation.RIGHT_OF,
    PlannerRelation.FRONT_OF: Relation.FRONT_OF,
    PlannerRelation.UNDER: Relation.UNDER,
}

_POSITIONAL_SELECTORS = {
    SelectorKind.LEFT: Relation.LEFTMOST,
    SelectorKind.RIGHT: Relation.RIGHTMOST,
    SelectorKind.FRONT: Relation.FRONTMOST,
    SelectorKind.BACK: Relation.BACKMOST,
    SelectorKind.MIDDLE: Relation.MIDDLE,
    SelectorKind.TOP: Relation.TOPMOST,
    SelectorKind.BOTTOM: Relation.BOTTOMMOST,
    SelectorKind.CENTER: Relation.CENTER,
    SelectorKind.FIRST: Relation.FIRST,
    SelectorKind.SECOND: Relation.SECOND,
}

_RELATIONAL_SELECTORS = {
    SelectorKind.ON: Relation.ON,
    SelectorKind.IN: Relation.IN,
    SelectorKind.NEXT_TO: Relation.NEXT_TO,
    SelectorKind.BETWEEN: Relation.BETWEEN,
}

class AtomicGoalGraphCompiler:
    """Lower one planner-selected motion goal to the strict graph schema.

    Selection is explicit so the old controller can retain its small
    ``compile(text)`` dependency without reparsing or truncating a compound
    instruction.  The coordinator is the only caller of :meth:`select`.
    """

    MOTION_KINDS = frozenset(
        {
            AtomicGoalKind.PLACE,
            AtomicGoalKind.STACK,
            AtomicGoalKind.PLACE_GROUP,
        }
    )
    LOWERABLE_KINDS = MOTION_KINDS

    def __init__(self) -> None:
        self._selected: AtomicGoal | None = None

    def select(self, goal: AtomicGoal) -> None:
        if goal.kind not in self.LOWERABLE_KINDS:
            raise GoalLoweringError(f"{goal.kind.value!r} is not a motion-graph goal")
        self._selected = goal

    def compile(self, instruction: str) -> ConstraintGraph:
        if self._selected is None:
            raise GoalLoweringError("no atomic goal was selected")
        return self.lower(self._selected, instruction)

    @classmethod
    def lower(cls, goal: AtomicGoal, instruction: str) -> ConstraintGraph:
        if goal.kind not in cls.LOWERABLE_KINDS:
            raise GoalLoweringError(f"{goal.kind.value!r} is not a motion-graph goal")
        if goal.target is None or goal.relation is None:
            raise GoalLoweringError("a motion goal requires a target and relation")
        relation = _GOAL_RELATIONS.get(goal.relation)
        if relation is None:
            raise GoalLoweringError(f"unsupported motion relation: {goal.relation.value}")

        # After stacking, carrying the lower object moves the mechanically
        # supported pair.  No synthetic group body or hidden identity enters
        # the controller.
        source_ref = goal.subjects[-1] if goal.kind is AtomicGoalKind.PLACE_GROUP else goal.subject
        if (
            goal.kind is AtomicGoalKind.PLACE_GROUP
            and len({item.label for item in goal.subjects}) == 1
        ):
            # Once an equal-label pair has been stacked, left/right selectors
            # collapse to the same XY.  The lower visible component is the
            # mechanically supporting object and therefore the only safe
            # group-carry anchor.
            source_ref = TaskEntityRef(
                source_ref.label,
                EntitySelector(SelectorKind.BOTTOM),
            )
        if (
            source_ref.selector is not None
            and source_ref.selector.kind is SelectorKind.SECOND
        ):
            # ``both`` goals run sequentially.  The first instance now lies at
            # the sensed destination, so the farthest same-label component is
            # the remaining instance.  This is episode geometry, not an
            # instance name or hidden object identity.
            source = EntityRef(
                "source",
                source_ref.label,
                SpatialSelector(Relation.FARTHEST_FROM, (goal.target.label,)),
            )
        else:
            source = cls._entity("source", source_ref)
        target = cls._entity("target", goal.target)
        graph = TemplateConstraintCompiler._build_graph(  # noqa: SLF001
            instruction.strip(),
            source.label,
            source.selector,
            target.label,
            relation,
        )
        graph = replace(graph, source=source, target=target)
        if goal.target_region is None:
            return graph

        constraints = []
        for constraint in graph.constraints:
            if constraint.kind is ConstraintKind.GOAL_RELATION:
                parameters = dict(constraint.parameters)
                parameters["target_subregion"] = goal.target_region.value
                constraint = replace(constraint, parameters=parameters)
            constraints.append(constraint)
        return replace(graph, constraints=tuple(constraints))

    @classmethod
    def _entity(cls, role: str, reference: TaskEntityRef) -> EntityRef:
        selector = (
            cls._selector(reference.selector, reference.label)
            if reference.selector is not None
            else None
        )
        return EntityRef(role, reference.label, selector)

    @classmethod
    def _selector(cls, selector: EntitySelector, label: str) -> SpatialSelector:
        if label == "drawer":
            drawer_parts = {
                SelectorKind.TOP: Relation.TOP_PART,
                SelectorKind.MIDDLE: Relation.MIDDLE_PART,
                SelectorKind.BOTTOM: Relation.BOTTOM_PART,
            }
            if selector.kind in drawer_parts:
                return SpatialSelector(drawer_parts[selector.kind])
        if selector.kind in _POSITIONAL_SELECTORS:
            return SpatialSelector(_POSITIONAL_SELECTORS[selector.kind])
        relation = _RELATIONAL_SELECTORS.get(selector.kind)
        if relation is None:
            raise GoalLoweringError(f"unsupported entity selector: {selector.kind.value}")
        references = tuple(cls._reference_label(item) for item in selector.references)
        return SpatialSelector(relation, references)

    @classmethod
    def _reference_label(cls, reference: TaskEntityRef) -> str:
        """Render a nested reference without discarding its qualifier.

        Constraint-graph selectors intentionally carry immutable language
        labels rather than planner objects.  Flattening ``top drawer of the
        wooden cabinet`` to merely ``top drawer`` loses the only observable
        fixture association before perception is queried.  Reconstruct the
        complete normalized phrase recursively so DINO and the RGB-D resolver
        receive the same qualifier that the instruction supplied.
        """

        selector = reference.selector
        if selector is not None and selector.kind in {
            SelectorKind.TOP,
            SelectorKind.MIDDLE,
            SelectorKind.BOTTOM,
        }:
            result = f"{selector.kind.value} {reference.label}"
            if selector.references:
                return f"{result} of {cls._reference_label(selector.references[0])}"
            return result
        if selector is not None and selector.references:
            rendered = tuple(cls._reference_label(item) for item in selector.references)
            if selector.kind is SelectorKind.BETWEEN:
                return f"{reference.label} between {rendered[0]} and {rendered[1]}"
            connector = {
                SelectorKind.ON: "on",
                SelectorKind.IN: "in",
                SelectorKind.NEXT_TO: "next to",
            }.get(selector.kind)
            if connector is None:
                raise GoalLoweringError(
                    f"cannot render nested selector reference: {selector.kind.value}"
                )
            return f"{reference.label} {connector} {rendered[0]}"
        return reference.label


@dataclass(frozen=True)
class ContactTargetEstimate:
    """Metric contact geometry reconstructed from current public sensors."""

    point_world: FloatArray
    outward_world: FloatArray
    feature_axis_world: FloatArray
    manipulation_axis_world: FloatArray
    manipulation_distance_m: float
    requested_labels: tuple[str, ...]
    confidence: float
    visual_progress_expected: bool = True
    rotation_center_world: FloatArray | None = None
    rotation_axis_world: FloatArray | None = None
    rotation_angle_rad: float | None = None
    approach_offset_world: FloatArray | None = None
    # Optional RGB-D drawer-front support used only by subsequent close
    # pushes.  The ordinary handle point remains the primary contact target;
    # these fields carry no simulator identity or evaluator state.
    drawer_front_point_world: FloatArray | None = None
    drawer_front_normal_world: FloatArray | None = None
    drawer_front_support_m: float | None = None
    # Typed planar-push geometry.  These values are reconstructed from the
    # public RGB-D plate/stove observations and deliberately travel together:
    # a contact point alone is not sufficient to prove that the same plate
    # reached the originally observed goal after a drag.
    push_object_center_world: FloatArray | None = None
    push_target_center_world: FloatArray | None = None
    push_object_radius_m: float | None = None
    push_direction_world: FloatArray | None = None

    def __post_init__(self) -> None:
        for name in (
            "point_world",
            "outward_world",
            "feature_axis_world",
            "manipulation_axis_world",
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite xyz vector")
            if name != "point_world":
                norm = float(np.linalg.norm(value))
                if norm < 1e-8:
                    raise ValueError(f"{name} must be non-zero")
                value = value / norm
            object.__setattr__(self, name, value.copy())
        if not np.isfinite(self.manipulation_distance_m) or self.manipulation_distance_m <= 0:
            raise ValueError("manipulation distance must be finite and positive")
        labels = tuple(dict.fromkeys(str(label).strip().lower() for label in self.requested_labels))
        if not labels or any(not label for label in labels):
            raise ValueError("at least one non-empty requested label is required")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must lie in [0, 1]")
        if not isinstance(self.visual_progress_expected, bool):
            raise ValueError("visual_progress_expected must be boolean")
        if self.approach_offset_world is not None:
            approach_offset = np.asarray(
                self.approach_offset_world, dtype=np.float64
            )
            if (
                approach_offset.shape != (3,)
                or not np.all(np.isfinite(approach_offset))
                or float(np.linalg.norm(approach_offset)) > 0.30
            ):
                raise ValueError(
                    "approach_offset_world must be a finite bounded xyz vector"
                )
            object.__setattr__(
                self, "approach_offset_world", approach_offset.copy()
            )
        if self.drawer_front_point_world is not None:
            front_point = np.asarray(self.drawer_front_point_world, dtype=np.float64)
            if front_point.shape != (3,) or not np.all(np.isfinite(front_point)):
                raise ValueError(
                    "drawer_front_point_world must be a finite xyz vector"
                )
            object.__setattr__(self, "drawer_front_point_world", front_point.copy())
        if self.drawer_front_normal_world is not None:
            front_normal = np.asarray(
                self.drawer_front_normal_world, dtype=np.float64
            )
            if front_normal.shape != (3,) or not np.all(np.isfinite(front_normal)):
                raise ValueError(
                    "drawer_front_normal_world must be a finite xyz vector"
                )
            front_norm = float(np.linalg.norm(front_normal))
            if front_norm < 1e-8:
                raise ValueError("drawer_front_normal_world must be non-zero")
            object.__setattr__(
                self,
                "drawer_front_normal_world",
                front_normal / front_norm,
            )
        if self.drawer_front_support_m is not None:
            if (
                not np.isfinite(self.drawer_front_support_m)
                or self.drawer_front_support_m <= 0.0
            ):
                raise ValueError("drawer front support must be positive")
        if (self.drawer_front_point_world is None) != (
            self.drawer_front_normal_world is None
        ):
            raise ValueError(
                "drawer front point and normal must be provided together"
            )
        push_values = (
            self.push_object_center_world,
            self.push_target_center_world,
            self.push_object_radius_m,
            self.push_direction_world,
        )
        if any(value is not None for value in push_values):
            if any(value is None for value in push_values):
                raise ValueError(
                    "planar push geometry requires object center, target "
                    "center, radius, and direction"
                )
            push_center = np.asarray(
                self.push_object_center_world, dtype=np.float64
            )
            push_target = np.asarray(
                self.push_target_center_world, dtype=np.float64
            )
            push_direction = np.asarray(
                self.push_direction_world, dtype=np.float64
            )
            direction_norm = float(np.linalg.norm(push_direction))
            push_radius = float(self.push_object_radius_m)
            if (
                push_center.shape != (3,)
                or push_target.shape != (3,)
                or push_direction.shape != (3,)
                or not np.all(np.isfinite(push_center))
                or not np.all(np.isfinite(push_target))
                or not np.all(np.isfinite(push_direction))
                or direction_norm < 1e-8
                or not np.isfinite(push_radius)
                or not 0.005 <= push_radius <= 0.20
            ):
                raise ValueError("planar push geometry is invalid")
            push_direction /= direction_norm
            if abs(float(push_direction[2])) > 0.10:
                raise ValueError("planar push direction must be horizontal")
            object.__setattr__(
                self, "push_object_center_world", push_center.copy()
            )
            object.__setattr__(
                self, "push_target_center_world", push_target.copy()
            )
            object.__setattr__(self, "push_object_radius_m", push_radius)
            object.__setattr__(
                self, "push_direction_world", push_direction.copy()
            )
        rotary_values = (
            self.rotation_center_world,
            self.rotation_axis_world,
            self.rotation_angle_rad,
        )
        if any(value is not None for value in rotary_values):
            if any(value is None for value in rotary_values):
                raise ValueError(
                    "rotary contact geometry requires center, axis, and angle"
                )
            center = np.asarray(self.rotation_center_world, dtype=np.float64)
            axis = np.asarray(self.rotation_axis_world, dtype=np.float64)
            angle = float(self.rotation_angle_rad)
            axis_norm = float(np.linalg.norm(axis))
            if (
                center.shape != (3,)
                or axis.shape != (3,)
                or not np.all(np.isfinite(center))
                or not np.all(np.isfinite(axis))
                or axis_norm < 1e-8
                or not np.isfinite(angle)
                or not 0.0 < abs(angle) <= np.pi
            ):
                raise ValueError("rotary contact geometry is invalid")
            object.__setattr__(self, "rotation_center_world", center.copy())
            object.__setattr__(
                self, "rotation_axis_world", axis / axis_norm
            )
            object.__setattr__(self, "rotation_angle_rad", angle)
        object.__setattr__(self, "requested_labels", labels)


@runtime_checkable
class ContactTargetProvider(Protocol):
    def estimate(self, goal: AtomicGoal) -> ContactTargetEstimate: ...


@runtime_checkable
class FormedStackHandler(Protocol):
    """Sensor-only lifecycle for carrying a newly formed object stack."""

    def capture_stack(
        self, goal: AtomicGoal, result: RouteCResult
    ) -> Verification: ...

    def prepare_carry(self, goal: AtomicGoal) -> Verification: ...

    def verify_carry(
        self, goal: AtomicGoal, result: RouteCResult
    ) -> Verification: ...

    def clear_carry_requirement(self) -> None: ...


@runtime_checkable
class RankedTargetHandler(Protocol):
    """Bind future ranked destinations before earlier actions occlude them."""

    def prefetch(self, plan: TaskPlan) -> str | None: ...

    def prepare_target(self, goal: AtomicGoal) -> str | None: ...

    def clear_target_requirement(self) -> None: ...


@dataclass(frozen=True)
class RouteCContactConfig:
    max_replans: int = 8
    position_tolerance_m: float = 0.010
    orientation_tolerance_rad: float = 0.14
    free_clearance_m: float = 0.012
    free_tool_radius_m: float = 0.018
    precontact_clearance_m: float = 0.070
    safe_height_m: float = 0.090
    # Short typed drawer-front exit before restoring ordinary inflated SDF
    # clearance after an opening pull.
    drawer_contact_exit_m: float = 0.025
    # A typed exit is not complete merely because the low-level controller
    # reports a bounded contact residual.  Require public EE displacement
    # along the observed outward ray, and use short diagonal microsteps when
    # a pure outward command is blocked by the drawer lip.
    drawer_contact_exit_progress_m: float = 0.015
    drawer_contact_exit_microstep_outward_m: float = 0.0125
    drawer_contact_exit_microstep_lift_m: float = 0.020
    drawer_release_min_width_m: float = 0.060
    visual_progress_m: float = 0.025
    knob_rotation_rad: float = 1.82
    # A physical stove knob reaches its mechanical stop before the nominal
    # Cartesian wrist command.  Completion still requires the low-level
    # signed-rotation, blocked-width and contact-pose plateau proof; this is
    # only the minimum measured travel admitted by that typed proof.
    knob_mechanical_completion_rad: float = 0.95
    # A handle crop can be biased toward one visible side by roughly one pad
    # thickness.  Search a tiny symmetric set across the measured long axis,
    # and require a short loaded pull before committing to the full motion.
    # Both signals remain public: calibrated RGB-D and gripper proprioception.
    handle_retry_offset_m: float = 0.012
    handle_load_proof_m: float = 0.012
    handle_load_min_width_m: float = 0.008
    handle_load_max_settle_loss_m: float = 0.003
    linear_close_stop_residual_m: float = 0.045
    drawer_close_stop_residual_m: float = 0.090
    # The final closed-front re-observation can lose the drawer handle behind
    # the cabinet lip.  Require a measured, signed traverse that reaches the
    # public LIBERO closed predicate for the standard 160-mm drawer travel;
    # the bounded retry path remains available when the first typed push stops
    # early.  This is still a sensor threshold, never a simulator joint read.
    drawer_close_required_progress_m: float = 0.140
    # Add a bounded sensor-directed push margin beyond the nominal drawer
    # travel.  It lets a compliant contact stop reach the RGB-D closed front
    # instead of being accepted several centimetres short; completion remains
    # gated by the fresh RGB-D displacement check below.
    drawer_close_extra_push_m: float = 0.050
    drawer_close_load_proof_m: float = 0.025
    drawer_close_compact_width_max_m: float = 0.012
    drawer_close_max_attempts: int = 3
    # A released retry starts with roughly 79-mm jaws.  Compact it at the
    # high, sensor-cleared transit pose before descending beside the drawer
    # front.  Each pulse is bounded by the adapter's ordinary gripper hold;
    # public width must cross the unchanged strict pusher gate.
    drawer_retry_compact_max_pulses: int = 2
    # Later close attempts must descend on an episode-local column that has
    # already been reached at drawer height.  A moved drawer front can make
    # the ordinary 70-mm precontact offset lie underneath another protruding
    # front or handle.  Re-project that proven public-proprio column onto the
    # fresh horizontal front normal, but never route farther than this hard
    # sensor-space bound.
    drawer_retry_safe_column_max_clearance_m: float = 0.22
    # A fresh close-drawer estimate may expose a nearby collision-free slot
    # along the observed handle/front tangent.  Project only that horizontal
    # component into the retry column; calibration residual off the tangent
    # and the total shift both remain tightly bounded and fail closed.
    drawer_retry_staging_max_offset_m: float = 0.040
    drawer_retry_staging_max_nontangent_m: float = 0.002
    # Once the proven column has descended, retain ordinary inflated-SDF
    # clearance until a 70-mm front entry.  The final short, horizontal leg is
    # a typed drawer-only corridor: its target self-surface may be removed by
    # the contact observer, while every unrelated field keeps the ordinary
    # free-space clearance/tool inflation.
    drawer_retry_corridor_contact_inset_m: float = 0.005
    drawer_retry_corridor_max_length_m: float = 0.090
    drawer_retry_corridor_lateral_tolerance_m: float = 0.006
    drawer_retry_corridor_terminal_clearance_m: float = 0.030
    drawer_retry_corridor_stop_residual_m: float = 0.030
    push_retry_inset_m: float = 0.008
    push_retry_depth_m: float = 0.006
    push_retained_min_width_m: float = 0.008
    # A plate rim must survive a short physical load proof before any long
    # drag.  Continue only in short pieces so width loss is observed before a
    # stale contact can traverse the whole target distance.
    push_load_proof_m: float = 0.008
    push_load_proof_max_progress_m: float = 0.010
    push_load_proof_max_cross_drift_m: float = 0.002
    push_load_proof_max_rotation_drift_rad: float = 0.010
    push_segment_m: float = 0.035
    push_max_segments: int = 7
    push_retry_perimeter_angle_rad: float = 0.35
    push_required_progress_m: float = 0.18
    # A fourth perimeter contact is available only when an earlier retry side
    # passed the public load proof and then produced significant fresh RGB-D
    # plate progress.  Validation below keeps four as a hard ceiling.
    push_max_attempts: int = 4
    # A plate drag is complete only when a fresh, same-identity RGB-D plate
    # lies close to the goal frozen before contact.  These gates prevent a
    # switched red blob or a long but off-axis displacement from satisfying
    # the skill.  Retry travel is recomputed from the fresh centre toward the
    # frozen target and remains tightly bounded by measured remaining range.
    push_goal_tolerance_m: float = 0.035
    push_overshoot_tolerance_m: float = 0.035
    push_lateral_tolerance_m: float = 0.030
    push_target_anchor_tolerance_m: float = 0.005
    push_radius_tolerance_m: float = 0.008
    push_radius_relative_tolerance: float = 0.25
    push_direction_min_cosine: float = 0.95
    push_retry_margin_m: float = 0.010
    push_retry_max_distance_m: float = 0.20
    # After a retained plate-rim segment loses its public width/contact proof,
    # open the fingers and leave the known target contact through one short
    # radial/upward typed corridor.  Only after measured motion and released
    # width may ordinary inflated-SDF retreat and fresh reacquisition resume.
    push_contact_exit_radial_m: float = 0.0125
    push_contact_exit_lift_m: float = 0.020
    push_contact_exit_min_progress_m: float = 0.010
    push_contact_exit_max_cross_drift_m: float = 0.006
    push_contact_exit_max_rotation_drift_rad: float = 0.030
    push_contact_exit_release_min_width_m: float = 0.060
    # A normal release pulse can leave a recently pinched plate rim narrow
    # for several control ticks.  Complete that release in place before any
    # Cartesian contact-exit motion, with a small hard pulse budget.
    push_contact_exit_release_max_pulses: int = 3
    # Fewer, slightly longer Cartesian chords reduce per-waypoint settling
    # overhead while retaining a small enough arc approximation for the
    # public microwave geometry (the maximum chord is about 44 mm at the
    # nominal handle radius).
    microwave_arc_segment_rad: float = 0.18
    # Per-waypoint advancement tolerance only.  This absorbs millimetre-scale
    # OSC endpoint plateaus without widening the typed mechanical-stop gate or
    # any final visual/grasp verifier.
    microwave_arc_position_tolerance_m: float = 0.012
    # A retained contact that changes by no more than this span over two
    # consecutive OSC chunks is a proprioceptive contact plateau.  It may
    # enter the typed mechanical-stop path, but cannot itself establish task
    # success; retained width and the final fresh RGB-D verifier remain
    # mandatory.
    microwave_arc_plateau_intervals: int = 2
    microwave_arc_plateau_span_m: float = 0.002
    microwave_arc_plateau_rotation_span_rad: float = 0.010
    microwave_regrasp_arc_rad: float = 0.43
    microwave_open_state_tolerance_rad: float = 0.16
    # A first, load-proven two-pad pull establishes the moving edge and a
    # signed hinge direction.  Continuing to regrasp that vertical edge once
    # the door is oblique drives the Panda wrist toward its yaw limit.  After
    # one useful chord, route outside the *same* locally tracked RGB-D edge
    # and finish with compact, closed fingers as a compressive back-side
    # pusher.  These dimensions describe only sensor-space clearance and the
    # bounded public TCP path; they do not encode a task or simulator state.
    microwave_open_push_handoff_min_rad: float = 0.30
    microwave_open_push_edge_clearance_m: float = 0.070
    microwave_open_push_backside_clearance_m: float = 0.055
    microwave_open_push_lift_m: float = 0.040
    microwave_open_push_precontact_m: float = 0.030
    microwave_open_push_contact_inset_m: float = 0.006
    # Back-side continuation is a sequence of short, fresh-vision pulses on
    # the frozen hinge circle.  The Cartesian cap and angular cap are both
    # enforced, so neither a small nor a large sensed radius can create an
    # unbounded chord.  These are command bounds, not completion thresholds.
    microwave_open_push_chord_max_m: float = 0.075
    microwave_open_push_chord_max_angle_rad: float = 0.30
    # Retain the original typed-contact residual independently of the longer
    # circular command bound above; extending a safe pulse must not relax the
    # contact acceptance gate.
    microwave_open_push_segment_m: float = 0.055
    microwave_open_push_compact_width_m: float = 0.014
    # A fixed close pulse is not long enough after the first microwave pinch
    # has been released.  Finish the pusher preshape with a short, public-
    # proprioception servo while holding the routed TCP pose.  These are hard
    # safety gates rather than success tolerances: callers may make them more
    # conservative, but the validation below prevents relaxing their caps.
    microwave_compact_servo_max_steps: int = 8
    microwave_compact_servo_max_stall_steps: int = 2
    microwave_compact_servo_min_width_progress_m: float = 0.0002
    microwave_compact_servo_max_position_drift_m: float = 0.008
    microwave_compact_servo_max_rotation_drift_rad: float = 0.08
    microwave_compact_servo_max_force_delta_n: float = 20.0
    microwave_compact_servo_max_force_norm_n: float = 80.0
    microwave_compact_servo_max_torque_delta_nm: float = 5.0
    microwave_compact_servo_max_torque_norm_nm: float = 15.0
    microwave_open_push_reverse_tolerance_rad: float = 0.020
    microwave_open_push_min_segment_progress_rad: float = 0.010
    microwave_open_push_handoff_residual_m: float = 0.060
    # One OSC prefix is followed immediately by a fresh local RGB-D edge.
    # Twelve such bounded observations fit the fixed episode budget, unlike
    # seven 3--5-replan Cartesian settles in the v84 trace.
    microwave_open_push_max_segments: int = 12
    # Co-rotate enough to keep the vertical handle seated between the pads,
    # but cap wrist yaw before the Panda reaches the progressive stall seen
    # when the full door angle is imposed on the wrist.  The bound is applied
    # symmetrically for opening and closing arcs.
    microwave_wrist_corotation_limit_rad: float = 0.58
    # A locally reassociated close edge can have a shorter sensed radius than
    # the public closed-handle slot, causing the ray-to-ray arc to stop just
    # before the physical jamb.  Continue a second-cycle retained close by one
    # small, symmetric, bounded angular margin.  The resulting motion still
    # has to retain blocked width; any jamb is accepted only by the typed
    # proprioceptive plateau and the unchanged fresh RGB-D verifier.
    microwave_supplemental_close_extension_rad: float = 0.20
    # Keep the slightly deeper supplemental close within four smooth chords
    # for the public microwave radius.  This cap is used only after a fresh
    # local edge association; the initial grasp/arc path remains unchanged.
    microwave_supplemental_arc_segment_rad: float = 0.195
    # A supplemental arc is already a retained, sensor-checked continuation.
    # Intermediate chords need only enter this small positional tube before
    # advancing to the next point on the same frozen hinge circle.  The final
    # chord keeps the ordinary microwave arc tolerance below, so this saves
    # redundant settling without shortening the commanded terminal close.
    microwave_supplemental_intermediate_position_tolerance_m: float = 0.015
    # A closing door leaves the released fingers immediately beside the
    # visible edge.  Clear that *known contact* with one short mostly-outward
    # motion before asking the free-space SDF planner to inflate the tool.  A
    # small upward component clears low foreground geometry that can lie in
    # front of the appliance.  This carries no completion/verification
    # semantics.
    microwave_contact_exit_m: float = 0.035
    microwave_contact_exit_lift_m: float = 0.020
    microwave_contact_exit_progress_m: float = 0.015
    microwave_release_min_width_m: float = 0.060
    # A freshly observed microwave edge must remain on the same frozen
    # RGB-D hinge circle.  A large radius jump indicates that occlusion made
    # the detector switch from the moving edge to an appliance/body edge.
    microwave_verify_radius_tolerance_m: float = 0.035
    # After a retained typed stop, search only a small neighbourhood around
    # the last measured EE/door-edge contact.  The detector independently
    # requires consistency with the initially frozen hinge radius.
    microwave_reassociation_anchor_radius_m: float = 0.080
    microwave_max_regrasps: int = 4

    def __post_init__(self) -> None:
        values = (
            self.position_tolerance_m,
            self.orientation_tolerance_rad,
            self.free_clearance_m,
            self.free_tool_radius_m,
            self.precontact_clearance_m,
            self.safe_height_m,
            self.drawer_contact_exit_m,
            self.drawer_contact_exit_progress_m,
            self.drawer_contact_exit_microstep_outward_m,
            self.drawer_contact_exit_microstep_lift_m,
            self.drawer_release_min_width_m,
            self.visual_progress_m,
            self.knob_rotation_rad,
            self.knob_mechanical_completion_rad,
            self.handle_retry_offset_m,
            self.handle_load_proof_m,
            self.handle_load_min_width_m,
            self.handle_load_max_settle_loss_m,
            self.linear_close_stop_residual_m,
            self.drawer_close_stop_residual_m,
            self.drawer_close_required_progress_m,
            self.drawer_close_extra_push_m,
            self.drawer_close_load_proof_m,
            self.drawer_close_compact_width_max_m,
            self.drawer_retry_safe_column_max_clearance_m,
            self.drawer_retry_staging_max_offset_m,
            self.drawer_retry_staging_max_nontangent_m,
            self.drawer_retry_corridor_contact_inset_m,
            self.drawer_retry_corridor_max_length_m,
            self.drawer_retry_corridor_lateral_tolerance_m,
            self.drawer_retry_corridor_terminal_clearance_m,
            self.drawer_retry_corridor_stop_residual_m,
            self.push_retry_inset_m,
            self.push_retry_depth_m,
            self.push_retained_min_width_m,
            self.push_load_proof_m,
            self.push_load_proof_max_progress_m,
            self.push_load_proof_max_cross_drift_m,
            self.push_load_proof_max_rotation_drift_rad,
            self.push_segment_m,
            self.push_retry_perimeter_angle_rad,
            self.push_required_progress_m,
            self.push_goal_tolerance_m,
            self.push_overshoot_tolerance_m,
            self.push_lateral_tolerance_m,
            self.push_target_anchor_tolerance_m,
            self.push_radius_tolerance_m,
            self.push_radius_relative_tolerance,
            self.push_direction_min_cosine,
            self.push_retry_margin_m,
            self.push_retry_max_distance_m,
            self.push_contact_exit_radial_m,
            self.push_contact_exit_lift_m,
            self.push_contact_exit_min_progress_m,
            self.push_contact_exit_max_cross_drift_m,
            self.push_contact_exit_max_rotation_drift_rad,
            self.push_contact_exit_release_min_width_m,
            self.microwave_arc_segment_rad,
            self.microwave_arc_position_tolerance_m,
            self.microwave_arc_plateau_span_m,
            self.microwave_arc_plateau_rotation_span_rad,
            self.microwave_regrasp_arc_rad,
            self.microwave_open_state_tolerance_rad,
            self.microwave_open_push_handoff_min_rad,
            self.microwave_open_push_edge_clearance_m,
            self.microwave_open_push_backside_clearance_m,
            self.microwave_open_push_lift_m,
            self.microwave_open_push_precontact_m,
            self.microwave_open_push_contact_inset_m,
            self.microwave_open_push_chord_max_m,
            self.microwave_open_push_chord_max_angle_rad,
            self.microwave_open_push_segment_m,
            self.microwave_open_push_compact_width_m,
            self.microwave_compact_servo_min_width_progress_m,
            self.microwave_compact_servo_max_position_drift_m,
            self.microwave_compact_servo_max_rotation_drift_rad,
            self.microwave_compact_servo_max_force_delta_n,
            self.microwave_compact_servo_max_force_norm_n,
            self.microwave_compact_servo_max_torque_delta_nm,
            self.microwave_compact_servo_max_torque_norm_nm,
            self.microwave_open_push_reverse_tolerance_rad,
            self.microwave_open_push_min_segment_progress_rad,
            self.microwave_open_push_handoff_residual_m,
            self.microwave_wrist_corotation_limit_rad,
            self.microwave_supplemental_close_extension_rad,
            self.microwave_supplemental_arc_segment_rad,
            self.microwave_supplemental_intermediate_position_tolerance_m,
            self.microwave_contact_exit_m,
            self.microwave_contact_exit_lift_m,
            self.microwave_contact_exit_progress_m,
            self.microwave_release_min_width_m,
            self.microwave_verify_radius_tolerance_m,
            self.microwave_reassociation_anchor_radius_m,
        )
        if (
            not isinstance(self.push_max_attempts, int)
            or isinstance(self.push_max_attempts, bool)
            or not 1 <= self.push_max_attempts <= 4
        ):
            raise ValueError(
                "plate push attempts must be an integer from one through four"
            )
        if (
            self.max_replans < 1
            or self.push_max_segments < 1
            or self.drawer_close_max_attempts < 1
            or self.drawer_retry_compact_max_pulses < 1
            or self.push_contact_exit_release_max_pulses < 1
            or self.microwave_max_regrasps < 1
            or self.microwave_open_push_max_segments < 1
            or self.microwave_arc_plateau_intervals < 1
            or self.microwave_compact_servo_max_steps < 1
            or self.microwave_compact_servo_max_stall_steps < 0
            or any(not np.isfinite(x) or x <= 0 for x in values)
        ):
            raise ValueError("contact execution limits must be finite and positive")
        if self.microwave_open_push_compact_width_m > 0.014:
            raise ValueError(
                "microwave compact pusher width cannot exceed the strict 14 mm gate"
            )
        if (
            self.drawer_close_compact_width_max_m > 0.014
            or self.drawer_retry_compact_max_pulses > 3
        ):
            raise ValueError(
                "drawer retry pusher must remain within the strict compact bounds"
            )
        if self.push_contact_exit_release_max_pulses > 3:
            raise ValueError(
                "plate contact-exit release cannot exceed three bounded pulses"
            )
        if (
            self.drawer_retry_safe_column_max_clearance_m > 0.22
            or self.drawer_retry_safe_column_max_clearance_m
            < self.precontact_clearance_m
        ):
            raise ValueError(
                "drawer retry safe-column clearance must remain between the "
                "ordinary precontact clearance and 0.22 m"
            )
        if (
            self.drawer_retry_staging_max_offset_m > 0.040
            or self.drawer_retry_staging_max_nontangent_m > 0.002
            or self.drawer_retry_staging_max_nontangent_m
            >= self.drawer_retry_staging_max_offset_m
        ):
            raise ValueError(
                "drawer retry staging offset exceeds its calibrated tangent bounds"
            )
        if (
            self.drawer_retry_corridor_contact_inset_m > 0.005
            or self.drawer_retry_corridor_max_length_m > 0.090
            or self.drawer_retry_corridor_lateral_tolerance_m > 0.006
            or self.drawer_retry_corridor_terminal_clearance_m > 0.030
            or self.drawer_retry_corridor_stop_residual_m > 0.030
            or self.drawer_retry_corridor_terminal_clearance_m
            >= self.precontact_clearance_m
            or self.precontact_clearance_m
            + self.drawer_retry_corridor_contact_inset_m
            > self.drawer_retry_corridor_max_length_m
        ):
            raise ValueError(
                "drawer retry contact corridor may only use the calibrated "
                "compact target-adjacent bounds"
            )
        if (
            self.microwave_compact_servo_max_steps > 12
            or self.microwave_compact_servo_max_stall_steps > 2
            or self.microwave_compact_servo_max_stall_steps
            >= self.microwave_compact_servo_max_steps
            or self.microwave_compact_servo_min_width_progress_m < 0.0002
            or self.microwave_compact_servo_max_position_drift_m > 0.010
            or self.microwave_compact_servo_max_rotation_drift_rad > 0.10
            or self.microwave_compact_servo_max_force_delta_n > 20.0
            or self.microwave_compact_servo_max_force_norm_n > 80.0
            or self.microwave_compact_servo_max_torque_delta_nm > 5.0
            or self.microwave_compact_servo_max_torque_norm_nm > 15.0
        ):
            raise ValueError(
                "microwave compact servo limits may only be made more conservative"
            )
        if self.knob_mechanical_completion_rad >= self.knob_rotation_rad:
            raise ValueError(
                "mechanical knob completion must be below the commanded rotation"
            )
        if (
            not 0.95 <= self.push_direction_min_cosine <= 1.0
            or self.push_retained_min_width_m < 0.008
            or self.visual_progress_m < 0.025
            or self.push_goal_tolerance_m > 0.035
            or self.push_overshoot_tolerance_m > 0.035
            or self.push_lateral_tolerance_m > 0.030
            or self.push_radius_relative_tolerance > 0.25
            or self.push_target_anchor_tolerance_m > 0.005
            or self.push_radius_tolerance_m > 0.008
            or self.push_lateral_tolerance_m > self.push_goal_tolerance_m
            or self.push_retry_margin_m > 0.010
            or self.push_retry_max_distance_m > 0.20
            or not 0.005 <= self.push_load_proof_m <= 0.010
            or not self.push_load_proof_m
            <= self.push_load_proof_max_progress_m
            <= 0.010
            or self.push_load_proof_max_cross_drift_m > 0.002
            or self.push_load_proof_max_rotation_drift_rad > 0.010
            or self.push_segment_m > 0.035
            or self.push_max_segments > 7
            or not 0.15 <= self.push_retry_perimeter_angle_rad <= 0.45
            or self.push_contact_exit_radial_m > 0.015
            or self.push_contact_exit_lift_m > 0.025
            or self.push_contact_exit_min_progress_m < 0.008
            or self.push_contact_exit_min_progress_m
            > float(
                np.hypot(
                    self.push_contact_exit_radial_m,
                    self.push_contact_exit_lift_m,
                )
            )
            or self.push_contact_exit_max_cross_drift_m > 0.006
            or self.push_contact_exit_max_rotation_drift_rad > 0.030
            or not 0.060 <= self.push_contact_exit_release_min_width_m <= 0.080
        ):
            raise ValueError(
                "plate push identity, goal, and retry gates may only be made "
                "more conservative"
            )
        if (
            self.microwave_open_push_handoff_min_rad
            > self.microwave_regrasp_arc_rad
        ):
            raise ValueError(
                "microwave back-side handoff must fit inside the first pinch chord"
            )
        if (
            self.microwave_open_push_contact_inset_m
            >= self.microwave_open_push_edge_clearance_m
        ):
            raise ValueError(
                "microwave pusher inset must remain inside its radial clearance"
            )
        if (
            self.microwave_open_push_chord_max_m > 0.075
            or self.microwave_open_push_chord_max_angle_rad > 0.30
            or self.microwave_open_push_max_segments > 12
        ):
            raise ValueError(
                "microwave fresh-vision push pulses exceed their calibrated bounds"
            )
        if (
            self.microwave_open_push_compact_width_m
            >= self.microwave_release_min_width_m
        ):
            raise ValueError(
                "microwave compact pusher width must be below its released width"
            )


@dataclass(frozen=True)
class ContactGoalResult:
    success: bool
    attempts: tuple[PhaseAttempt, ...]
    failure: str | None = None


@dataclass(frozen=True)
class _MicrowaveOpenHandoff:
    """Public pose/edge evidence frozen before the first pinch is released."""

    release_pose: FloatArray
    predicted_edge_world: FloatArray
    achieved_angle_rad: float
    rotary_complete: bool


@dataclass(frozen=True)
class _DrawerRetrySafeColumn:
    """Episode-local public evidence for one completed low drawer descent."""

    pose: FloatArray
    planar_outward_world: FloatArray


@dataclass(frozen=True)
class _DrawerClosePlateauEvidence:
    """Public typed-motion evidence for a bounded drawer-close plateau."""

    fresh_target: ContactTargetEstimate
    command_length_m: float
    signed_inward_m: float
    cross_drift_m: float
    rotation_drift_rad: float
    width_m: float
    policy_actions: int


@dataclass(frozen=True)
class _DrawerCloseMotionPlateauEvidence:
    """Public motion-only evidence from a rejected typed drawer corridor."""

    command_length_m: float
    signed_inward_m: float
    cross_drift_m: float
    rotation_drift_rad: float
    width_m: float
    policy_actions: int


class _DrawerCloseTypedCorridorPlateau(ContactExecutionError):
    """Carry validated public motion evidence across the typed GRASP boundary."""

    def __init__(
        self,
        detail: str,
        evidence: _DrawerCloseMotionPlateauEvidence,
    ) -> None:
        super().__init__(detail)
        self.evidence = evidence


class MPCContactGoalExecutor:
    """Execute sensor-derived fixture goals with Route C continuous MPC."""

    SUPPORTED_KINDS = frozenset(
        {
            AtomicGoalKind.OPEN,
            AtomicGoalKind.CLOSE,
            AtomicGoalKind.TURN_ON,
            AtomicGoalKind.TURN_OFF,
            AtomicGoalKind.PUSH,
        }
    )
    _PLATE_CLEARANCE_RESTORE_MAX_DISTANCE_M = 0.030
    _PLATE_CLEARANCE_RESTORE_SAMPLE_SPACING_M = 0.0025
    _PLATE_CLEARANCE_RESTORE_MIN_PROGRESS_M = 0.005
    _PLATE_CLEARANCE_RESTORE_MAX_CROSS_DRIFT_M = 0.006
    _PLATE_CLEARANCE_RESTORE_MAX_Z_ERROR_M = 0.006
    _PLATE_CLEARANCE_RESTORE_MAX_DOWNWARD_M = 0.001
    _PLATE_CLEARANCE_RESTORE_MAX_ROTATION_RAD = 0.030
    _PLATE_MAX_FRESH_REGRESSION_M = 0.005
    _DRAWER_CLOSE_PLATEAU_MAX_VISUAL_GAP_M = 0.003
    _DRAWER_CLOSE_PLATEAU_MAX_COMMAND_M = 0.090
    _DRAWER_CLOSE_PLATEAU_MIN_INWARD_M = 0.001
    _DRAWER_CLOSE_PLATEAU_MAX_CROSS_M = 0.006
    _DRAWER_CLOSE_PLATEAU_MAX_WIDTH_M = 0.012

    def __init__(
        self,
        observer: SceneEstimator,
        robot: RobotController,
        mpc: RecedingHorizonOptimizer,
        provider: ContactTargetProvider,
        config: RouteCContactConfig | None = None,
    ) -> None:
        self.observer = observer
        self.robot = robot
        self.mpc = mpc
        self.provider = provider
        self.config = config or RouteCContactConfig()
        self._last_turn_progress_rad: float | None = None
        self._final_goal_context = False
        self._drawer_release_proven = False
        self._microwave_open_handoff: _MicrowaveOpenHandoff | None = None
        self._drawer_retry_safe_column: _DrawerRetrySafeColumn | None = None
        self._drawer_close_plateau: _DrawerClosePlateauEvidence | None = None

    def set_final_goal_context(self, is_final_goal: bool) -> None:
        """Set the language-plan boundary for the next contact goal.

        This is supplied by :class:`RouteCSequentialCoordinator`; it is not
        inferred from task metadata or evaluator state.
        """

        self._final_goal_context = bool(is_final_goal)

    def execute(self, goal: AtomicGoal) -> ContactGoalResult:
        if goal.kind not in self.SUPPORTED_KINDS:
            return ContactGoalResult(False, (), f"unsupported contact goal: {goal.kind.value}")
        attempts: list[PhaseAttempt] = []
        try:
            self._last_turn_progress_rad = None
            self._drawer_release_proven = False
            self._microwave_open_handoff = None
            self._drawer_retry_safe_column = None
            self._drawer_close_plateau = None
            set_mode = getattr(self.robot, "set_active_grasp_mode", None)
            if callable(set_mode):
                set_mode(GraspMode.PINCH)
            set_candidate = getattr(self.robot, "set_active_grasp_candidate", None)
            if callable(set_candidate):
                set_candidate(f"route-c-contact-{goal.kind.value}")
            target = self.provider.estimate(goal)
            if target.confidence <= 0.0:
                raise ContactExecutionError("sensor target confidence is zero")
            if goal.kind is AtomicGoalKind.PUSH:
                self._execute_push(goal, target, attempts)
                # Plate push owns a stronger typed terminal verifier: fresh
                # RGB-D identity and frozen-goal geometry plus retained rim
                # evidence.  Falling through to the generic displacement
                # check would weaken that conjunction and was the source of
                # a false G05 completion.
                return ContactGoalResult(True, tuple(attempts))
            elif goal.kind in {AtomicGoalKind.TURN_ON, AtomicGoalKind.TURN_OFF}:
                self._execute_turn(goal, target, attempts)
            elif (
                goal.kind is AtomicGoalKind.CLOSE
                and goal.subject.label == "drawer"
            ):
                self._execute_close_drawer_servo(goal, target, attempts)
                return ContactGoalResult(True, tuple(attempts))
            elif (
                goal.kind is AtomicGoalKind.OPEN
                and goal.subject.label == "microwave"
                and target.rotation_center_world is not None
            ):
                self._execute_open_microwave_backside_push(
                    goal,
                    target,
                    attempts,
                )
            else:
                self._execute_linear_fixture(goal, target, attempts)
            self._verify_progress(goal, target, attempts)
            return ContactGoalResult(True, tuple(attempts))
        except (
            ContactExecutionError,
            ExecutionError,
            OptimisationError,
            LookupError,
            ValueError,
        ) as exc:
            return ContactGoalResult(False, tuple(attempts), str(exc))

    def _execute_linear_fixture(
        self,
        goal: AtomicGoal,
        target: ContactTargetEstimate,
        attempts: list[PhaseAttempt],
        *,
        microwave_close_cycle: int = 1,
        microwave_frozen_target: ContactTargetEstimate | None = None,
        defer_drawer_release: bool = False,
    ) -> None:
        observation_pose = np.asarray(
            self.robot.current_ee_pose(), dtype=np.float64
        ).copy()
        closing = goal.kind is AtomicGoalKind.CLOSE
        # Closing a drawer or microwave is a compressive face/handle push.
        # Preserve the currently reachable vertical wrist rather than
        # demanding the horizontal handle orientation needed for an opening
        # pull.
        horizontal_close = closing and goal.subject.label == "microwave"
        if horizontal_close and microwave_frozen_target is None:
            microwave_frozen_target = target
        pose = (
            self._nearest_symmetric_jaw_pose(self._contact_pose(target))
            if horizontal_close or not closing
            else self._vertical_contact_pose(target)
        )
        if not closing:
            if (
                goal.subject.label == "microwave"
                and target.rotation_center_world is not None
            ):
                current_target = target
                current_pose = pose
                for cycle in range(1, self.config.microwave_max_regrasps + 1):
                    complete = self._execute_open_fixture_with_load_proof(
                        goal,
                        current_target,
                        current_pose,
                        attempts,
                        rotary_angle_limit_rad=self.config.microwave_regrasp_arc_rad,
                    )
                    if complete:
                        return
                    refreshed = self.provider.estimate(goal)
                    remaining = refreshed.rotation_angle_rad
                    if remaining is None:
                        raise ContactExecutionError(
                            "fresh microwave continuation lacked hinge geometry"
                        )
                    attempts.append(
                        PhaseAttempt(
                            cycle,
                            Phase.VERIFY,
                            1,
                            abs(remaining)
                            <= self.config.microwave_open_state_tolerance_rad,
                            "fresh RGB-D microwave continuation has "
                            f"{abs(remaining):.3f} rad remaining",
                        )
                    )
                    if (
                        abs(remaining)
                        <= self.config.microwave_open_state_tolerance_rad
                    ):
                        return
                    current_target = refreshed
                    current_pose = self._nearest_symmetric_jaw_pose(
                        self._contact_pose(current_target)
                    )
                raise ContactExecutionError(
                    "microwave remained visibly short of the open state after "
                    f"{self.config.microwave_max_regrasps} bounded regrasps"
                )
            self._execute_open_fixture_with_load_proof(
                goal, target, pose, attempts
            )
            return
        precontact = pose.copy()
        precontact[:3, 3] += target.outward_world * self.config.precontact_clearance_m
        approach_offset = (
            np.zeros(3, dtype=np.float64)
            if target.approach_offset_world is None
            else target.approach_offset_world
        )
        staged_precontact = precontact.copy()
        staged_precontact[:3, 3] += approach_offset
        safe = staged_precontact.copy()
        safe[2, 3] += self.config.safe_height_m
        local_continuation = horizontal_close and microwave_close_cycle > 1
        if horizontal_close and not local_continuation:
            # An open vertical door cannot be approached from directly above:
            # its top edge blocks the descent long before the mid-height
            # contact point.  Translate to clearance in the current frame,
            # rotate to the sensed door normal there, then descend outside the
            # door plane with open fingers.
            staged_safe = safe.copy()
            staged_safe[:3, :3] = np.asarray(
                self.robot.current_ee_pose(), dtype=np.float64
            )[:3, :3]
            self._record_move(
                staged_safe,
                target,
                Phase.APPROACH,
                GRIPPER_OPEN,
                False,
                attempts,
            )
        has_staging_offset = float(np.linalg.norm(approach_offset)) > 1e-8
        if local_continuation:
            # The first cycle's typed exit and ordinary retreat have already
            # established free space beside and above the last measured door
            # contact.  Repeating the reset-observation detour wastes the
            # bounded episode budget and reintroduces a long shoulder swing.
            # Use that *current* retreat height for one lateral/orientation
            # move, then descend outside the sensed door plane.  This is the
            # shortest axis-staged route that avoids combining a large XY
            # swing, descent and wrist rotation in one OSC request; contact
            # itself remains separately typed.
            local_safe = staged_precontact.copy()
            current_pose = np.asarray(
                self.robot.current_ee_pose(), dtype=np.float64
            )
            local_safe[2, 3] = max(
                float(current_pose[2, 3]),
                float(staged_precontact[2, 3] + self.config.safe_height_m),
            )
            self._record_move(
                local_safe,
                target,
                Phase.APPROACH,
                GRIPPER_OPEN,
                False,
                attempts,
            )
            self._record_move(
                staged_precontact,
                target,
                Phase.APPROACH,
                GRIPPER_OPEN,
                False,
                attempts,
            )
        elif has_staging_offset:
            # Move directly above the intended movable drawer-front slot.
            # The alternative globally clear descent slot can be across the
            # robot shoulder singularity; its sensor evidence is retained to
            # justify the bounded target-adjacent descent below, not imposed
            # as an unnecessary Cartesian detour.
            contact_safe = precontact.copy()
            contact_safe[2, 3] += self.config.safe_height_m
            self._record_move(
                contact_safe,
                target,
                Phase.APPROACH,
                GRIPPER_OPEN,
                False,
                attempts,
            )
        else:
            self._record_move(
                safe,
                target,
                Phase.APPROACH,
                GRIPPER_OPEN,
                False,
                attempts,
            )

        if closing and not horizontal_close:
            # A closed drawer or microwave door is pushed with closed fingers.
            # Requiring a blocked-width grasp here would reject the intended
            # surface contact.  Close while safely above the fixture, then
            # descend and approach the frozen RGB-D point in two segments.
            self._set_gripper(GRIPPER_CLOSE, Phase.GRASP, attempts)
        if not local_continuation:
            self._record_move(
                precontact if has_staging_offset else staged_precontact,
                target,
                Phase.APPROACH,
                GRIPPER_CLOSE if closing and not horizontal_close else GRIPPER_OPEN,
                has_staging_offset,
                attempts,
                target_adjacent_approach_residual_m=(
                    0.030 if has_staging_offset else None
                ),
            )
            if closing and goal.subject.label == "drawer":
                # Freeze only after the ordinary low precontact descent has
                # genuinely returned.  The stored pose is the reached public
                # TCP sample, never the commanded waypoint or drawer state.
                self._freeze_drawer_retry_safe_column(
                    target,
                    attempts,
                    expected_pose=precontact,
                )
        if horizontal_close:
            # Keep the jaws open until the visible vertical door edge is
            # between both pads.  Closing in free space produces a compact
            # one-millimetre tool that can touch with one fingertip and then
            # trace the requested hinge arc while sliding past the door.
            self._record_contact_move(
                pose,
                target,
                GRIPPER_OPEN,
                attempts,
                max_residual_m=0.055,
            )
            self._set_gripper(GRIPPER_CLOSE, Phase.GRASP, attempts)
            if not self.robot.grasp_confirmed(GraspMode.PINCH):
                raise ContactExecutionError(
                    "microwave close contact did not retain the visible door edge"
                )
        else:
            self._record_contact_move(
                pose,
                target,
                GRIPPER_CLOSE if closing else GRIPPER_OPEN,
                attempts,
                max_residual_m=0.055,
                allow_compact_unknown_contact=(
                    closing and goal.subject.label == "drawer"
                ),
            )
        rotary_start_pose: FloatArray | None = None
        grip_site_to_edge_world: FloatArray | None = None
        if target.rotation_center_world is not None:
            rotary_start_pose = np.asarray(
                self.robot.current_ee_pose(), dtype=np.float64
            ).copy()
            grip_site_to_edge_world = (
                target.point_world - rotary_start_pose[:3, 3]
            )
            if float(np.linalg.norm(grip_site_to_edge_world)) > 0.060:
                raise ContactExecutionError(
                    "microwave grip-site to visible-edge offset is implausible"
                )
            rotary_complete = self._execute_rotary_fixture_transfer(
                target,
                attempts,
                consumed_linear_distance_m=0.0,
                require_retained_grasp=horizontal_close,
                allow_mechanical_stop=horizontal_close,
                extra_close_angle_rad=(
                    self.config.microwave_supplemental_close_extension_rad
                    if horizontal_close and microwave_close_cycle > 1
                    else 0.0
                ),
                max_segment_angle_rad=(
                    self.config.microwave_supplemental_arc_segment_rad
                    if horizontal_close and microwave_close_cycle > 1
                    else None
                ),
                intermediate_position_tolerance_m=(
                    self.config.microwave_supplemental_intermediate_position_tolerance_m
                    if horizontal_close and microwave_close_cycle > 1
                    else None
                ),
            )
            manipulated = np.asarray(
                self.robot.current_ee_pose(), dtype=np.float64
            ).copy()
        else:
            rotary_complete = True
            manipulated = pose.copy()
            manipulated[:3, 3] += (
                target.manipulation_axis_world * target.manipulation_distance_m
            )
            self._record_move(
                manipulated,
                target,
                Phase.TRANSFER,
                GRIPPER_CLOSE,
                True,
                attempts,
                allow_compact_unknown_contact=(
                    closing and goal.subject.label == "drawer"
                ),
                mechanical_stop_residual_m=(
                    self.config.drawer_close_stop_residual_m
                    if goal.subject.label == "drawer"
                    else self.config.linear_close_stop_residual_m
                ),
            )
        if defer_drawer_release:
            if not (closing and goal.subject.label == "drawer"):
                raise ContactExecutionError(
                    "deferred release is only valid for a closing drawer"
                )
            return
        self._set_gripper(GRIPPER_OPEN, Phase.RELEASE, attempts)
        release_width: float | None = None
        microwave_release_proven = False
        # Build clearance from the *measured* contact endpoint.  A fixture can
        # stop before the commanded pose, so retreating from the nominal goal
        # can leave the wrist in front of the very handle the fresh RGB-D
        # verifier must see.  Return to the pre-detection public pose after a
        # short local clearance move; that pose already proved both views had
        # enough evidence to estimate this target.
        retreat = np.asarray(
            self.robot.current_ee_pose(), dtype=np.float64
        ).copy()
        remaining_outward_m = self.config.precontact_clearance_m
        remaining_lift_m = self.config.safe_height_m
        typed_exit_progress_m = 0.0
        if horizontal_close:
            # The released jaws begin at a sensor-proven contact, so a
            # free-space request is invalid at its first sample: inflated
            # tool clearance necessarily overlaps the just-manipulated door.
            # Make one tightly bounded, mostly-outward contact exit first;
            # its small upward component clears low foreground geometry.  It
            # is a RETREAT (not TRANSFER), has no mechanical-stop gate, and a
            # controller rejection remains fatal, so it cannot count as
            # semantic task completion.
            contact_exit_m = min(
                self.config.microwave_contact_exit_m,
                self.config.precontact_clearance_m,
            )
            contact_exit_lift_m = min(
                self.config.microwave_contact_exit_lift_m,
                self.config.safe_height_m,
            )
            contact_exit = retreat.copy()
            exit_delta = target.outward_world * contact_exit_m
            exit_delta = np.asarray(exit_delta, dtype=np.float64).copy()
            exit_delta[2] += contact_exit_lift_m
            contact_exit[:3, 3] += exit_delta
            exit_start = retreat[:3, 3].copy()
            self._record_move(
                contact_exit,
                target,
                Phase.RETREAT,
                GRIPPER_OPEN,
                True,
                attempts,
            )
            retreat = np.asarray(
                self.robot.current_ee_pose(), dtype=np.float64
            ).copy()
            exit_norm = float(np.linalg.norm(exit_delta))
            if exit_norm > 1e-8:
                typed_exit_progress_m = max(
                    0.0,
                    float(
                        np.dot(
                            retreat[:3, 3] - exit_start,
                            exit_delta / exit_norm,
                        )
                    ),
                )
            # The fingers continue opening during the short typed exit.  Read
            # release and Cartesian progress from this same terminal public
            # sample rather than from the earlier accepted command edge.
            release_width = self._public_gripper_width_m()
            microwave_release_proven = bool(
                release_width is not None
                and release_width >= self.config.microwave_release_min_width_m
            )
            remaining_outward_m -= contact_exit_m
            remaining_lift_m -= contact_exit_lift_m
        if self._microwave_terminal_exit_defer_allowed(
            final_goal=self._final_goal_context,
            supplemental_close=bool(
                horizontal_close
                and microwave_close_cycle > 1
                and rotary_complete
            ),
            release_proven=microwave_release_proven,
            typed_exit_progress_m=typed_exit_progress_m,
            required_progress_m=self.config.microwave_contact_exit_progress_m,
        ):
            attempts.append(
                PhaseAttempt(
                    microwave_close_cycle,
                    Phase.RETREAT,
                    1,
                    True,
                    "terminal supplemental microwave exit released at "
                    f"{release_width:.4f} m and advanced "
                    f"{typed_exit_progress_m:.4f} m; defer ordinary retreat "
                    "to strict fresh RGB-D verification",
                )
            )
            return
        retreat[:3, 3] += target.outward_world * remaining_outward_m
        retreat[2, 3] += remaining_lift_m
        self._record_move(retreat, target, Phase.RETREAT, GRIPPER_OPEN, False, attempts)
        needs_local_continuation = horizontal_close and not rotary_complete
        if not needs_local_continuation:
            current = np.asarray(self.robot.current_ee_pose(), dtype=np.float64)
            observation_position_error, observation_rotation_error = self._pose_errors(
                current, observation_pose
            )
            if (
                observation_position_error > self.config.position_tolerance_m
                or observation_rotation_error > self.config.orientation_tolerance_rad
            ):
                self._record_move(
                    observation_pose,
                    target,
                    Phase.RETREAT,
                    GRIPPER_OPEN,
                    False,
                    attempts,
                )
        if horizontal_close and not rotary_complete:
            if microwave_close_cycle >= self.config.microwave_max_regrasps:
                raise ContactExecutionError(
                    "microwave remained visibly short of the closed state after "
                    f"{self.config.microwave_max_regrasps} bounded local regrasps"
                )
            continuation = getattr(
                self.provider, "estimate_microwave_continuation", None
            )
            if not callable(continuation):
                raise ContactExecutionError(
                    "contact-target provider lacks microwave local reassociation"
                )
            frozen = microwave_frozen_target
            if (
                frozen is None
                or frozen.rotation_center_world is None
                or frozen.rotation_axis_world is None
            ):
                raise ContactExecutionError(
                    "microwave local continuation lacks a frozen hinge circle"
                )
            frozen_radial = frozen.point_world - frozen.rotation_center_world
            frozen_radial -= frozen.rotation_axis_world * float(
                np.dot(frozen_radial, frozen.rotation_axis_world)
            )
            frozen_radius = float(np.linalg.norm(frozen_radial))
            if not 0.10 <= frozen_radius <= 0.60:
                raise ContactExecutionError(
                    "microwave frozen hinge radius is implausible"
                )
            if rotary_start_pose is None or grip_site_to_edge_world is None:
                raise ContactExecutionError(
                    "microwave local continuation lacks the retained grip site"
                )
            predicted_anchor, achieved_angle = (
                self._co_rotated_microwave_edge_anchor(
                    target,
                    rotary_start_pose,
                    manipulated,
                )
            )
            refreshed = continuation(
                goal,
                predicted_anchor_world=predicted_anchor,
                frozen_hinge_world=frozen.rotation_center_world,
                frozen_rotation_axis_world=frozen.rotation_axis_world,
                frozen_radius_m=frozen_radius,
                anchor_radius_m=(
                    self.config.microwave_reassociation_anchor_radius_m
                ),
                radius_tolerance_m=(
                    self.config.microwave_verify_radius_tolerance_m
                ),
            )
            if (
                refreshed.rotation_center_world is None
                or refreshed.rotation_axis_world is None
                or refreshed.rotation_angle_rad is None
            ):
                raise ContactExecutionError(
                    "local microwave continuation lacked hinge geometry"
                )
            anchor_error = float(
                np.linalg.norm(refreshed.point_world - predicted_anchor)
            )
            refreshed_radial = (
                refreshed.point_world - frozen.rotation_center_world
            )
            refreshed_radial -= frozen.rotation_axis_world * float(
                np.dot(refreshed_radial, frozen.rotation_axis_world)
            )
            radius_error = abs(
                float(np.linalg.norm(refreshed_radial)) - frozen_radius
            )
            same_hinge = bool(
                np.linalg.norm(
                    refreshed.rotation_center_world
                    - frozen.rotation_center_world
                )
                <= 1e-6
                and abs(
                    float(
                        np.dot(
                            refreshed.rotation_axis_world,
                            frozen.rotation_axis_world,
                        )
                    )
                )
                >= 1.0 - 1e-6
            )
            associated = bool(
                same_hinge
                and anchor_error
                <= self.config.microwave_reassociation_anchor_radius_m
                and radius_error
                <= self.config.microwave_verify_radius_tolerance_m
            )
            attempts.append(
                PhaseAttempt(
                    microwave_close_cycle + 1,
                    Phase.VERIFY,
                    1,
                    associated,
                    "local RGB-D microwave edge reassociation: "
                    f"grip-site offset {float(np.linalg.norm(grip_site_to_edge_world)):.4f} m "
                    f"co-rotated through {achieved_angle:.4f} rad; "
                    f"anchor error {anchor_error:.4f} m, frozen-radius error "
                    f"{radius_error:.4f} m, same_hinge={same_hinge}",
                )
            )
            if not associated:
                raise ContactExecutionError(
                    "local microwave RGB-D edge failed frozen anchor-circle gates"
                )
            self._execute_linear_fixture(
                goal,
                refreshed,
                attempts,
                microwave_close_cycle=microwave_close_cycle + 1,
                microwave_frozen_target=frozen,
            )

    @staticmethod
    def _co_rotated_microwave_edge_anchor(
        target: ContactTargetEstimate,
        start_grip_pose: FloatArray,
        final_grip_pose: FloatArray,
    ) -> tuple[FloatArray, float]:
        """Carry the sensed grip-site-to-edge offset around the frozen hinge."""

        center = target.rotation_center_world
        axis = target.rotation_axis_world
        if center is None or axis is None:
            raise ContactExecutionError(
                "microwave edge anchor requires frozen hinge geometry"
            )
        start = np.asarray(start_grip_pose, dtype=np.float64)
        final = np.asarray(final_grip_pose, dtype=np.float64)
        if (
            start.shape != (4, 4)
            or final.shape != (4, 4)
            or not np.all(np.isfinite(start))
            or not np.all(np.isfinite(final))
        ):
            raise ContactExecutionError(
                "microwave grip poses must be finite transforms"
            )
        start_radial = start[:3, 3] - center
        start_radial -= axis * float(np.dot(start_radial, axis))
        final_radial = final[:3, 3] - center
        final_radial -= axis * float(np.dot(final_radial, axis))
        if (
            float(np.linalg.norm(start_radial)) < 1e-8
            or float(np.linalg.norm(final_radial)) < 1e-8
        ):
            raise ContactExecutionError(
                "microwave grip site is degenerate about the frozen hinge"
            )
        achieved_angle = float(
            np.arctan2(
                np.dot(axis, np.cross(start_radial, final_radial)),
                np.dot(start_radial, final_radial),
            )
        )
        grip_site_to_edge = target.point_world - start[:3, 3]
        rotation = Rotation.from_rotvec(axis * achieved_angle).as_matrix()
        anchor = final[:3, 3] + rotation @ grip_site_to_edge
        return anchor, achieved_angle

    def _execute_close_drawer_servo(
        self,
        goal: AtomicGoal,
        initial: ContactTargetEstimate,
        attempts: list[PhaseAttempt],
    ) -> None:
        """Close a drawer with bounded pushes and fresh RGB-D recontact."""

        current = initial
        required = min(
            self.config.drawer_close_required_progress_m,
            0.90 * initial.manipulation_distance_m,
        )
        for cycle in range(1, self.config.drawer_close_max_attempts + 1):
            self._drawer_close_plateau = None
            if cycle == 1:
                # Keep the measured compressive contact after the first push.
                # If fresh RGB-D says the drawer is still short, the next
                # bounded push can continue from this public endpoint instead
                # of attempting an unreachable second grasp.
                self._execute_linear_fixture(
                    goal, current, attempts, defer_drawer_release=True
                )
            else:
                # Re-observe the drawer after the first stop.  The handle is
                # retained as identity, but the pusher must be re-grounded on
                # a same-level RGB-D front plane.  This prevents a wrist that
                # has drifted into the cabinet middle frame from being
                # treated as a valid drawer contact.
                axis = np.asarray(initial.manipulation_axis_world, dtype=np.float64)
                axis /= float(np.linalg.norm(axis))
                front_point = current.drawer_front_point_world
                front_normal = current.drawer_front_normal_world
                if front_point is None or front_normal is None:
                    raise ContactExecutionError(
                        "fresh RGB-D drawer-front plane is unavailable"
                    )
                front_point = np.asarray(front_point, dtype=np.float64)
                front_normal = np.asarray(front_normal, dtype=np.float64)
                if (
                    front_point.shape != (3,)
                    or front_normal.shape != (3,)
                    or not np.all(np.isfinite(front_point))
                    or not np.all(np.isfinite(front_normal))
                    or float(np.dot(front_normal, current.outward_world)) < 0.82
                    or float(np.dot(axis, -front_normal)) < 0.82
                ):
                    raise ContactExecutionError(
                        "fresh drawer-front plane is inconsistent with close axis"
                    )
                try:
                    # Leave the old contact before re-entering.  First rise on
                    # the current public TCP column, then translate above the
                    # freshly measured front.  Separating those motions avoids
                    # the low diagonal cabinet crossing that failed after the
                    # first compliant push.
                    self._set_gripper(GRIPPER_OPEN, Phase.RELEASE, attempts)
                    self._recontact_drawer_from_high_clearance(current, attempts)
                    self._compact_drawer_retry_pusher(attempts)
                    high_pose = np.asarray(
                        self.robot.current_ee_pose(), dtype=np.float64
                    ).copy()
                    if (
                        high_pose.shape != (4, 4)
                        or not np.all(np.isfinite(high_pose))
                    ):
                        raise ContactExecutionError(
                            "public EE pose must be a finite 4x4 matrix"
                        )
                    # Descend on a column that is well outside the fresh
                    # front plane.  The plane fit may retain a modest vertical
                    # component, so use its horizontal projection for all
                    # approach offsets and keep the selected same-level RGB-D
                    # slot's height.  Mixing the plane's Z component into the
                    # v84 descent left a 128-mm, almost purely vertical stall.
                    front_surface, planar_outward = (
                        self._drawer_retry_planar_staging_geometry(current)
                    )
                    safe_column_xy = self._drawer_retry_safe_column_xy(
                        current,
                        front_surface=front_surface,
                        planar_outward=planar_outward,
                    )
                    high_column_error = float(
                        np.linalg.norm(high_pose[:2, 3] - safe_column_xy)
                    )
                    if high_column_error > self.config.position_tolerance_m:
                        raise ContactExecutionError(
                            "drawer retry high transit did not reach the proven "
                            f"safe column: error={high_column_error:.4f} m"
                        )
                    safe_pose = high_pose.copy()
                    # Preserve the actually reached high-column XY verbatim:
                    # no accepted lateral residual may be hidden inside the
                    # descent.  Fresh RGB-D/SDF still owns the new height and
                    # every following horizontal approach segment.
                    safe_pose[2, 3] = front_surface[2]
                    descent = float(high_pose[2, 3] - safe_pose[2, 3])
                    if not 0.0 <= descent <= 0.25:
                        raise ContactExecutionError(
                            "drawer retry staged descent exceeds its 0.25 m bound"
                        )
                    self._record_move(
                        safe_pose,
                        current,
                        Phase.APPROACH,
                        GRIPPER_CLOSE,
                        False,
                        attempts,
                    )
                    # Keep ordinary clearance while moving from the proven
                    # column to a 70-mm entry, then use one strictly typed,
                    # horizontal and monotonic target-adjacent corridor.  The
                    # contact observer may remove only the fresh drawer's
                    # sensor-bound self surface; unrelated SDF fields retain
                    # the unchanged free-space inflation throughout.
                    corridor_plateau = self._execute_drawer_retry_contact_corridor(
                        current,
                        front_surface=front_surface,
                        planar_outward=planar_outward,
                        attempts=attempts,
                    )
                    if corridor_plateau is not None:
                        # The target-adjacent GRASP itself remained rejected;
                        # only its structured public motion prefix is carried
                        # here.  Re-observe immediately, associate the same
                        # drawer level/front, and accept only the unchanged
                        # accumulated RGB-D close target with a <=3-mm gap.
                        try:
                            fresh_after = self.provider.estimate(goal)
                        except (LookupError, ValueError) as refresh_exc:
                            raise ContactExecutionError(
                                "typed drawer corridor plateau could not obtain "
                                "fresh public RGB-D"
                            ) from refresh_exc
                        plateau = self._bind_drawer_close_motion_plateau(
                            motion=corridor_plateau,
                            fresh_before=current,
                            fresh_after=fresh_after,
                            axis=axis,
                        )
                        self._drawer_close_plateau = plateau
                        projected = float(
                            np.dot(
                                fresh_after.point_world - initial.point_world,
                                initial.manipulation_axis_world,
                            )
                        )
                        plateau_completion = (
                            self._drawer_close_plateau_visual_completion(
                                evidence=plateau,
                                refreshed=fresh_after,
                                axis=initial.manipulation_axis_world,
                                projected_m=projected,
                                required_m=required,
                            )
                        )
                        attempts.append(
                            PhaseAttempt(
                                cycle,
                                Phase.VERIFY,
                                1,
                                plateau_completion,
                                "rejected typed drawer corridor public plateau: "
                                f"actions={corridor_plateau.policy_actions}, "
                                f"inward={corridor_plateau.signed_inward_m:.4f} m, "
                                f"cross={corridor_plateau.cross_drift_m:.4f} m, "
                                f"rotation={corridor_plateau.rotation_drift_rad:.4f} rad, "
                                f"width={corridor_plateau.width_m:.4f} m; fresh "
                                f"RGB-D close={projected:.4f}/{required:.4f} m",
                            )
                        )
                        if plateau_completion:
                            self._release_drawer_after_close(fresh_after, attempts)
                            return
                        raise ContactExecutionError(
                            "rejected typed drawer corridor lacked same-front "
                            "fresh RGB-D completion within the 3-mm gap"
                        )
                    load_start = self._public_motion_endpoint_pose()
                    load_goal = load_start.copy()
                    load_goal[:3, 3] += axis * self.config.drawer_close_load_proof_m
                    self._record_move(
                        load_goal,
                        current,
                        Phase.TRANSFER,
                        GRIPPER_CLOSE,
                        True,
                        attempts,
                        allow_compact_unknown_contact=True,
                        mechanical_stop_residual_m=self.config.drawer_close_stop_residual_m,
                    )
                    load_end = self._public_motion_endpoint_pose()
                    measured_load = float(
                        np.dot(load_end[:3, 3] - load_start[:3, 3], axis)
                    )
                    width = self._public_gripper_width_m()
                    if measured_load < 0.008 or (
                        width is not None
                        and width > self.config.drawer_close_compact_width_max_m
                    ):
                        raise ContactExecutionError(
                            "drawer front load proof was insufficient: "
                            f"inward={measured_load:.4f} m, width={width}"
                        )
                    # Freeze the *loaded* public endpoint after load proof.
                    # Reconstructing a fresh front point here can put the
                    # first continuation target back outside the drawer (the
                    # close axis is opposite outward), causing the controller
                    # to undo its measured load.  All continuation goals are
                    # therefore absolute poses based on this endpoint and
                    # advance monotonically along the frozen sensor axis.
                    loaded_base = load_end.copy()
                    push_start = loaded_base.copy()
                    cumulative_push = 0.0
                    segment_progress: list[float] = []
                    segment_length = min(
                        0.015,
                        self.config.drawer_close_extra_push_m / 3.0,
                    )
                    fresh_before = current
                    for push_goal in self._drawer_continuation_goals(
                        loaded_base, axis, segment_length, segment_count=3
                    ):
                        command_start = self._public_motion_endpoint_pose()
                        steps_before = getattr(self.robot, "steps_executed", None)
                        mechanical_stop = self._record_move(
                            push_goal,
                            current,
                            Phase.TRANSFER,
                            GRIPPER_CLOSE,
                            True,
                            attempts,
                            allow_compact_unknown_contact=True,
                            mechanical_stop_residual_m=self.config.drawer_close_stop_residual_m,
                        )
                        push_end = self._public_motion_endpoint_pose()
                        steps_after = getattr(self.robot, "steps_executed", None)
                        segment = float(
                            np.dot(push_end[:3, 3] - push_start[:3, 3], axis)
                        )
                        segment_progress.append(segment)
                        cumulative_push += max(0.0, segment)
                        push_start = push_end
                        fresh_after = self.provider.estimate(goal)
                        action_count = (
                            steps_after - steps_before
                            if isinstance(steps_before, int)
                            and not isinstance(steps_before, bool)
                            and isinstance(steps_after, int)
                            and not isinstance(steps_after, bool)
                            else -1
                        )
                        plateau = self._capture_drawer_close_plateau_evidence(
                            mechanical_stop=mechanical_stop,
                            command_start=command_start,
                            command_goal=push_goal,
                            command_end=push_end,
                            axis=axis,
                            policy_actions=action_count,
                            width_m=self._public_gripper_width_m(),
                            fresh_before=fresh_before,
                            fresh_after=fresh_after,
                        )
                        self._drawer_close_plateau = plateau
                        fresh_segment = float(
                            np.dot(
                                fresh_after.point_world - fresh_before.point_world,
                                axis,
                            )
                        )
                        attempts.append(
                            PhaseAttempt(
                                cycle,
                                Phase.VERIFY,
                                1,
                                plateau is not None or fresh_segment > 0.001,
                                "fresh RGB-D drawer segment progress="
                                f"{fresh_segment:.4f} m; "
                                f"typed_plateau={plateau is not None}",
                            )
                        )
                        if plateau is not None:
                            break
                        self._fresh_drawer_segment_progress(
                            fresh_before, fresh_after, axis
                        )
                        fresh_before = fresh_after
                    if (
                        cumulative_push < 0.008
                        and self._drawer_close_plateau is None
                    ):
                        raise ContactExecutionError(
                            "drawer close continuation made insufficient inward progress: "
                            f"{cumulative_push:.4f} m over segments={segment_progress}"
                        )
                except (ContactExecutionError, ExecutionError, OptimisationError) as exc:
                    attempts.append(
                        PhaseAttempt(
                            cycle,
                            Phase.TRANSFER,
                            1,
                            False,
                            f"sensor drawer-front pusher candidate rejected: {exc}",
                        )
                    )
                    if cycle < self.config.drawer_close_max_attempts:
                        # Motion during a rejected approach invalidates the old
                        # contact plane.  Reacquire it before the sole remaining
                        # retry; a stale front is never reused as a policy input.
                        try:
                            retry_target = self.provider.estimate(goal)
                        except (LookupError, ValueError) as refresh_exc:
                            raise ContactExecutionError(
                                "drawer retry could not reacquire fresh public RGB-D"
                            ) from refresh_exc
                        if retry_target.confidence <= 0.0:
                            raise ContactExecutionError(
                                "drawer retry fresh RGB-D confidence is zero"
                            )
                        current = retry_target
                        attempts.append(
                            PhaseAttempt(
                                cycle,
                                Phase.VERIFY,
                                1,
                                True,
                                "reacquired fresh RGB-D drawer front after "
                                "rejected bounded recontact",
                            )
                        )
                    continue
            refreshed = self.provider.estimate(goal)
            projected = float(
                np.dot(
                    refreshed.point_world - initial.point_world,
                    initial.manipulation_axis_world,
                )
            )
            plateau_completion = self._drawer_close_plateau_visual_completion(
                evidence=self._drawer_close_plateau,
                refreshed=refreshed,
                axis=initial.manipulation_axis_world,
                projected_m=projected,
                required_m=required,
            )
            success = projected >= required or plateau_completion
            attempts.append(
                PhaseAttempt(
                    cycle,
                    Phase.VERIFY,
                    1,
                    success,
                    "fresh RGB-D drawer-close progress="
                    f"{projected:.4f}/{required:.4f} m; "
                    f"typed_plateau_fusion={plateau_completion}",
                )
            )
            if success:
                self._release_drawer_after_close(refreshed, attempts)
                return
            current = refreshed
        self._release_drawer_after_close(current, attempts)
        raise ContactExecutionError(
            "fresh RGB-D drawer close remained short after "
            f"{self.config.drawer_close_max_attempts} bounded pushes"
        )

    @staticmethod
    def _drawer_continuation_goals(
        loaded_base: FloatArray,
        axis: FloatArray,
        segment_length: float,
        *,
        segment_count: int,
    ) -> tuple[FloatArray, ...]:
        """Build monotonic drawer continuation poses from a loaded endpoint.

        ``loaded_base`` is the public EE pose at which the closed fingers
        passed the load proof.  Keeping its height, lateral offset, and wrist
        orientation fixed prevents a re-approach to a stale/fresh RGB-D
        surface from retracting the drawer before the next push.
        """

        base = np.asarray(loaded_base, dtype=np.float64)
        direction = np.asarray(axis, dtype=np.float64)
        if (
            base.shape != (4, 4)
            or not np.all(np.isfinite(base))
            or direction.shape != (3,)
            or not np.all(np.isfinite(direction))
        ):
            raise ContactExecutionError(
                "drawer continuation base and axis must be finite"
            )
        norm = float(np.linalg.norm(direction))
        if norm < 1e-8 or not np.isfinite(segment_length) or segment_length <= 0:
            raise ContactExecutionError(
                "drawer continuation axis and segment length must be positive"
            )
        if not isinstance(segment_count, int) or segment_count < 1:
            raise ContactExecutionError(
                "drawer continuation segment count must be positive"
            )
        direction /= norm
        goals: list[FloatArray] = []
        for index in range(1, segment_count + 1):
            goal = base.copy()
            goal[:3, 3] += direction * (segment_length * index)
            goals.append(goal)
        return tuple(goals)

    @staticmethod
    def _fresh_drawer_segment_progress(
        before: ContactTargetEstimate,
        after: ContactTargetEstimate,
        axis: FloatArray,
        *,
        minimum_m: float = 0.001,
    ) -> float:
        """Return one positive fresh RGB-D segment displacement or reject it."""

        direction = np.asarray(axis, dtype=np.float64)
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ContactExecutionError("drawer fresh-progress axis is invalid")
        norm = float(np.linalg.norm(direction))
        if norm < 1e-8 or not np.isfinite(minimum_m) or minimum_m <= 0.0:
            raise ContactExecutionError("drawer fresh-progress gate is invalid")
        direction /= norm
        delta = float(np.dot(after.point_world - before.point_world, direction))
        if not np.isfinite(delta) or delta <= minimum_m:
            raise ContactExecutionError(
                "drawer continuation lacked fresh RGB-D progress: "
                f"{delta:.4f} m"
            )
        return delta

    def _same_drawer_close_front_association(
        self,
        before: ContactTargetEstimate,
        after: ContactTargetEstimate,
        axis: FloatArray,
    ) -> bool:
        """Conservatively bind two fresh estimates to one drawer-front layer."""

        direction = np.asarray(axis, dtype=np.float64)
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            return False
        norm = float(np.linalg.norm(direction))
        if norm < 1e-8:
            return False
        direction = direction / norm
        if abs(float(direction[2])) > 1e-6:
            return False
        if tuple(label.lower() for label in before.requested_labels) != tuple(
            label.lower() for label in after.requested_labels
        ):
            return False
        if before.confidence <= 0.0 or after.confidence <= 0.0:
            return False
        if (
            before.drawer_front_point_world is None
            or after.drawer_front_point_world is None
            or before.drawer_front_normal_world is None
            or after.drawer_front_normal_world is None
            or before.drawer_front_support_m is None
            or after.drawer_front_support_m is None
        ):
            return False
        before_front = np.asarray(before.drawer_front_point_world, dtype=np.float64)
        after_front = np.asarray(after.drawer_front_point_world, dtype=np.float64)
        before_normal = np.asarray(
            before.drawer_front_normal_world, dtype=np.float64
        )
        after_normal = np.asarray(after.drawer_front_normal_world, dtype=np.float64)
        values = (
            before.point_world,
            after.point_world,
            before_front,
            after_front,
            before_normal,
            after_normal,
        )
        if any(value.shape != (3,) or not np.all(np.isfinite(value)) for value in values):
            return False
        before_normal_norm = float(np.linalg.norm(before_normal))
        after_normal_norm = float(np.linalg.norm(after_normal))
        if before_normal_norm < 1e-8 or after_normal_norm < 1e-8:
            return False
        before_normal /= before_normal_norm
        after_normal /= after_normal_norm
        point_delta = after.point_world - before.point_world
        front_delta = after_front - before_front
        point_cross = point_delta - float(np.dot(point_delta, direction)) * direction
        front_cross = front_delta - float(np.dot(front_delta, direction)) * direction
        support = np.asarray(
            (before.drawer_front_support_m, after.drawer_front_support_m),
            dtype=np.float64,
        )
        return bool(
            np.all(np.isfinite(support))
            and np.all(support > 0.0)
            and abs(float(before.point_world[2] - after.point_world[2]))
            <= self.config.position_tolerance_m
            and abs(float(before_front[2] - after_front[2]))
            <= self.config.position_tolerance_m
            and float(np.linalg.norm(point_cross))
            <= self.config.position_tolerance_m
            and float(np.linalg.norm(front_cross))
            <= self.config.position_tolerance_m
            and float(np.dot(before.manipulation_axis_world, direction)) >= 0.95
            and float(np.dot(after.manipulation_axis_world, direction)) >= 0.95
            and float(np.dot(before_normal, after_normal)) >= 0.95
            and float(np.dot(before_normal, before.outward_world)) >= 0.82
            and float(np.dot(after_normal, after.outward_world)) >= 0.82
            and float(np.dot(direction, -before_normal)) >= 0.82
            and float(np.dot(direction, -after_normal)) >= 0.82
        )

    def _capture_drawer_close_motion_plateau(
        self,
        *,
        command_start: FloatArray,
        command_goal: FloatArray,
        command_end: FloatArray,
        axis: FloatArray,
        policy_actions: int,
        width_m: float | None,
    ) -> _DrawerCloseMotionPlateauEvidence | None:
        """Validate only public motion from a bounded typed close command."""

        start = np.asarray(command_start, dtype=np.float64)
        goal = np.asarray(command_goal, dtype=np.float64)
        endpoint = np.asarray(command_end, dtype=np.float64)
        direction = np.asarray(axis, dtype=np.float64)
        if (
            start.shape != (4, 4)
            or goal.shape != (4, 4)
            or endpoint.shape != (4, 4)
            or direction.shape != (3,)
            or not np.all(np.isfinite(start))
            or not np.all(np.isfinite(goal))
            or not np.all(np.isfinite(endpoint))
            or not np.all(np.isfinite(direction))
        ):
            return None
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm < 1e-8:
            return None
        direction /= direction_norm
        command = goal[:3, 3] - start[:3, 3]
        command_length = float(np.linalg.norm(command))
        command_signed = float(np.dot(command, direction))
        command_cross = float(
            np.linalg.norm(command - command_signed * direction)
        )
        actual = endpoint[:3, 3] - start[:3, 3]
        signed = float(np.dot(actual, direction))
        cross = float(np.linalg.norm(actual - signed * direction))
        _, rotation = self._pose_errors(endpoint, start)
        if (
            not isinstance(policy_actions, int)
            or isinstance(policy_actions, bool)
            or policy_actions <= 0
            or width_m is None
            or not np.isfinite(width_m)
            or width_m < 0.0
            or width_m
            > min(
                self.config.drawer_close_compact_width_max_m,
                self._DRAWER_CLOSE_PLATEAU_MAX_WIDTH_M,
            )
            or not np.all(
                np.isfinite(
                    (command_length, command_signed, command_cross, signed, cross, rotation)
                )
            )
            or abs(float(direction[2])) > 1e-6
            or abs(float(command[2])) > 1e-6
            or command_length <= 0.0
            or command_length > self._DRAWER_CLOSE_PLATEAU_MAX_COMMAND_M
            or command_signed <= 0.0
            or command_cross > 1e-6
            or signed < self._DRAWER_CLOSE_PLATEAU_MIN_INWARD_M
            or cross > self._DRAWER_CLOSE_PLATEAU_MAX_CROSS_M
            or rotation > self.config.orientation_tolerance_rad
        ):
            return None
        return _DrawerCloseMotionPlateauEvidence(
            command_length_m=command_length,
            signed_inward_m=signed,
            cross_drift_m=cross,
            rotation_drift_rad=rotation,
            width_m=float(width_m),
            policy_actions=policy_actions,
        )

    def _bind_drawer_close_motion_plateau(
        self,
        *,
        motion: _DrawerCloseMotionPlateauEvidence,
        fresh_before: ContactTargetEstimate,
        fresh_after: ContactTargetEstimate,
        axis: FloatArray,
    ) -> _DrawerClosePlateauEvidence | None:
        """Bind motion evidence to one freshly associated drawer-front layer."""

        if not self._same_drawer_close_front_association(
            fresh_before, fresh_after, axis
        ):
            return None
        return _DrawerClosePlateauEvidence(
            fresh_target=fresh_after,
            command_length_m=motion.command_length_m,
            signed_inward_m=motion.signed_inward_m,
            cross_drift_m=motion.cross_drift_m,
            rotation_drift_rad=motion.rotation_drift_rad,
            width_m=motion.width_m,
            policy_actions=motion.policy_actions,
        )

    def _capture_drawer_close_plateau_evidence(
        self,
        *,
        mechanical_stop: bool,
        command_start: FloatArray,
        command_goal: FloatArray,
        command_end: FloatArray,
        axis: FloatArray,
        policy_actions: int,
        width_m: float | None,
        fresh_before: ContactTargetEstimate,
        fresh_after: ContactTargetEstimate,
    ) -> _DrawerClosePlateauEvidence | None:
        """Validate a typed horizontal close command and its public plateau."""

        if not mechanical_stop:
            return None
        motion = self._capture_drawer_close_motion_plateau(
            command_start=command_start,
            command_goal=command_goal,
            command_end=command_end,
            axis=axis,
            policy_actions=policy_actions,
            width_m=width_m,
        )
        if motion is None:
            return None
        return self._bind_drawer_close_motion_plateau(
            motion=motion,
            fresh_before=fresh_before,
            fresh_after=fresh_after,
            axis=axis,
        )

    def _drawer_close_plateau_visual_completion(
        self,
        *,
        evidence: _DrawerClosePlateauEvidence | None,
        refreshed: ContactTargetEstimate,
        axis: FloatArray,
        projected_m: float,
        required_m: float,
    ) -> bool:
        """Fuse only a <=3-mm visual shortfall with a valid typed plateau."""

        gap = float(required_m - projected_m)
        return bool(
            evidence is not None
            and np.isfinite(gap)
            and 0.0
            < gap
            <= self._DRAWER_CLOSE_PLATEAU_MAX_VISUAL_GAP_M + 1e-12
            and self._same_drawer_close_front_association(
                evidence.fresh_target, refreshed, axis
            )
        )

    def _public_motion_endpoint_pose(self) -> FloatArray:
        """Return the latest endpoint sample exposed by the motion adapter.

        Some OSC adapters publish one observation behind the accepted
        partial-contact waypoint.  The controller's phase trace stores the
        goal-minus-current vector from that same public proprioceptive sample;
        reconstructing only its Cartesian position avoids restarting a drawer
        pusher from a stale observation.  No evaluator or simulator state is
        consulted, and a normal current-pose read remains the fallback for
        minimal test robots.
        """

        fallback = np.asarray(self.robot.current_ee_pose(), dtype=np.float64)
        trace = getattr(self.robot, "phase_trace", ())
        if not trace:
            return fallback
        latest = trace[-1]
        if not isinstance(latest, dict):
            return fallback
        goal_xyz = np.asarray(latest.get("optimizer_goal_xyz_m"), dtype=np.float64)
        residual_xyz = np.asarray(
            latest.get("final_goal_minus_current_xyz_m"), dtype=np.float64
        )
        if (
            goal_xyz.shape != (3,)
            or residual_xyz.shape != (3,)
            or not np.all(np.isfinite(goal_xyz))
            or not np.all(np.isfinite(residual_xyz))
        ):
            return fallback
        endpoint = fallback.copy()
        endpoint[:3, 3] = goal_xyz - residual_xyz
        return endpoint

    def _release_drawer_after_close(
        self,
        target: ContactTargetEstimate,
        attempts: list[PhaseAttempt],
    ) -> None:
        """Release a retained close contact and leave through sensor clearance."""

        self._set_gripper(GRIPPER_OPEN, Phase.RELEASE, attempts)
        retreat = np.asarray(self.robot.current_ee_pose(), dtype=np.float64).copy()
        if retreat.shape != (4, 4) or not np.all(np.isfinite(retreat)):
            raise ContactExecutionError("public EE pose must be a finite 4x4 matrix")
        retreat[:3, 3] += target.outward_world * self.config.precontact_clearance_m
        retreat[2, 3] += self.config.safe_height_m
        self._record_move(
            retreat,
            target,
            Phase.RETREAT,
            GRIPPER_OPEN,
            False,
            attempts,
        )

    def _freeze_drawer_retry_safe_column(
        self,
        target: ContactTargetEstimate,
        attempts: list[PhaseAttempt],
        *,
        expected_pose: FloatArray,
    ) -> None:
        """Freeze the first reached low drawer precontact column.

        This method is called only after the low precontact move returned
        successfully.  Its evidence is therefore public proprioception plus
        the sensor-derived outward ray that produced that move.  Later fresh
        observations may supply a new slot and height, but they cannot erase
        this episode-local reachability evidence.
        """

        if self._drawer_retry_safe_column is not None:
            return
        pose = np.asarray(self.robot.current_ee_pose(), dtype=np.float64)
        expected = np.asarray(expected_pose, dtype=np.float64)
        outward = np.asarray(target.outward_world, dtype=np.float64).copy()
        outward[2] = 0.0
        outward_norm = float(np.linalg.norm(outward))
        if (
            pose.shape != (4, 4)
            or expected.shape != (4, 4)
            or not np.all(np.isfinite(pose))
            or not np.all(np.isfinite(expected))
            or not np.all(np.isfinite(outward))
            or outward_norm < 0.82
        ):
            raise ContactExecutionError(
                "completed drawer descent lacks a finite horizontal public safe column"
            )
        position_error = float(np.linalg.norm(pose[:3, 3] - expected[:3, 3]))
        if position_error > self.config.position_tolerance_m:
            raise ContactExecutionError(
                "drawer low descent did not reach the public safe column: "
                f"error={position_error:.4f} m"
            )
        outward /= outward_norm
        self._drawer_retry_safe_column = _DrawerRetrySafeColumn(
            pose.copy(), outward.copy()
        )
        attempts.append(
            PhaseAttempt(
                1,
                Phase.VERIFY,
                1,
                True,
                "froze first completed low drawer descent from public EE "
                f"column=({pose[0, 3]:.4f}, {pose[1, 3]:.4f}) m",
            )
        )

    def _drawer_retry_safe_column_xy(
        self,
        target: ContactTargetEstimate,
        *,
        front_surface: FloatArray,
        planar_outward: FloatArray,
    ) -> FloatArray:
        """Combine a fresh slot with the proven column's outward coordinate."""

        frozen = self._drawer_retry_safe_column
        if frozen is None:
            raise ContactExecutionError(
                "drawer retry lacks a proven safe column from the first low descent"
            )
        surface = np.asarray(front_surface, dtype=np.float64)
        outward = np.asarray(planar_outward, dtype=np.float64)
        pose = np.asarray(frozen.pose, dtype=np.float64)
        frozen_outward = np.asarray(
            frozen.planar_outward_world, dtype=np.float64
        )
        if (
            surface.shape != (3,)
            or outward.shape != (3,)
            or pose.shape != (4, 4)
            or frozen_outward.shape != (3,)
            or not np.all(np.isfinite(surface))
            or not np.all(np.isfinite(outward))
            or not np.all(np.isfinite(pose))
            or not np.all(np.isfinite(frozen_outward))
        ):
            raise ContactExecutionError("drawer retry proven safe column is invalid")
        if float(np.dot(frozen_outward, outward)) < 0.95:
            raise ContactExecutionError(
                "drawer retry proven safe column disagrees with the fresh front normal"
            )
        delta_xy = pose[:2, 3] - surface[:2]
        proven_clearance = float(np.dot(delta_xy, outward[:2]))
        tangent = np.array((-outward[1], outward[0]), dtype=np.float64)
        lateral_delta = abs(float(np.dot(delta_xy, tangent)))
        support = target.drawer_front_support_m
        if support is None or not np.isfinite(support) or support <= 0.0:
            raise ContactExecutionError(
                "drawer retry fresh front lacks a finite support span"
            )
        lateral_limit = min(0.080, max(0.030, 0.5 * float(support)))
        if not np.isfinite(proven_clearance) or proven_clearance < 0.004:
            raise ContactExecutionError(
                "drawer retry proven safe column is not outside the fresh front"
            )
        if (
            proven_clearance
            > self.config.drawer_retry_safe_column_max_clearance_m
            or lateral_delta > lateral_limit
        ):
            raise ContactExecutionError(
                "drawer retry proven safe column is inconsistent with the fresh slot"
            )
        clearance = max(self.config.precontact_clearance_m, proven_clearance)
        safe_xy = surface[:2] + outward[:2] * clearance
        if safe_xy.shape != (2,) or not np.all(np.isfinite(safe_xy)):
            raise ContactExecutionError(
                "drawer retry proven safe column projection is invalid"
            )
        return safe_xy

    def _recontact_drawer_from_high_clearance(
        self,
        target: ContactTargetEstimate,
        attempts: list[PhaseAttempt],
    ) -> None:
        """Move later close retries above the cabinet before descending.

        After a compliant first push, the ordinary retreat can leave the arm
        on the opposite side of the cabinet from the freshly observed
        handle.  A direct diagonal re-approach then collides with the front
        panel or becomes unreachable.  Translate at the current public
        orientation and a sensor-derived high z first; the normal typed
        approach below still performs all contact and clearance checks.
        """

        current = np.asarray(self.robot.current_ee_pose(), dtype=np.float64)
        if current.shape != (4, 4) or not np.all(np.isfinite(current)):
            raise ContactExecutionError("public EE pose must be a finite 4x4 matrix")
        front_point = target.drawer_front_point_world
        front_normal = target.drawer_front_normal_world
        if front_point is None or front_normal is None:
            raise ContactExecutionError(
                "fresh RGB-D drawer-front plane is unavailable"
            )
        front_point = np.asarray(front_point, dtype=np.float64)
        front_normal = np.asarray(front_normal, dtype=np.float64)
        front_surface, planar_outward = (
            self._drawer_retry_planar_staging_geometry(target)
        )
        safe_column_xy = self._drawer_retry_safe_column_xy(
            target,
            front_surface=front_surface,
            planar_outward=planar_outward,
        )
        if float(np.dot(front_normal, target.outward_world)) < 0.82:
            raise ContactExecutionError(
                "fresh drawer-front plane disagrees with the public outward ray"
            )
        outside_distance = float(
            np.dot(current[:3, 3] - front_point, front_normal)
        )
        if not np.isfinite(outside_distance) or outside_distance < 0.004:
            raise ContactExecutionError(
                "public TCP is not outside fresh drawer front before retry"
            )
        high_z = max(
            float(current[2, 3]),
            float(target.point_world[2] + self.config.safe_height_m + 0.10),
        )
        lift_distance = high_z - float(current[2, 3])
        if lift_distance > 0.30:
            raise ContactExecutionError(
                "drawer retry high-clearance lift exceeds its 0.30 m bound"
            )
        lift = current.copy()
        lift[2, 3] = high_z
        self._record_move(
            lift,
            target,
            Phase.RETREAT,
            GRIPPER_OPEN,
            False,
            attempts,
        )
        transit = lift.copy()
        transit[:2, 3] = safe_column_xy
        transit[2, 3] = high_z
        lateral_distance = float(
            np.linalg.norm(transit[:2, 3] - lift[:2, 3])
        )
        if lateral_distance > 0.30:
            raise ContactExecutionError(
                "drawer retry high-clearance translation exceeds its 0.30 m bound"
            )
        self._record_move(
            transit,
            target,
            Phase.APPROACH,
            GRIPPER_OPEN,
            False,
            attempts,
        )
        reached = np.asarray(self.robot.current_ee_pose(), dtype=np.float64)
        if reached.shape != (4, 4) or not np.all(np.isfinite(reached)):
            raise ContactExecutionError(
                "drawer retry high transit lacks finite public proprioception"
            )
        column_error = float(
            np.linalg.norm(reached[:2, 3] - safe_column_xy)
        )
        if column_error > self.config.position_tolerance_m:
            raise ContactExecutionError(
                "drawer retry high transit did not reach the proven safe column: "
                f"error={column_error:.4f} m"
            )
        clearance = float(
            np.dot(safe_column_xy - front_surface[:2], planar_outward[:2])
        )
        attempts.append(
            PhaseAttempt(
                1,
                Phase.VERIFY,
                1,
                True,
                "drawer retry reached fresh-slot/proven-column high transit "
                f"at outward clearance={clearance:.4f} m",
            )
        )

    @staticmethod
    def _drawer_retry_planar_contact_geometry(
        target: ContactTargetEstimate,
    ) -> tuple[FloatArray, FloatArray]:
        """Project a fresh drawer slot onto its front at constant height.

        Both inputs are public RGB-D estimates.  The robust front fit is
        allowed a bounded vertical component, but a vertical component is not
        a valid direction for a horizontal drawer push.  Project the tracked,
        same-level slot onto that plane along the plane's horizontal normal;
        this preserves its lateral coordinate and contact height.
        """

        front_point = target.drawer_front_point_world
        front_normal = target.drawer_front_normal_world
        reference = np.asarray(target.point_world, dtype=np.float64)
        if front_point is None or front_normal is None:
            raise ContactExecutionError(
                "fresh RGB-D drawer-front plane is unavailable"
            )
        point = np.asarray(front_point, dtype=np.float64)
        normal = np.asarray(front_normal, dtype=np.float64)
        if (
            point.shape != (3,)
            or normal.shape != (3,)
            or reference.shape != (3,)
            or not np.all(np.isfinite(point))
            or not np.all(np.isfinite(normal))
            or not np.all(np.isfinite(reference))
        ):
            raise ContactExecutionError("fresh drawer-front plane is invalid")
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm < 1e-8:
            raise ContactExecutionError("fresh drawer-front normal is degenerate")
        normal /= normal_norm
        planar_outward = normal.copy()
        planar_outward[2] = 0.0
        planar_norm = float(np.linalg.norm(planar_outward))
        if planar_norm < 0.82:
            raise ContactExecutionError(
                "fresh drawer-front normal lacks a horizontal approach"
            )
        planar_outward /= planar_norm
        denominator = float(np.dot(planar_outward, normal))
        if denominator < 0.82:
            raise ContactExecutionError(
                "fresh drawer-front horizontal projection is unstable"
            )
        signed_distance = float(np.dot(reference - point, normal))
        surface = reference - planar_outward * (
            signed_distance / denominator
        )
        if (
            not np.all(np.isfinite(surface))
            or abs(float(surface[2] - reference[2])) > 1e-9
            or abs(float(np.dot(surface - point, normal))) > 1e-6
        ):
            raise ContactExecutionError(
                "fresh drawer-front constant-height projection is invalid"
            )
        return surface, planar_outward

    def _drawer_retry_planar_staging_geometry(
        self,
        target: ContactTargetEstimate,
    ) -> tuple[FloatArray, FloatArray]:
        """Project one fresh RGB-D approach offset onto the front tangent.

        The detector's close-drawer approach offset denotes a nearby slot that
        was clearer than the selected contact column in the same fresh SDF.
        It may move a retry only along the horizontal tangent of the measured
        drawer front.  Small normal-fit residual is projected away; larger
        outward/vertical content, an excessive shift, or leaving the measured
        front support all fail closed.
        """

        surface, outward = self._drawer_retry_planar_contact_geometry(target)
        if target.approach_offset_world is None:
            return surface, outward
        offset = np.asarray(
            target.approach_offset_world, dtype=np.float64
        ).copy()
        if (
            offset.shape != (3,)
            or not np.all(np.isfinite(offset))
        ):
            raise ContactExecutionError(
                "drawer retry staging offset is not a finite xyz vector"
            )
        if float(np.linalg.norm(offset)) <= 1e-8:
            return surface, outward

        tangent = np.array(
            (-outward[1], outward[0], 0.0), dtype=np.float64
        )
        signed_tangent = float(np.dot(offset, tangent))
        projected = tangent * signed_tangent
        nontangent = float(np.linalg.norm(offset - projected))
        if (
            nontangent > self.config.drawer_retry_staging_max_nontangent_m
            or nontangent > 0.10 * abs(signed_tangent)
        ):
            raise ContactExecutionError(
                "drawer retry staging offset is not purely tangential: "
                f"residual={nontangent:.4f} m"
            )
        if (
            not np.isfinite(signed_tangent)
            or abs(signed_tangent)
            > self.config.drawer_retry_staging_max_offset_m
        ):
            raise ContactExecutionError(
                "drawer retry staging offset exceeds its bounded amplitude"
            )

        front_point = target.drawer_front_point_world
        support = target.drawer_front_support_m
        if front_point is None or support is None:
            raise ContactExecutionError(
                "drawer retry staging slot lacks fresh front support span"
            )
        front_point = np.asarray(front_point, dtype=np.float64)
        if (
            front_point.shape != (3,)
            or not np.all(np.isfinite(front_point))
            or not np.isfinite(support)
            or support <= 0.0
        ):
            raise ContactExecutionError(
                "drawer retry staging slot has invalid fresh support span"
            )
        staged = surface + projected
        lateral_from_center = abs(
            float(np.dot(staged - front_point, tangent))
        )
        if lateral_from_center > 0.5 * float(support):
            raise ContactExecutionError(
                "drawer retry staging slot lies outside the fresh support span"
            )
        if (
            not np.all(np.isfinite(staged))
            or abs(float(staged[2] - surface[2])) > 1e-12
            or abs(float(np.dot(staged - surface, outward))) > 1e-12
        ):
            raise ContactExecutionError(
                "drawer retry staging projection is not horizontal and tangential"
            )
        return staged, outward

    def _execute_drawer_retry_contact_corridor(
        self,
        target: ContactTargetEstimate,
        *,
        front_surface: FloatArray,
        planar_outward: FloatArray,
        attempts: list[PhaseAttempt],
    ) -> _DrawerCloseMotionPlateauEvidence | None:
        """Enter one fresh drawer front through a compact typed corridor.

        The already proven low column can be far outside a drawer that moved
        during the first push.  We first move to the ordinary 70-mm precontact
        boundary with the unchanged free-space SDF.  Only the remaining short
        horizontal leg is allowed to ask the drawer contact observer to peel
        one unambiguous target self-surface.  Every endpoint and progress gate
        below is public robot proprioception or fresh RGB-D geometry.
        """

        surface = np.asarray(front_surface, dtype=np.float64)
        outward = np.asarray(planar_outward, dtype=np.float64)
        current = np.asarray(self.robot.current_ee_pose(), dtype=np.float64)
        width = self._public_gripper_width_m()
        labels = {
            " ".join(str(label).lower().replace("_", " ").split())
            for label in target.requested_labels
        }
        if (
            surface.shape != (3,)
            or outward.shape != (3,)
            or current.shape != (4, 4)
            or not np.all(np.isfinite(surface))
            or not np.all(np.isfinite(outward))
            or not np.all(np.isfinite(current))
            or abs(float(outward[2])) > 1e-9
            or abs(float(np.linalg.norm(outward)) - 1.0) > 1e-6
            or not {"drawer", "cabinet"}.issubset(labels)
        ):
            raise ContactExecutionError(
                "drawer retry target-adjacent corridor lacks typed finite geometry"
            )
        if width is None or width > self.config.drawer_close_compact_width_max_m:
            raise ContactExecutionError(
                "drawer retry target-adjacent corridor requires the strict "
                f"compact width gate; width={width}"
            )
        if abs(float(current[2, 3] - surface[2])) > self.config.position_tolerance_m:
            raise ContactExecutionError(
                "drawer retry target-adjacent corridor did not start at the "
                "fresh front height"
            )
        initial_clearance = float(
            np.dot(current[:2, 3] - surface[:2], outward[:2])
        )
        if (
            not np.isfinite(initial_clearance)
            or initial_clearance + self.config.position_tolerance_m
            < self.config.precontact_clearance_m
            or initial_clearance
            > self.config.drawer_retry_safe_column_max_clearance_m
            + self.config.position_tolerance_m
        ):
            raise ContactExecutionError(
                "drawer retry target-adjacent corridor did not start on the "
                "proven outside column"
            )

        entry = current.copy()
        entry[:2, 3] = (
            surface[:2]
            + outward[:2] * self.config.precontact_clearance_m
        )
        # Preserve the reached low height and wrist exactly: fresh RGB-D has
        # already supplied that height through the preceding pure-Z descent.
        entry[2, 3] = current[2, 3]
        if float(np.linalg.norm(entry[:3, 3] - current[:3, 3])) > 1e-8:
            self._record_move(
                entry,
                target,
                Phase.APPROACH,
                GRIPPER_CLOSE,
                False,
                attempts,
            )
        reached_entry = np.asarray(
            self.robot.current_ee_pose(), dtype=np.float64
        )
        if (
            reached_entry.shape != (4, 4)
            or not np.all(np.isfinite(reached_entry))
            or float(np.linalg.norm(reached_entry[:3, 3] - entry[:3, 3]))
            > self.config.position_tolerance_m
        ):
            raise ContactExecutionError(
                "drawer retry did not reach the ordinary-SDF corridor entry"
            )

        corridor_start = reached_entry.copy()
        entry_clearance = float(
            np.dot(corridor_start[:2, 3] - surface[:2], outward[:2])
        )
        if (
            not np.isfinite(entry_clearance)
            or abs(entry_clearance - self.config.precontact_clearance_m)
            > self.config.position_tolerance_m
        ):
            raise ContactExecutionError(
                "drawer retry corridor entry lost its fresh-front clearance"
            )
        contact_pose = corridor_start.copy()
        # The ordinary entry is accepted through the public pose gate, so its
        # reached endpoint can retain a small tangential/outward OSC residual.
        # Replacing that accepted tangential coordinate with the ideal RGB-D
        # surface coordinate would turn the next command into a diagonal by a
        # few millimetres (v99), even though the intended contact corridor is
        # purely normal.  Rebase on the actual public endpoint and consume its
        # measured remaining clearance plus the fixed inset.  This preserves
        # the reached tangent, height, and wrist while keeping the terminal
        # signed clearance exactly at the same -inset plane.
        contact_pose[:2, 3] -= outward[:2] * (
            entry_clearance
            + self.config.drawer_retry_corridor_contact_inset_m
        )
        command = contact_pose[:3, 3] - corridor_start[:3, 3]
        command_length = float(np.linalg.norm(command))
        commanded_inward = float(np.dot(command, -outward))
        if (
            not np.isfinite(command_length)
            or command_length <= self.config.position_tolerance_m
            or command_length > self.config.drawer_retry_corridor_max_length_m
            or abs(float(command[2])) > 1e-9
            or commanded_inward < command_length - 1e-6
        ):
            raise ContactExecutionError(
                "drawer retry target-adjacent corridor is not a bounded "
                "horizontal monotonic inward command"
            )
        try:
            self._record_contact_move(
                contact_pose,
                target,
                GRIPPER_CLOSE,
                attempts,
                max_residual_m=self.config.drawer_retry_corridor_stop_residual_m,
                allow_compact_unknown_contact=True,
                drawer_retry_target_adjacent_corridor=True,
            )
        except _DrawerCloseTypedCorridorPlateau as exc:
            # This is not contact completion.  It is a structured handoff of
            # public motion facts from the rejected typed GRASP.  The caller
            # must immediately obtain a fresh, associated RGB-D estimate and
            # pass the unchanged global close-progress gate.
            return exc.evidence
        endpoint = self._public_motion_endpoint_pose()
        actual = endpoint[:3, 3] - corridor_start[:3, 3]
        inward_progress = float(np.dot(actual, -outward))
        lateral = actual - inward_progress * (-outward)
        terminal_clearance = float(
            np.dot(endpoint[:2, 3] - surface[:2], outward[:2])
        )
        _, rotation_drift = self._pose_errors(endpoint, corridor_start)
        final_width = self._public_gripper_width_m()
        minimum_progress = max(
            0.001,
            entry_clearance
            - self.config.drawer_retry_corridor_terminal_clearance_m,
        )
        accepted = bool(
            np.all(np.isfinite(endpoint))
            and np.isfinite(inward_progress)
            and np.isfinite(terminal_clearance)
            and np.isfinite(rotation_drift)
            and inward_progress >= minimum_progress
            and inward_progress <= command_length + self.config.position_tolerance_m
            and float(np.linalg.norm(lateral))
            <= self.config.drawer_retry_corridor_lateral_tolerance_m
            and terminal_clearance
            <= self.config.drawer_retry_corridor_terminal_clearance_m
            and terminal_clearance
            >= -(
                self.config.drawer_retry_corridor_contact_inset_m
                + self.config.position_tolerance_m
            )
            and rotation_drift <= self.config.orientation_tolerance_rad
            and final_width is not None
            and final_width <= self.config.drawer_close_compact_width_max_m
        )
        attempts.append(
            PhaseAttempt(
                1,
                Phase.VERIFY,
                1,
                accepted,
                "typed monotonic drawer-front corridor public endpoint: "
                f"inward={inward_progress:.4f}/{minimum_progress:.4f} m; "
                f"terminal_clearance={terminal_clearance:.4f}/"
                f"{self.config.drawer_retry_corridor_terminal_clearance_m:.4f} m; "
                f"lateral={float(np.linalg.norm(lateral)):.4f} m; "
                f"width={final_width}",
            )
        )
        if not accepted:
            raise ContactExecutionError(
                "drawer retry typed corridor lacked a valid public endpoint"
            )
        return None

    def _compact_drawer_retry_pusher(
        self, attempts: list[PhaseAttempt]
    ) -> float:
        """Close a retry pusher at high clearance before front descent.

        The first ordinary close pulse can leave a fully released Panda hand
        roughly 40 mm wide.  That wide geometry was observed to catch the
        open drawer/cabinet during a vertical retry descent.  Apply only a
        tiny bounded number of ordinary CLOSE_FINGERS pulses at the already
        verified high transit pose, and require finite, monotonic public
        width at or below the strict compact contact gate before moving down.
        """

        width = self._public_gripper_width_m()
        if width is None:
            raise ContactExecutionError(
                "drawer retry compact pusher lacks public gripper width"
            )
        maximum = self.config.drawer_close_compact_width_max_m
        if width <= maximum:
            attempts.append(
                PhaseAttempt(
                    1,
                    Phase.VERIFY,
                    1,
                    True,
                    "drawer retry compact pusher already within gate: "
                    f"width={width:.4f}/{maximum:.4f} m",
                )
            )
            return width
        previous = width
        for pulse in range(1, self.config.drawer_retry_compact_max_pulses + 1):
            self._set_gripper(GRIPPER_CLOSE, Phase.GRASP, attempts)
            width = self._public_gripper_width_m()
            if width is None:
                raise ContactExecutionError(
                    "drawer retry compact pusher lost public gripper width"
                )
            monotonic = width <= previous + 1e-4
            accepted = monotonic and width <= maximum
            attempts.append(
                PhaseAttempt(
                    pulse,
                    Phase.VERIFY,
                    1,
                    accepted,
                    "drawer retry compact pusher: "
                    f"width={previous:.4f}->{width:.4f} m, "
                    f"required<={maximum:.4f} m",
                )
            )
            if not monotonic:
                raise ContactExecutionError(
                    "drawer retry compact pusher width reversed while closing"
                )
            if accepted:
                return width
            previous = width
        raise ContactExecutionError(
            "drawer retry compact pusher remained wider than the strict "
            f"{maximum:.4f} m gate after "
            f"{self.config.drawer_retry_compact_max_pulses} bounded pulses"
        )

    def _execute_open_fixture_with_load_proof(
        self,
        goal: AtomicGoal,
        target: ContactTargetEstimate,
        base_pose: FloatArray,
        attempts: list[PhaseAttempt],
        *,
        rotary_angle_limit_rad: float | None = None,
        defer_microwave_retreat: bool = False,
        allow_microwave_open_handoff_stall: bool = False,
    ) -> bool:
        """Pinch a sensed handle and reject one-pad tangent contacts.

        A blocked width immediately after closing is insufficient: one finger
        can be wedged against the handle or fixture while the opposite finger
        closes through empty space.  A 12-mm pull makes that false grasp lose
        its width, whereas a two-pad pinch retains it and starts moving the
        articulated fixture.  Failed candidates are released and retried at
        bounded offsets along the jaw-closing axis.
        """

        if defer_microwave_retreat and goal.subject.label != "microwave":
            raise ValueError(
                "deferred open-fixture retreat is only valid for a microwave"
            )
        if (
            allow_microwave_open_handoff_stall
            and not defer_microwave_retreat
        ):
            raise ValueError(
                "a microwave handoff stall requires a deferred contact exit"
            )

        jaw_axis = np.asarray(base_pose[:3, 1], dtype=np.float64)
        retry = self.config.handle_retry_offset_m
        offsets = (0.0, -retry, retry)
        proof_distance = min(
            self.config.handle_load_proof_m,
            0.25 * target.manipulation_distance_m,
        )

        for candidate_index, offset in enumerate(offsets):
            pose = np.asarray(base_pose, dtype=np.float64).copy()
            pose[:3, 3] += jaw_axis * offset
            precontact = pose.copy()
            precontact[:3, 3] += (
                target.outward_world * self.config.precontact_clearance_m
            )
            safe = precontact.copy()
            safe[2, 3] += self.config.safe_height_m

            if candidate_index == 0 and goal.subject.label == "microwave":
                # First translate above the sensor-bound handle in the
                # currently reachable wrist frame, then rotate in place at
                # clearance.  Subsequent candidates start from a retreat that
                # already preserves this reachable handle orientation.
                staged_safe = safe.copy()
                staged_safe[:3, :3] = np.asarray(
                    self.robot.current_ee_pose(), dtype=np.float64
                )[:3, :3]
                self._record_move(
                    staged_safe,
                    target,
                    Phase.APPROACH,
                    GRIPPER_OPEN,
                    False,
                    attempts,
                )

            self._record_move(
                safe, target, Phase.APPROACH, GRIPPER_OPEN, False, attempts
            )
            self._record_move(
                precontact, target, Phase.APPROACH, GRIPPER_OPEN, False, attempts
            )
            self._record_contact_move(
                pose,
                target,
                GRIPPER_OPEN,
                attempts,
                max_residual_m=0.055,
            )
            self._set_gripper(GRIPPER_CLOSE, Phase.GRASP, attempts)
            initially_blocked = self.robot.grasp_confirmed(GraspMode.PINCH)
            initial_width = self._public_gripper_width_m()

            retained_after_load = False
            loaded_width: float | None = None
            settled_width: float | None = None
            width_load_proven = initial_width is None
            if initially_blocked:
                loaded = np.asarray(
                    self.robot.current_ee_pose(), dtype=np.float64
                ).copy()
                loaded[:3, 3] += (
                    target.manipulation_axis_world * proof_distance
                )
                self._record_move(
                    loaded,
                    target,
                    Phase.TRANSFER,
                    GRIPPER_CLOSE,
                    True,
                    attempts,
                )
                retained_after_load = self.robot.grasp_confirmed(
                    GraspMode.PINCH
                )
                loaded_width = self._public_gripper_width_m()
                # The approach close is deliberately short, and the fingers
                # continue closing while OSC executes the proof pull.  Judge
                # stability only after one additional stationary close—not
                # against the still-wide initial preshape.
                self._set_gripper(GRIPPER_CLOSE, Phase.GRASP, attempts)
                retained_after_load = bool(
                    retained_after_load
                    and self.robot.grasp_confirmed(GraspMode.PINCH)
                )
                settled_width = self._public_gripper_width_m()
                if loaded_width is not None and settled_width is not None:
                    width_load_proven = bool(
                        settled_width >= self.config.handle_load_min_width_m
                        and loaded_width - settled_width
                        <= self.config.handle_load_max_settle_loss_m
                    )
                    retained_after_load = bool(
                        retained_after_load and width_load_proven
                    )

            attempts.append(
                PhaseAttempt(
                    candidate_index + 1,
                    Phase.VERIFY,
                    1,
                    bool(initially_blocked and retained_after_load),
                    (
                        "fixture two-pad load proof retained blocked width "
                        f"at jaw offset {offset:+.3f} m"
                        + (
                            ""
                            if loaded_width is None or settled_width is None
                            else (
                                f" ({loaded_width:.4f}->{settled_width:.4f} m settled)"
                            )
                        )
                        if initially_blocked and retained_after_load
                        else (
                            "fixture candidate rejected after load proof "
                            f"at jaw offset {offset:+.3f} m"
                            + (
                                ""
                                if loaded_width is None or settled_width is None
                                else (
                                    f" ({loaded_width:.4f}->{settled_width:.4f} m settled; "
                                    f"width_gate={width_load_proven})"
                                )
                            )
                        )
                    ),
                )
            )
            if retained_after_load:
                if target.rotation_center_world is not None:
                    rotary_start_pose = np.asarray(
                        self.robot.current_ee_pose(), dtype=np.float64
                    ).copy()
                    rotary_complete = self._execute_rotary_fixture_transfer(
                        target,
                        attempts,
                        consumed_linear_distance_m=proof_distance,
                        require_retained_grasp=True,
                        allow_mechanical_stop=False,
                        max_abs_angle_rad=rotary_angle_limit_rad,
                        allow_open_handoff_stall=(
                            allow_microwave_open_handoff_stall
                        ),
                    )
                    rotary_final_pose = self._public_motion_endpoint_pose()
                    if goal.subject.label == "microwave":
                        predicted_edge, achieved_angle = (
                            self._co_rotated_microwave_edge_anchor(
                                target,
                                rotary_start_pose,
                                rotary_final_pose,
                            )
                        )
                        self._microwave_open_handoff = _MicrowaveOpenHandoff(
                            rotary_final_pose.copy(),
                            predicted_edge.copy(),
                            achieved_angle,
                            bool(rotary_complete),
                        )
                else:
                    rotary_complete = True
                    remaining = target.manipulation_distance_m - proof_distance
                    manipulated = np.asarray(
                        self.robot.current_ee_pose(), dtype=np.float64
                    ).copy()
                    manipulated[:3, 3] += (
                        target.manipulation_axis_world * remaining
                    )
                    self._record_move(
                        manipulated,
                        target,
                        Phase.TRANSFER,
                        GRIPPER_CLOSE,
                        True,
                        attempts,
                    )
                    if not self.robot.grasp_confirmed(GraspMode.PINCH):
                        raise ContactExecutionError(
                            "fixture grasp was lost during the full sensor-directed pull"
                        )
                self._set_gripper(GRIPPER_OPEN, Phase.RELEASE, attempts)
                if defer_microwave_retreat:
                    return rotary_complete
                if goal.subject.label == "drawer":
                    self._drawer_release_proven = self._drawer_is_released()
                if goal.subject.label == "drawer":
                    self._execute_drawer_open_retreat(target, attempts)
                else:
                    retreat = np.asarray(
                        self.robot.current_ee_pose(), dtype=np.float64
                    ).copy()
                    retreat[:3, 3] += (
                        target.outward_world * self.config.precontact_clearance_m
                    )
                    retreat[2, 3] += self.config.safe_height_m
                    self._record_move(
                        retreat,
                        target,
                        Phase.RETREAT,
                        GRIPPER_OPEN,
                        False,
                        attempts,
                    )
                return rotary_complete

            self._set_gripper(GRIPPER_OPEN, Phase.RELEASE, attempts)
            if goal.subject.label == "drawer":
                self._drawer_release_proven = self._drawer_is_released()
            if goal.subject.label == "drawer":
                self._execute_drawer_open_retreat(target, attempts)
            else:
                retreat = np.asarray(
                    self.robot.current_ee_pose(), dtype=np.float64
                ).copy()
                retreat[:3, 3] += (
                    target.outward_world * self.config.precontact_clearance_m
                )
                retreat[2, 3] += self.config.safe_height_m
                self._record_move(
                    retreat,
                    target,
                    Phase.RETREAT,
                    GRIPPER_OPEN,
                    False,
                    attempts,
                )

        raise ContactExecutionError(
            "all sensor-bounded handle candidates failed the two-pad load proof"
        )

    def _execute_open_microwave_backside_push(
        self,
        goal: AtomicGoal,
        initial: ContactTargetEstimate,
        attempts: list[PhaseAttempt],
    ) -> None:
        """Open a microwave with one proven pinch and a compact back-side push.

        The first short chord supplies the signed motion and moving-edge
        identity.  All later geometry is reconstructed from fresh local RGB-D
        under the initially frozen hinge/radius gates.  The compact gripper is
        a pusher: it deliberately has no blocked-width or grasp-confirmation
        success path.
        """

        center = initial.rotation_center_world
        axis = initial.rotation_axis_world
        angle = initial.rotation_angle_rad
        if center is None or axis is None or angle is None:
            raise ContactExecutionError(
                "open microwave back-side continuation requires hinge geometry"
            )
        frozen_radius_vector = self._microwave_radial(
            initial.point_world,
            center,
            axis,
        )
        frozen_radius_m = float(np.linalg.norm(frozen_radius_vector))
        if not 0.10 <= frozen_radius_m <= 0.60:
            raise ContactExecutionError(
                "open microwave frozen RGB-D hinge radius is implausible"
            )
        direction_sign = float(np.sign(angle))
        if direction_sign == 0.0:
            raise ContactExecutionError(
                "open microwave frozen hinge direction is degenerate"
            )

        base_pose = self._nearest_symmetric_jaw_pose(
            self._contact_pose(initial)
        )
        first_chord_complete = self._execute_open_fixture_with_load_proof(
            goal,
            initial,
            base_pose,
            attempts,
            rotary_angle_limit_rad=self.config.microwave_regrasp_arc_rad,
            defer_microwave_retreat=True,
            allow_microwave_open_handoff_stall=True,
        )
        handoff = self._microwave_open_handoff
        if handoff is None:
            raise ContactExecutionError(
                "open microwave pinch did not publish a sensor-space handoff"
            )
        aligned_handoff = direction_sign * handoff.achieved_angle_rad
        if (
            not first_chord_complete
            and aligned_handoff
            < self.config.microwave_open_push_handoff_min_rad
        ):
            raise ContactExecutionError(
                "open microwave pinch released before a sufficient hinge chord "
                f"({aligned_handoff:.4f} rad)"
            )

        if first_chord_complete:
            final_edge, remaining, _ = self._reassociate_open_microwave_edge(
                goal,
                initial,
                initial,
                handoff.predicted_edge_world,
                frozen_radius_m,
                direction_sign,
                attempts,
                attempt_index=1,
            )
            if remaining > self.config.microwave_open_state_tolerance_rad:
                raise ContactExecutionError(
                    "completed pinch chord was not confirmed by fresh local "
                    f"RGB-D ({remaining:.4f} rad remaining)"
                )
            self._execute_open_microwave_release_exit(
                final_edge,
                direction_sign,
                attempts,
            )
            return

        release_pose = np.asarray(
            handoff.release_pose, dtype=np.float64
        ).copy()
        if release_pose.shape != (4, 4) or not np.all(np.isfinite(release_pose)):
            raise ContactExecutionError(
                "open microwave release pose must be a finite public transform"
            )
        radial = self._microwave_radial(
            handoff.predicted_edge_world,
            center,
            axis,
        )
        radial /= float(np.linalg.norm(radial))
        opening_tangent = self._microwave_opening_tangent(
            handoff.predicted_edge_world,
            center,
            axis,
            direction_sign,
        )
        lift_axis = np.asarray(axis, dtype=np.float64).copy()
        if lift_axis[2] < 0.0:
            lift_axis *= -1.0

        # W1: leave the released contact along the frozen free-edge ray.  It
        # is intentionally typed as a contact retreat; the hard topology
        # checks below prevent a bounded controller residual from pretending
        # that the edge was cleared.
        clear_edge = release_pose.copy()
        clear_edge[:3, 3] += (
            radial * self.config.microwave_open_push_edge_clearance_m
            + lift_axis * self.config.microwave_open_push_lift_m
        )
        self._record_move(
            clear_edge,
            initial,
            Phase.RETREAT,
            GRIPPER_OPEN,
            True,
            attempts,
            allow_compact_unknown_contact=True,
        )
        clear_endpoint = self._public_motion_endpoint_pose()
        clear_delta = clear_endpoint[:3, 3] - release_pose[:3, 3]
        route_tolerance = self.config.position_tolerance_m
        if (
            float(np.dot(clear_delta, radial))
            < self.config.microwave_open_push_edge_clearance_m - route_tolerance
            or float(np.dot(clear_delta, lift_axis))
            < self.config.microwave_open_push_lift_m - route_tolerance
        ):
            raise ContactExecutionError(
                "open microwave retreat did not clear the sensor-derived free edge"
            )

        # W2: only after radial/lift clearance, translate behind the opening
        # tangent.  Preserve the already reachable wrist orientation.
        backside_safe = release_pose.copy()
        backside_safe[:3, 3] += (
            radial * self.config.microwave_open_push_edge_clearance_m
            - opening_tangent
            * self.config.microwave_open_push_backside_clearance_m
            + lift_axis * self.config.microwave_open_push_lift_m
        )
        self._record_move(
            backside_safe,
            initial,
            Phase.APPROACH,
            GRIPPER_OPEN,
            False,
            attempts,
        )
        routed_pose = self._public_motion_endpoint_pose()
        routed_delta = routed_pose[:3, 3] - release_pose[:3, 3]
        if (
            float(np.dot(routed_delta, radial))
            < self.config.microwave_open_push_edge_clearance_m - route_tolerance
            or -float(np.dot(routed_delta, opening_tangent))
            < self.config.microwave_open_push_backside_clearance_m - route_tolerance
            or float(np.dot(routed_delta, lift_axis))
            < self.config.microwave_open_push_lift_m - route_tolerance
        ):
            raise ContactExecutionError(
                "open microwave back-side route failed its frozen topology gates"
            )

        # The first release must be physically visible before compacting.  A
        # later upper-width gate proves that the closed tool remains a pusher
        # rather than silently becoming another edge pinch.
        self._require_microwave_width(
            minimum_m=self.config.microwave_release_min_width_m,
            detail="first microwave pinch release",
        )
        self._set_gripper(GRIPPER_CLOSE, Phase.GRASP, attempts)
        self._servo_microwave_pusher_width(attempts)

        current_edge, remaining, _ = self._reassociate_open_microwave_edge(
            goal,
            initial,
            initial,
            handoff.predicted_edge_world,
            frozen_radius_m,
            direction_sign,
            attempts,
            attempt_index=1,
        )
        previous_remaining = remaining

        if remaining > self.config.microwave_open_state_tolerance_rad:
            tangent = self._microwave_opening_tangent(
                current_edge.point_world,
                center,
                axis,
                direction_sign,
            )
            edge_radial = self._microwave_radial(
                current_edge.point_world,
                center,
                axis,
            )
            edge_radial /= float(np.linalg.norm(edge_radial))
            contact_position = (
                current_edge.point_world
                - edge_radial * self.config.microwave_open_push_contact_inset_m
            )
            precontact_position = (
                contact_position
                - tangent * self.config.microwave_open_push_precontact_m
            )
            precontact = routed_pose.copy()
            precontact[:3, 3] = precontact_position
            self._record_move(
                precontact,
                current_edge,
                Phase.APPROACH,
                GRIPPER_CLOSE,
                False,
                attempts,
            )
            self._require_microwave_width(
                maximum_m=self.config.microwave_open_push_compact_width_m,
                detail="back-side microwave precontact",
            )
            contact = precontact.copy()
            contact[:3, 3] = contact_position
            self._record_contact_move(
                contact,
                current_edge,
                GRIPPER_CLOSE,
                attempts,
                max_residual_m=self.config.microwave_open_push_segment_m,
                allow_compact_unknown_contact=True,
            )
            self._require_microwave_width(
                maximum_m=self.config.microwave_open_push_compact_width_m,
                detail="back-side microwave contact",
            )

        completed = remaining <= self.config.microwave_open_state_tolerance_rad
        for segment_index in range(
            1, self.config.microwave_open_push_max_segments + 1
        ):
            if completed:
                break
            segment_start = self._public_motion_endpoint_pose()
            push_goal, commanded_angle = self._microwave_open_push_chord(
                segment_start,
                current_edge,
                center,
                axis,
                direction_sign,
                remaining,
            )
            self._record_move(
                push_goal,
                current_edge,
                Phase.TRANSFER,
                GRIPPER_CLOSE,
                True,
                attempts,
                allow_compact_unknown_contact=True,
                mechanical_stop_residual_m=(
                    self.config.microwave_open_push_handoff_residual_m
                ),
                mechanical_stop_plateau_intervals=(
                    self.config.microwave_arc_plateau_intervals
                ),
                mechanical_stop_plateau_span_m=(
                    self.config.microwave_arc_plateau_span_m
                ),
                bounded_contact_handoff=True,
                fresh_visual_handoff_after_chunk=True,
            )
            self._require_microwave_width(
                maximum_m=self.config.microwave_open_push_compact_width_m,
                detail=f"back-side microwave push segment {segment_index}",
            )
            segment_end = self._public_motion_endpoint_pose()
            ee_delta_angle = self._signed_microwave_angle(
                segment_start[:3, 3],
                segment_end[:3, 3],
                center,
                axis,
            )
            aligned_ee_delta = direction_sign * ee_delta_angle
            if (
                aligned_ee_delta
                < -self.config.microwave_open_push_reverse_tolerance_rad
            ):
                raise ContactExecutionError(
                    "back-side microwave TCP reversed the frozen opening direction"
                )
            predicted_edge = center + Rotation.from_rotvec(
                axis * ee_delta_angle
            ).apply(current_edge.point_world - center)
            refreshed, remaining, edge_delta = (
                self._reassociate_open_microwave_edge(
                    goal,
                    initial,
                    current_edge,
                    predicted_edge,
                    frozen_radius_m,
                    direction_sign,
                    attempts,
                    attempt_index=segment_index + 1,
                )
            )
            if (
                remaining
                > previous_remaining
                + self.config.microwave_open_push_reverse_tolerance_rad
            ):
                raise ContactExecutionError(
                    "fresh local microwave edge increased the frozen remaining arc"
                )
            completed = (
                remaining <= self.config.microwave_open_state_tolerance_rad
            )
            if (
                not completed
                and edge_delta
                < self.config.microwave_open_push_min_segment_progress_rad
            ):
                raise ContactExecutionError(
                    "back-side microwave push lacked fresh local RGB-D hinge progress "
                    f"({edge_delta:.4f} rad)"
                )
            attempts.append(
                PhaseAttempt(
                    segment_index,
                    Phase.VERIFY,
                    1,
                    True,
                    "compact back-side pusher retained the local edge: "
                    f"fresh progress={edge_delta:.4f} rad, "
                    f"remaining={remaining:.4f} rad, "
                    f"commanded_chord={commanded_angle:.4f} rad",
                )
            )
            current_edge = refreshed
            previous_remaining = remaining

        if not completed:
            raise ContactExecutionError(
                "microwave remained visibly short of the open state after "
                f"{self.config.microwave_open_push_max_segments} bounded "
                "back-side push segments"
            )
        attempts.append(
            PhaseAttempt(
                1,
                Phase.VERIFY,
                1,
                True,
                "fresh local RGB-D microwave edge reached the strict open "
                f"gate ({remaining:.4f} <= "
                f"{self.config.microwave_open_state_tolerance_rad:.4f} rad)",
            )
        )
        self._execute_open_microwave_release_exit(
            current_edge,
            direction_sign,
            attempts,
        )

    def _microwave_open_push_chord(
        self,
        segment_start: FloatArray,
        current_edge: ContactTargetEstimate,
        center: FloatArray,
        axis: FloatArray,
        direction_sign: float,
        remaining_rad: float,
    ) -> tuple[FloatArray, float]:
        """Build one bounded pusher chord from fresh hinge-circle geometry.

        The local RGB-D edge supplies the current radius and remaining arc;
        public TCP proprioception supplies the actual start ray.  Rotating
        that ray about the frozen hinge avoids the outward radial error of a
        straight tangent command.  A small radial correction keeps the
        compact pusher seated, while independent metric and angular caps keep
        the command bounded for every accepted microwave radius.
        """

        start = np.asarray(segment_start, dtype=np.float64)
        hinge = np.asarray(center, dtype=np.float64)
        rotation_axis = np.asarray(axis, dtype=np.float64)
        if (
            start.shape != (4, 4)
            or hinge.shape != (3,)
            or rotation_axis.shape != (3,)
            or not np.all(np.isfinite(start))
            or not np.all(np.isfinite(hinge))
            or not np.all(np.isfinite(rotation_axis))
            or not np.isfinite(direction_sign)
            or abs(abs(float(direction_sign)) - 1.0) > 1e-9
            or not np.isfinite(remaining_rad)
            or remaining_rad <= 0.0
        ):
            raise ContactExecutionError(
                "microwave fresh-vision chord geometry is invalid"
            )
        axis_norm = float(np.linalg.norm(rotation_axis))
        if axis_norm < 1e-8:
            raise ContactExecutionError(
                "microwave fresh-vision chord axis is degenerate"
            )
        rotation_axis /= axis_norm
        edge_radial = self._microwave_radial(
            current_edge.point_world, hinge, rotation_axis
        )
        edge_radius_m = float(np.linalg.norm(edge_radial))
        tcp_delta = start[:3, 3] - hinge
        tcp_axial_m = float(np.dot(tcp_delta, rotation_axis))
        tcp_radial = tcp_delta - rotation_axis * tcp_axial_m
        tcp_radius_m = float(np.linalg.norm(tcp_radial))
        if (
            not 0.10 <= edge_radius_m <= 0.60
            or not 0.09 <= tcp_radius_m <= 0.61
        ):
            raise ContactExecutionError(
                "microwave fresh-vision chord radius is implausible"
            )
        edge_radial /= edge_radius_m
        tcp_radial /= tcp_radius_m
        if float(np.dot(edge_radial, tcp_radial)) < np.cos(0.40):
            raise ContactExecutionError(
                "microwave pusher TCP is not on the fresh edge ray"
            )
        desired_tcp_radius_m = max(
            0.10,
            edge_radius_m - self.config.microwave_open_push_contact_inset_m,
        )
        radial_correction_m = float(
            np.clip(
                desired_tcp_radius_m - tcp_radius_m,
                -self.config.microwave_open_push_contact_inset_m,
                self.config.microwave_open_push_contact_inset_m,
            )
        )
        target_radius_m = tcp_radius_m + radial_correction_m
        chord_ratio = min(
            1.0,
            self.config.microwave_open_push_chord_max_m
            / (2.0 * target_radius_m),
        )
        metric_angle_cap = 2.0 * float(np.arcsin(chord_ratio))
        commanded_angle = min(
            float(remaining_rad),
            self.config.microwave_open_push_chord_max_angle_rad,
            metric_angle_cap,
        )
        if not np.isfinite(commanded_angle) or commanded_angle <= 0.0:
            raise ContactExecutionError(
                "microwave fresh-vision chord has no positive opening arc"
            )
        goal = start.copy()
        goal_radial = Rotation.from_rotvec(
            rotation_axis * float(direction_sign) * commanded_angle
        ).apply(tcp_radial)
        goal[:3, 3] = (
            hinge
            + rotation_axis * tcp_axial_m
            + goal_radial * target_radius_m
        )
        displacement_m = float(
            np.linalg.norm(goal[:3, 3] - start[:3, 3])
        )
        combined_bound_m = float(
            np.hypot(
                self.config.microwave_open_push_chord_max_m,
                self.config.microwave_open_push_contact_inset_m,
            )
        )
        if (
            not np.all(np.isfinite(goal))
            or displacement_m > combined_bound_m + 1e-9
        ):
            raise ContactExecutionError(
                "microwave fresh-vision chord exceeds its Cartesian bound"
            )
        return goal, commanded_angle

    def _reassociate_open_microwave_edge(
        self,
        goal: AtomicGoal,
        frozen: ContactTargetEstimate,
        previous: ContactTargetEstimate,
        predicted_anchor_world: FloatArray,
        frozen_radius_m: float,
        direction_sign: float,
        attempts: list[PhaseAttempt],
        *,
        attempt_index: int,
    ) -> tuple[ContactTargetEstimate, float, float]:
        """Require one fresh local edge on the frozen hinge circle."""

        continuation = getattr(
            self.provider, "estimate_microwave_continuation", None
        )
        if not callable(continuation):
            raise ContactExecutionError(
                "contact-target provider lacks microwave local reassociation"
            )
        center = frozen.rotation_center_world
        axis = frozen.rotation_axis_world
        if center is None or axis is None:
            raise ContactExecutionError(
                "open microwave local continuation lacks a frozen hinge circle"
            )
        predicted = np.asarray(predicted_anchor_world, dtype=np.float64)
        if predicted.shape != (3,) or not np.all(np.isfinite(predicted)):
            raise ContactExecutionError(
                "open microwave predicted edge anchor is invalid"
            )
        refreshed = continuation(
            goal,
            predicted_anchor_world=predicted,
            frozen_hinge_world=center,
            frozen_rotation_axis_world=axis,
            frozen_radius_m=frozen_radius_m,
            anchor_radius_m=self.config.microwave_reassociation_anchor_radius_m,
            radius_tolerance_m=self.config.microwave_verify_radius_tolerance_m,
        )
        if (
            refreshed.rotation_center_world is None
            or refreshed.rotation_axis_world is None
            or refreshed.rotation_angle_rad is None
        ):
            raise ContactExecutionError(
                "fresh local open microwave edge lacked hinge geometry"
            )
        same_hinge = bool(
            np.linalg.norm(refreshed.rotation_center_world - center) <= 1e-6
            and float(np.dot(refreshed.rotation_axis_world, axis))
            >= 1.0 - 1e-6
        )
        anchor_error_m = float(
            np.linalg.norm(refreshed.point_world - predicted)
        )
        refreshed_radial = self._microwave_radial(
            refreshed.point_world,
            center,
            axis,
        )
        radius_error_m = abs(
            float(np.linalg.norm(refreshed_radial)) - frozen_radius_m
        )
        edge_delta = direction_sign * self._signed_microwave_angle(
            previous.point_world,
            refreshed.point_world,
            center,
            axis,
        )
        remaining = abs(float(refreshed.rotation_angle_rad))
        sign_consistent = bool(
            np.sign(refreshed.rotation_angle_rad) == direction_sign
        )
        associated = bool(
            same_hinge
            and sign_consistent
            and anchor_error_m
            <= self.config.microwave_reassociation_anchor_radius_m
            and radius_error_m
            <= self.config.microwave_verify_radius_tolerance_m
            and edge_delta
            >= -self.config.microwave_open_push_reverse_tolerance_rad
        )
        attempts.append(
            PhaseAttempt(
                attempt_index,
                Phase.VERIFY,
                1,
                associated,
                "fresh local RGB-D open-microwave edge: "
                f"anchor error={anchor_error_m:.4f} m, "
                f"frozen-radius error={radius_error_m:.4f} m, "
                f"signed progress={edge_delta:.4f} rad, "
                f"remaining={remaining:.4f} rad, same_hinge={same_hinge}, "
                f"direction_consistent={sign_consistent}",
            )
        )
        if not associated:
            raise ContactExecutionError(
                "fresh local open microwave edge failed frozen "
                "identity/radius/direction gates"
            )
        return refreshed, remaining, edge_delta

    def _execute_open_microwave_release_exit(
        self,
        final_edge: ContactTargetEstimate,
        direction_sign: float,
        attempts: list[PhaseAttempt],
    ) -> None:
        """Open the jaws and prove a typed exit from the final door edge."""

        center = final_edge.rotation_center_world
        axis = final_edge.rotation_axis_world
        if center is None or axis is None:
            raise ContactExecutionError(
                "final microwave release lacks frozen hinge geometry"
            )
        start = self._public_motion_endpoint_pose()
        tangent = self._microwave_opening_tangent(
            final_edge.point_world,
            center,
            axis,
            direction_sign,
        )
        radial = self._microwave_radial(
            final_edge.point_world,
            center,
            axis,
        )
        radial /= float(np.linalg.norm(radial))
        lift_axis = np.asarray(axis, dtype=np.float64).copy()
        if lift_axis[2] < 0.0:
            lift_axis *= -1.0
        self._set_gripper(GRIPPER_OPEN, Phase.RELEASE, attempts)
        exit_pose = start.copy()
        exit_delta = (
            radial * self.config.microwave_open_push_edge_clearance_m
            - tangent * self.config.microwave_contact_exit_m
            + lift_axis * self.config.microwave_contact_exit_lift_m
        )
        exit_pose[:3, 3] += exit_delta
        self._record_move(
            exit_pose,
            final_edge,
            Phase.RETREAT,
            GRIPPER_OPEN,
            True,
            attempts,
            allow_compact_unknown_contact=True,
        )
        endpoint = self._public_motion_endpoint_pose()
        exit_progress_m = float(
            np.dot(
                endpoint[:3, 3] - start[:3, 3],
                exit_delta / float(np.linalg.norm(exit_delta)),
            )
        )
        width = self._public_gripper_width_m()
        released = bool(
            width is not None
            and width >= self.config.microwave_release_min_width_m
        )
        exited = bool(
            exit_progress_m >= self.config.microwave_contact_exit_progress_m
        )
        attempts.append(
            PhaseAttempt(
                1,
                Phase.RETREAT,
                1,
                released and exited,
                "open microwave final pusher release/exit: "
                f"width={width}, progress={exit_progress_m:.4f} m",
            )
        )
        if not released or not exited:
            raise ContactExecutionError(
                "open microwave final release/exit failed its public "
                "width/displacement gates"
            )

    def _require_microwave_width(
        self,
        *,
        detail: str,
        minimum_m: float | None = None,
        maximum_m: float | None = None,
    ) -> float:
        """Apply an explicit public-width gate to a release or compact tool."""

        if (minimum_m is None) == (maximum_m is None):
            raise ValueError("microwave width gate requires exactly one bound")
        width = self._public_gripper_width_m()
        if width is None:
            raise ContactExecutionError(
                f"{detail} lacks public gripper-width proprioception"
            )
        accepted = bool(
            width >= minimum_m
            if minimum_m is not None
            else width <= maximum_m
        )
        if not accepted:
            relation = "at least" if minimum_m is not None else "at most"
            bound = minimum_m if minimum_m is not None else maximum_m
            raise ContactExecutionError(
                f"{detail} requires width {relation} {float(bound):.4f} m; "
                f"measured {width:.4f} m"
            )
        return width

    def _servo_microwave_pusher_width(
        self, attempts: list[PhaseAttempt]
    ) -> float:
        """Close a released microwave pusher to the strict public-width gate.

        The ordinary gripper command is deliberately a short pulse, so a
        released 60-mm opening may still measure roughly 40 mm afterwards.
        Continue only through the narrowly typed robot capability that holds
        the current TCP pose and checks public wrist wrench, pose drift,
        monotonic finger progress, and the episode step budget on every tick.
        The final width is read independently; adapter success can never round
        or otherwise widen the 14-mm requirement.
        """

        maximum_width = self.config.microwave_open_push_compact_width_m
        width = self._public_gripper_width_m()
        if width is None:
            raise ContactExecutionError(
                "back-side microwave pusher preshape lacks public "
                "gripper-width proprioception"
            )
        if width <= maximum_width:
            return width
        servo = getattr(self.robot, "servo_gripper_width_at_pose", None)
        if not callable(servo):
            raise ContactExecutionError(
                "back-side microwave pusher preshape lacks the bounded "
                "public-proprioception close servo"
            )
        feedback: ControllerFeedback = servo(
            maximum_width_m=maximum_width,
            max_steps=self.config.microwave_compact_servo_max_steps,
            maximum_position_drift_m=(
                self.config.microwave_compact_servo_max_position_drift_m
            ),
            maximum_rotation_drift_rad=(
                self.config.microwave_compact_servo_max_rotation_drift_rad
            ),
            maximum_force_delta_n=(
                self.config.microwave_compact_servo_max_force_delta_n
            ),
            maximum_force_norm_n=(
                self.config.microwave_compact_servo_max_force_norm_n
            ),
            maximum_torque_delta_nm=(
                self.config.microwave_compact_servo_max_torque_delta_nm
            ),
            maximum_torque_norm_nm=(
                self.config.microwave_compact_servo_max_torque_norm_nm
            ),
            minimum_width_progress_m=(
                self.config.microwave_compact_servo_min_width_progress_m
            ),
            maximum_stall_steps=(
                self.config.microwave_compact_servo_max_stall_steps
            ),
        )
        attempts.append(
            PhaseAttempt(
                1,
                Phase.GRASP,
                1,
                feedback.accepted,
                feedback.detail or "bounded microwave width servo completed",
            )
        )
        if not feedback.accepted:
            raise ContactExecutionError(
                feedback.detail or "bounded microwave width servo rejected"
            )
        return self._require_microwave_width(
            maximum_m=maximum_width,
            detail="back-side microwave pusher preshape",
        )

    @staticmethod
    def _microwave_radial(
        point_world: FloatArray,
        center_world: FloatArray,
        axis_world: FloatArray,
    ) -> FloatArray:
        point = np.asarray(point_world, dtype=np.float64)
        center = np.asarray(center_world, dtype=np.float64)
        axis = np.asarray(axis_world, dtype=np.float64)
        if (
            point.shape != (3,)
            or center.shape != (3,)
            or axis.shape != (3,)
            or not np.all(np.isfinite(point))
            or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(axis))
        ):
            raise ContactExecutionError(
                "microwave hinge geometry must contain finite xyz vectors"
            )
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-8:
            raise ContactExecutionError("microwave hinge axis is degenerate")
        axis = axis / axis_norm
        radial = point - center
        radial -= axis * float(np.dot(radial, axis))
        if float(np.linalg.norm(radial)) < 1e-8:
            raise ContactExecutionError("microwave hinge radial is degenerate")
        return radial

    @classmethod
    def _signed_microwave_angle(
        cls,
        start_world: FloatArray,
        end_world: FloatArray,
        center_world: FloatArray,
        axis_world: FloatArray,
    ) -> float:
        axis = np.asarray(axis_world, dtype=np.float64)
        axis /= float(np.linalg.norm(axis))
        start = cls._microwave_radial(start_world, center_world, axis)
        end = cls._microwave_radial(end_world, center_world, axis)
        return float(
            np.arctan2(
                np.dot(axis, np.cross(start, end)),
                np.dot(start, end),
            )
        )

    @classmethod
    def _microwave_opening_tangent(
        cls,
        point_world: FloatArray,
        center_world: FloatArray,
        axis_world: FloatArray,
        direction_sign: float,
    ) -> FloatArray:
        if direction_sign not in (-1.0, 1.0):
            raise ContactExecutionError(
                "microwave opening direction must have a frozen sign"
            )
        axis = np.asarray(axis_world, dtype=np.float64)
        axis /= float(np.linalg.norm(axis))
        radial = cls._microwave_radial(point_world, center_world, axis)
        tangent = np.cross(axis, radial) * direction_sign
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm < 1e-8:
            raise ContactExecutionError(
                "microwave edge cannot define an opening tangent"
            )
        return tangent / tangent_norm

    def _execute_drawer_open_retreat(
        self,
        target: ContactTargetEstimate,
        attempts: list[PhaseAttempt],
    ) -> None:
        """Exit a drawer lip with measured typed progress.

        A bounded OSC residual is not evidence that the wrist actually left
        the lip.  Every typed waypoint therefore has to show public EE
        displacement along the observed outward ray (and, for the fallback
        arc, upward displacement).  The fallback uses two-or-more small
        diagonal steps with the wrist orientation frozen at the first sample;
        once the typed exit is proven, the remaining motion is ordinary SDF
        retreat.
        """

        initial = np.asarray(self.robot.current_ee_pose(), dtype=np.float64).copy()
        if initial.shape != (4, 4) or not np.all(np.isfinite(initial)):
            raise ContactExecutionError("public EE pose must be a finite 4x4 matrix")
        outward = np.asarray(target.outward_world, dtype=np.float64)
        norm = float(np.linalg.norm(outward))
        if outward.shape != (3,) or not np.all(np.isfinite(outward)) or norm <= 1e-8:
            raise ContactExecutionError("drawer outward ray must be finite and nonzero")
        outward = outward / norm
        fixed_rotation = initial[:3, :3].copy()

        def typed_step(
            displacement: np.ndarray,
            *,
            min_outward_m: float | None = None,
            min_lift_m: float | None = None,
            min_path_m: float | None = None,
        ) -> tuple[float, float]:
            """Run one contact step and require measured public EE motion."""

            start = np.asarray(self.robot.current_ee_pose(), dtype=np.float64).copy()
            goal = start.copy()
            goal[:3, :3] = fixed_rotation
            goal[:3, 3] += np.asarray(displacement, dtype=np.float64)
            self._record_move(
                goal,
                target,
                Phase.RETREAT,
                GRIPPER_OPEN,
                True,
                attempts,
                allow_compact_unknown_contact=True,
            )
            end = np.asarray(self.robot.current_ee_pose(), dtype=np.float64)
            delta = end[:3, 3] - start[:3, 3]
            measured_outward = float(np.dot(delta, outward))
            measured_lift = float(delta[2])
            if (
                min_outward_m is not None
                and measured_outward + 1e-6 < min_outward_m
            ) or (
                min_lift_m is not None
                and measured_lift + 1e-6 < min_lift_m
            ) or (
                min_path_m is not None
                and float(np.linalg.norm(delta)) + 1e-6 < min_path_m
            ):
                raise ContactExecutionError(
                    "typed drawer exit made insufficient measured progress: "
                    f"outward={measured_outward:.4f}/"
                    f"{min_outward_m if min_outward_m is not None else 0.0:.4f} m, "
                    f"up={measured_lift:.4f}/"
                    f"{min_lift_m if min_lift_m is not None else 0.0:.4f} m, "
                    f"path={np.linalg.norm(delta):.4f}/"
                    f"{min_path_m if min_path_m is not None else 0.0:.4f} m"
                )
            return measured_outward, measured_lift

        typed_outward = 0.0
        typed_lift = 0.0
        no_progress_evidence = 0
        safe_escape_executed = False
        pure_exit = min(
            self.config.drawer_contact_exit_progress_m,
            self.config.precontact_clearance_m,
        )
        try:
            measured_outward, measured_lift = typed_step(
                outward * pure_exit,
                min_outward_m=pure_exit,
            )
            typed_outward += max(0.0, measured_outward)
            typed_lift += max(0.0, measured_lift)
        except ContactExecutionError:
            # The residual gate may have accepted a zero-motion compliant
            # stop.  Do not advance the plan on that result; use the short
            # diagonal exit below and require actual motion for each step.
            # Count this failed proprioceptive observation as well: the
            # last-goal defer path requires repeated measured no-progress,
            # including the initial pure-outward probe.
            no_progress_evidence += 1
            pass

        step_outward = self.config.drawer_contact_exit_microstep_outward_m
        step_lift = self.config.drawer_contact_exit_microstep_lift_m
        required_outward = min(
            self.config.drawer_contact_exit_m,
            self.config.precontact_clearance_m,
        )
        required_lift = min(self.config.safe_height_m, 0.040)

        def sdf_safe_candidate(start: np.ndarray, displacement: np.ndarray) -> bool:
            """Check a short candidate against the current public RGB-D SDF."""

            try:
                scene = self.observer.observe(target.requested_labels)
                fractions = np.linspace(0.2, 1.0, 5)[:, None]
                samples = start[:3, 3][None, :] + fractions * displacement[None, :]
                distances = np.asarray(scene.obstacle_sdf.distance(samples), dtype=np.float64)
            except (LookupError, ValueError, TypeError, AttributeError):
                return False
            if distances.ndim != 1 or not np.all(np.isfinite(distances)):
                return False
            required = self.config.free_clearance_m + self.config.free_tool_radius_m
            return bool(float(np.min(distances)) >= required - 0.002)

        def escape_step(
            displacement: np.ndarray,
            rotation: np.ndarray,
            *,
            min_path_m: float = 0.005,
            min_rotation_rad: float = 0.0,
        ) -> tuple[float, float]:
            """Run a safe typed escape and require measured motion."""

            start = np.asarray(self.robot.current_ee_pose(), dtype=np.float64).copy()
            displacement = np.asarray(displacement, dtype=np.float64)
            if not sdf_safe_candidate(start, displacement):
                raise ContactExecutionError("RGB-D SDF rejected drawer escape candidate")
            goal = start.copy()
            goal[:3, :3] = np.asarray(rotation, dtype=np.float64)
            goal[:3, 3] += displacement
            self._record_move(
                goal,
                target,
                Phase.RETREAT,
                GRIPPER_OPEN,
                True,
                attempts,
                allow_compact_unknown_contact=True,
            )
            end = np.asarray(self.robot.current_ee_pose(), dtype=np.float64)
            delta = end[:3, 3] - start[:3, 3]
            path = float(np.linalg.norm(delta))
            rotation_delta = float(
                Rotation.from_matrix(start[:3, :3].T @ end[:3, :3]).magnitude()
            )
            command_norm = float(np.linalg.norm(displacement))
            directional_progress = (
                float(np.dot(delta, displacement / command_norm))
                if command_norm > 1e-8
                else 0.0
            )
            if path + 1e-6 < min_path_m and rotation_delta + 1e-6 < min_rotation_rad:
                raise ContactExecutionError(
                    "drawer escape candidate made insufficient measured motion: "
                    f"path={path:.4f}/{min_path_m:.4f} m, "
                    f"rotation={rotation_delta:.4f}/{min_rotation_rad:.4f} rad"
                )
            if (
                command_norm > 1e-8
                and directional_progress + 1e-6 < min_path_m
            ):
                raise ContactExecutionError(
                    "drawer escape candidate made insufficient directional progress: "
                    f"progress={directional_progress:.4f}/{min_path_m:.4f} m"
                )
            return float(np.dot(delta, outward)), float(delta[2])

        def reactive_escape_search() -> tuple[float, float] | None:
            """Try bounded sensor-derived translations and wrist unjams."""

            nonlocal fixed_rotation, safe_escape_executed

            feature = np.asarray(target.feature_axis_world, dtype=np.float64)
            feature_norm = float(np.linalg.norm(feature))
            if feature.shape != (3,) or not np.all(np.isfinite(feature)) or feature_norm <= 1e-8:
                feature = np.zeros(3, dtype=np.float64)
            else:
                feature = feature / feature_norm
            z_axis = np.array((0.0, 0.0, 1.0), dtype=np.float64)
            displacement_candidates: list[np.ndarray] = []
            for sign in (1.0, -1.0):
                displacement_candidates.append(
                    outward * step_outward + sign * z_axis * min(step_lift, 0.015)
                )
            if feature_norm > 1e-8:
                for sign in (1.0, -1.0):
                    displacement_candidates.append(
                        outward * step_outward + sign * feature * step_outward
                    )
            for sign in (1.0, -1.0):
                displacement_candidates.append(sign * z_axis * min(step_lift, 0.015))
            if feature_norm > 1e-8:
                for sign in (1.0, -1.0):
                    displacement_candidates.append(sign * feature * step_outward)

            rotation_candidates: list[np.ndarray] = []
            for axis in (fixed_rotation[:, 2], fixed_rotation[:, 0]):
                for sign in (1.0, -1.0):
                    rotation_candidates.append(
                        fixed_rotation
                        @ Rotation.from_rotvec(sign * 0.10 * axis).as_matrix()
                    )

            for candidate_index, displacement in enumerate(displacement_candidates):
                try:
                    measured_outward, measured_lift = escape_step(
                        displacement,
                        fixed_rotation,
                    )
                except ContactExecutionError:
                    continue
                attempts.append(
                    PhaseAttempt(
                        1,
                        Phase.RETREAT,
                        1,
                        True,
                        "RGB-D/SDF escape translation "
                        f"{candidate_index} measured outward="
                        f"{measured_outward:.4f} m, up={measured_lift:.4f} m",
                    )
                )
                safe_escape_executed = True
                return measured_outward, measured_lift
            for candidate_index, rotation in enumerate(rotation_candidates):
                try:
                    measured_outward, measured_lift = escape_step(
                        np.zeros(3, dtype=np.float64),
                        rotation,
                        min_path_m=0.0,
                        min_rotation_rad=0.03,
                    )
                except ContactExecutionError:
                    continue
                attempts.append(
                    PhaseAttempt(
                        1,
                        Phase.RETREAT,
                        1,
                        True,
                        "RGB-D/SDF wrist unjam "
                        f"{candidate_index} measured rotation",
                    )
                )
                fixed_rotation = np.asarray(rotation, dtype=np.float64).copy()
                return measured_outward, measured_lift
            return None

        # Two nominal 12.5-mm/20-mm steps satisfy the 25-mm/40-mm typed
        # exit.  Allow extra attempts to absorb compliant partial motion;
        # each step still needs nonzero measured proprioceptive displacement,
        # never a zero-motion residual.  Component deficits are accumulated
        # rather than discarded, since the lip can permit upward motion before
        # it permits the commanded outward component.
        lateral_escape_count = 0
        vertical_escape_count = 0
        reactive_search_attempts = 0
        for _ in range(8):
            if typed_outward >= required_outward - 1e-6 and typed_lift >= required_lift - 1e-6:
                break
            try:
                measured_outward, measured_lift = typed_step(
                    outward * step_outward + np.array((0.0, 0.0, step_lift)),
                    min_path_m=0.005,
                )
            except ContactExecutionError:
                no_progress_evidence += 1
                if reactive_search_attempts < 2:
                    reactive_search_attempts += 1
                    escape_progress = reactive_escape_search()
                    if escape_progress is not None:
                        measured_outward, measured_lift = escape_progress
                        typed_outward += max(0.0, measured_outward)
                        typed_lift += max(0.0, measured_lift)
                        continue
                # First try to slip the horizontal handle out of the two-pad
                # gap with a small vertical move.  The wrist orientation is
                # frozen, and either vertical sign is accepted only when
                # public EE motion proves it; no simulator contact state is
                # consulted here.
                vertical_succeeded = False
                if vertical_escape_count < 2:
                    for sign in (1.0, -1.0):
                        try:
                            _, measured_lift = typed_step(
                                np.array((0.0, 0.0, sign * min(step_lift, 0.015))),
                                min_path_m=0.005,
                            )
                        except ContactExecutionError:
                            continue
                        vertical_escape_count += 1
                        typed_lift += max(0.0, measured_lift)
                        vertical_succeeded = True
                        break
                if vertical_succeeded:
                    # Once either side has actually cleared the finger gap,
                    # try a short outward contact exit before returning to the
                    # diagonal accumulation.  A downward escape is valid for
                    # this purpose too; ordinary safe-height lifting remains
                    # below and is still SDF checked.
                    try:
                        measured_outward, measured_lift = typed_step(
                            outward * step_outward,
                            min_path_m=0.005,
                        )
                    except ContactExecutionError:
                        pass
                    else:
                        typed_outward += max(0.0, measured_outward)
                        typed_lift += max(0.0, measured_lift)
                    # Retry the diagonal outward/up step after the handle has
                    # moved out of the finger gap.
                    continue

                # A nearby fixture can still block the arm's outer link even
                # after the drawer lip has released.  Use a bounded,
                # sensor-derived handle-axis displacement to clear that side;
                # try both signs and accept a sign only after public EE motion
                # proves it.  This is never the old large lateral detour.
                feature = np.asarray(target.feature_axis_world, dtype=np.float64)
                feature_norm = float(np.linalg.norm(feature))
                if (
                    feature.shape != (3,)
                    or not np.all(np.isfinite(feature))
                    or feature_norm <= 1e-8
                ):
                    raise
                feature = feature / feature_norm
                lateral_succeeded = False
                if lateral_escape_count < 4:
                    for sign in (1.0, -1.0):
                        try:
                            typed_step(
                                sign * feature * step_outward,
                                min_path_m=0.005,
                            )
                        except ContactExecutionError:
                            continue
                        lateral_escape_count += 1
                        lateral_succeeded = True
                        break
                if not lateral_succeeded:
                    # Keep the search bounded but let the outer loop collect
                    # the required repeated public-proprio no-progress
                    # evidence.  The final-goal defer gate below decides
                    # whether a fresh RGB-D verification may be attempted;
                    # non-final goals still fail closed there.
                    continue
                # The lateral move is escape-only.  Retry the same small
                # outward/up typed step and accumulate only its measured
                # outward/up components.
                continue
            typed_outward += max(0.0, measured_outward)
            typed_lift += max(0.0, measured_lift)
        if typed_outward < required_outward - 1e-6 or typed_lift < required_lift - 1e-6:
            # A hardware/simulator gripper width sample can lag the accepted
            # open command by one motion chunk.  Re-sample the public width at
            # the defer decision, after the bounded escape attempts, so the
            # release gate reflects the actual released jaws rather than the
            # command timing.  Missing/closed width remains fail-closed.
            if not self._drawer_release_proven:
                self._drawer_release_proven = self._drawer_is_released()
            if (
                self._drawer_exit_defer_allowed(
                    final_goal=self._final_goal_context,
                    release_proven=self._drawer_release_proven,
                    safe_escape_executed=safe_escape_executed,
                    no_progress_evidence=no_progress_evidence,
                )
            ):
                attempts.append(
                    PhaseAttempt(
                        1,
                        Phase.RETREAT,
                        1,
                        True,
                        "bounded final drawer retreat deferred to fresh RGB-D "
                        f"after {no_progress_evidence} measured no-progress samples",
                    )
                )
                return
            raise ContactExecutionError(
                "typed drawer exit remained short after measured microsteps: "
                f"outward={typed_outward:.4f}/{required_outward:.4f} m, "
                f"up={typed_lift:.4f}/{required_lift:.4f} m"
            )

        current = np.asarray(self.robot.current_ee_pose(), dtype=np.float64).copy()
        raised = current.copy()
        raised[:3, :3] = fixed_rotation
        remaining_lift = max(0.0, self.config.safe_height_m - typed_lift)
        raised[2, 3] += remaining_lift
        if remaining_lift > 1e-6:
            self._record_move(
                raised,
                target,
                Phase.RETREAT,
                GRIPPER_OPEN,
                False,
                attempts,
            )
        remaining_outward = max(
            0.0,
            self.config.precontact_clearance_m - typed_outward,
        )
        if remaining_outward > 1e-6:
            retreat = np.asarray(self.robot.current_ee_pose(), dtype=np.float64).copy()
            retreat[:3, :3] = fixed_rotation
            retreat[:3, 3] += outward * remaining_outward
            self._record_move(
                retreat,
                target,
                Phase.RETREAT,
                GRIPPER_OPEN,
                False,
                attempts,
            )

    def _execute_rotary_fixture_transfer(
        self,
        target: ContactTargetEstimate,
        attempts: list[PhaseAttempt],
        *,
        consumed_linear_distance_m: float,
        require_retained_grasp: bool,
        allow_mechanical_stop: bool,
        max_abs_angle_rad: float | None = None,
        extra_close_angle_rad: float = 0.0,
        max_segment_angle_rad: float | None = None,
        intermediate_position_tolerance_m: float | None = None,
        allow_open_handoff_stall: bool = False,
    ) -> bool:
        """Follow a sensor-derived vertical hinge in bounded arc segments."""

        if allow_mechanical_stop and not require_retained_grasp:
            raise ValueError(
                "rotary mechanical stop requires a retained sensor-proprioceptive grasp"
            )
        if allow_open_handoff_stall and (
            allow_mechanical_stop or not require_retained_grasp
        ):
            raise ValueError(
                "open microwave handoff stalls require a retained non-jamb pinch"
            )
        if (
            not np.isfinite(extra_close_angle_rad)
            or extra_close_angle_rad < 0.0
            or (
                extra_close_angle_rad > 0.0
                and (not allow_mechanical_stop or not require_retained_grasp)
            )
        ):
            raise ValueError(
                "rotary close extension requires a finite retained typed-stop path"
            )
        if max_segment_angle_rad is not None and (
            not np.isfinite(max_segment_angle_rad)
            or max_segment_angle_rad <= 0.0
        ):
            raise ValueError("rotary segment angle must be finite and positive")
        if intermediate_position_tolerance_m is not None and (
            not np.isfinite(intermediate_position_tolerance_m)
            or intermediate_position_tolerance_m <= 0.0
        ):
            raise ValueError(
                "rotary intermediate position tolerance must be finite and positive"
            )

        center = target.rotation_center_world
        axis = target.rotation_axis_world
        angle_value = target.rotation_angle_rad
        if center is None or axis is None or angle_value is None:
            raise ContactExecutionError("rotary fixture geometry is incomplete")
        start = np.asarray(self.robot.current_ee_pose(), dtype=np.float64).copy()
        radial = start[:3, 3] - center
        radial -= axis * float(np.dot(radial, axis))
        radius = float(np.linalg.norm(radial))
        if not 0.10 <= radius <= 0.60:
            raise ContactExecutionError(
                f"sensor-derived microwave hinge radius {radius:.3f} m is implausible"
            )
        angle = float(angle_value)
        if extra_close_angle_rad > 0.0:
            angle += np.sign(angle) * float(extra_close_angle_rad)
            attempts.append(
                PhaseAttempt(
                    1,
                    Phase.TRANSFER,
                    1,
                    True,
                    "retained local microwave close adds bounded terminal arc "
                    f"{float(extra_close_angle_rad):.3f} rad",
                )
            )
        consumed_angle = min(
            abs(angle) * 0.25,
            consumed_linear_distance_m / radius,
        )
        remaining = angle - np.sign(angle) * consumed_angle
        full_remaining = remaining
        if max_abs_angle_rad is not None:
            if not np.isfinite(max_abs_angle_rad) or max_abs_angle_rad <= 0.0:
                raise ValueError("rotary angle limit must be finite and positive")
            remaining = np.sign(remaining) * min(
                abs(remaining), float(max_abs_angle_rad)
            )
        segment_angle = (
            self.config.microwave_arc_segment_rad
            if max_segment_angle_rad is None
            else float(max_segment_angle_rad)
        )
        segments = max(
            1,
            int(np.ceil(abs(remaining) / segment_angle)),
        )
        mechanical_stop_encountered = False
        for index in range(1, segments + 1):
            partial_angle = remaining * index / segments
            rotation = Rotation.from_rotvec(axis * partial_angle).as_matrix()
            waypoint = start.copy()
            # The sensed microwave handle/door edge is a vertical capsule and
            # therefore can roll between the retained pads.  Co-rotate only a
            # bounded amount: a fixed wrist can eventually shed the handle,
            # while following the entire door angle needlessly accumulates
            # Panda yaw and can hit a joint limit before reaching the jamb.
            wrist_angle = np.sign(partial_angle) * min(
                abs(partial_angle),
                self.config.microwave_wrist_corotation_limit_rad,
            )
            waypoint[:3, :3] = (
                Rotation.from_rotvec(axis * wrist_angle).as_matrix()
                @ start[:3, :3]
            )
            waypoint[:3, 3] = center + rotation @ (start[:3, 3] - center)
            mechanical_stop = self._record_move(
                waypoint,
                target,
                Phase.TRANSFER,
                GRIPPER_CLOSE,
                True,
                attempts,
                mechanical_stop_residual_m=(
                    self.config.linear_close_stop_residual_m
                    if allow_mechanical_stop
                    else (
                        self.config.microwave_open_push_handoff_residual_m
                        if allow_open_handoff_stall
                        else None
                    )
                ),
                waypoint_position_tolerance_m=(
                    intermediate_position_tolerance_m
                    if (
                        intermediate_position_tolerance_m is not None
                        and index < segments
                    )
                    else self.config.microwave_arc_position_tolerance_m
                ),
                mechanical_stop_plateau_intervals=(
                    self.config.microwave_arc_plateau_intervals
                    if allow_mechanical_stop or allow_open_handoff_stall
                    else None
                ),
                mechanical_stop_plateau_span_m=(
                    self.config.microwave_arc_plateau_span_m
                    if allow_mechanical_stop or allow_open_handoff_stall
                    else None
                ),
                bounded_contact_handoff=allow_open_handoff_stall,
            )
            if (
                require_retained_grasp
                and not self.robot.grasp_confirmed(GraspMode.PINCH)
            ):
                raise ContactExecutionError(
                    "microwave door-edge grasp was lost during hinge-arc transfer"
                )
            if allow_open_handoff_stall and mechanical_stop:
                endpoint = self._public_motion_endpoint_pose()
                achieved_angle = self._signed_microwave_angle(
                    start[:3, 3],
                    endpoint[:3, 3],
                    center,
                    axis,
                )
                aligned_angle = float(np.sign(angle) * achieved_angle)
                if (
                    aligned_angle
                    < self.config.microwave_open_push_handoff_min_rad
                ):
                    raise ContactExecutionError(
                        "retained microwave tangent stalled before the safe "
                        "back-side handoff chord "
                        f"({aligned_angle:.4f} rad)"
                    )
                attempts.append(
                    PhaseAttempt(
                        index,
                        Phase.TRANSFER,
                        1,
                        True,
                        "retained microwave tangent plateau after a sufficient "
                        f"{aligned_angle:.4f}-rad sensor-space chord; releasing "
                        "for local RGB-D back-side continuation",
                    )
                )
                return False
            if allow_mechanical_stop and mechanical_stop:
                # A closing door may meet its physical jamb before the
                # RGB-D OBB's approximate hinge arc reaches a later sampled
                # waypoint.  Every bounded segment is therefore eligible for
                # the typed mechanical-stop gate, but only while the public
                # gripper width still proves the same edge is retained.  Stop
                # issuing deeper arc commands once that gate was needed; the
                # mandatory fresh RGB-D verification remains the only success
                # signal.
                attempts.append(
                    PhaseAttempt(
                        index,
                        Phase.TRANSFER,
                        1,
                        True,
                        "bounded rotary mechanical stop; defer to fresh RGB-D",
                    )
                )
                mechanical_stop_encountered = True
                break
        return bool(
            not mechanical_stop_encountered
            and abs(full_remaining) <= abs(remaining) + 1e-8
        )

    def _public_gripper_width_m(self) -> float | None:
        """Read optional sanitized gripper proprioception from the adapter."""

        getter = getattr(self.robot, "current_gripper_width_m", None)
        if not callable(getter):
            return None
        raw_width = getter()
        if raw_width is None:
            return None
        width = float(raw_width)
        if not np.isfinite(width) or not 0.0 <= width <= 0.12:
            raise ContactExecutionError(
                "public gripper width must be finite and physically bounded"
            )
        return width

    def _execute_turn(
        self,
        goal: AtomicGoal,
        target: ContactTargetEstimate,
        attempts: list[PhaseAttempt],
    ) -> None:
        # LIBERO stove knobs are operated from above.  Retain the calibrated
        # reset/current wrist frame instead of rotating into a horizontal
        # drawer-handle frame, which is frequently unreachable near the stove.
        pose = self._vertical_contact_pose(target)
        precontact = pose.copy()
        precontact[2, 3] += self.config.precontact_clearance_m
        self._record_move(precontact, target, Phase.APPROACH, GRIPPER_OPEN, False, attempts)
        self._record_contact_move(
            pose,
            target,
            GRIPPER_OPEN,
            attempts,
            max_residual_m=0.030,
            allow_compact_unknown_contact=True,
        )
        self._set_gripper(GRIPPER_CLOSE, Phase.GRASP, attempts)
        if not self.robot.grasp_confirmed(GraspMode.PINCH):
            raise ContactExecutionError("knob grasp was not confirmed by proprioception")
        # Contact tracking can stop a few degrees from the requested pose.
        # Measure the turn from the actual retained grasp, the same baseline
        # used by the controller's signed mechanical-stop measurement.
        turn_start_rotation = np.asarray(
            self.robot.current_ee_pose(), dtype=np.float64
        )[:3, :3].copy()
        sign = 1.0 if goal.kind is AtomicGoalKind.TURN_ON else -1.0
        turned = pose.copy()
        rotation = Rotation.from_rotvec(
            target.manipulation_axis_world * sign * self.config.knob_rotation_rad
        ).as_matrix()
        turned[:3, :3] = rotation @ turn_start_rotation
        minimum_rotation = self.config.knob_mechanical_completion_rad
        configure_turn = getattr(self.robot, "set_turn_contact_enabled", None)
        if callable(configure_turn):
            configure_turn(
                True,
                axis_world=target.manipulation_axis_world * sign,
                minimum_rotation_rad=minimum_rotation,
            )
        try:
            self._record_move(
                turned,
                target,
                Phase.TRANSFER,
                GRIPPER_CLOSE,
                True,
                attempts,
                allow_compact_unknown_contact=True,
                typed_turn=True,
            )
            self._last_turn_progress_rad = self._axis_rotation_progress(
                turn_start_rotation,
                np.asarray(self.robot.current_ee_pose(), dtype=np.float64)[:3, :3],
                target.manipulation_axis_world * sign,
            )
            if self._last_turn_progress_rad < minimum_rotation:
                raise ContactExecutionError(
                    "knob turn lacked the required measured signed rotation "
                    f"({self._last_turn_progress_rad:.3f} < {minimum_rotation:.3f} rad)"
                )
        finally:
            if callable(configure_turn):
                configure_turn(False)
        self._set_gripper(GRIPPER_OPEN, Phase.RELEASE, attempts)
        retreat = turned.copy()
        retreat[2, 3] += self.config.safe_height_m
        self._record_move(retreat, target, Phase.RETREAT, GRIPPER_OPEN, False, attempts)

    def _execute_push(
        self,
        goal: AtomicGoal,
        initial: ContactTargetEstimate,
        attempts: list[PhaseAttempt],
    ) -> None:
        """Drag one RGB-D-bound plate toward its frozen sensor-space goal."""

        frozen = self._required_push_geometry(initial)
        frozen_center, frozen_goal, frozen_radius, frozen_direction = frozen
        initial_axis = np.asarray(
            initial.manipulation_axis_world, dtype=np.float64
        ).copy()
        initial_axis[2] = 0.0
        initial_axis_norm = float(np.linalg.norm(initial_axis))
        goal_bearing = np.asarray(frozen_goal - frozen_center, dtype=np.float64)
        goal_bearing[2] = 0.0
        goal_range = float(np.linalg.norm(goal_bearing))
        if (
            goal_range < 1e-5
            or initial_axis_norm < 0.95
            or float(
                np.dot(
                    initial_axis / max(initial_axis_norm, 1e-12),
                    frozen_direction,
                )
            )
            < self.config.push_direction_min_cosine
            or float(np.dot(goal_bearing / goal_range, frozen_direction))
            < self.config.push_direction_min_cosine
        ):
            raise ContactExecutionError(
                "initial plate contact axis disagrees with frozen RGB-D goal direction"
            )
        target = initial
        last_detail = "no plate push attempt completed"
        active_retry_side: float | None = None
        proven_retry_side: float | None = None
        proven_retry_attempt: int | None = None
        previous_fresh_signed_progress = 0.0
        for attempt_index in range(self.config.push_max_attempts):
            retained_proof_lost = False
            retained_segment_lost = False
            load_proof_passed = False
            target_center, _, _, _ = self._required_push_geometry(target)
            contact_radial = np.asarray(
                target.point_world - target_center, dtype=np.float64
            )
            contact_radial[2] = 0.0
            contact_radial_norm = float(np.linalg.norm(contact_radial))
            if contact_radial_norm < 0.005:
                raise ContactExecutionError(
                    "plate rim contact lacks a sensor-derived perimeter ray"
                )
            contact_radial /= contact_radial_norm
            pose = self._vertical_jaw_aligned_pose(
                target,
                contact_radial,
            )
            precontact = pose.copy()
            precontact[2, 3] += self.config.precontact_clearance_m
            above = precontact.copy()
            above[2, 3] += self.config.safe_height_m
            self._record_move(
                above, target, Phase.APPROACH, GRIPPER_OPEN, False, attempts
            )
            self._record_move(
                precontact, target, Phase.APPROACH, GRIPPER_OPEN, False, attempts
            )
            self._record_contact_move(
                pose,
                target,
                GRIPPER_OPEN,
                attempts,
                max_residual_m=0.055,
            )
            self._set_gripper(GRIPPER_CLOSE, Phase.GRASP, attempts)
            retained, width = self._plate_rim_retention_sample()
            if retained:
                proof_distance = min(
                    self.config.push_load_proof_m,
                    target.manipulation_distance_m,
                )
                if proof_distance < 0.005:
                    retained = False
                proof_start = np.asarray(
                    self.robot.current_ee_pose(), dtype=np.float64
                ).copy()
                measured_proof = 0.0
                proof_cross_drift = 0.0
                proof_rotation_drift = 0.0
                if retained:
                    proof_axis = np.asarray(
                        target.manipulation_axis_world,
                        dtype=np.float64,
                    ).copy()
                    proof_axis_norm = float(np.linalg.norm(proof_axis))
                    if (
                        proof_axis.shape != (3,)
                        or not np.all(np.isfinite(proof_axis))
                        or proof_axis_norm < 0.95
                    ):
                        raise ContactExecutionError(
                            "plate rim load proof lacks a finite unit direction"
                        )
                    proof_axis /= proof_axis_norm
                    proof_goal = proof_start.copy()
                    proof_goal[:3, 3] += proof_axis * proof_distance
                    self._record_move(
                        proof_goal,
                        target,
                        Phase.TRANSFER,
                        GRIPPER_CLOSE,
                        True,
                        attempts,
                        waypoint_position_tolerance_m=0.002,
                        plate_rim_load_proof_handoff=True,
                    )
                    proof_end = np.asarray(
                        self.robot.current_ee_pose(), dtype=np.float64
                    )
                    proof_delta = proof_end[:3, 3] - proof_start[:3, 3]
                    measured_proof = float(
                        np.dot(proof_delta, proof_axis)
                    )
                    proof_cross_drift = float(
                        np.linalg.norm(
                            proof_delta - measured_proof * proof_axis
                        )
                    )
                    _, proof_rotation_drift = self._pose_errors(
                        proof_end,
                        proof_start,
                    )
                    retained_after_proof, width = (
                        self._plate_rim_retention_sample()
                    )
                    proof_motion_valid = bool(
                        np.isfinite(measured_proof)
                        and np.isfinite(proof_cross_drift)
                        and np.isfinite(proof_rotation_drift)
                        and 0.005
                        <= measured_proof
                        <= self.config.push_load_proof_max_progress_m
                        and proof_cross_drift
                        <= self.config.push_load_proof_max_cross_drift_m
                        and proof_rotation_drift
                        <= self.config.push_load_proof_max_rotation_drift_rad
                    )
                    # Only a measured 5--10 mm typed load proof followed by
                    # actual loss of public retained-width/pinch evidence may
                    # unlock the plate-specific contact exit.  A zero/sub-gate
                    # prefix or other bad motion envelope is not upgraded.
                    retained_proof_lost = bool(
                        proof_motion_valid and not retained_after_proof
                    )
                    load_proof_passed = bool(
                        proof_motion_valid and retained_after_proof
                    )
                    retained = bool(
                        retained_after_proof and proof_motion_valid
                    )
                attempts.append(
                    PhaseAttempt(
                        attempt_index + 1,
                        Phase.VERIFY,
                        1,
                        retained,
                        "plate rim micro-drag load proof: "
                        f"retained={retained}; width={width}; "
                        f"signed_progress={measured_proof:.4f} m; "
                        f"cross_drift={proof_cross_drift:.4f} m; "
                        f"rotation_drift={proof_rotation_drift:.4f} rad; "
                        "required_progress=[0.0050,"
                        f"{self.config.push_load_proof_max_progress_m:.4f}] m, "
                        "cross_drift<="
                        f"{self.config.push_load_proof_max_cross_drift_m:.4f} m, "
                        "rotation_drift<="
                        f"{self.config.push_load_proof_max_rotation_drift_rad:.4f} rad, and "
                        f"width>={self.config.push_retained_min_width_m:.4f} m",
                    )
                )

                remaining_command = max(
                    0.0,
                    target.manipulation_distance_m - proof_distance,
                )
                segment_index = 0
                while retained and remaining_command > 1e-8:
                    segment_index += 1
                    if segment_index > self.config.push_max_segments:
                        raise ContactExecutionError(
                            "plate push remaining travel exceeds the bounded "
                            "short-segment budget"
                        )
                    segment_distance = min(
                        self.config.push_segment_m, remaining_command
                    )
                    segment_start = np.asarray(
                        self.robot.current_ee_pose(), dtype=np.float64
                    ).copy()
                    segment_goal = segment_start.copy()
                    segment_goal[:3, 3] += (
                        target.manipulation_axis_world * segment_distance
                    )
                    self._record_move(
                        segment_goal,
                        target,
                        Phase.TRANSFER,
                        GRIPPER_CLOSE,
                        True,
                        attempts,
                        waypoint_position_tolerance_m=0.005,
                    )
                    segment_end = np.asarray(
                        self.robot.current_ee_pose(), dtype=np.float64
                    )
                    measured_segment = float(
                        np.dot(
                            segment_end[:3, 3] - segment_start[:3, 3],
                            target.manipulation_axis_world,
                        )
                    )
                    retained, width = self._plate_rim_retention_sample()
                    minimum_segment_progress = max(
                        0.0, segment_distance - 0.005
                    )
                    retained = bool(
                        retained
                        and measured_segment >= minimum_segment_progress
                    )
                    if not retained:
                        retained_segment_lost = True
                    attempts.append(
                        PhaseAttempt(
                            attempt_index + 1,
                            Phase.VERIFY,
                            segment_index,
                            retained,
                            f"plate retained segment {segment_index}: "
                            f"retained={retained}; width={width}; "
                            f"signed_progress={measured_segment:.4f}/"
                            f"{segment_distance:.4f} m",
                        )
                    )
                    remaining_command -= segment_distance

            self._set_gripper(GRIPPER_OPEN, Phase.RELEASE, attempts)
            typed_exit_succeeded = False
            if retained_proof_lost or retained_segment_lost:
                self._execute_plate_rim_contact_exit(
                    target,
                    attempts,
                    attempt_index=attempt_index + 1,
                )
                typed_exit_succeeded = True
            retreat = np.asarray(
                self.robot.current_ee_pose(), dtype=np.float64
            ).copy()
            retreat[2, 3] += self.config.safe_height_m
            try:
                self._record_move(
                    retreat,
                    target,
                    Phase.RETREAT,
                    GRIPPER_OPEN,
                    False,
                    attempts,
                )
            except OptimisationError as exc:
                reported_raw = self._plate_post_exit_start_clearance_raw(exc)
                if not typed_exit_succeeded or reported_raw is None:
                    raise
                self._execute_plate_post_exit_clearance_restore(
                    target,
                    attempts,
                    reported_raw_sdf_m=reported_raw,
                    attempt_index=attempt_index + 1,
                )
                # The unchanged ordinary 12 mm clearance plus 18 mm tool
                # envelope gets the final word after the narrow bridge.
                self._record_move(
                    retreat,
                    target,
                    Phase.RETREAT,
                    GRIPPER_OPEN,
                    False,
                    attempts,
                )

            fresh = self.provider.estimate(goal)
            if not np.isfinite(fresh.confidence) or fresh.confidence <= 0.0:
                raise ContactExecutionError(
                    "fresh RGB-D plate confidence is non-positive or non-finite"
                )
            fresh_center, fresh_goal, fresh_radius, fresh_direction = (
                self._required_push_geometry(fresh)
            )
            target_anchor_error = float(np.linalg.norm(fresh_goal - frozen_goal))
            radius_error = abs(fresh_radius - frozen_radius)
            radius_tolerance = max(
                self.config.push_radius_tolerance_m,
                self.config.push_radius_relative_tolerance * frozen_radius,
            )
            direction_cosine = float(np.dot(fresh_direction, frozen_direction))
            displacement = fresh_center - frozen_center
            signed_progress = float(np.dot(displacement, frozen_direction))
            fresh_visual_gain = (
                signed_progress - previous_fresh_signed_progress
            )
            lateral_vector = displacement - signed_progress * frozen_direction
            lateral = float(np.linalg.norm(lateral_vector))
            goal_delta = frozen_goal - fresh_center
            signed_remaining = float(np.dot(goal_delta, frozen_direction))
            goal_distance = float(np.linalg.norm(goal_delta))
            identity_ok = bool(
                target_anchor_error <= self.config.push_target_anchor_tolerance_m
                and radius_error <= radius_tolerance
                and direction_cosine >= self.config.push_direction_min_cosine
                and lateral <= self.config.push_lateral_tolerance_m
            )
            goal_ok = bool(
                -self.config.push_overshoot_tolerance_m
                <= signed_remaining
                <= self.config.push_goal_tolerance_m
                and goal_distance <= self.config.push_goal_tolerance_m
            )
            visually_complete = bool(retained and identity_ok and goal_ok)
            last_detail = (
                f"retained={retained}; identity={identity_ok}; "
                f"radius_error={radius_error:.4f}/{radius_tolerance:.4f} m; "
                f"target_anchor_error={target_anchor_error:.4f} m; "
                f"direction_cosine={direction_cosine:.4f}; "
                f"lateral={lateral:.4f} m; "
                f"signed_progress={signed_progress:.4f} m; "
                f"fresh_visual_gain={fresh_visual_gain:.4f} m; "
                f"signed_remaining={signed_remaining:.4f} m; "
                f"goal_distance={goal_distance:.4f} m"
            )
            attempts.append(
                PhaseAttempt(
                    attempt_index + 1,
                    Phase.VERIFY,
                    1,
                    visually_complete,
                    "fresh RGB-D plate terminal gate after rim attempt "
                    f"{attempt_index + 1}: {last_detail}",
                )
            )
            if not identity_ok:
                raise ContactExecutionError(
                    "fresh RGB-D plate identity gates failed; " + last_detail
                )
            if (
                fresh_visual_gain
                < -self._PLATE_MAX_FRESH_REGRESSION_M - 1e-12
            ):
                raise ContactExecutionError(
                    "fresh RGB-D plate regressed by more than the 5 mm "
                    "retry-history bound; "
                    + last_detail
                )
            if visually_complete:
                return
            if (
                active_retry_side is not None
                and load_proof_passed
                and fresh_visual_gain >= self.config.visual_progress_m
            ):
                # This episode-local evidence comes only from the public EE
                # load proof and the immediately following fresh RGB-D plate
                # displacement.  It authorises reusing that perimeter side;
                # retry parity alone can never authorise a fourth contact.
                proven_retry_side = active_retry_side
                proven_retry_attempt = attempt_index + 1
            previous_fresh_signed_progress = signed_progress
            if attempt_index + 1 < self.config.push_max_attempts:
                retry_index = attempt_index + 1
                retry_side = None
                if retry_index == 3:
                    if proven_retry_side is None or proven_retry_attempt is None:
                        raise ContactExecutionError(
                            "fourth plate rim attempt lacks a load-proven side "
                            "with significant fresh RGB-D progress"
                        )
                    retry_side = proven_retry_side
                target = self._plate_push_retry_target(
                    fresh,
                    frozen_goal=frozen_goal,
                    retry_index=retry_index,
                    perimeter_side=retry_side,
                )
                active_retry_side = (
                    retry_side
                    if retry_side is not None
                    else (1.0 if retry_index % 2 == 1 else -1.0)
                )
                if retry_index == 3:
                    self._authorise_plate_fourth_retry(
                        target,
                        evidence_attempt=proven_retry_attempt,
                        evidence_side=proven_retry_side,
                        fresh_signed_progress_m=signed_progress,
                        fresh_visual_gain_m=fresh_visual_gain,
                    )

        raise ContactExecutionError(
            "plate push terminal verification failed; " + last_detail
        )

    @staticmethod
    def _required_push_geometry(
        target: ContactTargetEstimate,
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """Return complete typed RGB-D push geometry or fail closed."""

        if (
            target.push_object_center_world is None
            or target.push_target_center_world is None
            or target.push_object_radius_m is None
            or target.push_direction_world is None
        ):
            raise ContactExecutionError(
                "plate push requires complete typed RGB-D object/goal geometry"
            )
        center = np.asarray(
            target.push_object_center_world, dtype=np.float64
        ).copy()
        goal = np.asarray(
            target.push_target_center_world, dtype=np.float64
        ).copy()
        direction = np.asarray(
            target.push_direction_world, dtype=np.float64
        ).copy()
        direction[2] = 0.0
        norm = float(np.linalg.norm(direction))
        if norm < 0.95:
            raise ContactExecutionError(
                "plate push RGB-D direction is not reliably planar"
            )
        direction /= norm
        return center, goal, float(target.push_object_radius_m), direction

    def _plate_rim_retention_sample(self) -> tuple[bool, float | None]:
        """Require both typed pinch state and an absolute public-width gate."""

        width = self._public_gripper_width_m()
        retained = bool(
            width is not None
            and width >= self.config.push_retained_min_width_m
            and self.robot.grasp_confirmed(GraspMode.PINCH)
        )
        return retained, width

    def _plate_post_exit_start_clearance_raw(
        self, failure: BaseException
    ) -> float | None:
        """Recognise only the v101-style positive, start-only free violation."""

        diagnostic = getattr(failure, "clearance_diagnostic", None)
        if (
            not isinstance(failure, OptimisationError)
            or not isinstance(diagnostic, InfeasibleClearanceDiagnostic)
        ):
            return None
        try:
            raw = float(diagnostic.raw_sdf_m)
            minimum = float(diagnostic.minimum_clearance_m)
            clearance = float(diagnostic.clearance_limit_m)
            tool_radius = float(diagnostic.tool_radius_m)
            nearest = np.asarray(
                diagnostic.nearest_point_world_m, dtype=np.float64
            )
            center = np.asarray(
                diagnostic.field_center_world_m, dtype=np.float64
            )
            half_extents = np.asarray(
                diagnostic.field_half_extents_m, dtype=np.float64
            )
        except (TypeError, ValueError):
            return None
        full_envelope = self.config.free_clearance_m + self.config.free_tool_radius_m
        if (
            diagnostic.phase is not Phase.RETREAT
            or diagnostic.reach_ok is not True
            or diagnostic.height_ok is not True
            or diagnostic.clearance_ok is not False
            or not isinstance(diagnostic.field_index, int)
            or isinstance(diagnostic.field_index, bool)
            or diagnostic.field_index < 0
            or not isinstance(diagnostic.source_instance_id, str)
            or not diagnostic.source_instance_id.strip()
            or not isinstance(diagnostic.source_label, str)
            or not diagnostic.source_label.strip()
            or not isinstance(diagnostic.sample_index, int)
            or isinstance(diagnostic.sample_index, bool)
            or not isinstance(diagnostic.sample_count, int)
            or isinstance(diagnostic.sample_count, bool)
            or diagnostic.sample_index != 0
            or diagnostic.sample_count <= 1
            or diagnostic.is_start is not True
            or diagnostic.is_end is not False
            or nearest.shape != (3,)
            or center.shape != (3,)
            or half_extents.shape != (3,)
            or not np.all(np.isfinite(nearest))
            or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(half_extents))
            or np.any(half_extents <= 0.0)
            or not np.all(
                np.isfinite(
                    (raw, minimum, clearance, tool_radius, full_envelope)
                )
            )
            or raw <= 0.0
            or raw >= full_envelope
            or abs(clearance - self.config.free_clearance_m) > 0.00015
            or abs(tool_radius - self.config.free_tool_radius_m) > 0.00020
            or abs((raw - minimum) - tool_radius) > 0.00020
            or minimum >= clearance
        ):
            return None
        return raw

    def _plate_post_exit_clearance_corridor(
        self,
        scene: object,
        start: FloatArray,
    ) -> tuple[FloatArray, FloatArray] | None:
        """Rank short all-obstacle rays restoring the ordinary free envelope."""

        if not isinstance(scene, SceneEstimate):
            return None
        obstacle = scene.obstacle_sdf
        if not isinstance(obstacle, CompositeSDF):
            return None
        origin = np.asarray(start[:3, 3], dtype=np.float64)
        full_envelope = self.config.free_clearance_m + self.config.free_tool_radius_m
        initial = float(obstacle.distance(origin))
        if not np.isfinite(initial) or not 0.0 < initial < full_envelope:
            return None

        gradient = np.zeros(3, dtype=np.float64)
        gradient_step = 0.001
        for axis in range(3):
            offset = np.zeros(3, dtype=np.float64)
            offset[axis] = gradient_step
            positive = float(obstacle.distance(origin + offset))
            negative = float(obstacle.distance(origin - offset))
            if not np.all(np.isfinite((positive, negative))):
                return None
            gradient[axis] = (positive - negative) / (2.0 * gradient_step)
        gradient_norm = float(np.linalg.norm(gradient))
        away = (
            gradient / gradient_norm
            if gradient_norm >= 1e-8
            else np.array((0.0, 0.0, 1.0), dtype=np.float64)
        )
        # Never nominate a downward escape beside a released tabletop object.
        away = away.copy()
        away[2] = max(0.0, float(away[2]))
        away_norm = float(np.linalg.norm(away))
        if away_norm >= 1e-8:
            away /= away_norm
        else:
            away = np.array((0.0, 0.0, 1.0), dtype=np.float64)

        up = np.array((0.0, 0.0, 1.0), dtype=np.float64)
        candidates: list[np.ndarray] = [up, away]
        for up_weight in (0.35, 0.70):
            blended = away + up_weight * up
            norm = float(np.linalg.norm(blended))
            if norm >= 1e-8:
                candidates.append(blended / norm)
        for elevation in (0.0, 0.5):
            planar_scale = float(np.sqrt(1.0 - elevation**2))
            for angle in np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False):
                candidates.append(
                    np.array(
                        (
                            planar_scale * np.cos(angle),
                            planar_scale * np.sin(angle),
                            elevation,
                        ),
                        dtype=np.float64,
                    )
                )

        unique: list[np.ndarray] = []
        for direction in candidates:
            norm = float(np.linalg.norm(direction))
            if norm < 1e-8 or not np.all(np.isfinite(direction)):
                continue
            direction = direction / norm
            if direction[2] < -1e-12:
                continue
            if not any(np.linalg.norm(direction - item) <= 1e-6 for item in unique):
                unique.append(direction)

        samples = int(
            np.ceil(
                self._PLATE_CLEARANCE_RESTORE_MAX_DISTANCE_M
                / self._PLATE_CLEARANCE_RESTORE_SAMPLE_SPACING_M
            )
        )
        distances = np.linspace(
            0.0, self._PLATE_CLEARANCE_RESTORE_MAX_DISTANCE_M, samples + 1
        )
        lower = np.asarray(scene.workspace_min, dtype=np.float64) + 0.005
        upper = np.asarray(scene.workspace_max, dtype=np.float64) - 0.005
        ranked: list[
            tuple[tuple[float, float, float, float], FloatArray, FloatArray]
        ] = []
        for direction in unique:
            points = origin[None, :] + distances[:, None] * direction[None, :]
            if np.any(points < lower) or np.any(points > upper):
                continue
            raw = np.asarray(obstacle.distance(points), dtype=np.float64)
            if raw.shape != (len(points),) or not np.all(np.isfinite(raw)):
                continue
            reached = np.flatnonzero(raw >= full_envelope)
            if len(reached) == 0:
                continue
            stop = int(reached[0])
            if (
                stop < 1
                or distances[stop] < self._PLATE_CLEARANCE_RESTORE_MIN_PROGRESS_M
                or np.any(np.diff(raw[: stop + 1]) <= 1e-8)
            ):
                continue
            poses = np.repeat(start[None, :, :], stop + 1, axis=0)
            poses[:, :3, 3] = points[: stop + 1]
            distance = float(distances[stop])
            score = (
                -distance,
                float(direction[2]),
                float(np.dot(direction, away)),
                float(raw[stop]),
            )
            ranked.append((score, poses, raw[: stop + 1].copy()))
        if not ranked:
            return None
        _, poses, raw = max(ranked, key=lambda item: item[0])
        return poses, raw

    def _execute_plate_post_exit_clearance_restore(
        self,
        target: ContactTargetEstimate,
        attempts: list[PhaseAttempt],
        *,
        reported_raw_sdf_m: float | None = None,
        attempt_index: int = 1,
    ) -> float:
        """Restore 30 mm all-obstacle clearance after a proven rim exit."""

        if bool(getattr(self.robot, "step_budget_exhausted", False)):
            raise ContactExecutionError(
                "episode OSC step budget exhausted before plate clearance restore"
            )
        width_before = self._public_gripper_width_m()
        if (
            width_before is None
            or width_before < self.config.push_contact_exit_release_min_width_m
        ):
            raise ContactExecutionError(
                "plate clearance restore lacks released public width"
            )
        start = np.asarray(self.robot.current_ee_pose(), dtype=np.float64)
        if start.shape != (4, 4) or not np.all(np.isfinite(start)):
            raise ContactExecutionError(
                "plate clearance restore lacks finite public EE pose"
            )
        invalidate = getattr(self.observer, "invalidate_sensor_cache", None)
        if callable(invalidate):
            invalidate()
        scene = self.observer.observe(target.requested_labels)
        if not isinstance(scene, SceneEstimate) or not isinstance(
            scene.obstacle_sdf, CompositeSDF
        ):
            raise ContactExecutionError(
                "plate clearance restore requires a fresh CompositeSDF"
            )
        initial_raw = float(scene.obstacle_sdf.distance(start[:3, 3]))
        full_envelope = self.config.free_clearance_m + self.config.free_tool_radius_m
        nearest = scene.obstacle_sdf.nearest_field_diagnostic(start[:3, 3])
        if not np.isfinite(initial_raw) or initial_raw <= 0.0:
            clearance_state = "penetrating_or_invalid"
        elif initial_raw >= full_envelope:
            clearance_state = "already_clear"
        else:
            clearance_state = "bridge_required"
        trace = getattr(self.robot, "phase_trace", None)
        if isinstance(trace, list):
            trace.append(
                {
                    "phase": "plate_post_exit_clearance_revalidation",
                    "fresh_raw_sdf_m": initial_raw,
                    "required_raw_sdf_m": full_envelope,
                    "clearance_state": clearance_state,
                    "nearest_field_index": nearest.field_index,
                    "nearest_field_id": nearest.source_instance_id,
                    "nearest_field_label": nearest.source_label,
                    "nearest_field_center_world_m": nearest.field_center_world_m,
                    "nearest_field_half_extents_m": nearest.field_half_extents_m,
                    "obstacle_field_count": len(scene.obstacle_sdf.fields),
                    "bridge_policy_actions": 0,
                    "pre_action": True,
                }
            )
        if (
            reported_raw_sdf_m is not None
            and (
                not np.isfinite(reported_raw_sdf_m)
                or not 0.0 < reported_raw_sdf_m < full_envelope
            )
        ):
            raise ContactExecutionError(
                "plate clearance restore requires a valid reported start violation"
            )
        if not np.isfinite(initial_raw) or initial_raw <= 0.0:
            raise ContactExecutionError(
                "plate clearance restore fresh start is penetrating or non-finite"
            )
        if initial_raw >= full_envelope:
            detail = (
                "plate post-contact clearance revalidated without bridge: "
                f"initial_raw={initial_raw:.4f} m; required={full_envelope:.4f} m; "
                f"nearest_field_id={nearest.source_instance_id}; "
                f"nearest_field_label={nearest.source_label}; "
                "bridge_policy_actions=0"
            )
            attempts.append(
                PhaseAttempt(attempt_index, Phase.VERIFY, 1, True, detail)
            )
            return 0.0
        corridor = self._plate_post_exit_clearance_corridor(scene, start)
        if corridor is None:
            raise ContactExecutionError(
                "plate clearance restore has no strictly improving bounded corridor"
            )
        poses, commanded_raw = corridor
        command = poses[-1, :3, 3] - start[:3, 3]
        command_length = float(np.linalg.norm(command))
        if (
            not np.isfinite(command_length)
            or command_length < self._PLATE_CLEARANCE_RESTORE_MIN_PROGRESS_M
            or command_length > self._PLATE_CLEARANCE_RESTORE_MAX_DISTANCE_M + 1e-12
        ):
            raise ContactExecutionError(
                "plate clearance restore command exceeds its compact bound"
            )
        command_axis = command / command_length
        steps_before = getattr(self.robot, "steps_executed", None)
        feedback = self.robot.execute_waypoints(
            poses, Phase.RETREAT, GRIPPER_OPEN
        )
        steps_after = getattr(self.robot, "steps_executed", None)
        if bool(getattr(self.robot, "step_budget_exhausted", False)):
            raise ContactExecutionError(
                "episode OSC step budget exhausted during plate clearance restore"
            )
        if (
            not isinstance(steps_before, int)
            or isinstance(steps_before, bool)
            or not isinstance(steps_after, int)
            or isinstance(steps_after, bool)
            or steps_after <= steps_before
        ):
            raise ContactExecutionError(
                "plate clearance restore executed no policy action"
            )
        endpoint = self._public_motion_endpoint_pose()
        if endpoint.shape != (4, 4) or not np.all(np.isfinite(endpoint)):
            raise ContactExecutionError(
                "plate clearance restore lost finite public proprioception"
            )
        actual = endpoint[:3, 3] - start[:3, 3]
        actual_distance = float(np.linalg.norm(actual))
        signed_progress = float(np.dot(actual, command_axis))
        cross_drift = float(
            np.linalg.norm(actual - signed_progress * command_axis)
        )
        actual_z = float(actual[2])
        commanded_z = float(command[2])
        _, rotation_drift = self._pose_errors(endpoint, start)
        width_after = self._public_gripper_width_m()

        if callable(invalidate):
            invalidate()
        endpoint_scene = self.observer.observe(target.requested_labels)
        if not isinstance(endpoint_scene, SceneEstimate) or not isinstance(
            endpoint_scene.obstacle_sdf, CompositeSDF
        ):
            raise ContactExecutionError(
                "plate clearance restore lost its fresh CompositeSDF endpoint"
            )
        actual_samples = np.linspace(
            start[:3, 3], endpoint[:3, 3], len(poses), dtype=np.float64
        )
        actual_raw = np.asarray(
            endpoint_scene.obstacle_sdf.distance(actual_samples),
            dtype=np.float64,
        )
        accepted = bool(
            feedback.accepted
            and actual_raw.shape == (len(actual_samples),)
            and np.all(np.isfinite(actual_raw))
            and np.all(np.diff(actual_raw) > 1e-8)
            and actual_raw[-1] >= full_envelope
            and np.isfinite(actual_distance)
            and actual_distance
            <= self._PLATE_CLEARANCE_RESTORE_MAX_DISTANCE_M + 1e-12
            and signed_progress
            >= max(
                self._PLATE_CLEARANCE_RESTORE_MIN_PROGRESS_M,
                command_length - self.config.position_tolerance_m,
            )
            and signed_progress
            <= command_length + self.config.position_tolerance_m
            and cross_drift <= self._PLATE_CLEARANCE_RESTORE_MAX_CROSS_DRIFT_M
            and actual_z >= -self._PLATE_CLEARANCE_RESTORE_MAX_DOWNWARD_M
            and abs(actual_z - commanded_z)
            <= self._PLATE_CLEARANCE_RESTORE_MAX_Z_ERROR_M
            and rotation_drift <= self._PLATE_CLEARANCE_RESTORE_MAX_ROTATION_RAD
            and width_after is not None
            and width_after >= self.config.push_contact_exit_release_min_width_m
        )
        detail = (
            "plate post-contact clearance restore: "
            f"initial_raw={initial_raw:.4f} m; "
            f"commanded_final_raw={commanded_raw[-1]:.4f} m; "
            f"final_raw={actual_raw[-1] if len(actual_raw) else float('nan'):.4f} m; "
            f"required={full_envelope:.4f} m; "
            f"actual_distance={actual_distance:.4f} m; "
            f"signed_progress={signed_progress:.4f}/{command_length:.4f} m; "
            f"cross_drift={cross_drift:.4f} m; z={actual_z:.4f}/"
            f"{commanded_z:.4f} m; rotation={rotation_drift:.4f} rad; "
            f"width={width_after}; policy_actions={steps_after - steps_before}; "
            f"obstacle_fields={len(endpoint_scene.obstacle_sdf.fields)}"
        )
        attempts.append(
            PhaseAttempt(
                attempt_index,
                Phase.VERIFY,
                1,
                accepted,
                detail,
            )
        )
        if not accepted:
            raise ContactExecutionError(
                "plate clearance restore failed public/SDF endpoint gates"
            )
        return signed_progress

    def _complete_plate_rim_contact_exit_release(
        self,
        attempts: list[PhaseAttempt],
        *,
        attempt_index: int,
    ) -> float:
        """Prove an in-place OPEN release before leaving lost rim contact.

        ``set_gripper`` feedback only acknowledges the command; it does not
        prove that the fingers have physically opened.  Extra pulses are
        available only on the already-authorized lost-contact exit path and
        must each execute policy actions while public width increases.
        """

        minimum = self.config.push_contact_exit_release_min_width_m
        width = self._public_gripper_width_m()
        if width is None:
            raise ContactExecutionError(
                "plate-rim contact exit lacks finite released public evidence"
            )
        if width >= minimum:
            return width

        previous = width
        for pulse in range(
            1, self.config.push_contact_exit_release_max_pulses + 1
        ):
            steps_before = getattr(self.robot, "steps_executed", None)
            self._set_gripper(GRIPPER_OPEN, Phase.RELEASE, attempts)
            steps_after = getattr(self.robot, "steps_executed", None)
            if bool(getattr(self.robot, "step_budget_exhausted", False)):
                raise ContactExecutionError(
                    "episode OSC step budget exhausted during plate-rim release"
                )
            if (
                not isinstance(steps_before, int)
                or isinstance(steps_before, bool)
                or not isinstance(steps_after, int)
                or isinstance(steps_after, bool)
                or steps_after <= steps_before
            ):
                raise ContactExecutionError(
                    "plate-rim release completion executed no policy action"
                )

            width = self._public_gripper_width_m()
            if width is None:
                raise ContactExecutionError(
                    "plate-rim release completion lost public gripper width"
                )
            reversed_width = width + 1e-6 < previous
            reached = width >= minimum
            stalled = not reached and width <= previous + 1e-5
            accepted = bool(not reversed_width and not stalled and reached)
            attempts.append(
                PhaseAttempt(
                    attempt_index,
                    Phase.VERIFY,
                    pulse,
                    accepted,
                    "plate-rim in-place release completion: "
                    f"width={previous:.5f}->{width:.5f} m; "
                    f"required>={minimum:.5f} m; "
                    f"policy_actions={steps_after - steps_before}",
                )
            )
            if reversed_width:
                raise ContactExecutionError(
                    "plate-rim release width reversed while opening"
                )
            if stalled:
                raise ContactExecutionError(
                    "plate-rim release width stalled below the strict gate"
                )
            if reached:
                return width
            previous = width

        raise ContactExecutionError(
            "plate-rim contact exit remained narrow after the bounded "
            "OPEN-only release budget"
        )

    def _execute_plate_rim_contact_exit(
        self,
        target: ContactTargetEstimate,
        attempts: list[PhaseAttempt],
        *,
        attempt_index: int = 1,
    ) -> float:
        """Leave a lost plate-rim contact before ordinary SDF retreat.

        This path is called only after a previously retained load-proof or
        short drag loses its public blocked-width/contact proof.  The fingers
        are already open.
        One short radial/upward move uses zero target-contact inflation, then
        measured public EE progress, drift, and released width must all pass
        before the ordinary retreat planner is allowed to run.
        """

        center, _, _, _ = self._required_push_geometry(target)
        width_before = self._complete_plate_rim_contact_exit_release(
            attempts,
            attempt_index=attempt_index,
        )
        start = np.asarray(self.robot.current_ee_pose(), dtype=np.float64)
        radial = np.asarray(target.point_world - center, dtype=np.float64)
        radial[2] = 0.0
        radial_norm = float(np.linalg.norm(radial))
        if (
            start.shape != (4, 4)
            or not np.all(np.isfinite(start))
            or radial.shape != (3,)
            or not np.all(np.isfinite(radial))
            or radial_norm < 0.005
            or width_before is None
            or width_before < self.config.push_contact_exit_release_min_width_m
        ):
            raise ContactExecutionError(
                "plate-rim contact exit lacks finite released public evidence"
            )
        radial /= radial_norm
        goal_pose = start.copy()
        goal_pose[:3, 3] += radial * self.config.push_contact_exit_radial_m
        goal_pose[2, 3] += self.config.push_contact_exit_lift_m
        command = goal_pose[:3, 3] - start[:3, 3]
        command_length = float(np.linalg.norm(command))
        if (
            not np.isfinite(command_length)
            or command_length <= self.config.push_contact_exit_min_progress_m
            or command_length > 0.030
        ):
            raise ContactExecutionError(
                "plate-rim contact exit command exceeds its compact bound"
            )

        self._record_move(
            goal_pose,
            target,
            Phase.RETREAT,
            GRIPPER_OPEN,
            True,
            attempts,
            plate_rim_contact_exit=True,
        )
        endpoint = self._public_motion_endpoint_pose()
        actual = endpoint[:3, 3] - start[:3, 3]
        command_axis = command / command_length
        signed_progress = float(np.dot(actual, command_axis))
        radial_progress = float(np.dot(actual, radial))
        lift_progress = float(actual[2])
        cross_drift = float(
            np.linalg.norm(actual - signed_progress * command_axis)
        )
        _, rotation_drift = self._pose_errors(endpoint, start)
        width_after = self._public_gripper_width_m()
        accepted = bool(
            np.all(np.isfinite(endpoint))
            and np.isfinite(signed_progress)
            and np.isfinite(radial_progress)
            and np.isfinite(lift_progress)
            and np.isfinite(cross_drift)
            and np.isfinite(rotation_drift)
            and signed_progress >= self.config.push_contact_exit_min_progress_m
            and signed_progress <= command_length + self.config.position_tolerance_m
            and radial_progress > 0.0
            and lift_progress > 0.0
            and cross_drift <= self.config.push_contact_exit_max_cross_drift_m
            and rotation_drift
            <= self.config.push_contact_exit_max_rotation_drift_rad
            and width_after is not None
            and width_after >= self.config.push_contact_exit_release_min_width_m
        )
        attempts.append(
            PhaseAttempt(
                attempt_index,
                Phase.VERIFY,
                1,
                accepted,
                "typed plate-rim contact exit public endpoint: "
                f"signed_progress={signed_progress:.4f}/"
                f"{self.config.push_contact_exit_min_progress_m:.4f} m; "
                f"radial={radial_progress:.4f} m; lift={lift_progress:.4f} m; "
                f"cross_drift={cross_drift:.4f} m; "
                f"rotation_drift={rotation_drift:.4f} rad; width={width_after}",
            )
        )
        if not accepted:
            raise ContactExecutionError(
                "plate-rim typed contact exit lacked measured public progress"
            )
        return signed_progress

    def _plate_push_retry_target(
        self,
        fresh: ContactTargetEstimate,
        *,
        frozen_goal: np.ndarray,
        retry_index: int = 1,
        perimeter_side: float | None = None,
    ) -> ContactTargetEstimate:
        """Build a bounded fresh-goal retry at a different perimeter point."""

        if (
            not isinstance(retry_index, int)
            or isinstance(retry_index, bool)
            or retry_index < 1
        ):
            raise ValueError("plate retry index must be a positive integer")

        center, fresh_goal, radius, _ = self._required_push_geometry(fresh)
        if (
            float(np.linalg.norm(fresh_goal - frozen_goal))
            > self.config.push_target_anchor_tolerance_m
        ):
            raise ContactExecutionError(
                "fresh plate observation changed the frozen RGB-D push goal"
            )
        remaining_vector = np.asarray(frozen_goal - center, dtype=np.float64)
        remaining_vector[2] = 0.0
        remaining = float(np.linalg.norm(remaining_vector))
        if remaining < 1e-5:
            raise ContactExecutionError(
                "plate reached the frozen goal without retained rim evidence"
            )
        direction = remaining_vector / remaining
        lateral = np.cross(np.array((0.0, 0.0, 1.0)), direction)
        lateral_norm = float(np.linalg.norm(lateral))
        if lateral_norm < 1e-8:
            raise ContactExecutionError(
                "fresh plate goal direction has no planar perimeter tangent"
            )
        lateral /= lateral_norm
        if perimeter_side is None:
            side = 1.0 if retry_index % 2 == 1 else -1.0
        else:
            side = float(perimeter_side)
            if not np.isfinite(side) or abs(abs(side) - 1.0) > 1e-12:
                raise ValueError("plate retry perimeter side must be -1 or +1")
        angle = self.config.push_retry_perimeter_angle_rad
        contact_ray = (
            np.cos(angle) * direction + side * np.sin(angle) * lateral
        )
        contact_ray /= float(np.linalg.norm(contact_ray))
        contact_radius = max(
            0.005,
            0.82 * radius - self.config.push_retry_inset_m,
        )
        point = center + contact_ray * contact_radius
        point[2] = (
            center[2] + 0.008 - self.config.push_retry_depth_m
        )
        travel = min(
            remaining + self.config.push_retry_margin_m,
            self.config.push_retry_max_distance_m,
        )
        if travel <= 0.0:
            raise ContactExecutionError(
                "fresh plate observation has no bounded remaining correction"
            )
        return replace(
            fresh,
            point_world=point,
            outward_world=-contact_ray,
            manipulation_axis_world=direction,
            manipulation_distance_m=travel,
        )

    def _authorise_plate_fourth_retry(
        self,
        target: ContactTargetEstimate,
        *,
        evidence_attempt: int,
        evidence_side: float,
        fresh_signed_progress_m: float,
        fresh_visual_gain_m: float,
    ) -> None:
        """Gate attempt four on public side evidence and available actions."""

        proof_distance = min(
            self.config.push_load_proof_m,
            target.manipulation_distance_m,
        )
        segment_count = int(
            np.ceil(
                max(0.0, target.manipulation_distance_m - proof_distance)
                / self.config.push_segment_m
            )
        )
        if segment_count > self.config.push_max_segments:
            raise ContactExecutionError(
                "fourth plate rim retry exceeds the bounded segment count"
            )

        step_budget = getattr(self.robot, "step_budget", None)
        steps_executed = getattr(self.robot, "steps_executed", None)
        robot_config = getattr(self.robot, "config", None)
        max_steps_per_waypoint = getattr(
            robot_config, "max_steps_per_waypoint", None
        )
        gripper_hold_steps = getattr(robot_config, "gripper_hold_steps", None)
        budget_values = (
            step_budget,
            steps_executed,
            max_steps_per_waypoint,
            gripper_hold_steps,
        )
        budget_gate_applied = all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and value >= (0 if index == 1 else 1)
            for index, value in enumerate(budget_values)
        )
        remaining_step_budget: int | None = None
        required_policy_action_reserve: int | None = None
        # Missing or malformed budget capabilities must never turn this
        # optional fourth contact into an unmetered action path.  Production
        # exposes all four values, but capability-limited/test adapters still
        # fail closed instead of silently bypassing admission control.
        accepted = False
        if budget_gate_applied:
            remaining_step_budget = int(step_budget - steps_executed)
            # Minimum direct-success-path reserve: three approach/contact
            # moves, one load proof, every <=35-mm retained segment, one
            # retreat, and close/open holds, with each move settling in its
            # first bounded OSC chunk.  This admission threshold deliberately
            # does not claim to reserve every possible MPC replan, typed exit,
            # or recovery path; the robot's global step budget remains their
            # hard total-action ceiling.
            required_policy_action_reserve = int(
                (5 + segment_count) * max_steps_per_waypoint
                + 2 * gripper_hold_steps
            )
            accepted = bool(
                remaining_step_budget >= required_policy_action_reserve
            )

        trace = getattr(self.robot, "phase_trace", None)
        if isinstance(trace, list):
            trace.append(
                {
                    "phase": "plate_fourth_rim_retry_authorization",
                    "attempt_index": 4,
                    "retry_index": 3,
                    "accepted": accepted,
                    "evidence_attempt_index": int(evidence_attempt),
                    "evidence_retry_side": float(evidence_side),
                    "selected_retry_side": float(evidence_side),
                    "fresh_plate_signed_progress_m": float(
                        fresh_signed_progress_m
                    ),
                    "fresh_plate_visual_gain_m": float(fresh_visual_gain_m),
                    "fresh_regression_limit_m": (
                        self._PLATE_MAX_FRESH_REGRESSION_M
                    ),
                    "bounded_segment_count": segment_count,
                    "budget_gate_applied": budget_gate_applied,
                    "action_reserve_semantics": (
                        "minimum_direct_success_path"
                    ),
                    "remaining_step_budget": remaining_step_budget,
                    "required_policy_action_reserve": (
                        required_policy_action_reserve
                    ),
                }
            )
        if not accepted:
            raise ContactExecutionError(
                "fourth plate rim retry lacks a valid minimum "
                "direct-success-path policy-action reserve"
            )

    def _record_contact_move(
        self,
        goal_pose: FloatArray,
        target: ContactTargetEstimate,
        gripper: float,
        attempts: list[PhaseAttempt],
        *,
        max_residual_m: float,
        allow_compact_unknown_contact: bool = False,
        drawer_retry_target_adjacent_corridor: bool = False,
    ) -> None:
        """Run one typed, sensor/proprioceptive contact approach.

        The low-level adapter may accept a stalled GRASP only after its
        bounded Cartesian stall and residual gates fire.  Ordinary free-space
        moves never enable this completion path.
        """

        configure = getattr(self.robot, "set_grasp_contact_enabled", None)
        reset = getattr(self.robot, "reset_grasp_contact", None)
        if callable(configure):
            configure(True, max_residual_m=max_residual_m)
        if callable(reset):
            reset()
        try:
            self._record_move(
                goal_pose,
                target,
                Phase.GRASP,
                gripper,
                True,
                attempts,
                allow_compact_unknown_contact=allow_compact_unknown_contact,
                target_adjacent_approach_residual_m=(
                    max_residual_m
                    if drawer_retry_target_adjacent_corridor
                    else None
                ),
                drawer_retry_target_adjacent_corridor=(
                    drawer_retry_target_adjacent_corridor
                ),
            )
        finally:
            if callable(configure):
                configure(False)

    def _record_move(
        self,
        goal_pose: FloatArray,
        target: ContactTargetEstimate,
        phase: Phase,
        gripper: float,
        touching: bool,
        attempts: list[PhaseAttempt],
        *,
        allow_compact_unknown_contact: bool = False,
        typed_turn: bool = False,
        mechanical_stop_residual_m: float | None = None,
        target_adjacent_approach_residual_m: float | None = None,
        waypoint_position_tolerance_m: float | None = None,
        mechanical_stop_plateau_intervals: int | None = None,
        mechanical_stop_plateau_span_m: float | None = None,
        bounded_contact_handoff: bool = False,
        fresh_visual_handoff_after_chunk: bool = False,
        plate_rim_load_proof_handoff: bool = False,
        drawer_retry_target_adjacent_corridor: bool = False,
        plate_rim_contact_exit: bool = False,
    ) -> bool:
        try:
            mechanical_stop = self._move(
                goal_pose,
                target.requested_labels,
                phase,
                gripper,
                touching,
                allow_compact_unknown_contact=allow_compact_unknown_contact,
                typed_turn=typed_turn,
                mechanical_stop_residual_m=mechanical_stop_residual_m,
                target_adjacent_approach_residual_m=(
                    target_adjacent_approach_residual_m
                ),
                waypoint_position_tolerance_m=waypoint_position_tolerance_m,
                mechanical_stop_plateau_intervals=(
                    mechanical_stop_plateau_intervals
                ),
                mechanical_stop_plateau_span_m=mechanical_stop_plateau_span_m,
                bounded_contact_handoff=bounded_contact_handoff,
                fresh_visual_handoff_after_chunk=(
                    fresh_visual_handoff_after_chunk
                ),
                plate_rim_load_proof_handoff=(
                    plate_rim_load_proof_handoff
                ),
                drawer_retry_target_adjacent_corridor=(
                    drawer_retry_target_adjacent_corridor
                ),
                plate_rim_contact_exit=plate_rim_contact_exit,
            )
        except (ContactExecutionError, ExecutionError, OptimisationError) as exc:
            attempts.append(PhaseAttempt(1, phase, 1, False, str(exc)))
            raise
        if drawer_retry_target_adjacent_corridor and not mechanical_stop:
            detail = (
                "typed monotonic drawer-front corridor executed; defer to "
                "public endpoint, compact-width, and fresh RGB-D gates"
            )
        elif plate_rim_contact_exit and not mechanical_stop:
            detail = (
                "typed plate-rim contact exit executed; defer to public "
                "motion and released-width gates"
            )
        elif plate_rim_load_proof_handoff and not mechanical_stop:
            detail = (
                "bounded plate-rim load-proof OSC prefix executed; defer to "
                "public signed-progress and retained-width gates"
            )
        elif fresh_visual_handoff_after_chunk and not mechanical_stop:
            detail = (
                "bounded fresh-vision contact pulse executed; defer to "
                "fresh local RGB-D"
            )
        elif mechanical_stop and bounded_contact_handoff:
            detail = (
                "bounded retained-contact plateau accepted; defer to fresh "
                "local RGB-D"
            )
        elif mechanical_stop:
            detail = "bounded mechanical stop accepted; defer to fresh RGB-D"
        else:
            detail = "sensor MPC goal reached"
        attempts.append(
            PhaseAttempt(1, phase, 1, True, detail)
        )
        return mechanical_stop

    def _move(
        self,
        goal_pose: FloatArray,
        labels: Sequence[str],
        phase: Phase,
        gripper: float,
        touching: bool,
        *,
        allow_compact_unknown_contact: bool = False,
        typed_turn: bool = False,
        mechanical_stop_residual_m: float | None = None,
        target_adjacent_approach_residual_m: float | None = None,
        waypoint_position_tolerance_m: float | None = None,
        mechanical_stop_plateau_intervals: int | None = None,
        mechanical_stop_plateau_span_m: float | None = None,
        bounded_contact_handoff: bool = False,
        fresh_visual_handoff_after_chunk: bool = False,
        plate_rim_load_proof_handoff: bool = False,
        drawer_retry_target_adjacent_corridor: bool = False,
        plate_rim_contact_exit: bool = False,
    ) -> bool:
        if (
            mechanical_stop_residual_m is not None
            and (
                not np.isfinite(mechanical_stop_residual_m)
                or mechanical_stop_residual_m <= 0.0
            )
        ):
            raise ValueError("mechanical stop residual must be finite and positive")
        if (
            target_adjacent_approach_residual_m is not None
            and (
                not np.isfinite(target_adjacent_approach_residual_m)
                or target_adjacent_approach_residual_m <= 0.0
            )
        ):
            raise ValueError(
                "target-adjacent approach residual must be finite and positive"
            )
        if (
            waypoint_position_tolerance_m is not None
            and (
                not np.isfinite(waypoint_position_tolerance_m)
                or waypoint_position_tolerance_m <= 0.0
            )
        ):
            raise ValueError(
                "waypoint position tolerance must be finite and positive"
            )
        plateau_enabled = (
            mechanical_stop_plateau_intervals is not None
            or mechanical_stop_plateau_span_m is not None
        )
        if plateau_enabled:
            if (
                mechanical_stop_plateau_intervals is None
                or mechanical_stop_plateau_span_m is None
                or not isinstance(mechanical_stop_plateau_intervals, int)
                or mechanical_stop_plateau_intervals < 1
                or not np.isfinite(mechanical_stop_plateau_span_m)
                or mechanical_stop_plateau_span_m <= 0.0
            ):
                raise ValueError(
                    "mechanical-stop plateau requires positive intervals and span"
                )
            if (
                mechanical_stop_residual_m is None
                or not touching
                or phase is not Phase.TRANSFER
                or typed_turn
            ):
                raise ValueError(
                    "mechanical-stop plateau is only valid for a typed touching transfer"
                )
        if fresh_visual_handoff_after_chunk and (
            not touching
            or phase is not Phase.TRANSFER
            or typed_turn
            or mechanical_stop_residual_m is None
        ):
            raise ValueError(
                "fresh-visual handoff is only valid for a bounded touching transfer"
            )
        normalized_labels = {
            " ".join(str(label).lower().replace("_", " ").split())
            for label in labels
        }
        if drawer_retry_target_adjacent_corridor and (
            not {"drawer", "cabinet"}.issubset(normalized_labels)
            or not touching
            or phase is not Phase.GRASP
            or gripper != GRIPPER_CLOSE
            or not allow_compact_unknown_contact
            or target_adjacent_approach_residual_m is None
            or target_adjacent_approach_residual_m
            > self.config.drawer_retry_corridor_stop_residual_m
            or typed_turn
            or mechanical_stop_residual_m is not None
            or waypoint_position_tolerance_m is not None
            or plateau_enabled
            or bounded_contact_handoff
            or fresh_visual_handoff_after_chunk
            or plate_rim_load_proof_handoff
            or plate_rim_contact_exit
        ):
            raise ValueError(
                "drawer retry target-adjacent corridor is only valid for one "
                "compact closed-finger typed drawer approach"
            )
        if plate_rim_contact_exit and (
            normalized_labels != {"plate", "stove"}
            or not touching
            or phase is not Phase.RETREAT
            or gripper != GRIPPER_OPEN
            or allow_compact_unknown_contact
            or typed_turn
            or mechanical_stop_residual_m is not None
            or target_adjacent_approach_residual_m is not None
            or waypoint_position_tolerance_m is not None
            or plateau_enabled
            or bounded_contact_handoff
            or fresh_visual_handoff_after_chunk
            or plate_rim_load_proof_handoff
        ):
            raise ValueError(
                "plate-rim contact exit is only valid for one open-finger "
                "typed plate/stove retreat"
            )
        if plate_rim_load_proof_handoff and (
            "plate" not in normalized_labels
            or not touching
            or phase is not Phase.TRANSFER
            or gripper != GRIPPER_CLOSE
            or allow_compact_unknown_contact
            or typed_turn
            or mechanical_stop_residual_m is not None
            or target_adjacent_approach_residual_m is not None
            or mechanical_stop_plateau_intervals is not None
            or mechanical_stop_plateau_span_m is not None
            or fresh_visual_handoff_after_chunk
            or waypoint_position_tolerance_m is None
            or abs(float(waypoint_position_tolerance_m) - 0.002) > 1e-12
        ):
            raise ValueError(
                "plate-rim load-proof handoff is only valid for a closed-finger "
                "5--10 mm typed plate transfer with the unchanged 2 mm gate"
            )
        position_tolerance_m = (
            self.config.position_tolerance_m
            if waypoint_position_tolerance_m is None
            else waypoint_position_tolerance_m
        )
        typed_linear_name: str | None = None
        typed_linear_origin: np.ndarray | None = None
        typed_linear_axis: np.ndarray | None = None
        typed_linear_length: float | None = None
        typed_linear_lateral_tolerance_m: float | None = None
        typed_linear_start_pose: np.ndarray | None = None
        typed_linear_endpoint_pose: np.ndarray | None = None
        if drawer_retry_target_adjacent_corridor or plate_rim_contact_exit:
            if bool(getattr(self.robot, "step_budget_exhausted", False)):
                raise ContactExecutionError("episode OSC step budget exhausted")
            typed_start = np.asarray(
                self.robot.current_ee_pose(), dtype=np.float64
            )
            typed_goal = np.asarray(goal_pose, dtype=np.float64)
            if (
                typed_start.shape != (4, 4)
                or typed_goal.shape != (4, 4)
                or not np.all(np.isfinite(typed_start))
                or not np.all(np.isfinite(typed_goal))
            ):
                raise ContactExecutionError(
                    "typed contact corridor requires finite public transforms"
                )
            typed_delta = typed_goal[:3, 3] - typed_start[:3, 3]
            typed_length = float(np.linalg.norm(typed_delta))
            _, typed_rotation = self._pose_errors(typed_start, typed_goal)
            if drawer_retry_target_adjacent_corridor:
                valid_command = bool(
                    np.all(np.isfinite(typed_delta))
                    and self.config.precontact_clearance_m
                    - self.config.position_tolerance_m
                    <= typed_length
                    <= self.config.drawer_retry_corridor_max_length_m
                    and abs(float(typed_delta[2])) <= 1e-9
                    and typed_rotation <= 0.010
                )
                typed_linear_name = "drawer retry target-adjacent corridor"
                typed_linear_lateral_tolerance_m = (
                    self.config.drawer_retry_corridor_lateral_tolerance_m
                )
            else:
                horizontal = float(np.linalg.norm(typed_delta[:2]))
                valid_command = bool(
                    np.all(np.isfinite(typed_delta))
                    and 0.005 <= horizontal <= 0.015
                    and 0.008 <= float(typed_delta[2]) <= 0.025
                    and self.config.push_contact_exit_min_progress_m
                    < typed_length
                    <= 0.030
                    and typed_rotation <= 0.010
                )
                typed_linear_name = "plate-rim contact exit"
                typed_linear_lateral_tolerance_m = (
                    self.config.push_contact_exit_max_cross_drift_m
                )
            if not valid_command:
                raise ValueError(
                    f"{typed_linear_name} command is not finite and bounded"
                )
            typed_linear_origin = typed_start[:3, 3].copy()
            typed_linear_axis = typed_delta / typed_length
            typed_linear_length = typed_length
            typed_linear_start_pose = typed_start.copy()
        self.mpc.reset()
        typed_policy_steps = 0
        typed_last_progress_m = 0.0
        plateau_positions: list[np.ndarray] = []
        plateau_position_errors: list[float] = []
        plateau_rotation_errors: list[float] = []
        for replan_index in range(self.config.max_replans):
            current = np.asarray(self.robot.current_ee_pose(), dtype=np.float64)
            if plateau_enabled and not plateau_positions:
                plateau_positions.append(current[:3, 3].copy())
            position_error, rotation_error = self._pose_errors(current, goal_pose)
            if plate_rim_load_proof_handoff and replan_index == 0 and (
                not 0.005 <= position_error <= 0.010
                or rotation_error > 0.010
            ):
                raise ValueError(
                    "plate-rim load-proof handoff command must be a 5--10 mm "
                    "translation with fixed wrist orientation"
                )
            if (
                position_error <= position_tolerance_m
                and rotation_error <= self.config.orientation_tolerance_rad
            ):
                if (
                    (drawer_retry_target_adjacent_corridor or plate_rim_contact_exit)
                    and typed_policy_steps <= 0
                ):
                    raise ContactExecutionError(
                        f"{typed_linear_name} executed no policy action"
                    )
                return False
            contact_observer = getattr(self.observer, "observe_contact", None)
            if allow_compact_unknown_contact and callable(contact_observer):
                scene = contact_observer(labels, goal_pose[:3, 3])
            else:
                scene = self.observer.observe(labels)
            request = MotionRequest(
                phase=phase,
                start_pose=current,
                goal_pose=goal_pose,
                clearance_m=(
                    self.config.free_clearance_m
                    if drawer_retry_target_adjacent_corridor or not touching
                    else 0.0
                ),
                tool_radius_m=(
                    self.config.free_tool_radius_m
                    if drawer_retry_target_adjacent_corridor or not touching
                    else 0.0
                ),
                workspace_margin_m=0.005,
            )
            chunk = self.mpc.replan(request, scene)
            if typed_linear_name is not None:
                assert typed_linear_origin is not None
                assert typed_linear_axis is not None
                assert typed_linear_length is not None
                assert typed_linear_lateral_tolerance_m is not None
                self._validate_typed_monotonic_chunk(
                    chunk.poses,
                    origin=typed_linear_origin,
                    axis=typed_linear_axis,
                    length_m=typed_linear_length,
                    lateral_tolerance_m=typed_linear_lateral_tolerance_m,
                    minimum_progress_m=typed_last_progress_m,
                    name=typed_linear_name,
                )
            steps_before = getattr(self.robot, "steps_executed", None)
            feedback = self.robot.execute_waypoints(chunk.poses, phase, gripper)
            steps_after = getattr(self.robot, "steps_executed", None)
            chunk_position_gap, chunk_rotation_gap = self._pose_errors(
                np.asarray(chunk.poses[-1]), goal_pose
            )
            chunk_reaches_goal = bool(
                chunk_position_gap <= 1e-6 and chunk_rotation_gap <= 1e-6
            )
            if drawer_retry_target_adjacent_corridor or plate_rim_contact_exit:
                if bool(getattr(self.robot, "step_budget_exhausted", False)):
                    raise ContactExecutionError("episode OSC step budget exhausted")
                if (
                    not isinstance(steps_before, int)
                    or isinstance(steps_before, bool)
                    or not isinstance(steps_after, int)
                    or isinstance(steps_after, bool)
                    or steps_after <= steps_before
                ):
                    raise ContactExecutionError(
                        f"{typed_linear_name} executed no policy action"
                    )
                typed_policy_steps += steps_after - steps_before
                assert typed_linear_origin is not None
                assert typed_linear_axis is not None
                typed_endpoint = np.asarray(
                    self.robot.current_ee_pose(), dtype=np.float64
                )
                if typed_endpoint.shape != (4, 4) or not np.all(
                    np.isfinite(typed_endpoint)
                ):
                    raise ContactExecutionError(
                        f"{typed_linear_name} lost finite public proprioception"
                    )
                measured_progress = float(
                    np.dot(
                        typed_endpoint[:3, 3] - typed_linear_origin,
                        typed_linear_axis,
                    )
                )
                typed_relative = (
                    typed_endpoint[:3, 3] - typed_linear_origin
                )
                measured_lateral = float(
                    np.linalg.norm(
                        typed_relative
                        - measured_progress * typed_linear_axis
                    )
                )
                assert typed_linear_length is not None
                assert typed_linear_lateral_tolerance_m is not None
                if (
                    not np.isfinite(measured_progress)
                    or not np.isfinite(measured_lateral)
                    or measured_progress + 1e-5 < typed_last_progress_m
                    or measured_progress
                    > typed_linear_length + self.config.position_tolerance_m
                    or measured_lateral > typed_linear_lateral_tolerance_m
                ):
                    raise ContactExecutionError(
                        f"{typed_linear_name} public endpoint left the "
                        "bounded monotonic corridor"
                    )
                typed_last_progress_m = max(
                    typed_last_progress_m, measured_progress
                )
                typed_linear_endpoint_pose = typed_endpoint.copy()
            if plate_rim_load_proof_handoff:
                if bool(getattr(self.robot, "step_budget_exhausted", False)):
                    raise ContactExecutionError(
                        "episode OSC step budget exhausted"
                    )
                if (
                    not isinstance(steps_before, int)
                    or isinstance(steps_before, bool)
                    or not isinstance(steps_after, int)
                    or isinstance(steps_after, bool)
                    or steps_after <= steps_before
                ):
                    raise ContactExecutionError(
                        "plate-rim load-proof handoff executed no policy action"
                    )
                if not feedback.accepted:
                    if bool(getattr(self.robot, "step_budget_exhausted", False)):
                        raise ContactExecutionError(
                            "episode OSC step budget exhausted"
                        )
                    raise ContactExecutionError(
                        feedback.detail
                        or "plate-rim load-proof OSC prefix was rejected"
                    )
                # Exactly one non-zero OSC prefix is allowed.  The caller
                # immediately measures signed public-EE progress and retained
                # width; this flag never changes the ordinary 2-mm gate.
                return False
            # The accepted partial-prefix feedback carries the endpoint
            # errors from the exact public-proprioception sample that closed
            # the OSC chunk.  Advance immediately when that sample is already
            # inside the ordinary waypoint gate.  Re-observing before the
            # next outer iteration can expose a subsequent compliant-contact
            # sample and otherwise waste several full no-progress chunks.
            if (
                feedback.accepted
                and chunk_reaches_goal
                and feedback.position_error_m is not None
                and feedback.rotation_error_rad is not None
                and np.isfinite(feedback.position_error_m)
                and np.isfinite(feedback.rotation_error_rad)
                and 0.0 <= feedback.position_error_m <= position_tolerance_m
                and 0.0
                <= feedback.rotation_error_rad
                <= self.config.orientation_tolerance_rad
            ):
                return False
            if (
                plateau_enabled
                and feedback.accepted
                and feedback.position_error_m is not None
                and feedback.rotation_error_rad is not None
                and np.isfinite(feedback.position_error_m)
                and np.isfinite(feedback.rotation_error_rad)
                and feedback.position_error_m >= 0.0
                and feedback.rotation_error_rad >= 0.0
            ):
                plateau_position_errors.append(float(feedback.position_error_m))
                plateau_rotation_errors.append(float(feedback.rotation_error_rad))
            if (
                not feedback.accepted
                and bool(getattr(self.robot, "step_budget_exhausted", False))
            ):
                raise ContactExecutionError(
                    "episode OSC step budget exhausted"
                )
            if not feedback.accepted:
                # A compliant contact can finish millimetres short of the
                # Cartesian target.  Accept only a near-endpoint touching
                # transfer; long drawer/door/plate stalls remain failures.
                current = np.asarray(self.robot.current_ee_pose(), dtype=np.float64)
                contact_position_error, contact_rotation_error = self._pose_errors(
                    current, goal_pose
                )
                # Some adapters return the pose errors measured in the same
                # final public-proprioception sample that triggered their OSC
                # stall detector.  Reading current_ee_pose() immediately
                # afterwards can yield the next sample.  Admit either sample,
                # but only through the same bounded contact residual and
                # orientation gates; evaluator state is never involved.
                feedback_position_error = feedback.position_error_m
                feedback_rotation_error = feedback.rotation_error_rad
                current_goal_reached = bool(
                    contact_position_error <= position_tolerance_m
                    and contact_rotation_error
                    <= self.config.orientation_tolerance_rad
                )
                feedback_goal_reached = bool(
                    chunk_reaches_goal
                    and feedback_position_error is not None
                    and feedback_rotation_error is not None
                    and np.isfinite(feedback_position_error)
                    and np.isfinite(feedback_rotation_error)
                    and feedback_position_error >= 0.0
                    and feedback_rotation_error >= 0.0
                    and feedback_position_error
                    <= position_tolerance_m
                    and feedback_rotation_error
                    <= self.config.orientation_tolerance_rad
                )
                feedback_stop_within_gate = bool(
                    feedback_position_error is not None
                    and feedback_rotation_error is not None
                    and np.isfinite(feedback_position_error)
                    and np.isfinite(feedback_rotation_error)
                    and feedback_position_error >= 0.0
                    and feedback_rotation_error >= 0.0
                    and feedback_position_error
                    <= (
                        0.025
                        if mechanical_stop_residual_m is None
                        else mechanical_stop_residual_m
                    )
                    and feedback_rotation_error
                    <= self.config.orientation_tolerance_rad
                )
                # A low-level OSC stall may be reported on the same sample
                # that entered the executor's ordinary goal tolerance.  That
                # is a reached waypoint, not evidence of a physical jamb.
                # Continue the segmented arc; reserve the wider typed-stop
                # residual for an endpoint genuinely outside normal bounds.
                if (
                    touching
                    and phase is Phase.TRANSFER
                    and not typed_turn
                    and (current_goal_reached or feedback_goal_reached)
                ):
                    return False
                if (
                    touching
                    and phase is Phase.TRANSFER
                    and not typed_turn
                    and (
                        (
                            contact_position_error
                            <= (
                                0.025
                                if mechanical_stop_residual_m is None
                                else mechanical_stop_residual_m
                            )
                            and contact_rotation_error
                            <= self.config.orientation_tolerance_rad
                        )
                        or feedback_stop_within_gate
                    )
                ):
                    return mechanical_stop_residual_m is not None
                # A collision-checked free-space approach can settle just
                # outside the nominal 10-mm MPC tolerance under OSC.  Admit
                # only a small, well-oriented endpoint plateau; contact and
                # transfer phases retain their typed/mechanical gates.
                if (
                    not touching
                    and phase is Phase.APPROACH
                    and contact_position_error <= 0.025
                    and contact_rotation_error
                    <= self.config.orientation_tolerance_rad
                ):
                    return False
                if (
                    touching
                    and (
                        phase is Phase.APPROACH
                        or (
                            drawer_retry_target_adjacent_corridor
                            and phase is Phase.GRASP
                        )
                    )
                    and target_adjacent_approach_residual_m is not None
                    and contact_position_error
                    <= target_adjacent_approach_residual_m
                    and contact_rotation_error
                    <= self.config.orientation_tolerance_rad
                ):
                    return False
                if (
                    drawer_retry_target_adjacent_corridor
                    and phase is Phase.GRASP
                    and not bool(
                        getattr(self.robot, "step_budget_exhausted", False)
                    )
                    and typed_linear_start_pose is not None
                    and typed_linear_endpoint_pose is not None
                    and typed_linear_axis is not None
                ):
                    motion_plateau = self._capture_drawer_close_motion_plateau(
                        command_start=typed_linear_start_pose,
                        command_goal=goal_pose,
                        command_end=typed_linear_endpoint_pose,
                        axis=typed_linear_axis,
                        policy_actions=typed_policy_steps,
                        width_m=self._public_gripper_width_m(),
                    )
                    if motion_plateau is not None:
                        raise _DrawerCloseTypedCorridorPlateau(
                            "rejected typed drawer corridor retained bounded "
                            "public motion evidence",
                            motion_plateau,
                        )
                # A drawer-open retreat begins with one explicitly typed,
                # outward contact-exit step.  The freshly opened lip may
                # settle a few centimetres short; admit only this bounded
                # contact residual, then require the following lift and
                # outward segment to use ordinary SDF planning.
                if (
                    touching
                    and phase is Phase.RETREAT
                    and allow_compact_unknown_contact
                    and contact_position_error <= 0.035
                    and contact_rotation_error
                    <= self.config.orientation_tolerance_rad
                ):
                    return False
                raise ContactExecutionError(
                    feedback.detail or f"robot rejected {phase.value} motion"
                )
            if fresh_visual_handoff_after_chunk:
                if (
                    isinstance(steps_before, int)
                    and not isinstance(steps_before, bool)
                    and isinstance(steps_after, int)
                    and not isinstance(steps_after, bool)
                    and steps_after <= steps_before
                ):
                    raise ContactExecutionError(
                        "fresh-vision contact pulse executed no policy action"
                    )
                # Deliberately stop after one bounded OSC prefix.  The caller
                # must now reacquire the local edge and pass the frozen
                # hinge/radius/direction gates before issuing another pulse.
                return False
            if (
                touching
                and phase is Phase.GRASP
                and bool(getattr(self.robot, "grasp_contact_reached", False))
            ):
                return False
            if (
                typed_turn
                and phase is Phase.TRANSFER
                and bool(getattr(self.robot, "turn_contact_reached", False))
            ):
                return False
            if plateau_enabled:
                assert mechanical_stop_plateau_intervals is not None
                assert mechanical_stop_plateau_span_m is not None
                endpoint = np.asarray(
                    self.robot.current_ee_pose(), dtype=np.float64
                )
                plateau_positions.append(endpoint[:3, 3].copy())
                window_size = mechanical_stop_plateau_intervals + 1
                if len(plateau_positions) > window_size:
                    plateau_positions = plateau_positions[-window_size:]
                if len(plateau_position_errors) > window_size:
                    plateau_position_errors = plateau_position_errors[-window_size:]
                    plateau_rotation_errors = plateau_rotation_errors[-window_size:]
                endpoint_position_error, endpoint_rotation_error = self._pose_errors(
                    endpoint, goal_pose
                )
                if len(plateau_positions) == window_size:
                    window = np.stack(plateau_positions)
                    window_span = float(
                        np.max(
                            np.linalg.norm(
                                window - window[0][None, :],
                                axis=1,
                            )
                        )
                    )
                    feedback_error_plateau = bool(
                        len(plateau_position_errors) == window_size
                        and min(plateau_position_errors) > position_tolerance_m
                        and max(plateau_position_errors)
                        <= mechanical_stop_residual_m
                        and max(plateau_position_errors)
                        - min(plateau_position_errors)
                        <= mechanical_stop_plateau_span_m
                        and max(plateau_rotation_errors)
                        <= self.config.orientation_tolerance_rad
                        and max(plateau_rotation_errors)
                        - min(plateau_rotation_errors)
                        <= self.config.microwave_arc_plateau_rotation_span_rad
                    )
                    # This is an explicit measured pose plateau, not an
                    # iteration-limit success.  The caller immediately adds
                    # the retained-width gate, and final semantics still come
                    # only from fresh RGB-D.
                    if (
                        endpoint_position_error > position_tolerance_m
                        and endpoint_position_error
                        <= mechanical_stop_residual_m
                        and endpoint_rotation_error
                        <= self.config.orientation_tolerance_rad
                        and (
                            window_span <= mechanical_stop_plateau_span_m
                            or feedback_error_plateau
                        )
                    ):
                        return True
        current = np.asarray(self.robot.current_ee_pose(), dtype=np.float64)
        position_error, rotation_error = self._pose_errors(current, goal_pose)
        if (
            typed_turn
            and phase is Phase.TRANSFER
            and bool(getattr(self.robot, "turn_contact_reached", False))
        ):
            return False
        raise ContactExecutionError(
            f"{phase.value} did not converge: position={position_error:.4f} m, "
            f"rotation={rotation_error:.4f} rad"
        )

    @staticmethod
    def _validate_typed_monotonic_chunk(
        poses_world: FloatArray,
        *,
        origin: FloatArray,
        axis: FloatArray,
        length_m: float,
        lateral_tolerance_m: float,
        minimum_progress_m: float,
        name: str,
    ) -> None:
        """Reject detours or reversals inside a short typed contact corridor."""

        poses = np.asarray(poses_world, dtype=np.float64)
        start = np.asarray(origin, dtype=np.float64)
        direction = np.asarray(axis, dtype=np.float64)
        if (
            poses.ndim != 3
            or poses.shape[1:] != (4, 4)
            or len(poses) < 2
            or start.shape != (3,)
            or direction.shape != (3,)
            or not np.all(np.isfinite(poses))
            or not np.all(np.isfinite(start))
            or not np.all(np.isfinite(direction))
            or not np.isfinite(length_m)
            or length_m <= 0.0
            or not np.isfinite(lateral_tolerance_m)
            or lateral_tolerance_m <= 0.0
            or not np.isfinite(minimum_progress_m)
            or minimum_progress_m < 0.0
        ):
            raise ContactExecutionError(f"{name} produced invalid MPC waypoints")
        direction_norm = float(np.linalg.norm(direction))
        if abs(direction_norm - 1.0) > 1e-6:
            raise ContactExecutionError(f"{name} axis is not a finite unit vector")
        relative = poses[:, :3, 3] - start[None, :]
        progress = relative @ direction
        lateral = relative - progress[:, None] * direction[None, :]
        lateral_norm = np.linalg.norm(lateral, axis=1)
        tolerance = 1e-5
        if (
            progress[0] < minimum_progress_m - tolerance
            or np.any(progress < -tolerance)
            or np.any(progress > length_m + tolerance)
            or np.any(np.diff(progress) < -tolerance)
            or np.any(lateral_norm > lateral_tolerance_m)
        ):
            raise ContactExecutionError(
                f"{name} MPC waypoints left the bounded monotonic corridor"
            )

    def _drawer_is_released(self) -> bool:
        """Require a public proprioceptive open-width sample."""

        width = self._public_gripper_width_m()
        return bool(
            width is not None
            and width >= self.config.drawer_release_min_width_m
        )

    @staticmethod
    def _microwave_terminal_exit_defer_allowed(
        *,
        final_goal: bool,
        supplemental_close: bool,
        release_proven: bool,
        typed_exit_progress_m: float,
        required_progress_m: float,
    ) -> bool:
        """Gate direct fresh verification after a terminal close supplement."""

        return bool(
            final_goal
            and supplemental_close
            and release_proven
            and np.isfinite(typed_exit_progress_m)
            and np.isfinite(required_progress_m)
            and required_progress_m > 0.0
            and typed_exit_progress_m >= required_progress_m
        )

    @staticmethod
    def _drawer_exit_defer_allowed(
        *,
        final_goal: bool,
        release_proven: bool,
        safe_escape_executed: bool,
        no_progress_evidence: int,
    ) -> bool:
        """Gate the last-goal retreat exception before fresh RGB-D verify."""

        return bool(
            final_goal
            and release_proven
            and safe_escape_executed
            and no_progress_evidence >= 3
        )

    def _set_gripper(
        self, command: float, phase: Phase, attempts: list[PhaseAttempt]
    ) -> None:
        feedback: ControllerFeedback = self.robot.set_gripper(command)
        attempts.append(
            PhaseAttempt(
                1,
                phase,
                1,
                feedback.accepted,
                feedback.detail or "gripper command accepted",
            )
        )
        if not feedback.accepted:
            raise ContactExecutionError(feedback.detail or "gripper command rejected")

    def _verify_progress(
        self,
        goal: AtomicGoal,
        initial: ContactTargetEstimate,
        attempts: list[PhaseAttempt],
    ) -> None:
        if goal.kind in {AtomicGoalKind.TURN_ON, AtomicGoalKind.TURN_OFF}:
            progress = self._last_turn_progress_rad
            if (
                progress is None
                or progress < self.config.knob_mechanical_completion_rad
            ):
                raise ContactExecutionError(
                    "turn verification lacked measured signed wrist rotation"
                )
            attempts.append(
                PhaseAttempt(
                    1,
                    Phase.VERIFY,
                    1,
                    True,
                    f"blocked-width turn completed with {progress:.3f} rad measured wrist rotation",
                )
            )
            return
        if not initial.visual_progress_expected:
            attempts.append(
                PhaseAttempt(
                    1,
                    Phase.VERIFY,
                    1,
                    True,
                    "sensor/proprioceptive motion completed; target appearance is not state-observable",
                )
            )
            return
        current = self.provider.estimate(goal)
        projected = float(
            np.dot(
                current.point_world - initial.point_world,
                initial.manipulation_axis_world,
            )
        )
        required_progress = (
            self.config.push_required_progress_m
            if goal.kind is AtomicGoalKind.PUSH
            else min(
                self.config.visual_progress_m,
                0.5 * initial.manipulation_distance_m,
            )
        )
        radius_error_m: float | None = None
        radius_consistent = True
        if (
            goal.subject.label == "microwave"
            and initial.rotation_center_world is not None
            and initial.rotation_axis_world is not None
        ):
            # Project both fresh and frozen contact features onto the same
            # public hinge plane.  True door motion preserves this radius;
            # an occlusion-driven switch to the appliance frame/body does
            # not.  Use the initial frozen hinge so a second noisy estimate
            # cannot move both the point and its reference together.
            center = initial.rotation_center_world
            axis = initial.rotation_axis_world
            initial_radial = initial.point_world - center
            initial_radial -= axis * float(np.dot(initial_radial, axis))
            current_radial = current.point_world - center
            current_radial -= axis * float(np.dot(current_radial, axis))
            radius_error_m = abs(
                float(np.linalg.norm(current_radial))
                - float(np.linalg.norm(initial_radial))
            )
            radius_consistent = bool(
                radius_error_m
                <= self.config.microwave_verify_radius_tolerance_m
            )
        success = projected >= required_progress and radius_consistent
        radius_detail = (
            ""
            if radius_error_m is None
            else f", frozen-hinge radius error={radius_error_m:.4f} m"
        )
        attempts.append(
            PhaseAttempt(
                1,
                Phase.VERIFY,
                1,
                success,
                f"fresh RGB-D projected progress={projected:.4f} m"
                + radius_detail,
            )
        )
        if not success:
            raise ContactExecutionError(
                "fresh RGB-D did not confirm contact progress "
                f"({projected:.4f} m{radius_detail})"
            )

    @staticmethod
    def _contact_pose(target: ContactTargetEstimate) -> FloatArray:
        tool_z = -target.outward_world
        # ``feature_axis_world`` is the observed long axis of a drawer or
        # microwave handle.  The jaws must close *across* that axis, not along
        # it: vertical LIBERO handles therefore get a horizontal pinch and
        # horizontal bars get a vertical pinch.
        jaw_axis = np.cross(target.feature_axis_world, target.outward_world)
        finger = jaw_axis - tool_z * float(
            np.dot(jaw_axis, tool_z)
        )
        if float(np.linalg.norm(finger)) < 1e-6:
            finger = np.array([0.0, 0.0, 1.0])
            finger -= tool_z * float(np.dot(finger, tool_z))
        if float(np.linalg.norm(finger)) < 1e-6:
            finger = np.array([1.0, 0.0, 0.0])
            finger -= tool_z * float(np.dot(finger, tool_z))
        finger /= float(np.linalg.norm(finger))
        tool_x = np.cross(finger, tool_z)
        tool_x /= float(np.linalg.norm(tool_x))
        finger = np.cross(tool_z, tool_x)
        pose = np.eye(4)
        pose[:3, :3] = np.column_stack((tool_x, finger, tool_z))
        pose[:3, 3] = target.point_world
        return pose

    def _vertical_contact_pose(self, target: ContactTargetEstimate) -> FloatArray:
        pose = np.asarray(self.robot.current_ee_pose(), dtype=np.float64).copy()
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            raise ContactExecutionError("public EE pose must be a finite 4x4 matrix")
        pose[:3, 3] = target.point_world
        return pose

    def _vertical_jaw_aligned_pose(
        self,
        target: ContactTargetEstimate,
        jaw_axis_world: FloatArray,
    ) -> FloatArray:
        """Keep the reachable tool axis while aligning the symmetric jaws."""

        pose = self._vertical_contact_pose(target)
        tool_z = pose[:3, 2].copy()
        tool_z /= max(float(np.linalg.norm(tool_z)), 1e-12)
        jaw = np.asarray(jaw_axis_world, dtype=np.float64).copy()
        jaw -= tool_z * float(np.dot(jaw, tool_z))
        norm = float(np.linalg.norm(jaw))
        if not np.isfinite(norm) or norm < 1e-6:
            raise ContactExecutionError(
                "plate direction cannot define a finite jaw-closing axis"
            )
        jaw /= norm
        # The gripper is symmetric under a pi yaw.  Choose the equivalent
        # axis nearest the current wrist frame to avoid an unnecessary flip.
        if float(np.dot(jaw, pose[:3, 1])) < 0.0:
            jaw *= -1.0
        tool_x = np.cross(jaw, tool_z)
        tool_x /= max(float(np.linalg.norm(tool_x)), 1e-12)
        jaw = np.cross(tool_z, tool_x)
        pose[:3, :3] = np.column_stack((tool_x, jaw, tool_z))
        return pose

    def _nearest_symmetric_jaw_pose(self, pose_world: FloatArray) -> FloatArray:
        """Choose the pi-equivalent parallel-jaw frame nearest current wrist."""

        pose = np.asarray(pose_world, dtype=np.float64).copy()
        current = np.asarray(self.robot.current_ee_pose(), dtype=np.float64)
        flipped = pose.copy()
        flipped[:3, 0] *= -1.0
        flipped[:3, 1] *= -1.0
        direct_error = float(
            Rotation.from_matrix(pose[:3, :3] @ current[:3, :3].T).magnitude()
        )
        flipped_error = float(
            Rotation.from_matrix(flipped[:3, :3] @ current[:3, :3].T).magnitude()
        )
        return flipped if flipped_error < direct_error else pose

    @staticmethod
    def _axis_rotation_progress(
        start_rotation: FloatArray,
        current_rotation: FloatArray,
        expected_axis_world: FloatArray,
    ) -> float:
        axis = np.asarray(expected_axis_world, dtype=np.float64)
        norm = float(np.linalg.norm(axis))
        if axis.shape != (3,) or not np.isfinite(norm) or norm < 1e-8:
            raise ValueError("expected contact rotation axis must be finite and non-zero")
        axis /= norm
        relative = np.asarray(current_rotation) @ np.asarray(start_rotation).T
        rotvec = Rotation.from_matrix(relative).as_rotvec()
        return float(np.dot(rotvec, axis))

    @staticmethod
    def _pose_errors(current: FloatArray, goal: FloatArray) -> tuple[float, float]:
        position = float(np.linalg.norm(goal[:3, 3] - current[:3, 3]))
        relative = goal[:3, :3] @ current[:3, :3].T
        orientation = float(Rotation.from_matrix(relative).magnitude())
        return position, orientation


@dataclass(frozen=True)
class GoalExecutionRecord:
    index: int
    goal: AtomicGoal
    success: bool
    failure: str | None
    attempts: tuple[PhaseAttempt, ...]
    verification: Verification | None = None
    grasp_attempt_events: tuple[GraspAttemptEvent, ...] = ()


@dataclass(frozen=True)
class RouteCSequenceResult:
    success: bool
    task_text: str
    plan: TaskPlan
    goals: tuple[GoalExecutionRecord, ...]
    failure: str | None = None

    @property
    def attempts(self) -> tuple[PhaseAttempt, ...]:
        return tuple(item for goal in self.goals for item in goal.attempts)

    @property
    def verification(self) -> Verification | None:
        for goal in reversed(self.goals):
            if goal.verification is not None:
                return goal.verification
        return None

    @property
    def grasp_attempt_events(self) -> tuple[GraspAttemptEvent, ...]:
        """Episode-global grasp attempts, retained in goal execution order."""

        return tuple(
            event
            for goal in self.goals
            for event in goal.grasp_attempt_events
        )


class RouteCSequentialCoordinator:
    """Run every planned goal independently and in language order."""

    def __init__(
        self,
        motion_controller: object,
        graph_compiler: AtomicGoalGraphCompiler,
        contact_executor: MPCContactGoalExecutor,
        *,
        planner: RouteCTaskPlanner | None = None,
        between_goals: Callable[[], None] | None = None,
        source_consumed: Callable[[str], None] | None = None,
        formed_stack_handler: FormedStackHandler | None = None,
        ranked_target_handler: RankedTargetHandler | None = None,
    ) -> None:
        if not callable(getattr(motion_controller, "run", None)):
            raise TypeError("motion controller must implement run(text)")
        self.motion_controller = motion_controller
        self.graph_compiler = graph_compiler
        self.contact_executor = contact_executor
        self.planner = planner or RouteCTaskPlanner()
        self.between_goals = between_goals
        self.source_consumed = source_consumed
        self.formed_stack_handler = formed_stack_handler
        self.ranked_target_handler = ranked_target_handler

    def run(self, instruction: str) -> RouteCSequenceResult:
        plan = self.planner.plan(instruction)
        if (
            any(goal.kind is AtomicGoalKind.PLACE_GROUP for goal in plan.goals)
            and self.formed_stack_handler is None
        ):
            return RouteCSequenceResult(
                False,
                instruction,
                plan,
                (),
                "formed-stack carry requires a sensor proof handler",
            )
        # Freeze the episode-reset wrist hemisphere once, after language-only
        # planning has succeeded but before any sensor-prefetch or action.
        # This prevents a preceding contact goal from donating its tilted
        # wrist frame to later top-down grasps, while a genuine zero-step
        # semantic failure still needs no sensor handoff.
        grasp_provider = getattr(self.motion_controller, "grasp_provider", None)
        freeze_reset_pose = getattr(grasp_provider, "freeze_reset_pose", None)
        if callable(freeze_reset_pose):
            freeze_reset_pose()
        if self.ranked_target_handler is not None:
            prefetch_failure = self.ranked_target_handler.prefetch(plan)
            if prefetch_failure is not None:
                return RouteCSequenceResult(
                    False,
                    instruction,
                    plan,
                    (),
                    f"ranked-target prefetch failed: {prefetch_failure}",
                )
        records: list[GoalExecutionRecord] = []
        grasp_event_count = 0
        for index, goal in enumerate(plan.goals):
            if index and self.between_goals is not None:
                self.between_goals()
            preparation: Verification | None = None
            if goal.kind is AtomicGoalKind.PLACE_GROUP:
                assert self.formed_stack_handler is not None
                preparation = self.formed_stack_handler.prepare_carry(goal)
                if not preparation.success:
                    self.formed_stack_handler.clear_carry_requirement()
                    record = GoalExecutionRecord(
                        index,
                        goal,
                        False,
                        preparation.detail,
                        (
                            PhaseAttempt(
                                1,
                                Phase.VERIFY,
                                1,
                                False,
                                preparation.detail,
                            ),
                        ),
                        preparation,
                    )
                    records.append(record)
                    return RouteCSequenceResult(
                        False,
                        instruction,
                        plan,
                        tuple(records),
                        f"goal {index + 1}/{len(plan.goals)} failed: "
                        f"{preparation.detail}",
                    )
            if goal.kind in AtomicGoalGraphCompiler.MOTION_KINDS:
                target_failure = (
                    self.ranked_target_handler.prepare_target(goal)
                    if self.ranked_target_handler is not None
                    else None
                )
                if target_failure is not None:
                    if self.formed_stack_handler is not None:
                        self.formed_stack_handler.clear_carry_requirement()
                    record = GoalExecutionRecord(
                        index,
                        goal,
                        False,
                        target_failure,
                        (),
                    )
                    records.append(record)
                    return RouteCSequenceResult(
                        False,
                        instruction,
                        plan,
                        tuple(records),
                        f"goal {index + 1}/{len(plan.goals)} failed: "
                        f"{target_failure}",
                    )
                self.graph_compiler.select(goal)
                try:
                    result: RouteCResult = self.motion_controller.run(instruction)
                    group_verification = None
                    if (
                        goal.kind is AtomicGoalKind.PLACE_GROUP
                        and result.success
                    ):
                        assert self.formed_stack_handler is not None
                        group_verification = self.formed_stack_handler.verify_carry(
                            goal, result
                        )
                finally:
                    if (
                        goal.kind is AtomicGoalKind.PLACE_GROUP
                        and self.formed_stack_handler is not None
                    ):
                        self.formed_stack_handler.clear_carry_requirement()
                    if self.ranked_target_handler is not None:
                        self.ranked_target_handler.clear_target_requirement()
                group_attempts = ()
                if preparation is not None:
                    group_attempts += (
                        PhaseAttempt(
                            1,
                            Phase.VERIFY,
                            1,
                            preparation.success,
                            f"formed-stack precondition: {preparation.detail}",
                        ),
                    )
                if group_verification is not None:
                    group_attempts += (
                        PhaseAttempt(
                            1,
                            Phase.VERIFY,
                            2,
                            group_verification.success,
                            group_verification.detail,
                        ),
                    )
                success = result.success and (
                    group_verification is None or group_verification.success
                )
                failure = result.failure
                verification = result.verification
                if group_verification is not None:
                    verification = group_verification
                    if not group_verification.success:
                        failure = group_verification.detail
                goal_grasp_events = tuple(
                    replace(
                        event,
                        attempt_index=grasp_event_count + local_index,
                    )
                    for local_index, event in enumerate(
                        result.grasp_attempt_events, start=1
                    )
                )
                grasp_event_count += len(goal_grasp_events)
                record = GoalExecutionRecord(
                    index,
                    goal,
                    success,
                    failure,
                    (*group_attempts, *result.attempts),
                    verification,
                    goal_grasp_events,
                )
            elif goal.kind in MPCContactGoalExecutor.SUPPORTED_KINDS:
                set_final_goal_context = getattr(
                    self.contact_executor,
                    "set_final_goal_context",
                    None,
                )
                if callable(set_final_goal_context):
                    set_final_goal_context(index + 1 == len(plan.goals))
                contact = self.contact_executor.execute(goal)
                record = GoalExecutionRecord(
                    index,
                    goal,
                    contact.success,
                    contact.failure,
                    contact.attempts,
                )
            else:
                record = GoalExecutionRecord(
                    index,
                    goal,
                    False,
                    f"no Route C executor for {goal.kind.value!r}",
                    (),
                )
            if (
                record.success
                and goal.kind is AtomicGoalKind.STACK
                and index + 1 < len(plan.goals)
                and plan.goals[index + 1].kind is AtomicGoalKind.PLACE_GROUP
            ):
                assert self.formed_stack_handler is not None
                capture = self.formed_stack_handler.capture_stack(goal, result)
                if not capture.success:
                    record = replace(
                        record,
                        success=False,
                        failure=capture.detail,
                        attempts=(
                            *record.attempts,
                            PhaseAttempt(
                                1,
                                Phase.VERIFY,
                                2,
                                False,
                                capture.detail,
                            ),
                        ),
                        verification=capture,
                    )
            records.append(record)
            if (
                record.success
                and goal.kind in AtomicGoalGraphCompiler.MOTION_KINDS
                and result.source_id is not None
                and self.source_consumed is not None
            ):
                self.source_consumed(result.source_id)
            if not record.success:
                return RouteCSequenceResult(
                    False,
                    instruction,
                    plan,
                    tuple(records),
                    f"goal {index + 1}/{len(plan.goals)} failed: {record.failure}",
                )
        return RouteCSequenceResult(True, instruction, plan, tuple(records))


def executable_plan(plan: TaskPlan) -> bool:
    """Static fail-closed dispatch check; it performs no sensor or robot I/O."""

    supported = AtomicGoalGraphCompiler.MOTION_KINDS | MPCContactGoalExecutor.SUPPORTED_KINDS
    for goal in plan.goals:
        if goal.kind not in supported:
            return False
        if goal.kind is AtomicGoalKind.PLACE_GROUP and (
            len(goal.subjects) != 2 or goal.relation is not PlannerRelation.IN
        ):
            return False
        if goal.kind in AtomicGoalGraphCompiler.MOTION_KINDS:
            try:
                AtomicGoalGraphCompiler.lower(goal, plan.instruction)
            except GoalLoweringError:
                return False
    return True


__all__ = [
    "AtomicGoalGraphCompiler",
    "ContactExecutionError",
    "ContactGoalResult",
    "ContactTargetEstimate",
    "ContactTargetProvider",
    "FormedStackHandler",
    "GoalExecutionRecord",
    "GoalLoweringError",
    "MPCContactGoalExecutor",
    "RankedTargetHandler",
    "RouteCContactConfig",
    "RouteCSequenceResult",
    "RouteCSequentialCoordinator",
    "executable_plan",
]
