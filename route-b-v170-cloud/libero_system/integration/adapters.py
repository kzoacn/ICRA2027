"""Thin adapters from the shared sensor boundary to Route B and Route C."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, replace
import re
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from libero_system.goal_skills import (
    ContactTarget,
    DrawerEpisodeAnchor,
    DrawerHandleDetector,
    GoalContactPolicy,
    GoalExecutorStatus,
    GoalSkillKind,
    MicrowaveDoorHandleDetector,
    PlateFrontDetector,
    StoveKnobDetector,
)
from libero_system.common import (
    GraspAttemptEvent,
    PendingGraspEngagement,
    OSCAction,
    PolicyDecision,
    PolicyTask,
    RobotObservation,
)
from libero_system.integration.cavity import (
    CavityGeometryError,
    align_panda_finger_axis,
    infer_cavity_frame,
)
from libero_system.route_b import (
    CameraIntrinsics as BCameraIntrinsics,
    ControlDecision as BControlDecision,
    ExecutorStatus as BExecutorStatus,
    Pose as BPose,
    RGBDFrame as BRGBDFrame,
    RobotState as BRobotState,
    RouteBController,
    SceneObject as BSceneObject,
    SensorObservation as BSensorObservation,
    SkillKind as BSkillKind,
    SkillStep as BSkillStep,
    TaskSpec as BTaskSpec,
    route_b_execution_issue,
)
from libero_system.route_c import (
    AtomicGoal,
    AtomicGoalKind,
    BoxSDF,
    BoundConstraintGraph,
    CompositeSDF,
    ContactTargetEstimate,
    ControllerFeedback,
    EntityGraspModeSelector,
    EntityRef,
    EntityResolver,
    EmptySDF,
    GraspCandidate,
    GeometryRelationVerifier,
    GoalSynthesizer,
    GraspMode,
    GraspModeSelector,
    MotionRequest,
    Phase,
    Relation,
    RouteCController,
    RouteCControllerConfig,
    RouteCResult,
    SceneEstimate,
    same_sensor_capture,
    sensor_capture_advanced,
    SelectorKind,
    SpatialSelector,
    TaskPlan,
)
from libero_system.route_c.controller import (
    ExecutionError,
    TerminalExecutionError,
    Verification,
)
from libero_system.route_c.grasp import GraspBindingError
from libero_system.route_c.optimizer import OptimisationError
from libero_system.common.pan_affordance import (
    PanAffordanceError,
    PanHandleAffordance,
    infer_pan_handle_affordance,
)
from libero_system.route_c.perception import (
    PerceptionError,
    SceneEntity,
    SupportRelationEvidence,
)


# One sensor-safe motion tolerance is shared by the low-level public-proprio
# sampler and every controller-side proof that consumes those samples.  Keeping
# this outside either class prevents an 8-mm executor allowance from silently
# weakening the Route-C three-frame 6-mm evidence gate.
_FREE_RIM_SENSOR_SAFE_MAX_VERTICAL_DRIFT_M = 0.006


def to_route_b_observation(observation: RobotObservation) -> BSensorObservation:
    """Copy only the public RGB-D/calibration/proprioception whitelist."""

    cameras: dict[str, BRGBDFrame] = {}
    for name, frame in observation.cameras.items():
        calibration = frame.calibration
        cameras[name] = BRGBDFrame(
            rgb=np.asarray(frame.rgb),
            depth_m=np.asarray(frame.depth_m, dtype=np.float64),
            intrinsics=BCameraIntrinsics(
                fx=calibration.fx,
                fy=calibration.fy,
                cx=calibration.cx,
                cy=calibration.cy,
                width=calibration.width,
                height=calibration.height,
            ),
            world_from_camera=BPose.from_matrix(calibration.T_world_camera),
            observation_v_flipped=calibration.observation_v_flipped,
        )
    return BSensorObservation(
        cameras=cameras,
        robot=BRobotState(
            ee_pose=BPose.from_matrix(observation.proprio.T_world_ee),
            gripper_width_m=observation.proprio.gripper_width_m,
            joint_position=observation.proprio.joint_position,
        ),
    )


class RouteBMicrowaveDoorDetector:
    """Ground a microwave door contact from frozen DINO plus current RGB-D.

    DINO supplies only the appliance body OBB.  The compact handle / open-door
    edge is then measured by :class:`MicrowaveDoorHandleDetector` from the
    current calibrated RGB-D views.  Keeping those roles separate avoids
    treating a high-scoring body crop (or a crop merely labelled "handle") as
    the physical contact point.
    """

    def __init__(
        self,
        perception: Any,
        *,
        handle_detector: MicrowaveDoorHandleDetector | Any | None = None,
    ) -> None:
        self.perception = perception
        self.handle_detector = handle_detector or MicrowaveDoorHandleDetector()
        self._fixture_geometry: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._fixture_surface_points_world: np.ndarray | None = None
        self._reference_ee_position_world: np.ndarray | None = None
        self._selected_closed_slot_world: np.ndarray | None = None
        self._wall_articulation = None
        self.last_body_selection: dict[str, Any] | None = None
        self.last_contact_detection: dict[str, Any] | None = None
        self.last_contact_track: dict[str, Any] | None = None
        self.last_articulation_selection: dict[str, Any] | None = None

    def reset(self) -> None:
        """Discard the complete frozen microwave binding for one episode."""

        self._fixture_geometry = None
        self._fixture_surface_points_world = None
        self._reference_ee_position_world = None
        self._selected_closed_slot_world = None
        self._wall_articulation = None
        self.last_body_selection = None
        self.last_contact_detection = None
        self.last_contact_track = None
        self.last_articulation_selection = None
        reset_handle = getattr(type(self.handle_detector), "reset", None)
        if callable(reset_handle):
            reset_handle(self.handle_detector)
        else:
            # Preserve compatibility with injected test detectors while the
            # exact production detector always takes the explicit API above.
            if hasattr(self.handle_detector, "last_detection_trace"):
                self.handle_detector.last_detection_trace = None
            if hasattr(self.handle_detector, "last_articulation_trace"):
                self.handle_detector.last_articulation_trace = None

    def detect(
        self,
        observation: RobotObservation,
        kind: GoalSkillKind,
    ) -> ContactTarget:
        converted = to_route_b_observation(observation)
        # A new language action starts a new immutable fixture/slot binding.
        # Never let a prior episode's verified slot enter this detection.
        self._fixture_surface_points_world = None
        self._selected_closed_slot_world = None
        self._wall_articulation = None
        center, axes, half_extents = self._select_fixture_geometry(converted)
        self._fixture_geometry = (
            center.copy(),
            axes.copy(),
            half_extents.copy(),
        )
        self._reference_ee_position_world = (
            observation.proprio.ee_position_world.copy()
        )
        local_kwargs = {}
        if kind is GoalSkillKind.CLOSE_MICROWAVE and self._fixture_surface_points_world is not None:
            from ..route_b.microwave_interior import microwave_cavity_from_walls

            try:
                frame = MicrowaveDoorHandleDetector._fixture_frame(
                    self._reference_ee_position_world, center, axes, half_extents)
                cavity, fit = microwave_cavity_from_walls(self._fixture_surface_points_world, frame)
            except (LookupError, ValueError):
                pass
            else:
                control_side = np.asarray(fit["control_side_world"])
                outward = np.asarray(fit["outward_world"])
                hinge = cavity.centroid_world - .1325*control_side + .084*outward
                closed_slot = hinge + .2375*control_side + .054*outward
                axis = np.array((0., 0., 1.))
                self._wall_articulation = (hinge, axis, closed_slot)
                # Use the measured wall frame for image-edge gating too.
                # A completed OBB from the partly visible roof may be tilted
                # or centred above the physical handle's vertical extent.
                center = cavity.centroid_world + .040*control_side - .004*outward
                center[2] -= .0025
                axes = np.column_stack((control_side, -outward, axis))
                half_extents = np.array((.1725, .1105, .0935))
                self._fixture_geometry = (center.copy(), axes.copy(), half_extents.copy())
                local_kwargs = dict(local_anchor_world=hinge, local_anchor_radius_m=.285,
                                    frozen_hinge_world=hinge, frozen_rotation_axis_world=axis,
                                    frozen_radius_m=float(np.hypot(.2375, .054)),
                                    frozen_radius_tolerance_m=.030)
        target = self.handle_detector.detect(
            observation,
            center,
            axes,
            half_extents,
            kind is GoalSkillKind.CLOSE_MICROWAVE,
            reference_ee_position_world=self._reference_ee_position_world,
            **local_kwargs,
        )
        if self._wall_articulation is not None:
            hinge, axis, closed_slot = self._wall_articulation
            radial = target.point_world - hinge
            radial -= axis * float(radial @ axis)
            # The detected vertical feature provides angle. Centre the pads
            # on the public handle capsule's height and radius, avoiding a
            # grasp on the lower attachment or door contour.
            point = hinge + radial / np.linalg.norm(radial) * np.hypot(.2375, .054)
            from scipy.spatial.transform import Rotation

            closed_radial = closed_slot - hinge
            angle = np.arctan2(float(axis @ np.cross(closed_radial, radial)),
                               float(closed_radial @ radial))
            normal = Rotation.from_rotvec(axis * angle).apply(outward)
            if normal @ (observation.proprio.ee_position_world - point) < 0:
                normal *= -1
            target = replace(target, point_world=point, axis_world=normal, outward_world=normal)
        self.last_contact_detection = self._contact_trace(
            "route_b_microwave_contact_detection",
            target,
            requested_kind=kind,
            observed_state="open" if kind is GoalSkillKind.CLOSE_MICROWAVE else "closed",
        )
        feature_trace = getattr(
            self.handle_detector,
            "last_detection_trace",
            None,
        )
        if isinstance(feature_trace, dict):
            self.last_contact_detection["sensor_feature_selection"] = {
                key: feature_trace[key]
                for key in (
                    "selection_mode",
                    "selected_point_world_m",
                    "selected_handle_side_coordinate_m",
                )
                if key in feature_trace
            }
        self._append_selector_diagnostic(self.last_contact_detection)
        return replace(target, kind=kind)

    def panel_geometry(self, observation: RobotObservation):
        """Bind an appliance from RGB-D, then fit its movable door panel."""
        from ..route_b.microwave_interior import microwave_cavity_from_walls, observed_door_panel

        converted = to_route_b_observation(observation)
        center, axes, half = self._select_fixture_geometry(converted)
        frame = MicrowaveDoorHandleDetector._fixture_frame(
            observation.proprio.ee_position_world, center, axes, half)
        cavity, wall_trace = microwave_cavity_from_walls(self._fixture_surface_points_world, frame)
        side = np.asarray(wall_trace['control_side_world'])
        outward = np.asarray(wall_trace['outward_world'])
        hinge = cavity.centroid_world-.1325*side+.084*outward
        angle, radial, normal, trace = observed_door_panel(observation, hinge, side, outward, open_only=True)
        self._append_selector_diagnostic({'kind': 'microwave_panel_initial_fit', **trace,
            'hinge_world_m': hinge.tolist(), 'control_side_world': side.tolist(),
            'outward_world': outward.tolist()})
        return hinge, side, outward, angle, radial, normal

    def articulation_geometry(
        self,
        reference_ee_position_world: np.ndarray,
        fixture_center_world: np.ndarray,
        fixture_axes_world: np.ndarray,
        fixture_half_extents_m: np.ndarray,
        observed_handle_world: np.ndarray | None = None,
        *,
        initial_is_open: bool | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Delegate sensor hinge selection while preserving Route-B trace."""

        selection: dict[str, Any] | None = None
        surface_points = self._fixture_surface_points_world
        if self._wall_articulation is not None and initial_is_open is True:
            values = tuple(value.copy() for value in self._wall_articulation)
            selection = {"selection_mode": "rgbd_control_corner_public_hinge_and_handle_offsets"}
        elif (
            initial_is_open is True
            and surface_points is not None
            and observed_handle_world is not None
        ):
            # This is the same pure RGB-D/public-asset fit used by Route C,
            # called without sharing any provider state.  Route B freezes its
            # own selected body cloud and selected closed slot below.
            hinge, axis, closed_slot, surface_trace = (
                LiberoRouteCContactTargetProvider._surface_refined_microwave_articulation(
                    reference_ee_position_world,
                    fixture_center_world,
                    fixture_axes_world,
                    fixture_half_extents_m,
                    surface_points,
                    np.asarray(observed_handle_world, dtype=np.float64),
                    initial_is_open=True,
                )
            )
            values = (hinge, axis, closed_slot)
            selection = {
                "selection_mode": (
                    "open_frozen_rgbd_surface_plus_public_asset_dimensions"
                ),
                "surface_fit": surface_trace,
            }
        else:
            geometry = getattr(self.handle_detector, "articulation_geometry", None)
            if not callable(geometry):
                raise LookupError(
                    "microwave handle detector exposes no articulation geometry"
                )
            values = geometry(
                reference_ee_position_world,
                fixture_center_world,
                fixture_axes_world,
                fixture_half_extents_m,
                observed_handle_world=observed_handle_world,
                initial_is_open=initial_is_open,
            )
            candidate_selection = getattr(
                self.handle_detector,
                "last_articulation_trace",
                None,
            )
            if isinstance(candidate_selection, dict):
                selection = candidate_selection
        self._selected_closed_slot_world = np.asarray(
            values[2], dtype=np.float64
        ).copy()
        trace: dict[str, Any] = {
            "kind": "route_b_microwave_hinge_hypotheses",
            "initial_door_state": (
                "open" if initial_is_open else "closed"
            ),
        }
        if selection is not None:
            trace["sensor_selection"] = selection
        trace["selected_closed_slot_world_m"] = (
            self._selected_closed_slot_world.tolist()
        )
        self.last_articulation_selection = trace
        self._append_selector_diagnostic(trace)
        return values

    def track(
        self,
        observation: RobotObservation,
        reference: ContactTarget,
        kind: GoalSkillKind,
    ) -> ContactTarget:
        if (
            self._fixture_geometry is None
            or self._reference_ee_position_world is None
        ):
            raise RuntimeError("microwave track requires an initial RGB-D detection")
        center, axes, half_extents = self._fixture_geometry
        # Verification observes the requested final door state: after OPEN an
        # open edge must be visible; after CLOSE the conservative closed-door
        # slot is allowed.  The body OBB and EE-based sign stay frozen across
        # arm motion so a newly occluded DINO crop cannot change identity.
        detect_kwargs: dict[str, Any] = {
            "reference_ee_position_world": self._reference_ee_position_world,
        }
        if (
            kind is GoalSkillKind.CLOSE_MICROWAVE
            and self._selected_closed_slot_world is not None
        ):
            detect_kwargs["expected_closed_slot_world"] = (
                self._selected_closed_slot_world
            )
        refreshed = self.handle_detector.detect(
            observation,
            center,
            axes,
            half_extents,
            kind is GoalSkillKind.OPEN_MICROWAVE,
            **detect_kwargs,
        )
        if float(np.linalg.norm(refreshed.point_world - reference.point_world)) > 0.35:
            raise LookupError("microwave visual track left its episode anchor")
        self.last_contact_track = self._contact_trace(
            "route_b_microwave_contact_track",
            refreshed,
            requested_kind=kind,
            observed_state="open" if kind is GoalSkillKind.OPEN_MICROWAVE else "closed",
        )
        detection_trace = getattr(
            self.handle_detector,
            "last_detection_trace",
            None,
        )
        if isinstance(detection_trace, dict):
            self.last_contact_track["sensor_feature_selection"] = dict(
                detection_trace
            )
        if self._selected_closed_slot_world is not None:
            self.last_contact_track["frozen_closed_slot_world_m"] = (
                self._selected_closed_slot_world.tolist()
            )
        self._append_selector_diagnostic(self.last_contact_track)
        return replace(refreshed, kind=kind)

    @staticmethod
    def _contact_trace(
        trace_kind: str,
        target: ContactTarget,
        *,
        requested_kind: GoalSkillKind,
        observed_state: str,
    ) -> dict[str, Any]:
        return {
            "kind": trace_kind,
            "requested_kind": requested_kind.value,
            "observed_state": observed_state,
            "point_world_m": target.point_world.tolist(),
            "outward_world": target.outward_world.tolist(),
            "fixture_center_world_m": target.fixture_center_world.tolist(),
            "confidence": float(target.confidence),
            "source_cameras": list(target.source_cameras),
        }

    def _append_selector_diagnostic(self, trace: dict[str, Any]) -> None:
        diagnostics = getattr(self.perception, "_selector_diagnostics", None)
        if isinstance(diagnostics, list):
            diagnostics.append(trace)

    def _select_fixture_geometry(
        self,
        observation: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        candidate_api = getattr(self.perception, "_candidates", None)
        candidates: list[Any] = []
        if callable(candidate_api):
            _, by_query = candidate_api(observation, ("microwave",))
            candidates.extend(by_query.get("microwave", ()))
        else:
            snapshot = self.perception.observe(observation, ("microwave",))
            fixture = snapshot.objects.get("microwave")
            if fixture is not None:
                candidates.append(fixture)

        plausible = [
            item for item in candidates if self._plausible_microwave_body(item)
        ]
        if not plausible:
            raise LookupError("no plausible microwave body OBB was visible in either RGB-D view")
        # A completed language box can be a bare table crop with the public
        # appliance dimensions attached. Prefer candidates whose actual RGB-D
        # surface supports the broad roof, before applying the old tie-break.
        from ..route_b.microwave_interior import microwave_cavity_from_walls

        supported = []
        for candidate in plausible:
            points = getattr(candidate, "surface_points_world", None)
            surface_api = getattr(self.perception, "_fresh_surface_points", None)
            if callable(surface_api):
                points = surface_api(candidate, "microwave")
            if points is None:
                continue
            try:
                frame = MicrowaveDoorHandleDetector._fixture_frame(
                    observation.robot.ee_pose.position, self._body_center(candidate),
                    self._body_axes(candidate), self._body_extents(candidate) / 2,
                )
                microwave_cavity_from_walls(points, frame)
            except (LookupError, ValueError):
                continue
            supported.append(candidate)
        if supported:
            plausible = supported
        body = min(
            plausible,
            key=lambda item: (
                float(self._body_center(item)[2]),
                -float(np.prod(self._body_extents(item))),
                -float(getattr(item, "confidence", 0.0)),
                str(getattr(item, "instance_id", "")),
            ),
        )
        center = self._body_center(body)
        axes = self._body_axes(body)
        extents = self._body_extents(body)
        surface_points = getattr(body, "surface_points_world", None)
        surface_api = getattr(self.perception, "_fresh_surface_points", None)
        if callable(surface_api):
            candidate_points = surface_api(body, "microwave")
            if candidate_points is not None:
                surface_points = candidate_points
        if surface_points is not None:
            points = np.asarray(surface_points, dtype=np.float64)
            if (
                points.ndim == 2
                and points.shape[1:] == (3,)
                and len(points) >= 3
                and np.all(np.isfinite(points))
            ):
                self._fixture_surface_points_world = points.copy()
        trace = {
            "kind": "route_b_microwave_body_selection",
            "candidates": [
                {
                    "instance_id": str(getattr(item, "instance_id", "")),
                    "center_world_m": self._body_center(item).tolist(),
                    "extent_m": self._body_extents(item).tolist(),
                    "confidence": float(getattr(item, "confidence", 0.0)),
                }
                for item in plausible
            ],
            "selected_instance_id": str(getattr(body, "instance_id", "")),
            "selected_center_world_m": center.tolist(),
            "selected_surface_point_count": (
                0
                if self._fixture_surface_points_world is None
                else int(len(self._fixture_surface_points_world))
            ),
            "selected_surface_points_frozen_with_obb": bool(
                self._fixture_surface_points_world is not None
            ),
        }
        self.last_body_selection = trace
        self._append_selector_diagnostic(trace)
        return center, axes, extents / 2.0

    @classmethod
    def _plausible_microwave_body(cls, body: Any) -> bool:
        axes = cls._body_axes(body)
        full = cls._body_extents(body)
        if axes.shape != (3, 3) or full.shape != (3,):
            return False
        if not np.all(np.isfinite(axes)) or not np.all(np.isfinite(full)):
            return False
        vertical_index = int(np.argmax(np.abs(axes[2, :])))
        if abs(float(axes[2, vertical_index])) < 0.70:
            return False
        horizontal = [full[index] for index in range(3) if index != vertical_index]
        short, long = sorted(float(value) for value in horizontal)
        vertical = float(full[vertical_index])
        return bool(
            0.12 <= short <= 0.45
            and 0.20 <= long <= 0.70
            and 0.10 <= vertical <= 0.45
        )

    @staticmethod
    def _body_center(body: Any) -> np.ndarray:
        value = getattr(body, "center_world", None)
        if value is None:
            value = body.centroid_world
        return np.asarray(value, dtype=np.float64)

    @staticmethod
    def _body_axes(body: Any) -> np.ndarray:
        return np.asarray(body.axes_world, dtype=np.float64)

    @staticmethod
    def _body_extents(body: Any) -> np.ndarray:
        value = getattr(body, "extents_m", None)
        if value is None:
            value = body.bounds_max_world - body.bounds_min_world
        return np.asarray(value, dtype=np.float64)


@dataclass(frozen=True)
class _RouteBSegment:
    start: int
    steps: tuple[BSkillStep, ...]
    mode: str


class RouteBPolicy:
    """Sensor-only sequential dispatcher for the complete Route-B IR.

    Consecutive grasp/place steps retain the original Route-B visual servo.
    Atomic drawer/microwave, bidirectional stove-knob, and planar-push steps
    use the contact controller.  The complete IR is capability-checked before
    either delegate moves; formed-stack continuations stay in one manipulation
    segment and must pass Route B's stack, lift, and final-placement RGB-D
    gates.
    """

    _CONTACT_KINDS = frozenset(
        {
            BSkillKind.OPEN,
            BSkillKind.CLOSE,
            BSkillKind.TURN_ON,
            BSkillKind.TURN_OFF,
            BSkillKind.PUSH_TO,
        }
    )

    def __init__(
        self,
        controller: RouteBController,
        *,
        contact_policy: GoalContactPolicy | Any | None = None,
    ) -> None:
        self.controller = controller
        self.contact_policy = contact_policy or GoalContactPolicy(
            microwave_detector=RouteBMicrowaveDoorDetector(controller.perception)
        )
        self.last_decision: BControlDecision | None = None
        self._legacy_direct = not hasattr(controller, "compiler")
        self._spec: BTaskSpec | None = None
        self._segments: tuple[_RouteBSegment, ...] = ()
        self._segment_index = 0
        self._episode_id = ""
        self._execution_issue: str | None = None
        self._drawer_anchors: dict[str, DrawerEpisodeAnchor] = {}
        self._archived_segment_index: int | None = None
        self._grasp_target_archive: list[dict[str, Any]] = []
        self._grasp_verification_archive: list[dict[str, Any]] = []
        self._grasp_event_archive: list[GraspAttemptEvent] = []
        self._placement_target_archive: list[dict[str, Any]] = []
        self._selector_diagnostic_archive: list[dict[str, Any]] = []
        self._selector_diagnostic_cursor = 0
        self._drawer_contact_proof: dict[str, Any] | None = None
        self._boundary_observation_sequence = 0
        self._boundary_observation_sequence_token: int | None = None
        # Freeze the first public proprioceptive pose at the policy boundary.
        # A preceding drawer/door skill can rotate the wrist before a later
        # independent grasp segment starts, so that live contact pose is not
        # a valid top-down grasp reference.  The controller receives this
        # episode pose through its public API after each manipulation reset.
        self._episode_reset_pose: BPose | None = None
        self._episode_reset_pose_segment_index: int | None = None
        self._episode_reset_call_counts: dict[str, int] = {}
        self._precontact_stove_geometry = None
        self._precontact_pick_sources: dict[int, BSceneObject] = {}

    @property
    def execution_issue(self) -> str | None:
        return self._execution_issue

    @property
    def observation_sequence_token(self) -> int | None:
        """Episode-local token used by the most recent outer ``act`` call.

        Route B can dispatch both manipulation and contact subcontrollers.
        Owning this sequence at their common wrapper keeps it monotonic across
        segment resets without adding a clock field to ``RobotObservation``.
        """

        return self._boundary_observation_sequence_token

    @property
    def task_spec(self) -> BTaskSpec | None:
        return self._spec

    @property
    def active_mode(self) -> str:
        if not self._segments or self._segment_index >= len(self._segments):
            return "done"
        return self._segments[self._segment_index].mode

    @property
    def grasp_target_attempts(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            (*self._grasp_target_archive, *self._live_controller_trace("grasp_target_attempts"))
        )

    @property
    def grasp_verifications(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            (*self._grasp_verification_archive, *self._live_controller_trace("grasp_verifications"))
        )

    @property
    def grasp_attempt_events(self) -> tuple[GraspAttemptEvent, ...]:
        """Episode-global mechanical engagements across Route-B segments."""

        if self._legacy_direct:
            return tuple(getattr(self.controller, "grasp_attempt_events", ()))
        return tuple((*self._grasp_event_archive, *self._live_grasp_events()))

    @property
    def pending_grasp_engagement(self) -> PendingGraspEngagement | None:
        """Episode-indexed snapshot for post-control interruption auditing."""

        pending = getattr(self.controller, "pending_grasp_engagement", None)
        if pending is None:
            return None
        if self._legacy_direct:
            return pending
        if (
            not self._segments
            or self._segment_index >= len(self._segments)
            or self._segments[self._segment_index].mode != "manipulation"
            or self._archived_segment_index == self._segment_index
        ):
            return None
        return replace(
            pending,
            attempt_index=len(self._grasp_event_archive) + pending.attempt_index,
        )

    @property
    def placement_target_attempts(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            (*self._placement_target_archive, *self._live_controller_trace("placement_target_attempts"))
        )

    @property
    def selector_diagnostics(self) -> tuple[dict[str, Any], ...]:
        """Episode-level selector evidence, including completed segments."""

        return tuple(
            (*self._selector_diagnostic_archive, *self._live_selector_diagnostics())
        )

    @property
    def drawer_contact_proof(self) -> dict[str, Any] | None:
        """Latest public-sensor drawer proof for the episode audit row."""

        if self._drawer_contact_proof is None:
            return None
        return dict(self._drawer_contact_proof)

    @property
    def episode_reset_call_counts(self) -> dict[str, int]:
        """Return the calls made by the most recent complete episode reset."""

        return dict(self._episode_reset_call_counts)

    def reset(self, task: PolicyTask) -> None:
        # PolicyTask contains language and an opaque episode id only.  No
        # benchmark identity or evaluator signal crosses this boundary.
        self.last_decision = None
        self._episode_id = task.episode_id
        self._segment_index = 0
        self._execution_issue = None
        self._drawer_anchors.clear()
        self._archived_segment_index = None
        self._grasp_target_archive.clear()
        self._grasp_verification_archive.clear()
        self._grasp_event_archive.clear()
        self._placement_target_archive.clear()
        self._selector_diagnostic_archive.clear()
        self._selector_diagnostic_cursor = 0
        self._drawer_contact_proof = None
        self._boundary_observation_sequence = 0
        self._boundary_observation_sequence_token = None
        self._episode_reset_pose = None
        self._episode_reset_pose_segment_index = None
        self._episode_reset_call_counts = {}
        self._precontact_stove_geometry = None
        self._precontact_pick_sources.clear()

        if self._legacy_direct:
            self.controller.reset(task.instruction)
            return

        formal_reset_graph = (
            type(self.controller) is RouteBController
            and type(self.contact_policy) is GoalContactPolicy
            and type(getattr(self.contact_policy, "microwave_detector", None))
            is RouteBMicrowaveDoorDetector
            and type(self.contact_policy.microwave_detector.handle_detector)
            is MicrowaveDoorHandleDetector
        )
        if not formal_reset_graph:
            reset_perception = getattr(self.controller.perception, "reset", None)
            if callable(reset_perception):
                reset_perception()
            self._spec = self.controller.compiler.compile(task.instruction)
            self._execution_issue = route_b_execution_issue(self._spec)
            if self._execution_issue is not None:
                self._segments = ()
                return
            self._segments = self._build_segments(self._spec)
            self._activate_segment()
            return

        reset_counts = {
            "route_b_policy": 1,
            "route_b_controller_clear": 0,
            "route_b_controller_activate": 0,
            "goal_contact_policy_clear": 0,
            "goal_contact_policy_activate": 0,
            "route_b_microwave_detector_reset": 0,
            "microwave_handle_detector_reset": 0,
            "route_b_perception_adapter_reset": 0,
        }
        type(self.controller).clear_episode_state(self.controller)
        reset_counts["route_b_controller_clear"] += 1
        reset_counts["route_b_perception_adapter_reset"] += 1
        type(self.contact_policy).clear_episode_state(self.contact_policy)
        reset_counts["goal_contact_policy_clear"] += 1
        reset_counts["route_b_microwave_detector_reset"] += 1
        reset_counts["microwave_handle_detector_reset"] += 1
        self._spec = type(self.controller.compiler).compile(
            self.controller.compiler,
            task.instruction,
        )
        self._execution_issue = route_b_execution_issue(self._spec)
        if self._execution_issue is not None:
            self._segments = ()
            self._episode_reset_call_counts = reset_counts
            return
        self._segments = RouteBPolicy._build_segments(self._spec)
        RouteBPolicy._activate_segment(
            self,
            episode_state_cleared=True,
            reset_counts=reset_counts,
        )
        self._episode_reset_call_counts = reset_counts

    def act(self, observation: RobotObservation) -> PolicyDecision:
        self._boundary_observation_sequence_token = self._boundary_observation_sequence
        self._boundary_observation_sequence += 1
        if self._legacy_direct:
            decision = type(self.controller).act(
                self.controller,
                to_route_b_observation(observation),
            )
            self.last_decision = decision
            terminal = decision.status in {
                BExecutorStatus.SUCCEEDED,
                BExecutorStatus.FAILED,
            }
            return self._common_decision(decision, terminal=terminal)

        self._freeze_episode_reset_pose(observation)
        if (
            self._boundary_observation_sequence == 1
            and self._spec is not None
            and self._spec.steps
            and self._spec.steps[0].kind in {BSkillKind.TURN_ON, BSkillKind.TURN_OFF}
            and any(step.kind is BSkillKind.PLACE_ON and step.target == "stove"
                    for step in self._spec.steps)
        ):
            # Turning the knob leaves the wrist in front of the stove. Freeze
            # the future support while it is still unobscured, before contact.
            observe = getattr(self.controller.perception, "observe", None)
            if callable(observe):
                snapshot = observe(to_route_b_observation(observation), ("stove",))
                self._precontact_stove_geometry = snapshot.objects.get("stove")

        if (self._boundary_observation_sequence == 1 and self._segments
                and self._segments[0].mode == "contact"):
            observe = getattr(self.controller.perception, "observe", None)
            following = next((item for item in self._segments if item.mode == "manipulation"), None)
            if callable(observe) and following is not None:
                first = following.steps[0]
                if first.kind is BSkillKind.PICK and first.selector is None:
                    # Remember the independent source before drawer/knob
                    # contact occludes it. The controller still requires a
                    # fresh nearby RGB-D observation after contact.
                    snapshot = observe(to_route_b_observation(observation), (first.subject,))
                    source = snapshot.objects.get(first.subject)
                    if source is not None:
                        self._precontact_pick_sources[following.start] = source

        if self._execution_issue is not None:
            decision = BControlDecision(
                OSCAction.hold(-1.0).values,
                BExecutorStatus.FAILED,
                0,
                "unsupported",
                self._execution_issue,
            )
            self.last_decision = decision
            return self._common_decision(decision, terminal=True)
        if self._segment_index >= len(self._segments):
            decision = BControlDecision(
                OSCAction.hold(-1.0).values,
                BExecutorStatus.SUCCEEDED,
                len(self._spec.steps) if self._spec is not None else 0,
                "done",
                "all sensor-only Route-B segments completed",
            )
            self.last_decision = decision
            return self._common_decision(decision, terminal=True)

        segment = self._segments[self._segment_index]
        if segment.mode == "manipulation":
            local = type(self.controller).act(
                self.controller,
                to_route_b_observation(observation),
            )
            global_index = segment.start + min(local.skill_index, len(segment.steps))
            if local.status is BExecutorStatus.FAILED:
                decision = BControlDecision(
                    local.action,
                    BExecutorStatus.FAILED,
                    global_index,
                    local.phase,
                    local.message,
                )
                self.last_decision = decision
                return self._common_decision(decision, terminal=True)
            if local.status is BExecutorStatus.SUCCEEDED:
                self._archive_controller_trace(segment)
                self._segment_index += 1
                if self._segment_index < len(self._segments):
                    RouteBPolicy._activate_segment(self)
                    status = BExecutorStatus.RUNNING
                    phase = "dispatch"
                    message = "advanced to the next sensor-only skill segment"
                else:
                    status = BExecutorStatus.SUCCEEDED
                    phase = "done"
                    message = "all sensor-only Route-B segments completed"
                decision = BControlDecision(
                    local.action,
                    status,
                    segment.start + len(segment.steps),
                    phase,
                    message,
                )
                self.last_decision = decision
                return self._common_decision(
                    decision,
                    terminal=status is BExecutorStatus.SUCCEEDED,
                )
            decision = BControlDecision(
                local.action,
                BExecutorStatus.RUNNING,
                global_index,
                local.phase,
                local.message,
            )
            self.last_decision = decision
            return self._common_decision(decision, terminal=False)

        contact_decision = type(self.contact_policy).act(
            self.contact_policy,
            observation,
        )
        raw_drawer_proof = contact_decision.diagnostics.get(
            "drawer_contact_proof"
        )
        if isinstance(raw_drawer_proof, Mapping):
            self._drawer_contact_proof = {
                **dict(raw_drawer_proof),
                "segment_start_skill_index": segment.start,
            }
        contact_status = self.contact_policy.status
        anchor = getattr(self.contact_policy, "drawer_anchor", None)
        if anchor is not None:
            self._drawer_anchors[anchor.level] = anchor
        if contact_status in {GoalExecutorStatus.FAILED, GoalExecutorStatus.HANDOFF}:
            message = str(contact_decision.diagnostics.get("message", "contact skill failed"))
            decision = BControlDecision(
                contact_decision.action.values,
                BExecutorStatus.FAILED,
                segment.start,
                str(contact_decision.diagnostics.get("phase", "contact_failed")),
                message,
            )
            self.last_decision = decision
            return self._common_decision(decision, terminal=True)
        if contact_status is GoalExecutorStatus.SUCCEEDED:
            self._archive_selector_diagnostics(segment)
            self._segment_index += 1
            if self._segment_index < len(self._segments):
                RouteBPolicy._activate_segment(self)
                status = BExecutorStatus.RUNNING
                phase = "dispatch"
                message = "advanced to the next sensor-only skill segment"
            else:
                status = BExecutorStatus.SUCCEEDED
                phase = "done"
                message = "all sensor-only Route-B segments completed"
            decision = BControlDecision(
                contact_decision.action.values,
                status,
                segment.start + 1,
                phase,
                message,
            )
            self.last_decision = decision
            return self._common_decision(
                decision,
                terminal=status is BExecutorStatus.SUCCEEDED,
            )

        decision = BControlDecision(
            contact_decision.action.values,
            BExecutorStatus.RUNNING,
            segment.start,
            str(contact_decision.diagnostics.get("phase", "contact")),
            str(contact_decision.diagnostics.get("message", "")),
        )
        self.last_decision = decision
        return self._common_decision(decision, terminal=False)

    @classmethod
    def _build_segments(cls, spec: BTaskSpec) -> tuple[_RouteBSegment, ...]:
        segments: list[_RouteBSegment] = []
        current_start = 0
        current: list[BSkillStep] = []
        for index, step in enumerate(spec.steps):
            if step.kind in cls._CONTACT_KINDS:
                if current:
                    segments.append(
                        _RouteBSegment(current_start, tuple(current), "manipulation")
                    )
                    current = []
                segments.append(_RouteBSegment(index, (step,), "contact"))
            else:
                if not current:
                    current_start = index
                current.append(step)
        if current:
            segments.append(_RouteBSegment(current_start, tuple(current), "manipulation"))
        return tuple(segments)

    def _activate_segment(
        self,
        *,
        episode_state_cleared: bool = False,
        reset_counts: dict[str, int] | None = None,
    ) -> None:
        segment = self._segments[self._segment_index]
        if segment.mode == "manipulation":
            assert self._spec is not None
            task = BTaskSpec(self._spec.instruction, segment.steps)
            if episode_state_cleared:
                type(self.controller).activate_after_episode_clear(
                    self.controller,
                    task,
                )
                if reset_counts is not None:
                    reset_counts["route_b_controller_activate"] += 1
            else:
                type(self.controller).reset(self.controller, task)
            self._inject_episode_reset_pose()
            # Opening already established this handle's sensor identity.
            # Carry that reference into manipulation so occlusion cannot
            # relabel its current visible height rank during placement.
            self.controller._drawer_episode_anchors = dict(self._drawer_anchors)
            source = self._precontact_pick_sources.get(segment.start)
            if source is not None:
                self.controller._prefetched_pick_sources[0] = source
                self.controller._optional_pick_source_anchors.add(0)
            if self._precontact_stove_geometry is not None:
                for index, step in enumerate(segment.steps):
                    if step.kind is BSkillKind.PLACE_ON and step.target == "stove":
                        self.controller._prefetched_place_destinations[index] = (
                            "stove", step.target_selector, step.relation,
                            self._precontact_stove_geometry, "precontact_rgbd_stove",
                        )
            return
        step = segment.steps[0]
        instruction = RouteBPolicy._contact_instruction(step)
        contact_task = PolicyTask(instruction, self._episode_id)
        if episode_state_cleared:
            type(self.contact_policy).activate_after_episode_clear(
                self.contact_policy,
                contact_task,
            )
            if reset_counts is not None:
                reset_counts["goal_contact_policy_activate"] += 1
        else:
            type(self.contact_policy).reset(self.contact_policy, contact_task)
        level = step.subject.split(" ", maxsplit=1)[0]
        anchor = self._drawer_anchors.get(level)
        seed = getattr(self.contact_policy, "seed_drawer_anchor", None)
        if anchor is not None and callable(seed):
            seed(anchor)

    def _freeze_episode_reset_pose(self, observation: RobotObservation) -> None:
        """Capture and inject only the episode's first public EE pose."""

        if self._episode_reset_pose is None:
            self._episode_reset_pose = BPose.from_matrix(
                observation.proprio.T_world_ee
            )
        self._inject_episode_reset_pose()

    def _inject_episode_reset_pose(self) -> None:
        if (
            self._episode_reset_pose is None
            or not self._segments
            or self._segment_index >= len(self._segments)
            or self._segments[self._segment_index].mode != "manipulation"
            or self._episode_reset_pose_segment_index == self._segment_index
        ):
            return
        self.controller.freeze_episode_reset_pose(self._episode_reset_pose)
        self._episode_reset_pose_segment_index = self._segment_index

    @staticmethod
    def _contact_instruction(step: BSkillStep) -> str:
        if step.kind is BSkillKind.OPEN:
            return f"open the {step.subject}"
        if step.kind is BSkillKind.CLOSE:
            return f"close the {step.subject}"
        if step.kind is BSkillKind.TURN_ON:
            return "turn on the stove"
        if step.kind is BSkillKind.TURN_OFF:
            return "turn off the stove"
        if step.kind is BSkillKind.PUSH_TO:
            return "push the plate to the front of the stove"
        raise ValueError(f"unsupported contact dispatcher step: {step.kind.value}")

    def _archive_controller_trace(self, segment: _RouteBSegment) -> None:
        if self._archived_segment_index == self._segment_index:
            return
        for destination, attribute in (
            (self._grasp_target_archive, "grasp_target_attempts"),
            (self._grasp_verification_archive, "grasp_verifications"),
            (self._placement_target_archive, "placement_target_attempts"),
        ):
            for item in getattr(self.controller, attribute, ()):
                row = dict(item)
                row["segment_start_skill_index"] = segment.start
                destination.append(row)
        for item in getattr(self.controller, "grasp_attempt_events", ()):
            self._grasp_event_archive.append(
                replace(item, attempt_index=len(self._grasp_event_archive) + 1)
            )
        self._archive_selector_diagnostics(segment)
        self._archived_segment_index = self._segment_index

    def _archive_selector_diagnostics(self, segment: _RouteBSegment) -> None:
        """Archive new perception evidence for either dispatcher delegate."""

        items = tuple(
            getattr(self.controller.perception, "selector_diagnostics", ())
        )
        for item in items[self._selector_diagnostic_cursor :]:
            row = dict(item)
            row["segment_start_skill_index"] = segment.start
            self._selector_diagnostic_archive.append(row)
        self._selector_diagnostic_cursor = len(items)

    def _live_controller_trace(self, attribute: str) -> tuple[dict[str, Any], ...]:
        if (
            not self._segments
            or self._segment_index >= len(self._segments)
            or self._segments[self._segment_index].mode != "manipulation"
            or self._archived_segment_index == self._segment_index
        ):
            return ()
        start = self._segments[self._segment_index].start
        return tuple(
            {**dict(item), "segment_start_skill_index": start}
            for item in getattr(self.controller, attribute, ())
        )

    def _live_grasp_events(self) -> tuple[GraspAttemptEvent, ...]:
        if (
            not self._segments
            or self._segment_index >= len(self._segments)
            or self._segments[self._segment_index].mode != "manipulation"
            or self._archived_segment_index == self._segment_index
        ):
            return ()
        offset = len(self._grasp_event_archive)
        return tuple(
            replace(item, attempt_index=offset + local_index)
            for local_index, item in enumerate(
                getattr(self.controller, "grasp_attempt_events", ()), start=1
            )
        )

    def _live_selector_diagnostics(self) -> tuple[dict[str, Any], ...]:
        items = tuple(
            getattr(self.controller.perception, "selector_diagnostics", ())
        )
        if self._legacy_direct:
            return tuple(dict(item) for item in items)
        if (
            not self._segments
            or self._segment_index >= len(self._segments)
        ):
            return ()
        start = self._segments[self._segment_index].start
        return tuple(
            {**dict(item), "segment_start_skill_index": start}
            for item in items[self._selector_diagnostic_cursor :]
        )

    @staticmethod
    def _common_decision(
        decision: BControlDecision,
        *,
        terminal: bool,
    ) -> PolicyDecision:
        return PolicyDecision(
            OSCAction.from_array(decision.action),
            request_stop=terminal,
            diagnostics={
                "route": "b",
                "status": decision.status.value,
                "skill_index": decision.skill_index,
                "phase": decision.phase,
                "message": decision.message,
            },
        )

    def close(self) -> None:
        close = getattr(type(self.controller.perception), "close", None)
        if callable(close):
            close(self.controller.perception)
        contact_close = getattr(type(self.contact_policy), "close", None)
        if callable(contact_close):
            contact_close(self.contact_policy)


def _support_refinement_residuals(
    scene: SceneEstimate,
    evidence: SupportRelationEvidence,
    source: SceneEntity,
) -> tuple[float, float] | None:
    """Validate one measured support point against its re-anchored source."""

    support = np.asarray(evidence.support_point_world, dtype=np.float64)
    if (
        support.shape != (3,)
        or not np.all(np.isfinite(support))
        or np.any(support < scene.workspace_min)
        or np.any(support > scene.workspace_max)
    ):
        return None
    world_span = np.abs(source.pose[:3, :3]) @ source.extent
    source_half_z = 0.5 * float(world_span[2])
    source_bottom_z = float(source.position[2] - source_half_z)
    vertical_tolerance = float(
        np.clip(0.10 * world_span[2], 0.0025, 0.0050)
    )
    vertical_residual = abs(source_bottom_z - float(support[2]))
    planar_residual = float(np.linalg.norm(source.position[:2] - support[:2]))
    planar_tolerance = float(
        np.clip(0.10 * np.max(world_span[:2]), 0.005, 0.015)
    )
    if (
        support[2] >= source.position[2]
        or support[2] <= float(scene.scene_floor_z) + 0.001
        or vertical_residual > vertical_tolerance
        or planar_residual > planar_tolerance
    ):
        return None
    bottom_keypoint = source.keypoints.get("bottom")
    if bottom_keypoint is not None:
        bottom = np.asarray(bottom_keypoint, dtype=np.float64)
        if (
            bottom.shape != (3,)
            or abs(float(bottom[2] - support[2])) > vertical_tolerance
            or float(np.linalg.norm(bottom[:2] - support[:2]))
            > planar_tolerance
        ):
            return None
    return vertical_residual, planar_residual


class WorkspaceCenterEntityResolver(EntityResolver):
    """Resolve selectors against measured geometry, including singletons.

    The base resolver can return the only same-label candidate before applying
    its selector.  In a multi-bowl Spatial scene that silently turns "in the
    top drawer" into "any detected black bowl".  Route C instead validates
    ON/IN candidates against the RGB-D anchor region even when only one source
    proposal survived grounding.
    """

    def __init__(
        self,
        front_axis: int = 1,
        *,
        selector_rejection_sink: Callable[[], None] | None = None,
        visible_instance_ids_provider: Callable[[], Collection[str]] | None = None,
    ) -> None:
        super().__init__(front_axis=front_axis)
        self.selector_rejection_sink = selector_rejection_sink
        self.visible_instance_ids_provider = visible_instance_ids_provider
        self._required_source_id: str | None = None
        self._required_target_id: str | None = None
        self.last_resolution_trace: dict[str, Any] = {}
        self._resolution_history: list[dict[str, Any]] = []

    @property
    def resolution_history(self) -> tuple[dict[str, Any], ...]:
        """Append-only support-selector audit for the current episode."""

        return tuple(dict(item) for item in self._resolution_history)

    def reset_resolution_history(self) -> None:
        """Start a new episode-local selector audit."""

        self.last_resolution_trace = {}
        self._resolution_history.clear()

    def _record_resolution_trace(self, trace: Mapping[str, Any]) -> None:
        recorded = dict(trace)
        self.last_resolution_trace = recorded
        self._resolution_history.append(dict(recorded))

    def require_source_identity(self, instance_id: str) -> None:
        """Pin the next compound-goal pick to one sensor track."""

        if not instance_id:
            raise ValueError("required source identity cannot be empty")
        self._required_source_id = str(instance_id)

    def clear_required_source_identity(self) -> None:
        self._required_source_id = None

    def require_target_identity(self, instance_id: str) -> None:
        """Pin a ranked destination before an earlier goal can occlude it."""

        if not instance_id:
            raise ValueError("required target identity cannot be empty")
        self._required_target_id = str(instance_id)

    def clear_required_target_identity(self) -> None:
        self._required_target_id = None

    def _reject_selector(self, detail: str) -> None:
        if self.selector_rejection_sink is not None:
            # A relation-invalid singleton must not be replayed from the
            # episode scene cache on the next task attempt.  This only drops
            # RGB-D-derived cache state; it does not expose evaluator truth.
            self.selector_rejection_sink()
        raise PerceptionError(detail)

    def _resolve_support_relation_evidence(
        self,
        reference: EntityRef,
        scene: SceneEstimate,
        candidates: Sequence[SceneEntity],
        fresh_visible_ids: Collection[str] | None,
    ) -> SceneEntity | None:
        """Resolve one ON selector from strict same-observation plane evidence.

        A low-profile support may be directly measured around the source while
        remaining unsafe to represent as a complete fixture OBB.  The evidence
        path therefore returns only an existing, freshly visible source.  It
        neither creates an anchor entity nor changes the scene obstacle field.
        """

        selector = reference.selector
        if (
            reference.role != "source"
            or selector is None
            or selector.relation is not Relation.ON
            or len(selector.references) != 1
            or fresh_visible_ids is None
        ):
            return None
        source_label = reference.label
        reference_label = selector.references[0]
        relevant = [
            item
            for item in scene.support_relation_evidence
            if item.relation is Relation.ON
            and item.source_label == source_label
            and item.reference_label == reference_label
        ]
        # Multiple independent claims are ambiguous even if they happen to
        # name the same track.  The producer must establish one unique source.
        if len(relevant) != 1:
            return None
        evidence = relevant[0]
        if not same_sensor_capture(evidence, scene):
            return None

        visible_ids = {str(instance_id) for instance_id in fresh_visible_ids}
        if evidence.source_instance_id not in visible_ids:
            return None
        matches = [
            candidate
            for candidate in candidates
            if candidate.instance_id == evidence.source_instance_id
            and candidate.label == source_label
        ]
        if len(matches) != 1:
            return None
        source = matches[0]

        residuals = _support_refinement_residuals(scene, evidence, source)
        if residuals is None:
            return None
        vertical_residual, planar_residual = residuals
        support = np.asarray(evidence.support_point_world, dtype=np.float64)

        self._record_resolution_trace({
            "relation": Relation.ON.value,
            "selection_mode": "measured_low_profile_support_evidence",
            "selected_source_id": source.instance_id,
            "selected_anchor_id": None,
            "anchor_geometry_created": False,
            "evidence_timestamp_s": float(evidence.timestamp_s),
            "evidence_capture_id": evidence.capture_id,
            "scene_capture_id": scene.capture_id,
            "support_point_world_m": [float(value) for value in support],
            "support_vertical_residual_m": vertical_residual,
            "support_planar_residual_m": planar_residual,
        })
        return source

    def resolve(self, reference, scene: SceneEstimate) -> SceneEntity:
        if reference.role == "source" and self._excluded_source_ids:
            scene = replace(
                scene,
                entities=tuple(
                    item
                    for item in scene.entities
                    if item.instance_id not in self._excluded_source_ids
                ),
            )
        required_id = (
            self._required_source_id
            if reference.role == "source"
            else self._required_target_id
            if reference.role == "target"
            else None
        )
        if required_id is not None:
            if self.visible_instance_ids_provider is not None:
                visible_ids = {
                    str(instance_id)
                    for instance_id in self.visible_instance_ids_provider()
                }
                if required_id not in visible_ids:
                    raise PerceptionError(
                        f"required {reference.role} identity {required_id!r} is not "
                        "freshly visible"
                    )
            matches = [
                entity
                for entity in scene.entities
                if entity.instance_id == required_id
                and (
                    entity.label == reference.label
                    or reference.label in entity.label
                    or entity.label in reference.label
                )
            ]
            if len(matches) != 1:
                raise PerceptionError(
                    f"required {reference.role} identity {required_id!r} does not "
                    f"uniquely match {reference.label!r}"
                )
            return matches[0]
        selector = reference.selector
        if selector is None:
            return super().resolve(reference, scene)
        if selector.relation in {
            Relation.LEFTMOST,
            Relation.RIGHTMOST,
            Relation.FRONTMOST,
            Relation.BACKMOST,
            Relation.MIDDLE,
            Relation.MIDDLE_PART,
            Relation.TOPMOST,
            Relation.BOTTOMMOST,
            Relation.TOP_PART,
            Relation.BOTTOM_PART,
            Relation.FIRST,
            Relation.SECOND,
        }:
            return super().resolve(reference, scene)
        exact = [entity for entity in scene.entities if entity.label == reference.label]
        candidates = exact or [
            entity
            for entity in scene.entities
            if reference.label in entity.label or entity.label in reference.label
        ]
        if not candidates:
            return super().resolve(reference, scene)
        fresh_visible_ids: frozenset[str] | None = None
        if (
            selector.relation in {Relation.ON, Relation.IN}
            and self.visible_instance_ids_provider is not None
        ):
            try:
                fresh_visible_ids = frozenset(
                    str(instance_id)
                    for instance_id in self.visible_instance_ids_provider()
                )
            except (TypeError, ValueError):
                self._record_resolution_trace({
                    "relation": selector.relation.value,
                    "selection_mode": "fresh_visibility_gate",
                    "selected_source_id": None,
                    "selected_anchor_id": None,
                })
                self._reject_selector(
                    "fresh visual identity set is unavailable for support selector"
                )
            candidates = [
                candidate
                for candidate in candidates
                if candidate.instance_id in fresh_visible_ids
            ]
            if not candidates:
                self._record_resolution_trace({
                    "relation": selector.relation.value,
                    "selection_mode": "fresh_visibility_gate",
                    "selected_source_id": None,
                    "selected_anchor_id": None,
                })
                self._reject_selector(
                    f"no freshly visible entity matches selector for "
                    f"{reference.label!r}"
                )
        if selector.relation != Relation.CENTER and selector.references:
            anchor_groups = [
                self._label_candidates(label, scene)
                for label in selector.references
            ]
            if fresh_visible_ids is not None:
                anchor_groups = [
                    [
                        anchor
                        for anchor in group
                        if anchor.instance_id in fresh_visible_ids
                    ]
                    for group in anchor_groups
                ]
            if any(not group for group in anchor_groups):
                missing = selector.references[
                    next(index for index, group in enumerate(anchor_groups) if not group)
                ]
                evidence_source = self._resolve_support_relation_evidence(
                    reference,
                    scene,
                    candidates,
                    fresh_visible_ids,
                )
                if evidence_source is not None:
                    return evidence_source
                self._reject_selector(
                    f"selector reference {missing!r} is not visible"
                )
            anchors = [anchor for group in anchor_groups for anchor in group]
            retained = [
                candidate
                for candidate in candidates
                if not any(
                    self._same_small_component(candidate, anchor)
                    for anchor in anchors
                    if candidate.label != anchor.label
                )
            ]
            if len(retained) != len(candidates):
                retained_identity = {id(entity) for entity in retained}
                candidate_identity = {id(entity) for entity in candidates}
                scene = replace(
                    scene,
                    entities=tuple(
                        entity
                        for entity in scene.entities
                        if id(entity) not in candidate_identity
                        or id(entity) in retained_identity
                    ),
                )
                candidates = retained
            if not candidates:
                self._reject_selector(
                    f"no distinct visual entity matches selector for {reference.label!r}"
                )
            if selector.relation in {Relation.ON, Relation.IN}:
                pairs = []
                trace_candidates = []
                for candidate in candidates:
                    for anchor in anchor_groups[0]:
                        valid, cost, vertical, planar, area = (
                            self._support_selector_metrics(
                                candidate, anchor, selector.relation
                            )
                        )
                        trace_candidates.append(
                            {
                                "source_id": candidate.instance_id,
                                "anchor_id": anchor.instance_id,
                                "valid": valid,
                                "cost": cost if valid else None,
                                "vertical_residual_m": vertical,
                                "normalized_planar_distance": planar,
                                "anchor_planar_area_m2": area,
                            }
                        )
                        if valid:
                            pairs.append(
                                (
                                    cost,
                                    area,
                                    -candidate.confidence,
                                    -anchor.confidence,
                                    candidate,
                                    anchor,
                                )
                            )
                if not pairs:
                    self._record_resolution_trace({
                        "relation": selector.relation.value,
                        "selection_mode": "measured_source_reference_pair",
                        "candidates": trace_candidates,
                        "selected_source_id": None,
                        "selected_anchor_id": None,
                    })
                    self._reject_selector(
                        f"no visual {reference.label!r} satisfies "
                        f"{selector.relation.value} {selector.references[0]!r}"
                    )
                selected = min(pairs, key=lambda item: item[:4])
                source, anchor = selected[4], selected[5]
                self._record_resolution_trace({
                    "relation": selector.relation.value,
                    "selection_mode": "measured_source_reference_pair",
                    "candidates": trace_candidates,
                    "selected_source_id": source.instance_id,
                    "selected_anchor_id": anchor.instance_id,
                    "selected_cost": selected[0],
                })
                return source
            # Other relative selectors keep the base resolver's ranking after
            # same-component candidates have been removed.  Preserve every
            # remaining source proposal so BETWEEN/NEXT_TO can compare them.
            candidate_ids = {id(entity) for entity in candidates}
            source_ids = {id(entity) for entity in exact}
            filtered_scene = replace(
                scene,
                entities=tuple(
                    entity
                    for entity in scene.entities
                    if id(entity) not in source_ids or id(entity) in candidate_ids
                ),
            )
            return super().resolve(reference, filtered_scene)
        workspace_center = (scene.workspace_min[:2] + scene.workspace_max[:2]) / 2.0
        return min(
            candidates,
            key=lambda entity: float(np.linalg.norm(entity.position[:2] - workspace_center)),
        )

    @staticmethod
    def _satisfies_support_selector(
        source: SceneEntity,
        anchor: SceneEntity,
        relation: Relation,
        *,
        planar_margin_m: float = 0.035,
        vertical_margin_m: float = 0.050,
    ) -> bool:
        return WorkspaceCenterEntityResolver._support_selector_metrics(
            source,
            anchor,
            relation,
            planar_margin_m=planar_margin_m,
            vertical_margin_m=vertical_margin_m,
        )[0]

    @staticmethod
    def _support_selector_metrics(
        source: SceneEntity,
        anchor: SceneEntity,
        relation: Relation,
        *,
        planar_margin_m: float = 0.035,
        vertical_margin_m: float = 0.050,
    ) -> tuple[bool, float, float, float, float]:
        if anchor.region is None:
            return False, float("inf"), float("inf"), float("inf"), float("inf")
        local = anchor.region.local_coordinates(source.position[None, :])[0]
        planar_limit = anchor.region.half_extents[:2] + planar_margin_m
        planar_valid = not np.any(np.abs(local[:2]) > planar_limit)
        normalized_planar = float(
            np.linalg.norm(
                local[:2]
                / np.maximum(anchor.region.half_extents[:2], 0.020)
            )
        )
        source_half_z = float(
            np.abs(source.pose[:3, :3][2]) @ (source.extent / 2.0)
        )
        anchor_half_z = float(
            np.abs(anchor.region.axes[2]) @ anchor.region.half_extents
        )
        if relation == Relation.IN:
            vertical_residual = max(
                abs(float(local[2]))
                + source_half_z
                - anchor_half_z
                - vertical_margin_m,
                0.0,
            )
            vertical_valid = vertical_residual <= 1e-12
        else:
            source_bottom = source.position[2] - source_half_z
            anchor_top = anchor.region.center[2] + anchor_half_z
            vertical_residual = abs(float(source_bottom - anchor_top))
            vertical_valid = vertical_residual <= 0.080
        area = float(np.prod(2.0 * anchor.region.half_extents[:2]))
        cost = vertical_residual + 0.020 * normalized_planar
        return (
            bool(planar_valid and vertical_valid),
            cost,
            vertical_residual,
            normalized_planar,
            area,
        )

    @staticmethod
    def _label_candidates(label: str, scene: SceneEstimate) -> list[SceneEntity]:
        return EntityResolver._label_candidates(label, scene)  # noqa: SLF001

    @staticmethod
    def _same_small_component(source: SceneEntity, reference: SceneEntity) -> bool:
        if max(source.extent[:2]) >= 0.15 or max(reference.extent[:2]) >= 0.15:
            return False
        if float(np.linalg.norm(source.position - reference.position)) > 0.025:
            return False
        source_extent = np.maximum(source.extent, 1e-6)
        reference_extent = np.maximum(reference.extent, 1e-6)
        extent_similarity = np.minimum(source_extent, reference_extent) / np.maximum(
            source_extent,
            reference_extent,
        )
        # Cross-query clones preserve one crop's centre and all three measured
        # dimensions.  A real bowl stacked on a partly visible cookie box can
        # share almost the same centre while its round footprint and height
        # differ strongly from the thin support strip; keep that physical pair
        # for the subsequent ON/IN geometry test.
        return bool(np.all(extent_similarity >= 0.65))


class LiberoRouteCContactTargetProvider:
    """Adapt calibrated dual-view detectors to Route C contact geometry.

    The provider reads only the current public observation and the stable
    RGB-D estimator.  Episode-local drawer and plate anchors preserve physical
    identity across sequential goals without fixture names or hidden state.
    """

    _DRAWER_LEVELS = {
        SelectorKind.TOP: "top",
        SelectorKind.MIDDLE: "middle",
        SelectorKind.BOTTOM: "bottom",
    }
    # A rotary target cannot be represented by a zero angle in the public
    # ContactTargetEstimate schema.  Keep a small non-zero continuation step
    # for a visually completed door, while still rejecting a materially
    # over-open or direction-inconsistent RGB-D estimate.
    _MICROWAVE_OPEN_MIN_REMAINING_RAD = 0.05
    _MICROWAVE_OPEN_PROGRESS_DIRECTION_TOLERANCE_RAD = 0.10
    _MICROWAVE_OPEN_OVERRUN_TOLERANCE_RAD = 0.12
    # Public geometry of the LIBERO microwave family.  These are fixed asset
    # dimensions, not episode state: the closed handle centre is 237.5 mm
    # across the door from its hinge and 54 mm in front of it.  RGB-D still
    # determines both the fixture frame and which physical end is hinged.
    _MICROWAVE_PUBLIC_HANDLE_LONG_OFFSET_M = 0.2375
    _MICROWAVE_PUBLIC_HANDLE_OUTWARD_OFFSET_M = 0.054
    _MICROWAVE_SURFACE_MIN_POINTS = 64
    _MICROWAVE_SURFACE_LONG_QUANTILES = (0.02, 0.98)
    _MICROWAVE_SURFACE_FRONT_QUANTILE = 0.90
    # The public collision model puts the closed door's outer planar face
    # about 25 mm in front of the hinge.  A q90 surface statistic lies a few
    # millimetres inside that face, so use a conservative 20-mm correction
    # while the door is closed.  With an already-open door, that panel has
    # rotated out of the appliance crop and the robust front quantile itself
    # is the best measured hinge-depth anchor.
    _MICROWAVE_PUBLIC_CLOSED_FRONT_HINGE_INSET_M = 0.020
    _MICROWAVE_PUBLIC_OPEN_FRONT_HINGE_INSET_M = 0.0

    def __init__(
        self,
        observation_provider: Callable[[], RobotObservation],
        observer: Any,
        *,
        drawer_detector: DrawerHandleDetector | None = None,
        knob_detector: StoveKnobDetector | None = None,
        plate_detector: PlateFrontDetector | None = None,
        microwave_detector: MicrowaveDoorHandleDetector | None = None,
        microwave_open_angle_rad: float = 1.55,
        microwave_close_overshoot_rad: float = 0.10,
    ) -> None:
        if not callable(observation_provider):
            raise TypeError("observation_provider must be callable")
        self._observation_provider = observation_provider
        self.observer = observer
        self.drawer_detector = drawer_detector or DrawerHandleDetector()
        self.knob_detector = knob_detector or StoveKnobDetector()
        self.plate_detector = plate_detector or PlateFrontDetector(
            self.knob_detector
        )
        self.microwave_detector = microwave_detector or MicrowaveDoorHandleDetector()
        if (
            not np.isfinite(microwave_open_angle_rad)
            or not 0.50 <= microwave_open_angle_rad <= 2.0
            or not np.isfinite(microwave_close_overshoot_rad)
            or not 0.0 < microwave_close_overshoot_rad <= 0.30
        ):
            raise ValueError("microwave arc angles are outside safe bounds")
        self.microwave_open_angle_rad = float(microwave_open_angle_rad)
        self.microwave_close_overshoot_rad = float(
            microwave_close_overshoot_rad
        )
        self._drawer_anchors: dict[str, Any] = {}
        self._drawer_slot_offsets: dict[
            str, tuple[np.ndarray, np.ndarray]
        ] = {}
        self._push_anchor: Any | None = None
        self._microwave_fixture_geometry: tuple[
            np.ndarray, np.ndarray, np.ndarray
        ] | None = None
        self._microwave_fixture_surface_points: np.ndarray | None = None
        self._microwave_surface_articulation: tuple[
            np.ndarray, np.ndarray, np.ndarray
        ] | None = None
        self._microwave_surface_articulation_trace: dict[str, Any] = {}
        self._microwave_closed_slot_world: np.ndarray | None = None
        self._microwave_reference_ee: np.ndarray | None = None
        self._microwave_state_open: bool | None = None
        self._microwave_active_goal: AtomicGoal | None = None
        self._microwave_open_direction_sign: int | None = None
        self.last_contact_trace: dict[str, Any] = {}
        self._drawer_estimate_trace: list[dict[str, Any]] = []

    def estimate(self, goal: AtomicGoal) -> ContactTargetEstimate:
        if goal.kind is AtomicGoalKind.PUSH:
            return self._plate_push(goal)
        subject = goal.subject
        if subject.label == "drawer":
            return self._drawer(goal)
        if subject.label == "stove":
            return self._stove_knob(goal)
        if subject.label == "microwave":
            return self._microwave_door(goal)
        raise LookupError(
            f"no RGB-D contact-target adapter for {subject.label!r}"
        )

    def estimate_microwave_continuation(
        self,
        goal: AtomicGoal,
        *,
        predicted_anchor_world: np.ndarray,
        frozen_hinge_world: np.ndarray,
        frozen_rotation_axis_world: np.ndarray,
        frozen_radius_m: float,
        anchor_radius_m: float,
        radius_tolerance_m: float,
    ) -> ContactTargetEstimate:
        """Reassociate a moving microwave edge on one frozen hinge circle.

        This deliberately bypasses ordinary body/closed-door fallback.  A
        current RGB-D vertical edge must be both local to the last retained
        contact point and consistent with the initially frozen hinge circle.
        Closing uses it after a typed retained stop; opening uses it after the
        first proven pinch and between compact back-side push segments.  The
        ordinary :meth:`estimate` path remains the final state verifier.
        """

        if (
            goal.kind not in {AtomicGoalKind.OPEN, AtomicGoalKind.CLOSE}
            or goal.subject.label != "microwave"
        ):
            raise ValueError(
                "local microwave continuation is only valid for a microwave goal"
            )
        if (
            self._microwave_fixture_geometry is None
            or self._microwave_reference_ee is None
            or self._microwave_active_goal is not goal
            or self._microwave_closed_slot_world is None
        ):
            raise LookupError(
                "local microwave continuation requires an initialized frozen target"
            )
        return self._microwave_door(
            goal,
            local_anchor_world=predicted_anchor_world,
            local_anchor_radius_m=anchor_radius_m,
            frozen_hinge_world=frozen_hinge_world,
            frozen_rotation_axis_world=frozen_rotation_axis_world,
            frozen_radius_m=frozen_radius_m,
            frozen_radius_tolerance_m=radius_tolerance_m,
        )

    def _drawer(self, goal: AtomicGoal) -> ContactTargetEstimate:
        selector = goal.subject.selector
        kind = selector.kind if selector is not None else SelectorKind.MIDDLE
        level = self._DRAWER_LEVELS.get(kind)
        if level is None:
            raise LookupError("drawer contact requires a top/middle/bottom selector")
        previous = self._drawer_anchors.get(level)
        observation = self._observation_provider()
        if previous is None:
            target = self.drawer_detector.detect(observation, level)
        else:
            target = self.drawer_detector.track(observation, previous, level)
        self._drawer_anchors[level] = target
        direction = (
            target.outward_world
            if goal.kind is AtomicGoalKind.OPEN
            else -target.outward_world
        )
        point = target.point_world.copy()
        point[2] -= 0.008
        closing = goal.kind is AtomicGoalKind.CLOSE
        if level not in self._drawer_slot_offsets:
            self._drawer_slot_offsets[level] = self._select_drawer_contact_slot(
                target,
                point,
                closing=closing,
                reference_ee_position_world=(
                    observation.proprio.ee_position_world
                    if hasattr(observation, "proprio")
                    else None
                ),
            )
        contact_offset, staging_offset = self._drawer_slot_offsets[level]
        # Opening needs a two-pad pinch on the selected clear part of the
        # handle.  Closing only uses that offset as a collision-free vertical
        # staging corridor, then traverses outside the cabinet plane and
        # pushes the moving drawer at its robust RGB-D centre.
        point += contact_offset
        approach_offset = (
            staging_offset - contact_offset if closing else None
        )
        estimate = ContactTargetEstimate(
            point,
            target.outward_world,
            target.feature_axis_world,
            direction,
            0.16,
            ("drawer", "cabinet"),
            target.confidence,
            approach_offset_world=approach_offset,
        )
        if closing:
            front_point, front_normal, front_support = (
                self.drawer_detector.detect_front_plane(
                    observation,
                    target,
                    level,
                )
            )
            front_point = self._select_drawer_front_push_point(
                front_point,
                front_normal,
                target.feature_axis_world,
                reference_ee_position_world=(
                    observation.proprio.ee_position_world
                    if hasattr(observation, "proprio")
                    else None
                ),
            )
            estimate = replace(
                estimate,
                drawer_front_point_world=front_point,
                drawer_front_normal_world=front_normal,
                drawer_front_support_m=front_support,
            )
            self.last_contact_trace["drawer_front_plane"] = {
                "point_world_m": front_point.tolist(),
                "normal_world": np.asarray(front_normal, dtype=np.float64).tolist(),
                "support_axis_span_m": float(front_support),
            }
        # Public RGB-D/proprio trace only; this records the successive drawer
        # anchors used by the bounded close servo and is intentionally not
        # populated from simulator joints or evaluator state.
        self._drawer_estimate_trace.append(
            {
                "goal_kind": goal.kind.value,
                "level": level,
                "point_world_m": estimate.point_world.tolist(),
                "outward_world": estimate.outward_world.tolist(),
                "manipulation_axis_world": estimate.manipulation_axis_world.tolist(),
            }
        )
        self.last_contact_trace["drawer_estimate_trace"] = list(
            self._drawer_estimate_trace
        )
        return estimate

    def _select_drawer_front_push_point(
        self,
        point_world: np.ndarray,
        normal_world: np.ndarray,
        feature_axis_world: np.ndarray,
        *,
        reference_ee_position_world: np.ndarray | None,
    ) -> np.ndarray:
        """Select a same-level solid front point with an RGB-D/SDF corridor."""

        point = np.asarray(point_world, dtype=np.float64)
        normal = np.asarray(normal_world, dtype=np.float64)
        axis = np.asarray(feature_axis_world, dtype=np.float64)
        if (
            point.shape != (3,)
            or normal.shape != (3,)
            or axis.shape != (3,)
            or not np.all(np.isfinite(point))
            or not np.all(np.isfinite(normal))
            or not np.all(np.isfinite(axis))
        ):
            raise LookupError("drawer front point geometry is invalid")
        normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
        axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
        observe_contact = getattr(self.observer, "observe_contact", None)
        scene = (
            observe_contact(("drawer", "cabinet"), point)
            if callable(observe_contact)
            else self.observer.observe(("drawer", "cabinet"))
        )
        sdf = scene.obstacle_sdf
        candidates: list[tuple[float, float, np.ndarray]] = []
        # Stay below the handle and away from either vertical jamb.  The
        # detector has already established the same-level plane; these small
        # lateral offsets only choose a solid patch with a clear approach.
        for lateral_m in (0.0, -0.040, 0.040, -0.060, 0.060):
            candidate = point + axis * lateral_m
            free_end = candidate + normal * 0.020
            safe = free_end.copy()
            safe[2] += 0.090
            # The final 20-mm typed contact segment is intentionally omitted
            # from the free-space SDF gate; the closed-finger contact move
            # owns that bounded approach and its residual/load proof.
            samples = np.linspace(safe, free_end, 9)
            distances = np.asarray(sdf.distance(samples), dtype=np.float64)
            if distances.shape != (len(samples),) or not np.all(np.isfinite(distances)):
                continue
            clearance = float(np.min(distances))
            if clearance < 0.024:
                continue
            reach = (
                0.0
                if reference_ee_position_world is None
                else float(np.linalg.norm(safe - reference_ee_position_world))
            )
            candidates.append((abs(lateral_m), reach, candidate))
        if not candidates:
            raise LookupError(
                "no same-level drawer-front point has a bounded RGB-D/SDF corridor"
            )
        _, _, selected = min(candidates, key=lambda item: (item[0], item[1]))
        return selected.copy()

    def _select_drawer_contact_slot(
        self,
        target: Any,
        center_point: np.ndarray,
        *,
        closing: bool,
        reference_ee_position_world: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Freeze a collision-free centre/side slot on the visible handle.

        The detector's robust inlier axis tells us where the physical handle
        runs.  A real object in front of one part of the handle remains in the
        neutral SDF, so score three bounded slots by their complete high-to-
        precontact and free approach corridors.  No obstacle is removed here.
        """

        axis = np.asarray(target.feature_axis_world, dtype=np.float64).copy()
        axis[2] = 0.0
        norm = float(np.linalg.norm(axis))
        if norm < 1e-6:
            raise LookupError("drawer handle has no reliable planar feature axis")
        axis /= norm
        # A drawer handle is part of the articulated cabinet surface.  Ask
        # the stable observer for its typed contact view so a sensor component
        # that contains the measured handle is treated as self geometry only
        # under the strict drawer-specific rules; independent clutter remains
        # in the SDF.  Older test/detector observers may expose only observe.
        observe_contact = getattr(self.observer, "observe_contact", None)
        scene = (
            observe_contact(("drawer", "cabinet"), center_point)
            if callable(observe_contact)
            else self.observer.observe(("drawer", "cabinet"))
        )
        sdf = scene.obstacle_sdf
        # An opening pull stays on the central 90-mm handle span.  A closing
        # push does not need to retain the handle and may use a wider visible
        # point on the same long handle/face axis when foreground clutter
        # blocks the centre.  The clutter itself remains in the neutral SDF.
        offsets = (
            (
                0.0,
                -0.045,
                0.045,
                -0.070,
                0.070,
                -0.085,
                0.085,
                -0.100,
                0.100,
                -0.115,
                0.115,
                -0.125,
                0.125,
            )
            if closing
            else (0.0, -0.045, 0.045)
        )
        required_clearance = 0.024 if closing else 0.030
        reference_ee = (
            None
            if reference_ee_position_world is None
            else np.asarray(reference_ee_position_world, dtype=np.float64)
        )
        scored: list[tuple[float, float, np.ndarray, float]] = []
        for scalar in offsets:
            offset = axis * scalar
            contact = center_point + offset
            precontact = contact + target.outward_world * 0.070
            safe = precontact.copy()
            safe[2] += 0.090
            # Stop the collision-checked corridor 20 mm before the intended
            # surface.  The final segment is governed by typed contact, while
            # every surrounding clutter component remains active.
            free_end = contact + target.outward_world * 0.020
            vertical = np.linspace(safe, precontact, 9)
            approach = np.linspace(precontact, free_end, 8)
            samples = np.concatenate((vertical, approach[1:]), axis=0)
            distances = np.asarray(sdf.distance(samples), dtype=np.float64)
            if distances.shape != (len(samples),) or not np.all(np.isfinite(distances)):
                raise LookupError("drawer slot SDF returned invalid distances")
            reach_distance = (
                0.0
                if reference_ee is None
                else float(np.linalg.norm(safe - reference_ee))
            )
            scored.append(
                (float(np.min(distances)), -abs(scalar), offset, reach_distance)
            )
        self.last_contact_trace = {
            "kind": "drawer_slot_selection",
            "closing": bool(closing),
            "required_clearance_m": required_clearance,
            "center_point_world_m": center_point.tolist(),
            "handle_axis_world": axis.tolist(),
            "candidates": [
                {
                    "offset_m": float(offset_scalar),
                    "minimum_sdf_clearance_m": float(score[0]),
                    "safe_pose_distance_from_public_ee_m": float(score[3]),
                }
                for offset_scalar, score in zip(offsets, scored, strict=True)
            ],
        }
        # Prefer the handle centre whenever it is genuinely clear; small SDF
        # differences between otherwise safe slots must not push the grasp to
        # an easy-to-slip handle endpoint.  Side slots are a clutter escape,
        # not a default grasp location.
        if scored[0][0] >= required_clearance:
            self.last_contact_trace["selected_offset_m"] = 0.0
            selected = scored[0][2].copy()
            return selected, selected.copy()
        safe = [item for item in scored[1:] if item[0] >= required_clearance]
        if not safe:
            raise LookupError(
                "no drawer-handle slot has the required RGB-D clearance"
            )
        # Prefer the nearest safe displacement from the robust handle centre;
        # clearance only breaks the left/right tie at a given displacement.
        clearance, _, selected, _ = max(
            safe,
            key=lambda item: (item[1], -item[3], item[0]),
        )
        self.last_contact_trace["selected_offset_world_m"] = selected.tolist()
        self.last_contact_trace["selected_clearance_m"] = float(clearance)
        if not closing:
            return selected.copy(), selected.copy()

        # The globally clearest vertical corridor can lie just outside the
        # moving drawer front.  Choose a second, bounded contact slot on the
        # measured handle/front span, preferring the nearest slot with at
        # least 8 mm raw RGB-D clearance.  Execution crosses from ``selected``
        # to this slot above the clutter before descending with closed jaws.
        contact_candidates = [
            item
            for offset_scalar, item in zip(offsets, scored, strict=True)
            if abs(offset_scalar) <= 0.125 and item[0] >= 0.018
        ]
        if not contact_candidates:
            raise LookupError(
                "no movable drawer-front slot has bounded RGB-D clearance"
            )
        contact_clearance, _, contact, _ = max(
            contact_candidates,
            key=lambda item: (item[1], -item[3], item[0]),
        )
        self.last_contact_trace["contact_offset_world_m"] = contact.tolist()
        self.last_contact_trace["contact_clearance_m"] = float(contact_clearance)
        self.last_contact_trace["staging_offset_world_m"] = selected.tolist()
        return contact.copy(), selected.copy()

    def _stove_knob(self, goal: AtomicGoal) -> ContactTargetEstimate:
        target = self.knob_detector.detect(self._observation_provider())
        return ContactTargetEstimate(
            target.point_world,
            target.outward_world,
            target.feature_axis_world,
            target.axis_world,
            0.01,
            ("stove", "stove knob"),
            target.confidence,
            False,
        )

    def _plate_push(self, goal: AtomicGoal) -> ContactTargetEstimate:
        observation = self._observation_provider()
        if self._push_anchor is None:
            target = self.plate_detector.detect(observation)
        else:
            target = self.plate_detector.track(observation, self._push_anchor)
        self._push_anchor = target
        point = target.object_center_world.copy()
        # Pinch the leading rim and drag it toward the sensed goal.  The
        # contact executor may retry once 8 mm farther inside after a fresh
        # RGB-D displacement check; both points remain on the measured plate.
        point += target.direction_world * (0.82 * target.object_radius_m)
        point[2] += 0.008
        travel = float(
            np.linalg.norm(
                target.target_center_world - target.object_center_world
            )
            + 0.025
        )
        return ContactTargetEstimate(
            point,
            -target.direction_world,
            np.array([0.0, 0.0, 1.0]),
            target.direction_world,
            travel,
            ("plate", "stove"),
            target.confidence,
            push_object_center_world=target.object_center_world,
            push_target_center_world=target.target_center_world,
            push_object_radius_m=target.object_radius_m,
            push_direction_world=target.direction_world,
        )

    def _microwave_door(
        self,
        goal: AtomicGoal,
        *,
        local_anchor_world: np.ndarray | None = None,
        local_anchor_radius_m: float | None = None,
        frozen_hinge_world: np.ndarray | None = None,
        frozen_rotation_axis_world: np.ndarray | None = None,
        frozen_radius_m: float | None = None,
        frozen_radius_tolerance_m: float | None = None,
    ) -> ContactTargetEstimate:
        observation = self._observation_provider()
        local_values = (
            local_anchor_world,
            local_anchor_radius_m,
            frozen_hinge_world,
            frozen_rotation_axis_world,
            frozen_radius_m,
            frozen_radius_tolerance_m,
        )
        local_mode = any(value is not None for value in local_values)
        if local_mode and not all(value is not None for value in local_values):
            raise ValueError(
                "local microwave continuation requires complete frozen geometry"
            )
        if self._microwave_fixture_geometry is None:
            if local_mode:
                raise LookupError(
                    "local microwave continuation cannot select a new appliance body"
                )
            scene = self.observer.observe(("microwave",))
            entities = [
                item
                for item in scene.entities
                if item.label == "microwave"
                and item.region is not None
                and self._plausible_microwave_body(item)
            ]
            if not entities:
                raise LookupError(
                    "no plausible microwave body OBB was visible in either RGB-D view"
                )
            # Multiple DINO boxes can carry the microwave label.  The actual
            # countertop appliance is the lowest plausible supported body;
            # a large robot/upper-cabinet fragment can otherwise win on raw
            # volume and hand an unrelated OBB to the handle detector.
            body = min(
                entities,
                key=lambda item: (
                    float(item.region.center[2]),
                    -float(np.prod(item.extent)),
                    -float(item.confidence),
                ),
            )
            assert body.region is not None
            self.last_contact_trace = {
                "kind": "microwave_body_selection",
                "candidates": [
                    {
                        "instance_id": item.instance_id,
                        "center_world_m": item.region.center.tolist(),
                        "extent_m": item.extent.tolist(),
                        "confidence": float(item.confidence),
                    }
                    for item in entities
                    if item.region is not None
                ],
                "selected_instance_id": body.instance_id,
                "selected_center_world_m": body.region.center.tolist(),
            }
            self._microwave_fixture_geometry = (
                body.region.center.copy(),
                body.region.axes.copy(),
                body.region.half_extents.copy(),
            )
            self._microwave_fixture_surface_points = (
                None
                if body.surface_points_world is None
                else np.asarray(
                    body.surface_points_world,
                    dtype=np.float64,
                ).copy()
            )
            self._microwave_reference_ee = (
                observation.proprio.ee_position_world.copy()
            )
            self.last_contact_trace.update(
                {
                    "selected_surface_point_count": (
                        0
                        if self._microwave_fixture_surface_points is None
                        else int(len(self._microwave_fixture_surface_points))
                    ),
                    "selected_surface_points_frozen_with_obb": bool(
                        self._microwave_fixture_surface_points is not None
                    ),
                }
            )

        is_first_measurement_for_goal = (
            not local_mode and self._microwave_active_goal is not goal
        )
        if local_mode:
            if (
                self._microwave_active_goal is not goal
                or self._microwave_closed_slot_world is None
            ):
                raise LookupError(
                    "local microwave continuation has no frozen initial target"
                )
            # A continuation always looks for the still-open moving edge.  It
            # must never enter the closed-door/body fallback path.
            initial_is_open = True
        elif is_first_measurement_for_goal:
            initial_is_open = (
                self._microwave_state_open
                if self._microwave_state_open is not None
                else goal.kind is AtomicGoalKind.CLOSE
            )
            self._microwave_active_goal = goal
            # A direction is meaningful only inside one goal.  The first
            # valid OPEN estimate below freezes it; local continuations are
            # requested only after the controller has proven contact load.
            self._microwave_open_direction_sign = None
        else:
            # The second measurement is the fresh post-action verifier and
            # must look for the requested final state, not the initial one.
            initial_is_open = goal.kind is AtomicGoalKind.OPEN
        center, axes, half_extents = self._microwave_fixture_geometry
        reference_ee = self._microwave_reference_ee
        assert reference_ee is not None
        fresh_closed_state_query = bool(
            not local_mode
            and not is_first_measurement_for_goal
            and goal.kind is AtomicGoalKind.CLOSE
            and self._microwave_closed_slot_world is not None
        )
        if local_mode:
            try:
                target = self.microwave_detector.detect(
                    observation,
                    center,
                    axes,
                    half_extents,
                    initial_is_open,
                    reference_ee_position_world=reference_ee,
                    local_anchor_world=local_anchor_world,
                    local_anchor_radius_m=local_anchor_radius_m,
                    frozen_hinge_world=frozen_hinge_world,
                    frozen_rotation_axis_world=frozen_rotation_axis_world,
                    frozen_radius_m=frozen_radius_m,
                    frozen_radius_tolerance_m=frozen_radius_tolerance_m,
                )
            except LookupError:
                self.last_contact_trace[
                    "microwave_local_detection_trace"
                ] = getattr(
                    self.microwave_detector,
                    "last_detection_trace",
                    {},
                )
                raise
            anchor = np.asarray(local_anchor_world, dtype=np.float64)
            hinge_value = np.asarray(frozen_hinge_world, dtype=np.float64)
            axis_value = np.asarray(
                frozen_rotation_axis_world, dtype=np.float64
            )
            axis_norm = float(np.linalg.norm(axis_value))
            if axis_norm < 1e-8:
                raise ValueError("frozen microwave rotation axis is degenerate")
            axis_value = axis_value / axis_norm
            anchor_error = float(np.linalg.norm(target.point_world - anchor))
            radial = target.point_world - hinge_value
            radial -= axis_value * float(np.dot(radial, axis_value))
            measured_radius = float(np.linalg.norm(radial))
            radius_error = abs(measured_radius - float(frozen_radius_m))
            if anchor_error > float(local_anchor_radius_m):
                raise LookupError(
                    "local microwave RGB-D edge exceeded the predicted endpoint neighbourhood"
                )
            if radius_error > float(frozen_radius_tolerance_m):
                raise LookupError(
                    "local microwave RGB-D edge left the frozen hinge circle"
                )
            self.last_contact_trace.update(
                {
                    "microwave_local_reassociation": True,
                    "microwave_local_anchor_world_m": anchor.tolist(),
                    "microwave_local_selected_edge_world_m": (
                        target.point_world.tolist()
                    ),
                    "microwave_local_anchor_error_m": anchor_error,
                    "microwave_local_anchor_radius_m": float(
                        local_anchor_radius_m
                    ),
                    "microwave_local_frozen_radius_m": float(frozen_radius_m),
                    "microwave_local_measured_radius_m": measured_radius,
                    "microwave_local_radius_error_m": radius_error,
                    "microwave_local_radius_tolerance_m": float(
                        frozen_radius_tolerance_m
                    ),
                    "microwave_local_detection_trace": getattr(
                        self.microwave_detector,
                        "last_detection_trace",
                        {},
                    ),
                }
            )
        else:
            try:
                target = self.microwave_detector.detect(
                    observation,
                    center,
                    axes,
                    half_extents,
                    initial_is_open,
                    reference_ee_position_world=reference_ee,
                    expected_closed_slot_world=(
                        self._microwave_closed_slot_world
                        if fresh_closed_state_query
                        else None
                    ),
                )
            except LookupError:
                if fresh_closed_state_query:
                    self.last_contact_trace[
                        "microwave_final_detection_trace"
                    ] = getattr(
                        self.microwave_detector,
                        "last_detection_trace",
                        {},
                    )
                raise
            if fresh_closed_state_query:
                detection_trace = getattr(
                    self.microwave_detector,
                    "last_detection_trace",
                    None,
                )
                self.last_contact_trace[
                    "microwave_final_detection_trace"
                ] = detection_trace or {}
                if detection_trace is None or bool(
                    detection_trace.get("closed_fallback_used", True)
                ):
                    raise LookupError(
                        "fresh closed microwave query lacked an observed "
                        "vertical handle/edge near the frozen closed slot"
                    )
        if is_first_measurement_for_goal:
            self._microwave_state_open = goal.kind is AtomicGoalKind.OPEN
        outward = target.outward_world
        direction = outward if goal.kind is AtomicGoalKind.OPEN else -outward
        if local_mode:
            hinge = np.asarray(frozen_hinge_world, dtype=np.float64).copy()
            rotation_axis = np.asarray(
                frozen_rotation_axis_world, dtype=np.float64
            ).copy()
            rotation_axis /= float(np.linalg.norm(rotation_axis))
            assert self._microwave_closed_slot_world is not None
            closed_slot = self._microwave_closed_slot_world.copy()
            self.last_contact_trace.update(
                {
                    "microwave_articulation_source": (
                        "frozen_hinge_circle_local_rgbd_edge"
                    ),
                    "microwave_surface_articulation_cache_reused": True,
                }
            )
        elif self._microwave_surface_articulation is not None:
            hinge, rotation_axis, closed_slot = (
                value.copy() for value in self._microwave_surface_articulation
            )
            self.last_contact_trace.update(
                self._microwave_surface_articulation_trace
            )
            self.last_contact_trace[
                "microwave_surface_articulation_cache_reused"
            ] = True
        elif self._microwave_fixture_surface_points is not None:
            try:
                hinge, rotation_axis, closed_slot, refinement_trace = (
                    self._surface_refined_microwave_articulation(
                        reference_ee,
                        center,
                        axes,
                        half_extents,
                        self._microwave_fixture_surface_points,
                        target.point_world,
                        initial_is_open=initial_is_open,
                    )
                )
            except (LookupError, ValueError) as error:
                self.last_contact_trace.update(
                    {
                        "microwave_articulation_source": (
                            "rejected_frozen_rgbd_surface"
                        ),
                        "microwave_surface_articulation_rejection": str(error),
                    }
                )
                raise
            self._microwave_surface_articulation = tuple(
                value.copy() for value in (hinge, rotation_axis, closed_slot)
            )
            self._microwave_surface_articulation_trace = dict(refinement_trace)
            self.last_contact_trace.update(refinement_trace)
            self.last_contact_trace[
                "microwave_surface_articulation_cache_reused"
            ] = False
        else:
            hinge, rotation_axis, closed_slot = (
                self.microwave_detector.articulation_geometry(
                    reference_ee,
                    center,
                    axes,
                    half_extents,
                    observed_handle_world=target.point_world,
                )
            )
            self.last_contact_trace.update(
                {
                    "microwave_articulation_source": (
                        "detector_obb_fallback_no_surface_points"
                    ),
                    "microwave_surface_point_count": 0,
                    "microwave_surface_articulation_cache_reused": False,
                }
            )
        if is_first_measurement_for_goal:
            self._microwave_closed_slot_world = closed_slot.copy()
        start_radius = target.point_world - hinge
        start_radius -= rotation_axis * float(
            np.dot(start_radius, rotation_axis)
        )
        radius_norm = float(np.linalg.norm(start_radius))
        if not 0.10 <= radius_norm <= 0.60:
            raise LookupError(
                f"microwave RGB-D hinge radius {radius_norm:.3f} m is implausible"
            )
        if local_mode:
            # The detector's ordinary open-door outward sign points from the
            # visible edge toward the current EE.  Near closure that chord can
            # cut diagonally across foreground clutter.  The frozen hinge and
            # freshly reassociated radial provide the actual door plane, so
            # approach along its orthogonal normal and choose only the sign
            # facing the current public EE sample.  This is a contact-approach
            # sign only; it must not redefine the semantic articulation sign.
            door_normal = np.cross(rotation_axis, start_radius)
            door_normal_norm = float(np.linalg.norm(door_normal))
            if door_normal_norm < 1e-8:
                raise LookupError("local microwave door normal is degenerate")
            door_normal /= door_normal_norm
            ee_delta = observation.proprio.ee_position_world - target.point_world
            if float(np.dot(door_normal, ee_delta)) < 0.0:
                door_normal *= -1.0
            outward = door_normal
            direction = -outward
            self.last_contact_trace[
                "microwave_local_door_normal_world"
            ] = outward.tolist()
        if goal.kind is AtomicGoalKind.OPEN:
            direction_sign_frozen = bool(local_mode)
            if local_mode:
                if self._microwave_open_direction_sign not in {-1, 1}:
                    raise LookupError(
                        "local microwave continuation has no frozen opening direction"
                    )
                direction_sign = float(
                    self._microwave_open_direction_sign
                )
            else:
                tangent_sign = float(
                    np.dot(
                        np.cross(rotation_axis, start_radius),
                        direction,
                    )
                )
                if abs(tangent_sign) < 0.05 * radius_norm:
                    raise LookupError(
                        "microwave pull direction is ambiguous relative to sensed hinge"
                    )
                direction_sign = 1.0 if tangent_sign > 0.0 else -1.0
            closed_radius = closed_slot - hinge
            closed_radius -= rotation_axis * float(
                np.dot(closed_radius, rotation_axis)
            )
            closed_radius_norm = float(np.linalg.norm(closed_radius))
            if not 0.10 <= closed_radius_norm <= 0.60:
                raise LookupError(
                    "microwave RGB-D closed-slot hinge radius is implausible"
                )
            progress_angle = float(
                np.arctan2(
                    np.dot(
                        rotation_axis,
                        np.cross(closed_radius, start_radius),
                    ),
                    np.dot(closed_radius, start_radius),
                )
            )
            if not np.isfinite(progress_angle) or abs(progress_angle) > np.pi:
                raise LookupError(
                    "microwave RGB-D visual open progress is invalid"
                )
            aligned_progress = direction_sign * progress_angle
            if (
                aligned_progress
                < -self._MICROWAVE_OPEN_PROGRESS_DIRECTION_TOLERANCE_RAD
            ):
                raise LookupError(
                    "microwave RGB-D visual open progress conflicts with sensed pull direction"
                )
            current_progress = max(0.0, aligned_progress)
            if (
                current_progress
                > self.microwave_open_angle_rad
                + self._MICROWAVE_OPEN_OVERRUN_TOLERANCE_RAD
            ):
                raise LookupError(
                    "microwave RGB-D visual open progress exceeds the requested arc"
                )
            if (
                not local_mode
                and self._microwave_open_direction_sign is None
            ):
                # Freeze only after all public RGB-D articulation gates have
                # accepted the first usable OPEN estimate.
                self._microwave_open_direction_sign = int(direction_sign)
            remaining_magnitude = max(
                0.0,
                self.microwave_open_angle_rad - current_progress,
            )
            remaining_magnitude = max(
                remaining_magnitude,
                self._MICROWAVE_OPEN_MIN_REMAINING_RAD,
            )
            rotation_angle = direction_sign * remaining_magnitude
            self.last_contact_trace.update(
                {
                    "microwave_open_current_progress_rad": float(
                        direction_sign * current_progress
                    ),
                    "microwave_open_remaining_angle_rad": float(rotation_angle),
                    "microwave_open_direction_sign": int(direction_sign),
                    "microwave_open_direction_sign_frozen": (
                        direction_sign_frozen
                    ),
                }
            )
        else:
            goal_radius = closed_slot - hinge
            goal_radius -= rotation_axis * float(
                np.dot(goal_radius, rotation_axis)
            )
            goal_norm = float(np.linalg.norm(goal_radius))
            if goal_norm < 0.10:
                raise LookupError("microwave closed-slot hinge radius is implausible")
            signed_angle = float(
                np.arctan2(
                    np.dot(
                        rotation_axis,
                        np.cross(start_radius, goal_radius),
                    ),
                    np.dot(start_radius, goal_radius),
                )
            )
            if abs(signed_angle) < 0.10:
                # The fresh visual point already lies nearly on the closed
                # ray; push only the bounded mechanical-stop overshoot.
                tangent_sign = float(
                    np.dot(
                        np.cross(rotation_axis, start_radius),
                        direction,
                    )
                )
                signed_angle = (
                    1.0 if tangent_sign >= 0.0 else -1.0
                ) * 0.10
            rotation_angle = signed_angle + np.sign(signed_angle) * (
                self.microwave_close_overshoot_rad
            )
        self.last_contact_trace.update(
            {
                "microwave_hinge_world_m": hinge.tolist(),
                "microwave_rotation_axis_world": rotation_axis.tolist(),
                "microwave_closed_slot_world_m": closed_slot.tolist(),
                "microwave_rotation_angle_rad": float(rotation_angle),
                "microwave_handle_radius_m": radius_norm,
            }
        )
        return ContactTargetEstimate(
            target.point_world,
            outward,
            target.feature_axis_world,
            direction,
            0.14,
            ("microwave",),
            target.confidence,
            True,
            hinge,
            rotation_axis,
            rotation_angle,
        )

    @classmethod
    def _surface_refined_microwave_articulation(
        cls,
        reference_ee_position_world: np.ndarray,
        fixture_center_world: np.ndarray,
        fixture_axes_world: np.ndarray,
        fixture_half_extents_m: np.ndarray,
        surface_points_world: np.ndarray,
        observed_handle_world: np.ndarray,
        *,
        initial_is_open: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Fit a microwave hinge from frozen RGB-D and public asset offsets.

        The completed OBB supplies only a stable frame.  Robust endpoint and
        front-plane coordinates come from the raw visible surface component;
        the public fixture family supplies only the hinge-to-handle offsets.
        The currently observed handle resolves the otherwise ambiguous left
        versus right hinge without fixture names, task IDs, or simulator pose.
        """

        reference_ee = np.asarray(
            reference_ee_position_world,
            dtype=np.float64,
        )
        center = np.asarray(fixture_center_world, dtype=np.float64)
        axes = np.asarray(fixture_axes_world, dtype=np.float64)
        half_extents = np.asarray(fixture_half_extents_m, dtype=np.float64)
        points = np.asarray(surface_points_world, dtype=np.float64)
        observed = np.asarray(observed_handle_world, dtype=np.float64)
        if (
            reference_ee.shape != (3,)
            or center.shape != (3,)
            or axes.shape != (3, 3)
            or half_extents.shape != (3,)
            or observed.shape != (3,)
        ):
            raise ValueError("microwave surface refinement has invalid geometry shapes")
        if (
            points.ndim != 2
            or points.shape[1:] != (3,)
            or len(points) < cls._MICROWAVE_SURFACE_MIN_POINTS
        ):
            raise LookupError(
                "microwave body surface has fewer than 64 fresh RGB-D points"
            )
        if not (
            np.all(np.isfinite(reference_ee))
            and np.all(np.isfinite(center))
            and np.all(np.isfinite(axes))
            and np.all(np.isfinite(half_extents))
            and np.all(np.isfinite(points))
            and np.all(np.isfinite(observed))
        ):
            raise ValueError("microwave surface refinement requires finite geometry")
        if np.any(half_extents <= 0.0):
            raise ValueError("microwave surface refinement requires positive OBB extents")
        axis_norms = np.linalg.norm(axes, axis=0)
        if np.any(axis_norms < 1e-8):
            raise ValueError("microwave surface refinement has a degenerate OBB axis")
        axes = axes / axis_norms[None, :]
        if not np.allclose(axes.T @ axes, np.eye(3), atol=0.08):
            raise ValueError("microwave surface refinement requires orthogonal OBB axes")

        vertical_index = int(np.argmax(np.abs(axes[2, :])))
        if abs(float(axes[2, vertical_index])) < 0.70:
            raise LookupError("microwave surface OBB has no reliable vertical axis")
        horizontal_indices = [
            index for index in range(3) if index != vertical_index
        ]
        horizontal_indices.sort(key=lambda index: float(half_extents[index]))
        short_index, long_index = horizontal_indices
        full_long_extent = 2.0 * float(half_extents[long_index])
        full_short_extent = 2.0 * float(half_extents[short_index])
        if not (0.20 <= full_long_extent <= 0.70):
            raise LookupError("microwave surface OBB long extent is implausible")
        if not (0.12 <= full_short_extent <= 0.45):
            raise LookupError("microwave surface OBB depth extent is implausible")

        vertical = axes[:, vertical_index].copy()
        if vertical[2] < 0.0:
            vertical *= -1.0
        long_axis = axes[:, long_index].copy()
        long_axis[2] = 0.0
        long_norm = float(np.linalg.norm(long_axis))
        if long_norm < 0.70:
            raise LookupError("microwave long OBB axis is not reliably horizontal")
        long_axis /= long_norm
        outward = axes[:, short_index].copy()
        outward[2] = 0.0
        outward -= long_axis * float(np.dot(outward, long_axis))
        outward_norm = float(np.linalg.norm(outward))
        if outward_norm < 0.70:
            raise LookupError("microwave depth OBB axis is not reliably horizontal")
        outward /= outward_norm
        if float(np.dot(outward, reference_ee - center)) < 0.0:
            outward *= -1.0

        delta = points - center[None, :]
        obb_local = delta @ axes
        limits = 1.35 * half_extents + 0.015
        inlier_mask = np.all(np.abs(obb_local) <= limits[None, :], axis=1)
        inliers = points[inlier_mask]
        if len(inliers) < cls._MICROWAVE_SURFACE_MIN_POINTS:
            raise LookupError(
                "microwave body surface has too few points inside its sensed OBB"
            )
        inlier_delta = inliers - center[None, :]
        long_coordinates = inlier_delta @ long_axis
        outward_coordinates = inlier_delta @ outward
        low_long, high_long = np.quantile(
            long_coordinates,
            cls._MICROWAVE_SURFACE_LONG_QUANTILES,
        )
        front_coordinate = float(
            np.quantile(
                outward_coordinates,
                cls._MICROWAVE_SURFACE_FRONT_QUANTILE,
            )
        )
        low_long = float(low_long)
        high_long = float(high_long)
        visible_long_span = high_long - low_long
        if not (0.20 <= visible_long_span <= 0.45):
            raise LookupError(
                "microwave visible surface has an implausible robust long span"
            )
        maximum_long_coordinate = 1.35 * float(half_extents[long_index]) + 0.015
        if (
            low_long < -maximum_long_coordinate
            or high_long > maximum_long_coordinate
        ):
            raise LookupError("microwave surface endpoints exceed the sensed OBB")
        short_half_extent = float(half_extents[short_index])
        if not (
            -0.50 * short_half_extent - 0.010
            <= front_coordinate
            <= 1.50 * short_half_extent + 0.030
        ):
            raise LookupError("microwave visible front-plane coordinate is implausible")

        # The visible front statistic has different physical support in the
        # two observable articulation states.  A closed appliance includes
        # the public 25-mm-thick door face; once open, that panel rotates out
        # of the body crop and q90 directly measures the hinge-side frame.
        # State comes solely from the language-requested action and the
        # episode-local visual detector protocol, never simulator state.
        front_hinge_inset = (
            cls._MICROWAVE_PUBLIC_OPEN_FRONT_HINGE_INSET_M
            if initial_is_open
            else cls._MICROWAVE_PUBLIC_CLOSED_FRONT_HINGE_INSET_M
        )
        hinge_front_coordinate = front_coordinate - front_hinge_inset
        expected_radius = float(
            np.hypot(
                cls._MICROWAVE_PUBLIC_HANDLE_LONG_OFFSET_M,
                cls._MICROWAVE_PUBLIC_HANDLE_OUTWARD_OFFSET_M,
            )
        )
        candidates: list[dict[str, Any]] = []
        for endpoint_name, endpoint_coordinate, handle_sign in (
            ("q02", low_long, 1.0),
            ("q98", high_long, -1.0),
        ):
            hinge = (
                center
                + long_axis * endpoint_coordinate
                + outward * hinge_front_coordinate
            )
            closed_slot = (
                hinge
                + handle_sign
                * long_axis
                * cls._MICROWAVE_PUBLIC_HANDLE_LONG_OFFSET_M
                + outward * cls._MICROWAVE_PUBLIC_HANDLE_OUTWARD_OFFSET_M
            )
            radial = observed - hinge
            radial -= vertical * float(np.dot(radial, vertical))
            observed_radius = float(np.linalg.norm(radial))
            candidates.append(
                {
                    "endpoint": endpoint_name,
                    "hinge": hinge,
                    "closed_slot": closed_slot,
                    "observed_radius_m": observed_radius,
                    "radius_residual_m": abs(observed_radius - expected_radius),
                }
            )
        selected = min(
            candidates,
            key=lambda item: float(item["radius_residual_m"]),
        )
        selected_radius = float(selected["observed_radius_m"])
        selected_residual = float(selected["radius_residual_m"])
        if not 0.16 <= selected_radius <= 0.36 or selected_residual > 0.085:
            raise LookupError(
                "microwave observed handle cannot resolve a plausible surface hinge"
            )

        trace = {
            "microwave_articulation_source": (
                "frozen_fresh_rgbd_surface_plus_public_asset_dimensions"
            ),
            "microwave_surface_point_count": int(len(points)),
            "microwave_surface_inlier_count": int(len(inliers)),
            "microwave_surface_long_axis_world": long_axis.tolist(),
            "microwave_surface_outward_axis_world": outward.tolist(),
            "microwave_surface_vertical_axis_world": vertical.tolist(),
            "microwave_surface_long_quantiles_m": [low_long, high_long],
            "microwave_surface_visible_long_span_m": float(visible_long_span),
            "microwave_surface_front_q90_m": front_coordinate,
            "microwave_surface_initial_door_state": (
                "open" if initial_is_open else "closed"
            ),
            "microwave_surface_front_hinge_inset_m": float(
                front_hinge_inset
            ),
            "microwave_surface_hinge_front_coordinate_m": float(
                hinge_front_coordinate
            ),
            "microwave_public_handle_long_offset_m": (
                cls._MICROWAVE_PUBLIC_HANDLE_LONG_OFFSET_M
            ),
            "microwave_public_handle_outward_offset_m": (
                cls._MICROWAVE_PUBLIC_HANDLE_OUTWARD_OFFSET_M
            ),
            "microwave_public_expected_handle_radius_m": expected_radius,
            "microwave_surface_hinge_candidates": [
                {
                    "endpoint": str(candidate["endpoint"]),
                    "hinge_world_m": candidate["hinge"].tolist(),
                    "closed_slot_world_m": candidate["closed_slot"].tolist(),
                    "observed_radius_m": float(candidate["observed_radius_m"]),
                    "radius_residual_m": float(candidate["radius_residual_m"]),
                }
                for candidate in candidates
            ],
            "microwave_surface_selected_endpoint": str(selected["endpoint"]),
            "microwave_surface_selected_radius_residual_m": selected_residual,
        }
        return (
            np.asarray(selected["hinge"], dtype=np.float64).copy(),
            vertical.copy(),
            np.asarray(selected["closed_slot"], dtype=np.float64).copy(),
            trace,
        )

    @staticmethod
    def _plausible_microwave_body(entity: SceneEntity) -> bool:
        """Reject handle fragments and giant background boxes conservatively."""

        region = entity.region
        if region is None:
            return False
        axes = np.asarray(region.axes, dtype=np.float64)
        full = 2.0 * np.asarray(region.half_extents, dtype=np.float64)
        vertical_index = int(np.argmax(np.abs(axes[2, :])))
        if abs(float(axes[2, vertical_index])) < 0.70:
            return False
        horizontal = [full[index] for index in range(3) if index != vertical_index]
        short, long = sorted(float(value) for value in horizontal)
        vertical = float(full[vertical_index])
        return bool(
            0.12 <= short <= 0.45
            and 0.20 <= long <= 0.70
            and 0.10 <= vertical <= 0.45
        )


class PolicyStepBudgetExhausted(TerminalExecutionError):
    """Terminal policy failure after the action budget has been consumed."""


class CavityLoadProofReleaseUnconfirmed(TerminalExecutionError):
    """Stop recovery when proprioception cannot prove a failed rim was freed."""


class FreeRimViewEvidenceUnavailable(TerminalExecutionError):
    """Stop while high/open when a non-physical self-field proof fails closed."""


@dataclass(frozen=True)
class OSCWaypointConfig:
    translation_scale_m: float = 0.05
    rotation_scale_rad: float = 0.5
    # Must be tighter than RouteCController's 8 mm phase tolerance; otherwise
    # a shrinking MPC prefix can be accepted without taking a step and stall
    # just outside the phase goal.
    position_tolerance_m: float = 0.0025
    intermediate_position_tolerance_m: float = 0.006
    rim_intermediate_position_tolerance_m: float = 0.008
    rotation_tolerance_rad: float = 0.10
    max_steps_per_waypoint: int = 18
    gripper_hold_steps: int = 6
    expand_gripper_hold_steps: int = 12
    # A thin bowl wall is reached later than an ordinary package during a
    # closing stroke.  Keep the EE still until the proprioceptive contact band
    # is reached, matching the independently calibrated Route-C rim grasp.
    rim_pinch_gripper_hold_steps: int = 15
    rim_loaded_translation_action_limit: float = 0.35
    # A fully open 80-mm hand makes the outside finger strike a cabinet top
    # before either pad reaches a bowl rim.  At the collision-free pregrasp,
    # narrow to a measured 10--24 mm window around the calibrated 20-mm
    # target, then use proprioceptive bang-bang control around that target
    # during descent.  Robosuite integrates Panda gripper commands, so a
    # neutral command would retain the preceding closing force.
    rim_cavity_preclose_target_width_m: float = 0.020
    rim_cavity_preclose_min_width_m: float = 0.010
    rim_cavity_preclose_max_width_m: float = 0.024
    rim_cavity_preclose_max_steps: int = 15
    # A free bowl does not need the cabinet-clearance 20-mm pre-shape.  Keep
    # a wider gap so bounded RGB-D centre/radius error cannot place both pads
    # on the same side of a thin rim, while still moving the inner finger well
    # away from the bowl centre/bottom collision produced by an 80-mm hand.
    rim_free_space_preclose_target_width_m: float = 0.040
    rim_free_space_preclose_min_width_m: float = 0.030
    rim_free_space_preclose_max_width_m: float = 0.048
    rim_free_space_preclose_max_steps: int = 12
    # A free-standing thin wall does not have the fixture reaction force of a
    # cavity rim.  Its first blocked close can therefore pull the compliant EE
    # a few millimetres toward the bowl centre.  Re-centre by only a bounded
    # 4-mm outward stroke: live gripper width showed that the shared 8-mm
    # cavity stroke crossed the wall and ended in an empty close.  Direction,
    # start radius, and cap still come solely from frozen RGB-D geometry.
    rim_free_space_seat_pull_m: float = 0.004
    # A free thin wall can present a sharp proprioceptive width cliff during
    # that stroke.  Stop on the last *non-marginal* blocked-width band instead
    # of taking the next control tick into an empty close.  The lower edge is
    # still the existing strict marginal-contact boundary (blocked minimum +
    # 1 mm); the upper edge, minimum radial travel, and observed width drop
    # make this a typed contact transition rather than a generic early exit.
    rim_free_space_seat_edge_width_margin_m: float = 0.004
    rim_free_space_seat_edge_min_progress_m: float = 0.00075
    rim_free_space_seat_edge_min_width_drop_m: float = 0.00025
    rim_free_space_seat_edge_min_steps: int = 2
    # Once both pads have closed on a cavity rim, a short in-plane pull seats
    # the wall against the pads before the proof lift.  The direction is the
    # sensor-selected physical rim side (candidate local-Y times its side
    # sign), so this remains invariant to drawer/world orientation.  Five OSC
    # ticks are the minimum closed-loop duration; allow up to fifteen for the
    # measured EE to reach a strict support-plane pose gate under compliant
    # contact.
    rim_cavity_seat_pull_m: float = 0.008
    rim_cavity_seat_pull_steps: int = 5
    rim_cavity_seat_pull_max_steps: int = 15
    rim_cavity_seat_pull_min_progress_m: float = 0.007
    rim_cavity_seat_pull_max_goal_error_m: float = 0.0015
    rim_cavity_seat_pull_max_orthogonal_error_m: float = 0.002
    rim_cavity_seat_pull_max_vertical_error_m: float = 0.002
    # A fixture-supported wall can become mechanically seated before the
    # compliant arm realizes the full 8-mm Cartesian request.  Accept that
    # second, sensor-only completion only when the outward motion has reached
    # at least 2 mm and stalled over four live proprio samples, while the
    # blocked fingers have measurably opened under sustained closing.  A
    # complete 4-sample Cartesian plateau and the mandatory subsequent lift
    # proof keep this gate from treating transit as a seated rim.  Its relaxed
    # vertical / orthogonal allowances remain millimetre-bounded; the ordinary
    # 7-mm geometric completion above keeps its tighter 2-mm pose gates.
    rim_cavity_seat_contact_min_progress_m: float = 0.001
    rim_cavity_seat_contact_min_width_gain_m: float = 0.00035
    rim_cavity_seat_contact_stall_window_steps: int = 4
    rim_cavity_seat_contact_max_radial_span_m: float = 0.0007
    rim_cavity_seat_contact_max_cartesian_span_m: float = 0.0008
    rim_cavity_seat_contact_max_orthogonal_error_m: float = 0.0025
    rim_cavity_seat_contact_max_vertical_error_m: float = 0.004
    # Before committing to the 110-mm transport lift, load every cavity rim
    # contact with a short vertical proof.  Five driven samples raise about
    # 12 mm, then three samples hold that same Cartesian target.  The proof
    # sees only public EE/gripper proprioception: a wall that was merely
    # wedged against a drawer usually collapses during this small load, while
    # a retained two-pad rim remains blocked with a stable finger-width load.
    # The compliant EE may still settle several millimetres toward the same
    # frozen target during the hold, so its pose span is diagnostic only.
    rim_cavity_load_proof_lift_m: float = 0.012
    rim_cavity_load_proof_steps: int = 5
    rim_cavity_load_proof_hold_steps: int = 3
    rim_cavity_load_proof_min_lift_m: float = 0.009
    rim_cavity_load_proof_max_lift_m: float = 0.015
    rim_cavity_load_proof_max_width_loss_m: float = 0.001
    rim_cavity_load_proof_max_hold_width_span_m: float = 0.0005
    # A non-marginal free-space rim close has no fixture reaction force to
    # justify an in-plane seat.  Probe it directly with a strict 6-mm vertical
    # load instead.  The successful historical black-bowl trace settled from
    # 9.0 mm to 4.8 mm under load, so absolute non-marginal width and a stable
    # loaded hold are the valid evidence; cavity's 1-mm relative-loss gate is
    # intentionally not reused.
    rim_free_space_load_proof_lift_m: float = 0.006
    rim_free_space_load_proof_steps: int = 3
    rim_free_space_load_proof_hold_steps: int = 2
    rim_free_space_load_proof_min_lift_m: float = 0.005
    rim_free_space_load_proof_max_lift_m: float = 0.008
    rim_free_space_load_proof_min_width_m: float = 0.004
    rim_free_space_load_proof_max_hold_width_span_m: float = 0.0005
    # Releasing a thin wall also needs more than the ordinary package pulse:
    # retreating while the fingers are still opening can drag a bowl off the
    # centre of a small support.
    rim_pinch_release_hold_steps: int = 12
    pinch_blocked_min_width_m: float = 0.003
    rim_pinch_blocked_min_width_m: float = 0.003
    pinch_max_width_m: float = 0.070
    expansion_min_width_m: float = 0.020
    expansion_max_width_m: float = 0.070
    # LIBERO's Panda reports about 80 mm at its calibrated no-load opening.
    # Widths between the blocked threshold and this physical limit are
    # ambiguous: they may be a wide bowl or fully open fingers.  They are only
    # allowed to proceed tentatively to LIFT and require fresh visual evidence.
    expansion_physical_max_width_m: float = 0.080
    min_position_progress_m: float = 0.002
    min_rotation_progress_rad: float = 0.02
    grasp_contact_max_residual_m: float = 0.040
    # Contact approaches may settle with a small compliant wrist deflection;
    # nine degrees still preserves the intended jaw/normal geometry while
    # avoiding false rejection at a real handle or rim plateau.
    grasp_contact_orientation_rad: float = 0.16
    grasp_contact_stall_epsilon_m: float = 0.0008
    grasp_contact_window_steps: int = 4
    # A stove knob normally reaches its mechanical stop before the wrist can
    # realize the final few tenths of a radian.  This typed completion remains
    # sensor/proprio-only: it requires a nearly complete signed rotation, a
    # blocked and stable finger width, a bounded contact position, and a true
    # Cartesian/angular plateau over consecutive live samples.
    turn_contact_position_tolerance_m: float = 0.025
    turn_contact_cartesian_span_m: float = 0.002
    turn_contact_rotation_span_rad: float = 0.020
    turn_contact_width_span_m: float = 0.0015
    turn_contact_reverse_tolerance_rad: float = 0.025
    turn_contact_window_steps: int = 5
    place_contact_max_vertical_residual_m: float = 0.055
    place_contact_xy_tolerance_m: float = 0.008
    # A retained rim can stop about 15 mm off the nominal EE centre when the
    # bowl first touches a support.  This typed contact gate remains much
    # tighter than the plate footprint and additionally requires a downward
    # approach, retained-width evidence, orientation agreement, and a short
    # goal-distance progress plateau.
    rim_place_contact_xy_tolerance_m: float = 0.015
    rim_place_contact_max_vertical_residual_m: float = 0.025
    rim_place_contact_orientation_tolerance_rad: float = 0.10
    rim_place_contact_stall_epsilon_m: float = 0.0015
    rim_place_contact_cartesian_span_m: float = 0.008
    # A high support can leave the compliant arm a few millimetres above and
    # sideways from a collision-free pregrasp.  This is safe only while still
    # above the bowl; GRASP itself keeps the strict sensor/contact gates.
    rim_pregrasp_xy_tolerance_m: float = 0.008
    rim_pregrasp_vertical_clearance_m: float = 0.050
    rim_pregrasp_orientation_tolerance_rad: float = 0.10
    # A cavity APPROACH target is the nominal 90-mm pregrasp, not the rim.
    # Accepting a well-aligned pose up to 15 mm below it still leaves at least
    # 75 mm of vertical separation before the typed GRASP descent and avoids
    # spending repeated 18-step chunks correcting harmless OSC undershoot.
    rim_cavity_pregrasp_below_tolerance_m: float = 0.015
    # Rim contact can leave a held bowl slightly below the free-space lift
    # target.  A 20-mm residual still guarantees at least a 90-mm lift while
    # the post-phase blocked-width check independently proves retention.
    rim_lift_position_tolerance_m: float = 0.020
    rim_lift_orientation_tolerance_rad: float = 0.10
    # Once post-LIFT proprioception confirms the wall is still retained, use
    # the same 15-mm translation allowance as the legacy held-bowl transfer,
    # while retaining full-pose (not axisymmetric) orientation checking.  The
    # vertical component stays compliant, but XY must be centered before a
    # placement descent can begin.
    rim_transfer_position_tolerance_m: float = 0.015
    rim_transfer_planar_tolerance_m: float = 0.008
    rim_transfer_orientation_tolerance_rad: float = 0.10
    expand_lift_position_tolerance_m: float = 0.015
    expand_lift_tool_axis_tolerance_rad: float = 0.10
    expand_pregrasp_xy_tolerance_m: float = 0.008
    expand_pregrasp_vertical_clearance_m: float = 0.050
    expand_pregrasp_tool_axis_tolerance_rad: float = 0.10
    expand_grasp_xy_tolerance_m: float = 0.015
    expand_grasp_vertical_residual_m: float = 0.150
    expand_grasp_tool_axis_tolerance_rad: float = 0.12
    expand_transfer_position_tolerance_m: float = 0.015
    expand_transfer_tool_axis_tolerance_rad: float = 0.10

    def __post_init__(self) -> None:
        if not 0.0 < self.rim_loaded_translation_action_limit <= 1.0:
            raise ValueError("loaded rim translation limit must lie in (0, 1]")
        positive = (
            self.translation_scale_m,
            self.rotation_scale_rad,
            self.position_tolerance_m,
            self.intermediate_position_tolerance_m,
            self.rim_intermediate_position_tolerance_m,
            self.rotation_tolerance_rad,
            self.min_position_progress_m,
            self.min_rotation_progress_rad,
            self.grasp_contact_max_residual_m,
            self.grasp_contact_orientation_rad,
            self.grasp_contact_stall_epsilon_m,
            self.turn_contact_position_tolerance_m,
            self.turn_contact_cartesian_span_m,
            self.turn_contact_rotation_span_rad,
            self.turn_contact_width_span_m,
            self.turn_contact_reverse_tolerance_rad,
            self.place_contact_max_vertical_residual_m,
            self.place_contact_xy_tolerance_m,
            self.rim_place_contact_xy_tolerance_m,
            self.rim_place_contact_max_vertical_residual_m,
            self.rim_place_contact_orientation_tolerance_rad,
            self.rim_place_contact_stall_epsilon_m,
            self.rim_place_contact_cartesian_span_m,
            self.rim_pregrasp_xy_tolerance_m,
            self.rim_pregrasp_vertical_clearance_m,
            self.rim_pregrasp_orientation_tolerance_rad,
            self.rim_cavity_pregrasp_below_tolerance_m,
            self.rim_lift_position_tolerance_m,
            self.rim_lift_orientation_tolerance_rad,
            self.rim_transfer_position_tolerance_m,
            self.rim_transfer_planar_tolerance_m,
            self.rim_transfer_orientation_tolerance_rad,
            self.expand_lift_position_tolerance_m,
            self.expand_lift_tool_axis_tolerance_rad,
            self.expand_pregrasp_xy_tolerance_m,
            self.expand_pregrasp_vertical_clearance_m,
            self.expand_pregrasp_tool_axis_tolerance_rad,
            self.expand_grasp_xy_tolerance_m,
            self.expand_grasp_vertical_residual_m,
            self.expand_grasp_tool_axis_tolerance_rad,
            self.expand_transfer_position_tolerance_m,
            self.expand_transfer_tool_axis_tolerance_rad,
            self.rim_pinch_blocked_min_width_m,
            self.rim_cavity_preclose_target_width_m,
            self.rim_cavity_preclose_min_width_m,
            self.rim_cavity_preclose_max_width_m,
            self.rim_free_space_preclose_target_width_m,
            self.rim_free_space_preclose_min_width_m,
            self.rim_free_space_preclose_max_width_m,
            self.rim_free_space_seat_pull_m,
            self.rim_free_space_seat_edge_width_margin_m,
            self.rim_free_space_seat_edge_min_progress_m,
            self.rim_free_space_seat_edge_min_width_drop_m,
            self.rim_cavity_seat_pull_m,
            self.rim_cavity_seat_pull_min_progress_m,
            self.rim_cavity_seat_pull_max_goal_error_m,
            self.rim_cavity_seat_pull_max_orthogonal_error_m,
            self.rim_cavity_seat_pull_max_vertical_error_m,
            self.rim_cavity_seat_contact_min_progress_m,
            self.rim_cavity_seat_contact_min_width_gain_m,
            self.rim_cavity_seat_contact_max_radial_span_m,
            self.rim_cavity_seat_contact_max_cartesian_span_m,
            self.rim_cavity_seat_contact_max_orthogonal_error_m,
            self.rim_cavity_seat_contact_max_vertical_error_m,
            self.rim_cavity_load_proof_lift_m,
            self.rim_cavity_load_proof_min_lift_m,
            self.rim_cavity_load_proof_max_lift_m,
            self.rim_cavity_load_proof_max_width_loss_m,
            self.rim_cavity_load_proof_max_hold_width_span_m,
            self.rim_free_space_load_proof_lift_m,
            self.rim_free_space_load_proof_min_lift_m,
            self.rim_free_space_load_proof_max_lift_m,
            self.rim_free_space_load_proof_min_width_m,
            self.rim_free_space_load_proof_max_hold_width_span_m,
            self.expansion_max_width_m,
            self.expansion_physical_max_width_m,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("OSC scales and tolerances must be positive")
        if (
            self.max_steps_per_waypoint < 1
            or self.gripper_hold_steps < 1
            or self.expand_gripper_hold_steps < 1
            or self.rim_pinch_gripper_hold_steps < 1
            or self.rim_pinch_release_hold_steps < 1
            or self.rim_cavity_preclose_max_steps < 1
            or self.rim_free_space_preclose_max_steps < 1
            or self.rim_free_space_seat_edge_min_steps < 2
            or self.rim_cavity_seat_pull_steps < 1
            or self.rim_cavity_seat_pull_max_steps < 1
            or self.rim_cavity_seat_contact_stall_window_steps < 2
            or not 4 <= self.rim_cavity_load_proof_steps <= 6
            or not 2 <= self.rim_cavity_load_proof_hold_steps <= 3
            or not 2 <= self.rim_free_space_load_proof_steps <= 4
            or not 2 <= self.rim_free_space_load_proof_hold_steps <= 3
        ):
            raise ValueError("waypoint and gripper step limits must be positive")
        if self.rim_cavity_seat_pull_steps > self.rim_cavity_seat_pull_max_steps:
            raise ValueError("cavity rim seat-pull step budget must be ordered")
        if self.rim_cavity_seat_pull_min_progress_m > self.rim_cavity_seat_pull_m:
            raise ValueError("cavity rim seat-pull progress cannot exceed its goal")
        if (
            self.rim_cavity_seat_contact_min_progress_m
            > self.rim_cavity_seat_pull_min_progress_m
        ):
            raise ValueError(
                "cavity contact-seat progress cannot exceed geometric progress"
            )
        if not 0.010 <= self.rim_cavity_load_proof_lift_m <= 0.015:
            raise ValueError(
                "cavity load-proof commanded lift must remain within 10--15 mm"
            )
        if not (
            0.009
            <= self.rim_cavity_load_proof_min_lift_m
            <= self.rim_cavity_load_proof_lift_m
            <= self.rim_cavity_load_proof_max_lift_m
            <= 0.015
        ):
            raise ValueError(
                "cavity load-proof actual lift window must remain within 9--15 mm"
            )
        if self.rim_cavity_load_proof_max_width_loss_m > 0.001:
            raise ValueError("cavity load-proof width loss cannot exceed 1 mm")
        if self.rim_cavity_load_proof_max_hold_width_span_m > 0.0005:
            raise ValueError(
                "cavity load-proof hold-width span cannot exceed 0.5 mm"
            )
        if not 0.005 <= self.rim_free_space_load_proof_lift_m <= 0.008:
            raise ValueError(
                "free-space rim load-proof command must remain within 5--8 mm"
            )
        if not (
            0.005
            <= self.rim_free_space_load_proof_min_lift_m
            <= self.rim_free_space_load_proof_lift_m
            <= self.rim_free_space_load_proof_max_lift_m
            <= 0.008
        ):
            raise ValueError(
                "free-space rim actual load-proof window must remain within 5--8 mm"
            )
        if not (
            self.rim_pinch_blocked_min_width_m
            < self.rim_free_space_load_proof_min_width_m
            <= self.rim_pinch_blocked_min_width_m + 0.001
        ):
            raise ValueError(
                "free-space rim load proof must require a non-marginal width"
            )
        if self.rim_free_space_load_proof_max_hold_width_span_m > 0.0005:
            raise ValueError(
                "free-space rim load-proof hold-width span cannot exceed 0.5 mm"
            )
        if self.grasp_contact_window_steps < 2:
            raise ValueError("grasp contact window must contain at least two steps")
        if self.turn_contact_window_steps < 4:
            raise ValueError("turn contact window must contain at least four steps")
        if not (
            0.0
            < self.expansion_min_width_m
            <= self.expansion_max_width_m
            < self.expansion_physical_max_width_m
            <= 0.12
        ):
            raise ValueError("expansion width thresholds are inconsistent")
        if self.rim_pinch_blocked_min_width_m >= self.pinch_max_width_m:
            raise ValueError("rim-pinch blocked width must be below pinch maximum")
        if not (
            0.001
            < self.rim_free_space_seat_edge_width_margin_m
            < self.pinch_max_width_m - self.rim_pinch_blocked_min_width_m
        ):
            raise ValueError("free-space rim seat edge-width band is inconsistent")
        if not (
            self.rim_cavity_preclose_min_width_m
            < self.rim_cavity_preclose_target_width_m
            < self.rim_cavity_preclose_max_width_m
            < self.pinch_max_width_m
        ):
            raise ValueError("cavity preclose width window is inconsistent")
        if not (
            self.rim_free_space_preclose_min_width_m
            < self.rim_free_space_preclose_target_width_m
            < self.rim_free_space_preclose_max_width_m
            < self.pinch_max_width_m
        ):
            raise ValueError("free-space rim preclose width window is inconsistent")


class LiberoRouteCRobot:
    """Execute Route C optimiser waypoints through real normalized OSC actions.

    The adapter is intentionally capability-limited: it can read only a
    sanitized :class:`RobotObservation` and dispatch an :class:`OSCAction` to
    a callable that returns another sanitized observation.  Episode scoring,
    recording, and simulator lifecycle remain outside this object.
    """

    def __init__(
        self,
        observation_provider: Callable[[], RobotObservation],
        action_executor: Callable[[OSCAction], RobotObservation],
        *,
        step_budget: int,
        config: OSCWaypointConfig | None = None,
    ) -> None:
        if not callable(observation_provider):
            raise TypeError("observation_provider must be callable")
        if not callable(action_executor):
            raise TypeError("action_executor must be callable")
        if step_budget < 1:
            raise ValueError("step_budget must be positive")
        self._observation_provider = observation_provider
        self._action_executor = action_executor
        self.step_budget = int(step_budget)
        self.config = config or OSCWaypointConfig()
        self.steps_executed = 0
        self._last_gripper_command = -1.0
        self._active_grasp_mode = GraspMode.PINCH
        self._active_grasp_candidate_id: str | None = None
        self._allow_tentative_expansion = False
        self._grasp_contact_reached = False
        self._grasp_contact_servo_goal: np.ndarray | None = None
        self._grasp_contact_enabled = False
        self._grasp_contact_residual_m = self.config.grasp_contact_max_residual_m
        self._turn_contact_enabled = False
        self._turn_contact_reached = False
        self._turn_contact_axis_world: np.ndarray | None = None
        self._turn_contact_start_pose: np.ndarray | None = None
        self._turn_contact_min_rotation_rad = 0.0
        self._turn_contact_progress_rad = 0.0
        self._turn_contact_monotonic = True
        self._turn_contact_previous_progress_rad = 0.0
        self._place_contact_reached = False
        self._expand_lift_tolerance_reached = False
        self._rim_lift_tolerance_reached = False
        self._expand_safe_pregrasp_reached = False
        self._rim_safe_pregrasp_reached = False
        self._rim_transfer_tolerance_reached = False
        self._expand_transfer_tolerance_reached = False
        self._expand_retention_confirmed = False
        self._rim_retention_confirmed = False
        self._rim_grasp_marginal = False
        self._cavity_rim_preclosed = False
        self._cavity_rim_load_baseline_width_m: float | None = None
        self._cavity_rim_load_proof_passed = False
        self._cavity_rim_load_proof_failed = False
        self._active_cavity_rim_execution_profile: dict[str, Any] | None = None
        self._last_sensor_safe_view_motion_samples: tuple[
            dict[str, object], ...
        ] = ()
        self.phase_trace: list[dict[str, Any]] = []
        self.grasp_checks: list[dict[str, Any]] = []

    @property
    def grasp_contact_reached(self) -> bool:
        return self._grasp_contact_reached

    def reset_grasp_contact(self) -> None:
        self._grasp_contact_reached = False
        self._grasp_contact_servo_goal = None

    def set_grasp_contact_enabled(
        self, enabled: bool, *, max_residual_m: float | None = None
    ) -> None:
        self._grasp_contact_enabled = bool(enabled)
        self._grasp_contact_residual_m = (
            self.config.grasp_contact_max_residual_m
            if max_residual_m is None
            else float(max_residual_m)
        )
        if self._grasp_contact_residual_m <= 0:
            raise ValueError("grasp contact residual must be positive")
        self._grasp_contact_reached = False
        self._grasp_contact_servo_goal = None

    @property
    def turn_contact_reached(self) -> bool:
        return self._turn_contact_reached

    @property
    def turn_contact_progress_rad(self) -> float:
        return float(self._turn_contact_progress_rad)

    def set_turn_contact_enabled(
        self,
        enabled: bool,
        *,
        axis_world: np.ndarray | None = None,
        minimum_rotation_rad: float | None = None,
    ) -> None:
        """Arm or clear the typed mechanical-stop completion for one turn."""

        self._turn_contact_enabled = bool(enabled)
        self._turn_contact_reached = False
        self._turn_contact_progress_rad = 0.0
        self._turn_contact_previous_progress_rad = 0.0
        self._turn_contact_monotonic = True
        if not enabled:
            self._turn_contact_axis_world = None
            self._turn_contact_start_pose = None
            self._turn_contact_min_rotation_rad = 0.0
            return
        if axis_world is None or minimum_rotation_rad is None:
            raise ValueError(
                "turn contact requires an axis and minimum signed rotation"
            )
        axis = np.asarray(axis_world, dtype=np.float64)
        norm = float(np.linalg.norm(axis))
        minimum = float(minimum_rotation_rad)
        if (
            axis.shape != (3,)
            or not np.all(np.isfinite(axis))
            or norm < 1e-8
            or not np.isfinite(minimum)
            or minimum <= 0.0
        ):
            raise ValueError("turn contact geometry must be finite and positive")
        self._turn_contact_axis_world = axis / norm
        self._turn_contact_start_pose = self.current_ee_pose()
        self._turn_contact_min_rotation_rad = minimum

    def reset_turn_contact(self) -> None:
        self._turn_contact_reached = False
        self._turn_contact_progress_rad = 0.0
        self._turn_contact_previous_progress_rad = 0.0
        self._turn_contact_monotonic = True
        if self._turn_contact_enabled:
            self._turn_contact_start_pose = self.current_ee_pose()

    @property
    def place_contact_reached(self) -> bool:
        return self._place_contact_reached

    def reset_place_contact(self) -> None:
        self._place_contact_reached = False

    @property
    def expand_lift_tolerance_reached(self) -> bool:
        return self._expand_lift_tolerance_reached

    def reset_expand_lift_tolerance(self) -> None:
        self._expand_lift_tolerance_reached = False

    @property
    def rim_lift_tolerance_reached(self) -> bool:
        return self._rim_lift_tolerance_reached

    def reset_rim_lift_tolerance(self) -> None:
        self._rim_lift_tolerance_reached = False

    @property
    def expand_safe_pregrasp_reached(self) -> bool:
        return self._expand_safe_pregrasp_reached

    def reset_expand_safe_pregrasp(self) -> None:
        self._expand_safe_pregrasp_reached = False

    @property
    def rim_safe_pregrasp_reached(self) -> bool:
        return self._rim_safe_pregrasp_reached

    def reset_rim_safe_pregrasp(self) -> None:
        self._rim_safe_pregrasp_reached = False

    @property
    def rim_transfer_tolerance_reached(self) -> bool:
        return self._rim_transfer_tolerance_reached

    def reset_rim_transfer_tolerance(self) -> None:
        self._rim_transfer_tolerance_reached = False

    @property
    def expand_transfer_tolerance_reached(self) -> bool:
        return self._expand_transfer_tolerance_reached

    def reset_expand_transfer_tolerance(self) -> None:
        self._expand_transfer_tolerance_reached = False

    @property
    def expand_retention_confirmed(self) -> bool:
        return self._expand_retention_confirmed

    def set_expand_retention_confirmed(self, confirmed: bool) -> None:
        """Record the sensor/proprio post-LIFT retention gate for TRANSFER."""

        self._expand_retention_confirmed = bool(confirmed)

    def set_rim_retention_confirmed(self, confirmed: bool) -> None:
        """Record a blocked-width post-LIFT rim gate for typed TRANSFER."""

        self._rim_retention_confirmed = bool(confirmed)

    @property
    def rim_retention_confirmed(self) -> bool:
        return self._rim_retention_confirmed

    @property
    def rim_grasp_marginal(self) -> bool:
        """Whether the latest rim-width evidence was just above empty-close."""

        return self._rim_grasp_marginal

    def reset_rim_grasp_marginal(self) -> None:
        self._rim_grasp_marginal = False

    def current_ee_pose(self) -> np.ndarray:
        return np.array(
            self._observation().proprio.T_world_ee,
            dtype=np.float64,
            copy=True,
        )

    def current_public_view_snapshot(self) -> dict[str, object]:
        """Return synchronized public proprioception and pinhole calibrations.

        This deliberately omits RGB values, task metadata, and every scoring or
        simulator capability.  The controller uses the immutable calibration
        copies only to determine whether a sensor-derived OBB can intersect the
        wrist-camera frustum before requesting another RGB-D frame.
        """

        observation = self._observation()
        cameras: list[dict[str, object]] = []
        for policy_name, frame in sorted(observation.cameras.items()):
            calibration = frame.calibration
            if (
                type(policy_name) is not str
                or policy_name not in {"agentview", "wrist"}
                or type(calibration.name) is not str
                or not calibration.name.strip()
                or type(calibration.width) is not int
                or type(calibration.height) is not int
                or type(calibration.observation_v_flipped) is not bool
            ):
                raise ValueError(
                    "public camera metadata must retain exact builtin types"
                )
            cameras.append(
                {
                    "policy_name": policy_name,
                    "sensor_name": calibration.name,
                    "width_px": calibration.width,
                    "height_px": calibration.height,
                    "intrinsic": np.array(
                        calibration.intrinsic, dtype=np.float64, copy=True
                    ),
                    "world_from_camera": np.array(
                        calibration.T_world_camera,
                        dtype=np.float64,
                        copy=True,
                    ),
                    "observation_v_flipped": (
                        calibration.observation_v_flipped
                    ),
                }
            )
        gripper_width = observation.proprio.gripper_width_m
        if isinstance(gripper_width, (bool, np.bool_)) or not np.isfinite(
            gripper_width
        ):
            raise ValueError("public gripper width must be a finite scalar")
        return {
            "ee_pose_world": np.array(
                observation.proprio.T_world_ee, dtype=np.float64, copy=True
            ),
            "gripper_width_m": float(gripper_width),
            "cameras": tuple(cameras),
        }

    def capture_fresh_sensor_frame_at_pose(
        self, gripper_command: float
    ) -> ControllerFeedback:
        """Dispatch exactly one zero-Cartesian action for a new RGB-D frame.

        Cache invalidation alone cannot make two observations temporally
        independent when the environment publishes camera data only after a
        policy action.  This narrow capability keeps the empty hand at its
        current public-proprio pose, preserves the requested gripper state,
        and advances the environment by one action.  It exposes neither the
        dispatcher's generation counter nor any reward / done / evaluator
        signal; the caller still has to verify that the next sensor timestamp
        actually advanced.
        """

        if gripper_command not in (-1.0, 1.0):
            return ControllerFeedback(
                False,
                "sensor refresh gripper command must be -1 or +1",
            )
        if self.steps_executed >= self.step_budget:
            # No action remains, so this is not a recoverable candidate-level
            # rejection.  Propagate the typed terminal signal consumed by
            # ``ContactAwareRouteCController.run``; otherwise the base
            # controller could schedule a retreat or another grasp attempt
            # despite having no executable policy step.
            raise PolicyStepBudgetExhausted(
                "episode OSC step budget exhausted before sensor refresh"
            )
        start_steps = self.steps_executed
        before = self.current_ee_pose()
        self._last_gripper_command = float(gripper_command)
        self._step(
            self._pose_action(
                before,
                before,
                float(gripper_command),
            )
        )
        after = self.current_ee_pose()
        position_error, rotation_error = self._pose_errors(after, before)
        accepted = bool(
            self.steps_executed == start_steps + 1
            and position_error <= self.config.position_tolerance_m
            and rotation_error <= self.config.rotation_tolerance_rad
        )
        self.phase_trace.append(
            {
                "phase": "sensor_refresh_hold",
                "policy_actions": self.steps_executed - start_steps,
                "gripper_command": float(gripper_command),
                "start_xyz_m": before[:3, 3].tolist(),
                "end_xyz_m": after[:3, 3].tolist(),
                "position_drift_m": position_error,
                "rotation_drift_rad": rotation_error,
                "accepted": accepted,
                "purpose": "advance_public_dual_rgbd_frame_at_fixed_pose",
            }
        )
        return ControllerFeedback(
            accepted,
            (
                "fresh sensor frame action dispatched"
                if accepted
                else "sensor refresh hold exceeded public pose drift gate"
            ),
            position_error,
            rotation_error,
        )

    @property
    def last_sensor_safe_view_motion_samples(
        self,
    ) -> tuple[dict[str, object], ...]:
        """Return sanitized per-action proprioception from the last view move."""

        return tuple(
            {
                "step": int(item["step"]),
                "ee_pose_world": np.asarray(
                    item["ee_pose_world"], dtype=np.float64
                ).copy(),
                "gripper_width_m": float(item["gripper_width_m"]),
            }
            for item in self._last_sensor_safe_view_motion_samples
        )

    def execute_sensor_safe_view_waypoints(
        self,
        poses_world: np.ndarray,
        gripper_command: float,
        *,
        maximum_policy_actions: int,
        reserved_followup_actions: int,
    ) -> ControllerFeedback:
        """Track every <=20-mm view waypoint under one pre-reserved budget.

        This narrow path exists because the ordinary phase executor follows a
        dense MPC chunk's endpoint.  A visibility-proof motion instead needs
        each frozen-SDF waypoint to be an actual commanded endpoint and needs
        every issued action's public proprioception available to the caller.
        """

        self._last_sensor_safe_view_motion_samples = ()
        poses = np.asarray(poses_world, dtype=np.float64)
        if (
            poses.ndim != 3
            or poses.shape[1:] != (4, 4)
            or len(poses) < 2
            or not np.all(np.isfinite(poses))
            or type(maximum_policy_actions) is not int
            or type(reserved_followup_actions) is not int
            or maximum_policy_actions < 1
            or reserved_followup_actions < 1
            or type(gripper_command) not in (int, float)
            or float(gripper_command) != -1.0
        ):
            return ControllerFeedback(False, "invalid sensor-safe view request")
        if any(
            not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6)
            or not np.allclose(
                pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=2e-3
            )
            or float(np.linalg.det(pose[:3, :3])) <= 0.0
            for pose in poses
        ):
            return ControllerFeedback(False, "malformed sensor-safe view pose")
        segment_lengths = np.linalg.norm(
            np.diff(poses[:, :3, 3], axis=0), axis=1
        )
        if (
            np.any(segment_lengths <= 0.0)
            or np.any(segment_lengths > 0.020 + 1e-12)
            or np.max(np.abs(poses[:, 2, 3] - poses[0, 2, 3])) > 1e-12
            or any(
                self._pose_errors(pose, poses[0])[1] > 1e-12
                for pose in poses[1:]
            )
        ):
            return ControllerFeedback(
                False, "sensor-safe view waypoints violate geometry bounds"
            )
        current = self.current_ee_pose()
        start_position_error, start_rotation_error = self._pose_errors(
            current, poses[0]
        )
        start_width = self._gripper_width()
        if (
            start_position_error > 0.003
            or start_rotation_error > 0.020
            or not np.isfinite(start_width)
            or start_width < 0.070
        ):
            return ControllerFeedback(
                False, "sensor-safe view start is not synchronized high/open"
            )
        # Reserve both the whole view transaction and its mandatory fresh-frame
        # hold before dispatching the first action.  Partial motion followed by
        # a known budget failure is never admissible evidence.
        if (
            self.step_budget - self.steps_executed
            < maximum_policy_actions + reserved_followup_actions
        ):
            return ControllerFeedback(
                False, "insufficient pre-reserved sensor-safe view action budget"
            )

        samples: list[dict[str, object]] = []
        start_steps = self.steps_executed
        self._last_gripper_command = -1.0
        last_position_error = start_position_error
        last_rotation_error = start_rotation_error
        for target in poses[1:]:
            reached = False
            while self.steps_executed - start_steps < maximum_policy_actions:
                current = self.current_ee_pose()
                last_position_error, last_rotation_error = self._pose_errors(
                    current, target
                )
                if (
                    last_position_error <= self.config.position_tolerance_m
                    and last_rotation_error <= 0.020
                ):
                    reached = True
                    break
                self._step(self._pose_action(current, target, -1.0))
                measured = self.current_ee_pose()
                width = self._gripper_width()
                samples.append(
                    {
                        "step": self.steps_executed,
                        "ee_pose_world": measured.copy(),
                        "gripper_width_m": width,
                    }
                )
                _, rotation_from_start = self._pose_errors(
                    measured, poses[0]
                )
                if (
                    not np.all(np.isfinite(measured))
                    or not np.isfinite(width)
                    or width < 0.070
                    or abs(float(measured[2, 3] - poses[0, 2, 3]))
                    > _FREE_RIM_SENSOR_SAFE_MAX_VERTICAL_DRIFT_M
                    or rotation_from_start > 0.020
                ):
                    self._last_sensor_safe_view_motion_samples = tuple(samples)
                    return ControllerFeedback(
                        False,
                        "sensor-safe view public-proprio/open gate failed",
                    )
            if not reached:
                current = self.current_ee_pose()
                last_position_error, last_rotation_error = self._pose_errors(
                    current, target
                )
                reached = bool(
                    last_position_error <= self.config.position_tolerance_m
                    and last_rotation_error <= 0.020
                )
            if not reached:
                self._last_sensor_safe_view_motion_samples = tuple(samples)
                return ControllerFeedback(
                    False,
                    "sensor-safe view waypoint tracking budget exhausted",
                    last_position_error,
                    last_rotation_error,
                )
        self._last_sensor_safe_view_motion_samples = tuple(samples)
        accepted = bool(
            0 < self.steps_executed - start_steps <= maximum_policy_actions
        )
        self.phase_trace.append(
            {
                "phase": "sensor_safe_view_waypoint_motion",
                "accepted": accepted,
                "policy_actions": self.steps_executed - start_steps,
                "commanded_waypoints": len(poses) - 1,
                "maximum_policy_actions": maximum_policy_actions,
                "reserved_followup_actions": reserved_followup_actions,
                "maximum_commanded_segment_m": float(np.max(segment_lengths)),
                "gripper_command": -1.0,
                "public_sample_count": len(samples),
            }
        )
        return ControllerFeedback(
            accepted,
            (
                "sensor-safe view waypoints tracked"
                if accepted
                else "sensor-safe view emitted no policy action"
            ),
            last_position_error,
            last_rotation_error,
        )

    def set_active_grasp_mode(self, mode: GraspMode) -> None:
        self._active_grasp_mode = mode

    @property
    def active_grasp_mode(self) -> GraspMode:
        return self._active_grasp_mode

    def set_active_grasp_candidate(self, candidate_id: str) -> None:
        candidate_id = str(candidate_id)
        if candidate_id != self._active_grasp_candidate_id:
            self._cavity_rim_preclosed = False
            self._cavity_rim_load_baseline_width_m = None
            self._cavity_rim_load_proof_passed = False
            self._cavity_rim_load_proof_failed = False
            self._active_cavity_rim_execution_profile = None
        self._active_grasp_candidate_id = candidate_id

    @property
    def cavity_rim_load_proof_failed(self) -> bool:
        """Whether the active candidate lost its load using public proprio."""

        return self._cavity_rim_load_proof_failed

    def set_cavity_rim_execution_profile(
        self, profile: Mapping[str, object] | None
    ) -> None:
        """Install narrow sensor geometry for a typed recovery candidate.

        The provider supplies only its frozen/fresh RGB-D centre, radial line,
        and radii.  No simulator body, contact, reward, task id, or evaluator
        capability crosses this boundary.
        """

        if profile is None:
            self._active_cavity_rim_execution_profile = None
            return
        candidate_id = str(profile.get("candidate_id", ""))
        if candidate_id != self._active_grasp_candidate_id:
            raise ValueError("cavity execution profile does not match candidate")
        center = np.asarray(profile.get("source_center_world_m"), dtype=np.float64)
        direction = np.asarray(
            profile.get("radial_direction_world"), dtype=np.float64
        )
        if (
            center.shape != (3,)
            or direction.shape != (3,)
            or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(direction))
        ):
            raise ValueError("cavity execution profile vectors must be finite xyz")
        direction[2] = 0.0
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm < 0.80:
            raise ValueError("cavity execution radial direction must be planar")
        direction /= direction_norm
        observed_radius = float(profile.get("observed_rim_radius_m", np.nan))
        target_radius = float(profile.get("target_radius_m", np.nan))
        profile_kind = str(profile.get("profile_kind", ""))
        direct_free_space = profile_kind == "free_space_rim_direct_load"
        maximum_seat_radius = (
            None
            if direct_free_space
            else float(profile.get("maximum_seat_radius_m", np.nan))
        )
        radius_consistent = bool(
            np.isfinite(observed_radius)
            and np.isfinite(target_radius)
            and 0.0 < target_radius < observed_radius
            and (
                direct_free_space
                or (
                    maximum_seat_radius is not None
                    and np.isfinite(maximum_seat_radius)
                    and observed_radius < maximum_seat_radius
                )
            )
        )
        if not radius_consistent:
            raise ValueError("cavity execution profile radii are inconsistent")
        if direct_free_space and not candidate_id.startswith("analytic-rim-"):
            raise ValueError(
                "free-space direct-load profile requires an external rim candidate"
            )
        self._active_cavity_rim_execution_profile = {
            "candidate_id": candidate_id,
            "source_center_world_m": center.copy(),
            "radial_direction_world": direction.copy(),
            "observed_rim_radius_m": observed_radius,
            "target_radius_m": target_radius,
            "maximum_seat_radius_m": maximum_seat_radius,
            "profile_kind": profile_kind,
        }

    def preclose_cavity_rim(self) -> ControllerFeedback:
        """Narrow an open hand at a sensor-bound rim pregrasp."""

        if (
            self._active_grasp_mode != GraspMode.RIM_PINCH
            or not str(self._active_grasp_candidate_id).startswith(
                ("analytic-cavity-rim-", "analytic-rim-")
            )
        ):
            return ControllerFeedback(False, "rim preclose is not active")
        free_space = str(self._active_grasp_candidate_id).startswith(
            "analytic-rim-"
        )
        target_width = (
            self.config.rim_free_space_preclose_target_width_m
            if free_space
            else self.config.rim_cavity_preclose_target_width_m
        )
        minimum_width = (
            self.config.rim_free_space_preclose_min_width_m
            if free_space
            else self.config.rim_cavity_preclose_min_width_m
        )
        maximum_width = (
            self.config.rim_free_space_preclose_max_width_m
            if free_space
            else self.config.rim_cavity_preclose_max_width_m
        )
        maximum_steps = (
            self.config.rim_free_space_preclose_max_steps
            if free_space
            else self.config.rim_cavity_preclose_max_steps
        )
        width_before = self._gripper_width()
        executed_steps = 0
        while (
            self._gripper_width() > maximum_width
            and executed_steps < maximum_steps
        ):
            if self.steps_executed >= self.step_budget:
                return ControllerFeedback(False, "episode OSC step budget exhausted")
            self._step(OSCAction.hold(1.0))
            executed_steps += 1
        width_after = self._gripper_width()
        accepted = bool(
            minimum_width <= width_after <= maximum_width
        )
        self._cavity_rim_preclosed = accepted
        self.phase_trace.append(
            {
                "phase": "rim_safe_preclose",
                "active_grasp_candidate": self._active_grasp_candidate_id,
                "preclose_profile": (
                    "free_space_wide_capture"
                    if free_space
                    else "cavity_narrow_clearance"
                ),
                "preclose_steps": executed_steps,
                "target_width_m": target_width,
                "accepted_width_min_m": minimum_width,
                "accepted_width_max_m": maximum_width,
                "gripper_width_before_m": width_before,
                "gripper_width_after_m": width_after,
                "accepted": accepted,
            }
        )
        return ControllerFeedback(
            accepted,
            (
                "cavity rim preclose reached width window"
                if accepted
                else "cavity rim preclose missed width window"
            ),
        )

    @property
    def step_budget_exhausted(self) -> bool:
        return self.steps_executed >= self.step_budget

    def set_tentative_expansion_allowed(self, allowed: bool) -> None:
        """Permit a wide EXPAND width only as provisional GRASP evidence."""

        self._allow_tentative_expansion = bool(allowed)

    def expansion_width_is_tentative(self) -> bool:
        width = self._gripper_width()
        return bool(
            self._last_gripper_command == -1.0
            and self.config.expansion_max_width_m
            < width
            <= self.config.expansion_physical_max_width_m
        )

    def _finish_failed_cavity_load_proof(
        self,
        trace: dict[str, Any],
        failure_kind: str,
    ) -> ControllerFeedback:
        """Record a failed public-proprio load proof and open immediately."""

        self._cavity_rim_load_proof_failed = True
        self._cavity_rim_load_proof_passed = False
        trace.update({"accepted": False, "failure_kind": failure_kind})
        self.phase_trace.append(trace)
        # Release before any task-level retreat.  If evaluator ownership ends
        # the episode while applying this action, its RuntimeError sentinel is
        # deliberately allowed to propagate and is never converted into a
        # candidate failure or another action.
        released = self.set_gripper(-1.0)
        release_width_m = self._gripper_width()
        release_open_proven = bool(
            released.accepted
            and release_width_m > self.config.pinch_max_width_m
        )
        trace["immediate_release_width_m"] = release_width_m
        trace["immediate_release_open_proven"] = release_open_proven
        if not release_open_proven:
            # A generic rim RELEASE historically reports command acceptance
            # even when its full pulse cannot visibly open the fingers.  That
            # is sufficient for a normal episode handoff, but not for this
            # recovery: moving to a fresh high view with a still-pinched or
            # mechanically wedged bowl can drag it through the drawer wall.
            # Stop before candidate marking, visual rebind, or retreat.  A
            # consumed action budget retains its existing typed termination.
            if (
                failure_kind == "step_budget_exhausted"
                or self.step_budget_exhausted
            ):
                raise PolicyStepBudgetExhausted(
                    "episode OSC step budget exhausted"
                )
            raise CavityLoadProofReleaseUnconfirmed(
                "cavity rim load-proof release was not confirmed by "
                "gripper proprioception"
            )
        detail = (
            "episode OSC step budget exhausted"
            if failure_kind == "step_budget_exhausted"
            else "cavity rim load proof rejected the grasp"
        )
        if not released.accepted:
            detail = f"{detail}; immediate release failed: {released.detail}"
        return ControllerFeedback(False, detail)

    def _run_cavity_rim_load_proof(self) -> ControllerFeedback:
        """Apply a typed vertical load before the controller's full LIFT.

        The outer Route-C controller freezes the full LIFT goal before this
        method is called.  Returning one accepted partial chunk therefore
        makes the next MPC replan continue toward that same absolute goal;
        the proof displacement is never added to the configured full lift.
        """

        candidate_id = str(self._active_grasp_candidate_id)
        profile = self._active_cavity_rim_execution_profile
        free_space = bool(
            candidate_id.startswith("analytic-rim-")
            and profile is not None
            and profile.get("candidate_id") == candidate_id
            and profile.get("profile_kind") == "free_space_rim_direct_load"
        )
        lift_m = (
            self.config.rim_free_space_load_proof_lift_m
            if free_space
            else self.config.rim_cavity_load_proof_lift_m
        )
        lift_steps = (
            self.config.rim_free_space_load_proof_steps
            if free_space
            else self.config.rim_cavity_load_proof_steps
        )
        hold_steps = (
            self.config.rim_free_space_load_proof_hold_steps
            if free_space
            else self.config.rim_cavity_load_proof_hold_steps
        )
        minimum_lift_m = (
            self.config.rim_free_space_load_proof_min_lift_m
            if free_space
            else self.config.rim_cavity_load_proof_min_lift_m
        )
        maximum_lift_m = (
            self.config.rim_free_space_load_proof_max_lift_m
            if free_space
            else self.config.rim_cavity_load_proof_max_lift_m
        )
        minimum_width_m = (
            self.config.rim_free_space_load_proof_min_width_m
            if free_space
            else self.config.rim_pinch_blocked_min_width_m
        )
        maximum_hold_span_m = (
            self.config.rim_free_space_load_proof_max_hold_width_span_m
            if free_space
            else self.config.rim_cavity_load_proof_max_hold_width_span_m
        )
        baseline = self._cavity_rim_load_baseline_width_m
        start = self.current_ee_pose()
        goal = start.copy()
        goal[2, 3] += lift_m
        trace: dict[str, Any] = {
            "phase": (
                "rim_free_space_load_proof"
                if free_space
                else "rim_cavity_load_proof"
            ),
            "active_grasp_candidate": self._active_grasp_candidate_id,
            "strategy": "vertical_micro_lift_then_loaded_hold",
            "start_position_world_m": start[:3, 3].tolist(),
            "commanded_goal_xyz_m": goal[:3, 3].tolist(),
            "commanded_lift_m": lift_m,
            "minimum_actual_lift_m": minimum_lift_m,
            "maximum_actual_lift_m": maximum_lift_m,
            "lift_steps_required": lift_steps,
            "hold_steps_required": hold_steps,
            "settle_steps_limit": 8 if free_space else 0,
            "baseline_width_m": baseline,
            "minimum_blocked_width_m": minimum_width_m,
            "minimum_width_is_strict": free_space,
            "maximum_width_loss_m": (
                None
                if free_space
                else self.config.rim_cavity_load_proof_max_width_loss_m
            ),
            "maximum_hold_width_span_m": maximum_hold_span_m,
            "hold_pose_span_role": "diagnostic_only_not_contact_evidence",
            "samples": [],
        }
        baseline_valid = bool(
            baseline is not None
            and (
                minimum_width_m < baseline
                if free_space
                else minimum_width_m <= baseline
            )
            and baseline < self.config.pinch_max_width_m
        )
        if not baseline_valid:
            return self._finish_failed_cavity_load_proof(
                trace, "missing_confirmed_close_or_seat_baseline"
            )
        assert baseline is not None

        samples = trace["samples"]
        assert isinstance(samples, list)
        hold_positions: list[np.ndarray] = []
        hold_widths: list[float] = []
        for stage, count in (
            ("lift", lift_steps),
            ("settle", 8 if free_space else 0),
            ("hold", hold_steps),
        ):
            for stage_index in range(count):
                # OSC does not instantaneously reach each commanded pose.
                # Wait for the measured micro-lift before beginning the
                # loaded hold, keeping the same absolute goal and closed jaw.
                # An unresponsive/blocked lift still exhausts this finite
                # settling window and fails the final displacement check.
                if stage == "settle" and (
                    self.current_ee_pose()[2, 3] - start[2, 3] >= minimum_lift_m
                ):
                    break
                if self.steps_executed >= self.step_budget:
                    trace["lift_steps_executed"] = sum(
                        sample["stage"] == "lift" for sample in samples
                    )
                    trace["hold_steps_executed"] = sum(
                        sample["stage"] == "hold" for sample in samples
                    )
                    return self._finish_failed_cavity_load_proof(
                        trace, "step_budget_exhausted"
                    )
                current = self.current_ee_pose()
                stage_goal = goal.copy()
                if stage == "lift":
                    # Apply the load progressively across the configured
                    # driven ticks.  Repeating the final 12-mm target would be
                    # a one-tick jump in an ideal OSC plant followed by four
                    # nominal "lift" holds, defeating the mechanical proof's
                    # intended gradual loading semantics.
                    stage_goal[2, 3] = start[2, 3] + (
                        lift_m
                        * (stage_index + 1)
                        / lift_steps
                    )
                width_before = self._gripper_width()
                self._step(self._pose_action(current, stage_goal, 1.0))
                updated = self.current_ee_pose()
                width_after = self._gripper_width()
                width_loss = float(baseline - width_after)
                sample = {
                    "step": self.steps_executed,
                    "stage": stage,
                    "stage_index": stage_index,
                    "stage_goal_xyz_m": stage_goal[:3, 3].tolist(),
                    "position_world_m": updated[:3, 3].tolist(),
                    "actual_lift_m": float(updated[2, 3] - start[2, 3]),
                    "width_before_m": width_before,
                    "width_after_m": width_after,
                    "width_loss_from_baseline_m": width_loss,
                }
                samples.append(sample)
                if stage == "hold":
                    hold_positions.append(updated[:3, 3].copy())
                    hold_widths.append(width_after)
                width_too_small = bool(
                    width_after <= minimum_width_m
                    if free_space
                    else width_after < minimum_width_m
                )
                if width_too_small:
                    return self._finish_failed_cavity_load_proof(
                        trace, "blocked_width_below_absolute_minimum"
                    )
                if width_after >= self.config.pinch_max_width_m:
                    return self._finish_failed_cavity_load_proof(
                        trace, "blocked_width_above_physical_maximum"
                    )
                if (
                    not free_space
                    and
                    width_loss
                    > self.config.rim_cavity_load_proof_max_width_loss_m
                    + 1e-12
                ):
                    return self._finish_failed_cavity_load_proof(
                        trace, "width_loss_exceeded_baseline_limit"
                    )

        final = self.current_ee_pose()
        actual_lift = float(final[2, 3] - start[2, 3])
        hold_width_span = float(np.ptp(np.asarray(hold_widths)))
        hold_points = np.asarray(hold_positions, dtype=np.float64)
        hold_pairwise = hold_points[:, None, :] - hold_points[None, :, :]
        hold_pose_span = float(
            np.max(np.linalg.norm(hold_pairwise, axis=2))
        )
        trace.update(
            {
                "lift_steps_executed": lift_steps,
                "hold_steps_executed": hold_steps,
                "settle_steps_executed": sum(sample["stage"] == "settle" for sample in samples),
                "actual_lift_m": actual_lift,
                "final_width_m": hold_widths[-1],
                "final_width_loss_from_baseline_m": (
                    baseline - hold_widths[-1]
                ),
                "hold_width_span_m": hold_width_span,
                "hold_pose_span_m": hold_pose_span,
            }
        )
        if not (
            minimum_lift_m <= actual_lift <= maximum_lift_m
        ):
            return self._finish_failed_cavity_load_proof(
                trace, "micro_lift_displacement_out_of_range"
            )
        if (
            hold_width_span
            > maximum_hold_span_m + 1e-12
        ):
            return self._finish_failed_cavity_load_proof(
                trace, "loaded_hold_width_was_not_stable"
            )
        self._cavity_rim_load_proof_passed = True
        self._cavity_rim_load_proof_failed = False
        trace["accepted"] = True
        trace["full_lift_anchor_semantics"] = "continue_same_absolute_goal"
        self.phase_trace.append(trace)
        return ControllerFeedback(
            True,
            (
                "free-space rim load proof passed"
                if free_space
                else "cavity rim load proof passed"
            ),
        )

    def execute_waypoints(
        self, poses_world: np.ndarray, phase: Phase, gripper_command: float
    ) -> ControllerFeedback:
        poses = np.asarray(poses_world, dtype=np.float64)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 1:
            return ControllerFeedback(False, "optimizer waypoints must have shape (N,4,4)")
        if not np.all(np.isfinite(poses)):
            return ControllerFeedback(False, "optimizer returned non-finite waypoints")
        self._last_gripper_command = float(gripper_command)
        if phase == Phase.GRASP:
            # A new grasp may not inherit post-LIFT evidence from a previous
            # task attempt or candidate.
            self._expand_retention_confirmed = False
            self._rim_retention_confirmed = False
            self._rim_grasp_marginal = False
            self._cavity_rim_load_baseline_width_m = None
            self._cavity_rim_load_proof_passed = False
            self._cavity_rim_load_proof_failed = False
        if phase != Phase.GRASP:
            self._grasp_contact_reached = False
        if phase != Phase.PLACE:
            self._place_contact_reached = False
        start_steps = self.steps_executed
        optimizer_waypoint_count = len(poses)
        axisymmetric_orientation = self._active_grasp_mode == GraspMode.EXPAND
        rim_width_servo_samples: list[dict[str, float | int]] = []
        cavity_rim_candidate = str(self._active_grasp_candidate_id).startswith(
            "analytic-cavity-rim-"
        )
        free_space_rim_candidate = str(
            self._active_grasp_candidate_id
        ).startswith("analytic-rim-")
        free_space_direct_load_candidate = bool(
            free_space_rim_candidate
            and self._active_cavity_rim_execution_profile is not None
            and self._active_cavity_rim_execution_profile.get("candidate_id")
            == self._active_grasp_candidate_id
            and self._active_cavity_rim_execution_profile.get("profile_kind")
            == "free_space_rim_direct_load"
        )
        if (
            phase == Phase.LIFT
            and self._active_grasp_mode == GraspMode.RIM_PINCH
            and (cavity_rim_candidate or free_space_direct_load_candidate)
            and not self._cavity_rim_load_proof_passed
        ):
            return self._run_cavity_rim_load_proof()
        cavity_rim_grasp = bool(
            phase == Phase.GRASP
            and self._cavity_rim_preclosed
            and (cavity_rim_candidate or free_space_rim_candidate)
        )
        rim_width_servo_target_m = (
            self.config.rim_free_space_preclose_target_width_m
            if free_space_rim_candidate
            else self.config.rim_cavity_preclose_target_width_m
        )
        if cavity_rim_grasp:
            # Freeze the sensor-derived goal for the subsequent close pulse
            # even when Cartesian tracking converges normally.  Previously
            # only the compliant-stall completion path populated this target.
            # Both cavity and free-space rim grasps now preserve their
            # pre-shaped descent target while the pads close.
            self._grasp_contact_servo_goal = poses[-1].copy()
        trace_context = {
            "grasp_mode": self._active_grasp_mode.value,
            "active_grasp_candidate": self._active_grasp_candidate_id,
            "optimizer_goal_xyz_m": poses[-1, :3, 3].tolist(),
            "rotation_error_kind": (
                "tool_axis" if axisymmetric_orientation else "full_pose"
            ),
        }
        if cavity_rim_grasp:
            trace_context["rim_width_servo_target_m"] = (
                rim_width_servo_target_m
            )
            trace_context["rim_width_servo_samples"] = rim_width_servo_samples
        if phase == Phase.APPROACH and cavity_rim_candidate:
            trace_context["rim_cavity_pregrasp_below_tolerance_m"] = (
                self.config.rim_cavity_pregrasp_below_tolerance_m
            )
        # A chunk contains the current pose followed by a short optimiser
        # prefix.  Tracking every dense interpolation sample through a 20 Hz
        # impedance controller wastes most of an episode.  For ordinary phases
        # the safe prefix endpoint is sufficient; transfer retains a midpoint
        # as well so an obstacle-avoiding bend is not shortcut.  A retained
        # rim placement first aligns XY / orientation at the measured transfer
        # height.  Descending while still correcting lateral error lets a bowl
        # catch a plate edge and converts that correction into a sideways
        # contact force.
        if phase == Phase.TRANSFER and len(poses) >= 4:
            tracked = poses[[len(poses) // 2, len(poses) - 1]]
        elif (
            phase == Phase.PLACE
            and self._active_grasp_mode == GraspMode.RIM_PINCH
            and self._rim_retention_confirmed
        ):
            final_target = poses[-1].copy()
            planar_alignment = final_target.copy()
            planar_alignment[2, 3] = self.current_ee_pose()[2, 3]
            tracked = np.stack((planar_alignment, final_target))
        else:
            tracked = poses[-1:]
        for target_index, target in enumerate(tracked):
            position_tolerance = (
                self.config.position_tolerance_m
                if target_index == len(tracked) - 1
                else (
                    self.config.rim_intermediate_position_tolerance_m
                    if self._active_grasp_mode == GraspMode.RIM_PINCH
                    else self.config.intermediate_position_tolerance_m
                )
            )
            reached = False
            initial_pose = self.current_ee_pose()
            initial_position_error, initial_rotation_error = self._pose_errors(
                initial_pose,
                target,
                axisymmetric=axisymmetric_orientation,
            )
            initial_goal_minus_current = target[:3, 3] - initial_pose[:3, 3]

            def typed_mode_completion(
                current_pose: np.ndarray,
                current_position_error: float,
                current_rotation_error: float,
            ) -> ControllerFeedback | None:
                residual = target[:3, 3] - current_pose[:3, 3]
                completion_kind: str | None = None
                if self._active_grasp_mode == GraspMode.RIM_PINCH:
                    below_tolerance = (
                        self.config.rim_cavity_pregrasp_below_tolerance_m
                        if cavity_rim_candidate
                        else 0.0
                    )
                    if (
                        phase == Phase.APPROACH
                        and float(np.linalg.norm(residual[:2]))
                        <= self.config.rim_pregrasp_xy_tolerance_m
                        and -below_tolerance <= -float(residual[2])
                        <= self.config.rim_pregrasp_vertical_clearance_m
                        and current_rotation_error
                        <= self.config.rim_pregrasp_orientation_tolerance_rad
                    ):
                        self._rim_safe_pregrasp_reached = True
                        completion_kind = "rim_safe_pregrasp"
                    elif (
                        phase == Phase.LIFT
                        and current_position_error
                        <= self.config.rim_lift_position_tolerance_m
                        and current_rotation_error
                        <= self.config.rim_lift_orientation_tolerance_rad
                    ):
                        self._rim_lift_tolerance_reached = True
                        completion_kind = "rim_lift_pose_tolerance"
                    elif (
                        phase == Phase.TRANSFER
                        and target_index == len(tracked) - 1
                        and self._rim_retention_confirmed
                        and current_position_error
                        <= self.config.rim_transfer_position_tolerance_m
                        and float(np.linalg.norm(residual[:2]))
                        <= self.config.rim_transfer_planar_tolerance_m
                        and current_rotation_error
                        <= self.config.rim_transfer_orientation_tolerance_rad
                    ):
                        self._rim_transfer_tolerance_reached = True
                        completion_kind = "rim_transfer_pose_tolerance"
                    else:
                        return None
                elif self._active_grasp_mode != GraspMode.EXPAND:
                    return None
                elif (
                    phase == Phase.LIFT
                    and current_position_error
                    <= self.config.expand_lift_position_tolerance_m
                    and current_rotation_error
                    <= self.config.expand_lift_tool_axis_tolerance_rad
                ):
                    self._expand_lift_tolerance_reached = True
                    completion_kind = "expand_lift_pose_tolerance"
                elif (
                    phase == Phase.APPROACH
                    and float(np.linalg.norm(residual[:2]))
                    <= self.config.expand_pregrasp_xy_tolerance_m
                    and 0.0 <= -float(residual[2])
                    <= self.config.expand_pregrasp_vertical_clearance_m
                    and current_rotation_error
                    <= self.config.expand_pregrasp_tool_axis_tolerance_rad
                ):
                    # Only a pose vertically above the sensor-bound goal is a
                    # safe pregrasp.  Being below the goal or laterally offset
                    # remains a hard rejection even when the norm is small.
                    self._expand_safe_pregrasp_reached = True
                    completion_kind = "expand_safe_pregrasp"
                elif (
                    phase == Phase.TRANSFER
                    and target_index == len(tracked) - 1
                    and self._expand_retention_confirmed
                    and current_position_error
                    <= self.config.expand_transfer_position_tolerance_m
                    and current_rotation_error
                    <= self.config.expand_transfer_tool_axis_tolerance_rad
                ):
                    # This relaxed endpoint is available only after the
                    # post-LIFT proprio/fresh-RGB-D retention gate.  PLACE
                    # remains governed by its own precise/contact completion.
                    self._expand_transfer_tolerance_reached = True
                    completion_kind = "expand_transfer_pose_tolerance"
                if completion_kind is None:
                    return None
                self.phase_trace.append(
                    {
                        "phase": phase.value,
                        "osc_steps": self.steps_executed - start_steps,
                        "optimizer_waypoints": optimizer_waypoint_count,
                        "tracked_waypoints": target_index + 1,
                        "partial_progress": False,
                        "typed_completion": completion_kind,
                        "initial_position_error_m": initial_position_error,
                        "initial_rotation_error_rad": initial_rotation_error,
                        "final_position_error_m": current_position_error,
                        "final_rotation_error_rad": current_rotation_error,
                        "initial_goal_minus_current_xyz_m": (
                            initial_goal_minus_current.tolist()
                        ),
                        "final_goal_minus_current_xyz_m": residual.tolist(),
                        "gripper_width_m": self._gripper_width(),
                        **trace_context,
                    }
                )
                return ControllerFeedback(
                    True,
                    f"typed {completion_kind} completion",
                    current_position_error,
                    current_rotation_error,
                )

            position_history = [initial_position_error]
            cartesian_history = [initial_pose[:3, 3].copy()]
            turn_progress_history: list[float] = []
            turn_width_history: list[float] = []
            if self._turn_contact_enabled and phase == Phase.TRANSFER:
                progress = self._turn_contact_progress(initial_pose)
                self._record_turn_progress(progress)
                turn_progress_history.append(progress)
                turn_width_history.append(self._gripper_width())

            def typed_turn_contact_stall_completion(
                current_pose: np.ndarray,
                current_position_error: float,
                current_rotation_error: float,
            ) -> ControllerFeedback | None:
                if (
                    phase != Phase.TRANSFER
                    or not self._turn_contact_enabled
                    or self._turn_contact_start_pose is None
                    or self._turn_contact_axis_world is None
                    or self._active_grasp_mode != GraspMode.PINCH
                    or self._last_gripper_command != 1.0
                ):
                    return None
                window_steps = self.config.turn_contact_window_steps
                if (
                    len(turn_progress_history) < window_steps
                    or len(turn_width_history) < window_steps
                    or len(cartesian_history) < window_steps
                ):
                    return None
                progress_window = np.asarray(
                    turn_progress_history[-window_steps:], dtype=np.float64
                )
                width_window = np.asarray(
                    turn_width_history[-window_steps:], dtype=np.float64
                )
                contact_window = cartesian_history[-window_steps:]
                cartesian_span = max(
                    float(np.linalg.norm(first - second))
                    for first in contact_window
                    for second in contact_window
                )
                rotation_span = float(np.ptp(progress_window))
                width_span = float(np.ptp(width_window))
                progress = float(progress_window[-1])
                contact_position_error = float(
                    np.linalg.norm(
                        current_pose[:3, 3]
                        - self._turn_contact_start_pose[:3, 3]
                    )
                )
                width = float(width_window[-1])
                blocked_width = bool(
                    self.config.pinch_blocked_min_width_m
                    <= width
                    <= self.config.pinch_max_width_m
                )
                if not (
                    self._turn_contact_monotonic
                    and progress >= self._turn_contact_min_rotation_rad
                    and blocked_width
                    and contact_position_error
                    <= self.config.turn_contact_position_tolerance_m
                    and cartesian_span
                    <= self.config.turn_contact_cartesian_span_m
                    and rotation_span
                    <= self.config.turn_contact_rotation_span_rad
                    and width_span <= self.config.turn_contact_width_span_m
                ):
                    return None
                self._turn_contact_reached = True
                self._turn_contact_progress_rad = progress
                self.phase_trace.append(
                    {
                        "phase": phase.value,
                        "osc_steps": self.steps_executed - start_steps,
                        "optimizer_waypoints": optimizer_waypoint_count,
                        "tracked_waypoints": target_index + 1,
                        "partial_progress": False,
                        "contact_reached": True,
                        "typed_completion": "stove_knob_mechanical_stop",
                        "signed_rotation_progress_rad": progress,
                        "minimum_signed_rotation_rad": (
                            self._turn_contact_min_rotation_rad
                        ),
                        "contact_position_error_m": contact_position_error,
                        "recent_cartesian_span_m": cartesian_span,
                        "recent_rotation_span_rad": rotation_span,
                        "recent_width_span_m": width_span,
                        "blocked_width_confirmed": blocked_width,
                        "monotonic_direction_confirmed": (
                            self._turn_contact_monotonic
                        ),
                        "initial_position_error_m": initial_position_error,
                        "initial_rotation_error_rad": initial_rotation_error,
                        "final_position_error_m": current_position_error,
                        "final_rotation_error_rad": current_rotation_error,
                        "gripper_width_m": width,
                        **trace_context,
                    }
                )
                return ControllerFeedback(
                    True,
                    "typed stove_knob_mechanical_stop completion",
                    current_position_error,
                    current_rotation_error,
                )

            def typed_expand_grasp_stall_completion(
                current_pose: np.ndarray,
                current_position_error: float,
                current_rotation_error: float,
            ) -> ControllerFeedback | None:
                if (
                    phase != Phase.GRASP
                    or self._active_grasp_mode != GraspMode.EXPAND
                    or not self._grasp_contact_enabled
                ):
                    return None
                contact_window = cartesian_history[
                    -self.config.grasp_contact_window_steps :
                ]
                if len(contact_window) < self.config.grasp_contact_window_steps:
                    return None
                stall_span = max(
                    float(np.linalg.norm(first - second))
                    for first in contact_window
                    for second in contact_window
                )
                residual = target[:3, 3] - current_pose[:3, 3]
                planar_residual = float(np.linalg.norm(residual[:2]))
                if not (
                    stall_span <= self.config.grasp_contact_stall_epsilon_m
                    and planar_residual
                    <= self.config.expand_grasp_xy_tolerance_m
                    and -self.config.expand_grasp_vertical_residual_m
                    <= float(residual[2])
                    <= 0.0
                    and current_rotation_error
                    <= self.config.expand_grasp_tool_axis_tolerance_rad
                ):
                    return None
                self._grasp_contact_reached = True
                self.phase_trace.append(
                    {
                        "phase": phase.value,
                        "osc_steps": self.steps_executed - start_steps,
                        "optimizer_waypoints": optimizer_waypoint_count,
                        "tracked_waypoints": target_index + 1,
                        "partial_progress": False,
                        "contact_reached": True,
                        "typed_completion": "expand_grasp_stall_contact",
                        "contact_residual_m": current_position_error,
                        "contact_xy_residual_m": planar_residual,
                        "contact_signed_z_residual_m": float(residual[2]),
                        "stall_window_span_m": stall_span,
                        "initial_position_error_m": initial_position_error,
                        "initial_rotation_error_rad": initial_rotation_error,
                        "final_position_error_m": current_position_error,
                        "final_rotation_error_rad": current_rotation_error,
                        "initial_goal_minus_current_xyz_m": (
                            initial_goal_minus_current.tolist()
                        ),
                        "final_goal_minus_current_xyz_m": residual.tolist(),
                        "gripper_width_m": self._gripper_width(),
                        **trace_context,
                    }
                )
                return ControllerFeedback(
                    True,
                    "typed expand_grasp_stall_contact completion",
                    current_position_error,
                    current_rotation_error,
                )

            def typed_rim_place_stall_completion(
                current_pose: np.ndarray,
                current_position_error: float,
                current_rotation_error: float,
            ) -> ControllerFeedback | None:
                if (
                    phase != Phase.PLACE
                    or self._active_grasp_mode != GraspMode.RIM_PINCH
                    or not self._rim_retention_confirmed
                    or self._last_gripper_command != 1.0
                ):
                    return None
                progress_window = position_history[
                    -self.config.grasp_contact_window_steps :
                ]
                if len(progress_window) < self.config.grasp_contact_window_steps:
                    return None
                progress_span = max(progress_window) - min(progress_window)
                contact_window = cartesian_history[
                    -self.config.grasp_contact_window_steps :
                ]
                cartesian_span = max(
                    float(np.linalg.norm(first - second))
                    for first in contact_window
                    for second in contact_window
                )
                residual = target[:3, 3] - current_pose[:3, 3]
                planar_residual = float(np.linalg.norm(residual[:2]))
                vertical_residual = -float(residual[2])
                gripper_width = self._gripper_width()
                blocked_width = bool(
                    self.config.rim_pinch_blocked_min_width_m
                    <= gripper_width
                    <= self.config.pinch_max_width_m
                )
                downward_nearly_complete = bool(
                    float(initial_goal_minus_current[2]) < 0.0
                    and 0.0
                    <= vertical_residual
                    <= self.config.rim_place_contact_max_vertical_residual_m
                )
                if not (
                    blocked_width
                    and downward_nearly_complete
                    and planar_residual
                    <= self.config.rim_place_contact_xy_tolerance_m
                    and current_rotation_error
                    <= self.config.rim_place_contact_orientation_tolerance_rad
                    and progress_span
                    <= self.config.rim_place_contact_stall_epsilon_m
                    and cartesian_span
                    <= self.config.rim_place_contact_cartesian_span_m
                ):
                    return None
                self._place_contact_reached = True
                self.phase_trace.append(
                    {
                        "phase": phase.value,
                        "osc_steps": self.steps_executed - start_steps,
                        "optimizer_waypoints": optimizer_waypoint_count,
                        "tracked_waypoints": target_index + 1,
                        "partial_progress": False,
                        "contact_reached": True,
                        "typed_completion": "rim_place_stall_contact",
                        "contact_residual_m": current_position_error,
                        "contact_xy_residual_m": planar_residual,
                        "contact_vertical_residual_m": vertical_residual,
                        "stall_window_span_m": progress_span,
                        "stall_cartesian_span_m": cartesian_span,
                        "retained_width_confirmed": blocked_width,
                        "initial_position_error_m": initial_position_error,
                        "initial_rotation_error_rad": initial_rotation_error,
                        "final_position_error_m": current_position_error,
                        "final_rotation_error_rad": current_rotation_error,
                        "initial_goal_minus_current_xyz_m": (
                            initial_goal_minus_current.tolist()
                        ),
                        "final_goal_minus_current_xyz_m": residual.tolist(),
                        "gripper_width_m": gripper_width,
                        **trace_context,
                    }
                )
                return ControllerFeedback(
                    True,
                    "typed rim_place_stall_contact completion",
                    current_position_error,
                    current_rotation_error,
                )


            for _ in range(self.config.max_steps_per_waypoint):
                current = self.current_ee_pose()
                position_error, rotation_error = self._pose_errors(
                    current,
                    target,
                    axisymmetric=axisymmetric_orientation,
                )
                if (
                    position_error <= position_tolerance
                    and rotation_error <= self.config.rotation_tolerance_rad
                ):
                    reached = True
                    break
                typed_feedback = typed_mode_completion(
                    current, position_error, rotation_error
                )
                if typed_feedback is not None:
                    return typed_feedback
                typed_feedback = typed_turn_contact_stall_completion(
                    current, position_error, rotation_error
                )
                if typed_feedback is not None:
                    return typed_feedback
                if self.steps_executed >= self.step_budget:
                    self.phase_trace.append(
                        {
                            "phase": phase.value,
                            "osc_steps": self.steps_executed - start_steps,
                            "optimizer_waypoints": optimizer_waypoint_count,
                            "tracked_waypoints": target_index + 1,
                            "partial_progress": False,
                            "accepted": False,
                            "failure_kind": "step_budget_exhausted",
                            "initial_position_error_m": initial_position_error,
                            "initial_rotation_error_rad": initial_rotation_error,
                            "final_position_error_m": position_error,
                            "final_rotation_error_rad": rotation_error,
                            "initial_goal_minus_current_xyz_m": (
                                initial_goal_minus_current.tolist()
                            ),
                            "final_goal_minus_current_xyz_m": (
                                target[:3, 3] - current[:3, 3]
                            ).tolist(),
                            "gripper_width_m": self._gripper_width(),
                            **trace_context,
                        }
                    )
                    return ControllerFeedback(
                        False,
                        "episode OSC step budget exhausted",
                        position_error,
                        rotation_error,
                    )
                step_gripper_command = float(gripper_command)
                width_before_step: float | None = None
                if phase == Phase.GRASP and self._cavity_rim_preclosed:
                    width_before_step = self._gripper_width()
                    # Robosuite integrates Panda gripper commands; zero keeps
                    # the previous closing force and collapsed the 20-mm
                    # pre-shape to ~1 mm.  Reverse the sign around the measured
                    # 20-mm target on every OSC step instead.
                    step_gripper_command = (
                        1.0
                        if width_before_step
                        > rim_width_servo_target_m
                        else -1.0
                    )
                self._step(
                    self._pose_action(
                        current,
                        target,
                        step_gripper_command,
                    )
                )
                if width_before_step is not None:
                    rim_width_servo_samples.append(
                        {
                            "step": self.steps_executed,
                            "width_before_m": width_before_step,
                            "command": step_gripper_command,
                            "width_after_m": self._gripper_width(),
                        }
                    )
                updated_pose = self.current_ee_pose()
                updated_position_error, updated_rotation_error = self._pose_errors(
                    updated_pose,
                    target,
                    axisymmetric=axisymmetric_orientation,
                )
                position_history.append(updated_position_error)
                cartesian_history.append(updated_pose[:3, 3].copy())
                if self._turn_contact_enabled and phase == Phase.TRANSFER:
                    progress = self._turn_contact_progress(updated_pose)
                    self._record_turn_progress(progress)
                    turn_progress_history.append(progress)
                    turn_width_history.append(self._gripper_width())
                if (
                    updated_position_error > position_tolerance
                    or updated_rotation_error > self.config.rotation_tolerance_rad
                ):
                    typed_feedback = typed_turn_contact_stall_completion(
                        updated_pose,
                        updated_position_error,
                        updated_rotation_error,
                    )
                    if typed_feedback is not None:
                        return typed_feedback
                    typed_feedback = typed_expand_grasp_stall_completion(
                        updated_pose,
                        updated_position_error,
                        updated_rotation_error,
                    )
                    if typed_feedback is not None:
                        return typed_feedback
                    typed_feedback = typed_rim_place_stall_completion(
                        updated_pose,
                        updated_position_error,
                        updated_rotation_error,
                    )
                    if typed_feedback is not None:
                        return typed_feedback
            if not reached:
                current = self.current_ee_pose()
                position_error, rotation_error = self._pose_errors(
                    current,
                    target,
                    axisymmetric=axisymmetric_orientation,
                )
                reached = (
                    position_error <= position_tolerance
                    and rotation_error <= self.config.rotation_tolerance_rad
                )
            if not reached:
                typed_feedback = typed_mode_completion(
                    current, position_error, rotation_error
                )
                if typed_feedback is not None:
                    return typed_feedback
                typed_feedback = typed_turn_contact_stall_completion(
                    current, position_error, rotation_error
                )
                if typed_feedback is not None:
                    return typed_feedback
                typed_feedback = typed_expand_grasp_stall_completion(
                    current, position_error, rotation_error
                )
                if typed_feedback is not None:
                    return typed_feedback
                typed_feedback = typed_rim_place_stall_completion(
                    current, position_error, rotation_error
                )
                if typed_feedback is not None:
                    return typed_feedback
                position_progress = initial_position_error - position_error
                rotation_progress = initial_rotation_error - rotation_error
                contact_window = position_history[-self.config.grasp_contact_window_steps :]
                stalled = bool(
                    len(contact_window) >= self.config.grasp_contact_window_steps
                    and max(contact_window) - min(contact_window)
                    <= self.config.grasp_contact_stall_epsilon_m
                )
                delta = target[:3, 3] - current[:3, 3]
                pinch_grasp_contact = bool(
                    phase == Phase.GRASP
                    and self._active_grasp_mode != GraspMode.EXPAND
                    and self._grasp_contact_enabled
                    and stalled
                    and position_error <= self._grasp_contact_residual_m
                    and rotation_error <= self.config.grasp_contact_orientation_rad
                )
                if pinch_grasp_contact:
                    self._grasp_contact_reached = True
                    if self._active_grasp_mode == GraspMode.RIM_PINCH:
                        # Preserve the frozen sensor-bound target so a close
                        # pulse can keep seating the fingers through compliant
                        # top contact instead of holding several millimetres
                        # above the observed rim.  Contextual cavity grasps and
                        # free-space rim grasps share this proprioceptive
                        # contact signal; only their bounded radial profiles
                        # differ after the close.
                        self._grasp_contact_servo_goal = target.copy()
                    self.phase_trace.append(
                        {
                            "phase": phase.value,
                            "osc_steps": self.steps_executed - start_steps,
                            "optimizer_waypoints": optimizer_waypoint_count,
                            "tracked_waypoints": target_index + 1,
                            "partial_progress": False,
                            "contact_reached": True,
                            "contact_residual_m": position_error,
                            "contact_vertical_residual_m": -float(delta[2]),
                            "initial_position_error_m": initial_position_error,
                            "initial_rotation_error_rad": initial_rotation_error,
                            "final_position_error_m": position_error,
                            "final_rotation_error_rad": rotation_error,
                            "initial_goal_minus_current_xyz_m": (
                                initial_goal_minus_current.tolist()
                            ),
                            "final_goal_minus_current_xyz_m": (
                                delta
                            ).tolist(),
                            "gripper_width_m": self._gripper_width(),
                            **trace_context,
                        }
                    )
                    return ControllerFeedback(True, "grasp contact completion")
                if (
                    phase == Phase.PLACE
                    and self._active_grasp_mode != GraspMode.RIM_PINCH
                    and len(contact_window) >= self.config.grasp_contact_window_steps
                    and float(np.linalg.norm(delta[:2]))
                    <= self.config.place_contact_xy_tolerance_m
                    and 0.0 <= -float(delta[2])
                    <= self.config.place_contact_max_vertical_residual_m
                    and rotation_error <= self.config.grasp_contact_orientation_rad
                    and max(contact_window) - min(contact_window)
                    <= self.config.grasp_contact_stall_epsilon_m
                ):
                    self._place_contact_reached = True
                    self.phase_trace.append(
                        {
                            "phase": phase.value,
                            "osc_steps": self.steps_executed - start_steps,
                            "optimizer_waypoints": optimizer_waypoint_count,
                            "tracked_waypoints": target_index + 1,
                            "partial_progress": False,
                            "contact_reached": True,
                            "contact_residual_m": position_error,
                            "contact_vertical_residual_m": -float(delta[2]),
                            "initial_position_error_m": initial_position_error,
                            "initial_rotation_error_rad": initial_rotation_error,
                            "final_position_error_m": position_error,
                            "final_rotation_error_rad": rotation_error,
                            "initial_goal_minus_current_xyz_m": (
                                initial_goal_minus_current.tolist()
                            ),
                            "final_goal_minus_current_xyz_m": delta.tolist(),
                            "gripper_width_m": self._gripper_width(),
                            **trace_context,
                        }
                    )
                    return ControllerFeedback(True, "place contact completion")
                if (
                    position_progress >= self.config.min_position_progress_m
                    or rotation_progress >= self.config.min_rotation_progress_rad
                ):
                    # A receding-horizon prefix is not itself the phase goal.
                    # Accept useful partial tracking so the outer controller
                    # can observe fresh RGB-D and optimise the next prefix.
                    # TRANSFER also tracks an internal midpoint. Feedback
                    # errors must refer to the requested chunk endpoint, not
                    # that midpoint, or the caller can stop halfway through
                    # a turn while seeing a small reported rotation error.
                    chunk_position_error, chunk_rotation_error = self._pose_errors(
                        current, poses[-1], axisymmetric=axisymmetric_orientation
                    )
                    self.phase_trace.append(
                        {
                            "phase": phase.value,
                            "osc_steps": self.steps_executed - start_steps,
                            "optimizer_waypoints": optimizer_waypoint_count,
                            "tracked_waypoints": target_index + 1,
                            "partial_progress": True,
                            "chunk_endpoint_position_error_m": chunk_position_error,
                            "chunk_endpoint_rotation_error_rad": chunk_rotation_error,
                            "position_progress_m": position_progress,
                            "rotation_progress_rad": rotation_progress,
                            "initial_position_error_m": initial_position_error,
                            "initial_rotation_error_rad": initial_rotation_error,
                            "final_position_error_m": position_error,
                            "final_rotation_error_rad": rotation_error,
                            "initial_goal_minus_current_xyz_m": (
                                initial_goal_minus_current.tolist()
                            ),
                            "final_goal_minus_current_xyz_m": delta.tolist(),
                            "gripper_width_m": self._gripper_width(),
                            **trace_context,
                        }
                    )
                    return ControllerFeedback(
                        True,
                        "partial MPC prefix made progress",
                        chunk_position_error,
                        chunk_rotation_error,
                    )
                turn_stall_diagnostic: dict[str, Any] = {}
                if (
                    self._turn_contact_enabled
                    and phase == Phase.TRANSFER
                    and self._turn_contact_start_pose is not None
                    and turn_progress_history
                    and turn_width_history
                    and cartesian_history
                ):
                    window_steps = self.config.turn_contact_window_steps
                    recent_progress = np.asarray(
                        turn_progress_history[-window_steps:], dtype=np.float64
                    )
                    recent_width = np.asarray(
                        turn_width_history[-window_steps:], dtype=np.float64
                    )
                    recent_cartesian = cartesian_history[-window_steps:]
                    turn_stall_diagnostic = {
                        "turn_contact_diagnostic": {
                            "signed_rotation_progress_rad": float(
                                recent_progress[-1]
                            ),
                            "minimum_signed_rotation_rad": float(
                                self._turn_contact_min_rotation_rad
                            ),
                            "recent_rotation_span_rad": float(
                                np.ptp(recent_progress)
                            ),
                            "recent_width_span_m": float(np.ptp(recent_width)),
                            "recent_cartesian_span_m": max(
                                float(np.linalg.norm(first - second))
                                for first in recent_cartesian
                                for second in recent_cartesian
                            ),
                            "contact_position_error_m": float(
                                np.linalg.norm(
                                    current[:3, 3]
                                    - self._turn_contact_start_pose[:3, 3]
                                )
                            ),
                            "monotonic_direction_confirmed": bool(
                                self._turn_contact_monotonic
                            ),
                        }
                    }
                self.phase_trace.append(
                    {
                        "phase": phase.value,
                        "osc_steps": self.steps_executed - start_steps,
                        "optimizer_waypoints": optimizer_waypoint_count,
                        "tracked_waypoints": target_index + 1,
                        "partial_progress": False,
                        "accepted": False,
                        "failure_kind": "osc_stalled",
                        "initial_position_error_m": initial_position_error,
                        "initial_rotation_error_rad": initial_rotation_error,
                        "final_position_error_m": position_error,
                        "final_rotation_error_rad": rotation_error,
                        "initial_goal_minus_current_xyz_m": (
                            initial_goal_minus_current.tolist()
                        ),
                        "final_goal_minus_current_xyz_m": delta.tolist(),
                        "gripper_width_m": self._gripper_width(),
                        **turn_stall_diagnostic,
                        **trace_context,
                    }
                )
                return ControllerFeedback(
                    False,
                    f"OSC tracking did not converge in {phase.value}; "
                    f"position error {position_error:.4f} m, "
                    f"rotation error {rotation_error:.4f} rad "
                    f"({trace_context['rotation_error_kind']})",
                    position_error,
                    rotation_error,
                )
        self.phase_trace.append(
            {
                "phase": phase.value,
                "osc_steps": self.steps_executed - start_steps,
                "optimizer_waypoints": optimizer_waypoint_count,
                "tracked_waypoints": len(tracked),
                "partial_progress": False,
                "initial_position_error_m": initial_position_error,
                "initial_rotation_error_rad": initial_rotation_error,
                "final_position_error_m": position_error,
                "final_rotation_error_rad": rotation_error,
                "initial_goal_minus_current_xyz_m": (
                    initial_goal_minus_current.tolist()
                ),
                "final_goal_minus_current_xyz_m": (
                    target[:3, 3] - current[:3, 3]
                ).tolist(),
                "gripper_width_m": self._gripper_width(),
                **trace_context,
            }
        )
        return ControllerFeedback(True)

    def set_gripper(self, gripper_command: float) -> ControllerFeedback:
        if gripper_command not in (-1.0, 1.0):
            return ControllerFeedback(False, "gripper command must be -1 or +1")
        self._last_gripper_command = float(gripper_command)
        rim_release = bool(
            self._active_grasp_mode == GraspMode.RIM_PINCH
            and gripper_command == -1.0
        )
        release_width_before = self._gripper_width() if rim_release else None
        if self._active_grasp_mode == GraspMode.EXPAND and gripper_command == 1.0:
            # Closing the fingers releases an internal expansion grasp.
            self._expand_retention_confirmed = False
        if self._active_grasp_mode == GraspMode.RIM_PINCH and gripper_command == -1.0:
            # Opening releases a rim pinch, invalidating its transfer gate.
            self._rim_retention_confirmed = False
            self._rim_grasp_marginal = False
            self._cavity_rim_preclosed = False
            self._cavity_rim_load_baseline_width_m = None
            self._cavity_rim_load_proof_passed = False
        if self._active_grasp_mode == GraspMode.EXPAND and gripper_command == -1.0:
            hold_steps = self.config.expand_gripper_hold_steps
        elif self._active_grasp_mode == GraspMode.RIM_PINCH:
            hold_steps = (
                self.config.rim_pinch_gripper_hold_steps
                if gripper_command == 1.0
                else self.config.rim_pinch_release_hold_steps
            )
        else:
            hold_steps = self.config.gripper_hold_steps
        candidate_id = str(self._active_grasp_candidate_id)
        execution_profile = self._active_cavity_rim_execution_profile
        free_space_rim_profile = bool(
            candidate_id.startswith("analytic-rim-")
            and execution_profile is not None
            and execution_profile.get("candidate_id") == candidate_id
            and execution_profile.get("profile_kind")
            in {
                "free_space_rim_contact_seat",
                "free_space_rim_direct_load",
            }
        )
        free_space_direct_load_profile = bool(
            free_space_rim_profile
            and execution_profile is not None
            and execution_profile.get("profile_kind")
            == "free_space_rim_direct_load"
        )
        free_space_contact_seat = bool(
            self._grasp_contact_reached and free_space_rim_profile
        )
        close_servo_goal = (
            self._grasp_contact_servo_goal
            if (
                self._active_grasp_mode == GraspMode.RIM_PINCH
                and gripper_command == 1.0
                and (
                    free_space_contact_seat
                    or (
                        self._cavity_rim_preclosed
                        and candidate_id.startswith(
                            ("analytic-cavity-rim-", "analytic-rim-")
                        )
                    )
                )
            )
            else None
        )
        servo_start = self.current_ee_pose() if close_servo_goal is not None else None
        width_before = self._gripper_width() if close_servo_goal is not None else None
        executed_steps = 0
        for _ in range(hold_steps):
            # A recovery retreat already commands an open rim gripper.  Do
            # not spend another fixed twelve ticks once public proprioception
            # proves that the fingers are outside the entire blocked-width
            # interval.  Equality remains classified as blocked, matching
            # grasp_confirmed(), so this cannot weaken retention checks.
            if (
                rim_release
                and self._gripper_width() > self.config.pinch_max_width_m
            ):
                break
            if self.steps_executed >= self.step_budget:
                return ControllerFeedback(False, "episode OSC step budget exhausted")
            action = (
                self._pose_action(
                    self.current_ee_pose(),
                    close_servo_goal,
                    gripper_command,
                )
                if close_servo_goal is not None
                else OSCAction.hold(gripper_command)
            )
            self._step(action)
            executed_steps += 1
        if rim_release:
            release_width_after = self._gripper_width()
            self.phase_trace.append(
                {
                    "phase": "rim_release",
                    "active_grasp_candidate": self._active_grasp_candidate_id,
                    "release_steps": executed_steps,
                    "maximum_release_steps": hold_steps,
                    "gripper_width_before_m": release_width_before,
                    "gripper_width_after_m": release_width_after,
                    "open_width_min_exclusive_m": self.config.pinch_max_width_m,
                    "open_width_proven": bool(
                        release_width_after > self.config.pinch_max_width_m
                    ),
                }
            )
        if close_servo_goal is not None:
            assert servo_start is not None and width_before is not None
            servo_end = self.current_ee_pose()
            initial_position_error, initial_rotation_error = self._pose_errors(
                servo_start,
                close_servo_goal,
            )
            final_position_error, final_rotation_error = self._pose_errors(
                servo_end,
                close_servo_goal,
            )
            self.phase_trace.append(
                {
                    "phase": "grasp_close_servo",
                    "active_grasp_candidate": self._active_grasp_candidate_id,
                    "servo_trigger": (
                        "typed_contact_stall"
                        if self._grasp_contact_reached
                        else "normal_pose_convergence"
                    ),
                    "servo_steps": executed_steps,
                    "initial_position_error_m": initial_position_error,
                    "initial_rotation_error_rad": initial_rotation_error,
                    "final_position_error_m": final_position_error,
                    "final_rotation_error_rad": final_rotation_error,
                    "initial_goal_minus_current_xyz_m": (
                        close_servo_goal[:3, 3] - servo_start[:3, 3]
                    ).tolist(),
                    "final_goal_minus_current_xyz_m": (
                        close_servo_goal[:3, 3] - servo_end[:3, 3]
                    ).tolist(),
                    "gripper_width_before_m": width_before,
                    "gripper_width_after_m": self._gripper_width(),
                }
            )
            nominal_direct_lift = bool(
                candidate_id.startswith("analytic-cavity-rim-")
                and candidate_id.endswith("-z0-yaw0")
            )
            free_space_contact_seat = bool(
                free_space_rim_profile
                and not free_space_direct_load_profile
                and execution_profile is not None
            )
            seat_phase = (
                "rim_free_space_direct_load"
                if free_space_direct_load_profile
                else (
                    "rim_free_space_contact_seat"
                    if free_space_contact_seat
                    else "rim_cavity_seat_pull"
                )
            )
            seat_width_before = self._gripper_width()
            if not (
                self.config.rim_pinch_blocked_min_width_m
                <= seat_width_before
                < self.config.pinch_max_width_m
            ):
                # A pull is a post-contact seating manoeuvre, never another
                # blind grasp search.  Empty-close (or implausibly wide)
                # proprioception cannot authorize lateral motion beside a
                # fixture; fail the candidate before taking any seat action.
                self.phase_trace.append(
                    {
                        "phase": (
                            "rim_cavity_direct_lift"
                            if nominal_direct_lift
                            else seat_phase
                        ),
                        "active_grasp_candidate": self._active_grasp_candidate_id,
                        "servo_steps": 0,
                        "minimum_servo_steps": (
                            self.config.rim_cavity_seat_pull_steps
                        ),
                        "maximum_servo_steps": (
                            self.config.rim_cavity_seat_pull_max_steps
                        ),
                        "gripper_width_before_m": seat_width_before,
                        "gripper_width_after_m": seat_width_before,
                        "accepted_width_min_m": (
                            self.config.rim_pinch_blocked_min_width_m
                        ),
                        "accepted_width_max_exclusive_m": (
                            self.config.pinch_max_width_m
                        ),
                        "accepted": False,
                        "failure_kind": "unconfirmed_rim_width",
                    }
                )
                return ControllerFeedback(
                    False,
                    (
                        "rim close did not confirm a blocked width"
                        if free_space_rim_profile
                        else "cavity rim close did not confirm a blocked width"
                    ),
                )
            if free_space_direct_load_profile:
                # A stable free-space wall needs no horizontal seating motion:
                # that v28 manoeuvre crossed the thin rim and converted a
                # 9.813-mm close into an empty 2.956-mm pinch.  Preserve the
                # external CLOSE_FINGERS contact exactly where it formed and
                # let the next LIFT call apply the strict 5--8-mm vertical
                # proof.  Marginal widths remain accepted only as a close;
                # the coordinator's existing marginal gate releases and
                # reacquires another physical rim before any lift.
                load_eligible = bool(
                    seat_width_before
                    > self.config.rim_free_space_load_proof_min_width_m
                )
                self.phase_trace.append(
                    {
                        "phase": "rim_free_space_direct_load",
                        "active_grasp_candidate": self._active_grasp_candidate_id,
                        "servo_steps": 0,
                        "gripper_width_before_m": seat_width_before,
                        "gripper_width_after_m": seat_width_before,
                        "accepted_width_min_m": (
                            self.config.rim_pinch_blocked_min_width_m
                        ),
                        "load_proof_width_min_exclusive_m": (
                            self.config.rim_free_space_load_proof_min_width_m
                        ),
                        "accepted_width_max_exclusive_m": (
                            self.config.pinch_max_width_m
                        ),
                        "load_proof_eligible": load_eligible,
                        "motion_semantics": "vertical_only_no_xy_seat",
                        "grasp_semantics": "external_rim_close_fingers",
                        "accepted": True,
                    }
                )
                self._cavity_rim_load_baseline_width_m = (
                    seat_width_before if load_eligible else None
                )
                self._cavity_rim_load_proof_passed = False
                self._cavity_rim_load_proof_failed = False
                return ControllerFeedback(True)
            if nominal_direct_lift:
                # The conservative sensor profile is already level and sits at
                # the observed rim top.  Its historical successful mechanics
                # use the two-pad blocked close directly: a lateral seat pull
                # changes that contact without increasing the measured width.
                # Skip all seat actions, but retain the same proprioceptive
                # hard gate above and the normal post-LIFT RGB-D/width proof.
                self.phase_trace.append(
                    {
                        "phase": "rim_cavity_direct_lift",
                        "active_grasp_candidate": self._active_grasp_candidate_id,
                        "profile_height": "z0",
                        "profile_yaw_offset_deg": 0.0,
                        "profile_outer_finger_tilt_deg": 0.0,
                        "servo_steps": 0,
                        "gripper_width_before_m": seat_width_before,
                        "gripper_width_after_m": seat_width_before,
                        "accepted_width_min_m": (
                            self.config.rim_pinch_blocked_min_width_m
                        ),
                        "accepted_width_max_exclusive_m": (
                            self.config.pinch_max_width_m
                        ),
                        "ordering_basis": "sensor_nominal_rim_profile",
                        "accepted": True,
                    }
                )
                self._cavity_rim_load_baseline_width_m = seat_width_before
                self._cavity_rim_load_proof_passed = False
                self._cavity_rim_load_proof_failed = False
                return ControllerFeedback(True)
            # Closing alone can form a valid two-pad rim contact yet lose it
            # immediately when lifting: the wall slides out along the support
            # plane as vertical load rises.  Seat that already sensed contact
            # with a small outward pull before the proof lift.  Both the side
            # and the planar axis come from the active RGB-D grasp candidate;
            # no task metadata or simulator contact is consulted.
            side_sign = (
                1.0
                if "-positive-" in candidate_id
                else -1.0
                if "-negative-" in candidate_id
                else 0.0
            )
            planar_axis = np.asarray(close_servo_goal[:3, 1], dtype=np.float64).copy()
            planar_axis[2] = 0.0
            planar_axis_norm = float(np.linalg.norm(planar_axis))
            if free_space_contact_seat:
                assert execution_profile is not None
                outward_direction = np.asarray(
                    execution_profile["radial_direction_world"],
                    dtype=np.float64,
                ).copy()
                side_sign = float(
                    np.sign(np.dot(outward_direction, planar_axis))
                )
            else:
                outward_direction = side_sign * planar_axis
            if side_sign == 0.0 or planar_axis_norm < 0.80:
                self.phase_trace.append(
                    {
                        "phase": seat_phase,
                        "active_grasp_candidate": self._active_grasp_candidate_id,
                        "accepted": False,
                        "failure_kind": "invalid_sensor_candidate_axis",
                        "candidate_side_sign": side_sign,
                        "candidate_planar_axis_norm": planar_axis_norm,
                    }
                )
                return ControllerFeedback(
                    False, "rim seat pull has no valid sensor direction"
                )
            planar_axis /= planar_axis_norm
            if not free_space_contact_seat:
                outward_direction = side_sign * planar_axis
            seat_start = self.current_ee_pose()
            recovery_profile = self._active_cavity_rim_execution_profile
            recovery_seat_cap_active = bool(
                recovery_profile is not None
                and recovery_profile.get("candidate_id") == candidate_id
                and recovery_profile.get("profile_kind")
                in {
                    "fresh_axis_rinset2_zminus6",
                    "free_space_rim_contact_seat",
                }
            )
            seat_command_distance = (
                self.config.rim_free_space_seat_pull_m
                if free_space_contact_seat
                else self.config.rim_cavity_seat_pull_m
            )
            seat_start_radius_m: float | None = None
            maximum_seat_radius_m: float | None = None
            if recovery_seat_cap_active:
                assert recovery_profile is not None
                profile_direction = np.asarray(
                    recovery_profile["radial_direction_world"],
                    dtype=np.float64,
                )
                if float(np.dot(profile_direction, outward_direction)) < 0.98:
                    self.phase_trace.append(
                        {
                            "phase": seat_phase,
                            "active_grasp_candidate": candidate_id,
                            "accepted": False,
                            "failure_kind": "recovery_profile_axis_mismatch",
                        }
                    )
                    return ControllerFeedback(
                        False,
                        "rim seat axis disagrees with sensor profile",
                    )
                outward_direction = profile_direction.copy()
                source_center = np.asarray(
                    recovery_profile["source_center_world_m"],
                    dtype=np.float64,
                )
                maximum_seat_radius_m = float(
                    recovery_profile["maximum_seat_radius_m"]
                )
                seat_start_radius_m = float(
                    np.dot(
                        seat_start[:3, 3] - source_center,
                        outward_direction,
                    )
                )
                remaining_radius = maximum_seat_radius_m - seat_start_radius_m
                if remaining_radius <= 0.0:
                    self.phase_trace.append(
                        {
                            "phase": seat_phase,
                            "active_grasp_candidate": candidate_id,
                            "accepted": False,
                            "failure_kind": "recovery_seat_radius_already_exceeded",
                            "seat_start_radius_m": seat_start_radius_m,
                            "maximum_seat_radius_m": maximum_seat_radius_m,
                        }
                    )
                    return ControllerFeedback(
                        False,
                        "rim seat starts outside its sensor radius cap",
                    )
                seat_command_distance = min(
                    seat_command_distance,
                    remaining_radius,
                )
            seat_goal = close_servo_goal.copy()
            seat_goal[:3, 3] = (
                seat_start[:3, 3]
                + seat_command_distance * outward_direction
            )
            geometric_min_progress_m = (
                min(
                    self.config.rim_cavity_seat_pull_min_progress_m,
                    max(
                        self.config.rim_cavity_seat_contact_min_progress_m,
                        seat_command_distance
                        - self.config.rim_cavity_seat_pull_max_goal_error_m,
                    ),
                )
                if recovery_seat_cap_active
                else self.config.rim_cavity_seat_pull_min_progress_m
            )
            seat_samples: list[dict[str, Any]] = []
            seat_steps = 0
            seat_accepted = False
            seat_accepted_kind: str | None = None
            seat_failure_kind: str | None = None
            all_widths_retained = True
            recent_radial_span: float | None = None
            recent_cartesian_span: float | None = None
            edge_predicted_next_width_m: float | None = None
            for _ in range(self.config.rim_cavity_seat_pull_max_steps):
                if self.steps_executed >= self.step_budget:
                    seat_failure_kind = "step_budget_exhausted"
                    break
                current = self.current_ee_pose()
                step_width_before = self._gripper_width()
                self._step(self._pose_action(current, seat_goal, 1.0))
                seat_steps += 1
                updated = self.current_ee_pose()
                updated_width = self._gripper_width()
                displacement = updated[:3, 3] - seat_start[:3, 3]
                radial_progress = float(
                    np.dot(displacement, outward_direction)
                )
                absolute_radial_position: float | None = None
                if recovery_seat_cap_active:
                    assert recovery_profile is not None
                    absolute_radial_position = float(
                        np.dot(
                            updated[:3, 3]
                            - np.asarray(
                                recovery_profile["source_center_world_m"],
                                dtype=np.float64,
                            ),
                            outward_direction,
                        )
                    )
                planar_orthogonal = (
                    displacement[:2]
                    - radial_progress * outward_direction[:2]
                )
                orthogonal_error = float(np.linalg.norm(planar_orthogonal))
                vertical_error = abs(float(displacement[2]))
                goal_error, rotation_error = self._pose_errors(updated, seat_goal)
                seat_samples.append(
                    {
                        "step": self.steps_executed,
                        "command": 1.0,
                        "width_before_m": step_width_before,
                        "width_after_m": updated_width,
                        "radial_progress_m": radial_progress,
                        "absolute_radial_position_m": (
                            absolute_radial_position
                        ),
                        "orthogonal_error_m": orthogonal_error,
                        "vertical_error_m": vertical_error,
                        "goal_error_after_m": goal_error,
                        "rotation_error_after_rad": rotation_error,
                        "position_world_m": updated[:3, 3].tolist(),
                    }
                )
                width_retained = bool(
                    self.config.rim_pinch_blocked_min_width_m
                    <= updated_width
                    < self.config.pinch_max_width_m
                )
                if not width_retained:
                    all_widths_retained = False
                    seat_failure_kind = "proprioceptive_blocked_width_lost"
                    break
                if (
                    maximum_seat_radius_m is not None
                    and absolute_radial_position is not None
                    and absolute_radial_position
                    > maximum_seat_radius_m + 1e-6
                ):
                    seat_failure_kind = "recovery_seat_radius_cap_exceeded"
                    break
                geometry_seat_accepted = bool(
                    seat_steps >= self.config.rim_cavity_seat_pull_steps
                    and radial_progress
                    >= geometric_min_progress_m
                    and goal_error
                    <= self.config.rim_cavity_seat_pull_max_goal_error_m
                    and orthogonal_error
                    <= self.config.rim_cavity_seat_pull_max_orthogonal_error_m
                    and vertical_error
                    <= self.config.rim_cavity_seat_pull_max_vertical_error_m
                    and rotation_error <= self.config.rotation_tolerance_rad
                )
                contact_seat_accepted = False
                thin_wall_edge_accepted = False
                stall_window = self.config.rim_cavity_seat_contact_stall_window_steps
                if len(seat_samples) >= stall_window:
                    recent_samples = seat_samples[-stall_window:]
                    recent_radial = np.asarray(
                        [sample["radial_progress_m"] for sample in recent_samples],
                        dtype=np.float64,
                    )
                    recent_radial_span = float(np.ptp(recent_radial))
                    recent_positions = np.asarray(
                        [sample["position_world_m"] for sample in recent_samples],
                        dtype=np.float64,
                    )
                    pairwise = (
                        recent_positions[:, None, :]
                        - recent_positions[None, :, :]
                    )
                    recent_cartesian_span = float(
                        np.max(np.linalg.norm(pairwise, axis=2))
                    )
                    width_gain = updated_width - seat_width_before
                    contact_seat_accepted = bool(
                        seat_steps >= self.config.rim_cavity_seat_pull_steps
                        and radial_progress
                        >= self.config.rim_cavity_seat_contact_min_progress_m
                        and recent_radial_span
                        <= self.config.rim_cavity_seat_contact_max_radial_span_m
                        and recent_cartesian_span
                        <= self.config.rim_cavity_seat_contact_max_cartesian_span_m
                        and width_gain
                        >= self.config.rim_cavity_seat_contact_min_width_gain_m
                        and all_widths_retained
                        and orthogonal_error
                        <= self.config.rim_cavity_seat_contact_max_orthogonal_error_m
                        and vertical_error
                        <= self.config.rim_cavity_seat_contact_max_vertical_error_m
                        and rotation_error <= self.config.rotation_tolerance_rad
                    )
                if free_space_contact_seat and len(seat_samples) >= 2:
                    previous_width = float(
                        seat_samples[-2]["width_after_m"]
                    )
                    marginal_width_max = (
                        self.config.rim_pinch_blocked_min_width_m + 0.001
                    )
                    edge_width_max = (
                        self.config.rim_pinch_blocked_min_width_m
                        + self.config.rim_free_space_seat_edge_width_margin_m
                    )
                    width_drop = previous_width - updated_width
                    edge_predicted_next_width_m = updated_width - max(
                        width_drop, 0.0
                    )
                    thin_wall_edge_accepted = bool(
                        seat_steps
                        >= self.config.rim_free_space_seat_edge_min_steps
                        and radial_progress
                        >= self.config.rim_free_space_seat_edge_min_progress_m
                        and marginal_width_max < updated_width <= edge_width_max
                        and width_drop
                        >= self.config.rim_free_space_seat_edge_min_width_drop_m
                        and edge_predicted_next_width_m <= marginal_width_max
                        and all_widths_retained
                        and orthogonal_error
                        <= self.config.rim_cavity_seat_contact_max_orthogonal_error_m
                        and vertical_error
                        <= self.config.rim_cavity_seat_contact_max_vertical_error_m
                        and rotation_error <= self.config.rotation_tolerance_rad
                    )
                if geometry_seat_accepted:
                    seat_accepted = True
                    seat_accepted_kind = "geometric_pose_gate"
                elif contact_seat_accepted:
                    seat_accepted = True
                    seat_accepted_kind = "proprio_width_gain_contact_seated"
                elif thin_wall_edge_accepted:
                    seat_accepted = True
                    seat_accepted_kind = "proprio_nonmarginal_thin_wall_edge"
                if seat_accepted:
                    break
            seat_end = self.current_ee_pose()
            realised_delta = seat_end[:3, 3] - seat_start[:3, 3]
            radial_progress = float(
                np.dot(realised_delta, outward_direction)
            )
            planar_orthogonal = (
                realised_delta[:2]
                - radial_progress * outward_direction[:2]
            )
            orthogonal_error = float(np.linalg.norm(planar_orthogonal))
            vertical_error = abs(float(realised_delta[2]))
            final_goal_error, final_rotation_error = self._pose_errors(
                seat_end, seat_goal
            )
            if not seat_accepted and seat_failure_kind is None:
                seat_failure_kind = "measured_pose_gate_not_reached_within_budget"
            self.phase_trace.append(
                {
                    "phase": seat_phase,
                    "active_grasp_candidate": self._active_grasp_candidate_id,
                    "candidate_side_sign": side_sign,
                    "candidate_planar_axis_world": planar_axis.tolist(),
                    "outward_direction_world": outward_direction.tolist(),
                    "commanded_distance_m": seat_command_distance,
                    "nominal_commanded_distance_m": (
                        self.config.rim_free_space_seat_pull_m
                        if free_space_contact_seat
                        else self.config.rim_cavity_seat_pull_m
                    ),
                    "recovery_seat_radius_cap_active": (
                        recovery_seat_cap_active
                    ),
                    "seat_start_radius_m": seat_start_radius_m,
                    "maximum_seat_radius_m": maximum_seat_radius_m,
                    "seat_radius_numeric_tolerance_m": (
                        1e-6 if recovery_seat_cap_active else None
                    ),
                    "start_position_world_m": seat_start[:3, 3].tolist(),
                    "commanded_goal_xyz_m": seat_goal[:3, 3].tolist(),
                    "servo_steps": seat_steps,
                    "minimum_servo_steps": self.config.rim_cavity_seat_pull_steps,
                    "maximum_servo_steps": (
                        self.config.rim_cavity_seat_pull_max_steps
                    ),
                    "minimum_radial_progress_m": (
                        geometric_min_progress_m
                    ),
                    "maximum_goal_error_m": (
                        self.config.rim_cavity_seat_pull_max_goal_error_m
                    ),
                    "maximum_orthogonal_error_m": (
                        self.config.rim_cavity_seat_pull_max_orthogonal_error_m
                    ),
                    "maximum_vertical_error_m": (
                        self.config.rim_cavity_seat_pull_max_vertical_error_m
                    ),
                    "contact_seated_minimum_radial_progress_m": (
                        self.config.rim_cavity_seat_contact_min_progress_m
                    ),
                    "contact_seated_minimum_width_gain_m": (
                        self.config.rim_cavity_seat_contact_min_width_gain_m
                    ),
                    "contact_seated_stall_window_steps": (
                        self.config.rim_cavity_seat_contact_stall_window_steps
                    ),
                    "contact_seated_maximum_radial_span_m": (
                        self.config.rim_cavity_seat_contact_max_radial_span_m
                    ),
                    "contact_seated_maximum_cartesian_span_m": (
                        self.config.rim_cavity_seat_contact_max_cartesian_span_m
                    ),
                    "contact_seated_maximum_orthogonal_error_m": (
                        self.config.rim_cavity_seat_contact_max_orthogonal_error_m
                    ),
                    "contact_seated_maximum_vertical_error_m": (
                        self.config.rim_cavity_seat_contact_max_vertical_error_m
                    ),
                    "free_space_edge_width_min_exclusive_m": (
                        self.config.rim_pinch_blocked_min_width_m + 0.001
                        if free_space_contact_seat
                        else None
                    ),
                    "free_space_edge_width_max_m": (
                        self.config.rim_pinch_blocked_min_width_m
                        + self.config.rim_free_space_seat_edge_width_margin_m
                        if free_space_contact_seat
                        else None
                    ),
                    "free_space_edge_minimum_progress_m": (
                        self.config.rim_free_space_seat_edge_min_progress_m
                        if free_space_contact_seat
                        else None
                    ),
                    "free_space_edge_minimum_width_drop_m": (
                        self.config.rim_free_space_seat_edge_min_width_drop_m
                        if free_space_contact_seat
                        else None
                    ),
                    "free_space_edge_minimum_steps": (
                        self.config.rim_free_space_seat_edge_min_steps
                        if free_space_contact_seat
                        else None
                    ),
                    "free_space_edge_predicted_next_width_m": (
                        edge_predicted_next_width_m
                        if free_space_contact_seat
                        else None
                    ),
                    "maximum_rotation_error_rad": (
                        self.config.rotation_tolerance_rad
                    ),
                    "actual_delta_xyz_m": realised_delta.tolist(),
                    "actual_projected_displacement_m": radial_progress,
                    "actual_orthogonal_error_m": orthogonal_error,
                    "actual_vertical_error_m": vertical_error,
                    "actual_width_gain_m": (
                        self._gripper_width() - seat_width_before
                    ),
                    "recent_radial_span_m": recent_radial_span,
                    "recent_cartesian_span_m": recent_cartesian_span,
                    "all_widths_retained": all_widths_retained,
                    "final_goal_error_m": final_goal_error,
                    "final_rotation_error_rad": final_rotation_error,
                    "gripper_width_before_m": seat_width_before,
                    "gripper_width_after_m": self._gripper_width(),
                    "servo_samples": seat_samples,
                    "accepted": seat_accepted,
                    "accepted_kind": seat_accepted_kind,
                    **({} if seat_accepted else {"failure_kind": seat_failure_kind}),
                }
            )
            if not seat_accepted:
                return ControllerFeedback(
                    False,
                    (
                        "episode OSC step budget exhausted"
                        if seat_failure_kind == "step_budget_exhausted"
                        else "rim seat pull failed its sensor/proprioceptive gate"
                    ),
                )
            self._cavity_rim_load_baseline_width_m = self._gripper_width()
            self._cavity_rim_load_proof_passed = False
            self._cavity_rim_load_proof_failed = False
        return ControllerFeedback(True)

    def servo_gripper_width_at_pose(
        self,
        *,
        maximum_width_m: float,
        max_steps: int,
        maximum_position_drift_m: float,
        maximum_rotation_drift_rad: float,
        maximum_force_delta_n: float,
        maximum_force_norm_n: float,
        maximum_torque_delta_nm: float,
        maximum_torque_norm_nm: float,
        minimum_width_progress_m: float,
        maximum_stall_steps: int,
    ) -> ControllerFeedback:
        """Compact only the typed microwave pusher under public safety gates.

        Every issued close action actively holds the frozen public TCP pose.
        The servo terminates on the exact finger-width bound and fails closed
        on wrist-wrench growth, pose drift, non-monotonic width, lack of
        progress, or either bounded step budget.  It has no access to contacts,
        object joints, rewards, task predicates, or simulator identities.
        """

        scalar_limits = (
            maximum_width_m,
            maximum_position_drift_m,
            maximum_rotation_drift_rad,
            maximum_force_delta_n,
            maximum_force_norm_n,
            maximum_torque_delta_nm,
            maximum_torque_norm_nm,
            minimum_width_progress_m,
        )
        valid_integer_limits = bool(
            isinstance(max_steps, int)
            and not isinstance(max_steps, bool)
            and 1 <= max_steps <= 12
            and isinstance(maximum_stall_steps, int)
            and not isinstance(maximum_stall_steps, bool)
            and 0 <= maximum_stall_steps <= 2
            and maximum_stall_steps < max_steps
        )
        valid_scalar_limits = bool(
            all(np.isfinite(value) and value > 0.0 for value in scalar_limits)
            and maximum_width_m <= 0.014
            and maximum_position_drift_m <= 0.010
            and maximum_rotation_drift_rad <= 0.10
            and maximum_force_delta_n <= 20.0
            and maximum_force_norm_n <= 80.0
            and maximum_torque_delta_nm <= 5.0
            and maximum_torque_norm_nm <= 15.0
            and minimum_width_progress_m >= 0.0002
        )
        if not valid_integer_limits or not valid_scalar_limits:
            return ControllerFeedback(
                False,
                "microwave compact servo limits violate the strict safety caps",
            )
        if (
            self._active_grasp_mode is not GraspMode.PINCH
            or self._active_grasp_candidate_id != "route-c-contact-open"
        ):
            return ControllerFeedback(
                False,
                "microwave compact servo requires the typed Route C open contact",
            )

        initial_observation = self._observation()
        frozen_pose = np.asarray(
            initial_observation.proprio.T_world_ee, dtype=np.float64
        ).copy()
        initial_width = float(initial_observation.proprio.gripper_width_m)
        baseline_force = np.asarray(
            initial_observation.proprio.ee_force_sensor, dtype=np.float64
        ).copy()
        baseline_torque = np.asarray(
            initial_observation.proprio.ee_torque_sensor, dtype=np.float64
        ).copy()
        samples: list[dict[str, Any]] = []
        start_steps = self.steps_executed

        def finish(
            accepted: bool,
            detail: str,
            final_width: float,
            *,
            failure_kind: str | None = None,
        ) -> ControllerFeedback:
            trace: dict[str, Any] = {
                "phase": "microwave_compact_width_servo",
                "active_grasp_candidate": self._active_grasp_candidate_id,
                "accepted": accepted,
                "servo_steps": self.steps_executed - start_steps,
                "maximum_servo_steps": max_steps,
                "initial_width_m": initial_width,
                "final_width_m": final_width,
                "maximum_width_m": maximum_width_m,
                "maximum_position_drift_m": maximum_position_drift_m,
                "maximum_rotation_drift_rad": maximum_rotation_drift_rad,
                "maximum_force_delta_n": maximum_force_delta_n,
                "maximum_force_norm_n": maximum_force_norm_n,
                "maximum_torque_delta_nm": maximum_torque_delta_nm,
                "maximum_torque_norm_nm": maximum_torque_norm_nm,
                "minimum_width_progress_m": minimum_width_progress_m,
                "maximum_stall_steps": maximum_stall_steps,
                "servo_samples": samples,
            }
            if failure_kind is not None:
                trace["failure_kind"] = failure_kind
            self.phase_trace.append(trace)
            return ControllerFeedback(accepted, detail)

        if (
            frozen_pose.shape != (4, 4)
            or not np.all(np.isfinite(frozen_pose))
            or not np.all(np.isfinite(baseline_force))
            or not np.all(np.isfinite(baseline_torque))
        ):
            return finish(
                False,
                "microwave compact servo public proprioception is invalid",
                initial_width,
                failure_kind="sensor",
            )
        baseline_force_norm = float(np.linalg.norm(baseline_force))
        baseline_torque_norm = float(np.linalg.norm(baseline_torque))
        if (
            baseline_force_norm > maximum_force_norm_n
            or baseline_torque_norm > maximum_torque_norm_nm
        ):
            return finish(
                False,
                "microwave compact servo force/torque baseline exceeds safety gate",
                initial_width,
                failure_kind="force",
            )
        if initial_width <= maximum_width_m:
            return finish(
                True,
                "microwave compact servo width was already within the strict gate",
                initial_width,
            )

        self._last_gripper_command = 1.0
        previous_width = initial_width
        stalled_steps = 0
        for index in range(1, max_steps + 1):
            if self.steps_executed >= self.step_budget:
                return finish(
                    False,
                    "microwave compact servo episode step budget exhausted",
                    previous_width,
                    failure_kind="budget",
                )
            current_pose = self.current_ee_pose()
            self._step(self._pose_action(current_pose, frozen_pose, 1.0))
            observation = self._observation()
            pose = np.asarray(observation.proprio.T_world_ee, dtype=np.float64)
            width = float(observation.proprio.gripper_width_m)
            force = np.asarray(
                observation.proprio.ee_force_sensor, dtype=np.float64
            )
            torque = np.asarray(
                observation.proprio.ee_torque_sensor, dtype=np.float64
            )
            if (
                pose.shape != (4, 4)
                or not np.all(np.isfinite(pose))
                or not np.all(np.isfinite(force))
                or not np.all(np.isfinite(torque))
                or not np.isfinite(width)
            ):
                return finish(
                    False,
                    "microwave compact servo received invalid public proprioception",
                    previous_width,
                    failure_kind="sensor",
                )
            try:
                position_drift, rotation_drift = self._pose_errors(
                    pose, frozen_pose
                )
            except ValueError:
                return finish(
                    False,
                    "microwave compact servo received an invalid public TCP pose",
                    width,
                    failure_kind="sensor",
                )
            force_norm = float(np.linalg.norm(force))
            force_delta = float(np.linalg.norm(force - baseline_force))
            torque_norm = float(np.linalg.norm(torque))
            torque_delta = float(np.linalg.norm(torque - baseline_torque))
            width_progress = previous_width - width
            samples.append(
                {
                    "index": index,
                    "width_m": width,
                    "width_progress_m": width_progress,
                    "position_drift_m": position_drift,
                    "rotation_drift_rad": rotation_drift,
                    "force_norm_n": force_norm,
                    "force_delta_n": force_delta,
                    "torque_norm_nm": torque_norm,
                    "torque_delta_nm": torque_delta,
                }
            )
            if (
                force_norm > maximum_force_norm_n
                or force_delta > maximum_force_delta_n
                or torque_norm > maximum_torque_norm_nm
                or torque_delta > maximum_torque_delta_nm
            ):
                return finish(
                    False,
                    "microwave compact servo force/torque safety gate exceeded",
                    width,
                    failure_kind="force",
                )
            if (
                position_drift > maximum_position_drift_m
                or rotation_drift > maximum_rotation_drift_rad
            ):
                return finish(
                    False,
                    "microwave compact servo motion safety gate exceeded",
                    width,
                    failure_kind="motion",
                )
            if width > previous_width + minimum_width_progress_m:
                return finish(
                    False,
                    "microwave compact servo width reversed while closing",
                    width,
                    failure_kind="width_reverse",
                )
            if width <= maximum_width_m:
                return finish(
                    True,
                    "microwave compact servo reached the strict width gate",
                    width,
                )
            if width_progress < minimum_width_progress_m:
                stalled_steps += 1
            else:
                stalled_steps = 0
            if stalled_steps > maximum_stall_steps:
                return finish(
                    False,
                    "microwave compact servo stalled before the strict width gate",
                    width,
                    failure_kind="stall",
                )
            previous_width = width

        return finish(
            False,
            "microwave compact servo timed out before the strict width gate",
            previous_width,
            failure_kind="timeout",
        )

    def grasp_confirmed(self, mode: GraspMode) -> bool:
        # This deliberately uses gripper proprioception only.  Simulator
        # contacts, body poses, task predicates, and evaluator success are not
        # accessible through RobotObservation.
        width = self._observation().proprio.gripper_width_m
        if mode != GraspMode.EXPAND:
            blocked_min_width = (
                self.config.rim_pinch_blocked_min_width_m
                if mode == GraspMode.RIM_PINCH
                else self.config.pinch_blocked_min_width_m
            )
            confirmed = bool(
                self._last_gripper_command == 1.0
                and blocked_min_width <= width
                <= self.config.pinch_max_width_m
            )
            evidence_kind = (
                "proprio_blocked_width"
                if confirmed
                else "proprio_width_out_of_range"
            )
            tentative = False
        else:
            strict = bool(
                self._last_gripper_command == -1.0
                and self.config.expansion_min_width_m
                <= width
                <= self.config.expansion_max_width_m
            )
            tentative = bool(
                not strict
                and self._allow_tentative_expansion
                and self.expansion_width_is_tentative()
            )
            confirmed = strict or tentative
            evidence_kind = (
                "proprio_blocked_width"
                if strict
                else (
                    "tentative_wide_width"
                    if tentative
                    else "proprio_width_out_of_range"
                )
            )
        marginal = bool(
            mode == GraspMode.RIM_PINCH
            and confirmed
            and width <= self.config.rim_pinch_blocked_min_width_m + 0.001
        )
        if mode == GraspMode.RIM_PINCH:
            self._rim_grasp_marginal = marginal
        self.grasp_checks.append(
            {
                "step": self.steps_executed,
                "mode": mode.value,
                "gripper_width_m": float(width),
                "confirmed": confirmed,
                "tentative": tentative,
                "marginal": marginal,
                "evidence_kind": evidence_kind,
                "active_grasp_candidate": self._active_grasp_candidate_id,
            }
        )
        return confirmed

    def record_expansion_visual_check(
        self,
        *,
        confirmed: bool,
        nearest_distance_m: float | None,
        visible_candidates: int,
    ) -> None:
        """Record the fresh RGB-D evidence used to resolve a wide opening."""

        self.grasp_checks.append(
            {
                "step": self.steps_executed,
                "mode": GraspMode.EXPAND.value,
                "gripper_width_m": self._gripper_width(),
                "confirmed": bool(confirmed),
                "tentative": False,
                "evidence_kind": (
                    "fresh_rgbd_near_ee"
                    if confirmed
                    else "fresh_rgbd_not_near_ee"
                ),
                "nearest_visible_source_distance_m": nearest_distance_m,
                "visible_source_candidates": int(visible_candidates),
                "active_grasp_candidate": self._active_grasp_candidate_id,
            }
        )

    def record_rim_visual_binding(self, evidence: Mapping[str, Any]) -> None:
        """Persist a fresh RGB-D held-offset / retention decision."""

        self.grasp_checks.append(
            {
                "step": self.steps_executed,
                "mode": GraspMode.RIM_PINCH.value,
                "gripper_width_m": self._gripper_width(),
                "confirmed": bool(evidence.get("accepted", False)),
                "evidence_kind": "fresh_rgbd_held_binding",
                "active_grasp_candidate": self._active_grasp_candidate_id,
                **dict(evidence),
            }
        )

    def _gripper_width(self) -> float:
        return float(self._observation().proprio.gripper_width_m)

    def current_gripper_width_m(self) -> float:
        """Return the sanitized public Panda finger separation.

        Contact skills use this read-only proprioceptive value to distinguish
        a retained two-pad load from a single finger wedged on a fixture.  It
        exposes no simulator contact, body state, reward, or task predicate.
        """

        return self._gripper_width()

    def _turn_contact_progress(self, pose: np.ndarray) -> float:
        if self._turn_contact_start_pose is None or self._turn_contact_axis_world is None:
            return 0.0
        relative = np.asarray(pose, dtype=np.float64)[:3, :3] @ (
            self._turn_contact_start_pose[:3, :3].T
        )
        rotvec = Rotation.from_matrix(relative).as_rotvec()
        return float(np.dot(rotvec, self._turn_contact_axis_world))

    def _record_turn_progress(self, progress_rad: float) -> None:
        progress = float(progress_rad)
        if (
            progress
            < self._turn_contact_previous_progress_rad
            - self.config.turn_contact_reverse_tolerance_rad
        ):
            self._turn_contact_monotonic = False
        self._turn_contact_previous_progress_rad = progress
        self._turn_contact_progress_rad = progress

    def _observation(self) -> RobotObservation:
        observation = self._observation_provider()
        if not isinstance(observation, RobotObservation):
            raise TypeError("observation_provider must return RobotObservation")
        return observation

    def _step(self, action: OSCAction) -> None:
        try:
            observation = self._action_executor(action)
        finally:
            # A dispatcher may terminate control flow after applying an action
            # (for example, when its owner ends an episode).  Count the issued
            # command without inspecting the reason or any hidden state.
            self.steps_executed += 1
        if not isinstance(observation, RobotObservation):
            raise TypeError("action_executor must return RobotObservation")

    def _pose_action(
        self, current: np.ndarray, target: np.ndarray, gripper_command: float
    ) -> OSCAction:
        translation = (
            target[:3, 3] - current[:3, 3]
        ) / self.config.translation_scale_m
        if (self._active_grasp_mode == GraspMode.RIM_PINCH
                and self._rim_retention_confirmed and gripper_command > 0.5):
            magnitude = float(np.linalg.norm(translation))
            limit = self.config.rim_loaded_translation_action_limit
            if magnitude > limit:
                translation *= limit / magnitude
        relative_world = target[:3, :3] @ current[:3, :3].T
        rotation = Rotation.from_matrix(relative_world).as_rotvec() / self.config.rotation_scale_rad
        action = np.concatenate((translation, rotation, (float(gripper_command),)))
        return OSCAction.from_array(action, clip=True)

    @staticmethod
    def _pose_errors(
        current: np.ndarray,
        target: np.ndarray,
        *,
        axisymmetric: bool = False,
    ) -> tuple[float, float]:
        position = float(np.linalg.norm(target[:3, 3] - current[:3, 3]))
        if axisymmetric:
            cosine = float(
                np.clip(np.dot(current[:3, 2], target[:3, 2]), -1.0, 1.0)
            )
            orientation = float(np.arccos(cosine))
        else:
            relative = target[:3, :3] @ current[:3, :3].T
            orientation = float(Rotation.from_matrix(relative).magnitude())
        return position, orientation


class ContactAwareRouteCController(RouteCController):
    """Treat a typed, proprioceptive GRASP contact stall as phase completion."""

    _FREE_RIM_ESCAPE_DISTANCE_M = 0.040
    _FREE_RIM_ESCAPE_PATH_SAMPLES = 7
    _FREE_RIM_ESCAPE_MIN_PROGRESS_M = 0.004
    _FREE_RIM_ESCAPE_MIN_HEIGHT_ABOVE_SOURCE_M = 0.080
    _FREE_RIM_ESCAPE_MAX_CROSS_DRIFT_M = 0.008
    _FREE_RIM_ESCAPE_MAX_VERTICAL_DRIFT_M = 0.008
    _FREE_RIM_TARGET_EGRESS_MAX_DISTANCE_M = 0.100
    _FREE_RIM_TARGET_EGRESS_WAYPOINT_SPACING_M = 0.020
    _FREE_RIM_TARGET_EGRESS_PATH_SAMPLES = 21
    _FREE_RIM_TARGET_EGRESS_SAFE_POSE_POSITION_TOLERANCE_M = 0.003
    _FREE_RIM_TARGET_EGRESS_SAFE_POSE_ROTATION_TOLERANCE_RAD = 0.020
    # A frame-local fused detector component can occasionally describe the
    # wrist/tool rather than a world-fixed fixture.  Reclassifying such a
    # recognised component as proprioceptive self geometry requires two
    # observations separated by a large, released-hand motion.  These hard
    # bounds are deliberately much tighter than scene-track association.
    _FREE_RIM_VIEW_FIELD_MIN_EE_MOTION_M = 0.050
    _FREE_RIM_VIEW_FIELD_MIN_FIELD_MOTION_M = 0.040
    _FREE_RIM_VIEW_FIELD_MAX_COMOTION_RESIDUAL_M = 0.015
    _FREE_RIM_VIEW_FIELD_MIN_DIRECTION_COSINE = 0.98
    _FREE_RIM_VIEW_FIELD_MIN_MOTION_RATIO = 0.65
    _FREE_RIM_VIEW_FIELD_MAX_MOTION_RATIO = 1.25
    _FREE_RIM_VIEW_FIELD_MAX_HALF_EXTENT_DRIFT_M = 0.012
    _FREE_RIM_VIEW_FIELD_MIN_OPEN_WIDTH_M = 0.070
    _FREE_RIM_VIEW_FIELD_MIN_CAMERAS = 2
    _FREE_RIM_VIEW_FIELD_MIN_POINTS_PER_CAMERA = 24
    _FREE_RIM_VIEW_FIELD_MAX_POINT_ERROR_M = 0.012
    _FREE_RIM_VIEW_FIELD_MIN_STATIC_ADVANTAGE_M = 0.008
    _FREE_RIM_VIEW_FIELD_MAX_STATIC_ERROR_RATIO = 0.55
    # A one-camera positive proof may be strengthened only by one independent
    # high/open view transaction.  Its second motion is deliberately
    # non-collinear with the already accepted target-egress segment so a
    # frame-local static crop cannot pass by sliding along one repeated edge.
    _FREE_RIM_VIEW_REACQUIRE_DISTANCE_M = 0.080
    _FREE_RIM_VIEW_REACQUIRE_PATH_SAMPLES = 15
    _FREE_RIM_VIEW_REACQUIRE_MAX_POLICY_ACTIONS = 19
    _FREE_RIM_VIEW_REACQUIRE_MAX_DIRECTION_COSINE = 0.50
    _FREE_RIM_VIEW_REACQUIRE_MAX_CROSS_DRIFT_M = 0.008
    _FREE_RIM_VIEW_REACQUIRE_MAX_VERTICAL_DRIFT_M = (
        _FREE_RIM_SENSOR_SAFE_MAX_VERTICAL_DRIFT_M
    )
    _FREE_RIM_VIEW_REACQUIRE_WORKSPACE_MARGIN_M = 0.010
    _FREE_RIM_VIEW_CAMERA_RIGID_POSITION_TOLERANCE_M = 0.002
    _FREE_RIM_VIEW_CAMERA_RIGID_ROTATION_TOLERANCE_RAD = 0.010
    _FREE_RIM_VIEW_OBB_UNCERTAINTY_M = 0.032
    _FREE_RIM_VIEW_FRUSTUM_NEAR_M = 0.025
    _FREE_RIM_VIEW_FRUSTUM_FAR_M = 3.0
    _FREE_RIM_VIEW_FRUSTUM_PIXEL_MARGIN = 2.0
    # Public Panda-hand geometry expressed in the public EE-site frame.  The
    # fixed envelope conservatively covers the hand body and both fully open
    # fingers (including a small calibration/depth margin), but not the swept
    # workspace around them.  Raw crops must predominantly occupy this volume
    # in every camera and every temporal endpoint.  This blocks a translated
    # crop of a static drawer plane from impersonating tool-rigid geometry.
    # This is a union, not one swept bounding box.  The rear box covers the
    # released ``hand.stl`` after its public +90-degree Z rotation and the
    # grip-site's +97-mm Z offset.  The narrower forward box covers either
    # finger over the public +/-40-mm slide range.  Small margins absorb RGB-D
    # and calibration noise without admitting the empty front corners of the
    # hand's overall AABB.
    _FREE_RIM_VIEW_TOOL_PALM_LOCAL_MIN_M = (-0.112, -0.040, -0.132)
    _FREE_RIM_VIEW_TOOL_PALM_LOCAL_MAX_M = (0.112, 0.040, -0.023)
    _FREE_RIM_VIEW_TOOL_FINGER_SWEEP_LOCAL_MIN_M = (-0.075, -0.020, -0.052)
    _FREE_RIM_VIEW_TOOL_FINGER_SWEEP_LOCAL_MAX_M = (0.075, 0.020, 0.020)
    _FREE_RIM_VIEW_TOOL_MIN_ENVELOPE_FRACTION = 0.80
    _FREE_RIM_VIEW_TOOL_MIN_ROBUST_SPAN_M = (0.030, 0.010, 0.020)
    _FREE_RIM_VIEW_TOOL_MIN_THICKNESS_M = 0.002
    _FREE_RIM_VIEW_TOOL_MIN_EIGENVALUE_RATIO = 0.003
    _FREE_RIM_VIEW_TOOL_MIN_NEGATIVE_X_M = 0.012
    _FREE_RIM_VIEW_TOOL_MIN_POSITIVE_X_M = 0.012
    _FREE_RIM_VIEW_TOOL_MIN_NEGATIVE_Y_M = 0.003
    _FREE_RIM_VIEW_TOOL_MIN_POSITIVE_Y_M = 0.003
    _FREE_RIM_VIEW_TOOL_MAX_CENTROID_ABS_X_M = 0.040
    _FREE_RIM_VIEW_TOOL_MAX_CENTROID_ABS_Y_M = 0.022
    _FREE_RIM_VIEW_TOOL_CENTROID_Z_RANGE_M = (-0.105, -0.010)
    _FREE_RIM_VIEW_TOOL_MIN_REAR_Z_M = 0.030
    _FREE_RIM_VIEW_TOOL_MIN_FORWARD_Z_M = -0.010

    def __init__(
        self,
        *args,
        expansion_visual_max_distance_m: float = 0.065,
        free_rim_view_clearance_m: float = 0.140,
        free_rim_view_retreat_m: float = 0.080,
        free_rim_temporal_center_tolerance_m: float = 0.015,
        free_rim_temporal_radius_tolerance_m: float = 0.008,
        free_rim_temporal_top_tolerance_m: float = 0.012,
        **kwargs,
    ) -> None:
        if expansion_visual_max_distance_m <= 0:
            raise ValueError("expansion visual distance must be positive")
        if (
            not 0.10 <= free_rim_view_clearance_m <= 0.18
            or not 0.06 <= free_rim_view_retreat_m <= 0.10
            or not 0.0 < free_rim_temporal_center_tolerance_m <= 0.015
            or not 0.0 < free_rim_temporal_radius_tolerance_m <= 0.008
            or not 0.0 < free_rim_temporal_top_tolerance_m <= 0.012
        ):
            raise ValueError(
                "free-rim active-view and temporal gates exceed calibrated bounds"
            )
        super().__init__(*args, **kwargs)
        self._integration_phase: Phase | None = None
        self.expansion_visual_max_distance_m = float(
            expansion_visual_max_distance_m
        )
        self._cavity_approach_safe_pose: np.ndarray | None = None
        self._cavity_recovery_pending = False
        self._cavity_recovery_labels: tuple[str, ...] = ()
        self._cavity_failed_candidate_id: str | None = None
        self._free_rim_escape_pending = False
        self._free_rim_escape_candidate_id: str | None = None
        self._free_rim_escape_source_id: str | None = None
        self._free_rim_escape_labels: tuple[str, ...] = ()
        self._free_rim_escape_nominal_envelope_m: float | None = None
        self._free_rim_escape_attempted_candidates: set[str] = set()
        self._free_rim_target_egress_pending = False
        self._free_rim_target_egress_candidate_id: str | None = None
        self._free_rim_target_egress_source_id: str | None = None
        self._free_rim_target_egress_source_label: str | None = None
        self._free_rim_target_egress_target_id: str | None = None
        self._free_rim_target_egress_target_label: str | None = None
        self._free_rim_target_egress_labels: tuple[str, ...] = ()
        self._free_rim_target_egress_diagnostic: dict[str, object] | None = None
        self._free_rim_target_egress_attempted_candidates: set[str] = set()
        self._free_rim_last_assembly_egress_proof: dict[str, object] | None = None
        self._free_rim_view_field_retried_candidates: set[str] = set()
        self._free_rim_view_reacquired_candidates: set[str] = set()
        self._free_rim_view_field_retry_active = False
        self.free_rim_view_clearance_m = float(free_rim_view_clearance_m)
        self.free_rim_view_retreat_m = float(free_rim_view_retreat_m)
        self.free_rim_temporal_center_tolerance_m = float(
            free_rim_temporal_center_tolerance_m
        )
        self.free_rim_temporal_radius_tolerance_m = float(
            free_rim_temporal_radius_tolerance_m
        )
        self.free_rim_temporal_top_tolerance_m = float(
            free_rim_temporal_top_tolerance_m
        )
        self._free_rim_active_view_count = 0

    def run(self, task_text: str) -> RouteCResult:
        """Stop immediately once no further policy action can be executed."""

        clear_self_filter = getattr(
            self.observer, "clear_temporal_proprio_self_filter", None
        )
        if callable(clear_self_filter):
            clear_self_filter()
        reset_grasps = getattr(self.grasp_provider, "reset_goal_context", None)
        if callable(reset_grasps):
            reset_grasps()
        self._cavity_approach_safe_pose = None
        self._cavity_recovery_pending = False
        self._cavity_recovery_labels = ()
        self._cavity_failed_candidate_id = None
        self._free_rim_escape_pending = False
        self._free_rim_escape_candidate_id = None
        self._free_rim_escape_source_id = None
        self._free_rim_escape_labels = ()
        self._free_rim_escape_nominal_envelope_m = None
        self._free_rim_escape_attempted_candidates.clear()
        self._free_rim_target_egress_pending = False
        self._free_rim_target_egress_candidate_id = None
        self._free_rim_target_egress_source_id = None
        self._free_rim_target_egress_source_label = None
        self._free_rim_target_egress_target_id = None
        self._free_rim_target_egress_target_label = None
        self._free_rim_target_egress_labels = ()
        self._free_rim_target_egress_diagnostic = None
        self._free_rim_target_egress_attempted_candidates.clear()
        self._free_rim_last_assembly_egress_proof = None
        self._free_rim_view_field_retried_candidates.clear()
        self._free_rim_view_reacquired_candidates.clear()
        self._free_rim_view_field_retry_active = False
        self._free_rim_active_view_count = 0
        try:
            return super().run(task_text)
        except PolicyStepBudgetExhausted as exc:
            # This is a policy-side terminal failure, not evaluator success.
            # The base controller journals completed attempts plus the active
            # interrupted phase without fabricating any recovery action.
            return self._interrupted_result(task_text, exc)

    def _append_robot_phase_trace(self, item: Mapping[str, Any]) -> None:
        trace = getattr(self.robot, "phase_trace", None)
        # A list subclass can override append/delete and make an ostensibly
        # transactional proof trace impossible to roll back.  Ordinary trace
        # writes therefore use only the exact built-in capability and invoke
        # its unoverrideable native method.
        if type(trace) is list:
            list.append(trace, dict(item))

    def _require_fresh_free_rim_visibility(
        self,
        source: SceneEntity,
        *,
        stage: str,
    ) -> None:
        """Require the resolved bowl id to have a current RGB-D detection.

        Both Route-C tracking layers deliberately retain a requested entity
        through a brief occlusion.  That cached geometry is useful as an
        association prior, but a newer global frame timestamp does not make
        it a fresh bowl observation.  Active-view reconstruction therefore
        admits only stable ids listed by the observer's latest public
        visibility set and fails closed when that capability is absent.
        """

        visible_value = getattr(self.observer, "visible_instance_ids", None)
        capability_available = visible_value is not None
        try:
            visible_ids = (
                {str(instance_id) for instance_id in visible_value}
                if capability_available
                else set()
            )
        except TypeError as exc:
            raise GraspBindingError(
                "free-space rim visibility capability is not a collection"
            ) from exc
        accepted = bool(source.instance_id in visible_ids)
        self._append_robot_phase_trace(
            {
                "phase": "free_rim_fresh_visibility_gate",
                "stage": str(stage),
                "source_id": source.instance_id,
                "visibility_capability_available": capability_available,
                "freshly_visible": accepted,
                "accepted": accepted,
                "grasp_semantics": "external_rim_close_fingers_only",
            }
        )
        if not capability_available:
            raise GraspBindingError(
                "free-space rim temporal reconstruction lacks a fresh RGB-D "
                "visibility capability"
            )
        if not accepted:
            raise GraspBindingError(
                f"free-space rim {stage} source is not freshly visible in RGB-D"
            )

    def _observe_and_resolve_source(
        self,
        graph,
        labels: Sequence[str],
    ) -> tuple[SceneEstimate, SceneEntity, object]:
        """Recover a malformed free-space rim OBB before grasp proposal.

        The ordinary observation is already fused from the public agent and
        wrist RGB-D cameras.  If that fused bowl OBB is not shallow and
        axisymmetric, move the empty hand vertically and then farther away in
        XY, drop pre-grasp tracks, and require two new cache-separated fused
        observations with mutually stable centre/radius/top geometry.  No
        physical candidate exists yet, so this recovery cannot consume the
        near or antipodal rim failure keys.
        """

        observed = super()._observe_and_resolve_source(graph, labels)
        scene, source, _ = observed
        geometry_for = getattr(
            self.grasp_provider, "free_space_rim_geometry_evidence", None
        )
        selector = getattr(graph.source, "selector", None)
        if (
            not callable(geometry_for)
            or selector is not None
            or self.grasp_mode_selector.select(source.label)
            is not GraspMode.RIM_PINCH
        ):
            return observed
        initial_evidence = geometry_for(source)
        if bool(initial_evidence.get("accepted", False)):
            return observed
        if self._free_rim_active_view_count >= 1:
            raise GraspBindingError(
                "free-space rim geometry remained invalid after its bounded "
                "active view"
            )

        self._execute_free_rim_active_view(scene, source, initial_evidence)
        self._invalidate_for_selector_reacquisition()
        self._free_rim_active_view_count += 1
        first = super()._observe_and_resolve_source(graph, labels)
        first_scene, first_source, _ = first
        self._require_fresh_free_rim_visibility(
            first_source,
            stage="first reconstructed frame",
        )
        first_evidence = geometry_for(first_source)

        # RGB-D frames in the real evaluator are published on environment
        # actions.  Invalidating only the perception cache therefore rebuilds
        # the same raw frame and leaves both estimates with the same public
        # timestamp.  Ask the capability-limited robot for exactly one
        # zero-Cartesian, open-hand hold before requesting the second frame.
        # Legacy/test adapters without content commitments remain admissible
        # only if their independently supplied scene timestamp advances below.
        refresh_hold = getattr(
            self.robot,
            "capture_fresh_sensor_frame_at_pose",
            None,
        )
        refresh_hold_available = callable(refresh_hold)
        refresh_hold_executed = False
        if refresh_hold_available:
            refresh_feedback = refresh_hold(
                self._released_gripper_command(GraspMode.RIM_PINCH)
            )
            if not isinstance(refresh_feedback, ControllerFeedback):
                raise GraspBindingError(
                    "free-space rim sensor refresh returned an invalid feedback type"
                )
            if not refresh_feedback.accepted:
                raise GraspBindingError(
                    refresh_feedback.detail
                    or "free-space rim fixed-pose sensor refresh failed"
                )
            refresh_hold_executed = True

        invalidate = getattr(self.observer, "invalidate_sensor_cache", None)
        if not callable(invalidate):
            raise GraspBindingError(
                "free-space rim temporal reconstruction cannot force fresh RGB-D"
            )
        invalidate()
        second = super()._observe_and_resolve_source(graph, labels)
        second_scene, second_source, _ = second
        self._require_fresh_free_rim_visibility(
            second_source,
            stage="second reconstructed frame",
        )
        second_evidence = geometry_for(second_source)

        center_error = float(
            np.linalg.norm(
                np.asarray(second_evidence["source_center_world_m"])
                - np.asarray(first_evidence["source_center_world_m"])
            )
        )
        radius_error = abs(
            float(second_evidence["rim_radius_m"])
            - float(first_evidence["rim_radius_m"])
        )
        top_error = abs(
            float(second_evidence["top_z_m"])
            - float(first_evidence["top_z_m"])
        )
        capture_advanced = sensor_capture_advanced(first_scene, second_scene)
        same_stable_source = bool(
            second_source.instance_id == first_source.instance_id
        )
        stable = bool(
            bool(first_evidence.get("accepted", False))
            and bool(second_evidence.get("accepted", False))
            and capture_advanced
            and same_stable_source
            and center_error <= self.free_rim_temporal_center_tolerance_m
            and radius_error <= self.free_rim_temporal_radius_tolerance_m
            and top_error <= self.free_rim_temporal_top_tolerance_m
        )
        self._append_robot_phase_trace(
            {
                "phase": "free_rim_dual_rgbd_temporal_reconstruction",
                "initial_geometry": initial_evidence,
                "first_fresh_geometry": first_evidence,
                "second_fresh_geometry": second_evidence,
                "first_timestamp_s": float(first_scene.timestamp_s),
                "second_timestamp_s": float(second_scene.timestamp_s),
                "first_capture_id": first_scene.capture_id,
                "second_capture_id": second_scene.capture_id,
                "sensor_capture_advanced": capture_advanced,
                "same_stable_source": same_stable_source,
                "sensor_refresh_hold_available": refresh_hold_available,
                "sensor_refresh_hold_executed": refresh_hold_executed,
                "sensor_refresh_hold_policy_actions": (
                    1 if refresh_hold_executed else 0
                ),
                "center_error_m": center_error,
                "radius_error_m": radius_error,
                "top_error_m": top_error,
                "accepted": stable,
                "candidate_failure_keys_consumed": False,
                "grasp_semantics": "external_rim_close_fingers_only",
            }
        )
        if not stable:
            raise GraspBindingError(
                "free-space rim fresh dual-RGB-D geometry is temporally "
                "inconsistent"
            )
        return second

    def _execute_free_rim_active_view(
        self,
        scene: SceneEstimate,
        source: SceneEntity,
        evidence: Mapping[str, object],
    ) -> None:
        """Move an empty hand up, then away from an occluded bowl in XY."""

        current = np.asarray(self.robot.current_ee_pose(), dtype=np.float64)
        if current.shape != (4, 4) or not np.all(np.isfinite(current)):
            raise GraspBindingError(
                "free-space rim active view lacks a finite public EE pose"
            )
        top_z = float(evidence["top_z_m"])
        ceiling = min(
            self.config.recovery_motion_ceiling_z_m,
            float(scene.workspace_max[2]),
        )
        safe_z = max(float(current[2, 3]), top_z + self.free_rim_view_clearance_m)
        if not np.isfinite(safe_z) or safe_z > ceiling + 1e-9:
            raise GraspBindingError(
                "free-space rim has no calibrated high active-view clearance"
            )
        command = self._released_gripper_command(GraspMode.RIM_PINCH)
        raised = current.copy()
        raised[2, 3] = safe_z
        if safe_z > float(current[2, 3]) + 1e-4:
            feedback = self.robot.execute_waypoints(
                np.stack((current, raised)), Phase.RETREAT, command
            )
            if not feedback.accepted:
                if bool(
                    getattr(self.robot, "step_budget_exhausted", False)
                ):
                    raise PolicyStepBudgetExhausted(
                        "episode OSC step budget exhausted during vertical "
                        "free-space rim active view"
                    )
                raise GraspBindingError(
                    feedback.detail or "free-space rim vertical active view failed"
                )
        self._append_robot_phase_trace(
            {
                "phase": "free_rim_active_view_vertical",
                "start_xyz_m": current[:3, 3].tolist(),
                "goal_xyz_m": raised[:3, 3].tolist(),
                "accepted": True,
            }
        )

        bearing = raised[:2, 3] - source.position[:2]
        bearing_norm = float(np.linalg.norm(bearing))
        if bearing_norm < 1e-6:
            bearing = np.asarray(raised[:2, 0], dtype=np.float64)
            bearing_norm = float(np.linalg.norm(bearing))
        if bearing_norm < 1e-6:
            raise GraspBindingError(
                "free-space rim active view lacks an outward planar bearing"
            )
        bearing /= bearing_norm
        view = raised.copy()
        view[:2, 3] += bearing * self.free_rim_view_retreat_m
        lower = scene.workspace_min[:2] + self.config.selector_view_workspace_margin_m
        upper = scene.workspace_max[:2] - self.config.selector_view_workspace_margin_m
        if np.any(view[:2, 3] < lower) or np.any(view[:2, 3] > upper):
            raise GraspBindingError(
                "free-space rim active-view retreat leaves calibrated workspace"
            )
        feedback = self.robot.execute_waypoints(
            np.stack((raised, view)), Phase.RETREAT, command
        )
        if not feedback.accepted:
            if bool(getattr(self.robot, "step_budget_exhausted", False)):
                raise PolicyStepBudgetExhausted(
                    "episode OSC step budget exhausted during lateral "
                    "free-space rim active view"
                )
            raise GraspBindingError(
                feedback.detail or "free-space rim lateral active view failed"
            )
        self._append_robot_phase_trace(
            {
                "phase": "free_rim_active_view_retreat",
                "start_xyz_m": raised[:3, 3].tolist(),
                "goal_xyz_m": view[:3, 3].tolist(),
                "retreat_from_source_world": bearing.tolist(),
                "retreat_distance_m": self.free_rim_view_retreat_m,
                "accepted": True,
                "next_observation_requires_fresh_dual_rgbd": True,
            }
        )

    @staticmethod
    def _recovery_pose_errors(
        current: np.ndarray, goal: np.ndarray
    ) -> tuple[float, float]:
        position_error = float(np.linalg.norm(goal[:3, 3] - current[:3, 3]))
        relative = goal[:3, :3] @ current[:3, :3].T
        rotation_error = float(Rotation.from_matrix(relative).magnitude())
        return position_error, rotation_error

    @staticmethod
    def _public_ee_pose(robot: object) -> np.ndarray | None:
        pose_provider = getattr(robot, "current_ee_pose", None)
        if not callable(pose_provider):
            return None
        try:
            pose = np.asarray(pose_provider(), dtype=np.float64)
        except (TypeError, ValueError, RuntimeError):
            return None
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            return None
        return pose.copy()

    @classmethod
    def _public_ee_xyz(cls, robot: object) -> tuple[list[float] | None, bool]:
        """Read only a finite public proprioceptive EE position, if exposed."""

        pose = cls._public_ee_pose(robot)
        return (None, False) if pose is None else (pose[:3, 3].tolist(), True)

    @staticmethod
    def _canonical_view_name(value: object) -> str:
        # The typed negative-visibility proof is defined over the two policy
        # cameras, not over aliases supplied by a detector or simulator.  Do
        # not silently promote e.g. ``robot0_eye_in_hand`` or a substring to a
        # calibrated policy view.
        return value if type(value) is str and value in {"agentview", "wrist"} else ""

    @staticmethod
    def _builtin_finite_number(value: object) -> float | None:
        # ``bool`` and numpy scalar subclasses are intentionally rejected.  A
        # proof counter or calibration scalar must be an ordinary serialized
        # policy value, not a truth value masquerading as an integer.
        if type(value) not in (int, float):
            return None
        result = float(value)
        return result if np.isfinite(result) else None

    @classmethod
    def _public_view_snapshot(cls, robot: object) -> dict[str, object] | None:
        provider = getattr(robot, "current_public_view_snapshot", None)
        if not callable(provider):
            return None
        try:
            raw = provider()
        except (TypeError, ValueError, RuntimeError):
            return None
        if not isinstance(raw, Mapping) or set(raw) != {
            "ee_pose_world",
            "gripper_width_m",
            "cameras",
        }:
            return None
        pose = np.asarray(raw["ee_pose_world"], dtype=np.float64)
        width = cls._builtin_finite_number(raw["gripper_width_m"])
        if (
            pose.shape != (4, 4)
            or not np.all(np.isfinite(pose))
            or not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6)
            or not np.allclose(
                pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=2e-3
            )
            or float(np.linalg.det(pose[:3, :3])) <= 0.0
            or width is None
            or not 0.0 <= width <= 0.12
        ):
            return None
        try:
            rows = tuple(raw["cameras"])
        except TypeError:
            return None
        cameras: dict[str, dict[str, object]] = {}
        required = {
            "policy_name",
            "sensor_name",
            "width_px",
            "height_px",
            "intrinsic",
            "world_from_camera",
            "observation_v_flipped",
        }
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != required:
                return None
            policy_name = cls._canonical_view_name(row["policy_name"])
            sensor_name = row["sensor_name"]
            width_px = row["width_px"]
            height_px = row["height_px"]
            flipped = row["observation_v_flipped"]
            intrinsic = np.asarray(row["intrinsic"], dtype=np.float64)
            world_from_camera = np.asarray(
                row["world_from_camera"], dtype=np.float64
            )
            if (
                not policy_name
                or policy_name in cameras
                or type(sensor_name) is not str
                or not sensor_name.strip()
                or type(width_px) is not int
                or type(height_px) is not int
                or width_px < 8
                or height_px < 8
                or type(flipped) is not bool
                or intrinsic.shape != (3, 3)
                or world_from_camera.shape != (4, 4)
                or not np.all(np.isfinite(intrinsic))
                or not np.all(np.isfinite(world_from_camera))
                or intrinsic[0, 0] <= 0.0
                or intrinsic[1, 1] <= 0.0
                or abs(float(intrinsic[0, 1])) > 1e-12
                or abs(float(intrinsic[1, 0])) > 1e-12
                or not np.allclose(intrinsic[2], (0.0, 0.0, 1.0), atol=1e-6)
                or not 0.0 <= intrinsic[0, 2] <= float(width_px - 1)
                or not 0.0 <= intrinsic[1, 2] <= float(height_px - 1)
                or not np.allclose(
                    world_from_camera[3],
                    (0.0, 0.0, 0.0, 1.0),
                    atol=1e-6,
                )
                or not np.allclose(
                    world_from_camera[:3, :3].T
                    @ world_from_camera[:3, :3],
                    np.eye(3),
                    atol=2e-3,
                )
                or float(np.linalg.det(world_from_camera[:3, :3])) <= 0.0
            ):
                return None
            cameras[policy_name] = {
                "policy_name": policy_name,
                "sensor_name": sensor_name.strip(),
                "width_px": width_px,
                "height_px": height_px,
                "intrinsic": intrinsic.copy(),
                "world_from_camera": world_from_camera.copy(),
                "observation_v_flipped": flipped,
            }
        if set(cameras) != {"agentview", "wrist"}:
            return None
        return {
            "ee_pose_world": pose.copy(),
            "gripper_width_m": width,
            "cameras": cameras,
        }

    @staticmethod
    def _rotation_distance_rad(first: np.ndarray, second: np.ndarray) -> float:
        return float(
            Rotation.from_matrix(first[:3, :3] @ second[:3, :3].T).magnitude()
        )

    @classmethod
    def _camera_snapshot_pair_evidence(
        cls,
        first: Mapping[str, object],
        second: Mapping[str, object],
    ) -> dict[str, object]:
        """Check fixed-view stability and the public wrist/EE rigid transform."""

        try:
            first_ee = np.asarray(first["ee_pose_world"], dtype=np.float64)
            second_ee = np.asarray(second["ee_pose_world"], dtype=np.float64)
            first_cameras = first["cameras"]
            second_cameras = second["cameras"]
            if not isinstance(first_cameras, Mapping) or not isinstance(
                second_cameras, Mapping
            ):
                raise TypeError
            first_agent = first_cameras["agentview"]
            second_agent = second_cameras["agentview"]
            first_wrist = first_cameras["wrist"]
            second_wrist = second_cameras["wrist"]
            if not all(
                isinstance(item, Mapping)
                for item in (first_agent, second_agent, first_wrist, second_wrist)
            ):
                raise TypeError
            required_camera_keys = {
                "policy_name",
                "sensor_name",
                "width_px",
                "height_px",
                "intrinsic",
                "world_from_camera",
                "observation_v_flipped",
            }
            if (
                set(first) != {"ee_pose_world", "gripper_width_m", "cameras"}
                or set(second) != {"ee_pose_world", "gripper_width_m", "cameras"}
                or set(first_cameras) != {"agentview", "wrist"}
                or set(second_cameras) != {"agentview", "wrist"}
                or any(
                    set(camera) != required_camera_keys
                    for camera in (
                        first_agent,
                        second_agent,
                        first_wrist,
                        second_wrist,
                    )
                )
                or first_agent.get("policy_name") != "agentview"
                or second_agent.get("policy_name") != "agentview"
                or first_wrist.get("policy_name") != "wrist"
                or second_wrist.get("policy_name") != "wrist"
                or any(
                    type(camera.get(key)) is not int
                    for camera in (
                        first_agent,
                        second_agent,
                        first_wrist,
                        second_wrist,
                    )
                    for key in ("width_px", "height_px")
                )
                or any(
                    type(camera.get("observation_v_flipped")) is not bool
                    for camera in (
                        first_agent,
                        second_agent,
                        first_wrist,
                        second_wrist,
                    )
                )
                or cls._builtin_finite_number(first.get("gripper_width_m"))
                is None
                or cls._builtin_finite_number(second.get("gripper_width_m"))
                is None
            ):
                raise TypeError
            first_agent_pose = np.asarray(
                first_agent["world_from_camera"], dtype=np.float64
            )
            second_agent_pose = np.asarray(
                second_agent["world_from_camera"], dtype=np.float64
            )
            first_wrist_pose = np.asarray(
                first_wrist["world_from_camera"], dtype=np.float64
            )
            second_wrist_pose = np.asarray(
                second_wrist["world_from_camera"], dtype=np.float64
            )
            first_ee_from_wrist = np.linalg.inv(first_ee) @ first_wrist_pose
            second_ee_from_wrist = np.linalg.inv(second_ee) @ second_wrist_pose
        except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
            return {"accepted": False, "reason": "malformed_camera_snapshot_pair"}
        transforms = (
            first_ee,
            second_ee,
            first_agent_pose,
            second_agent_pose,
            first_wrist_pose,
            second_wrist_pose,
            first_ee_from_wrist,
            second_ee_from_wrist,
        )
        if any(
            transform.shape != (4, 4)
            or not np.all(np.isfinite(transform))
            or not np.allclose(
                transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6
            )
            or not np.allclose(
                transform[:3, :3].T @ transform[:3, :3],
                np.eye(3),
                atol=2e-3,
            )
            or float(np.linalg.det(transform[:3, :3])) <= 0.0
            for transform in transforms
        ):
            return {"accepted": False, "reason": "malformed_camera_snapshot_pair"}
        camera_models = (
            first_agent,
            second_agent,
            first_wrist,
            second_wrist,
        )
        for camera in camera_models:
            try:
                intrinsic = np.asarray(camera["intrinsic"], dtype=np.float64)
                width_px = camera["width_px"]
                height_px = camera["height_px"]
            except (KeyError, TypeError, ValueError):
                return {
                    "accepted": False,
                    "reason": "malformed_camera_snapshot_pair",
                }
            if (
                type(camera.get("sensor_name")) is not str
                or not str(camera.get("sensor_name")).strip()
                or width_px < 8
                or height_px < 8
                or intrinsic.shape != (3, 3)
                or not np.all(np.isfinite(intrinsic))
                or intrinsic[0, 0] <= 0.0
                or intrinsic[1, 1] <= 0.0
                or abs(float(intrinsic[0, 1])) > 1e-12
                or abs(float(intrinsic[1, 0])) > 1e-12
                or not np.allclose(
                    intrinsic[2], (0.0, 0.0, 1.0), atol=1e-12
                )
                or not 0.0 <= intrinsic[0, 2] <= float(width_px - 1)
                or not 0.0 <= intrinsic[1, 2] <= float(height_px - 1)
            ):
                return {
                    "accepted": False,
                    "reason": "malformed_camera_snapshot_pair",
                }
        same_calibration = all(
            bool(
                first_camera[key] == second_camera[key]
                if key in {"sensor_name", "width_px", "height_px", "observation_v_flipped"}
                else np.allclose(
                    np.asarray(first_camera[key], dtype=np.float64),
                    np.asarray(second_camera[key], dtype=np.float64),
                    rtol=0.0,
                    atol=1e-9,
                )
            )
            for first_camera, second_camera in (
                (first_agent, second_agent),
                (first_wrist, second_wrist),
            )
            for key in (
                "sensor_name",
                "width_px",
                "height_px",
                "intrinsic",
                "observation_v_flipped",
            )
        )
        fixed_position_drift = float(
            np.linalg.norm(first_agent_pose[:3, 3] - second_agent_pose[:3, 3])
        )
        fixed_rotation_drift = cls._rotation_distance_rad(
            first_agent_pose, second_agent_pose
        )
        wrist_relative_position_drift = float(
            np.linalg.norm(
                first_ee_from_wrist[:3, 3] - second_ee_from_wrist[:3, 3]
            )
        )
        wrist_relative_rotation_drift = cls._rotation_distance_rad(
            first_ee_from_wrist, second_ee_from_wrist
        )
        accepted = bool(
            same_calibration
            and fixed_position_drift
            <= cls._FREE_RIM_VIEW_CAMERA_RIGID_POSITION_TOLERANCE_M
            and fixed_rotation_drift
            <= cls._FREE_RIM_VIEW_CAMERA_RIGID_ROTATION_TOLERANCE_RAD
            and wrist_relative_position_drift
            <= cls._FREE_RIM_VIEW_CAMERA_RIGID_POSITION_TOLERANCE_M
            and wrist_relative_rotation_drift
            <= cls._FREE_RIM_VIEW_CAMERA_RIGID_ROTATION_TOLERANCE_RAD
        )
        return {
            "accepted": accepted,
            "reason": (
                "public_fixed_and_wrist_camera_calibration_stable"
                if accepted
                else "public_camera_calibration_or_rigid_transform_drift"
            ),
            "fixed_position_drift_m": fixed_position_drift,
            "fixed_rotation_drift_rad": fixed_rotation_drift,
            "wrist_relative_position_drift_m": wrist_relative_position_drift,
            "wrist_relative_rotation_drift_rad": wrist_relative_rotation_drift,
        }

    @classmethod
    def _wrist_frustum_evidence(
        cls,
        center_world_m: object,
        half_extents_m: object,
        axes_world: object,
        wrist_model: Mapping[str, object],
        surface_points_world: object | None = None,
    ) -> dict[str, object]:
        """Conservatively classify an uncertainty-inflated OBB against a frustum."""

        center = np.asarray(center_world_m, dtype=np.float64)
        half = np.asarray(half_extents_m, dtype=np.float64)
        axes = np.asarray(axes_world, dtype=np.float64)
        try:
            intrinsic = np.asarray(wrist_model["intrinsic"], dtype=np.float64)
            world_from_camera = np.asarray(
                wrist_model["world_from_camera"], dtype=np.float64
            )
            width_px = wrist_model["width_px"]
            height_px = wrist_model["height_px"]
            flipped = wrist_model["observation_v_flipped"]
        except (KeyError, TypeError, ValueError):
            return {"accepted": False, "reason": "malformed_wrist_camera_model"}
        if (
            center.shape != (3,)
            or half.shape != (3,)
            or axes.shape != (3, 3)
            or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(half))
            or np.any(half <= 0.0)
            or not np.all(np.isfinite(axes))
            or not np.allclose(axes.T @ axes, np.eye(3), atol=2e-3)
            or intrinsic.shape != (3, 3)
            or world_from_camera.shape != (4, 4)
            or not np.all(np.isfinite(intrinsic))
            or not np.all(np.isfinite(world_from_camera))
            or type(width_px) is not int
            or type(height_px) is not int
            or type(flipped) is not bool
            or width_px < 8
            or height_px < 8
            or intrinsic[0, 0] <= 0.0
            or intrinsic[1, 1] <= 0.0
            or abs(float(intrinsic[0, 1])) > 1e-12
            or abs(float(intrinsic[1, 0])) > 1e-12
            or not np.allclose(intrinsic[2], (0.0, 0.0, 1.0), atol=1e-12)
            or not 0.0 <= intrinsic[0, 2] <= float(width_px - 1)
            or not 0.0 <= intrinsic[1, 2] <= float(height_px - 1)
            or not np.allclose(
                world_from_camera[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6
            )
            or not np.allclose(
                world_from_camera[:3, :3].T
                @ world_from_camera[:3, :3],
                np.eye(3),
                atol=2e-3,
            )
            or float(np.linalg.det(world_from_camera[:3, :3])) <= 0.0
        ):
            return {"accepted": False, "reason": "malformed_obb_or_wrist_calibration"}
        expanded = half + cls._FREE_RIM_VIEW_OBB_UNCERTAINTY_M
        camera_center_local = (
            world_from_camera[:3, 3] - center
        ) @ axes
        if np.all(np.abs(camera_center_local) <= expanded + 1e-12):
            return {
                "accepted": False,
                "reason": "wrist_camera_origin_inside_uncertainty_inflated_obb",
            }
        signs = np.array(
            [
                (x, y, z)
                for x in (-1.0, 1.0)
                for y in (-1.0, 1.0)
                for z in (-1.0, 1.0)
            ],
            dtype=np.float64,
        )
        corners_world = center[None, :] + (signs * expanded[None, :]) @ axes.T
        try:
            camera_from_world = np.linalg.inv(world_from_camera)
        except np.linalg.LinAlgError:
            return {"accepted": False, "reason": "singular_wrist_calibration"}
        corners_camera = (
            corners_world @ camera_from_world[:3, :3].T
            + camera_from_world[:3, 3]
        )
        x = corners_camera[:, 0]
        y = corners_camera[:, 1]
        z = corners_camera[:, 2]
        fx = float(intrinsic[0, 0])
        fy = float(intrinsic[1, 1])
        cx = float(intrinsic[0, 2])
        cy = float(intrinsic[1, 2])
        margin = cls._FREE_RIM_VIEW_FRUSTUM_PIXEL_MARGIN
        visible_cy = float(height_px - 1) - cy if flipped else cy
        visible_fy = -fy if flipped else fy
        plane_values = {
            "near": z - cls._FREE_RIM_VIEW_FRUSTUM_NEAR_M,
            "far": cls._FREE_RIM_VIEW_FRUSTUM_FAR_M - z,
            # Expand, rather than shrink, the image rectangle by the public
            # calibration uncertainty: u/v in [-margin, size-1+margin].
            "left": fx * x + (cx + margin) * z,
            "right": (float(width_px - 1) + margin - cx) * z - fx * x,
            "top": visible_fy * y + (visible_cy + margin) * z,
            "bottom": (
                (float(height_px - 1) + margin - visible_cy) * z
                - visible_fy * y
            ),
        }
        if not all(np.all(np.isfinite(values)) for values in plane_values.values()):
            return {"accepted": False, "reason": "nonfinite_frustum_projection"}
        separating = [
            name
            for name, values in plane_values.items()
            if float(np.max(values)) < -1e-9
        ]
        negative_source = "uncertainty_inflated_field_obb"
        if not separating and surface_points_world is not None:
            points = np.asarray(surface_points_world, dtype=np.float64)
            if (
                points.ndim == 2
                and points.shape[1:] == (3,)
                and len(points) >= cls._FREE_RIM_VIEW_FIELD_MIN_POINTS_PER_CAMERA
                and np.all(np.isfinite(points))
            ):
                camera_points = (
                    points @ camera_from_world[:3, :3].T
                    + camera_from_world[:3, 3]
                )
                px = camera_points[:, 0]
                py = camera_points[:, 1]
                pz = camera_points[:, 2]
                point_plane_values = {
                    "near": pz - cls._FREE_RIM_VIEW_FRUSTUM_NEAR_M,
                    "far": cls._FREE_RIM_VIEW_FRUSTUM_FAR_M - pz,
                    "left": fx * px + (cx + margin) * pz,
                    "right": (
                        (float(width_px - 1) + margin - cx) * pz - fx * px
                    ),
                    "top": visible_fy * py + (visible_cy + margin) * pz,
                    "bottom": (
                        (float(height_px - 1) + margin - visible_cy) * pz
                        - visible_fy * py
                    ),
                }
                # The field's independently gated raw points carry metric RGB-D
                # uncertainty.  Expanding every point by the full alignment
                # tolerance keeps a same-plane separation a strict negative
                # visibility proof instead of treating detector absence as one.
                coefficients = {
                    "near": np.array((0.0, 0.0, 1.0)),
                    "far": np.array((0.0, 0.0, -1.0)),
                    "left": np.array((fx, 0.0, cx + margin)),
                    "right": np.array(
                        (-fx, 0.0, float(width_px - 1) + margin - cx)
                    ),
                    "top": np.array((0.0, visible_fy, visible_cy + margin)),
                    "bottom": np.array(
                        (
                            0.0,
                            -visible_fy,
                            float(height_px - 1) + margin - visible_cy,
                        )
                    ),
                }
                separating = [
                    name
                    for name, values in point_plane_values.items()
                    if float(np.max(values))
                    + cls._FREE_RIM_VIEW_FIELD_MAX_POINT_ERROR_M
                    * float(np.linalg.norm(coefficients[name]))
                    < -1e-9
                ]
                if separating:
                    negative_source = (
                        "tool_morphology_gated_agentview_raw_surface"
                    )
        strictly_outside = bool(separating)
        if strictly_outside:
            reason = (
                "uncertainty_inflated_obb_strictly_outside_wrist_frustum"
                if negative_source == "uncertainty_inflated_field_obb"
                else "uncertainty_bounded_tool_surface_strictly_outside_wrist_frustum"
            )
        else:
            reason = "uncertainty_inflated_obb_may_intersect_wrist_frustum"
        return {
            "accepted": True,
            "reason": reason,
            "strictly_outside": strictly_outside,
            "theoretically_visible": not strictly_outside,
            "separating_planes": sorted(separating),
            "negative_geometry_source": negative_source,
            "minimum_camera_depth_m": float(np.min(z)),
            "maximum_camera_depth_m": float(np.max(z)),
            "obb_uncertainty_inflation_m": cls._FREE_RIM_VIEW_OBB_UNCERTAINTY_M,
        }

    @staticmethod
    def _start_clearance_envelope_from_failure(
        failure: BaseException,
    ) -> float | None:
        """Recover the failed nominal envelope from an MPC diagnostic.

        The escape is intentionally unavailable unless the optimiser itself
        identifies the raw-SDF argmin as the fixed start sample and reports
        reachability/height as feasible.  Thus an interior or goal obstacle
        can never be reclassified merely from a fresh point query.
        """

        if not isinstance(failure, OptimisationError):
            return None
        detail = str(failure)
        required = (
            "approach has no feasible trajectory:",
            "clearance violation=",
            "sample_location=start",
            "is_start=True",
            "is_end=False",
            "checks reach=True, height=True, clearance=False",
        )
        if not all(token in detail for token in required):
            return None
        minimum_match = re.search(
            r"\(minimum=([-+0-9.eE]+), limit=([-+0-9.eE]+)\)",
            detail,
        )
        raw_match = re.search(r"raw_sdf=([-+0-9.eE]+)", detail)
        if minimum_match is None or raw_match is None:
            return None
        try:
            minimum_clearance = float(minimum_match.group(1))
            clearance_limit = float(minimum_match.group(2))
            raw_distance = float(raw_match.group(1))
        except ValueError:
            return None
        # ``trajectory.min_clearance`` is raw SDF minus tool radius.  Recover
        # the exact combined envelope used by that failed request without
        # inventing a controller-side collision radius.
        tool_radius = raw_distance - minimum_clearance
        nominal_envelope = tool_radius + clearance_limit
        if (
            not np.all(
                np.isfinite(
                    (minimum_clearance, clearance_limit, raw_distance, tool_radius)
                )
            )
            or raw_distance <= 0.0
            or tool_radius < 0.0
            or clearance_limit < 0.0
            or nominal_envelope <= raw_distance
        ):
            return None
        return float(nominal_envelope)

    @staticmethod
    def _negative_start_clearance_diagnostic(
        failure: BaseException,
    ) -> dict[str, object] | None:
        """Parse one explicit negative-raw, start-only MPC diagnostic."""

        if not isinstance(failure, OptimisationError):
            return None
        detail = str(failure)
        required = (
            "approach has no feasible trajectory:",
            "clearance violation=",
            "sample_location=start",
            "is_start=True",
            "is_end=False",
            "checks reach=True, height=True, clearance=False",
        )
        if not all(token in detail for token in required):
            return None
        minimum_match = re.search(
            r"\(minimum=([-+0-9.eE]+), limit=([-+0-9.eE]+)\)",
            detail,
        )
        raw_match = re.search(r"raw_sdf=([-+0-9.eE]+)", detail)
        id_match = re.search(r"field_id=([^,]+)", detail)
        label_match = re.search(r"field_label=([^,]+)", detail)
        center_match = re.search(r"field_center=\[([^\]]+)\]", detail)
        half_match = re.search(r"field_half_extents=\[([^\]]+)\]", detail)
        point_match = re.search(r"point_xyz=\[([^\]]+)\]", detail)
        if any(
            match is None
            for match in (
                minimum_match,
                raw_match,
                id_match,
                label_match,
                center_match,
                half_match,
                point_match,
            )
        ):
            return None

        def vector(match: re.Match[str]) -> np.ndarray:
            return np.asarray(
                [float(item.strip()) for item in match.group(1).split(",")],
                dtype=np.float64,
            )

        assert minimum_match is not None
        assert raw_match is not None
        assert id_match is not None
        assert label_match is not None
        assert center_match is not None
        assert half_match is not None
        assert point_match is not None
        try:
            minimum_clearance = float(minimum_match.group(1))
            clearance_limit = float(minimum_match.group(2))
            raw_distance = float(raw_match.group(1))
            center = vector(center_match)
            half_extents = vector(half_match)
            point = vector(point_match)
        except ValueError:
            return None
        tool_radius = raw_distance - minimum_clearance
        nominal_envelope = tool_radius + clearance_limit
        field_id = id_match.group(1).strip()
        field_label = label_match.group(1).strip()
        if (
            center.shape != (3,)
            or half_extents.shape != (3,)
            or point.shape != (3,)
            or not np.all(
                np.isfinite(
                    (
                        minimum_clearance,
                        clearance_limit,
                        raw_distance,
                        tool_radius,
                        nominal_envelope,
                    )
                )
            )
            or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(half_extents))
            or np.any(half_extents <= 0.0)
            or raw_distance >= 0.0
            or tool_radius < 0.0
            or clearance_limit < 0.0
            or nominal_envelope <= 0.0
            or not field_id
            or field_id == "unknown"
            or not field_label
            or field_label == "unknown"
        ):
            return None
        return {
            "raw_distance_m": raw_distance,
            "minimum_clearance_m": minimum_clearance,
            "clearance_limit_m": clearance_limit,
            "tool_radius_m": tool_radius,
            "nominal_envelope_m": nominal_envelope,
            "field_id": field_id,
            "field_label": field_label,
            "field_center_world_m": center,
            "field_half_extents_m": half_extents,
            "point_world_m": point,
        }

    @staticmethod
    def _normalise_semantic_label(label: object) -> str:
        if not isinstance(label, str):
            return ""
        return " ".join(label.lower().replace("_", " ").split())

    @classmethod
    def _is_target_container_diagnostic(
        cls,
        bound: object,
        diagnostic: Mapping[str, object],
    ) -> bool:
        field_label = cls._normalise_semantic_label(
            diagnostic.get("field_label")
        )
        target_label = cls._normalise_semantic_label(
            getattr(bound, "target_label", "")
        )
        source_label = cls._normalise_semantic_label(
            getattr(bound, "source_label", "")
        )
        field_id = str(diagnostic.get("field_id", ""))
        source_id = str(getattr(bound, "source_id", ""))
        target_id = str(getattr(bound, "target_id", ""))
        graph = getattr(bound, "graph", None)
        graph_target = getattr(graph, "target", None)
        graph_target_label = cls._normalise_semantic_label(
            getattr(graph_target, "label", "")
        )
        graph_target_role = getattr(graph_target, "role", None)
        graph_relation = getattr(graph, "goal_relation", None)
        container_tokens = {"drawer", "cabinet"}
        field_tokens = set(field_label.split())
        target_tokens = set(target_label.split())
        return bool(
            field_tokens & container_tokens
            and target_tokens & container_tokens
            and graph_target_role == "target"
            and graph_target_label == target_label
            and graph_relation == Relation.IN
            and field_label != source_label
            and field_id != source_id
            and source_id
            and target_id
        )

    def _mark_free_space_environment_blocked(
        self, candidate_id: str | None
    ) -> None:
        block = getattr(
            self.grasp_provider, "mark_free_space_environment_blocked", None
        )
        if callable(block):
            block()
            return
        mark_failed = getattr(self.grasp_provider, "mark_candidate_failed", None)
        if callable(mark_failed) and candidate_id:
            mark_failed(candidate_id)

    def _mark_free_rim_escape_candidate_failed(self) -> None:
        candidate_id = self._free_rim_escape_candidate_id
        if candidate_id is None:
            return
        mark_failed = getattr(self.grasp_provider, "mark_candidate_failed", None)
        if callable(mark_failed):
            mark_failed(candidate_id)
        self._cavity_failed_candidate_id = candidate_id

    def _free_rim_high_corridor(
        self,
        scene: SceneEstimate,
        start: np.ndarray,
        source: SceneEntity,
    ) -> tuple[np.ndarray, np.ndarray, int] | None:
        """Rank bounded constant-height escape rays by the fresh RGB-D SDF."""

        bearing = start[:2, 3] - source.position[:2]
        bearing_norm = float(np.linalg.norm(bearing))
        if bearing_norm < 1e-6:
            # The public EE orientation supplies a deterministic fallback when
            # the camera reconstruction lies directly below the tool column.
            bearing = np.asarray(start[:2, 0], dtype=np.float64)
            bearing_norm = float(np.linalg.norm(bearing))
        if bearing_norm < 1e-6:
            return None
        bearing /= bearing_norm

        angles = np.deg2rad((0.0, 45.0, -45.0, 90.0, -90.0, 135.0, -135.0, 180.0))
        fractions = np.linspace(
            0.0, 1.0, self._FREE_RIM_ESCAPE_PATH_SAMPLES
        )
        lower = scene.workspace_min[:2] + 0.010
        upper = scene.workspace_max[:2] - 0.010
        ranked: list[tuple[tuple[float, float, float], np.ndarray, np.ndarray]] = []
        evaluated = 0
        for angle in angles:
            cosine = float(np.cos(angle))
            sine = float(np.sin(angle))
            direction = np.array(
                (
                    cosine * bearing[0] - sine * bearing[1],
                    sine * bearing[0] + cosine * bearing[1],
                ),
                dtype=np.float64,
            )
            points = np.repeat(start[None, :3, 3], len(fractions), axis=0)
            points[:, :2] += (
                fractions[:, None]
                * self._FREE_RIM_ESCAPE_DISTANCE_M
                * direction[None, :]
            )
            if np.any(points[:, :2] < lower) or np.any(points[:, :2] > upper):
                continue
            evaluated += 1
            distances = np.asarray(
                scene.obstacle_sdf.distance(points), dtype=np.float64
            )
            if distances.shape != (len(points),) or not np.all(
                np.isfinite(distances)
            ):
                continue
            if np.any(np.diff(distances) < -1e-12):
                continue
            if (
                distances[-1]
                < distances[0] + self._FREE_RIM_ESCAPE_MIN_PROGRESS_M
            ):
                continue
            goal = start.copy()
            goal[:2, 3] = points[-1, :2]
            # Highest final clearance wins; ties prefer the greatest minimum
            # clearance and the sensor-observed outward bearing.
            score = (
                float(distances[-1]),
                float(np.min(distances)),
                float(np.dot(direction, bearing)),
            )
            ranked.append((score, goal, distances))
        if not ranked:
            return None
        _, goal, distances = max(ranked, key=lambda item: item[0])
        return goal, distances, evaluated

    def _fresh_target_container_field(
        self,
        scene: SceneEstimate,
        start: np.ndarray,
        diagnostic: Mapping[str, object],
    ) -> tuple[SceneEntity, SceneEntity, BoxSDF, str] | None:
        """Bind one fresh SDF field uniquely to the current semantic target."""

        source_id = self._free_rim_target_egress_source_id
        target_id = self._free_rim_target_egress_target_id
        if source_id is None or target_id is None:
            return None
        try:
            source = scene.by_id(source_id)
            target = scene.by_id(target_id)
        except PerceptionError:
            return None
        if source.instance_id == target.instance_id:
            return None
        visible_value = getattr(self.observer, "visible_instance_ids", None)
        try:
            visible_ids = (
                {str(instance_id) for instance_id in visible_value}
                if visible_value is not None
                else None
            )
        except TypeError:
            return None
        if (
            visible_ids is None
            or source.instance_id not in visible_ids
            or target.instance_id not in visible_ids
            or self._normalise_semantic_label(source.label)
            != self._normalise_semantic_label(
                self._free_rim_target_egress_source_label
            )
            or self._normalise_semantic_label(target.label)
            != self._normalise_semantic_label(
                self._free_rim_target_egress_target_label
            )
        ):
            return None

        obstacle = scene.obstacle_sdf
        if not isinstance(obstacle, CompositeSDF):
            return None
        expected_id = str(diagnostic["field_id"])
        expected_label = self._normalise_semantic_label(
            diagnostic["field_label"]
        )
        expected_center = np.asarray(
            diagnostic["field_center_world_m"], dtype=np.float64
        )
        expected_half = np.asarray(
            diagnostic["field_half_extents_m"], dtype=np.float64
        )
        matching_fields = [
            field
            for field in obstacle.fields
            if isinstance(field, BoxSDF)
            and field.source_instance_id == expected_id
            and self._normalise_semantic_label(field.source_label)
            == expected_label
            and np.linalg.norm(field.center - expected_center) <= 0.020
            and np.max(np.abs(field.half_extents - expected_half)) <= 0.012
        ]
        if len(matching_fields) != 1:
            return None
        field = matching_fields[0]
        distances = np.asarray(
            [float(item.distance(start[:3, 3])) for item in obstacle.fields],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(distances)):
            return None
        minimum = float(np.min(distances))
        if int(
            np.count_nonzero(
                np.isclose(distances, minimum, rtol=0.0, atol=1e-9)
            )
        ) != 1:
            return None
        matching_indices = [
            index
            for index, item in enumerate(obstacle.fields)
            if item is field
            and np.isclose(distances[index], minimum, rtol=0.0, atol=1e-9)
        ]
        if len(matching_indices) != 1 or minimum >= 0.0:
            return None

        target_label = self._normalise_semantic_label(target.label)
        bound_target_label = self._normalise_semantic_label(
            self._free_rim_target_egress_target_label
        )
        field_label = self._normalise_semantic_label(field.source_label)
        if not (
            {"drawer", "cabinet"} & set(target_label.split())
            and {"drawer", "cabinet"} & set(bound_target_label.split())
            and {"drawer", "cabinet"} & set(field_label.split())
        ):
            return None
        exact_public_provenance = field.source_instance_id == target.instance_id
        container_entities = [
            entity
            for entity in scene.entities
            if {"drawer", "cabinet"}
            & set(self._normalise_semantic_label(entity.label).split())
        ]
        obb_matches = [
            entity
            for entity in container_entities
            if StableSceneEstimator._obb_surface_overlaps(entity, field)
            and StableSceneEstimator._obb_contains_either_center(entity, field)
        ]
        unique_obb_binding = bool(
            len(obb_matches) == 1
            and obb_matches[0].instance_id == target.instance_id
        )
        same_family_fields = [
            item
            for item in obstacle.fields
            if {"drawer", "cabinet"}
            & set(
                self._normalise_semantic_label(
                    getattr(item, "source_label", "")
                ).split()
            )
        ]
        target_as_field = BoxSDF(
            target.position,
            target.extent / 2.0,
            target.pose[:3, :3],
        )
        overlapping_visible_targets = [
            entity
            for entity in container_entities
            if entity.instance_id != target.instance_id
            and visible_ids is not None
            and entity.instance_id in visible_ids
            and StableSceneEstimator._obb_surface_overlaps(
                entity, target_as_field
            )
            and StableSceneEstimator._obb_contains_either_center(
                entity, target_as_field
            )
        ]
        # An opened drawer front can move away from the cabinet/body component
        # that encloses the high public EE start.  Admit that articulated
        # assembly topology only when the failed optimiser fingerprint selects
        # the very same field again in a fresh frame, the bound moving target
        # is freshly visible, and no second container-family field exists.
        # Geometry/ID drift and multi-field ambiguity therefore remain closed;
        # a shared semantic label alone can never establish this association.
        unique_stable_assembly_field = bool(
            not exact_public_provenance
            and not unique_obb_binding
            and visible_ids is not None
            and target.instance_id in visible_ids
            and len(same_family_fields) == 1
            and same_family_fields[0] is field
            and not overlapping_visible_targets
        )
        if exact_public_provenance:
            binding_kind = "exact_field_provenance"
        elif unique_obb_binding:
            binding_kind = "unique_same_frame_obb"
        elif unique_stable_assembly_field:
            binding_kind = "unique_stable_assembly_field"
        else:
            return None
        return source, target, field, binding_kind

    def _free_rim_target_egress_corridor(
        self,
        scene: SceneEstimate,
        start: np.ndarray,
        source: SceneEntity,
        nominal_envelope_m: float,
    ) -> tuple[np.ndarray, np.ndarray, int] | None:
        """Find the shortest strict-SDF-improving constant-height egress."""

        bearing = start[:2, 3] - source.position[:2]
        bearing_norm = float(np.linalg.norm(bearing))
        if bearing_norm < 1e-6:
            bearing = np.asarray(start[:2, 0], dtype=np.float64)
            bearing_norm = float(np.linalg.norm(bearing))
        if bearing_norm < 1e-6:
            return None
        bearing /= bearing_norm
        angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
        fractions = np.linspace(
            0.0, 1.0, self._FREE_RIM_TARGET_EGRESS_PATH_SAMPLES
        )
        lower = scene.workspace_min[:2] + 0.010
        upper = scene.workspace_max[:2] - 0.010
        ranked: list[
            tuple[tuple[float, float, float], np.ndarray, np.ndarray]
        ] = []
        evaluated = 0
        for angle in angles:
            cosine = float(np.cos(angle))
            sine = float(np.sin(angle))
            direction = np.array(
                (
                    cosine * bearing[0] - sine * bearing[1],
                    sine * bearing[0] + cosine * bearing[1],
                ),
                dtype=np.float64,
            )
            dense = np.repeat(start[None, :3, 3], len(fractions), axis=0)
            dense[:, :2] += (
                fractions[:, None]
                * self._FREE_RIM_TARGET_EGRESS_MAX_DISTANCE_M
                * direction[None, :]
            )
            if np.any(dense[:, :2] < lower) or np.any(dense[:, :2] > upper):
                continue
            evaluated += 1
            raw = np.asarray(scene.obstacle_sdf.distance(dense), dtype=np.float64)
            if raw.shape != (len(dense),) or not np.all(np.isfinite(raw)):
                continue
            envelope_indices = np.flatnonzero(raw >= nominal_envelope_m)
            if len(envelope_indices) == 0:
                continue
            stop = int(envelope_indices[0])
            if stop < 1 or np.any(np.diff(raw[: stop + 1]) <= 1e-8):
                continue
            goal = start.copy()
            goal[:2, 3] = dense[stop, :2]
            distance = float(np.linalg.norm(goal[:2, 3] - start[:2, 3]))
            score = (
                -distance,
                float(raw[stop]),
                float(np.dot(direction, bearing)),
            )
            ranked.append((score, goal, raw[: stop + 1].copy()))
        if not ranked:
            return None
        _, goal, raw = max(ranked, key=lambda item: item[0])
        return goal, raw, evaluated

    @staticmethod
    def _camera_surface_map(value: object) -> dict[str, np.ndarray] | None:
        try:
            rows = tuple(value)  # type: ignore[arg-type]
        except TypeError:
            return None
        result: dict[str, np.ndarray] = {}
        for row in rows:
            if not isinstance(row, tuple) or len(row) != 2:
                return None
            name, raw_points = row
            points = np.asarray(raw_points, dtype=np.float64)
            if (
                not isinstance(name, str)
                or not name.strip()
                or name in result
                or points.ndim != 2
                or points.shape[1:] != (3,)
                or not np.all(np.isfinite(points))
            ):
                return None
            result[name] = points
        return result

    @staticmethod
    def _surface_alignment_error_m(
        reference_world: np.ndarray,
        current_world: np.ndarray,
        transform_world: np.ndarray,
    ) -> float:
        """Robust symmetric nearest-surface error for two raw RGB-D crops."""

        first = np.asarray(reference_world, dtype=np.float64)
        second = np.asarray(current_world, dtype=np.float64)
        transform = np.asarray(transform_world, dtype=np.float64)
        if (
            first.ndim != 2
            or first.shape[1:] != (3,)
            or second.ndim != 2
            or second.shape[1:] != (3,)
            or transform.shape != (4, 4)
            or not np.all(np.isfinite(first))
            or not np.all(np.isfinite(second))
            or not np.all(np.isfinite(transform))
        ):
            return float("inf")
        if len(first) > 2048:
            first = first[
                np.linspace(0, len(first) - 1, 2048, dtype=np.int64)
            ]
        if len(second) > 2048:
            second = second[
                np.linspace(0, len(second) - 1, 2048, dtype=np.int64)
            ]
        moved = first @ transform[:3, :3].T + transform[:3, 3]
        forward, _ = cKDTree(second).query(moved, k=1)
        backward, _ = cKDTree(moved).query(second, k=1)
        return float(
            max(np.quantile(forward, 0.75), np.quantile(backward, 0.75))
        )

    @staticmethod
    def _strict_surface_alignment_error_m(
        reference_world: np.ndarray,
        current_world: np.ndarray,
        transform_world: np.ndarray,
    ) -> float:
        """Symmetric 95th-percentile error for typed-negative tool proofs."""

        first = np.asarray(reference_world, dtype=np.float64)
        second = np.asarray(current_world, dtype=np.float64)
        transform = np.asarray(transform_world, dtype=np.float64)
        if (
            first.ndim != 2
            or first.shape[1:] != (3,)
            or second.ndim != 2
            or second.shape[1:] != (3,)
            or transform.shape != (4, 4)
            or not np.all(np.isfinite(first))
            or not np.all(np.isfinite(second))
            or not np.all(np.isfinite(transform))
        ):
            return float("inf")
        if len(first) > 2048:
            first = first[
                np.linspace(0, len(first) - 1, 2048, dtype=np.int64)
            ]
        if len(second) > 2048:
            second = second[
                np.linspace(0, len(second) - 1, 2048, dtype=np.int64)
            ]
        moved = first @ transform[:3, :3].T + transform[:3, 3]
        forward, _ = cKDTree(second).query(moved, k=1)
        backward, _ = cKDTree(moved).query(second, k=1)
        return float(
            max(np.quantile(forward, 0.95), np.quantile(backward, 0.95))
        )

    @classmethod
    def _released_panda_tool_surface_evidence(
        cls,
        points_world: np.ndarray,
        ee_pose_world: np.ndarray,
    ) -> dict[str, object]:
        """Bind one raw RGB-D component to the fixed public Panda hand volume.

        Nearest-neighbour alignment alone is insufficient: two disjoint crops
        of a homogeneous static plane can be exact translations.  This gate
        additionally requires most raw points to lie inside the public
        released-hand envelope in EE coordinates and requires robust,
        bilateral, genuinely three-dimensional occupancy.  It consumes no
        simulator geometry or contact state.
        """

        points = np.asarray(points_world, dtype=np.float64)
        pose = np.asarray(ee_pose_world, dtype=np.float64)
        evidence: dict[str, object] = {
            "accepted": False,
            "raw_point_count": int(len(points)) if points.ndim >= 1 else 0,
        }
        if (
            points.ndim != 2
            or points.shape[1:] != (3,)
            or len(points) < cls._FREE_RIM_VIEW_FIELD_MIN_POINTS_PER_CAMERA
            or pose.shape != (4, 4)
            or not np.all(np.isfinite(points))
            or not np.all(np.isfinite(pose))
            or not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6)
            or not np.allclose(
                pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=2e-3
            )
            or float(np.linalg.det(pose[:3, :3])) <= 0.0
        ):
            evidence["reason"] = "malformed_points_or_public_ee_pose"
            return evidence

        local = (points - pose[:3, 3]) @ pose[:3, :3]
        palm_lower = np.asarray(
            cls._FREE_RIM_VIEW_TOOL_PALM_LOCAL_MIN_M, dtype=np.float64
        )
        palm_upper = np.asarray(
            cls._FREE_RIM_VIEW_TOOL_PALM_LOCAL_MAX_M, dtype=np.float64
        )
        finger_lower = np.asarray(
            cls._FREE_RIM_VIEW_TOOL_FINGER_SWEEP_LOCAL_MIN_M,
            dtype=np.float64,
        )
        finger_upper = np.asarray(
            cls._FREE_RIM_VIEW_TOOL_FINGER_SWEEP_LOCAL_MAX_M,
            dtype=np.float64,
        )
        numeric_margin = 1e-9
        inside_palm = np.all(
            (local >= palm_lower - numeric_margin)
            & (local <= palm_upper + numeric_margin),
            axis=1,
        )
        inside_finger_sweep = np.all(
            (local >= finger_lower - numeric_margin)
            & (local <= finger_upper + numeric_margin),
            axis=1,
        )
        inside = inside_palm | inside_finger_sweep
        inliers = local[inside]
        fraction = float(len(inliers) / len(local))
        evidence.update(
            {
                "tool_envelope_point_count": int(len(inliers)),
                "tool_envelope_fraction": fraction,
            }
        )
        if (
            len(inliers) < cls._FREE_RIM_VIEW_FIELD_MIN_POINTS_PER_CAMERA
            or fraction < cls._FREE_RIM_VIEW_TOOL_MIN_ENVELOPE_FRACTION
        ):
            evidence["reason"] = "insufficient_public_tool_envelope_occupancy"
            return evidence

        low = np.quantile(inliers, 0.05, axis=0)
        high = np.quantile(inliers, 0.95, axis=0)
        spans = high - low
        centroid = np.median(inliers, axis=0)
        robust_center = 0.5 * (low + high)
        covariance = np.cov(inliers, rowvar=False, bias=True)
        eigenvalues = np.maximum(
            np.linalg.eigvalsh(covariance), 0.0
        )
        thickness = float(np.sqrt(eigenvalues[0]))
        eigenvalue_ratio = float(
            eigenvalues[0] / max(eigenvalues[-1], 1e-12)
        )
        gates = {
            "robust_span": bool(
                np.all(
                    spans
                    >= np.asarray(
                        cls._FREE_RIM_VIEW_TOOL_MIN_ROBUST_SPAN_M,
                        dtype=np.float64,
                    )
                )
            ),
            "nonplanar_thickness": bool(
                thickness >= cls._FREE_RIM_VIEW_TOOL_MIN_THICKNESS_M
                and eigenvalue_ratio
                >= cls._FREE_RIM_VIEW_TOOL_MIN_EIGENVALUE_RATIO
            ),
            "bilateral_x": bool(
                low[0] <= -cls._FREE_RIM_VIEW_TOOL_MIN_NEGATIVE_X_M
                and high[0] >= cls._FREE_RIM_VIEW_TOOL_MIN_POSITIVE_X_M
            ),
            "bilateral_y": bool(
                low[1] <= -cls._FREE_RIM_VIEW_TOOL_MIN_NEGATIVE_Y_M
                and high[1] >= cls._FREE_RIM_VIEW_TOOL_MIN_POSITIVE_Y_M
            ),
            "centred_on_tool": bool(
                abs(float(robust_center[0]))
                <= cls._FREE_RIM_VIEW_TOOL_MAX_CENTROID_ABS_X_M
                and abs(float(robust_center[1]))
                <= cls._FREE_RIM_VIEW_TOOL_MAX_CENTROID_ABS_Y_M
                and cls._FREE_RIM_VIEW_TOOL_CENTROID_Z_RANGE_M[0]
                <= float(robust_center[2])
                <= cls._FREE_RIM_VIEW_TOOL_CENTROID_Z_RANGE_M[1]
            ),
            "palm_and_finger_depth": bool(
                low[2] <= -cls._FREE_RIM_VIEW_TOOL_MIN_REAR_Z_M
                and high[2] >= cls._FREE_RIM_VIEW_TOOL_MIN_FORWARD_Z_M
            ),
        }
        accepted = all(gates.values())
        evidence.update(
            {
                "accepted": accepted,
                "reason": (
                    "public_panda_tool_volume_and_3d_occupancy"
                    if accepted
                    else "public_panda_tool_morphology_gate_failed"
                ),
                "robust_low_local_m": low.tolist(),
                "robust_high_local_m": high.tolist(),
                "robust_span_local_m": spans.tolist(),
                "median_local_m": centroid.tolist(),
                "robust_center_local_m": robust_center.tolist(),
                "smallest_axis_rms_m": thickness,
                "smallest_to_largest_covariance_ratio": eigenvalue_ratio,
                "gates": gates,
            }
        )
        return evidence

    @classmethod
    def _typed_negative_panda_tool_surface_evidence(
        cls,
        points_world: np.ndarray,
        ee_pose_world: np.ndarray,
    ) -> dict[str, object]:
        """Apply the ordinary morphology proof plus a 95% contamination cap."""

        evidence = dict(
            cls._released_panda_tool_surface_evidence(
                points_world, ee_pose_world
            )
        )
        strict_fraction = bool(
            evidence.get("accepted") is True
            and cls._builtin_finite_number(
                evidence.get("tool_envelope_fraction")
            )
            is not None
            and float(evidence["tool_envelope_fraction"]) >= 0.95
        )
        evidence["typed_negative_minimum_tool_envelope_fraction"] = 0.95
        evidence["accepted"] = strict_fraction
        if not strict_fraction:
            evidence["reason"] = (
                "typed_negative_tool_surface_contamination_exceeds_5_percent"
            )
        else:
            evidence["reason"] = (
                "typed_negative_public_panda_tool_volume_and_3d_occupancy"
            )
        return evidence

    def _current_sensor_scene(self) -> SceneEstimate | None:
        value = getattr(self.observer, "current_sensor_scene", None)
        if callable(value):
            try:
                value = value()
            except (TypeError, ValueError, RuntimeError):
                return None
        return value if isinstance(value, SceneEstimate) else None

    def _unique_three_frame_field(
        self,
        scene: SceneEstimate,
        proof: Mapping[str, object],
        *,
        expected_center_world_m: object | None = None,
    ) -> BoxSDF | None:
        """Rebind one fresh field without weakening the estimator's identity gates."""

        obstacle = scene.obstacle_sdf
        if not isinstance(obstacle, CompositeSDF):
            return None
        try:
            field_id = proof["field_id"]
            field_label = self._normalise_semantic_label(proof["field_label"])
            source_id = str(proof["source_id"])
            target_id = str(proof["target_id"])
            anchor_half = np.asarray(
                proof["field_half_extents_m"], dtype=np.float64
            )
            source = scene.by_id(source_id)
            target = scene.by_id(target_id)
        except (KeyError, TypeError, ValueError, PerceptionError):
            return None
        if (
            not isinstance(field_id, str)
            or not field_id.startswith("fused-")
            or not field_label
            or anchor_half.shape != (3,)
            or not np.all(np.isfinite(anchor_half))
            or np.any(anchor_half <= 0.0)
            or source.instance_id == target.instance_id
            or self._normalise_semantic_label(source.label)
            != self._normalise_semantic_label(proof.get("source_label"))
            or self._normalise_semantic_label(target.label)
            != self._normalise_semantic_label(proof.get("target_label"))
        ):
            return None
        visible_value = getattr(self.observer, "visible_instance_ids", None)
        try:
            visible_ids = {str(value) for value in visible_value}
        except TypeError:
            return None
        if source_id not in visible_ids or target_id not in visible_ids:
            return None
        same_family = [
            field
            for field in obstacle.fields
            if isinstance(field, BoxSDF)
            and {"drawer", "cabinet"}
            & set(self._normalise_semantic_label(field.source_label or "").split())
        ]
        matches = [
            field
            for field in same_family
            if field.source_instance_id == field_id
            and self._normalise_semantic_label(field.source_label) == field_label
            and float(np.max(np.abs(field.half_extents - anchor_half)))
            <= self._FREE_RIM_VIEW_FIELD_MAX_HALF_EXTENT_DRIFT_M
        ]
        if len(same_family) != 1 or len(matches) != 1:
            return None
        field = matches[0]
        if expected_center_world_m is not None:
            expected_center = np.asarray(
                expected_center_world_m, dtype=np.float64
            )
            if (
                expected_center.shape != (3,)
                or not np.all(np.isfinite(expected_center))
                or float(np.linalg.norm(field.center - expected_center)) > 0.020
            ):
                return None
        visible_containers = [
            entity
            for entity in scene.entities
            if entity.instance_id in visible_ids
            and entity.instance_id != field.source_instance_id
            and {"drawer", "cabinet"}
            & set(self._normalise_semantic_label(entity.label).split())
        ]
        if any(
            StableSceneEstimator._obb_surface_overlaps(entity, field)
            and StableSceneEstimator._obb_contains_either_center(entity, field)
            for entity in visible_containers
        ):
            return None
        return field

    @classmethod
    def _canonical_surface_map(cls, value: object) -> dict[str, np.ndarray] | None:
        raw = cls._camera_surface_map(value)
        if raw is None:
            return None
        result: dict[str, np.ndarray] = {}
        for name, points in raw.items():
            canonical = cls._canonical_view_name(name)
            if not canonical or canonical in result:
                return None
            result[canonical] = points
        return result

    @staticmethod
    def _strict_provider_float_array(
        value: object, shape: tuple[int, ...]
    ) -> np.ndarray | None:
        """Accept only native real floating arrays at a proof-provider boundary.

        ``np.asarray(..., dtype=float)`` is intentionally not used as the type
        gate: it silently discards complex components and converts strings or
        booleans into plausible geometry.  Conversion to float64 happens only
        after the native ndarray/dtype/shape/finite contract has passed.
        """

        if (
            type(value) is not np.ndarray
            or value.dtype.kind != "f"
            or value.shape != shape
            or not np.all(np.isfinite(value))
        ):
            return None
        result = value.astype(np.float64, copy=True)
        return result if np.all(np.isfinite(result)) else None

    @classmethod
    def _strict_provider_surface_map(
        cls,
        value: object,
        *,
        canonical_policy_names: bool = True,
    ) -> dict[str, np.ndarray] | None:
        """Parse exact policy-camera surfaces without numeric type coercion."""

        if type(value) is not tuple:
            return None
        result: dict[str, np.ndarray] = {}
        for row in value:
            if type(row) is not tuple or len(row) != 2:
                return None
            name, raw_points = row
            canonical = (
                cls._canonical_view_name(name)
                if canonical_policy_names
                else name
                if type(name) is str and name.strip()
                else ""
            )
            if (
                not canonical
                or canonical in result
                or type(raw_points) is not np.ndarray
                or raw_points.dtype.kind != "f"
                or raw_points.ndim != 2
                or raw_points.shape[1:] != (3,)
                or not np.all(np.isfinite(raw_points))
            ):
                return None
            points = raw_points.astype(np.float64, copy=True)
            if not np.all(np.isfinite(points)):
                return None
            result[canonical] = points
        return result

    def _freeze_field_provider_authority(
        self,
        *,
        scene: SceneEstimate,
        field: BoxSDF,
        bound: BoundConstraintGraph,
        proof: Mapping[str, object],
    ) -> dict[str, object] | None:
        """Freeze the complete public-scene authority before a provider call."""

        try:
            graph = bound.graph
            graph_source = graph.source
            graph_target = graph.target
            proof_source_id = proof["source_id"]
            proof_source_label = proof["source_label"]
            proof_target_id = proof["target_id"]
            proof_target_label = proof["target_label"]
            proof_field_id = proof["field_id"]
            proof_field_label = proof["field_label"]
            bound_source_id = bound.source_id
            bound_source_label = bound.source_label
            bound_target_id = bound.target_id
            bound_target_label = bound.target_label
            graph_source_label = graph_source.label
            graph_target_label = graph_target.label
            source = scene.by_id(proof_source_id)
            target = scene.by_id(proof_target_id)
        except BaseException:
            return None
        native_strings = (
            proof_source_id,
            proof_source_label,
            proof_target_id,
            proof_target_label,
            proof_field_id,
            proof_field_label,
            bound_source_id,
            bound_source_label,
            bound_target_id,
            bound_target_label,
            graph_source_label,
            graph_target_label,
            source.instance_id,
            source.label,
            target.instance_id,
            target.label,
            field.source_instance_id,
            field.source_label,
        )
        if (
            any(type(value) is not str or not value.strip() for value in native_strings)
            or getattr(graph_source, "role", None) != "source"
            or getattr(graph_target, "role", None) != "target"
            or proof_source_id != bound_source_id
            or proof_source_id != source.instance_id
            or proof_source_label != bound_source_label
            or proof_source_label != graph_source_label
            or proof_source_label != source.label
            or proof_target_id != bound_target_id
            or proof_target_id != target.instance_id
            or proof_target_label != bound_target_label
            or proof_target_label != graph_target_label
            or proof_target_label != target.label
            or proof_field_id != field.source_instance_id
            or proof_field_label != field.source_label
        ):
            return None
        center = self._strict_provider_float_array(field.center, (3,))
        half = self._strict_provider_float_array(field.half_extents, (3,))
        axes = self._strict_provider_float_array(field.axes, (3, 3))
        surface_rows = tuple(
            (name, points.copy()) for name, points in field.surface_points_by_camera
        )
        surfaces = self._strict_provider_surface_map(
            surface_rows, canonical_policy_names=False
        )
        if center is None or half is None or axes is None or surfaces is None:
            return None
        return {
            "field_id": proof_field_id,
            "field_label": proof_field_label,
            "field_center_world_m": center,
            "field_half_extents_m": half,
            "field_axes_world": axes,
            "surface_points_by_camera": tuple(
                (name, points.copy()) for name, points in surface_rows
            ),
            "source_id": proof_source_id,
            "source_label": proof_source_label,
            "target_id": proof_target_id,
            "target_label": proof_target_label,
        }

    def _strict_provider_mapping_for_authority(
        self,
        value: object,
        authority: Mapping[str, object],
    ) -> dict[str, object] | None:
        """Accept only an exact typed echo of the pre-call frozen authority."""

        if not isinstance(value, Mapping):
            return None
        identity_keys = (
            "field_id",
            "field_label",
            "source_id",
            "source_label",
            "target_id",
            "target_label",
        )
        if any(
            type(value.get(key)) is not str
            or value.get(key) != authority.get(key)
            for key in identity_keys
        ):
            return None
        try:
            center = self._strict_provider_float_array(
                value["field_center_world_m"], (3,)
            )
            half = self._strict_provider_float_array(
                value["field_half_extents_m"], (3,)
            )
            axes = self._strict_provider_float_array(
                value["field_axes_world"], (3, 3)
            )
            surfaces = self._strict_provider_surface_map(
                value["surface_points_by_camera"],
                canonical_policy_names=False,
            )
            authority_surfaces = self._strict_provider_surface_map(
                authority["surface_points_by_camera"],
                canonical_policy_names=False,
            )
        except (KeyError, TypeError, ValueError):
            return None
        authority_center = authority["field_center_world_m"]
        authority_half = authority["field_half_extents_m"]
        authority_axes = authority["field_axes_world"]
        if (
            center is None
            or half is None
            or axes is None
            or surfaces is None
            or authority_surfaces is None
            or not np.array_equal(center, authority_center)
            or not np.array_equal(half, authority_half)
            or not np.array_equal(axes, authority_axes)
            or set(surfaces) != set(authority_surfaces)
            or any(
                surfaces[name].shape != authority_surfaces[name].shape
                or not np.array_equal(surfaces[name], authority_surfaces[name])
                for name in surfaces
            )
        ):
            return None
        return {
            **{key: authority[key] for key in identity_keys},
            "field_center_world_m": np.asarray(authority_center).copy(),
            "field_half_extents_m": np.asarray(authority_half).copy(),
            "field_axes_world": np.asarray(authority_axes).copy(),
            "surface_points_by_camera": tuple(
                (name, points.copy())
                for name, points in authority["surface_points_by_camera"]
            ),
        }

    @classmethod
    def _three_frame_motion_pair_evidence(
        cls,
        first: Mapping[str, object],
        second: Mapping[str, object],
    ) -> dict[str, object]:
        """Apply the unchanged motion, morphology, and point-hypothesis gates."""

        try:
            first_center = np.asarray(first["field_center_world_m"], dtype=np.float64)
            second_center = np.asarray(second["field_center_world_m"], dtype=np.float64)
            first_half = np.asarray(first["field_half_extents_m"], dtype=np.float64)
            second_half = np.asarray(second["field_half_extents_m"], dtype=np.float64)
            first_axes = np.asarray(first["field_axes_world"], dtype=np.float64)
            second_axes = np.asarray(second["field_axes_world"], dtype=np.float64)
            first_pose = np.asarray(first["ee_pose_world"], dtype=np.float64)
            second_pose = np.asarray(second["ee_pose_world"], dtype=np.float64)
            first_width = cls._builtin_finite_number(first["gripper_width_m"])
            second_width = cls._builtin_finite_number(second["gripper_width_m"])
            first_raw = cls._builtin_finite_number(first["raw_sdf_m"])
            second_raw = cls._builtin_finite_number(second["raw_sdf_m"])
            first_stamp = cls._builtin_finite_number(first["timestamp_s"])
            second_stamp = cls._builtin_finite_number(second["timestamp_s"])
            capture_advanced = sensor_capture_advanced(first, second)
            first_surfaces = cls._canonical_surface_map(
                first["surface_points_by_camera"]
            )
            second_surfaces = cls._canonical_surface_map(
                second["surface_points_by_camera"]
            )
        except (KeyError, TypeError, ValueError):
            return {"accepted": False, "reason": "malformed_three_frame_endpoint"}
        arrays = (
            first_center,
            second_center,
            first_half,
            second_half,
            first_axes,
            second_axes,
            first_pose,
            second_pose,
        )
        if (
            any(value.shape != (3,) for value in arrays[:4])
            or any(value.shape != (3, 3) for value in arrays[4:6])
            or any(value.shape != (4, 4) for value in arrays[6:])
            or not all(np.all(np.isfinite(value)) for value in arrays)
            or first_width is None
            or second_width is None
            or first_raw is None
            or second_raw is None
            or first_stamp is None
            or second_stamp is None
            or not capture_advanced
            or first_surfaces is None
            or second_surfaces is None
            or "agentview" not in first_surfaces
            or "agentview" not in second_surfaces
            or len(first_surfaces["agentview"])
            < cls._FREE_RIM_VIEW_FIELD_MIN_POINTS_PER_CAMERA
            or len(second_surfaces["agentview"])
            < cls._FREE_RIM_VIEW_FIELD_MIN_POINTS_PER_CAMERA
            or any(
                not np.allclose(axes.T @ axes, np.eye(3), atol=2e-3)
                or float(np.linalg.det(axes)) <= 0.0
                for axes in (first_axes, second_axes)
            )
            or any(
                not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6)
                or not np.allclose(
                    pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=2e-3
                )
                or float(np.linalg.det(pose[:3, :3])) <= 0.0
                for pose in (first_pose, second_pose)
            )
        ):
            return {"accepted": False, "reason": "invalid_three_frame_endpoint"}
        ee_motion = second_pose[:3, 3] - first_pose[:3, 3]
        field_motion = second_center - first_center
        ee_distance = float(np.linalg.norm(ee_motion))
        field_distance = float(np.linalg.norm(field_motion))
        if ee_distance <= 0.0 or field_distance <= 0.0:
            return {"accepted": False, "reason": "nonpositive_ee_or_field_motion"}
        residual = float(np.linalg.norm(field_motion - ee_motion))
        relative_drift = float(
            np.linalg.norm(
                (second_center - second_pose[:3, 3])
                - (first_center - first_pose[:3, 3])
            )
        )
        direction_cosine = float(
            np.dot(field_motion, ee_motion) / (field_distance * ee_distance)
        )
        motion_ratio = field_distance / ee_distance
        half_drift = float(np.max(np.abs(second_half - first_half)))
        axes_drift = float(
            Rotation.from_matrix(second_axes @ first_axes.T).magnitude()
        )
        ee_rotation = cls._rotation_distance_rad(second_pose, first_pose)
        first_tool = cls._typed_negative_panda_tool_surface_evidence(
            first_surfaces["agentview"], first_pose
        )
        second_tool = cls._typed_negative_panda_tool_surface_evidence(
            second_surfaces["agentview"], second_pose
        )
        try:
            tool_transform = second_pose @ np.linalg.inv(first_pose)
        except np.linalg.LinAlgError:
            return {"accepted": False, "reason": "singular_three_frame_pose"}
        tool_error = cls._strict_surface_alignment_error_m(
            first_surfaces["agentview"],
            second_surfaces["agentview"],
            tool_transform,
        )
        static_error = cls._strict_surface_alignment_error_m(
            first_surfaces["agentview"],
            second_surfaces["agentview"],
            np.eye(4, dtype=np.float64),
        )
        accepted = bool(
            first_raw < 0.0
            and second_raw < 0.0
            and first_width >= cls._FREE_RIM_VIEW_FIELD_MIN_OPEN_WIDTH_M
            and second_width >= cls._FREE_RIM_VIEW_FIELD_MIN_OPEN_WIDTH_M
            and ee_distance >= cls._FREE_RIM_VIEW_FIELD_MIN_EE_MOTION_M
            and field_distance >= cls._FREE_RIM_VIEW_FIELD_MIN_FIELD_MOTION_M
            and residual <= cls._FREE_RIM_VIEW_FIELD_MAX_COMOTION_RESIDUAL_M
            and relative_drift
            <= cls._FREE_RIM_VIEW_FIELD_MAX_COMOTION_RESIDUAL_M
            and direction_cosine >= cls._FREE_RIM_VIEW_FIELD_MIN_DIRECTION_COSINE
            and cls._FREE_RIM_VIEW_FIELD_MIN_MOTION_RATIO
            <= motion_ratio
            <= cls._FREE_RIM_VIEW_FIELD_MAX_MOTION_RATIO
            and half_drift <= cls._FREE_RIM_VIEW_FIELD_MAX_HALF_EXTENT_DRIFT_M
            and axes_drift
            <= cls._FREE_RIM_TARGET_EGRESS_SAFE_POSE_ROTATION_TOLERANCE_RAD
            and ee_rotation
            <= cls._FREE_RIM_TARGET_EGRESS_SAFE_POSE_ROTATION_TOLERANCE_RAD
            and bool(first_tool["accepted"])
            and bool(second_tool["accepted"])
            and np.isfinite(tool_error)
            and np.isfinite(static_error)
            and tool_error <= cls._FREE_RIM_VIEW_FIELD_MAX_POINT_ERROR_M
            and tool_error + cls._FREE_RIM_VIEW_FIELD_MIN_STATIC_ADVANTAGE_M
            <= static_error
            and tool_error
            <= cls._FREE_RIM_VIEW_FIELD_MAX_STATIC_ERROR_RATIO * static_error
        )
        return {
            "accepted": accepted,
            "reason": (
                "released_hand_agentview_tool_comotion_and_static_rejection"
                if accepted
                else "three_frame_motion_or_point_hypothesis_gate_failed"
            ),
            "ee_motion_norm_m": ee_distance,
            "field_motion_norm_m": field_distance,
            "comotion_residual_m": residual,
            "relative_offset_drift_m": relative_drift,
            "direction_cosine": direction_cosine,
            "motion_ratio": motion_ratio,
            "half_extent_drift_m": half_drift,
            "field_axes_drift_rad": axes_drift,
            "ee_rotation_drift_rad": ee_rotation,
            "agentview_tool_motion_error_m": tool_error,
            "agentview_world_static_error_m": static_error,
            "first_public_tool_evidence": first_tool,
            "second_public_tool_evidence": second_tool,
            "first_capture_id": first.get("capture_id", ""),
            "second_capture_id": second.get("capture_id", ""),
            "sensor_capture_advanced": capture_advanced,
        }

    @classmethod
    def _three_frame_wrist_endpoint_evidence(
        cls,
        endpoint: Mapping[str, object],
        snapshot: Mapping[str, object],
    ) -> dict[str, object]:
        try:
            pose = np.asarray(endpoint["ee_pose_world"], dtype=np.float64)
            surfaces = cls._canonical_surface_map(
                endpoint["surface_points_by_camera"]
            )
            snapshot_pose = np.asarray(snapshot["ee_pose_world"], dtype=np.float64)
            cameras = snapshot["cameras"]
            if not isinstance(cameras, Mapping):
                raise TypeError
            wrist = cameras["wrist"]
            if not isinstance(wrist, Mapping):
                raise TypeError
        except (KeyError, TypeError, ValueError):
            return {"accepted": False, "reason": "malformed_wrist_endpoint"}
        if surfaces is None or "agentview" not in surfaces:
            return {"accepted": False, "reason": "agentview_surface_unavailable"}
        try:
            position_error = float(
                np.linalg.norm(snapshot_pose[:3, 3] - pose[:3, 3])
            )
            rotation_error = cls._rotation_distance_rad(snapshot_pose, pose)
        except ValueError:
            return {"accepted": False, "reason": "malformed_synchronized_pose"}
        if (
            position_error
            > cls._FREE_RIM_VIEW_CAMERA_RIGID_POSITION_TOLERANCE_M
            or rotation_error
            > cls._FREE_RIM_VIEW_CAMERA_RIGID_ROTATION_TOLERANCE_RAD
        ):
            return {
                "accepted": False,
                "reason": "camera_snapshot_not_synchronized_with_public_ee",
                "position_error_m": position_error,
                "rotation_error_rad": rotation_error,
            }
        frustum = cls._wrist_frustum_evidence(
            endpoint["field_center_world_m"],
            endpoint["field_half_extents_m"],
            endpoint["field_axes_world"],
            wrist,
            surfaces["agentview"],
        )
        if not bool(frustum.get("accepted", False)):
            return {
                "accepted": False,
                "reason": "wrist_frustum_classification_failed",
                "frustum": frustum,
            }
        wrist_points = surfaces.get("wrist")
        wrist_count = 0 if wrist_points is None else len(wrist_points)
        if bool(frustum["strictly_outside"]):
            accepted = wrist_count == 0
            return {
                "accepted": accepted,
                "reason": (
                    "typed_wrist_negative_visibility"
                    if accepted
                    else "wrist_surface_contradicts_negative_visibility"
                ),
                "visibility_mode": "strict_negative",
                "wrist_point_count": wrist_count,
                "frustum": frustum,
                "position_error_m": position_error,
                "rotation_error_rad": rotation_error,
            }
        wrist_tool = (
            cls._released_panda_tool_surface_evidence(wrist_points, pose)
            if wrist_points is not None
            else {"accepted": False, "reason": "wrist_surface_unavailable"}
        )
        accepted = bool(
            wrist_count >= cls._FREE_RIM_VIEW_FIELD_MIN_POINTS_PER_CAMERA
            and wrist_tool["accepted"]
        )
        return {
            "accepted": accepted,
            "reason": (
                "positive_wrist_surface_required_and_valid"
                if accepted
                else "theoretically_visible_wrist_surface_insufficient"
            ),
            "visibility_mode": "positive",
            "wrist_point_count": wrist_count,
            "frustum": frustum,
            "wrist_public_tool_evidence": wrist_tool,
            "position_error_m": position_error,
            "rotation_error_rad": rotation_error,
        }

    def _plan_three_frame_noncollinear_path(
        self,
        scene: SceneEstimate,
        field: BoxSDF,
        start: np.ndarray,
        first_motion_world: np.ndarray,
        nominal_envelope_m: float,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, list[float]]] | None:
        """Plan one high constant-Z segment against every frozen sensor field."""

        motion = np.asarray(first_motion_world, dtype=np.float64)
        if (
            start.shape != (4, 4)
            or motion.shape != (3,)
            or not np.all(np.isfinite(start))
            or not np.all(np.isfinite(motion))
            or not np.isfinite(nominal_envelope_m)
            or nominal_envelope_m <= 0.0
        ):
            return None
        first_xy = motion[:2]
        first_norm = float(np.linalg.norm(first_xy))
        if first_norm < self._FREE_RIM_VIEW_FIELD_MIN_EE_MOTION_M:
            return None
        first_xy /= first_norm
        lower = (
            scene.workspace_min[:2]
            + self._FREE_RIM_VIEW_REACQUIRE_WORKSPACE_MARGIN_M
        )
        upper = (
            scene.workspace_max[:2]
            - self._FREE_RIM_VIEW_REACQUIRE_WORKSPACE_MARGIN_M
        )
        fractions = np.linspace(
            0.0, 1.0, self._FREE_RIM_VIEW_REACQUIRE_PATH_SAMPLES
        )
        ranked: list[
            tuple[
                tuple[float, float, float],
                np.ndarray,
                np.ndarray,
                dict[str, list[float]],
            ]
        ] = []
        for angle in np.deg2rad((90.0, -90.0, 60.0, -60.0, 120.0, -120.0)):
            cosine = float(np.cos(angle))
            sine = float(np.sin(angle))
            direction = np.array(
                (
                    cosine * first_xy[0] - sine * first_xy[1],
                    sine * first_xy[0] + cosine * first_xy[1],
                ),
                dtype=np.float64,
            )
            noncollinear_cosine = abs(float(np.dot(direction, first_xy)))
            if (
                noncollinear_cosine
                > self._FREE_RIM_VIEW_REACQUIRE_MAX_DIRECTION_COSINE + 1e-12
            ):
                continue
            points = np.repeat(start[None, :3, 3], len(fractions), axis=0)
            points[:, :2] += (
                fractions[:, None]
                * self._FREE_RIM_VIEW_REACQUIRE_DISTANCE_M
                * direction[None, :]
            )
            if np.any(points[:, :2] < lower) or np.any(points[:, :2] > upper):
                continue
            candidate_raw = np.asarray(field.distance(points), dtype=np.float64)
            if (
                candidate_raw.shape != (len(points),)
                or not np.all(np.isfinite(candidate_raw))
                or np.any(np.diff(candidate_raw) <= 1e-8)
                or candidate_raw[-1] < nominal_envelope_m
            ):
                continue
            retained_distances: dict[str, list[float]] = {}
            retained_minimum = float("inf")
            retained_ok = True
            for index, retained in enumerate(scene.obstacle_sdf.fields):
                if retained is field:
                    continue
                values = np.asarray(retained.distance(points), dtype=np.float64)
                key = str(
                    getattr(retained, "source_instance_id", None)
                    or f"sensor-field-{index}"
                )
                retained_distances[key] = values.tolist()
                if (
                    values.shape != (len(points),)
                    or not np.all(np.isfinite(values))
                    or float(np.min(values)) < nominal_envelope_m
                ):
                    retained_ok = False
                    break
                retained_minimum = min(retained_minimum, float(np.min(values)))
            if not retained_ok:
                continue
            goal = start.copy()
            goal[:2, 3] = points[-1, :2]
            # Keep commanded endpoints comfortably below the 20-mm hard
            # bound, leaving room for the strict per-action measured gate.
            segments = max(
                1,
                int(
                    np.ceil(
                        self._FREE_RIM_VIEW_REACQUIRE_DISTANCE_M
                        / (
                            0.8
                            * self._FREE_RIM_TARGET_EGRESS_WAYPOINT_SPACING_M
                        )
                    )
                ),
            )
            waypoint_fractions = np.linspace(0.0, 1.0, segments + 1)
            waypoints = np.repeat(start[None, :, :], len(waypoint_fractions), axis=0)
            waypoints[:, :2, 3] += (
                waypoint_fractions[:, None]
                * self._FREE_RIM_VIEW_REACQUIRE_DISTANCE_M
                * direction[None, :]
            )
            score = (
                retained_minimum,
                float(candidate_raw[-1]),
                -noncollinear_cosine,
            )
            retained_distances["proved-candidate"] = candidate_raw.tolist()
            ranked.append((score, goal, waypoints, retained_distances))
        if not ranked:
            return None
        _, goal, waypoints, distances = max(ranked, key=lambda item: item[0])
        return goal, waypoints, distances

    def _sensor_safe_path_sample_evidence(
        self,
        *,
        scene: SceneEstimate,
        candidate_field: BoxSDF,
        start_pose_world: np.ndarray,
        goal_pose_world: np.ndarray,
        samples: object,
        steps_before: int,
        steps_after: int,
        nominal_envelope_m: float,
    ) -> dict[str, object]:
        """Validate every public-proprio action in a frozen sensor-SDF path.

        The ordinary waypoint executor intentionally tracks only a phase
        endpoint.  A path that will become temporal deletion authority needs a
        stronger transaction: every issued action must remain high/open and in
        the exact frozen candidate/retained-field corridor.  The returned pose
        is internal typed evidence used to bind the subsequent endpoint query;
        it is never exposed as simulator state.
        """

        evidence: dict[str, object] = {"accepted": False}
        start = np.asarray(start_pose_world, dtype=np.float64)
        goal = np.asarray(goal_pose_world, dtype=np.float64)
        nominal_envelope = self._builtin_finite_number(nominal_envelope_m)
        if (
            type(steps_before) is not int
            or type(steps_after) is not int
            or steps_before < 0
            or steps_after <= steps_before
            or type(samples) is not tuple
            or len(samples) != steps_after - steps_before
            or start.shape != (4, 4)
            or goal.shape != (4, 4)
            or not np.all(np.isfinite(start))
            or not np.all(np.isfinite(goal))
            or nominal_envelope is None
            or nominal_envelope <= 0.0
        ):
            evidence["reason"] = "public_motion_sample_count_or_header_mismatch"
            return evidence
        commanded_delta = goal[:3, 3] - start[:3, 3]
        commanded_distance = float(np.linalg.norm(commanded_delta))
        if not np.isfinite(commanded_distance) or commanded_distance <= 0.0:
            evidence["reason"] = "public_motion_command_direction_is_invalid"
            return evidence
        direction = commanded_delta / commanded_distance
        measured_poses: list[np.ndarray] = []
        measured_widths: list[float] = []
        for offset, sample in enumerate(samples, start=1):
            if not isinstance(sample, Mapping) or set(sample) != {
                "step",
                "ee_pose_world",
                "gripper_width_m",
            }:
                evidence["reason"] = "public_motion_sample_is_malformed"
                return evidence
            step_value = sample.get("step")
            width = self._builtin_finite_number(sample.get("gripper_width_m"))
            pose = np.asarray(sample.get("ee_pose_world"), dtype=np.float64)
            if (
                type(step_value) is not int
                or step_value != steps_before + offset
                or width is None
                or width < self._FREE_RIM_VIEW_FIELD_MIN_OPEN_WIDTH_M
                or pose.shape != (4, 4)
                or not np.all(np.isfinite(pose))
                or not np.allclose(
                    pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6
                )
                or not np.allclose(
                    pose[:3, :3].T @ pose[:3, :3],
                    np.eye(3),
                    atol=2e-3,
                )
                or float(np.linalg.det(pose[:3, :3])) <= 0.0
            ):
                evidence["reason"] = "public_motion_sample_fails_type_open_or_pose_gate"
                return evidence
            measured_poses.append(pose.copy())
            measured_widths.append(width)

        measured_xyz = np.vstack(
            (start[:3, 3], *(pose[:3, 3] for pose in measured_poses))
        )
        step_lengths = np.linalg.norm(np.diff(measured_xyz, axis=0), axis=1)
        deltas = measured_xyz - start[:3, 3]
        progress = deltas @ direction
        cross = np.linalg.norm(
            deltas - progress[:, None] * direction[None, :], axis=1
        )
        rotations = np.asarray(
            [self._rotation_distance_rad(pose, start) for pose in measured_poses],
            dtype=np.float64,
        )
        candidate_raw = np.asarray(
            candidate_field.distance(measured_xyz), dtype=np.float64
        )
        lower = np.asarray(scene.workspace_min, dtype=np.float64)
        upper = np.asarray(scene.workspace_max, dtype=np.float64)
        workspace_ok = bool(
            lower.shape == (3,)
            and upper.shape == (3,)
            and np.all(np.isfinite(lower))
            and np.all(np.isfinite(upper))
            and np.all(
                measured_xyz[:, :2]
                >= lower[:2] + self._FREE_RIM_VIEW_REACQUIRE_WORKSPACE_MARGIN_M
            )
            and np.all(
                measured_xyz[:, :2]
                <= upper[:2] - self._FREE_RIM_VIEW_REACQUIRE_WORKSPACE_MARGIN_M
            )
            and np.all(measured_xyz[:, 2] >= lower[2])
            and np.all(measured_xyz[:, 2] <= upper[2])
        )
        retained_raw: dict[str, list[float]] = {}
        retained_ok = True
        obstacle = scene.obstacle_sdf
        if not isinstance(obstacle, CompositeSDF):
            retained_ok = False
        else:
            for index, retained in enumerate(obstacle.fields):
                if retained is candidate_field:
                    continue
                values = np.asarray(retained.distance(measured_xyz), dtype=np.float64)
                key = str(
                    getattr(retained, "source_instance_id", None)
                    or f"sensor-field-{index}"
                )
                retained_raw[key] = values.tolist()
                retained_ok = bool(
                    retained_ok
                    and values.shape == (len(measured_xyz),)
                    and np.all(np.isfinite(values))
                    and float(np.min(values)) >= nominal_envelope
                )
        accepted = bool(
            np.all(np.isfinite(step_lengths))
            and np.all(
                step_lengths
                <= self._FREE_RIM_TARGET_EGRESS_WAYPOINT_SPACING_M + 1e-12
            )
            and np.all(np.isfinite(rotations))
            and np.all(
                rotations
                <= self._FREE_RIM_TARGET_EGRESS_SAFE_POSE_ROTATION_TOLERANCE_RAD
            )
            and np.all(np.isfinite(cross))
            and float(np.max(cross))
            <= self._FREE_RIM_VIEW_REACQUIRE_MAX_CROSS_DRIFT_M
            and float(np.max(np.abs(measured_xyz[:, 2] - start[2, 3])))
            <= self._FREE_RIM_VIEW_REACQUIRE_MAX_VERTICAL_DRIFT_M
            and workspace_ok
            and candidate_raw.shape == (len(measured_xyz),)
            and np.all(np.isfinite(candidate_raw))
            and np.all(np.diff(candidate_raw) > 1e-8)
            and candidate_raw[-1] >= nominal_envelope
            and retained_ok
        )
        evidence.update(
            {
                "accepted": accepted,
                "reason": (
                    "every_public_action_passed_frozen_sensor_corridor"
                    if accepted
                    else "public_action_failed_frozen_sensor_corridor"
                ),
                "motion_policy_actions": steps_after - steps_before,
                "measured_step_lengths_m": step_lengths.tolist(),
                "measured_cross_drift_m": cross.tolist(),
                "measured_rotation_drift_rad": rotations.tolist(),
                "maximum_vertical_drift_m": float(
                    np.max(np.abs(measured_xyz[:, 2] - start[2, 3]))
                ),
                "workspace_xyz_gate": workspace_ok,
                "candidate_raw_sdf_m": candidate_raw.tolist(),
                "retained_raw_sdf_m": retained_raw,
                "measured_gripper_widths_m": measured_widths,
                "last_pose_world": measured_poses[-1].copy(),
            }
        )
        return evidence

    def _reacquire_three_frame_self_field_evidence(
        self,
        bound: BoundConstraintGraph,
        candidate_id: str,
        diagnostic: Mapping[str, object],
        provider_diagnostic: Mapping[str, object],
        current_pose: np.ndarray,
        proof: Mapping[str, object],
        fresh_evidence: Callable[..., object],
    ) -> tuple[dict[str, object], dict[str, object], np.ndarray]:
        """Run the sole non-collinear, high/open A-B-C evidence transaction."""

        if candidate_id in self._free_rim_view_reacquired_candidates:
            raise FreeRimViewEvidenceUnavailable(
                "free-rim self-field view evidence budget already consumed"
            )
        # Consume the independent budget before any validation or command.  A
        # malformed provider result can therefore never be retried on a later
        # frame and cannot roll into the physical primary/antipodal budget.
        self._free_rim_view_reacquired_candidates.add(candidate_id)
        action_started = False
        cleanup_done = False

        def cleanup_after_action() -> None:
            nonlocal cleanup_done
            if cleanup_done or not action_started:
                return
            cleanup_done = True
            self._cleanup_free_rim_typed_view_transaction(
                invalidate_cache=True,
                reset_retry_active=False,
            )

        def fail(reason: str, **details: object):
            cleanup_after_action()
            self._append_robot_phase_trace(
                {
                    "phase": "free_rim_three_frame_view_reacquisition",
                    "active_grasp_candidate": candidate_id,
                    "strategy": (
                        "three_fresh_frames_two_noncollinear_released_hand_motions_"
                        "with_typed_wrist_negative_visibility"
                    ),
                    "accepted": False,
                    "proof_state": "REJECT",
                    "reason": reason,
                    "candidate_failure_keys_consumed": False,
                    "view_budget_consumed": True,
                    **details,
                }
            )
            raise FreeRimViewEvidenceUnavailable(
                f"free-rim non-physical view evidence failed closed: {reason}"
            )

        if (
            provider_diagnostic.get("accepted") is not False
            or provider_diagnostic.get("reason")
            != "insufficient_camera_surface_provenance"
            or type(provider_diagnostic.get("reason")) is not str
            or type(provider_diagnostic.get("field_id")) is not str
            or provider_diagnostic.get("field_id") != proof.get("field_id")
            or type(provider_diagnostic.get("field_label")) is not str
            or self._normalise_semantic_label(
                provider_diagnostic.get("field_label")
            )
            != self._normalise_semantic_label(proof.get("field_label"))
        ):
            fail("provider_rejection_is_not_the_unique_one_camera_case")
        camera_count = provider_diagnostic.get("camera_count")
        counts = provider_diagnostic.get("per_camera_point_counts")
        if (
            type(camera_count) is not int
            or camera_count != 1
            or not isinstance(counts, Mapping)
            or set(counts) != {"agentview"}
            or any(type(name) is not str for name in counts)
            or type(counts.get("agentview")) is not int
            or counts["agentview"]
            < self._FREE_RIM_VIEW_FIELD_MIN_POINTS_PER_CAMERA
        ):
            fail("one_camera_provider_diagnostic_is_malformed_or_not_agentview")
        if (
            str(proof.get("candidate_id", "")) != candidate_id
            or str(proof.get("source_id", ""))
            != str(getattr(bound, "source_id", ""))
            or str(proof.get("target_id", ""))
            != str(getattr(bound, "target_id", ""))
            or proof.get("field_binding") != "unique_stable_assembly_field"
        ):
            fail("frozen_A_B_proof_binding_mismatch")
        labels = proof.get("labels")
        if (
            not isinstance(labels, tuple)
            or not labels
            or any(not isinstance(label, str) or not label.strip() for label in labels)
        ):
            fail("frozen_requested_sensor_labels_unavailable")
        first_snapshot = proof.get("start_view_snapshot")
        if not isinstance(first_snapshot, Mapping):
            fail("frame_A_public_camera_snapshot_unavailable")
        second_snapshot = self._public_view_snapshot(self.robot)
        if second_snapshot is None:
            fail("frame_B_public_camera_snapshot_unavailable")
        first_pose = np.asarray(proof.get("start_ee_pose_world"), dtype=np.float64)
        second_pose = np.asarray(current_pose, dtype=np.float64)
        if first_pose.shape != (4, 4) or second_pose.shape != (4, 4):
            fail("frame_A_or_B_public_pose_malformed")
        second_snapshot_pose = np.asarray(
            second_snapshot["ee_pose_world"], dtype=np.float64
        )
        endpoint_position_error, endpoint_rotation_error = self._recovery_pose_errors(
            second_snapshot_pose, second_pose
        )
        if (
            endpoint_position_error
            > self._FREE_RIM_VIEW_CAMERA_RIGID_POSITION_TOLERANCE_M
            or endpoint_rotation_error
            > self._FREE_RIM_VIEW_CAMERA_RIGID_ROTATION_TOLERANCE_RAD
        ):
            fail(
                "frame_B_camera_and_public_pose_are_not_synchronized",
                position_error_m=endpoint_position_error,
                rotation_error_rad=endpoint_rotation_error,
            )
        scene_b = self._current_sensor_scene()
        if scene_b is None:
            fail("fresh_frame_B_sensor_scene_unavailable")
        field_b = self._unique_three_frame_field(
            scene_b,
            proof,
            expected_center_world_m=diagnostic.get("field_center_world_m"),
        )
        if field_b is None:
            fail("frame_B_field_identity_size_or_unique_family_gate_failed")
        first_surfaces = self._canonical_surface_map(
            proof.get("surface_points_by_camera")
        )
        second_surfaces = self._canonical_surface_map(
            field_b.surface_points_by_camera
        )
        if (
            first_surfaces is None
            or second_surfaces is None
            or set(first_surfaces) != {"agentview"}
            or set(second_surfaces) != {"agentview"}
            or counts["agentview"] != len(second_surfaces["agentview"])
        ):
            fail("frames_A_B_are_not_exactly_agentview_positive")
        first_stamp = self._builtin_finite_number(
            proof.get("start_scene_timestamp_s")
        )
        second_stamp = self._builtin_finite_number(scene_b.timestamp_s)
        if first_stamp is None or second_stamp is None:
            fail("frame_A_or_B_timestamp_is_malformed")
        first_capture_id = proof.get("start_scene_capture_id", "")
        first_camera_capture_ids = proof.get(
            "start_scene_camera_capture_ids", {}
        )
        try:
            capture_advanced_ab = sensor_capture_advanced(
                {
                    "capture_id": first_capture_id,
                    "camera_capture_ids": first_camera_capture_ids,
                    "timestamp_s": first_stamp,
                },
                scene_b,
            )
        except ValueError:
            fail("frame_A_or_B_capture_id_is_malformed")
        if not capture_advanced_ab:
            fail("frame_B_sensor_capture_did_not_advance_from_A")
        first_endpoint = {
            "field_center_world_m": proof.get("field_center_world_m"),
            "field_half_extents_m": proof.get("field_half_extents_m"),
            "field_axes_world": proof.get("field_axes_world"),
            "surface_points_by_camera": proof.get("surface_points_by_camera"),
            "ee_pose_world": first_pose,
            "gripper_width_m": proof.get("start_gripper_width_m"),
            "raw_sdf_m": proof.get("initial_raw_sdf_m"),
            "timestamp_s": first_stamp,
            "capture_id": first_capture_id,
            "camera_capture_ids": first_camera_capture_ids,
        }
        second_endpoint = {
            "field_center_world_m": field_b.center,
            "field_half_extents_m": field_b.half_extents,
            "field_axes_world": field_b.axes,
            "surface_points_by_camera": field_b.surface_points_by_camera,
            "ee_pose_world": second_pose,
            "gripper_width_m": second_snapshot["gripper_width_m"],
            "raw_sdf_m": diagnostic.get("raw_distance_m"),
            "timestamp_s": second_stamp,
            "capture_id": scene_b.capture_id,
            "camera_capture_ids": dict(scene_b.camera_capture_ids),
        }
        pair_ab = self._three_frame_motion_pair_evidence(
            first_endpoint, second_endpoint
        )
        camera_ab = self._camera_snapshot_pair_evidence(
            first_snapshot, second_snapshot
        )
        wrist_a = self._three_frame_wrist_endpoint_evidence(
            first_endpoint, first_snapshot
        )
        wrist_b = self._three_frame_wrist_endpoint_evidence(
            second_endpoint, second_snapshot
        )
        if not all(
            bool(item.get("accepted", False))
            for item in (pair_ab, camera_ab, wrist_a, wrist_b)
        ):
            fail(
                "frozen_A_B_motion_camera_or_wrist_gate_failed",
                pair_A_B=pair_ab,
                camera_A_B=camera_ab,
                wrist_A=wrist_a,
                wrist_B=wrist_b,
            )
        if any(
            item.get("visibility_mode") != "strict_negative"
            for item in (wrist_a, wrist_b)
        ):
            fail("frames_A_B_lack_a_complete_dual_positive_or_typed_negative_chain")
        try:
            source = scene_b.by_id(str(proof["source_id"]))
            target = scene_b.by_id(str(proof["target_id"]))
            visible_value = getattr(self.observer, "visible_instance_ids", None)
            visible_ids = {
                str(instance_id) for instance_id in visible_value
            }
        except (KeyError, PerceptionError, TypeError):
            fail("frame_B_source_target_are_not_freshly_visible")
        if (
            source.instance_id not in visible_ids
            or target.instance_id not in visible_ids
            or self._normalise_semantic_label(source.label)
            != self._normalise_semantic_label(proof.get("source_label"))
            or self._normalise_semantic_label(target.label)
            != self._normalise_semantic_label(proof.get("target_label"))
        ):
            fail("frame_B_source_target_are_not_freshly_visible")
        source_top_z = float(source.position[2] + 0.5 * source.extent[2])
        minimum_height = source_top_z + self._FREE_RIM_ESCAPE_MIN_HEIGHT_ABOVE_SOURCE_M
        ceiling = min(
            self.config.recovery_motion_ceiling_z_m,
            float(scene_b.workspace_max[2]),
        )
        if (
            float(second_pose[2, 3]) < minimum_height
            or float(second_pose[2, 3]) > ceiling
        ):
            fail(
                "frame_B_is_not_a_sensor_safe_high_pose",
                minimum_height_m=minimum_height,
                ceiling_m=ceiling,
            )
        first_motion = second_pose[:3, 3] - first_pose[:3, 3]
        nominal_envelope = self._builtin_finite_number(
            proof.get("nominal_envelope_m")
        )
        if nominal_envelope is None:
            fail("frozen_inflated_clearance_envelope_unavailable")
        proof_final_pose = np.asarray(
            proof.get("final_ee_pose_world"), dtype=np.float64
        )
        frozen_direction = np.asarray(
            proof.get("egress_command_direction_world"), dtype=np.float64
        )
        if (
            proof_final_pose.shape != (4, 4)
            or not np.all(np.isfinite(proof_final_pose))
            or frozen_direction.shape != (3,)
            or not np.all(np.isfinite(frozen_direction))
            or not np.isclose(
                float(np.linalg.norm(frozen_direction)),
                1.0,
                rtol=0.0,
                atol=1e-9,
            )
        ):
            fail("frozen_A_B_public_path_evidence_is_malformed")
        refresh_position_drift, refresh_rotation_drift = self._recovery_pose_errors(
            second_pose, proof_final_pose
        )
        ab_delta = second_pose[:3, 3] - first_pose[:3, 3]
        ab_progress = float(np.dot(ab_delta, frozen_direction))
        ab_cross = float(
            np.linalg.norm(ab_delta - ab_progress * frozen_direction)
        )
        ab_vertical = abs(float(second_pose[2, 3] - first_pose[2, 3]))
        ab_rotation = self._rotation_distance_rad(second_pose, first_pose)
        try:
            frozen_field_a = BoxSDF(
                np.asarray(proof["field_center_world_m"], dtype=np.float64),
                np.asarray(proof["field_half_extents_m"], dtype=np.float64),
                np.asarray(proof["field_axes_world"], dtype=np.float64),
            )
            frozen_candidate_b_raw = float(
                frozen_field_a.distance(second_pose[:3, 3])
            )
        except (KeyError, TypeError, ValueError):
            fail("frozen_A_candidate_field_is_malformed")
        if (
            refresh_position_drift
            > self._FREE_RIM_VIEW_CAMERA_RIGID_POSITION_TOLERANCE_M
            or refresh_rotation_drift
            > self._FREE_RIM_VIEW_CAMERA_RIGID_ROTATION_TOLERANCE_RAD
            or float(np.linalg.norm(second_pose[:3, 3] - proof_final_pose[:3, 3]))
            > self._FREE_RIM_TARGET_EGRESS_WAYPOINT_SPACING_M + 1e-12
            or ab_progress < self._FREE_RIM_ESCAPE_MIN_PROGRESS_M
            or float(np.linalg.norm(ab_delta))
            > self._FREE_RIM_TARGET_EGRESS_MAX_DISTANCE_M
            + self._FREE_RIM_ESCAPE_MAX_CROSS_DRIFT_M
            + 1e-12
            or ab_cross > self._FREE_RIM_ESCAPE_MAX_CROSS_DRIFT_M
            or ab_vertical > self._FREE_RIM_VIEW_REACQUIRE_MAX_VERTICAL_DRIFT_M
            or ab_rotation
            > self._FREE_RIM_TARGET_EGRESS_SAFE_POSE_ROTATION_TOLERANCE_RAD
            or not np.isfinite(frozen_candidate_b_raw)
            or frozen_candidate_b_raw < nominal_envelope
            or float(second_snapshot["gripper_width_m"])
            < self._FREE_RIM_VIEW_FIELD_MIN_OPEN_WIDTH_M
        ):
            fail(
                "frame_B_refresh_failed_frozen_A_B_path_gate",
                refresh_position_drift_m=refresh_position_drift,
                refresh_rotation_drift_rad=refresh_rotation_drift,
                signed_progress_m=ab_progress,
                cross_drift_m=ab_cross,
                vertical_drift_m=ab_vertical,
                rotation_drift_rad=ab_rotation,
                frozen_candidate_raw_sdf_m=frozen_candidate_b_raw,
            )
        frame_b_retained_raw: dict[str, float] = {}
        frame_b_retained_ok = True
        for index, retained in enumerate(scene_b.obstacle_sdf.fields):
            if retained is field_b:
                continue
            value = float(retained.distance(second_pose[:3, 3]))
            frame_b_retained_raw[
                str(
                    getattr(retained, "source_instance_id", None)
                    or f"sensor-field-{index}"
                )
            ] = value
            frame_b_retained_ok = bool(
                frame_b_retained_ok
                and np.isfinite(value)
                and value >= nominal_envelope
            )
        if (
            not frame_b_retained_ok
            or np.any(
                second_pose[:2, 3]
                < scene_b.workspace_min[:2]
                + self._FREE_RIM_VIEW_REACQUIRE_WORKSPACE_MARGIN_M
            )
            or np.any(
                second_pose[:2, 3]
                > scene_b.workspace_max[:2]
                - self._FREE_RIM_VIEW_REACQUIRE_WORKSPACE_MARGIN_M
            )
            or second_pose[2, 3] < scene_b.workspace_min[2]
            or second_pose[2, 3] > scene_b.workspace_max[2]
        ):
            fail(
                "frame_B_fresh_workspace_or_retained_field_gate_failed",
                retained_raw_sdf_m=frame_b_retained_raw,
            )
        planned = self._plan_three_frame_noncollinear_path(
            scene_b,
            field_b,
            second_pose,
            first_motion,
            nominal_envelope,
        )
        if planned is None:
            fail("no_fresh_SDF_safe_noncollinear_B_C_path")
        goal, waypoints, path_distances = planned
        segment_lengths = np.linalg.norm(
            np.diff(waypoints[:, :3, 3], axis=0), axis=1
        )
        if (
            len(segment_lengths) < 1
            or not np.all(np.isfinite(segment_lengths))
            or float(np.max(segment_lengths))
            > self._FREE_RIM_TARGET_EGRESS_WAYPOINT_SPACING_M + 1e-12
        ):
            fail("B_C_waypoint_spacing_gate_failed")
        command = self._released_gripper_command(GraspMode.RIM_PINCH)
        if command != -1.0:
            fail("B_C_view_motion_lacks_the_released_command")
        execute_view = getattr(
            self.robot, "execute_sensor_safe_view_waypoints", None
        )
        if not callable(execute_view):
            fail("B_C_sensor_safe_waypoint_capability_unavailable")
        steps_before = getattr(self.robot, "steps_executed", None)
        if type(steps_before) is not int or steps_before < 0:
            fail("B_C_start_policy_action_counter_is_not_a_builtin_int")
        step_budget = getattr(self.robot, "step_budget", None)
        if (
            type(step_budget) is not int
            or step_budget - steps_before
            < self._FREE_RIM_VIEW_REACQUIRE_MAX_POLICY_ACTIONS
        ):
            fail("B_C_complete_view_and_refresh_budget_not_pre_reserved")
        action_started = True
        try:
            feedback = execute_view(
                waypoints,
                command,
                maximum_policy_actions=(
                    self._FREE_RIM_VIEW_REACQUIRE_MAX_POLICY_ACTIONS - 1
                ),
                reserved_followup_actions=1,
            )
        except PolicyStepBudgetExhausted:
            cleanup_after_action()
            raise
        except (TypeError, ValueError, RuntimeError) as exc:
            fail(
                "B_C_sensor_safe_waypoint_execution_error",
                error_type=type(exc).__name__,
            )
        if not isinstance(feedback, ControllerFeedback):
            fail("B_C_view_motion_returned_invalid_feedback")
        steps_after_motion = getattr(self.robot, "steps_executed", None)
        if type(steps_after_motion) is not int or steps_after_motion < steps_before:
            fail("B_C_end_policy_action_counter_is_not_a_builtin_int")
        motion_steps = steps_after_motion - steps_before
        if (
            type(feedback.accepted) is not bool
            or motion_steps < 1
            or motion_steps
            > self._FREE_RIM_VIEW_REACQUIRE_MAX_POLICY_ACTIONS - 1
        ):
            fail("B_C_feedback_or_policy_action_count_gate_failed")
        budget_exhausted = getattr(self.robot, "step_budget_exhausted", False)
        if type(budget_exhausted) is not bool:
            fail("B_C_step_budget_flag_is_not_a_builtin_bool")
        if budget_exhausted:
            cleanup_after_action()
            raise PolicyStepBudgetExhausted(
                "episode OSC step budget exhausted during three-frame view motion"
            )
        measured_value = getattr(
            self.robot, "last_sensor_safe_view_motion_samples", None
        )
        if callable(measured_value):
            try:
                measured_value = measured_value()
            except (TypeError, ValueError, RuntimeError) as exc:
                fail(
                    "B_C_public_motion_samples_error",
                    error_type=type(exc).__name__,
                )
        if type(measured_value) is not tuple or len(measured_value) != motion_steps:
            fail("B_C_public_motion_sample_count_mismatch")
        measured_poses: list[np.ndarray] = []
        measured_widths: list[float] = []
        for offset, sample in enumerate(measured_value, start=1):
            if not isinstance(sample, Mapping) or set(sample) != {
                "step",
                "ee_pose_world",
                "gripper_width_m",
            }:
                fail("B_C_public_motion_sample_is_malformed")
            step_value = sample.get("step")
            width_value = self._builtin_finite_number(
                sample.get("gripper_width_m")
            )
            pose_value = np.asarray(
                sample.get("ee_pose_world"), dtype=np.float64
            )
            if (
                type(step_value) is not int
                or step_value != steps_before + offset
                or width_value is None
                or width_value < self._FREE_RIM_VIEW_FIELD_MIN_OPEN_WIDTH_M
                or pose_value.shape != (4, 4)
                or not np.all(np.isfinite(pose_value))
                or not np.allclose(
                    pose_value[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6
                )
                or not np.allclose(
                    pose_value[:3, :3].T @ pose_value[:3, :3],
                    np.eye(3),
                    atol=2e-3,
                )
                or float(np.linalg.det(pose_value[:3, :3])) <= 0.0
            ):
                fail("B_C_public_motion_sample_fails_type_or_open_pose_gate")
            measured_poses.append(pose_value.copy())
            measured_widths.append(width_value)
        measured_xyz = np.vstack(
            (second_pose[:3, 3], *(pose[:3, 3] for pose in measured_poses))
        )
        measured_step_lengths = np.linalg.norm(
            np.diff(measured_xyz, axis=0), axis=1
        )
        measured_path_arc_length = float(np.sum(measured_step_lengths))
        measured_rotations = np.asarray(
            [
                self._rotation_distance_rad(pose, second_pose)
                for pose in measured_poses
            ],
            dtype=np.float64,
        )
        commanded_delta_for_samples = goal[:3, 3] - second_pose[:3, 3]
        commanded_distance_for_samples = float(
            np.linalg.norm(commanded_delta_for_samples)
        )
        direction_for_samples = (
            commanded_delta_for_samples / commanded_distance_for_samples
        )
        measured_deltas = measured_xyz - second_pose[:3, 3]
        measured_progress = measured_deltas @ direction_for_samples
        measured_cross = np.linalg.norm(
            measured_deltas
            - measured_progress[:, None] * direction_for_samples[None, :],
            axis=1,
        )
        candidate_measured_raw = np.asarray(
            field_b.distance(measured_xyz), dtype=np.float64
        )
        retained_measured_raw: dict[str, list[float]] = {}
        measured_path_ok = bool(
            np.all(np.isfinite(measured_step_lengths))
            and np.isfinite(measured_path_arc_length)
            and measured_path_arc_length
            <= self._FREE_RIM_TARGET_EGRESS_MAX_DISTANCE_M
            and np.all(
                measured_step_lengths
                <= self._FREE_RIM_TARGET_EGRESS_WAYPOINT_SPACING_M + 1e-12
            )
            and np.all(np.isfinite(candidate_measured_raw))
            and np.all(np.diff(candidate_measured_raw) > 1e-8)
            and candidate_measured_raw[-1] >= nominal_envelope
            and np.all(np.isfinite(measured_rotations))
            and np.all(
                measured_rotations
                <= self._FREE_RIM_TARGET_EGRESS_SAFE_POSE_ROTATION_TOLERANCE_RAD
            )
            and np.max(np.abs(measured_xyz[:, 2] - second_pose[2, 3]))
            <= self._FREE_RIM_VIEW_REACQUIRE_MAX_VERTICAL_DRIFT_M
            and np.max(measured_cross)
            <= self._FREE_RIM_VIEW_REACQUIRE_MAX_CROSS_DRIFT_M
            and np.all(
                measured_xyz[:, :2]
                >= scene_b.workspace_min[:2]
                + self._FREE_RIM_VIEW_REACQUIRE_WORKSPACE_MARGIN_M
            )
            and np.all(
                measured_xyz[:, :2]
                <= scene_b.workspace_max[:2]
                - self._FREE_RIM_VIEW_REACQUIRE_WORKSPACE_MARGIN_M
            )
            and np.all(measured_xyz[:, 2] >= scene_b.workspace_min[2])
            and np.all(measured_xyz[:, 2] <= scene_b.workspace_max[2])
        )
        for index, retained in enumerate(scene_b.obstacle_sdf.fields):
            if retained is field_b:
                continue
            values = np.asarray(retained.distance(measured_xyz), dtype=np.float64)
            retained_measured_raw[
                str(
                    getattr(retained, "source_instance_id", None)
                    or f"sensor-field-{index}"
                )
            ] = values.tolist()
            measured_path_ok = bool(
                measured_path_ok
                and values.shape == (len(measured_xyz),)
                and np.all(np.isfinite(values))
                and float(np.min(values)) >= nominal_envelope
            )
        if not measured_path_ok:
            fail(
                "B_C_measured_path_failed_frozen_per_field_SDF_gate",
                candidate_raw_sdf_m=candidate_measured_raw.tolist(),
                retained_raw_sdf_m=retained_measured_raw,
                measured_step_lengths_m=measured_step_lengths.tolist(),
                measured_path_arc_length_m=measured_path_arc_length,
                measured_cross_drift_m=measured_cross.tolist(),
                measured_rotation_drift_rad=measured_rotations.tolist(),
            )
        actual = self._public_ee_pose(self.robot)
        if actual is None:
            fail("frame_C_public_pose_unavailable_after_motion")
        commanded_delta = goal[:3, 3] - second_pose[:3, 3]
        commanded_distance = float(np.linalg.norm(commanded_delta))
        direction = commanded_delta / commanded_distance
        actual_delta = actual[:3, 3] - second_pose[:3, 3]
        actual_progress = float(np.dot(actual_delta, direction))
        cross_drift = float(
            np.linalg.norm(actual_delta - actual_progress * direction)
        )
        actual_distance = float(np.linalg.norm(actual_delta))
        vertical_drift = abs(float(actual[2, 3] - second_pose[2, 3]))
        rotation_drift = self._rotation_distance_rad(actual, second_pose)
        last_motion_position_error, last_motion_rotation_error = (
            self._recovery_pose_errors(actual, measured_poses[-1])
        )
        post_motion_arc_segment_m = float(
            np.linalg.norm(actual[:3, 3] - measured_poses[-1][:3, 3])
        )
        path_arc_through_post_motion_m = (
            measured_path_arc_length + post_motion_arc_segment_m
        )
        try:
            post_motion_width = self._builtin_finite_number(
                self.robot.current_gripper_width_m()
            )
        except (AttributeError, TypeError, ValueError, RuntimeError):
            post_motion_width = None
        post_motion_raw_sdf = float(
            scene_b.obstacle_sdf.distance(actual[:3, 3])
        )
        first_direction = first_motion[:2] / float(np.linalg.norm(first_motion[:2]))
        actual_planar = actual_delta[:2]
        actual_planar_norm = float(np.linalg.norm(actual_planar))
        noncollinear_cosine = (
            abs(float(np.dot(first_direction, actual_planar / actual_planar_norm)))
            if actual_planar_norm > 0.0
            else float("inf")
        )
        if (
            not feedback.accepted
            or motion_steps < 1
            or motion_steps > self._FREE_RIM_VIEW_REACQUIRE_MAX_POLICY_ACTIONS - 1
            or actual_distance < self._FREE_RIM_VIEW_FIELD_MIN_EE_MOTION_M
            or actual_distance
            > self._FREE_RIM_TARGET_EGRESS_MAX_DISTANCE_M + 1e-12
            or actual_progress < commanded_distance - 0.008
            or cross_drift > self._FREE_RIM_VIEW_REACQUIRE_MAX_CROSS_DRIFT_M
            or vertical_drift > self._FREE_RIM_VIEW_REACQUIRE_MAX_VERTICAL_DRIFT_M
            or rotation_drift
            > self._FREE_RIM_TARGET_EGRESS_SAFE_POSE_ROTATION_TOLERANCE_RAD
            or last_motion_position_error
            > self._FREE_RIM_VIEW_CAMERA_RIGID_POSITION_TOLERANCE_M
            or last_motion_rotation_error
            > self._FREE_RIM_VIEW_CAMERA_RIGID_ROTATION_TOLERANCE_RAD
            or not np.isfinite(post_motion_arc_segment_m)
            or not np.isfinite(path_arc_through_post_motion_m)
            or path_arc_through_post_motion_m
            > self._FREE_RIM_TARGET_EGRESS_MAX_DISTANCE_M
            or post_motion_width is None
            or post_motion_width < self._FREE_RIM_VIEW_FIELD_MIN_OPEN_WIDTH_M
            or not np.isfinite(post_motion_raw_sdf)
            or post_motion_raw_sdf < nominal_envelope
            or noncollinear_cosine
            > self._FREE_RIM_VIEW_REACQUIRE_MAX_DIRECTION_COSINE + 1e-12
            or np.any(
                actual[:2, 3]
                < scene_b.workspace_min[:2]
                + self._FREE_RIM_VIEW_REACQUIRE_WORKSPACE_MARGIN_M
            )
            or np.any(
                actual[:2, 3]
                > scene_b.workspace_max[:2]
                - self._FREE_RIM_VIEW_REACQUIRE_WORKSPACE_MARGIN_M
            )
            or actual[2, 3] < scene_b.workspace_min[2]
            or actual[2, 3] > scene_b.workspace_max[2]
        ):
            fail(
                "B_C_public_proprio_motion_gate_failed",
                feedback_accepted=bool(feedback.accepted),
                motion_policy_actions=motion_steps,
                commanded_distance_m=commanded_distance,
                actual_distance_m=actual_distance,
                signed_progress_m=actual_progress,
                cross_drift_m=cross_drift,
                vertical_drift_m=vertical_drift,
                rotation_drift_rad=rotation_drift,
                last_motion_sample_position_error_m=(
                    last_motion_position_error
                ),
                last_motion_sample_rotation_error_rad=(
                    last_motion_rotation_error
                ),
                motion_sample_arc_length_m=measured_path_arc_length,
                post_motion_arc_segment_m=post_motion_arc_segment_m,
                path_arc_through_post_motion_m=(
                    path_arc_through_post_motion_m
                ),
                post_motion_gripper_width_m=post_motion_width,
                post_motion_raw_sdf_m=post_motion_raw_sdf,
                noncollinear_abs_cosine=noncollinear_cosine,
            )
        refresh = getattr(self.robot, "capture_fresh_sensor_frame_at_pose", None)
        if not callable(refresh):
            fail("frame_C_fixed_pose_sensor_refresh_capability_unavailable")
        try:
            refresh_feedback = refresh(command)
        except PolicyStepBudgetExhausted:
            cleanup_after_action()
            raise
        except (TypeError, ValueError, RuntimeError) as exc:
            fail(
                "frame_C_sensor_refresh_error",
                error_type=type(exc).__name__,
            )
        if not isinstance(refresh_feedback, ControllerFeedback):
            fail("frame_C_sensor_refresh_returned_invalid_feedback")
        if type(refresh_feedback.accepted) is not bool:
            fail("frame_C_sensor_refresh_acceptance_is_not_a_builtin_bool")
        steps_after_refresh = getattr(self.robot, "steps_executed", None)
        if (
            type(steps_after_refresh) is not int
            or steps_after_refresh != steps_after_motion + 1
            or steps_after_refresh - steps_before
            > self._FREE_RIM_VIEW_REACQUIRE_MAX_POLICY_ACTIONS
        ):
            fail("frame_C_sensor_refresh_policy_action_budget_gate_failed")
        refresh_budget_exhausted = getattr(
            self.robot, "step_budget_exhausted", False
        )
        if type(refresh_budget_exhausted) is not bool:
            fail("frame_C_step_budget_flag_is_not_a_builtin_bool")
        if refresh_budget_exhausted:
            cleanup_after_action()
            raise PolicyStepBudgetExhausted(
                "episode OSC step budget exhausted during three-frame refresh"
            )
        if not refresh_feedback.accepted:
            fail("frame_C_fixed_pose_sensor_refresh_failed")
        invalidate = getattr(self.observer, "invalidate_sensor_cache", None)
        if not callable(invalidate):
            fail("frame_C_sensor_cache_invalidation_unavailable")
        try:
            invalidate()
        except (TypeError, ValueError, RuntimeError) as exc:
            fail(
                "frame_C_sensor_cache_invalidation_error",
                error_type=type(exc).__name__,
            )
        if self._current_sensor_scene() is not None:
            fail("frame_C_invalidation_retained_a_stale_sensor_scene")
        try:
            scene_c = self.observer.observe(labels)
        except (AttributeError, TypeError, ValueError, PerceptionError) as exc:
            fail("fresh_frame_C_observation_failed", error_type=type(exc).__name__)
        if not isinstance(scene_c, SceneEstimate):
            fail("fresh_frame_C_observation_has_invalid_type")
        third_snapshot = self._public_view_snapshot(self.robot)
        if third_snapshot is None:
            fail("frame_C_public_camera_snapshot_unavailable")
        refreshed_pose = np.asarray(
            third_snapshot["ee_pose_world"], dtype=np.float64
        )
        refresh_arc_segment_m = float(
            np.linalg.norm(refreshed_pose[:3, 3] - actual[:3, 3])
        )
        complete_path_segment_lengths = np.concatenate(
            (
                measured_step_lengths,
                np.asarray(
                    (post_motion_arc_segment_m, refresh_arc_segment_m),
                    dtype=np.float64,
                ),
            )
        )
        complete_path_arc_length_m = float(
            np.sum(complete_path_segment_lengths)
        )
        if (
            not np.all(np.isfinite(complete_path_segment_lengths))
            or not np.isfinite(complete_path_arc_length_m)
            or complete_path_arc_length_m
            > self._FREE_RIM_TARGET_EGRESS_MAX_DISTANCE_M
        ):
            fail(
                "B_C_complete_public_path_arc_length_gate_failed",
                motion_sample_segment_lengths_m=(
                    measured_step_lengths.tolist()
                ),
                post_motion_arc_segment_m=post_motion_arc_segment_m,
                refresh_arc_segment_m=refresh_arc_segment_m,
                complete_path_segment_lengths_m=(
                    complete_path_segment_lengths.tolist()
                ),
                complete_path_arc_length_m=complete_path_arc_length_m,
                hard_path_arc_length_upper_bound_m=(
                    self._FREE_RIM_TARGET_EGRESS_MAX_DISTANCE_M
                ),
            )
        hold_position_drift, hold_rotation_drift = self._recovery_pose_errors(
            refreshed_pose, actual
        )
        refreshed_delta = refreshed_pose[:3, 3] - second_pose[:3, 3]
        refreshed_distance = float(np.linalg.norm(refreshed_delta))
        refreshed_progress = float(np.dot(refreshed_delta, direction))
        refreshed_cross_drift = float(
            np.linalg.norm(
                refreshed_delta - refreshed_progress * direction
            )
        )
        refreshed_vertical_drift = abs(
            float(refreshed_pose[2, 3] - second_pose[2, 3])
        )
        refreshed_rotation_drift = self._rotation_distance_rad(
            refreshed_pose, second_pose
        )
        refreshed_planar = refreshed_delta[:2]
        refreshed_planar_norm = float(np.linalg.norm(refreshed_planar))
        refreshed_noncollinear_cosine = (
            abs(
                float(
                    np.dot(
                        first_direction,
                        refreshed_planar / refreshed_planar_norm,
                    )
                )
            )
            if refreshed_planar_norm > 0.0
            else float("inf")
        )
        refreshed_frozen_raw_sdf = float(
            scene_b.obstacle_sdf.distance(refreshed_pose[:3, 3])
        )
        if (
            hold_position_drift
            > self._FREE_RIM_VIEW_CAMERA_RIGID_POSITION_TOLERANCE_M
            or hold_rotation_drift
            > self._FREE_RIM_VIEW_CAMERA_RIGID_ROTATION_TOLERANCE_RAD
            or refreshed_distance < self._FREE_RIM_VIEW_FIELD_MIN_EE_MOTION_M
            or refreshed_distance
            > self._FREE_RIM_TARGET_EGRESS_MAX_DISTANCE_M + 1e-12
            or refreshed_progress < commanded_distance - 0.008
            or refreshed_cross_drift
            > self._FREE_RIM_VIEW_REACQUIRE_MAX_CROSS_DRIFT_M
            or refreshed_vertical_drift
            > self._FREE_RIM_VIEW_REACQUIRE_MAX_VERTICAL_DRIFT_M
            or refreshed_rotation_drift
            > self._FREE_RIM_TARGET_EGRESS_SAFE_POSE_ROTATION_TOLERANCE_RAD
            or refreshed_noncollinear_cosine
            > self._FREE_RIM_VIEW_REACQUIRE_MAX_DIRECTION_COSINE + 1e-12
            or float(third_snapshot["gripper_width_m"])
            < self._FREE_RIM_VIEW_FIELD_MIN_OPEN_WIDTH_M
            or not np.isfinite(refreshed_frozen_raw_sdf)
            or refreshed_frozen_raw_sdf < nominal_envelope
            or np.any(
                refreshed_pose[:2, 3]
                < scene_b.workspace_min[:2]
                + self._FREE_RIM_VIEW_REACQUIRE_WORKSPACE_MARGIN_M
            )
            or np.any(
                refreshed_pose[:2, 3]
                > scene_b.workspace_max[:2]
                - self._FREE_RIM_VIEW_REACQUIRE_WORKSPACE_MARGIN_M
            )
            or refreshed_pose[2, 3] < scene_b.workspace_min[2]
            or refreshed_pose[2, 3] > scene_b.workspace_max[2]
        ):
            fail(
                "frame_C_refresh_drift_or_open_gate_failed",
                hold_position_drift_m=hold_position_drift,
                hold_rotation_drift_rad=hold_rotation_drift,
                refreshed_distance_m=refreshed_distance,
                refreshed_signed_progress_m=refreshed_progress,
                refreshed_cross_drift_m=refreshed_cross_drift,
                refreshed_vertical_drift_m=refreshed_vertical_drift,
                refreshed_rotation_drift_rad=refreshed_rotation_drift,
                refreshed_noncollinear_abs_cosine=(
                    refreshed_noncollinear_cosine
                ),
                refreshed_frozen_raw_sdf_m=refreshed_frozen_raw_sdf,
            )
        actual = refreshed_pose.copy()
        third_stamp = self._builtin_finite_number(scene_c.timestamp_s)
        if third_stamp is None:
            fail("frame_C_timestamp_is_malformed")
        try:
            capture_advanced_bc = sensor_capture_advanced(scene_b, scene_c)
        except ValueError:
            fail("frame_B_or_C_capture_id_is_malformed")
        if not capture_advanced_bc:
            fail("frame_C_sensor_capture_did_not_advance_from_B")
        field_c = self._unique_three_frame_field(scene_c, proof)
        if field_c is None:
            fail("frame_C_field_identity_size_or_unique_family_gate_failed")
        # Freeze field, source and target authority *before* invoking a
        # provider.  Every ndarray handed across the boundary is a copy and
        # every scalar used after the call comes from this immutable snapshot.
        frame_c_authority = self._freeze_field_provider_authority(
            scene=scene_c,
            field=field_c,
            bound=bound,
            proof=proof,
        )
        if frame_c_authority is None:
            fail("frame_C_authority_snapshot_is_malformed_or_inconsistent")
        try:
            frozen_field_c_id = frame_c_authority["field_id"]
            frozen_field_c_label = frame_c_authority["field_label"]
            frozen_field_c_center = np.asarray(
                frame_c_authority["field_center_world_m"]
            ).copy()
            frozen_field_c_half = np.asarray(
                frame_c_authority["field_half_extents_m"]
            ).copy()
            frozen_field_c_axes = np.asarray(
                frame_c_authority["field_axes_world"]
            ).copy()
            frozen_field_c_surface_rows = tuple(
                (name, points.copy())
                for name, points in frame_c_authority[
                    "surface_points_by_camera"
                ]
            )
            frozen_source_c_id = frame_c_authority["source_id"]
            frozen_source_c_label = frame_c_authority["source_label"]
            frozen_target_c_id = frame_c_authority["target_id"]
            frozen_target_c_label = frame_c_authority["target_label"]
            frozen_field_c_surfaces = self._strict_provider_surface_map(
                frozen_field_c_surface_rows
            )
            frozen_field_c_sdf = BoxSDF(
                frozen_field_c_center,
                frozen_field_c_half,
                frozen_field_c_axes,
                source_instance_id=frozen_field_c_id,
                source_label=frozen_field_c_label,
            )
            frozen_field_c_raw = float(
                frozen_field_c_sdf.distance(actual[:3, 3])
            )
        except (KeyError, TypeError, ValueError, PerceptionError):
            fail("frame_C_authority_snapshot_is_malformed")
        if (
            type(frozen_field_c_id) is not str
            or type(frozen_field_c_label) is not str
            or type(frozen_source_c_id) is not str
            or type(frozen_source_c_label) is not str
            or type(frozen_target_c_id) is not str
            or type(frozen_target_c_label) is not str
            or frozen_field_c_surfaces is None
            or not np.isfinite(frozen_field_c_raw)
        ):
            fail("frame_C_authority_snapshot_is_malformed")
        fresh_c_retained_raw: dict[str, float] = {}
        fresh_c_retained_ok = True
        for index, retained in enumerate(scene_c.obstacle_sdf.fields):
            if retained is field_c:
                continue
            value = float(retained.distance(actual[:3, 3]))
            fresh_c_retained_raw[
                str(
                    getattr(retained, "source_instance_id", None)
                    or f"sensor-field-{index}"
                )
            ] = value
            fresh_c_retained_ok = bool(
                fresh_c_retained_ok
                and np.isfinite(value)
                and value >= nominal_envelope
            )
        if (
            not fresh_c_retained_ok
            or np.any(
                actual[:2, 3]
                < scene_c.workspace_min[:2]
                + self._FREE_RIM_VIEW_REACQUIRE_WORKSPACE_MARGIN_M
            )
            or np.any(
                actual[:2, 3]
                > scene_c.workspace_max[:2]
                - self._FREE_RIM_VIEW_REACQUIRE_WORKSPACE_MARGIN_M
            )
            or actual[2, 3] < scene_c.workspace_min[2]
            or actual[2, 3] > scene_c.workspace_max[2]
        ):
            fail(
                "frame_C_fresh_workspace_or_retained_field_gate_failed",
                retained_raw_sdf_m=fresh_c_retained_raw,
            )
        try:
            current_field = fresh_evidence(
                field_id=frozen_field_c_id,
                field_label=frozen_field_c_label,
                field_center_world_m=frozen_field_c_center.copy(),
                field_half_extents_m=frozen_field_c_half.copy(),
                field_axes_world=frozen_field_c_axes.copy(),
                surface_points_by_camera=tuple(
                    (name, points.copy())
                    for name, points in frozen_field_c_surface_rows
                ),
                source_id=frozen_source_c_id,
                source_label=frozen_source_c_label,
                target_id=frozen_target_c_id,
                target_label=frozen_target_c_label,
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            fail("frame_C_field_provider_error", error_type=type(exc).__name__)
        if not isinstance(current_field, Mapping):
            third_provider_diagnostic = getattr(
                self.observer,
                "last_temporal_proprio_field_evidence_diagnostic",
                None,
            )
            third_surfaces = frozen_field_c_surfaces
            if (
                not isinstance(third_provider_diagnostic, Mapping)
                or third_provider_diagnostic.get("accepted") is not False
                or type(third_provider_diagnostic.get("reason")) is not str
                or third_provider_diagnostic.get("reason")
                != "insufficient_camera_surface_provenance"
                or type(third_provider_diagnostic.get("field_id")) is not str
                or third_provider_diagnostic.get("field_id")
                != frozen_field_c_id
                or type(third_provider_diagnostic.get("field_label")) is not str
                or third_provider_diagnostic.get("field_label")
                != frozen_field_c_label
                or type(third_provider_diagnostic.get("camera_count")) is not int
                or third_provider_diagnostic.get("camera_count") != 1
                or not isinstance(
                    third_provider_diagnostic.get("per_camera_point_counts"),
                    Mapping,
                )
                or set(
                    third_provider_diagnostic["per_camera_point_counts"]
                )
                != {"agentview"}
                or any(
                    type(name) is not str
                    for name in third_provider_diagnostic[
                        "per_camera_point_counts"
                    ]
                )
                or type(
                    third_provider_diagnostic["per_camera_point_counts"].get(
                        "agentview"
                    )
                )
                is not int
                or third_surfaces is None
                or set(third_surfaces) != {"agentview"}
                or len(third_surfaces["agentview"])
                < self._FREE_RIM_VIEW_FIELD_MIN_POINTS_PER_CAMERA
                or third_provider_diagnostic["per_camera_point_counts"][
                    "agentview"
                ]
                != len(third_surfaces["agentview"])
            ):
                fail(
                    "frame_C_provider_did_not_preserve_the_typed_one_camera_case",
                    field_evidence_provider_diagnostic=(
                        dict(third_provider_diagnostic)
                        if isinstance(third_provider_diagnostic, Mapping)
                        else {}
                    ),
                )
            current_field = {
                "field_id": frozen_field_c_id,
                "field_label": frozen_field_c_label,
                "source_id": frozen_source_c_id,
                "source_label": frozen_source_c_label,
                "target_id": frozen_target_c_id,
                "target_label": frozen_target_c_label,
                "field_center_world_m": frozen_field_c_center.copy(),
                "field_half_extents_m": frozen_field_c_half.copy(),
                "field_axes_world": frozen_field_c_axes.copy(),
                "surface_points_by_camera": tuple(
                    (name, points.copy())
                    for name, points in frozen_field_c_surface_rows
                ),
                "raw_sdf_m": frozen_field_c_raw,
            }
        else:
            strict_current_field = self._strict_provider_mapping_for_authority(
                current_field,
                frame_c_authority,
            )
            if strict_current_field is None:
                fail(
                    "frame_C_provider_mapping_does_not_match_current_field",
                )
            current_field = strict_current_field
            current_field["raw_sdf_m"] = frozen_field_c_raw
        third_endpoint = {
            "field_center_world_m": current_field.get("field_center_world_m"),
            "field_half_extents_m": current_field.get("field_half_extents_m"),
            "field_axes_world": current_field.get("field_axes_world"),
            "surface_points_by_camera": current_field.get(
                "surface_points_by_camera"
            ),
            "ee_pose_world": actual,
            "gripper_width_m": third_snapshot["gripper_width_m"],
            "raw_sdf_m": frozen_field_c_raw,
            "timestamp_s": third_stamp,
            "capture_id": scene_c.capture_id,
            "camera_capture_ids": dict(scene_c.camera_capture_ids),
        }
        pair_bc = self._three_frame_motion_pair_evidence(
            second_endpoint, third_endpoint
        )
        pair_ac = self._three_frame_motion_pair_evidence(
            first_endpoint, third_endpoint
        )
        camera_bc = self._camera_snapshot_pair_evidence(
            second_snapshot, third_snapshot
        )
        camera_ac = self._camera_snapshot_pair_evidence(
            first_snapshot, third_snapshot
        )
        wrist_c = self._three_frame_wrist_endpoint_evidence(
            third_endpoint, third_snapshot
        )
        if not all(
            bool(item.get("accepted", False))
            for item in (pair_bc, pair_ac, camera_bc, camera_ac, wrist_c)
        ):
            fail(
                "frame_C_motion_camera_or_wrist_gate_failed",
                pair_B_C=pair_bc,
                pair_A_C=pair_ac,
                camera_B_C=camera_bc,
                camera_A_C=camera_ac,
                wrist_C=wrist_c,
            )
        if wrist_c.get("visibility_mode") != "strict_negative":
            fail("three_frame_chain_is_not_uniform_typed_wrist_negative")
        wrist_cameras = third_snapshot["cameras"]
        assert isinstance(wrist_cameras, Mapping)
        wrist_model = wrist_cameras["wrist"]
        assert isinstance(wrist_model, Mapping)
        ee_from_wrist = np.linalg.inv(actual) @ np.asarray(
            wrist_model["world_from_camera"], dtype=np.float64
        )
        updated_proof = dict(proof)
        updated_proof.update(
            {
                # Provider-side mutation of the live scene/binding cannot
                # alter the proof handed to the installer/application path.
                # These scalar identities come only from the authority frozen
                # immediately before the frame-C provider call.
                "field_id": frozen_field_c_id,
                "field_label": frozen_field_c_label,
                "source_id": frozen_source_c_id,
                "source_label": frozen_source_c_label,
                "target_id": frozen_target_c_id,
                "target_label": frozen_target_c_label,
                "final_ee_pose_world": actual.copy(),
                "end_gripper_width_m": third_snapshot["gripper_width_m"],
                "visibility_evidence": {
                    "mode": (
                        "three_frame_noncollinear_typed_wrist_negative"
                    ),
                    "endpoint_count": 3,
                    "motion_segment_count": 2,
                    "view_budget_consumed": True,
                    "wrist_camera_model": {
                        key: (
                            np.asarray(value, dtype=np.float64).copy()
                            if key in {"intrinsic", "world_from_camera"}
                            else value
                        )
                        for key, value in wrist_model.items()
                    },
                    "ee_from_wrist": ee_from_wrist.copy(),
                    "wrist_endpoint_modes": (
                        str(wrist_a["visibility_mode"]),
                        str(wrist_b["visibility_mode"]),
                        str(wrist_c["visibility_mode"]),
                    ),
                    "obb_uncertainty_inflation_m": (
                        self._FREE_RIM_VIEW_OBB_UNCERTAINTY_M
                    ),
                },
            }
        )
        self._append_robot_phase_trace(
            {
                "phase": "free_rim_three_frame_view_reacquisition",
                "active_grasp_candidate": candidate_id,
                "strategy": (
                    "three_fresh_frames_two_noncollinear_released_hand_motions_"
                    "with_typed_wrist_negative_visibility"
                ),
                "accepted": True,
                "candidate_failure_keys_consumed": False,
                "view_budget_consumed": True,
                "start_xyz_m": second_pose[:3, 3].tolist(),
                "goal_xyz_m": goal[:3, 3].tolist(),
                "final_xyz_m": actual[:3, 3].tolist(),
                "commanded_distance_m": commanded_distance,
                "actual_distance_m": refreshed_distance,
                "noncollinear_abs_cosine": refreshed_noncollinear_cosine,
                "cross_drift_m": refreshed_cross_drift,
                "vertical_drift_m": refreshed_vertical_drift,
                "rotation_drift_rad": refreshed_rotation_drift,
                "post_motion_pre_refresh_distance_m": actual_distance,
                "post_motion_pre_refresh_cross_drift_m": cross_drift,
                "post_motion_pre_refresh_vertical_drift_m": vertical_drift,
                "post_motion_pre_refresh_rotation_drift_rad": rotation_drift,
                "refresh_hold_position_drift_m": hold_position_drift,
                "refresh_hold_rotation_drift_rad": hold_rotation_drift,
                "motion_policy_actions": motion_steps,
                "total_policy_actions": steps_after_refresh - steps_before,
                "maximum_policy_actions": (
                    self._FREE_RIM_VIEW_REACQUIRE_MAX_POLICY_ACTIONS
                ),
                "maximum_waypoint_spacing_m": (
                    self._FREE_RIM_TARGET_EGRESS_WAYPOINT_SPACING_M
                ),
                "frozen_path_raw_sdf_m": path_distances,
                "measured_candidate_raw_sdf_m": candidate_measured_raw.tolist(),
                "measured_retained_raw_sdf_m": retained_measured_raw,
                "motion_sample_segment_lengths_m": (
                    measured_step_lengths.tolist()
                ),
                "post_motion_arc_segment_m": post_motion_arc_segment_m,
                "refresh_arc_segment_m": refresh_arc_segment_m,
                "measured_step_lengths_m": (
                    complete_path_segment_lengths.tolist()
                ),
                "measured_path_arc_length_m": complete_path_arc_length_m,
                "hard_path_arc_length_upper_bound_m": (
                    self._FREE_RIM_TARGET_EGRESS_MAX_DISTANCE_M
                ),
                "frame_timestamps_s": [first_stamp, second_stamp, third_stamp],
                "frame_capture_ids": [
                    first_capture_id,
                    scene_b.capture_id,
                    scene_c.capture_id,
                ],
                "frame_camera_capture_ids": [
                    dict(first_camera_capture_ids),
                    dict(scene_b.camera_capture_ids),
                    dict(scene_c.camera_capture_ids),
                ],
                "pair_A_B": pair_ab,
                "pair_B_C": pair_bc,
                "pair_A_C": pair_ac,
                "camera_A_B": camera_ab,
                "camera_B_C": camera_bc,
                "camera_A_C": camera_ac,
                "wrist_A": wrist_a,
                "wrist_B": wrist_b,
                "wrist_C": wrist_c,
                "next_step": "install_one_shot_exact_self_filter",
            }
        )
        return updated_proof, dict(current_field), actual.copy()

    def _cleanup_free_rim_typed_view_transaction(
        self,
        *,
        invalidate_cache: bool,
        reset_retry_active: bool,
    ) -> None:
        """Best-effort independent cleanup for a typed-view transaction.

        Each operation is isolated under ``BaseException`` so a broken
        observer cleanup hook cannot suppress the remaining cleanup or leave
        the controller in retry mode.
        """

        try:
            try:
                clear_filter = getattr(
                    self.observer, "clear_temporal_proprio_self_filter", None
                )
            except BaseException:
                clear_filter = None
            if callable(clear_filter):
                try:
                    clear_filter()
                except BaseException:
                    pass
            if invalidate_cache:
                try:
                    invalidate = getattr(
                        self.observer, "invalidate_sensor_cache", None
                    )
                except BaseException:
                    invalidate = None
                if callable(invalidate):
                    try:
                        invalidate()
                    except BaseException:
                        pass
        finally:
            # This assignment is deliberately in the outermost finally: even
            # observer attribute lookup is untrusted at this cleanup boundary.
            if reset_retry_active:
                self._free_rim_view_field_retry_active = False

    def _try_install_free_rim_view_self_filter(
        self,
        bound: BoundConstraintGraph,
        candidate_id: str,
        diagnostic: Mapping[str, object] | None,
        current_pose: np.ndarray | None,
    ) -> bool:
        """Run proof/install with a BaseException-safe typed-view boundary."""

        view_consumed_before = (
            candidate_id in self._free_rim_view_reacquired_candidates
        )
        try:
            return self._try_install_free_rim_view_self_filter_impl(
                bound, candidate_id, diagnostic, current_pose
            )
        except BaseException:
            if (
                not view_consumed_before
                and candidate_id in self._free_rim_view_reacquired_candidates
            ):
                self._cleanup_free_rim_typed_view_transaction(
                    invalidate_cache=True,
                    reset_retry_active=True,
                )
            raise

    def _try_install_free_rim_view_self_filter_impl(
        self,
        bound: BoundConstraintGraph,
        candidate_id: str,
        diagnostic: Mapping[str, object] | None,
        current_pose: np.ndarray | None,
    ) -> bool:
        """Prove that a recognised fused field moved rigidly with the tool.

        A single RGB-D box, its semantic label, or its frame-local ordinal id
        can never authorise obstacle removal.  This bridge is available only
        after a successful released-hand, constant-height target egress.  The
        same negative field must then translate with public EE proprioception
        over at least 50 mm while preserving its size and tool-relative
        offset.  A world-fixed cabinet cannot satisfy that two-frame motion
        test.  The observer receives a narrowly scoped filter for the immediate
        fresh replan; all other fields remain in the CompositeSDF.
        """

        proof = self._free_rim_last_assembly_egress_proof
        if self._free_rim_view_field_retry_active:
            return False
        if candidate_id in self._free_rim_view_reacquired_candidates:
            raise FreeRimViewEvidenceUnavailable(
                "free-rim self-field view evidence budget already consumed"
            )
        if (
            proof is None
            or candidate_id in self._free_rim_view_field_retried_candidates
        ):
            return False

        # Exactly the first post-egress failure for this physical candidate
        # may use the two-frame proof.  A rejected diagnostic can never be
        # combined with a later frame after the wrist has moved away/back.
        self._free_rim_last_assembly_egress_proof = None
        nonphysical_view_mode = False

        def reject(reason: str, **details: object) -> bool:
            self._cleanup_free_rim_typed_view_transaction(
                invalidate_cache=nonphysical_view_mode,
                reset_retry_active=nonphysical_view_mode,
            )
            self._append_robot_phase_trace(
                {
                    "phase": "free_rim_view_dependent_field_rejection",
                    "active_grasp_candidate": candidate_id,
                    "strategy": (
                        "three_frame_noncollinear_typed_wrist_negative"
                        if nonphysical_view_mode
                        else "two_frame_released_hand_public_proprio_point_hypothesis"
                    ),
                    "accepted": False,
                    "proof_state": "REJECT",
                    "reason": reason,
                    "candidate_failure_keys_consumed": False,
                    **details,
                }
            )
            if nonphysical_view_mode:
                raise FreeRimViewEvidenceUnavailable(
                    "free-rim non-physical field proof failed closed after "
                    f"view reacquisition: {reason}"
                )
            return False

        if current_pose is None:
            return reject("post_egress_public_ee_pose_unavailable")
        if type(proof.get("candidate_id")) is not str or proof.get(
            "candidate_id"
        ) != candidate_id:
            return reject("post_egress_candidate_identity_mismatch")

        if (
            type(proof.get("source_id")) is not str
            or type(proof.get("target_id")) is not str
            or type(getattr(bound, "source_id", None)) is not str
            or type(getattr(bound, "target_id", None)) is not str
            or proof.get("source_id") != getattr(bound, "source_id", None)
            or proof.get("target_id") != getattr(bound, "target_id", None)
            or proof.get("field_binding") != "unique_stable_assembly_field"
        ):
            return reject("source_target_or_field_binding_mismatch")
        if diagnostic is None or not self._is_target_container_diagnostic(
            bound, diagnostic
        ):
            return reject("not_unique_target_container_start_diagnostic")

        field_id = diagnostic.get("field_id")
        field_label = diagnostic.get("field_label")
        diagnostic_center = self._strict_provider_float_array(
            diagnostic.get("field_center_world_m"), (3,)
        )
        diagnostic_half = self._strict_provider_float_array(
            diagnostic.get("field_half_extents_m"), (3,)
        )
        if (
            type(field_id) is not str
            or type(field_label) is not str
            or not field_id.startswith("fused-")
            or type(proof.get("field_id")) is not str
            or type(proof.get("field_label")) is not str
            or field_id != proof.get("field_id")
            or field_label != proof.get("field_label")
            or diagnostic_center is None
            or diagnostic_half is None
        ):
            return reject(
                "field_identity_or_label_mismatch",
                diagnostic_field_id=field_id,
                diagnostic_field_label=field_label,
            )
        provider_scene = self._current_sensor_scene()
        if provider_scene is None:
            return reject("fresh_field_provider_scene_unavailable")
        provider_field = self._unique_three_frame_field(
            provider_scene,
            proof,
            expected_center_world_m=diagnostic_center,
        )
        if provider_field is None:
            return reject("fresh_field_provider_authority_is_not_unique")
        provider_authority = self._freeze_field_provider_authority(
            scene=provider_scene,
            field=provider_field,
            bound=bound,
            proof=proof,
        )
        if (
            provider_authority is None
            or provider_authority.get("field_id") != field_id
            or provider_authority.get("field_label") != field_label
        ):
            return reject("fresh_field_provider_authority_is_inconsistent")
        fresh_evidence = getattr(
            self.observer, "temporal_proprio_field_evidence", None
        )
        if not callable(fresh_evidence):
            return reject("fresh_field_evidence_provider_unavailable")
        try:
            current_field = fresh_evidence(
                field_id=field_id,
                field_label=field_label,
                field_center_world_m=np.asarray(
                    provider_authority["field_center_world_m"]
                ).copy(),
                field_half_extents_m=np.asarray(
                    provider_authority["field_half_extents_m"]
                ).copy(),
                field_axes_world=np.asarray(
                    provider_authority["field_axes_world"]
                ).copy(),
                surface_points_by_camera=tuple(
                    (name, points.copy())
                    for name, points in provider_authority[
                        "surface_points_by_camera"
                    ]
                ),
                source_id=provider_authority["source_id"],
                source_label=provider_authority["source_label"],
                target_id=provider_authority["target_id"],
                target_label=provider_authority["target_label"],
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            return reject(
                "fresh_field_evidence_error", error_type=type(exc).__name__
            )
        if not isinstance(current_field, Mapping):
            provider_diagnostic = getattr(
                self.observer,
                "last_temporal_proprio_field_evidence_diagnostic",
                None,
            )
            if (
                isinstance(provider_diagnostic, Mapping)
                and provider_diagnostic.get("accepted") is False
                and provider_diagnostic.get("reason")
                == "insufficient_camera_surface_provenance"
            ):
                self._append_robot_phase_trace(
                    {
                        "phase": "free_rim_view_dependent_field_rejection",
                        "active_grasp_candidate": candidate_id,
                        "strategy": (
                            "three_frame_noncollinear_typed_wrist_negative"
                        ),
                        "accepted": False,
                        "proof_state": "NEEDS_VIEW",
                        "reason": "unique_agentview_only_provider_evidence",
                        "candidate_failure_keys_consumed": False,
                    }
                )
                try:
                    proof, current_field, current_pose = (
                        self._reacquire_three_frame_self_field_evidence(
                            bound,
                            candidate_id,
                            diagnostic,
                            provider_diagnostic,
                            current_pose,
                            proof,
                            fresh_evidence,
                        )
                    )
                except BaseException:
                    # The view helper itself cleans every expected closed-gate
                    # path.  This outer finally-equivalent also covers an
                    # unexpected adapter/runtime exception after motion.
                    self._cleanup_free_rim_typed_view_transaction(
                        invalidate_cache=True,
                        reset_retry_active=True,
                    )
                    raise
                nonphysical_view_mode = True
            else:
                return reject(
                    "fresh_field_evidence_rejected",
                    field_evidence_provider_diagnostic=(
                        dict(provider_diagnostic)
                        if isinstance(provider_diagnostic, Mapping)
                        else {}
                    ),
                )
        else:
            strict_current_field = self._strict_provider_mapping_for_authority(
                current_field,
                provider_authority,
            )
            if strict_current_field is None:
                return reject(
                    "fresh_field_provider_mapping_does_not_match_frozen_authority"
                )
            current_field = strict_current_field
        install_identity = {
            key: current_field.get(key)
            for key in (
                "field_id",
                "field_label",
                "source_id",
                "source_label",
                "target_id",
                "target_label",
            )
        }
        if any(
            type(value) is not str or not value
            for value in install_identity.values()
        ):
            return reject("frozen_install_identity_is_malformed")
        field_id = install_identity["field_id"]
        field_label = install_identity["field_label"]
        try:
            prior_center = np.asarray(
                proof["field_center_world_m"], dtype=np.float64
            )
            current_center = np.asarray(
                current_field["field_center_world_m"], dtype=np.float64
            )
            prior_half = np.asarray(
                proof["field_half_extents_m"], dtype=np.float64
            )
            current_half = np.asarray(
                current_field["field_half_extents_m"], dtype=np.float64
            )
            prior_axes = np.asarray(proof["field_axes_world"], dtype=np.float64)
            current_axes = np.asarray(
                current_field["field_axes_world"], dtype=np.float64
            )
            start_pose = np.asarray(
                proof["start_ee_pose_world"], dtype=np.float64
            )
            final_pose = np.asarray(
                proof["final_ee_pose_world"], dtype=np.float64
            )
            current_matrix = np.asarray(current_pose, dtype=np.float64)
            start_ee = start_pose[:3, 3]
            final_ee = final_pose[:3, 3]
            prior_raw = float(proof["initial_raw_sdf_m"])
            current_raw = float(
                current_field.get(
                    "raw_sdf_m",
                    diagnostic["raw_distance_m"],
                )
            )
            start_open_width = float(proof["start_gripper_width_m"])
            end_open_width = float(proof["end_gripper_width_m"])
        except (KeyError, TypeError, ValueError) as exc:
            return reject("invalid_motion_proof", error_type=type(exc).__name__)
        vectors = (
            prior_center,
            current_center,
            prior_half,
            current_half,
            prior_axes,
            current_axes,
            start_ee,
            final_ee,
        )
        if (
            any(value.shape != (3,) for value in vectors[:4])
            or any(value.shape != (3, 3) for value in vectors[4:6])
            or any(value.shape != (3,) for value in vectors[6:])
            or start_pose.shape != (4, 4)
            or final_pose.shape != (4, 4)
            or current_matrix.shape != (4, 4)
            or not all(np.all(np.isfinite(value)) for value in vectors)
            or not np.all(np.isfinite(start_pose))
            or not np.all(np.isfinite(final_pose))
            or not np.all(np.isfinite(current_matrix))
            or any(
                not np.allclose(axes.T @ axes, np.eye(3), atol=2e-3)
                or float(np.linalg.det(axes)) <= 0.0
                for axes in (prior_axes, current_axes)
            )
            or any(
                not np.allclose(
                    pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6
                )
                or not np.allclose(
                    pose[:3, :3].T @ pose[:3, :3],
                    np.eye(3),
                    atol=2e-3,
                )
                or float(np.linalg.det(pose[:3, :3])) <= 0.0
                for pose in (start_pose, final_pose, current_matrix)
            )
        ):
            return reject("nonfinite_or_malformed_motion_proof")
        current_ee = current_matrix[:3, 3]
        ee_motion = current_ee - start_ee
        field_motion = current_center - prior_center
        ee_distance = float(np.linalg.norm(ee_motion))
        field_distance = float(np.linalg.norm(field_motion))
        comotion_residual = float(np.linalg.norm(field_motion - ee_motion))
        if ee_distance <= 0.0 or field_distance <= 0.0:
            return reject("nonpositive_ee_or_field_motion")
        direction_cosine = float(
            np.dot(field_motion, ee_motion) / (field_distance * ee_distance)
        )
        motion_ratio = field_distance / ee_distance
        endpoint_error = float(np.linalg.norm(current_ee - final_ee))
        half_extent_drift = float(np.max(np.abs(current_half - prior_half)))
        axes_drift = float(
            Rotation.from_matrix(current_axes @ prior_axes.T).magnitude()
        )
        ee_rotation = float(
            Rotation.from_matrix(
                current_matrix[:3, :3] @ start_pose[:3, :3].T
            ).magnitude()
        )
        endpoint_rotation_error = float(
            Rotation.from_matrix(
                current_matrix[:3, :3] @ final_pose[:3, :3].T
            ).magnitude()
        )
        relative_offset_drift = float(
            np.linalg.norm(
                (current_center - current_ee) - (prior_center - start_ee)
            )
        )
        width_provider = getattr(self.robot, "current_gripper_width_m", None)
        if not callable(width_provider):
            return reject("public_gripper_width_unavailable")
        try:
            raw_open_width = width_provider()
            open_width = (
                self._builtin_finite_number(raw_open_width)
                if nonphysical_view_mode
                else float(raw_open_width)
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            return reject(
                "public_gripper_width_error", error_type=type(exc).__name__
            )
        if open_width is None:
            return reject("public_gripper_width_is_not_a_builtin_finite_number")
        visibility_evidence = proof.get("visibility_evidence")
        typed_negative_mode = bool(
            nonphysical_view_mode
            and isinstance(visibility_evidence, Mapping)
            and visibility_evidence.get("mode")
            == "three_frame_noncollinear_typed_wrist_negative"
        )
        surface_parser = (
            self._canonical_surface_map
            if typed_negative_mode
            else self._camera_surface_map
        )
        prior_surfaces = surface_parser(proof.get("surface_points_by_camera"))
        current_surfaces = surface_parser(
            current_field.get("surface_points_by_camera")
        )
        positive_surface_gate = bool(
            prior_surfaces is not None
            and current_surfaces is not None
            and set(prior_surfaces) == set(current_surfaces)
            and (
                set(prior_surfaces) == {"agentview"}
                if typed_negative_mode
                else len(prior_surfaces)
                >= self._FREE_RIM_VIEW_FIELD_MIN_CAMERAS
            )
            and all(
                len(points) >= self._FREE_RIM_VIEW_FIELD_MIN_POINTS_PER_CAMERA
                for points in (
                    *prior_surfaces.values(),
                    *current_surfaces.values(),
                )
            )
        )
        if not positive_surface_gate:
            return reject(
                "insufficient_same_camera_surface_evidence",
                prior_cameras=(
                    sorted(prior_surfaces) if prior_surfaces is not None else []
                ),
                current_cameras=(
                    sorted(current_surfaces)
                    if current_surfaces is not None
                    else []
                ),
                prior_point_counts=(
                    {name: len(points) for name, points in prior_surfaces.items()}
                    if prior_surfaces is not None
                    else {}
                ),
                current_point_counts=(
                    {
                        name: len(points)
                        for name, points in current_surfaces.items()
                    }
                    if current_surfaces is not None
                    else {}
                ),
            )
        tool_volume_evidence: dict[str, dict[str, object]] = {}
        for camera_name in sorted(prior_surfaces):
            morphology = (
                self._typed_negative_panda_tool_surface_evidence
                if typed_negative_mode
                else self._released_panda_tool_surface_evidence
            )
            prior_tool = morphology(
                prior_surfaces[camera_name], start_pose
            )
            current_tool = morphology(
                current_surfaces[camera_name], current_matrix
            )
            tool_volume_evidence[camera_name] = {
                "prior": prior_tool,
                "current": current_tool,
            }
        if not all(
            bool(endpoint["accepted"])
            for camera in tool_volume_evidence.values()
            for endpoint in (camera["prior"], camera["current"])
        ):
            return reject(
                "raw_surface_not_bound_to_public_panda_tool_volume",
                per_camera_public_tool_evidence=tool_volume_evidence,
            )
        try:
            tool_transform = current_matrix @ np.linalg.inv(start_pose)
        except np.linalg.LinAlgError:
            return reject("singular_public_ee_pose")
        identity = np.eye(4, dtype=np.float64)
        per_camera_errors: dict[str, tuple[float, float]] = {}
        for camera_name in sorted(prior_surfaces):
            alignment = (
                self._strict_surface_alignment_error_m
                if typed_negative_mode
                else self._surface_alignment_error_m
            )
            tool_error = alignment(
                prior_surfaces[camera_name],
                current_surfaces[camera_name],
                tool_transform,
            )
            static_error = alignment(
                prior_surfaces[camera_name],
                current_surfaces[camera_name],
                identity,
            )
            per_camera_errors[camera_name] = (tool_error, static_error)
        point_hypothesis_ok = all(
            np.all(np.isfinite(errors))
            and errors[0] <= self._FREE_RIM_VIEW_FIELD_MAX_POINT_ERROR_M
            and errors[0] + self._FREE_RIM_VIEW_FIELD_MIN_STATIC_ADVANTAGE_M
            <= errors[1]
            and errors[0]
            <= self._FREE_RIM_VIEW_FIELD_MAX_STATIC_ERROR_RATIO * errors[1]
            for errors in per_camera_errors.values()
        )
        if not (
            np.all(
                np.isfinite(
                    (
                        prior_raw,
                        current_raw,
                        ee_distance,
                        field_distance,
                        comotion_residual,
                        direction_cosine,
                        motion_ratio,
                        endpoint_error,
                        half_extent_drift,
                        relative_offset_drift,
                        open_width,
                        start_open_width,
                        end_open_width,
                        axes_drift,
                        ee_rotation,
                        endpoint_rotation_error,
                    )
                )
            )
            and prior_raw < 0.0
            and current_raw < 0.0
            and ee_distance >= self._FREE_RIM_VIEW_FIELD_MIN_EE_MOTION_M
            and field_distance >= self._FREE_RIM_VIEW_FIELD_MIN_FIELD_MOTION_M
            and comotion_residual
            <= self._FREE_RIM_VIEW_FIELD_MAX_COMOTION_RESIDUAL_M
            and relative_offset_drift
            <= self._FREE_RIM_VIEW_FIELD_MAX_COMOTION_RESIDUAL_M
            and direction_cosine
            >= self._FREE_RIM_VIEW_FIELD_MIN_DIRECTION_COSINE
            and self._FREE_RIM_VIEW_FIELD_MIN_MOTION_RATIO
            <= motion_ratio
            <= self._FREE_RIM_VIEW_FIELD_MAX_MOTION_RATIO
            and endpoint_error
            <= self._FREE_RIM_TARGET_EGRESS_SAFE_POSE_POSITION_TOLERANCE_M
            and half_extent_drift
            <= self._FREE_RIM_VIEW_FIELD_MAX_HALF_EXTENT_DRIFT_M
            and open_width >= self._FREE_RIM_VIEW_FIELD_MIN_OPEN_WIDTH_M
            and start_open_width >= self._FREE_RIM_VIEW_FIELD_MIN_OPEN_WIDTH_M
            and end_open_width >= self._FREE_RIM_VIEW_FIELD_MIN_OPEN_WIDTH_M
            and ee_rotation
            <= self._FREE_RIM_TARGET_EGRESS_SAFE_POSE_ROTATION_TOLERANCE_RAD
            and endpoint_rotation_error
            <= self._FREE_RIM_TARGET_EGRESS_SAFE_POSE_ROTATION_TOLERANCE_RAD
            and axes_drift <= 0.020
            and point_hypothesis_ok
        ):
            return reject(
                "motion_or_point_hypothesis_gate_failed",
                ee_motion_norm_m=ee_distance,
                field_motion_norm_m=field_distance,
                comotion_residual_m=comotion_residual,
                relative_offset_drift_m=relative_offset_drift,
                direction_cosine=direction_cosine,
                motion_ratio=motion_ratio,
                endpoint_error_m=endpoint_error,
                half_extent_drift_m=half_extent_drift,
                gripper_width_m=open_width,
                start_gripper_width_m=start_open_width,
                egress_end_gripper_width_m=end_open_width,
                ee_rotation_drift_rad=ee_rotation,
                endpoint_rotation_error_rad=endpoint_rotation_error,
                field_axes_drift_rad=axes_drift,
                per_camera_public_tool_evidence=tool_volume_evidence,
                per_camera_point_hypothesis={
                    name: {
                        "tool_motion_error_m": errors[0],
                        "world_static_error_m": errors[1],
                    }
                    for name, errors in per_camera_errors.items()
                },
            )

        install = getattr(
            self.observer, "install_temporal_proprio_self_filter", None
        )
        if not callable(install):
            return reject("self_filter_installer_unavailable")

        # Installation is allowed only when its audit sink is an exact
        # built-in list.  A list subclass may append and then throw while also
        # overriding deletion, which makes an INSTALLED record impossible to
        # roll back reliably after the filter has acquired side effects.
        try:
            phase_trace = getattr(self.robot, "phase_trace", None)
        except BaseException as exc:
            raise ExecutionError(
                "self-filter install trace capability is unavailable"
            ) from exc
        if type(phase_trace) is not list:
            raise ExecutionError(
                "self-filter install trace requires an exact built-in list"
            )
        trace_transaction_start = list.__len__(phase_trace)

        def rollback_possible_install() -> None:
            """Best-effort rollback that never masks the triggering failure."""

            try:
                retry_ledger = self._free_rim_view_field_retried_candidates
                if isinstance(retry_ledger, set):
                    set.discard(retry_ledger, candidate_id)
                else:
                    retry_ledger.discard(candidate_id)
            except BaseException:
                pass
            try:
                if list.__len__(phase_trace) > trace_transaction_start:
                    list.__delitem__(
                        phase_trace,
                        slice(trace_transaction_start, None),
                    )
            except BaseException:
                pass
            try:
                self._cleanup_free_rim_typed_view_transaction(
                    invalidate_cache=True,
                    reset_retry_active=True,
                )
            except BaseException:
                pass

        try:
            install_result = install(
                    field_id=field_id,
                    field_label=field_label,
                    field_center_world_m=current_center,
                    field_half_extents_m=current_half,
                    ee_world_m=current_ee,
                    maximum_relative_error_m=(
                        self._FREE_RIM_VIEW_FIELD_MAX_COMOTION_RESIDUAL_M
                    ),
                    field_axes_world=current_axes,
                    surface_points_by_camera=tuple(
                        (name, points.copy())
                        for name, points in current_surfaces.items()
                    ),
                    ee_pose_world=current_matrix,
                    source_id=install_identity["source_id"],
                    source_label=install_identity["source_label"],
                    target_id=install_identity["target_id"],
                    target_label=install_identity["target_label"],
                    visibility_evidence=(
                        visibility_evidence if typed_negative_mode else None
                    ),
            )
        except BaseException:
            rollback_possible_install()
            raise
        if typed_negative_mode and type(install_result) is not bool:
            rollback_possible_install()
            return reject("self_filter_install_result_is_not_a_builtin_bool")
        try:
            installed = bool(install_result)
        except BaseException:
            rollback_possible_install()
            raise
        if not installed:
            rollback_possible_install()
            return reject("self_filter_install_rejected")

        try:
            invalidate = getattr(
                self.observer, "invalidate_sensor_cache", None
            )
        except BaseException as exc:
            rollback_possible_install()
            return reject(
                "post_install_sensor_cache_invalidation_error",
                error_type=type(exc).__name__,
            )
        if typed_negative_mode and not callable(invalidate):
            rollback_possible_install()
            return reject("post_install_sensor_cache_invalidation_unavailable")
        if callable(invalidate):
            try:
                invalidate()
            except BaseException as exc:
                rollback_possible_install()
                return reject(
                    "post_install_sensor_cache_invalidation_error",
                    error_type=type(exc).__name__,
                )

        try:
            current_phase_trace = getattr(self.robot, "phase_trace", None)
        except BaseException:
            rollback_possible_install()
            raise
        if (
            type(current_phase_trace) is not list
            or current_phase_trace is not phase_trace
        ):
            rollback_possible_install()
            raise ExecutionError(
                "self-filter install trace capability changed during transaction"
            )

        # Commit only after installation and every mandatory post-install
        # operation succeeded.  Append with the frozen built-in capability
        # before mutating the retry ledger and roll back both on every
        # BaseException.
        install_trace_record = {
                "phase": "free_rim_view_dependent_field_rejection",
                "active_grasp_candidate": candidate_id,
                "strategy": (
                    "three_frame_noncollinear_typed_wrist_negative"
                    if typed_negative_mode
                    else "two_frame_released_hand_public_proprio_comotion"
                ),
                "proof_state": "INSTALLED",
                "field_id": field_id,
                "field_label": field_label,
                "prior_field_center_world_m": prior_center.tolist(),
                "current_field_center_world_m": current_center.tolist(),
                "prior_ee_world_m": start_ee.tolist(),
                "current_ee_world_m": current_ee.tolist(),
                "ee_motion_m": ee_motion.tolist(),
                "field_motion_m": field_motion.tolist(),
                "ee_motion_norm_m": ee_distance,
                "field_motion_norm_m": field_distance,
                "comotion_residual_m": comotion_residual,
                "relative_offset_drift_m": relative_offset_drift,
                "direction_cosine": direction_cosine,
                "motion_ratio": motion_ratio,
                "half_extent_drift_m": half_extent_drift,
                "gripper_width_m": open_width,
                "start_gripper_width_m": start_open_width,
                "egress_end_gripper_width_m": end_open_width,
                "ee_rotation_drift_rad": ee_rotation,
                "endpoint_rotation_error_rad": endpoint_rotation_error,
                "field_axes_drift_rad": axes_drift,
                "per_camera_public_tool_evidence": tool_volume_evidence,
                "per_camera_point_hypothesis": {
                    name: {
                        "tool_motion_error_m": errors[0],
                        "world_static_error_m": errors[1],
                    }
                    for name, errors in per_camera_errors.items()
                },
                "candidate_failure_keys_consumed": False,
                "accepted": True,
                "next_step": "fresh_same_phase_replan_with_exact_self_filter",
        }
        try:
            list.append(phase_trace, dict(install_trace_record))
            self._free_rim_view_field_retried_candidates.add(candidate_id)
        except BaseException:
            rollback_possible_install()
            raise
        return True

    def _try_free_rim_target_container_egress(self) -> bool | None:
        """Exit a recognised target-container self field from a cached high pose."""

        if not self._free_rim_target_egress_pending:
            return None
        self._free_rim_target_egress_pending = False
        candidate_id = self._free_rim_target_egress_candidate_id
        diagnostic = self._free_rim_target_egress_diagnostic
        labels = self._free_rim_target_egress_labels
        invalidate = getattr(self.observer, "invalidate_sensor_cache", None)

        def reject(detail: str, **extra: object) -> bool:
            self._mark_free_space_environment_blocked(candidate_id)
            self._cavity_failed_candidate_id = candidate_id
            self._cavity_recovery_pending = False
            self._append_robot_phase_trace(
                {
                    "phase": "free_rim_target_container_high_egress",
                    "active_grasp_candidate": candidate_id,
                    "accepted": False,
                    "detail": detail,
                    "osc_steps": 0,
                    "gripper_command": self._released_gripper_command(
                        GraspMode.RIM_PINCH
                    ),
                    **extra,
                }
            )
            if callable(invalidate):
                invalidate()
            return False

        if diagnostic is None or self._cavity_approach_safe_pose is None:
            return reject("missing frozen diagnostic or cached high safe pose")
        if callable(invalidate):
            invalidate()
        try:
            scene = self.observer.observe(labels)
        except (AttributeError, TypeError, ValueError, PerceptionError) as exc:
            return reject(f"fresh target-container observation failed: {exc}")
        start = self._public_ee_pose(self.robot)
        if start is None:
            return reject("public EE pose is unavailable or non-finite")
        start_view_snapshot = self._public_view_snapshot(self.robot)
        if start_view_snapshot is not None:
            snapshot_pose = np.asarray(
                start_view_snapshot["ee_pose_world"], dtype=np.float64
            )
            snapshot_position_error, snapshot_rotation_error = (
                self._recovery_pose_errors(snapshot_pose, start)
            )
            if (
                snapshot_position_error
                > self._FREE_RIM_VIEW_CAMERA_RIGID_POSITION_TOLERANCE_M
                or snapshot_rotation_error
                > self._FREE_RIM_VIEW_CAMERA_RIGID_ROTATION_TOLERANCE_RAD
            ):
                start_view_snapshot = None
        safe_position_error, safe_rotation_error = self._recovery_pose_errors(
            start, self._cavity_approach_safe_pose
        )
        if (
            safe_position_error
            > self._FREE_RIM_TARGET_EGRESS_SAFE_POSE_POSITION_TOLERANCE_M
            or safe_rotation_error
            > self._FREE_RIM_TARGET_EGRESS_SAFE_POSE_ROTATION_TOLERANCE_RAD
        ):
            return reject(
                "public EE no longer equals the cached high safe pose",
                safe_pose_position_error_m=safe_position_error,
                safe_pose_rotation_error_rad=safe_rotation_error,
            )
        failed_point = np.asarray(diagnostic["point_world_m"], dtype=np.float64)
        if np.linalg.norm(start[:3, 3] - failed_point) > 0.0035:
            return reject(
                "fresh public EE does not match the failed start sample",
                failed_start_point_world_m=failed_point.tolist(),
                public_ee_xyz_m=start[:3, 3].tolist(),
            )
        binding = self._fresh_target_container_field(
            scene, start, diagnostic
        )
        if binding is None:
            return reject(
                "fresh SDF field has no unique current semantic target-container binding"
            )
        source, target, field, field_binding = binding
        source_top_z = float(source.position[2] + 0.5 * source.extent[2])
        minimum_escape_z = (
            source_top_z + self._FREE_RIM_ESCAPE_MIN_HEIGHT_ABOVE_SOURCE_M
        )
        if float(start[2, 3]) < minimum_escape_z:
            return reject(
                "public EE is below the sensor-derived high target-container egress",
                source_top_z_m=source_top_z,
                minimum_escape_z_m=minimum_escape_z,
            )
        start_raw_sdf = float(scene.obstacle_sdf.distance(start[:3, 3]))
        nominal_envelope = float(diagnostic["nominal_envelope_m"])
        if not np.isfinite(start_raw_sdf) or start_raw_sdf >= 0.0:
            return reject(
                "fresh SDF no longer confirms a negative-raw start",
                initial_raw_sdf_m=start_raw_sdf,
            )
        corridor = self._free_rim_target_egress_corridor(
            scene, start, source, nominal_envelope
        )
        if corridor is None:
            return reject(
                "no constant-height path reaches positive full inflated envelope within hard upper bound",
                initial_raw_sdf_m=start_raw_sdf,
                nominal_inflated_envelope_m=nominal_envelope,
                hard_distance_upper_bound_m=(
                    self._FREE_RIM_TARGET_EGRESS_MAX_DISTANCE_M
                ),
            )
        goal, dense_raw, corridors_evaluated = corridor
        delta_xy = goal[:2, 3] - start[:2, 3]
        command_distance = float(np.linalg.norm(delta_xy))
        direction = delta_xy / command_distance
        waypoint_segments = max(
            1,
            int(
                np.ceil(
                    command_distance
                    / self._FREE_RIM_TARGET_EGRESS_WAYPOINT_SPACING_M
                )
            ),
        )
        fractions = np.linspace(0.0, 1.0, waypoint_segments + 1)
        waypoints = np.repeat(start[None, :, :], len(fractions), axis=0)
        waypoints[:, :2, 3] += fractions[:, None] * delta_xy[None, :]
        waypoint_raw = np.asarray(
            scene.obstacle_sdf.distance(waypoints[:, :3, 3]),
            dtype=np.float64,
        )
        candidate_waypoint_raw = np.asarray(
            field.distance(waypoints[:, :3, 3]), dtype=np.float64
        )
        retained_waypoint_raw: dict[str, list[float]] = {}
        retained_waypoints_safe = True
        for index, retained in enumerate(scene.obstacle_sdf.fields):
            if retained is field:
                continue
            values = np.asarray(
                retained.distance(waypoints[:, :3, 3]), dtype=np.float64
            )
            retained_waypoint_raw[
                str(
                    getattr(retained, "source_instance_id", None)
                    or f"sensor-field-{index}"
                )
            ] = values.tolist()
            retained_waypoints_safe = bool(
                retained_waypoints_safe
                and values.shape == (len(waypoints),)
                and np.all(np.isfinite(values))
                and float(np.min(values)) >= nominal_envelope
            )
        if (
            waypoint_raw.shape != (len(waypoints),)
            or not np.all(np.isfinite(waypoint_raw))
            or np.any(np.diff(waypoint_raw) <= 1e-8)
            or waypoint_raw[-1] < nominal_envelope
            or candidate_waypoint_raw.shape != (len(waypoints),)
            or not np.all(np.isfinite(candidate_waypoint_raw))
            or np.any(np.diff(candidate_waypoint_raw) <= 1e-8)
            or candidate_waypoint_raw[-1] < nominal_envelope
            or not retained_waypoints_safe
        ):
            return reject(
                "command waypoints fail candidate-improvement or retained-field envelope gates",
                commanded_raw_sdf_m=waypoint_raw.tolist(),
                candidate_commanded_raw_sdf_m=candidate_waypoint_raw.tolist(),
                retained_commanded_raw_sdf_m=retained_waypoint_raw,
            )

        width_provider = getattr(self.robot, "current_gripper_width_m", None)
        if not callable(width_provider):
            return reject("released-hand public width is unavailable")
        try:
            start_open_width = self._builtin_finite_number(width_provider())
        except (TypeError, ValueError, RuntimeError):
            return reject("released-hand public width is unavailable")
        if (
            start_open_width is None
            or start_open_width < self._FREE_RIM_VIEW_FIELD_MIN_OPEN_WIDTH_M
        ):
            return reject(
                "target-container egress start is not publicly open",
                start_gripper_width_m=start_open_width,
            )
        command = self._released_gripper_command(GraspMode.RIM_PINCH)
        if command != -1.0:
            return reject("target-container egress lacks the released command")
        steps_before = getattr(self.robot, "steps_executed", None)
        if type(steps_before) is not int or steps_before < 0:
            return reject("target-container egress start step is not a builtin int")
        step_budget = getattr(self.robot, "step_budget", None)
        if (
            type(step_budget) is not int
            or step_budget - steps_before
            < self._FREE_RIM_VIEW_REACQUIRE_MAX_POLICY_ACTIONS
        ):
            return reject(
                "target-container complete egress and refresh budget was not pre-reserved"
            )
        execute_safe = getattr(
            self.robot, "execute_sensor_safe_view_waypoints", None
        )
        if not callable(execute_safe):
            return reject(
                "target-container egress lacks per-action public-proprio execution"
            )
        try:
            feedback = execute_safe(
                waypoints,
                command,
                maximum_policy_actions=(
                    self._FREE_RIM_VIEW_REACQUIRE_MAX_POLICY_ACTIONS - 1
                ),
                reserved_followup_actions=1,
            )
        except PolicyStepBudgetExhausted:
            raise
        except (TypeError, ValueError, RuntimeError) as exc:
            return reject(
                "target-container sensor-safe egress execution failed",
                error_type=type(exc).__name__,
            )
        if (
            not isinstance(feedback, ControllerFeedback)
            or type(feedback.accepted) is not bool
        ):
            return reject(
                "target-container egress feedback is not typed public evidence"
            )
        steps_after = getattr(self.robot, "steps_executed", None)
        if type(steps_after) is not int or steps_after < steps_before:
            return reject("target-container egress end step is not a builtin int")
        osc_steps = steps_after - steps_before
        step_budget_exhausted = getattr(
            self.robot, "step_budget_exhausted", False
        )
        if type(step_budget_exhausted) is not bool:
            return reject("target-container egress budget flag is not a builtin bool")
        if step_budget_exhausted:
            reject(
                "episode policy-action budget exhausted during target-container egress",
                osc_steps=osc_steps,
                step_budget_exhausted=True,
            )
            raise PolicyStepBudgetExhausted(
                "episode OSC step budget exhausted during target-container egress"
            )
        if osc_steps == 0:
            return reject(
                "accepted target-container egress reported zero policy actions",
                osc_steps=0,
                step_budget_exhausted=False,
            )
        if osc_steps > self._FREE_RIM_VIEW_REACQUIRE_MAX_POLICY_ACTIONS - 1:
            return reject(
                "target-container egress exceeded its native policy-action cap",
                osc_steps=osc_steps,
                maximum_policy_actions=(
                    self._FREE_RIM_VIEW_REACQUIRE_MAX_POLICY_ACTIONS - 1
                ),
                step_budget_exhausted=False,
            )
        sample_value = getattr(
            self.robot, "last_sensor_safe_view_motion_samples", None
        )
        if callable(sample_value):
            try:
                sample_value = sample_value()
            except (TypeError, ValueError, RuntimeError) as exc:
                return reject(
                    "target-container public motion samples failed",
                    error_type=type(exc).__name__,
                )
        sample_evidence = self._sensor_safe_path_sample_evidence(
            scene=scene,
            candidate_field=field,
            start_pose_world=start,
            goal_pose_world=goal,
            samples=sample_value,
            steps_before=steps_before,
            steps_after=steps_after,
            nominal_envelope_m=nominal_envelope,
        )
        if not feedback.accepted or sample_evidence.get("accepted") is not True:
            return reject(
                "target-container measured path failed per-action sensor gates",
                feedback_accepted=feedback.accepted,
                public_motion_sample_evidence={
                    key: value
                    for key, value in sample_evidence.items()
                    if key != "last_pose_world"
                },
            )
        actual = self._public_ee_pose(self.robot)
        if actual is None:
            return reject(
                "post-egress public EE pose is unavailable",
                osc_steps=osc_steps,
            )
        actual_delta_xy = actual[:2, 3] - start[:2, 3]
        signed_progress = float(np.dot(actual_delta_xy, direction))
        cross_drift = float(
            np.linalg.norm(actual_delta_xy - signed_progress * direction)
        )
        vertical_drift = abs(float(actual[2, 3] - start[2, 3]))
        actual_distance = float(np.linalg.norm(actual_delta_xy))
        actual_raw_sdf = float(scene.obstacle_sdf.distance(actual[:3, 3]))
        _, actual_rotation_drift = self._recovery_pose_errors(actual, start)
        last_sample_pose = np.asarray(
            sample_evidence["last_pose_world"], dtype=np.float64
        )
        sample_endpoint_position_error, sample_endpoint_rotation_error = (
            self._recovery_pose_errors(actual, last_sample_pose)
        )
        try:
            end_open_width = self._builtin_finite_number(width_provider())
        except (TypeError, ValueError, RuntimeError):
            end_open_width = None
        workspace_min = np.asarray(scene.workspace_min, dtype=np.float64)
        workspace_max = np.asarray(scene.workspace_max, dtype=np.float64)
        actual_workspace_ok = bool(
            workspace_min.shape == (3,)
            and workspace_max.shape == (3,)
            and np.all(np.isfinite(workspace_min))
            and np.all(np.isfinite(workspace_max))
            and np.all(
                actual[:2, 3]
                >= workspace_min[:2]
                + self._FREE_RIM_VIEW_REACQUIRE_WORKSPACE_MARGIN_M
            )
            and np.all(
                actual[:2, 3]
                <= workspace_max[:2]
                - self._FREE_RIM_VIEW_REACQUIRE_WORKSPACE_MARGIN_M
            )
            and workspace_min[2] <= actual[2, 3] <= workspace_max[2]
        )
        accepted = bool(
            feedback.accepted
            and osc_steps > 0
            and not step_budget_exhausted
            and signed_progress >= max(
                self._FREE_RIM_ESCAPE_MIN_PROGRESS_M,
                command_distance - 0.008,
            )
            and cross_drift <= self._FREE_RIM_ESCAPE_MAX_CROSS_DRIFT_M
            and vertical_drift
            <= self._FREE_RIM_VIEW_REACQUIRE_MAX_VERTICAL_DRIFT_M
            and actual_rotation_drift
            <= self._FREE_RIM_TARGET_EGRESS_SAFE_POSE_ROTATION_TOLERANCE_RAD
            and sample_endpoint_position_error
            <= self._FREE_RIM_VIEW_CAMERA_RIGID_POSITION_TOLERANCE_M
            and sample_endpoint_rotation_error
            <= self._FREE_RIM_VIEW_CAMERA_RIGID_ROTATION_TOLERANCE_RAD
            and end_open_width is not None
            and end_open_width >= self._FREE_RIM_VIEW_FIELD_MIN_OPEN_WIDTH_M
            and actual_distance
            <= self._FREE_RIM_TARGET_EGRESS_MAX_DISTANCE_M
            + self._FREE_RIM_ESCAPE_MAX_CROSS_DRIFT_M
            and np.isfinite(actual_raw_sdf)
            and actual_raw_sdf >= nominal_envelope
            and actual_workspace_ok
        )
        if not feedback.accepted:
            detail = feedback.detail or "robot rejected target-container egress"
        elif osc_steps <= 0:
            detail = "accepted target-container egress reported zero policy actions"
        elif not accepted:
            detail = (
                "public proprio/SDF gates rejected target-container egress; "
                "full inflated envelope was not restored"
            )
        else:
            detail = (
                "target-container egress restored the ordinary inflated envelope"
            )
        self._append_robot_phase_trace(
            {
                "phase": "free_rim_target_container_high_egress",
                "active_grasp_candidate": candidate_id,
                "strategy": (
                    "fresh_target_field_strict_monotonic_constant_height_egress"
                ),
                "target_id": target.instance_id,
                "target_label": target.label,
                "target_field_id": field.source_instance_id,
                "target_field_label": field.source_label,
                "target_field_binding": field_binding,
                "start_xyz_m": start[:3, 3].tolist(),
                "goal_xyz_m": goal[:3, 3].tolist(),
                "final_xyz_m": actual[:3, 3].tolist(),
                "source_top_z_m": source_top_z,
                "minimum_commanded_height_m": float(
                    np.min(waypoints[:, 2, 3])
                ),
                "initial_raw_sdf_m": start_raw_sdf,
                "final_raw_sdf_m": actual_raw_sdf,
                "nominal_inflated_envelope_m": nominal_envelope,
                "dense_corridor_raw_sdf_m": dense_raw.tolist(),
                "commanded_waypoint_raw_sdf_m": waypoint_raw.tolist(),
                "candidate_commanded_raw_sdf_m": candidate_waypoint_raw.tolist(),
                "retained_commanded_raw_sdf_m": retained_waypoint_raw,
                "public_motion_sample_evidence": {
                    key: value
                    for key, value in sample_evidence.items()
                    if key != "last_pose_world"
                },
                "corridors_evaluated": corridors_evaluated,
                "commanded_distance_m": command_distance,
                "hard_distance_upper_bound_m": (
                    self._FREE_RIM_TARGET_EGRESS_MAX_DISTANCE_M
                ),
                "maximum_waypoint_spacing_m": (
                    self._FREE_RIM_TARGET_EGRESS_WAYPOINT_SPACING_M
                ),
                "public_signed_progress_m": signed_progress,
                "public_cross_drift_m": cross_drift,
                "public_vertical_drift_m": vertical_drift,
                "public_rotation_drift_rad": actual_rotation_drift,
                "sample_endpoint_position_error_m": (
                    sample_endpoint_position_error
                ),
                "sample_endpoint_rotation_error_rad": (
                    sample_endpoint_rotation_error
                ),
                "final_workspace_xyz_gate": actual_workspace_ok,
                "start_gripper_width_m": start_open_width,
                "end_gripper_width_m": end_open_width,
                "osc_steps": osc_steps,
                "step_budget_exhausted": step_budget_exhausted,
                "gripper_command": command,
                "accepted": accepted,
                "detail": detail,
                "next_approach_requires_fresh_rgbd": True,
            }
        )
        if callable(invalidate):
            invalidate()
        self._cavity_recovery_pending = False
        if accepted:
            self._cavity_approach_safe_pose = actual.copy()
            self._cavity_failed_candidate_id = None
            start_scene_timestamp = self._builtin_finite_number(scene.timestamp_s)
            start_scene_capture_id = scene.capture_id
            start_scene_camera_capture_ids = dict(scene.camera_capture_ids)
            self._free_rim_last_assembly_egress_proof = (
                {
                    "candidate_id": candidate_id,
                    "source_id": source.instance_id,
                    "source_label": source.label,
                    "target_id": target.instance_id,
                    "target_label": target.label,
                    "field_id": field.source_instance_id,
                    "field_label": field.source_label,
                    "field_binding": field_binding,
                    "field_center_world_m": np.asarray(
                        field.center, dtype=np.float64
                    ).copy(),
                    "field_half_extents_m": np.asarray(
                        field.half_extents, dtype=np.float64
                    ).copy(),
                    "field_axes_world": np.asarray(
                        field.axes, dtype=np.float64
                    ).copy(),
                    "surface_points_by_camera": tuple(
                        (name, np.asarray(points, dtype=np.float64).copy())
                        for name, points in field.surface_points_by_camera
                    ),
                    "start_ee_pose_world": start.copy(),
                    "final_ee_pose_world": actual.copy(),
                    "egress_command_direction_world": np.asarray(
                        direction, dtype=np.float64
                    ).copy(),
                    "start_gripper_width_m": start_open_width,
                    "end_gripper_width_m": end_open_width,
                    "start_ee_world_m": start[:3, 3].copy(),
                    "final_ee_world_m": actual[:3, 3].copy(),
                    "initial_raw_sdf_m": start_raw_sdf,
                    "nominal_envelope_m": nominal_envelope,
                    "start_scene_timestamp_s": start_scene_timestamp,
                    "start_scene_capture_id": start_scene_capture_id,
                    "start_scene_camera_capture_ids": (
                        start_scene_camera_capture_ids
                    ),
                    "labels": (
                        tuple(labels)
                        if isinstance(labels, (tuple, list))
                        and labels
                        and all(
                            isinstance(label, str) and label.strip()
                            for label in labels
                        )
                        else ()
                    ),
                    "start_view_snapshot": start_view_snapshot,
                    "candidate_commanded_raw_sdf_m": (
                        candidate_waypoint_raw.copy()
                    ),
                    "retained_commanded_raw_sdf_m": {
                        key: tuple(value)
                        for key, value in retained_waypoint_raw.items()
                    },
                }
                if field_binding == "unique_stable_assembly_field"
                and start_scene_timestamp is not None
                else None
            )
            return True
        self._free_rim_last_assembly_egress_proof = None
        self._mark_free_space_environment_blocked(candidate_id)
        self._cavity_failed_candidate_id = candidate_id
        if step_budget_exhausted:
            raise PolicyStepBudgetExhausted(
                "episode OSC step budget exhausted during target-container egress"
            )
        return False

    def _try_free_rim_high_start_escape(self) -> bool | None:
        """Try one sensor-proven free-space start escape.

        ``True``/``False`` means an escape command was issued and fully
        handled here.  ``None`` means a safety prerequisite failed before any
        command, so the pre-existing safe-column recovery remains in force.
        """

        if not self._free_rim_escape_pending:
            return None
        self._free_rim_escape_pending = False
        candidate_id = self._free_rim_escape_candidate_id
        source_id = self._free_rim_escape_source_id
        labels = self._free_rim_escape_labels
        nominal_envelope = self._free_rim_escape_nominal_envelope_m
        invalidate = getattr(self.observer, "invalidate_sensor_cache", None)

        def reject(detail: str, **extra: object) -> None:
            self._mark_free_rim_escape_candidate_failed()
            self._append_robot_phase_trace(
                {
                    "phase": "free_rim_high_start_escape",
                    "active_grasp_candidate": candidate_id,
                    "accepted": False,
                    "detail": detail,
                    "osc_steps": 0,
                    "gripper_command": self._released_gripper_command(
                        GraspMode.RIM_PINCH
                    ),
                    **extra,
                }
            )

        if source_id is None or nominal_envelope is None:
            reject("missing sensor-bound source or failed nominal envelope")
            return None
        if callable(invalidate):
            invalidate()
        try:
            scene = self.observer.observe(labels)
            start = np.asarray(self.robot.current_ee_pose(), dtype=np.float64)
            source = scene.by_id(source_id)
        except (AttributeError, TypeError, ValueError, PerceptionError) as exc:
            reject(f"fresh sensor/public-proprio escape binding failed: {exc}")
            return None
        if start.shape != (4, 4) or not np.all(np.isfinite(start)):
            reject("public EE pose is unavailable or non-finite")
            return None

        source_top_z = float(source.position[2] + 0.5 * source.extent[2])
        minimum_escape_z = (
            source_top_z + self._FREE_RIM_ESCAPE_MIN_HEIGHT_ABOVE_SOURCE_M
        )
        start_raw_sdf = float(scene.obstacle_sdf.distance(start[:3, 3]))
        if (
            not np.isfinite(start_raw_sdf)
            or start_raw_sdf <= 0.0
            or start_raw_sdf >= nominal_envelope
        ):
            reject(
                "fresh SDF does not confirm a positive start inside only the inflated envelope",
                start_raw_sdf_m=start_raw_sdf,
                nominal_inflated_envelope_m=nominal_envelope,
            )
            return None
        if float(start[2, 3]) < minimum_escape_z:
            reject(
                "public EE is below the sensor-derived high escape corridor",
                start_z_m=float(start[2, 3]),
                minimum_escape_z_m=minimum_escape_z,
            )
            return None

        corridor = self._free_rim_high_corridor(scene, start, source)
        if corridor is None:
            reject(
                "no bounded workspace corridor has monotonically non-decreasing fresh SDF distance",
                start_raw_sdf_m=start_raw_sdf,
                nominal_inflated_envelope_m=nominal_envelope,
            )
            return None
        goal, commanded_distances, corridors_evaluated = corridor
        direction = goal[:2, 3] - start[:2, 3]
        command_distance = float(np.linalg.norm(direction))
        direction /= command_distance
        command = self._released_gripper_command(GraspMode.RIM_PINCH)
        steps_before = int(getattr(self.robot, "steps_executed", 0))
        feedback = self.robot.execute_waypoints(
            np.stack((start, goal)), Phase.RETREAT, command
        )
        steps_after = int(getattr(self.robot, "steps_executed", 0))
        actual = np.asarray(self.robot.current_ee_pose(), dtype=np.float64)
        osc_steps = steps_after - steps_before
        actual_delta_xy = actual[:2, 3] - start[:2, 3]
        signed_progress = float(np.dot(actual_delta_xy, direction))
        cross_drift = float(
            np.linalg.norm(actual_delta_xy - signed_progress * direction)
        )
        vertical_drift = abs(float(actual[2, 3] - start[2, 3]))
        actual_distance = float(np.linalg.norm(actual_delta_xy))
        actual_raw_sdf = float(scene.obstacle_sdf.distance(actual[:3, 3]))
        accepted = bool(
            feedback.accepted
            and osc_steps > 0
            and signed_progress >= self._FREE_RIM_ESCAPE_MIN_PROGRESS_M
            and cross_drift <= self._FREE_RIM_ESCAPE_MAX_CROSS_DRIFT_M
            and vertical_drift <= self._FREE_RIM_ESCAPE_MAX_VERTICAL_DRIFT_M
            and actual_distance
            <= self._FREE_RIM_ESCAPE_DISTANCE_M
            + self._FREE_RIM_ESCAPE_MAX_CROSS_DRIFT_M
            and np.isfinite(actual_raw_sdf)
            and actual_raw_sdf + 1e-12 >= start_raw_sdf
        )
        if not feedback.accepted:
            detail = feedback.detail or "robot rejected high start escape"
        elif osc_steps <= 0:
            detail = "accepted escape reported zero policy actions"
        elif not accepted:
            detail = "public proprio/SDF gates rejected executed high start escape"
        else:
            detail = "bounded high start escape made sensor-proven progress"
        self._append_robot_phase_trace(
            {
                "phase": "free_rim_high_start_escape",
                "active_grasp_candidate": candidate_id,
                "strategy": "fresh_rgbd_sdf_ranked_constant_height_corridor",
                "start_xyz_m": start[:3, 3].tolist(),
                "goal_xyz_m": goal[:3, 3].tolist(),
                "final_xyz_m": actual[:3, 3].tolist(),
                "source_top_z_m": source_top_z,
                "minimum_commanded_height_m": float(
                    min(start[2, 3], goal[2, 3])
                ),
                "start_raw_sdf_m": start_raw_sdf,
                "final_raw_sdf_m": actual_raw_sdf,
                "nominal_inflated_envelope_m": nominal_envelope,
                "commanded_corridor_raw_sdf_m": commanded_distances.tolist(),
                "corridors_evaluated": corridors_evaluated,
                "commanded_distance_m": command_distance,
                "public_signed_progress_m": signed_progress,
                "public_cross_drift_m": cross_drift,
                "public_vertical_drift_m": vertical_drift,
                "osc_steps": osc_steps,
                "gripper_command": command,
                "accepted": accepted,
                "detail": detail,
                "next_approach_requires_fresh_rgbd": True,
            }
        )
        if callable(invalidate):
            invalidate()
        self._cavity_recovery_pending = False
        if accepted:
            # The next task attempt starts from the observed escaped pose and
            # rebinds the rim from a fresh frame.  Do not consume a physical
            # rim edge for an environment/start-envelope failure.
            self._cavity_approach_safe_pose = actual.copy()
            self._cavity_failed_candidate_id = None
            return True
        self._mark_free_rim_escape_candidate_failed()
        if bool(getattr(self.robot, "step_budget_exhausted", False)):
            raise PolicyStepBudgetExhausted(
                "episode OSC step budget exhausted during free-rim high escape"
            )
        return False

    def _safe_retreat(self) -> None:
        """Recover a failed cavity approach without a low lateral sweep.

        The first pose is public proprioception captured before entering the
        cavity.  Recovery first raises vertically at the current XY, then uses
        the ordinary receding-horizon optimiser and fresh RGB-D SDF to return
        to that safe XY/orientation while remaining at the raised height.
        """

        target_egress_handled = self._try_free_rim_target_container_egress()
        if target_egress_handled is not None:
            return

        escape_handled = self._try_free_rim_high_start_escape()
        if escape_handled is not None:
            return

        if (
            not self._cavity_recovery_pending
            or self._cavity_approach_safe_pose is None
        ):
            invalidate = getattr(self.observer, "invalidate_sensor_cache", None)
            try:
                super()._safe_retreat()
            finally:
                # Any recovery action changes the camera pose.  Even a failed
                # or externally interrupted attempt must not let the next task
                # attempt reuse RGB-D captured before that motion.
                if callable(invalidate):
                    invalidate()
            return

        self._cavity_recovery_pending = False
        safe_pose = self._cavity_approach_safe_pose.copy()
        candidate_id = self._cavity_failed_candidate_id
        labels = self._cavity_recovery_labels
        released_command = self._released_gripper_command(
            self._active_grasp_mode
        )
        start = np.asarray(self.robot.current_ee_pose(), dtype=np.float64)
        raised_goal = start.copy()
        raised_goal[2, 3] = max(float(start[2, 3]), float(safe_pose[2, 3]))
        vertical_start_steps = int(getattr(self.robot, "steps_executed", 0))
        vertical_accepted = True
        vertical_detail = "already_at_safe_height"
        if raised_goal[2, 3] > start[2, 3] + 1e-4:
            feedback = self.robot.execute_waypoints(
                np.stack((start, raised_goal)),
                Phase.RETREAT,
                released_command,
            )
            vertical_accepted = bool(feedback.accepted)
            vertical_detail = feedback.detail
        raised = np.asarray(self.robot.current_ee_pose(), dtype=np.float64)
        vertical_error = abs(float(raised_goal[2, 3] - raised[2, 3]))
        vertical_xy_drift = float(
            np.linalg.norm(raised[:2, 3] - start[:2, 3])
        )
        self._append_robot_phase_trace(
            {
                "phase": "cavity_recovery_vertical",
                "failed_candidate": candidate_id,
                "start_xyz_m": start[:3, 3].tolist(),
                "goal_xyz_m": raised_goal[:3, 3].tolist(),
                "final_xyz_m": raised[:3, 3].tolist(),
                "target_height_source": "max_current_and_initial_public_proprio",
                "osc_steps": int(getattr(self.robot, "steps_executed", 0))
                - vertical_start_steps,
                "final_vertical_error_m": vertical_error,
                "actual_xy_drift_m": vertical_xy_drift,
                "accepted": vertical_accepted,
                "detail": vertical_detail,
            }
        )
        if not vertical_accepted:
            invalidate = getattr(self.observer, "invalidate_sensor_cache", None)
            if callable(invalidate):
                invalidate()
            return

        # Keep the measured raised Z for the lateral return.  The next fresh
        # APPROACH may descend from here; recovery itself never combines a
        # lateral sweep with a descent toward the fixture.
        return_goal = safe_pose.copy()
        return_goal[2, 3] = raised[2, 3]
        return_start = raised.copy()
        return_start_steps = int(getattr(self.robot, "steps_executed", 0))
        replans = 0
        return_detail = "safe pose return did not converge"
        fresh_scene_observations = 0
        replan_min_heights: list[float] = []
        invalidate = getattr(self.observer, "invalidate_sensor_cache", None)
        if callable(invalidate):
            invalidate()
        try:
            self.mpc.reset()
            for _ in range(self.config.max_mpc_replans):
                current = np.asarray(self.robot.current_ee_pose(), dtype=np.float64)
                position_error, rotation_error = self._recovery_pose_errors(
                    current, return_goal
                )
                if (
                    position_error <= self.config.position_tolerance_m
                    and rotation_error <= self.config.orientation_tolerance_rad
                ):
                    return_detail = "initial safe column reached"
                    break
                scene = self.observer.observe(labels)
                fresh_scene_observations += 1
                tool_radius_m = 0.025
                clearance_m = 0.025
                start_surface_distance = float(
                    scene.obstacle_sdf.distance(current[:3, 3])
                )
                if (
                    np.isfinite(start_surface_distance)
                    and 0.0 < start_surface_distance < tool_radius_m + clearance_m
                ):
                    tool_radius_m = min(tool_radius_m, start_surface_distance)
                    clearance_m = max(
                        0.0, start_surface_distance - tool_radius_m
                    )
                replan_min_height = max(
                    float(safe_pose[2, 3]),
                    float(current[2, 3]) - 0.003,
                )
                replan_min_heights.append(replan_min_height)
                request = MotionRequest(
                    phase=Phase.RETREAT,
                    start_pose=current,
                    goal_pose=return_goal,
                    clearance_m=clearance_m,
                    tool_radius_m=tool_radius_m,
                    min_height_m=replan_min_height,
                )
                chunk = self.mpc.replan(request, scene)
                replans += 1
                feedback = self.robot.execute_waypoints(
                    chunk.poses,
                    Phase.RETREAT,
                    released_command,
                )
                if not feedback.accepted:
                    return_detail = feedback.detail or "safe return motion rejected"
                    break
            final = np.asarray(self.robot.current_ee_pose(), dtype=np.float64)
            final_position_error, final_rotation_error = self._recovery_pose_errors(
                final, return_goal
            )
            return_accepted = bool(
                final_position_error <= self.config.position_tolerance_m
                and final_rotation_error <= self.config.orientation_tolerance_rad
            )
            if return_accepted:
                return_detail = "initial safe column reached"
        except (ExecutionError, OptimisationError, PerceptionError) as exc:
            final = np.asarray(self.robot.current_ee_pose(), dtype=np.float64)
            final_position_error, final_rotation_error = self._recovery_pose_errors(
                final, return_goal
            )
            return_accepted = False
            return_detail = str(exc)
        finally:
            # The next task attempt must not bind from the SDF snapshot used
            # for recovery planning.  It performs a fresh dual-RGB-D observe.
            if callable(invalidate):
                invalidate()
        self._append_robot_phase_trace(
            {
                "phase": "cavity_recovery_safe_pose_return",
                "failed_candidate": candidate_id,
                "strategy": "fresh_rgbd_sdf_receding_horizon_at_raised_height",
                "initial_public_proprio_safe_xyz_m": safe_pose[:3, 3].tolist(),
                "start_xyz_m": return_start[:3, 3].tolist(),
                "goal_xyz_m": return_goal[:3, 3].tolist(),
                "final_xyz_m": final[:3, 3].tolist(),
                "minimum_height_floor_m": float(safe_pose[2, 3]),
                "replan_min_height_m": replan_min_heights,
                "replans": replans,
                "fresh_scene_observations": fresh_scene_observations,
                "osc_steps": int(getattr(self.robot, "steps_executed", 0))
                - return_start_steps,
                "final_position_error_m": final_position_error,
                "final_rotation_error_rad": final_rotation_error,
                "accepted": return_accepted,
                "detail": return_detail,
                "next_binding_requires_fresh_rgbd": True,
            }
        )

    def _execute_phase(
        self,
        phase: Phase,
        bound: BoundConstraintGraph,
        labels: Sequence[str],
        grasp_mode: GraspMode,
    ) -> BoundConstraintGraph:
        # Enforce the black-bowl contract before even updating adapter state or
        # configuring a robot hook.  Every grasp mode requires three exact
        # native-string views of the source label to agree after normalization;
        # an inconsistent graph/binding/request cannot choose the least strict
        # label and bypass the black-bowl external-rim contract.
        try:
            bound_source_label = getattr(bound, "source_label", "")
            requested_source_label = (
                labels[0]
                if isinstance(labels, Sequence)
                and not isinstance(labels, (str, bytes))
                and len(labels) > 0
                else ""
            )
        except BaseException as exc:
            raise ExecutionError(
                "source labels are unavailable before phase side effects"
            ) from exc
        try:
            graph_source_label = getattr(bound.graph.source, "label")
        except BaseException as exc:
            raise ExecutionError(
                "graph source label is unavailable before phase side effects"
            ) from exc
        source_labels = (
            graph_source_label,
            bound_source_label,
            requested_source_label,
        )
        if any(
            type(value) is not str or not value.strip()
            for value in source_labels
        ):
            raise ExecutionError(
                "source labels require exact non-empty native strings"
            )
        for source_label in source_labels:
            self._require_black_bowl_rim_pinch(source_label, grasp_mode)
        normalized_source_labels = tuple(
            self._normalise_semantic_label(value) for value in source_labels
        )
        if len(set(normalized_source_labels)) != 1:
            raise ExecutionError(
                "source labels disagree across graph, binding, and request"
            )
        self._integration_phase = phase
        set_mode = getattr(self.robot, "set_active_grasp_mode", None)
        if callable(set_mode):
            set_mode(grasp_mode)
        set_candidate = getattr(self.robot, "set_active_grasp_candidate", None)
        if callable(set_candidate):
            set_candidate(bound.grasp.candidate.candidate_id)
        configure_cavity_profile = getattr(
            self.robot, "set_cavity_rim_execution_profile", None
        )
        if callable(configure_cavity_profile):
            profile_for = getattr(
                self.grasp_provider, "cavity_rim_execution_profile", None
            )
            profile = (
                profile_for(bound.grasp.candidate.candidate_id)
                if callable(profile_for)
                else None
            )
            configure_cavity_profile(profile)
        requires_reacquire = getattr(
            self.grasp_provider, "requires_full_reacquire", None
        )
        cavity_approach = bool(
            phase == Phase.APPROACH
            and grasp_mode == GraspMode.RIM_PINCH
            and callable(requires_reacquire)
            and requires_reacquire(bound.grasp.candidate.candidate_id)
        )
        if cavity_approach:
            candidate_id = bound.grasp.candidate.candidate_id
            if self._cavity_approach_safe_pose is None:
                current_pose = getattr(self.robot, "current_ee_pose", None)
                if callable(current_pose):
                    self._cavity_approach_safe_pose = np.asarray(
                        current_pose(), dtype=np.float64
                    ).copy()
                    self._append_robot_phase_trace(
                        {
                            "phase": "cavity_recovery_safe_pose_cached",
                            "active_grasp_candidate": candidate_id,
                            "safe_pose_xyz_m": (
                                self._cavity_approach_safe_pose[:3, 3].tolist()
                            ),
                        }
                    )
            elif (
                self._cavity_failed_candidate_id is not None
                and candidate_id != self._cavity_failed_candidate_id
            ):
                self._append_robot_phase_trace(
                    {
                        "phase": "cavity_candidate_switch",
                        "failed_candidate": self._cavity_failed_candidate_id,
                        "active_grasp_candidate": candidate_id,
                        "fresh_task_rebind": True,
                    }
                )
                self._cavity_failed_candidate_id = None
        allow_tentative = getattr(
            self.robot, "set_tentative_expansion_allowed", None
        )
        if callable(allow_tentative):
            allow_tentative(phase == Phase.GRASP and grasp_mode == GraspMode.EXPAND)
        if phase == Phase.APPROACH:
            reset = getattr(self.robot, "reset_expand_safe_pregrasp", None)
            if callable(reset):
                reset()
            reset_rim = getattr(self.robot, "reset_rim_safe_pregrasp", None)
            if callable(reset_rim):
                reset_rim()
        if phase == Phase.GRASP:
            retain = getattr(self.robot, "set_expand_retention_confirmed", None)
            if callable(retain):
                # A candidate must earn a new post-LIFT retention result.
                retain(False)
            rim_retain = getattr(self.robot, "set_rim_retention_confirmed", None)
            if callable(rim_retain):
                rim_retain(False)
            reset_marginal = getattr(self.robot, "reset_rim_grasp_marginal", None)
            if callable(reset_marginal):
                reset_marginal()
        if phase == Phase.LIFT:
            reset = getattr(self.robot, "reset_expand_lift_tolerance", None)
            if callable(reset):
                reset()
            reset_rim_lift = getattr(self.robot, "reset_rim_lift_tolerance", None)
            if callable(reset_rim_lift):
                reset_rim_lift()
            retain = getattr(self.robot, "set_expand_retention_confirmed", None)
            if callable(retain):
                retain(False)
            rim_retain = getattr(self.robot, "set_rim_retention_confirmed", None)
            if callable(rim_retain):
                rim_retain(False)
        if phase == Phase.TRANSFER:
            reset = getattr(self.robot, "reset_expand_transfer_tolerance", None)
            if callable(reset):
                reset()
            reset_rim_transfer = getattr(
                self.robot, "reset_rim_transfer_tolerance", None
            )
            if callable(reset_rim_transfer):
                reset_rim_transfer()
        if phase == Phase.GRASP:
            allows = getattr(self.grasp_provider, "allows_contact_completion", None)
            enabled = bool(allows(bound.grasp.candidate.candidate_id)) if callable(allows) else False
            residual_for = getattr(
                self.grasp_provider, "contact_completion_residual_m", None
            )
            max_residual = (
                residual_for(bound.grasp.candidate.candidate_id)
                if callable(residual_for)
                else None
            )
            configure = getattr(self.robot, "set_grasp_contact_enabled", None)
            if callable(configure):
                configure(enabled, max_residual_m=max_residual)
            reset = getattr(self.robot, "reset_grasp_contact", None)
            if callable(reset):
                reset()
        if phase == Phase.PLACE:
            reset = getattr(self.robot, "reset_place_contact", None)
            if callable(reset):
                reset()
        try:
            try:
                executed = super()._execute_phase(
                    phase, bound, labels, grasp_mode
                )
            except (ExecutionError, OptimisationError) as exc:
                if bool(getattr(self.robot, "step_budget_exhausted", False)):
                    raise PolicyStepBudgetExhausted(
                        "episode OSC step budget exhausted"
                    ) from exc
                if (
                    phase == Phase.LIFT
                    and grasp_mode == GraspMode.RIM_PINCH
                    and bool(
                        getattr(
                            self.robot,
                            "cavity_rim_load_proof_failed",
                            False,
                        )
                    )
                ):
                    # The low-level proof has already opened the hand before
                    # returning this failure.  Mark exactly this physical
                    # profile and force a fresh high approach; retrying LIFT
                    # with an empty hand or changing sides at drawer height is
                    # never allowed.
                    mark_failed = getattr(
                        self.grasp_provider, "mark_candidate_failed", None
                    )
                    if callable(mark_failed):
                        mark_failed(bound.grasp.candidate.candidate_id)
                    invalidate = getattr(
                        self.observer, "invalidate_sensor_cache", None
                    )
                    if callable(invalidate):
                        invalidate()
                    raise GraspBindingError(
                        "cavity rim load proof failed; fresh physical "
                        "candidate reacquisition required"
                    ) from exc
                if phase == Phase.GRASP:
                    mark_failed = getattr(
                        self.grasp_provider, "mark_candidate_failed", None
                    )
                    if callable(mark_failed):
                        mark_failed(bound.grasp.candidate.candidate_id)
                elif cavity_approach:
                    candidate_id = bound.grasp.candidate.candidate_id
                    public_ee_xyz_m, pose_available = self._public_ee_xyz(
                        self.robot
                    )
                    # Preserve the optimiser/executor exception verbatim.  In
                    # particular, a later recovery classification must never
                    # replace its raw-SDF argmin diagnostic.
                    self._append_robot_phase_trace(
                        {
                            "phase": "cavity_approach_failure",
                            "active_grasp_candidate": candidate_id,
                            "failure_type": type(exc).__name__,
                            "failure_detail": str(exc),
                            "public_ee_xyz_m": public_ee_xyz_m,
                            "pose_available": pose_available,
                            "recovery_required": True,
                        }
                    )
                    start_envelope = self._start_clearance_envelope_from_failure(
                        exc
                    )
                    negative_diagnostic = (
                        self._negative_start_clearance_diagnostic(exc)
                    )
                    free_space_rim = candidate_id.startswith("analytic-rim-")
                    source_id = getattr(bound, "source_id", None)
                    current_pose = self._public_ee_pose(self.robot)
                    safe_pose_matches = False
                    if (
                        current_pose is not None
                        and self._cavity_approach_safe_pose is not None
                    ):
                        safe_position_error, safe_rotation_error = (
                            self._recovery_pose_errors(
                                current_pose, self._cavity_approach_safe_pose
                            )
                        )
                        safe_pose_matches = bool(
                            safe_position_error
                            <= self._FREE_RIM_TARGET_EGRESS_SAFE_POSE_POSITION_TOLERANCE_M
                            and safe_rotation_error
                            <= self._FREE_RIM_TARGET_EGRESS_SAFE_POSE_ROTATION_TOLERANCE_RAD
                        )
                    if self._try_install_free_rim_view_self_filter(
                        bound,
                        candidate_id,
                        negative_diagnostic,
                        current_pose,
                    ):
                        # Retry inside this phase attempt.  The task-level
                        # attempt budget represents physical grasp attempts;
                        # a proved wrist/view self component consumes neither
                        # a rim edge nor another task attempt.
                        self._free_rim_view_field_retry_active = True
                        try:
                            return self._execute_phase(
                                phase, bound, labels, grasp_mode
                            )
                        finally:
                            self._cleanup_free_rim_typed_view_transaction(
                                invalidate_cache=(
                                    candidate_id
                                    in self._free_rim_view_reacquired_candidates
                                ),
                                reset_retry_active=True,
                            )
                    if (
                        self._free_rim_view_field_retry_active
                        and candidate_id
                        in self._free_rim_view_reacquired_candidates
                    ):
                        self._append_robot_phase_trace(
                            {
                                "phase": (
                                    "free_rim_three_frame_view_reacquisition"
                                ),
                                "active_grasp_candidate": candidate_id,
                                "accepted": False,
                                "proof_state": "REJECT",
                                "reason": (
                                    "fresh_application_rejected_after_typed_"
                                    "negative_install"
                                ),
                                "candidate_failure_keys_consumed": False,
                                "view_budget_consumed": True,
                            }
                        )
                        raise FreeRimViewEvidenceUnavailable(
                            "free-rim typed-negative self-field application "
                            "failed closed"
                        )
                    can_target_egress = bool(
                        free_space_rim
                        and pose_available
                        and safe_pose_matches
                        and negative_diagnostic is not None
                        and self._is_target_container_diagnostic(
                            bound, negative_diagnostic
                        )
                        and candidate_id
                        not in self._free_rim_target_egress_attempted_candidates
                    )
                    if can_target_egress:
                        self._free_rim_target_egress_attempted_candidates.add(
                            candidate_id
                        )
                        self._free_rim_target_egress_pending = True
                        self._free_rim_target_egress_candidate_id = candidate_id
                        self._free_rim_target_egress_source_id = str(source_id)
                        self._free_rim_target_egress_source_label = str(
                            getattr(bound, "source_label", "")
                        )
                        self._free_rim_target_egress_target_id = str(
                            getattr(bound, "target_id")
                        )
                        self._free_rim_target_egress_target_label = str(
                            getattr(bound, "target_label")
                        )
                        self._free_rim_target_egress_labels = tuple(labels)
                        self._free_rim_target_egress_diagnostic = dict(
                            negative_diagnostic
                        )
                        self._cavity_recovery_labels = tuple(labels)
                        self._cavity_recovery_pending = True
                        self._append_robot_phase_trace(
                            {
                                "phase": "free_rim_target_egress_eligibility",
                                "active_grasp_candidate": candidate_id,
                                "optimizer_argmin_location": "start",
                                "raw_sdf_m": negative_diagnostic[
                                    "raw_distance_m"
                                ],
                                "target_id": getattr(bound, "target_id"),
                                "target_label": getattr(bound, "target_label"),
                                "diagnostic_field_id": negative_diagnostic[
                                    "field_id"
                                ],
                                "diagnostic_field_label": negative_diagnostic[
                                    "field_label"
                                ],
                                "public_ee_equals_cached_safe_pose": True,
                                "accepted": True,
                                "next_step": (
                                    "fresh_target_identity_and_high_egress_validation"
                                ),
                            }
                        )
                        raise GraspBindingError(
                            "free-space rim approach starts inside the current "
                            "semantic target field; target-container high egress "
                            "required"
                        ) from exc

                    can_escape = bool(
                        free_space_rim
                        and pose_available
                        and isinstance(source_id, str)
                        and source_id
                        and start_envelope is not None
                        and candidate_id
                        not in self._free_rim_escape_attempted_candidates
                    )
                    if can_escape:
                        self._free_rim_escape_attempted_candidates.add(candidate_id)
                        self._free_rim_escape_pending = True
                        self._free_rim_escape_candidate_id = candidate_id
                        self._free_rim_escape_source_id = source_id
                        self._free_rim_escape_labels = tuple(labels)
                        self._free_rim_escape_nominal_envelope_m = start_envelope
                        self._cavity_recovery_labels = tuple(labels)
                        self._cavity_recovery_pending = True
                        self._append_robot_phase_trace(
                            {
                                "phase": "free_rim_start_escape_eligibility",
                                "active_grasp_candidate": candidate_id,
                                "optimizer_argmin_location": "start",
                                "nominal_inflated_envelope_m": start_envelope,
                                "accepted": True,
                                "next_step": (
                                    "fresh_rgbd_sdf_high_corridor_validation"
                                ),
                            }
                        )
                        raise GraspBindingError(
                            "free-space rim approach failed at the inflated "
                            "SDF start; high start-envelope escape required"
                        ) from exc

                    negative_environment_blocked = bool(
                        free_space_rim and negative_diagnostic is not None
                    )
                    if negative_environment_blocked:
                        # The same negative start is independent of which bowl
                        # wall is selected.  If strict target identity/safe-pose
                        # gates fail, suppress both walls rather than repeating
                        # an identical zero-motion plan for the antipodal side.
                        self._mark_free_space_environment_blocked(candidate_id)

                    # All contextual-cavity failures, repeated free-space
                    # start failures, and every endpoint/interior/reach/height
                    # failure keep the original fail-closed behaviour.  Such
                    # evidence consumes exactly the active physical profile
                    # before the safe-column task-level reacquisition.
                    mark_failed = getattr(
                        self.grasp_provider, "mark_candidate_failed", None
                    )
                    if callable(mark_failed) and not negative_environment_blocked:
                        mark_failed(candidate_id)
                    self._cavity_failed_candidate_id = candidate_id
                    self._cavity_recovery_labels = tuple(labels)
                    self._cavity_recovery_pending = True
                    raise GraspBindingError(
                        "cavity approach failed; public-proprio safe-pose "
                        "reacquisition required"
                    ) from exc
                raise
            if phase == Phase.APPROACH and grasp_mode == GraspMode.RIM_PINCH:
                requires_reacquire = getattr(
                    self.grasp_provider,
                    "requires_full_reacquire",
                    None,
                )
                if callable(requires_reacquire) and bool(
                    requires_reacquire(bound.grasp.candidate.candidate_id)
                ):
                    preclose = getattr(self.robot, "preclose_cavity_rim", None)
                    if not callable(preclose):
                        raise ExecutionError(
                            "cavity rim preclose is unavailable"
                        )
                    preclose_feedback = preclose()
                    if not preclose_feedback.accepted:
                        if bool(
                            getattr(self.robot, "step_budget_exhausted", False)
                        ):
                            raise PolicyStepBudgetExhausted(
                                "episode OSC step budget exhausted"
                            )
                        mark_failed = getattr(
                            self.grasp_provider,
                            "mark_candidate_failed",
                            None,
                        )
                        if callable(mark_failed):
                            mark_failed(bound.grasp.candidate.candidate_id)
                        raise GraspBindingError(
                            preclose_feedback.detail
                            or "cavity rim preclose failed"
                        )
            if (
                phase == Phase.GRASP
                and grasp_mode == GraspMode.RIM_PINCH
                and bool(getattr(self.robot, "rim_grasp_marginal", False))
                and callable(
                    reject_marginal := getattr(
                        self.grasp_provider,
                        "reject_marginal_before_lift",
                        None,
                    )
                )
                and bool(
                    reject_marginal(bound.grasp.candidate.candidate_id)
                )
            ):
                # A width only just above empty-close consistently slides
                # below the calibrated contact band during LIFT / TRANSFER.
                # Release it while stationary, then restart the whole task
                # attempt.  A different rim point needs a fresh pregrasp and
                # approach; changing candidates inside the GRASP phase leaves
                # the wrist beside the old rim and produced repeated marginal
                # closes without ever reacquiring the opposite wall.
                mark_failed = getattr(
                    self.grasp_provider, "mark_candidate_failed", None
                )
                if callable(mark_failed):
                    mark_failed(bound.grasp.candidate.candidate_id)
                released = self.robot.set_gripper(
                    self._released_gripper_command(grasp_mode)
                )
                if not released.accepted:
                    raise ExecutionError(
                        "marginal rim grasp could not be safely released"
                    )
                raise GraspBindingError(
                    "marginal rim grasp width; reacquiring the next candidate"
                )
            retained = True
            if phase == Phase.LIFT:
                retained = self.robot.grasp_confirmed(grasp_mode)
                if (
                    not retained
                    and grasp_mode == GraspMode.EXPAND
                    and self._expansion_width_is_tentative()
                ):
                    retained = self._fresh_visual_expansion_confirmed(
                        bound, labels
                    )
                if grasp_mode == GraspMode.EXPAND:
                    remember_retention = getattr(
                        self.robot, "set_expand_retention_confirmed", None
                    )
                    if callable(remember_retention):
                        remember_retention(retained)
                elif grasp_mode == GraspMode.RIM_PINCH:
                    remember_rim_retention = getattr(
                        self.robot, "set_rim_retention_confirmed", None
                    )
                    if callable(remember_rim_retention):
                        remember_rim_retention(retained)
                    if retained:
                        self._fresh_visual_rim_binding(
                            bound, labels, evidence_stage="post_lift"
                        )
            if phase == Phase.LIFT and not retained:
                # A thin package can briefly block the fingers against the
                # table edge and then slip out as lift begins.  Re-check after
                # lift using only persistent gripper proprioception before any
                # held-object propagation is trusted by later phases.
                reject_binding = getattr(self.goals, "reject_held_binding", None)
                if callable(reject_binding):
                    reject_binding(bound.source_id)
                mark_failed = getattr(self.grasp_provider, "mark_candidate_failed", None)
                if callable(mark_failed):
                    mark_failed(bound.grasp.candidate.candidate_id)
                invalidate = getattr(self.observer, "invalidate_sensor_cache", None)
                if callable(invalidate):
                    invalidate()
                # GraspBindingError intentionally bypasses the ordinary
                # same-phase retry in RouteCController and is caught by its
                # task-attempt recovery.  Repeating LIFT with empty fingers
                # would only propagate a stale track farther into free space.
                raise GraspBindingError("grasp was not retained after lift")
            if (
                phase == Phase.TRANSFER
                and grasp_mode == GraspMode.PINCH
                and bound.grasp.candidate.candidate_id.startswith(
                    "analytic-pan-handle-"
                )
                and not self.robot.grasp_confirmed(grasp_mode)
            ):
                # A long pan can pass the vertical LIFT proof yet slide along
                # its handle under the first lateral transport load.  Do not
                # propagate an empty, stale held-object transform into PLACE:
                # blocked gripper width is public proprioception and provides
                # a fresh physical retention check at the transfer endpoint.
                reject_binding = getattr(self.goals, "reject_held_binding", None)
                if callable(reject_binding):
                    reject_binding(bound.source_id)
                mark_failed = getattr(
                    self.grasp_provider, "mark_candidate_failed", None
                )
                if callable(mark_failed):
                    mark_failed(bound.grasp.candidate.candidate_id)
                invalidate = getattr(self.observer, "invalidate_sensor_cache", None)
                if callable(invalidate):
                    invalidate()
                raise GraspBindingError(
                    "pan handle grasp was not retained after transfer"
                )
            if (
                phase == Phase.TRANSFER
                and grasp_mode == GraspMode.RIM_PINCH
            ):
                # A rim-held bowl can finish settling while it is translated
                # across the workspace.  Bind one more fresh RGB-D centre at
                # the final transfer pose before freezing the PLACE target;
                # otherwise a post-LIFT swing is preserved as a plate-scale
                # lateral placement error.
                proprio_retained = self.robot.grasp_confirmed(grasp_mode)
                visual_evidence = self._fresh_visual_rim_binding(
                    bound, labels, evidence_stage="post_transfer"
                )
                visually_retained = bool(visual_evidence.get("accepted", False))
                # Near-EE geometry proves identity / pose, but not contact: a
                # bowl that has already slipped onto a support can remain in
                # the same image gate.  Continue to PLACE only when the
                # fingers are still physically blocked *and* fresh RGB-D sees
                # the bound bowl below them.
                retained = proprio_retained and visually_retained
                remember_rim_retention = getattr(
                    self.robot, "set_rim_retention_confirmed", None
                )
                if callable(remember_rim_retention):
                    remember_rim_retention(retained)
                if not retained:
                    reject_binding = getattr(self.goals, "reject_held_binding", None)
                    if callable(reject_binding):
                        reject_binding(bound.source_id)
                    mark_failed = getattr(
                        self.grasp_provider, "mark_candidate_failed", None
                    )
                    if callable(mark_failed):
                        mark_failed(bound.grasp.candidate.candidate_id)
                    raise GraspBindingError("grasp was not retained after transfer")
            if phase == Phase.RETREAT:
                predicted = getattr(self.goals, "predicted_release_source", None)
                update_track = getattr(self.observer, "update_track", None)
                if predicted is not None and callable(update_track):
                    # This association prior is not visibility evidence.
                    # Gravity releases should search at the resting position,
                    # rather than bind the old airborne id to a basket wall.
                    update_track(bound.source_id, predicted)
                invalidate = getattr(self.observer, "invalidate_sensor_cache", None)
                if callable(invalidate):
                    # VERIFY must inspect a post-release image rather than the
                    # static episode cache used for motion planning.
                    invalidate()
            return executed
        finally:
            if callable(allow_tentative):
                allow_tentative(False)
            self._integration_phase = None

    def _expansion_width_is_tentative(self) -> bool:
        check = getattr(self.robot, "expansion_width_is_tentative", None)
        return bool(check()) if callable(check) else False

    def _fresh_visual_rim_binding(
        self,
        bound: BoundConstraintGraph,
        labels: Sequence[str],
        *,
        evidence_stage: str,
    ) -> dict[str, Any]:
        """Refresh a retained rim grasp from fresh dual RGB-D geometry.

        A bowl may remain rigidly offset at the sampled rim or settle toward
        the finger centre.  Gripper width cannot distinguish those cases, so
        no offset correction is inferred from proprioception alone.  After
        retention has been confirmed, a cache-invalidated visible component
        may update only the held translation; absent or rejected evidence
        leaves the original rigid transform untouched.
        """

        refresh = getattr(self.goals, "refresh_rim_held_binding", None)
        if not callable(refresh):
            return {
                "accepted": False,
                "reason": "rim_binding_refresh_unavailable",
                "evidence_stage": evidence_stage,
            }
        invalidate = getattr(self.observer, "invalidate_sensor_cache", None)
        if callable(invalidate):
            invalidate()
        try:
            scene = self.observer.observe(labels)
            current = np.asarray(self.robot.current_ee_pose(), dtype=np.float64)
            trace = refresh(
                bound,
                scene,
                current,
                getattr(self.observer, "visible_instance_ids", ()),
            )
        except (PerceptionError, ValueError) as exc:
            trace = {
                "accepted": False,
                "reason": f"{evidence_stage}_observation_failed: {exc}",
            }
        trace = {**trace, "evidence_stage": evidence_stage}
        record = getattr(self.robot, "record_rim_visual_binding", None)
        if callable(record):
            record(trace)
        return trace

    def _fresh_visual_expansion_confirmed(
        self,
        bound: BoundConstraintGraph,
        labels: Sequence[str],
    ) -> bool:
        """Resolve a wide opening from a cache-invalidated dual-RGB-D view."""

        invalidate = getattr(self.observer, "invalidate_sensor_cache", None)
        if callable(invalidate):
            invalidate()
        visible_candidates: list[SceneEntity] = []
        nearest_distance: float | None = None
        try:
            scene = self.observer.observe(labels)
            visible_ids = set(
                getattr(self.observer, "visible_instance_ids", ())
            )
            visible_candidates = [
                entity
                for entity in scene.entities
                if entity.instance_id in visible_ids
                and entity.label == bound.source_label
            ]
            if visible_candidates:
                ee_position = np.asarray(
                    self.robot.current_ee_pose(), dtype=np.float64
                )[:3, 3]
                nearest_distance = min(
                    float(np.linalg.norm(entity.position - ee_position))
                    for entity in visible_candidates
                )
        except PerceptionError:
            # Missing visual evidence is conservative failure, never success.
            visible_candidates = []
            nearest_distance = None
        confirmed = bool(
            nearest_distance is not None
            and nearest_distance <= self.expansion_visual_max_distance_m
        )
        record = getattr(self.robot, "record_expansion_visual_check", None)
        if callable(record):
            record(
                confirmed=confirmed,
                nearest_distance_m=nearest_distance,
                visible_candidates=len(visible_candidates),
            )
        return confirmed

    def _at_goal(self, current: np.ndarray, goal: np.ndarray) -> bool:
        if self._integration_phase == Phase.APPROACH and bool(
            getattr(self.robot, "expand_safe_pregrasp_reached", False)
        ):
            return True
        if self._integration_phase == Phase.APPROACH and bool(
            getattr(self.robot, "rim_safe_pregrasp_reached", False)
        ):
            return True
        if self._integration_phase == Phase.GRASP and bool(
            getattr(self.robot, "grasp_contact_reached", False)
        ):
            return True
        if self._integration_phase == Phase.LIFT and bool(
            getattr(self.robot, "expand_lift_tolerance_reached", False)
        ):
            return True
        if self._integration_phase == Phase.LIFT and bool(
            getattr(self.robot, "rim_lift_tolerance_reached", False)
        ):
            return True
        if self._integration_phase == Phase.TRANSFER and bool(
            getattr(self.robot, "expand_transfer_tolerance_reached", False)
        ):
            return True
        if self._integration_phase == Phase.TRANSFER and bool(
            getattr(self.robot, "rim_transfer_tolerance_reached", False)
        ):
            return True
        if self._integration_phase == Phase.PLACE and bool(
            getattr(self.robot, "place_contact_reached", False)
        ):
            return True
        return super()._at_goal(current, goal)


class AnalyticTopDownGraspProvider:
    """Frozen, sensor-derived grasp proposals used when GraspGenX is absent."""

    def __init__(
        self,
        pose_provider: Callable[[], np.ndarray],
        *,
        finger_plane_floor_clearance_m: float = 0.012,
        finger_pad_top_inset_m: float = 0.006,
        free_space_rim_grasp_top_inset_m: float = 0.010,
        rim_pinch_radial_inset_m: float = 0.003,
        rim_pinch_min_radius_m: float = 0.025,
        rim_pinch_max_radius_m: float = 0.060,
        free_space_rim_min_planar_axis_ratio: float = 0.80,
        free_space_rim_max_height_ratio: float = 0.75,
        cavity_outer_finger_lift_rad: float = np.deg2rad(10.0),
        cavity_rim_min_safe_wall_clearance_m: float = 0.025,
        cavity_nominal_direct_min_target_sdf_m: float = 0.040,
        grasp_mode_selector: GraspModeSelector | None = None,
    ) -> None:
        if (
            finger_plane_floor_clearance_m <= 0
            or finger_pad_top_inset_m <= 0
            or free_space_rim_grasp_top_inset_m <= 0
            or rim_pinch_radial_inset_m <= 0
            or rim_pinch_min_radius_m <= 0
            or rim_pinch_max_radius_m <= 0
        ):
            raise ValueError("finger-plane clearances must be positive")
        if rim_pinch_min_radius_m >= rim_pinch_max_radius_m:
            raise ValueError("rim-pinch radius bounds must be ordered")
        if (
            not 0.80 <= free_space_rim_min_planar_axis_ratio <= 1.0
            or not 0.45 <= free_space_rim_max_height_ratio <= 0.75
        ):
            raise ValueError(
                "free-space rim geometry gates may only be made more conservative"
            )
        if (
            not np.isfinite(cavity_outer_finger_lift_rad)
            or not 0.0 < cavity_outer_finger_lift_rad <= np.deg2rad(25.0)
        ):
            raise ValueError(
                "cavity outer-finger lift must be in (0, 25] degrees"
            )
        if (
            not np.isfinite(cavity_rim_min_safe_wall_clearance_m)
            or cavity_rim_min_safe_wall_clearance_m <= 0.0
        ):
            raise ValueError(
                "cavity rim minimum safe wall clearance must be positive"
            )
        if (
            not np.isfinite(cavity_nominal_direct_min_target_sdf_m)
            or cavity_nominal_direct_min_target_sdf_m <= 0.0
        ):
            raise ValueError(
                "cavity nominal-direct target SDF clearance must be positive"
            )
        self.pose_provider = pose_provider
        # ``freeze_reset_pose`` is invoked by the sequential coordinator only
        # after semantic planning succeeds, but before its first robot action.
        # Keeping construction lazy preserves the formal guarantee that a
        # zero-step semantic failure receives no sensor observation at all.
        self._reset_tool_z_hemisphere_sign: float | None = None
        self._reset_tool_z_world_up_dot: float | None = None
        self._reset_planar_jaw_axis_world: np.ndarray | None = None
        self._reset_planar_jaw_axis_source: str | None = None
        self.finger_plane_floor_clearance_m = float(finger_plane_floor_clearance_m)
        self.finger_pad_top_inset_m = float(finger_pad_top_inset_m)
        self.free_space_rim_grasp_top_inset_m = float(
            free_space_rim_grasp_top_inset_m
        )
        self.rim_pinch_radial_inset_m = float(rim_pinch_radial_inset_m)
        self.rim_pinch_min_radius_m = float(rim_pinch_min_radius_m)
        self.rim_pinch_max_radius_m = float(rim_pinch_max_radius_m)
        self.free_space_rim_min_planar_axis_ratio = float(
            free_space_rim_min_planar_axis_ratio
        )
        self.free_space_rim_max_height_ratio = float(
            free_space_rim_max_height_ratio
        )
        self.cavity_outer_finger_lift_rad = float(
            cavity_outer_finger_lift_rad
        )
        self.cavity_rim_min_safe_wall_clearance_m = float(
            cavity_rim_min_safe_wall_clearance_m
        )
        self.cavity_nominal_direct_min_target_sdf_m = float(
            cavity_nominal_direct_min_target_sdf_m
        )
        self.grasp_mode_selector = grasp_mode_selector or EntityGraspModeSelector()
        self._contact_completion: dict[str, bool] = {}
        self._contact_residual_m: dict[str, float] = {}
        self._failed_candidates: set[str] = set()
        # Contextual cavity ids include the *current* unsigned PCA-axis sign.
        # Map the ids from the latest proposal to a role relative to the
        # episode's frozen physical rim direction, so a recovery-time local-Y
        # sign flip neither resurrects a failed profile nor suppresses its
        # antipodal counterpart.
        self._candidate_failure_keys_by_id: dict[str, str] = {}
        self._marginal_reacquire_candidates: set[str] = set()
        # A contextual drawer retry can start from a different EE pose after
        # recovery.  Preserve the first public proprioception-to-RGB-D bearing
        # for the episode so retries cannot silently rotate the cavity frame.
        self._initial_cavity_approach_world: np.ndarray | None = None
        self._initial_cavity_lateral_axis_world: np.ndarray | None = None
        self._initial_cavity_source_center_world: np.ndarray | None = None
        self._initial_cavity_rim_radius_m: float | None = None
        # Preserve the first collision-free *physical* rim side as a world
        # direction too.  ``align_panda_finger_axis`` may reverse local-Y to
        # avoid a 180-degree wrist turn after recovery; caching a direction
        # instead of a signed PCA index makes that harmless and keeps retries
        # on the same side of the visible cavity.
        self._initial_cavity_rim_direction_world: np.ndarray | None = None
        # Free-space proposals also render the near wall twice (indices 0 and
        # 2).  Freeze its first world direction so a failed physical wall is
        # not resurrected by a later wrist-axis sign flip or duplicate id.
        self._initial_free_space_rim_direction_world: np.ndarray | None = None
        self._cavity_rim_execution_profiles: dict[str, dict[str, Any]] = {}
        self.last_proposal_trace: dict[str, Any] = {}

    def freeze_reset_pose(self) -> None:
        """Freeze the episode-reset top-down frame from proprioception.

        The call is idempotent so compound atomic goals cannot replace the
        reset frame after a stove/drawer skill has rotated the wrist.  Direct
        provider users that do not have a coordinator are supported by a lazy
        call from ``_world_up_top_down_base`` at their first proposal.
        """

        if self._reset_tool_z_hemisphere_sign is not None:
            return
        reset_pose = np.asarray(self.pose_provider(), dtype=np.float64)
        if reset_pose.shape != (4, 4) or not np.all(np.isfinite(reset_pose)):
            raise GraspBindingError(
                "analytic grasp provider requires a finite reset-time EE pose"
            )
        if not np.allclose(reset_pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
            raise GraspBindingError(
                "analytic grasp provider received an invalid reset-time EE pose"
            )
        reset_rotation = reset_pose[:3, :3]
        if not np.allclose(
            reset_rotation.T @ reset_rotation,
            np.eye(3),
            atol=2e-3,
        ) or not np.isclose(np.linalg.det(reset_rotation), 1.0, atol=2e-3):
            raise GraspBindingError(
                "analytic grasp provider requires a right-handed reset rotation"
            )
        reset_dot = float(reset_rotation[:3, 2] @ (0.0, 0.0, 1.0))
        self._reset_tool_z_hemisphere_sign = 1.0 if reset_dot >= 0.0 else -1.0
        self._reset_tool_z_world_up_dot = reset_dot
        world_up = np.array((0.0, 0.0, 1.0), dtype=np.float64)
        for column, name in ((1, "reset_local_y"), (0, "reset_local_x_fallback")):
            candidate = np.asarray(reset_rotation[:, column], dtype=np.float64).copy()
            candidate -= world_up * float(np.dot(candidate, world_up))
            norm = float(np.linalg.norm(candidate))
            if norm >= 0.25:
                self._reset_planar_jaw_axis_world = candidate / norm
                self._reset_planar_jaw_axis_source = name
                break
        if self._reset_planar_jaw_axis_world is None:
            # A valid rotation cannot have both local X and local Y parallel
            # to world-up.  Keep the invariant explicit so malformed reset
            # proprioception cannot silently inject an arbitrary grasp yaw.
            raise GraspBindingError(
                "reset-time EE pose has no reliable planar wrist heading"
            )

    def reset_goal_context(self) -> None:
        """Forget grasp failures and frozen geometry from the previous goal.

        Candidate ids such as ``analytic-top-0`` describe a proposal role,
        not an episode-global physical object.  Their recovery state must
        persist across retries of one atomic goal but must never bias the next
        source in a compound instruction.
        """

        self._contact_completion.clear()
        self._contact_residual_m.clear()
        self._failed_candidates.clear()
        self._candidate_failure_keys_by_id.clear()
        self._marginal_reacquire_candidates.clear()
        self._initial_cavity_approach_world = None
        self._initial_cavity_lateral_axis_world = None
        self._initial_cavity_source_center_world = None
        self._initial_cavity_rim_radius_m = None
        self._initial_cavity_rim_direction_world = None
        self._initial_free_space_rim_direction_world = None
        self._cavity_rim_execution_profiles.clear()
        self.last_proposal_trace = {}

    def cavity_rim_execution_profile(
        self, candidate_id: str
    ) -> Mapping[str, object] | None:
        """Return defensive sensor geometry for a typed recovery profile."""

        profile = self._cavity_rim_execution_profiles.get(str(candidate_id))
        if profile is None:
            return None
        return {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in profile.items()
        }

    def allows_contact_completion(self, candidate_id: str) -> bool:
        return bool(self._contact_completion.get(candidate_id, False))

    def contact_completion_residual_m(self, candidate_id: str) -> float | None:
        return self._contact_residual_m.get(candidate_id)

    def mark_candidate_failed(self, candidate_id: str) -> None:
        """De-prioritise a sensor grasp that failed the post-lift check."""

        if candidate_id:
            candidate_id = str(candidate_id)
            failure_key = self._candidate_failure_keys_by_id.get(
                candidate_id,
                self._candidate_failure_key(candidate_id),
            )
            self._failed_candidates.add(failure_key)

    def mark_free_space_environment_blocked(self) -> None:
        """Suppress both rim walls after a candidate-independent start block."""

        self._failed_candidates.update(
            ("free-rim:primary", "free-rim:antipodal")
        )

    @staticmethod
    def _candidate_failure_key(candidate_id: str) -> str:
        """Return the legacy failure key for a candidate without role context.

        Contextual cavity proposals install a stronger primary/antipodal key
        in ``_candidate_failure_keys_by_id``.  The sign-canonical form remains
        as a conservative fallback for callers that supply a legacy cavity id
        without first requesting a contextual proposal.  Ordinary/free-space
        ids remain exact.
        """

        candidate_id = str(candidate_id)
        if candidate_id.startswith("analytic-cavity-rim-"):
            return candidate_id.replace(
                "-positive-", "-physical-side-"
            ).replace("-negative-", "-physical-side-")
        return candidate_id

    @staticmethod
    def _is_pan_handle_source(label: str) -> bool:
        normalized = " ".join(label.lower().replace("_", " ").split())
        return normalized in {"frying pan", "fry pan", "skillet"}

    def free_space_rim_geometry_evidence(
        self, source: SceneEntity
    ) -> dict[str, object]:
        """Classify a fresh bowl OBB before any physical edge is consumed.

        A free-space bowl is a shallow, approximately axisymmetric vessel in
        its sensor OBB.  A tall or strongly elliptical box is normally an
        occluded completion containing the robot/drawer rather than a usable
        rim.  This gate uses only the current RGB-D entity geometry and
        intentionally runs before candidate ids or failure keys are created.
        """

        vertical_axis = int(np.argmax(np.abs(source.pose[2, :3])))
        planar_axes = tuple(index for index in range(3) if index != vertical_axis)
        planar_values = np.asarray(source.extent[list(planar_axes)], dtype=np.float64)
        planar_min = float(np.min(planar_values))
        planar_max = float(np.max(planar_values))
        vertical_span = float(
            np.abs(source.pose[:3, :3][2]) @ source.extent
        )
        planar_axis_ratio = planar_min / max(planar_max, 1e-12)
        height_ratio = vertical_span / max(planar_min, 1e-12)
        top_z = float(source.position[2] + 0.5 * vertical_span)
        accepted = bool(
            np.isfinite(planar_axis_ratio)
            and np.isfinite(height_ratio)
            and planar_axis_ratio
            >= self.free_space_rim_min_planar_axis_ratio
            and height_ratio <= self.free_space_rim_max_height_ratio
        )
        return {
            "accepted": accepted,
            "source_id": source.instance_id,
            "source_center_world_m": source.position.tolist(),
            "vertical_axis_index": vertical_axis,
            "vertical_span_m": vertical_span,
            "planar_min_diameter_m": planar_min,
            "planar_max_diameter_m": planar_max,
            "planar_axis_ratio": planar_axis_ratio,
            "minimum_planar_axis_ratio": (
                self.free_space_rim_min_planar_axis_ratio
            ),
            "height_to_planar_diameter_ratio": height_ratio,
            "maximum_height_ratio": self.free_space_rim_max_height_ratio,
            "rim_radius_m": 0.5 * planar_min,
            "top_z_m": top_z,
            "geometry_source": "fresh_dual_rgbd_obb",
        }

    def _world_up_top_down_base(
        self,
        current_world_from_ee: object,
    ) -> tuple[np.ndarray, str]:
        """Level a grasp without inheriting pose changes from a prior skill.

        Panda's fingers close along local Y.  Both the planar jaw heading and
        the vertical tool hemisphere are frozen from reset-time public
        proprioception.  A drawer or door skill can rotate local Y by roughly
        ninety degrees before a later object grasp; inheriting that contact
        yaw made an otherwise top-down rim descent stall.  The frozen episode
        heading gives independent object grasps the same collision geometry
        before and after a fixture skill.  Pan and cavity proposals still
        align this base to their freshly sensed handle / opening axes.  No
        task, simulator, or evaluator state participates in this construction.
        """

        self.freeze_reset_pose()
        assert self._reset_tool_z_hemisphere_sign is not None
        assert self._reset_planar_jaw_axis_world is not None
        assert self._reset_planar_jaw_axis_source is not None
        current = np.asarray(current_world_from_ee, dtype=np.float64)
        if current.shape != (4, 4) or not np.all(np.isfinite(current)):
            raise GraspBindingError(
                "top-down grasp requires a finite 4x4 proprioceptive EE pose"
            )
        if not np.allclose(current[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
            raise GraspBindingError(
                "top-down grasp received an invalid homogeneous EE pose"
            )
        rotation = current[:3, :3]
        if not np.allclose(
            rotation.T @ rotation, np.eye(3), atol=2e-3
        ) or not np.isclose(
            np.linalg.det(rotation),
            1.0,
            atol=2e-3,
        ):
            raise GraspBindingError(
                "top-down grasp requires a right-handed proprioceptive rotation"
            )

        world_up = np.array((0.0, 0.0, 1.0), dtype=np.float64)
        jaw_axis = self._reset_planar_jaw_axis_world.copy()
        reference_name = f"episode_{self._reset_planar_jaw_axis_source}"
        tool_z = self._reset_tool_z_hemisphere_sign * world_up
        tool_x = np.cross(jaw_axis, tool_z)
        tool_x /= float(np.linalg.norm(tool_x))
        jaw_axis = np.cross(tool_z, tool_x)
        jaw_axis /= float(np.linalg.norm(jaw_axis))
        result = current.copy()
        result[:3, :3] = np.column_stack((tool_x, jaw_axis, tool_z))
        return result, reference_name

    def _propose_pan_handle(
        self,
        source: SceneEntity,
        scene: SceneEstimate,
        current: np.ndarray,
    ) -> tuple[GraspCandidate, ...]:
        """Build object-aligned grasps from fresh public RGB-D pan surfaces."""

        points = source.surface_points_world
        if points is None:
            raise GraspBindingError(
                "sensor-only frying-pan handle inference requires a fresh "
                "RGB-D surface component"
            )
        top_down_base, top_down_reference = self._world_up_top_down_base(current)
        try:
            affordance = infer_pan_handle_affordance(points, top_down_base)
        except (PanAffordanceError, ValueError) as exc:
            raise GraspBindingError(
                f"sensor-only frying-pan handle inference failed: {exc}"
            ) from exc

        aligned = align_panda_finger_axis(
            top_down_base,
            affordance.jaw_axis_world,
        )
        candidates: list[GraspCandidate] = []
        candidate_trace: list[dict[str, object]] = []
        # For an object whose mass is concentrated in the pan body, a safe
        # solid slot nearer the neck has the shortest gravitational load arm.
        # Keep physically distinct sensor slots, but make that general
        # mechanics prior strong enough that an SDF bonus cannot promote a
        # distal, high-torque grip over it.
        ordered_slots = tuple(
            sorted(
                affordance.slots,
                key=lambda slot: (slot.longitudinal_fraction, -slot.score),
            )
        )
        mechanical_scores = (0.96, 0.68, 0.40, 0.12)
        for index, slot in enumerate(ordered_slots):
            pose = aligned.copy()
            pose[:3, 3] = slot.position_world
            pose[2, 3] = max(
                slot.local_top_z_m - self.finger_pad_top_inset_m,
                scene.scene_floor_z + self.finger_plane_floor_clearance_m,
            )
            candidate_id = f"analytic-pan-{slot.slot_id}"
            failure_key = f"pan:{slot.slot_id}"
            self._candidate_failure_keys_by_id[candidate_id] = failure_key
            failed = failure_key in self._failed_candidates
            base_score = mechanical_scores[min(index, len(mechanical_scores) - 1)]
            candidate_score = float(base_score if not failed else 0.0)
            self._contact_completion[candidate_id] = True
            self._contact_residual_m[candidate_id] = 0.012
            candidates.append(
                GraspCandidate(
                    candidate_id=candidate_id,
                    object_id=source.instance_id,
                    world_from_ee=pose,
                    approach_world=np.array((0.0, 0.0, -1.0)),
                    score=candidate_score,
                    clearance_m=0.04,
                    gripper_width_m=float(
                        np.clip(slot.local_width_m + 0.025, 0.03, 0.079)
                    ),
                )
            )
            candidate_trace.append(
                {
                    "candidate_id": candidate_id,
                    "slot_id": slot.slot_id,
                    "target_world_m": pose[:3, 3].tolist(),
                    "longitudinal_fraction": slot.longitudinal_fraction,
                    "local_top_z_m": slot.local_top_z_m,
                    "local_width_m": slot.local_width_m,
                    "center_occupancy_gap_m": slot.center_gap_m,
                    "sensor_score": slot.score,
                    "mechanical_load_rank": index,
                    "mechanical_base_score": base_score,
                    "previously_failed": failed,
                }
            )
        if not candidates:
            raise GraspBindingError(
                "sensor-only frying-pan handle inference returned no solid slots"
            )
        self._record_pan_handle_trace(
            source,
            scene,
            points,
            affordance,
            aligned,
            candidate_trace,
            top_down_reference,
        )
        return tuple(candidates)

    def _record_pan_handle_trace(
        self,
        source: SceneEntity,
        scene: SceneEstimate,
        points: np.ndarray,
        affordance: PanHandleAffordance,
        aligned_pose: np.ndarray,
        candidates: Sequence[Mapping[str, object]],
        top_down_reference: str,
    ) -> None:
        """Expose only reproducible language/RGB-D/proprioception evidence."""

        self.last_proposal_trace = {
            "source_id": source.instance_id,
            "source_label": source.label,
            "source_position_m": source.position.tolist(),
            "source_extent_m": source.extent.tolist(),
            "grasp_mode": GraspMode.PINCH.value,
            "scene_floor_z_m": scene.scene_floor_z,
            "pan_handle_strategy": "rgbd_wide_body_unique_narrow_tail",
            "pan_surface_point_count": len(points),
            "pan_handle_axis_world": affordance.handle_axis_world.tolist(),
            "pan_jaw_axis_world": affordance.jaw_axis_world.tolist(),
            "pan_aligned_local_y_world": aligned_pose[:3, 1].tolist(),
            "pan_body_center_world_m": affordance.body_center_world.tolist(),
            "pan_handle_start_world_m": affordance.handle_start_world.tolist(),
            "pan_handle_end_world_m": affordance.handle_end_world.tolist(),
            "pan_body_diameter_m": affordance.body_diameter_m,
            "pan_handle_length_m": affordance.handle_length_m,
            "pan_median_handle_width_m": affordance.median_handle_width_m,
            "pan_tail_bin_coverage": affordance.tail_bin_coverage,
            "pan_narrow_bin_fraction": affordance.narrow_bin_fraction,
            "pan_candidate_slots": [dict(item) for item in candidates],
            "pan_slot_order_strategy": "proximal_low_load_arm_then_sensor_score",
            "primary_ee_z_m": float(candidates[0]["target_world_m"][2]),
            "orientation_source": (
                "rgbd_handle_axis_reset_hemisphere_top_down_with_proprio_shortest_sign"
            ),
            "top_down_reference_axis": top_down_reference,
            "reset_tool_z_hemisphere_sign": self._reset_tool_z_hemisphere_sign,
            "reset_tool_z_world_up_dot": self._reset_tool_z_world_up_dot,
        }

    @staticmethod
    def _cavity_profile_names(
        yaw_offset_rad: float,
        z_offset_m: float,
    ) -> tuple[str, str]:
        """Return stable height/yaw labels shared by ids and failure keys."""

        height_name = "z0" if np.isclose(z_offset_m, 0.0) else "zminus4"
        yaw_degrees = int(round(np.rad2deg(yaw_offset_rad)))
        yaw_name = (
            f"yawplus{yaw_degrees}"
            if yaw_degrees > 0
            else f"yawminus{abs(yaw_degrees)}"
            if yaw_degrees < 0
            else "yaw0"
        )
        return height_name, yaw_name

    @classmethod
    def _cavity_physical_failure_key(
        cls,
        physical_role: str,
        yaw_offset_rad: float,
        z_offset_m: float,
        profile_kind: str = "standard",
    ) -> str:
        """Key a profile by its frozen primary/antipodal physical role."""

        if physical_role not in {"primary", "antipodal"}:
            raise ValueError(f"unknown cavity physical role {physical_role!r}")
        if profile_kind != "standard":
            return f"cavity:{physical_role}:{profile_kind}"
        height_name, yaw_name = cls._cavity_profile_names(
            yaw_offset_rad,
            z_offset_m,
        )
        return f"cavity:{physical_role}:{height_name}:{yaw_name}"

    def _cavity_candidate_profiles(
        self, selected_rim_approach_dot: float
    ) -> tuple[tuple[float, float, float], ...]:
        """Order cavity poses by the sensed rim side relative to approach."""

        yaw_minus_10 = -np.deg2rad(10.0)
        yaw_plus_10 = np.deg2rad(10.0)
        tilted_minus = (
            yaw_minus_10,
            -0.004,
            self.cavity_outer_finger_lift_rad,
        )
        tilted_plus = (
            yaw_plus_10,
            -0.004,
            self.cavity_outer_finger_lift_rad,
        )
        nominal = (0.0, 0.0, 0.0)
        # A selected rim pointing back toward the initial EE is the near side:
        # use the least articulated pose before trying tilted yaws beside the
        # fixture.  A far-side rim has free approach space behind the centre,
        # where the calibrated yaw/tilt profiles remain the stronger contact.
        return (
            (nominal, tilted_minus, tilted_plus)
            if selected_rim_approach_dot < 0.0
            else (tilted_minus, tilted_plus, nominal)
        )

    def reject_marginal_before_lift(self, candidate_id: str) -> bool:
        """Whether a marginal close should trigger a full free-space reacquire.

        Free-space rim candidates can safely retry the opposite wall.  Cavity
        candidates deliberately do not opt in: fixture clearance may leave
        only one reachable side, and their existing post-LIFT blocked-width
        check remains the hard retention gate.
        """

        return candidate_id in self._marginal_reacquire_candidates

    @staticmethod
    def requires_full_reacquire(candidate_id: str) -> bool:
        """Whether a failed grasp must return through a safe pregrasp.

        Switching between opposite rim walls at grasp height creates a low
        lateral sweep through the bowl (and through a drawer for contextual
        candidates).  Every analytic rim candidate is therefore retried only
        by the controller's task-level retreat/re-observe/APPROACH path, which
        also re-establishes its measured 20-mm pre-shape safely above the rim.
        """

        candidate_id = str(candidate_id)
        return candidate_id.startswith(
            (
                "analytic-cavity-rim-",
                "analytic-rim-",
                "analytic-pan-handle-",
            )
        )

    def propose(self, scene: SceneEstimate, object_id: str) -> Sequence[GraspCandidate]:
        source = scene.by_id(object_id)
        current = np.asarray(self.pose_provider(), dtype=np.float64)
        top_down_base, top_down_reference = self._world_up_top_down_base(current)
        grasp_mode = self.grasp_mode_selector.select(source.label)
        if grasp_mode == GraspMode.RIM_PINCH:
            rim_geometry = self.free_space_rim_geometry_evidence(source)
            provisional_radius = (
                float(rim_geometry["rim_radius_m"])
                - self.rim_pinch_radial_inset_m
            )
            radius_is_calibrated = bool(
                self.rim_pinch_min_radius_m
                <= provisional_radius
                <= self.rim_pinch_max_radius_m
            )
            if radius_is_calibrated and not bool(rim_geometry["accepted"]):
                # Publish the rejection without allocating candidate ids or
                # physical failure keys.  The contact-aware controller may
                # now move the empty arm for a fresh dual-view reconstruction;
                # no bad near/antipodal edge has been physically attempted.
                self.last_proposal_trace = {
                    "grasp_mode": GraspMode.RIM_PINCH.value,
                    "rim_geometry_gate": rim_geometry,
                    "free_space_failed_physical_edges": sorted(
                        key
                        for key in self._failed_candidates
                        if key.startswith("free-rim:")
                    ),
                }
                raise GraspBindingError(
                    "free-space rim requires shallow axisymmetric fresh "
                    "RGB-D geometry"
                )
        if (
            grasp_mode == GraspMode.PINCH
            and self._is_pan_handle_source(source.label)
        ):
            return self._propose_pan_handle(source, scene, current)
        width_aligned = False
        if grasp_mode == GraspMode.PINCH:
            vertical_axis = int(np.argmax(np.abs(source.pose[2, :3])))
            planar_axes = [axis for axis in range(3) if axis != vertical_axis]
            narrow_axis = min(planar_axes, key=lambda axis: source.extent[axis])
            short_side = float(source.extent[narrow_axis])
            long_side = float(max(source.extent[axis] for axis in planar_axes))
            if long_side >= .055 and long_side >= 1.4 * short_side:
                jaw_axis = source.pose[:3, narrow_axis].copy()
                jaw_axis[2] = 0.0
                norm = float(np.linalg.norm(jaw_axis))
                if norm >= .8:
                    jaw_axis /= norm
                    if float(jaw_axis @ top_down_base[:3, 1]) < 0.0:
                        jaw_axis *= -1.0
                    top_down_base[:3, 1] = jaw_axis
                    top_down_base[:3, 0] = np.cross(jaw_axis, top_down_base[:3, 2])
                    width_aligned = True
        # For Panda pinch grasps the finger closing plane must cross the object
        # body.  The visible top keypoint is useful for pre-grasp clearance but
        # would close above short packages, so proposals are centred on the
        # sensor-estimated object centroid.  The protected bowl branch below
        # then replaces XY with an external rim target; it never uses the
        # explicitly injected EXPAND compatibility mode.
        grasp_point = np.array(source.position, dtype=np.float64, copy=True)
        vertical_half_extent = float(
            np.abs(source.pose[:3, :3][2]) @ (source.extent / 2.0)
        )
        bottom_z = source.position[2] - vertical_half_extent
        top_z = source.position[2] + vertical_half_extent
        expansion_source = grasp_mode == GraspMode.EXPAND
        rim_pinch_source = grasp_mode == GraspMode.RIM_PINCH
        # This is the OSC EE target itself.  For a thin supported package the
        # Panda stops roughly 30 mm above it on table contact, so adding the
        # 35-mm tool offset again would close entirely above the object.  The
        # same sensor formula is shared with the validated Route-B grasp:
        # slightly inside the visible top and never below support+12 mm.
        # Tall narrow bodies use an upper-body grasp to clear their top;
        # other objects retain the calibrated centre+35 mm upper bound.
        primary_top_inset_m = (
            self.free_space_rim_grasp_top_inset_m
            if rim_pinch_source
            else self.finger_pad_top_inset_m
        )
        world_span = np.abs(source.pose[:3, :3]) @ source.extent
        tall_narrow = (
            not rim_pinch_source
            and not expansion_source
            and world_span[2] > 2.0 * float(np.max(world_span[:2]))
        )
        if tall_narrow:
            primary_top_inset_m = max(
                primary_top_inset_m, min(0.030, 0.15 * float(world_span[2]))
            )
        primary_z = min(
            max(
                top_z - primary_top_inset_m,
                bottom_z + self.finger_plane_floor_clearance_m,
            ),
            top_z - primary_top_inset_m
            if tall_narrow or (grasp_mode == GraspMode.PINCH and source.label == "moka pot")
            else source.position[2] + 0.035,
        )
        flat_pinch_pitch = 0.0
        if (
            grasp_mode == GraspMode.PINCH and width_aligned
            and world_span[2] <= .040 and short_side <= .055
        ):
            # A shallow object on a low support can drive the Panda wrist
            # into a folded joint limit with a vertical tool. Incline around
            # the closing axis while leaving the two pad centres level.
            flat_pinch_pitch = -np.deg2rad(20.)
            cosine, sine = np.cos(flat_pinch_pitch), np.sin(flat_pinch_pitch)
            top_down_base[:3, :3] = top_down_base[:3, :3] @ np.array(
                ((cosine, 0., sine), (0., 1., 0.), (-sine, 0., cosine))
            )
            primary_z += .004
        self.last_proposal_trace = {
            "source_id": source.instance_id,
            "source_label": source.label,
            "source_position_m": source.position.tolist(),
            "source_extent_m": source.extent.tolist(),
            "grasp_mode": grasp_mode.value,
            "observed_bottom_z_m": bottom_z,
            "observed_top_z_m": top_z,
            "primary_ee_z_m": primary_z,
            "scene_floor_z_m": scene.scene_floor_z,
            "orientation_source": (
                "reset_hemisphere_top_down_from_public_proprioception"
            ),
            "top_down_reference_axis": top_down_reference,
            "reset_tool_z_hemisphere_sign": self._reset_tool_z_hemisphere_sign,
            "reset_tool_z_world_up_dot": self._reset_tool_z_world_up_dot,
            "pinch_aligned_to_observed_width": width_aligned,
            "flat_pinch_pitch_rad": float(flat_pinch_pitch),
        }
        if rim_pinch_source:
            self.last_proposal_trace.update(
                {
                    "rim_height_strategy": "sensor_completed_top_surface",
                    "free_space_rim_grasp_top_inset_m": (
                        self.free_space_rim_grasp_top_inset_m
                    ),
                    "rim_height_from_sensor_top_m": primary_z - top_z,
                }
            )
        candidates: list[GraspCandidate] = []
        # Four-millimetre AABB-relative alternatives handle depth / support
        # bias without jumping entirely across a thin package's closing band.
        # This spacing is deliberately sensor-relative rather than task-specific.
        # An explicitly injected non-bowl EXPAND mechanism tries z0, -4 mm,
        # then +4 mm.  Free-space
        # RIM_PINCH diversity comes from its measured near/opposite rim side;
        # every retry therefore stays at the independently calibrated RGB-D
        # top-minus-10-mm plane rather than mixing side and height changes.
        offsets = (
            (0.0, -0.004, 0.004)
            if expansion_source
            else (
                (0.0, 0.0, 0.0)
                if rim_pinch_source
                else ((0.0, -0.004, 0.004) if tall_narrow else (0.0, 0.004, -0.004))
            )
        )
        if expansion_source:
            self.last_proposal_trace["expansion_candidate_ee_z_m"] = [
                source.position[2] + offset for offset in offsets
            ]
        finger_axis_xy: np.ndarray | None = None
        rim_radius_m: float | None = None
        near_sign = 1.0
        if rim_pinch_source:
            # Panda's pads move along local Y in the public raw EEF-body
            # frame.  Project that measured axis onto the table so one open
            # finger descends inside the bowl and the other outside.
            finger_axis_xy = np.array(
                top_down_base[:2, 1], dtype=np.float64, copy=True
            )
            axis_norm = float(np.linalg.norm(finger_axis_xy))
            if axis_norm < 0.5:
                raise GraspBindingError(
                    "Panda local-Y finger axis is not sufficiently planar for rim pinch"
                )
            finger_axis_xy /= axis_norm
            vertical_axis = int(np.argmax(np.abs(source.pose[2, :3])))
            planar_axes = tuple(index for index in range(3) if index != vertical_axis)
            planar_diameter = float(np.min(source.extent[list(planar_axes)]))
            rim_radius_m = planar_diameter / 2.0 - self.rim_pinch_radial_inset_m
            if not self.rim_pinch_min_radius_m <= rim_radius_m <= self.rim_pinch_max_radius_m:
                raise GraspBindingError(
                    "sensor-derived rim radius "
                    f"{rim_radius_m:.4f} m is outside calibrated bounds "
                    f"[{self.rim_pinch_min_radius_m:.4f}, "
                    f"{self.rim_pinch_max_radius_m:.4f}] m"
                )
            center_to_ee_xy = current[:2, 3] - source.position[:2]
            near_sign = (
                1.0
                if float(np.dot(center_to_ee_xy, finger_axis_xy)) >= 0.0
                else -1.0
            )
            first_radial = np.array(
                (
                    near_sign * finger_axis_xy[0],
                    near_sign * finger_axis_xy[1],
                    0.0,
                ),
                dtype=np.float64,
            )
            if self._initial_free_space_rim_direction_world is None:
                self._initial_free_space_rim_direction_world = (
                    first_radial.copy()
                )
            self.last_proposal_trace.update(
                {
                    "rim_radius_m": rim_radius_m,
                    "rim_planar_diameter_m": planar_diameter,
                    "finger_axis_world": [
                        float(finger_axis_xy[0]),
                        float(finger_axis_xy[1]),
                        0.0,
                    ],
                    "rim_candidate_sides": ["near", "opposite", "near"],
                }
            )
        rim_targets: list[list[float]] = []
        for index, z_offset in enumerate(offsets):
            pose = top_down_base.copy()
            pose[:3, 3] = grasp_point
            if rim_pinch_source:
                assert finger_axis_xy is not None and rim_radius_m is not None
                use_near_side = index != 1
                side_sign = near_sign if use_near_side else -near_sign
                pose[:2, 3] = (
                    source.position[:2]
                    + side_sign * finger_axis_xy * rim_radius_m
                )
            pose[2, 3] = (
                source.position[2] + z_offset
                if expansion_source
                else primary_z + z_offset
            )
            width = float(np.clip(max(source.extent[:2]) * 0.8, 0.01, 0.079))
            candidate_id = (
                f"analytic-rim-{index}"
                if rim_pinch_source
                else f"analytic-top-{index}"
            )
            failure_key = self._candidate_failure_key(candidate_id)
            if rim_pinch_source:
                rim_targets.append(pose[:3, 3].tolist())
                self._marginal_reacquire_candidates.add(candidate_id)
                assert finger_axis_xy is not None and rim_radius_m is not None
                use_near_side = index != 1
                side_sign = near_sign if use_near_side else -near_sign
                radial_direction = np.array(
                    (
                        side_sign * finger_axis_xy[0],
                        side_sign * finger_axis_xy[1],
                        0.0,
                    ),
                    dtype=np.float64,
                )
                assert self._initial_free_space_rim_direction_world is not None
                physical_role = (
                    "primary"
                    if float(
                        np.dot(
                            radial_direction,
                            self._initial_free_space_rim_direction_world,
                        )
                    )
                    >= 0.0
                    else "antipodal"
                )
                failure_key = f"free-rim:{physical_role}"
                self._candidate_failure_keys_by_id[candidate_id] = failure_key
                observed_radius = planar_diameter / 2.0
                # Preserve the frozen RGB-D centre and physical side, but do
                # not move laterally after a non-marginal close.  Production
                # uses an external CLOSE_FINGERS rim pinch followed by a
                # strict 5--8-mm vertical load proof; it never expands inside
                # the bowl and never seats outward across the thin wall.
                self._cavity_rim_execution_profiles[candidate_id] = {
                    "candidate_id": candidate_id,
                    "source_center_world_m": source.position.copy(),
                    "radial_direction_world": radial_direction,
                    "observed_rim_radius_m": observed_radius,
                    "target_radius_m": rim_radius_m,
                    "profile_kind": "free_space_rim_direct_load",
                }
                if failure_key in self._failed_candidates:
                    self._cavity_rim_execution_profiles.pop(candidate_id, None)
                    continue
            # Successful sensor-only Object grasps complete within 12 mm of
            # the commanded pose.  A broader thin-object allowance admitted a
            # table-edge block that disappeared during lift, so keep the same
            # strict proprioceptive residual for every pinch candidate.
            self._contact_completion[candidate_id] = True
            self._contact_residual_m[candidate_id] = (
                0.014 if rim_pinch_source else 0.012
            )
            candidates.append(
                GraspCandidate(
                    candidate_id=candidate_id,
                    object_id=source.instance_id,
                    world_from_ee=pose,
                    approach_world=np.array((0.0, 0.0, -1.0)),
                    score=(
                        0.9 - 0.05 * index
                        if failure_key not in self._failed_candidates
                        else 0.1 - 0.01 * index
                    ),
                    clearance_m=0.04,
                    gripper_width_m=width,
                )
            )
        if rim_pinch_source:
            self.last_proposal_trace["rim_candidate_targets_world_m"] = rim_targets
            self.last_proposal_trace["free_space_failed_physical_edges"] = sorted(
                key
                for key in self._failed_candidates
                if key.startswith("free-rim:")
            )
            if not candidates:
                raise GraspBindingError(
                    "all sensor-bound free-space physical rim edges failed"
                )
        return tuple(candidates)

    def propose_with_reference(
        self,
        scene: SceneEstimate,
        object_id: str,
        reference: SceneEntity,
    ) -> Sequence[GraspCandidate]:
        """Propose a fixture-width rim pinch for a bowl inside a cavity.

        The reference is the RGB-D entity that satisfied the source's ``IN``
        selector.  A drawer or cabinet supplies an observable opening frame:
        aligning Panda local-Y with the horizontal OBB line most parallel to
        the initial EE-to-bowl bearing prevents both fingers from descending
        into the drawer's front/back walls.  All semantic and geometric gates
        are intentionally local to public proprioception and the current
        sensor scene; an absent or unreliable cavity preserves ``propose``'s
        free-space candidates byte-for-byte.
        """

        self._cavity_rim_execution_profiles.clear()
        fallback = tuple(self.propose(scene, object_id))
        source = scene.by_id(object_id)
        if self.grasp_mode_selector.select(source.label) != GraspMode.RIM_PINCH:
            return fallback
        reference_label = getattr(reference, "label", "")
        if not isinstance(reference_label, str) or not any(
            token in reference_label for token in ("drawer", "cabinet")
        ):
            return fallback

        try:
            current = np.asarray(self.pose_provider(), dtype=np.float64)
            top_down_base, top_down_reference = self._world_up_top_down_base(
                current
            )
            if self._initial_cavity_approach_world is None:
                approach = np.asarray(source.position, dtype=np.float64) - current[:3, 3]
                # Drawer disambiguation is deliberately planar.  Depth bias in
                # the visible source centroid and the initial safe EE height
                # must not affect which of the two horizontal OBB lines wins.
                approach[2] = 0.0
                approach_norm = float(np.linalg.norm(approach))
                if approach_norm < 1e-6 or not np.all(np.isfinite(approach)):
                    raise CavityGeometryError(
                        "initial EE-to-source horizontal bearing is unavailable"
                    )
                self._initial_cavity_approach_world = approach / approach_norm
            initial_approach = self._initial_cavity_approach_world.copy()
            cavity = infer_cavity_frame(
                reference,
                lateral_hint_world=initial_approach,
            )
            current_inferred_lateral = cavity.lateral_axis_world
            if self._initial_cavity_lateral_axis_world is None:
                self._initial_cavity_lateral_axis_world = (
                    current_inferred_lateral.copy()
                )
            # PCA can rotate a nearly square drawer footprint between fresh
            # RGB-D observations even when the same OBB line remains the best
            # match to the frozen approach bearing.  Keep the first measured
            # line for all retry yaws, reprojecting it onto the freshly sensed
            # top plane so a noisy normal cannot introduce vertical motion.
            stable_lateral = self._initial_cavity_lateral_axis_world.copy()
            stable_lateral -= cavity.up_axis_world * float(
                np.dot(stable_lateral, cavity.up_axis_world)
            )
            stable_lateral_norm = float(np.linalg.norm(stable_lateral))
            if stable_lateral_norm < 0.80:
                raise CavityGeometryError(
                    "initial cavity lateral axis is inconsistent with fresh top plane"
                )
            stable_lateral /= stable_lateral_norm
            aligned_pose = align_panda_finger_axis(
                top_down_base,
                stable_lateral,
            )
            finger_axis = np.array(aligned_pose[:3, 1], dtype=np.float64, copy=True)
            # A top-down rim pinch needs a horizontal closing line.  The
            # helper rejects axes parallel to tool-Z; this additional gate
            # rejects a badly tilted RGB-D fixture without changing the
            # original free-space behavior.
            horizontal_norm = float(np.linalg.norm(finger_axis[:2]))
            if horizontal_norm < 0.80:
                raise CavityGeometryError(
                    "aligned cavity finger axis is not sufficiently horizontal"
                )
            finger_axis[2] = 0.0
            finger_axis /= float(np.linalg.norm(finger_axis))

            # Recovery deliberately gets a second, current-view axis rather
            # than another yaw around the episode's frozen PCA line.  Align
            # its unsigned sign to the frozen line so "primary" continues to
            # denote the same physical rim side after a PCA sign flip.
            fresh_finger_axis = np.asarray(
                current_inferred_lateral, dtype=np.float64
            ).copy()
            fresh_finger_axis[2] = 0.0
            fresh_axis_norm = float(np.linalg.norm(fresh_finger_axis))
            if fresh_axis_norm < 0.80:
                raise CavityGeometryError(
                    "fresh cavity lateral axis is not sufficiently horizontal"
                )
            fresh_finger_axis /= fresh_axis_norm
            if float(np.dot(fresh_finger_axis, finger_axis)) < 0.0:
                fresh_finger_axis *= -1.0

            rim_radius_m = float(self.last_proposal_trace["rim_radius_m"])
            if self._initial_cavity_source_center_world is None:
                self._initial_cavity_source_center_world = source.position.copy()
            if self._initial_cavity_rim_radius_m is None:
                self._initial_cavity_rim_radius_m = rim_radius_m
            frozen_source_center = self._initial_cavity_source_center_world.copy()
            frozen_rim_radius_m = float(self._initial_cavity_rim_radius_m)
            # The recovery radius must remain inside both the first and the
            # fresh RGB-D bowl estimates.  ``rim_radius_m`` already includes
            # the ordinary 3-mm raw-radius inset; this profile moves another
            # 2 mm inward, yielding a mechanically distinct pad position.
            recovery_observed_radius_m = min(
                frozen_rim_radius_m, rim_radius_m
            )
            recovery_target_radius_m = recovery_observed_radius_m - 0.002
            if recovery_target_radius_m <= self.rim_pinch_min_radius_m - 0.010:
                raise CavityGeometryError(
                    "fresh/frozen bowl radius is too small for recovery inset"
                )
            # ``propose`` deliberately puts every free-space rim retry at the
            # deeper sensor-top-minus-10-mm plane.  A cavity has separate
            # fixture mechanics: keep its nominal base at the original
            # top-minus-6-mm level, then let the typed z-minus-4 profiles reach
            # that same 10-mm depth with yaw/outer-finger clearance.  Computing
            # this from the same RGB-D top/bottom estimates keeps the two
            # calibrations independent and avoids silently deepening cavity
            # nominal candidates.
            observed_top_z = float(self.last_proposal_trace["observed_top_z_m"])
            observed_bottom_z = float(
                self.last_proposal_trace["observed_bottom_z_m"]
            )
            free_space_primary_z = float(
                self.last_proposal_trace["primary_ee_z_m"]
            )
            primary_z = min(
                max(
                    observed_top_z - self.finger_pad_top_inset_m,
                    observed_bottom_z + self.finger_plane_floor_clearance_m,
                ),
                source.position[2] + 0.035,
            )
            recovery_z = max(
                observed_top_z - 0.012,
                observed_bottom_z + self.finger_plane_floor_clearance_m,
            )
            recovery_z_offset = recovery_z - primary_z
            self.last_proposal_trace.update(
                {
                    "free_space_primary_ee_z_m": free_space_primary_z,
                    "primary_ee_z_m": primary_z,
                    "cavity_base_grasp_top_inset_m": self.finger_pad_top_inset_m,
                    "rim_height_strategy": (
                        "cavity_sensor_top_base_with_typed_offsets"
                    ),
                    "rim_height_from_sensor_top_m": primary_z - observed_top_z,
                    "cavity_recovery_profile_kind": (
                        "fresh_axis_rinset2_zminus6"
                    ),
                    "cavity_recovery_frozen_source_center_world_m": (
                        frozen_source_center.tolist()
                    ),
                    "cavity_recovery_fresh_source_center_world_m": (
                        source.position.tolist()
                    ),
                    "cavity_recovery_source_center_shift_m": float(
                        np.linalg.norm(source.position - frozen_source_center)
                    ),
                    "cavity_recovery_frozen_rim_radius_m": (
                        frozen_rim_radius_m
                    ),
                    "cavity_recovery_fresh_rim_radius_m": rim_radius_m,
                    "cavity_recovery_observed_rim_radius_m": (
                        recovery_observed_radius_m
                    ),
                    "cavity_recovery_target_radius_m": (
                        recovery_target_radius_m
                    ),
                    "cavity_recovery_radial_extra_inset_m": 0.002,
                    "cavity_recovery_depth_from_sensor_top_m": (
                        observed_top_z - recovery_z
                    ),
                    "cavity_recovery_z_offset_from_base_m": (
                        recovery_z_offset
                    ),
                    "cavity_recovery_fresh_finger_axis_world": (
                        fresh_finger_axis.tolist()
                    ),
                    "orientation_source": (
                        "rgbd_cavity_axis_reset_hemisphere_top_down_with_proprio_shortest_sign"
                    ),
                    "top_down_reference_axis": top_down_reference,
                    "reset_tool_z_hemisphere_sign": (
                        self._reset_tool_z_hemisphere_sign
                    ),
                    "reset_tool_z_world_up_dot": (
                        self._reset_tool_z_world_up_dot
                    ),
                }
            )
            stable_depth_axis = np.cross(cavity.up_axis_world, finger_axis)
            stable_depth_axis /= float(np.linalg.norm(stable_depth_axis))
            assert reference.region is not None
            region_axes = np.asarray(reference.region.axes, dtype=np.float64)
            region_half_extents = np.asarray(
                reference.region.half_extents,
                dtype=np.float64,
            )

            def region_support(direction: np.ndarray) -> float:
                return float(
                    np.sum(
                        region_half_extents
                        * np.abs(region_axes.T @ direction)
                    )
                )

            lateral_half_extent = region_support(finger_axis)
            depth_half_extent = region_support(stable_depth_axis)
            up_half_extent = region_support(cavity.up_axis_world)
            # A useful cavity must be visibly larger than the bowl, and the
            # bowl centre must lie in (or just inside the noisy edge of) the
            # measured opening.  These gates reject bowl-sized relation crops
            # and partial vertical cabinet faces before they can rotate the
            # wrist.
            if lateral_half_extent < max(0.070, rim_radius_m + 0.008):
                raise CavityGeometryError("visible cavity is too narrow for a rim pinch")
            if depth_half_extent < rim_radius_m + 0.003:
                raise CavityGeometryError("visible cavity is too shallow for the bowl")

            source_delta = source.position - cavity.center_world
            lateral_coordinate = float(np.dot(source_delta, finger_axis))
            depth_coordinate = float(
                np.dot(source_delta, stable_depth_axis)
            )
            if abs(lateral_coordinate) > lateral_half_extent + 0.025:
                raise CavityGeometryError("bowl is outside the visible cavity width")
            if abs(depth_coordinate) > depth_half_extent + 0.025:
                raise CavityGeometryError("bowl is outside the visible cavity depth")

            wall_clearances = {
                sign: lateral_half_extent
                - rim_radius_m
                - sign * lateral_coordinate
                for sign in (1.0, -1.0)
            }
            if max(wall_clearances.values()) < 0.005:
                raise CavityGeometryError("neither sensed rim side has wall clearance")
            center_to_ee = current[:3, 3] - source.position
            near_sign = (
                1.0 if float(np.dot(center_to_ee, finger_axis)) >= 0.0 else -1.0
            )
            wall_side_order = tuple(
                sorted(
                    (1.0, -1.0),
                    key=lambda sign: (
                        -wall_clearances[sign],
                        0 if sign == near_sign else 1,
                    ),
                )
            )
        except (CavityGeometryError, KeyError, TypeError, ValueError):
            return fallback

        width = fallback[0].gripper_width_m
        def planar_axis(
            yaw_offset_rad: float,
            *,
            base_axis: np.ndarray = finger_axis,
        ) -> np.ndarray:
            perpendicular = np.cross(cavity.up_axis_world, base_axis)
            perpendicular /= float(np.linalg.norm(perpendicular))
            result = (
                np.cos(yaw_offset_rad) * base_axis
                + np.sin(yaw_offset_rad) * perpendicular
            )
            result[2] = 0.0
            result /= float(np.linalg.norm(result))
            return result

        def rim_target(
            side_sign: float,
            yaw_offset_rad: float,
            z_offset_m: float = -0.004,
            *,
            base_axis: np.ndarray = finger_axis,
            radius_m: float = rim_radius_m,
            absolute_z_m: float | None = None,
        ) -> np.ndarray:
            target = (
                source.position
                + side_sign
                * planar_axis(yaw_offset_rad, base_axis=base_axis)
                * radius_m
            )
            target = np.asarray(target, dtype=np.float64)
            target[2] = (
                primary_z + z_offset_m
                if absolute_z_m is None
                else absolute_z_m
            )
            return target

        # Preserve the original sensor-selected first candidate: the fused
        # public RGB-D SDF picks the initial physical primary side and the
        # direction is then frozen across retries.  Whether an *antipodal*
        # retry is safe is decided separately from the visible cavity OBB wall
        # clearance below; distant SDF values are not a wall-clearance proxy.
        side_probe_clearances = {
            sign: float(scene.obstacle_sdf.distance(rim_target(sign, 0.0)))
            for sign in (1.0, -1.0)
        }
        if not all(np.isfinite(tuple(side_probe_clearances.values()))):
            return fallback
        if self._initial_cavity_rim_direction_world is None:
            preferred_side_sign = max(
                wall_side_order,
                key=lambda sign: side_probe_clearances[sign],
            )
            self._initial_cavity_rim_direction_world = (
                preferred_side_sign * finger_axis
            )
        else:
            preferred_side_sign = (
                1.0
                if float(
                    np.dot(
                        self._initial_cavity_rim_direction_world,
                        finger_axis,
                    )
                )
                >= 0.0
                else -1.0
            )

        # Use the initial public-proprio bearing to distinguish the selected
        # physical near and far rim sides.  A near-side tilted wrist can catch
        # the cabinet before reaching pregrasp, so it starts with the nominal
        # cavity axis / top-minus-6-mm pose.  A far-side grasp starts with the
        # independently calibrated yaw/tilt contact profiles.  Equality is
        # assigned deterministically to the far/orthogonal branch.
        assert self._initial_cavity_rim_direction_world is not None
        selected_rim_approach_dot = float(
            np.dot(
                self._initial_cavity_rim_direction_world,
                initial_approach,
            )
        )
        candidate_profiles = self._cavity_candidate_profiles(
            selected_rim_approach_dot
        )
        candidate_ordering_reason = (
            "selected_rim_near_side_nominal_first"
            if selected_rim_approach_dot < 0.0
            else "selected_rim_far_or_orthogonal_yaw_first"
        )
        primary_side_sign = preferred_side_sign
        antipodal_side_sign = -preferred_side_sign
        primary_wall_clearance_m = float(wall_clearances[primary_side_sign])
        antipodal_wall_clearance_m = float(
            wall_clearances[antipodal_side_sign]
        )
        antipodal_nominal_promoted = bool(
            np.isfinite(antipodal_wall_clearance_m)
            and antipodal_wall_clearance_m + 1e-12
            >= self.cavity_rim_min_safe_wall_clearance_m
        )

        primary_profile_specs = [
            ("primary", primary_side_sign, *profile, "standard")
            for profile in candidate_profiles
        ]
        antipodal_profile_specs = [
            ("antipodal", antipodal_side_sign, *profile, "standard")
            for profile in candidate_profiles
        ]
        all_profile_specs = primary_profile_specs + antipodal_profile_specs
        nominal_profile_spec = next(
            spec
            for spec in antipodal_profile_specs
            if np.isclose(spec[2], 0.0)
            and np.isclose(spec[3], 0.0)
            and np.isclose(spec[4], 0.0)
            and spec[5] == "standard"
        )
        recovery_profile_spec = (
            "primary",
            primary_side_sign,
            0.0,
            recovery_z_offset,
            self.cavity_outer_finger_lift_rad,
            "fresh_axis_rinset2_zminus6",
        )
        antipodal_nominal_target_sdf_m = float(
            scene.obstacle_sdf.distance(
                rim_target(antipodal_side_sign, 0.0, 0.0)
            )
        )
        antipodal_wall_dominates_primary = bool(
            antipodal_wall_clearance_m + 1e-12
            >= primary_wall_clearance_m
        )
        antipodal_nominal_rank0_promoted = bool(
            antipodal_nominal_promoted
            and antipodal_wall_dominates_primary
            and np.isfinite(antipodal_nominal_target_sdf_m)
            and antipodal_nominal_target_sdf_m + 1e-12
            >= self.cavity_nominal_direct_min_target_sdf_m
        )
        if antipodal_nominal_rank0_promoted:
            leading_specs = (
                nominal_profile_spec,
                primary_profile_specs[0],
                primary_profile_specs[1],
            )
            candidate_profile_specs = [
                *leading_specs,
                recovery_profile_spec,
                *(
                    spec
                    for spec in all_profile_specs
                    if spec not in leading_specs
                ),
            ]
            selection_reasons = (
                "safer_antipodal_nominal_direct_first",
                "previous_primary_after_sensor_clearance_promotion",
                "same_side_yaw_compensation",
                "fresh_rgbd_deep_inset_load_recovery",
                "remaining_unique_profile",
                "remaining_unique_profile",
                "remaining_unique_profile",
            )
        elif antipodal_nominal_promoted:
            leading_specs = (
                primary_profile_specs[0],
                nominal_profile_spec,
                primary_profile_specs[1],
            )
            candidate_profile_specs = [
                *leading_specs,
                recovery_profile_spec,
                *(
                    spec
                    for spec in all_profile_specs
                    if spec not in leading_specs
                ),
            ]
            selection_reasons = (
                "retain_existing_first_choice",
                "safe_physical_side_diversity_direct_proof",
                "same_side_yaw_compensation",
                "fresh_rgbd_deep_inset_load_recovery",
                "remaining_unique_profile",
                "remaining_unique_profile",
                "remaining_unique_profile",
            )
        else:
            candidate_profile_specs = [
                *primary_profile_specs,
                recovery_profile_spec,
                *antipodal_profile_specs,
            ]
            selection_reasons = (
                "retain_existing_first_choice",
                "primary_profile_retained_antipodal_below_obb_threshold",
                "primary_profile_retained_antipodal_below_obb_threshold",
                "fresh_rgbd_deep_inset_load_recovery",
                "antipodal_tail_below_obb_threshold",
                "antipodal_tail_below_obb_threshold",
                "antipodal_tail_below_obb_threshold",
            )

        candidates: list[GraspCandidate] = []
        targets: list[list[float]] = []
        candidate_sides: list[str] = []
        candidate_physical_roles: list[str] = []
        candidate_failure_keys: list[str] = []
        candidate_base_scores: list[float] = []
        candidate_effective_scores: list[float] = []
        candidate_yaw_offsets: list[float] = []
        candidate_planar_axes: list[list[float]] = []
        candidate_finger_axes: list[list[float]] = []
        candidate_target_clearances: list[float] = []
        candidate_height_offsets: list[float] = []
        candidate_outer_finger_lifts: list[float] = []
        physical_candidate_plan: list[dict[str, Any]] = []
        candidate_ids: set[str] = set()
        # Adjacent leading scores differ by more than GraspBinder's maximum
        # 0.2 SDF bonus.  Thus the OBB-gated physical-side plan stays ordered
        # even when target SDF values differ, while failed profiles fall to
        # zero and expose the next planned physical alternative.
        base_scores = (0.96, 0.72, 0.48, 0.24, 0.0, 0.0, 0.0)
        for rank, (
            physical_role,
            side_sign,
            yaw_offset,
            z_offset,
            outer_finger_lift,
            profile_kind,
        ) in enumerate(
            candidate_profile_specs,
        ):
            score = base_scores[rank]
            side_name = "positive" if side_sign > 0.0 else "negative"
            recovery_profile = profile_kind != "standard"
            if recovery_profile:
                height_name = "zminus6"
                yaw_name = "fresh"
                candidate_id = (
                    f"analytic-cavity-rim-{side_name}-rinset2-"
                    "zminus6-fresh"
                )
            else:
                height_name, yaw_name = self._cavity_profile_names(
                    yaw_offset,
                    z_offset,
                )
                candidate_id = (
                    f"analytic-cavity-rim-{side_name}-{height_name}-{yaw_name}"
                )
            if candidate_id in candidate_ids:
                raise GraspBindingError(
                    f"duplicate contextual cavity candidate {candidate_id!r}"
                )
            candidate_ids.add(candidate_id)
            physical_failure_key = self._cavity_physical_failure_key(
                physical_role,
                yaw_offset,
                z_offset,
                profile_kind,
            )
            self._candidate_failure_keys_by_id[candidate_id] = (
                physical_failure_key
            )
            candidate_failed = physical_failure_key in self._failed_candidates
            effective_score = score if not candidate_failed else 0.0
            candidate_base_axis = (
                fresh_finger_axis if recovery_profile else finger_axis
            )
            final_planar_axis = planar_axis(
                yaw_offset, base_axis=candidate_base_axis
            )
            pose = align_panda_finger_axis(
                top_down_base,
                final_planar_axis,
            )
            # Tilt about local-X so the finger outside the bowl is visibly
            # higher and the inside finger can enter before the outer pad
            # contacts the cabinet/rim.  Which physical finger is outside is
            # derived from the sensor-selected rim side, never a task id.
            tilt = outer_finger_lift
            tilted_finger_axis = (
                np.cos(tilt) * final_planar_axis
                + side_sign * np.sin(tilt) * cavity.up_axis_world
            )
            tilted_finger_axis /= float(np.linalg.norm(tilted_finger_axis))
            tool_x = np.asarray(pose[:3, 0], dtype=np.float64).copy()
            tool_x -= tilted_finger_axis * float(
                np.dot(tool_x, tilted_finger_axis)
            )
            tool_x /= float(np.linalg.norm(tool_x))
            tool_z = np.cross(tool_x, tilted_finger_axis)
            tool_z /= float(np.linalg.norm(tool_z))
            tilted_finger_axis = np.cross(tool_z, tool_x)
            tilted_finger_axis /= float(np.linalg.norm(tilted_finger_axis))
            pose[:3, :3] = np.column_stack(
                (tool_x, tilted_finger_axis, tool_z)
            )
            pose[:3, 3] = rim_target(
                side_sign,
                yaw_offset,
                z_offset,
                base_axis=candidate_base_axis,
                radius_m=(
                    recovery_target_radius_m
                    if recovery_profile
                    else rim_radius_m
                ),
                absolute_z_m=recovery_z if recovery_profile else None,
            )
            target_clearance = float(scene.obstacle_sdf.distance(pose[:3, 3]))
            self._contact_completion[candidate_id] = True
            # Compliant drawer/rim contact can stop the EE up to 14 mm above
            # the frozen sensor goal.  This tolerance is cavity-only and the
            # subsequent close-servo plus blocked-width check remains the hard
            # grasp gate.
            self._contact_residual_m[candidate_id] = 0.014
            candidates.append(
                GraspCandidate(
                    candidate_id=candidate_id,
                    object_id=source.instance_id,
                    world_from_ee=pose,
                    approach_world=np.array((0.0, 0.0, -1.0)),
                    score=effective_score,
                    # A failed physical profile must not regain up to 0.2 in
                    # GraspBinder's SDF bonus and outrank the fourth recovery
                    # option.  Zero measured-clearance credit keeps it in the
                    # trace without allowing an accidental repeat.
                    clearance_m=0.0 if candidate_failed else 0.04,
                    gripper_width_m=width,
                )
            )
            if recovery_profile:
                radial_direction = side_sign * final_planar_axis
                self._cavity_rim_execution_profiles[candidate_id] = {
                    "candidate_id": candidate_id,
                    "profile_kind": profile_kind,
                    "source_center_world_m": source.position.copy(),
                    "radial_direction_world": radial_direction.copy(),
                    "observed_rim_radius_m": recovery_observed_radius_m,
                    "target_radius_m": recovery_target_radius_m,
                    "maximum_seat_radius_m": (
                        recovery_observed_radius_m + 0.0015
                    ),
                }
            targets.append(pose[:3, 3].tolist())
            candidate_sides.append(side_name)
            candidate_physical_roles.append(physical_role)
            candidate_failure_keys.append(physical_failure_key)
            candidate_base_scores.append(score)
            candidate_effective_scores.append(effective_score)
            candidate_yaw_offsets.append(float(yaw_offset))
            candidate_planar_axes.append(final_planar_axis.tolist())
            candidate_finger_axes.append(tilted_finger_axis.tolist())
            candidate_target_clearances.append(target_clearance)
            candidate_height_offsets.append(float(z_offset))
            candidate_outer_finger_lifts.append(float(outer_finger_lift))
            profile_name = (
                profile_kind
                if recovery_profile
                else (
                    "nominal_direct"
                    if np.isclose(yaw_offset, 0.0)
                    and np.isclose(z_offset, 0.0)
                    and np.isclose(outer_finger_lift, 0.0)
                    else f"tilted_{yaw_name}"
                )
            )
            physical_candidate_plan.append(
                {
                    "rank": rank,
                    "candidate_id": candidate_id,
                    "physical_role": physical_role,
                    "transient_side_sign": side_name,
                    "transient_axis_side": side_name,
                    "profile": profile_name,
                    "profile_id": (
                        "rinset2-zminus6-fresh"
                        if recovery_profile
                        else f"{height_name}-{yaw_name}"
                    ),
                    "recovery_profile": recovery_profile,
                    "yaw_offset_rad": float(yaw_offset),
                    "height_offset_m": float(z_offset),
                    "outer_finger_lift_rad": float(outer_finger_lift),
                    "wall_clearance_m": float(wall_clearances[side_sign]),
                    "obb_wall_clearance_m": float(wall_clearances[side_sign]),
                    "target_sdf_clearance_m": target_clearance,
                    "approach_dot": float(
                        np.dot(
                            side_sign * candidate_base_axis,
                            initial_approach,
                        )
                    ),
                    "failure_key": physical_failure_key,
                    "failed_before_proposal": candidate_failed,
                    "base_score": score,
                    "effective_score": effective_score,
                    "selection_reason": selection_reasons[rank],
                }
            )

        self.last_proposal_trace.update(
            {
                "rim_strategy": "cavity_width_axis",
                "cavity_anchor_id": reference.instance_id,
                "cavity_anchor_label": reference.label,
                "cavity_initial_ee_to_source_axis_world": initial_approach.tolist(),
                "cavity_lateral_approach_alignment_abs": abs(
                    float(np.dot(finger_axis, initial_approach))
                ),
                "cavity_typed_approach_clearance": (
                    "observed_start_distance_then_nominal"
                ),
                "cavity_outer_finger_lift_rad": (
                    self.cavity_outer_finger_lift_rad
                ),
                "cavity_grasp_levels_finger_axis_after_preclose": True,
                "cavity_current_inferred_lateral_axis_world": (
                    current_inferred_lateral.tolist()
                ),
                "cavity_base_finger_axis_world": finger_axis.tolist(),
                "cavity_selected_rim_side": (
                    "positive" if preferred_side_sign > 0.0 else "negative"
                ),
                "cavity_selected_rim_direction_world": (
                    preferred_side_sign * finger_axis
                ).tolist(),
                "cavity_selected_rim_approach_dot": (
                    selected_rim_approach_dot
                ),
                "cavity_candidate_ordering_reason": (
                    candidate_ordering_reason
                ),
                "cavity_final_candidate_ordering_reason": (
                    "roomier_antipodal_nominal_direct_first"
                    if antipodal_nominal_rank0_promoted
                    else (
                        "primary_then_safe_antipodal_nominal"
                        if antipodal_nominal_promoted
                        else "primary_profiles_before_unsafe_antipodal"
                    )
                ),
                "cavity_physical_diversity_clearance_source": (
                    "rgbd_reference_obb_wall_clearance"
                ),
                "cavity_physical_side_ranking_signal": (
                    "rgbd_obb_wall_clearance_then_target_sdf"
                ),
                "cavity_side_probe_sdf_role": (
                    "initial_primary_side_selection_and_motion_clearance_only"
                ),
                "cavity_sdf_role": (
                    "initial_primary_selection_then_nominal_direct_rank0_"
                    "and_trajectory_feasibility"
                ),
                "cavity_antipodal_promotion_uses_sdf": False,
                "cavity_nominal_direct_rank0_uses_sdf": True,
                "cavity_antipodal_nominal_promoted": (
                    antipodal_nominal_promoted
                ),
                "cavity_antipodal_nominal_min_wall_clearance_m": (
                    self.cavity_rim_min_safe_wall_clearance_m
                ),
                "cavity_antipodal_wall_clearance_m": (
                    antipodal_wall_clearance_m
                ),
                "cavity_primary_wall_clearance_m": (
                    primary_wall_clearance_m
                ),
                "cavity_antipodal_wall_dominates_primary": (
                    antipodal_wall_dominates_primary
                ),
                "cavity_nominal_direct_rank0_promoted": (
                    antipodal_nominal_rank0_promoted
                ),
                "cavity_nominal_direct_rank0_signal": (
                    "rgbd_obb_wall_clearance_plus_sensor_sdf"
                ),
                "cavity_nominal_direct_target_sdf_m": (
                    antipodal_nominal_target_sdf_m
                ),
                "cavity_nominal_direct_min_target_sdf_m": (
                    self.cavity_nominal_direct_min_target_sdf_m
                ),
                "cavity_side_probe_clearances_m": {
                    "positive_axis": side_probe_clearances[1.0],
                    "negative_axis": side_probe_clearances[-1.0],
                },
                "cavity_candidate_physical_roles": (
                    candidate_physical_roles
                ),
                "cavity_candidate_failure_keys": candidate_failure_keys,
                "cavity_candidate_base_scores": candidate_base_scores,
                "cavity_candidate_effective_scores": (
                    candidate_effective_scores
                ),
                "cavity_physical_candidate_plan": physical_candidate_plan,
                "rim_candidate_yaw_offsets_rad": candidate_yaw_offsets,
                "rim_candidate_planar_finger_axes_world": (
                    candidate_planar_axes
                ),
                "rim_candidate_finger_axes_world": candidate_finger_axes,
                "rim_candidate_outer_finger_lift_rad": (
                    candidate_outer_finger_lifts
                ),
                "rim_candidate_target_sdf_clearances_m": (
                    candidate_target_clearances
                ),
                "cavity_lateral_axis_world": finger_axis.tolist(),
                "finger_axis_world": finger_axis.tolist(),
                "cavity_half_extents_m": [
                    lateral_half_extent,
                    depth_half_extent,
                    up_half_extent,
                ],
                "cavity_source_lateral_coordinate_m": lateral_coordinate,
                "cavity_source_depth_coordinate_m": depth_coordinate,
                "cavity_wall_clearances_m": {
                    "positive_axis": float(wall_clearances[1.0]),
                    "negative_axis": float(wall_clearances[-1.0]),
                },
                "rim_candidate_sides": candidate_sides,
                "rim_candidate_height_offsets_m": candidate_height_offsets,
                "rim_candidate_targets_world_m": targets,
            }
        )
        return tuple(candidates)


class LabelCheckedGoalSynthesizer(GoalSynthesizer):
    """Reject a recycled frame-local id when its semantic label changed."""

    @staticmethod
    def _entity(scene: SceneEstimate, instance_id: str, label: str) -> SceneEntity:
        try:
            entity = scene.by_id(instance_id)
            if entity.label == label:
                return entity
        except PerceptionError:
            pass
        matches = [entity for entity in scene.entities if entity.label == label]
        if not matches:
            raise PerceptionError(
                f"tracked entity {instance_id!r} no longer matches visible label {label!r}"
            )
        return max(matches, key=lambda entity: entity.confidence)


class SensorBoundGoalSynthesizer(LabelCheckedGoalSynthesizer):
    """Keep held-object geometry attached to the measured end-effector pose.

    The first LIFT request can only occur after Route C has closed the gripper
    and confirmed a blocked width from proprioception.  At that point we bind
    the last sensor-derived source pose to the *actual* EE pose.  LIFT,
    TRANSFER, and PLACE then propagate the cached source geometry from
    proprioception instead of trusting a newly segmented object that may be
    hidden by the wrist camera or assigned a recycled frame-local id.
    """

    _HELD_PHASES = {Phase.LIFT, Phase.TRANSFER, Phase.PLACE}

    def __init__(
        self,
        track_updater: Callable[[str, SceneEntity], None] | None = None,
        track_invalidator: Callable[[], None] | None = None,
        track_discarder: Callable[[str], None] | None = None,
        *,
        container_transfer_clearance_m: float = 0.025,
        expand_grasp_z_offset_m: float = 0.0,
        rim_visual_binding_max_ee_distance_m: float = 0.070,
        rim_visual_binding_min_below_ee_m: float = 0.012,
        rim_visual_binding_stable_min_below_ee_m: float = 0.005,
        rim_visual_binding_max_shift_diameters: float = 0.35,
        rim_visual_binding_min_planar_size_ratio: float = 0.55,
        rim_visual_binding_max_planar_size_ratio: float = 1.80,
    ) -> None:
        super().__init__()
        if (
            container_transfer_clearance_m <= 0
            or expand_grasp_z_offset_m < 0
            or rim_visual_binding_max_ee_distance_m <= 0
            or rim_visual_binding_min_below_ee_m <= 0
            or rim_visual_binding_stable_min_below_ee_m <= 0
            or rim_visual_binding_stable_min_below_ee_m
            > rim_visual_binding_min_below_ee_m
            or rim_visual_binding_max_shift_diameters <= 0
            or rim_visual_binding_min_planar_size_ratio <= 0
            or rim_visual_binding_max_planar_size_ratio
            <= rim_visual_binding_min_planar_size_ratio
        ):
            raise ValueError("goal-synthesis distances must be positive")
        self._track_updater = track_updater
        self._track_invalidator = track_invalidator
        self._track_discarder = track_discarder
        self.container_transfer_clearance_m = float(container_transfer_clearance_m)
        self.expand_grasp_z_offset_m = float(expand_grasp_z_offset_m)
        self.rim_visual_binding_max_ee_distance_m = float(
            rim_visual_binding_max_ee_distance_m
        )
        self.rim_visual_binding_min_below_ee_m = float(
            rim_visual_binding_min_below_ee_m
        )
        self.rim_visual_binding_stable_min_below_ee_m = float(
            rim_visual_binding_stable_min_below_ee_m
        )
        self.rim_visual_binding_max_shift_diameters = float(
            rim_visual_binding_max_shift_diameters
        )
        self.rim_visual_binding_min_planar_size_ratio = float(
            rim_visual_binding_min_planar_size_ratio
        )
        self.rim_visual_binding_max_planar_size_ratio = float(
            rim_visual_binding_max_planar_size_ratio
        )
        self._held_object_from_ee: np.ndarray | None = None
        self._held_source: SceneEntity | None = None
        self._predicted_release_source: SceneEntity | None = None
        self._place_target: SceneEntity | None = None
        self._verified_basket_placements: list[tuple[np.ndarray, SceneEntity]] = []
        self._basket_drop_xy_world: np.ndarray | None = None
        self.last_rim_binding_trace: dict[str, Any] = {}

    @property
    def predicted_release_source(self) -> SceneEntity | None:
        return self._predicted_release_source

    @property
    def place_target(self) -> SceneEntity | None:
        return self._place_target

    @property
    def initial_source(self) -> SceneEntity | None:
        return self._held_source

    def remember_verified_basket_placement(self, source: SceneEntity, target: SceneEntity) -> None:
        """Keep a visually verified occupant for later goals in this episode."""
        if target.label != "basket":
            return
        self._verified_basket_placements = [
            (center, entity) for center, entity in self._verified_basket_placements
            if entity.instance_id != source.instance_id
        ]
        self._verified_basket_placements.append((target.position.copy(), source))

    def _basket_unoccupied_center(self, object_pose, source, target):
        center = object_pose[:3, 3].copy()
        region = target.region
        occupants = [entity for anchor, entity in self._verified_basket_placements
                     if np.linalg.norm(anchor - target.position) <= .06]
        if region is None or not occupants:
            return center
        projected_half = np.abs(region.axes.T @ object_pose[:3, :3]) @ (source.extent / 2)
        available = np.maximum(0., region.half_extents[:2] - projected_half[:2] - .012)
        candidates = [center + region.axes[:, :2] @ (available * (x, y))
                      for x in (-1., 0., 1.) for y in (-1., 0., 1.)]
        # Avoid putting the next package on top of the previous one. Choose
        # among positions whose entire measured footprint fits inside the rim.
        def score(candidate):
            separation = min(np.linalg.norm(candidate[:2] - entity.position[:2])
                             for entity in occupants)
            return (separation, -np.linalg.norm(candidate[:2] - center[:2]))
        return max(candidates, key=score)

    def refresh_rim_held_binding(
        self,
        bound: BoundConstraintGraph,
        scene: SceneEstimate,
        current_pose: np.ndarray,
        visible_instance_ids: Collection[str],
    ) -> dict[str, Any]:
        """Update held translation from a gated, fresh post-LIFT RGB-D crop.

        The nominal rigid transform remains the fallback.  A visible stable
        source (or an unambiguous same-label component near the end effector)
        must be physically below the fingers, match the bound object's planar
        scale, and cannot jump farther than 0.35 measured diameters from the
        nominal prediction.  Only the
        relative translation changes; the original sensor-bound orientation
        and dimensions remain fixed.
        """

        trace: dict[str, Any] = {
            "accepted": False,
            "reason": "held_binding_unavailable",
            "visible_source_candidates": 0,
        }
        if self._held_object_from_ee is None or self._held_source is None:
            self.last_rim_binding_trace = trace
            return trace
        current = np.asarray(current_pose, dtype=np.float64)
        if current.shape != (4, 4) or not np.all(np.isfinite(current)):
            raise ValueError("current_pose must be a finite 4x4 pose")

        visible_ids = {str(instance_id) for instance_id in visible_instance_ids}
        visible = [
            entity
            for entity in scene.entities
            if entity.instance_id in visible_ids
            and entity.label == bound.source_label
        ]
        trace["visible_source_candidates"] = len(visible)
        if not visible:
            trace["reason"] = "no_visible_same_label_source"
            self.last_rim_binding_trace = trace
            return trace

        old_binding = self._held_object_from_ee.copy()
        nominal_pose = current @ np.linalg.inv(old_binding)
        nominal_center = nominal_pose[:3, 3]
        ee_center = current[:3, 3]
        object_diameter = float(max(self._held_source.extent[:2]))
        nominal_half_extents = np.asarray(
            self._held_source.extent, dtype=np.float64
        ) / 2.0
        nominal_axes = np.asarray(nominal_pose[:3, :3], dtype=np.float64)
        nominal_vertical_half_extent = float(
            np.abs(nominal_axes[2, :]) @ nominal_half_extents
        )
        max_shift = (
            object_diameter * self.rim_visual_binding_max_shift_diameters
        )
        diagnostics: list[dict[str, Any]] = []
        valid: list[tuple[int, float, float, SceneEntity, dict[str, Any]]] = []
        for entity in visible:
            stable_source_id = entity.instance_id == bound.source_id
            center_to_ee = float(np.linalg.norm(entity.position - ee_center))
            below_ee = float(ee_center[2] - entity.position[2])
            nominal_shift = float(
                np.linalg.norm(entity.position - nominal_center)
            )
            observed_diameter = float(max(entity.extent[:2]))
            planar_size_ratio = observed_diameter / object_diameter
            observed_half_extents = np.asarray(
                entity.extent, dtype=np.float64
            ) / 2.0
            observed_axes = np.asarray(entity.pose[:3, :3], dtype=np.float64)
            observed_vertical_half_extent = float(
                np.abs(observed_axes[2, :]) @ observed_half_extents
            )
            observed_bottom_z = float(
                entity.position[2] - observed_vertical_half_extent
            )
            obb_bottom_below_ee = float(ee_center[2] - observed_bottom_z)
            obb_vertical_size_ratio = (
                observed_vertical_half_extent / nominal_vertical_half_extent
            )
            nominal_center_local_observed = (
                nominal_center - entity.position
            ) @ observed_axes
            observed_center_local_nominal = (
                entity.position - nominal_center
            ) @ nominal_axes
            nominal_center_in_observed_obb = bool(
                np.all(
                    np.abs(nominal_center_local_observed)
                    <= observed_half_extents + 1e-9
                )
            )
            observed_center_in_nominal_obb = bool(
                np.all(
                    np.abs(observed_center_local_nominal)
                    <= nominal_half_extents + 1e-9
                )
            )
            mutual_center_containment = bool(
                nominal_center_in_observed_obb
                and observed_center_in_nominal_obb
            )
            reasons: list[str] = []
            if center_to_ee > self.rim_visual_binding_max_ee_distance_m:
                reasons.append("too_far_from_ee")
            required_below_ee = (
                self.rim_visual_binding_stable_min_below_ee_m
                if stable_source_id
                else self.rim_visual_binding_min_below_ee_m
            )
            if below_ee < required_below_ee:
                reasons.append("not_below_ee")
            if nominal_shift > max_shift:
                reasons.append("nominal_shift_too_large")
            if not (
                self.rim_visual_binding_min_planar_size_ratio
                <= planar_size_ratio
                <= self.rim_visual_binding_max_planar_size_ratio
            ):
                reasons.append("planar_size_mismatch")
            # A rim crop can be top-biased by the hand: its fitted centre may
            # sit slightly above the EE even though the measured bowl volume
            # still extends well below the finger plane.  Only the exact,
            # freshly associated stable source may use this fallback, and
            # only when centre height is the sole failed legacy gate.  Mutual
            # OBB-centre containment prevents a tall/nearby same-label box
            # from manufacturing that identity.  This evidence deliberately
            # never authorises use of the crop's Z as the held COM.
            obb_bottom_fallback_attempted = bool(
                stable_source_id and reasons == ["not_below_ee"]
            )
            obb_bottom_fallback_accepted = bool(
                obb_bottom_fallback_attempted
                and obb_bottom_below_ee
                >= self.rim_visual_binding_min_below_ee_m
                and mutual_center_containment
                and self.rim_visual_binding_min_planar_size_ratio
                <= obb_vertical_size_ratio
                <= self.rim_visual_binding_max_planar_size_ratio
            )
            if obb_bottom_fallback_accepted:
                reasons.clear()
            diagnostic = {
                "instance_id": entity.instance_id,
                "stable_source_id": stable_source_id,
                "center_to_ee_m": center_to_ee,
                "below_ee_m": below_ee,
                "required_below_ee_m": required_below_ee,
                "nominal_shift_m": nominal_shift,
                "observed_planar_diameter_m": observed_diameter,
                "planar_size_ratio": planar_size_ratio,
                "observed_obb_vertical_half_extent_m": (
                    observed_vertical_half_extent
                ),
                "nominal_obb_vertical_half_extent_m": (
                    nominal_vertical_half_extent
                ),
                "obb_vertical_size_ratio": obb_vertical_size_ratio,
                "min_obb_vertical_size_ratio": (
                    self.rim_visual_binding_min_planar_size_ratio
                ),
                "max_obb_vertical_size_ratio": (
                    self.rim_visual_binding_max_planar_size_ratio
                ),
                "observed_obb_bottom_z_m": observed_bottom_z,
                "observed_obb_bottom_below_ee_m": obb_bottom_below_ee,
                "required_obb_bottom_below_ee_m": (
                    self.rim_visual_binding_min_below_ee_m
                ),
                "nominal_center_local_observed_obb_m": (
                    nominal_center_local_observed.tolist()
                ),
                "observed_center_local_nominal_obb_m": (
                    observed_center_local_nominal.tolist()
                ),
                "nominal_center_in_observed_obb": (
                    nominal_center_in_observed_obb
                ),
                "observed_center_in_nominal_obb": (
                    observed_center_in_nominal_obb
                ),
                "obb_mutual_center_containment": mutual_center_containment,
                "obb_bottom_identity_fallback_attempted": (
                    obb_bottom_fallback_attempted
                ),
                "obb_bottom_identity_fallback_accepted": (
                    obb_bottom_fallback_accepted
                ),
                "valid": not reasons,
                "rejections": reasons,
            }
            diagnostics.append(diagnostic)
            if not reasons:
                valid.append(
                    (
                        0 if stable_source_id else 1,
                        center_to_ee,
                        nominal_shift,
                        entity,
                        diagnostic,
                    )
                )
        trace["candidates"] = diagnostics
        if not valid:
            trace["reason"] = "visible_candidates_failed_geometry_gates"
            self.last_rim_binding_trace = trace
            return trace

        stable_valid = [item for item in valid if item[0] == 0]
        if len(stable_valid) == 1:
            selected_tuple = stable_valid[0]
        elif len(stable_valid) > 1 or len(valid) != 1:
            trace["reason"] = "ambiguous_visible_same_label_sources"
            trace["valid_source_ids"] = [item[3].instance_id for item in valid]
            self.last_rim_binding_trace = trace
            return trace
        else:
            selected_tuple = valid[0]
        _, center_to_ee, nominal_shift, selected, selected_diagnostic = (
            selected_tuple
        )
        # A rim-held bowl can swing along its grasp radius, while an RGB-D
        # crop partly occupied by the hand often shifts the detected centre
        # orthogonally.  Project only horizontal visual motion onto the
        # sensor-bound rim radial axis, retaining fresh measured Z.  This is
        # a mechanical constraint, not a benchmark/task prior.
        raw_offset_from_ee = selected.position - ee_center
        nominal_offset_from_ee = nominal_center - ee_center
        constrained_center = np.array(selected.position, copy=True)
        fresh_z_accepted = bool(
            not selected_diagnostic[
                "obb_bottom_identity_fallback_accepted"
            ]
            and selected_diagnostic["below_ee_m"]
            >= self.rim_visual_binding_min_below_ee_m
        )
        if not fresh_z_accepted:
            # A shallow stable crop still supplies useful identity and radial
            # XY evidence, but its top-biased depth is not safe as an object
            # COM estimate.  Preserve the rigidly propagated Z in that case.
            constrained_center[2] = nominal_center[2]
        radial_norm = float(np.linalg.norm(nominal_offset_from_ee[:2]))
        radial_axis_xy: np.ndarray | None = None
        dropped_orthogonal = np.zeros(3, dtype=np.float64)
        if radial_norm > 1e-6:
            radial_axis_xy = nominal_offset_from_ee[:2] / radial_norm
            radial_component = float(
                np.dot(raw_offset_from_ee[:2], radial_axis_xy)
            )
            constrained_offset_xy = radial_axis_xy * radial_component
            constrained_center[:2] = ee_center[:2] + constrained_offset_xy
            dropped_orthogonal[:2] = (
                raw_offset_from_ee[:2] - constrained_offset_xy
            )

        measured_pose = nominal_pose.copy()
        measured_pose[:3, 3] = constrained_center
        updated_binding = np.linalg.inv(measured_pose) @ current
        # Preserve the pre-LIFT relative rotation bit-for-bit; only the
        # translation is sensor-refreshed.
        updated_binding[:3, :3] = old_binding[:3, :3]
        self._held_object_from_ee = updated_binding
        trace.update(
            {
                "accepted": True,
                "reason": "visible_stable_source"
                if selected.instance_id == bound.source_id
                else "visible_same_label_near_ee",
                "selected_source_id": selected.instance_id,
                "selected_center_to_ee_m": center_to_ee,
                "selected_below_ee_m": selected_diagnostic["below_ee_m"],
                "selected_obb_bottom_below_ee_m": selected_diagnostic[
                    "observed_obb_bottom_below_ee_m"
                ],
                "selected_nominal_shift_m": nominal_shift,
                "identity_evidence": (
                    "stable_source_obb_bottom_mutual_center"
                    if selected_diagnostic[
                        "obb_bottom_identity_fallback_accepted"
                    ]
                    else "center_below_ee"
                ),
                "fresh_z_accepted": fresh_z_accepted,
                "z_binding_source": (
                    "fresh_rgbd" if fresh_z_accepted else "nominal_rigid"
                ),
                "nominal_center_world_m": nominal_center.tolist(),
                "measured_center_world_m": selected.position.tolist(),
                "constrained_center_world_m": constrained_center.tolist(),
                "nominal_to_measured_offset_world_m": (
                    selected.position - nominal_center
                ).tolist(),
                "nominal_to_constrained_offset_world_m": (
                    constrained_center - nominal_center
                ).tolist(),
                "rim_visual_radial_axis_world": (
                    [
                        float(radial_axis_xy[0]),
                        float(radial_axis_xy[1]),
                        0.0,
                    ]
                    if radial_axis_xy is not None
                    else None
                ),
                "rim_visual_dropped_orthogonal_world_m": (
                    dropped_orthogonal.tolist()
                ),
                "rim_visual_orthogonal_residual_m": float(
                    np.linalg.norm(dropped_orthogonal[:2])
                ),
                "relative_translation_before_m": old_binding[:3, 3].tolist(),
                "relative_translation_after_m": updated_binding[:3, 3].tolist(),
                "max_nominal_shift_m": max_shift,
                "min_planar_size_ratio": (
                    self.rim_visual_binding_min_planar_size_ratio
                ),
                "max_planar_size_ratio": (
                    self.rim_visual_binding_max_planar_size_ratio
                ),
            }
        )
        self.last_rim_binding_trace = trace
        return trace

    def reject_held_binding(self, stable_id: str) -> None:
        """Roll a proprio-propagated track back to its last RGB-D binding."""

        if self._held_source is not None and self._track_updater is not None:
            self._track_updater(
                stable_id,
                replace(self._held_source, instance_id=stable_id),
            )
        self._held_object_from_ee = None
        self._held_source = None
        self._predicted_release_source = None
        self._place_target = None
        self.last_rim_binding_trace = {}
        if self._track_invalidator is not None:
            self._track_invalidator()

    def discard_recovery_released_binding(self, stable_id: str) -> None:
        """Drop a proprio-propagated track after an abnormal held release.

        A failed PLACE is recovered by physically opening the gripper.  The
        last propagated source pose near the raised EE is no longer valid and
        must not seed the next task attempt.  Removing that one stable track
        and invalidating the RGB-D cache forces re-acquisition from visible
        sensor components without inventing a release pose.
        """

        self._held_object_from_ee = None
        self._held_source = None
        self._predicted_release_source = None
        self._place_target = None
        self.last_rim_binding_trace = {}
        if self._track_discarder is not None:
            self._track_discarder(stable_id)
        if self._track_invalidator is not None:
            self._track_invalidator()

    def motion_request(
        self,
        phase: Phase,
        bound: BoundConstraintGraph,
        scene: SceneEstimate,
        current_pose: np.ndarray,
        config: RouteCControllerConfig,
        grasp_mode: GraspMode,
    ) -> MotionRequest:
        current = np.asarray(current_pose, dtype=np.float64)
        if phase == Phase.APPROACH:
            # APPROACH is the first phase of each task attempt, including a
            # recovery retry after the controller has released the object.
            self._held_object_from_ee = None
            self._held_source = None
            self._predicted_release_source = None
            self._place_target = None
            self._basket_drop_xy_world = None
            self.last_rim_binding_trace = {}
        elif phase == Phase.LIFT and self._held_object_from_ee is None:
            source = self._entity(scene, bound.source_id, bound.source_label)
            self._held_source = source
            self._held_object_from_ee = np.linalg.inv(source.pose) @ current

        if phase in {Phase.TRANSFER, Phase.PLACE}:
            if self._place_target is None:
                self._place_target = self._entity(
                    scene, bound.target_id, bound.target_label
                )
            frozen_target = replace(
                self._place_target,
                instance_id=bound.target_id,
            )
            target_entities = tuple(
                frozen_target
                if entity.instance_id == bound.target_id
                else entity
                for entity in scene.entities
            )
            if not any(
                entity.instance_id == bound.target_id
                for entity in target_entities
            ):
                target_entities = (*target_entities, frozen_target)
            scene = replace(scene, entities=target_entities)

        if phase in self._HELD_PHASES and self._held_object_from_ee is not None:
            assert self._held_source is not None
            propagated_pose = current @ np.linalg.inv(self._held_object_from_ee)
            propagated_source = replace(
                self._held_source,
                instance_id=bound.source_id,
                pose=propagated_pose,
            )
            if self._track_updater is not None:
                # Feed the proprioceptively propagated, originally
                # sensor-bound source back into the stable visual track.  A
                # same-label component left at the pick location can no longer
                # steal the held object's stable id after release.
                self._track_updater(bound.source_id, propagated_source)
            entities = tuple(
                propagated_source if entity.instance_id == bound.source_id else entity
                for entity in scene.entities
            )
            if not any(entity.instance_id == bound.source_id for entity in entities):
                entities = (*entities, propagated_source)
            scene = replace(scene, entities=entities)
            bindings = list(bound.grasp_options)
            bindings[bound.active_grasp_index] = replace(
                bound.grasp,
                object_from_ee=self._held_object_from_ee,
            )
            bound = replace(bound, grasp_options=tuple(bindings))

        request = super().motion_request(phase, bound, scene, current, config, grasp_mode)
        if (
            phase in {Phase.TRANSFER, Phase.PLACE}
            and grasp_mode == GraspMode.PINCH
            and bound.graph.goal_relation == Relation.IN
            and bound.target_label == "basket"
            and self._held_object_from_ee is not None
        ):
            destination = self._entity(scene, bound.target_id, bound.target_label)
            if destination.region is not None:
                assert self._held_source is not None
                object_goal = request.goal_pose @ np.linalg.inv(self._held_object_from_ee)
                if self._basket_drop_xy_world is None:
                    self._basket_drop_xy_world = self._basket_unoccupied_center(
                        object_goal, self._held_source, destination
                    )[:2].copy()
                object_goal[:2, 3] = self._basket_drop_xy_world
                source_half = float(np.abs(object_goal[2, :3]) @ (self._held_source.extent / 2))
                target_half = float(np.abs(destination.region.axes[2]) @ destination.region.half_extents)
                # Release ordinary pinched payloads above the opening so the
                # palm need not enter the deep basket with the object.
                object_goal[2, 3] = destination.region.center[2] + target_half + source_half + .025
                request = replace(request, goal_pose=object_goal @ self._held_object_from_ee)
        if (
            phase == Phase.GRASP
            and grasp_mode == GraspMode.RIM_PINCH
            and bound.grasp.candidate.candidate_id.startswith(
                "analytic-cavity-rim-"
            )
        ):
            # The 10-degree approach tilt raises the finger outside the bowl
            # while the hand is still wide.  After the safe 20-mm pre-shape,
            # retaining that tilt through CLOSE leaves the two pads at
            # different heights and can produce a one-sided wedge that
            # disappears during proof-lift.  Level only the final GRASP pose:
            # planar yaw and measured translation stay frozen, while local-Y
            # becomes horizontal and local-Z keeps its original hemisphere.
            goal_pose = request.goal_pose.copy()
            world_up = np.array((0.0, 0.0, 1.0), dtype=np.float64)
            finger_axis = goal_pose[:3, 1].copy()
            finger_axis -= world_up * float(np.dot(finger_axis, world_up))
            finger_axis_norm = float(np.linalg.norm(finger_axis))
            if finger_axis_norm < 0.80:
                raise ExecutionError(
                    "cavity rim grasp finger axis cannot be levelled safely"
                )
            finger_axis /= finger_axis_norm
            tool_z_sign = (
                1.0
                if float(np.dot(goal_pose[:3, 2], world_up)) >= 0.0
                else -1.0
            )
            tool_z = tool_z_sign * world_up
            tool_x = np.cross(finger_axis, tool_z)
            tool_x /= float(np.linalg.norm(tool_x))
            finger_axis = np.cross(tool_z, tool_x)
            finger_axis /= float(np.linalg.norm(finger_axis))
            goal_pose[:3, :3] = np.column_stack(
                (tool_x, finger_axis, tool_z)
            )
            request = replace(request, goal_pose=goal_pose)
        if (
            phase == Phase.APPROACH
            and grasp_mode == GraspMode.RIM_PINCH
            and bound.grasp.candidate.candidate_id.startswith(
                "analytic-cavity-rim-"
            )
        ):
            # A neutral RGB-D component can include the wrist itself.  The
            # resulting SDF box then surrounds the measured initial EE even
            # though LIBERO's proprioceptive start state is collision-free.
            # Requiring the nominal 25-mm tool radius plus 25-mm clearance at
            # that first sample makes *every* trajectory infeasible before it
            # can move away.  For this typed cavity approach only, cap the
            # combined envelope at the currently observed positive surface
            # distance.  Each replan recomputes the cap, so the full nominal
            # envelope is restored as soon as the wrist leaves the spurious
            # component; the planned path may never reduce clearance below
            # the measured start state.
            start_surface_distance = float(
                scene.obstacle_sdf.distance(current[:3, 3])
            )
            nominal_total = request.tool_radius_m + request.clearance_m
            if (
                np.isfinite(start_surface_distance)
                and 0.0 < start_surface_distance < nominal_total
            ):
                adaptive_total = start_surface_distance
                adaptive_tool_radius = min(
                    request.tool_radius_m,
                    adaptive_total,
                )
                request = replace(
                    request,
                    tool_radius_m=adaptive_tool_radius,
                    clearance_m=adaptive_total - adaptive_tool_radius,
                )
        if phase == Phase.LIFT and self._held_object_from_ee is not None:
            assert self._held_source is not None
            target = self._entity(scene, bound.target_id, bound.target_label)
            if target.region is not None:
                target_top = target.region.center[2] + float(
                    np.abs(target.region.axes[2]) @ target.region.half_extents
                )
                source_half_height = float(
                    np.abs(self._held_source.pose[:3, :3][2])
                    @ (self._held_source.extent / 2.0)
                )
                object_at_goal = request.goal_pose @ np.linalg.inv(self._held_object_from_ee)
                required_object_z = target_top + source_half_height + self.container_transfer_clearance_m
                # Establish vertical clearance before crossing a raised rim.
                # A diagonal transfer from the ordinary 110-mm pickup lift can
                # sweep the carried bowl into an open drawer's front edge.
                goal = request.goal_pose.copy()
                goal[2, 3] += max(0.0, required_object_z - object_at_goal[2, 3])
                goal[2, 3] = max(request.goal_pose[2, 3], min(
                    goal[2, 3], config.placement_clearance_ceiling_z_m
                ))
                request = replace(request, goal_pose=goal)
        if phase == Phase.TRANSFER and self._held_object_from_ee is not None:
            assert self._held_source is not None
            target = self._entity(scene, bound.target_id, bound.target_label)
            if target.region is not None:
                target_half_z = float(
                    np.abs(target.region.axes[2]) @ target.region.half_extents
                )
                source_half_z = float(
                    np.abs(self._held_source.pose[:3, :3][2])
                    @ (self._held_source.extent / 2.0)
                )
                object_at_goal = (
                    request.goal_pose @ np.linalg.inv(self._held_object_from_ee)
                )
                rim_safe_object_z = (
                    target.region.center[2]
                    + target_half_z
                    + source_half_z
                    + self.container_transfer_clearance_m
                )
                if object_at_goal[2, 3] < rim_safe_object_z:
                    goal_pose = request.goal_pose.copy()
                    goal_pose[2, 3] += rim_safe_object_z - object_at_goal[2, 3]
                    request = replace(request, goal_pose=goal_pose)
            # LIFT, horizontal TRANSFER, then PLACE form a sensor-bound
            # three-stage motion.  If the source starts on a cabinet or in a
            # drawer, keep the measured post-lift EE clearance until it has
            # moved horizontally away; descending happens only in PLACE.
            transfer_goal = request.goal_pose.copy()
            transfer_goal[2, 3] = max(transfer_goal[2, 3], current[2, 3])
            minimum_height = max(
                current[2, 3],
                request.min_height_m if request.min_height_m is not None else current[2, 3],
            )
            request = replace(
                request,
                goal_pose=transfer_goal,
                min_height_m=minimum_height,
            )
        if phase == Phase.PLACE and self._held_object_from_ee is not None:
            assert self._held_source is not None
            predicted_pose = request.goal_pose @ np.linalg.inv(self._held_object_from_ee)
            if (
                grasp_mode == GraspMode.PINCH
                and bound.graph.goal_relation == Relation.IN
                and bound.target_label == "basket"
            ):
                destination = self._entity(scene, bound.target_id, bound.target_label)
                if destination.region is not None:
                    half_height = float(np.abs(predicted_pose[2, :3]) @ (self._held_source.extent / 2))
                    target_half = float(np.abs(destination.region.axes[2]) @ destination.region.half_extents)
                    predicted_pose[2, 3] = destination.region.center[2] - target_half + half_height + .006
            self._predicted_release_source = replace(
                self._held_source,
                instance_id=bound.source_id,
                pose=predicted_pose,
            )
            if self._place_target is None:
                self._place_target = self._entity(
                    scene, bound.target_id, bound.target_label
                )
        return request

    def _effective_object_from_ee(
        self,
        bound: BoundConstraintGraph,
        source: SceneEntity,
        grasp_mode: GraspMode,
        config: RouteCControllerConfig,
    ) -> np.ndarray:
        if self._held_object_from_ee is not None:
            return self._held_object_from_ee.copy()
        if grasp_mode != GraspMode.EXPAND:
            # Before contact the package or bowl is static.  Freeze the
            # initial sensor-derived world grasp, including a RIM_PINCH's
            # lateral offset and wrist orientation; fresh PCA axes can
            # otherwise flip by ~1 rad between APPROACH and GRASP.
            return np.linalg.inv(source.pose) @ bound.grasp.candidate.world_from_ee
        # Expansion grasps insert the closed fingers at the measured bowl
        # centre, then open against the inner wall.  The previous +25 mm
        # offset closed above shallow bowls; zero is the sensor-calibrated
        # centre-relative offset shared with the validated Spatial route.
        expand_grasp = bound.grasp.candidate.world_from_ee.copy()
        expand_grasp[:2, 3] = source.position[:2]
        # Preserve the analytic / frozen model candidate's measured depth
        # alternative.  Resetting every option to source.z made Route C retry
        # the identical empty expansion three times.  The primary candidate
        # remains exactly z=0 relative to the observed bowl centre; the two
        # sensor-scale recovery candidates are +4/-4 mm.
        expand_grasp[2, 3] += self.expand_grasp_z_offset_m
        return np.linalg.inv(source.pose) @ expand_grasp

    def _placement_ee_pose(
        self,
        bound: BoundConstraintGraph,
        source: SceneEntity,
        target: SceneEntity,
        scene: SceneEstimate,
        object_from_ee: np.ndarray,
    ) -> np.ndarray:
        target_subregion = next(
            (
                constraint.parameters.get("target_subregion")
                for constraint in bound.graph.constraints
                if constraint.kind.value == "goal_relation"
            ),
            None,
        )
        if (
            bound.graph.goal_relation == Relation.IN
            and target_subregion not in {"upper_shelf", "lower_shelf"}
        ):
            # An open basket constrains the object's horizontal footprint, not
            # its full height: bottles are intentionally allowed to protrude.
            # A shelf cavity is vertically bounded, however, so its public
            # upper/lower geometry must receive the true sensed object height.
            planar_extent = source.extent.copy()
            planar_extent[2] = 1e-6
            source = replace(source, extent=planar_extent)
        return super()._placement_ee_pose(
            bound, source, target, scene, object_from_ee
        )


class LabelCheckedGeometryVerifier(GeometryRelationVerifier):
    """Apply the same identity guard during final visual verification."""

    @staticmethod
    def _tracked_or_label(scene: SceneEstimate, instance_id: str, label: str) -> SceneEntity:
        return LabelCheckedGoalSynthesizer._entity(scene, instance_id, label)


class SensorBoundGeometryVerifier(LabelCheckedGeometryVerifier):
    """Verify a placement only after fresh RGB-D source reacquisition.

    The sensor-bound release prediction is an identity-association prior, not
    visual evidence that the object was actually released at that pose.  A
    retained stable track is likewise only a cached sensor/proprio estimate
    unless its id occurs in the observer's latest ``visible_instance_ids``.
    Consequently neither prediction nor an occluded track can produce a
    successful verification on its own.
    """

    def __init__(
        self,
        goals: SensorBoundGoalSynthesizer,
        observer: Any,
        *,
        reacquisition_distance_m: float = 0.08,
        target_drift_tolerance_m: float = 0.06,
        initial_planar_log_tolerance: float = 0.35,
        initial_height_log_tolerance: float = 0.50,
    ) -> None:
        super().__init__()
        if min(
            reacquisition_distance_m,
            target_drift_tolerance_m,
            initial_planar_log_tolerance,
            initial_height_log_tolerance,
        ) <= 0:
            raise ValueError("verification distances must be positive")
        self.goals = goals
        self.observer = observer
        self.reacquisition_distance_m = float(reacquisition_distance_m)
        self.target_drift_tolerance_m = float(target_drift_tolerance_m)
        self.initial_planar_log_tolerance = float(initial_planar_log_tolerance)
        self.initial_height_log_tolerance = float(initial_height_log_tolerance)

    def verify(self, bound: BoundConstraintGraph, scene: SceneEstimate):
        visible_ids = set(getattr(self.observer, "visible_instance_ids", ()))
        relation = bound.graph.goal_relation
        predicted = self.goals.predicted_release_source
        geometry_scene = scene
        planned_target = self.goals.place_target
        if planned_target is not None:
            visible_targets = [
                entity
                for entity in scene.entities
                if entity.instance_id in visible_ids
                and entity.label == bound.target_label
            ]
            if visible_targets:
                observed_target = min(
                    visible_targets,
                    key=lambda entity: float(
                        np.linalg.norm(entity.position - planned_target.position)
                    ),
                )
                target_drift = float(
                    np.linalg.norm(
                        observed_target.position - planned_target.position
                    )
                )
                target_drift = max(
                    target_drift,
                    float(
                        np.linalg.norm(
                            observed_target.extent - planned_target.extent
                        )
                    ),
                )
                if (
                    observed_target.region is not None
                    and planned_target.region is not None
                ):
                    target_drift = max(
                        target_drift,
                        float(
                            np.linalg.norm(
                                observed_target.region.center
                                - planned_target.region.center
                            )
                        ),
                        float(
                            np.linalg.norm(
                                observed_target.region.half_extents
                                - planned_target.region.half_extents
                            )
                        ),
                    )
                if target_drift > self.target_drift_tolerance_m:
                    return Verification(
                        False,
                        relation,
                        target_drift,
                        "fresh target drifted from the frozen transfer anchor",
                    )
            frozen_target = replace(
                planned_target,
                instance_id=bound.target_id,
            )
            target_entities = tuple(
                frozen_target
                if entity.instance_id == bound.target_id
                else entity
                for entity in scene.entities
            )
            if not any(
                entity.instance_id == bound.target_id
                for entity in target_entities
            ):
                target_entities = (*target_entities, frozen_target)
            geometry_scene = replace(scene, entities=target_entities)
        visible_source = [
            entity
            for entity in scene.entities
            if entity.instance_id in visible_ids and entity.label == bound.source_label
        ]
        initial = self.goals.initial_source
        if not visible_source:
            return Verification(
                False,
                relation,
                float("inf"),
                "source not freshly reacquired after release",
            )

        shape_consistent = [
            entity
            for entity in visible_source
            if initial is None or self._shape_matches_initial(entity, initial)
        ]
        if not shape_consistent:
            return Verification(
                False,
                relation,
                float("inf"),
                "fresh source candidates failed the initial RGB-D shape gate",
            )

        visible_stable = [
            entity
            for entity in shape_consistent
            if entity.instance_id == bound.source_id
        ]
        if visible_stable:
            selected = visible_stable[0]
            prediction_error = (
                float(np.linalg.norm(selected.position - predicted.position))
                if predicted is not None
                else 0.0
            )
            if (
                predicted is not None
                and prediction_error > self.reacquisition_distance_m
            ):
                if (
                    initial is not None
                    and float(np.linalg.norm(selected.position - initial.position))
                    <= 0.04
                ):
                    detail = "source visibly remained at pick location"
                else:
                    detail = (
                        "fresh stable source contradicts the sensor-bound "
                        "release prediction"
                    )
                return Verification(False, relation, prediction_error, detail)
            result = self._verify_geometry(
                bound,
                self._replace_source(
                    geometry_scene, bound.source_id, selected
                ),
            )
            if (
                initial is not None
                and float(np.linalg.norm(selected.position - initial.position)) <= 0.04
            ):
                movement = float(
                    np.linalg.norm(selected.position - initial.position)
                )
                return Verification(
                    False,
                    relation,
                    max(result.position_error_m, 0.04 - movement),
                    f"source visibly remained at pick location; {result.detail}",
                )
            self._remember_verified_placement(result, selected, planned_target)
            return replace(
                result,
                detail=f"visible stable source track; {result.detail}",
            )

        if predicted is None:
            candidates = shape_consistent
        else:
            candidates = [
                entity
                for entity in shape_consistent
                if float(np.linalg.norm(entity.position - predicted.position))
                <= self.reacquisition_distance_m
            ]

        if not candidates and initial is not None:
            at_initial = min(
                shape_consistent,
                key=lambda entity: float(
                    np.linalg.norm(entity.position - initial.position)
                ),
            )
            if float(np.linalg.norm(at_initial.position - initial.position)) <= 0.04:
                result = self._verify_geometry(
                    bound,
                    self._replace_source(
                        geometry_scene, bound.source_id, at_initial
                    ),
                )
                movement = float(
                    np.linalg.norm(at_initial.position - initial.position)
                )
                return Verification(
                    False,
                    relation,
                    max(result.position_error_m, 0.04 - movement),
                    f"source visibly remained at pick location; {result.detail}",
                )

        if not candidates:
            nearest_error = (
                min(
                    float(np.linalg.norm(entity.position - predicted.position))
                    for entity in shape_consistent
                )
                if predicted is not None
                else float("inf")
            )
            return Verification(
                False,
                relation,
                nearest_error,
                "fresh source does not agree with the sensor-bound release prediction",
            )
        candidates, alias_count = self._collapse_surface_aliases(
            candidates,
            predicted,
        )
        if len(candidates) != 1:
            return Verification(
                False,
                relation,
                float("inf"),
                "ambiguous fresh post-release source reacquisition: "
                + ", ".join(sorted(entity.instance_id for entity in candidates)),
            )

        selected = candidates[0]
        rebind = getattr(self.observer, "rebind_track", None)
        if callable(rebind):
            # Collapse the newly visible post-release component back onto the
            # episode identity bound before grasp.  A later same-label goal
            # can then exclude this exact sensor track rather than reranking a
            # changed scene and accidentally selecting it again.
            rebind(bound.source_id, selected)
        result = self._verify_geometry(
            bound,
            self._replace_source(
                geometry_scene, bound.source_id, selected
            ),
        )
        evidence = (
            "visible stable source track"
            if selected.instance_id == bound.source_id
            else "visible post-release source reacquisition"
        )
        if alias_count:
            evidence += f"; deduplicated {alias_count} overlapping RGB-D alias"
        self._remember_verified_placement(result, selected, planned_target)
        return replace(result, detail=f"{evidence}; {result.detail}")

    def _remember_verified_placement(self, result, source, target):
        remember = getattr(self.goals, "remember_verified_basket_placement", None)
        if result.success and target is not None and callable(remember):
            remember(source, target)

    @staticmethod
    def _collapse_surface_aliases(
        candidates: Sequence[SceneEntity],
        predicted: SceneEntity | None,
    ) -> tuple[list[SceneEntity], int]:
        """Collapse tracker ids backed by the same fresh RGB-D surface.

        Cross-camera partial OBB centres can differ by more than the tracker's
        ordinary 45-mm association radius, especially for a long pan.  This
        must not turn one physical observation into two identities at final
        verification.  Conversely, proximity alone is not identity evidence:
        candidates are aliases only when a majority of the *smaller* public
        RGB-D surface has a near-coincident point in the other component.
        Entities without fresh surface points remain distinct and therefore
        retain the verifier's existing fail-closed ambiguity behavior.
        """

        remaining = list(candidates)
        if len(remaining) < 2:
            return remaining, 0

        def aliases(first: SceneEntity, second: SceneEntity) -> bool:
            first_points = first.surface_points_world
            second_points = second.surface_points_world
            if first_points is None or second_points is None:
                return False
            first_planar = float(np.max(first.extent[:2]))
            second_planar = float(np.max(second.extent[:2]))
            maximum_center_separation = min(
                0.080,
                max(0.015, 0.35 * min(first_planar, second_planar)),
            )
            if (
                float(np.linalg.norm(first.position - second.position))
                > maximum_center_separation
            ):
                return False
            smaller, larger = (
                (first_points, second_points)
                if len(first_points) <= len(second_points)
                else (second_points, first_points)
            )
            # Deterministic thinning bounds verification cost without changing
            # which metric surface is queried.  At the production point counts
            # this normally keeps the complete cloud.
            if len(smaller) > 4096:
                indices = np.linspace(0, len(smaller) - 1, 4096, dtype=np.int64)
                smaller = smaller[indices]
            distances, _ = cKDTree(larger).query(smaller, k=1)
            return bool(float(np.mean(distances <= 0.012)) >= 0.60)

        groups: list[list[SceneEntity]] = []
        while remaining:
            group = [remaining.pop(0)]
            changed = True
            while changed:
                changed = False
                for candidate in tuple(remaining):
                    if any(aliases(candidate, member) for member in group):
                        group.append(candidate)
                        remaining.remove(candidate)
                        changed = True
            groups.append(group)

        collapsed: list[SceneEntity] = []
        for group in groups:
            collapsed.append(
                min(
                    group,
                    key=lambda entity: (
                        float(np.linalg.norm(entity.position - predicted.position))
                        if predicted is not None
                        else -float(entity.confidence),
                        -float(entity.confidence),
                        entity.instance_id,
                    ),
                )
            )
        return collapsed, len(candidates) - len(collapsed)

    def _shape_matches_initial(
        self, candidate: SceneEntity, initial: SceneEntity
    ) -> bool:
        rigid_size_error = np.abs(np.log(
            np.maximum(np.sort(candidate.extent), 1e-5)
            / np.maximum(np.sort(initial.extent), 1e-5)
        ))
        if np.max(rigid_size_error) <= self.initial_planar_log_tolerance:
            # Gravity can turn a released package onto another face. Its
            # intrinsic OBB edge lengths still identify the same rigid shape.
            return True
        candidate_span = np.abs(candidate.pose[:3, :3]) @ candidate.extent
        initial_span = np.abs(initial.pose[:3, :3]) @ initial.extent
        planar_log_ratio = np.abs(
            np.log(
                np.maximum(np.sort(candidate_span[:2]), 1e-5)
                / np.maximum(np.sort(initial_span[:2]), 1e-5)
            )
        )
        height_log_ratio = abs(
            float(
                np.log(
                    max(candidate_span[2], 1e-5)
                    / max(initial_span[2], 1e-5)
                )
            )
        )
        return bool(
            np.max(planar_log_ratio) <= self.initial_planar_log_tolerance
            and height_log_ratio <= self.initial_height_log_tolerance
        )

    def _verify_geometry(
        self, bound: BoundConstraintGraph, scene: SceneEstimate
    ):
        if bound.graph.goal_relation != Relation.IN:
            return super().verify(bound, scene)
        source = self._tracked_or_label(scene, bound.source_id, bound.source_label)
        target = self._tracked_or_label(scene, bound.target_id, bound.target_label)
        if target.region is None:
            return super().verify(bound, scene)
        # Project the source centre onto the target opening plane and collapse
        # only its vertical extent.  The base oriented-box verifier then tests
        # the true rotated XY footprint while ignoring permitted protrusion.
        local_center = target.region.local_coordinates(source.position[None, :])[0]
        local_center[2] = 0.0
        planar_pose = source.pose.copy()
        planar_pose[:3, 3] = target.region.center + local_center @ target.region.axes.T
        planar_extent = source.extent.copy()
        planar_extent[2] = 1e-6
        planar_source = replace(source, pose=planar_pose, extent=planar_extent)
        return super().verify(
            bound, self._replace_source(scene, bound.source_id, planar_source)
        )

    @staticmethod
    def _replace_source(
        scene: SceneEstimate, source_id: str, source: SceneEntity
    ) -> SceneEstimate:
        replacement = replace(source, instance_id=source_id)
        entities = tuple(
            replacement if entity.instance_id == source_id else entity
            for entity in scene.entities
        )
        if not any(entity.instance_id == source_id for entity in entities):
            entities = (*entities, replacement)
        return replace(scene, entities=entities)


class StableSceneEstimator:
    """Maintain sensor-derived Route C identities across frame-local reordering.

    The neutral geometry pipeline intentionally assigns deterministic IDs only
    within one frame.  Once an object moves, sorting can recycle an ID for a
    different package.  This wrapper uses label plus nearest-neighbour visual
    tracking and retains requested entities through short gripper occlusions.
    It never reads simulator bodies or contacts.
    """

    def __init__(
        self,
        observer: Any,
        *,
        ee_pose_provider: Callable[[], np.ndarray] | None = None,
        association_distance_m: float = 0.06,
        association_extent_scale: float = 1.5,
        minimum_association_distance_m: float = 0.025,
    ) -> None:
        if (
            association_distance_m <= 0
            or association_extent_scale <= 0
            or minimum_association_distance_m <= 0
            or minimum_association_distance_m > association_distance_m
        ):
            raise ValueError("scene-track association distances must be positive and ordered")
        self.observer = observer
        self._ee_pose_provider = ee_pose_provider
        # Same-label objects are common in LIBERO.  Treat this value as a hard
        # ceiling, while the actual gate below scales with the last RGB-D planar
        # footprint.  The previous fixed 150-mm radius could silently exchange
        # two bowls or packages during a one-frame occlusion.
        self.association_distance_m = float(association_distance_m)
        self.association_extent_scale = float(association_extent_scale)
        self.minimum_association_distance_m = float(minimum_association_distance_m)
        self._tracks: dict[str, SceneEntity] = {}
        self._next_id = 0
        self._last_visible_ids: frozenset[str] = frozenset()
        self._last_obstacle_filter_trace: tuple[dict[str, object], ...] = ()
        self._obstacle_filter_history: list[dict[str, object]] = []
        # A typed drawer contact can identify one RGB-D component that is the
        # articulated cabinet self surface.  Keep only its public provenance
        # long enough for the controller's subsequent MPC observations to
        # use the same contact-safe scene; no simulator identity is stored.
        self._drawer_self_filter_provenance: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._selector_diagnostic_archive: list[dict[str, object]] = []
        # A support-plane observation may complete a partially occluded
        # source's vertical pose.  Preserve that validated sensor refinement
        # only until the first post-grasp proprioceptive track update.
        self._support_refinements: dict[str, SceneEntity] = {}
        # Installed only after the contact-aware controller proves, across a
        # large released-hand motion, that one recognised fused field is
        # rigidly camera/tool-relative rather than world-fixed.  It is scoped
        # to the immediate same-phase replan and cleared by that controller.
        self._temporal_proprio_self_filter: dict[str, object] | None = None
        # Sensor-only audit for the latest attempt to capture the second frame
        # of a temporal self-field proof.  Keep only typed reasons and bounded
        # counts/identities here; raw RGB-D points remain inside the estimator.
        self._last_temporal_proprio_field_evidence_diagnostic: dict[
            str, object
        ] = {}
        self._last_stabilized_scene: SceneEstimate | None = None

    @property
    def visible_instance_ids(self) -> frozenset[str]:
        """Stable ids backed by an RGB-D component in the latest observation."""

        return self._last_visible_ids

    @property
    def last_obstacle_filter_trace(self) -> tuple[dict[str, object], ...]:
        """Sensor provenance for obstacle fields removed as requested duplicates."""

        return tuple(dict(item) for item in self._last_obstacle_filter_trace)

    @property
    def obstacle_filter_history(self) -> tuple[dict[str, object], ...]:
        """Append-only obstacle-removal audit for the current policy episode."""

        return tuple(dict(item) for item in self._obstacle_filter_history)

    @property
    def selector_diagnostics(self) -> tuple[dict[str, object], ...]:
        """Append-only Route-C selector evidence across inner track resets."""

        live = getattr(self.observer, "selector_diagnostics", ())
        return tuple(
            dict(item)
            for item in (*self._selector_diagnostic_archive, *tuple(live))
        )

    @property
    def last_temporal_proprio_field_evidence_diagnostic(
        self,
    ) -> dict[str, object]:
        """Return the bounded sensor-only reason for the latest proof capture."""

        return dict(self._last_temporal_proprio_field_evidence_diagnostic)

    @property
    def current_sensor_scene(self) -> SceneEstimate | None:
        """Expose only the latest stabilized sensor scene for frozen-path checks."""

        return self._last_stabilized_scene

    @property
    def last_selector_view_hint(self):
        """Forward the latest strict sensor hint without promoting geometry.

        The wrapped estimator owns the 2-D provenance.  This wrapper only
        exposes that bounded active-view request to Route C; it never inserts
        the hinted source or a fixture into the stable scene tracks.
        """

        return getattr(self.observer, "last_selector_view_hint", None)

    def update_track(self, stable_id: str, entity: SceneEntity) -> None:
        """Update a visual track from sensor-bound EE proprio propagation."""

        if not stable_id:
            raise ValueError("stable track id is required")
        self._support_refinements.pop(stable_id, None)
        self._tracks[stable_id] = replace(entity, instance_id=stable_id)
        # Proprio propagation is not a claim of current RGB-D visibility.
        self._last_visible_ids = frozenset(
            item for item in self._last_visible_ids if item != stable_id
        )

    def rebind_track(self, stable_id: str, entity: SceneEntity) -> None:
        """Merge a freshly reacquired component into an existing visual id."""

        if not stable_id:
            raise ValueError("stable track id is required")
        observed_id = entity.instance_id
        self._support_refinements.pop(stable_id, None)
        self._support_refinements.pop(observed_id, None)
        if observed_id != stable_id:
            self._tracks.pop(observed_id, None)
        self._tracks[stable_id] = replace(entity, instance_id=stable_id)
        visible = set(self._last_visible_ids)
        visible.discard(observed_id)
        visible.add(stable_id)
        self._last_visible_ids = frozenset(visible)

    def discard_track(self, stable_id: str) -> None:
        """Forget one released dynamic track before fresh RGB-D association."""

        self._support_refinements.pop(stable_id, None)
        self._tracks.pop(stable_id, None)
        self._last_visible_ids = frozenset(
            item for item in self._last_visible_ids if item != stable_id
        )

    @staticmethod
    def _validated_typed_wrist_negative_visibility(
        value: object,
        ee_pose_world: np.ndarray,
    ) -> dict[str, object] | None:
        """Validate the serialized public three-frame visibility authority."""

        required = {
            "mode",
            "endpoint_count",
            "motion_segment_count",
            "view_budget_consumed",
            "wrist_camera_model",
            "ee_from_wrist",
            "wrist_endpoint_modes",
            "obb_uncertainty_inflation_m",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            return None
        endpoint_count = value.get("endpoint_count")
        segment_count = value.get("motion_segment_count")
        budget_consumed = value.get("view_budget_consumed")
        endpoint_modes = value.get("wrist_endpoint_modes")
        inflation = ContactAwareRouteCController._builtin_finite_number(
            value.get("obb_uncertainty_inflation_m")
        )
        model = value.get("wrist_camera_model")
        model_keys = {
            "policy_name",
            "sensor_name",
            "width_px",
            "height_px",
            "intrinsic",
            "world_from_camera",
            "observation_v_flipped",
        }
        if (
            type(value.get("mode")) is not str
            or value.get("mode")
            != "three_frame_noncollinear_typed_wrist_negative"
            or type(endpoint_count) is not int
            or endpoint_count != 3
            or type(segment_count) is not int
            or segment_count != 2
            or type(budget_consumed) is not bool
            or budget_consumed is not True
            or type(endpoint_modes) is not tuple
            or any(type(mode) is not str for mode in endpoint_modes)
            or endpoint_modes
            != ("strict_negative", "strict_negative", "strict_negative")
            or inflation is None
            or abs(
                inflation
                - ContactAwareRouteCController._FREE_RIM_VIEW_OBB_UNCERTAINTY_M
            )
            > 1e-12
            or not isinstance(model, Mapping)
            or set(model) != model_keys
            or type(model.get("policy_name")) is not str
            or model.get("policy_name") != "wrist"
            or type(model.get("sensor_name")) is not str
            or not str(model.get("sensor_name")).strip()
            or type(model.get("width_px")) is not int
            or type(model.get("height_px")) is not int
            or type(model.get("observation_v_flipped")) is not bool
        ):
            return None
        try:
            ee_pose = np.asarray(ee_pose_world, dtype=np.float64)
            intrinsic = np.asarray(model["intrinsic"], dtype=np.float64)
            world_from_camera = np.asarray(
                model["world_from_camera"], dtype=np.float64
            )
            ee_from_wrist = np.asarray(
                value["ee_from_wrist"], dtype=np.float64
            )
        except (KeyError, TypeError, ValueError):
            return None
        width_px = model["width_px"]
        height_px = model["height_px"]
        transforms = (ee_pose, world_from_camera, ee_from_wrist)
        if (
            ee_pose.shape != (4, 4)
            or intrinsic.shape != (3, 3)
            or world_from_camera.shape != (4, 4)
            or ee_from_wrist.shape != (4, 4)
            or width_px < 8
            or height_px < 8
            or not np.all(np.isfinite(intrinsic))
            or intrinsic[0, 0] <= 0.0
            or intrinsic[1, 1] <= 0.0
            or abs(float(intrinsic[0, 1])) > 1e-12
            or abs(float(intrinsic[1, 0])) > 1e-12
            or not np.allclose(intrinsic[2], (0.0, 0.0, 1.0), atol=1e-12)
            or not 0.0 <= intrinsic[0, 2] <= float(width_px - 1)
            or not 0.0 <= intrinsic[1, 2] <= float(height_px - 1)
            or any(
                not np.all(np.isfinite(transform))
                or not np.allclose(
                    transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6
                )
                or not np.allclose(
                    transform[:3, :3].T @ transform[:3, :3],
                    np.eye(3),
                    atol=2e-3,
                )
                or float(np.linalg.det(transform[:3, :3])) <= 0.0
                for transform in transforms
            )
        ):
            return None
        predicted_world_from_camera = ee_pose @ ee_from_wrist
        position_error = float(
            np.linalg.norm(
                predicted_world_from_camera[:3, 3]
                - world_from_camera[:3, 3]
            )
        )
        rotation_error = ContactAwareRouteCController._rotation_distance_rad(
            predicted_world_from_camera, world_from_camera
        )
        if (
            position_error
            > ContactAwareRouteCController._FREE_RIM_VIEW_CAMERA_RIGID_POSITION_TOLERANCE_M
            or rotation_error
            > ContactAwareRouteCController._FREE_RIM_VIEW_CAMERA_RIGID_ROTATION_TOLERANCE_RAD
        ):
            return None
        return {
            "mode": value["mode"],
            "endpoint_count": endpoint_count,
            "motion_segment_count": segment_count,
            "view_budget_consumed": True,
            "wrist_camera_model": {
                "policy_name": "wrist",
                "sensor_name": str(model["sensor_name"]),
                "width_px": width_px,
                "height_px": height_px,
                "intrinsic": intrinsic.copy(),
                "world_from_camera": world_from_camera.copy(),
                "observation_v_flipped": model["observation_v_flipped"],
            },
            "ee_from_wrist": ee_from_wrist.copy(),
            "wrist_endpoint_modes": endpoint_modes,
            "obb_uncertainty_inflation_m": inflation,
        }

    def install_temporal_proprio_self_filter(
        self,
        *,
        field_id: str,
        field_label: str,
        field_center_world_m: np.ndarray,
        field_half_extents_m: np.ndarray,
        ee_world_m: np.ndarray,
        maximum_relative_error_m: float,
        field_axes_world: np.ndarray,
        surface_points_by_camera: Sequence[tuple[str, np.ndarray]],
        ee_pose_world: np.ndarray,
        source_id: str,
        source_label: str,
        target_id: str,
        target_label: str,
        visibility_evidence: Mapping[str, object] | None = None,
    ) -> bool:
        """Install one exact, short-lived two-frame self-field proof.

        This method does not infer motion or semantic identity.  Its caller
        supplies the already validated temporal evidence; this layer merely
        applies it to future fresh SDFs using public EE proprioception.  A
        changed id/label/size/tool offset or an ambiguous duplicate preserves
        every obstacle and therefore fails closed.
        """

        # A failed re-install must never leave an older deletion authority.
        self._temporal_proprio_self_filter = None
        try:
            center = np.asarray(field_center_world_m, dtype=np.float64)
            half = np.asarray(field_half_extents_m, dtype=np.float64)
            ee = np.asarray(ee_world_m, dtype=np.float64)
            ee_pose = np.asarray(ee_pose_world, dtype=np.float64)
            axes = np.asarray(field_axes_world, dtype=np.float64)
            label = self._normalised_label(field_label)
            maximum_error = float(maximum_relative_error_m)
        except (TypeError, ValueError):
            return False
        if (
            not isinstance(field_id, str)
            or not field_id.startswith("fused-")
            or not label
            or center.shape != (3,)
            or half.shape != (3,)
            or ee.shape != (3,)
            or ee_pose.shape != (4, 4)
            or axes.shape != (3, 3)
            or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(half))
            or not np.all(np.isfinite(ee))
            or not np.all(np.isfinite(ee_pose))
            or not np.all(np.isfinite(axes))
            or not np.allclose(ee_pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6)
            or not np.allclose(
                ee_pose[:3, :3].T @ ee_pose[:3, :3],
                np.eye(3),
                atol=2e-3,
            )
            or float(np.linalg.det(ee_pose[:3, :3])) <= 0.0
            or not np.allclose(axes.T @ axes, np.eye(3), atol=2e-3)
            or float(np.linalg.det(axes)) <= 0.0
            or float(np.linalg.norm(ee_pose[:3, 3] - ee)) > 0.002
            or np.any(half <= 0.0)
            or not np.isfinite(maximum_error)
            or not 0.0 < maximum_error <= 0.015
            or (
                visibility_evidence is not None
                and ContactAwareRouteCController._builtin_finite_number(
                    maximum_relative_error_m
                )
                is None
            )
            or not all(
                isinstance(value, str) and value.strip()
                for value in (source_id, source_label, target_id, target_label)
            )
            or source_id == target_id
        ):
            return False
        typed_visibility = None
        if visibility_evidence is not None:
            typed_visibility = self._validated_typed_wrist_negative_visibility(
                visibility_evidence,
                ee_pose,
            )
            if typed_visibility is None:
                return False
        camera_points: list[tuple[str, np.ndarray]] = []
        seen_cameras: set[str] = set()
        for camera_name, raw_points in surface_points_by_camera:
            points = np.asarray(raw_points, dtype=np.float64)
            canonical_name = (
                ContactAwareRouteCController._canonical_view_name(camera_name)
                if typed_visibility is not None
                else camera_name
            )
            if (
                not isinstance(camera_name, str)
                or not camera_name.strip()
                or not canonical_name
                or canonical_name in seen_cameras
                or points.ndim != 2
                or points.shape[1:] != (3,)
                or len(points) < 24
                or not np.all(np.isfinite(points))
            ):
                return False
            seen_cameras.add(canonical_name)
            camera_points.append((canonical_name, points.copy()))
        if typed_visibility is not None:
            if set(seen_cameras) != {"agentview"}:
                return False
        elif len(camera_points) < 2:
            return False
        public_tool_evidence = {
            camera_name: (
                (
                    ContactAwareRouteCController._typed_negative_panda_tool_surface_evidence
                    if typed_visibility is not None
                    else ContactAwareRouteCController._released_panda_tool_surface_evidence
                )(points, ee_pose)
            )
            for camera_name, points in camera_points
        }
        if not all(
            bool(evidence["accepted"])
            for evidence in public_tool_evidence.values()
        ):
            return False
        if typed_visibility is not None:
            wrist_model = dict(typed_visibility["wrist_camera_model"])
            frustum = ContactAwareRouteCController._wrist_frustum_evidence(
                center,
                half,
                axes,
                wrist_model,
                dict(camera_points)["agentview"],
            )
            if (
                frustum.get("accepted") is not True
                or frustum.get("strictly_outside") is not True
            ):
                return False
        self._temporal_proprio_self_filter = {
            "field_id": field_id,
            "field_label": label,
            "field_half_extents_m": half.copy(),
            "field_relative_to_ee_m": (center - ee).copy(),
            "field_axes_world": axes.copy(),
            "surface_points_by_camera": tuple(camera_points),
            "ee_pose_world": ee_pose.copy(),
            "source_id": source_id,
            "source_label": self._normalised_label(source_label),
            "target_id": target_id,
            "target_label": self._normalised_label(target_label),
            "maximum_relative_error_m": maximum_error,
            "per_camera_public_tool_evidence": public_tool_evidence,
            "visibility_evidence": typed_visibility,
        }
        return True

    def clear_temporal_proprio_self_filter(self) -> None:
        """Clear the immediate-replan recognised self-field proof."""

        self._temporal_proprio_self_filter = None

    def temporal_proprio_field_evidence(
        self,
        *,
        field_id: str,
        field_label: str,
        field_center_world_m: np.ndarray,
        field_half_extents_m: np.ndarray,
        source_id: str,
        source_label: str,
        target_id: str,
        target_label: str,
        field_axes_world: np.ndarray | None = None,
        surface_points_by_camera: object | None = None,
    ) -> dict[str, object] | None:
        """Return one fresh, target-consistent field with raw camera support."""

        self._last_temporal_proprio_field_evidence_diagnostic = {}

        def reject(reason: str, **details: object) -> None:
            self._last_temporal_proprio_field_evidence_diagnostic = {
                "accepted": False,
                "reason": reason,
                "field_id": str(field_id),
                "field_label": self._normalised_label(field_label),
                **details,
            }
            return None

        scene = self._last_stabilized_scene
        if scene is None or not isinstance(scene.obstacle_sdf, CompositeSDF):
            return reject("fresh_composite_sdf_unavailable")
        try:
            source = scene.by_id(source_id)
            target = scene.by_id(target_id)
        except PerceptionError:
            return reject("source_or_target_missing")
        if source.instance_id == target.instance_id:
            return reject("source_target_identity_alias")
        if source_id not in self._last_visible_ids:
            return reject("source_not_freshly_visible")
        if target_id not in self._last_visible_ids:
            return reject("target_not_freshly_visible")
        if self._normalised_label(source.label) != self._normalised_label(
            source_label
        ):
            return reject("source_label_mismatch")
        if self._normalised_label(target.label) != self._normalised_label(
            target_label
        ):
            return reject("target_label_mismatch")
        expected_center = np.asarray(field_center_world_m, dtype=np.float64)
        expected_half = np.asarray(field_half_extents_m, dtype=np.float64)
        expected_label = self._normalised_label(field_label)
        if (
            expected_center.shape != (3,)
            or expected_half.shape != (3,)
            or not np.all(np.isfinite(expected_center))
            or not np.all(np.isfinite(expected_half))
        ):
            return reject("malformed_expected_field_geometry")
        same_family = [
            item
            for item in scene.obstacle_sdf.fields
            if isinstance(item, BoxSDF)
            and {"drawer", "cabinet"}
            & set(self._normalised_label(item.source_label or "").split())
        ]
        matches = [
            item
            for item in same_family
            if item.source_instance_id == field_id
            and self._normalised_label(item.source_label or "") == expected_label
            and float(np.linalg.norm(item.center - expected_center)) <= 0.020
            and float(np.max(np.abs(item.half_extents - expected_half))) <= 0.012
        ]
        if len(same_family) != 1:
            return reject(
                "container_family_field_ambiguity",
                same_family_field_count=len(same_family),
            )
        if len(matches) != 1:
            return reject(
                "field_identity_label_or_geometry_mismatch",
                matching_field_count=len(matches),
            )
        field = matches[0]
        if field_axes_world is not None or surface_points_by_camera is not None:
            expected_axes = (
                ContactAwareRouteCController._strict_provider_float_array(
                    field_axes_world, (3, 3)
                )
            )
            expected_surfaces = (
                ContactAwareRouteCController._strict_provider_surface_map(
                    surface_points_by_camera,
                    canonical_policy_names=False,
                )
            )
            actual_surfaces = (
                ContactAwareRouteCController._strict_provider_surface_map(
                    tuple(
                        (name, points.copy())
                        for name, points in field.surface_points_by_camera
                    ),
                    canonical_policy_names=False,
                )
            )
            if (
                expected_axes is None
                or expected_surfaces is None
                or actual_surfaces is None
                or not np.array_equal(expected_axes, field.axes)
                or set(expected_surfaces) != set(actual_surfaces)
                or any(
                    not np.array_equal(
                        expected_surfaces[name], actual_surfaces[name]
                    )
                    for name in expected_surfaces
                )
            ):
                return reject("expected_field_axes_or_surfaces_mismatch")
        visible_containers = [
            entity
            for entity in scene.entities
            if entity.instance_id in self._last_visible_ids
            # The perception scene deliberately exposes a generic component as
            # both a SceneEntity and its obstacle BoxSDF.  That corresponding
            # entity is provenance for this field, not a second container.  It
            # must not veto itself; every *different* visible container keeps
            # the conservative overlap veto below.
            and entity.instance_id != field.source_instance_id
            and {"drawer", "cabinet"}
            & set(self._normalised_label(entity.label).split())
        ]
        overlapping_visible_containers = [
            entity
            for entity in visible_containers
            if self._obb_surface_overlaps(entity, field)
            and self._obb_contains_either_center(entity, field)
        ]
        if overlapping_visible_containers:
            return reject(
                "distinct_visible_container_overlap",
                overlapping_visible_container_ids=sorted(
                    entity.instance_id
                    for entity in overlapping_visible_containers
                ),
            )
        cameras = tuple(
            (name, np.asarray(points, dtype=np.float64).copy())
            for name, points in field.surface_points_by_camera
        )
        if len(cameras) < 2:
            return reject(
                "insufficient_camera_surface_provenance",
                camera_count=len(cameras),
                per_camera_point_counts={
                    name: int(len(points)) for name, points in cameras
                },
            )
        result = {
            "field_id": field.source_instance_id,
            "field_label": field.source_label,
            "field_center_world_m": field.center.copy(),
            "field_half_extents_m": field.half_extents.copy(),
            "field_axes_world": field.axes.copy(),
            "surface_points_by_camera": cameras,
            "source_id": source.instance_id,
            "source_label": source.label,
            "target_id": target.instance_id,
            "target_label": target.label,
        }
        self._last_temporal_proprio_field_evidence_diagnostic = {
            "accepted": True,
            "reason": "unique_fresh_field_with_dual_camera_surface_provenance",
            "field_id": str(field.source_instance_id),
            "field_label": self._normalised_label(field.source_label or ""),
            "camera_count": len(cameras),
            "per_camera_point_counts": {
                name: int(len(points)) for name, points in cameras
            },
        }
        return result

    def invalidate_sensor_cache(self) -> None:
        """Drop only episode-level sensor caches; stable ids remain sensor-bound."""

        # A caller asking for a fresh frame must not be able to retrieve the
        # previous field snapshot through ``current_sensor_scene`` before the
        # wrapped RGB-D estimator has actually observed again.
        self._last_stabilized_scene = None
        self._last_temporal_proprio_field_evidence_diagnostic = {}
        invalidate = getattr(self.observer, "invalidate_sensor_cache", None)
        if callable(invalidate):
            invalidate()

    def invalidate_selector_reacquisition(self) -> None:
        """Drop all pre-grasp tracks before a changed wrist-camera view.

        This path is used only before source binding, when the empty hand has
        moved to a high observation pose.  Retaining unmatched requested
        tracks here could let a stale fixture OBB satisfy the next selector;
        resetting both association layers guarantees the resolver sees only
        geometry extracted from the new RGB-D frame.
        """

        self._tracks.clear()
        self._support_refinements.clear()
        self._next_id = 0
        self._last_visible_ids = frozenset()
        self._last_obstacle_filter_trace = ()
        self._temporal_proprio_self_filter = None
        self._last_stabilized_scene = None
        reset = getattr(self.observer, "reset", None)
        if callable(reset):
            self._selector_diagnostic_archive.extend(
                dict(item)
                for item in getattr(self.observer, "selector_diagnostics", ())
            )
            reset()
            return
        invalidate = getattr(self.observer, "invalidate_sensor_cache", None)
        if callable(invalidate):
            invalidate()

    def observe(self, requested_labels: Sequence[str]) -> SceneEstimate:
        scene = self.observer.observe(requested_labels)
        stabilized = self._stabilize_observation(scene, requested_labels)
        result = self._apply_cached_drawer_self_filter(
            stabilized, requested_labels
        )
        self._last_stabilized_scene = result
        return result

    def observe_contact(
        self,
        requested_labels: Sequence[str],
        contact_point_world: np.ndarray,
    ) -> SceneEstimate:
        """Expose one tightly bounded intended fixture contact surface.

        Neutral RGB-D geometry can represent a small black knob as an
        ``unknown`` obstacle even when the language-conditioned knob detector
        has reconstructed its contact point.  Only the exact stove-knob
        request may remove one compact unknown OBB containing that point.  If
        the evidence is absent or ambiguous, preserve every field so planning
        fails closed.  A typed drawer transfer may similarly remove exactly
        one broad unknown field containing its sensor-bound endpoint.  The
        exact microwave request admits only one upright, thin door-like OBB;
        compact clutter such as a black bowl remains active.
        """

        point = np.asarray(contact_point_world, dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise PerceptionError("contact point must be a finite xyz vector")
        labels = {
            self._normalised_label(label) for label in requested_labels
        }
        stove_request = "stove knob" in labels
        drawer_request = "drawer" in labels and "cabinet" in labels
        microwave_request = labels == {"microwave"}
        if not (stove_request or drawer_request or microwave_request):
            raise PerceptionError(
                "unknown contact filtering is restricted to typed fixtures"
            )
        scene = self.observe(requested_labels)
        obstacle = scene.obstacle_sdf
        if not isinstance(obstacle, CompositeSDF):
            return scene

        candidates: list[tuple[int, BoxSDF, float]] = []
        for index, field in enumerate(obstacle.fields):
            if not isinstance(field, BoxSDF):
                continue
            if (
                field.source_instance_id is None
                or field.source_label is None
                or self._normalised_label(field.source_label) != "unknown"
            ):
                continue
            full_extents = 2.0 * np.asarray(field.half_extents, dtype=np.float64)
            if stove_request:
                if float(np.max(full_extents)) > 0.100:
                    continue
            elif microwave_request:
                # PCA OBB axes are expressed in world coordinates.  Identify
                # the most vertical local axis, then require a genuinely
                # upright, broad and thin panel.  This excludes bowls, plates,
                # cups and appliance-body blobs even if one happens to cover
                # the intended contact point.
                vertical_components = np.abs(
                    np.asarray(field.axes, dtype=np.float64)[2, :]
                )
                vertical_index = int(np.argmax(vertical_components))
                vertical_alignment = float(vertical_components[vertical_index])
                vertical_extent = float(full_extents[vertical_index])
                horizontal_extents = np.delete(full_extents, vertical_index)
                horizontal_long = float(np.max(horizontal_extents))
                horizontal_thin = float(np.min(horizontal_extents))
                upright_door_surface = (
                    vertical_alignment >= 0.88
                    and 0.12 <= vertical_extent <= 0.36
                    and 0.12 <= horizontal_long <= 0.50
                    and horizontal_thin <= 0.090
                )
                if not upright_door_surface:
                    continue
            else:
                horizontal = np.sort(full_extents[:2])
                planar_swept_surface = (
                    full_extents[2] <= 0.065
                    and horizontal[0] >= 0.115
                    and horizontal[1] <= 0.42
                )
                # RGB-D segmentation can fuse the handle/front with the
                # cabinet spine. Such a self component is a broad, upright
                # facade rather than the thin swept panel above. Require a
                # substantial horizontal footprint and a vertical extent;
                # bowls and other low clutter therefore cannot be peeled by
                # this rule. The measured contact point must still lie in
                # the component, so unrelated cabinet-sized clutter remains
                # an active obstacle.
                upright_articulated_surface = (
                    0.12 <= full_extents[2] <= 0.36
                    and horizontal[0] >= 0.14
                    and horizontal[1] >= 0.08
                    and horizontal[1] <= 0.42
                )
                if not (planar_swept_surface or upright_articulated_surface):
                    continue
            raw_distance = float(field.distance(point))
            maximum_contact_distance = 0.004 if stove_request else 0.010
            if raw_distance <= maximum_contact_distance:
                candidates.append((index, field, raw_distance))
        if len(candidates) != 1:
            return scene

        removed_index, field, raw_distance = candidates[0]
        if drawer_request and (
            2.0 * float(field.half_extents[2]) > 0.065
        ):
            self._drawer_self_filter_provenance[field.source_instance_id] = (
                np.asarray(field.center, dtype=np.float64).copy(),
                np.asarray(field.half_extents, dtype=np.float64).copy(),
            )
        retained = tuple(
            item
            for index, item in enumerate(obstacle.fields)
            if index != removed_index
        )
        trace = {
            "field_index": removed_index,
            "removed_source_instance_id": field.source_instance_id,
            "removed_source_label": field.source_label,
            "reason": (
                "sensor_bound_stove_knob_compact_contact_surface"
                if stove_request
                else (
                    "sensor_bound_microwave_articulated_self_surface"
                    if microwave_request
                    else (
                        "sensor_bound_drawer_swept_planar_surface"
                        if 2.0 * float(field.half_extents[2]) <= 0.065
                        else "sensor_bound_drawer_articulated_self_surface"
                    )
                )
            ),
            "raw_sdf_m": raw_distance,
            "contact_point_world_m": [float(value) for value in point],
            "field_center_world_m": [float(value) for value in field.center],
            "field_half_extents_m": [
                float(value) for value in field.half_extents
            ],
        }
        self._last_obstacle_filter_trace = (
            *self._last_obstacle_filter_trace,
            trace,
        )
        self._obstacle_filter_history.append(dict(trace))
        filtered = CompositeSDF(retained) if retained else EmptySDF()
        return replace(scene, obstacle_sdf=filtered)

    def _apply_cached_drawer_self_filter(
        self, scene: SceneEstimate, requested_labels: Sequence[str]
    ) -> SceneEstimate:
        """Reuse one fresh typed self-surface decision during MPC replans.

        Contact selection and motion planning observe separately.  Without
        this narrow hand-off, the next ordinary ``observe`` would restore the
        same RGB-D cabinet fusion and reject the already validated handle
        approach.  Match only the recorded public provenance id, unknown
        label, and small sensor-scale OBB motion; all other fields (including
        black-bowl clutter) remain untouched.
        """

        labels = {self._normalised_label(label) for label in requested_labels}
        if not ("drawer" in labels and "cabinet" in labels):
            return scene
        if not self._drawer_self_filter_provenance:
            return scene
        obstacle = scene.obstacle_sdf
        if not isinstance(obstacle, CompositeSDF):
            return scene
        removals: list[dict[str, object]] = []
        retained: list[object] = []
        for index, field in enumerate(obstacle.fields):
            provenance = self._drawer_self_filter_provenance.get(
                getattr(field, "source_instance_id", None)
            )
            if (
                provenance is None
                or not isinstance(field, BoxSDF)
                or self._normalised_label(field.source_label or "") != "unknown"
            ):
                retained.append(field)
                continue
            previous_center, previous_half = provenance
            center = np.asarray(field.center, dtype=np.float64)
            half = np.asarray(field.half_extents, dtype=np.float64)
            if (
                np.linalg.norm(center - previous_center) > 0.080
                or np.max(np.abs(half - previous_half)) > 0.035
                or not (
                    0.12 <= 2.0 * float(half[2]) <= 0.36
                    and np.min(2.0 * np.sort(half[:2])) >= 0.08
                )
            ):
                retained.append(field)
                continue
            removals.append(
                {
                    "field_index": index,
                    "removed_source_instance_id": field.source_instance_id,
                    "removed_source_label": field.source_label,
                    "reason": "sensor_bound_drawer_articulated_self_surface_cached",
                    "field_center_world_m": [float(value) for value in center],
                    "field_half_extents_m": [float(value) for value in half],
                }
            )
        if not removals:
            return scene
        self._last_obstacle_filter_trace = (
            *self._last_obstacle_filter_trace,
            *removals,
        )
        self._obstacle_filter_history.extend(dict(item) for item in removals)
        filtered = CompositeSDF(tuple(retained)) if retained else EmptySDF()
        return replace(scene, obstacle_sdf=filtered)

    def observe_source_selector(
        self,
        requested_labels: Sequence[str],
        source_reference: EntityRef,
    ) -> SceneEstimate:
        """Observe one source selector without retaining hidden context.

        New sensor adapters can use the typed reference to run bounded
        selector-specific measurements for this frame.  Older observers stay
        compatible through the ordinary ``observe`` call.  In either case the
        resulting scene goes through exactly the same stable-id and freshness
        pipeline.
        """

        # A new selector-scoped measurement supersedes any earlier support
        # refinement.  Clearing before the synchronous call also fails closed
        # if the inner observation raises.
        self._support_refinements.clear()
        contextual = getattr(self.observer, "observe_source_selector", None)
        scene = (
            contextual(requested_labels, source_reference)
            if callable(contextual)
            else self.observer.observe(requested_labels)
        )
        return self._stabilize_observation(
            scene,
            requested_labels,
            source_reference=source_reference,
        )

    def _stabilize_observation(
        self,
        scene: SceneEstimate,
        requested_labels: Sequence[str],
        *,
        source_reference: EntityRef | None = None,
    ) -> SceneEstimate:
        requested = {" ".join(label.lower().replace("_", " ").split()) for label in requested_labels}
        # The inner Route-C estimator retains requested entities through
        # occlusion.  When it exposes a visibility set, only components in
        # that set are fresh association candidates; cached ghosts remain an
        # internal prior and must not be promoted to visual evidence here.
        inner_visible = getattr(self.observer, "visible_instance_ids", None)
        if inner_visible is None:
            candidates = list(scene.entities)
        else:
            visible_frame_ids = {str(instance_id) for instance_id in inner_visible}
            candidates = [
                entity
                for entity in scene.entities
                if entity.instance_id in visible_frame_ids
            ]
        if not self._tracks:
            entities = []
            visible: set[str] = set()
            inner_to_stable: dict[str, str] = {}
            for entity in candidates:
                stable_id = self._unique_id(entity.instance_id)
                tracked = replace(entity, instance_id=stable_id)
                self._tracks[stable_id] = tracked
                entities.append(tracked)
                visible.add(stable_id)
                inner_to_stable[entity.instance_id] = stable_id
            self._last_visible_ids = frozenset(visible)
            tracked_scene = replace(
                scene,
                entities=tuple(entities),
                support_relation_evidence=self._translate_support_relation_evidence(
                    scene,
                    inner_to_stable,
                    visible,
                    tuple(entities),
                ),
            )
            tracked_scene = self._without_requested_obstacle_duplicates(
                tracked_scene,
                requested,
                tuple(entities),
            )
            tracked_scene = self._without_temporal_proprio_self_component(
                tracked_scene
            )
            tracked_scene = self._without_public_proprio_self_component(
                tracked_scene
            )
            self._capture_support_refinement(tracked_scene, source_reference)
            return tracked_scene

        used: set[int] = set()
        updated: dict[str, SceneEntity] = {}
        visible: set[str] = set()
        inner_to_stable: dict[str, str] = {}
        for stable_id, previous in self._tracks.items():
            same_label = [
                (index, entity)
                for index, entity in enumerate(candidates)
                if index not in used and entity.label == previous.label
            ]
            if same_label:
                index, candidate = min(
                    same_label,
                    key=lambda item: float(
                        np.linalg.norm(item[1].position - previous.position)
                    ),
                )
                distance = float(np.linalg.norm(candidate.position - previous.position))
                association_limit = self._association_limit_m(previous)
                if distance <= association_limit + 1e-9:
                    used.add(index)
                    tracked = replace(candidate, instance_id=stable_id)
                    if source_reference is None:
                        tracked = self._support_refinements.get(
                            stable_id, tracked
                        )
                    updated[stable_id] = tracked
                    visible.add(stable_id)
                    inner_to_stable[candidate.instance_id] = stable_id
                    continue
            if previous.label in requested:
                # A requested package under the gripper can disappear from RGB
                # segmentation for several phases.  Its last visual geometry
                # remains safer than silently binding its id to another object.
                updated[stable_id] = previous

        for index, candidate in enumerate(candidates):
            if index in used:
                continue
            stable_id = self._unique_id(candidate.instance_id, existing=updated)
            updated[stable_id] = replace(candidate, instance_id=stable_id)
            visible.add(stable_id)
            inner_to_stable[candidate.instance_id] = stable_id
        self._tracks = updated
        self._support_refinements = {
            stable_id: entity
            for stable_id, entity in self._support_refinements.items()
            if stable_id in updated
        }
        self._last_visible_ids = frozenset(visible)
        tracked_entities = tuple(updated.values())
        tracked_scene = replace(
            scene,
            entities=tracked_entities,
            support_relation_evidence=self._translate_support_relation_evidence(
                scene,
                inner_to_stable,
                visible,
                tracked_entities,
            ),
        )
        tracked_scene = self._without_requested_obstacle_duplicates(
            tracked_scene,
            requested,
            tracked_entities,
        )
        tracked_scene = self._without_temporal_proprio_self_component(
            tracked_scene
        )
        tracked_scene = self._without_public_proprio_self_component(
            tracked_scene
        )
        self._capture_support_refinement(tracked_scene, source_reference)
        return tracked_scene

    def _capture_support_refinement(
        self,
        scene: SceneEstimate,
        source_reference: EntityRef | None,
    ) -> None:
        """Retain one physically validated, selector-scoped source entity."""

        if source_reference is None:
            return
        selector = source_reference.selector
        if (
            source_reference.role != "source"
            or selector is None
            or selector.relation is not Relation.ON
            or len(selector.references) != 1
            or len(scene.support_relation_evidence) != 1
        ):
            return
        evidence = scene.support_relation_evidence[0]
        if (
            evidence.source_label != source_reference.label
            or evidence.reference_label != selector.references[0]
            or evidence.source_instance_id not in self._last_visible_ids
        ):
            return
        try:
            source = scene.by_id(evidence.source_instance_id)
        except PerceptionError:
            return
        if source.label != evidence.source_label:
            return
        if _support_refinement_residuals(scene, evidence, source) is None:
            return
        self._support_refinements[source.instance_id] = source
        # Keep the track and the retained refinement identical.  Later plain
        # frames may refresh visibility while this exact geometry remains the
        # pre-grasp source pose.
        self._tracks[source.instance_id] = source

    @staticmethod
    def _translate_support_relation_evidence(
        scene: SceneEstimate,
        inner_to_stable: Mapping[str, str],
        visible_stable_ids: Collection[str],
        stable_entities: Sequence[SceneEntity],
    ) -> tuple[SupportRelationEvidence, ...]:
        """Translate only evidence attached to a fresh one-to-one association."""

        visible = {str(instance_id) for instance_id in visible_stable_ids}
        entities_by_id = {entity.instance_id: entity for entity in stable_entities}
        translated: list[SupportRelationEvidence] = []
        for evidence in scene.support_relation_evidence:
            if not same_sensor_capture(evidence, scene):
                continue
            stable_id = inner_to_stable.get(evidence.source_instance_id)
            if stable_id is None or stable_id not in visible:
                continue
            source = entities_by_id.get(stable_id)
            if source is None or source.label != evidence.source_label:
                continue
            translated.append(replace(evidence, source_instance_id=stable_id))
        return tuple(translated)

    @staticmethod
    def _normalised_label(label: str) -> str:
        return " ".join(label.lower().replace("_", " ").split())

    @staticmethod
    def _obb_surface_overlaps(entity: SceneEntity, field: BoxSDF) -> bool:
        """Conservatively test two world-frame OBBs with a sensor-scale margin."""

        entity_center = np.asarray(entity.position, dtype=np.float64)
        entity_axes = np.asarray(entity.pose[:3, :3], dtype=np.float64)
        entity_half = np.asarray(entity.extent, dtype=np.float64) / 2.0
        field_center = np.asarray(field.center, dtype=np.float64)
        field_axes = np.asarray(field.axes, dtype=np.float64)
        field_half = np.asarray(field.half_extents, dtype=np.float64)
        minimum_span = float(
            min(np.min(2.0 * entity_half), np.min(2.0 * field_half))
        )
        tolerance = max(0.001, min(0.006, 0.10 * minimum_span))

        # Cheap world-AABB rejection comes first.  This also prevents a shared
        # detector provenance id from deleting a spatially distinct component.
        entity_world_half = np.abs(entity_axes) @ entity_half
        field_world_half = np.abs(field_axes) @ field_half
        if np.any(
            np.abs(entity_center - field_center)
            > entity_world_half + field_world_half + tolerance
        ):
            return False

        # Full 15-axis separating-axis test avoids the false overlap that a
        # rotated OBB's enclosing AABB can create.  The bounded tolerance only
        # bridges RGB-D/track surface jitter; it does not alter MPC clearance.
        rotation = entity_axes.T @ field_axes
        translation = entity_axes.T @ (field_center - entity_center)
        absolute_rotation = np.abs(rotation) + 1e-10
        for axis in range(3):
            radius_entity = entity_half[axis]
            radius_field = float(field_half @ absolute_rotation[axis, :])
            if abs(translation[axis]) > radius_entity + radius_field + tolerance:
                return False
        for axis in range(3):
            radius_entity = float(entity_half @ absolute_rotation[:, axis])
            radius_field = field_half[axis]
            projected = abs(float(translation @ rotation[:, axis]))
            if projected > radius_entity + radius_field + tolerance:
                return False
        for entity_axis in range(3):
            entity_next = (entity_axis + 1) % 3
            entity_last = (entity_axis + 2) % 3
            for field_axis in range(3):
                field_next = (field_axis + 1) % 3
                field_last = (field_axis + 2) % 3
                radius_entity = (
                    entity_half[entity_next]
                    * absolute_rotation[entity_last, field_axis]
                    + entity_half[entity_last]
                    * absolute_rotation[entity_next, field_axis]
                )
                radius_field = (
                    field_half[field_next]
                    * absolute_rotation[entity_axis, field_last]
                    + field_half[field_last]
                    * absolute_rotation[entity_axis, field_next]
                )
                projected = abs(
                    translation[entity_last] * rotation[entity_next, field_axis]
                    - translation[entity_next] * rotation[entity_last, field_axis]
                )
                if projected > radius_entity + radius_field + tolerance:
                    return False
        return True

    @staticmethod
    def _obb_contains_either_center(entity: SceneEntity, field: BoxSDF) -> bool:
        """Require strong overlap, not merely two OBB surfaces touching."""

        entity_center = np.asarray(entity.position, dtype=np.float64)
        entity_axes = np.asarray(entity.pose[:3, :3], dtype=np.float64)
        entity_half = np.asarray(entity.extent, dtype=np.float64) / 2.0
        field_center = np.asarray(field.center, dtype=np.float64)
        field_axes = np.asarray(field.axes, dtype=np.float64)
        field_half = np.asarray(field.half_extents, dtype=np.float64)
        minimum_span = float(
            min(np.min(2.0 * entity_half), np.min(2.0 * field_half))
        )
        tolerance = max(0.001, min(0.006, 0.10 * minimum_span))
        entity_center_in_field = bool(
            np.all(
                np.abs((entity_center - field_center) @ field_axes)
                <= field_half + tolerance
            )
        )
        field_center_in_entity = bool(
            np.all(
                np.abs((field_center - entity_center) @ entity_axes)
                <= entity_half + tolerance
            )
        )
        return entity_center_in_field or field_center_in_entity

    def _without_requested_obstacle_duplicates(
        self,
        scene: SceneEstimate,
        requested: set[str],
        entities: Sequence[SceneEntity],
    ) -> SceneEstimate:
        """Remove tagged raw boxes that duplicate an outer requested track."""

        self._last_obstacle_filter_trace = ()
        obstacle = scene.obstacle_sdf
        if not isinstance(obstacle, CompositeSDF):
            return scene
        requested_entities = tuple(
            entity
            for entity in entities
            if self._normalised_label(entity.label) in requested
        )
        if not requested_entities:
            return scene

        direct_matches: dict[int, tuple[SceneEntity, str]] = {}
        matched_provenance_ids: set[str] = set()
        for index, field in enumerate(obstacle.fields):
            if not isinstance(field, BoxSDF):
                continue
            field_label = (
                self._normalised_label(field.source_label)
                if field.source_label is not None
                else None
            )
            for entity in requested_entities:
                same_label = field_label == self._normalised_label(entity.label)
                same_id = (
                    field.source_instance_id is not None
                    and field.source_instance_id == entity.instance_id
                )
                unlabelled = (
                    field_label == "unknown"
                    and not same_label
                    and not same_id
                )
                if not (same_label or same_id or unlabelled):
                    continue
                if not self._obb_surface_overlaps(entity, field):
                    continue
                if unlabelled and (
                    entity.instance_id not in self._last_visible_ids
                    or not self._obb_contains_either_center(entity, field)
                ):
                    continue
                if same_label:
                    reason = "requested_label_obb_overlap"
                elif same_id:
                    reason = "requested_source_id_obb_overlap"
                else:
                    reason = "unlabelled_requested_obb_overlap"
                direct_matches[index] = (entity, reason)
                if field.source_instance_id is not None:
                    matched_provenance_ids.add(field.source_instance_id)
                break

        removals = dict(direct_matches)
        if matched_provenance_ids:
            for index, field in enumerate(obstacle.fields):
                if index in removals or not isinstance(field, BoxSDF):
                    continue
                if field.source_instance_id not in matched_provenance_ids:
                    continue
                for entity in requested_entities:
                    if self._obb_surface_overlaps(entity, field):
                        removals[index] = (
                            entity,
                            "shared_source_instance_id_obb_overlap",
                        )
                        break
        if not removals:
            return scene

        retained = tuple(
            field
            for index, field in enumerate(obstacle.fields)
            if index not in removals
        )
        trace: list[dict[str, object]] = []
        for index in sorted(removals):
            entity, reason = removals[index]
            field = obstacle.fields[index]
            trace.append(
                {
                    "field_index": index,
                    "removed_source_instance_id": getattr(
                        field, "source_instance_id", None
                    ),
                    "removed_source_label": getattr(field, "source_label", None),
                    "matched_requested_instance_id": entity.instance_id,
                    "matched_requested_label": entity.label,
                    "reason": reason,
                }
            )
        self._last_obstacle_filter_trace = tuple(trace)
        self._obstacle_filter_history.extend(dict(item) for item in trace)
        filtered = CompositeSDF(retained) if retained else EmptySDF()
        return replace(scene, obstacle_sdf=filtered)

    def _without_temporal_proprio_self_component(
        self, scene: SceneEstimate
    ) -> SceneEstimate:
        """Remove exactly one recognised field proved to co-move with the EE.

        The proof is installed by ``ContactAwareRouteCController`` only after
        two negative-start observations separated by a released-hand high
        egress.  This application step still requires the same fused id,
        semantic label, dimensions, tool-relative world offset, and a field
        enclosing the current public EE.  Any ambiguity leaves the SDF intact.
        """

        proof = self._temporal_proprio_self_filter
        # Each fresh scene consumes the installed authority.  A fully accepted
        # application below may re-arm it from that same fresh frame for the
        # next replan in this one controller phase; every rejection therefore
        # fails closed permanently instead of being retried against a later
        # view.
        if proof is not None:
            self._temporal_proprio_self_filter = None
        obstacle = scene.obstacle_sdf
        if (
            proof is None
            or self._ee_pose_provider is None
            or not isinstance(obstacle, CompositeSDF)
        ):
            return scene
        ee_pose = np.asarray(self._ee_pose_provider(), dtype=np.float64)
        if (
            ee_pose.shape != (4, 4)
            or not np.all(np.isfinite(ee_pose))
            or not np.allclose(ee_pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6)
            or not np.allclose(
                ee_pose[:3, :3].T @ ee_pose[:3, :3],
                np.eye(3),
                atol=2e-3,
            )
            or float(np.linalg.det(ee_pose[:3, :3])) <= 0.0
        ):
            return scene
        ee = ee_pose[:3, 3]
        try:
            expected_half = np.asarray(
                proof["field_half_extents_m"], dtype=np.float64
            )
            expected_relative = np.asarray(
                proof["field_relative_to_ee_m"], dtype=np.float64
            )
            maximum_relative_error = float(proof["maximum_relative_error_m"])
            installed_ee_pose = np.asarray(
                proof["ee_pose_world"], dtype=np.float64
            )
            source = scene.by_id(str(proof["source_id"]))
            target = scene.by_id(str(proof["target_id"]))
        except (KeyError, TypeError, ValueError, PerceptionError):
            return scene
        raw_visibility = proof.get("visibility_evidence")
        typed_negative_mode = raw_visibility is not None
        typed_visibility = None
        if typed_negative_mode:
            typed_visibility = self._validated_typed_wrist_negative_visibility(
                raw_visibility,
                installed_ee_pose,
            )
            if typed_visibility is None:
                return scene
        if (
            expected_half.shape != (3,)
            or expected_relative.shape != (3,)
            or not np.all(np.isfinite(expected_half))
            or not np.all(np.isfinite(expected_relative))
            or not np.isfinite(maximum_relative_error)
            or source.instance_id == target.instance_id
            or source.instance_id not in self._last_visible_ids
            or target.instance_id not in self._last_visible_ids
            or self._normalised_label(source.label) != proof["source_label"]
            or self._normalised_label(target.label) != proof["target_label"]
        ):
            return scene
        same_family = [
            item
            for item in obstacle.fields
            if isinstance(item, BoxSDF)
            and {"drawer", "cabinet"}
            & set(self._normalised_label(item.source_label or "").split())
        ]
        if len(same_family) != 1:
            return scene
        matches: list[tuple[int, BoxSDF, float, float, float]] = []
        for index, field in enumerate(obstacle.fields):
            if not isinstance(field, BoxSDF):
                continue
            if (
                field.source_instance_id != proof["field_id"]
                or self._normalised_label(field.source_label or "")
                != proof["field_label"]
            ):
                continue
            half_drift = float(
                np.max(np.abs(field.half_extents - expected_half))
            )
            relative_drift = float(
                np.linalg.norm((field.center - ee) - expected_relative)
            )
            raw_distance = float(field.distance(ee))
            try:
                expected_axes = np.asarray(
                    proof["field_axes_world"], dtype=np.float64
                )
                prior_ee_pose = np.asarray(
                    proof["ee_pose_world"], dtype=np.float64
                )
            except (KeyError, TypeError, ValueError):
                continue
            if expected_axes.shape != (3, 3) or prior_ee_pose.shape != (4, 4):
                continue
            delta_rotation = ee_pose[:3, :3] @ prior_ee_pose[:3, :3].T
            predicted_axes = delta_rotation @ expected_axes
            try:
                axes_drift = float(
                    Rotation.from_matrix(field.axes @ predicted_axes.T).magnitude()
                )
            except ValueError:
                continue
            if (
                np.all(
                    np.isfinite((half_drift, relative_drift, raw_distance))
                )
                and half_drift <= 0.012
                and relative_drift <= maximum_relative_error
                and raw_distance <= 0.0
                and np.isfinite(axes_drift)
                and axes_drift <= 0.020
            ):
                matches.append(
                    (
                        index,
                        field,
                        half_drift,
                        relative_drift,
                        raw_distance,
                    )
                )
        if len(matches) != 1:
            return scene
        index, field, half_drift, relative_drift, raw_distance = matches[0]
        visible_containers = [
            entity
            for entity in scene.entities
            if entity.instance_id in self._last_visible_ids
            # The generic SceneEntity carrying this BoxSDF's own provenance is
            # not a second container.  Distinct visible containers still veto
            # the removal below exactly as they did during proof capture.
            and entity.instance_id != field.source_instance_id
            and {"drawer", "cabinet"}
            & set(self._normalised_label(entity.label).split())
        ]
        if any(
            self._obb_surface_overlaps(entity, field)
            and self._obb_contains_either_center(entity, field)
            for entity in visible_containers
        ):
            return scene
        surface_parser = (
            ContactAwareRouteCController._canonical_surface_map
            if typed_negative_mode
            else ContactAwareRouteCController._camera_surface_map
        )
        prior_surfaces = surface_parser(proof.get("surface_points_by_camera"))
        current_surfaces = surface_parser(field.surface_points_by_camera)
        surface_gate = bool(
            prior_surfaces is not None
            and current_surfaces is not None
            and set(prior_surfaces) == set(current_surfaces)
            and (
                set(prior_surfaces) == {"agentview"}
                if typed_negative_mode
                else len(prior_surfaces)
                >= ContactAwareRouteCController._FREE_RIM_VIEW_FIELD_MIN_CAMERAS
            )
            and all(
                len(points)
                >= ContactAwareRouteCController._FREE_RIM_VIEW_FIELD_MIN_POINTS_PER_CAMERA
                for points in (
                    *prior_surfaces.values(),
                    *current_surfaces.values(),
                )
            )
        )
        if not surface_gate:
            return scene
        morphology = (
            ContactAwareRouteCController._typed_negative_panda_tool_surface_evidence
            if typed_negative_mode
            else ContactAwareRouteCController._released_panda_tool_surface_evidence
        )
        current_tool_evidence = {
            name: morphology(points, ee_pose)
            for name, points in current_surfaces.items()
        }
        if not all(
            bool(evidence["accepted"])
            for evidence in current_tool_evidence.values()
        ):
            return scene
        try:
            tool_transform = ee_pose @ np.linalg.inv(
                np.asarray(proof["ee_pose_world"], dtype=np.float64)
            )
        except (KeyError, np.linalg.LinAlgError):
            return scene
        point_errors = {
            name: (
                ContactAwareRouteCController._strict_surface_alignment_error_m
                if typed_negative_mode
                else ContactAwareRouteCController._surface_alignment_error_m
            )(prior_surfaces[name], current_surfaces[name], tool_transform)
            for name in prior_surfaces
        }
        if not all(
            np.isfinite(error)
            and error
            <= ContactAwareRouteCController._FREE_RIM_VIEW_FIELD_MAX_POINT_ERROR_M
            for error in point_errors.values()
        ):
            return scene
        next_visibility = typed_visibility
        if typed_negative_mode:
            assert typed_visibility is not None
            wrist_model = dict(typed_visibility["wrist_camera_model"])
            ee_from_wrist = np.asarray(
                typed_visibility["ee_from_wrist"], dtype=np.float64
            )
            wrist_model["world_from_camera"] = ee_pose @ ee_from_wrist
            frustum = ContactAwareRouteCController._wrist_frustum_evidence(
                field.center,
                field.half_extents,
                field.axes,
                wrist_model,
                current_surfaces["agentview"],
            )
            if (
                frustum.get("accepted") is not True
                or frustum.get("strictly_outside") is not True
            ):
                return scene
            next_visibility = {
                **typed_visibility,
                "wrist_camera_model": wrist_model,
            }
        retained = tuple(
            item
            for field_index, item in enumerate(obstacle.fields)
            if field_index != index
        )
        trace = {
            "field_index": index,
            "removed_source_instance_id": field.source_instance_id,
            "removed_source_label": field.source_label,
            "reason": (
                "three_frame_noncollinear_released_hand_public_proprio_"
                "typed_wrist_negative_self_field"
                if typed_negative_mode
                else "two_frame_released_hand_public_proprio_comoving_self_field"
            ),
            "raw_sdf_m": raw_distance,
            "ee_position_world_m": [float(value) for value in ee],
            "field_center_world_m": [
                float(value) for value in field.center
            ],
            "field_half_extents_m": [
                float(value) for value in field.half_extents
            ],
            "field_relative_to_ee_m": [
                float(value) for value in field.center - ee
            ],
            "relative_offset_drift_m": relative_drift,
            "half_extent_drift_m": half_drift,
            "field_axes_drift_rad": axes_drift,
            "per_camera_tool_local_point_error_m": point_errors,
            "source_id": source.instance_id,
            "target_id": target.instance_id,
        }
        self._last_obstacle_filter_trace = (
            *self._last_obstacle_filter_trace,
            trace,
        )
        self._obstacle_filter_history.append(dict(trace))
        # Continue only as a fresh adjacent-frame chain inside the recursive
        # same-phase retry.  The controller's finally block clears this even
        # after a successful phase; any intervening rejected frame has already
        # consumed it above.
        self._temporal_proprio_self_filter = {
            **proof,
            "field_half_extents_m": field.half_extents.copy(),
            "field_relative_to_ee_m": (field.center - ee).copy(),
            "field_axes_world": field.axes.copy(),
            "surface_points_by_camera": tuple(
                (name, points.copy())
                for name, points in current_surfaces.items()
            ),
            "ee_pose_world": ee_pose.copy(),
            "per_camera_public_tool_evidence": current_tool_evidence,
            "visibility_evidence": next_visibility,
        }
        filtered = CompositeSDF(retained) if retained else EmptySDF()
        return replace(scene, obstacle_sdf=filtered)

    def _without_public_proprio_self_component(
        self, scene: SceneEstimate
    ) -> SceneEstimate:
        """Peel only an unlabelled SDF component enclosing the measured EE.

        RGB-D can segment the wrist and held package as one large neutral
        component.  Such a box makes the policy's measured start pose collide
        with itself.  At each peel step, require that this box is the global
        SDF argmin at public EE proprioception, is explicitly tagged
        ``unknown``, and actually contains the EE.  A recognised or merely
        nearby field stops peeling, preserving the ordinary MPC clearances.
        """

        if self._ee_pose_provider is None:
            return scene
        obstacle = scene.obstacle_sdf
        if not isinstance(obstacle, CompositeSDF):
            return scene

        ee_pose = np.asarray(self._ee_pose_provider(), dtype=np.float64)
        if (
            ee_pose.shape != (4, 4)
            or not np.all(np.isfinite(ee_pose))
            or not np.allclose(ee_pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6)
        ):
            raise PerceptionError("public EE proprioception must be a finite 4x4 pose")
        ee_position = ee_pose[:3, 3]
        remaining = list(enumerate(obstacle.fields))
        removals: list[dict[str, object]] = []

        while remaining:
            distances = np.asarray(
                [float(field.distance(ee_position)) for _, field in remaining],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(distances)):
                # Do not apply a partial self-filter when SDF ordering itself
                # cannot be established from finite sensor geometry.
                return scene
            argmin = int(np.argmin(distances))
            original_index, field = remaining[argmin]
            raw_distance = float(distances[argmin])
            if not isinstance(field, BoxSDF):
                break
            if field.source_instance_id is None or field.source_label is None:
                break
            if self._normalised_label(field.source_label) != "unknown":
                break
            if raw_distance > 0.0:
                break

            removals.append(
                {
                    "field_index": original_index,
                    "removed_source_instance_id": field.source_instance_id,
                    "removed_source_label": field.source_label,
                    "reason": "public_proprio_ee_inside_unknown_self_component",
                    "raw_sdf_m": raw_distance,
                    "ee_position_world_m": [float(value) for value in ee_position],
                    "field_center_world_m": [
                        float(value) for value in field.center
                    ],
                    "field_half_extents_m": [
                        float(value) for value in field.half_extents
                    ],
                }
            )
            del remaining[argmin]

        if not removals:
            return scene
        retained = tuple(field for _, field in remaining)
        self._last_obstacle_filter_trace = (
            *self._last_obstacle_filter_trace,
            *removals,
        )
        self._obstacle_filter_history.extend(dict(item) for item in removals)
        filtered = CompositeSDF(retained) if retained else EmptySDF()
        return replace(scene, obstacle_sdf=filtered)

    def _association_limit_m(self, previous: SceneEntity) -> float:
        """Object-scale nearest-neighbour gate, capped at the configured radius."""

        world_span = np.abs(previous.pose[:3, :3]) @ previous.extent
        planar_span = float(np.max(world_span[:2]))
        scaled = max(
            self.minimum_association_distance_m,
            self.association_extent_scale * planar_span,
        )
        return min(self.association_distance_m, scaled)

    def _unique_id(
        self, proposed: str, *, existing: Mapping[str, SceneEntity] | None = None
    ) -> str:
        occupied = set(self._tracks)
        if existing is not None:
            occupied.update(existing)
        if proposed not in occupied:
            return proposed
        while True:
            candidate = f"tracked-{self._next_id:04d}"
            self._next_id += 1
            if candidate not in occupied:
                return candidate


@dataclass(frozen=True)
class _FormedStackBinding:
    top_id: str
    bottom_id: str
    top_label: str
    bottom_label: str
    top_shape: SceneEntity
    bottom_shape: SceneEntity


class SensorRankedTargetHandler:
    """Prefetch sensor identities for future left/right/front/back targets."""

    _RELATIONS = {
        SelectorKind.LEFT: Relation.LEFTMOST,
        SelectorKind.RIGHT: Relation.RIGHTMOST,
        SelectorKind.FRONT: Relation.FRONTMOST,
        SelectorKind.BACK: Relation.BACKMOST,
        SelectorKind.MIDDLE: Relation.MIDDLE,
        SelectorKind.FIRST: Relation.FIRST,
        SelectorKind.SECOND: Relation.SECOND,
    }

    def __init__(
        self,
        observer: StableSceneEstimator,
        resolver: WorkspaceCenterEntityResolver,
    ) -> None:
        self.observer = observer
        self.resolver = resolver
        self._bindings: dict[int, str] = {}

    def prefetch(self, plan: TaskPlan) -> str | None:
        self._bindings.clear()
        self.resolver.clear_required_target_identity()
        ranked = [
            goal
            for goal in plan.goals
            if goal.target is not None
            and goal.target.selector is not None
            and goal.target.selector.kind in self._RELATIONS
        ]
        if not ranked:
            return None
        labels = tuple(
            dict.fromkeys(goal.target.label for goal in ranked if goal.target)
        )
        try:
            scene = self.observer.observe(labels)
            visible = set(self.observer.visible_instance_ids)
            selected_by_label: dict[str, list[tuple[SelectorKind, str]]] = {}
            for goal in ranked:
                assert goal.target is not None
                assert goal.target.selector is not None
                relation = self._RELATIONS[goal.target.selector.kind]
                selected = self.resolver.resolve(
                    EntityRef(
                        "target",
                        goal.target.label,
                        SpatialSelector(relation),
                    ),
                    scene,
                )
                if selected.instance_id not in visible:
                    return (
                        f"ranked target {selected.instance_id!r} is not backed "
                        "by the pre-action RGB-D frame"
                    )
                prior = selected_by_label.setdefault(goal.target.label, [])
                if any(
                    kind is not goal.target.selector.kind
                    and instance_id == selected.instance_id
                    for kind, instance_id in prior
                ):
                    self._bindings.clear()
                    return (
                        f"distinct ranked {goal.target.label!r} destinations "
                        "collapsed to one visual identity"
                    )
                prior.append(
                    (goal.target.selector.kind, selected.instance_id)
                )
                self._bindings[id(goal)] = selected.instance_id
        except (PerceptionError, LookupError, ValueError) as exc:
            self._bindings.clear()
            return str(exc)
        return None

    def prepare_target(self, goal: AtomicGoal) -> str | None:
        instance_id = self._bindings.get(id(goal))
        if instance_id is None:
            return None
        try:
            self.resolver.require_target_identity(instance_id)
        except ValueError as exc:
            return str(exc)
        return None

    def clear_target_requirement(self) -> None:
        self.resolver.clear_required_target_identity()


class SensorFormedStackHandler:
    """Prove and preserve a newly stacked pair using fresh RGB-D only.

    The handler records the stable identities produced by the first Route-C
    stack goal, pins the following carry to the visible lower object, and
    finally verifies both the destination containment and the retained ON
    relation.  A fused component, an occluded member, or an ambiguous
    reacquisition fails closed.
    """

    def __init__(
        self,
        observer: StableSceneEstimator,
        resolver: WorkspaceCenterEntityResolver,
        *,
        on_tolerance_m: float = 0.035,
        containment_tolerance_m: float = 0.025,
        shape_log_tolerance: float = 0.45,
    ) -> None:
        if min(
            on_tolerance_m,
            containment_tolerance_m,
            shape_log_tolerance,
        ) <= 0:
            raise ValueError("formed-stack proof tolerances must be positive")
        self.observer = observer
        self.resolver = resolver
        self.on_tolerance_m = float(on_tolerance_m)
        self.containment_tolerance_m = float(containment_tolerance_m)
        self.shape_log_tolerance = float(shape_log_tolerance)
        self._binding: _FormedStackBinding | None = None

    def capture_stack(
        self, goal: AtomicGoal, result: RouteCResult
    ) -> Verification:
        self._binding = None
        if (
            result.source_id is None
            or result.target_id is None
            or result.source_id == result.target_id
        ):
            return self._failure(
                Relation.ON,
                "stack result did not preserve distinct top/bottom identities",
            )
        top_label = goal.subject.label
        assert goal.target is not None
        bottom_label = goal.target.label
        try:
            scene, visible = self._fresh_scene((top_label, bottom_label))
            top = self._fresh_exact(
                scene, visible, result.source_id, top_label, "top"
            )
            bottom = self._fresh_exact(
                scene, visible, result.target_id, bottom_label, "bottom"
            )
        except (PerceptionError, LookupError, ValueError) as exc:
            return self._failure(Relation.ON, str(exc))
        proof = self._on_proof(top, bottom, prefix="post-stack")
        if proof.success:
            self._binding = _FormedStackBinding(
                result.source_id,
                result.target_id,
                top_label,
                bottom_label,
                top,
                bottom,
            )
        return proof

    def prepare_carry(self, goal: AtomicGoal) -> Verification:
        binding = self._binding
        if binding is None:
            return self._failure(
                Relation.ON,
                "formed-stack identities were not captured before carry",
            )
        if goal.relation is None or goal.relation.value != Relation.IN.value:
            return self._failure(
                Relation.IN,
                "formed-stack verifier supports only containment carries",
            )
        if len(goal.subjects) != 2:
            return self._failure(
                Relation.ON, "formed-stack carry requires exactly two members"
            )
        try:
            scene, visible = self._fresh_scene(
                (binding.top_label, binding.bottom_label)
            )
            top = self._fresh_exact(
                scene, visible, binding.top_id, binding.top_label, "top"
            )
            bottom = self._fresh_exact(
                scene,
                visible,
                binding.bottom_id,
                binding.bottom_label,
                "bottom",
            )
        except (PerceptionError, LookupError, ValueError) as exc:
            return self._failure(Relation.ON, str(exc))
        proof = self._on_proof(top, bottom, prefix="pre-carry")
        if proof.success:
            self.resolver.require_source_identity(binding.bottom_id)
        return proof

    def verify_carry(
        self, goal: AtomicGoal, result: RouteCResult
    ) -> Verification:
        binding = self._binding
        if binding is None:
            return self._failure(
                Relation.IN,
                "formed-stack binding disappeared before final verification",
            )
        if goal.relation is None or goal.relation.value != Relation.IN.value:
            return self._failure(
                Relation.IN,
                "formed-stack verifier supports only containment carries",
            )
        if result.source_id != binding.bottom_id:
            return self._failure(
                Relation.IN,
                "carry controller did not manipulate the captured bottom identity",
            )
        if result.target_id is None or goal.target is None:
            return self._failure(
                Relation.IN,
                "carry result did not preserve the destination identity",
            )
        if result.verification is None or not result.verification.success:
            return self._failure(
                Relation.IN,
                "ordinary bottom-to-destination verification did not succeed",
            )

        try:
            scene, visible = self._fresh_scene(
                (binding.top_label, binding.bottom_label, goal.target.label)
            )
            bottom = self._fresh_exact(
                scene,
                visible,
                binding.bottom_id,
                binding.bottom_label,
                "bottom",
            )
            target = self._fresh_exact(
                scene,
                visible,
                result.target_id,
                goal.target.label,
                "destination",
            )
            top = self._reacquire_top(scene, visible, binding, bottom)
        except (PerceptionError, LookupError, ValueError) as exc:
            return self._failure(Relation.IN, str(exc))

        on_proof = self._on_proof(top, bottom, prefix="post-carry")
        inside_error = self._containment_error(bottom, target)
        inside_success = inside_error <= self.containment_tolerance_m
        success = on_proof.success and inside_success
        error = max(on_proof.position_error_m, inside_error)
        return Verification(
            success,
            Relation.IN,
            error,
            "formed-stack dual proof: "
            f"bottom containment={inside_error:.4f} m; {on_proof.detail}",
        )

    def clear_carry_requirement(self) -> None:
        self.resolver.clear_required_source_identity()
        self._binding = None

    def _fresh_scene(
        self, labels: Sequence[str]
    ) -> tuple[SceneEstimate, set[str]]:
        scene = self.observer.observe(tuple(dict.fromkeys(labels)))
        visible = {
            str(instance_id)
            for instance_id in self.observer.visible_instance_ids
        }
        return scene, visible

    @staticmethod
    def _fresh_exact(
        scene: SceneEstimate,
        visible: Collection[str],
        instance_id: str,
        label: str,
        role: str,
    ) -> SceneEntity:
        if instance_id not in visible:
            raise PerceptionError(
                f"formed-stack {role} identity {instance_id!r} is not freshly visible"
            )
        try:
            entity = scene.by_id(instance_id)
        except PerceptionError as exc:
            raise PerceptionError(
                f"formed-stack {role} identity {instance_id!r} is absent"
            ) from exc
        if entity.label != label:
            raise PerceptionError(
                f"formed-stack {role} identity changed label from "
                f"{label!r} to {entity.label!r}"
            )
        return entity

    def _reacquire_top(
        self,
        scene: SceneEstimate,
        visible: set[str],
        binding: _FormedStackBinding,
        bottom: SceneEntity,
    ) -> SceneEntity:
        if binding.top_id in visible:
            return self._fresh_exact(
                scene,
                visible,
                binding.top_id,
                binding.top_label,
                "top",
            )
        candidates = [
            entity
            for entity in scene.entities
            if entity.instance_id in visible
            and entity.instance_id != bottom.instance_id
            and entity.label == binding.top_label
            and self._shape_matches(entity, binding.top_shape)
            and self._on_error(entity, bottom) <= self.on_tolerance_m
        ]
        if len(candidates) != 1:
            raise PerceptionError(
                "formed-stack top reacquisition is ambiguous after carry: "
                f"{len(candidates)} valid fresh candidates"
            )
        selected = candidates[0]
        self.observer.rebind_track(binding.top_id, selected)
        return replace(selected, instance_id=binding.top_id)

    def _shape_matches(
        self, candidate: SceneEntity, initial: SceneEntity
    ) -> bool:
        candidate_span = np.sort(
            np.abs(candidate.pose[:3, :3]) @ candidate.extent
        )
        initial_span = np.sort(np.abs(initial.pose[:3, :3]) @ initial.extent)
        log_ratio = np.abs(
            np.log(
                np.maximum(candidate_span, 1e-5)
                / np.maximum(initial_span, 1e-5)
            )
        )
        return bool(np.max(log_ratio) <= self.shape_log_tolerance)

    def _on_proof(
        self, top: SceneEntity, bottom: SceneEntity, *, prefix: str
    ) -> Verification:
        error = self._on_error(top, bottom)
        return Verification(
            error <= self.on_tolerance_m,
            Relation.ON,
            error,
            f"{prefix} top-on-bottom error={error:.4f} m",
        )

    @staticmethod
    def _on_error(top: SceneEntity, bottom: SceneEntity) -> float:
        if bottom.region is None:
            return float("inf")
        local = bottom.region.local_coordinates(top.position[None, :])[0]
        planar_overflow = float(
            np.linalg.norm(
                np.maximum(
                    np.abs(local[:2]) - bottom.region.half_extents[:2],
                    0.0,
                )
            )
        )
        top_half_z = float(
            np.abs(top.pose[:3, :3][2]) @ (top.extent / 2.0)
        )
        bottom_half_z = float(
            np.abs(bottom.pose[:3, :3][2]) @ (bottom.extent / 2.0)
        )
        contact_error = abs(
            float(
                (top.position[2] - top_half_z)
                - (bottom.position[2] + bottom_half_z)
            )
        )
        order_error = max(float(bottom.position[2] - top.position[2]), 0.0)
        return max(planar_overflow, contact_error, order_error)

    @staticmethod
    def _containment_error(source: SceneEntity, target: SceneEntity) -> float:
        if target.region is None:
            return float("inf")
        signs = np.array(
            [
                (x, y, z)
                for x in (-1.0, 1.0)
                for y in (-1.0, 1.0)
                for z in (-1.0, 1.0)
            ],
            dtype=np.float64,
        )
        local_corners = signs * (source.extent / 2.0)
        corners = (
            local_corners @ source.pose[:3, :3].T + source.position
        )
        target_local = np.abs(target.region.local_coordinates(corners))
        overflow = np.maximum(
            target_local - target.region.half_extents,
            0.0,
        )
        return float(np.max(np.linalg.norm(overflow, axis=1)))

    @staticmethod
    def _failure(relation: Relation, detail: str) -> Verification:
        return Verification(False, relation, float("inf"), detail)
