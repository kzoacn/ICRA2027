"""Fail-closed, evaluator-free Route-B episode runtime.

The production worker owns one exactly assembled :class:`RouteBPolicy` and a
single frozen perception bundle for its lifetime.  Each episode receives only
an exact two-field :class:`PolicyTask`, an independent process episode nonce,
the initial dual-RGB-D/proprioception DTO, and an action capability returning
the next DTO.  No environment, benchmark identity, reward, evaluator signal,
BDDL state, simulator object state, or simulator package is imported here.

The formal action horizon is exactly 520.  A terminal policy decision consumes
one observation/``act`` call but its action is not executed.  A timeout executes
exactly 520 actions, receives observation 521, and performs no hidden 521st
``act``.  Therefore every successful return proves ``O = A + 1``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import re
import secrets
import threading
from types import MappingProxyType
import unicodedata
from typing import Any, Callable

import numpy as np

from libero_system.common.grasp_journal import (
    GraspAttemptEvent,
    GraspEvidenceCategory,
    GraspOutcome,
    GraspReason,
    JawBehavior,
    PendingGraspEngagement,
)
from libero_system.common.observation import (
    CameraCalibration,
    CameraFrame,
    Proprioception,
    RobotObservation,
)
from libero_system.common.policy import OSCAction, PolicyDecision, PolicyTask
from libero_system.integration.formal_process_transport import (
    MAX_JSON_CONTAINER_ITEMS,
    MAX_JSON_DEPTH,
    MAX_JSON_FRAME_BYTES,
    MAX_JSON_NODES,
    strict_json_bytes,
    strict_json_loads,
)


ActionExecutor = Callable[[OSCAction], RobotObservation]

MAX_ROUTE_B_POLICY_STEPS = 520
MAX_ROUTE_B_INSTRUCTION_CHARS = 4_096
ROUTE_B_WORKER_RUNTIME_SCHEMA = "libero-route-b-worker-runtime.v2"
ROUTE_B_EPISODE_RUNTIME_SCHEMA = "libero-route-b-episode-runtime.v2"

_OPAQUE_EPISODE_ID = re.compile(r"episode-v2-[0-9a-f]{64}")
_LOWER_HEX_64 = re.compile(r"[0-9a-f]{64}")
_BLACK_BOWL = re.compile(r"(?<!\w)black(?:[\s_-]+)bowl(?!\w)")
_POLICY_STATUSES = frozenset({"idle", "running", "succeeded", "failed"})
_RESULT_STATUSES = frozenset({"succeeded", "failed", "timeout"})
_PRODUCTION_ASSEMBLY_TOKEN = object()
_TEST_ASSEMBLY_TOKEN = object()
_RESET_CALL_KEYS = frozenset(
    {
        "route_b_policy",
        "route_b_controller_clear",
        "route_b_controller_activate",
        "goal_contact_policy_clear",
        "goal_contact_policy_activate",
        "route_b_microwave_detector_reset",
        "microwave_handle_detector_reset",
        "route_b_perception_adapter_reset",
    }
)


def _type_name(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _is_nonce(value: object) -> bool:
    return type(value) is str and _LOWER_HEX_64.fullmatch(value) is not None


def _new_child_nonce() -> str:
    """Call the child entropy source exactly at the nonce construction point."""

    value = secrets.token_hex(32)
    if not _is_nonce(value):
        raise RuntimeError("child entropy source returned a non-canonical nonce")
    return value


def _strict_json_detach(value: object) -> dict[str, Any]:
    """Validate transport limits and return a detached exact JSON object."""

    if type(value) is not dict:
        raise TypeError("Route B trace must be an exact dict")
    # The shared codec enforces 2 MiB, depth 18, 24k nodes, 2048 container
    # items, finite numbers, signed int64, and <=256-character native keys.
    return strict_json_loads(strict_json_bytes(value))


def _detach_observation(value: object) -> RobotObservation:
    """Rebuild an exact sensor DTO, validating every nested field/array."""

    if type(value) is not RobotObservation:
        raise RuntimeError("Route B received a non-whitelist observation DTO")
    cameras_value = value.cameras
    if type(cameras_value) is not MappingProxyType or set(cameras_value) != {
        "agentview",
        "wrist",
    }:
        raise TypeError("Route B observation camera mapping is not canonical")
    cameras: dict[str, CameraFrame] = {}
    for name in ("agentview", "wrist"):
        frame = cameras_value[name]
        if type(frame) is not CameraFrame:
            raise TypeError("Route B camera frame must use the exact whitelist DTO")
        calibration = frame.calibration
        if type(calibration) is not CameraCalibration:
            raise TypeError("Route B camera calibration must use the exact whitelist DTO")
        if (
            type(calibration.name) is not str
            or calibration.name != name
            or type(calibration.width) is not int
            or type(calibration.height) is not int
            or type(calibration.observation_v_flipped) is not bool
        ):
            raise TypeError("Route B camera calibration scalars are not canonical")
        rebuilt_calibration = CameraCalibration(
            name=calibration.name,
            width=calibration.width,
            height=calibration.height,
            intrinsic=np.array(calibration.intrinsic, copy=True),
            T_world_camera=np.array(calibration.T_world_camera, copy=True),
            observation_v_flipped=calibration.observation_v_flipped,
        )
        cameras[name] = CameraFrame(
            rgb=np.array(frame.rgb, copy=True),
            depth_m=np.array(frame.depth_m, copy=True),
            calibration=rebuilt_calibration,
        )
    proprio = value.proprio
    if type(proprio) is not Proprioception:
        raise TypeError("Route B proprioception must use the exact whitelist DTO")
    width = proprio.gripper_width_m
    if type(width) is not float:
        raise TypeError("Route B gripper width must be a native float")
    return RobotObservation(
        cameras=cameras,
        proprio=Proprioception(
            T_world_ee=np.array(proprio.T_world_ee, copy=True),
            ee_quat_xyzw=np.array(proprio.ee_quat_xyzw, copy=True),
            joint_position=np.array(proprio.joint_position, copy=True),
            joint_velocity=np.array(proprio.joint_velocity, copy=True),
            gripper_qpos=np.array(proprio.gripper_qpos, copy=True),
            gripper_qvel=np.array(proprio.gripper_qvel, copy=True),
            gripper_width_m=width,
            ee_force_sensor=np.array(proprio.ee_force_sensor, copy=True),
            ee_torque_sensor=np.array(proprio.ee_torque_sensor, copy=True),
        ),
    )


def _detach_action(value: object) -> OSCAction:
    """Copy and revalidate one exact normalized action before capability use."""

    if type(value) is not OSCAction:
        raise TypeError("Route B action must be the exact whitelist DTO")
    return OSCAction(np.array(value.values, dtype=np.float32, copy=True))


def _json_sha256(value: dict[str, Any], *, domain: bytes) -> str:
    return hashlib.sha256(domain + strict_json_bytes(value)).hexdigest()


def _text_sha256(value: str, *, domain: bytes) -> str:
    encoded = value.encode("utf-8")
    return hashlib.sha256(
        domain + len(encoded).to_bytes(8, "big") + encoded
    ).hexdigest()


def _enum_string(value: object, *, name: str) -> str:
    raw = getattr(value, "value", value)
    if type(raw) is not str:
        raise RuntimeError(f"Route B {name} is not a native string")
    return raw


def _bounded_semantic(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise RuntimeError(f"Route B {name} is not a native string")
    if not value or len(value) > 256:
        raise RuntimeError(f"Route B {name} is empty or too long")
    if any(unicodedata.category(character).startswith("C") for character in value):
        raise RuntimeError(f"Route B {name} contains a control character")
    normalized = " ".join(value.casefold().replace("_", " ").split())
    if not normalized:
        raise RuntimeError(f"Route B {name} is empty after normalization")
    return normalized


def _black_bowl_contract(
    *,
    source_text: object,
    source_class: object,
    grasp_mode: object,
    jaw_behavior: object,
) -> None:
    normalized_text = _bounded_semantic(source_text, name="grasp source_text")
    normalized_class = _bounded_semantic(source_class, name="grasp source_class")
    if type(grasp_mode) is not str or grasp_mode not in {
        "pinch",
        "rim_pinch",
        "expand",
    }:
        raise RuntimeError("Route B grasp mode is invalid")
    if type(jaw_behavior) is not JawBehavior:
        raise RuntimeError("Route B jaw behavior is invalid")
    expected_jaw = (
        JawBehavior.OPEN_FINGERS_INTERIOR_BRACE
        if grasp_mode == "expand"
        else JawBehavior.CLOSE_FINGERS
    )
    if jaw_behavior is not expected_jaw:
        raise RuntimeError("Route B grasp mode and jaw behavior disagree")
    text_tokens = tuple(re.findall(r"[^\W_]+", normalized_text))
    class_tokens = tuple(re.findall(r"[^\W_]+", normalized_class))
    is_black_bowl = bool(
        class_tokens in {("black", "bowl"), ("bowl",)}
        or any(
            text_tokens[index : index + 2] == ("black", "bowl")
            for index in range(max(0, len(text_tokens) - 1))
        )
        or _BLACK_BOWL.search(normalized_text)
    )
    if is_black_bowl and (
        grasp_mode != "rim_pinch"
        or jaw_behavior is not JawBehavior.CLOSE_FINGERS
    ):
        raise RuntimeError(
            "black bowl requires closed-finger rim-pinch at the bowl edge"
        )


def _validate_grasp_lifecycle(
    *,
    events: object,
    pending: object,
    status: str,
    action_count: int,
) -> tuple[tuple[GraspAttemptEvent, ...], PendingGraspEngagement | None]:
    if type(events) is not tuple:
        raise TypeError("Route B grasp_attempt_events must be an exact tuple")
    if len(events) > action_count:
        raise RuntimeError("Route B has more grasp engagements than actions")
    event_rows: list[dict[str, Any]] = []
    for expected_index, event in enumerate(events, start=1):
        if type(event) is not GraspAttemptEvent:
            raise TypeError("Route B grasp events must use exact common DTOs")
        attempt_index = event.attempt_index
        source_text = event.source_text
        source_class = event.source_class
        grasp_mode = event.grasp_mode
        jaw_behavior = event.jaw_behavior
        accepted = event.accepted
        outcome = event.outcome
        reason = event.reason
        evidence_source = event.evidence_source
        if type(attempt_index) is not int or attempt_index != expected_index:
            raise RuntimeError("Route B grasp attempt indices are not contiguous")
        _black_bowl_contract(
            source_text=source_text,
            source_class=source_class,
            grasp_mode=grasp_mode,
            jaw_behavior=jaw_behavior,
        )
        if type(accepted) is not bool:
            raise RuntimeError("Route B grasp accepted flag is invalid")
        if type(outcome) is not GraspOutcome:
            raise RuntimeError("Route B grasp outcome is invalid")
        if type(reason) is not GraspReason:
            raise RuntimeError("Route B grasp reason is invalid")
        if type(evidence_source) is not GraspEvidenceCategory:
            raise RuntimeError("Route B grasp evidence source is invalid")
        if accepted is not (outcome is GraspOutcome.ACCEPTED):
            raise RuntimeError("Route B grasp accepted/outcome values disagree")
        if accepted is not (reason is GraspReason.ACCEPTED):
            raise RuntimeError("Route B grasp accepted/reason values disagree")
        event_rows.append(
            {
                "attempt_index": attempt_index,
                "source_text": source_text,
                "source_class": source_class,
                "grasp_mode": grasp_mode,
                "jaw_behavior": jaw_behavior.value,
                "accepted": accepted,
                "outcome": outcome.value,
                "reason": reason.value,
                "evidence_source": evidence_source.value,
            }
        )

    pending_row: dict[str, Any] | None = None
    if pending is not None:
        if type(pending) is not PendingGraspEngagement:
            raise TypeError("Route B pending grasp must use the exact common DTO")
        if status != "timeout":
            raise RuntimeError("Route B terminal policy left a pending grasp")
        attempt_index = pending.attempt_index
        source_text = pending.source_text
        source_class = pending.source_class
        grasp_mode = pending.grasp_mode
        jaw_behavior = pending.jaw_behavior
        if (
            type(attempt_index) is not int
            or attempt_index != len(event_rows) + 1
            or attempt_index > action_count
        ):
            raise RuntimeError("Route B pending grasp index/action lifecycle is invalid")
        _black_bowl_contract(
            source_text=source_text,
            source_class=source_class,
            grasp_mode=grasp_mode,
            jaw_behavior=jaw_behavior,
        )
        pending_row = {
            "attempt_index": attempt_index,
            "source_text": source_text,
            "source_class": source_class,
            "grasp_mode": grasp_mode,
            "jaw_behavior": jaw_behavior.value,
        }

    # Enforce the same frame bound on all non-trace evidence, then rebuild
    # exact common DTOs so the result never retains a caller-owned object.
    detached = _strict_json_detach(
        {"events": event_rows, "pending": pending_row}
    )
    rebuilt_events = tuple(
        GraspAttemptEvent(
            attempt_index=row["attempt_index"],
            source_text=row["source_text"],
            source_class=row["source_class"],
            grasp_mode=row["grasp_mode"],
            jaw_behavior=JawBehavior(row["jaw_behavior"]),
            accepted=row["accepted"],
            outcome=GraspOutcome(row["outcome"]),
            reason=GraspReason(row["reason"]),
            evidence_source=GraspEvidenceCategory(row["evidence_source"]),
        )
        for row in detached["events"]
    )
    row = detached["pending"]
    rebuilt_pending = (
        None
        if row is None
        else PendingGraspEngagement(
            attempt_index=row["attempt_index"],
            source_text=row["source_text"],
            source_class=row["source_class"],
            grasp_mode=row["grasp_mode"],
            jaw_behavior=JawBehavior(row["jaw_behavior"]),
        )
    )
    return rebuilt_events, rebuilt_pending


@dataclass(frozen=True, slots=True)
class RouteBPolicyExecution:
    """Detached policy result matching Route C's common six-field wire shape."""

    success: bool
    failure: str | None
    steps_executed: int
    trace: dict[str, Any]
    grasp_attempt_events: tuple[GraspAttemptEvent, ...]
    pending_grasp_engagement: PendingGraspEngagement | None

    def __post_init__(self) -> None:
        if type(self.success) is not bool:
            raise TypeError("Route B policy success must be a native boolean")
        if type(self.failure) not in {str, type(None)}:
            raise TypeError("Route B policy failure must be a native string or None")
        if type(self.failure) is str and len(self.failure) > MAX_ROUTE_B_INSTRUCTION_CHARS:
            raise ValueError("Route B policy failure text is too long")
        if type(self.steps_executed) is not int or not (
            0 <= self.steps_executed <= MAX_ROUTE_B_POLICY_STEPS
        ):
            raise ValueError("Route B steps_executed is outside the fixed horizon")
        detached_trace = _strict_json_detach(self.trace)
        status = detached_trace.get("status")
        if type(status) is not str or status not in _RESULT_STATUSES:
            raise ValueError("Route B policy status is invalid")
        if self.success is not (status == "succeeded"):
            raise ValueError("Route B success and status disagree")
        if status == "succeeded" and self.failure is not None:
            raise ValueError("a succeeded Route B result cannot have a failure")
        if status != "succeeded" and self.failure is None:
            raise ValueError("a non-succeeded Route B result requires a failure")
        events, pending = _validate_grasp_lifecycle(
            events=self.grasp_attempt_events,
            pending=self.pending_grasp_engagement,
            status=status,
            action_count=self.steps_executed,
        )
        complete_wire = _strict_json_detach(
            {
                "success": self.success,
                "failure": self.failure,
                "steps_executed": self.steps_executed,
                "trace": detached_trace,
                "grasp_attempt_events": [
                    GraspAttemptEvent.to_dict(event) for event in events
                ],
                "pending_grasp_engagement": (
                    None
                    if pending is None
                    else PendingGraspEngagement.to_dict(pending)
                ),
            }
        )
        object.__setattr__(self, "trace", complete_wire["trace"])
        object.__setattr__(self, "grasp_attempt_events", events)
        object.__setattr__(self, "pending_grasp_engagement", pending)

    @property
    def status(self) -> str:
        return self.trace["status"]


@dataclass(frozen=True, slots=True)
class _AssemblyEvidence:
    token: object
    production_assembly: bool
    assembly_nonce: str
    component_type_manifest: dict[str, str]
    component_instance_counts: dict[str, int]
    component_construction_counts: dict[str, int]
    component_identity_sha256: str

    def __post_init__(self) -> None:
        if type(self.production_assembly) is not bool:
            raise TypeError("Route B production_assembly must be a native boolean")
        expected_token = (
            _PRODUCTION_ASSEMBLY_TOKEN
            if self.production_assembly
            else _TEST_ASSEMBLY_TOKEN
        )
        if self.token is not expected_token or not _is_nonce(self.assembly_nonce):
            raise RuntimeError("Route B assembly evidence is not authentic")
        for value in (
            self.component_type_manifest,
            self.component_instance_counts,
            self.component_construction_counts,
        ):
            if type(value) is not dict:
                raise TypeError("Route B assembly manifest must use exact dicts")
        if self.production_assembly:
            expected_manifest = {
                "perception_bundle",
                "route_b_perception_adapter",
                "route_b_controller",
                "route_b_policy",
                "route_b_runtime",
            }
            expected_instances = set(expected_manifest)
            expected_constructions = {
                "route_b_perception_adapter",
                "route_b_controller",
                "route_b_policy",
                "route_b_runtime",
            }
        else:
            expected_manifest = {"test_policy", "route_b_runtime"}
            expected_instances = {"test_policy", "route_b_runtime"}
            expected_constructions = {"route_b_runtime"}
        if (
            set(self.component_type_manifest) != expected_manifest
            or set(self.component_instance_counts) != expected_instances
            or set(self.component_construction_counts) != expected_constructions
            or any(
                type(value) is not str or not value or len(value) > 256
                for value in self.component_type_manifest.values()
            )
            or any(
                type(value) is not int or value != 1
                for value in self.component_instance_counts.values()
            )
            or any(
                type(value) is not int or value != 1
                for value in self.component_construction_counts.values()
            )
        ):
            raise RuntimeError("Route B assembly manifest/count set is invalid")
        detached = _strict_json_detach(
            {
                "component_type_manifest": self.component_type_manifest,
                "component_instance_counts": self.component_instance_counts,
                "component_construction_counts": self.component_construction_counts,
            }
        )
        object.__setattr__(
            self,
            "component_type_manifest",
            detached["component_type_manifest"],
        )
        object.__setattr__(
            self,
            "component_instance_counts",
            detached["component_instance_counts"],
        )
        object.__setattr__(
            self,
            "component_construction_counts",
            detached["component_construction_counts"],
        )
        if not _is_nonce(self.component_identity_sha256):
            raise RuntimeError("Route B component identity commitment is invalid")


def _component_identity(
    assembly_nonce: str,
    components: dict[str, object],
) -> str:
    digest = hashlib.sha256(
        b"libero-route-b-production-components.v2\0"
        + bytes.fromhex(assembly_nonce)
    )
    for name, component in sorted(components.items()):
        encoded_name = name.encode("utf-8")
        encoded_type = _type_name(component).encode("utf-8")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(len(encoded_type).to_bytes(4, "big"))
        digest.update(encoded_type)
        digest.update(str(id(component)).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _decision_state(policy: Any) -> tuple[str, int, str, str]:
    decision = getattr(policy, "last_decision")
    if decision is None:
        raise RuntimeError("Route B policy did not retain its last decision")
    status = _enum_string(getattr(decision, "status"), name="status")
    if status not in _POLICY_STATUSES:
        raise RuntimeError(f"Route B policy returned invalid status {status!r}")
    skill_index = getattr(decision, "skill_index")
    if type(skill_index) is not int or skill_index < 0:
        raise RuntimeError("Route B decision skill_index is invalid")
    phase = getattr(decision, "phase")
    message = getattr(decision, "message")
    if type(phase) is not str or not phase or len(phase) > 256:
        raise RuntimeError("Route B decision phase is invalid")
    if type(message) is not str or len(message) > MAX_ROUTE_B_INSTRUCTION_CHARS:
        raise RuntimeError("Route B decision message is invalid")
    return status, skill_index, phase, message


def _rows(policy: Any, attribute: str) -> list[dict[str, Any]]:
    value = getattr(policy, attribute)
    if type(value) is not tuple:
        raise RuntimeError(f"Route B {attribute} must be an exact tuple")
    rows: list[dict[str, Any]] = []
    for item in value:
        if type(item) is not dict:
            raise RuntimeError(f"Route B {attribute} contains a non-exact dict")
        rows.append(item)
    return rows


def _events(policy: Any) -> tuple[GraspAttemptEvent, ...]:
    value = getattr(policy, "grasp_attempt_events")
    if type(value) is not tuple:
        raise RuntimeError("Route B grasp event collection must be an exact tuple")
    return value


def _pending(policy: Any) -> PendingGraspEngagement | None:
    return getattr(policy, "pending_grasp_engagement")


def _post_reset_state(
    policy: Any,
    *,
    production: bool,
    task: PolicyTask,
    reset_call_counts: dict[str, int],
) -> dict[str, Any]:
    """Commit every wrapper journal plus critical controller reset state."""

    if (
        type(reset_call_counts) is not dict
        or set(reset_call_counts) != _RESET_CALL_KEYS
        or any(type(value) is not int or value < 0 for value in reset_call_counts.values())
        or reset_call_counts["route_b_policy"] != 1
    ):
        raise RuntimeError("Route B reset call-count receipt is invalid")
    state: dict[str, Any] = {
        "drawer_contact_proof_present": (
            getattr(policy, "drawer_contact_proof") is not None
        ),
        "grasp_attempt_events": len(_events(policy)),
        "grasp_target_attempts": len(_rows(policy, "grasp_target_attempts")),
        "grasp_verifications": len(_rows(policy, "grasp_verifications")),
        "last_decision_present": getattr(policy, "last_decision") is not None,
        "observation_sequence_token_present": (
            getattr(policy, "observation_sequence_token") is not None
        ),
        "pending_grasp_engagement_present": _pending(policy) is not None,
        "placement_target_attempts": len(
            _rows(policy, "placement_target_attempts")
        ),
        "selector_diagnostics": len(_rows(policy, "selector_diagnostics")),
    }
    if production:
        from libero_system.goal_skills.controller import GoalContactPolicy
        from libero_system.goal_skills.detectors import (
            DrawerHandleDetector,
            MicrowaveDoorHandleDetector,
            PlateFrontDetector,
            StoveKnobDetector,
        )
        from libero_system.integration.adapters import (
            RouteBMicrowaveDoorDetector,
            RouteBPolicy,
        )
        from libero_system.goal_skills.schema import GoalExecutorStatus
        from libero_system.perception.adapters import RouteBScenePerceptionAdapter
        from libero_system.route_b.controller import RouteBController
        from libero_system.route_b.task_compiler import (
            TaskCompiler,
            TaskSpec,
            route_b_execution_issue,
        )

        policy_dict = vars(policy)
        controller = getattr(policy, "controller")
        controller_dict = vars(controller)
        contact_policy = getattr(policy, "contact_policy")
        contact_dict = vars(contact_policy)
        microwave_detector = getattr(contact_policy, "microwave_detector")
        microwave_dict = vars(microwave_detector)
        handle_detector = getattr(microwave_detector, "handle_detector")
        handle_dict = vars(handle_detector)
        perception = getattr(controller, "perception")
        perception_dict = vars(perception)
        spec = policy_dict["_spec"]
        segments = policy_dict["_segments"]
        expected_issue = route_b_execution_issue(spec)
        first_mode = segments[0].mode if segments else "done"
        expected_counts = {
            "route_b_policy": 1,
            "route_b_controller_clear": 1,
            "route_b_controller_activate": int(first_mode == "manipulation"),
            "goal_contact_policy_clear": 1,
            "goal_contact_policy_activate": int(first_mode == "contact"),
            "route_b_microwave_detector_reset": 1,
            "microwave_handle_detector_reset": 1,
            "route_b_perception_adapter_reset": 1,
        }
        if (
            reset_call_counts != expected_counts
            or type(policy) is not RouteBPolicy
        ):
            raise RuntimeError("production Route B reset call counts drifted")
        state.update(
            {
                "archived_segment_index_present": (
                    policy_dict["_archived_segment_index"] is not None
                ),
                "boundary_observation_sequence": policy_dict[
                    "_boundary_observation_sequence"
                ],
                "drawer_anchor_entries": len(policy_dict["_drawer_anchors"]),
                "episode_reset_pose_present": (
                    policy_dict["_episode_reset_pose"] is not None
                ),
                "episode_reset_pose_segment_present": (
                    policy_dict["_episode_reset_pose_segment_index"] is not None
                ),
                "grasp_event_archive_entries": len(
                    policy_dict["_grasp_event_archive"]
                ),
                "grasp_target_archive_entries": len(
                    policy_dict["_grasp_target_archive"]
                ),
                "grasp_verification_archive_entries": len(
                    policy_dict["_grasp_verification_archive"]
                ),
                "placement_target_archive_entries": len(
                    policy_dict["_placement_target_archive"]
                ),
                "selector_archive_entries": len(
                    policy_dict["_selector_diagnostic_archive"]
                ),
                "selector_diagnostic_cursor": policy_dict[
                    "_selector_diagnostic_cursor"
                ],
                "segment_index": policy_dict["_segment_index"],
                "segment_container_exact_tuple": type(segments) is tuple,
                "segment_entries_structurally_valid": all(
                    type(getattr(segment, "start", None)) is int
                    and getattr(segment, "start") >= 0
                    and type(getattr(segment, "steps", None)) is tuple
                    and getattr(segment, "mode", None)
                    in {"manipulation", "contact"}
                    for segment in segments
                ),
                "compiler_type_exact": (
                    type(getattr(controller, "compiler")) is TaskCompiler
                ),
                "spec_type_exact": type(spec) is TaskSpec,
                "spec_instruction_matches": (
                    type(spec) is TaskSpec
                    and spec.instruction == task.instruction
                ),
                "execution_issue_matches_spec": (
                    policy_dict["_execution_issue"] == expected_issue
                ),
                "segments_match_compiled_spec": (
                    segments
                    == (
                        ()
                        if expected_issue is not None
                        else RouteBPolicy._build_segments(spec)
                    )
                ),
                "controller_grasp_attempt_events": len(
                    getattr(controller, "grasp_attempt_events")
                ),
                "controller_grasp_target_attempts": len(
                    getattr(controller, "grasp_target_attempts")
                ),
                "controller_grasp_verifications": len(
                    getattr(controller, "grasp_verifications")
                ),
                "controller_observation_sequence": controller_dict[
                    "_observation_sequence"
                ],
                "controller_observation_timestamp_s": controller_dict[
                    "_observation_timestamp_s"
                ],
                "controller_pending_grasp_present": (
                    getattr(controller, "pending_grasp_engagement") is not None
                ),
                "controller_phase": controller_dict["_phase"],
                "controller_phase_ticks": controller_dict["_phase_ticks"],
                "controller_placement_target_attempts": len(
                    getattr(controller, "placement_target_attempts")
                ),
                "controller_skill_index": controller_dict["_skill_index"],
                "controller_status": _enum_string(
                    getattr(controller, "status"),
                    name="post-reset controller status",
                ),
                "controller_type_exact": type(controller) is RouteBController,
                "controller_spec_present": controller_dict["_spec"] is not None,
                "controller_spec_matches_active_segment": (
                    controller_dict["_spec"]
                    == (
                        TaskSpec(spec.instruction, segments[0].steps)
                        if first_mode == "manipulation"
                        else None
                    )
                ),
                "contact_policy_type_exact": (
                    type(contact_policy) is GoalContactPolicy
                ),
                "contact_plan_present": contact_dict["_plan"] is not None,
                "contact_status": _enum_string(
                    getattr(contact_policy, "status"),
                    name="post-reset contact status",
                ),
                "contact_phase": getattr(contact_policy, "phase"),
                "contact_phase_ticks": contact_dict["_phase_ticks"],
                "contact_step_index": contact_dict["_step_index"],
                "contact_detection_misses": contact_dict["_detection_misses"],
                "contact_message_empty": contact_dict["_message"] == "",
                "contact_target_present": contact_dict["_target"] is not None,
                "contact_push_present": contact_dict["_push"] is not None,
                "contact_drawer_anchor_present": (
                    getattr(contact_policy, "drawer_anchor") is not None
                ),
                "contact_base_rotation": contact_dict["_base_rotation"].tolist(),
                "contact_motion_position": contact_dict["_motion_position"].tolist(),
                "contact_motion_rotation": contact_dict["_motion_rotation"].tolist(),
                "contact_initial_feature_point": contact_dict[
                    "_initial_feature_point"
                ].tolist(),
                "contact_initial_feature_axis": contact_dict[
                    "_initial_feature_axis"
                ].tolist(),
                "contact_force_baseline": contact_dict["_force_baseline"].tolist(),
                "contact_manipulation_start_position": contact_dict[
                    "_manipulation_start_position"
                ].tolist(),
                "contact_previous_error_present": (
                    contact_dict["_previous_error"] is not None
                ),
                "contact_stall_ticks": contact_dict["_stall_ticks"],
                "delegate_activation_matches_first_segment": (
                    (
                        first_mode == "manipulation"
                        and getattr(controller, "status").value == "running"
                        and controller_dict["_phase"] == "acquire"
                        and getattr(contact_policy, "status")
                        is GoalExecutorStatus.IDLE
                        and contact_dict["_phase"] == "idle"
                    )
                    or (
                        first_mode == "contact"
                        and getattr(controller, "status").value == "idle"
                        and controller_dict["_phase"] == "idle"
                        and getattr(contact_policy, "status")
                        is GoalExecutorStatus.RUNNING
                        and contact_dict["_phase"] == "detect"
                    )
                    or (
                        first_mode == "done"
                        and getattr(controller, "status").value == "idle"
                        and controller_dict["_phase"] == "idle"
                        and getattr(contact_policy, "status")
                        is GoalExecutorStatus.IDLE
                        and contact_dict["_phase"] == "idle"
                    )
                ),
                "policy_reset_receipt_matches": (
                    policy_dict["_episode_reset_call_counts"]
                    == reset_call_counts
                ),
                "contact_stateless_detector_graph_exact": (
                    type(contact_policy.drawer_detector) is DrawerHandleDetector
                    and set(vars(contact_policy.drawer_detector)) == {"config"}
                    and type(contact_policy.knob_detector) is StoveKnobDetector
                    and set(vars(contact_policy.knob_detector)) == {"config"}
                    and type(contact_policy.plate_detector) is PlateFrontDetector
                    and set(vars(contact_policy.plate_detector))
                    == {"config", "knob_detector"}
                    and contact_policy.plate_detector.knob_detector
                    is contact_policy.knob_detector
                ),
                "microwave_detector_type_exact": (
                    type(microwave_detector) is RouteBMicrowaveDoorDetector
                ),
                "microwave_fixture_geometry_present": (
                    microwave_dict["_fixture_geometry"] is not None
                ),
                "microwave_fixture_surface_points_present": (
                    microwave_dict["_fixture_surface_points_world"] is not None
                ),
                "microwave_reference_ee_present": (
                    microwave_dict["_reference_ee_position_world"] is not None
                ),
                "microwave_selected_closed_slot_present": (
                    microwave_dict["_selected_closed_slot_world"] is not None
                ),
                "microwave_last_body_selection_present": (
                    microwave_dict["last_body_selection"] is not None
                ),
                "microwave_last_contact_detection_present": (
                    microwave_dict["last_contact_detection"] is not None
                ),
                "microwave_last_contact_track_present": (
                    microwave_dict["last_contact_track"] is not None
                ),
                "microwave_last_articulation_selection_present": (
                    microwave_dict["last_articulation_selection"] is not None
                ),
                "microwave_handle_detector_type_exact": (
                    type(handle_detector) is MicrowaveDoorHandleDetector
                ),
                "microwave_handle_state_keys_exact": (
                    set(handle_dict)
                    == {
                        "config",
                        "last_detection_trace",
                        "last_articulation_trace",
                    }
                ),
                "microwave_handle_last_detection_trace_present": (
                    handle_dict["last_detection_trace"] is not None
                ),
                "microwave_handle_last_articulation_trace_present": (
                    handle_dict["last_articulation_trace"] is not None
                ),
                "perception_type_exact": (
                    type(perception) is RouteBScenePerceptionAdapter
                ),
                "perception_dino_cache_entries": len(
                    perception_dict["_dino_cache"]
                ),
                "perception_dino_timestamp_entries": len(
                    perception_dict["_dino_cache_timestamp"]
                ),
                "perception_dino_provenance_entries": len(
                    perception_dict["_dino_provenance"]
                ),
                "perception_track_entries": len(perception_dict["_tracks"]),
                "perception_selector_diagnostics": len(
                    perception_dict["_selector_diagnostics"]
                ),
                "perception_observation_sequence": perception_dict[
                    "_observation_sequence"
                ],
                "perception_last_observation_present": (
                    perception_dict["_last_observation"] is not None
                ),
                "perception_last_observation_timestamp_s": perception_dict[
                    "_last_observation_timestamp_s"
                ],
                "perception_last_selector_reference_present": (
                    perception_dict["last_selector_reference"] is not None
                ),
            }
        )
    return _strict_json_detach(state)


_EMPTY_BASE_RESET_STATE: dict[str, Any] = {
    "drawer_contact_proof_present": False,
    "grasp_attempt_events": 0,
    "grasp_target_attempts": 0,
    "grasp_verifications": 0,
    "last_decision_present": False,
    "observation_sequence_token_present": False,
    "pending_grasp_engagement_present": False,
    "placement_target_attempts": 0,
    "selector_diagnostics": 0,
}

_EMPTY_PRODUCTION_RESET_EXTENSION: dict[str, Any] = {
    "archived_segment_index_present": False,
    "boundary_observation_sequence": 0,
    "drawer_anchor_entries": 0,
    "episode_reset_pose_present": False,
    "episode_reset_pose_segment_present": False,
    "grasp_event_archive_entries": 0,
    "grasp_target_archive_entries": 0,
    "grasp_verification_archive_entries": 0,
    "placement_target_archive_entries": 0,
    "selector_archive_entries": 0,
    "selector_diagnostic_cursor": 0,
    "segment_index": 0,
    "segment_container_exact_tuple": True,
    "segment_entries_structurally_valid": True,
    "compiler_type_exact": True,
    "spec_type_exact": True,
    "spec_instruction_matches": True,
    "execution_issue_matches_spec": True,
    "segments_match_compiled_spec": True,
    "controller_grasp_attempt_events": 0,
    "controller_grasp_target_attempts": 0,
    "controller_grasp_verifications": 0,
    "controller_observation_sequence": 0,
    "controller_observation_timestamp_s": 0.0,
    "controller_pending_grasp_present": False,
    "controller_phase": "idle",
    "controller_phase_ticks": 0,
    "controller_placement_target_attempts": 0,
    "controller_skill_index": 0,
    "controller_status": "idle",
    "controller_type_exact": True,
    "controller_spec_present": False,
    "controller_spec_matches_active_segment": True,
    "contact_policy_type_exact": True,
    "contact_plan_present": False,
    "contact_status": "idle",
    "contact_phase": "idle",
    "contact_phase_ticks": 0,
    "contact_step_index": 0,
    "contact_detection_misses": 0,
    "contact_message_empty": True,
    "contact_target_present": False,
    "contact_push_present": False,
    "contact_drawer_anchor_present": False,
    "contact_base_rotation": [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    "contact_motion_position": [0.0, 0.0, 0.0],
    "contact_motion_rotation": [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    "contact_initial_feature_point": [0.0, 0.0, 0.0],
    "contact_initial_feature_axis": [1.0, 0.0, 0.0],
    "contact_force_baseline": [0.0, 0.0, 0.0],
    "contact_manipulation_start_position": [0.0, 0.0, 0.0],
    "contact_previous_error_present": False,
    "contact_stall_ticks": 0,
    "delegate_activation_matches_first_segment": True,
    "policy_reset_receipt_matches": True,
    "contact_stateless_detector_graph_exact": True,
    "microwave_detector_type_exact": True,
    "microwave_fixture_geometry_present": False,
    "microwave_fixture_surface_points_present": False,
    "microwave_reference_ee_present": False,
    "microwave_selected_closed_slot_present": False,
    "microwave_last_body_selection_present": False,
    "microwave_last_contact_detection_present": False,
    "microwave_last_contact_track_present": False,
    "microwave_last_articulation_selection_present": False,
    "microwave_handle_detector_type_exact": True,
    "microwave_handle_state_keys_exact": True,
    "microwave_handle_last_detection_trace_present": False,
    "microwave_handle_last_articulation_trace_present": False,
    "perception_type_exact": True,
    "perception_dino_cache_entries": 0,
    "perception_dino_timestamp_entries": 0,
    "perception_dino_provenance_entries": 0,
    "perception_track_entries": 0,
    "perception_selector_diagnostics": 0,
    "perception_observation_sequence": 0,
    "perception_last_observation_present": False,
    "perception_last_observation_timestamp_s": 0.0,
    "perception_last_selector_reference_present": False,
}


class RouteBPolicyRuntime:
    """One serial worker runtime; any episode-side exception poisons it."""

    __slots__ = (
        "_policy",
        "_assembly",
        "_episodes_reset",
        "_used_child_nonces",
        "_used_episode_nonces",
        "_used_opaque_episode_ids",
        "_state_lock",
        "_active",
        "_poisoned",
        "_closed",
        "_policy_close_attempted",
    )

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError(
            "construct RouteBPolicyRuntime with from_perception_bundle(); "
            "tests may use the private _from_test_policy() entry"
        )

    @classmethod
    def _create(
        cls,
        instance: "RouteBPolicyRuntime",
        policy: Any,
        evidence: _AssemblyEvidence,
    ) -> "RouteBPolicyRuntime":
        if cls is not RouteBPolicyRuntime or type(instance) is not RouteBPolicyRuntime:
            raise TypeError("Route B runtime construction requires the exact base type")
        instance._policy = policy
        instance._assembly = evidence
        instance._episodes_reset = 0
        instance._used_child_nonces = {evidence.assembly_nonce}
        instance._used_episode_nonces: set[str] = set()
        instance._used_opaque_episode_ids: set[str] = set()
        instance._state_lock = threading.Lock()
        instance._active = False
        instance._poisoned = False
        instance._closed = False
        instance._policy_close_attempted = False
        return instance

    @classmethod
    def from_perception_bundle(cls, bundle: Any) -> "RouteBPolicyRuntime":
        """Build and type-lock the exact production component graph once."""

        from libero_system.integration.adapters import RouteBPolicy
        from libero_system.integration.components import PerceptionBundle
        from libero_system.perception.adapters import RouteBScenePerceptionAdapter
        from libero_system.route_b.controller import RouteBController

        if cls is not RouteBPolicyRuntime:
            raise TypeError("production Route B runtime cannot be subclassed")
        if (
            type(bundle) is not PerceptionBundle
            or set(vars(bundle))
            != {
                "backend",
                "box_detector",
                "box_extractor",
                "route_c_episode_cache",
            }
        ):
            raise TypeError("production Route B requires the exact PerceptionBundle")
        assembly_nonce = _new_child_nonce()
        counts = {
            "perception_bundle": 1,
            "route_b_perception_adapter": 0,
            "route_b_controller": 0,
            "route_b_policy": 0,
            "route_b_runtime": 0,
        }
        constructed: list[object] = []
        try:
            # The exact class descriptor prevents a mutable bundle instance
            # from substituting a prebuilt/dirty adapter while retaining the
            # nominal PerceptionBundle type.
            adapter = PerceptionBundle.for_route_b(bundle)
            counts["route_b_perception_adapter"] += 1
            if type(adapter) is not RouteBScenePerceptionAdapter:
                raise TypeError("production Route B perception adapter type drifted")
            constructed.append(adapter)

            controller = RouteBController(adapter)
            counts["route_b_controller"] += 1
            if type(controller) is not RouteBController:
                raise TypeError("production Route B controller type drifted")
            constructed.append(controller)

            policy = RouteBPolicy(controller)
            counts["route_b_policy"] += 1
            if type(policy) is not RouteBPolicy:
                raise TypeError("production Route B policy type drifted")
            constructed.append(policy)

            runtime_instance = object.__new__(RouteBPolicyRuntime)
            counts["route_b_runtime"] += 1
            components = {
                "perception_bundle": bundle,
                "route_b_perception_adapter": adapter,
                "route_b_controller": controller,
                "route_b_policy": policy,
                "route_b_runtime": runtime_instance,
            }
            evidence = _AssemblyEvidence(
                token=_PRODUCTION_ASSEMBLY_TOKEN,
                production_assembly=True,
                assembly_nonce=assembly_nonce,
                component_type_manifest={
                    name: _type_name(component)
                    for name, component in components.items()
                },
                component_instance_counts=dict(counts),
                component_construction_counts={
                    "route_b_perception_adapter": counts[
                        "route_b_perception_adapter"
                    ],
                    "route_b_controller": counts["route_b_controller"],
                    "route_b_policy": counts["route_b_policy"],
                    "route_b_runtime": counts["route_b_runtime"],
                },
                component_identity_sha256=_component_identity(
                    assembly_nonce,
                    components,
                ),
            )
            return cls._create(runtime_instance, policy, evidence)
        except BaseException:
            # Factory failures have no runtime instance to poison.  Close the
            # deepest constructed owner once, best effort, without replacing
            # the construction exception.
            target = constructed[-1] if constructed else None
            try:
                close = getattr(target, "close", None)
                if callable(close):
                    close()
            except BaseException:
                pass
            raise

    @classmethod
    def _from_test_policy(cls, policy: Any) -> "RouteBPolicyRuntime":
        """Private fake-policy entry that can never claim production assembly."""

        if cls is not RouteBPolicyRuntime:
            raise TypeError("test Route B runtime cannot be subclassed")
        if not callable(getattr(policy, "reset", None)) or not callable(
            getattr(policy, "act", None)
        ):
            raise TypeError("test Route B runtime requires reset/act capabilities")
        assembly_nonce = _new_child_nonce()
        runtime_instance = object.__new__(RouteBPolicyRuntime)
        components = {
            "test_policy": policy,
            "route_b_runtime": runtime_instance,
        }
        evidence = _AssemblyEvidence(
            token=_TEST_ASSEMBLY_TOKEN,
            production_assembly=False,
            assembly_nonce=assembly_nonce,
            component_type_manifest={
                name: _type_name(component)
                for name, component in components.items()
            },
            component_instance_counts={"test_policy": 1, "route_b_runtime": 1},
            component_construction_counts={"route_b_runtime": 1},
            component_identity_sha256=_component_identity(
                assembly_nonce,
                components,
            ),
        )
        return cls._create(runtime_instance, policy, evidence)

    @property
    def poisoned(self) -> bool:
        with self._state_lock:
            return self._poisoned

    @property
    def worker_runtime_receipt(self) -> dict[str, Any]:
        with self._state_lock:
            active = self._active
            poisoned = self._poisoned
            closed = self._closed
            episodes_reset = self._episodes_reset
        evidence = self._assembly
        return _strict_json_detach(
            {
                "schema": ROUTE_B_WORKER_RUNTIME_SCHEMA,
                "production_assembly": evidence.production_assembly,
                "assembly_nonce": evidence.assembly_nonce,
                "assembly_count": evidence.component_instance_counts[
                    "route_b_runtime"
                ],
                "component_type_manifest": evidence.component_type_manifest,
                "component_instance_counts": evidence.component_instance_counts,
                "component_construction_counts": (
                    evidence.component_construction_counts
                ),
                "component_identity_sha256": evidence.component_identity_sha256,
                "episodes_reset": episodes_reset,
                "active": active,
                "poisoned": poisoned,
                "closed": closed,
            }
        )

    def _enter_episode(self) -> None:
        active_violation = False
        with self._state_lock:
            if self._active:
                active_violation = True
                self._poisoned = True
                self._closed = True
            elif self._poisoned:
                raise RuntimeError("Route B runtime is poisoned")
            elif self._closed:
                raise RuntimeError("Route B runtime is closed")
            else:
                self._active = True
        if active_violation:
            # Re-entrancy/concurrency makes episode counters and callbacks
            # ambiguous.  Poison under the same lock that observed ``active``
            # so the in-flight completion cannot win a TOCTOU race.
            self._best_effort_policy_close()
            raise RuntimeError(
                "Route B runtime concurrent/recursive active episode violation"
            )

    def _leave_episode(self) -> None:
        with self._state_lock:
            self._active = False

    def _best_effort_policy_close(self) -> None:
        with self._state_lock:
            if self._policy_close_attempted:
                return
            self._policy_close_attempted = True
        try:
            if self._assembly.production_assembly:
                from libero_system.integration.adapters import RouteBPolicy

                RouteBPolicy.close(self._policy)
            else:
                close = getattr(self._policy, "close")
                if callable(close):
                    close()
        except BaseException:
            pass

    def _poison(self) -> None:
        with self._state_lock:
            self._poisoned = True
            self._closed = True
        self._best_effort_policy_close()

    def _require_healthy_active_episode(self) -> None:
        with self._state_lock:
            if self._poisoned or self._closed:
                raise RuntimeError("Route B runtime was poisoned during its episode")
            if not self._active:
                raise RuntimeError("Route B runtime lost its active episode guard")

    def _commit_episode_completion(self) -> None:
        """Atomically close the active interval or reject a racing violation."""

        with self._state_lock:
            if self._poisoned or self._closed:
                raise RuntimeError("Route B runtime was poisoned during its episode")
            if not self._active:
                raise RuntimeError("Route B runtime lost its active episode guard")
            self._active = False

    def _reserve_episode_tokens(
        self,
        *,
        task: PolicyTask,
        episode_nonce: str,
    ) -> None:
        opaque_entropy = task.episode_id.removeprefix("episode-v2-")
        with self._state_lock:
            if (
                episode_nonce in self._used_episode_nonces
                or episode_nonce in self._used_child_nonces
                or episode_nonce == opaque_entropy
                or task.episode_id in self._used_opaque_episode_ids
                or opaque_entropy in self._used_child_nonces
                or opaque_entropy in self._used_episode_nonces
            ):
                raise RuntimeError("Route B episode/opaque nonce is replayed or aliased")
            self._used_episode_nonces.add(episode_nonce)
            self._used_opaque_episode_ids.add(task.episode_id)
            # Child nonces must also remain distinct from both parent-provided
            # entropy fields for the full worker lifetime.
            self._used_child_nonces.add(episode_nonce)
            self._used_child_nonces.add(opaque_entropy)

    def _fresh_reset_nonce(self) -> str:
        # Exactly one child entropy call is made per candidate.  Retry is
        # bounded and every invalid/colliding construction remains fail-closed.
        for _ in range(8):
            candidate = _new_child_nonce()
            with self._state_lock:
                if candidate not in self._used_child_nonces:
                    self._used_child_nonces.add(candidate)
                    return candidate
        raise RuntimeError("could not allocate a unique Route B reset nonce")

    @staticmethod
    def _validate_start(
        *,
        task: object,
        episode_nonce: object,
        initial_observation: object,
        execute_action: object,
        step_budget: object,
        allow_short_horizon: bool,
    ) -> PolicyTask:
        if type(task) is not PolicyTask:
            raise TypeError("Route B START task must be the exact whitelist DTO")
        instruction = task.instruction
        opaque_episode_id = task.episode_id
        if (
            type(instruction) is not str
            or not instruction.strip()
            or len(instruction) > MAX_ROUTE_B_INSTRUCTION_CHARS
            or any(
                unicodedata.category(character).startswith("C")
                for character in instruction
            )
        ):
            raise ValueError("Route B instruction must be non-empty and bounded")
        if (
            type(opaque_episode_id) is not str
            or _OPAQUE_EPISODE_ID.fullmatch(opaque_episode_id) is None
        ):
            raise ValueError("Route B opaque id must be episode-v2 plus 64 lower hex")
        if not _is_nonce(episode_nonce):
            raise ValueError("Route B episode_nonce must be exact 64 lower hex")
        if type(initial_observation) is not RobotObservation:
            raise TypeError("Route B initial observation must be the whitelist DTO")
        if not callable(execute_action):
            raise TypeError("Route B action capability must be callable")
        if type(step_budget) is not int:
            raise ValueError("Route B step_budget must be a native integer")
        if allow_short_horizon:
            if not 1 <= step_budget <= MAX_ROUTE_B_POLICY_STEPS:
                raise ValueError("test Route B step_budget is outside its bound")
        elif step_budget != MAX_ROUTE_B_POLICY_STEPS:
            raise ValueError("formal Route B step_budget must be exactly 520")
        return PolicyTask(instruction, opaque_episode_id)

    def run_episode(
        self,
        *,
        task: PolicyTask,
        episode_nonce: str,
        initial_observation: RobotObservation,
        execute_action: ActionExecutor,
        step_budget: int,
    ) -> RouteBPolicyExecution:
        """Run one formal, exact-520 episode."""

        self._enter_episode()
        try:
            detached_task = self._validate_start(
                task=task,
                episode_nonce=episode_nonce,
                initial_observation=initial_observation,
                execute_action=execute_action,
                step_budget=step_budget,
                allow_short_horizon=False,
            )
            detached_initial_observation = _detach_observation(initial_observation)
            self._require_healthy_active_episode()
        except BaseException:
            self._leave_episode()
            raise
        return self._run_validated_episode(
            task=detached_task,
            episode_nonce=episode_nonce,
            initial_observation=detached_initial_observation,
            execute_action=execute_action,
            step_budget=step_budget,
        )

    def _run_test_episode(
        self,
        *,
        task: PolicyTask,
        episode_nonce: str,
        initial_observation: RobotObservation,
        execute_action: ActionExecutor,
        step_budget: int,
    ) -> RouteBPolicyExecution:
        """Private short-horizon hook, unavailable to production assemblies."""

        if self._assembly.production_assembly:
            raise RuntimeError("production Route B cannot use the short test horizon")
        self._enter_episode()
        try:
            detached_task = self._validate_start(
                task=task,
                episode_nonce=episode_nonce,
                initial_observation=initial_observation,
                execute_action=execute_action,
                step_budget=step_budget,
                allow_short_horizon=True,
            )
            detached_initial_observation = _detach_observation(initial_observation)
            self._require_healthy_active_episode()
        except BaseException:
            self._leave_episode()
            raise
        return self._run_validated_episode(
            task=detached_task,
            episode_nonce=episode_nonce,
            initial_observation=detached_initial_observation,
            execute_action=execute_action,
            step_budget=step_budget,
        )

    def _run_validated_episode(
        self,
        *,
        task: PolicyTask,
        episode_nonce: str,
        initial_observation: RobotObservation,
        execute_action: ActionExecutor,
        step_budget: int,
    ) -> RouteBPolicyExecution:
        try:
            self._require_healthy_active_episode()
            self._reserve_episode_tokens(task=task, episode_nonce=episode_nonce)
            if self._assembly.production_assembly:
                from libero_system.integration.adapters import RouteBPolicy

                RouteBPolicy.reset(self._policy, task)
                reset_call_counts = RouteBPolicy.episode_reset_call_counts.__get__(
                    self._policy,
                    RouteBPolicy,
                )
            else:
                self._policy.reset(task)
                reset_call_counts = {
                    key: int(key == "route_b_policy")
                    for key in _RESET_CALL_KEYS
                }
            initial_state = _post_reset_state(
                self._policy,
                production=self._assembly.production_assembly,
                task=task,
                reset_call_counts=reset_call_counts,
            )
            expected_state = dict(_EMPTY_BASE_RESET_STATE)
            if self._assembly.production_assembly:
                expected_state.update(_EMPTY_PRODUCTION_RESET_EXTENSION)
                segments = vars(self._policy)["_segments"]
                first_mode = segments[0].mode if segments else "done"
                if first_mode == "manipulation":
                    expected_state.update(
                        {
                            "controller_phase": "acquire",
                            "controller_status": "running",
                            "controller_spec_present": True,
                        }
                    )
                elif first_mode == "contact":
                    expected_state.update(
                        {
                            "contact_plan_present": True,
                            "contact_status": "running",
                            "contact_phase": "detect",
                        }
                    )
            if initial_state != expected_state:
                raise RuntimeError(
                    "Route B policy reset left stale episode-local state"
                )

            reset_nonce = self._fresh_reset_nonce()
            with self._state_lock:
                self._episodes_reset += 1
                reset_count = self._episodes_reset

            result = self._execute_loop(
                task=task,
                episode_nonce=episode_nonce,
                reset_nonce=reset_nonce,
                reset_count=reset_count,
                reset_call_counts=reset_call_counts,
                initial_state=initial_state,
                initial_observation=initial_observation,
                execute_action=execute_action,
                step_budget=step_budget,
            )
            self._commit_episode_completion()
            return result
        except BaseException:
            self._poison()
            raise
        finally:
            self._leave_episode()

    def _execute_loop(
        self,
        *,
        task: PolicyTask,
        episode_nonce: str,
        reset_nonce: str,
        reset_count: int,
        reset_call_counts: dict[str, int],
        initial_state: dict[str, Any],
        initial_observation: RobotObservation,
        execute_action: ActionExecutor,
        step_budget: int,
    ) -> RouteBPolicyExecution:
        observation = initial_observation
        action_count = 0
        observation_count = 1
        act_count = 0
        phase_transitions: list[dict[str, Any]] = []
        last_phase: str | None = None
        last_state: tuple[str, int, str, str] | None = None
        terminal_status: str | None = None
        terminal_failure: str | None = None

        while action_count < step_budget:
            if self._assembly.production_assembly:
                from libero_system.integration.adapters import RouteBPolicy

                decision = RouteBPolicy.act(self._policy, observation)
            else:
                decision = self._policy.act(observation)
            self._require_healthy_active_episode()
            if type(decision) is not PolicyDecision:
                raise RuntimeError("Route B policy returned a non-whitelist decision")
            action_value = decision.action
            request_stop = decision.request_stop
            detached_action = _detach_action(action_value)
            if type(request_stop) is not bool:
                raise RuntimeError("Route B request_stop must be a native boolean")
            state = _decision_state(self._policy)
            sequence_token = getattr(self._policy, "observation_sequence_token")
            if type(sequence_token) is not int or sequence_token != act_count:
                raise RuntimeError("Route B controller observation sequence drifted")
            act_count += 1
            last_state = state
            status, skill_index, phase, message = state
            if phase != last_phase:
                phase_transitions.append(
                    {
                        "step": action_count,
                        "skill_index": skill_index,
                        "phase": phase,
                    }
                )
                last_phase = phase

            if request_stop:
                if status not in {"succeeded", "failed"}:
                    raise RuntimeError(
                        "Route B requested stop without a terminal policy status"
                    )
                terminal_status = status
                terminal_failure = message if status == "failed" else None
                break
            if status != "running":
                raise RuntimeError(
                    "Route B terminal/non-running status omitted request_stop"
                )

            next_observation = execute_action(detached_action)
            self._require_healthy_active_episode()
            next_observation = _detach_observation(next_observation)
            action_count += 1
            observation_count += 1
            observation = next_observation

        if last_state is None:
            raise RuntimeError("Route B loop produced no policy decision")
        final_controller_status, final_skill_index, final_phase, final_message = (
            last_state
        )
        if terminal_status is None:
            if not (
                action_count == act_count == step_budget
                and final_controller_status == "running"
            ):
                raise RuntimeError("Route B timeout lifecycle counts are inconsistent")
            terminal_status = "timeout"
            terminal_failure = (
                f"policy remained running at the {step_budget}-step limit"
            )
        elif not (
            act_count == action_count + 1
            and final_controller_status == terminal_status
        ):
            raise RuntimeError("Route B terminal act/action counts are inconsistent")
        if observation_count != action_count + 1:
            raise RuntimeError("Route B observation/action lifecycle is inconsistent")

        controller = getattr(self._policy, "controller")
        grasp_mode_value = getattr(controller, "grasp_mode")
        grasp_mode = _enum_string(grasp_mode_value, name="grasp mode")
        active_mode = getattr(self._policy, "active_mode")
        if type(active_mode) is not str or not active_mode or len(active_mode) > 256:
            raise RuntimeError("Route B active dispatcher mode is invalid")

        events = _events(self._policy)
        pending = _pending(self._policy)
        trace: dict[str, Any] = {
            "status": terminal_status,
            "controller_status": final_controller_status,
            "skill_index": final_skill_index,
            "phase": final_phase,
            "message": final_message,
            "phase_transitions": phase_transitions,
            "active_mode": active_mode,
            "grasp_mode": grasp_mode,
            "grasp_target_attempts": _rows(
                self._policy, "grasp_target_attempts"
            ),
            "grasp_verifications": _rows(
                self._policy, "grasp_verifications"
            ),
            "placement_target_attempts": _rows(
                self._policy, "placement_target_attempts"
            ),
            "selector_diagnostics": _rows(
                self._policy, "selector_diagnostics"
            ),
        }
        drawer_proof = getattr(self._policy, "drawer_contact_proof")
        if drawer_proof is not None:
            if type(drawer_proof) is not dict:
                raise RuntimeError("Route B drawer contact proof is not an exact dict")
            trace["drawer_contact_proof"] = drawer_proof

        instruction_sha256 = _text_sha256(
            task.instruction,
            domain=b"libero-route-b-instruction.v2\0",
        )
        opaque_id_sha256 = _text_sha256(
            task.episode_id,
            domain=b"libero-route-b-opaque-episode-id.v2\0",
        )
        episode_binding = {
            "assembly_nonce": self._assembly.assembly_nonce,
            "episode_nonce": episode_nonce,
            "reset_nonce": reset_nonce,
            "reset_count": reset_count,
            "reset_call_counts": reset_call_counts,
            "instruction_sha256": instruction_sha256,
            "opaque_episode_id_sha256": opaque_id_sha256,
        }
        episode_receipt: dict[str, Any] = {
            "schema": ROUTE_B_EPISODE_RUNTIME_SCHEMA,
            "production_assembly": self._assembly.production_assembly,
            "assembly_nonce": self._assembly.assembly_nonce,
            "assembly_count": self._assembly.component_instance_counts[
                "route_b_runtime"
            ],
            "component_type_manifest": self._assembly.component_type_manifest,
            "component_instance_counts": self._assembly.component_instance_counts,
            "component_construction_counts": (
                self._assembly.component_construction_counts
            ),
            "component_identity_sha256": (
                self._assembly.component_identity_sha256
            ),
            "episode_nonce": episode_nonce,
            "reset_nonce": reset_nonce,
            "reset_count": reset_count,
            "reset_call_counts": reset_call_counts,
            "instruction_sha256": instruction_sha256,
            "opaque_episode_id_sha256": opaque_id_sha256,
            "episode_binding_sha256": _json_sha256(
                episode_binding,
                domain=b"libero-route-b-episode-binding.v2\0",
            ),
            "initial_state": initial_state,
            "initial_state_sha256": _json_sha256(
                initial_state,
                domain=b"libero-route-b-post-reset-state.v2\0",
            ),
            "act_count": act_count,
            "final_action_count": action_count,
            "observation_count": observation_count,
            "status": terminal_status,
        }
        trace["episode_runtime"] = episode_receipt
        result = RouteBPolicyExecution(
            success=terminal_status == "succeeded",
            failure=terminal_failure,
            steps_executed=action_count,
            trace=trace,
            grasp_attempt_events=events,
            pending_grasp_engagement=pending,
        )
        self._require_healthy_active_episode()
        return result

    def close(self) -> None:
        active_violation = False
        with self._state_lock:
            if self._active:
                active_violation = True
                self._poisoned = True
                self._closed = True
            elif self._closed:
                return
            else:
                self._closed = True
        if active_violation:
            self._best_effort_policy_close()
            raise RuntimeError(
                "cannot close Route B during an active episode; runtime poisoned"
            )
        try:
            with self._state_lock:
                self._policy_close_attempted = True
            if self._assembly.production_assembly:
                from libero_system.integration.adapters import RouteBPolicy

                RouteBPolicy.close(self._policy)
            else:
                close = getattr(self._policy, "close")
                if callable(close):
                    close()
        except BaseException:
            with self._state_lock:
                self._poisoned = True
            raise


def build_route_b_policy_runtime(bundle: Any) -> RouteBPolicyRuntime:
    """Production-only worker factory."""

    return RouteBPolicyRuntime.from_perception_bundle(bundle)


__all__ = [
    "MAX_ROUTE_B_INSTRUCTION_CHARS",
    "MAX_ROUTE_B_POLICY_STEPS",
    "ROUTE_B_EPISODE_RUNTIME_SCHEMA",
    "ROUTE_B_WORKER_RUNTIME_SCHEMA",
    "RouteBPolicyExecution",
    "RouteBPolicyRuntime",
    "build_route_b_policy_runtime",
]
