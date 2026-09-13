"""Runtime audit of the sensor-only policy boundary.

This module observes values *at* the Route B / Route C handoff.  It never asks
the environment for simulator state or evaluator feedback.  Forbidden inputs
are represented only by field-name counters, so detecting a violation does not
require forwarding the forbidden value to a policy.

The builder is append-only: reset, observation, sequence, forbidden-name, and
evaluator-count events can be added, but no event can be replaced or removed.
The compact episode report is composed entirely of JSON-native values and has
an independent strict validator for result/verifier use.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import math
import re
from typing import Any, Literal

import numpy as np

from libero_system.common.observation import (
    CameraCalibration,
    CameraFrame,
    Proprioception,
    RobotObservation,
)
from libero_system.common.policy import PolicyTask


POLICY_BOUNDARY_AUDIT_SCHEMA = "libero-policy-boundary-runtime-audit.v3"
CONTROLLER_SEQUENCE_SOURCE = "controller_owned"
EVALUATOR_STEP_COUNT_SOURCE = "LiberoEnvAdapter.episode_steps"

PolicyRoute = Literal["b", "c"]
DeliveryDestination = Literal["policy", "controller"]

_OPAQUE_EPISODE_ID = re.compile(
    r"episode-(?:[0-9a-f]{20}|v2-[0-9a-f]{64})"
)
_CAMERA_NAMES = ("agentview", "wrist")

FIELD_CONTRACTS: dict[str, tuple[str, ...]] = {
    "PolicyTask": ("instruction", "episode_id"),
    "RobotObservation": ("cameras", "proprio"),
    "CameraFrame": ("rgb", "depth_m", "calibration"),
    "CameraCalibration": (
        "name",
        "width",
        "height",
        "intrinsic",
        "T_world_camera",
        "observation_v_flipped",
    ),
    "Proprioception": (
        "T_world_ee",
        "ee_quat_xyzw",
        "joint_position",
        "joint_velocity",
        "gripper_qpos",
        "gripper_qvel",
        "gripper_width_m",
        "ee_force_sensor",
        "ee_torque_sensor",
    ),
}

_FIELD_TYPES: dict[str, type[Any]] = {
    "PolicyTask": PolicyTask,
    "RobotObservation": RobotObservation,
    "CameraFrame": CameraFrame,
    "CameraCalibration": CameraCalibration,
    "Proprioception": Proprioception,
}

# Canonical, explicit counters.  These names describe values that would have
# crossed the boundary; the audit API accepts only a name and count, never the
# value itself.
FORBIDDEN_DELIVERY_FIELDS: tuple[str, ...] = (
    "suite_id",
    "task_id",
    "init_state_id",
    "bddl",
    "simulator_object_pose",
    "simulator_segmentation",
    "simulator_contact_state",
    "step_index",
    "sim_time_s",
    "monotonic_time_s",
    "reward",
    "success",
    "done",
    "terminated",
    "truncated",
    "evaluator_feedback",
)

_FORBIDDEN_ALIASES: dict[str, str] = {
    "suite": "suite_id",
    "suite_name": "suite_id",
    "suite_id": "suite_id",
    "task_id": "task_id",
    "task_index": "task_id",
    "init": "init_state_id",
    "init_id": "init_state_id",
    "init_state": "init_state_id",
    "init_state_id": "init_state_id",
    "init_state_index": "init_state_id",
    "bddl": "bddl",
    "bddl_file": "bddl",
    "bddl_path": "bddl",
    "object_pose": "simulator_object_pose",
    "object_poses": "simulator_object_pose",
    "sim_object_pose": "simulator_object_pose",
    "simulator_object_pose": "simulator_object_pose",
    "segmentation": "simulator_segmentation",
    "segmentation_id": "simulator_segmentation",
    "segmentation_ids": "simulator_segmentation",
    "simulator_segmentation": "simulator_segmentation",
    "contact": "simulator_contact_state",
    "contact_state": "simulator_contact_state",
    "sim_contact": "simulator_contact_state",
    "simulator_contact_state": "simulator_contact_state",
    "step": "step_index",
    "step_index": "step_index",
    "sim_time": "sim_time_s",
    "sim_time_s": "sim_time_s",
    "monotonic_time": "monotonic_time_s",
    "monotonic_time_s": "monotonic_time_s",
    "reward": "reward",
    "success": "success",
    "evaluator_success": "success",
    "done": "done",
    "raw_done": "done",
    "terminated": "terminated",
    "truncated": "truncated",
    "evaluation": "evaluator_feedback",
    "evaluator_feedback": "evaluator_feedback",
}


class PolicyBoundaryAuditValidationError(ValueError):
    """Raised when an episode boundary report fails its strict contract."""


@dataclass(frozen=True, slots=True)
class _TaskEvent:
    type_exact: bool
    field_names: tuple[str, ...]
    instruction_valid: bool
    opaque_episode_id_valid: bool
    opaque_episode_id: str | None
    forbidden_names: tuple[str, ...]
    receiver: str
    boundary_adapter_applied: bool


@dataclass(frozen=True, slots=True)
class _ObservationEvent:
    type_exact: bool
    field_names_exact: bool
    camera_names: tuple[str, ...]
    dual_camera_exact: bool
    camera_field_names_exact: bool
    calibration_field_names_exact: bool
    proprio_field_names_exact: bool
    shape_checks_pass: bool
    dtype_checks_pass: bool
    finiteness_checks_pass: bool
    calibration_checks_pass: bool
    proprio_checks_pass: bool
    invalid_paths: tuple[str, ...]
    forbidden_names: tuple[str, ...]
    destination: DeliveryDestination


@dataclass(frozen=True, slots=True)
class _SequenceEvent:
    act_index: int | None
    value: int | None
    source: str | None
    exposed_to_policy: bool | None


@dataclass(frozen=True, slots=True)
class _EvaluatorStepEvent:
    value: int | None
    source: str | None
    exposed_to_policy: bool | None


@dataclass(frozen=True, slots=True)
class _ForbiddenEvent:
    field_name: str
    destination: DeliveryDestination
    count: int


def _field_names(value: object) -> tuple[str, ...]:
    if not is_dataclass(value):
        return ()
    try:
        return tuple(field.name for field in fields(value))
    except (TypeError, ValueError):
        return ()


def _runtime_field_contracts() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, expected in FIELD_CONTRACTS.items():
        reflected = tuple(field.name for field in fields(_FIELD_TYPES[name]))
        result[name] = {
            "expected_fields": list(expected),
            "reflected_fields": list(reflected),
            "exact": reflected == expected,
        }
    return result


def _forbidden_names(names: tuple[str, ...] | list[str] | set[str]) -> tuple[str, ...]:
    canonical = {
        mapped
        for name in names
        if (mapped := _FORBIDDEN_ALIASES.get(str(name).lower())) is not None
    }
    return tuple(sorted(canonical))


def _array_checks(
    value: object,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype[Any] | type[Any],
) -> tuple[bool, bool, bool]:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False, False, False
    shape_valid = tuple(array.shape) == shape
    dtype_valid = array.dtype == np.dtype(dtype)
    if array.dtype.kind in "biufc":
        try:
            finite = bool(np.isfinite(array).all())
        except TypeError:
            finite = False
    else:
        finite = False
    return shape_valid, dtype_valid, finite


def _valid_nonnegative_integer(value: object) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        return None
    return value


def _valid_native_bool(value: object) -> bool | None:
    return value if type(value) is bool else None


def _task_event(
    task: object,
    *,
    receiver: str,
    boundary_adapter_applied: bool,
) -> _TaskEvent:
    names = _field_names(task)
    instruction = getattr(task, "instruction", None)
    episode_id = getattr(task, "episode_id", None)
    opaque_valid = bool(
        isinstance(episode_id, str) and _OPAQUE_EPISODE_ID.fullmatch(episode_id)
    )
    extras = tuple(name for name in names if name not in FIELD_CONTRACTS["PolicyTask"])
    return _TaskEvent(
        type_exact=type(task) is PolicyTask,
        field_names=names,
        instruction_valid=bool(isinstance(instruction, str) and instruction.strip()),
        opaque_episode_id_valid=opaque_valid,
        opaque_episode_id=episode_id if opaque_valid else None,
        forbidden_names=_forbidden_names(extras),
        receiver=receiver,
        boundary_adapter_applied=boundary_adapter_applied,
    )


def _observation_event(
    observation: object,
    *,
    expected_height: int,
    expected_width: int,
    destination: DeliveryDestination,
) -> _ObservationEvent:
    invalid: list[str] = []
    outer_names = _field_names(observation)
    outer_exact = outer_names == FIELD_CONTRACTS["RobotObservation"]
    forbidden_candidates = list(
        name for name in outer_names if name not in FIELD_CONTRACTS["RobotObservation"]
    )

    cameras = getattr(observation, "cameras", None)
    if isinstance(cameras, Mapping):
        camera_names = tuple(sorted(str(name) for name in cameras))
        camera_mapping: Mapping[object, object] = cameras
    else:
        camera_names = ()
        camera_mapping = {}
        invalid.append("RobotObservation.cameras.mapping")
    forbidden_candidates.extend(camera_names)
    dual_exact = camera_names == _CAMERA_NAMES

    camera_fields_exact = True
    calibration_fields_exact = True
    shape_pass = True
    dtype_pass = True
    finite_pass = True
    calibration_pass = True
    for camera_name in _CAMERA_NAMES:
        frame = camera_mapping.get(camera_name)
        if frame is None:
            camera_fields_exact = False
            calibration_fields_exact = False
            shape_pass = False
            dtype_pass = False
            finite_pass = False
            calibration_pass = False
            invalid.append(f"cameras.{camera_name}.missing")
            continue
        frame_names = _field_names(frame)
        frame_exact = type(frame) is CameraFrame and (
            frame_names == FIELD_CONTRACTS["CameraFrame"]
        )
        camera_fields_exact = camera_fields_exact and frame_exact
        forbidden_candidates.extend(
            name for name in frame_names if name not in FIELD_CONTRACTS["CameraFrame"]
        )
        calibration = getattr(frame, "calibration", None)
        calibration_names = _field_names(calibration)
        calibration_exact = type(calibration) is CameraCalibration and (
            calibration_names == FIELD_CONTRACTS["CameraCalibration"]
        )
        calibration_fields_exact = calibration_fields_exact and calibration_exact
        forbidden_candidates.extend(
            name
            for name in calibration_names
            if name not in FIELD_CONTRACTS["CameraCalibration"]
        )

        rgb_shape, rgb_dtype, rgb_finite = _array_checks(
            getattr(frame, "rgb", None),
            shape=(expected_height, expected_width, 3),
            dtype=np.uint8,
        )
        depth_shape, depth_dtype, depth_finite = _array_checks(
            getattr(frame, "depth_m", None),
            shape=(expected_height, expected_width),
            dtype=np.float32,
        )
        if not rgb_shape:
            invalid.append(f"cameras.{camera_name}.rgb.shape")
        if not depth_shape:
            invalid.append(f"cameras.{camera_name}.depth_m.shape")
        if not rgb_dtype:
            invalid.append(f"cameras.{camera_name}.rgb.dtype")
        if not depth_dtype:
            invalid.append(f"cameras.{camera_name}.depth_m.dtype")
        if not rgb_finite:
            invalid.append(f"cameras.{camera_name}.rgb.finite")
        if not depth_finite:
            invalid.append(f"cameras.{camera_name}.depth_m.finite")
        shape_pass = shape_pass and rgb_shape and depth_shape
        dtype_pass = dtype_pass and rgb_dtype and depth_dtype
        finite_pass = finite_pass and rgb_finite and depth_finite

        intrinsic_shape, intrinsic_dtype, intrinsic_finite = _array_checks(
            getattr(calibration, "intrinsic", None),
            shape=(3, 3),
            dtype=np.float64,
        )
        transform_shape, transform_dtype, transform_finite = _array_checks(
            getattr(calibration, "T_world_camera", None),
            shape=(4, 4),
            dtype=np.float64,
        )
        intrinsic = np.asarray(getattr(calibration, "intrinsic", ()))
        transform = np.asarray(getattr(calibration, "T_world_camera", ()))
        intrinsic_physical = bool(
            intrinsic_shape
            and intrinsic_finite
            and float(intrinsic[0, 0]) > 0.0
            and float(intrinsic[1, 1]) > 0.0
        )
        transform_physical = bool(
            transform_shape
            and transform_finite
            and np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9)
            and np.allclose(
                transform[:3, :3].T @ transform[:3, :3],
                np.eye(3),
                atol=1e-5,
            )
        )
        width = getattr(calibration, "width", None)
        height = getattr(calibration, "height", None)
        calibration_name = getattr(calibration, "name", None)
        v_flipped = getattr(calibration, "observation_v_flipped", None)
        metadata_valid = bool(
            width == expected_width
            and not isinstance(width, bool)
            and height == expected_height
            and not isinstance(height, bool)
            and calibration_name == camera_name
            and type(v_flipped) is bool
        )
        calibration_valid = bool(
            calibration_exact
            and intrinsic_shape
            and intrinsic_dtype
            and intrinsic_finite
            and intrinsic_physical
            and transform_shape
            and transform_dtype
            and transform_finite
            and transform_physical
            and metadata_valid
        )
        if not calibration_valid:
            invalid.append(f"cameras.{camera_name}.calibration")
        shape_pass = shape_pass and intrinsic_shape and transform_shape
        dtype_pass = dtype_pass and intrinsic_dtype and transform_dtype
        finite_pass = finite_pass and intrinsic_finite and transform_finite
        calibration_pass = calibration_pass and calibration_valid

    proprio = getattr(observation, "proprio", None)
    proprio_names = _field_names(proprio)
    proprio_fields_exact = type(proprio) is Proprioception and (
        proprio_names == FIELD_CONTRACTS["Proprioception"]
    )
    forbidden_candidates.extend(
        name for name in proprio_names if name not in FIELD_CONTRACTS["Proprioception"]
    )
    proprio_specs = {
        "T_world_ee": ((4, 4), np.float64),
        "ee_quat_xyzw": ((4,), np.float64),
        "joint_position": ((7,), np.float64),
        "joint_velocity": ((7,), np.float64),
        "gripper_qpos": ((2,), np.float64),
        "gripper_qvel": ((2,), np.float64),
        "ee_force_sensor": ((3,), np.float64),
        "ee_torque_sensor": ((3,), np.float64),
    }
    proprio_valid = proprio_fields_exact
    for name, (shape, dtype) in proprio_specs.items():
        valid_shape, valid_dtype, finite = _array_checks(
            getattr(proprio, name, None), shape=shape, dtype=dtype
        )
        if not valid_shape:
            invalid.append(f"proprio.{name}.shape")
        if not valid_dtype:
            invalid.append(f"proprio.{name}.dtype")
        if not finite:
            invalid.append(f"proprio.{name}.finite")
        shape_pass = shape_pass and valid_shape
        dtype_pass = dtype_pass and valid_dtype
        finite_pass = finite_pass and finite
        proprio_valid = proprio_valid and valid_shape and valid_dtype and finite
    gripper_width = getattr(proprio, "gripper_width_m", None)
    gripper_valid = bool(
        isinstance(gripper_width, (int, float))
        and not isinstance(gripper_width, bool)
        and math.isfinite(float(gripper_width))
        and 0.0 <= float(gripper_width) <= 0.12
    )
    if not gripper_valid:
        invalid.append("proprio.gripper_width_m")
    finite_pass = finite_pass and gripper_valid
    proprio_valid = proprio_valid and gripper_valid

    return _ObservationEvent(
        type_exact=type(observation) is RobotObservation,
        field_names_exact=outer_exact,
        camera_names=camera_names,
        dual_camera_exact=dual_exact,
        camera_field_names_exact=camera_fields_exact,
        calibration_field_names_exact=calibration_fields_exact,
        proprio_field_names_exact=proprio_fields_exact,
        shape_checks_pass=shape_pass,
        dtype_checks_pass=dtype_pass,
        finiteness_checks_pass=finite_pass,
        calibration_checks_pass=calibration_pass,
        proprio_checks_pass=proprio_valid,
        invalid_paths=tuple(sorted(set(invalid))),
        forbidden_names=_forbidden_names(forbidden_candidates),
        destination=destination,
    )


def _all(events: list[Any], attribute: str) -> bool:
    return bool(events) and all(bool(getattr(event, attribute)) for event in events)


def _exact_int(value: object, *, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


class PolicyBoundaryAuditor:
    """Append-only per-episode boundary auditor for either Route B or Route C."""

    def __init__(
        self,
        route: PolicyRoute,
        *,
        expected_height: int = 256,
        expected_width: int = 256,
    ) -> None:
        if route not in {"b", "c"}:
            raise ValueError("route must be 'b' or 'c'")
        if not _exact_int(expected_height, minimum=1) or not _exact_int(
            expected_width, minimum=1
        ):
            raise ValueError("expected camera dimensions must be positive integers")
        self._route: PolicyRoute = route
        self._expected_height = expected_height
        self._expected_width = expected_width
        self._task_events: list[_TaskEvent] = []
        self._observation_events: list[_ObservationEvent] = []
        self._sequence_events: list[_SequenceEvent] = []
        self._evaluator_step_events: list[_EvaluatorStepEvent] = []
        self._forbidden_events: list[_ForbiddenEvent] = []
        self._route_c_process_boundary_events = 0
        self._route_c_process_bindings: list[dict[str, Any]] = []
        # Route C can ask several consumers for the very same immutable sensor
        # snapshot within one dispatcher generation.  Reuse only the most
        # recent structural check while still appending every actual handoff;
        # this bounds memory and avoids repeated full-image finite scans.
        self._last_observation: object | None = None
        self._last_observation_event: _ObservationEvent | None = None

    @property
    def route(self) -> PolicyRoute:
        return self._route

    @property
    def reset_delivery_count(self) -> int:
        return len(self._task_events)

    @property
    def act_delivery_count(self) -> int:
        return len(self._observation_events)

    def _destination(self) -> DeliveryDestination:
        return "policy" if self._route == "b" else "controller"

    def record_reset(self, task: object) -> None:
        """Append a Route B task envelope presented directly to ``policy.reset``.

        Route C must instead call :meth:`adapt_route_c_task_to_instruction` so
        a completed report proves that the full ``PolicyTask`` stopped at the
        evaluator-owned adapter and only its language crossed downstream.
        Calling this lower-level method for Route C is retained as an
        append-only misuse record; its adapter gate will fail closed.
        """

        event = _task_event(
            task,
            receiver=(
                "route_b_policy" if self._route == "b" else "route_c_boundary_adapter"
            ),
            boundary_adapter_applied=self._route == "b",
        )
        self._task_events.append(event)
        # Route B delivers the object directly to policy.reset, so unexpected
        # task fields are actual policy deliveries.  A Route C adapter rejects
        # a malformed envelope before forwarding anything; its exact-schema
        # gate still fails, without falsely claiming controller exposure.
        if self._route == "b":
            for name in event.forbidden_names:
                self._forbidden_events.append(_ForbiddenEvent(name, "policy", 1))

    def adapt_route_c_task_to_instruction(self, task: object) -> str:
        """Audit a Route C ``PolicyTask`` and forward only its instruction.

        This method is the explicit evaluator-to-coordinator boundary adapter.
        It receives the exact task envelope, records its whitelist contract,
        and returns one plain string.  The opaque episode id and the
        ``PolicyTask`` object itself are not made available to Route C.
        """

        if self._route != "c":
            raise ValueError("Route C task adapter can only be used for route 'c'")
        event = _task_event(
            task,
            receiver="route_c_boundary_adapter",
            boundary_adapter_applied=True,
        )
        self._task_events.append(event)
        if (
            not event.type_exact
            or event.field_names != FIELD_CONTRACTS["PolicyTask"]
            or not event.instruction_valid
            or not event.opaque_episode_id_valid
        ):
            raise PolicyBoundaryAuditValidationError(
                "Route C boundary adapter rejected a non-conforming PolicyTask"
            )
        # Exact type/field validation above makes this attribute access safe,
        # while the return annotation documents the only downstream payload.
        return task.instruction

    def record_route_c_process_boundary(self, attestation: object) -> None:
        """Bind a validated spawned-process transport to this Route C episode.

        The attestation contains channel schemas and capability-absence facts,
        never a simulator or evaluator object.  Repeated calls are retained as
        misuse and make the resulting handoff pattern fail closed.
        """

        if self._route != "c":
            raise ValueError("spawned Route C process applies only to route 'c'")
        from libero_system.integration.formal_route_c_process import (
            validate_process_boundary_attestation,
        )

        validated = validate_process_boundary_attestation(attestation)
        canonical = json.dumps(
            validated,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        receipt = validated.get("latest_episode_receipt")
        runtime = validated["runtime_receipt"]
        self._route_c_process_bindings.append(
            {
                "attestation_sha256": hashlib.sha256(
                    b"libero-route-c-process-attestation.v1\0" + canonical
                ).hexdigest(),
                "worker_nonce": runtime["worker_nonce"],
                "episode_nonce": (
                    receipt.get("episode_nonce")
                    if isinstance(receipt, Mapping)
                    else None
                ),
                "episode_index": (
                    receipt.get("episode_index")
                    if isinstance(receipt, Mapping)
                    else None
                ),
                "runtime_assembly_nonce": (
                    receipt.get("runtime_assembly_nonce")
                    if isinstance(receipt, Mapping)
                    else None
                ),
                "action_count": (
                    receipt.get("action_count")
                    if isinstance(receipt, Mapping)
                    else None
                ),
                "observation_count": (
                    receipt.get("observation_count")
                    if isinstance(receipt, Mapping)
                    else None
                ),
            }
        )
        self._route_c_process_boundary_events += 1

    def record_act(
        self,
        observation: object,
        *,
        act_index: object,
        internal_sequence: object,
        sequence_source: object = CONTROLLER_SEQUENCE_SOURCE,
        sequence_exposed_to_policy: object = False,
    ) -> None:
        """Append one observation handoff and its controller-owned sequence token.

        ``act_index`` and ``internal_sequence`` are audit-side integers.  They
        are never attached to ``RobotObservation`` and never forwarded to the
        policy.  Repeated calls with the same ``act_index`` prove that all
        sensor reads within one action saw a stable sequence value.
        """

        if observation is self._last_observation and self._last_observation_event is not None:
            event = self._last_observation_event
        else:
            event = _observation_event(
                observation,
                expected_height=self._expected_height,
                expected_width=self._expected_width,
                destination=self._destination(),
            )
            self._last_observation = observation
            self._last_observation_event = event
        self._observation_events.append(event)
        self._sequence_events.append(
            _SequenceEvent(
                act_index=_valid_nonnegative_integer(act_index),
                value=_valid_nonnegative_integer(internal_sequence),
                source=(sequence_source if isinstance(sequence_source, str) else None),
                exposed_to_policy=_valid_native_bool(sequence_exposed_to_policy),
            )
        )
        for name in event.forbidden_names:
            self._forbidden_events.append(_ForbiddenEvent(name, event.destination, 1))

    def record_forbidden_delivery(
        self,
        field_name: str,
        *,
        destination: DeliveryDestination | None = None,
        count: int = 1,
    ) -> None:
        """Append only a forbidden *name/count*, never its sensitive value."""

        canonical = _FORBIDDEN_ALIASES.get(str(field_name).lower())
        if canonical not in FORBIDDEN_DELIVERY_FIELDS:
            raise ValueError(f"unknown forbidden delivery field {field_name!r}")
        target = destination or self._destination()
        if target not in {"policy", "controller"}:
            raise ValueError("destination must be 'policy' or 'controller'")
        if not _exact_int(count, minimum=1):
            raise ValueError("forbidden delivery count must be a positive integer")
        self._forbidden_events.append(_ForbiddenEvent(canonical, target, count))

    def record_evaluator_step_count(
        self,
        value: object,
        *,
        source: object = EVALUATOR_STEP_COUNT_SOURCE,
        exposed_to_policy: object = False,
    ) -> None:
        """Append the final evaluator-owned action count and its provenance."""

        self._evaluator_step_events.append(
            _EvaluatorStepEvent(
                value=_valid_nonnegative_integer(value),
                source=source if isinstance(source, str) else None,
                exposed_to_policy=_valid_native_bool(exposed_to_policy),
            )
        )

    def to_report(self) -> dict[str, Any]:
        """Return the current deterministic, JSON-native episode report."""

        field_contracts = _runtime_field_contracts()
        task_events = self._task_events
        observation_events = self._observation_events
        sequence_events = self._sequence_events
        observation_applicable = bool(observation_events)
        sequence_applicable = bool(sequence_events)

        opaque_ids = [
            event.opaque_episode_id
            for event in task_events
            if event.opaque_episode_id is not None
        ]
        opaque_id_consistent = bool(opaque_ids) and (
            len(set(opaque_ids)) == 1 and len(opaque_ids) == len(task_events)
        )
        camera_counts = Counter(
            name for event in observation_events for name in event.camera_names
        )
        invalid_delivery_indices = [
            index
            for index, event in enumerate(observation_events)
            if not all(
                (
                    event.type_exact,
                    event.field_names_exact,
                    event.dual_camera_exact,
                    event.camera_field_names_exact,
                    event.calibration_field_names_exact,
                    event.proprio_field_names_exact,
                    event.shape_checks_pass,
                    event.dtype_checks_pass,
                    event.finiteness_checks_pass,
                    event.calibration_checks_pass,
                    event.proprio_checks_pass,
                )
            )
        ]
        invalid_paths = sorted(
            {
                path
                for event in observation_events
                for path in event.invalid_paths
            }
        )

        sequences_valid = (
            all(
                event.act_index is not None and event.value is not None
                for event in sequence_events
            )
            if sequence_applicable
            else None
        )
        values = [event.value for event in sequence_events if event.value is not None]
        monotonic = (
            bool(values)
            and all(
                previous <= current
                for previous, current in zip(values, values[1:])
            )
            if sequence_applicable
            else None
        )
        per_act_values: dict[int, set[int]] = defaultdict(set)
        act_sample_counts: Counter[int] = Counter()
        for event in sequence_events:
            if event.act_index is not None and event.value is not None:
                per_act_values[event.act_index].add(event.value)
                act_sample_counts[event.act_index] += 1
        same_act_stable = (
            bool(per_act_values)
            and all(len(group) == 1 for group in per_act_values.values())
            if sequence_applicable
            else None
        )
        sources_valid = (
            all(
                event.source == CONTROLLER_SEQUENCE_SOURCE
                for event in sequence_events
            )
            if sequence_applicable
            else None
        )
        sequence_exposure_valid = (
            all(event.exposed_to_policy is False for event in sequence_events)
            if sequence_applicable
            else None
        )

        combined_forbidden = Counter({name: 0 for name in FORBIDDEN_DELIVERY_FIELDS})
        by_destination = {
            destination: Counter({name: 0 for name in FORBIDDEN_DELIVERY_FIELDS})
            for destination in ("policy", "controller")
        }
        for event in self._forbidden_events:
            combined_forbidden[event.field_name] += event.count
            by_destination[event.destination][event.field_name] += event.count

        evaluator_event = (
            self._evaluator_step_events[0]
            if len(self._evaluator_step_events) == 1
            else None
        )
        report: dict[str, Any] = {
            "schema": POLICY_BOUNDARY_AUDIT_SCHEMA,
            "route": self._route,
            "handoff_pattern": (
                "route_b_policy_reset_act"
                if self._route == "b"
                else (
                    "route_c_spawned_process_serialized_rpc"
                    if self._route_c_process_boundary_events == 1
                    else (
                        "route_c_instruction_sensor_callbacks"
                        if self._route_c_process_boundary_events == 0
                        else "route_c_invalid_multiple_process_boundaries"
                    )
                )
            ),
            "formal_pass": False,
            "field_contracts": field_contracts,
            "task_delivery": {
                "allowed_fields": list(FIELD_CONTRACTS["PolicyTask"]),
                "receiver": (
                    "route_b_policy"
                    if self._route == "b"
                    else "route_c_boundary_adapter"
                ),
                "downstream_payload_kind": (
                    "PolicyTask" if self._route == "b" else "instruction"
                ),
                "downstream_field_names": (
                    list(FIELD_CONTRACTS["PolicyTask"])
                    if self._route == "b"
                    else ["instruction"]
                ),
                "policy_task_object_forwarded_downstream": self._route == "b",
                "reset_delivery_count": len(task_events),
                "boundary_adapter_applicable": self._route == "c",
                "boundary_adapter_applied": (
                    _all(task_events, "boundary_adapter_applied")
                    if self._route == "c"
                    else None
                ),
                "all_types_exact": _all(task_events, "type_exact"),
                "all_field_names_exact": bool(task_events)
                and all(
                    event.field_names == FIELD_CONTRACTS["PolicyTask"]
                    for event in task_events
                ),
                "all_instructions_valid": _all(task_events, "instruction_valid"),
                "all_opaque_episode_ids_valid": _all(
                    task_events, "opaque_episode_id_valid"
                ),
                "opaque_episode_id_consistent": opaque_id_consistent,
                "opaque_episode_id_sha256": (
                    hashlib.sha256(opaque_ids[0].encode("utf-8")).hexdigest()
                    if opaque_id_consistent
                    else None
                ),
            },
            "route_c_process_boundary": {
                "applicable": self._route == "c",
                "event_count": self._route_c_process_boundary_events,
                "binding": (
                    dict(self._route_c_process_bindings[0])
                    if len(self._route_c_process_bindings) == 1
                    else None
                ),
            },
            "observation_delivery": {
                "applicable": observation_applicable,
                "act_delivery_count": len(observation_events),
                "expected_height": self._expected_height,
                "expected_width": self._expected_width,
                "expected_camera_names": list(_CAMERA_NAMES),
                "dual_camera_delivery_count": sum(
                    event.dual_camera_exact for event in observation_events
                ),
                "camera_delivery_counts": {
                    name: int(camera_counts[name]) for name in _CAMERA_NAMES
                },
                "all_types_exact": (
                    _all(observation_events, "type_exact")
                    if observation_applicable
                    else None
                ),
                "all_field_names_exact": (
                    _all(observation_events, "field_names_exact")
                    if observation_applicable
                    else None
                ),
                "all_camera_names_exact": (
                    _all(observation_events, "dual_camera_exact")
                    if observation_applicable
                    else None
                ),
                "all_camera_field_names_exact": (
                    _all(observation_events, "camera_field_names_exact")
                    if observation_applicable
                    else None
                ),
                "all_calibration_field_names_exact": (
                    _all(observation_events, "calibration_field_names_exact")
                    if observation_applicable
                    else None
                ),
                "all_proprio_field_names_exact": (
                    _all(observation_events, "proprio_field_names_exact")
                    if observation_applicable
                    else None
                ),
                "shape_checks_pass": (
                    _all(observation_events, "shape_checks_pass")
                    if observation_applicable
                    else None
                ),
                "dtype_checks_pass": (
                    _all(observation_events, "dtype_checks_pass")
                    if observation_applicable
                    else None
                ),
                "finiteness_checks_pass": (
                    _all(observation_events, "finiteness_checks_pass")
                    if observation_applicable
                    else None
                ),
                "calibration_checks_pass": (
                    _all(observation_events, "calibration_checks_pass")
                    if observation_applicable
                    else None
                ),
                "proprio_checks_pass": (
                    _all(observation_events, "proprio_checks_pass")
                    if observation_applicable
                    else None
                ),
                "invalid_delivery_indices": invalid_delivery_indices,
                "invalid_paths": invalid_paths,
            },
            "internal_sequence": {
                "applicable": sequence_applicable,
                "source": CONTROLLER_SEQUENCE_SOURCE,
                "sample_count": len(sequence_events),
                "unique_act_count": len(per_act_values),
                "repeated_same_act_sample_count": sum(
                    count - 1 for count in act_sample_counts.values()
                ),
                "values_valid": sequences_valid,
                "all_sources_controller_owned": sources_valid,
                "same_act_stable": same_act_stable,
                "monotonic": monotonic,
                "exposed_to_policy": False,
                "all_exposure_flags_false": sequence_exposure_valid,
            },
            "evaluator_step_count": {
                "event_count": len(self._evaluator_step_events),
                "source": evaluator_event.source if evaluator_event else None,
                "value": evaluator_event.value if evaluator_event else None,
                "exposed_to_policy": (
                    evaluator_event.exposed_to_policy if evaluator_event else None
                ),
            },
            "forbidden_delivery_counts": {
                name: int(combined_forbidden[name])
                for name in FORBIDDEN_DELIVERY_FIELDS
            },
            "forbidden_delivery_counts_by_destination": {
                destination: {
                    name: int(by_destination[destination][name])
                    for name in FORBIDDEN_DELIVERY_FIELDS
                }
                for destination in ("policy", "controller")
            },
            "violations": [],
        }
        violations = _strict_report_violations(report, require_declared_pass=False)
        report["violations"] = violations
        report["formal_pass"] = not violations
        return report


def _strict_report_violations(
    report: Mapping[str, Any], *, require_declared_pass: bool
) -> list[str]:
    violations: list[str] = []
    if report.get("schema") != POLICY_BOUNDARY_AUDIT_SCHEMA:
        violations.append("schema_mismatch")
    route = report.get("route")
    if route not in {"b", "c"}:
        violations.append("route_invalid")
    allowed_patterns = {
        "b": {"route_b_policy_reset_act"},
        "c": {
            "route_c_instruction_sensor_callbacks",
            "route_c_spawned_process_serialized_rpc",
        },
    }.get(route, set())
    if report.get("handoff_pattern") not in allowed_patterns:
        violations.append("handoff_pattern_invalid")

    contracts = report.get("field_contracts")
    if not isinstance(contracts, Mapping) or set(contracts) != set(FIELD_CONTRACTS):
        violations.append("field_contracts_invalid")
    else:
        for name, expected in FIELD_CONTRACTS.items():
            contract = contracts.get(name)
            if not isinstance(contract, Mapping) or (
                set(contract) != {"expected_fields", "reflected_fields", "exact"}
                or
                contract.get("expected_fields") != list(expected)
                or contract.get("reflected_fields") != list(expected)
                or contract.get("exact") is not True
            ):
                violations.append(f"field_contract_{name}_invalid")

    task = report.get("task_delivery")
    if not isinstance(task, Mapping):
        violations.append("task_delivery_invalid")
    else:
        expected_task_keys = {
            "allowed_fields",
            "receiver",
            "downstream_payload_kind",
            "downstream_field_names",
            "policy_task_object_forwarded_downstream",
            "reset_delivery_count",
            "boundary_adapter_applicable",
            "boundary_adapter_applied",
            "all_types_exact",
            "all_field_names_exact",
            "all_instructions_valid",
            "all_opaque_episode_ids_valid",
            "opaque_episode_id_consistent",
            "opaque_episode_id_sha256",
        }
        if set(task) != expected_task_keys:
            violations.append("task_delivery_fields_invalid")
        if task.get("allowed_fields") != list(FIELD_CONTRACTS["PolicyTask"]):
            violations.append("task_allowed_fields_invalid")
        expected_task_semantics = {
            "b": {
                "receiver": "route_b_policy",
                "downstream_payload_kind": "PolicyTask",
                "downstream_field_names": list(FIELD_CONTRACTS["PolicyTask"]),
                "policy_task_object_forwarded_downstream": True,
                "boundary_adapter_applicable": False,
                "boundary_adapter_applied": None,
            },
            "c": {
                "receiver": "route_c_boundary_adapter",
                "downstream_payload_kind": "instruction",
                "downstream_field_names": ["instruction"],
                "policy_task_object_forwarded_downstream": False,
                "boundary_adapter_applicable": True,
                "boundary_adapter_applied": True,
            },
        }.get(route, {})
        for name, expected in expected_task_semantics.items():
            if task.get(name) != expected or type(task.get(name)) is not type(expected):
                violations.append(f"task_{name}_invalid")
        if task.get("reset_delivery_count") != 1:
            violations.append("reset_delivery_count_invalid")
        for name in (
            "all_types_exact",
            "all_field_names_exact",
            "all_instructions_valid",
            "all_opaque_episode_ids_valid",
            "opaque_episode_id_consistent",
        ):
            if task.get(name) is not True:
                violations.append(f"task_{name}_invalid")
        opaque_commitment = task.get("opaque_episode_id_sha256")
        if not isinstance(opaque_commitment, str) or re.fullmatch(
            r"[0-9a-f]{64}", opaque_commitment
        ) is None:
            violations.append("task_opaque_episode_id_sha256_invalid")

    process_boundary = report.get("route_c_process_boundary")
    if not isinstance(process_boundary, Mapping) or set(process_boundary) != {
        "applicable",
        "event_count",
        "binding",
    }:
        violations.append("route_c_process_boundary_invalid")
    elif route == "b":
        if process_boundary != {
            "applicable": False,
            "event_count": 0,
            "binding": None,
        }:
            violations.append("route_b_process_boundary_must_be_absent")
    else:
        count = process_boundary.get("event_count")
        binding = process_boundary.get("binding")
        if process_boundary.get("applicable") is not True or type(count) is not int:
            violations.append("route_c_process_boundary_count_invalid")
        elif count == 1:
            if not isinstance(binding, Mapping) or set(binding) != {
                "attestation_sha256",
                "worker_nonce",
                "episode_nonce",
                "episode_index",
                "runtime_assembly_nonce",
                "action_count",
                "observation_count",
            }:
                violations.append("route_c_process_boundary_binding_invalid")
            elif any(
                not isinstance(binding.get(name), str)
                or re.fullmatch(r"[0-9a-f]{64}", binding[name]) is None
                for name in (
                    "attestation_sha256",
                    "worker_nonce",
                    "episode_nonce",
                    "runtime_assembly_nonce",
                )
            ):
                violations.append("route_c_process_boundary_nonce_invalid")
            elif any(
                type(binding.get(name)) is not int or binding[name] < minimum
                for name, minimum in (
                    ("episode_index", 1),
                    ("action_count", 0),
                    ("observation_count", 1),
                )
            ):
                violations.append("route_c_process_boundary_counts_invalid")
        elif count == 0 and binding is not None:
            violations.append("route_c_process_boundary_unexpected_binding")
        elif count not in {0, 1}:
            violations.append("route_c_process_boundary_count_invalid")

    observation = report.get("observation_delivery")
    act_count: int | None = None
    if not isinstance(observation, Mapping):
        violations.append("observation_delivery_invalid")
    else:
        expected_observation_keys = {
            "applicable",
            "act_delivery_count",
            "expected_height",
            "expected_width",
            "expected_camera_names",
            "dual_camera_delivery_count",
            "camera_delivery_counts",
            "all_types_exact",
            "all_field_names_exact",
            "all_camera_names_exact",
            "all_camera_field_names_exact",
            "all_calibration_field_names_exact",
            "all_proprio_field_names_exact",
            "shape_checks_pass",
            "dtype_checks_pass",
            "finiteness_checks_pass",
            "calibration_checks_pass",
            "proprio_checks_pass",
            "invalid_delivery_indices",
            "invalid_paths",
        }
        if set(observation) != expected_observation_keys:
            violations.append("observation_delivery_fields_invalid")
        if not _exact_int(observation.get("expected_height"), minimum=1):
            violations.append("observation_expected_height_invalid")
        if not _exact_int(observation.get("expected_width"), minimum=1):
            violations.append("observation_expected_width_invalid")
        raw_act_count = observation.get("act_delivery_count")
        if not _exact_int(raw_act_count, minimum=0):
            violations.append("act_delivery_count_invalid")
        else:
            act_count = raw_act_count
        observation_applicable = act_count is not None and act_count > 0
        if observation.get("applicable") is not observation_applicable:
            violations.append("observation_applicable_invalid")
        if route == "b" and act_count == 0:
            violations.append("route_b_observation_delivery_required")
        if observation.get("expected_camera_names") != list(_CAMERA_NAMES):
            violations.append("expected_camera_names_invalid")
        if (
            act_count is not None
            and (
                type(observation.get("dual_camera_delivery_count")) is not int
                or observation.get("dual_camera_delivery_count") != act_count
            )
        ):
            violations.append("dual_camera_delivery_count_invalid")
        camera_counts = observation.get("camera_delivery_counts")
        if not isinstance(camera_counts, Mapping) or set(camera_counts) != set(_CAMERA_NAMES):
            violations.append("camera_delivery_counts_invalid")
        elif act_count is not None and any(
            type(camera_counts.get(name)) is not int
            or camera_counts.get(name) != act_count
            for name in _CAMERA_NAMES
        ):
            violations.append("camera_delivery_count_mismatch")
        for name in (
            "all_types_exact",
            "all_field_names_exact",
            "all_camera_names_exact",
            "all_camera_field_names_exact",
            "all_calibration_field_names_exact",
            "all_proprio_field_names_exact",
            "shape_checks_pass",
            "dtype_checks_pass",
            "finiteness_checks_pass",
            "calibration_checks_pass",
            "proprio_checks_pass",
        ):
            expected = True if observation_applicable else None
            if observation.get(name) is not expected:
                violations.append(f"observation_{name}_invalid")
        if observation.get("invalid_delivery_indices") != []:
            violations.append("invalid_observation_deliveries")
        if observation.get("invalid_paths") != []:
            violations.append("invalid_observation_paths")

    sequence = report.get("internal_sequence")
    if not isinstance(sequence, Mapping):
        violations.append("internal_sequence_invalid")
    else:
        expected_sequence_keys = {
            "applicable",
            "source",
            "sample_count",
            "unique_act_count",
            "repeated_same_act_sample_count",
            "values_valid",
            "all_sources_controller_owned",
            "same_act_stable",
            "monotonic",
            "exposed_to_policy",
            "all_exposure_flags_false",
        }
        if set(sequence) != expected_sequence_keys:
            violations.append("internal_sequence_fields_invalid")
        sequence_applicable = act_count is not None and act_count > 0
        if sequence.get("applicable") is not sequence_applicable:
            violations.append("internal_sequence_applicable_invalid")
        if sequence.get("source") != CONTROLLER_SEQUENCE_SOURCE:
            violations.append("internal_sequence_source_invalid")
        if not _exact_int(sequence.get("sample_count"), minimum=0):
            violations.append("internal_sequence_sample_count_invalid")
        elif act_count is not None and sequence.get("sample_count") != act_count:
            violations.append("internal_sequence_sample_count_mismatch")
        minimum_unique = 1 if sequence_applicable else 0
        if not _exact_int(
            sequence.get("unique_act_count"), minimum=minimum_unique
        ):
            violations.append("internal_sequence_unique_act_count_invalid")
        elif not sequence_applicable and sequence.get("unique_act_count") != 0:
            violations.append("internal_sequence_unique_act_count_invalid")
        if not _exact_int(
            sequence.get("repeated_same_act_sample_count"), minimum=0
        ):
            violations.append(
                "internal_sequence_repeated_same_act_sample_count_invalid"
            )
        elif (
            _exact_int(sequence.get("sample_count"), minimum=0)
            and _exact_int(sequence.get("unique_act_count"), minimum=0)
            and sequence.get("repeated_same_act_sample_count")
            != sequence.get("sample_count") - sequence.get("unique_act_count")
        ):
            violations.append("internal_sequence_repeat_count_mismatch")
        for name in (
            "values_valid",
            "all_sources_controller_owned",
            "same_act_stable",
            "monotonic",
            "all_exposure_flags_false",
        ):
            expected = True if sequence_applicable else None
            if sequence.get(name) is not expected:
                violations.append(f"internal_sequence_{name}_invalid")
        if sequence.get("exposed_to_policy") is not False:
            violations.append("internal_sequence_exposure_invalid")

    evaluator_count = report.get("evaluator_step_count")
    evaluator_step_value: int | None = None
    if not isinstance(evaluator_count, Mapping):
        violations.append("evaluator_step_count_invalid")
    else:
        if set(evaluator_count) != {
            "event_count",
            "source",
            "value",
            "exposed_to_policy",
        }:
            violations.append("evaluator_step_count_fields_invalid")
        if evaluator_count.get("event_count") != 1:
            violations.append("evaluator_step_count_event_count_invalid")
        if evaluator_count.get("source") != EVALUATOR_STEP_COUNT_SOURCE:
            violations.append("evaluator_step_count_source_invalid")
        if not _exact_int(evaluator_count.get("value"), minimum=0):
            violations.append("evaluator_step_count_value_invalid")
        else:
            evaluator_step_value = evaluator_count.get("value")
        if evaluator_count.get("exposed_to_policy") is not False:
            violations.append("evaluator_step_count_exposure_invalid")

    # Route C may reject language before requesting its first sensor snapshot.
    # That is a genuine zero-step policy failure, not infrastructure failure.
    # Conversely, executed actions without a single audited sensor handoff
    # prove that the wrapper missed the real controller boundary.
    if route == "c" and act_count == 0 and evaluator_step_value != 0:
        violations.append("route_c_zero_handoff_requires_zero_evaluator_steps")

    forbidden = report.get("forbidden_delivery_counts")
    if not isinstance(forbidden, Mapping) or set(forbidden) != set(FORBIDDEN_DELIVERY_FIELDS):
        violations.append("forbidden_delivery_counts_invalid")
    else:
        for name in FORBIDDEN_DELIVERY_FIELDS:
            if forbidden.get(name) != 0 or type(forbidden.get(name)) is not int:
                violations.append(f"forbidden_delivery_{name}_nonzero")
    by_destination = report.get("forbidden_delivery_counts_by_destination")
    if not isinstance(by_destination, Mapping) or set(by_destination) != {
        "policy",
        "controller",
    }:
        violations.append("forbidden_destination_counts_invalid")
    else:
        for destination in ("policy", "controller"):
            counts = by_destination.get(destination)
            if not isinstance(counts, Mapping) or set(counts) != set(
                FORBIDDEN_DELIVERY_FIELDS
            ):
                violations.append(f"forbidden_{destination}_counts_invalid")
                continue
            for name in FORBIDDEN_DELIVERY_FIELDS:
                if counts.get(name) != 0 or type(counts.get(name)) is not int:
                    violations.append(f"forbidden_{destination}_{name}_nonzero")

    if require_declared_pass:
        if report.get("formal_pass") is not True:
            violations.append("formal_pass_not_true")
        declared = report.get("violations")
        if declared != []:
            violations.append("declared_violations_not_empty")
    return list(dict.fromkeys(violations))


def validate_policy_boundary_audit(report: Mapping[str, Any]) -> dict[str, Any]:
    """Strictly validate a completed report and return a detached JSON copy."""

    if not isinstance(report, Mapping):
        raise PolicyBoundaryAuditValidationError("policy boundary audit must be an object")
    expected_top_level = {
        "schema",
        "route",
        "handoff_pattern",
        "formal_pass",
        "field_contracts",
        "task_delivery",
        "route_c_process_boundary",
        "observation_delivery",
        "internal_sequence",
        "evaluator_step_count",
        "forbidden_delivery_counts",
        "forbidden_delivery_counts_by_destination",
        "violations",
    }
    if set(report) != expected_top_level:
        missing = sorted(expected_top_level - set(report))
        extra = sorted(set(report) - expected_top_level)
        raise PolicyBoundaryAuditValidationError(
            f"policy boundary audit fields mismatch: missing={missing}, extra={extra}"
        )
    try:
        normalized = json.loads(
            json.dumps(
                dict(report),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    except (TypeError, ValueError) as exc:
        raise PolicyBoundaryAuditValidationError(
            f"policy boundary audit is not strict JSON: {exc}"
        ) from exc
    violations = _strict_report_violations(normalized, require_declared_pass=True)
    if violations:
        raise PolicyBoundaryAuditValidationError(
            "policy boundary audit failed: " + ", ".join(violations)
        )
    return normalized


__all__ = [
    "CONTROLLER_SEQUENCE_SOURCE",
    "EVALUATOR_STEP_COUNT_SOURCE",
    "FIELD_CONTRACTS",
    "FORBIDDEN_DELIVERY_FIELDS",
    "POLICY_BOUNDARY_AUDIT_SCHEMA",
    "PolicyBoundaryAuditValidationError",
    "PolicyBoundaryAuditor",
    "validate_policy_boundary_audit",
]
