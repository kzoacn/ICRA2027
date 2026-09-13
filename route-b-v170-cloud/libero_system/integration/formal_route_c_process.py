"""Sandboxed, bounded, replay-resistant Route-C policy process boundary.

The parent retains MuJoCo, video, score, and all benchmark metadata.  The
policy is launched through bubblewrap in isolated user/PID/mount/network/IPC/
UTS namespaces.  Its filesystem contains a fixed policy-source projection,
the exact static gallery, one frozen model repository, and a read-only Conda
runtime with LIBERO/BDDL/robosuite/MuJoCo packages masked.

This is an auditable process separation boundary, not a cryptographic defence
against a parent that forges its own result artifacts.  Runtime receipts bind
the live child, namespaces, mounts, environment, model-load count, per-episode
nonces, strict message sequence, and sensor/action/result transcript.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import secrets
import socket
import subprocess
import sys
from typing import Any, Callable, Mapping

import numpy as np

from libero_system.common.grasp_journal import (
    GraspAttemptEvent,
    GraspEvidenceCategory,
    GraspOutcome,
    GraspReason,
    JawBehavior,
    PendingGraspEngagement,
)
from libero_system.common.observation import RobotObservation
from libero_system.common.policy import OSCAction
from libero_system.integration.components import (
    PerceptionFactoryContext,
    build_perception_bundle_from_context,
)
from libero_system.integration.formal_process_sandbox import (
    BWRAP_PATH,
    EXCLUDED_POLICY_FILES,
    MASKED_SITE_PACKAGES,
    POLICY_SOURCE_FILES,
    POLICY_SOURCE_ROOT,
    SANDBOX_ASSET_ROOT,
    SANDBOX_HOSTNAME,
    SANDBOX_RUNTIME_ROOT,
    SYSTEM_RUNTIME_FILES,
    SandboxProjection,
    environment_sha256,
    fixed_child_environment,
    sha256_file,
    tree_manifest_sha256,
)
from libero_system.integration.formal_process_transport import (
    BoundedFrameSocket,
    FormalProcessTransportError,
    strict_json_bytes,
    strict_json_loads,
)
from libero_system.integration.policy_ipc import (
    ACTION_WIRE_SCHEMA,
    OBSERVATION_WIRE_SCHEMA,
    PolicyIPCValidationError,
    deserialize_policy_action,
    deserialize_policy_observation,
    sensor_content_commitments,
    serialize_policy_action,
    serialize_policy_observation,
)
from libero_system.integration.policy_result import RouteCPolicyExecution


PROCESS_INIT_SCHEMA = "libero-formal-route-c-process-init.v2"
PROCESS_MESSAGE_SCHEMA = "libero-formal-route-c-process-message.v2"
PROCESS_BOUNDARY_ATTESTATION_SCHEMA = (
    "libero-formal-route-c-process-boundary-attestation.v2"
)
PROCESS_RUNTIME_RECEIPT_SCHEMA = "libero-formal-route-c-runtime-receipt.v2"
PROCESS_EPISODE_RECEIPT_SCHEMA = "libero-formal-route-c-episode-receipt.v2"
SENSOR_COMMITMENT_SCHEMA = "libero-sensor-content-commitments.v2"
EPISODE_RUNTIME_SCHEMA = "libero-route-c-episode-runtime.v1"

MAX_POLICY_STEPS = 1_024
MAX_INSTRUCTION_CHARS = 4_096
MAX_GRASP_EVENTS = MAX_POLICY_STEPS
_CHILD_IO_TIMEOUT_S = 900.0
_MODEL_BACKENDS = {"auto", "gallery", "grounding-dino"}
_DEVICE = re.compile(r"(?:cpu|cuda(?::\d+)?)")
_HASH = re.compile(r"[0-9a-f]{64}")
_NAMESPACE_NAMES = ("ipc", "mnt", "net", "pid", "user", "uts")
_FORBIDDEN_VISIBLE_PATHS = (
    "/mnt/d/work/libero/background",
    "/mnt/d/work/libero/.local/libero",
    "/home/kzoacn/.cache/libero/assets/scenes",
    "/policy/libero_system/common/env_adapter.py",
    "/policy/libero_system/common/runner.py",
    "/policy/libero_system/integration/campaign_manifest.py",
    "/policy/libero_system/integration/campaign_plan.py",
    "/policy/libero_system/integration/evaluator.py",
    "/policy/libero_system/integration/results.py",
    "/policy/libero_system/integration/video.py",
    "/runtime/lib/python3.12/site-packages/libero/__init__.py",
    "/runtime/lib/python3.12/site-packages/libero/libero/bddl_files",
    "/runtime/lib/python3.12/site-packages/libero/libero/init_files",
    "/runtime/lib/python3.12/site-packages/bddl/__init__.py",
    "/runtime/lib/python3.12/site-packages/robosuite/__init__.py",
    "/runtime/lib/python3.12/site-packages/mujoco/__init__.py",
)
_EMBEDDED_TEST_POLICY_CAPABILITY = object()
_BOOTSTRAP = (
    "import sys;"
    "sys.path[:]=['/policy','/runtime/lib/python3.12/site-packages',"
    "'/runtime/lib/python3.12','/runtime/lib/python3.12/lib-dynload'];"
    "from libero_system.integration.formal_route_c_process import _child_bootstrap;"
    "sys.exit(_child_bootstrap(int(sys.argv[1]),sys.argv[2]=='1'))"
)


class FormalRouteCProcessError(RuntimeError):
    """Raised when the formal policy child violates its narrow contract."""


def _hash_valid(value: object) -> bool:
    return type(value) is str and _HASH.fullmatch(value) is not None


def _uses_accelerator_environment(environment: object) -> bool:
    return isinstance(environment, dict) and environment.get("LD_LIBRARY_PATH") == (
        fixed_child_environment(accelerator=True)["LD_LIBRARY_PATH"]
    )


def _accelerator_runtime_paths() -> list[str]:
    """Collect the GPU driver files actually opened/mapped by this worker."""
    paths = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
             if "/usr/lib/wsl/" in line}
    for entry in Path("/proc/self/fd").iterdir():
        try:
            target = os.readlink(entry)
        except FileNotFoundError:
            continue
        if target.startswith("/usr/lib/wsl/"):
            paths.add(target)
    return sorted(path for path in paths if Path(path).is_file())


def _strict_json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return strict_json_loads(strict_json_bytes(dict(value)))
    except FormalProcessTransportError as exc:
        raise FormalRouteCProcessError(str(exc)) from exc


def _json_sha256(value: Mapping[str, Any], domain: bytes) -> str:
    try:
        payload = strict_json_bytes(dict(value))
    except FormalProcessTransportError as exc:
        raise FormalRouteCProcessError(str(exc)) from exc
    return hashlib.sha256(domain + payload).hexdigest()


def _init_payload(context: PerceptionFactoryContext) -> dict[str, Any]:
    if type(context) is not PerceptionFactoryContext:
        raise TypeError("formal Route C context must use the exact safe DTO")
    if (
        type(context.device) is not str
        or _DEVICE.fullmatch(context.device) is None
        or type(context.perception_backend) is not str
        or context.perception_backend not in _MODEL_BACKENDS
        or type(context.image_size) is not int
        or not 64 <= context.image_size <= 512
        or (
            context.perception_model is not None
            and not isinstance(context.perception_model, Path)
        )
    ):
        raise ValueError("formal Route C perception context is invalid")
    return {
        "schema": PROCESS_INIT_SCHEMA,
        "device": context.device,
        "perception_backend": context.perception_backend,
        "perception_model": (
            str(context.perception_model)
            if context.perception_model is not None
            else None
        ),
        "image_size": context.image_size,
    }


def _context_from_init(payload: object) -> PerceptionFactoryContext:
    keys = {
        "schema",
        "device",
        "perception_backend",
        "perception_model",
        "image_size",
    }
    if type(payload) is not dict or set(payload) != keys:
        raise FormalRouteCProcessError("child init payload has invalid fields")
    if payload["schema"] != PROCESS_INIT_SCHEMA or type(payload["schema"]) is not str:
        raise FormalRouteCProcessError("child init schema is invalid")
    device = payload["device"]
    backend = payload["perception_backend"]
    image_size = payload["image_size"]
    model = payload["perception_model"]
    if (
        type(device) is not str
        or _DEVICE.fullmatch(device) is None
        or type(backend) is not str
        or backend not in _MODEL_BACKENDS
        or type(image_size) is not int
        or not 64 <= image_size <= 512
        or (model is not None and (type(model) is not str or not model))
    ):
        raise FormalRouteCProcessError("child init deployment values are invalid")
    model_path = Path(model).resolve(strict=True) if model is not None else None
    return PerceptionFactoryContext(device, backend, model_path, image_size)


def _sanitized_child_environment() -> dict[str, str]:
    """Compatibility name for the fixed, non-inherited child environment."""

    return fixed_child_environment()


def _event_from_dict(value: object) -> GraspAttemptEvent:
    keys = {
        "attempt_index",
        "source_text",
        "source_class",
        "grasp_mode",
        "jaw_behavior",
        "accepted",
        "outcome",
        "reason",
        "evidence_source",
    }
    if type(value) is not dict or set(value) != keys:
        raise FormalRouteCProcessError("child grasp event has invalid fields")
    try:
        return GraspAttemptEvent(
            attempt_index=value["attempt_index"],
            source_text=value["source_text"],
            source_class=value["source_class"],
            grasp_mode=value["grasp_mode"],
            jaw_behavior=JawBehavior(value["jaw_behavior"]),
            accepted=value["accepted"],
            outcome=GraspOutcome(value["outcome"]),
            reason=GraspReason(value["reason"]),
            evidence_source=GraspEvidenceCategory(value["evidence_source"]),
        )
    except (TypeError, ValueError) as exc:
        raise FormalRouteCProcessError(f"child grasp event is invalid: {exc}") from exc


def _pending_from_dict(value: object) -> PendingGraspEngagement | None:
    if value is None:
        return None
    keys = {
        "attempt_index",
        "source_text",
        "source_class",
        "grasp_mode",
        "jaw_behavior",
    }
    if type(value) is not dict or set(value) != keys:
        raise FormalRouteCProcessError("child pending engagement has invalid fields")
    try:
        return PendingGraspEngagement(
            attempt_index=value["attempt_index"],
            source_text=value["source_text"],
            source_class=value["source_class"],
            grasp_mode=value["grasp_mode"],
            jaw_behavior=JawBehavior(value["jaw_behavior"]),
        )
    except (TypeError, ValueError) as exc:
        raise FormalRouteCProcessError(
            f"child pending engagement is invalid: {exc}"
        ) from exc


def _trace_json_copy(trace: Mapping[str, Any]) -> dict[str, Any]:
    """Convert numerical diagnostics to native JSON before transport validation.

    Geometry calculations naturally produce NumPy scalars and arrays. They are
    policy-generated diagnostics. Unavailable geometric errors use infinity as
    a sentinel; preserve those as explicit strings so a failed visual check
    cannot terminate the worker or discard the next episode.
    """

    def numerical_value(value: object) -> Any:
        if isinstance(value, (np.floating, np.integer, np.bool_)):
            return numerical_value(value.item())
        if isinstance(value, np.ndarray):
            return numerical_value(value.tolist())
        if isinstance(value, float) and not np.isfinite(value):
            return "NaN" if np.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
        if isinstance(value, dict):
            return {key: numerical_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [numerical_value(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError(f"unsupported policy trace value: {type(value).__name__}")

    try:
        encoded = json.dumps(numerical_value(dict(trace)), allow_nan=False)
        return strict_json_loads(encoded.encode("utf-8"))
    except (TypeError, ValueError, FormalProcessTransportError) as exc:
        raise FormalRouteCProcessError(f"cannot encode policy trace: {exc}") from exc


def _result_core(
    result: RouteCPolicyExecution,
    episode_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    pending = result.pending_grasp_engagement
    return {
        "type": "result",
        "success": result.success,
        "failure": result.failure,
        "steps_executed": result.steps_executed,
        "trace": _trace_json_copy(result.trace),
        "grasp_attempt_events": [item.to_dict() for item in result.grasp_attempt_events],
        "pending_grasp_engagement": pending.to_dict() if pending is not None else None,
        "episode_receipt": dict(episode_receipt),
    }


def _result_from_core(payload: object) -> RouteCPolicyExecution:
    keys = {
        "type",
        "success",
        "failure",
        "steps_executed",
        "trace",
        "grasp_attempt_events",
        "pending_grasp_engagement",
        "episode_receipt",
    }
    if type(payload) is not dict or set(payload) != keys or payload["type"] != "result":
        raise FormalRouteCProcessError("child result has invalid fields")
    if type(payload["success"]) is not bool:
        raise FormalRouteCProcessError("child result success is not a native boolean")
    if payload["failure"] is not None and type(payload["failure"]) is not str:
        raise FormalRouteCProcessError("child result failure is invalid")
    if type(payload["steps_executed"]) is not int or not (
        0 <= payload["steps_executed"] <= MAX_POLICY_STEPS
    ):
        raise FormalRouteCProcessError("child result step count is invalid")
    if type(payload["trace"]) is not dict:
        raise FormalRouteCProcessError("child result trace is invalid")
    events = payload["grasp_attempt_events"]
    if type(events) is not list or len(events) > MAX_GRASP_EVENTS:
        raise FormalRouteCProcessError("child result grasp events are invalid")
    return RouteCPolicyExecution(
        success=payload["success"],
        failure=payload["failure"],
        steps_executed=payload["steps_executed"],
        trace=payload["trace"],
        grasp_attempt_events=tuple(_event_from_dict(item) for item in events),
        pending_grasp_engagement=_pending_from_dict(payload["pending_grasp_engagement"]),
    )


def _transcript_seed(
    worker_nonce: str,
    episode_nonce: str,
    episode_index: int,
) -> str:
    return hashlib.sha256(
        b"libero-route-c-transcript-seed.v1\0"
        + bytes.fromhex(worker_nonce)
        + bytes.fromhex(episode_nonce)
        + episode_index.to_bytes(8, "big")
    ).hexdigest()


class _Transcript:
    __slots__ = (
        "worker_nonce",
        "episode_nonce",
        "episode_index",
        "next_sequence",
        "sha256",
    )

    def __init__(
        self,
        *,
        worker_nonce: str,
        episode_nonce: str,
        episode_index: int,
    ) -> None:
        if not _hash_valid(worker_nonce) or not _hash_valid(episode_nonce):
            raise FormalRouteCProcessError("transcript nonce is invalid")
        if type(episode_index) is not int or episode_index < 1:
            raise FormalRouteCProcessError("transcript episode index is invalid")
        self.worker_nonce = worker_nonce
        self.episode_nonce = episode_nonce
        self.episode_index = episode_index
        self.next_sequence = 0
        self.sha256 = _transcript_seed(worker_nonce, episode_nonce, episode_index)

    def _next_hash(
        self,
        *,
        direction: str,
        message_type: str,
        core: Mapping[str, Any],
    ) -> str:
        payload = strict_json_bytes(dict(core))
        return hashlib.sha256(
            b"libero-route-c-transcript-link.v1\0"
            + bytes.fromhex(self.sha256)
            + direction.encode("ascii")
            + self.next_sequence.to_bytes(8, "big")
            + message_type.encode("ascii")
            + payload
        ).hexdigest()

    def wrap(self, core: Mapping[str, Any], *, direction: str) -> dict[str, Any]:
        if type(core) is not dict or type(core.get("type")) is not str:
            raise FormalRouteCProcessError("transcript core is invalid")
        result = {
            "schema": PROCESS_MESSAGE_SCHEMA,
            "worker_nonce": self.worker_nonce,
            "episode_nonce": self.episode_nonce,
            "episode_index": self.episode_index,
            "message_sequence": self.next_sequence,
            "previous_transcript_sha256": self.sha256,
            **core,
        }
        next_hash = self._next_hash(
            direction=direction,
            message_type=core["type"],
            core=core,
        )
        result["transcript_sha256"] = next_hash
        self.next_sequence += 1
        self.sha256 = next_hash
        return result

    def unwrap(self, message: object, *, direction: str) -> dict[str, Any]:
        if type(message) is not dict:
            raise FormalRouteCProcessError("chained message must be an exact object")
        envelope = {
            "schema",
            "worker_nonce",
            "episode_nonce",
            "episode_index",
            "message_sequence",
            "previous_transcript_sha256",
            "transcript_sha256",
        }
        if not envelope < set(message):
            raise FormalRouteCProcessError("chained message envelope is incomplete")
        if (
            message["schema"] != PROCESS_MESSAGE_SCHEMA
            or message["worker_nonce"] != self.worker_nonce
            or message["episode_nonce"] != self.episode_nonce
            or message["episode_index"] != self.episode_index
            or type(message["message_sequence"]) is not int
            or message["message_sequence"] != self.next_sequence
            or message["previous_transcript_sha256"] != self.sha256
            or not _hash_valid(message["transcript_sha256"])
        ):
            raise FormalRouteCProcessError("stale, replayed, or reordered process message")
        core = {name: value for name, value in message.items() if name not in envelope}
        if type(core.get("type")) is not str:
            raise FormalRouteCProcessError("chained message type is invalid")
        expected = self._next_hash(
            direction=direction,
            message_type=core["type"],
            core=core,
        )
        if message["transcript_sha256"] != expected:
            raise FormalRouteCProcessError("process transcript hash is invalid")
        self.next_sequence += 1
        self.sha256 = expected
        return core


def _observation_envelope(
    observation: RobotObservation,
    transcript: _Transcript,
) -> tuple[bytes, dict[str, Any]]:
    wire = serialize_policy_observation(observation)
    commitments = sensor_content_commitments(observation)
    core = {"type": "observation", "commitments": commitments}
    chain = transcript.wrap(core, direction="parent_to_child")
    envelope = {**chain, "observation": wire}
    try:
        payload = pickle.dumps(envelope, protocol=5)
    except Exception as exc:
        raise FormalRouteCProcessError(f"cannot encode observation: {exc}") from exc
    return payload, commitments


def _decode_observation(
    payload: bytes,
    transcript: _Transcript,
) -> tuple[RobotObservation, dict[str, Any]]:
    try:
        value = pickle.loads(payload)
    except Exception as exc:
        raise FormalRouteCProcessError(
            f"cannot decode trusted-parent observation: {type(exc).__name__}: {exc}"
        ) from exc
    if type(value) is not dict or "observation" not in value:
        raise FormalRouteCProcessError("observation envelope is invalid")
    wire = value.pop("observation")
    core = transcript.unwrap(value, direction="parent_to_child")
    if set(core) != {"type", "commitments"} or core["type"] != "observation":
        raise FormalRouteCProcessError("observation chain fields are invalid")
    try:
        observation = deserialize_policy_observation(wire)
    except PolicyIPCValidationError as exc:
        raise FormalRouteCProcessError(f"invalid parent observation: {exc}") from exc
    commitments = sensor_content_commitments(observation)
    if core["commitments"] != commitments:
        raise FormalRouteCProcessError("observation sensor commitment is invalid")
    return observation, commitments


def _send_observation(
    channel: BoundedFrameSocket,
    observation: RobotObservation,
    transcript: _Transcript,
    *,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    payload, commitments = _observation_envelope(observation, transcript)
    try:
        channel.send_observation_bytes(payload, timeout_s=timeout_s)
    except FormalProcessTransportError as exc:
        raise FormalRouteCProcessError(str(exc)) from exc
    return commitments


def _recv_observation(
    channel: BoundedFrameSocket,
    transcript: _Transcript,
    *,
    timeout_s: float | None = None,
) -> tuple[RobotObservation, dict[str, Any]]:
    try:
        payload = channel.recv_observation_bytes(timeout_s=timeout_s)
    except FormalProcessTransportError as exc:
        raise FormalRouteCProcessError(str(exc)) from exc
    return _decode_observation(payload, transcript)


def _send_chained_json(
    channel: BoundedFrameSocket,
    transcript: _Transcript,
    core: Mapping[str, Any],
    *,
    direction: str,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    message = transcript.wrap(core, direction=direction)
    try:
        channel.send_json(message, timeout_s=timeout_s)
    except FormalProcessTransportError as exc:
        raise FormalRouteCProcessError(str(exc)) from exc
    return message


def _recv_chained_json(
    channel: BoundedFrameSocket,
    transcript: _Transcript,
    *,
    direction: str,
    timeout_s: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        message = channel.recv_json(timeout_s=timeout_s)
    except FormalProcessTransportError as exc:
        raise FormalRouteCProcessError(str(exc)) from exc
    core = transcript.unwrap(message, direction=direction)
    return core, message


def _mount_status(destination: str) -> dict[str, Any]:
    matches: list[tuple[int, list[str]]] = []
    for raw in Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
        fields = raw.split()
        if len(fields) < 10 or "-" not in fields:
            continue
        mount_point = fields[4].replace("\\040", " ")
        if mount_point == destination:
            matches.append((len(fields[3]), fields))
    if not matches:
        raise FormalRouteCProcessError(f"expected sandbox mount is absent: {destination}")
    fields = max(matches, key=lambda item: item[0])[1]
    separator = fields.index("-")
    # The bind mount's effective access mode is in the mount-options field.
    # Superblock options may remain ``rw`` even when this particular bind is
    # remounted read-only.
    options = set(fields[5].split(","))
    return {
        "destination": destination,
        "filesystem": fields[separator + 1],
        "read_only": "ro" in options and "rw" not in options,
    }


def _proc_starttime(pid: int | str) -> int:
    payload = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    close = payload.rfind(")")
    fields_after_command = payload[close + 2 :].split()
    # Field 22 overall; the suffix starts at field 3.
    value = int(fields_after_command[19])
    if value <= 0:
        raise FormalRouteCProcessError("process starttime is invalid")
    return value


def _namespace_inodes(pid: int | str = "self") -> dict[str, int]:
    return {
        name: int(Path(f"/proc/{pid}/ns/{name}").stat().st_ino)
        for name in _NAMESPACE_NAMES
    }


def _open_fd_manifest(protocol_fd: int, *, accelerator: bool = False) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for entry in sorted(Path("/proc/self/fd").iterdir(), key=lambda path: int(path.name)):
        fd = int(entry.name)
        try:
            target = os.readlink(entry)
        except FileNotFoundError:
            continue
        kind = "socket" if target.startswith("socket:[") else "null" if target == "/dev/null" else "other"
        if accelerator and (
            target == "/dev/dxg" or target.startswith("/usr/lib/wsl/")
            or target.startswith("anon_inode:[event") or target.startswith("pipe:[")
        ):
            kind = "accelerator"
        records.append({"fd": fd, "kind": kind})
    expected = [
        {"fd": 0, "kind": "null"},
        {"fd": 1, "kind": "null"},
        {"fd": 2, "kind": "null"},
        {"fd": protocol_fd, "kind": "socket"},
    ]
    if [item for item in records if item["kind"] != "accelerator"] != expected:
        raise FormalRouteCProcessError(f"child has unexpected open descriptors: {records}")
    return records


def _network_interfaces() -> list[str]:
    result: list[str] = []
    for line in Path("/proc/net/dev").read_text(encoding="utf-8").splitlines()[2:]:
        if ":" in line:
            result.append(line.split(":", 1)[0].strip())
    return sorted(result)


def _runtime_receipt(
    *,
    context: PerceptionFactoryContext,
    protocol_fd: int,
    worker_nonce: str,
    bundle_build_count: int,
    model_load_count: int,
    policy_runner: str,
) -> dict[str, Any]:
    environment = dict(os.environ)
    accelerator = context.device.startswith("cuda")
    expected_environment = fixed_child_environment(accelerator=accelerator)
    if environment != expected_environment:
        extra = sorted(set(environment) - set(expected_environment))
        missing = sorted(set(expected_environment) - set(environment))
        raise FormalRouteCProcessError(
            f"child environment differs from fixed contract: extra={extra}, missing={missing}"
        )
    source_sha, source_count = tree_manifest_sha256(POLICY_SOURCE_ROOT)
    gallery_sha, gallery_count = tree_manifest_sha256(SANDBOX_ASSET_ROOT)

    model_root: Path | None = None
    if context.perception_backend in {"auto", "grounding-dino"}:
        if context.perception_model is None:
            from libero_system.perception.grounding_dino import DEFAULT_GROUNDING_DINO_TINY

            model_root = DEFAULT_GROUNDING_DINO_TINY
        else:
            model_root = context.perception_model
        for candidate in (model_root, *model_root.parents):
            if (candidate / "blobs").is_dir() and (candidate / "snapshots").is_dir():
                model_root = candidate
                break
    model_sha = None
    model_count = 0
    if model_root is not None:
        if model_root.is_dir():
            model_sha, model_count = tree_manifest_sha256(model_root)
        else:
            model_sha, model_count = sha256_file(model_root), 1

    expected_mounts = [
        str(POLICY_SOURCE_ROOT),
        str(SANDBOX_ASSET_ROOT),
        str(SANDBOX_RUNTIME_ROOT),
        "/tmp",
        "/proc",
        "/dev/null",
        "/dev/urandom",
        "/etc/passwd",
        "/etc/group",
        *SYSTEM_RUNTIME_FILES,
    ]
    if model_root is not None:
        expected_mounts.append(str(model_root))
    if accelerator:
        expected_mounts.extend(("/usr/lib/wsl", "/dev/dxg"))
    site_packages = SANDBOX_RUNTIME_ROOT / "lib/python3.12/site-packages"
    expected_mounts.extend(
        str(site_packages / package)
        for package in MASKED_SITE_PACKAGES
        if (site_packages / package).exists()
    )
    mount_status = [_mount_status(path) for path in expected_mounts]
    by_destination = {item["destination"]: item for item in mount_status}
    for readonly_path in (
        str(POLICY_SOURCE_ROOT),
        str(SANDBOX_ASSET_ROOT),
        str(SANDBOX_RUNTIME_ROOT),
    ):
        if not by_destination[readonly_path]["read_only"]:
            raise FormalRouteCProcessError(f"sandbox mount is not read-only: {readonly_path}")
    if model_root is not None and not by_destination[str(model_root)]["read_only"]:
        raise FormalRouteCProcessError("model repository mount is not read-only")
    if any(not by_destination[path]["read_only"] for path in SYSTEM_RUNTIME_FILES):
        raise FormalRouteCProcessError("fixed system runtime file is not read-only")
    system_runtime_files = [
        {
            "destination": path,
            "sha256": sha256_file(path),
            "read_only": by_destination[path]["read_only"],
        }
        for path in SYSTEM_RUNTIME_FILES
    ]
    if accelerator:
        if not by_destination["/usr/lib/wsl"]["read_only"]:
            raise FormalRouteCProcessError("GPU driver runtime must be read-only")
        system_runtime_files.extend({
            "destination": path, "sha256": sha256_file(path), "read_only": True,
        } for path in _accelerator_runtime_paths())
    identity_files = [
        {
            "destination": path,
            "sha256": sha256_file(path),
            "read_only": by_destination[path]["read_only"],
        }
        for path in ("/etc/passwd", "/etc/group")
    ]
    if any(not item["read_only"] for item in identity_files):
        raise FormalRouteCProcessError("fixed identity file is not read-only")

    proc_pids = sorted(
        name for name in os.listdir("/proc") if name.isdigit()
    )
    interfaces = _network_interfaces()
    status = Path("/proc/self/status").read_text(encoding="utf-8")
    cap_eff = next(
        line.split(":", 1)[1].strip()
        for line in status.splitlines()
        if line.startswith("CapEff:")
    )
    forbidden = {path: Path(path).exists() for path in _FORBIDDEN_VISIBLE_PATHS}
    loaded_policy_modules = sorted(
        name for name in sys.modules if name.startswith("libero_system")
    )
    if any(
        name in loaded_policy_modules
        for name in (
            "libero_system.common.env_adapter",
            "libero_system.common.runner",
            "libero_system.integration.evaluator",
            "libero_system.integration.results",
            "libero_system.integration.video",
        )
    ):
        raise FormalRouteCProcessError("child eagerly loaded evaluator-side modules")
    if any(forbidden.values()):
        raise FormalRouteCProcessError("child can access a forbidden policy path")
    if os.getpid() != 1 or os.getppid() != 0 or proc_pids != ["1"]:
        raise FormalRouteCProcessError("child is not isolated as PID 1 with private proc")
    if interfaces != ["lo"]:
        raise FormalRouteCProcessError("child network namespace has non-loopback interfaces")
    if cap_eff != "0000000000000000":
        raise FormalRouteCProcessError("child retained Linux capabilities")
    if socket.gethostname() != SANDBOX_HOSTNAME:
        raise FormalRouteCProcessError("child UTS namespace hostname is invalid")

    return {
        "schema": PROCESS_RUNTIME_RECEIPT_SCHEMA,
        "worker_nonce": worker_nonce,
        "inner_pid": os.getpid(),
        "inner_ppid": os.getppid(),
        "proc_starttime_ticks": _proc_starttime("self"),
        "namespace_inodes": _namespace_inodes(),
        "environment_values": environment,
        "environment_sha256": environment_sha256(environment),
        "source_sha256": source_sha,
        "source_file_count": source_count,
        "gallery_sha256": gallery_sha,
        "gallery_file_count": gallery_count,
        "model_sha256": model_sha,
        "model_file_count": model_count,
        "bundle_build_count": bundle_build_count,
        "model_load_count": model_load_count,
        "policy_runner": policy_runner,
        "mount_status": mount_status,
        "open_fds": _open_fd_manifest(protocol_fd, accelerator=accelerator),
        "private_proc_pids": proc_pids,
        "network_interfaces": interfaces,
        "effective_capabilities": cap_eff,
        "hostname": socket.gethostname(),
        "forbidden_path_access": forbidden,
        "loaded_policy_modules": loaded_policy_modules,
        "system_runtime_files": system_runtime_files,
        "fixed_identity_files": identity_files,
    }


def _validate_sensor_commitment(value: object) -> dict[str, Any]:
    keys = {
        "schema",
        "camera_capture_ids",
        "proprioception_sha256",
        "snapshot_sha256",
    }
    if type(value) is not dict or set(value) != keys:
        raise FormalRouteCProcessError("episode sensor commitment has invalid fields")
    cameras = value["camera_capture_ids"]
    if (
        value["schema"] != SENSOR_COMMITMENT_SCHEMA
        or type(cameras) is not dict
        or set(cameras) != {"agentview", "wrist"}
        or not all(_hash_valid(item) for item in cameras.values())
        or not _hash_valid(value["proprioception_sha256"])
        or not _hash_valid(value["snapshot_sha256"])
    ):
        raise FormalRouteCProcessError("episode sensor commitment is invalid")
    return value


_EMPTY_EPISODE_RUNTIME_STATE: dict[str, int | bool] = {
    "contact_trace_entries": 0,
    "controller_grasp_events": 0,
    "controller_pending_grasp": False,
    "observer_obstacle_filter_entries": 0,
    "observer_selector_diagnostics": 0,
    "resolver_resolution_history": 0,
    "robot_grasp_checks": 0,
    "robot_phase_trace_entries": 0,
    "robot_steps_executed": 0,
}


def _validate_episode_runtime_receipt(value: object) -> dict[str, Any]:
    keys = {
        "schema",
        "assembly_nonce",
        "assembly_count",
        "component_count",
        "component_identity_sha256",
        "initial_state",
        "initial_state_sha256",
        "final_action_count",
    }
    if type(value) is not dict or set(value) != keys:
        raise FormalRouteCProcessError(
            "Route C episode runtime receipt has invalid fields"
        )
    initial_state = value["initial_state"]
    if (
        value["schema"] != EPISODE_RUNTIME_SCHEMA
        or not _hash_valid(value["assembly_nonce"])
        or type(value["assembly_count"]) is not int
        or value["assembly_count"] < 1
        or type(value["component_count"]) is not int
        or value["component_count"] != 16
        or not _hash_valid(value["component_identity_sha256"])
        or type(initial_state) is not dict
        or set(initial_state) != set(_EMPTY_EPISODE_RUNTIME_STATE)
        or any(
            type(initial_state[name]) is not type(expected)
            or initial_state[name] != expected
            for name, expected in _EMPTY_EPISODE_RUNTIME_STATE.items()
        )
        or not _hash_valid(value["initial_state_sha256"])
        or value["initial_state_sha256"]
        != _json_sha256(
            initial_state,
            b"libero-route-c-runtime-initial-state.v1\0",
        )
        or type(value["final_action_count"]) is not int
        or not 0 <= value["final_action_count"] <= MAX_POLICY_STEPS
    ):
        raise FormalRouteCProcessError(
            "Route C episode runtime receipt values are invalid"
        )
    return value


def _validate_episode_receipt(value: object) -> dict[str, Any]:
    keys = {
        "schema",
        "worker_nonce",
        "episode_nonce",
        "episode_index",
        "message_count",
        "observation_count",
        "action_count",
        "bundle_build_count",
        "runtime_assembly_nonce",
        "runtime_assembly_count",
        "runtime_component_count",
        "runtime_component_identity_sha256",
        "runtime_initial_state_sha256",
        "transcript_before_result_sha256",
        "final_transcript_sha256",
        "result_core_sha256",
        "observation_commitments",
        "action_commitments",
    }
    if type(value) is not dict or set(value) != keys:
        raise FormalRouteCProcessError("Route C episode receipt has invalid fields")
    integer_fields = (
        "episode_index",
        "message_count",
        "observation_count",
        "action_count",
        "bundle_build_count",
        "runtime_assembly_count",
        "runtime_component_count",
    )
    if (
        value["schema"] != PROCESS_EPISODE_RECEIPT_SCHEMA
        or not _hash_valid(value["worker_nonce"])
        or not _hash_valid(value["episode_nonce"])
        or not _hash_valid(value["transcript_before_result_sha256"])
        or not _hash_valid(value["final_transcript_sha256"])
        or not _hash_valid(value["result_core_sha256"])
        or any(type(value[name]) is not int for name in integer_fields)
        or not 1 <= value["episode_index"]
        or not 0 <= value["action_count"] <= MAX_POLICY_STEPS
        or value["observation_count"] != value["action_count"] + 1
        or value["bundle_build_count"] != 1
        or not _hash_valid(value["runtime_assembly_nonce"])
        or value["runtime_assembly_count"] != value["episode_index"]
        or value["runtime_component_count"] != 16
        or not _hash_valid(value["runtime_component_identity_sha256"])
        or not _hash_valid(value["runtime_initial_state_sha256"])
        or value["runtime_initial_state_sha256"]
        != _json_sha256(
            _EMPTY_EPISODE_RUNTIME_STATE,
            b"libero-route-c-runtime-initial-state.v1\0",
        )
        or value["message_count"] != 2 * value["action_count"] + 3
    ):
        raise FormalRouteCProcessError("Route C episode receipt values are invalid")
    observations = value["observation_commitments"]
    actions = value["action_commitments"]
    if (
        type(observations) is not list
        or len(observations) != value["observation_count"]
        or type(actions) is not list
        or len(actions) != value["action_count"]
        or any(not _hash_valid(item) for item in actions)
    ):
        raise FormalRouteCProcessError("Route C episode transcript lists are invalid")
    for item in observations:
        _validate_sensor_commitment(item)
    return value


def _validate_runtime_receipt(value: object) -> dict[str, Any]:
    keys = {
        "schema",
        "worker_nonce",
        "inner_pid",
        "inner_ppid",
        "proc_starttime_ticks",
        "namespace_inodes",
        "environment_values",
        "environment_sha256",
        "source_sha256",
        "source_file_count",
        "gallery_sha256",
        "gallery_file_count",
        "model_sha256",
        "model_file_count",
        "bundle_build_count",
        "model_load_count",
        "policy_runner",
        "mount_status",
        "open_fds",
        "private_proc_pids",
        "network_interfaces",
        "effective_capabilities",
        "hostname",
        "forbidden_path_access",
        "loaded_policy_modules",
        "system_runtime_files",
        "fixed_identity_files",
    }
    if type(value) is not dict or set(value) != keys:
        raise FormalRouteCProcessError("Route C runtime receipt has invalid fields")
    accelerator = _uses_accelerator_environment(value.get("environment_values"))
    environment = fixed_child_environment(accelerator=accelerator)
    namespaces = value["namespace_inodes"]
    if (
        value["schema"] != PROCESS_RUNTIME_RECEIPT_SCHEMA
        or not _hash_valid(value["worker_nonce"])
        or value["inner_pid"] != 1
        or value["inner_ppid"] != 0
        or type(value["proc_starttime_ticks"]) is not int
        or value["proc_starttime_ticks"] <= 0
        or type(namespaces) is not dict
        or set(namespaces) != set(_NAMESPACE_NAMES)
        or any(type(item) is not int or item <= 0 for item in namespaces.values())
        or value["environment_values"] != environment
        or value["environment_sha256"] != environment_sha256(environment)
        or not _hash_valid(value["source_sha256"])
        or type(value["source_file_count"]) is not int
        or value["source_file_count"] != len(POLICY_SOURCE_FILES)
        or not _hash_valid(value["gallery_sha256"])
        or type(value["gallery_file_count"]) is not int
        or value["gallery_file_count"] < 1
        or (
            value["model_sha256"] is not None
            and not _hash_valid(value["model_sha256"])
        )
        or type(value["model_file_count"]) is not int
        or value["model_file_count"] < 0
        or value["bundle_build_count"] != 1
        or type(value["model_load_count"]) is not int
        or value["model_load_count"] not in (0, 1)
        or value["policy_runner"]
        not in {"production_route_c_policy_episode", "embedded_test_fixture"}
        or value["private_proc_pids"] != ["1"]
        or value["network_interfaces"] != ["lo"]
        or value["effective_capabilities"] != "0000000000000000"
        or value["hostname"] != SANDBOX_HOSTNAME
    ):
        raise FormalRouteCProcessError("Route C runtime receipt values are invalid")
    if type(value["forbidden_path_access"]) is not dict or set(
        value["forbidden_path_access"]
    ) != set(_FORBIDDEN_VISIBLE_PATHS) or any(
        type(item) is not bool or item
        for item in value["forbidden_path_access"].values()
    ):
        raise FormalRouteCProcessError("Route C forbidden-path receipt is invalid")
    if type(value["mount_status"]) is not list or len(value["mount_status"]) < 6:
        raise FormalRouteCProcessError("Route C mount receipt is invalid")
    mount_destinations: set[str] = set()
    for item in value["mount_status"]:
        if (
            type(item) is not dict
            or set(item) != {"destination", "filesystem", "read_only"}
            or type(item["destination"]) is not str
            or not item["destination"].startswith("/")
            or type(item["filesystem"]) is not str
            or not item["filesystem"]
            or type(item["read_only"]) is not bool
            or item["destination"] in mount_destinations
        ):
            raise FormalRouteCProcessError("Route C mount receipt item is invalid")
        mount_destinations.add(item["destination"])
    required_mounts = {
        "/policy",
        str(SANDBOX_ASSET_ROOT),
        "/runtime",
        "/tmp",
        "/proc",
        "/dev/null",
        "/dev/urandom",
        "/etc/passwd",
        "/etc/group",
        *SYSTEM_RUNTIME_FILES,
    }
    if accelerator:
        required_mounts.update(("/usr/lib/wsl", "/dev/dxg"))
    if not required_mounts <= mount_destinations:
        raise FormalRouteCProcessError("Route C required mount receipt is incomplete")
    readonly_destinations = {
        item["destination"] for item in value["mount_status"] if item["read_only"]
    }
    if not {"/policy", str(SANDBOX_ASSET_ROOT), "/runtime"} <= readonly_destinations:
        raise FormalRouteCProcessError("Route C policy mounts are not read-only")
    if type(value["open_fds"]) is not list:
        raise FormalRouteCProcessError("Route C descriptor receipt is invalid")
    all_fds = value["open_fds"]
    if any(type(item) is not dict or set(item) != {"fd", "kind"}
           or type(item["fd"]) is not int or item["fd"] < 0 for item in all_fds):
        raise FormalRouteCProcessError("Route C descriptor receipt is invalid")
    if len({item["fd"] for item in all_fds}) != len(all_fds):
        raise FormalRouteCProcessError("Route C descriptor numbers are repeated")
    core_fds = [item for item in all_fds if not (accelerator and item["kind"] == "accelerator")]
    if [item.get("kind") for item in core_fds] != [
        "null",
        "null",
        "null",
        "socket",
    ]:
        raise FormalRouteCProcessError("Route C descriptor kinds are invalid")
    if (
        [item.get("fd") for item in core_fds[:3]] != [0, 1, 2]
        or type(core_fds[3].get("fd")) is not int
        or core_fds[3]["fd"] < 3
        or any(
            type(item) is not dict or set(item) != {"fd", "kind"}
            for item in value["open_fds"]
        )
    ):
        raise FormalRouteCProcessError("Route C descriptor numbers are invalid")
    if (value["model_sha256"] is None) != (value["model_file_count"] == 0):
        raise FormalRouteCProcessError("Route C model commitment/count disagree")
    if value["model_load_count"] != (1 if value["model_sha256"] else 0):
        raise FormalRouteCProcessError("Route C model load receipt disagrees")
    modules = value["loaded_policy_modules"]
    if type(modules) is not list or modules != sorted(set(modules)):
        raise FormalRouteCProcessError("Route C loaded-module receipt is invalid")
    system_files = value["system_runtime_files"]
    if (
        type(system_files) is not list
        or (len(system_files) < len(SYSTEM_RUNTIME_FILES) if accelerator
            else len(system_files) != len(SYSTEM_RUNTIME_FILES))
        or [item.get("destination") for item in system_files[:len(SYSTEM_RUNTIME_FILES)]]
        != list(SYSTEM_RUNTIME_FILES)
        or any(
            type(item) is not dict
            or set(item) != {"destination", "sha256", "read_only"}
            or not _hash_valid(item["sha256"])
            or item["read_only"] is not True
            for item in system_files
        )
    ):
        raise FormalRouteCProcessError("Route C system runtime receipt is invalid")
    if any(not item["destination"].startswith("/usr/lib/wsl/")
           for item in system_files[len(SYSTEM_RUNTIME_FILES):]):
        raise FormalRouteCProcessError("GPU driver receipt has an invalid path")
    identity_files = value["fixed_identity_files"]
    if (
        type(identity_files) is not list
        or [item.get("destination") for item in identity_files]
        != ["/etc/passwd", "/etc/group"]
        or any(
            type(item) is not dict
            or set(item) != {"destination", "sha256", "read_only"}
            or not _hash_valid(item["sha256"])
            or item["read_only"] is not True
            for item in identity_files
        )
    ):
        raise FormalRouteCProcessError("Route C fixed identity receipt is invalid")
    return value


def _synthetic_runtime_receipt() -> dict[str, Any]:
    environment = fixed_child_environment()
    return {
        "schema": PROCESS_RUNTIME_RECEIPT_SCHEMA,
        "worker_nonce": "0" * 64,
        "inner_pid": 1,
        "inner_ppid": 0,
        "proc_starttime_ticks": 1,
        "namespace_inodes": {name: 1 for name in _NAMESPACE_NAMES},
        "environment_values": environment,
        "environment_sha256": environment_sha256(environment),
        "source_sha256": "1" * 64,
        "source_file_count": len(POLICY_SOURCE_FILES),
        "gallery_sha256": "2" * 64,
        "gallery_file_count": 1,
        "model_sha256": None,
        "model_file_count": 0,
        "bundle_build_count": 1,
        "model_load_count": 0,
        "policy_runner": "production_route_c_policy_episode",
        "mount_status": [
            {"destination": path, "filesystem": "bind", "read_only": True}
            for path in (
                "/policy",
                str(SANDBOX_ASSET_ROOT),
                "/runtime",
                "/tmp",
                "/proc",
                "/dev/null",
                "/dev/urandom",
                "/etc/passwd",
                "/etc/group",
                *SYSTEM_RUNTIME_FILES,
            )
        ],
        "open_fds": [
            {"fd": 0, "kind": "null"},
            {"fd": 1, "kind": "null"},
            {"fd": 2, "kind": "null"},
            {"fd": 3, "kind": "socket"},
        ],
        "private_proc_pids": ["1"],
        "network_interfaces": ["lo"],
        "effective_capabilities": "0000000000000000",
        "hostname": SANDBOX_HOSTNAME,
        "forbidden_path_access": {path: False for path in _FORBIDDEN_VISIBLE_PATHS},
        "loaded_policy_modules": ["libero_system"],
        "system_runtime_files": [
            {
                "destination": path,
                "sha256": "a" * 64,
                "read_only": True,
            }
            for path in SYSTEM_RUNTIME_FILES
        ],
        "fixed_identity_files": [
            {
                "destination": path,
                "sha256": "b" * 64,
                "read_only": True,
            }
            for path in ("/etc/passwd", "/etc/group")
        ],
    }


def _synthetic_live_process(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "verified": True,
        "launcher_pid": 1,
        "payload_host_pid": 1,
        "proc_starttime_ticks": receipt["proc_starttime_ticks"],
        "namespace_inodes": dict(receipt["namespace_inodes"]),
    }


def _synthetic_episode_receipt(
    runtime: Mapping[str, Any],
    action_count: int,
) -> dict[str, Any]:
    if type(action_count) is not int or not 0 <= action_count <= MAX_POLICY_STEPS:
        raise ValueError("synthetic episode action count is invalid")
    sensor = {
        "schema": SENSOR_COMMITMENT_SCHEMA,
        "camera_capture_ids": {
            "agentview": "6" * 64,
            "wrist": "7" * 64,
        },
        "proprioception_sha256": "8" * 64,
        "snapshot_sha256": "9" * 64,
    }
    initial_state_sha256 = _json_sha256(
        _EMPTY_EPISODE_RUNTIME_STATE,
        b"libero-route-c-runtime-initial-state.v1\0",
    )
    return {
        "schema": PROCESS_EPISODE_RECEIPT_SCHEMA,
        "worker_nonce": runtime["worker_nonce"],
        "episode_nonce": "a" * 64,
        "episode_index": 1,
        "message_count": 2 * action_count + 3,
        "observation_count": action_count + 1,
        "action_count": action_count,
        "bundle_build_count": 1,
        "runtime_assembly_nonce": "f" * 64,
        "runtime_assembly_count": 1,
        "runtime_component_count": 16,
        "runtime_component_identity_sha256": "0" * 64,
        "runtime_initial_state_sha256": initial_state_sha256,
        "transcript_before_result_sha256": "b" * 64,
        "final_transcript_sha256": "c" * 64,
        "result_core_sha256": "d" * 64,
        "observation_commitments": [deepcopy(sensor) for _ in range(action_count + 1)],
        "action_commitments": ["e" * 64 for _ in range(action_count)],
    }


def validate_process_boundary_attestation(value: object) -> dict[str, Any]:
    expected_keys = {
        "schema",
        "transport",
        "policy_runtime_module",
        "policy_runner",
        "child_startup_payload_fields",
        "child_environment_fields",
        "child_environment_values",
        "child_environment_sha256",
        "child_working_directory",
        "child_command_shape",
        "observation_wire_schema",
        "action_wire_schema",
        "parent_to_child_observation_encoding",
        "child_to_parent_encoding",
        "frame_limits",
        "freshness_authority",
        "legacy_timestamp_s",
        "policy_input_fields",
        "policy_output_fields",
        "forwarded_object_capabilities",
        "forwarded_evaluator_signal_fields",
        "evaluator_clock_contract",
        "sandbox_launcher",
        "namespace_contract",
        "policy_source_projection",
        "mount_commitments",
        "masked_site_packages",
        "runtime_receipt",
        "live_process",
        "latest_episode_receipt",
    }
    if type(value) is not dict or set(value) != expected_keys:
        raise FormalRouteCProcessError(
            "Route C process-boundary attestation has invalid fields"
        )
    accelerator = _uses_accelerator_environment(value.get("child_environment_values"))
    environment = fixed_child_environment(accelerator=accelerator)
    exact_values: dict[str, object] = {
        "schema": PROCESS_BOUNDARY_ATTESTATION_SCHEMA,
        "transport": "bwrap_namespaced_bounded_socket_rpc",
        "policy_runtime_module": "libero_system.integration.route_c_policy_runtime",
        "policy_runner": "production_route_c_policy_episode",
        "child_startup_payload_fields": sorted(
            {"schema", "device", "perception_backend", "perception_model", "image_size"}
        ),
        "child_environment_fields": sorted(environment),
        "child_environment_values": environment,
        "child_environment_sha256": environment_sha256(environment),
        "child_working_directory": "/tmp",
        "child_command_shape": [
            "/usr/bin/bwrap",
            "isolated_namespaces",
            "fixed_read_only_projection",
            "python_executable",
            "-I",
            "-S",
            "-B",
            "-X",
            "pycache_prefix=/dev/null",
            "controlled_bootstrap",
            "opaque_socket_fd",
            "production_runner_literal_0",
        ],
        "observation_wire_schema": OBSERVATION_WIRE_SCHEMA,
        "action_wire_schema": ACTION_WIRE_SCHEMA,
        "parent_to_child_observation_encoding": "bounded_validated_pickle_from_trusted_parent",
        "child_to_parent_encoding": "bounded_strict_json_only",
        "frame_limits": {"json_bytes": 2 * 1024 * 1024, "observation_bytes": 8 * 1024 * 1024},
        "freshness_authority": "both_camera_rgbd_calibration_sha256_separate_from_proprioception",
        "legacy_timestamp_s": 0.0,
        "policy_input_fields": [
            "instruction",
            "fixed_step_budget",
            "agentview_rgbd_and_calibration",
            "wrist_rgbd_and_calibration",
            "proprioception",
        ],
        "policy_output_fields": ["osc_action", "policy_result"],
        "forwarded_object_capabilities": {
            "environment": False,
            "evaluator_score": False,
            "video_recorder": False,
            "benchmark_metadata": False,
        },
        "forwarded_evaluator_signal_fields": [],
        "evaluator_clock_contract": "no_evaluator_derived_clock_or_step_input",
        "namespace_contract": {
            "user": "isolated",
            "pid": "isolated_pid1_ppid0_private_proc",
            "mount": "isolated_fixed_projection",
            "network": "isolated_loopback_only",
            "ipc": "isolated",
            "uts": "isolated_fixed_hostname",
            "nested_user_namespaces": "disabled_and_asserted",
            "capabilities": "all_dropped",
        },
        "masked_site_packages": list(MASKED_SITE_PACKAGES),
    }
    for name, expected in exact_values.items():
        if value.get(name) != expected or type(value.get(name)) is not type(expected):
            raise FormalRouteCProcessError(
                f"Route C process-boundary attestation {name} is invalid"
            )
    launcher = value["sandbox_launcher"]
    if (
        type(launcher) is not dict
        or set(launcher) != {"path", "sha256", "version"}
        or launcher["path"] != str(BWRAP_PATH)
        or not _hash_valid(launcher["sha256"])
        or type(launcher["version"]) is not str
        or not launcher["version"].startswith("bubblewrap ")
    ):
        raise FormalRouteCProcessError("Route C sandbox launcher attestation is invalid")
    projection = value["policy_source_projection"]
    if (
        type(projection) is not dict
        or set(projection) != {"sha256", "file_count", "included_files", "excluded_files", "clock_source_gate_sha256"}
        or not _hash_valid(projection["sha256"])
        or projection["file_count"] != len(POLICY_SOURCE_FILES)
        or projection["included_files"] != list(POLICY_SOURCE_FILES)
        or projection["excluded_files"] != list(EXCLUDED_POLICY_FILES)
        or not _hash_valid(projection["clock_source_gate_sha256"])
    ):
        raise FormalRouteCProcessError("Route C policy projection attestation is invalid")
    if type(value["mount_commitments"]) is not dict:
        raise FormalRouteCProcessError("Route C mount commitments are invalid")
    runtime = _validate_runtime_receipt(value["runtime_receipt"])
    if runtime["policy_runner"] != "production_route_c_policy_episode":
        raise FormalRouteCProcessError(
            "formal attestation rejects the embedded test policy runner"
        )
    mounts = value["mount_commitments"]
    expected_mount_fields = {
        "policy_source",
        "gallery_assets",
        "model_repository",
        "conda_runtime",
        "private_tmp",
        "private_proc",
        "device_nodes",
        "system_runtime_files",
        "fixed_identity_files",
    }
    if accelerator:
        expected_mount_fields.add("accelerator_runtime")
        if mounts.get("accelerator_runtime") != {
            "destination": "/usr/lib/wsl", "read_only": True, "device": "/dev/dxg",
        }:
            raise FormalRouteCProcessError("Route C GPU mount commitment is invalid")
        if runtime["environment_values"] != environment:
            raise FormalRouteCProcessError("Route C GPU environment differs from runtime")
    if set(mounts) != expected_mount_fields:
        raise FormalRouteCProcessError("Route C mount commitment fields are invalid")

    def committed_tree(
        item: object,
        *,
        destination: str,
        sha256: str,
        file_count: int,
    ) -> bool:
        return bool(
            type(item) is dict
            and set(item) == {"destination", "sha256", "file_count", "read_only"}
            and item["destination"] == destination
            and item["sha256"] == sha256
            and item["file_count"] == file_count
            and item["read_only"] is True
        )

    if (
        projection["sha256"] != runtime["source_sha256"]
        or not committed_tree(
            mounts["policy_source"],
            destination="/policy",
            sha256=runtime["source_sha256"],
            file_count=runtime["source_file_count"],
        )
        or not committed_tree(
            mounts["gallery_assets"],
            destination=str(SANDBOX_ASSET_ROOT),
            sha256=runtime["gallery_sha256"],
            file_count=runtime["gallery_file_count"],
        )
    ):
        raise FormalRouteCProcessError("Route C source/gallery commitments disagree")
    model_mount = mounts["model_repository"]
    if runtime["model_sha256"] is None:
        if model_mount is not None:
            raise FormalRouteCProcessError("Route C unexpected model mount commitment")
    elif (
        type(model_mount) is not dict
        or set(model_mount) != {"destination", "sha256", "file_count", "read_only"}
        or type(model_mount["destination"]) is not str
        or not model_mount["destination"].startswith("/")
        or model_mount["sha256"] != runtime["model_sha256"]
        or model_mount["file_count"] != runtime["model_file_count"]
        or model_mount["read_only"] is not True
    ):
        raise FormalRouteCProcessError("Route C model mount commitment disagrees")
    conda = mounts["conda_runtime"]
    system_mounts = mounts["system_runtime_files"]
    identity_mounts = mounts["fixed_identity_files"]
    if (
        type(conda) is not dict
        or set(conda) != {"destination", "python_sha256", "read_only"}
        or conda["destination"] != "/runtime"
        or not _hash_valid(conda["python_sha256"])
        or conda["read_only"] is not True
        or mounts["private_tmp"] != {"destination": "/tmp", "kind": "tmpfs"}
        or mounts["private_proc"] != {"destination": "/proc", "kind": "proc"}
        or mounts["device_nodes"]
        != [
            {
                "destination": "/dev/null",
                "kind": "dev-bind",
                "access": "read_write",
            },
            {
                "destination": "/dev/urandom",
                "kind": "dev-bind",
                "access": "read_write",
            },
        ]
        or system_mounts != runtime["system_runtime_files"]
        or identity_mounts != runtime["fixed_identity_files"]
    ):
        raise FormalRouteCProcessError("Route C runtime/private mount commitment is invalid")
    live = value["live_process"]
    if (
        type(live) is not dict
        or set(live) != {"verified", "launcher_pid", "payload_host_pid", "proc_starttime_ticks", "namespace_inodes"}
        or live["verified"] is not True
        or any(type(live[name]) is not int or live[name] < 1 for name in ("launcher_pid", "payload_host_pid", "proc_starttime_ticks"))
        or live["proc_starttime_ticks"] != runtime["proc_starttime_ticks"]
        or live["namespace_inodes"] != runtime["namespace_inodes"]
    ):
        raise FormalRouteCProcessError("Route C live-process attestation is invalid")
    episode = value["latest_episode_receipt"]
    if episode is not None:
        _validate_episode_receipt(episode)
        if episode["worker_nonce"] != runtime["worker_nonce"]:
            raise FormalRouteCProcessError("episode receipt belongs to another worker")
    return _strict_json_copy(value)


def _build_process_boundary_attestation(
    child_environment_fields: list[str] | None = None,
    *,
    projection: SandboxProjection | None = None,
    runtime_receipt: Mapping[str, Any] | None = None,
    live_process: Mapping[str, Any] | None = None,
    latest_episode_receipt: Mapping[str, Any] | None = None,
    synthetic_episode_action_count: int | None = None,
) -> dict[str, Any]:
    """Build the exact attestation; defaults retain lightweight fixture use."""

    environment = dict(runtime_receipt["environment_values"]) if runtime_receipt else fixed_child_environment()
    if child_environment_fields is not None and sorted(child_environment_fields) != sorted(environment):
        raise FormalRouteCProcessError("child environment field list is not the fixed contract")
    runtime = dict(runtime_receipt or _synthetic_runtime_receipt())
    live = dict(live_process or _synthetic_live_process(runtime))
    if latest_episode_receipt is not None and synthetic_episode_action_count is not None:
        raise ValueError("provide only one episode receipt source")
    if synthetic_episode_action_count is not None:
        latest_episode_receipt = _synthetic_episode_receipt(
            runtime,
            synthetic_episode_action_count,
        )
    if projection is None:
        source_sha = "1" * 64
        clock_sha = "3" * 64
        launcher_sha = "4" * 64
        launcher_version = "bubblewrap synthetic-test-fixture"
        mount_commitments: dict[str, Any] = {
            "policy_source": {"destination": "/policy", "sha256": source_sha, "file_count": len(POLICY_SOURCE_FILES), "read_only": True},
            "gallery_assets": {"destination": str(SANDBOX_ASSET_ROOT), "sha256": "2" * 64, "file_count": 1, "read_only": True},
            "model_repository": None,
            "conda_runtime": {"destination": "/runtime", "python_sha256": "5" * 64, "read_only": True},
            "private_tmp": {"destination": "/tmp", "kind": "tmpfs"},
            "private_proc": {"destination": "/proc", "kind": "proc"},
            "device_nodes": [
                {
                    "destination": "/dev/null",
                    "kind": "dev-bind",
                    "access": "read_write",
                },
                {
                    "destination": "/dev/urandom",
                    "kind": "dev-bind",
                    "access": "read_write",
                },
            ],
            "system_runtime_files": deepcopy(runtime["system_runtime_files"]),
            "fixed_identity_files": deepcopy(runtime["fixed_identity_files"]),
        }
    else:
        source_sha = projection.source_sha256
        clock_sha = projection.clock_source_gate_sha256
        launcher_sha = projection.bwrap_sha256
        launcher_version = projection.bwrap_version
        mount_commitments = projection.mount_commitments()
        if projection.accelerator:
            mount_commitments["system_runtime_files"] = deepcopy(runtime["system_runtime_files"])
    return validate_process_boundary_attestation(
        {
            "schema": PROCESS_BOUNDARY_ATTESTATION_SCHEMA,
            "transport": "bwrap_namespaced_bounded_socket_rpc",
            "policy_runtime_module": "libero_system.integration.route_c_policy_runtime",
            "policy_runner": "production_route_c_policy_episode",
            "child_startup_payload_fields": sorted({"schema", "device", "perception_backend", "perception_model", "image_size"}),
            "child_environment_fields": sorted(environment),
            "child_environment_values": environment,
            "child_environment_sha256": environment_sha256(environment),
            "child_working_directory": "/tmp",
            "child_command_shape": ["/usr/bin/bwrap", "isolated_namespaces", "fixed_read_only_projection", "python_executable", "-I", "-S", "-B", "-X", "pycache_prefix=/dev/null", "controlled_bootstrap", "opaque_socket_fd", "production_runner_literal_0"],
            "observation_wire_schema": OBSERVATION_WIRE_SCHEMA,
            "action_wire_schema": ACTION_WIRE_SCHEMA,
            "parent_to_child_observation_encoding": "bounded_validated_pickle_from_trusted_parent",
            "child_to_parent_encoding": "bounded_strict_json_only",
            "frame_limits": {"json_bytes": 2 * 1024 * 1024, "observation_bytes": 8 * 1024 * 1024},
            "freshness_authority": "both_camera_rgbd_calibration_sha256_separate_from_proprioception",
            "legacy_timestamp_s": 0.0,
            "policy_input_fields": ["instruction", "fixed_step_budget", "agentview_rgbd_and_calibration", "wrist_rgbd_and_calibration", "proprioception"],
            "policy_output_fields": ["osc_action", "policy_result"],
            "forwarded_object_capabilities": {"environment": False, "evaluator_score": False, "video_recorder": False, "benchmark_metadata": False},
            "forwarded_evaluator_signal_fields": [],
            "evaluator_clock_contract": "no_evaluator_derived_clock_or_step_input",
            "sandbox_launcher": {"path": str(BWRAP_PATH), "sha256": launcher_sha, "version": launcher_version},
            "namespace_contract": {"user": "isolated", "pid": "isolated_pid1_ppid0_private_proc", "mount": "isolated_fixed_projection", "network": "isolated_loopback_only", "ipc": "isolated", "uts": "isolated_fixed_hostname", "nested_user_namespaces": "disabled_and_asserted", "capabilities": "all_dropped"},
            "policy_source_projection": {"sha256": source_sha, "file_count": len(POLICY_SOURCE_FILES), "included_files": list(POLICY_SOURCE_FILES), "excluded_files": list(EXCLUDED_POLICY_FILES), "clock_source_gate_sha256": clock_sha},
            "mount_commitments": mount_commitments,
            "masked_site_packages": list(MASKED_SITE_PACKAGES),
            "runtime_receipt": runtime,
            "live_process": live,
            "latest_episode_receipt": None if latest_episode_receipt is None else dict(latest_episode_receipt),
        }
    )


class _ChildActionChannel:
    __slots__ = (
        "_channel",
        "_transcript",
        "_current",
        "observation_commitments",
        "action_commitments",
    )

    def __init__(
        self,
        channel: BoundedFrameSocket,
        transcript: _Transcript,
        initial: RobotObservation,
        initial_commitment: dict[str, Any],
    ) -> None:
        self._channel = channel
        self._transcript = transcript
        self._current = initial
        self.observation_commitments = [initial_commitment]
        self.action_commitments: list[str] = []

    def current_observation(self) -> RobotObservation:
        return self._current

    def execute(self, action: OSCAction) -> RobotObservation:
        action_wire = serialize_policy_action(action)
        self.action_commitments.append(
            _json_sha256(action_wire, b"libero-policy-action.v1\0")
        )
        _send_chained_json(
            self._channel,
            self._transcript,
            {"type": "action", "action": action_wire},
            direction="child_to_parent",
        )
        self._current, commitment = _recv_observation(
            self._channel,
            self._transcript,
        )
        self.observation_commitments.append(commitment)
        return self._current


def _child_loop(
    channel: BoundedFrameSocket,
    *,
    protocol_fd: int,
    test_policy: bool,
) -> int:
    try:
        init = channel.recv_json()
    except FormalProcessTransportError as exc:
        raise FormalRouteCProcessError(str(exc)) from exc
    context = _context_from_init(init)
    bundle_build_count = 0
    bundle = build_perception_bundle_from_context(context)
    bundle_build_count += 1
    from libero_system.integration.route_c_policy_runtime import (
        run_route_c_policy_episode,
        run_route_c_policy_test_episode,
    )

    policy_runner = (
        run_route_c_policy_test_episode
        if test_policy
        else run_route_c_policy_episode
    )
    policy_runner_name = (
        "embedded_test_fixture"
        if test_policy
        else "production_route_c_policy_episode"
    )
    # Some binary wheels opportunistically add Qt helper variables while
    # importing.  They are not policy inputs: erase every mutation and restore
    # the complete fixed environment before producing runtime evidence.
    os.environ.clear()
    os.environ.update(fixed_child_environment(accelerator=context.device.startswith("cuda")))
    worker_nonce = secrets.token_hex(32)
    receipt = _runtime_receipt(
        context=context,
        protocol_fd=protocol_fd,
        worker_nonce=worker_nonce,
        bundle_build_count=bundle_build_count,
        model_load_count=1 if bundle.box_detector is not None else 0,
        policy_runner=policy_runner_name,
    )
    channel.send_json(
        {
            "schema": PROCESS_MESSAGE_SCHEMA,
            "type": "ready",
            "runtime_receipt": receipt,
        }
    )
    episode_index = 0
    control_sequence = 0
    used_nonces: set[str] = set()
    used_runtime_nonces: set[str] = set()
    try:
        while True:
            message = channel.recv_json()
            if message.get("type") == "shutdown":
                if message != {
                    "schema": PROCESS_MESSAGE_SCHEMA,
                    "type": "shutdown",
                    "worker_nonce": worker_nonce,
                    "control_sequence": control_sequence,
                }:
                    raise FormalRouteCProcessError("invalid or replayed shutdown message")
                channel.send_json(
                    {
                        "schema": PROCESS_MESSAGE_SCHEMA,
                        "type": "closed",
                        "worker_nonce": worker_nonce,
                        "control_sequence": control_sequence,
                    }
                )
                return 0
            next_index = episode_index + 1
            episode_nonce = message.get("episode_nonce")
            if not _hash_valid(episode_nonce) or episode_nonce in used_nonces:
                raise FormalRouteCProcessError("episode nonce is invalid or replayed")
            transcript = _Transcript(
                worker_nonce=worker_nonce,
                episode_nonce=episode_nonce,
                episode_index=next_index,
            )
            core = transcript.unwrap(message, direction="parent_to_child")
            if (
                set(core) != {"type", "instruction", "step_budget"}
                or core["type"] != "start_episode"
                or type(core["instruction"]) is not str
                or not core["instruction"].strip()
                or len(core["instruction"]) > MAX_INSTRUCTION_CHARS
                or type(core["step_budget"]) is not int
                or not 1 <= core["step_budget"] <= MAX_POLICY_STEPS
            ):
                raise FormalRouteCProcessError("invalid start_episode message")
            used_nonces.add(episode_nonce)
            episode_index = next_index
            initial, initial_commitment = _recv_observation(channel, transcript)
            action_channel = _ChildActionChannel(
                channel,
                transcript,
                initial,
                initial_commitment,
            )
            try:
                result = policy_runner(
                    bundle=bundle,
                    instruction=core["instruction"],
                    current_observation=action_channel.current_observation,
                    execute_action=action_channel.execute,
                    step_budget=core["step_budget"],
                )
                if result.steps_executed != len(action_channel.action_commitments):
                    raise FormalRouteCProcessError(
                        "policy result count differs from serialized actions"
                    )
                if len(action_channel.observation_commitments) != result.steps_executed + 1:
                    raise FormalRouteCProcessError(
                        "policy observation deliveries must equal action count plus one"
                    )
                runtime_episode = _validate_episode_runtime_receipt(
                    result.trace.get("episode_runtime")
                )
                if (
                    runtime_episode["assembly_count"] != episode_index
                    or runtime_episode["assembly_nonce"] in used_runtime_nonces
                    or runtime_episode["final_action_count"]
                    != result.steps_executed
                ):
                    raise FormalRouteCProcessError(
                        "policy episode runtime was stale, reused, or count-mismatched"
                    )
                used_runtime_nonces.add(runtime_episode["assembly_nonce"])
                child_receipt = {
                    "schema": PROCESS_EPISODE_RECEIPT_SCHEMA,
                    "worker_nonce": worker_nonce,
                    "episode_nonce": episode_nonce,
                    "episode_index": episode_index,
                    # Includes start, all observations/actions, and result.
                    "message_count": 2 * result.steps_executed + 3,
                    "observation_count": len(action_channel.observation_commitments),
                    "action_count": len(action_channel.action_commitments),
                    "bundle_build_count": bundle_build_count,
                    "runtime_assembly_nonce": runtime_episode["assembly_nonce"],
                    "runtime_assembly_count": runtime_episode["assembly_count"],
                    "runtime_component_count": runtime_episode["component_count"],
                    "runtime_component_identity_sha256": runtime_episode[
                        "component_identity_sha256"
                    ],
                    "runtime_initial_state_sha256": runtime_episode[
                        "initial_state_sha256"
                    ],
                    "transcript_before_result_sha256": transcript.sha256,
                    "observation_commitments": action_channel.observation_commitments,
                    "action_commitments": action_channel.action_commitments,
                }
                core_result = _result_core(result, child_receipt)
                _send_chained_json(
                    channel,
                    transcript,
                    core_result,
                    direction="child_to_parent",
                )
            except BaseException as exc:
                try:
                    _send_chained_json(
                        channel,
                        transcript,
                        {
                            "type": "error",
                            "exception_type": type(exc).__name__,
                            "detail": str(exc)[:2_000],
                        },
                        direction="child_to_parent",
                    )
                except BaseException:
                    pass
    finally:
        bundle.close()


def _descendant_pids(root_pid: int) -> list[int]:
    result: list[int] = []
    pending = [root_pid]
    seen = {root_pid}
    while pending:
        current = pending.pop()
        children_path = Path(f"/proc/{current}/task/{current}/children")
        try:
            children = [int(item) for item in children_path.read_text().split()]
        except (FileNotFoundError, ProcessLookupError):
            continue
        for child in children:
            if child not in seen:
                seen.add(child)
                result.append(child)
                pending.append(child)
    return result


def _verify_live_child(
    process: subprocess.Popen[bytes],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_runtime_receipt(receipt)
    if process.poll() is not None:
        raise FormalRouteCProcessError("sandbox child exited before live verification")
    candidates = [process.pid, *_descendant_pids(process.pid)]
    matches: list[int] = []
    for pid in candidates:
        try:
            if (
                _proc_starttime(pid) == receipt["proc_starttime_ticks"]
                and _namespace_inodes(pid) == receipt["namespace_inodes"]
            ):
                matches.append(pid)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    if len(matches) != 1:
        raise FormalRouteCProcessError(
            f"runtime receipt does not identify one live sandbox payload: {matches}"
        )
    return {
        "verified": True,
        "launcher_pid": process.pid,
        "payload_host_pid": matches[0],
        "proc_starttime_ticks": receipt["proc_starttime_ticks"],
        "namespace_inodes": dict(receipt["namespace_inodes"]),
    }


class FormalRouteCPolicyProcess:
    """Parent-owned handle to one persistent capability-limited policy child."""

    def __init__(
        self,
        context: PerceptionFactoryContext,
        *,
        python_executable: str | Path = sys.executable,
        startup_timeout_s: float = 180.0,
        message_timeout_s: float = 600.0,
        _test_policy_capability: object | None = None,
    ) -> None:
        executable = Path(python_executable).resolve(strict=True)
        if not executable.is_file():
            raise ValueError("formal Route C Python executable must be a file")
        for value, lower, upper, name in (
            (startup_timeout_s, 1.0, 600.0, "startup"),
            (message_timeout_s, 1.0, 3600.0, "message"),
        ):
            if (
                type(value) not in (int, float)
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or not lower <= float(value) <= upper
            ):
                raise ValueError(f"{name} timeout must be within {lower:g}..{upper:g} seconds")
        if (
            _test_policy_capability is not None
            and _test_policy_capability is not _EMBEDDED_TEST_POLICY_CAPABILITY
        ):
            raise TypeError("embedded test policy requires its private capability")
        test_policy = (
            _test_policy_capability is _EMBEDDED_TEST_POLICY_CAPABILITY
        )
        init = _init_payload(context)
        self._closed = True
        self._projection: SandboxProjection | None = None
        self._channel: BoundedFrameSocket | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._message_timeout_s = float(message_timeout_s)
        self._episodes_started = 0
        self._control_sequence = 0
        self._latest_episode_receipt: dict[str, Any] | None = None
        self._projection = SandboxProjection.create(
            python_executable=executable,
            perception_backend=context.perception_backend,
            perception_model=context.perception_model,
            accelerator=context.device.startswith("cuda"),
        )
        parent_socket, child_socket = socket.socketpair()
        child_fd = child_socket.fileno()
        command = self._projection.command(
            child_fd=child_fd,
            bootstrap=_BOOTSTRAP,
            test_policy=test_policy,
        )
        self.launch_argv = command
        try:
            process = subprocess.Popen(
                command,
                cwd="/tmp",
                env={},
                pass_fds=(child_fd,),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
            self._process = process
            child_socket.close()
            self._channel = BoundedFrameSocket(
                parent_socket,
                default_timeout_s=self._message_timeout_s,
            )
            self._closed = False
            self._channel.send_json(init, timeout_s=float(startup_timeout_s))
            ready = self._channel.recv_json(timeout_s=float(startup_timeout_s))
            if (
                type(ready) is dict
                and ready.get("schema") == PROCESS_MESSAGE_SCHEMA
                and ready.get("type") == "startup_error"
                and set(ready)
                == {"schema", "type", "exception_type", "detail"}
            ):
                raise FormalRouteCProcessError(
                    "formal Route C child startup failed: "
                    f"{ready['exception_type']}: {ready['detail']}"
                )
            if type(ready) is not dict or set(ready) != {
                "schema",
                "type",
                "runtime_receipt",
            } or ready.get("schema") != PROCESS_MESSAGE_SCHEMA or ready.get("type") != "ready":
                raise FormalRouteCProcessError("formal Route C child returned invalid ready")
            runtime_receipt = _validate_runtime_receipt(ready["runtime_receipt"])
            if runtime_receipt["source_sha256"] != self._projection.source_sha256:
                raise FormalRouteCProcessError("child policy projection hash differs")
            if runtime_receipt["gallery_sha256"] != self._projection.gallery_sha256:
                raise FormalRouteCProcessError("child gallery projection hash differs")
            if runtime_receipt["model_sha256"] != self._projection.model_sha256:
                raise FormalRouteCProcessError("child model repository hash differs")
            self._runtime_receipt = runtime_receipt
            for item in runtime_receipt["system_runtime_files"][len(SYSTEM_RUNTIME_FILES):]:
                if item["sha256"] != sha256_file(item["destination"]):
                    raise FormalRouteCProcessError("GPU driver hash differs from host runtime")
            expected_runner = (
                "embedded_test_fixture"
                if test_policy
                else "production_route_c_policy_episode"
            )
            if runtime_receipt["policy_runner"] != expected_runner:
                raise FormalRouteCProcessError(
                    "child policy runner differs from the fixed launch selector"
                )
            self._worker_nonce = runtime_receipt["worker_nonce"]
            self._live_process = _verify_live_child(process, runtime_receipt)
        except BaseException as exc:
            child_socket.close()
            parent_socket.close()
            self._abort()
            if isinstance(exc, FormalRouteCProcessError):
                raise
            if isinstance(exc, FormalProcessTransportError):
                raise FormalRouteCProcessError(str(exc)) from exc
            raise

    @classmethod
    def _create_embedded_test_fixture(
        cls,
        context: PerceptionFactoryContext,
        *,
        python_executable: str | Path = sys.executable,
        startup_timeout_s: float = 180.0,
        message_timeout_s: float = 600.0,
    ) -> "FormalRouteCPolicyProcess":
        """Create a non-attestable test child for transport/runtime smoke tests."""

        return cls(
            context,
            python_executable=python_executable,
            startup_timeout_s=startup_timeout_s,
            message_timeout_s=message_timeout_s,
            _test_policy_capability=_EMBEDDED_TEST_POLICY_CAPABILITY,
        )

    def run_episode(
        self,
        *,
        instruction: str,
        initial_observation: RobotObservation,
        step_budget: int,
        action_handler: Callable[[OSCAction], RobotObservation],
        observation_delivery: Callable[[RobotObservation, int], None] | None = None,
    ) -> RouteCPolicyExecution:
        if (
            self._closed
            or self._process is None
            or self._process.poll() is not None
            or self._channel is None
        ):
            raise FormalRouteCProcessError("formal Route C child is unavailable")
        if type(instruction) is not str or not instruction.strip() or len(instruction) > MAX_INSTRUCTION_CHARS:
            raise ValueError("instruction must be a non-empty bounded native string")
        if type(step_budget) is not int or not 1 <= step_budget <= MAX_POLICY_STEPS:
            raise ValueError("step_budget must be within the fixed formal range")
        if not callable(action_handler):
            raise TypeError("action_handler must be callable")
        if observation_delivery is not None and not callable(observation_delivery):
            raise TypeError("observation_delivery must be callable")
        episode_index = self._episodes_started + 1
        episode_nonce = secrets.token_hex(32)
        transcript = _Transcript(
            worker_nonce=self._worker_nonce,
            episode_nonce=episode_nonce,
            episode_index=episode_index,
        )
        parent_observations: list[dict[str, Any]] = []
        parent_actions: list[str] = []
        try:
            _send_chained_json(
                self._channel,
                transcript,
                {
                    "type": "start_episode",
                    "instruction": instruction,
                    "step_budget": step_budget,
                },
                direction="parent_to_child",
                timeout_s=self._message_timeout_s,
            )
            self._episodes_started = episode_index
            if observation_delivery is not None:
                observation_delivery(initial_observation, 0)
            parent_observations.append(
                _send_observation(
                    self._channel,
                    initial_observation,
                    transcript,
                    timeout_s=self._message_timeout_s,
                )
            )
        except BaseException:
            self._abort()
            raise
        actions = 0
        while True:
            try:
                core, raw_message = _recv_chained_json(
                    self._channel,
                    transcript,
                    direction="child_to_parent",
                    timeout_s=self._message_timeout_s,
                )
            except BaseException as exc:
                self._abort()
                if isinstance(exc, FormalRouteCProcessError):
                    raise
                raise FormalRouteCProcessError(
                    f"formal Route C child disconnected: {exc}"
                ) from exc
            message_type = core.get("type")
            if message_type == "action":
                if set(core) != {"type", "action"}:
                    self._abort()
                    raise FormalRouteCProcessError("child action has invalid fields")
                if actions >= step_budget:
                    self._abort()
                    raise FormalRouteCProcessError("child exceeded its fixed action horizon")
                try:
                    action = deserialize_policy_action(core["action"])
                except PolicyIPCValidationError as exc:
                    self._abort()
                    raise FormalRouteCProcessError(f"child action is invalid: {exc}") from exc
                parent_actions.append(
                    _json_sha256(core["action"], b"libero-policy-action.v1\0")
                )
                try:
                    observation = action_handler(action)
                except BaseException:
                    self._abort()
                    raise
                if type(observation) is not RobotObservation:
                    self._abort()
                    raise FormalRouteCProcessError(
                        "evaluator action handler returned a non-whitelist observation"
                    )
                actions += 1
                try:
                    if observation_delivery is not None:
                        observation_delivery(observation, actions)
                    parent_observations.append(
                        _send_observation(
                            self._channel,
                            observation,
                            transcript,
                            timeout_s=self._message_timeout_s,
                        )
                    )
                except BaseException:
                    self._abort()
                    raise
                continue
            if message_type == "error":
                if set(core) != {"type", "exception_type", "detail"}:
                    self._abort()
                    raise FormalRouteCProcessError("child error message is malformed")
                self._abort()
                raise FormalRouteCProcessError(
                    f"policy child {core.get('exception_type')}: {core.get('detail')}"
                )
            try:
                result = _result_from_core(core)
                runtime_episode = _validate_episode_runtime_receipt(
                    result.trace.get("episode_runtime")
                )
                child_receipt = core["episode_receipt"]
                if type(child_receipt) is not dict:
                    raise FormalRouteCProcessError("child episode receipt is invalid")
                complete_receipt = {
                    **child_receipt,
                    "final_transcript_sha256": transcript.sha256,
                    "result_core_sha256": _json_sha256(
                        {name: value for name, value in core.items() if name != "episode_receipt"},
                        b"libero-policy-result-core.v1\0",
                    ),
                }
                _validate_episode_receipt(complete_receipt)
            except BaseException:
                self._abort()
                raise
            if (
                result.steps_executed != actions
                or complete_receipt["action_count"] != actions
                or complete_receipt["observation_count"] != actions + 1
                or complete_receipt["worker_nonce"] != self._worker_nonce
                or complete_receipt["episode_nonce"] != episode_nonce
                or complete_receipt["episode_index"] != episode_index
                or runtime_episode["assembly_nonce"]
                != complete_receipt["runtime_assembly_nonce"]
                or runtime_episode["assembly_count"]
                != complete_receipt["runtime_assembly_count"]
                or runtime_episode["component_count"]
                != complete_receipt["runtime_component_count"]
                or runtime_episode["component_identity_sha256"]
                != complete_receipt["runtime_component_identity_sha256"]
                or runtime_episode["initial_state_sha256"]
                != complete_receipt["runtime_initial_state_sha256"]
                or runtime_episode["final_action_count"] != actions
                or complete_receipt["observation_commitments"] != parent_observations
                or complete_receipt["action_commitments"] != parent_actions
                or complete_receipt["transcript_before_result_sha256"]
                != raw_message["previous_transcript_sha256"]
            ):
                self._abort()
                raise FormalRouteCProcessError(
                    "child episode receipt differs from parent-observed transcript"
                )
            self._latest_episode_receipt = _strict_json_copy(complete_receipt)
            return result

    def boundary_attestation(self) -> dict[str, Any]:
        if self._closed or self._process is None or self._projection is None:
            raise FormalRouteCProcessError("formal Route C child is unavailable")
        if self._runtime_receipt["policy_runner"] != (
            "production_route_c_policy_episode"
        ):
            raise FormalRouteCProcessError(
                "embedded test policy processes cannot produce formal attestation"
            )
        self._live_process = _verify_live_child(
            self._process,
            self._runtime_receipt,
        )
        return _build_process_boundary_attestation(
            projection=self._projection,
            runtime_receipt=self._runtime_receipt,
            live_process=self._live_process,
            latest_episode_receipt=self._latest_episode_receipt,
        )

    @property
    def latest_episode_receipt(self) -> dict[str, Any] | None:
        return (
            None
            if self._latest_episode_receipt is None
            else _strict_json_copy(self._latest_episode_receipt)
        )

    @property
    def runtime_receipt(self) -> dict[str, Any]:
        if self._closed:
            raise FormalRouteCProcessError("formal Route C child is unavailable")
        return _strict_json_copy(self._runtime_receipt)

    def _abort(self) -> None:
        if not getattr(self, "_closed", True):
            self._closed = True
            if self._channel is not None:
                try:
                    self._channel.close()
                except BaseException:
                    pass
            if self._process is not None and self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=5.0)
        projection = getattr(self, "_projection", None)
        if projection is not None:
            self._projection = None
            projection.cleanup()

    def close(self) -> None:
        if self._closed:
            if self._projection is not None:
                projection = self._projection
                self._projection = None
                projection.cleanup()
            return
        assert self._channel is not None and self._process is not None
        try:
            request = {
                "schema": PROCESS_MESSAGE_SCHEMA,
                "type": "shutdown",
                "worker_nonce": self._worker_nonce,
                "control_sequence": self._control_sequence,
            }
            self._channel.send_json(request, timeout_s=10.0)
            reply = self._channel.recv_json(timeout_s=10.0)
            if reply != {
                "schema": PROCESS_MESSAGE_SCHEMA,
                "type": "closed",
                "worker_nonce": self._worker_nonce,
                "control_sequence": self._control_sequence,
            }:
                raise FormalRouteCProcessError("child returned invalid close receipt")
            self._control_sequence += 1
            self._channel.close()
            self._process.wait(timeout=10.0)
            if self._process.returncode != 0:
                raise FormalRouteCProcessError(
                    f"sandbox child exited with status {self._process.returncode}"
                )
            self._closed = True
            projection = self._projection
            self._projection = None
            if projection is not None:
                projection.cleanup()
        except BaseException:
            self._abort()
            raise

    def __enter__(self) -> "FormalRouteCPolicyProcess":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self._abort()
        except BaseException:
            pass


def _child_bootstrap(child_fd: int, test_policy: bool = False) -> int:
    if type(child_fd) is not int or child_fd < 3 or type(test_policy) is not bool:
        return 2
    raw_socket = socket.socket(fileno=child_fd)
    channel = BoundedFrameSocket(raw_socket, default_timeout_s=_CHILD_IO_TIMEOUT_S)
    try:
        return _child_loop(
            channel,
            protocol_fd=child_fd,
            test_policy=test_policy,
        )
    except BaseException as exc:
        try:
            channel.send_json(
                {
                    "schema": PROCESS_MESSAGE_SCHEMA,
                    "type": "startup_error",
                    "exception_type": type(exc).__name__,
                    "detail": str(exc)[:2_000],
                },
                timeout_s=5.0,
            )
        except BaseException:
            pass
        return 1
    finally:
        channel.close()


def _child_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--child-fd", type=int, required=True)
    parser.add_argument("--test-policy", action="store_true")
    arguments = parser.parse_args(argv)
    return _child_bootstrap(arguments.child_fd, arguments.test_policy)


if __name__ == "__main__":
    raise SystemExit(_child_main())


__all__ = [
    "FormalRouteCPolicyProcess",
    "FormalRouteCProcessError",
    "MAX_POLICY_STEPS",
    "PROCESS_BOUNDARY_ATTESTATION_SCHEMA",
    "PROCESS_EPISODE_RECEIPT_SCHEMA",
    "PROCESS_INIT_SCHEMA",
    "PROCESS_MESSAGE_SCHEMA",
    "PROCESS_RUNTIME_RECEIPT_SCHEMA",
    "validate_process_boundary_attestation",
]
