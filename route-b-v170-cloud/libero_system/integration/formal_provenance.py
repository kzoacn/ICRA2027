"""Immutable formal-campaign provenance shared by runners and verifiers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping
import uuid


PROVENANCE_SCHEMA = "libero-route-shard-provenance.v5"
POLICY_CONTRACT_SCHEMA = "libero-formal-policy-contract.v4"
EPISODE_SCHEMA = "libero-route-episode.v5"
LIFECYCLE_SCHEMA = "libero-formal-episode-lifecycle.v4"
SUMMARY_SCHEMA = "libero-routes-eval.v4"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_OPAQUE_POLICY_TOKEN_DOMAIN = b"libero-formal-policy-attempt-token.v2\0"

_REQUIRED_EXECUTION_SCHEDULE: dict[str, Any] = {
    "per_route_max_workers": 2,
    "global_max_workers": 2,
    "route_order": ["b", "c"],
    "routes_must_not_overlap": True,
    "evidence": "runner_execution_schedule_intervals",
}

_REQUIRED_POLICY_CONTRACT: dict[str, Any] = {
    "schema": POLICY_CONTRACT_SCHEMA,
    "training": False,
    "policy_inputs": {
        "task": ["instruction", "opaque_episode_id"],
        "observation": [
            "agentview_rgbd",
            "wrist_rgbd",
            "camera_calibration",
            "proprioception",
        ],
    },
    "forbidden_inputs": [
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
    ],
    "evaluation": {
        "success_scoring": "external_sticky_any_success",
        "evaluator_signal_deliveries_to_policy_or_controller": 0,
        "evaluator_derived_clock_or_step_inputs_to_policy_or_controller": 0,
        "evaluator_driven_early_stops": 0,
        "policy_action_horizon": 520,
    },
    "black_bowl_grasp": {
        "allowed": "closed_finger_rim_pinch",
        "expansion_allowed": False,
    },
}

_EFFECTIVE_CONFIG_FIELDS = (
    "route",
    "suite",
    "task_ids",
    "episodes_per_task",
    "output_dir",
    "run_name",
    "device",
    "perception_backend",
    "perception_model",
    "perception_factory",
    "libero_config_path",
    "max_steps",
    "image_size",
    "seed",
    "record_video",
    "video_fps",
    "video_stride",
    "render_backend",
    "init_state_start",
)


class FormalProvenanceError(ValueError):
    """Raised when a formal provenance lock is absent or inconsistent."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def structured_sha256(domain: str, value: Any) -> str:
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical_json_bytes(value))
    return digest.hexdigest()


def valid_uuid4_hex(value: object) -> bool:
    """Return true only for a canonical lowercase RFC-4122 UUIDv4 hex value."""

    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{32}", value) is None:
        return False
    try:
        parsed = uuid.UUID(hex=value)
    except (AttributeError, ValueError):
        return False
    return parsed.hex == value and parsed.version == 4 and parsed.variant == uuid.RFC_4122


def opaque_policy_token_for_attempt(attempt_id: object) -> str:
    """Derive the sole policy-visible opaque token for a formal attempt."""

    if not valid_uuid4_hex(attempt_id):
        raise FormalProvenanceError("attempt_id must be canonical UUIDv4 hex")
    # Hash the complete 128-bit UUID payload.  Truncating the printable UUID
    # before hashing made distinct attempts that shared a prefix collapse to
    # the same policy-visible identity.  The domain prefix prevents this
    # digest from being confused with any other UUID commitment.
    digest = hashlib.sha256(
        _OPAQUE_POLICY_TOKEN_DOMAIN + uuid.UUID(hex=str(attempt_id)).bytes
    ).hexdigest()
    return f"episode-v2-{digest}"


def opaque_policy_token_sha256(attempt_id: object) -> str:
    return hashlib.sha256(
        opaque_policy_token_for_attempt(attempt_id).encode("utf-8")
    ).hexdigest()


def effective_run_config(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the stable, resume-independent portion of EvaluationConfig."""

    result: dict[str, Any] = {}
    for field in _EFFECTIVE_CONFIG_FIELDS:
        item = value.get(field)
        if isinstance(item, Path):
            item = str(item)
        elif isinstance(item, tuple):
            item = list(item)
        result[field] = item
    return result


def effective_run_config_sha256(value: Mapping[str, Any]) -> str:
    return structured_sha256(
        "libero-effective-run-config.v2", effective_run_config(value)
    )


def default_policy_contract_path() -> Path:
    return Path(__file__).resolve().parents[2] / "manifests" / "formal_policy_contract.v4.json"


def validate_policy_contract(path: str | Path, expected_sha256: str | None = None) -> dict[str, Any]:
    contract_path = Path(path).expanduser().resolve(strict=True)
    try:
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalProvenanceError(f"cannot read policy contract {contract_path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != POLICY_CONTRACT_SCHEMA:
        raise FormalProvenanceError(
            f"policy contract schema must be {POLICY_CONTRACT_SCHEMA!r}"
        )
    if payload != _REQUIRED_POLICY_CONTRACT:
        raise FormalProvenanceError(
            "policy contract does not match the frozen sensor-only formal contract"
        )
    digest = file_sha256(contract_path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise FormalProvenanceError(
            "formal policy contract hash changed: "
            f"expected={expected_sha256}, current={digest}"
        )
    return payload


def _expected_config_from_plan(
    plan: Mapping[str, Any], shard: Mapping[str, Any]
) -> dict[str, Any]:
    protocol = plan["protocol"]
    run_directory = Path(str(shard["run_directory"]))
    return {
        "route": plan["route"],
        "suite": shard["suite"],
        "task_ids": list(shard["task_ids"]),
        "episodes_per_task": protocol["episodes_per_task"],
        "output_dir": str(run_directory.parent),
        "run_name": shard["run_name"],
        "device": protocol["device"],
        "perception_backend": protocol["perception"]["backend"],
        "perception_model": protocol["perception"]["model"],
        "perception_factory": None,
        "libero_config_path": protocol["libero_config_path"],
        "max_steps": protocol["max_steps"][shard["suite"]],
        "image_size": protocol["image_size"],
        "seed": protocol["base_seed"],
        "record_video": True,
        "video_fps": protocol["video"]["fps"],
        "video_stride": protocol["video"]["stride"],
        "render_backend": protocol["render_backend"],
        "init_state_start": protocol["init_state_start"],
    }


def build_shard_provenance(
    plan: Mapping[str, Any],
    shard: Mapping[str, Any],
    *,
    plan_path: str | Path,
    execution_nonce: str,
    shard_execution_nonce: str,
    execution_started_at_unix_ns: int,
    cohort: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the fresh, execution-bound launch ticket for one planned shard."""

    if not valid_uuid4_hex(execution_nonce):
        raise FormalProvenanceError("execution_nonce must be a canonical UUIDv4 hex value")
    if not valid_uuid4_hex(shard_execution_nonce):
        raise FormalProvenanceError(
            "shard_execution_nonce must be a canonical UUIDv4 hex value"
        )
    if (
        not isinstance(execution_started_at_unix_ns, int)
        or isinstance(execution_started_at_unix_ns, bool)
        or execution_started_at_unix_ns <= 0
    ):
        raise FormalProvenanceError("execution_started_at_unix_ns must be a positive integer")

    resolved_plan = Path(plan_path).expanduser().resolve(strict=True)
    policy_contract = plan["policy_contract"]
    expected_config = _expected_config_from_plan(plan, shard)
    episode_ids = [
        f"{plan['route']}:{shard['suite']}:task{int(task_id):02d}:ep{int(index):04d}"
        for task_id in shard["task_ids"]
        for index in plan["protocol"]["episode_indices"]
    ]
    dependencies = plan["dependency_provenance"]
    if not isinstance(cohort, Mapping):
        raise FormalProvenanceError("cohort binding must be an object")
    return {
        "schema": PROVENANCE_SCHEMA,
        "formal": True,
        "campaign_plan": {
            "path": str(resolved_plan),
            "file_sha256": file_sha256(resolved_plan),
        },
        "execution": {
            "execution_nonce": execution_nonce,
            "shard_execution_nonce": shard_execution_nonce,
            "started_at_unix_ns": execution_started_at_unix_ns,
        },
        "cohort": dict(cohort),
        "shard": {
            "index": shard["index"],
            "shard_id": shard["shard_id"],
            "route": plan["route"],
            "suite": shard["suite"],
            "task_ids": list(shard["task_ids"]),
            "run_name": shard["run_name"],
            "run_directory": shard["run_directory"],
            "spec_sha256": structured_sha256("libero-shard-spec.v3", shard),
        },
        "expected_episode_ids": episode_ids,
        "execution_schedule_contract": dict(
            plan["protocol"]["execution_schedule"]
        ),
        "locks": {
            "manifest_sha256": plan["manifest"]["sha256"],
            "source_tree_sha256": plan["source_tree_sha256"],
            "orchestration_source_sha256": plan[
                "orchestration_source_sha256"
            ],
            "protocol_sha256": structured_sha256(
                "libero-campaign-protocol.v3", plan["protocol"]
            ),
            "dependency_provenance_sha256": structured_sha256(
                "libero-dependency-provenance.v3", dependencies
            ),
            "python_executable_sha256": dependencies["python"]["executable_sha256"],
            "libero_config_sha256": dependencies["libero_config"]["sha256"],
            "perception_model_tree_sha256": dependencies["perception_model"][
                "tree_sha256"
            ],
            "libero_assets_tree_sha256": dependencies["libero_assets"]["tree_sha256"],
            "policy_contract_path": policy_contract["path"],
            "policy_contract_sha256": policy_contract["sha256"],
            "execution_schedule_sha256": structured_sha256(
                "libero-formal-execution-schedule-contract.v1",
                plan["protocol"]["execution_schedule"],
            ),
        },
        "runtime_binding": {
            "python_executable": dependencies["python"]["executable"],
            "python_executable_sha256": dependencies["python"][
                "executable_sha256"
            ],
            "python_runtime": dependencies["python"]["python_runtime"],
            "formal_environment": plan["protocol"]["formal_environment"],
            "formal_runtime_environment": dependencies["python"][
                "formal_runtime_environment"
            ],
            "media_binaries": dependencies["python"]["media_binaries"],
            "route_c_process_plan_binding": dependencies["route_c_process"],
        },
        "effective_run_config_sha256": effective_run_config_sha256(expected_config),
    }


def _serialized_provenance(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def ensure_shard_provenance(payload: Mapping[str, Any], path: str | Path) -> tuple[Path, str]:
    """Exclusively create one immutable, fresh-only shard launch ticket."""

    validate_shard_provenance(payload)
    raw_destination = Path(path).expanduser()
    destination = raw_destination.resolve()
    if not raw_destination.is_absolute() or str(raw_destination) != str(destination):
        raise FormalProvenanceError(
            "formal shard provenance path must be exact and canonical"
        )
    expected = _serialized_provenance(payload)
    if destination.exists():
        raise FormalProvenanceError(
            f"fresh formal shard already has provenance: {destination}"
        )
    destination.parent.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.parent.mkdir()
    except FileExistsError as exc:
        raise FormalProvenanceError(
            f"fresh formal shard directory already exists: {destination.parent}"
        ) from exc
    with tempfile.NamedTemporaryFile(
        "wb",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        stream.write(expected)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    try:
        try:
            os.link(temporary, destination)
        except FileExistsError:
            raise FormalProvenanceError(
                f"concurrent formal shard provenance already exists: {destination}"
            ) from None
    finally:
        temporary.unlink(missing_ok=True)
    directory_descriptor = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return destination, file_sha256(destination)


def validate_shard_provenance(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping) or payload.get("schema") != PROVENANCE_SCHEMA:
        raise FormalProvenanceError(f"provenance schema must be {PROVENANCE_SCHEMA!r}")
    if payload.get("formal") is not True:
        raise FormalProvenanceError("formal shard provenance must set formal=true")
    plan = payload.get("campaign_plan")
    execution = payload.get("execution")
    cohort = payload.get("cohort")
    shard = payload.get("shard")
    locks = payload.get("locks")
    runtime_binding = payload.get("runtime_binding")
    if not all(
        isinstance(item, Mapping)
        for item in (plan, execution, cohort, shard, locks, runtime_binding)
    ):
        raise FormalProvenanceError(
            "provenance plan, execution, cohort, shard, locks, and runtime must be objects"
        )
    if set(cohort) != {
        "cohort_nonce",
        "cohort_directory",
        "ticket_path",
        "ticket_sha256",
        "state_path",
        "lock_path",
        "lock_sha256",
    }:
        raise FormalProvenanceError("provenance cohort fields are invalid")
    if not valid_uuid4_hex(cohort.get("cohort_nonce")):
        raise FormalProvenanceError("provenance cohort_nonce is invalid")
    for field in ("cohort_directory", "ticket_path", "state_path", "lock_path"):
        value = cohort.get(field)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise FormalProvenanceError(f"provenance cohort {field} is invalid")
    ticket_sha = cohort.get("ticket_sha256")
    if not isinstance(ticket_sha, str) or _SHA256.fullmatch(ticket_sha) is None:
        raise FormalProvenanceError("provenance cohort ticket hash is invalid")
    lock_sha = cohort.get("lock_sha256")
    if not isinstance(lock_sha, str) or _SHA256.fullmatch(lock_sha) is None:
        raise FormalProvenanceError("provenance cohort lock hash is invalid")
    if set(execution) != {
        "execution_nonce",
        "shard_execution_nonce",
        "started_at_unix_ns",
    }:
        raise FormalProvenanceError("provenance execution fields are invalid")
    if not valid_uuid4_hex(execution.get("execution_nonce")):
        raise FormalProvenanceError("provenance execution_nonce is invalid")
    if not valid_uuid4_hex(execution.get("shard_execution_nonce")):
        raise FormalProvenanceError("provenance shard_execution_nonce is invalid")
    started_at = execution.get("started_at_unix_ns")
    if (
        not isinstance(started_at, int)
        or isinstance(started_at, bool)
        or started_at <= 0
    ):
        raise FormalProvenanceError(
            "provenance execution started_at_unix_ns is invalid"
        )
    for owner, field in (
        (plan, "file_sha256"),
        (shard, "spec_sha256"),
        (locks, "manifest_sha256"),
        (locks, "source_tree_sha256"),
        (locks, "orchestration_source_sha256"),
        (locks, "protocol_sha256"),
        (locks, "dependency_provenance_sha256"),
        (locks, "python_executable_sha256"),
        (locks, "libero_config_sha256"),
        (locks, "perception_model_tree_sha256"),
        (locks, "libero_assets_tree_sha256"),
        (locks, "policy_contract_sha256"),
        (locks, "execution_schedule_sha256"),
        (payload, "effective_run_config_sha256"),
    ):
        value = owner.get(field)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise FormalProvenanceError(f"provenance {field} is not a SHA-256")
    if set(runtime_binding) != {
        "python_executable",
        "python_executable_sha256",
        "python_runtime",
        "formal_environment",
        "formal_runtime_environment",
        "media_binaries",
        "route_c_process_plan_binding",
    }:
        raise FormalProvenanceError("provenance runtime-binding fields are invalid")
    if (
        not isinstance(runtime_binding.get("python_executable"), str)
        or not Path(runtime_binding["python_executable"]).is_absolute()
        or runtime_binding.get("python_executable_sha256")
        != locks.get("python_executable_sha256")
    ):
        raise FormalProvenanceError("provenance runtime Python binding is invalid")
    environment = runtime_binding.get("formal_runtime_environment")
    if (
        not isinstance(environment, Mapping)
        or environment.get("schema") != "libero-formal-runtime-environment.v2"
        or not isinstance(environment.get("modules"), list)
        or not isinstance(environment.get("distributions"), list)
        or not isinstance(environment.get("bootstrap_sys_path"), list)
        or environment.get("sha256")
        != structured_sha256(
            "libero-formal-runtime-environment.v2",
            {
                "modules": environment.get("modules"),
                "distributions": environment.get("distributions"),
                "bootstrap_sys_path": environment.get("bootstrap_sys_path"),
            },
        )
    ):
        raise FormalProvenanceError("provenance formal runtime environment is invalid")
    python_runtime = runtime_binding.get("python_runtime")
    if (
        not isinstance(python_runtime, Mapping)
        or python_runtime.get("schema") != "libero-python-runtime-closure.v1"
        or not isinstance(python_runtime.get("stdlib_roots"), list)
        or not isinstance(python_runtime.get("native_closure"), Mapping)
    ):
        raise FormalProvenanceError("provenance Python runtime closure is invalid")
    formal_environment = runtime_binding.get("formal_environment")
    if not isinstance(formal_environment, Mapping) or not formal_environment:
        raise FormalProvenanceError("provenance formal child environment is invalid")
    media = runtime_binding.get("media_binaries")
    media_executables = media.get("executables") if isinstance(media, Mapping) else None
    if (
        not isinstance(media, Mapping)
        or media.get("schema") != "libero-formal-media-binaries.v3"
        or not isinstance(media_executables, list)
        or [
            item.get("role") if isinstance(item, Mapping) else None
            for item in media_executables
        ]
        != ["ffprobe", "imageio_ffmpeg", "bwrap"]
        or media.get("sha256")
        != structured_sha256(
            "libero-formal-media-binaries.v3", media_executables
        )
    ):
        raise FormalProvenanceError("provenance formal media binding is invalid")
    route_c_binding = runtime_binding.get("route_c_process_plan_binding")
    if (
        not isinstance(route_c_binding, Mapping)
        or route_c_binding.get("schema")
        != "libero-formal-route-c-plan-binding.v1"
    ):
        raise FormalProvenanceError("provenance Route C process binding is invalid")
    expected_ids = payload.get("expected_episode_ids")
    if not isinstance(expected_ids, list) or not expected_ids or not all(
        isinstance(item, str) and item for item in expected_ids
    ) or len(expected_ids) != len(set(expected_ids)):
        raise FormalProvenanceError("expected_episode_ids must be unique non-empty strings")
    execution_schedule = payload.get("execution_schedule_contract")
    if (
        not isinstance(execution_schedule, Mapping)
        or set(execution_schedule) != set(_REQUIRED_EXECUTION_SCHEDULE)
        or execution_schedule != _REQUIRED_EXECUTION_SCHEDULE
        or type(execution_schedule.get("per_route_max_workers")) is not int
        or type(execution_schedule.get("global_max_workers")) is not int
        or type(execution_schedule.get("routes_must_not_overlap")) is not bool
    ):
        raise FormalProvenanceError(
            "provenance execution_schedule_contract must require two workers "
            "per route and sequential B/C execution"
        )
    expected_schedule_hash = structured_sha256(
        "libero-formal-execution-schedule-contract.v1", execution_schedule
    )
    if locks.get("execution_schedule_sha256") != expected_schedule_hash:
        raise FormalProvenanceError(
            "provenance execution_schedule_sha256 does not match its contract"
        )
    for owner, field in (
        (plan, "path"),
        (shard, "route"),
        (shard, "suite"),
        (shard, "run_name"),
        (shard, "run_directory"),
        (locks, "policy_contract_path"),
    ):
        if not isinstance(owner.get(field), str) or not owner[field]:
            raise FormalProvenanceError(f"provenance {field} must be a non-empty string")


def load_shard_provenance(
    path: str | Path, *, expected_sha256: str | None = None
) -> tuple[dict[str, Any], str]:
    raw_path = Path(path).expanduser()
    provenance_path = raw_path.resolve(strict=True)
    if (
        not raw_path.is_absolute()
        or str(raw_path) != str(provenance_path)
        or raw_path.is_symlink()
    ):
        raise FormalProvenanceError(
            "formal shard provenance path must be exact and must not be a symlink"
        )
    digest = file_sha256(provenance_path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise FormalProvenanceError(
            f"shard provenance hash mismatch: expected={expected_sha256}, current={digest}"
        )
    try:
        payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalProvenanceError(f"cannot read shard provenance: {exc}") from exc
    if not isinstance(payload, dict):
        raise FormalProvenanceError("shard provenance top-level value must be an object")
    validate_shard_provenance(payload)
    plan = payload["campaign_plan"]
    if file_sha256(plan["path"]) != plan["file_sha256"]:
        raise FormalProvenanceError("campaign plan file changed after shard provenance creation")
    locks = payload["locks"]
    validate_policy_contract(
        locks["policy_contract_path"], locks["policy_contract_sha256"]
    )
    return payload, digest


def validate_provenance_for_config(
    payload: Mapping[str, Any], config: Mapping[str, Any], *, run_directory: str | Path
) -> None:
    actual_hash = effective_run_config_sha256(config)
    if actual_hash != payload.get("effective_run_config_sha256"):
        raise FormalProvenanceError("effective run configuration differs from shard provenance")
    shard = payload["shard"]
    raw_directory = Path(run_directory).expanduser()
    actual_directory = raw_directory.resolve()
    expected_directory = Path(str(shard["run_directory"])).expanduser()
    if (
        not raw_directory.is_absolute()
        or str(raw_directory) != str(actual_directory)
        or str(expected_directory) != str(expected_directory.resolve())
        or actual_directory != expected_directory
    ):
        raise FormalProvenanceError("run directory differs from shard provenance")
