"""Resumable, append-only evaluation records for Routes B and C.

The JSONL file is the source of truth.  The summary is derived from it, so an
interrupted 50-episode run can be resumed without rerunning completed episodes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Iterable, Mapping

from .formal_provenance import (
    EPISODE_SCHEMA,
    LIFECYCLE_SCHEMA,
    SUMMARY_SCHEMA,
    file_sha256,
    opaque_policy_token_for_attempt,
    opaque_policy_token_sha256,
    valid_uuid4_hex,
)
from .grasp_audit import (
    AuditRoute,
    GraspAuditValidationError,
    validate_grasp_attempt_counts,
    validate_grasp_audit_report,
)
from .policy_boundary_audit import (
    POLICY_BOUNDARY_AUDIT_SCHEMA,
    PolicyBoundaryAuditValidationError,
    validate_policy_boundary_audit,
)
from .video_audit import VIDEO_CONTENT_AUDIT_SCHEMA


@dataclass(frozen=True, order=True)
class EpisodeKey:
    route: str
    suite: str
    task_id: int
    episode_index: int

    def __post_init__(self) -> None:
        if self.route not in {"b", "c"}:
            raise ValueError("route must be 'b' or 'c'")
        if not self.suite or self.task_id < 0 or self.episode_index < 0:
            raise ValueError("suite, task_id, and episode_index must be valid")

    @property
    def id(self) -> str:
        return f"{self.route}:{self.suite}:task{self.task_id:02d}:ep{self.episode_index:04d}"


@dataclass(frozen=True)
class EpisodeRecord:
    key: EpisodeKey
    instruction: str
    evaluator_success: bool
    policy_status: str
    steps: int
    elapsed_s: float
    seed: int | None = None
    failure: str | None = None
    video_paths: Mapping[str, str] = field(default_factory=dict)
    route_trace: Mapping[str, Any] = field(default_factory=dict)
    provenance_sha256: str | None = None
    lifecycle: Mapping[str, Any] | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def __post_init__(self) -> None:
        if not self.instruction.strip():
            raise ValueError("instruction cannot be empty")
        if self.steps < 0 or self.elapsed_s < 0:
            raise ValueError("steps and elapsed_s must be non-negative")
        if (self.provenance_sha256 is None) != (self.lifecycle is None):
            raise ValueError("formal provenance and lifecycle must be provided together")
        if self.provenance_sha256 is not None:
            if type(self.evaluator_success) is not bool:
                raise ValueError("formal evaluator_success must be a boolean")
            if not re.fullmatch(r"[0-9a-f]{64}", self.provenance_sha256):
                raise ValueError("formal provenance_sha256 is invalid")
            lifecycle = self.lifecycle
            if not isinstance(lifecycle, Mapping):
                raise ValueError("formal lifecycle must be an object")
            expected_lifecycle_fields = {
                "schema",
                "state",
                "attempt_id",
                "execution_nonce",
                "shard_execution_nonce",
                "started_at_unix_ns",
                "ended_at_unix_ns",
                "reset_completed",
                "score_finalized",
                "video_finalized",
                "stop_owner",
                "stop_reason",
                "action_steps",
                "video_content_audit",
            }
            if set(lifecycle) != expected_lifecycle_fields:
                raise ValueError("formal lifecycle fields are invalid")
            if lifecycle.get("schema") != LIFECYCLE_SCHEMA:
                raise ValueError("formal lifecycle schema is invalid")
            if lifecycle.get("state") != "completed":
                raise ValueError("formal lifecycle state must be completed")
            for field_name in (
                "attempt_id",
                "execution_nonce",
                "shard_execution_nonce",
            ):
                if not valid_uuid4_hex(lifecycle.get(field_name)):
                    raise ValueError(f"formal lifecycle {field_name} must be UUIDv4 hex")
            started_at_ns = lifecycle.get("started_at_unix_ns")
            ended_at_ns = lifecycle.get("ended_at_unix_ns")
            if (
                not isinstance(started_at_ns, int)
                or isinstance(started_at_ns, bool)
                or not isinstance(ended_at_ns, int)
                or isinstance(ended_at_ns, bool)
                or started_at_ns <= 0
                or started_at_ns >= ended_at_ns
            ):
                raise ValueError("formal lifecycle unix-ns interval is invalid")
            for field_name in (
                "reset_completed",
                "score_finalized",
                "video_finalized",
            ):
                if lifecycle.get(field_name) is not True:
                    raise ValueError(f"formal lifecycle {field_name} must be true")
            if lifecycle.get("stop_owner") not in {"policy", "fixed_horizon"}:
                raise ValueError("formal lifecycle stop_owner is invalid")
            if lifecycle.get("stop_reason") not in {"succeeded", "failed", "timeout"}:
                raise ValueError("formal lifecycle stop_reason is invalid")
            if lifecycle.get("action_steps") != self.steps:
                raise ValueError("formal lifecycle action_steps must equal steps")
            video_audit = lifecycle.get("video_content_audit")
            if not isinstance(video_audit, Mapping):
                raise ValueError("formal lifecycle video_content_audit must be an object")
            if video_audit.get("schema") != VIDEO_CONTENT_AUDIT_SCHEMA:
                raise ValueError("formal lifecycle video content audit schema is invalid")
            if video_audit.get("formal_pass") is not True:
                raise ValueError("formal lifecycle video content audit must pass")
            video_sha256 = video_audit.get("sha256")
            if not isinstance(video_sha256, str) or not re.fullmatch(
                r"[0-9a-f]{64}", video_sha256
            ):
                raise ValueError("formal lifecycle video content SHA-256 is invalid")
            if self.policy_status == "exception":
                raise ValueError("formal infrastructure exceptions are not episodes")
            if not isinstance(self.route_trace, Mapping):
                raise ValueError("formal route_trace must be an object")
            boundary_audit = self.route_trace.get("policy_boundary_audit")
            if not isinstance(boundary_audit, Mapping):
                raise ValueError(
                    "formal route_trace policy_boundary_audit must be an object"
                )
            if boundary_audit.get("schema") != POLICY_BOUNDARY_AUDIT_SCHEMA:
                raise ValueError("formal policy boundary audit schema is invalid")
            try:
                validated_boundary = validate_policy_boundary_audit(boundary_audit)
            except PolicyBoundaryAuditValidationError as exc:
                raise ValueError(f"formal policy boundary audit is invalid: {exc}") from exc
            if validated_boundary["route"] != self.key.route:
                raise ValueError("formal policy boundary audit route does not match row")
            if validated_boundary["evaluator_step_count"]["value"] != self.steps:
                raise ValueError(
                    "formal policy boundary evaluator step count must equal steps"
                )
            isolation = self.route_trace.get("evaluator_isolation")
            if not isinstance(isolation, Mapping):
                raise ValueError("formal route_trace evaluator_isolation must be an object")
            evaluator_fields = {
                "reward",
                "success",
                "done",
                "terminated",
                "truncated",
            }
            for destination in (
                "delivered_to_policy",
                "delivered_to_controller",
            ):
                deliveries = isolation.get(destination)
                if (
                    not isinstance(deliveries, Mapping)
                    or set(deliveries) != evaluator_fields
                    or any(
                        type(value) is not int or value != 0
                        for value in deliveries.values()
                    )
                ):
                    raise ValueError(
                        f"formal evaluator isolation {destination} must contain "
                        "exact integer-zero signal deliveries"
                    )
            if (
                type(
                    isolation.get(
                        "evaluator_derived_clock_or_step_inputs_to_policy_or_controller"
                    )
                )
                is not int
                or isolation.get(
                    "evaluator_derived_clock_or_step_inputs_to_policy_or_controller"
                )
                != 0
            ):
                raise ValueError(
                    "formal evaluator isolation derived clock/step inputs must be zero"
                )
            if (
                type(isolation.get("evaluator_driven_early_stops")) is not int
                or isolation.get("evaluator_driven_early_stops") != 0
            ):
                raise ValueError(
                    "formal evaluator isolation evaluator_driven_early_stops must be zero"
                )
            grasp_audit = self.route_trace.get("grasp_audit")
            if not isinstance(grasp_audit, Mapping):
                raise ValueError("formal route_trace grasp_audit must be an object")
            grasp_counts = self.route_trace.get("grasp_attempt_counts")
            if not isinstance(grasp_counts, Mapping):
                raise ValueError(
                    "formal route_trace grasp_attempt_counts must be an object"
                )
            try:
                audit_route = AuditRoute(self.key.route)
                validated_grasps = validate_grasp_audit_report(
                    grasp_audit,
                    expected_route=audit_route,
                )
                validated_counts = validate_grasp_attempt_counts(
                    grasp_counts,
                    validated_grasps["records"],
                    route=audit_route,
                )
            except (GraspAuditValidationError, ValueError) as exc:
                raise ValueError(f"formal grasp audit is invalid: {exc}") from exc
            if validated_counts["pending"]:
                if (
                    lifecycle.get("stop_owner") != "fixed_horizon"
                    or lifecycle.get("stop_reason") != "timeout"
                ):
                    raise ValueError(
                        "formal pending grasp audit requires fixed-horizon timeout"
                    )
            grasp_episode_id = validated_grasps["episode_id"]
            expected_opaque_token = opaque_policy_token_for_attempt(
                lifecycle["attempt_id"]
            )
            expected_opaque_commitment = opaque_policy_token_sha256(
                lifecycle["attempt_id"]
            )
            if grasp_episode_id != expected_opaque_token:
                raise ValueError(
                    "formal grasp audit episode token is not derived from "
                    "lifecycle attempt_id"
                )
            if validated_boundary["task_delivery"].get(
                "opaque_episode_id_sha256"
            ) != expected_opaque_commitment:
                raise ValueError(
                    "formal policy boundary opaque token commitment is not "
                    "derived from lifecycle attempt_id"
                )
            grasp_episode_commitment = hashlib.sha256(
                str(grasp_episode_id).encode("utf-8")
            ).hexdigest()
            if (
                grasp_episode_commitment
                != validated_boundary["task_delivery"].get(
                    "opaque_episode_id_sha256"
                )
            ):
                raise ValueError(
                    "formal grasp audit episode token does not match the "
                    "policy boundary commitment"
                )

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["episode_id"] = self.key.id
        if self.provenance_sha256 is None:
            row.pop("provenance_sha256", None)
            row.pop("lifecycle", None)
        else:
            row["schema"] = EPISODE_SCHEMA
        return row

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "EpisodeRecord":
        formal = row.get("schema") == EPISODE_SCHEMA
        if row.get("schema") is not None and not formal:
            raise ValueError("unknown episode record schema")
        raw_success = row["evaluator_success"]
        if formal and type(raw_success) is not bool:
            raise ValueError("formal evaluator_success must be a boolean")
        key = EpisodeKey(**dict(row["key"]))
        return cls(
            key=key,
            instruction=str(row["instruction"]),
            evaluator_success=(raw_success if formal else bool(raw_success)),
            policy_status=str(row["policy_status"]),
            steps=int(row["steps"]),
            elapsed_s=float(row["elapsed_s"]),
            seed=int(row["seed"]) if row.get("seed") is not None else None,
            failure=str(row["failure"]) if row.get("failure") is not None else None,
            video_paths=dict(row.get("video_paths", {})),
            route_trace=dict(row.get("route_trace", {})),
            provenance_sha256=(
                str(row["provenance_sha256"]) if formal else None
            ),
            lifecycle=(dict(row["lifecycle"]) if formal else None),
            created_at=str(row.get("created_at", "")),
        )


def episode_schedule(
    route: str,
    suite: str,
    task_ids: Iterable[int],
    episodes_per_task: int,
    init_state_start: int = 0,
) -> tuple[EpisodeKey, ...]:
    """Return a deterministic task-major schedule.

    Ten task IDs with ``episodes_per_task=5`` is exactly the requested 50
    episodes.  Task-major ordering also makes partial per-task metrics useful
    while a long run is still in progress.
    """

    ids = tuple(dict.fromkeys(int(task_id) for task_id in task_ids))
    if not ids or any(task_id < 0 for task_id in ids):
        raise ValueError("task_ids must contain unique non-negative ids")
    if episodes_per_task < 1:
        raise ValueError("episodes_per_task must be positive")
    if init_state_start < 0:
        raise ValueError("init_state_start must be non-negative")
    return tuple(
        EpisodeKey(route, suite, task_id, episode_index)
        for task_id in ids
        for episode_index in range(
            init_state_start,
            init_state_start + episodes_per_task,
        )
    )


class JsonlResultStore:
    """Append episode results and derive a compact summary atomically."""

    def __init__(
        self,
        trace_path: str | Path,
        summary_path: str | Path,
        *,
        formal_provenance_sha256: str | None = None,
        expected_episode_ids: Iterable[str] | None = None,
        source_check: Callable[[], str] | None = None,
    ) -> None:
        self.trace_path = Path(trace_path)
        self.summary_path = Path(summary_path)
        self.formal_provenance_sha256 = formal_provenance_sha256
        self.expected_episode_ids = (
            frozenset(expected_episode_ids)
            if expected_episode_ids is not None
            else None
        )
        self.source_check = source_check

    def records(self) -> tuple[EpisodeRecord, ...]:
        if not self.trace_path.exists():
            return ()
        rows: list[EpisodeRecord] = []
        lines = self.trace_path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                rows.append(EpisodeRecord.from_dict(json.loads(line)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                # A process can be killed between a write and fsync.  Only a
                # malformed final line is safely treated as a partial append.
                if index != len(lines) - 1:
                    raise ValueError(
                        f"malformed evaluation JSONL at {self.trace_path}:{index + 1}"
                    ) from None
        return tuple(rows)

    def completed_ids(self) -> frozenset[str]:
        return frozenset(record.key.id for record in self.records())

    def _prepare_trace_for_append(self) -> None:
        """Make a crash-truncated tail safe before appending another row.

        ``records`` deliberately ignores one malformed final line because a
        process can be killed during the last write.  Appending directly after
        that fragment would turn it into a malformed *middle* line and make
        every later resume unreadable.  Validate existing non-empty lines,
        discard only a malformed final fragment, and ensure a valid final row
        is newline terminated before the next append.
        """

        if not self.trace_path.exists():
            return
        payload = self.trace_path.read_bytes()
        if not payload:
            return
        lines = payload.splitlines(keepends=True)
        nonempty = [index for index, line in enumerate(lines) if line.strip()]
        if not nonempty:
            return
        final_nonempty = nonempty[-1]
        offsets: list[int] = []
        offset = 0
        for line in lines:
            offsets.append(offset)
            offset += len(line)
        for index in nonempty:
            try:
                json.loads(lines[index])
            except (UnicodeDecodeError, json.JSONDecodeError):
                if index != final_nonempty:
                    raise ValueError(
                        f"malformed evaluation JSONL at {self.trace_path}:{index + 1}"
                    ) from None
                with self.trace_path.open("r+b") as stream:
                    stream.truncate(offsets[index])
                    stream.flush()
                    os.fsync(stream.fileno())
                return
        if not payload.endswith((b"\n", b"\r")):
            with self.trace_path.open("ab") as stream:
                stream.write(b"\n")
                stream.flush()
                os.fsync(stream.fileno())

    def append(self, record: EpisodeRecord) -> None:
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._prepare_trace_for_append()
        if self.formal_provenance_sha256 is not None:
            if self.source_check is None:
                raise ValueError("formal result store requires a source check")
            self.source_check()
            if record.provenance_sha256 != self.formal_provenance_sha256:
                raise ValueError("episode provenance does not match formal result store")
            if (
                self.expected_episode_ids is None
                or record.key.id not in self.expected_episode_ids
            ):
                raise ValueError(f"unscheduled formal episode {record.key.id}")
            existing_records = self.records()
            if record.key.id in {item.key.id for item in existing_records}:
                raise ValueError(f"duplicate formal episode {record.key.id}")
            if existing_records:
                previous_lifecycle = existing_records[-1].lifecycle
                current_lifecycle = record.lifecycle
                if (
                    not isinstance(previous_lifecycle, Mapping)
                    or not isinstance(current_lifecycle, Mapping)
                    or previous_lifecycle.get("ended_at_unix_ns")
                    > current_lifecycle.get("started_at_unix_ns")
                ):
                    raise ValueError(
                        "formal episode lifecycle overlaps the previous JSONL row"
                    )
        payload = json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True)
        with self.trace_path.open("a", encoding="utf-8") as stream:
            stream.write(payload + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def write_summary(self, *, run_config: Mapping[str, Any]) -> dict[str, Any]:
        records = self.records()
        by_task: dict[int, list[EpisodeRecord]] = {}
        for record in records:
            by_task.setdefault(record.key.task_id, []).append(record)

        def metrics(group: Iterable[EpisodeRecord]) -> dict[str, Any]:
            values = tuple(group)
            successes = sum(record.evaluator_success for record in values)
            return {
                "episodes": len(values),
                "successes": successes,
                "success_rate": successes / len(values) if values else None,
                "steps": sum(record.steps for record in values),
                "elapsed_s": round(sum(record.elapsed_s for record in values), 3),
            }

        formal = self.formal_provenance_sha256 is not None
        summary = {
            "schema": SUMMARY_SCHEMA if formal else "libero-routes-eval.v1",
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "run_config": dict(run_config),
            "overall": metrics(records),
            "per_task": {str(task_id): metrics(rows) for task_id, rows in sorted(by_task.items())},
        }
        if formal:
            summary["provenance_sha256"] = self.formal_provenance_sha256
            summary["episodes_jsonl_sha256"] = (
                file_sha256(self.trace_path)
                if self.trace_path.is_file()
                else hashlib.sha256(b"").hexdigest()
            )
            summary["episode_rows"] = len(records)
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=self.summary_path.parent,
            prefix=f".{self.summary_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            temporary = Path(stream.name)
        temporary.replace(self.summary_path)
        return summary
