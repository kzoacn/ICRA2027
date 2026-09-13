"""Uniform, policy-boundary-safe grasp-attempt audit records.

The runtime adapters for Routes B and C expose different diagnostic shapes.
This module defines the small common record that formal result code can build
from those diagnostics.  It intentionally has no suite, task, initial-state,
simulator, oracle, reward, or evaluator fields.

Records are immutable and an :class:`GraspAuditTrail` can only append the next
episode-global attempt.  Canonical JSON uses sorted keys and rejects NaN so the
same record always has the same serialized bytes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
import json
import re
from typing import TypeVar


GRASP_AUDIT_SCHEMA = "libero-grasp-attempt.v1"
GRASP_AUDIT_REPORT_SCHEMA = "libero-grasp-audit-report.v1"
GRASP_ATTEMPT_COUNTS_SCHEMA = "libero-grasp-attempt-counts.v1"

_OPAQUE_EPISODE_ID = re.compile(
    r"episode-(?:[0-9a-f]{20}|v2-[0-9a-f]{64})"
)
_BLACK_BOWL = re.compile(r"(?<![a-z0-9])black[\s-]+bowl(?![a-z0-9])")
_FORBIDDEN_SEMANTIC_METADATA = re.compile(
    r"(?:^|\s)(?:libero(?:[_-][a-z0-9_-]+)?|suite|task(?:[_-]?id)?|"
    r"init(?:ial[_-]?state|[_-]?id)?|episode(?:[_-]?id)?|oracle|evaluator)"
    r"(?:$|\s|[:#_-]*\d)",
)


class GraspAuditValidationError(ValueError):
    """Raised when a grasp audit is incomplete or violates its safety rules."""


class AuditRoute(StrEnum):
    B = "b"
    C = "c"


class ObservedGraspMode(StrEnum):
    PINCH = "pinch"
    RIM_PINCH = "rim_pinch"
    EXPAND = "expand"


class CommandedJawBehavior(StrEnum):
    """Mechanical engagement command for one grasp attempt."""

    CLOSE_FINGERS = "close_fingers"
    OPEN_FINGERS_INTERIOR_BRACE = "open_fingers_interior_brace"


class GraspEvidenceSource(StrEnum):
    """Allowed sensor/controller evidence categories, never evaluator state."""

    RGBD_GEOMETRY = "rgbd_geometry"
    PROPRIOCEPTION = "proprioception"
    RGBD_AND_PROPRIOCEPTION = "rgbd_and_proprioception"
    CONTROLLER_EXECUTION = "controller_execution"
    SAFETY_GATE = "safety_gate"


class GraspReasonCode(StrEnum):
    """Route-independent terminal reason for one grasp attempt."""

    ACCEPTED = "accepted"
    PERCEPTION_REJECTED = "perception_rejected"
    GEOMETRY_REJECTED = "geometry_rejected"
    TRAJECTORY_REJECTED = "trajectory_rejected"
    EXECUTION_STALLED = "execution_stalled"
    JAW_COMMAND_REJECTED = "jaw_command_rejected"
    RETENTION_REJECTED = "retention_rejected"
    IDENTITY_REJECTED = "identity_rejected"
    AMBIGUOUS_EVIDENCE = "ambiguous_evidence"
    SAFETY_REJECTED = "safety_rejected"
    STEP_BUDGET_EXHAUSTED = "step_budget_exhausted"


_EnumT = TypeVar("_EnumT", bound=StrEnum)


def _semantic_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise GraspAuditValidationError(f"{field_name} must be a string")
    if any(ord(character) < 32 for character in value):
        raise GraspAuditValidationError(
            f"{field_name} must not contain control characters"
        )
    normalized = " ".join(value.lower().replace("_", " ").split())
    if not normalized or len(normalized) > 256:
        raise GraspAuditValidationError(
            f"{field_name} must contain 1 to 256 normalized characters"
        )
    if _FORBIDDEN_SEMANTIC_METADATA.search(normalized):
        raise GraspAuditValidationError(
            f"{field_name} contains evaluator-side metadata"
        )
    return normalized


def validate_opaque_episode_id(value: object) -> str:
    """Require the exact non-descriptive token exposed at the policy boundary."""

    if not isinstance(value, str) or _OPAQUE_EPISODE_ID.fullmatch(value) is None:
        raise GraspAuditValidationError(
            "episode_id must be a canonical opaque episode token"
        )
    return value


def _enum_value(
    enum_type: type[_EnumT], value: object, field_name: str
) -> _EnumT:
    if not isinstance(value, str):
        raise GraspAuditValidationError(f"{field_name} must be a string")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise GraspAuditValidationError(
            f"unknown {field_name} {value!r}"
        ) from exc


@dataclass(frozen=True, slots=True)
class GraspAuditRecord:
    """One completed mechanical grasp attempt from allowed policy evidence."""

    route: AuditRoute
    episode_id: str
    attempt_index: int
    target_text: str
    target_class: str
    grasp_mode: ObservedGraspMode
    jaw_behavior: CommandedJawBehavior
    evidence_source: GraspEvidenceSource
    accepted: bool
    reason_code: GraspReasonCode
    schema: str = field(default=GRASP_AUDIT_SCHEMA, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.route, AuditRoute):
            raise GraspAuditValidationError("route must be an AuditRoute")
        object.__setattr__(
            self,
            "episode_id",
            validate_opaque_episode_id(self.episode_id),
        )
        if (
            isinstance(self.attempt_index, bool)
            or not isinstance(self.attempt_index, int)
            or self.attempt_index < 1
        ):
            raise GraspAuditValidationError(
                "attempt_index must be a positive, one-based integer"
            )
        object.__setattr__(
            self, "target_text", _semantic_text(self.target_text, "target_text")
        )
        object.__setattr__(
            self,
            "target_class",
            _semantic_text(self.target_class, "target_class"),
        )
        if not isinstance(self.grasp_mode, ObservedGraspMode):
            raise GraspAuditValidationError(
                "grasp_mode must be an ObservedGraspMode"
            )
        if not isinstance(self.jaw_behavior, CommandedJawBehavior):
            raise GraspAuditValidationError(
                "jaw_behavior must be a CommandedJawBehavior"
            )
        if not isinstance(self.evidence_source, GraspEvidenceSource):
            raise GraspAuditValidationError(
                "evidence_source must be a GraspEvidenceSource"
            )
        if type(self.accepted) is not bool:
            raise GraspAuditValidationError("accepted must be a JSON boolean")
        if not isinstance(self.reason_code, GraspReasonCode):
            raise GraspAuditValidationError(
                "reason_code must be a GraspReasonCode"
            )
        self._validate_outcome()
        self._validate_jaw_contract()
        self._validate_black_bowl_contract()

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "GraspAuditRecord":
        """Parse one exact-schema JSON object and reject extra metadata."""

        if not isinstance(payload, Mapping):
            raise GraspAuditValidationError("grasp audit record must be an object")
        required = {
            "schema",
            "route",
            "episode_id",
            "attempt_index",
            "target_text",
            "target_class",
            "grasp_mode",
            "jaw_behavior",
            "evidence_source",
            "accepted",
            "reason_code",
        }
        keys = set(payload)
        missing = sorted(required - keys)
        extra = sorted(keys - required)
        if missing:
            raise GraspAuditValidationError(
                f"grasp audit record is missing fields: {missing}"
            )
        if extra:
            raise GraspAuditValidationError(
                f"grasp audit record has forbidden fields: {extra}"
            )
        if payload["schema"] != GRASP_AUDIT_SCHEMA:
            raise GraspAuditValidationError(
                f"grasp audit schema must be {GRASP_AUDIT_SCHEMA!r}"
            )
        accepted = payload["accepted"]
        if type(accepted) is not bool:
            raise GraspAuditValidationError("accepted must be a JSON boolean")
        attempt_index = payload["attempt_index"]
        if isinstance(attempt_index, bool) or not isinstance(attempt_index, int):
            raise GraspAuditValidationError("attempt_index must be an integer")
        return cls(
            route=_enum_value(AuditRoute, payload["route"], "route"),
            episode_id=validate_opaque_episode_id(payload["episode_id"]),
            attempt_index=attempt_index,
            target_text=_semantic_text(payload["target_text"], "target_text"),
            target_class=_semantic_text(payload["target_class"], "target_class"),
            grasp_mode=_enum_value(
                ObservedGraspMode, payload["grasp_mode"], "grasp_mode"
            ),
            jaw_behavior=_enum_value(
                CommandedJawBehavior,
                payload["jaw_behavior"],
                "jaw_behavior",
            ),
            evidence_source=_enum_value(
                GraspEvidenceSource,
                payload["evidence_source"],
                "evidence_source",
            ),
            accepted=accepted,
            reason_code=_enum_value(
                GraspReasonCode, payload["reason_code"], "reason_code"
            ),
        )

    def _validate_outcome(self) -> None:
        if self.accepted != (self.reason_code is GraspReasonCode.ACCEPTED):
            raise GraspAuditValidationError(
                "accepted records require reason_code='accepted'; rejected records do not"
            )

    def _validate_jaw_contract(self) -> None:
        expected = (
            CommandedJawBehavior.OPEN_FINGERS_INTERIOR_BRACE
            if self.grasp_mode is ObservedGraspMode.EXPAND
            else CommandedJawBehavior.CLOSE_FINGERS
        )
        if self.jaw_behavior is not expected:
            raise GraspAuditValidationError(
                f"grasp_mode={self.grasp_mode.value!r} requires "
                f"jaw_behavior={expected.value!r}"
            )

    def _validate_black_bowl_contract(self) -> None:
        normalized_text = self.target_text.replace("-", " ")
        black_bowl = bool(
            self.target_class.replace("-", " ") in {"black bowl", "bowl"}
            or _BLACK_BOWL.search(normalized_text)
        )
        if not black_bowl:
            return
        if (
            self.grasp_mode is not ObservedGraspMode.RIM_PINCH
            or self.jaw_behavior is not CommandedJawBehavior.CLOSE_FINGERS
        ):
            raise GraspAuditValidationError(
                "black bowl attempts require closed-finger rim_pinch; "
                "expansion and interior bracing are forbidden"
            )

    def to_dict(self) -> dict[str, object]:
        """Return the complete primitive-only representation."""

        return {
            "schema": self.schema,
            "route": self.route.value,
            "episode_id": self.episode_id,
            "attempt_index": self.attempt_index,
            "target_text": self.target_text,
            "target_class": self.target_class,
            "grasp_mode": self.grasp_mode.value,
            "jaw_behavior": self.jaw_behavior.value,
            "evidence_source": self.evidence_source.value,
            "accepted": self.accepted,
            "reason_code": self.reason_code.value,
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )


GraspAuditInput = GraspAuditRecord | Mapping[str, object]


def validate_grasp_audit(
    records: Sequence[GraspAuditInput],
) -> tuple[GraspAuditRecord, ...]:
    """Validate one route/episode trail with exact one-based contiguous indices."""

    if isinstance(records, (str, bytes, bytearray)) or not isinstance(
        records, Sequence
    ):
        raise GraspAuditValidationError("grasp audit must be a sequence")
    parsed = tuple(
        record
        if isinstance(record, GraspAuditRecord)
        else GraspAuditRecord.from_mapping(record)
        for record in records
    )
    if not parsed:
        return parsed
    route = parsed[0].route
    episode_id = parsed[0].episode_id
    for expected_index, record in enumerate(parsed, start=1):
        if record.attempt_index != expected_index:
            raise GraspAuditValidationError(
                "grasp attempt indices must be contiguous and one-based: "
                f"expected {expected_index}, got {record.attempt_index}"
            )
        if record.route is not route:
            raise GraspAuditValidationError(
                "one grasp audit trail cannot mix routes"
            )
        if record.episode_id != episode_id:
            raise GraspAuditValidationError(
                "one grasp audit trail cannot mix opaque episode ids"
            )
    return parsed


def canonical_grasp_audit_jsonl(records: Sequence[GraspAuditInput]) -> str:
    """Serialize a validated audit as deterministic JSON Lines."""

    parsed = validate_grasp_audit(records)
    if not parsed:
        return ""
    return "".join(f"{record.canonical_json()}\n" for record in parsed)


def canonical_grasp_audit_json(records: Sequence[GraspAuditInput]) -> str:
    """Serialize a validated audit as one embeddable canonical JSON array."""

    parsed = validate_grasp_audit(records)
    return json.dumps(
        [record.to_dict() for record in parsed],
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def build_grasp_audit_report(
    route: AuditRoute,
    episode_id: str,
    records: Sequence[GraspAuditInput],
) -> dict[str, object]:
    """Build the exact JSON-native report embedded in a formal route trace."""

    if not isinstance(route, AuditRoute):
        raise GraspAuditValidationError("route must be an AuditRoute")
    opaque_id = validate_opaque_episode_id(episode_id)
    parsed = validate_grasp_audit(records)
    for record in parsed:
        if record.route is not route:
            raise GraspAuditValidationError(
                "grasp audit report route does not match its records"
            )
        if record.episode_id != opaque_id:
            raise GraspAuditValidationError(
                "grasp audit report episode_id does not match its records"
            )
    return {
        "schema": GRASP_AUDIT_REPORT_SCHEMA,
        "formal_pass": True,
        "route": route.value,
        "episode_id": opaque_id,
        "records": [record.to_dict() for record in parsed],
    }


def validate_grasp_audit_report(
    payload: Mapping[str, object],
    *,
    expected_route: AuditRoute | None = None,
    expected_episode_id: str | None = None,
) -> dict[str, object]:
    """Strictly validate one embedded formal report and normalize its records."""

    if not isinstance(payload, Mapping):
        raise GraspAuditValidationError("grasp audit report must be an object")
    required = {"schema", "formal_pass", "route", "episode_id", "records"}
    missing = sorted(required - set(payload))
    extra = sorted(set(payload) - required)
    if missing:
        raise GraspAuditValidationError(
            f"grasp audit report is missing fields: {missing}"
        )
    if extra:
        raise GraspAuditValidationError(
            f"grasp audit report has forbidden fields: {extra}"
        )
    if payload["schema"] != GRASP_AUDIT_REPORT_SCHEMA:
        raise GraspAuditValidationError(
            f"grasp audit report schema must be {GRASP_AUDIT_REPORT_SCHEMA!r}"
        )
    if payload["formal_pass"] is not True:
        raise GraspAuditValidationError("grasp audit report must formally pass")
    route = _enum_value(AuditRoute, payload["route"], "route")
    episode_id = validate_opaque_episode_id(payload["episode_id"])
    raw_records = payload["records"]
    if not isinstance(raw_records, list):
        raise GraspAuditValidationError(
            "grasp audit report records must be a JSON list"
        )
    parsed = validate_grasp_audit(raw_records)
    for record in parsed:
        if record.route is not route:
            raise GraspAuditValidationError(
                "grasp audit report route does not match its records"
            )
        if record.episode_id != episode_id:
            raise GraspAuditValidationError(
                "grasp audit report episode_id does not match its records"
            )
    if expected_route is not None and route is not expected_route:
        raise GraspAuditValidationError(
            "grasp audit report route does not match the episode row"
        )
    if expected_episode_id is not None:
        expected_id = validate_opaque_episode_id(expected_episode_id)
        if episode_id != expected_id:
            raise GraspAuditValidationError(
                "grasp audit report episode_id does not match the episode row"
            )
    return build_grasp_audit_report(route, episode_id, parsed)


def build_grasp_attempt_counts(
    *, completed: int, pending: int
) -> dict[str, object]:
    """Build independent issued-engagement counts for cross-checking a report."""

    for name, value in (("completed", completed), ("pending", pending)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise GraspAuditValidationError(f"{name} must be a non-negative integer")
    if pending > 1:
        raise GraspAuditValidationError(
            "pending grasp engagement count cannot exceed one"
        )
    return {
        "schema": GRASP_ATTEMPT_COUNTS_SCHEMA,
        "completed": completed,
        "pending": pending,
        "issued": completed + pending,
    }


def validate_grasp_attempt_counts(
    payload: Mapping[str, object],
    records: Sequence[GraspAuditInput],
    *,
    route: AuditRoute,
) -> dict[str, object]:
    """Cross-check independent controller counts against persisted audit records."""

    if not isinstance(payload, Mapping):
        raise GraspAuditValidationError("grasp attempt counts must be an object")
    required = {"schema", "completed", "pending", "issued"}
    missing = sorted(required - set(payload))
    extra = sorted(set(payload) - required)
    if missing:
        raise GraspAuditValidationError(
            f"grasp attempt counts are missing fields: {missing}"
        )
    if extra:
        raise GraspAuditValidationError(
            f"grasp attempt counts have forbidden fields: {extra}"
        )
    if payload["schema"] != GRASP_ATTEMPT_COUNTS_SCHEMA:
        raise GraspAuditValidationError(
            f"grasp attempt counts schema must be {GRASP_ATTEMPT_COUNTS_SCHEMA!r}"
        )
    values: dict[str, int] = {}
    for name in ("completed", "pending", "issued"):
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise GraspAuditValidationError(
                f"grasp attempt count {name} must be a non-negative integer"
            )
        values[name] = value
    if values["pending"] > 1:
        raise GraspAuditValidationError(
            "pending grasp engagement count cannot exceed one"
        )
    if values["issued"] != values["completed"] + values["pending"]:
        raise GraspAuditValidationError(
            "issued grasp count must equal completed plus pending"
        )
    parsed = validate_grasp_audit(records)
    if len(parsed) != values["issued"]:
        raise GraspAuditValidationError(
            "grasp audit record count must equal issued grasp count"
        )
    route_b_budget_records = sum(
        record.reason_code is GraspReasonCode.STEP_BUDGET_EXHAUSTED
        for record in parsed
    )
    if route is AuditRoute.B and route_b_budget_records != values["pending"]:
        raise GraspAuditValidationError(
            "Route B step_budget_exhausted records must exactly equal pending count"
        )
    if values["pending"]:
        if route is not AuditRoute.B:
            raise GraspAuditValidationError(
                "only Route B may persist a fixed-horizon pending engagement"
            )
        terminal = parsed[-1]
        if (
            terminal.accepted
            or terminal.reason_code is not GraspReasonCode.STEP_BUDGET_EXHAUSTED
        ):
            raise GraspAuditValidationError(
                "a persisted pending engagement must terminate as step_budget_exhausted"
            )
    return {
        "schema": GRASP_ATTEMPT_COUNTS_SCHEMA,
        **values,
    }


class GraspAuditTrail:
    """Append-only builder that owns episode-global attempt numbering."""

    def __init__(self, route: AuditRoute, episode_id: str) -> None:
        if not isinstance(route, AuditRoute):
            raise GraspAuditValidationError("route must be an AuditRoute")
        self._route = route
        self._episode_id = validate_opaque_episode_id(episode_id)
        self._records: list[GraspAuditRecord] = []

    @property
    def records(self) -> tuple[GraspAuditRecord, ...]:
        return tuple(self._records)

    def append(
        self,
        *,
        target_text: str,
        target_class: str,
        grasp_mode: ObservedGraspMode,
        jaw_behavior: CommandedJawBehavior,
        evidence_source: GraspEvidenceSource,
        accepted: bool,
        reason_code: GraspReasonCode,
    ) -> GraspAuditRecord:
        """Append exactly the next attempt and return its immutable record."""

        record = GraspAuditRecord(
            route=self._route,
            episode_id=self._episode_id,
            attempt_index=len(self._records) + 1,
            target_text=target_text,
            target_class=target_class,
            grasp_mode=grasp_mode,
            jaw_behavior=jaw_behavior,
            evidence_source=evidence_source,
            accepted=accepted,
            reason_code=reason_code,
        )
        self._records.append(record)
        return record

    def append_record(self, record: GraspAuditRecord) -> None:
        """Append a prebuilt record without permitting replacement or gaps."""

        validate_grasp_audit((*self._records, record))
        if record.route is not self._route or record.episode_id != self._episode_id:
            raise GraspAuditValidationError(
                "record does not belong to this grasp audit trail"
            )
        self._records.append(record)

    def to_jsonable(self) -> list[dict[str, object]]:
        return [record.to_dict() for record in self._records]

    def canonical_jsonl(self) -> str:
        return canonical_grasp_audit_jsonl(self._records)

    def canonical_json(self) -> str:
        return canonical_grasp_audit_json(self._records)


__all__ = [
    "AuditRoute",
    "CommandedJawBehavior",
    "GRASP_AUDIT_SCHEMA",
    "GRASP_AUDIT_REPORT_SCHEMA",
    "GRASP_ATTEMPT_COUNTS_SCHEMA",
    "GraspAuditRecord",
    "GraspAuditTrail",
    "GraspAuditValidationError",
    "GraspEvidenceSource",
    "GraspReasonCode",
    "ObservedGraspMode",
    "build_grasp_attempt_counts",
    "build_grasp_audit_report",
    "canonical_grasp_audit_json",
    "canonical_grasp_audit_jsonl",
    "validate_opaque_episode_id",
    "validate_grasp_audit",
    "validate_grasp_attempt_counts",
    "validate_grasp_audit_report",
]
