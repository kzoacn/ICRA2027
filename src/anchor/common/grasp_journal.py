"""Route-owned records for completed mechanical grasp engagements.

This is deliberately smaller than the integration audit schema.  A route
owns and resets the journal, while integration code may later attach its
route name and opaque episode token.  Consequently these records cannot carry
benchmark, simulator, evaluator, task-id, or initial-state metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re


_BLACK_BOWL = re.compile(r"(?<!\w)black[\s-]+bowl(?!\w)")
_GRASP_MODES = frozenset({"pinch", "rim_pinch", "expand"})


class JawBehavior(StrEnum):
    """Direction of the one mechanical jaw engagement."""

    CLOSE_FINGERS = "close_fingers"
    OPEN_FINGERS_INTERIOR_BRACE = "open_fingers_interior_brace"


class GraspOutcome(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class GraspEvidenceCategory(StrEnum):
    """Policy-available evidence used to terminate the engagement cycle."""

    RGBD_GEOMETRY = "rgbd_geometry"
    PROPRIOCEPTION = "proprioception"
    RGBD_AND_PROPRIOCEPTION = "rgbd_and_proprioception"
    CONTROLLER_EXECUTION = "controller_execution"
    SAFETY_GATE = "safety_gate"


class GraspReason(StrEnum):
    """Generic route-independent terminal reason codes."""

    ACCEPTED = "accepted"
    TRAJECTORY_REJECTED = "trajectory_rejected"
    EXECUTION_STALLED = "execution_stalled"
    JAW_COMMAND_REJECTED = "jaw_command_rejected"
    RETENTION_REJECTED = "retention_rejected"
    SAFETY_REJECTED = "safety_rejected"
    STEP_BUDGET_EXHAUSTED = "step_budget_exhausted"


def _semantic(value: str, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    normalized = " ".join(value.lower().replace("_", " ").split())
    if not normalized:
        raise ValueError(f"{field} must not be empty")
    if any(ord(character) < 32 for character in normalized):
        raise ValueError(f"{field} must not contain control characters")
    return normalized


@dataclass(frozen=True, slots=True)
class PendingGraspEngagement:
    """Immutable allowed-field snapshot of one issued, unfinished engagement."""

    attempt_index: int
    source_text: str
    source_class: str
    grasp_mode: str
    jaw_behavior: JawBehavior

    def __post_init__(self) -> None:
        if type(self.attempt_index) is not int or self.attempt_index < 1:
            raise ValueError("attempt_index must be a positive one-based integer")
        object.__setattr__(
            self, "source_text", _semantic(self.source_text, "source_text")
        )
        object.__setattr__(
            self, "source_class", _semantic(self.source_class, "source_class")
        )
        if self.grasp_mode not in _GRASP_MODES:
            raise ValueError(f"unknown grasp_mode {self.grasp_mode!r}")
        if not isinstance(self.jaw_behavior, JawBehavior):
            raise TypeError("jaw_behavior must be a JawBehavior")
        expected_jaw = (
            JawBehavior.OPEN_FINGERS_INTERIOR_BRACE
            if self.grasp_mode == "expand"
            else JawBehavior.CLOSE_FINGERS
        )
        if self.jaw_behavior is not expected_jaw:
            raise ValueError(
                f"grasp_mode={self.grasp_mode!r} requires "
                f"jaw_behavior={expected_jaw.value!r}"
            )
        black_bowl = bool(
            self.source_class.replace("-", " ") in {"black bowl", "bowl"}
            or _BLACK_BOWL.search(self.source_text.replace("-", " "))
        )
        if black_bowl and (
            self.grasp_mode != "rim_pinch"
            or self.jaw_behavior is not JawBehavior.CLOSE_FINGERS
        ):
            raise ValueError(
                "black bowl engagements require closed-finger rim_pinch; "
                "expansion and interior bracing are forbidden"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "attempt_index": self.attempt_index,
            "source_text": self.source_text,
            "source_class": self.source_class,
            "grasp_mode": self.grasp_mode,
            "jaw_behavior": self.jaw_behavior.value,
        }


@dataclass(frozen=True, slots=True)
class GraspAttemptEvent:
    """One terminal record for one physical finger engagement cycle."""

    attempt_index: int
    source_text: str
    source_class: str
    grasp_mode: str
    jaw_behavior: JawBehavior
    accepted: bool
    outcome: GraspOutcome
    reason: GraspReason
    evidence_source: GraspEvidenceCategory

    def __post_init__(self) -> None:
        if type(self.attempt_index) is not int or self.attempt_index < 1:
            raise ValueError("attempt_index must be a positive one-based integer")
        object.__setattr__(self, "source_text", _semantic(self.source_text, "source_text"))
        object.__setattr__(self, "source_class", _semantic(self.source_class, "source_class"))
        if self.grasp_mode not in _GRASP_MODES:
            raise ValueError(f"unknown grasp_mode {self.grasp_mode!r}")
        if not isinstance(self.jaw_behavior, JawBehavior):
            raise TypeError("jaw_behavior must be a JawBehavior")
        if type(self.accepted) is not bool:
            raise TypeError("accepted must be a boolean")
        if not isinstance(self.outcome, GraspOutcome):
            raise TypeError("outcome must be a GraspOutcome")
        if not isinstance(self.reason, GraspReason):
            raise TypeError("reason must be a GraspReason")
        if not isinstance(self.evidence_source, GraspEvidenceCategory):
            raise TypeError("evidence_source must be a GraspEvidenceCategory")
        if self.accepted != (self.outcome is GraspOutcome.ACCEPTED):
            raise ValueError("accepted and outcome disagree")
        if self.accepted != (self.reason is GraspReason.ACCEPTED):
            raise ValueError("accepted and reason disagree")

        expected_jaw = (
            JawBehavior.OPEN_FINGERS_INTERIOR_BRACE
            if self.grasp_mode == "expand"
            else JawBehavior.CLOSE_FINGERS
        )
        if self.jaw_behavior is not expected_jaw:
            raise ValueError(
                f"grasp_mode={self.grasp_mode!r} requires "
                f"jaw_behavior={expected_jaw.value!r}"
            )

        black_bowl = bool(
            self.source_class.replace("-", " ") in {"black bowl", "bowl"}
            or _BLACK_BOWL.search(self.source_text.replace("-", " "))
        )
        if black_bowl and (
            self.grasp_mode != "rim_pinch"
            or self.jaw_behavior is not JawBehavior.CLOSE_FINGERS
        ):
            raise ValueError(
                "black bowl engagements require closed-finger rim_pinch; "
                "expansion and interior bracing are forbidden"
            )

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-primitive-only route-local payload."""

        return {
            "attempt_index": self.attempt_index,
            "source_text": self.source_text,
            "source_class": self.source_class,
            "grasp_mode": self.grasp_mode,
            "jaw_behavior": self.jaw_behavior.value,
            "accepted": self.accepted,
            "outcome": self.outcome.value,
            "reason": self.reason.value,
            "evidence_source": self.evidence_source.value,
        }


class GraspAttemptJournal:
    """Append-only, run-local owner of contiguous attempt numbering."""

    def __init__(self) -> None:
        self._records: list[GraspAttemptEvent] = []

    @property
    def records(self) -> tuple[GraspAttemptEvent, ...]:
        return tuple(self._records)

    def reset(self) -> None:
        self._records.clear()

    def append(
        self,
        *,
        source_text: str,
        source_class: str,
        grasp_mode: str,
        jaw_behavior: JawBehavior,
        accepted: bool,
        reason: GraspReason,
        evidence_source: GraspEvidenceCategory,
    ) -> GraspAttemptEvent:
        record = GraspAttemptEvent(
            attempt_index=len(self._records) + 1,
            source_text=source_text,
            source_class=source_class,
            grasp_mode=grasp_mode,
            jaw_behavior=jaw_behavior,
            accepted=accepted,
            outcome=(
                GraspOutcome.ACCEPTED if accepted else GraspOutcome.REJECTED
            ),
            reason=reason,
            evidence_source=evidence_source,
        )
        self._records.append(record)
        return record


__all__ = [
    "GraspAttemptEvent",
    "GraspAttemptJournal",
    "GraspEvidenceCategory",
    "GraspOutcome",
    "GraspReason",
    "JawBehavior",
    "PendingGraspEngagement",
]
