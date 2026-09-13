"""Small policy-side result DTO shared across the Route-C process boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from libero_system.common.grasp_journal import (
    GraspAttemptEvent,
    PendingGraspEngagement,
)


@dataclass(frozen=True, slots=True)
class RouteCPolicyExecution:
    """Policy result containing no evaluator-owned score or termination signal."""

    success: bool
    failure: str | None
    steps_executed: int
    trace: Mapping[str, Any]
    grasp_attempt_events: tuple[GraspAttemptEvent, ...]
    pending_grasp_engagement: PendingGraspEngagement | None

    def __post_init__(self) -> None:
        if type(self.success) is not bool:
            raise TypeError("Route C policy success must be a native boolean")
        if self.failure is not None and type(self.failure) is not str:
            raise TypeError("Route C policy failure must be a string or None")
        if type(self.steps_executed) is not int or self.steps_executed < 0:
            raise ValueError("Route C policy steps_executed must be non-negative")
        if not isinstance(self.trace, Mapping):
            raise TypeError("Route C policy trace must be a mapping")
        if not all(
            type(item) is GraspAttemptEvent for item in self.grasp_attempt_events
        ):
            raise TypeError("Route C grasp events must use exact common DTOs")
        if self.pending_grasp_engagement is not None and type(
            self.pending_grasp_engagement
        ) is not PendingGraspEngagement:
            raise TypeError(
                "Route C pending engagement must use the exact common DTO"
            )


__all__ = ["RouteCPolicyExecution"]
