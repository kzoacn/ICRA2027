"""Sensor-only contracts for articulated-fixture and planar-push skills.

None of these types can carry simulator bodies, joints, contacts, BDDL state,
or evaluator success.  Targets are reconstructed from calibrated RGB-D only.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _vector(value: ArrayLike, name: str, *, unit: bool = False) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite 3-vector")
    if unit:
        norm = float(np.linalg.norm(array))
        if norm < 1e-9:
            raise ValueError(f"{name} cannot be zero")
        array = array / norm
    return np.array(array, copy=True)


class GoalSkillKind(str, Enum):
    OPEN_DRAWER = "open_drawer"
    CLOSE_DRAWER = "close_drawer"
    TURN_KNOB = "turn_knob"
    PUSH_OBJECT = "push_object"
    OPEN_MICROWAVE = "open_microwave"
    CLOSE_MICROWAVE = "close_microwave"
    ROUTE_B_PLACE_IN = "route_b_place_in"


class GoalExecutorStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    HANDOFF = "handoff"


@dataclass(frozen=True, slots=True)
class GoalSkillStep:
    kind: GoalSkillKind
    subject: str
    target: str | None = None
    level: str | None = None
    # Signed language intent for rotary controls.  Existing turn-on plans keep
    # +1; turn-off uses the mechanically opposite -1 direction.
    turn_direction: int = 1

    def __post_init__(self) -> None:
        if self.turn_direction not in {-1, 1}:
            raise ValueError("turn_direction must be -1 or +1")
        if self.kind is not GoalSkillKind.TURN_KNOB and self.turn_direction != 1:
            raise ValueError("turn_direction applies only to a knob skill")


@dataclass(frozen=True, slots=True)
class GoalSkillPlan:
    instruction: str
    steps: tuple[GoalSkillStep, ...]

    def __post_init__(self) -> None:
        if not self.instruction.strip() or not self.steps:
            raise ValueError("a goal plan needs language and at least one step")


@dataclass(frozen=True, slots=True)
class ContactTarget:
    """A fixture contact point and manipulation frame estimated from RGB-D."""

    kind: GoalSkillKind
    point_world: FloatArray
    axis_world: FloatArray
    outward_world: FloatArray
    fixture_center_world: FloatArray
    feature_axis_world: FloatArray
    confidence: float
    source_cameras: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "point_world", _vector(self.point_world, "point_world"))
        object.__setattr__(self, "axis_world", _vector(self.axis_world, "axis_world", unit=True))
        object.__setattr__(self, "outward_world", _vector(self.outward_world, "outward_world", unit=True))
        object.__setattr__(
            self,
            "fixture_center_world",
            _vector(self.fixture_center_world, "fixture_center_world"),
        )
        object.__setattr__(
            self,
            "feature_axis_world",
            _vector(self.feature_axis_world, "feature_axis_world", unit=True),
        )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must lie in [0, 1]")
        if not self.source_cameras:
            raise ValueError("at least one RGB-D source camera is required")


@dataclass(frozen=True, slots=True)
class DrawerEpisodeAnchor:
    """Episode-local identity for one drawer handle reconstructed from RGB-D.

    The anchor contains no simulator identity or articulated state.  It binds
    a language level to the metric appearance frame first observed by the
    policy, so a later view can keep the same physical handle even when
    occlusion changes the current bottom/middle/top ordering.
    """

    level: str
    target: ContactTarget

    def __post_init__(self) -> None:
        if self.level not in {"top", "middle", "bottom"}:
            raise ValueError("drawer anchor level must be top, middle, or bottom")
        if self.target.kind not in {
            GoalSkillKind.OPEN_DRAWER,
            GoalSkillKind.CLOSE_DRAWER,
        }:
            raise ValueError("drawer anchor target must describe a drawer handle")


@dataclass(frozen=True, slots=True)
class PushTarget:
    """Start/end geometry for a sensor-derived planar push."""

    object_center_world: FloatArray
    target_center_world: FloatArray
    direction_world: FloatArray
    object_radius_m: float
    confidence: float
    source_cameras: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "object_center_world",
            _vector(self.object_center_world, "object_center_world"),
        )
        object.__setattr__(
            self,
            "target_center_world",
            _vector(self.target_center_world, "target_center_world"),
        )
        object.__setattr__(
            self,
            "direction_world",
            _vector(self.direction_world, "direction_world", unit=True),
        )
        if not 0.005 <= self.object_radius_m <= 0.20:
            raise ValueError("object_radius_m is implausible")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must lie in [0, 1]")
        if not self.source_cameras:
            raise ValueError("at least one RGB-D source camera is required")
