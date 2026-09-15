"""Policy and native LIBERO OSC action contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .observation import RobotObservation


@dataclass(frozen=True, slots=True)
class PolicyTask:
    """Information exposed to a policy (intentionally excludes benchmark ids)."""

    instruction: str
    episode_id: str


@dataclass(frozen=True, slots=True)
class OSCAction:
    """Native normalized robosuite OSC_POSE action.

    Components are ``dx,dy,dz,dRx,dRy,dRz,gripper`` in [-1, 1].  With the
    installed controller, unit translation maps to 0.05 m and unit rotation to
    0.5 rad in a 20 Hz policy step.  Panda gripper semantics are -1=open,
    +1=close.  This is a command, not an open-fraction.
    """

    values: NDArray[np.float32]

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float32)
        if values.shape != (7,):
            raise ValueError(f"OSC action must have shape (7,), got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("OSC action contains non-finite values")
        if np.any(values < -1.0) or np.any(values > 1.0):
            raise ValueError("OSC action must already be normalized to [-1, 1]")
        values = np.array(values, copy=True)
        values.setflags(write=False)
        object.__setattr__(self, "values", values)

    @classmethod
    def from_array(cls, values: ArrayLike, *, clip: bool = False) -> "OSCAction":
        array = np.asarray(values, dtype=np.float32)
        if clip:
            array = np.clip(array, -1.0, 1.0)
        return cls(array)

    @classmethod
    def hold(cls, gripper_command: float = -1.0) -> "OSCAction":
        values = np.zeros(7, dtype=np.float32)
        values[-1] = gripper_command
        return cls(values)


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    action: OSCAction
    request_stop: bool = False
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class Policy(Protocol):
    def reset(self, task: PolicyTask) -> None: ...

    def act(self, observation: RobotObservation) -> PolicyDecision: ...

    def close(self) -> None: ...

