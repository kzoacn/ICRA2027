"""Route-B boundary types.

The types in this module deliberately contain sensor data only.  In particular,
there is no field for a simulator object pose, segmentation id, BDDL state, or
success flag.  A runtime may adapt its native observation to these types at the
policy boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.floating]
UInt8Array = npt.NDArray[np.uint8]
BoolArray = npt.NDArray[np.bool_]


def _finite_array(value: npt.ArrayLike, shape: tuple[int, ...], name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


@dataclass(frozen=True)
class Pose:
    """A rigid transform expressed in the world frame."""

    position: FloatArray
    rotation: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _finite_array(self.position, (3,), "position"))
        rotation = _finite_array(self.rotation, (3, 3), "rotation")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ValueError("rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
            raise ValueError("rotation must have determinant +1")
        object.__setattr__(self, "rotation", rotation)

    @classmethod
    def identity(cls, position: npt.ArrayLike = (0.0, 0.0, 0.0)) -> "Pose":
        return cls(np.asarray(position, dtype=np.float64), np.eye(3, dtype=np.float64))

    @classmethod
    def from_matrix(cls, matrix: npt.ArrayLike) -> "Pose":
        transform = _finite_array(matrix, (4, 4), "matrix")
        if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
            raise ValueError("matrix must be a homogeneous transform")
        return cls(transform[:3, 3], transform[:3, :3])

    @property
    def matrix(self) -> FloatArray:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = self.rotation
        transform[:3, 3] = self.position
        return transform

    def transform_points(self, points: npt.ArrayLike) -> FloatArray:
        points_array = np.asarray(points, dtype=np.float64)
        if points_array.ndim != 2 or points_array.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        return points_array @ self.rotation.T + self.position


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("focal lengths must be positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image dimensions must be positive")
        if not all(np.isfinite(v) for v in (self.fx, self.fy, self.cx, self.cy)):
            raise ValueError("intrinsics must be finite")


@dataclass(frozen=True)
class RGBDFrame:
    """Calibrated RGB-D frame; depth is metric camera-z depth."""

    rgb: UInt8Array
    depth_m: FloatArray
    intrinsics: CameraIntrinsics
    world_from_camera: Pose
    observation_v_flipped: bool = False

    def __post_init__(self) -> None:
        rgb = np.asarray(self.rgb)
        depth = np.asarray(self.depth_m, dtype=np.float64)
        expected_hw = (self.intrinsics.height, self.intrinsics.width)
        if rgb.shape != (*expected_hw, 3):
            raise ValueError(f"rgb must have shape {(*expected_hw, 3)}; got {rgb.shape}")
        if rgb.dtype != np.uint8:
            raise ValueError("rgb must have dtype uint8")
        if depth.shape != expected_hw:
            raise ValueError(f"depth_m must have shape {expected_hw}; got {depth.shape}")
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "depth_m", depth)
        object.__setattr__(self, "observation_v_flipped", bool(self.observation_v_flipped))


@dataclass(frozen=True)
class RobotState:
    ee_pose: Pose
    gripper_width_m: float
    joint_position: FloatArray | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.gripper_width_m) or self.gripper_width_m < 0:
            raise ValueError("gripper_width_m must be finite and non-negative")
        if self.joint_position is not None:
            object.__setattr__(self, "joint_position",
                               _finite_array(self.joint_position, (7,), "joint_position"))


@dataclass(frozen=True)
class SensorObservation:
    """The only observation accepted by the route-B controller.

    Capture ordering is owned inside perception/controller state.  Keeping it
    out of this value prevents an environment clock or episode counter from
    becoming a policy input.
    """

    cameras: Mapping[str, RGBDFrame]
    robot: RobotState

    def __post_init__(self) -> None:
        if not self.cameras:
            raise ValueError("at least one calibrated RGB-D camera is required")


@dataclass(frozen=True)
class RegionDetection:
    """Output of a frozen open-vocabulary region model."""

    query: str
    mask: BoolArray
    score: float

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask, dtype=bool)
        if mask.ndim != 2:
            raise ValueError("mask must be two-dimensional")
        if not np.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be in [0, 1]")
        object.__setattr__(self, "mask", mask)


@dataclass(frozen=True)
class SceneObject:
    """Geometry reconstructed exclusively from masks and RGB-D."""

    name: str
    centroid_world: FloatArray
    axes_world: FloatArray
    extents_m: FloatArray
    bounds_min_world: FloatArray
    bounds_max_world: FloatArray
    confidence: float
    point_count: int
    surface_points_world: FloatArray | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "centroid_world", _finite_array(self.centroid_world, (3,), "centroid_world")
        )
        axes = _finite_array(self.axes_world, (3, 3), "axes_world")
        if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-5):
            raise ValueError("axes_world must be orthonormal")
        object.__setattr__(self, "axes_world", axes)
        extents = _finite_array(self.extents_m, (3,), "extents_m")
        if np.any(extents < 0):
            raise ValueError("extents_m cannot be negative")
        object.__setattr__(self, "extents_m", extents)
        bounds_min = _finite_array(self.bounds_min_world, (3,), "bounds_min_world")
        bounds_max = _finite_array(self.bounds_max_world, (3,), "bounds_max_world")
        if np.any(bounds_min > bounds_max):
            raise ValueError("bounds_min_world cannot exceed bounds_max_world")
        object.__setattr__(self, "bounds_min_world", bounds_min)
        object.__setattr__(self, "bounds_max_world", bounds_max)
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.point_count <= 0:
            raise ValueError("point_count must be positive")
        points = self.surface_points_world
        if points is not None:
            points = np.asarray(points, dtype=np.float64)
            if (
                points.ndim != 2
                or points.shape[1:] != (3,)
                or len(points) < 3
                or not np.all(np.isfinite(points))
            ):
                raise ValueError(
                    "surface_points_world must be a finite Nx3 RGB-D cloud"
                )
            object.__setattr__(self, "surface_points_world", points.copy())

    @property
    def height_m(self) -> float:
        return float(self.bounds_max_world[2] - self.bounds_min_world[2])


@dataclass(frozen=True)
class SceneSnapshot:
    timestamp_s: float
    objects: Mapping[str, SceneObject]

    def require(self, name: str) -> SceneObject:
        try:
            return self.objects[name]
        except KeyError as exc:
            raise KeyError(f"scene object not observed: {name!r}") from exc


class ExecutorStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class ControlDecision:
    """One normalized LIBERO OSC_POSE action plus transparent controller state."""

    action: FloatArray
    status: ExecutorStatus
    skill_index: int
    phase: str
    message: str = ""

    def __post_init__(self) -> None:
        action = np.asarray(self.action, dtype=np.float32)
        if action.shape != (7,):
            raise ValueError(f"OSC_POSE action must have shape (7,); got {action.shape}")
        if not np.all(np.isfinite(action)):
            raise ValueError("action must contain only finite values")
        if np.any(action < -1.0) or np.any(action > 1.0):
            raise ValueError("action must be normalized to [-1, 1]")
        object.__setattr__(self, "action", action)
