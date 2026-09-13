"""Sensor-only observations shared by the LIBERO control routes.

The types in this module deliberately cannot represent simulator object poses,
segmentation ids, or BDDL predicates.  This makes the policy boundary auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
from numpy.typing import NDArray


def _readonly_array(value: object, shape: tuple[int, ...], dtype: np.dtype | type) -> NDArray:
    array = np.asarray(value, dtype=dtype)
    if array.shape != shape:
        raise ValueError(f"expected shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("sensor arrays must contain only finite values")
    array = np.array(array, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class CameraCalibration:
    """Pinhole calibration using OpenCV camera axes and world z-up.

    ``T_world_camera`` maps ``[x right, y down, z forward]`` camera points to
    MuJoCo world coordinates.  LIBERO's default OpenGL observation array is
    visually upright, but its row index is vertically reversed relative to the
    OpenCV projection convention; ``observation_v_flipped`` records that pixel
    transform without rotating the image supplied to vision models.
    """

    name: str
    width: int
    height: int
    intrinsic: NDArray[np.float64]
    T_world_camera: NDArray[np.float64]
    observation_v_flipped: bool = False

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera dimensions must be positive")
        object.__setattr__(self, "intrinsic", _readonly_array(self.intrinsic, (3, 3), np.float64))
        transform = _readonly_array(self.T_world_camera, (4, 4), np.float64)
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
            raise ValueError("T_world_camera must be homogeneous")
        if not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5):
            raise ValueError("T_world_camera rotation must be orthonormal")
        object.__setattr__(self, "T_world_camera", transform)

    @property
    def T_camera_world(self) -> NDArray[np.float64]:
        result = np.linalg.inv(self.T_world_camera)
        result.setflags(write=False)
        return result

    @property
    def fx(self) -> float:
        return float(self.intrinsic[0, 0])

    @property
    def fy(self) -> float:
        return float(self.intrinsic[1, 1])

    @property
    def cx(self) -> float:
        return float(self.intrinsic[0, 2])

    @property
    def cy(self) -> float:
        return float(self.intrinsic[1, 2])


@dataclass(frozen=True, slots=True)
class CameraFrame:
    """Visually upright HWC RGB plus aligned metric depth."""

    rgb: NDArray[np.uint8]
    depth_m: NDArray[np.float32]
    calibration: CameraCalibration

    def __post_init__(self) -> None:
        height, width = self.calibration.height, self.calibration.width
        rgb = np.asarray(self.rgb)
        if rgb.shape != (height, width, 3) or rgb.dtype != np.uint8:
            raise ValueError(f"rgb must be uint8 HWC {(height, width, 3)}, got {rgb.shape}/{rgb.dtype}")
        depth = np.asarray(self.depth_m, dtype=np.float32)
        if depth.shape != (height, width):
            raise ValueError(f"depth_m must be HW {(height, width)}, got {depth.shape}")
        if not np.all(np.isfinite(depth)) or np.any(depth <= 0):
            raise ValueError("depth_m must contain finite positive metric depths")
        rgb = np.array(rgb, copy=True)
        depth = np.array(depth, copy=True)
        rgb.setflags(write=False)
        depth.setflags(write=False)
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "depth_m", depth)


@dataclass(frozen=True, slots=True)
class Proprioception:
    """Robot-side measurements only; no task or object state is present."""

    T_world_ee: NDArray[np.float64]
    ee_quat_xyzw: NDArray[np.float64]
    joint_position: NDArray[np.float64]
    joint_velocity: NDArray[np.float64]
    gripper_qpos: NDArray[np.float64]
    gripper_qvel: NDArray[np.float64]
    gripper_width_m: float
    ee_force_sensor: NDArray[np.float64]
    ee_torque_sensor: NDArray[np.float64]

    def __post_init__(self) -> None:
        object.__setattr__(self, "T_world_ee", _readonly_array(self.T_world_ee, (4, 4), np.float64))
        object.__setattr__(self, "ee_quat_xyzw", _readonly_array(self.ee_quat_xyzw, (4,), np.float64))
        object.__setattr__(self, "joint_position", _readonly_array(self.joint_position, (7,), np.float64))
        object.__setattr__(self, "joint_velocity", _readonly_array(self.joint_velocity, (7,), np.float64))
        object.__setattr__(self, "gripper_qpos", _readonly_array(self.gripper_qpos, (2,), np.float64))
        object.__setattr__(self, "gripper_qvel", _readonly_array(self.gripper_qvel, (2,), np.float64))
        object.__setattr__(self, "ee_force_sensor", _readonly_array(self.ee_force_sensor, (3,), np.float64))
        object.__setattr__(self, "ee_torque_sensor", _readonly_array(self.ee_torque_sensor, (3,), np.float64))
        if not np.isfinite(self.gripper_width_m) or not 0.0 <= self.gripper_width_m <= 0.12:
            raise ValueError("gripper_width_m is outside the Panda's physical range")

    @property
    def ee_position_world(self) -> NDArray[np.float64]:
        return self.T_world_ee[:3, 3]


@dataclass(frozen=True, slots=True)
class RobotObservation:
    """The complete, sensor-only policy input."""

    cameras: Mapping[str, CameraFrame]
    proprio: Proprioception

    def __post_init__(self) -> None:
        cameras = dict(self.cameras)
        required = {"agentview", "wrist"}
        if set(cameras) != required:
            raise ValueError(f"expected exactly cameras {sorted(required)}, got {sorted(cameras)}")
        object.__setattr__(self, "cameras", MappingProxyType(cameras))

    @property
    def agentview(self) -> CameraFrame:
        return self.cameras["agentview"]

    @property
    def wrist(self) -> CameraFrame:
        return self.cameras["wrist"]
