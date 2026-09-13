"""Strict serializers for the formal evaluator-to-policy process boundary.

The wire format is deliberately smaller than the evaluator's internal state:
it can represent only the two public RGB-D cameras, their calibration, robot
proprioception, and a normalized seven-dimensional OSC action.  In
particular, benchmark identity, clocks, action indices, rewards, termination
signals, evaluator success, and environment objects have no wire fields.

The helpers return ordinary ``dict``/``list``/NumPy values for the bounded
formal socket codec.  Receivers validate exact keys and native scalar types
before reconstructing the immutable common DTOs.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

from libero_system.common import (
    CameraCalibration,
    CameraFrame,
    OSCAction,
    Proprioception,
    RobotObservation,
)


OBSERVATION_WIRE_SCHEMA = "libero-policy-observation-wire.v1"
ACTION_WIRE_SCHEMA = "libero-policy-osc-action-wire.v1"
_CAMERA_KEYS = ("agentview", "wrist")
MAX_CAMERA_DIMENSION = 512
MAX_CAMERA_PIXELS = MAX_CAMERA_DIMENSION * MAX_CAMERA_DIMENSION
_CAMERA_FIELDS = {
    "policy_name",
    "sensor_name",
    "width",
    "height",
    "rgb",
    "depth_m",
    "intrinsic",
    "T_world_camera",
    "observation_v_flipped",
}
_PROPRIO_FIELDS = {
    "T_world_ee",
    "ee_quat_xyzw",
    "joint_position",
    "joint_velocity",
    "gripper_qpos",
    "gripper_qvel",
    "gripper_width_m",
    "ee_force_sensor",
    "ee_torque_sensor",
}


class PolicyIPCValidationError(ValueError):
    """Raised when a process-boundary message is not exactly whitelisted."""


def _exact_dict(value: object, keys: set[str], name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise PolicyIPCValidationError(f"{name} must be an exact dict with fixed keys")
    return value


def _native_int(value: object, name: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise PolicyIPCValidationError(f"{name} must be a native integer >= {minimum}")
    return value


def _native_float(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise PolicyIPCValidationError(f"{name} must be a finite native float")
    return value


def _array(
    value: object,
    *,
    dtype: np.dtype | type,
    shape: tuple[int, ...],
    name: str,
    positive: bool = False,
) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise PolicyIPCValidationError(f"{name} must be an exact NumPy array")
    expected_dtype = np.dtype(dtype)
    if value.dtype != expected_dtype or value.shape != shape:
        raise PolicyIPCValidationError(
            f"{name} must have dtype {expected_dtype} and shape {shape}"
        )
    if np.issubdtype(expected_dtype, np.floating):
        if not np.all(np.isfinite(value)):
            raise PolicyIPCValidationError(f"{name} must contain only finite values")
        if positive and np.any(value <= 0):
            raise PolicyIPCValidationError(f"{name} must contain only positive values")
    return np.array(value, dtype=expected_dtype, copy=True, order="C")


def serialize_policy_observation(observation: RobotObservation) -> dict[str, Any]:
    """Return the exact whitelist payload for one immutable sensor snapshot."""

    if type(observation) is not RobotObservation:
        raise PolicyIPCValidationError("observation must be an exact RobotObservation")
    cameras: dict[str, dict[str, Any]] = {}
    for policy_name in _CAMERA_KEYS:
        frame = observation.cameras[policy_name]
        if type(frame) is not CameraFrame or type(frame.calibration) is not CameraCalibration:
            raise PolicyIPCValidationError("camera values must use exact common DTO types")
        calibration = frame.calibration
        if type(calibration.name) is not str or not calibration.name:
            raise PolicyIPCValidationError("camera sensor_name must be a non-empty string")
        if type(calibration.width) is not int or type(calibration.height) is not int:
            raise PolicyIPCValidationError("camera dimensions must be native integers")
        if (
            calibration.width > MAX_CAMERA_DIMENSION
            or calibration.height > MAX_CAMERA_DIMENSION
            or calibration.width * calibration.height > MAX_CAMERA_PIXELS
        ):
            raise PolicyIPCValidationError(
                f"camera.{policy_name} dimensions exceed the formal wire limit"
            )
        if type(calibration.observation_v_flipped) is not bool:
            raise PolicyIPCValidationError("camera flip flag must be a native boolean")
        cameras[policy_name] = {
            "policy_name": policy_name,
            "sensor_name": calibration.name,
            "width": calibration.width,
            "height": calibration.height,
            "rgb": np.array(frame.rgb, dtype=np.uint8, copy=True, order="C"),
            "depth_m": np.array(frame.depth_m, dtype=np.float32, copy=True, order="C"),
            "intrinsic": np.array(
                calibration.intrinsic, dtype=np.float64, copy=True, order="C"
            ),
            "T_world_camera": np.array(
                calibration.T_world_camera,
                dtype=np.float64,
                copy=True,
                order="C",
            ),
            "observation_v_flipped": calibration.observation_v_flipped,
        }
    proprio = observation.proprio
    if type(proprio) is not Proprioception:
        raise PolicyIPCValidationError("proprio must be an exact Proprioception")
    return {
        "schema": OBSERVATION_WIRE_SCHEMA,
        "cameras": cameras,
        "proprio": {
            "T_world_ee": np.array(
                proprio.T_world_ee, dtype=np.float64, copy=True, order="C"
            ),
            "ee_quat_xyzw": np.array(
                proprio.ee_quat_xyzw, dtype=np.float64, copy=True, order="C"
            ),
            "joint_position": np.array(
                proprio.joint_position, dtype=np.float64, copy=True, order="C"
            ),
            "joint_velocity": np.array(
                proprio.joint_velocity, dtype=np.float64, copy=True, order="C"
            ),
            "gripper_qpos": np.array(
                proprio.gripper_qpos, dtype=np.float64, copy=True, order="C"
            ),
            "gripper_qvel": np.array(
                proprio.gripper_qvel, dtype=np.float64, copy=True, order="C"
            ),
            "gripper_width_m": float(proprio.gripper_width_m),
            "ee_force_sensor": np.array(
                proprio.ee_force_sensor, dtype=np.float64, copy=True, order="C"
            ),
            "ee_torque_sensor": np.array(
                proprio.ee_torque_sensor, dtype=np.float64, copy=True, order="C"
            ),
        },
    }


def deserialize_policy_observation(payload: object) -> RobotObservation:
    """Validate a wire payload and rebuild a detached immutable observation."""

    top = _exact_dict(payload, {"schema", "cameras", "proprio"}, "observation")
    if top["schema"] != OBSERVATION_WIRE_SCHEMA or type(top["schema"]) is not str:
        raise PolicyIPCValidationError("observation wire schema is invalid")
    raw_cameras = _exact_dict(
        top["cameras"], set(_CAMERA_KEYS), "observation.cameras"
    )
    cameras: dict[str, CameraFrame] = {}
    for policy_name in _CAMERA_KEYS:
        raw = _exact_dict(
            raw_cameras[policy_name], _CAMERA_FIELDS, f"camera.{policy_name}"
        )
        if type(raw["policy_name"]) is not str or raw["policy_name"] != policy_name:
            raise PolicyIPCValidationError("camera policy_name is invalid")
        if type(raw["sensor_name"]) is not str or not raw["sensor_name"]:
            raise PolicyIPCValidationError("camera sensor_name is invalid")
        width = _native_int(raw["width"], f"camera.{policy_name}.width")
        height = _native_int(raw["height"], f"camera.{policy_name}.height")
        if (
            width > MAX_CAMERA_DIMENSION
            or height > MAX_CAMERA_DIMENSION
            or width * height > MAX_CAMERA_PIXELS
        ):
            raise PolicyIPCValidationError(
                f"camera.{policy_name} dimensions exceed the formal wire limit"
            )
        if type(raw["observation_v_flipped"]) is not bool:
            raise PolicyIPCValidationError("camera flip flag must be a native boolean")
        calibration = CameraCalibration(
            name=raw["sensor_name"],
            width=width,
            height=height,
            intrinsic=_array(
                raw["intrinsic"],
                dtype=np.float64,
                shape=(3, 3),
                name=f"camera.{policy_name}.intrinsic",
            ),
            T_world_camera=_array(
                raw["T_world_camera"],
                dtype=np.float64,
                shape=(4, 4),
                name=f"camera.{policy_name}.T_world_camera",
            ),
            observation_v_flipped=raw["observation_v_flipped"],
        )
        cameras[policy_name] = CameraFrame(
            rgb=_array(
                raw["rgb"],
                dtype=np.uint8,
                shape=(height, width, 3),
                name=f"camera.{policy_name}.rgb",
            ),
            depth_m=_array(
                raw["depth_m"],
                dtype=np.float32,
                shape=(height, width),
                name=f"camera.{policy_name}.depth_m",
                positive=True,
            ),
            calibration=calibration,
        )
    raw_proprio = _exact_dict(top["proprio"], _PROPRIO_FIELDS, "observation.proprio")
    proprio = Proprioception(
        T_world_ee=_array(
            raw_proprio["T_world_ee"],
            dtype=np.float64,
            shape=(4, 4),
            name="proprio.T_world_ee",
        ),
        ee_quat_xyzw=_array(
            raw_proprio["ee_quat_xyzw"],
            dtype=np.float64,
            shape=(4,),
            name="proprio.ee_quat_xyzw",
        ),
        joint_position=_array(
            raw_proprio["joint_position"],
            dtype=np.float64,
            shape=(7,),
            name="proprio.joint_position",
        ),
        joint_velocity=_array(
            raw_proprio["joint_velocity"],
            dtype=np.float64,
            shape=(7,),
            name="proprio.joint_velocity",
        ),
        gripper_qpos=_array(
            raw_proprio["gripper_qpos"],
            dtype=np.float64,
            shape=(2,),
            name="proprio.gripper_qpos",
        ),
        gripper_qvel=_array(
            raw_proprio["gripper_qvel"],
            dtype=np.float64,
            shape=(2,),
            name="proprio.gripper_qvel",
        ),
        gripper_width_m=_native_float(
            raw_proprio["gripper_width_m"], "proprio.gripper_width_m"
        ),
        ee_force_sensor=_array(
            raw_proprio["ee_force_sensor"],
            dtype=np.float64,
            shape=(3,),
            name="proprio.ee_force_sensor",
        ),
        ee_torque_sensor=_array(
            raw_proprio["ee_torque_sensor"],
            dtype=np.float64,
            shape=(3,),
            name="proprio.ee_torque_sensor",
        ),
    )
    return RobotObservation(cameras=cameras, proprio=proprio)


def serialize_policy_action(action: OSCAction) -> dict[str, Any]:
    """Return the only policy-to-evaluator command representable on the wire."""

    if type(action) is not OSCAction:
        raise PolicyIPCValidationError("action must be an exact OSCAction")
    return {
        "schema": ACTION_WIRE_SCHEMA,
        "values": [float(value) for value in action.values],
    }


def deserialize_policy_action(payload: object) -> OSCAction:
    """Validate one exact seven-float OSC command from a policy worker."""

    message = _exact_dict(payload, {"schema", "values"}, "action")
    if message["schema"] != ACTION_WIRE_SCHEMA or type(message["schema"]) is not str:
        raise PolicyIPCValidationError("action wire schema is invalid")
    values = message["values"]
    if type(values) is not list or len(values) != 7:
        raise PolicyIPCValidationError("action values must be an exact seven-item list")
    if any(type(value) is not float or not math.isfinite(value) for value in values):
        raise PolicyIPCValidationError("action values must be finite native floats")
    try:
        return OSCAction.from_array(values)
    except ValueError as exc:
        raise PolicyIPCValidationError(str(exc)) from exc


def _add_text(digest: Any, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(4, "big"))
    digest.update(encoded)


def _add_array(digest: Any, value: np.ndarray) -> None:
    _add_text(digest, value.dtype.str)
    digest.update(len(value.shape).to_bytes(2, "big"))
    for dimension in value.shape:
        digest.update(int(dimension).to_bytes(8, "big"))
    contiguous = np.ascontiguousarray(value)
    digest.update(contiguous.nbytes.to_bytes(8, "big"))
    digest.update(contiguous.tobytes(order="C"))


def _camera_payload_sha256(camera: dict[str, Any], policy_name: str) -> str:
    digest = hashlib.sha256(b"libero-camera-rgbd-calibration.v1\0")
    _add_text(digest, policy_name)
    _add_text(digest, camera["sensor_name"])
    digest.update(camera["width"].to_bytes(8, "big"))
    digest.update(camera["height"].to_bytes(8, "big"))
    digest.update(b"\x01" if camera["observation_v_flipped"] else b"\x00")
    for name in ("rgb", "depth_m", "intrinsic", "T_world_camera"):
        _add_text(digest, name)
        _add_array(digest, camera[name])
    return digest.hexdigest()


def _proprio_payload_sha256(proprio: dict[str, Any]) -> str:
    digest = hashlib.sha256(b"libero-proprioception.v1\0")
    for name in (
        "T_world_ee",
        "ee_quat_xyzw",
        "joint_position",
        "joint_velocity",
        "gripper_qpos",
        "gripper_qvel",
        "ee_force_sensor",
        "ee_torque_sensor",
    ):
        _add_text(digest, name)
        _add_array(digest, proprio[name])
    _add_text(digest, "gripper_width_m")
    digest.update(np.float64(proprio["gripper_width_m"]).tobytes())
    return digest.hexdigest()


def camera_content_sha256(
    observation: RobotObservation,
    camera_name: str,
) -> str:
    """Commit one camera's RGB, depth, and calibration independently."""

    if type(camera_name) is not str or camera_name not in _CAMERA_KEYS:
        raise PolicyIPCValidationError(
            f"camera_name must be one of {list(_CAMERA_KEYS)!r}"
        )
    payload = serialize_policy_observation(observation)
    return _camera_payload_sha256(payload["cameras"][camera_name], camera_name)


def proprio_content_sha256(observation: RobotObservation) -> str:
    """Commit proprioception separately from camera-capture freshness."""

    payload = serialize_policy_observation(observation)
    return _proprio_payload_sha256(payload["proprio"])


def sensor_content_commitments(observation: RobotObservation) -> dict[str, Any]:
    """Return independent dual-camera, proprio, and aggregate commitments."""

    payload = serialize_policy_observation(observation)
    cameras = {
        name: _camera_payload_sha256(payload["cameras"][name], name)
        for name in _CAMERA_KEYS
    }
    proprio = _proprio_payload_sha256(payload["proprio"])
    digest = hashlib.sha256(b"libero-sensor-snapshot.v2\0")
    for name in _CAMERA_KEYS:
        _add_text(digest, name)
        _add_text(digest, cameras[name])
    _add_text(digest, "proprioception")
    _add_text(digest, proprio)
    return {
        "schema": "libero-sensor-content-commitments.v2",
        "camera_capture_ids": cameras,
        "proprioception_sha256": proprio,
        "snapshot_sha256": digest.hexdigest(),
    }


def sensor_content_sha256(observation: RobotObservation) -> str:
    """Commit the full whitelist snapshot; retained as an aggregate API.

    Camera freshness gates must use the independent camera commitments.  A
    proprioception-only change therefore changes this aggregate token but
    cannot masquerade as a fresh pair of RGB-D captures.
    """

    return sensor_content_commitments(observation)["snapshot_sha256"]


__all__ = [
    "ACTION_WIRE_SCHEMA",
    "MAX_CAMERA_DIMENSION",
    "MAX_CAMERA_PIXELS",
    "OBSERVATION_WIRE_SCHEMA",
    "PolicyIPCValidationError",
    "camera_content_sha256",
    "deserialize_policy_action",
    "deserialize_policy_observation",
    "proprio_content_sha256",
    "sensor_content_commitments",
    "sensor_content_sha256",
    "serialize_policy_action",
    "serialize_policy_observation",
]
