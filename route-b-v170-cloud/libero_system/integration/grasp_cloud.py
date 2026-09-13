"""Strict adapter for the frozen ``grasp-cloud:20260830`` service.

The service returns grasp poses in the submitted camera frame and currently
ships a ``piper_hand`` geometry.  Route C uses a Panda end-effector in the
world frame, so this module deliberately requires both camera calibration and
an explicit service-gripper-to-Panda tool transform.  It never guesses either
transform and never consumes simulator object poses.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import math
import time
from typing import Any
from urllib import error, request
import uuid

import cv2
import numpy as np

from libero_system.route_c import GraspCandidate, SceneEstimate


def _pose(value: object, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4, 4) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite 4x4 pose")
    if not np.allclose(result[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
        raise ValueError(f"{name} has an invalid homogeneous row")
    rotation = result[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
        raise ValueError(f"{name} rotation is not orthonormal")
    if np.linalg.det(rotation) < 0.998:
        raise ValueError(f"{name} rotation must be right handed")
    return result.copy()


def _unit(value: object, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite 3-vector")
    norm = float(np.linalg.norm(result))
    if norm < 1e-9:
        raise ValueError(f"{name} cannot be zero")
    return result / norm


@dataclass(frozen=True)
class GraspCloudInference:
    """One frozen-service response plus its ordinary camera calibration."""

    payload: Mapping[str, Any]
    world_from_camera: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "world_from_camera",
            _pose(self.world_from_camera, "world_from_camera"),
        )


@dataclass(frozen=True)
class GraspCloudAdapterConfig:
    """Calibration needed to make a service pose executable by Panda.

    ``service_gripper_from_panda_ee`` must be measured or generated after the
    service is configured with a Panda-compatible hand.  The 20260830 bundle's
    default ``piper_hand`` result must not be executed with an identity guess.
    """

    service_gripper_from_panda_ee: np.ndarray
    approach_axis_service_gripper: np.ndarray
    panda_gripper_width_m: float = 0.079
    maximum_source_distance_m: float = 0.22
    maximum_plans: int = 8

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "service_gripper_from_panda_ee",
            _pose(
                self.service_gripper_from_panda_ee,
                "service_gripper_from_panda_ee",
            ),
        )
        object.__setattr__(
            self,
            "approach_axis_service_gripper",
            _unit(
                self.approach_axis_service_gripper,
                "approach_axis_service_gripper",
            ),
        )
        if not 0.001 <= self.panda_gripper_width_m <= 0.12:
            raise ValueError("panda_gripper_width_m is implausible")
        if self.maximum_source_distance_m <= 0 or self.maximum_plans < 1:
            raise ValueError("source distance and plan count must be positive")


class GraspCloudResultProvider:
    """Route-C ``GraspProvider`` backed by an injected frozen-service call.

    The callback owns image submission and may use either calibrated RGB-D
    camera.  Returning calibration with the payload keeps camera/world
    conversion explicit and makes the provider easy to test offline.
    """

    def __init__(
        self,
        infer: Callable[[SceneEstimate, str], GraspCloudInference],
        config: GraspCloudAdapterConfig,
    ) -> None:
        self.infer = infer
        self.config = config
        self.last_proposal_trace: dict[str, Any] = {}

    def propose(self, scene: SceneEstimate, object_id: str) -> Sequence[GraspCandidate]:
        source = scene.by_id(object_id)
        inference = self.infer(scene, object_id)
        payload = inference.payload
        if payload.get("result") != "success":
            raise RuntimeError(
                f"grasp-cloud did not return success: {payload.get('result')!r}"
            )
        plans = payload.get("plans")
        if not isinstance(plans, list):
            raise ValueError("grasp-cloud result must contain a plans list")
        candidates: list[GraspCandidate] = []
        rejected = 0
        for index, plan in enumerate(plans):
            if len(candidates) >= self.config.maximum_plans:
                break
            if not isinstance(plan, Mapping):
                rejected += 1
                continue
            pick = plan.get("pick_pose")
            if not isinstance(pick, Mapping) or pick.get("frame") != "camera":
                rejected += 1
                continue
            try:
                camera_from_service_gripper = _pose(
                    pick.get("matrix_4x4"),
                    f"plans[{index}].pick_pose.matrix_4x4",
                )
            except ValueError:
                rejected += 1
                continue
            world_from_service_gripper = (
                inference.world_from_camera @ camera_from_service_gripper
            )
            world_from_ee = (
                world_from_service_gripper
                @ self.config.service_gripper_from_panda_ee
            )
            if (
                np.linalg.norm(world_from_ee[:3, 3] - source.position)
                > self.config.maximum_source_distance_m
            ):
                rejected += 1
                continue
            score = float(plan.get("combined_score", plan.get("graspgenx_score", 0.0)))
            clearance_mm = float(plan.get("scene_clearance_mm", 0.0))
            if not math.isfinite(score) or not math.isfinite(clearance_mm):
                rejected += 1
                continue
            approach_world = (
                world_from_service_gripper[:3, :3]
                @ self.config.approach_axis_service_gripper
            )
            rank = int(plan.get("rank", index + 1))
            candidates.append(
                GraspCandidate(
                    candidate_id=f"grasp-cloud-{rank}",
                    object_id=object_id,
                    world_from_ee=world_from_ee,
                    approach_world=approach_world,
                    score=float(np.clip(score, 0.0, 1.0)),
                    clearance_m=max(0.0, clearance_mm / 1000.0),
                    gripper_width_m=self.config.panda_gripper_width_m,
                )
            )
        if not candidates:
            raise RuntimeError("grasp-cloud returned no calibrated candidate near the visual source")
        candidates.sort(key=lambda item: (item.score, item.clearance_m), reverse=True)
        self.last_proposal_trace = {
            "provider": "grasp-cloud:20260830",
            "coordinate_frame": "camera_to_world_calibrated",
            "object_id": object_id,
            "plans_received": len(plans),
            "plans_rejected": rejected,
            "candidates": len(candidates),
            "config_sha256": payload.get("config_sha256"),
        }
        return tuple(candidates)


@dataclass(frozen=True)
class GraspCloudHTTPConfig:
    base_url: str = "http://127.0.0.1:6006"
    token: str = "grasp-demo"
    request_timeout_s: float = 30.0
    job_timeout_s: float = 240.0
    poll_interval_s: float = 0.5

    def __post_init__(self) -> None:
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must be HTTP(S)")
        if not self.token:
            raise ValueError("API token cannot be empty")
        if min(self.request_timeout_s, self.job_timeout_s, self.poll_interval_s) <= 0:
            raise ValueError("HTTP and polling timeouts must be positive")


class GraspCloudJobClient:
    """Small dependency-free client for the bundle's asynchronous HTTP API."""

    def __init__(
        self,
        config: GraspCloudHTTPConfig | None = None,
        *,
        urlopen: Callable[..., Any] = request.urlopen,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config or GraspCloudHTTPConfig()
        self._urlopen = urlopen
        self._sleep = sleeper

    def submit(self, rgb: np.ndarray, depth_m: np.ndarray, instruction: str) -> str:
        rgb = np.asarray(rgb)
        depth = np.asarray(depth_m, dtype=np.float64)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("rgb must be HxWx3 uint8")
        if depth.shape != rgb.shape[:2]:
            raise ValueError("depth_m must match rgb height and width")
        if not instruction.strip():
            raise ValueError("instruction cannot be empty")
        valid = np.isfinite(depth) & (depth > 0.0) & (depth < 65.535)
        depth_mm = np.zeros(depth.shape, dtype=np.uint16)
        depth_mm[valid] = np.rint(depth[valid] * 1000.0).astype(np.uint16)
        ok_rgb, encoded_rgb = cv2.imencode(
            ".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        )
        ok_depth, encoded_depth = cv2.imencode(".png", depth_mm)
        if not ok_rgb or not ok_depth:
            raise RuntimeError("failed to encode grasp-cloud RGB-D submission")
        boundary = f"----libero-grasp-{uuid.uuid4().hex}"
        body = self._multipart(
            boundary,
            (
                ("instruction", None, "text/plain; charset=utf-8", instruction.encode()),
                ("rgb", "rgb.png", "image/png", encoded_rgb.tobytes()),
                ("depth", "depth.png", "image/png", encoded_depth.tobytes()),
            ),
        )
        payload = self._json(
            "/api/jobs",
            method="POST",
            data=body,
            content_type=f"multipart/form-data; boundary={boundary}",
        )
        job_id = payload.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise RuntimeError("grasp-cloud submit response omitted job_id")
        return job_id

    def infer(self, rgb: np.ndarray, depth_m: np.ndarray, instruction: str) -> Mapping[str, Any]:
        job_id = self.submit(rgb, depth_m, instruction)
        deadline = time.monotonic() + self.config.job_timeout_s
        while time.monotonic() < deadline:
            status = self._json(f"/api/jobs/{job_id}")
            state = str(status.get("status", status.get("state", ""))).lower()
            result_state = str(status.get("result", "")).lower()
            if state == "finished":
                return self._json(f"/api/jobs/{job_id}/result")
            if state in {"failed", "error", "rejected"} or result_state in {
                "failed",
                "error",
                "rejected",
            }:
                raise RuntimeError(f"grasp-cloud job {job_id} ended as {state or result_state}")
            self._sleep(self.config.poll_interval_s)
        raise TimeoutError(f"grasp-cloud job {job_id} exceeded {self.config.job_timeout_s}s")

    def _json(
        self,
        path: str,
        *,
        method: str = "GET",
        data: bytes | None = None,
        content_type: str | None = None,
    ) -> Mapping[str, Any]:
        headers = {"X-API-Token": self.config.token, "Accept": "application/json"}
        if content_type is not None:
            headers["Content-Type"] = content_type
        target = self.config.base_url.rstrip("/") + path
        outgoing = request.Request(target, data=data, headers=headers, method=method)
        try:
            with self._urlopen(outgoing, timeout=self.config.request_timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"grasp-cloud request failed for {path}") from exc
        if not isinstance(payload, Mapping):
            raise RuntimeError("grasp-cloud response must be a JSON object")
        return payload

    @staticmethod
    def _multipart(
        boundary: str,
        fields: Sequence[tuple[str, str | None, str, bytes]],
    ) -> bytes:
        chunks: list[bytes] = []
        marker = boundary.encode("ascii")
        for name, filename, content_type, value in fields:
            chunks.extend((b"--" + marker + b"\r\n",))
            disposition = f'Content-Disposition: form-data; name="{name}"'
            if filename is not None:
                disposition += f'; filename="{filename}"'
            chunks.extend(
                (
                    disposition.encode("utf-8") + b"\r\n",
                    f"Content-Type: {content_type}\r\n\r\n".encode("ascii"),
                    value,
                    b"\r\n",
                )
            )
        chunks.append(b"--" + marker + b"--\r\n")
        return b"".join(chunks)
