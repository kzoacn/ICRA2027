"""Simulator-agnostic sensor and scene types for LIBERO perception.

Only calibrated RGB-D pixels enter this module.  The types intentionally have
no simulator object ids, task predicates, privileged poses, or segmentation
buffers.  ``world_from_camera`` is ordinary camera calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
import hashlib
import math
import re
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
UInt8Array = NDArray[np.uint8]
IntArray = NDArray[np.int32]


_CAPTURE_ID = re.compile(r"[0-9a-f]{64}")


def _validated_capture_id(value: object, name: str = "capture_id") -> str:
    """Validate an equality-only commitment derived from public sensors.

    The empty string remains available to non-formal fixtures.  A populated
    value is a SHA-256 commitment, not a clock, action index, or sortable
    sequence number.
    """

    if type(value) is not str or (value and _CAPTURE_ID.fullmatch(value) is None):
        raise ValueError(f"{name} must be empty or 64 lowercase hex characters")
    return value


def _validated_camera_capture_ids(
    value: object,
    *,
    camera_names: Sequence[str],
) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError("camera_capture_ids must be a mapping")
    raw = dict(value)
    if not raw:
        return MappingProxyType({})
    if any(type(name) is not str or not name for name in raw):
        raise ValueError("camera_capture_ids names must be non-empty strings")
    if not camera_names or set(raw) != set(camera_names):
        raise ValueError("camera_capture_ids must cover every scene camera exactly")
    validated = {
        name: _validated_capture_id(raw[name], f"camera_capture_ids.{name}")
        for name in sorted(raw)
    }
    if any(not commitment for commitment in validated.values()):
        raise ValueError("camera_capture_ids cannot contain empty commitments")
    return MappingProxyType(validated)


def _combined_camera_capture_id(camera_capture_ids: Mapping[str, str]) -> str:
    if not camera_capture_ids:
        return ""
    digest = hashlib.sha256(b"libero-camera-capture-set.v1\0")
    for name, commitment in sorted(camera_capture_ids.items()):
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(commitment))
    return digest.hexdigest()


def _finite_array(value: object, shape: tuple[int, ...], name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {shape}; got {result.shape}")
    return result.copy()


def _points(value: object, name: str, *, dtype=np.float64) -> NDArray:
    result = np.asarray(value, dtype=dtype)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3); got {result.shape}")
    if np.issubdtype(result.dtype, np.floating) and not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result.copy()


def normalize_label(label: str) -> str:
    result = " ".join(str(label).lower().replace("_", " ").strip().split())
    if not result:
        raise ValueError("label cannot be empty")
    return result


@dataclass(frozen=True)
class RGBDFrame:
    """One upright RGB-D image with metric camera-z depth.

    ``intrinsics`` uses the camera projection convention.  LIBERO's observation
    array is visually upright but vertically reversed relative to that
    convention; set ``observation_v_flipped`` to preserve the image for vision
    models while geometry transparently uses ``H - 1 - v_observation``.
    """

    name: str
    rgb: UInt8Array
    depth_m: FloatArray
    intrinsics: FloatArray
    world_from_camera: FloatArray
    timestamp_s: float = 0.0
    observation_v_flipped: bool = False
    capture_id: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("camera name is required")
        rgb = np.asarray(self.rgb)
        depth = np.asarray(self.depth_m, dtype=np.float64)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("rgb must be an HxWx3 uint8 array")
        if depth.shape != rgb.shape[:2]:
            raise ValueError("depth_m must match the RGB image height and width")
        intrinsics = _finite_array(self.intrinsics, (3, 3), "intrinsics")
        if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
            raise ValueError("camera focal lengths must be positive")
        transform = _finite_array(self.world_from_camera, (4, 4), "world_from_camera")
        if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
            raise ValueError("world_from_camera must be homogeneous")
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
            raise ValueError("world_from_camera rotation must be orthonormal")
        if not math.isfinite(float(self.timestamp_s)):
            raise ValueError("timestamp_s must be finite")
        capture_id = _validated_capture_id(self.capture_id)
        object.__setattr__(self, "rgb", rgb.copy())
        object.__setattr__(self, "depth_m", depth.copy())
        object.__setattr__(self, "intrinsics", intrinsics)
        object.__setattr__(self, "world_from_camera", transform)
        object.__setattr__(self, "observation_v_flipped", bool(self.observation_v_flipped))
        object.__setattr__(self, "capture_id", capture_id)

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])


@dataclass(frozen=True)
class PointCloud:
    """Fused points with the sensor provenance needed for ray completion."""

    points_world: FloatArray
    colors_rgb: UInt8Array
    camera_indices: IntArray
    pixels_uv: FloatArray

    def __post_init__(self) -> None:
        points = _points(self.points_world, "points_world")
        colors = _points(self.colors_rgb, "colors_rgb", dtype=np.uint8)
        camera_indices = np.asarray(self.camera_indices, dtype=np.int32)
        pixels = np.asarray(self.pixels_uv, dtype=np.float64)
        count = len(points)
        if len(colors) != count or camera_indices.shape != (count,) or pixels.shape != (count, 2):
            raise ValueError("point cloud attributes must have the same leading length")
        if np.any(camera_indices < 0) or not np.all(np.isfinite(pixels)):
            raise ValueError("point provenance is invalid")
        object.__setattr__(self, "points_world", points)
        object.__setattr__(self, "colors_rgb", colors)
        object.__setattr__(self, "camera_indices", camera_indices.copy())
        object.__setattr__(self, "pixels_uv", pixels.copy())

    def subset(self, indices: NDArray[np.integer] | NDArray[np.bool_]) -> "PointCloud":
        return PointCloud(
            self.points_world[indices],
            self.colors_rgb[indices],
            self.camera_indices[indices],
            self.pixels_uv[indices],
        )


@dataclass(frozen=True)
class ObservedGeometry:
    """Geometry of visible surfaces, before prior-based occlusion completion."""

    center_world: FloatArray
    axes_world: FloatArray
    extents_m: FloatArray
    bounds_min_world: FloatArray
    bounds_max_world: FloatArray

    def __post_init__(self) -> None:
        center = _finite_array(self.center_world, (3,), "observed.center_world")
        axes = _finite_array(self.axes_world, (3, 3), "observed.axes_world")
        extents = _finite_array(self.extents_m, (3,), "observed.extents_m")
        lower = _finite_array(self.bounds_min_world, (3,), "observed.bounds_min_world")
        upper = _finite_array(self.bounds_max_world, (3,), "observed.bounds_max_world")
        if not np.allclose(axes.T @ axes, np.eye(3), atol=2e-3):
            raise ValueError("observed axes must be orthonormal")
        if np.any(extents < 0) or np.any(lower > upper):
            raise ValueError("observed extents or bounds are invalid")
        object.__setattr__(self, "center_world", center)
        object.__setattr__(self, "axes_world", axes)
        object.__setattr__(self, "extents_m", extents)
        object.__setattr__(self, "bounds_min_world", lower)
        object.__setattr__(self, "bounds_max_world", upper)


@dataclass(frozen=True)
class ObjectInstance:
    """A neutral multi-instance result usable by both ANCHOR and Planning.

    ``observed`` is never overwritten: it describes only visible RGB-D
    surfaces.  ``center_world`` and ``extents_m`` may be completed with static
    asset dimensions and the 2-D component-centre ray, which avoids the common
    two-centimetre camera-side bias of a visible-point median.
    """

    instance_id: str
    center_world: FloatArray
    grasp_point_world: FloatArray
    axes_world: FloatArray
    extents_m: FloatArray
    observed: ObservedGeometry
    label_scores: Mapping[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    point_count: int = 0
    color_histogram: FloatArray | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.instance_id:
            raise ValueError("instance_id is required")
        center = _finite_array(self.center_world, (3,), "center_world")
        grasp = _finite_array(self.grasp_point_world, (3,), "grasp_point_world")
        axes = _finite_array(self.axes_world, (3, 3), "axes_world")
        extents = _finite_array(self.extents_m, (3,), "extents_m")
        if not np.allclose(axes.T @ axes, np.eye(3), atol=2e-3):
            raise ValueError("axes_world must be orthonormal")
        if np.any(extents <= 0):
            raise ValueError("extents_m must be positive")
        scores: dict[str, float] = {}
        for raw_label, raw_score in self.label_scores.items():
            label = normalize_label(raw_label)
            score = float(raw_score)
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError("label scores must lie in [0, 1]")
            scores[label] = score
        if not math.isfinite(float(self.confidence)) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must lie in [0, 1]")
        if self.point_count <= 0:
            raise ValueError("point_count must be positive")
        histogram = None
        if self.color_histogram is not None:
            histogram = np.asarray(self.color_histogram, dtype=np.float64).reshape(-1)
            if np.any(histogram < 0) or not np.all(np.isfinite(histogram)):
                raise ValueError("color_histogram must be finite and non-negative")
            total = float(histogram.sum())
            if total <= 0:
                raise ValueError("color_histogram cannot be empty")
            histogram = histogram / total
        object.__setattr__(self, "center_world", center)
        object.__setattr__(self, "grasp_point_world", grasp)
        object.__setattr__(self, "axes_world", axes)
        object.__setattr__(self, "extents_m", extents)
        object.__setattr__(self, "label_scores", MappingProxyType(scores))
        object.__setattr__(self, "color_histogram", histogram)

    @property
    def label(self) -> str:
        if not self.label_scores:
            return "unknown"
        return max(self.label_scores, key=self.label_scores.__getitem__)

    def score_for(self, label: str) -> float:
        return float(self.label_scores.get(normalize_label(label), 0.0))

    def with_semantics(
        self,
        label_scores: Mapping[str, float],
        *,
        center_world: FloatArray | None = None,
        grasp_point_world: FloatArray | None = None,
        extents_m: FloatArray | None = None,
        confidence: float | None = None,
    ) -> "ObjectInstance":
        return replace(
            self,
            label_scores=label_scores,
            center_world=self.center_world if center_world is None else center_world,
            grasp_point_world=(
                self.grasp_point_world if grasp_point_world is None else grasp_point_world
            ),
            extents_m=self.extents_m if extents_m is None else extents_m,
            confidence=self.confidence if confidence is None else confidence,
        )


class SelectorKind(StrEnum):
    HIGHEST_SCORE = "highest_score"
    MIN_X = "min_x"
    MAX_X = "max_x"
    MIN_Y = "min_y"
    MAX_Y = "max_y"
    NEAREST = "nearest"
    FARTHEST = "farthest"
    BETWEEN = "between"
    INDEX = "index"


@dataclass(frozen=True)
class InstanceSelector:
    """Observable geometry selector for repeated labels.

    Axis selectors are named explicitly in the world frame so route adapters do
    not have to guess what "front" means in a particular LIBERO scene.
    """

    kind: SelectorKind = SelectorKind.HIGHEST_SCORE
    reference_points: tuple[FloatArray, ...] = ()
    index: int = 0

    def __post_init__(self) -> None:
        points = tuple(_finite_array(point, (3,), "reference_point") for point in self.reference_points)
        required = 1 if self.kind in {SelectorKind.NEAREST, SelectorKind.FARTHEST} else 2 if self.kind == SelectorKind.BETWEEN else 0
        if len(points) != required:
            raise ValueError(f"selector {self.kind.value} requires {required} reference point(s)")
        if self.kind == SelectorKind.INDEX and self.index < 0:
            raise ValueError("selector index cannot be negative")
        object.__setattr__(self, "reference_points", points)


@dataclass(frozen=True)
class SceneObservation:
    timestamp_s: float
    table_height_m: float
    instances: tuple[ObjectInstance, ...]
    camera_names: tuple[str, ...] = ()
    capture_id: str = ""
    camera_capture_ids: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.timestamp_s)) or not math.isfinite(float(self.table_height_m)):
            raise ValueError("scene time and table height must be finite")
        ids = [instance.instance_id for instance in self.instances]
        if len(ids) != len(set(ids)):
            raise ValueError("instance ids must be unique")
        if len(self.camera_names) != len(set(self.camera_names)):
            raise ValueError("scene camera names must be unique")
        camera_ids = _validated_camera_capture_ids(
            self.camera_capture_ids,
            camera_names=self.camera_names,
        )
        capture_id = _validated_capture_id(self.capture_id)
        if camera_ids and capture_id != _combined_camera_capture_id(camera_ids):
            raise ValueError(
                "scene capture_id must commit the exact per-camera capture IDs"
            )
        object.__setattr__(self, "capture_id", capture_id)
        object.__setattr__(self, "camera_capture_ids", camera_ids)

    def candidates(self, label: str | None = None, *, min_score: float = 0.0) -> tuple[ObjectInstance, ...]:
        if label is None:
            return self.instances
        normalized = normalize_label(label)
        return tuple(x for x in self.instances if x.label_scores.get(normalized, 0.0) >= min_score)

    def select(
        self,
        label: str | None = None,
        selector: InstanceSelector | None = None,
        *,
        min_score: float = 0.0,
    ) -> ObjectInstance:
        candidates = list(self.candidates(label, min_score=min_score))
        if not candidates:
            raise LookupError(f"no observed instance matches {label!r}")
        selector = selector or InstanceSelector()
        label_key = normalize_label(label) if label is not None else None
        if selector.kind == SelectorKind.HIGHEST_SCORE:
            return max(candidates, key=lambda x: x.label_scores.get(label_key, x.confidence) if label_key else x.confidence)
        if selector.kind == SelectorKind.MIN_X:
            return min(candidates, key=lambda x: x.center_world[0])
        if selector.kind == SelectorKind.MAX_X:
            return max(candidates, key=lambda x: x.center_world[0])
        if selector.kind == SelectorKind.MIN_Y:
            return min(candidates, key=lambda x: x.center_world[1])
        if selector.kind == SelectorKind.MAX_Y:
            return max(candidates, key=lambda x: x.center_world[1])
        if selector.kind == SelectorKind.INDEX:
            ordered = sorted(candidates, key=lambda x: (x.center_world[0], x.center_world[1], x.instance_id))
            if selector.index >= len(ordered):
                raise LookupError(f"selector index {selector.index} exceeds {len(ordered)} candidates")
            return ordered[selector.index]
        target = selector.reference_points[0]
        if selector.kind == SelectorKind.BETWEEN:
            target = (selector.reference_points[0] + selector.reference_points[1]) / 2.0
        distances = [float(np.linalg.norm(x.center_world[:2] - target[:2])) for x in candidates]
        if selector.kind == SelectorKind.FARTHEST:
            return candidates[int(np.argmax(distances))]
        return candidates[int(np.argmin(distances))]


@dataclass(frozen=True)
class BoxDetection:
    query: str
    xyxy: FloatArray
    score: float
    camera_name: str

    def __post_init__(self) -> None:
        query = normalize_label(self.query)
        box = _finite_array(self.xyxy, (4,), "xyxy")
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError("box must have positive width and height")
        if not 0.0 <= float(self.score) <= 1.0:
            raise ValueError("score must lie in [0, 1]")
        if not self.camera_name:
            raise ValueError("camera_name is required")
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "xyxy", box)


def scene_timestamp(frames: Sequence[RGBDFrame]) -> float:
    if not frames:
        raise ValueError("at least one RGB-D frame is required")
    return float(max(frame.timestamp_s for frame in frames))


def scene_capture_id(frames: Sequence[RGBDFrame]) -> str:
    """Commit the complete named camera-capture set without a clock."""

    return _combined_camera_capture_id(scene_camera_capture_ids(frames))


def scene_camera_capture_ids(frames: Sequence[RGBDFrame]) -> Mapping[str, str]:
    """Return per-camera commitments, rejecting partial/mixed snapshots."""

    if not frames:
        raise ValueError("at least one RGB-D frame is required")
    names = [frame.name for frame in frames]
    if len(names) != len(set(names)):
        raise ValueError("RGB-D frame names must be unique")
    populated = [bool(frame.capture_id) for frame in frames]
    if any(populated) and not all(populated):
        raise ValueError("camera capture commitments must be all populated or all empty")
    if not any(populated):
        return MappingProxyType({})
    return MappingProxyType(
        {
            frame.name: frame.capture_id
            for frame in sorted(frames, key=lambda item: item.name)
        }
    )
