"""Sensor-only RGB-D scene interfaces and basic 3-D geometry extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import re
from typing import Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .schema import EntityRef, Relation


FloatArray = NDArray[np.float64]
_CAPTURE_ID = re.compile(r"[0-9a-f]{64}")


def _capture_id(value: object, name: str) -> str:
    if type(value) is not str or (value and _CAPTURE_ID.fullmatch(value) is None):
        raise ValueError(f"{name} must be empty or 64 lowercase hex characters")
    return value


def _capture_field(value: object, field: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(field, default)
    return getattr(value, field, default)


def _camera_capture_ids(value: object, name: str) -> Mapping[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    raw = dict(value)
    if not raw:
        return {}
    if set(raw) != {"agentview", "wrist"}:
        raise ValueError(f"{name} must contain exact agentview and wrist keys")
    validated = {
        camera: _capture_id(raw[camera], f"{name}.{camera}")
        for camera in ("agentview", "wrist")
    }
    if any(not commitment for commitment in validated.values()):
        raise ValueError(f"{name} commitments cannot be empty")
    return validated


def _combined_camera_capture_id(value: Mapping[str, str]) -> str:
    if not value:
        return ""
    digest = hashlib.sha256(b"libero-camera-capture-set.v1\0")
    for name, commitment in sorted(value.items()):
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(commitment))
    return digest.hexdigest()


def _legacy_capture_time(value: object) -> float | None:
    raw = _capture_field(value, "timestamp_s")
    if type(raw) not in (int, float) or not math.isfinite(raw):
        return None
    return float(raw)


def same_sensor_capture(first: object, second: object) -> bool:
    """Compare capture identity without treating a clock as policy evidence.

    Populated content commitments are equality-only.  Numeric timestamps are
    retained solely for legacy/test providers that predate capture IDs.
    """

    first_cameras = _camera_capture_ids(
        _capture_field(first, "camera_capture_ids", {}),
        "first.camera_capture_ids",
    )
    second_cameras = _camera_capture_ids(
        _capture_field(second, "camera_capture_ids", {}),
        "second.camera_capture_ids",
    )
    if first_cameras or second_cameras:
        return bool(
            first_cameras
            and second_cameras
            and all(
                first_cameras[name] == second_cameras[name]
                for name in ("agentview", "wrist")
            )
        )
    first_id = _capture_id(
        _capture_field(first, "capture_id", ""), "first.capture_id"
    )
    second_id = _capture_id(
        _capture_field(second, "capture_id", ""), "second.capture_id"
    )
    if first_id or second_id:
        return bool(first_id and second_id and first_id == second_id)
    first_time = _legacy_capture_time(first)
    second_time = _legacy_capture_time(second)
    return bool(
        first_time is not None
        and second_time is not None
        and first_time == second_time
    )


def sensor_capture_advanced(previous: object, current: object) -> bool:
    """Require both camera captures to advance, with legacy time fallback."""

    previous_cameras = _camera_capture_ids(
        _capture_field(previous, "camera_capture_ids", {}),
        "previous.camera_capture_ids",
    )
    current_cameras = _camera_capture_ids(
        _capture_field(current, "camera_capture_ids", {}),
        "current.camera_capture_ids",
    )
    if previous_cameras or current_cameras:
        return bool(
            previous_cameras
            and current_cameras
            and all(
                previous_cameras[name] != current_cameras[name]
                for name in ("agentview", "wrist")
            )
        )

    previous_id = _capture_id(
        _capture_field(previous, "capture_id", ""), "previous.capture_id"
    )
    current_id = _capture_id(
        _capture_field(current, "capture_id", ""), "current.capture_id"
    )
    if previous_id or current_id:
        return bool(previous_id and current_id and previous_id != current_id)
    previous_time = _legacy_capture_time(previous)
    current_time = _legacy_capture_time(current)
    return bool(
        previous_time is not None
        and current_time is not None
        and current_time > previous_time
    )


class PerceptionError(RuntimeError):
    pass


def _array(value: object, shape: tuple[int, ...], name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {shape}, got {result.shape}")
    return result.copy()


def _pose(value: object, name: str = "pose") -> FloatArray:
    result = _array(value, (4, 4), name)
    if not np.allclose(result[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    if not np.allclose(result[:3, :3].T @ result[:3, :3], np.eye(3), atol=2e-3):
        raise ValueError(f"{name} rotation is not orthonormal")
    return result


@runtime_checkable
class DistanceField(Protocol):
    """Positive in free space and negative inside an obstacle."""

    def distance(self, points_world: FloatArray) -> FloatArray: ...


@dataclass(frozen=True)
class RGBDFrame:
    rgb: NDArray[np.uint8]
    depth_m: FloatArray
    intrinsics: FloatArray
    world_from_camera: FloatArray
    timestamp_s: float
    camera_name: str
    capture_id: str = ""

    def __post_init__(self) -> None:
        rgb = np.asarray(self.rgb)
        depth = np.asarray(self.depth_m, dtype=np.float64)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[:2] != depth.shape:
            raise ValueError("rgb must be HxWx3 and depth_m must be matching HxW")
        if rgb.dtype != np.uint8:
            raise ValueError("rgb must use uint8 pixels")
        if not math.isfinite(float(self.timestamp_s)):
            raise ValueError("timestamp_s must be finite")
        if not self.camera_name:
            raise ValueError("camera_name is required")
        capture_id = _capture_id(self.capture_id, "frame.capture_id")
        object.__setattr__(self, "rgb", rgb.copy())
        object.__setattr__(self, "depth_m", depth.copy())
        object.__setattr__(self, "intrinsics", _array(self.intrinsics, (3, 3), "intrinsics"))
        object.__setattr__(
            self, "world_from_camera", _pose(self.world_from_camera, "world_from_camera")
        )
        object.__setattr__(self, "capture_id", capture_id)


@dataclass(frozen=True)
class MaskObservation:
    """Open-vocabulary mask output; it is not a simulator instance mask."""

    label: str
    instance_id: str
    camera_name: str
    mask: NDArray[np.bool_]
    confidence: float

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask, dtype=np.bool_)
        if mask.ndim != 2:
            raise ValueError("mask must be HxW")
        if not self.label.strip() or not self.instance_id or not self.camera_name:
            raise ValueError("label, instance_id and camera_name are required")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        object.__setattr__(self, "label", " ".join(self.label.lower().replace("_", " ").split()))
        object.__setattr__(self, "mask", mask.copy())


@dataclass(frozen=True)
class TargetRegion:
    """Oriented box inferred from visible RGB-D geometry."""

    center: FloatArray
    axes: FloatArray
    half_extents: FloatArray
    surface_normal: FloatArray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))

    def __post_init__(self) -> None:
        center = _array(self.center, (3,), "region.center")
        axes = _array(self.axes, (3, 3), "region.axes")
        extents = _array(self.half_extents, (3,), "region.half_extents")
        normal = _array(self.surface_normal, (3,), "region.surface_normal")
        if np.any(extents <= 0):
            raise ValueError("region.half_extents must be positive")
        if not np.allclose(axes.T @ axes, np.eye(3), atol=2e-3):
            raise ValueError("region.axes must be orthonormal")
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            raise ValueError("region.surface_normal must be non-zero")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "half_extents", extents)
        object.__setattr__(self, "surface_normal", normal / norm)

    def local_coordinates(self, points_world: FloatArray) -> FloatArray:
        points = np.asarray(points_world, dtype=np.float64)
        return (points - self.center) @ self.axes

    def contains(self, points_world: FloatArray, margin: float = 0.0) -> NDArray[np.bool_]:
        local = np.abs(self.local_coordinates(points_world))
        limits = np.maximum(self.half_extents - float(margin), 0.0)
        return np.all(local <= limits, axis=-1)

    @property
    def top_center(self) -> FloatArray:
        return self.center + self.surface_normal * self.half_extents[2]


@dataclass(frozen=True)
class SceneEntity:
    instance_id: str
    label: str
    pose: FloatArray
    extent: FloatArray
    confidence: float
    keypoints: Mapping[str, FloatArray] = field(default_factory=dict)
    region: TargetRegion | None = None
    # Raw visible surface samples are optional because tracked entities may
    # survive a short occlusion.  When present they come only from the current
    # calibrated RGB-D component; completed asset dimensions never fabricate
    # points.  Geometry-specific grasp providers must fail closed when their
    # required samples are absent.
    surface_points_world: FloatArray | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not self.instance_id or not self.label.strip():
            raise ValueError("entity instance_id and label are required")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("entity confidence must be in [0, 1]")
        extent = _array(self.extent, (3,), "entity.extent")
        if np.any(extent <= 0):
            raise ValueError("entity.extent must be positive")
        keypoints = {
            name: _array(point, (3,), f"keypoint.{name}")
            for name, point in self.keypoints.items()
        }
        surface_points = None
        if self.surface_points_world is not None:
            surface_points = np.asarray(self.surface_points_world, dtype=np.float64)
            if (
                surface_points.ndim != 2
                or surface_points.shape[1:] != (3,)
                or len(surface_points) < 8
                or not np.all(np.isfinite(surface_points))
            ):
                raise ValueError(
                    "entity.surface_points_world must be a finite Nx3 array "
                    "with at least eight points"
                )
            surface_points = surface_points.copy()
        object.__setattr__(self, "label", " ".join(self.label.lower().replace("_", " ").split()))
        object.__setattr__(self, "pose", _pose(self.pose, "entity.pose"))
        object.__setattr__(self, "extent", extent)
        object.__setattr__(self, "keypoints", keypoints)
        object.__setattr__(self, "surface_points_world", surface_points)

    @property
    def position(self) -> FloatArray:
        return self.pose[:3, 3]


@dataclass(frozen=True)
class SupportRelationEvidence:
    """Observation-scoped proof that a visible source rests on a support.

    This record deliberately carries no inferred fixture pose, extent, region,
    or obstacle geometry.  It names one already observed source and one exact
    language reference, plus the measured support point used to re-anchor that
    source.  Consumers must still match the timestamp and fresh source identity
    against the :class:`SceneEstimate` in which the evidence appears.
    """

    timestamp_s: float
    relation: Relation
    source_label: str
    reference_label: str
    source_instance_id: str
    support_point_world: FloatArray
    capture_id: str = ""
    camera_capture_ids: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.timestamp_s, bool) or not isinstance(
            self.timestamp_s, (int, float)
        ):
            raise ValueError("support evidence timestamp_s must be numeric")
        timestamp_s = float(self.timestamp_s)
        if not math.isfinite(timestamp_s):
            raise ValueError("support evidence timestamp_s must be finite")
        if self.relation is not Relation.ON:
            raise ValueError("support evidence is restricted to Relation.ON")

        def normalise_label(value: object, name: str) -> str:
            if not isinstance(value, str):
                raise ValueError(f"support evidence {name} must be a string")
            result = " ".join(value.lower().replace("_", " ").strip().split())
            if not result or len(result) > 96:
                raise ValueError(
                    f"support evidence {name} must contain 1..96 characters"
                )
            return result

        source_label = normalise_label(self.source_label, "source_label")
        reference_label = normalise_label(self.reference_label, "reference_label")
        if (
            not isinstance(self.source_instance_id, str)
            or not self.source_instance_id.strip()
        ):
            raise ValueError(
                "support evidence source_instance_id must be a non-empty string"
            )
        source_instance_id = self.source_instance_id.strip()
        if len(source_instance_id) > 256:
            raise ValueError(
                "support evidence source_instance_id must contain at most 256 characters"
            )

        object.__setattr__(self, "timestamp_s", timestamp_s)
        object.__setattr__(
            self,
            "capture_id",
            _capture_id(self.capture_id, "support_evidence.capture_id"),
        )
        camera_ids = _camera_capture_ids(
            self.camera_capture_ids,
            "support_evidence.camera_capture_ids",
        )
        if camera_ids and self.capture_id != _combined_camera_capture_id(camera_ids):
            raise ValueError(
                "support evidence capture_id must commit its two camera IDs"
            )
        object.__setattr__(self, "camera_capture_ids", camera_ids)
        object.__setattr__(self, "source_label", source_label)
        object.__setattr__(self, "reference_label", reference_label)
        object.__setattr__(self, "source_instance_id", source_instance_id)
        object.__setattr__(
            self,
            "support_point_world",
            _array(
                self.support_point_world,
                (3,),
                "support_evidence.support_point_world",
            ),
        )


@dataclass(frozen=True)
class SceneEstimate:
    """All state consumed by Planning; every field is sensor-derived."""

    timestamp_s: float
    entities: tuple[SceneEntity, ...]
    obstacle_sdf: DistanceField
    workspace_min: FloatArray
    workspace_max: FloatArray
    scene_floor_z: float
    support_relation_evidence: tuple[SupportRelationEvidence, ...] = ()
    capture_id: str = ""
    camera_capture_ids: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        lower = _array(self.workspace_min, (3,), "workspace_min")
        upper = _array(self.workspace_max, (3,), "workspace_max")
        if np.any(lower >= upper):
            raise ValueError("workspace_min must be less than workspace_max")
        if isinstance(self.timestamp_s, bool) or not isinstance(
            self.timestamp_s, (int, float)
        ) or not math.isfinite(float(self.timestamp_s)):
            raise ValueError("scene timestamp_s must be finite numeric")
        if len({x.instance_id for x in self.entities}) != len(self.entities):
            raise ValueError("entity instance_id values must be unique")
        if not isinstance(self.obstacle_sdf, DistanceField):
            raise TypeError("obstacle_sdf must implement DistanceField")
        evidence = tuple(self.support_relation_evidence)
        if not all(isinstance(item, SupportRelationEvidence) for item in evidence):
            raise TypeError(
                "support_relation_evidence must contain SupportRelationEvidence values"
            )
        object.__setattr__(self, "workspace_min", lower)
        object.__setattr__(self, "workspace_max", upper)
        object.__setattr__(self, "support_relation_evidence", evidence)
        capture_id = _capture_id(self.capture_id, "scene.capture_id")
        camera_ids = _camera_capture_ids(
            self.camera_capture_ids,
            "scene.camera_capture_ids",
        )
        if camera_ids and capture_id != _combined_camera_capture_id(camera_ids):
            raise ValueError("scene capture_id must commit its two camera IDs")
        object.__setattr__(self, "capture_id", capture_id)
        object.__setattr__(self, "camera_capture_ids", camera_ids)

    def by_id(self, instance_id: str) -> SceneEntity:
        for entity in self.entities:
            if entity.instance_id == instance_id:
                return entity
        raise PerceptionError(f"entity disappeared: {instance_id!r}")


@runtime_checkable
class OpenVocabularySegmenter(Protocol):
    def segment(
        self, frames: Sequence[RGBDFrame], requested_labels: Sequence[str]
    ) -> Sequence[MaskObservation]: ...


@runtime_checkable
class SceneEstimator(Protocol):
    def observe(self, requested_labels: Sequence[str]) -> SceneEstimate: ...


class RGBDGeometryBuilder:
    """Convert a model-predicted mask and calibrated depth into 3-D geometry."""

    def __init__(self, min_depth_m: float = 0.05, max_depth_m: float = 3.0) -> None:
        if not 0 < min_depth_m < max_depth_m:
            raise ValueError("invalid depth interval")
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)

    def backproject(self, frame: RGBDFrame, mask: NDArray[np.bool_]) -> FloatArray:
        mask = np.asarray(mask, dtype=np.bool_)
        if mask.shape != frame.depth_m.shape:
            raise PerceptionError("mask and frame dimensions differ")
        depth = frame.depth_m
        valid = mask & np.isfinite(depth) & (depth >= self.min_depth_m) & (depth <= self.max_depth_m)
        rows, cols = np.nonzero(valid)
        if len(rows) < 8:
            raise PerceptionError("too few valid depth pixels in mask")
        z = depth[rows, cols]
        fx, fy = frame.intrinsics[0, 0], frame.intrinsics[1, 1]
        cx, cy = frame.intrinsics[0, 2], frame.intrinsics[1, 2]
        camera = np.column_stack(((cols - cx) * z / fx, (rows - cy) * z / fy, z))
        homogeneous = np.column_stack((camera, np.ones(len(camera))))
        return (frame.world_from_camera @ homogeneous.T).T[:, :3]

    @staticmethod
    def robust_points(points_world: FloatArray, z_limit: float = 4.5) -> FloatArray:
        points = np.asarray(points_world, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 8:
            raise PerceptionError("point cloud must be Nx3 with at least eight points")
        median = np.median(points, axis=0)
        mad = np.median(np.abs(points - median), axis=0)
        scale = np.maximum(1.4826 * mad, 1e-5)
        keep = np.all(np.abs(points - median) <= z_limit * scale, axis=1)
        filtered = points[keep]
        if len(filtered) < 8:
            raise PerceptionError("robust filtering removed too many points")
        return filtered

    def entity_from_mask(self, frame: RGBDFrame, mask: MaskObservation) -> SceneEntity:
        if mask.camera_name != frame.camera_name:
            raise PerceptionError("mask camera does not match RGB-D frame")
        points = self.robust_points(self.backproject(frame, mask.mask))
        center = np.median(points, axis=0)
        centered = points - center
        covariance = centered.T @ centered / max(len(points) - 1, 1)
        _, axes = np.linalg.eigh(covariance)
        axes = axes[:, ::-1]
        if np.linalg.det(axes) < 0:
            axes[:, -1] *= -1
        local = centered @ axes
        low, high = np.quantile(local, [0.02, 0.98], axis=0)
        half_extents = np.maximum((high - low) / 2.0, 0.003)
        local_center = (high + low) / 2.0
        obb_center = center + axes @ local_center
        pose = np.eye(4)
        pose[:3, :3] = axes
        pose[:3, 3] = obb_center
        world_z = np.array([0.0, 0.0, 1.0])
        top = points[np.argmax(points[:, 2])]
        bottom = points[np.argmin(points[:, 2])]
        region = TargetRegion(obb_center, axes, half_extents, world_z)
        return SceneEntity(
            instance_id=mask.instance_id,
            label=mask.label,
            pose=pose,
            extent=2.0 * half_extents,
            confidence=mask.confidence,
            keypoints={"centroid": center, "top": top, "bottom": bottom},
            region=region,
            surface_points_world=points,
        )


class EntityResolver:
    """Resolve language references using labels and measured 3-D relations."""

    def __init__(self, front_axis: int = 1) -> None:
        if front_axis not in (0, 1):
            raise ValueError("front_axis must be 0 or 1")
        self.front_axis = front_axis
        self._excluded_source_ids: set[str] = set()

    def exclude_source(self, instance_id: str) -> None:
        """Exclude one episode-local visual identity from later pick goals."""

        if not instance_id:
            raise ValueError("excluded source identity cannot be empty")
        self._excluded_source_ids.add(str(instance_id))

    def reset_source_exclusions(self) -> None:
        self._excluded_source_ids.clear()

    def resolve(self, reference: EntityRef, scene: SceneEstimate) -> SceneEntity:
        label = reference.label
        candidates = self._label_candidates(label, scene)
        if reference.role == "source" and self._excluded_source_ids:
            candidates = [
                item
                for item in candidates
                if item.instance_id not in self._excluded_source_ids
            ]
        if not candidates:
            raise PerceptionError(f"no visual entity matches {label!r}")
        if reference.selector is None:
            return max(candidates, key=lambda x: x.confidence)
        selector = reference.selector
        minimum_count = {
            Relation.LEFTMOST: 2,
            Relation.RIGHTMOST: 2,
            Relation.FRONTMOST: 2,
            Relation.BACKMOST: 2,
            Relation.FIRST: 2,
            Relation.SECOND: 2,
            Relation.MIDDLE: 3,
            Relation.TOPMOST: 2,
            Relation.BOTTOMMOST: 2,
        }.get(selector.relation, 1)
        if len(candidates) < minimum_count:
            raise PerceptionError(
                f"selector {selector.relation.value!r} requires at least "
                f"{minimum_count} visible {label!r} entities"
            )
        if len(candidates) == 1:
            return candidates[0]
        if selector.relation == Relation.CENTER:
            return min(candidates, key=lambda x: np.linalg.norm(x.position[:2]))
        rank_coordinates = self._view_rank_coordinates(candidates)
        if selector.relation in {Relation.LEFTMOST, Relation.FIRST}:
            return min(
                candidates,
                key=lambda x: (
                    float(rank_coordinates[x.instance_id][0]),
                    -x.confidence,
                    x.instance_id,
                ),
            )
        if selector.relation in {Relation.RIGHTMOST, Relation.SECOND}:
            return min(
                candidates,
                key=lambda x: (
                    -float(rank_coordinates[x.instance_id][0]),
                    -x.confidence,
                    x.instance_id,
                ),
            )
        if selector.relation == Relation.FRONTMOST:
            return min(
                candidates,
                key=lambda x: (
                    float(rank_coordinates[x.instance_id][1]),
                    -x.confidence,
                    x.instance_id,
                ),
            )
        if selector.relation == Relation.BACKMOST:
            return min(
                candidates,
                key=lambda x: (
                    -float(rank_coordinates[x.instance_id][1]),
                    -x.confidence,
                    x.instance_id,
                ),
            )
        if selector.relation in {Relation.TOPMOST, Relation.TOP_PART}:
            return max(candidates, key=lambda x: x.position[2])
        if selector.relation in {Relation.BOTTOMMOST, Relation.BOTTOM_PART}:
            return min(candidates, key=lambda x: x.position[2])
        if selector.relation in {Relation.MIDDLE, Relation.MIDDLE_PART}:
            coordinates = np.stack(
                [rank_coordinates[item.instance_id] for item in candidates]
            )
            median = np.median(coordinates, axis=0)
            return min(
                candidates,
                key=lambda x: (
                    float(
                        np.linalg.norm(
                            rank_coordinates[x.instance_id] - median
                        )
                    ),
                    -x.confidence,
                    x.instance_id,
                ),
            )
        refs = [self._best_label(x, scene) for x in selector.references]

        if selector.relation == Relation.BETWEEN:
            midpoint = (refs[0].position + refs[1].position) / 2.0
            return min(candidates, key=lambda x: np.linalg.norm(x.position[:2] - midpoint[:2]))
        anchor = refs[0].position
        if selector.relation == Relation.FARTHEST_FROM:
            return max(candidates, key=lambda x: np.linalg.norm(x.position - anchor))
        if selector.relation == Relation.NEXT_TO:
            return min(candidates, key=lambda x: np.linalg.norm(x.position[:2] - anchor[:2]))
        if selector.relation in {Relation.ON, Relation.IN}:
            return min(candidates, key=lambda x: np.linalg.norm(x.position - anchor))
        if selector.relation == Relation.LEFT_OF:
            return min(candidates, key=lambda x: (x.position[0] >= anchor[0], x.position[0] - anchor[0]))
        if selector.relation == Relation.RIGHT_OF:
            return min(candidates, key=lambda x: (x.position[0] <= anchor[0], anchor[0] - x.position[0]))
        axis = self.front_axis
        if selector.relation == Relation.FRONT_OF:
            return min(candidates, key=lambda x: (x.position[axis] >= anchor[axis], x.position[axis] - anchor[axis]))
        if selector.relation == Relation.BEHIND:
            return min(candidates, key=lambda x: (x.position[axis] <= anchor[axis], anchor[axis] - x.position[axis]))
        raise PerceptionError(f"unsupported visual selector: {selector.relation.value}")

    @staticmethod
    def _view_rank_coordinates(
        candidates: Sequence[SceneEntity],
    ) -> dict[str, FloatArray]:
        """Project ranks into the calibrated fixed-camera image/depth frame."""

        calibrated: dict[str, FloatArray] = {}
        for candidate in candidates:
            origin = candidate.keypoints.get("view_origin")
            right = candidate.keypoints.get("view_right_axis")
            forward = candidate.keypoints.get("view_forward_axis")
            if origin is None or right is None or forward is None:
                calibrated.clear()
                break
            origin = np.asarray(origin, dtype=np.float64)
            right = np.asarray(right, dtype=np.float64)
            forward = np.asarray(forward, dtype=np.float64)
            if not all(
                value.shape == (3,) and np.all(np.isfinite(value))
                for value in (origin, right, forward)
            ):
                calibrated.clear()
                break
            right_norm = float(np.linalg.norm(right))
            forward_norm = float(np.linalg.norm(forward))
            if right_norm < 1e-8 or forward_norm < 1e-8:
                calibrated.clear()
                break
            delta = candidate.position - origin
            depth = float(np.dot(delta, forward / forward_norm))
            if depth <= 1e-5:
                calibrated.clear()
                break
            image_u = float(np.dot(delta, right / right_norm)) / depth
            calibrated[candidate.instance_id] = np.array(
                (image_u, depth), dtype=np.float64
            )
        if len(calibrated) == len(candidates):
            return calibrated

        # Equivalent public LIBERO frame convention for synthetic/minimal
        # observations without fixed-camera calibration: image-right is
        # +world-Y and optical-near/front is +world-X.
        return {
            candidate.instance_id: np.array(
                (candidate.position[1], -candidate.position[0]),
                dtype=np.float64,
            )
            for candidate in candidates
        }

    @staticmethod
    def _best_label(label: str, scene: SceneEstimate) -> SceneEntity:
        candidates = EntityResolver._label_candidates(label, scene)
        if not candidates:
            raise PerceptionError(f"selector reference {label!r} is not visible")
        return max(candidates, key=lambda x: x.confidence)

    @staticmethod
    def _label_candidates(label: str, scene: SceneEstimate) -> list[SceneEntity]:
        exact = [entity for entity in scene.entities if entity.label == label]
        if exact:
            return exact
        # A nested language qualifier is executable evidence, not decoration.
        # Falling back from ``top drawer of wooden cabinet`` to either a bare
        # drawer or an unrelated high-confidence cabinet would discard that
        # evidence and can select a bowl from the wrong fixture.  The sensor
        # adapter emits the complete canonical query label when DINO grounds
        # it; without that exact qualified anchor the selector must fail closed.
        if " of " in label:
            return []
        return [
            entity
            for entity in scene.entities
            if label in entity.label or entity.label in label
        ]
