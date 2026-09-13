"""Sensor-only object geometry for route B.

The module consumes masks predicted by a frozen RGB model and reconstructs
geometry from calibrated depth.  It never asks the simulator for object poses,
instance ids, contacts, or task predicates.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Protocol

import numpy as np
import numpy.typing as npt

from .models import RegionDetection, SceneObject, SceneSnapshot, SensorObservation, UInt8Array


class FrozenRegionModel(Protocol):
    """Grounding/segmentation adapter implemented by a frozen model service."""

    def detect(self, rgb: UInt8Array, queries: Sequence[str]) -> Sequence[RegionDetection]:
        ...


class ScenePerception(Protocol):
    def observe(self, observation: SensorObservation, queries: Sequence[str]) -> SceneSnapshot:
        ...


class RGBDScenePerception:
    """Fuse open-vocabulary masks from all calibrated RGB-D cameras."""

    def __init__(
        self,
        region_model: FrozenRegionModel,
        *,
        min_score: float = 0.35,
        min_points: int = 24,
        max_points_per_detection: int = 4_000,
        min_depth_m: float = 0.05,
        max_depth_m: float = 3.0,
        depth_trim_quantile: float = 0.01,
        mad_scale: float = 7.0,
    ) -> None:
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be in [0, 1]")
        if min_points < 3:
            raise ValueError("min_points must be at least 3")
        if max_points_per_detection < min_points:
            raise ValueError("max_points_per_detection cannot be less than min_points")
        if not 0.0 <= depth_trim_quantile < 0.5:
            raise ValueError("depth_trim_quantile must be in [0, 0.5)")
        self.region_model = region_model
        self.min_score = min_score
        self.min_points = min_points
        self.max_points_per_detection = max_points_per_detection
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m
        self.depth_trim_quantile = depth_trim_quantile
        self.mad_scale = mad_scale
        self._observation_sequence = 0
        self._last_observation: SensorObservation | None = None
        self._observation_timestamp_s = 0.0

    def reset(self) -> None:
        """Reset the episode-local capture token used in scene provenance."""

        self._observation_sequence = 0
        self._last_observation = None
        self._observation_timestamp_s = 0.0

    def begin_observation(
        self,
        observation: SensorObservation,
        sequence: int,
    ) -> None:
        """Bind every query in one controller act to the same capture token."""

        if sequence < 0:
            raise ValueError("observation sequence must be non-negative")
        self._last_observation = observation
        self._observation_timestamp_s = float(sequence)
        self._observation_sequence = max(self._observation_sequence, sequence + 1)

    def _capture_timestamp_s(self, observation: SensorObservation) -> float:
        if observation is not self._last_observation:
            self._last_observation = observation
            self._observation_timestamp_s = float(self._observation_sequence)
            self._observation_sequence += 1
        return self._observation_timestamp_s

    def observe(self, observation: SensorObservation, queries: Sequence[str]) -> SceneSnapshot:
        timestamp_s = self._capture_timestamp_s(observation)
        normalized_queries = tuple(
            dict.fromkeys(query.strip().lower() for query in queries if query.strip())
        )
        if not normalized_queries:
            return SceneSnapshot(timestamp_s, {})

        points_by_query: dict[str, list[npt.NDArray[np.float64]]] = defaultdict(list)
        scores_by_query: dict[str, list[tuple[float, int]]] = defaultdict(list)

        for frame in observation.cameras.values():
            detections = self.region_model.detect(frame.rgb, normalized_queries)
            for detection in detections:
                query = detection.query.strip().lower()
                if query not in normalized_queries or detection.score < self.min_score:
                    continue
                if detection.mask.shape != frame.depth_m.shape:
                    raise ValueError(
                        f"mask for {query!r} has shape {detection.mask.shape}, "
                        f"expected {frame.depth_m.shape}"
                    )
                camera_points = self._backproject_mask(
                    frame.depth_m,
                    detection.mask,
                    frame.intrinsics,
                    observation_v_flipped=frame.observation_v_flipped,
                )
                if camera_points.shape[0] < self.min_points:
                    continue
                world_points = frame.world_from_camera.transform_points(camera_points)
                points_by_query[query].append(world_points)
                scores_by_query[query].append((float(detection.score), world_points.shape[0]))

        objects: dict[str, SceneObject] = {}
        for query, point_groups in points_by_query.items():
            points = np.concatenate(point_groups, axis=0)
            points = self._reject_world_outliers(points)
            if points.shape[0] < self.min_points:
                continue
            weighted_score = sum(score * count for score, count in scores_by_query[query]) / sum(
                count for _, count in scores_by_query[query]
            )
            objects[query] = self._fit_object(query, points, float(weighted_score))
        return SceneSnapshot(timestamp_s, objects)

    def _backproject_mask(
        self,
        depth_m,
        mask,
        intrinsics,
        *,
        observation_v_flipped: bool = False,
    ) -> npt.NDArray[np.float64]:
        depth = np.asarray(depth_m, dtype=np.float64)
        valid = (
            np.asarray(mask, dtype=bool)
            & np.isfinite(depth)
            & (depth >= self.min_depth_m)
            & (depth <= self.max_depth_m)
        )
        if np.count_nonzero(valid) < self.min_points:
            return np.empty((0, 3), dtype=np.float64)

        masked_depth = depth[valid]
        if self.depth_trim_quantile > 0 and masked_depth.size >= 2 * self.min_points:
            lower, upper = np.quantile(
                masked_depth, (self.depth_trim_quantile, 1.0 - self.depth_trim_quantile)
            )
            # Keep constant-depth masks intact.
            if upper > lower:
                valid &= (depth >= lower) & (depth <= upper)

        rows, cols = np.nonzero(valid)
        z = depth[rows, cols]
        if z.size > self.max_points_per_detection:
            # Deterministic, spatially ordered subsampling keeps tests reproducible.
            indices = np.linspace(0, z.size - 1, self.max_points_per_detection, dtype=np.int64)
            rows, cols, z = rows[indices], cols[indices], z[indices]
        x = (cols.astype(np.float64) - intrinsics.cx) * z / intrinsics.fx
        projection_rows = (
            intrinsics.height - 1 - rows if observation_v_flipped else rows
        )
        y = (projection_rows.astype(np.float64) - intrinsics.cy) * z / intrinsics.fy
        return np.column_stack((x, y, z))

    def _reject_world_outliers(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if points.shape[0] < 2 * self.min_points:
            return points
        median = np.median(points, axis=0)
        absolute = np.abs(points - median)
        mad = np.median(absolute, axis=0)
        # A planar or axis-aligned cloud legitimately has zero MAD on one axis.
        scale = np.maximum(1.4826 * mad, 1e-4)
        keep = np.all(absolute <= self.mad_scale * scale, axis=1)
        return points[keep] if np.count_nonzero(keep) >= self.min_points else points

    @staticmethod
    def _fit_object(
        name: str, points: npt.NDArray[np.float64], confidence: float
    ) -> SceneObject:
        centroid = np.median(points, axis=0)
        centered = points - centroid
        covariance = centered.T @ centered / max(points.shape[0] - 1, 1)
        _, eigenvectors = np.linalg.eigh(covariance)
        axes = eigenvectors[:, ::-1]
        if np.linalg.det(axes) < 0:
            axes[:, -1] *= -1
        projected = centered @ axes
        projected_min = np.quantile(projected, 0.01, axis=0)
        projected_max = np.quantile(projected, 0.99, axis=0)
        extents = np.maximum(projected_max - projected_min, 0.0)
        bounds_min = np.quantile(points, 0.01, axis=0)
        bounds_max = np.quantile(points, 0.99, axis=0)
        return SceneObject(
            name=name,
            centroid_world=centroid,
            axes_world=axes,
            extents_m=extents,
            bounds_min_world=bounds_min,
            bounds_max_world=bounds_max,
            confidence=confidence,
            point_count=int(points.shape[0]),
            surface_points_world=points,
        )
