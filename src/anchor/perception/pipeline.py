"""End-to-end sensor-only tabletop instance perception."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .gallery import TextureSizeGallery, hsv_histogram
from .geometry import (
    ComponentConfig,
    GeometryError,
    center_ray_intersection,
    connected_components_3d,
    estimate_horizontal_support,
    fit_observed_geometry,
    fuse_rgbd_frames,
    provisional_instance,
)
from .schema import (
    ObjectInstance,
    PointCloud,
    RGBDFrame,
    SceneObservation,
    normalize_label,
    scene_camera_capture_ids,
    scene_capture_id,
    scene_timestamp,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PerceptionConfig:
    """Geometry thresholds in world metres, with LIBERO-safe defaults."""

    workspace_min: FloatArray = field(
        default_factory=lambda: np.array([-0.55, -0.45, -0.15], dtype=np.float64)
    )
    workspace_max: FloatArray = field(
        default_factory=lambda: np.array([0.45, 0.65, 1.35], dtype=np.float64)
    )
    # LIBERO-Object uses a table near world z=0 while Spatial/Goal use one near
    # z=0.90.  The support plane therefore must be data-driven across suites.
    table_z_bounds: tuple[float, float] = (-0.10, 1.05)
    projection_stride: int = 2
    # 2.5 mm admits quantised table-depth ridges that bridge every object into
    # one component at 256 px.  4 mm removes those ridges while retaining the
    # roughly 8--12 mm LIBERO plate / bowl rims.
    min_object_height_m: float = 0.004
    max_object_height_m: float = 0.24
    min_planar_extent_m: float = 0.010
    max_planar_extent_m: float = 0.28
    max_support_gap_m: float | None = None
    reject_workspace_boundary_m: float = 0.012
    component: ComponentConfig = field(default_factory=ComponentConfig)

    def __post_init__(self) -> None:
        lower = np.asarray(self.workspace_min, dtype=np.float64)
        upper = np.asarray(self.workspace_max, dtype=np.float64)
        if lower.shape != (3,) or upper.shape != (3,) or np.any(lower >= upper):
            raise ValueError("workspace bounds must be ordered 3-vectors")
        if self.projection_stride <= 0:
            raise ValueError("projection_stride must be positive")
        if not 0 <= self.min_object_height_m < self.max_object_height_m:
            raise ValueError("invalid object-height interval")
        if not 0 < self.min_planar_extent_m < self.max_planar_extent_m:
            raise ValueError("invalid object-footprint interval")
        if self.max_support_gap_m is not None and self.max_support_gap_m <= 0:
            raise ValueError("max_support_gap_m must be positive or None")
        if self.reject_workspace_boundary_m < 0:
            raise ValueError("reject_workspace_boundary_m cannot be negative")
        object.__setattr__(self, "workspace_min", lower.copy())
        object.__setattr__(self, "workspace_max", upper.copy())


@dataclass(frozen=True)
class PerceptionDiagnostics:
    """All fields remain sensor-derived and are safe to log."""

    fused_point_count: int
    workspace_point_count: int
    object_candidate_point_count: int
    raw_component_count: int
    kept_component_count: int


class SensorOnlyScenePerception:
    """Segment tabletop instances with RGB-D geometry and static asset priors."""

    def __init__(
        self,
        config: PerceptionConfig | None = None,
        *,
        classifier: TextureSizeGallery | None = None,
    ) -> None:
        self.config = config or PerceptionConfig()
        self.classifier = classifier

    def observe(
        self,
        frames: Sequence[RGBDFrame],
        *,
        requested_labels: Sequence[str] | None = None,
    ) -> SceneObservation:
        scene, _ = self.observe_with_diagnostics(frames, requested_labels=requested_labels)
        return scene

    def observe_with_diagnostics(
        self,
        frames: Sequence[RGBDFrame],
        *,
        requested_labels: Sequence[str] | None = None,
    ) -> tuple[SceneObservation, PerceptionDiagnostics]:
        if not frames:
            raise ValueError("at least one RGB-D frame is required")
        cloud = fuse_rgbd_frames(frames, stride=self.config.projection_stride)
        inside = np.all(
            (cloud.points_world >= self.config.workspace_min)
            & (cloud.points_world <= self.config.workspace_max),
            axis=1,
        )
        workspace_cloud = cloud.subset(inside)
        if len(workspace_cloud.points_world) == 0:
            raise GeometryError("no RGB-D points lie inside the configured workspace")
        table_height = estimate_horizontal_support(
            workspace_cloud.points_world,
            z_bounds=self.config.table_z_bounds,
            min_inliers=max(30, self.config.component.min_points),
        )
        relative_height = workspace_cloud.points_world[:, 2] - table_height
        object_mask = (
            (relative_height >= self.config.min_object_height_m)
            & (relative_height <= self.config.max_object_height_m)
        )
        candidate_cloud = workspace_cloud.subset(object_mask)
        raw_components = connected_components_3d(candidate_cloud.points_world, self.config.component)

        component_clouds: list[PointCloud] = []
        for indices in raw_components:
            component = candidate_cloud.subset(indices)
            component_lower = np.quantile(component.points_world, 0.01, axis=0)
            component_upper = np.quantile(component.points_world, 0.99, axis=0)
            margin = self.config.reject_workspace_boundary_m
            if margin > 0 and (
                np.any(component_lower[:2] <= self.config.workspace_min[:2] + margin)
                or np.any(component_upper[:2] >= self.config.workspace_max[:2] - margin)
            ):
                continue
            raw_bottom = float(np.quantile(component.points_world[:, 2], 0.01))
            if (
                self.config.max_support_gap_m is not None
                and raw_bottom - table_height > self.config.max_support_gap_m
            ):
                continue
            observed = fit_observed_geometry(
                component.points_world,
                support_height_m=table_height,
                force_support_plane=True,
            )
            planar_extent = float(np.max(observed.extents_m[:2]))
            if not self.config.min_planar_extent_m <= planar_extent <= self.config.max_planar_extent_m:
                continue
            component_clouds.append(component)

        # Stable IDs do not depend on cKDTree pair ordering.
        component_clouds.sort(
            key=lambda component: tuple(np.median(component.points_world[:, :2], axis=0))
        )
        instances: list[ObjectInstance] = []
        gallery_labels: list[str] | None = None
        classify_instances = self.classifier is not None
        if self.classifier is not None and requested_labels is not None:
            normalized = {normalize_label(label) for label in requested_labels}
            gallery_labels = [label for label in self.classifier.prototypes if label in normalized]
            if not gallery_labels:
                classify_instances = False

        for index, component in enumerate(component_clouds):
            histogram = hsv_histogram(component.colors_rgb)
            instance = provisional_instance(
                f"object-{index:03d}",
                component,
                frames,
                support_height_m=table_height,
                color_histogram=histogram,
            )
            if self.classifier is not None and classify_instances:
                classification = self.classifier.classify(
                    instance,
                    allowed_labels=gallery_labels,
                )
                instance = self._complete_from_prior(
                    instance,
                    component,
                    frames,
                    table_height,
                    classification.dimensions_xyz_m,
                    classification.scores,
                    classification.confidence,
                )
            instances.append(instance)

        scene = SceneObservation(
            timestamp_s=scene_timestamp(frames),
            table_height_m=table_height,
            instances=tuple(instances),
            camera_names=tuple(frame.name for frame in frames),
            capture_id=scene_capture_id(frames),
            camera_capture_ids=scene_camera_capture_ids(frames),
        )
        diagnostics = PerceptionDiagnostics(
            fused_point_count=len(cloud.points_world),
            workspace_point_count=len(workspace_cloud.points_world),
            object_candidate_point_count=len(candidate_cloud.points_world),
            raw_component_count=len(raw_components),
            kept_component_count=len(instances),
        )
        return scene, diagnostics

    @staticmethod
    def _complete_from_prior(
        instance: ObjectInstance,
        component: PointCloud,
        frames: Sequence[RGBDFrame],
        support_height_m: float,
        dimensions_xyz_m: FloatArray,
        scores,
        confidence: float,
    ) -> ObjectInstance:
        dimensions = np.asarray(dimensions_xyz_m, dtype=np.float64)
        planar = sorted(dimensions[:2], reverse=True)
        completed_extents = np.array([planar[0], planar[1], dimensions[2]], dtype=np.float64)
        centre_height = support_height_m + completed_extents[2] / 2.0
        ray_center = center_ray_intersection(
            component,
            frames,
            centre_height,
            fallback_xy=instance.observed.center_world[:2],
        )
        observed_planar = np.sort(instance.observed.extents_m[:2])[::-1]
        prior_planar = np.sort(completed_extents[:2])[::-1]
        coverage = float(np.min(observed_planar / np.maximum(prior_planar, 1e-6)))
        # A complete two-view footprint has a much less biased bounds centre
        # than an oblique centre ray evaluated at the prior mid-height.  Blend
        # toward the ray only as the visible footprint becomes incomplete.
        ray_weight = float(np.clip((0.55 - coverage) / 0.35, 0.0, 1.0))
        completed_center = ray_center
        completed_center[:2] = (
            ray_weight * ray_center[:2]
            + (1.0 - ray_weight) * instance.observed.center_world[:2]
        )
        grasp = completed_center.copy()
        grasp[2] = support_height_m + completed_extents[2]
        geometric_confidence = min(1.0, 0.45 + 0.12 * np.log1p(instance.point_count))
        combined_confidence = float(np.clip(0.65 * confidence + 0.35 * geometric_confidence, 0.0, 1.0))
        return instance.with_semantics(
            scores,
            center_world=completed_center,
            grasp_point_world=grasp,
            extents_m=completed_extents,
            confidence=combined_confidence,
        )
