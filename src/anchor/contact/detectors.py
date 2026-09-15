"""RGB-D fixture, handle, knob, and plate detectors for LIBERO-Goal.

The implementation intentionally uses visible appearance, metric depth and
camera calibration only.  Static dimensional thresholds describe fixture
families; no simulator model names, poses, segmentation ids, or joint values
are inspected.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np

from anchor.common import (
    CameraFrame,
    RobotObservation,
    backproject_depth,
    project_world_points,
)

from .schema import ContactTarget, GoalSkillKind, PushTarget


def _components(mask: np.ndarray) -> list[tuple[int, np.ndarray, np.ndarray]]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return [(index, stats[index], labels == index) for index in range(1, count)]


def _points(frame: CameraFrame, mask: np.ndarray) -> np.ndarray:
    if np.count_nonzero(mask) < 4:
        return np.empty((0, 3), dtype=np.float64)
    return backproject_depth(frame, mask=mask, world=True)


def _horizontal_axis(points: np.ndarray) -> np.ndarray:
    if len(points) < 4:
        return np.array([1.0, 0.0, 0.0])
    xy = points[:, :2] - np.median(points[:, :2], axis=0)
    covariance = xy.T @ xy / max(len(xy) - 1, 1)
    _, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, -1]
    if axis[0] < 0:
        axis *= -1
    return np.array([axis[0], axis[1], 0.0], dtype=np.float64)


def _planar_median_neighborhood(
    points: np.ndarray,
    *,
    radius_m: float,
    min_points: int,
) -> np.ndarray:
    """Remove disconnected planar tails while retaining a real depth cluster."""

    if len(points) < min_points:
        return points
    median_xy = np.median(points[:, :2], axis=0)
    inliers = points[
        np.linalg.norm(points[:, :2] - median_xy[None, :], axis=1) <= radius_m
    ]
    return inliers if len(inliers) >= min_points else points


def _projected_quantile_center(
    points: np.ndarray,
    basis_world: np.ndarray,
    *,
    tail_quantile: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the midpoint and spans of a robust projected bounding box.

    A surface contour can be sampled much more densely on one visible side.
    Its coordinatewise sample median then follows pixel density instead of
    the feature's geometric centre.  Midpoints of paired projection
    quantiles are invariant to that imbalance while trimming isolated tails.
    """

    basis = np.asarray(basis_world, dtype=np.float64)
    if basis.shape != (3, 3) or not np.all(np.isfinite(basis)):
        raise ValueError("projected centre basis must be a finite 3x3 matrix")
    if not np.allclose(basis.T @ basis, np.eye(3), atol=0.08):
        raise ValueError("projected centre basis must have orthonormal columns")
    if not 0.0 <= tail_quantile < 0.5:
        raise ValueError("tail quantile must lie in [0, 0.5)")
    if len(points) < 4:
        raise ValueError("projected centre needs at least four points")
    coordinates = np.asarray(points, dtype=np.float64) @ basis
    lower, upper = np.quantile(
        coordinates,
        (tail_quantile, 1.0 - tail_quantile),
        axis=0,
    )
    midpoint = (lower + upper) / 2.0
    return basis @ midpoint, upper - lower


def _unit_xy(vector: np.ndarray, fallback: tuple[float, float] = (0.0, -1.0)) -> np.ndarray:
    result = np.array([vector[0], vector[1], 0.0], dtype=np.float64)
    norm = float(np.linalg.norm(result))
    if norm < 1e-8:
        return np.array([fallback[0], fallback[1], 0.0], dtype=np.float64)
    return result / norm


def _kmeans_levels(points: np.ndarray, count: int = 3) -> list[np.ndarray]:
    if len(points) < count * 12:
        return []
    z = points[:, 2]
    centers = np.quantile(z, np.linspace(0.18, 0.82, count))
    for _ in range(20):
        assignment = np.argmin(np.abs(z[:, None] - centers[None, :]), axis=1)
        updated = np.array(
            [np.median(z[assignment == index]) if np.any(assignment == index) else centers[index]
             for index in range(count)]
        )
        if np.allclose(updated, centers, atol=1e-5):
            break
        centers = updated
    groups = [points[assignment == index] for index in np.argsort(centers)]
    if any(len(group) < 12 for group in groups):
        return []
    return groups


@dataclass(frozen=True, slots=True)
class FixtureDetectorConfig:
    cabinet_dark_threshold: int = 55
    neutral_low: int = 45
    neutral_high: int = 145
    neutral_chroma: int = 24
    # Drawer handles need a narrower appearance band than the generic neutral
    # fixture mask: in the white-cabinet view, the cabinet face is connected
    # to the handles when the upper bound is 145.
    drawer_handle_neutral_low: int = 50
    drawer_handle_neutral_high: int = 115
    drawer_handle_neutral_chroma: int = 24
    drawer_handle_depth_window_m: float = 0.012
    drawer_handle_depth_window_step_m: float = 0.004
    handle_z_range: tuple[float, float] = (0.92, 1.14)
    knob_z_range: tuple[float, float] = (0.88, 1.07)
    front_clearance_m: float = 0.10
    # The same episode-local handle must be trackable in both articulation
    # directions.  OPEN moves along the frozen outward normal; CLOSE moves an
    # equal bounded distance against it.  Axis, height and lateral gates below
    # still prevent a neighbouring drawer from becoming an alias.
    drawer_track_longitudinal_range_m: tuple[float, float] = (-0.24, 0.24)
    drawer_track_lateral_m: float = 0.055
    drawer_track_height_m: float = 0.030
    drawer_track_axis_cosine: float = 0.72
    drawer_track_outward_cosine: float = 0.72
    drawer_planar_inlier_radius_m: float = 0.050
    drawer_planar_inlier_min_points: int = 12
    drawer_center_tail_quantile: float = 0.05
    drawer_axis_span_range_m: tuple[float, float] = (0.025, 0.18)
    drawer_max_normal_span_m: float = 0.070
    drawer_max_vertical_span_m: float = 0.050
    drawer_max_center_correction_m: float = 0.025


@dataclass(frozen=True, slots=True)
class MicrowaveDoorHandleConfig:
    """Metric and appearance gates for a microwave door feature.

    The fixture OBB is supplied by a separate sensor-only scene estimator.
    These thresholds only describe the appearance and dimensions of the
    handle / vertical door edge inside that OBB-derived search volume.
    """

    neutral_low: int = 12
    neutral_high: int = 185
    neutral_chroma: int = 30
    dark_high: int = 92
    local_contrast_threshold: int = 9
    min_component_pixels: int = 10
    max_component_area_fraction: float = 0.12
    max_component_width_fraction: float = 0.45
    max_component_height_fraction: float = 0.88
    component_border_margin_px: int = 1
    min_vertical_span_m: float = 0.035
    max_vertical_span_m: float = 0.34
    min_vertical_aspect: float = 1.35
    closed_max_lateral_span_m: float = 0.075
    closed_max_outward_span_m: float = 0.070
    closed_front_inner_margin_m: float = 0.025
    closed_front_outer_margin_m: float = 0.060
    closed_handle_side_min_fraction: float = 0.25
    closed_handle_side_max_fraction: float = 0.95
    closed_fallback_front_offset_m: float = 0.004
    closed_fallback_handle_side_fraction: float = 0.55
    closed_candidate_slot_radius_m: float = 0.060
    closed_candidate_vertical_offset_m: float = 0.055
    open_front_clearance_m: float = 0.030
    open_outward_reach_m: float = 0.38
    open_handle_side_min_fraction: float = 0.10
    open_handle_side_reach_m: float = 0.28
    # Several vertical contours can survive the open-door structural gates
    # (window frame, panel seam, and the actual handle / free edge).  Among
    # otherwise valid contours, retain the small handle-side frontier and use
    # confidence only inside that metric band.  This is reconstructed from
    # calibrated RGB-D and the frozen appliance OBB; it is not an object-id or
    # simulator-geometry prior.
    open_free_edge_side_tolerance_m: float = 0.018
    # An explicitly observed open edge must admit a non-trivial rotation to a
    # corresponding closed handle slot.  Keeping both OBB long-side hinge
    # hypotheses and applying this signed-angle gate prevents a partial body
    # box (which can include the open door) from selecting the opposite jamb
    # merely because its radius happens to be closer to a nominal width.
    open_hinge_close_angle_range_rad: tuple[float, float] = (0.55, 2.40)
    open_max_lateral_span_m: float = 0.105
    open_max_outward_span_m: float = 0.105
    open_vertical_core_radius_m: float = 0.026
    open_vertical_core_min_points: int = 12
    open_vertical_core_min_height_fraction: float = 0.35
    open_vertical_core_min_span_m: float = 0.055
    open_vertical_core_aspect: float = 2.0
    open_vertical_core_max_frame_span_m: float = 0.22
    center_tail_quantile: float = 0.05
    max_center_correction_m: float = 0.035
    vertical_margin_m: float = 0.040
    robot_exclusion_radius_m: float = 0.070
    cross_view_agreement_m: float = 0.075

    def __post_init__(self) -> None:
        if not 0 <= self.neutral_low < self.neutral_high <= 255:
            raise ValueError("microwave neutral intensity range is invalid")
        if not 0 <= self.dark_high <= 255 or self.neutral_chroma < 0:
            raise ValueError("microwave appearance thresholds are invalid")
        positive = (
            self.local_contrast_threshold,
            self.min_component_pixels,
            self.max_component_area_fraction,
            self.max_component_width_fraction,
            self.max_component_height_fraction,
            self.component_border_margin_px,
            self.min_vertical_span_m,
            self.max_vertical_span_m,
            self.min_vertical_aspect,
            self.closed_max_lateral_span_m,
            self.closed_max_outward_span_m,
            self.closed_front_inner_margin_m,
            self.closed_front_outer_margin_m,
            self.closed_handle_side_min_fraction,
            self.closed_handle_side_max_fraction,
            self.closed_fallback_front_offset_m,
            self.closed_fallback_handle_side_fraction,
            self.closed_candidate_slot_radius_m,
            self.closed_candidate_vertical_offset_m,
            self.open_front_clearance_m,
            self.open_outward_reach_m,
            self.open_handle_side_min_fraction,
            self.open_handle_side_reach_m,
            self.open_free_edge_side_tolerance_m,
            *self.open_hinge_close_angle_range_rad,
            self.open_max_lateral_span_m,
            self.open_max_outward_span_m,
            self.open_vertical_core_radius_m,
            self.open_vertical_core_min_points,
            self.open_vertical_core_min_height_fraction,
            self.open_vertical_core_min_span_m,
            self.open_vertical_core_aspect,
            self.open_vertical_core_max_frame_span_m,
            self.max_center_correction_m,
            self.vertical_margin_m,
            self.robot_exclusion_radius_m,
            self.cross_view_agreement_m,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("microwave detector metric gates must be positive")
        if not 0.0 <= self.center_tail_quantile < 0.5:
            raise ValueError("microwave centre tail quantile must lie in [0, 0.5)")
        if self.max_vertical_span_m <= self.min_vertical_span_m:
            raise ValueError("microwave vertical span bounds are invalid")
        close_angle_min, close_angle_max = self.open_hinge_close_angle_range_rad
        if not 0.0 < close_angle_min < close_angle_max < np.pi:
            raise ValueError("microwave open-state close-angle bounds are invalid")


class DrawerHandleDetector:
    """Recover three aligned drawer-handle height modes from RGB-D."""

    def __init__(self, config: FixtureDetectorConfig | None = None) -> None:
        self.config = config or FixtureDetectorConfig()

    def detect(self, observation: RobotObservation, level: str = "middle") -> ContactTarget:
        if level not in {"top", "middle", "bottom"}:
            raise ValueError("drawer level must be top, middle, or bottom")
        proposals: list[ContactTarget] = []
        for name in ("agentview", "wrist"):
            proposal = self._detect_frame(observation.cameras[name], level)
            if proposal is not None:
                proposals.append(proposal)
        if not proposals:
            raise LookupError(f"no {level} drawer handle was visible in either RGB-D view")
        anchor = proposals[0]
        agreeing = [
            item for item in proposals
            if np.linalg.norm(item.point_world - anchor.point_world) <= 0.065
            and abs(float(item.point_world[2] - anchor.point_world[2])) <= 0.025
            and float(item.outward_world @ anchor.outward_world) >= 0.95
            and np.linalg.norm(item.fixture_center_world[:2] - anchor.fixture_center_world[:2]) <= .040
        ]
        if len(agreeing) == 1:
            return anchor
        weights = np.asarray([item.confidence for item in agreeing], dtype=np.float64)
        weights /= weights.sum()
        point = sum(weight * item.point_world for weight, item in zip(weights, agreeing))
        fixture = sum(weight * item.fixture_center_world for weight, item in zip(weights, agreeing))
        axis = sum(weight * item.feature_axis_world for weight, item in zip(weights, agreeing))
        outward = sum(weight * item.outward_world for weight, item in zip(weights, agreeing))
        return ContactTarget(
            GoalSkillKind.OPEN_DRAWER,
            point,
            outward,
            outward,
            fixture,
            axis,
            min(0.98, float(max(weights) + 0.35)),
            tuple(item.source_cameras[0] for item in agreeing),
        )

    def track(
        self,
        observation: RobotObservation,
        reference: ContactTarget,
        level: str = "middle",
    ) -> ContactTarget:
        """Associate the same visible handle with an episode-local RGB-D anchor.

        Opening one drawer can occlude a lower handle, so a fresh three-mode
        height sort may relabel the requested middle handle as the current
        ``bottom`` mode.  Tracking therefore enumerates every current height
        mode in both cameras and accepts only motion compatible with the
        reference fixture frame: bounded travel along the drawer normal,
        little handle-axis/height drift, and consistent handle/outward axes.
        """

        if level not in {"top", "middle", "bottom"}:
            raise ValueError("drawer level must be top, middle, or bottom")
        if reference.kind not in {
            GoalSkillKind.OPEN_DRAWER,
            GoalSkillKind.CLOSE_DRAWER,
        }:
            raise ValueError("drawer tracking requires a drawer-handle reference")

        lower, upper = self.config.drawer_track_longitudinal_range_m
        accepted: list[tuple[float, ContactTarget]] = []
        for camera_name in ("agentview", "wrist"):
            frame = observation.cameras[camera_name]
            for visible_rank in ("bottom", "middle", "top"):
                candidate = self._detect_frame(frame, visible_rank)
                if candidate is None:
                    continue
                delta = candidate.point_world - reference.point_world
                longitudinal = float(np.dot(delta, reference.outward_world))
                lateral = abs(float(np.dot(delta, reference.feature_axis_world)))
                height = abs(float(delta[2]))
                axis_agreement = abs(
                    float(
                        np.dot(
                            candidate.feature_axis_world,
                            reference.feature_axis_world,
                        )
                    )
                )
                outward_agreement = float(
                    np.dot(candidate.outward_world, reference.outward_world)
                )
                if not (
                    lower <= longitudinal <= upper
                    and lateral <= self.config.drawer_track_lateral_m
                    and height <= self.config.drawer_track_height_m
                    and axis_agreement >= self.config.drawer_track_axis_cosine
                    and outward_agreement >= self.config.drawer_track_outward_cosine
                ):
                    continue
                score = (
                    lateral / self.config.drawer_track_lateral_m
                    + height / self.config.drawer_track_height_m
                    + (1.0 - axis_agreement)
                    / (1.0 - self.config.drawer_track_axis_cosine)
                    + (1.0 - outward_agreement)
                    / (1.0 - self.config.drawer_track_outward_cosine)
                    - 0.20 * candidate.confidence
                )
                accepted.append((score, candidate))

        if not accepted:
            raise LookupError(
                f"no {level} drawer handle matched the episode visual anchor"
            )
        _, selected = min(accepted, key=lambda item: item[0])
        # Refresh only the moving point.  The opening-time fixture frame is the
        # stable sensor-derived identity; current dark-body medians can move
        # under arm occlusion or when the drawer front dominates the image.
        return ContactTarget(
            reference.kind,
            selected.point_world,
            reference.axis_world,
            reference.outward_world,
            reference.fixture_center_world,
            reference.feature_axis_world,
            selected.confidence,
            selected.source_cameras,
        )

    def detect_front_plane(
        self,
        observation: RobotObservation,
        handle: ContactTarget,
        level: str,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Fit a same-level drawer-front plane around a tracked handle.

        The fit is deliberately local to the visible handle in each public
        RGB-D view.  It rejects the cabinet's middle spine/jamb by requiring
        a lower, broad support patch in the handle's level and a normal
        consistent with the tracked drawer outward ray.  No segmentation id,
        simulator body, or task metadata participates in this measurement.
        """

        if level not in {"top", "middle", "bottom"}:
            raise ValueError("drawer level must be top, middle, or bottom")
        normal = _unit_xy(handle.outward_world)
        axis = _unit_xy(handle.feature_axis_world, fallback=(1.0, 0.0))
        if float(abs(np.dot(normal, axis))) > 0.25:
            raise LookupError("drawer handle frame is not planar")
        support_by_camera: dict[str, np.ndarray] = {}
        for camera_name in ("agentview", "wrist"):
            frame = observation.cameras[camera_name]
            uv, camera_z = project_world_points(handle.point_world[None], frame)
            if not np.isfinite(camera_z[0]) or camera_z[0] <= 0.0:
                continue
            u, v = uv[0]
            rows, cols = np.indices(frame.depth_m.shape)
            # A 90x95 px local crop covers the front facade but avoids most
            # unrelated cabinet shelves and neighbouring drawer levels.
            local = (np.abs(cols - u) <= 45.0) & (np.abs(rows - v) <= 48.0)
            world = backproject_depth(frame, mask=local, stride=1, world=True)
            if len(world) < 24:
                continue
            delta = world - handle.point_world[None, :]
            lateral = delta @ axis
            vertical = delta[:, 2]
            outward = delta @ normal
            # The front facade is below the handle, slightly inward from the
            # protruding bar.  Keep a generous metric band for perspective,
            # but not enough to admit the middle frame or cabinet rear.
            support = world[
                (np.abs(lateral) <= 0.085)
                & (vertical >= -0.070)
                & (vertical <= -0.012)
                & (outward >= -0.080)
                & (outward <= 0.015)
            ]
            if len(support) >= 24:
                support_by_camera[camera_name] = support
        if not support_by_camera:
            raise LookupError(
                "no same-level RGB-D drawer-front support"
            )

        # A drawer bar protrudes from its facade.  In the local crop, the
        # facade is therefore the closest *negative* depth mode along the
        # frozen outward ray.  Fitting all eligible samples at once is not
        # robust: a cabinet shelf spanning -80..-50 mm can outnumber the true
        # face at roughly -20 mm and make an unconstrained SVD return a nearly
        # vertical normal.  Find narrow normal-coordinate modes first, demand
        # agreement across available RGB-D views, and only then fit a plane.
        # A wrist view that cannot see the drawer must not veto a well-measured
        # plane in the fixed camera.
        minimum_behind_m = 0.004
        maximum_behind_m = 0.050
        mode_bin_m = 0.005
        mode_half_width_m = 0.008
        per_view_minimum = 12
        depths_by_camera = {
            camera_name: (points - handle.point_world[None, :]) @ normal
            for camera_name, points in support_by_camera.items()
        }
        pooled_depth = np.concatenate(tuple(depths_by_camera.values()))
        eligible = pooled_depth[
            (pooled_depth <= -minimum_behind_m)
            & (pooled_depth >= -maximum_behind_m)
        ]
        if len(eligible) < len(support_by_camera) * per_view_minimum:
            raise LookupError(
                "no negative behind-handle drawer-front mode"
            )
        mode_indices = np.floor(eligible / mode_bin_m).astype(np.int64)
        modes: list[tuple[float, np.ndarray, np.ndarray]] = []
        for mode_index in np.unique(mode_indices):
            mode_values = eligible[mode_indices == mode_index]
            if len(mode_values) < len(support_by_camera) * per_view_minimum:
                continue
            mode_center = float(np.median(mode_values))
            selected_by_camera: dict[str, np.ndarray] = {}
            for camera_name, points in support_by_camera.items():
                selected = points[
                    np.abs(depths_by_camera[camera_name] - mode_center)
                    <= mode_half_width_m
                ]
                if len(selected) < per_view_minimum:
                    break
                selected_by_camera[camera_name] = selected
            if set(selected_by_camera) != set(support_by_camera):
                continue
            view_depth_medians = [
                float(
                    np.median(
                        (points - handle.point_world[None, :]) @ normal
                    )
                )
                for points in selected_by_camera.values()
            ]
            if np.ptp(view_depth_medians) > 0.010:
                continue
            points = np.concatenate(tuple(selected_by_camera.values()), axis=0)
            center = np.median(points, axis=0)
            centered = points - center
            _, _, vectors = np.linalg.svd(centered, full_matrices=False)
            fitted_normal = vectors[-1]
            if float(np.dot(fitted_normal, normal)) < 0.0:
                fitted_normal *= -1.0
            if float(np.dot(fitted_normal, normal)) < 0.82:
                continue
            residual = np.abs((points - center) @ fitted_normal)
            inlier_mask = residual <= 0.008
            inliers = points[inlier_mask]
            if len(inliers) < max(24, int(np.ceil(0.70 * len(points)))):
                continue
            camera_ids = np.concatenate(
                tuple(
                    np.full(len(points), index, dtype=np.int8)
                    for index, points in enumerate(selected_by_camera.values())
                )
            )
            if any(
                int(np.count_nonzero(camera_ids[inlier_mask] == index))
                < per_view_minimum
                for index in range(len(selected_by_camera))
            ):
                continue
            spans = np.ptp(
                np.column_stack((inliers @ axis, inliers[:, 2])), axis=0
            )
            if float(spans[0]) < 0.060 or float(spans[1]) < 0.020:
                continue
            depth_median = float(
                np.median((inliers - handle.point_world[None, :]) @ normal)
            )
            if not -maximum_behind_m <= depth_median <= -minimum_behind_m:
                continue
            modes.append((depth_median, inliers, fitted_normal))
        if not modes:
            raise LookupError(
                "no negative drawer-front plane mode passed residual gates"
            )
        # Largest negative coordinate is the nearest validated facade behind
        # the protruding bar; farther parallel modes are cabinet internals.
        _, inliers, fitted_normal = max(modes, key=lambda item: item[0])
        point = np.median(inliers, axis=0)
        # Keep the contact below the bar and away from its vertical jambs.
        point -= axis * float(np.dot(point - handle.point_world, axis))
        point[2] = min(point[2], handle.point_world[2] - 0.018)
        return point, fitted_normal, float(np.ptp(inliers @ axis))

    def _detect_frame(self, frame: CameraFrame, level: str) -> ContactTarget | None:
        legacy = self._detect_frame_legacy(frame, level)
        global_proposal = self._detect_frame_global_triplet(frame, level)
        if global_proposal is None:
            return legacy
        if legacy is None:
            return global_proposal
        # Preserve the historical centre estimator when it is already
        # consistent with the globally coherent triplet.  If it lands far
        # from every global handle, it is typically the unrelated dark stove
        # body that motivated this detector path.
        if (
            np.linalg.norm(global_proposal.point_world - legacy.point_world)
            <= 0.080
            and abs(float(global_proposal.point_world[2] - legacy.point_world[2])) <= 0.018
            and float(global_proposal.outward_world @ legacy.outward_world) >= 0.82
            and abs(
                float(
                    np.dot(
                        global_proposal.feature_axis_world,
                        legacy.feature_axis_world,
                    )
                )
            )
            >= 0.82
        ):
            return legacy
        return global_proposal

    def _detect_frame_legacy(
        self,
        frame: CameraFrame,
        level: str,
    ) -> ContactTarget | None:
        # Historical detector retained as a conservative fallback for
        # cropped/occluded views where the global triplet is unavailable.
        rgb = frame.rgb
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        dark = cv2.morphologyEx(
            (gray < self.config.cabinet_dark_threshold).astype(np.uint8),
            cv2.MORPH_CLOSE,
            np.ones((5, 5), dtype=np.uint8),
        )
        candidates: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
        for _, stats, mask in _components(dark):
            x, y, width, height, area = map(int, stats)
            if area < 700 or width < 30 or height < 30:
                continue
            world = _points(frame, mask)
            if len(world) < 100:
                continue
            median_z = float(np.median(world[:, 2]))
            if not 0.88 <= median_z <= 1.16:
                continue
            # Cabinet bodies have much more visible area than isolated robot links.
            candidates.append((area * (1.2 - abs(median_z - 1.02)), stats, mask, world))
        if not candidates:
            return None
        _, stats, _body_mask, body_points = max(candidates, key=lambda item: item[0])
        x, y, width, height, _ = map(int, stats)
        margin = 24
        roi = np.zeros(gray.shape, dtype=bool)
        roi[max(0, y - margin): min(frame.calibration.height, y + height + margin),
            max(0, x - margin): min(frame.calibration.width, x + width + margin)] = True
        chroma = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
        neutral = (
            roi
            & (gray >= self.config.neutral_low)
            & (gray <= self.config.neutral_high)
            & (chroma <= self.config.neutral_chroma)
        )
        handle_points = _points(frame, neutral)
        lower, upper = self.config.handle_z_range
        body_xy = np.median(body_points[:, :2], axis=0)
        handle_points = handle_points[
            (handle_points[:, 2] >= lower)
            & (handle_points[:, 2] <= upper)
            & (np.linalg.norm(handle_points[:, :2] - body_xy, axis=1) <= 0.26)
        ]
        groups = _kmeans_levels(handle_points, 3)
        if len(groups) != 3:
            return None
        selected = groups[{"bottom": 0, "middle": 1, "top": 2}[level]]
        selected = _planar_median_neighborhood(
            selected,
            radius_m=self.config.drawer_planar_inlier_radius_m,
            min_points=self.config.drawer_planar_inlier_min_points,
        )
        sample_median = np.median(selected, axis=0)
        axis = _horizontal_axis(selected)
        normal = np.array([-axis[1], axis[0], 0.0])
        if np.dot(normal[:2], sample_median[:2] - body_xy) < 0:
            normal *= -1
        outward = _unit_xy(normal)
        basis = np.column_stack(
            (axis, outward, np.array((0.0, 0.0, 1.0), dtype=np.float64))
        )
        try:
            point, spans = _projected_quantile_center(
                selected,
                basis,
                tail_quantile=self.config.drawer_center_tail_quantile,
            )
        except ValueError:
            return None
        axis_lower, axis_upper = self.config.drawer_axis_span_range_m
        if not (
            axis_lower <= spans[0] <= axis_upper
            and spans[1] <= self.config.drawer_max_normal_span_m
            and spans[2] <= self.config.drawer_max_vertical_span_m
            and np.linalg.norm(point - sample_median)
            <= self.config.drawer_max_center_correction_m
        ):
            # A tiny fragment, broad panel, or strongly one-sided crop does
            # not support a defensible handle centre.  Let the other camera
            # vote; if neither does, detection fails closed.
            return None
        separation = min(
            abs(float(np.median(groups[1][:, 2]) - np.median(groups[0][:, 2]))),
            abs(float(np.median(groups[2][:, 2]) - np.median(groups[1][:, 2]))),
        )
        confidence = float(np.clip(0.45 + 2.5 * separation + len(selected) / 800.0, 0, 0.95))
        fixture = np.array([body_xy[0], body_xy[1], point[2]])
        return ContactTarget(
            GoalSkillKind.OPEN_DRAWER,
            point,
            outward,
            outward,
            fixture,
            axis,
            confidence,
            (frame.calibration.name,),
        )

    def _detect_frame_global_triplet(
        self,
        frame: CameraFrame,
        level: str,
        *,
        allow_roof_fallback: bool = True,
    ) -> ContactTarget | None:
        """Find three aligned neutral handle bars without a dark body ROI.

        The candidates are deliberately assembled from public RGB-D geometry:
        each must have a bounded horizontal span and thin depth/height extent,
        while the selected triplet must have three ordered height levels,
        parallel axes, and a common lateral line.  One drawer may be displaced
        along the inferred outward normal by up to 240 mm, so that direction is
        not used as an equality constraint.  The camera origin signs the
        normal because the cabinet body may be white or partially absent.
        """

        if level not in {"top", "middle", "bottom"}:
            raise ValueError("drawer level must be top, middle, or bottom")
        rgb = frame.rgb
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        chroma = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
        neutral = (
            (gray >= self.config.drawer_handle_neutral_low)
            & (gray <= self.config.drawer_handle_neutral_high)
            & (chroma <= self.config.drawer_handle_neutral_chroma)
        )
        candidates: list[dict[str, object]] = []
        image_area = frame.calibration.width * frame.calibration.height
        for _, stats, component in _components(neutral):
            x, y, width, height, area = map(int, stats)
            if area < 24 or area > 0.035 * image_area:
                continue
            if width < 4 or height < 3:
                continue
            points = _points(frame, component)
            if len(points) < 12 or not np.all(np.isfinite(points)):
                continue
            # A single RGB component can contain several handles at different
            # metric depths (the dark cabinet trim joins them in projection).
            # Re-windowing by depth recovers each visible bar without using a
            # body mask or simulator geometry.
            lower, upper = self.config.handle_z_range
            window = self.config.drawer_handle_depth_window_m
            step = self.config.drawer_handle_depth_window_step_m
            component_candidates: list[dict[str, object]] = []
            for z_center in np.arange(lower, upper + step * 0.5, step):
                subset = points[np.abs(points[:, 2] - z_center) <= window]
                if len(subset) < 12:
                    continue
                median_z = float(np.median(subset[:, 2]))
                if not lower <= median_z <= upper:
                    continue
                axis = _horizontal_axis(subset)
                axis_norm = float(np.linalg.norm(axis[:2]))
                if axis_norm < 1e-6:
                    continue
                axis /= axis_norm
                normal = np.array((-axis[1], axis[0], 0.0), dtype=np.float64)
                basis = np.column_stack((axis, normal, np.array((0.0, 0.0, 1.0))))
                try:
                    center, spans = _projected_quantile_center(
                        subset,
                        basis,
                        tail_quantile=self.config.drawer_center_tail_quantile,
                    )
                except ValueError:
                    continue
                axis_span, normal_span, vertical_span = map(float, spans)
                axis_lower, axis_upper = self.config.drawer_axis_span_range_m
                if not (
                    axis_lower <= axis_span <= axis_upper
                    and normal_span <= self.config.drawer_max_normal_span_m
                    and vertical_span <= self.config.drawer_max_vertical_span_m
                ):
                    continue
                proposal = {
                    "center": center,
                    "axis": axis,
                    "normal": normal,
                    "axis_span": axis_span,
                    "normal_span": normal_span,
                    "vertical_span": vertical_span,
                    "confidence": float(np.clip(0.40 + len(subset) / 800.0, 0.0, 0.90)),
                    "z": median_z,
                    "pixels": len(subset),
                }
                duplicate = next(
                    (
                        index
                        for index, existing in enumerate(component_candidates)
                        if abs(float(existing["z"]) - median_z) < 0.025
                    ),
                    None,
                )
                if duplicate is None:
                    component_candidates.append(proposal)
                elif len(subset) > int(component_candidates[duplicate]["pixels"]):
                    component_candidates[duplicate] = proposal
            for proposal in component_candidates:
                proposal.pop("z")
                proposal.pop("pixels")
                candidates.append(proposal)
        if len(candidates) < 2:
            return None

        best: tuple[float, tuple[dict[str, object], ...]] | None = None
        for group in combinations(candidates, 3):
            ordered = tuple(sorted(group, key=lambda item: float(np.asarray(item["center"])[2])))
            centers = np.asarray([item["center"] for item in ordered], dtype=np.float64)
            axes = np.asarray([item["axis"] for item in ordered], dtype=np.float64)
            reference_axis = axes[1].copy()
            for index in (0, 2):
                if float(np.dot(axes[index], reference_axis)) < 0.0:
                    axes[index] *= -1.0
            axis_cosine = np.abs(axes @ reference_axis)
            if np.min(axis_cosine) < 0.82:
                continue
            z_gaps = np.diff(centers[:, 2])
            if np.any(z_gaps < 0.040) or np.any(z_gaps > 0.120):
                continue
            normal = np.array(
                (-reference_axis[1], reference_axis[0], 0.0), dtype=np.float64
            )
            lateral = centers @ reference_axis
            if float(np.ptp(lateral)) > 0.085:
                continue
            normal_coordinates = centers @ normal
            if float(np.ptp(normal_coordinates)) > 0.240:
                continue
            spacing_error = abs(float(z_gaps[1] - z_gaps[0]))
            lateral_error = float(np.ptp(lateral))
            axis_error = float(np.sum(1.0 - axis_cosine))
            shape_reward = float(
                sum(float(item["axis_span"]) for item in ordered)
            )
            score = (
                spacing_error / 0.080
                + lateral_error / 0.085
                + axis_error
                - 0.35 * shape_reward
                - 0.10 * sum(float(item["confidence"]) for item in ordered)
            )
            if best is None or score < best[0]:
                best = (score, ordered)
        if best is None and level in {"top", "bottom"}:
            # An open outer drawer can occlude the middle bar. Two aligned
            # bars separated by two drawer pitches still identify the top and
            # bottom levels; they do not locate an unseen middle handle.
            for group in combinations(candidates, 2):
                ordered = tuple(sorted(group, key=lambda item: float(np.asarray(item["center"])[2])))
                centers = np.asarray([item["center"] for item in ordered])
                axes = np.asarray([item["axis"] for item in ordered])
                gap = float(centers[1, 2] - centers[0, 2])
                if not 0.125 <= gap <= 0.165 or abs(float(axes[0] @ axes[1])) < 0.90:
                    continue
                pair_axis = axes[0] + np.sign(float(axes[0] @ axes[1])) * axes[1]
                pair_axis /= np.linalg.norm(pair_axis)
                lateral_error = abs(float((centers[1] - centers[0]) @ pair_axis))
                if lateral_error > 0.045:
                    continue
                normal = np.array((-pair_axis[1], pair_axis[0], 0.0))
                if abs(float((centers[1] - centers[0]) @ normal)) > 0.240:
                    continue
                score = abs(gap - 0.145) + lateral_error
                if best is None or score < best[0]:
                    best = (score, ordered)
        if best is None:
            return (self._detect_top_from_roof(frame, candidates)
                    if level == "top" and allow_roof_fallback else None)

        _, ordered = best
        selected = ordered[{"bottom": 0, "middle": 1, "top": -1}[level]]
        point = np.asarray(selected["center"], dtype=np.float64).copy()
        axis = np.asarray(selected["axis"], dtype=np.float64).copy()
        # The initial camera origin provides a stable sign even when the
        # visible cabinet spine is absent.  Use the triplet median as the
        # fixture reference, never a semantic/task or simulator location.
        centers = np.asarray(
            [np.asarray(item["center"], dtype=np.float64) for item in ordered]
        )
        fixture = np.median(centers, axis=0)
        normal = np.array((-axis[1], axis[0], 0.0), dtype=np.float64)
        camera_origin = frame.calibration.T_world_camera[:3, 3]
        sign_reference = camera_origin - fixture
        # A side-on camera can lie almost in the handle plane. In that view,
        # one false bar shifts the triplet median enough to reverse the sign.
        # A substantial local dark cabinet surface gives the front/back
        # direction directly: the protruding handle lies outside that body.
        body_points = _points(frame, gray < self.config.cabinet_dark_threshold)
        if len(body_points):
            delta = body_points - point
            local_body = body_points[
                (np.abs(delta @ axis) <= 0.16)
                & (np.abs(delta @ normal) <= 0.30)
                & (delta[:, 2] >= -0.21)
                & (delta[:, 2] <= 0.065)
            ]
            # A body patch is a tie-breaker for a side-on camera.  In an
            # oblique view, dark countertop clutter beside an extended drawer
            # can sit in front of the handle and must not reverse the clearly
            # observed camera-facing normal.
            if (
                len(local_body) >= 150
                and abs(float(sign_reference @ normal)) < 0.080
            ):
                body_reference = point - np.median(local_body, axis=0)
                if abs(float(body_reference @ normal)) >= 0.045:
                    sign_reference = body_reference
        if float(np.dot(normal, sign_reference)) < 0.0:
            normal *= -1.0
        outward = _unit_xy(normal)
        # Consumers interpret fixture_center_world as the cabinet body
        # reference, not a moving-handle median. Anchor its normal coordinate
        # behind the most recessed visible handle using the same public
        # closed-handle offset as the close controller.
        closed_handle_normal = float(np.min(centers @ outward))
        fixture += outward * (closed_handle_normal - 0.085 - float(fixture @ outward))
        point[2] = float(np.asarray(selected["center"], dtype=np.float64)[2])
        confidence = float(
            np.clip(
                0.48
                + 0.35 * min(1.0, float(sum(float(item["confidence"]) for item in ordered)) / 2.0),
                0.0,
                0.95,
            )
        )
        return ContactTarget(
            GoalSkillKind.OPEN_DRAWER,
            point,
            outward,
            outward,
            fixture,
            axis,
            confidence,
            (frame.calibration.name,),
        )

    def _detect_top_from_roof(self, frame, candidates) -> ContactTarget | None:
        """Bind an isolated upper handle to a broad roof behind and above it.

        An extended top drawer hides the two lower handles. Its visible bar
        can still be identified by the nearby horizontal cabinet roof; thin
        side edges do not supply this two-dimensional depth support.
        """
        points = _points(frame, np.ones(frame.depth_m.shape, dtype=bool))
        camera = frame.calibration.T_world_camera[:3, 3]
        proposals = []
        for candidate in candidates:
            if candidate["axis_span"] < .060 or candidate["vertical_span"] > .016:
                continue
            point = np.asarray(candidate["center"])
            axis = np.asarray(candidate["axis"])
            outward = np.array((-axis[1], axis[0], 0.0))
            if float(outward @ (camera - point)) < 0.0:
                outward *= -1.0
            delta = points - point
            normal = delta @ outward
            local = points[(np.abs(delta @ axis) < .18)
                           & (normal < -.030) & (normal > -.34)
                           & (delta[:, 2] > .018) & (delta[:, 2] < .060)]
            if len(local) < 150:
                continue
            edges = np.arange(point[2] + .018, point[2] + .063, .003)
            counts, edges = np.histogram(local[:, 2], edges)
            mode = int(np.argmax(counts))
            height = (edges[mode] + edges[mode + 1]) / 2.0
            plane = local[np.abs(local[:, 2] - height) < .002]
            if len(plane) < 150:
                continue
            basis = np.column_stack((axis, outward, (0., 0., 1.)))
            lower, upper = np.quantile(plane @ basis, (.02, .98), axis=0)
            spans = upper - lower
            if min(spans[:2]) < .075 or max(spans[:2]) > .36 or spans[2] > .003:
                continue
            fixture = basis @ ((lower + upper) / 2.0)
            lateral = abs(float((fixture - point) @ axis))
            if lateral > .055:
                continue
            score = len(plane) * float(candidate["confidence"]) / (1.0 + 20 * lateral)
            proposals.append((score, ContactTarget(
                GoalSkillKind.OPEN_DRAWER, point, outward, outward, fixture, axis,
                min(.85, float(candidate["confidence"]) + .1), (frame.calibration.name,),
            )))
        return max(proposals, key=lambda item: item[0])[1] if proposals else None


@dataclass(frozen=True, slots=True)
class _MicrowaveFixtureFrame:
    center: np.ndarray
    outward: np.ndarray
    handle_side: np.ndarray
    vertical: np.ndarray
    outward_half_extent_m: float
    handle_side_half_extent_m: float
    vertical_half_extent_m: float


class MicrowaveDoorHandleDetector:
    """Find a microwave handle or open-door edge from calibrated RGB-D.

    The caller supplies an episode-local fixture OBB reconstructed by its
    scene estimator.  Axis columns and half extents must correspond.  Camera
    position never decides a sign: the shorter horizontal OBB axis is signed
    toward the first measured end-effector pose and forms the closed-door
    outward normal; the longer horizontal axis is likewise signed toward the
    end effector to select the handle side.  Pass that frozen public pose as
    ``reference_ee_position_world`` on later observations to retain both
    signs while the robot moves.

    A closed door has a conservative OBB fallback because its rigid front
    surface is known.  An open door does not: without a visible vertical edge
    or handle the detector raises ``LookupError`` instead of aiming at a far
    appliance corner.
    """

    _PUBLIC_HANDLE_LONG_OFFSET_M = 0.2375
    _PUBLIC_HANDLE_OUTWARD_OFFSET_M = 0.054

    def __init__(self, config: MicrowaveDoorHandleConfig | None = None) -> None:
        self.config = config or MicrowaveDoorHandleConfig()
        self.last_detection_trace: dict[str, object] | None = None
        self.last_articulation_trace: dict[str, object] | None = None

    def reset(self) -> None:
        """Discard all observation-derived state at an episode boundary."""

        self.last_detection_trace = None
        self.last_articulation_trace = None

    def detect(
        self,
        observation: RobotObservation,
        fixture_center_world: np.ndarray,
        fixture_axes_world: np.ndarray,
        fixture_half_extents_m: np.ndarray,
        initial_is_open: bool,
        *,
        reference_ee_position_world: np.ndarray | None = None,
        local_anchor_world: np.ndarray | None = None,
        local_anchor_radius_m: float | None = None,
        frozen_hinge_world: np.ndarray | None = None,
        frozen_rotation_axis_world: np.ndarray | None = None,
        frozen_radius_m: float | None = None,
        frozen_radius_tolerance_m: float | None = None,
        expected_closed_slot_world: np.ndarray | None = None,
    ) -> ContactTarget:
        signing_ee = (
            observation.proprio.ee_position_world
            if reference_ee_position_world is None
            else np.asarray(reference_ee_position_world, dtype=np.float64)
        )
        if signing_ee.shape != (3,) or not np.all(np.isfinite(signing_ee)):
            raise ValueError("reference EE position must be a finite 3-vector")
        local_values = (
            local_anchor_world,
            local_anchor_radius_m,
            frozen_hinge_world,
            frozen_rotation_axis_world,
            frozen_radius_m,
            frozen_radius_tolerance_m,
        )
        local_mode = any(value is not None for value in local_values)
        if local_mode and not all(value is not None for value in local_values):
            raise ValueError(
                "microwave local association requires anchor and frozen hinge-radius geometry"
            )
        local_anchor: np.ndarray | None = None
        local_hinge: np.ndarray | None = None
        local_axis: np.ndarray | None = None
        expected_closed_slot: np.ndarray | None = None
        if expected_closed_slot_world is not None:
            expected_closed_slot = np.asarray(
                expected_closed_slot_world, dtype=np.float64
            )
            if (
                expected_closed_slot.shape != (3,)
                or not np.all(np.isfinite(expected_closed_slot))
            ):
                raise ValueError(
                    "expected microwave closed slot must be a finite 3-vector"
                )
            if initial_is_open:
                raise ValueError(
                    "expected microwave closed slot is only valid for a closed-state query"
                )
        if local_mode:
            local_anchor = np.asarray(local_anchor_world, dtype=np.float64)
            local_hinge = np.asarray(frozen_hinge_world, dtype=np.float64)
            local_axis = np.asarray(
                frozen_rotation_axis_world, dtype=np.float64
            )
            if (
                local_anchor.shape != (3,)
                or local_hinge.shape != (3,)
                or local_axis.shape != (3,)
                or not np.all(np.isfinite(local_anchor))
                or not np.all(np.isfinite(local_hinge))
                or not np.all(np.isfinite(local_axis))
            ):
                raise ValueError(
                    "microwave local anchor and hinge geometry must be finite 3-vectors"
                )
            axis_norm = float(np.linalg.norm(local_axis))
            if axis_norm < 1e-8:
                raise ValueError("microwave frozen rotation axis is degenerate")
            local_axis = local_axis / axis_norm
            if any(
                not np.isfinite(float(value)) or float(value) <= 0.0
                for value in (
                    local_anchor_radius_m,
                    frozen_radius_m,
                    frozen_radius_tolerance_m,
                )
            ):
                raise ValueError(
                    "microwave local association radii must be finite and positive"
                )
        self.last_detection_trace = {
            "local_mode": local_mode,
            "initial_is_open": bool(initial_is_open),
            "cameras": {},
        }
        if local_mode:
            assert local_anchor is not None
            assert local_hinge is not None
            assert local_axis is not None
            self.last_detection_trace.update(
                {
                    "local_anchor_world_m": local_anchor.tolist(),
                    "local_anchor_radius_m": float(local_anchor_radius_m),
                    "frozen_hinge_world_m": local_hinge.tolist(),
                    "frozen_rotation_axis_world": local_axis.tolist(),
                    "frozen_radius_m": float(frozen_radius_m),
                    "frozen_radius_tolerance_m": float(
                        frozen_radius_tolerance_m
                    ),
                }
            )
        if expected_closed_slot is not None:
            self.last_detection_trace[
                "expected_closed_slot_world_m"
            ] = expected_closed_slot.tolist()
        frame = self._fixture_frame(
            signing_ee,
            fixture_center_world,
            fixture_axes_world,
            fixture_half_extents_m,
        )
        proposals: list[tuple[np.ndarray, float, str]] = []
        for camera_name in ("agentview", "wrist"):
            proposal = self._detect_frame(
                observation.cameras[camera_name],
                observation.proprio.ee_position_world,
                frame,
                initial_is_open=initial_is_open,
                local_anchor_world=local_anchor,
                local_anchor_radius_m=local_anchor_radius_m,
                frozen_hinge_world=local_hinge,
                frozen_rotation_axis_world=local_axis,
                frozen_radius_m=frozen_radius_m,
                frozen_radius_tolerance_m=frozen_radius_tolerance_m,
                expected_closed_slot_world=expected_closed_slot,
            )
            if proposal is not None:
                point, confidence = proposal
                proposals.append((point, confidence, camera_name))

        kind = (
            GoalSkillKind.CLOSE_MICROWAVE
            if initial_is_open
            else GoalSkillKind.OPEN_MICROWAVE
        )
        closed_slot = (
            expected_closed_slot.copy()
            if expected_closed_slot is not None
            else self._closed_fallback_point(frame)
        )
        if not initial_is_open:
            proposals = [
                item
                for item in proposals
                if np.linalg.norm(item[0] - closed_slot)
                <= self.config.closed_candidate_slot_radius_m
                and abs(float(np.dot(item[0] - frame.center, frame.vertical)))
                <= self.config.closed_candidate_vertical_offset_m
            ]
        if not proposals:
            if initial_is_open:
                raise LookupError(
                    "no reliable open microwave door edge was visible in either RGB-D view"
                )
            self.last_detection_trace["closed_fallback_used"] = True
            return ContactTarget(
                kind,
                closed_slot,
                frame.outward,
                frame.outward,
                frame.center,
                np.array((0.0, 0.0, 1.0), dtype=np.float64),
                0.34,
                tuple(observation.cameras),
            )
        self.last_detection_trace["closed_fallback_used"] = False

        if local_anchor is not None:
            anchor = min(
                proposals,
                key=lambda item: (
                    float(np.linalg.norm(item[0] - local_anchor)),
                    -item[1],
                ),
            )
            selection_mode = "local_anchor"
        elif initial_is_open:
            # A window-frame bar can be taller and denser than the movable
            # leading edge, so raw component confidence alone is the wrong
            # first key.  The handle / free edge is the handle-side extremum
            # among already structure-gated RGB-D vertical features.  Keep a
            # narrow metric frontier to absorb depth noise, then select the
            # most reliable member of that frontier.
            side_coordinates = [
                float(np.dot(item[0] - frame.center, frame.handle_side))
                for item in proposals
            ]
            maximum_side = max(side_coordinates)
            frontier = [
                item
                for item, side_coordinate in zip(
                    proposals, side_coordinates, strict=True
                )
                if side_coordinate
                >= maximum_side - self.config.open_free_edge_side_tolerance_m
            ]
            anchor = max(frontier, key=lambda item: item[1])
            selection_mode = "open_handle_side_free_edge"
        else:
            anchor = max(proposals, key=lambda item: item[1])
            selection_mode = "closed_feature_confidence"
        agreeing = [
            item
            for item in proposals
            if np.linalg.norm(item[0] - anchor[0])
            <= self.config.cross_view_agreement_m
        ]
        if local_anchor is not None:
            # The local continuation must preserve the closest vertical edge,
            # not average it back toward a second appliance/body fragment.
            agreeing = [anchor]
        elif initial_is_open:
            anchor_side = float(
                np.dot(anchor[0] - frame.center, frame.handle_side)
            )
            agreeing = [
                item
                for item in agreeing
                if abs(
                    float(np.dot(item[0] - frame.center, frame.handle_side))
                    - anchor_side
                )
                <= self.config.open_free_edge_side_tolerance_m
            ]
        weights = np.asarray([item[1] for item in agreeing], dtype=np.float64)
        weights /= max(float(weights.sum()), 1e-12)
        point = sum(weight * item[0] for weight, item in zip(weights, agreeing))
        if initial_is_open:
            outward = _unit_xy(
                observation.proprio.ee_position_world - point,
                fallback=(float(frame.outward[0]), float(frame.outward[1])),
            )
        else:
            outward = frame.outward
            # A vertical closed-door handle is intentionally contacted at its
            # OBB mid-height.  Fragmented top/bottom edge pixels still refine
            # its visible planar surface, but cannot drag the contact toward a
            # mounting bracket or the worktop below the door.
            point -= frame.vertical * float(
                np.dot(point - frame.center, frame.vertical)
            )
        confidence = float(
            np.clip(max(item[1] for item in agreeing) + 0.04 * (len(agreeing) - 1), 0, 0.97)
        )
        self.last_detection_trace.update(
            {
                "selection_mode": selection_mode,
                "selected_point_world_m": point.tolist(),
                "selected_handle_side_coordinate_m": float(
                    np.dot(point - frame.center, frame.handle_side)
                ),
                "cross_view_proposals": [
                    {
                        "point_world_m": item[0].tolist(),
                        "confidence": float(item[1]),
                        "camera": item[2],
                        "handle_side_coordinate_m": float(
                            np.dot(item[0] - frame.center, frame.handle_side)
                        ),
                    }
                    for item in proposals
                ],
            }
        )
        return ContactTarget(
            kind,
            point,
            outward,
            outward,
            frame.center,
            np.array((0.0, 0.0, 1.0), dtype=np.float64),
            confidence,
            tuple(item[2] for item in agreeing),
        )

    def _closed_fallback_point(
        self,
        fixture: _MicrowaveFixtureFrame,
    ) -> np.ndarray:
        return (
            fixture.center
            + fixture.outward
            * (
                fixture.outward_half_extent_m
                + self.config.closed_fallback_front_offset_m
            )
            + fixture.handle_side
            * (
                self.config.closed_fallback_handle_side_fraction
                * fixture.handle_side_half_extent_m
            )
        )

    def articulation_geometry_hypotheses(
        self,
        reference_ee_position_world: np.ndarray,
        fixture_center_world: np.ndarray,
        fixture_axes_world: np.ndarray,
        fixture_half_extents_m: np.ndarray,
        observed_handle_world: np.ndarray,
    ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
        """Return both sensor-derived open-door hinge hypotheses, ranked.

        A frozen appliance OBB has two long-side/front corners.  An open door
        can bias that OBB toward its own point cloud, so radius proximity by
        itself is not sufficient to discard either jamb.  Rank both using the
        signed rotation from the observed moving edge to the corresponding
        closed handle slot.  The method consumes calibrated RGB-D geometry
        and the public reset EE pose only.
        """

        fixture = self._fixture_frame(
            np.asarray(reference_ee_position_world, dtype=np.float64),
            np.asarray(fixture_center_world, dtype=np.float64),
            np.asarray(fixture_axes_world, dtype=np.float64),
            np.asarray(fixture_half_extents_m, dtype=np.float64),
        )
        observed = np.asarray(observed_handle_world, dtype=np.float64)
        if observed.shape != (3,) or not np.all(np.isfinite(observed)):
            raise ValueError("observed microwave handle must be a finite 3-vector")
        rows = self._rank_open_articulation_hypotheses(fixture, observed)
        return tuple(
            (
                np.array(row["hinge_world_m"], dtype=np.float64),
                fixture.vertical.copy(),
                np.array(row["closed_slot_world_m"], dtype=np.float64),
            )
            for row in rows
        )

    def _rank_open_articulation_hypotheses(
        self,
        fixture: _MicrowaveFixtureFrame,
        observed: np.ndarray,
    ) -> list[dict[str, object]]:
        """Rank both OBB jambs by a sensor-only open-to-close rotation."""

        expected_radius = (
            1.0 + self.config.closed_fallback_handle_side_fraction
        ) * fixture.handle_side_half_extent_m
        close_angle_min, close_angle_max = (
            self.config.open_hinge_close_angle_range_rad
        )
        rows: list[dict[str, object]] = []
        for hinge_sign in (-1.0, 1.0):
            hinge = (
                fixture.center
                + fixture.outward * fixture.outward_half_extent_m
                + hinge_sign
                * fixture.handle_side
                * fixture.handle_side_half_extent_m
            )
            closed_slot = (
                fixture.center
                + fixture.outward
                * (
                    fixture.outward_half_extent_m
                    + self.config.closed_fallback_front_offset_m
                )
                - hinge_sign
                * fixture.handle_side
                * (
                    self.config.closed_fallback_handle_side_fraction
                    * fixture.handle_side_half_extent_m
                )
            )
            start_radius = observed - hinge
            start_radius -= fixture.vertical * float(
                np.dot(start_radius, fixture.vertical)
            )
            closed_radius = closed_slot - hinge
            closed_radius -= fixture.vertical * float(
                np.dot(closed_radius, fixture.vertical)
            )
            observed_radius_m = float(np.linalg.norm(start_radius))
            closed_radius_m = float(np.linalg.norm(closed_radius))
            signed_close_angle_rad = float(
                np.arctan2(
                    np.dot(
                        fixture.vertical,
                        np.cross(start_radius, closed_radius),
                    ),
                    np.dot(start_radius, closed_radius),
                )
            )
            angle_feasible = bool(
                close_angle_min
                <= abs(signed_close_angle_rad)
                <= close_angle_max
            )
            rows.append(
                {
                    "hinge_sign": hinge_sign,
                    "hinge_world_m": hinge.tolist(),
                    "closed_slot_world_m": closed_slot.tolist(),
                    "signed_close_angle_rad": signed_close_angle_rad,
                    "observed_radius_m": observed_radius_m,
                    "closed_radius_m": closed_radius_m,
                    "nominal_radius_residual_m": abs(
                        observed_radius_m - expected_radius
                    ),
                    "signed_close_angle_feasible": angle_feasible,
                }
            )
        rows.sort(
            key=lambda row: (
                not bool(row["signed_close_angle_feasible"]),
                float(row["nominal_radius_residual_m"]),
                -abs(float(row["signed_close_angle_rad"])),
            )
        )
        for rank, row in enumerate(rows):
            row["rank"] = rank
            row["selected"] = rank == 0
        self.last_articulation_trace = {
            "selection_mode": "open_signed_close_feasibility",
            "close_angle_range_rad": [close_angle_min, close_angle_max],
            "observed_edge_world_m": observed.tolist(),
            "hypotheses": [dict(row) for row in rows],
            "selected_hypothesis_feasible": bool(
                rows[0]["signed_close_angle_feasible"]
            ),
        }
        return rows

    def articulation_geometry(
        self,
        reference_ee_position_world: np.ndarray,
        fixture_center_world: np.ndarray,
        fixture_axes_world: np.ndarray,
        fixture_half_extents_m: np.ndarray,
        observed_handle_world: np.ndarray | None = None,
        *,
        initial_is_open: bool | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return RGB-D OBB hinge, vertical axis, and closed handle slot.

        A microwave door is a vertical revolute panel.  The handle lies on
        the sensor-signed long side of the appliance, so its hinge is the
        opposite long-side/front OBB edge.  This is fixture-family geometry
        reconstructed from calibrated RGB-D, not a simulator joint pose.
        """

        fixture = self._fixture_frame(
            np.asarray(reference_ee_position_world, dtype=np.float64),
            np.asarray(fixture_center_world, dtype=np.float64),
            np.asarray(fixture_axes_world, dtype=np.float64),
            np.asarray(fixture_half_extents_m, dtype=np.float64),
        )
        if initial_is_open not in (None, True, False):
            raise ValueError("initial microwave door state must be boolean")
        self.last_articulation_trace = None
        hinge_sign = -1.0
        if observed_handle_world is not None:
            observed = np.asarray(observed_handle_world, dtype=np.float64)
            if observed.shape != (3,) or not np.all(np.isfinite(observed)):
                raise ValueError("observed microwave handle must be a finite 3-vector")
            if initial_is_open is False:
                # In the closed state the calibrated RGB-D point is the
                # handle itself.  The public microwave family fixes the
                # handle-to-hinge offsets, so anchor both the hinge depth and
                # long-axis side directly from that measurement instead of
                # inheriting the much thicker appliance-body OBB front face.
                handle_side = fixture.handle_side.copy()
                if float(np.dot(observed - fixture.center, handle_side)) < 0.0:
                    handle_side *= -1.0
                hinge = (
                    observed
                    - handle_side * self._PUBLIC_HANDLE_LONG_OFFSET_M
                    - fixture.outward * self._PUBLIC_HANDLE_OUTWARD_OFFSET_M
                )
                self.last_articulation_trace = {
                    "selection_mode": "closed_handle_public_offset",
                    "observed_handle_world_m": observed.tolist(),
                    "selected_hinge_world_m": hinge.tolist(),
                    "selected_closed_slot_world_m": observed.tolist(),
                }
                return (
                    hinge.copy(),
                    fixture.vertical.copy(),
                    observed.copy(),
                )
            if initial_is_open is True:
                ranked = self._rank_open_articulation_hypotheses(
                    fixture,
                    observed,
                )
                selected = ranked[0]
                return (
                    np.asarray(
                        selected["hinge_world_m"], dtype=np.float64
                    ).copy(),
                    fixture.vertical.copy(),
                    np.asarray(
                        selected["closed_slot_world_m"], dtype=np.float64
                    ).copy(),
                )
            expected_radius = (
                1.0 + self.config.closed_fallback_handle_side_fraction
            ) * fixture.handle_side_half_extent_m
            candidates: list[tuple[float, float, np.ndarray]] = []
            for sign in (-1.0, 1.0):
                candidate = (
                    fixture.center
                    + fixture.outward * fixture.outward_half_extent_m
                    + sign * fixture.handle_side * fixture.handle_side_half_extent_m
                )
                radial = observed - candidate
                radial -= fixture.vertical * float(np.dot(radial, fixture.vertical))
                radius = float(np.linalg.norm(radial))
                candidates.append(
                    (abs(radius - expected_radius), sign, candidate)
                )
            _, hinge_sign, hinge = min(candidates, key=lambda item: item[0])
        else:
            hinge = (
                fixture.center
                + fixture.outward * fixture.outward_half_extent_m
                - fixture.handle_side * fixture.handle_side_half_extent_m
            )
        closed_slot = (
            fixture.center
            + fixture.outward
            * (
                fixture.outward_half_extent_m
                + self.config.closed_fallback_front_offset_m
            )
            - hinge_sign
            * fixture.handle_side
            * (
                self.config.closed_fallback_handle_side_fraction
                * fixture.handle_side_half_extent_m
            )
        )
        self.last_articulation_trace = {
            "selection_mode": "legacy_obb_radius_or_default",
            "observed_handle_world_m": (
                None if observed_handle_world is None else observed.tolist()
            ),
            "selected_hinge_world_m": hinge.tolist(),
            "selected_closed_slot_world_m": closed_slot.tolist(),
        }
        return (
            hinge.copy(),
            fixture.vertical.copy(),
            closed_slot.copy(),
        )

    @staticmethod
    def _fixture_frame(
        reference_ee_position_world: np.ndarray,
        center_value: np.ndarray,
        axes_value: np.ndarray,
        half_extents_value: np.ndarray,
    ) -> _MicrowaveFixtureFrame:
        center = np.asarray(center_value, dtype=np.float64)
        axes = np.asarray(axes_value, dtype=np.float64)
        half_extents = np.asarray(half_extents_value, dtype=np.float64)
        if center.shape != (3,) or axes.shape != (3, 3) or half_extents.shape != (3,):
            raise ValueError("microwave OBB requires center (3), axes (3,3), and half extents (3)")
        if not (
            np.all(np.isfinite(center))
            and np.all(np.isfinite(axes))
            and np.all(np.isfinite(half_extents))
        ):
            raise ValueError("microwave OBB must be finite")
        if np.any(half_extents <= 0):
            raise ValueError("microwave OBB half extents must be positive")
        norms = np.linalg.norm(axes, axis=0)
        if np.any(norms < 1e-8):
            raise ValueError("microwave OBB axes cannot be degenerate")
        axes = axes / norms[None, :]
        if not np.allclose(axes.T @ axes, np.eye(3), atol=0.08):
            raise ValueError("microwave OBB axes must be orthogonal columns")

        vertical_index = int(np.argmax(np.abs(axes[2, :])))
        if abs(float(axes[2, vertical_index])) < 0.70:
            raise ValueError("microwave OBB has no reliable vertical axis")
        horizontal_indices = sorted(
            (index for index in range(3) if index != vertical_index),
            key=lambda index: half_extents[index],
        )
        short_index, long_index = horizontal_indices[0], horizontal_indices[-1]

        outward = _unit_xy(axes[:, short_index])
        handle_side = _unit_xy(axes[:, long_index], fallback=(1.0, 0.0))
        ee_delta = reference_ee_position_world - center
        if float(np.dot(outward, ee_delta)) < 0:
            outward *= -1
        if float(np.dot(handle_side, ee_delta)) < 0:
            handle_side *= -1
        # Re-orthogonalise after projecting slightly tilted OBB axes onto XY.
        handle_side -= outward * float(np.dot(handle_side, outward))
        handle_side = _unit_xy(handle_side, fallback=(-outward[1], outward[0]))
        if float(np.dot(handle_side, ee_delta)) < 0:
            handle_side *= -1
        vertical = axes[:, vertical_index].copy()
        if vertical[2] < 0:
            vertical *= -1
        return _MicrowaveFixtureFrame(
            np.array(center, copy=True),
            outward,
            handle_side,
            vertical,
            float(half_extents[short_index]),
            float(half_extents[long_index]),
            float(half_extents[vertical_index]),
        )

    def _detect_frame(
        self,
        camera: CameraFrame,
        ee_position_world: np.ndarray,
        fixture: _MicrowaveFixtureFrame,
        *,
        initial_is_open: bool,
        local_anchor_world: np.ndarray | None = None,
        local_anchor_radius_m: float | None = None,
        frozen_hinge_world: np.ndarray | None = None,
        frozen_rotation_axis_world: np.ndarray | None = None,
        frozen_radius_m: float | None = None,
        frozen_radius_tolerance_m: float | None = None,
        expected_closed_slot_world: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float] | None:
        frame_trace: dict[str, object] = {
            "raw_components": 0,
            "unique_components": 0,
            "bounded_components": 0,
            "depth_components": 0,
            "structural_components": 0,
            "localized_components": 0,
            "vertical_components": 0,
            "nonrobot_components": 0,
            "anchor_components": 0,
            "radius_components": 0,
            "structural_candidates": [],
            "vertical_core_candidates": [],
            "prelocal_candidates": [],
        }
        rgb = camera.rgb
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        chroma = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
        low_chroma = (
            (gray >= self.config.neutral_low)
            & (gray <= self.config.neutral_high)
            & (chroma <= self.config.neutral_chroma)
        )
        blurred = cv2.GaussianBlur(gray, (0, 0), 3.0)
        contrast = cv2.absdiff(gray, blurred) >= self.config.local_contrast_threshold
        edge = cv2.Canny(gray, 28, 84) > 0
        masks = (
            low_chroma & (gray <= self.config.dark_high),
            low_chroma & contrast,
            low_chroma & edge,
            low_chroma,
        )
        candidates: list[tuple[float, np.ndarray]] = []
        seen: set[tuple[int, int, int, int]] = set()
        image_area = camera.calibration.width * camera.calibration.height
        for appearance in masks:
            for _, stats, component in _components(appearance):
                frame_trace["raw_components"] = int(
                    frame_trace["raw_components"]
                ) + 1
                x, y, width, height, area = map(int, stats)
                signature = (x, y, width, height)
                if signature in seen:
                    continue
                seen.add(signature)
                frame_trace["unique_components"] = int(
                    frame_trace["unique_components"]
                ) + 1
                if area < self.config.min_component_pixels:
                    continue
                if area > self.config.max_component_area_fraction * image_area:
                    continue
                if width > self.config.max_component_width_fraction * camera.calibration.width:
                    continue
                if height > self.config.max_component_height_fraction * camera.calibration.height:
                    continue
                border = self.config.component_border_margin_px
                if (
                    x < border
                    or y < border
                    or x + width > camera.calibration.width - border
                    or y + height > camera.calibration.height - border
                ):
                    # A component cut by the frame boundary has no observable
                    # width/extent closure.  In wrist views this is commonly
                    # the appliance body or a robot link entering the image,
                    # not an isolated door handle.
                    continue
                frame_trace["bounded_components"] = int(
                    frame_trace["bounded_components"]
                ) + 1
                points = _points(camera, component)
                if len(points) < self.config.min_component_pixels:
                    continue
                frame_trace["depth_components"] = int(
                    frame_trace["depth_components"]
                ) + 1
                delta = points - fixture.center[None, :]
                outward_coordinate = delta @ fixture.outward
                side_coordinate = delta @ fixture.handle_side
                vertical_coordinate = delta @ fixture.vertical
                vertical_ok = (
                    np.abs(vertical_coordinate)
                    <= fixture.vertical_half_extent_m + self.config.vertical_margin_m
                )
                if initial_is_open:
                    # During a close continuation the retained edge can
                    # legitimately return to the appliance-front slab.  The
                    # ordinary open-door search keeps its 30-mm free-front
                    # requirement; local mode substitutes the closed-front
                    # inner bound because its independent endpoint-neighbour
                    # and frozen-radius gates exclude a remote body fallback.
                    open_front_min = (
                        fixture.outward_half_extent_m
                        - self.config.closed_front_inner_margin_m
                        if local_anchor_world is not None
                        else fixture.outward_half_extent_m
                        + self.config.open_front_clearance_m
                    )
                    if local_anchor_world is not None:
                        assert local_anchor_radius_m is not None
                        # A closing free edge crosses the appliance body's
                        # signed long-axis midpoint before reaching the jamb.
                        # The ordinary open-door query can use the static
                        # handle-side prior, but a continuation must instead
                        # stay local to the co-rotated retained-contact
                        # anchor.  Keeping the old positive-side bound here
                        # discards the moving edge near closure and leaves a
                        # fixed front-frame edge as the only admissible
                        # candidate.  The independent anchor and frozen-circle
                        # gates below remain mandatory.
                        anchor_side_coordinate = float(
                            np.dot(
                                local_anchor_world - fixture.center,
                                fixture.handle_side,
                            )
                        )
                        open_side_min = (
                            anchor_side_coordinate - local_anchor_radius_m
                        )
                        open_side_max = (
                            anchor_side_coordinate + local_anchor_radius_m
                        )
                    else:
                        open_side_min = (
                            self.config.open_handle_side_min_fraction
                            * fixture.handle_side_half_extent_m
                        )
                        open_side_max = (
                            fixture.handle_side_half_extent_m
                            + self.config.open_handle_side_reach_m
                        )
                    structural = (
                        vertical_ok
                        & (
                            outward_coordinate >= open_front_min
                        )
                        & (
                            outward_coordinate
                            <= fixture.outward_half_extent_m
                            + self.config.open_outward_reach_m
                        )
                        & (
                            side_coordinate >= open_side_min
                        )
                        & (
                            side_coordinate <= open_side_max
                        )
                    )
                else:
                    if expected_closed_slot_world is not None:
                        expected_delta = (
                            expected_closed_slot_world - fixture.center
                        )
                        expected_side = float(
                            np.dot(expected_delta, fixture.handle_side)
                        )
                        expected_outward = float(
                            np.dot(expected_delta, fixture.outward)
                        )
                        slot_radius = self.config.closed_candidate_slot_radius_m
                        structural = (
                            vertical_ok
                            & (
                                np.abs(side_coordinate - expected_side)
                                <= slot_radius
                            )
                            & (
                                np.abs(outward_coordinate - expected_outward)
                                <= slot_radius
                            )
                        )
                    else:
                        structural = (
                            vertical_ok
                            & (
                                outward_coordinate
                                >= fixture.outward_half_extent_m
                                - self.config.closed_front_inner_margin_m
                            )
                            & (
                                outward_coordinate
                                <= fixture.outward_half_extent_m
                                + self.config.closed_front_outer_margin_m
                            )
                            & (
                                side_coordinate
                                >= self.config.closed_handle_side_min_fraction
                                * fixture.handle_side_half_extent_m
                            )
                            & (
                                side_coordinate
                                <= self.config.closed_handle_side_max_fraction
                                * fixture.handle_side_half_extent_m
                            )
                        )
                selected = points[structural]
                if len(selected) < self.config.min_component_pixels:
                    continue
                if len(selected) < 0.22 * len(points):
                    continue
                frame_trace["structural_components"] = int(
                    frame_trace["structural_components"]
                ) + 1
                if local_anchor_world is not None:
                    assert local_anchor_radius_m is not None
                    assert frozen_hinge_world is not None
                    assert frozen_rotation_axis_world is not None
                    assert frozen_radius_m is not None
                    raw_delta = selected - fixture.center[None, :]
                    raw_coordinates = np.column_stack(
                        (
                            raw_delta @ fixture.handle_side,
                            raw_delta @ fixture.outward,
                            raw_delta @ fixture.vertical,
                        )
                    )
                    raw_lower, raw_upper = np.quantile(
                        raw_coordinates,
                        (
                            self.config.center_tail_quantile,
                            1.0 - self.config.center_tail_quantile,
                        ),
                        axis=0,
                    )
                    raw_point = np.median(selected, axis=0)
                    raw_point += fixture.vertical * (
                        0.5 * (raw_lower[2] + raw_upper[2])
                        - float(
                            np.dot(
                                raw_point - fixture.center,
                                fixture.vertical,
                            )
                        )
                    )
                    raw_radial = raw_point - frozen_hinge_world
                    raw_radial -= frozen_rotation_axis_world * float(
                        np.dot(raw_radial, frozen_rotation_axis_world)
                    )
                    raw_local_delta = selected - local_anchor_world[None, :]
                    raw_local_planar = raw_local_delta - np.outer(
                        raw_local_delta @ frozen_rotation_axis_world,
                        frozen_rotation_axis_world,
                    )
                    structural_candidates = frame_trace[
                        "structural_candidates"
                    ]
                    assert isinstance(structural_candidates, list)
                    structural_candidates.append(
                        {
                            "point_world_m": raw_point.tolist(),
                            "anchor_error_m": float(
                                np.linalg.norm(
                                    raw_point - local_anchor_world
                                )
                            ),
                            "minimum_anchor_planar_error_m": float(
                                np.min(
                                    np.linalg.norm(
                                        raw_local_planar,
                                        axis=1,
                                    )
                                )
                            ),
                            "frozen_radius_residual_m": abs(
                                float(np.linalg.norm(raw_radial))
                                - frozen_radius_m
                            ),
                            "lateral_span_m": float(
                                raw_upper[0] - raw_lower[0]
                            ),
                            "outward_span_m": float(
                                raw_upper[1] - raw_lower[1]
                            ),
                            "vertical_span_m": float(
                                raw_upper[2] - raw_lower[2]
                            ),
                            "point_count": int(len(selected)),
                        }
                    )
                    # Connected door contours near the jamb can include most
                    # of the dark appliance frame and fail a global planar
                    # thinness test.  Crop only in the hinge plane around the
                    # predicted retained contact while preserving the full
                    # vertical extent needed to prove a real door edge.
                    local_delta = selected - local_anchor_world[None, :]
                    local_planar = local_delta - np.outer(
                        local_delta @ frozen_rotation_axis_world,
                        frozen_rotation_axis_world,
                    )
                    selected = selected[
                        np.linalg.norm(local_planar, axis=1)
                        <= local_anchor_radius_m
                    ]
                    if len(selected) < self.config.min_component_pixels:
                        continue
                    frame_trace["localized_components"] = int(
                        frame_trace["localized_components"]
                    ) + 1
                selected_delta = selected - fixture.center[None, :]
                coordinates = np.column_stack(
                    (
                        selected_delta @ fixture.handle_side,
                        selected_delta @ fixture.outward,
                        selected_delta @ fixture.vertical,
                    )
                )
                tail = self.config.center_tail_quantile
                lower, upper = np.quantile(
                    coordinates,
                    (tail, 1.0 - tail),
                    axis=0,
                )
                lateral_span, outward_span, vertical_span = upper - lower
                vertical_core_point: np.ndarray | None = None
                max_lateral = (
                    self.config.open_max_lateral_span_m
                    if initial_is_open
                    else self.config.closed_max_lateral_span_m
                )
                max_outward = (
                    self.config.open_max_outward_span_m
                    if initial_is_open
                    else self.config.closed_max_outward_span_m
                )
                direct_vertical_shape = bool(
                    self.config.min_vertical_span_m
                    <= vertical_span
                    <= self.config.max_vertical_span_m
                    and lateral_span <= max_lateral
                    and outward_span <= max_outward
                    and vertical_span
                    >= self.config.min_vertical_aspect
                    * max(lateral_span, outward_span, 0.006)
                )
                # A locally cropped door outline can satisfy the permissive
                # direct aspect gate while still spanning a broad panel/frame
                # cross-section.  Its midpoint is not a graspable free edge.
                # Require the existing narrow vertical-core proof whenever
                # the local planar span exceeds the core's own diameter
                # bound; an isolated thin capsule keeps the direct path.
                local_broad_outline = bool(
                    local_anchor_world is not None
                    and max(lateral_span, outward_span)
                    > 2.15 * self.config.open_vertical_core_radius_m
                )
                if not direct_vertical_shape or local_broad_outline:
                    if not initial_is_open:
                        continue
                    core_diagnostics = frame_trace[
                        "vertical_core_candidates"
                    ]
                    assert isinstance(core_diagnostics, list)
                    vertical_core = self._open_vertical_core(
                        selected,
                        fixture,
                        local_anchor_world=local_anchor_world,
                        local_anchor_radius_m=local_anchor_radius_m,
                        frozen_hinge_world=frozen_hinge_world,
                        frozen_rotation_axis_world=(
                            frozen_rotation_axis_world
                        ),
                        frozen_radius_m=frozen_radius_m,
                        frozen_radius_tolerance_m=(
                            frozen_radius_tolerance_m
                        ),
                        diagnostics=core_diagnostics,
                    )
                    if vertical_core is None:
                        continue
                    selected, vertical_core_point = vertical_core
                    selected_delta = selected - fixture.center[None, :]
                    coordinates = np.column_stack(
                        (
                            selected_delta @ fixture.handle_side,
                            selected_delta @ fixture.outward,
                            selected_delta @ fixture.vertical,
                        )
                    )
                    lower, upper = np.quantile(
                        coordinates,
                        (tail, 1.0 - tail),
                        axis=0,
                    )
                    lateral_span, outward_span, vertical_span = upper - lower
                frame_trace["vertical_components"] = int(
                    frame_trace["vertical_components"]
                ) + 1
                if vertical_core_point is not None:
                    point = vertical_core_point
                elif initial_is_open:
                    # For an open door, the planar sample median is the
                    # measured visible surface.  Moving to the centre of its
                    # full planar extent can put the contact inside the door
                    # slab.  Only de-bias height, whose two visible endpoints
                    # directly support a geometric midpoint.
                    point = np.median(selected, axis=0)
                    visible_vertical_midpoint = 0.5 * (lower[2] + upper[2])
                    point += fixture.vertical * (
                        visible_vertical_midpoint
                        - float(np.dot(point - fixture.center, fixture.vertical))
                    )
                else:
                    # Pixel density is viewpoint-dependent: one side of a
                    # cylindrical handle or thin door edge can contribute far
                    # more samples than the other.  Centre the robust visible
                    # cross-section in the signed fixture frame rather than
                    # following the sample median toward that denser side.
                    midpoint = 0.5 * (lower + upper)
                    point = (
                        fixture.center
                        + fixture.handle_side * midpoint[0]
                        + fixture.outward * midpoint[1]
                        + fixture.vertical * midpoint[2]
                    )
                    if (
                        np.linalg.norm(point - np.median(selected, axis=0))
                        > self.config.max_center_correction_m
                    ):
                        # A much larger correction indicates a severely
                        # clipped or multimodal component, not a defensible
                        # estimate of an occluded handle centre.
                        continue
                if initial_is_open and np.linalg.norm(point - ee_position_world) < self.config.robot_exclusion_radius_m:
                    continue
                frame_trace["nonrobot_components"] = int(
                    frame_trace["nonrobot_components"]
                ) + 1
                if local_anchor_world is not None:
                    assert local_anchor_radius_m is not None
                    assert frozen_hinge_world is not None
                    assert frozen_rotation_axis_world is not None
                    assert frozen_radius_m is not None
                    assert frozen_radius_tolerance_m is not None
                    if (
                        float(np.linalg.norm(point - local_anchor_world))
                        > local_anchor_radius_m
                    ):
                        cast_candidates = frame_trace["prelocal_candidates"]
                        assert isinstance(cast_candidates, list)
                        cast_candidates.append(
                            {
                                "point_world_m": point.tolist(),
                                "anchor_error_m": float(
                                    np.linalg.norm(point - local_anchor_world)
                                ),
                                "radius_error_m": None,
                            }
                        )
                        continue
                    frame_trace["anchor_components"] = int(
                        frame_trace["anchor_components"]
                    ) + 1
                    radial = point - frozen_hinge_world
                    radial -= frozen_rotation_axis_world * float(
                        np.dot(radial, frozen_rotation_axis_world)
                    )
                    if (
                        abs(float(np.linalg.norm(radial)) - frozen_radius_m)
                        > frozen_radius_tolerance_m
                    ):
                        cast_candidates = frame_trace["prelocal_candidates"]
                        assert isinstance(cast_candidates, list)
                        cast_candidates.append(
                            {
                                "point_world_m": point.tolist(),
                                "anchor_error_m": float(
                                    np.linalg.norm(point - local_anchor_world)
                                ),
                                "radius_error_m": abs(
                                    float(np.linalg.norm(radial))
                                    - frozen_radius_m
                                ),
                            }
                        )
                        continue
                    frame_trace["radius_components"] = int(
                        frame_trace["radius_components"]
                    ) + 1
                thinness = vertical_span / max(
                    vertical_span + lateral_span + outward_span, 1e-9
                )
                vertical_coverage = min(
                    vertical_span / max(2.0 * fixture.vertical_half_extent_m, 1e-9),
                    1.0,
                )
                side_progress = float(
                    np.median(selected_delta @ fixture.handle_side)
                    / max(fixture.handle_side_half_extent_m, 1e-9)
                )
                score = (
                    0.35 * thinness
                    + 0.30 * vertical_coverage
                    + 0.20 * min(max(side_progress, 0.0), 1.0)
                    + 0.15 * min(len(selected) / 240.0, 1.0)
                )
                confidence = float(np.clip(0.42 + 0.48 * score, 0, 0.94))
                candidates.append((confidence, point))
        if not candidates and local_anchor_world is not None:
            assert local_anchor_radius_m is not None
            assert frozen_hinge_world is not None
            assert frozen_rotation_axis_world is not None
            assert frozen_radius_m is not None
            assert frozen_radius_tolerance_m is not None
            aggregate = self._local_rgbd_vertical_edge(
                camera,
                fixture,
                local_anchor_world=local_anchor_world,
                local_anchor_radius_m=local_anchor_radius_m,
                frozen_hinge_world=frozen_hinge_world,
                frozen_rotation_axis_world=frozen_rotation_axis_world,
                frozen_radius_m=frozen_radius_m,
                frozen_radius_tolerance_m=frozen_radius_tolerance_m,
                diagnostics=frame_trace,
            )
            if aggregate is not None:
                candidates.append(aggregate)
        if not candidates:
            self.last_detection_trace["cameras"][
                camera.calibration.name
            ] = frame_trace
            return None
        if local_anchor_world is not None:
            confidence, point = min(
                candidates,
                key=lambda item: (
                    float(np.linalg.norm(item[1] - local_anchor_world)),
                    -item[0],
                ),
            )
        elif initial_is_open:
            side_coordinates = [
                float(np.dot(item[1] - fixture.center, fixture.handle_side))
                for item in candidates
            ]
            maximum_side = max(side_coordinates)
            frontier = [
                item
                for item, side_coordinate in zip(
                    candidates, side_coordinates, strict=True
                )
                if side_coordinate
                >= maximum_side - self.config.open_free_edge_side_tolerance_m
            ]
            confidence, point = max(frontier, key=lambda item: item[0])
            frame_trace["selection_mode"] = "open_handle_side_free_edge"
            frame_trace["selected_handle_side_coordinate_m"] = float(
                np.dot(point - fixture.center, fixture.handle_side)
            )
        else:
            slot = (
                expected_closed_slot_world
                if expected_closed_slot_world is not None
                else self._closed_fallback_point(fixture)
            )
            reliable = [
                item
                for item in candidates
                if np.linalg.norm(item[1] - slot)
                <= self.config.closed_candidate_slot_radius_m
                and abs(float(np.dot(item[1] - fixture.center, fixture.vertical)))
                <= self.config.closed_candidate_vertical_offset_m
            ]
            if not reliable:
                return None
            # Connected edge fragments often represent the top and bottom of
            # one physical vertical handle.  Prefer the fragment nearest the
            # sensor-derived handle slot; raw component area alone favours a
            # large appliance/body edge in the wrist view.
            confidence, point = min(
                reliable,
                key=lambda item: (
                    float(np.linalg.norm(item[1] - slot)),
                    -item[0],
                ),
            )
        frame_trace["selected_point_world_m"] = point.tolist()
        frame_trace["selected_confidence"] = float(confidence)
        self.last_detection_trace["cameras"][
            camera.calibration.name
        ] = frame_trace
        return point, confidence

    def _local_rgbd_vertical_edge(
        self,
        camera: CameraFrame,
        fixture: _MicrowaveFixtureFrame,
        *,
        local_anchor_world: np.ndarray,
        local_anchor_radius_m: float,
        frozen_hinge_world: np.ndarray,
        frozen_rotation_axis_world: np.ndarray,
        frozen_radius_m: float,
        frozen_radius_tolerance_m: float,
        diagnostics: dict[str, object],
    ) -> tuple[float, np.ndarray] | None:
        """Aggregate a local RGB-D silhouette before fitting a vertical core.

        Near the jamb, the free edge can be connected to the entire appliance
        contour, so component-level shape gates discard it before the local
        geometry is considered.  This path is available only with both the
        predicted-contact and frozen-circle gates.  It never emits a body or
        closed-slot fallback.
        """

        gray = cv2.cvtColor(camera.rgb, cv2.COLOR_RGB2GRAY)
        colour_edge = cv2.Canny(gray, 28, 84) > 0
        depth = np.asarray(camera.depth_m, dtype=np.float64)
        valid = np.isfinite(depth) & (depth > 0.0)
        depth_edge = np.zeros(depth.shape, dtype=bool)
        horizontal_valid = valid[:, 1:] & valid[:, :-1]
        horizontal_jump = np.abs(depth[:, 1:] - depth[:, :-1]) >= 0.008
        depth_edge[:, 1:] |= horizontal_valid & horizontal_jump
        depth_edge[:, :-1] |= horizontal_valid & horizontal_jump
        vertical_valid = valid[1:, :] & valid[:-1, :]
        vertical_jump = np.abs(depth[1:, :] - depth[:-1, :]) >= 0.008
        depth_edge[1:, :] |= vertical_valid & vertical_jump
        depth_edge[:-1, :] |= vertical_valid & vertical_jump
        points = _points(camera, (colour_edge | depth_edge) & valid)
        diagnostics["local_aggregate_raw_edge_points"] = int(len(points))
        if len(points) < self.config.open_vertical_core_min_points:
            return None

        delta = points - fixture.center[None, :]
        outward_coordinate = delta @ fixture.outward
        side_coordinate = delta @ fixture.handle_side
        vertical_coordinate = delta @ fixture.vertical
        anchor_side_coordinate = float(
            np.dot(
                local_anchor_world - fixture.center,
                fixture.handle_side,
            )
        )
        local_delta = points - local_anchor_world[None, :]
        local_planar = local_delta - np.outer(
            local_delta @ frozen_rotation_axis_world,
            frozen_rotation_axis_world,
        )
        radial = points - frozen_hinge_world[None, :]
        radial -= np.outer(
            radial @ frozen_rotation_axis_world,
            frozen_rotation_axis_world,
        )
        selected = points[
            (
                np.abs(vertical_coordinate)
                <= fixture.vertical_half_extent_m + self.config.vertical_margin_m
            )
            & (
                outward_coordinate
                >= fixture.outward_half_extent_m
                - self.config.closed_front_inner_margin_m
            )
            & (
                outward_coordinate
                <= fixture.outward_half_extent_m
                + self.config.open_outward_reach_m
            )
            & (
                side_coordinate
                >= anchor_side_coordinate - local_anchor_radius_m
            )
            & (
                side_coordinate
                <= anchor_side_coordinate + local_anchor_radius_m
            )
            & (np.linalg.norm(local_planar, axis=1) <= local_anchor_radius_m)
            & (
                np.abs(np.linalg.norm(radial, axis=1) - frozen_radius_m)
                <= frozen_radius_tolerance_m
            )
        ]
        diagnostics["local_aggregate_gated_edge_points"] = int(len(selected))
        if len(selected) < self.config.open_vertical_core_min_points:
            return None
        core_diagnostics = diagnostics["vertical_core_candidates"]
        assert isinstance(core_diagnostics, list)
        core = self._open_vertical_core(
            selected,
            fixture,
            local_anchor_world=local_anchor_world,
            local_anchor_radius_m=local_anchor_radius_m,
            frozen_hinge_world=frozen_hinge_world,
            frozen_rotation_axis_world=frozen_rotation_axis_world,
            frozen_radius_m=frozen_radius_m,
            frozen_radius_tolerance_m=frozen_radius_tolerance_m,
            diagnostics=core_diagnostics,
        )
        if core is None:
            return None
        core_points, point = core
        diagnostics["local_aggregate_selected_point_world_m"] = point.tolist()
        coverage = min(
            1.0,
            float(np.ptp(core_points @ fixture.vertical))
            / max(2.0 * fixture.vertical_half_extent_m, 1e-9),
        )
        confidence = float(np.clip(0.46 + 0.28 * coverage, 0.0, 0.80))
        return confidence, point

    def _open_vertical_core(
        self,
        points: np.ndarray,
        fixture: _MicrowaveFixtureFrame,
        *,
        local_anchor_world: np.ndarray | None = None,
        local_anchor_radius_m: float | None = None,
        frozen_hinge_world: np.ndarray | None = None,
        frozen_rotation_axis_world: np.ndarray | None = None,
        frozen_radius_m: float | None = None,
        frozen_radius_tolerance_m: float | None = None,
        diagnostics: list[dict[str, object]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Extract one vertical free edge from a connected open-door frame.

        An open microwave door is often observed as a rectangular contour.
        Treating the complete contour as a handle fails the thinness gate,
        while simply relaxing that gate admits cups and broad appliance
        borders.  This bounded RANSAC-like search finds a narrow cluster in
        the fixture's planar coordinates with door-height vertical support.
        """

        if len(points) < self.config.open_vertical_core_min_points:
            return None
        delta = points - fixture.center[None, :]
        planar = np.column_stack(
            (delta @ fixture.handle_side, delta @ fixture.outward)
        )
        vertical = delta @ fixture.vertical
        tail = self.config.center_tail_quantile
        planar_lower, planar_upper = np.quantile(
            planar,
            (tail, 1.0 - tail),
            axis=0,
        )
        if float(np.max(planar_upper - planar_lower)) > (
            self.config.open_vertical_core_max_frame_span_m
        ):
            return None
        sample_count = min(len(points), 96)
        seed_indices = np.linspace(
            0,
            len(points) - 1,
            sample_count,
            dtype=np.int64,
        )
        seed_indices = np.unique(
            np.concatenate(
                (
                    seed_indices,
                    np.argmin(planar, axis=0),
                    np.argmax(planar, axis=0),
                )
            )
        )
        minimum_span = max(
            self.config.open_vertical_core_min_span_m,
            self.config.open_vertical_core_min_height_fraction
            * 2.0
            * fixture.vertical_half_extent_m,
        )
        candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
        for index in seed_indices:
            inlier = (
                np.linalg.norm(planar - planar[index][None, :], axis=1)
                <= self.config.open_vertical_core_radius_m
            )
            if np.count_nonzero(inlier) < self.config.open_vertical_core_min_points:
                continue
            core_planar = planar[inlier]
            core_vertical = vertical[inlier]
            planar_lower, planar_upper = np.quantile(
                core_planar,
                (tail, 1.0 - tail),
                axis=0,
            )
            vertical_lower, vertical_upper = np.quantile(
                core_vertical,
                (tail, 1.0 - tail),
            )
            planar_spans = planar_upper - planar_lower
            vertical_span = float(vertical_upper - vertical_lower)
            planar_major = float(max(*planar_spans, 0.006))
            if not (
                minimum_span <= vertical_span <= self.config.max_vertical_span_m
                and planar_major
                <= 2.15 * self.config.open_vertical_core_radius_m
                and vertical_span
                >= self.config.open_vertical_core_aspect * planar_major
            ):
                continue
            median_side = float(np.median(core_planar[:, 0]))
            side_progress = float(
                np.clip(
                    median_side
                    / max(
                        fixture.handle_side_half_extent_m
                        + self.config.open_handle_side_reach_m,
                        1e-9,
                    ),
                    0.0,
                    1.0,
                )
            )
            height_coverage = min(
                vertical_span
                / max(2.0 * fixture.vertical_half_extent_m, 1e-9),
                1.0,
            )
            score = (
                height_coverage
                + 0.30 * side_progress
                + 0.001 * min(int(np.count_nonzero(inlier)), 100)
                - 0.5 * float(np.sum(planar_spans))
            )
            core = points[inlier].copy()
            # Contour sampling is rarely uniform: a long horizontal bottom
            # edge can contribute many pixels while only a few samples reach
            # the top of the same physical vertical edge.  The coordinatewise
            # median would then collapse onto the bottom corner.  Interpolate
            # to the midpoint of the robust *visible* vertical endpoints.
            # Keep planar coordinates on the median measured surface: the
            # midpoint of a door slab's planar extent can lie inside it.
            visible_vertical_midpoint = 0.5 * (
                float(vertical_lower) + float(vertical_upper)
            )
            midpoint = np.median(core, axis=0)
            sampled_vertical_median = float(
                np.dot(midpoint - fixture.center, fixture.vertical)
            )
            midpoint += fixture.vertical * (
                visible_vertical_midpoint - sampled_vertical_median
            )
            if local_anchor_world is not None:
                assert local_anchor_radius_m is not None
                assert frozen_hinge_world is not None
                assert frozen_rotation_axis_world is not None
                assert frozen_radius_m is not None
                assert frozen_radius_tolerance_m is not None
                anchor_error = float(
                    np.linalg.norm(midpoint - local_anchor_world)
                )
                radial = midpoint - frozen_hinge_world
                radial -= frozen_rotation_axis_world * float(
                    np.dot(radial, frozen_rotation_axis_world)
                )
                radius_error = abs(
                    float(np.linalg.norm(radial)) - frozen_radius_m
                )
                passes_local_geometry = bool(
                    anchor_error <= local_anchor_radius_m
                    and radius_error <= frozen_radius_tolerance_m
                )
                if diagnostics is not None:
                    diagnostics.append(
                        {
                            "point_world_m": midpoint.tolist(),
                            "anchor_error_m": anchor_error,
                            "frozen_radius_residual_m": radius_error,
                            "vertical_span_m": vertical_span,
                            "score": float(score),
                            "passes_local_geometry": passes_local_geometry,
                        }
                    )
                if not passes_local_geometry:
                    continue
            candidates.append((score, core, midpoint))
        if not candidates:
            return None
        if local_anchor_world is not None:
            _, core, midpoint = min(
                candidates,
                key=lambda item: (
                    float(np.linalg.norm(item[2] - local_anchor_world)),
                    -item[0],
                ),
            )
        else:
            _, core, midpoint = max(candidates, key=lambda item: item[0])
        return core, midpoint


class StoveKnobDetector:
    """Find the compact dark knob adjacent to a low-chroma metal fixture."""

    def __init__(self, config: FixtureDetectorConfig | None = None) -> None:
        self.config = config or FixtureDetectorConfig()

    def detect(self, observation: RobotObservation) -> ContactTarget:
        proposals: list[ContactTarget] = []
        for name in ("agentview", "wrist"):
            target = self._detect_frame(observation.cameras[name])
            if target is not None:
                proposals.append(target)
        if not proposals:
            raise LookupError("no stove knob was visible in either RGB-D view")
        # Agent view is normally unobstructed; wrist becomes a consistency vote.
        anchor = proposals[0]
        agreeing = [item for item in proposals if np.linalg.norm(item.point_world - anchor.point_world) < 0.06]
        if len(agreeing) == 1:
            return anchor
        point = np.mean([item.point_world for item in agreeing], axis=0)
        fixture = np.mean([item.fixture_center_world for item in agreeing], axis=0)
        outward = _unit_xy(point - fixture)
        return ContactTarget(
            GoalSkillKind.TURN_KNOB,
            point,
            np.array([0.0, 0.0, 1.0]),
            outward,
            fixture,
            anchor.feature_axis_world,
            min(0.98, max(item.confidence for item in agreeing) + 0.08),
            tuple(item.source_cameras[0] for item in agreeing),
        )

    def _detect_frame(self, frame: CameraFrame) -> ContactTarget | None:
        rgb = frame.rgb
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        chroma = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
        dark = cv2.morphologyEx((gray < 55).astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        metal = (gray >= 55) & (gray <= 210) & (chroma <= 24)
        candidates: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for _, stats, mask in _components(dark):
            x, y, width, height, area = map(int, stats)
            fill = area / max(width * height, 1)
            aspect = width / max(height, 1)
            if not (120 <= area <= 900 and 10 <= width <= 45 and 10 <= height <= 45):
                continue
            if not (0.5 <= aspect <= 1.8 and fill >= 0.34):
                continue
            world = _points(frame, mask)
            if len(world) < 40:
                continue
            median_z = float(np.median(world[:, 2]))
            if not self.config.knob_z_range[0] <= median_z <= self.config.knob_z_range[1]:
                continue
            pad = 42
            local = np.zeros_like(metal)
            local[max(0, y - pad): min(frame.calibration.height, y + height + pad),
                  max(0, x - pad): min(frame.calibration.width, x + width + pad)] = True
            nearby_metal = metal & local
            metal_count = int(np.count_nonzero(nearby_metal))
            if metal_count < 180:
                continue
            fixture_points = _points(frame, nearby_metal)
            fixture_points = fixture_points[
                (fixture_points[:, 2] >= 0.86)
                & (fixture_points[:, 2] <= 1.08)
                & (np.abs(fixture_points[:, 0]) <= 0.70)
                & (np.abs(fixture_points[:, 1]) <= 0.70)
            ]
            if len(fixture_points) < 40:
                continue
            # A compact dark cabinet feature or bottle can have nearby gray
            # pixels in image space. Bind the control to an actual broad,
            # shallow metal surface before accepting it as a stove knob.
            point = np.median(world, axis=0)
            provisional = ContactTarget(
                GoalSkillKind.TURN_KNOB, point, np.array([0., 0., 1.]),
                _unit_xy(point - np.median(fixture_points, axis=0)),
                np.median(fixture_points, axis=0), _horizontal_axis(world),
                .5, (frame.calibration.name,),
            )
            surface = PlateFrontDetector._stove_surface_in_frame(frame, provisional)
            if surface is None:
                continue
            distances = np.linalg.norm(surface[:, :2] - point[:2], axis=1)
            if float(np.quantile(distances, .01)) > .075:
                continue
            fixture_points = surface
            score = 1.5 * fill + min(area, 360) / 500.0 + min(metal_count, 1800) / 2500.0
            candidates.append((score, mask, world, nearby_metal, fixture_points))
        if not candidates:
            return None
        score, _mask, world, _metal_mask, fixture_points = max(candidates, key=lambda item: item[0])
        point = np.median(world, axis=0)
        fixture = np.median(fixture_points, axis=0)
        outward = _unit_xy(point - fixture)
        axis = _horizontal_axis(world)
        confidence = float(np.clip(0.35 + score / 4.0, 0, 0.96))
        return ContactTarget(
            GoalSkillKind.TURN_KNOB,
            point,
            np.array([0.0, 0.0, 1.0]),
            outward,
            fixture,
            axis,
            confidence,
            (frame.calibration.name,),
        )


class PlateFrontDetector:
    """Find the red-rimmed plate and derive the free-space stove-front goal."""

    def __init__(
        self,
        knob_detector: StoveKnobDetector | None = None,
        config: FixtureDetectorConfig | None = None,
    ) -> None:
        self.config = config or FixtureDetectorConfig()
        self.knob_detector = knob_detector or StoveKnobDetector(self.config)

    def detect(self, observation: RobotObservation) -> PushTarget:
        knob = self.knob_detector.detect(observation)
        plate_proposals: list[tuple[np.ndarray, float, float, str]] = []
        for name in ("agentview", "wrist"):
            proposal = self._plate_in_frame(observation.cameras[name])
            if proposal is not None:
                center, radius, confidence = proposal
                plate_proposals.append((center, radius, confidence, name))
        if not plate_proposals:
            raise LookupError("no red-rimmed plate was visible in either RGB-D view")
        center, radius, confidence, camera = max(plate_proposals, key=lambda item: item[2])
        stove_groups: list[np.ndarray] = []
        for name in ("agentview", "wrist"):
            surface = self._stove_surface_in_frame(observation.cameras[name], knob)
            if surface is not None:
                stove_groups.append(surface)
        if not stove_groups:
            raise LookupError("no metric stove surface was visible in either RGB-D view")
        stove_points = np.concatenate(stove_groups, axis=0)
        stove_center = np.median(stove_points, axis=0)
        # The control / knob edge is the rear of LIBERO's flat stove.  Its
        # opposite long-axis edge is the semantic front.  Derive that axis
        # from the knob-to-surface vector, then step into visible free space
        # beyond the robust front edge.
        outward = _unit_xy(stove_center - knob.point_world)
        along = stove_points @ outward
        front_edge = float(np.quantile(along, 0.95))
        target = stove_center.copy()
        target += outward * (
            front_edge - float(stove_center @ outward) + self.config.front_clearance_m
        )
        target[2] = center[2]
        direction = _unit_xy(target - center)
        cameras = tuple(dict.fromkeys((camera, *knob.source_cameras)))
        return PushTarget(
            center,
            target,
            direction,
            radius,
            min(confidence, knob.confidence),
            cameras,
        )

    def track(self, observation: RobotObservation, reference: PushTarget) -> PushTarget:
        """Re-localize the plate while retaining the detected stove goal.

        The arm can occlude the compact knob after a long drag even when the
        red plate remains visible.  Requiring a second knob detection would
        therefore turn an object-visibility check into a fixture-visibility
        failure.  The goal geometry was already measured before motion; this
        method refreshes only the moving object from the current RGB-D views.
        """

        proposals: list[tuple[np.ndarray, float, float, str]] = []
        for name in ("agentview", "wrist"):
            proposal = self._plate_in_frame(observation.cameras[name])
            if proposal is not None:
                center, radius, confidence = proposal
                proposals.append((center, radius, confidence, name))
        if not proposals:
            raise LookupError("no red-rimmed plate was visible in either RGB-D view")
        center, radius, confidence, camera = max(proposals, key=lambda item: item[2])
        return PushTarget(
            center,
            reference.target_center_world,
            reference.direction_world,
            radius,
            confidence,
            (camera,),
        )

    @staticmethod
    def _stove_surface_in_frame(
        frame: CameraFrame,
        knob: ContactTarget,
    ) -> np.ndarray | None:
        rgb = frame.rgb
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        chroma = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
        metal = (gray >= 55) & (gray <= 210) & (chroma <= 24)
        candidates: list[tuple[float, np.ndarray]] = []
        for _, stats, mask in _components(metal):
            _, _, width, height, area = map(int, stats)
            if area < 450 or width < 25 or height < 20:
                continue
            world = _points(frame, mask)
            world = world[(world[:, 2] >= 0.86) & (world[:, 2] <= 1.02)]
            if len(world) < 300:
                continue
            lower = np.quantile(world, 0.05, axis=0)
            upper = np.quantile(world, 0.95, axis=0)
            planar = np.sort(upper[:2] - lower[:2])
            vertical = float(upper[2] - lower[2])
            if not (0.08 <= planar[0] <= 0.35 and 0.11 <= planar[1] <= 0.42):
                continue
            if vertical > 0.065:
                continue
            surface_center = np.median(world, axis=0)
            knob_distance = float(np.linalg.norm(surface_center[:2] - knob.point_world[:2]))
            if knob_distance > 0.36:
                continue
            score = float(len(world)) + 1600.0 * planar.sum() - 500.0 * knob_distance
            candidates.append((score, world))
        return max(candidates, key=lambda item: item[0])[1] if candidates else None

    @staticmethod
    def _plate_in_frame(frame: CameraFrame) -> tuple[np.ndarray, float, float] | None:
        hsv = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2HSV)
        red = (
            (hsv[:, :, 0] <= 12)
            & (hsv[:, :, 1] >= 40)
            & (hsv[:, :, 2] >= 60)
        )
        red = cv2.morphologyEx(red.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        candidates: list[tuple[float, np.ndarray, np.ndarray, int, int]] = []
        for _, stats, mask in _components(red):
            _, _, width, height, area = map(int, stats)
            aspect = width / max(height, 1)
            if not (100 <= area <= 1800 and 20 <= width <= 80 and 18 <= height <= 70):
                continue
            if not 0.65 <= aspect <= 1.55:
                continue
            world = _points(frame, mask)
            if len(world) < 50:
                continue
            planar = np.ptp(world[:, :2], axis=0)
            radius = float(np.clip(np.mean(planar) / 2.0, 0.025, 0.09))
            circularity = min(aspect, 1.0 / aspect)
            candidates.append((area * circularity, world, mask, width, height))
        if not candidates:
            return None
        score, world, _mask, width, height = max(candidates, key=lambda item: item[0])
        center = np.median(world, axis=0)
        planar = np.ptp(world[:, :2], axis=0)
        radius = float(np.clip(np.mean(planar) / 2.0, 0.025, 0.09))
        confidence = float(np.clip(0.45 + score / 2200.0 + min(width, height) / 160.0, 0, 0.96))
        return center, radius, confidence
