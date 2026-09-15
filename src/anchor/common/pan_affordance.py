"""Route-independent frying-pan handle geometry from a segmented RGB-D cloud.

The helpers in this module deliberately know nothing about LIBERO task ids,
initial states, simulator bodies, or evaluator state.  They consume only a
whole-pan point cloud in the calibrated world frame, an optional per-point RGB
occupancy predicate, and the current public end-effector pose.

The estimator is conservative.  A pan must contain one wide, approximately
round body and exactly one sufficiently long, narrow tail.  Candidate grasp
slots must additionally contain observed surface near their centre, which
keeps a distal hanging hole from being treated as a solid grasp surface.
Ambiguous geometry raises :class:`PanAffordanceError` instead of inventing a
handle direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


class PanAffordanceError(RuntimeError):
    """The public sensor evidence is insufficient for a safe handle grasp."""


@dataclass(frozen=True)
class PanAffordanceConfig:
    """Metric and scale-relative gates for pan-handle inference.

    ``rgb_occupancy`` passed to :func:`infer_pan_handle_affordance` is a
    per-point boolean mask or confidence in ``[0, 1]``.  It is intended for a
    public RGB foreground predicate registered to the depth cloud, not a
    simulator segmentation mask.
    """

    min_points: int = 160
    longitudinal_bins: int = 44
    geometry_quantile: float = 0.01
    minimum_pca_aspect_ratio: float = 1.08
    body_core_width_ratio: float = 0.68
    minimum_body_core_bins: int = 4
    tail_probe_radius_ratio: float = 0.88
    minimum_tail_length_body_ratio: float = 0.22
    minimum_tail_length_m: float = 0.035
    maximum_tail_width_body_ratio: float = 0.48
    minimum_narrow_bin_fraction: float = 0.62
    minimum_tail_bin_coverage: float = 0.58
    neck_clearance_body_ratio: float = 0.055
    tip_clearance_handle_ratio: float = 0.08
    minimum_tip_clearance_m: float = 0.006
    slot_fractions: tuple[float, ...] = (0.50, 0.32, 0.68, 0.80)
    slot_window_body_ratio: float = 0.065
    minimum_slot_window_m: float = 0.006
    minimum_handle_width_m: float = 0.008
    maximum_handle_width_m: float = 0.065
    maximum_center_gap_width_ratio: float = 0.15
    maximum_center_gap_m: float = 0.004
    minimum_distinct_slot_distance_m: float = 0.012
    minimum_slots: int = 2
    occupancy_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.min_points < 32:
            raise ValueError("min_points must be at least 32")
        if self.longitudinal_bins < 16:
            raise ValueError("longitudinal_bins must be at least 16")
        if not 0.0 < self.geometry_quantile < 0.1:
            raise ValueError("geometry_quantile must be in (0, 0.1)")
        if self.minimum_pca_aspect_ratio <= 1.0:
            raise ValueError("minimum_pca_aspect_ratio must exceed one")
        for name in (
            "body_core_width_ratio",
            "tail_probe_radius_ratio",
            "minimum_tail_length_body_ratio",
            "maximum_tail_width_body_ratio",
            "minimum_narrow_bin_fraction",
            "minimum_tail_bin_coverage",
            "neck_clearance_body_ratio",
            "tip_clearance_handle_ratio",
            "slot_window_body_ratio",
            "maximum_center_gap_width_ratio",
            "occupancy_threshold",
        ):
            value = float(getattr(self, name))
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be in (0, 1)")
        if self.minimum_body_core_bins < 2:
            raise ValueError("minimum_body_core_bins must be at least two")
        positive_lengths = (
            self.minimum_tail_length_m,
            self.minimum_tip_clearance_m,
            self.minimum_slot_window_m,
            self.minimum_handle_width_m,
            self.maximum_handle_width_m,
            self.maximum_center_gap_m,
            self.minimum_distinct_slot_distance_m,
        )
        if any(float(value) <= 0.0 for value in positive_lengths):
            raise ValueError("metric pan-affordance gates must be positive")
        if self.minimum_handle_width_m >= self.maximum_handle_width_m:
            raise ValueError("handle width bounds are reversed")
        if self.minimum_slots < 2:
            raise ValueError("minimum_slots must be at least two")
        if len(self.slot_fractions) < self.minimum_slots:
            raise ValueError("slot_fractions cannot satisfy minimum_slots")
        fractions = tuple(float(value) for value in self.slot_fractions)
        if len(set(fractions)) != len(fractions) or any(
            not 0.0 < value < 1.0 for value in fractions
        ):
            raise ValueError("slot_fractions must be unique and in (0, 1)")
        object.__setattr__(self, "slot_fractions", fractions)


@dataclass(frozen=True)
class PanHandleSlot:
    """One sensor-supported, physically distinct grasp point on the handle."""

    slot_id: str
    position_world: FloatArray
    longitudinal_fraction: float
    local_top_z_m: float
    local_width_m: float
    center_gap_m: float
    score: float

    def __post_init__(self) -> None:
        point = _finite_array(self.position_world, (3,), "slot.position_world")
        if not self.slot_id:
            raise ValueError("slot_id is required")
        for name in (
            "longitudinal_fraction",
            "local_top_z_m",
            "local_width_m",
            "center_gap_m",
            "score",
        ):
            if not np.isfinite(float(getattr(self, name))):
                raise ValueError(f"slot.{name} must be finite")
        if not 0.0 < float(self.longitudinal_fraction) < 1.0:
            raise ValueError("slot.longitudinal_fraction must be in (0, 1)")
        if float(self.local_width_m) <= 0.0 or float(self.center_gap_m) < 0.0:
            raise ValueError("slot width must be positive and gap non-negative")
        object.__setattr__(self, "position_world", point)


@dataclass(frozen=True)
class PanHandleAffordance:
    """Signed handle geometry and solid mid-handle grasp candidates."""

    handle_axis_world: FloatArray
    jaw_axis_world: FloatArray
    body_center_world: FloatArray
    handle_start_world: FloatArray
    handle_end_world: FloatArray
    body_diameter_m: float
    handle_length_m: float
    median_handle_width_m: float
    tail_bin_coverage: float
    narrow_bin_fraction: float
    slots: tuple[PanHandleSlot, ...]

    def __post_init__(self) -> None:
        handle = _unit_vector(self.handle_axis_world, "handle_axis_world")
        jaw = _unit_vector(self.jaw_axis_world, "jaw_axis_world")
        if abs(float(np.dot(handle, jaw))) > 1e-6:
            raise ValueError("handle and jaw axes must be perpendicular")
        for name in ("body_center_world", "handle_start_world", "handle_end_world"):
            object.__setattr__(self, name, _finite_array(getattr(self, name), (3,), name))
        for name in (
            "body_diameter_m",
            "handle_length_m",
            "median_handle_width_m",
            "tail_bin_coverage",
            "narrow_bin_fraction",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        slots = tuple(self.slots)
        if not slots:
            raise ValueError("at least one handle slot is required")
        object.__setattr__(self, "handle_axis_world", handle)
        object.__setattr__(self, "jaw_axis_world", jaw)
        object.__setattr__(self, "slots", slots)


@dataclass(frozen=True)
class _SideEvidence:
    sign: float
    far_u_m: float
    extra_length_m: float
    median_width_m: float
    bin_coverage: float
    narrow_fraction: float


def _finite_array(value: object, shape: tuple[int, ...], name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {shape}, got {result.shape}")
    return result.copy()


def _unit_vector(value: object, name: str) -> FloatArray:
    result = _finite_array(value, (3,), name)
    norm = float(np.linalg.norm(result))
    if norm < 1e-9:
        raise ValueError(f"{name} must be non-zero")
    return result / norm


def _validated_pose(value: object) -> FloatArray:
    pose = _finite_array(value, (4, 4), "current_world_from_ee")
    if not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
        raise ValueError("current_world_from_ee has an invalid homogeneous row")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
        raise ValueError("current_world_from_ee rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-3):
        raise ValueError("current_world_from_ee rotation must be right handed")
    return pose


def _occupied_points(
    points_world: object,
    rgb_occupancy: object | None,
    config: PanAffordanceConfig,
) -> FloatArray:
    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("points_world must be a finite Nx3 array")
    if not np.all(np.isfinite(points)):
        raise ValueError("points_world must not contain NaN or infinity")
    if rgb_occupancy is not None:
        occupancy = np.asarray(rgb_occupancy)
        if occupancy.shape != (len(points),):
            raise ValueError("rgb_occupancy must have one value per world point")
        if occupancy.dtype == np.bool_:
            keep = occupancy
        else:
            confidence = np.asarray(occupancy, dtype=np.float64)
            if not np.all(np.isfinite(confidence)) or np.any(confidence < 0.0) or np.any(
                confidence > 1.0
            ):
                raise ValueError("rgb_occupancy confidences must be finite and in [0, 1]")
            keep = confidence >= config.occupancy_threshold
        points = points[keep]
    if len(points) < config.min_points:
        raise PanAffordanceError(
            f"only {len(points)} occupied pan points; need at least {config.min_points}"
        )
    return points.copy()


def _largest_true_run(mask: NDArray[np.bool_]) -> tuple[int, int] | None:
    best: tuple[int, int] | None = None
    start: int | None = None
    for index, value in enumerate((*mask.tolist(), False)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            candidate = (start, index)
            if best is None or candidate[1] - candidate[0] > best[1] - best[0]:
                best = candidate
            start = None
    return best


def _slice_profile(
    longitudinal: FloatArray,
    transverse: FloatArray,
    config: PanAffordanceConfig,
) -> tuple[FloatArray, FloatArray, NDArray[np.int64], float, float]:
    quantile = config.geometry_quantile
    lower, upper = np.quantile(longitudinal, (quantile, 1.0 - quantile))
    if not np.isfinite(lower) or not np.isfinite(upper) or upper - lower < 0.08:
        raise PanAffordanceError("pan planar extent is too small for handle inference")
    edges = np.linspace(lower, upper, config.longitudinal_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2.0
    bin_index = np.searchsorted(edges, longitudinal, side="right") - 1
    bin_index = np.clip(bin_index, 0, config.longitudinal_bins - 1)
    minimum_count = max(4, len(longitudinal) // (config.longitudinal_bins * 30))
    widths = np.full(config.longitudinal_bins, np.nan, dtype=np.float64)
    counts = np.zeros(config.longitudinal_bins, dtype=np.int64)
    for index in range(config.longitudinal_bins):
        values = transverse[bin_index == index]
        counts[index] = len(values)
        if len(values) >= minimum_count:
            low, high = np.quantile(values, (0.05, 0.95))
            widths[index] = max(float(high - low), 0.0)
    return centres, widths, counts, float(lower), float(upper)


def _body_geometry(
    centres: FloatArray,
    widths: FloatArray,
    config: PanAffordanceConfig,
) -> tuple[float, float, tuple[int, int]]:
    valid = np.isfinite(widths) & (widths > 0.0)
    if int(np.count_nonzero(valid)) < config.minimum_body_core_bins:
        raise PanAffordanceError("too few occupied longitudinal slices")
    body_diameter = float(np.quantile(widths[valid], 0.88))
    if body_diameter < 2.0 * config.minimum_handle_width_m:
        raise PanAffordanceError("wide pan body was not observed")
    body_like = valid & (widths >= config.body_core_width_ratio * body_diameter)
    run = _largest_true_run(body_like)
    if run is None or run[1] - run[0] < config.minimum_body_core_bins:
        raise PanAffordanceError("no contiguous wide pan body was observed")
    body_indices = np.arange(run[0], run[1])
    weights = np.square(widths[body_indices])
    body_s = float(np.average(centres[body_indices], weights=weights))
    return body_s, body_diameter, run


def _side_evidence(
    sign: float,
    body_s: float,
    body_radius: float,
    body_diameter: float,
    longitudinal: FloatArray,
    centres: FloatArray,
    widths: FloatArray,
    config: PanAffordanceConfig,
) -> _SideEvidence | None:
    signed_points = sign * (longitudinal - body_s)
    far_u = float(np.quantile(signed_points, 1.0 - config.geometry_quantile))
    extra_length = far_u - body_radius
    minimum_length = max(
        config.minimum_tail_length_m,
        config.minimum_tail_length_body_ratio * body_diameter,
    )
    if extra_length < minimum_length:
        return None

    signed_centres = sign * (centres - body_s)
    tail_start = config.tail_probe_radius_ratio * body_radius
    selected = (signed_centres >= tail_start) & (signed_centres <= far_u)
    selected_count = int(np.count_nonzero(selected))
    if selected_count < 3:
        return None
    valid = selected & np.isfinite(widths) & (widths > 0.0)
    valid_count = int(np.count_nonzero(valid))
    bin_coverage = valid_count / selected_count
    if valid_count < 3 or bin_coverage < config.minimum_tail_bin_coverage:
        return None
    selected_widths = widths[valid]
    width_limit = config.maximum_tail_width_body_ratio * body_diameter
    narrow_fraction = float(np.mean(selected_widths <= width_limit))
    median_width = float(np.median(selected_widths))
    if narrow_fraction < config.minimum_narrow_bin_fraction or median_width > width_limit:
        return None
    if not config.minimum_handle_width_m <= median_width <= config.maximum_handle_width_m:
        return None
    return _SideEvidence(
        sign=float(sign),
        far_u_m=far_u,
        extra_length_m=extra_length,
        median_width_m=median_width,
        bin_coverage=float(bin_coverage),
        narrow_fraction=narrow_fraction,
    )


def _solid_slots(
    points: FloatArray,
    origin_xy: FloatArray,
    major_xy: FloatArray,
    minor_xy: FloatArray,
    longitudinal: FloatArray,
    transverse: FloatArray,
    body_s: float,
    body_diameter: float,
    side: _SideEvidence,
    config: PanAffordanceConfig,
) -> tuple[tuple[PanHandleSlot, ...], float, float, float]:
    body_radius = body_diameter / 2.0
    start_u = body_radius + config.neck_clearance_body_ratio * body_diameter
    tip_clearance = max(
        config.minimum_tip_clearance_m,
        config.tip_clearance_handle_ratio * side.extra_length_m,
    )
    end_u = side.far_u_m - tip_clearance
    if end_u - start_u < config.minimum_distinct_slot_distance_m:
        raise PanAffordanceError("observed handle has no safe mid-section")

    signed_u = side.sign * (longitudinal - body_s)
    window = max(
        config.minimum_slot_window_m,
        config.slot_window_body_ratio * body_diameter,
    )
    tail_band = (signed_u >= start_u) & (signed_u <= end_u)
    if int(np.count_nonzero(tail_band)) < 12:
        raise PanAffordanceError("too few points on the safe handle mid-section")
    # Use one line fitted across the complete safe tail.  Computing a separate
    # median transverse coordinate in every slot would let the centre jump to
    # one edge of a hanging hole, incorrectly re-labelling that edge as solid.
    handle_t_center = float(np.median(transverse[tail_band]))
    slots: list[PanHandleSlot] = []
    for fraction in config.slot_fractions:
        target_u = start_u + fraction * (end_u - start_u)
        local = np.abs(signed_u - target_u) <= window
        if int(np.count_nonzero(local)) < 8:
            continue
        local_t = transverse[local]
        t_low, t_high = np.quantile(local_t, (0.05, 0.95))
        local_width = float(t_high - t_low)
        width_limit = min(
            config.maximum_handle_width_m,
            config.maximum_tail_width_body_ratio * body_diameter,
        )
        if not config.minimum_handle_width_m <= local_width <= width_limit:
            continue
        t_center = handle_t_center
        planar_gap = np.sqrt(
            np.square(signed_u[local] - target_u)
            + np.square(local_t - t_center)
        )
        center_gap = float(np.min(planar_gap))
        allowed_gap = max(
            config.maximum_center_gap_m,
            config.maximum_center_gap_width_ratio * local_width,
        )
        # A distal hanging hole has points on both lateral edges and therefore
        # a plausible slice width, but no registered RGB-D surface at the
        # candidate centre.  This gap gate rejects exactly that geometry.
        if center_gap > allowed_gap:
            continue
        central = local & (np.abs(transverse - t_center) <= 0.35 * local_width)
        if int(np.count_nonzero(central)) < 4:
            continue
        local_top = float(np.quantile(points[central, 2], 0.90))
        signed_offset = side.sign * target_u
        xy = (
            origin_xy
            + (body_s + signed_offset) * major_xy
            + t_center * minor_xy
        )
        point = np.array((xy[0], xy[1], local_top), dtype=np.float64)
        centre_preference = 1.0 - min(abs(fraction - 0.5) / 0.5, 1.0)
        fill_score = max(0.0, 1.0 - center_gap / allowed_gap)
        score = 0.65 * fill_score + 0.35 * centre_preference
        candidate = PanHandleSlot(
            slot_id=f"handle-u{int(round(100.0 * fraction)):02d}",
            position_world=point,
            longitudinal_fraction=fraction,
            local_top_z_m=local_top,
            local_width_m=local_width,
            center_gap_m=center_gap,
            score=score,
        )
        if all(
            np.linalg.norm(candidate.position_world[:2] - old.position_world[:2])
            >= config.minimum_distinct_slot_distance_m
            for old in slots
        ):
            slots.append(candidate)
    if len(slots) < config.minimum_slots:
        raise PanAffordanceError(
            f"only {len(slots)} solid handle slots; need at least {config.minimum_slots}"
        )
    slots.sort(key=lambda item: (-item.score, item.longitudinal_fraction))
    return tuple(slots), start_u, end_u, handle_t_center


def infer_pan_handle_affordance(
    points_world: object,
    current_world_from_ee: object,
    *,
    rgb_occupancy: object | None = None,
    config: PanAffordanceConfig | None = None,
) -> PanHandleAffordance:
    """Infer solid mid-handle grasp slots from public RGB-D geometry.

    Args:
        points_world: ``Nx3`` whole-pan foreground points in the calibrated
            world frame.  Simulator object state and instance masks are not
            valid inputs.
        current_world_from_ee: Public proprioceptive ``4x4`` end-effector pose.
            Its local-Y sign is used only to choose the closest equivalent jaw
            axis; the preceding task's wrist yaw is never copied as a grasp
            orientation.
        rgb_occupancy: Optional per-point boolean foreground decision or RGB
            occupancy confidence in ``[0, 1]`` registered to ``points_world``.
        config: Conservative geometry thresholds.

    Returns:
        A signed body-to-handle axis, a perpendicular Panda jaw axis, local
        handle height, and at least two physically distinct solid slots.

    Raises:
        PanAffordanceError: if the geometry has no unique narrow tail, contains
            too little evidence, or cannot supply enough solid mid-handle slots.
        ValueError: if an input has an invalid shape or non-finite value.
    """

    cfg = config or PanAffordanceConfig()
    points = _occupied_points(points_world, rgb_occupancy, cfg)
    current = _validated_pose(current_world_from_ee)

    xy = points[:, :2]
    # The arithmetic centroid is exactly equivariant under planar rigid-body
    # transforms.  A component-wise median is attractive for robustness but
    # silently changes the PCA frame when the same cloud is merely rotated.
    # Foreground/outlier rejection belongs in the registered occupancy input.
    origin_xy = np.mean(xy, axis=0)
    centred = xy - origin_xy
    covariance = centred.T @ centred / max(len(centred) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if eigenvalues[0] <= 1e-12:
        raise PanAffordanceError("pan cloud is planar-degenerate")
    pca_aspect = float(np.sqrt(eigenvalues[1] / eigenvalues[0]))
    if pca_aspect < cfg.minimum_pca_aspect_ratio:
        raise PanAffordanceError(
            f"pan major axis is ambiguous (PCA aspect {pca_aspect:.3f})"
        )
    major_xy = np.asarray(eigenvectors[:, 1], dtype=np.float64)
    major_xy /= float(np.linalg.norm(major_xy))
    minor_xy = np.array((-major_xy[1], major_xy[0]), dtype=np.float64)
    longitudinal = centred @ major_xy
    transverse = centred @ minor_xy
    centres, widths, _, _, _ = _slice_profile(
        longitudinal,
        transverse,
        cfg,
    )
    body_s, body_diameter, body_run = _body_geometry(centres, widths, cfg)
    body_radius = body_diameter / 2.0

    side_candidates = tuple(
        evidence
        for sign in (-1.0, 1.0)
        if (
            evidence := _side_evidence(
                sign,
                body_s,
                body_radius,
                body_diameter,
                longitudinal,
                centres,
                widths,
                cfg,
            )
        )
        is not None
    )
    if not side_candidates:
        raise PanAffordanceError("no sufficiently long, narrow handle tail was observed")
    if len(side_candidates) != 1:
        raise PanAffordanceError("ambiguous pan geometry: narrow tails exist on both sides")
    side = side_candidates[0]

    body_slice = (longitudinal >= centres[body_run[0]]) & (
        longitudinal <= centres[body_run[1] - 1]
    )
    body_t = float(np.median(transverse[body_slice]))
    body_z = float(np.median(points[body_slice, 2]))
    body_xy = origin_xy + body_s * major_xy + body_t * minor_xy
    body_center = np.array((body_xy[0], body_xy[1], body_z), dtype=np.float64)

    slots, start_u, end_u, handle_t_center = _solid_slots(
        points,
        origin_xy,
        major_xy,
        minor_xy,
        longitudinal,
        transverse,
        body_s,
        body_diameter,
        side,
        cfg,
    )
    handle_axis = np.array(
        (side.sign * major_xy[0], side.sign * major_xy[1], 0.0),
        dtype=np.float64,
    )
    jaw_axis = np.cross(np.array((0.0, 0.0, 1.0)), handle_axis)
    jaw_axis /= float(np.linalg.norm(jaw_axis))
    current_jaw_xy = np.array(current[:3, 1], dtype=np.float64, copy=True)
    current_jaw_xy[2] = 0.0
    current_jaw_norm = float(np.linalg.norm(current_jaw_xy))
    if current_jaw_norm < 0.25:
        raise PanAffordanceError("current Panda local-Y jaw axis is not sufficiently planar")
    current_jaw_xy /= current_jaw_norm
    if float(np.dot(jaw_axis, current_jaw_xy)) < 0.0:
        jaw_axis *= -1.0

    handle_line_origin_xy = (
        origin_xy + body_s * major_xy + handle_t_center * minor_xy
    )
    handle_start = body_center.copy()
    handle_start[:2] = handle_line_origin_xy + start_u * handle_axis[:2]
    handle_start[2] = float(np.median([slot.local_top_z_m for slot in slots]))
    handle_end = handle_start.copy()
    handle_end[:2] = handle_line_origin_xy + end_u * handle_axis[:2]

    return PanHandleAffordance(
        handle_axis_world=handle_axis,
        jaw_axis_world=jaw_axis,
        body_center_world=body_center,
        handle_start_world=handle_start,
        handle_end_world=handle_end,
        body_diameter_m=body_diameter,
        handle_length_m=side.extra_length_m,
        median_handle_width_m=side.median_width_m,
        tail_bin_coverage=side.bin_coverage,
        narrow_bin_fraction=side.narrow_fraction,
        slots=slots,
    )


__all__: Sequence[str] = (
    "PanAffordanceConfig",
    "PanAffordanceError",
    "PanHandleAffordance",
    "PanHandleSlot",
    "infer_pan_handle_affordance",
)
