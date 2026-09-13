"""Deterministic content audit for formal dual-camera episode videos.

The ordinary run verifier checks container metadata.  This module deliberately
goes one step further and decodes the complete video: it proves the number and
shape of the decoded frames, checks that both camera halves contain image
content and change during the episode, and detects accidentally duplicated
camera views.

The audit is read-only.  Its return value contains JSON-native values only so
it can be embedded verbatim in an episode record or verification report.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np


VIDEO_CONTENT_AUDIT_SCHEMA = "libero-dual-view-video-content-audit.v1"


@dataclass(frozen=True)
class VideoContentAuditThresholds:
    """Conservative thresholds for rendered LIBERO RGB evidence.

    Spatial content is checked on every decoded frame.  Temporal change is
    measured on a small grayscale copy of every frame, which makes long static
    pauses harmless while still rejecting an episode that is constant from
    start to finish.  Cross-view similarity is more expensive and is therefore
    evaluated on a deterministic, uniformly spaced sample.
    """

    min_spatial_luma_std: float = 1.0
    min_mean_luma: float = 1.0
    max_mean_luma: float = 254.0
    min_temporal_luma_mad: float = 0.25
    # Lossy MP4 block prediction can make two originally identical side-by-
    # side halves differ by roughly four RGB levels after decode.  Five keeps
    # that encoding noise inside the duplicate gate while remaining far below
    # the difference between independent LIBERO viewpoints.
    near_duplicate_rgb_mad: float = 5.0
    max_near_duplicate_fraction: float = 0.95
    cross_view_sample_limit: int = 64
    temporal_downsample_size: int = 64

    def __post_init__(self) -> None:
        nonnegative = (
            "min_spatial_luma_std",
            "min_mean_luma",
            "max_mean_luma",
            "min_temporal_luma_mad",
            "near_duplicate_rgb_mad",
        )
        for name in nonnegative:
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be a finite number")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be a finite non-negative number")
        if self.min_mean_luma >= self.max_mean_luma:
            raise ValueError("min_mean_luma must be less than max_mean_luma")
        fraction = self.max_near_duplicate_fraction
        if not isinstance(fraction, (int, float)) or isinstance(fraction, bool):
            raise TypeError("max_near_duplicate_fraction must be a finite number")
        if not math.isfinite(float(fraction)) or not 0.0 <= float(fraction) <= 1.0:
            raise ValueError("max_near_duplicate_fraction must be in [0, 1]")
        for name in ("cross_view_sample_limit", "temporal_downsample_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


def _positive_integer(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _sample_indices(frame_count: int, limit: int) -> frozenset[int]:
    """Uniform zero-based indices including both endpoints, without duplicates."""

    if frame_count <= limit:
        return frozenset(range(frame_count))
    if limit == 1:
        return frozenset((0,))
    denominator = limit - 1
    return frozenset(
        (sample * (frame_count - 1) + denominator // 2) // denominator
        for sample in range(limit)
    )


def _rounded(value: float | None) -> float | None:
    if value is None:
        return None
    if not math.isfinite(value):
        return None
    return round(float(value), 6)


def _view_metrics(
    *,
    means: list[float],
    spatial_stds: list[float],
    blank_indices: list[int],
    max_temporal_mad: float,
) -> dict[str, Any]:
    return {
        "frames_measured": len(means),
        "blank_frame_count": len(blank_indices),
        "blank_frame_indices": list(blank_indices),
        "mean_luma_min": _rounded(min(means) if means else None),
        "mean_luma_max": _rounded(max(means) if means else None),
        "spatial_luma_std_min": _rounded(min(spatial_stds) if spatial_stds else None),
        "spatial_luma_std_max": _rounded(max(spatial_stds) if spatial_stds else None),
        "max_consecutive_temporal_luma_mad": _rounded(max_temporal_mad),
    }


def _failure_report(
    *,
    path: Path,
    expected_frames: int,
    expected_height: int,
    expected_half_width: int,
    thresholds: VideoContentAuditThresholds,
    reasons: list[str],
    sha256: str | None = None,
    file_size_bytes: int | None = None,
) -> dict[str, Any]:
    return {
        "schema": VIDEO_CONTENT_AUDIT_SCHEMA,
        "formal_pass": False,
        "path": str(path),
        "sha256": sha256,
        "file_size_bytes": file_size_bytes,
        "expected": {
            "frames": expected_frames,
            "height": expected_height,
            "half_width": expected_half_width,
            "full_width": expected_half_width * 2,
        },
        "thresholds": asdict(thresholds),
        "decoded": {
            "frames": 0,
            "height": None,
            "width": None,
        },
        "metrics": {
            "temporal_variation_applicable": expected_frames > 1,
            "invalid_shape_frame_count": 0,
            "nonfinite_frame_count": 0,
            "dimension_mismatch_frame_count": 0,
            "cross_view_sample_count": 0,
            "near_duplicate_sample_count": 0,
            "near_duplicate_fraction": None,
            "cross_view_rgb_mad_min": None,
            "cross_view_rgb_mad_max": None,
            "external": _view_metrics(
                means=[],
                spatial_stds=[],
                blank_indices=[],
                max_temporal_mad=0.0,
            ),
            "wrist": _view_metrics(
                means=[],
                spatial_stds=[],
                blank_indices=[],
                max_temporal_mad=0.0,
            ),
        },
        "reasons": list(dict.fromkeys(reasons)),
    }


def audit_dual_view_video(
    path: str | Path,
    *,
    expected_frames: int,
    expected_height: int = 256,
    expected_half_width: int = 256,
    thresholds: VideoContentAuditThresholds | None = None,
) -> dict[str, Any]:
    """Decode and audit one side-by-side external/wrist MP4.

    ``expected_frames`` is the exact number dictated by the episode's action
    count and recorder stride (currently ``steps // stride + 1``).  A report is
    returned for missing, corrupt, or non-conforming files; invalid caller
    arguments raise ``ValueError`` before any file access.
    """

    expected_frames = _positive_integer("expected_frames", expected_frames)
    expected_height = _positive_integer("expected_height", expected_height)
    expected_half_width = _positive_integer("expected_half_width", expected_half_width)
    limits = thresholds or VideoContentAuditThresholds()
    video_path = Path(path)

    if not video_path.exists():
        return _failure_report(
            path=video_path,
            expected_frames=expected_frames,
            expected_height=expected_height,
            expected_half_width=expected_half_width,
            thresholds=limits,
            reasons=["file_missing"],
        )
    if not video_path.is_file():
        return _failure_report(
            path=video_path,
            expected_frames=expected_frames,
            expected_height=expected_height,
            expected_half_width=expected_half_width,
            thresholds=limits,
            reasons=["not_a_file"],
        )

    try:
        digest = _sha256(video_path)
        file_size = int(video_path.stat().st_size)
    except OSError:
        return _failure_report(
            path=video_path,
            expected_frames=expected_frames,
            expected_height=expected_height,
            expected_half_width=expected_half_width,
            thresholds=limits,
            reasons=["file_read_failed"],
        )

    initial_reasons: list[str] = []
    if video_path.suffix.lower() != ".mp4":
        initial_reasons.append("not_mp4")

    try:
        import cv2
    except ImportError:  # pragma: no cover - OpenCV is part of the runtime env
        return _failure_report(
            path=video_path,
            expected_frames=expected_frames,
            expected_height=expected_height,
            expected_half_width=expected_half_width,
            thresholds=limits,
            reasons=[*initial_reasons, "decoder_dependency_unavailable"],
            sha256=digest,
            file_size_bytes=file_size,
        )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        return _failure_report(
            path=video_path,
            expected_frames=expected_frames,
            expected_height=expected_height,
            expected_half_width=expected_half_width,
            thresholds=limits,
            reasons=[*initial_reasons, "decoder_open_failed"],
            sha256=digest,
            file_size_bytes=file_size,
        )

    sampled_indices = _sample_indices(expected_frames, limits.cross_view_sample_limit)
    decoded_frames = 0
    first_height: int | None = None
    first_width: int | None = None
    invalid_shape_count = 0
    nonfinite_count = 0
    dimension_mismatch_count = 0
    odd_width_count = 0
    external_means: list[float] = []
    wrist_means: list[float] = []
    external_stds: list[float] = []
    wrist_stds: list[float] = []
    external_blanks: list[int] = []
    wrist_blanks: list[int] = []
    previous_small: tuple[np.ndarray, np.ndarray] | None = None
    max_temporal = [0.0, 0.0]
    cross_view_mads: list[float] = []
    decode_exception = False

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index = decoded_frames
            decoded_frames += 1
            if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
                invalid_shape_count += 1
                continue
            height, width = int(frame.shape[0]), int(frame.shape[1])
            if first_height is None:
                first_height, first_width = height, width
            if frame.dtype.kind in "fc" and not bool(np.isfinite(frame).all()):
                nonfinite_count += 1
                continue
            if width % 2:
                odd_width_count += 1
                dimension_mismatch_count += 1
                continue
            if height != expected_height or width != expected_half_width * 2:
                dimension_mismatch_count += 1

            half_width = width // 2
            external = frame[:, :half_width]
            wrist = frame[:, half_width:]
            external_gray = cv2.cvtColor(external, cv2.COLOR_BGR2GRAY)
            wrist_gray = cv2.cvtColor(wrist, cv2.COLOR_BGR2GRAY)
            grays = (external_gray, wrist_gray)
            view_means = (external_means, wrist_means)
            view_stds = (external_stds, wrist_stds)
            view_blanks = (external_blanks, wrist_blanks)
            small_views: list[np.ndarray] = []
            for view_index, gray in enumerate(grays):
                mean, std = cv2.meanStdDev(gray)
                mean_value = float(mean[0, 0])
                std_value = float(std[0, 0])
                view_means[view_index].append(mean_value)
                view_stds[view_index].append(std_value)
                if (
                    std_value < limits.min_spatial_luma_std
                    or mean_value <= limits.min_mean_luma
                    or mean_value >= limits.max_mean_luma
                ):
                    view_blanks[view_index].append(frame_index)
                small = cv2.resize(
                    gray,
                    (limits.temporal_downsample_size, limits.temporal_downsample_size),
                    interpolation=cv2.INTER_AREA,
                )
                small_views.append(small)

            current_small = (small_views[0], small_views[1])
            if previous_small is not None:
                for view_index in (0, 1):
                    delta = cv2.absdiff(current_small[view_index], previous_small[view_index])
                    max_temporal[view_index] = max(
                        max_temporal[view_index], float(cv2.mean(delta)[0])
                    )
            previous_small = current_small

            if frame_index in sampled_indices:
                difference = cv2.absdiff(external, wrist)
                cross_view_mads.append(float(sum(cv2.mean(difference)[:3]) / 3.0))
    except (cv2.error, MemoryError, OverflowError, ValueError):
        decode_exception = True
    finally:
        capture.release()

    reasons = list(initial_reasons)
    if decode_exception:
        reasons.append("decode_exception")
    if decoded_frames == 0:
        reasons.append("zero_decoded_frames")
    if decoded_frames != expected_frames:
        reasons.append("decoded_frame_count_mismatch")
    if invalid_shape_count:
        reasons.append("invalid_frame_shape")
    if nonfinite_count:
        reasons.append("nonfinite_frame")
    if odd_width_count:
        reasons.append("odd_frame_width")
    if dimension_mismatch_count:
        reasons.append("frame_dimension_mismatch")
    if external_blanks:
        reasons.append("external_blank_frame")
    if wrist_blanks:
        reasons.append("wrist_blank_frame")
    # A policy may validly reject a task immediately after reset.  Its
    # complete evidence is one frame, so temporal variation is inapplicable;
    # count, shape, nonblank halves, cross-view independence, and bytes are
    # still audited.  Requiring an artificial action here would bias formal
    # results by turning a genuine zero-action policy failure into infra loss.
    temporal_variation_applicable = expected_frames > 1
    if temporal_variation_applicable and (
        not external_means or max_temporal[0] < limits.min_temporal_luma_mad
    ):
        reasons.append("external_no_temporal_variation")
    if temporal_variation_applicable and (
        not wrist_means or max_temporal[1] < limits.min_temporal_luma_mad
    ):
        reasons.append("wrist_no_temporal_variation")

    near_duplicate_count = sum(
        difference <= limits.near_duplicate_rgb_mad for difference in cross_view_mads
    )
    near_duplicate_fraction = (
        near_duplicate_count / len(cross_view_mads) if cross_view_mads else None
    )
    if not cross_view_mads:
        reasons.append("no_cross_view_samples")
    elif near_duplicate_fraction is not None and (
        near_duplicate_fraction >= limits.max_near_duplicate_fraction
    ):
        reasons.append("views_near_duplicate")

    report = {
        "schema": VIDEO_CONTENT_AUDIT_SCHEMA,
        "formal_pass": not reasons,
        "path": str(video_path),
        "sha256": digest,
        "file_size_bytes": file_size,
        "expected": {
            "frames": expected_frames,
            "height": expected_height,
            "half_width": expected_half_width,
            "full_width": expected_half_width * 2,
        },
        "thresholds": asdict(limits),
        "decoded": {
            "frames": decoded_frames,
            "height": first_height,
            "width": first_width,
        },
        "metrics": {
            "temporal_variation_applicable": temporal_variation_applicable,
            "invalid_shape_frame_count": invalid_shape_count,
            "nonfinite_frame_count": nonfinite_count,
            "dimension_mismatch_frame_count": dimension_mismatch_count,
            "cross_view_sample_count": len(cross_view_mads),
            "near_duplicate_sample_count": near_duplicate_count,
            "near_duplicate_fraction": _rounded(near_duplicate_fraction),
            "cross_view_rgb_mad_min": _rounded(
                min(cross_view_mads) if cross_view_mads else None
            ),
            "cross_view_rgb_mad_max": _rounded(
                max(cross_view_mads) if cross_view_mads else None
            ),
            "external": _view_metrics(
                means=external_means,
                spatial_stds=external_stds,
                blank_indices=external_blanks,
                max_temporal_mad=max_temporal[0],
            ),
            "wrist": _view_metrics(
                means=wrist_means,
                spatial_stds=wrist_stds,
                blank_indices=wrist_blanks,
                max_temporal_mad=max_temporal[1],
            ),
        },
        "reasons": list(dict.fromkeys(reasons)),
    }
    # Keep this an exact native bool even if downstream code uses numpy values.
    report["formal_pass"] = bool(report["formal_pass"])
    return report


__all__ = [
    "VIDEO_CONTENT_AUDIT_SCHEMA",
    "VideoContentAuditThresholds",
    "audit_dual_view_video",
]
