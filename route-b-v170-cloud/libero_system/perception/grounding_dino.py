"""Optional frozen Grounding-DINO boxes and sensor-only box geometry.

Importing this module does not load a model.  The adapter defaults to the local
Hugging Face snapshot and ``local_files_only=True`` so a benchmark run cannot
silently download or update weights.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .gallery import hsv_histogram
from .geometry import (
    ComponentConfig,
    backproject_frame,
    center_ray_intersection,
    connected_components_3d,
    fit_observed_geometry,
)
from .schema import BoxDetection, ObjectInstance, ObservedGeometry, PointCloud, RGBDFrame, normalize_label


import os

DEFAULT_GROUNDING_DINO_TINY = Path(os.environ.get(
    "ROUTE_B_MODEL_PATH",
    str(Path(__file__).resolve().parents[2] / "resources" / "grounding-dino-tiny"),
)).expanduser()


def rectangular_fixture_geometry(points: np.ndarray, observed: ObservedGeometry) -> ObservedGeometry:
    """Fit the outer fixture edges without weighting densely visible panels."""
    import cv2

    xy = np.asarray(points[:, :2], dtype=np.float32)
    lower, upper = np.quantile(xy, [.01, .99], axis=0)
    trimmed = xy[np.all((xy >= lower) & (xy <= upper), axis=1)]
    if len(trimmed) < 8:
        return observed
    center, size, angle = cv2.minAreaRect(trimmed)
    if min(size) < .020 or max(size) < .20:
        return observed
    radians = np.deg2rad(angle)
    major = np.array((np.cos(radians), np.sin(radians)))
    if size[1] > size[0]:
        major = np.array((-major[1], major[0]))
    if major[0] < 0 or (abs(major[0]) < 1e-9 and major[1] < 0):
        major *= -1
    axes = np.array(((major[0], -major[1], 0.), (major[1], major[0], 0.), (0., 0., 1.)))
    return replace(
        observed,
        center_world=np.array((*center, observed.center_world[2])),
        axes_world=axes,
        extents_m=np.array((max(size), min(size), observed.extents_m[2])),
    )


def workspace_rgb(
    frame: RGBDFrame, lower: NDArray[np.floating], upper: NDArray[np.floating]
) -> NDArray[np.uint8]:
    """Suppress distant scene distractors using calibrated metric depth.

    Pixel coordinates stay unchanged so detections still refer to the original
    RGB-D frame used for segmentation and geometry extraction.
    """
    cloud = backproject_frame(frame)
    inside = np.all(
        (cloud.points_world >= lower) & (cloud.points_world <= upper), axis=1
    )
    rgb = np.full_like(frame.rgb, 127)
    pixels = cloud.pixels_uv[inside].astype(np.intp)
    rgb[pixels[:, 1], pixels[:, 0]] = frame.rgb[pixels[:, 1], pixels[:, 0]]
    return rgb


def _match_grounded_query(
    detected_label: str,
    queries: Sequence[str],
) -> str | None:
    """Map one model phrase to the prompt phrase without substring stealing."""

    detected = normalize_label(detected_label.strip(" ."))
    normalized = tuple(normalize_label(query) for query in queries)
    exact = [query for query in normalized if query == detected]
    if exact:
        return exact[0]
    containment = [
        query for query in normalized if query in detected or detected in query
    ]
    if containment:
        return max(containment, key=len)
    detected_words = set(detected.split())
    overlap = [len(detected_words & set(query.split())) for query in normalized]
    if max(overlap, default=0) == 0:
        return None
    return normalized[int(np.argmax(overlap))]


def _aligned_detection_rows(
    boxes: NDArray[np.floating],
    scores: NDArray[np.floating],
    text_labels: Sequence[object],
    queries: Sequence[str],
) -> tuple[tuple[NDArray[np.float64], float, object], ...]:
    """Defensively align a Transformers Grounding-DINO postprocess result.

    Transformers 5.x can occasionally return stale surplus ``text_labels``
    after score filtering.  Boxes and scores are the authoritative filtered
    tensors and must agree exactly; surplus labels preserve order and can be
    truncated.  If labels are short, a missing row is unambiguous only for a
    single-query prompt, otherwise it is skipped instead of assigning a wrong
    semantic class or aborting the episode.
    """

    boxes_array = np.asarray(boxes)
    scores_array = np.asarray(scores).reshape(-1)
    if len(boxes_array) != len(scores_array):
        raise RuntimeError(
            "Grounding-DINO postprocess returned different box and score counts"
        )
    labels = list(text_labels)[: len(boxes_array)]
    if len(labels) < len(boxes_array) and len(queries) == 1:
        labels.extend([queries[0]] * (len(boxes_array) - len(labels)))
    row_count = min(len(boxes_array), len(labels))
    return tuple(
        (np.asarray(boxes_array[index], dtype=np.float64), float(scores_array[index]), labels[index])
        for index in range(row_count)
    )


@runtime_checkable
class FrozenBoxDetector(Protocol):
    def detect(self, frame: RGBDFrame, queries: Sequence[str]) -> Sequence[BoxDetection]: ...


class GroundingDINOBoxDetector:
    """Per-query Grounding-DINO inference with permanently frozen parameters."""

    def __init__(
        self,
        model_name_or_path: str | Path = DEFAULT_GROUNDING_DINO_TINY,
        *,
        device: str = "auto",
        score_threshold: float = 0.18,
        text_threshold: float = 0.15,
        max_detections_per_query: int = 6,
        local_files_only: bool = True,
        workspace_bounds: tuple[NDArray[np.floating], NDArray[np.floating]] | None = None,
    ) -> None:
        if not 0 <= score_threshold <= 1 or not 0 <= text_threshold <= 1:
            raise ValueError("DINO thresholds must lie in [0, 1]")
        if max_detections_per_query <= 0:
            raise ValueError("max_detections_per_query must be positive")
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("torch and transformers are required for Grounding-DINO") from exc
        model_path = str(model_name_or_path)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=local_files_only,
        )
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_path,
            local_files_only=local_files_only,
        ).to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.score_threshold = float(score_threshold)
        self.text_threshold = float(text_threshold)
        self.max_detections_per_query = int(max_detections_per_query)
        self.workspace_bounds = None
        if workspace_bounds is not None:
            lower, upper = (np.array(value, dtype=np.float64, copy=True)
                            for value in workspace_bounds)
            if (lower.shape != (3,) or upper.shape != (3,)
                    or not np.all(np.isfinite([lower, upper]))
                    or np.any(lower >= upper)):
                raise ValueError("detector workspace must be ordered finite 3-vectors")
            self.workspace_bounds = (lower, upper)
        self._torch = torch

    def detect(self, frame: RGBDFrame, queries: Sequence[str]) -> tuple[BoxDetection, ...]:
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Pillow is required for Grounding-DINO images") from exc
        rgb = (frame.rgb if self.workspace_bounds is None
               else workspace_rgb(frame, *self.workspace_bounds))
        image = Image.fromarray(rgb, mode="RGB")
        normalized_queries = tuple(
            dict.fromkeys(normalize_label(query) for query in queries if str(query).strip())
        )
        if not normalized_queries:
            return ()
        # One compound prompt is one model forward.  Running one forward per
        # entity made a three-entity Spatial observation roughly three times
        # slower on CPU.
        prompt = ". ".join(normalized_queries) + "."
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with self._torch.inference_mode():
            outputs = self.model(**inputs)
        processed = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.get("input_ids"),
            threshold=self.score_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(frame.height, frame.width)],
        )[0]
        boxes = processed["boxes"].detach().cpu().numpy()
        scores = processed["scores"].detach().cpu().numpy()
        text_labels = processed.get("text_labels", processed.get("labels", []))
        grouped: dict[str, list[tuple[float, NDArray[np.float64]]]] = {
            query: [] for query in normalized_queries
        }
        for box, score, raw_label in _aligned_detection_rows(
            boxes, scores, text_labels, normalized_queries
        ):
            detected_label = normalize_label(str(raw_label).strip(" ."))
            # An exact phrase owns its box.  Without this precedence, a
            # detected ``rack`` could be incorrectly assigned to the longer
            # prompt ``wine rack`` when both phrases are present.  Longest
            # containment remains useful only when the model emits a phrase
            # absent from the prompt.
            query = _match_grounded_query(detected_label, normalized_queries)
            if query is None:
                continue
            grouped[query].append((float(score), np.asarray(box, dtype=np.float64)))
        detections: list[BoxDetection] = []
        for query, rows in grouped.items():
            for score, raw_box in sorted(rows, key=lambda row: row[0], reverse=True)[
                : self.max_detections_per_query
            ]:
                box = raw_box.copy()
                box[[0, 2]] = np.clip(box[[0, 2]], 0, frame.width - 1)
                box[[1, 3]] = np.clip(box[[1, 3]], 0, frame.height - 1)
                if box[2] - box[0] < 2 or box[3] - box[1] < 2:
                    continue
                detections.append(
                    BoxDetection(
                        query=query,
                        xyxy=box,
                        score=score,
                        camera_name=frame.name,
                    )
                )
        return tuple(detections)


class BoxGeometryExtractor:
    """Turn a frozen-model box into a foreground mask and visible 3-D OBB."""

    def __init__(
        self,
        *,
        depth_band_m: float = 0.10,
        table_margin_m: float = 0.0015,
        use_grabcut: bool = True,
        grabcut_rng_seed: int = 0,
        component: ComponentConfig | None = None,
    ) -> None:
        if depth_band_m <= 0 or table_margin_m < 0:
            raise ValueError("invalid box-extraction distances")
        if not -(2**31) <= int(grabcut_rng_seed) < 2**31:
            raise ValueError("grabcut_rng_seed must fit a signed 32-bit integer")
        self.depth_band_m = float(depth_band_m)
        self.table_margin_m = float(table_margin_m)
        self.use_grabcut = bool(use_grabcut)
        self.grabcut_rng_seed = int(grabcut_rng_seed)
        self.component = component or ComponentConfig(
            voxel_size_m=0.004,
            connectivity_radius_m=0.012,
            min_points=12,
            min_voxels=3,
        )

    @staticmethod
    def _integer_box(frame: RGBDFrame, box: NDArray[np.floating]) -> tuple[int, int, int, int]:
        x0 = max(0, int(np.floor(box[0])))
        y0 = max(0, int(np.floor(box[1])))
        x1 = min(frame.width, int(np.ceil(box[2])) + 1)
        y1 = min(frame.height, int(np.ceil(box[3])) + 1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            raise ValueError("detection box is outside the frame")
        return x0, y0, x1, y1

    def foreground_mask(self, frame: RGBDFrame, detection: BoxDetection) -> NDArray[np.bool_]:
        if detection.camera_name != frame.name:
            raise ValueError("detection and RGB-D camera names differ")
        x0, y0, x1, y1 = self._integer_box(frame, detection.xyxy)
        rectangle = np.zeros(frame.depth_m.shape, dtype=np.bool_)
        rectangle[y0:y1, x0:x1] = True
        foreground = rectangle.copy()
        if self.use_grabcut and x0 > 0 and y0 > 0 and x1 < frame.width and y1 < frame.height:
            try:
                import cv2

                labels = np.zeros(frame.depth_m.shape, dtype=np.uint8)
                background = np.zeros((1, 65), dtype=np.float64)
                foreground_model = np.zeros((1, 65), dtype=np.float64)
                # OpenCV's GrabCut initialises its colour GMM with the
                # process-global RNG.  Without reseeding here, an identical
                # RGB-D frame can produce a different foreground after earlier
                # episodes have consumed a different number of detections.
                # Evaluation is serial, so resetting immediately before the
                # call makes extraction depend only on this frame and box.
                cv2.setRNGSeed(self.grabcut_rng_seed)
                cv2.grabCut(
                    frame.rgb,
                    labels,
                    (x0, y0, x1 - x0, y1 - y0),
                    background,
                    foreground_model,
                    3,
                    cv2.GC_INIT_WITH_RECT,
                )
                proposed = rectangle & ((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD))
                if np.count_nonzero(proposed) >= 12:
                    foreground = proposed
            except Exception:  # OpenCV is optional; deterministic depth fallback follows.
                # Depth filtering below remains deterministic when OpenCV is
                # absent or GrabCut cannot fit a tiny crop.
                foreground = rectangle.copy()

        depth = frame.depth_m
        centre_x = int(round((x0 + x1 - 1) / 2))
        centre_y = int(round((y0 + y1 - 1) / 2))
        radius_x = max(1, (x1 - x0) // 8)
        radius_y = max(1, (y1 - y0) // 8)
        central = depth[
            max(y0, centre_y - radius_y) : min(y1, centre_y + radius_y + 1),
            max(x0, centre_x - radius_x) : min(x1, centre_x + radius_x + 1),
        ]
        central = central[np.isfinite(central) & (central > 0)]
        if len(central):
            median = float(np.median(central))
            mad = float(np.median(np.abs(central - median)))
            band = max(self.depth_band_m, 6.0 * 1.4826 * mad)
            foreground &= np.isfinite(depth) & (np.abs(depth - median) <= band)
        else:
            foreground &= np.isfinite(depth) & (depth > 0)
        return foreground

    def extract(
        self,
        frame: RGBDFrame,
        detection: BoxDetection,
        *,
        support_height_m: float | None = None,
        above_support_only: bool = True,
        instance_id: str | None = None,
    ) -> tuple[NDArray[np.bool_], ObjectInstance]:
        """Return the established mask/instance pair.

        Consumers that need affordance geometry can call
        :meth:`extract_with_cloud`; keeping this wrapper preserves the shared
        Route-B and third-party extractor contract.
        """

        mask, instance, _ = self.extract_with_cloud(
            frame,
            detection,
            support_height_m=support_height_m,
            above_support_only=above_support_only,
            instance_id=instance_id,
        )
        return mask, instance

    def extract_with_cloud(
        self,
        frame: RGBDFrame,
        detection: BoxDetection,
        *,
        support_height_m: float | None = None,
        above_support_only: bool = True,
        instance_id: str | None = None,
    ) -> tuple[NDArray[np.bool_], ObjectInstance, PointCloud]:
        """Also return the exact public RGB-D component used for the OBB."""

        mask = self.foreground_mask(frame, detection)
        cloud = backproject_frame(frame, mask=mask, camera_index=0)
        if above_support_only:
            if support_height_m is None:
                raise ValueError("support_height_m is required for above-support extraction")
            keep = cloud.points_world[:, 2] >= support_height_m + self.table_margin_m
            cloud = cloud.subset(keep)
        if len(cloud.points_world) < 8:
            raise ValueError(f"box for {detection.query!r} has too few foreground depth points")
        components = connected_components_3d(cloud.points_world, self.component)
        if components:
            if any(token in normalize_label(detection.query)
                   for token in ("tray", "stove", "burner", "hot plate")):
                # Flat fixtures expose a broad connected surface. Tiny table
                # strips visible through a handle/rim can cross the exact box
                # centre and otherwise outrank the fixture itself. Keep
                # substantial components before applying the centre-ray cue.
                minimum_points = .10 * max(len(indices) for indices in components)
                components = [indices for indices in components
                              if len(indices) >= minimum_points]
            # The centre ray generally crosses the target; prefer the component
            # whose projected bbox centre is closest to the detection centre.
            target_uv = np.array(
                [(detection.xyxy[0] + detection.xyxy[2]) / 2, (detection.xyxy[1] + detection.xyxy[3]) / 2]
            )
            selected = min(
                components,
                key=lambda indices: float(
                    np.linalg.norm(np.median(cloud.pixels_uv[indices], axis=0) - target_uv)
                ),
            )
            cloud = cloud.subset(selected)
        observed = fit_observed_geometry(cloud.points_world, support_height_m=support_height_m)
        if any(token in normalize_label(detection.query) for token in ("caddy", "desk organizer")):
            observed = rectangular_fixture_geometry(cloud.points_world, observed)
        if support_height_m is None:
            center = observed.center_world.copy()
            grasp = observed.bounds_max_world.copy()
            grasp[:2] = center[:2]
        else:
            height = max(float(observed.extents_m[2]), 0.004)
            center = center_ray_intersection(
                cloud,
                [frame],
                support_height_m + height / 2.0,
                fallback_xy=observed.center_world[:2],
            )
            grasp = center.copy()
            grasp[2] = support_height_m + height
        instance = ObjectInstance(
            instance_id=(
                instance_id
                or f"{normalize_label(detection.query).replace(' ', '-')}-0"
            ),
            center_world=center,
            grasp_point_world=grasp,
            axes_world=observed.axes_world,
            extents_m=np.maximum(observed.extents_m, 0.002),
            observed=observed,
            label_scores={detection.query: float(detection.score)},
            confidence=float(detection.score),
            point_count=len(cloud.points_world),
            color_histogram=hsv_histogram(cloud.colors_rgb),
        )
        return mask, instance, cloud
