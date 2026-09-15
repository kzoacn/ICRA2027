"""Thin ANCHOR / Planning bridges around the neutral sensor-only API.

The imports of route-specific types are intentionally local so the perception
package remains usable on its own and does not create a dependency cycle.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .grounding_dino import BoxGeometryExtractor, FrozenBoxDetector
from .fusion import fuse_instances_3d, resolve_geometric_selector
from .gallery import (
    GALLERY_QUERY_PRIORS,
    LIBERO_OBJECT_LABELS,
    GalleryQueryPrior,
    TextureSizeGallery,
    hsv_histogram,
)
from .geometry import (
    ComponentConfig,
    connected_components_3d,
    fit_observed_geometry,
    fuse_rgbd_frames,
)
from .pipeline import SensorOnlyScenePerception
from .schema import BoxDetection, ObjectInstance, RGBDFrame, SceneObservation, normalize_label


def _uses_object_gallery(queries: Sequence[str]) -> bool:
    """Object mode also applies when a held-object phase asks only for basket."""

    return any(
        normalize_label(query) in LIBERO_OBJECT_LABELS
        or normalize_label(query) == "basket"
        for query in queries
    )


_SPATIAL_GALLERY_LABELS = frozenset({"black bowl", "plate", "ramekin", "cookie box"})
_STACKABLE_GALLERY_LABELS = _SPATIAL_GALLERY_LABELS
_IN_FIXTURE_TOKENS = (
    "drawer",
    "cabinet",
    "microwave",
    "stove",
    "rack",
    "shelf",
    "caddy",
    "tray",
)
# These labels are grounded by the frozen open-vocabulary detector first.  A
# gallery family may validate/rank its crop, but is never allowed to turn an
# arbitrary connected component into a detection by one-class classification.
_DINO_GALLERY_LABELS = frozenset((*GALLERY_QUERY_PRIORS, *_SPATIAL_GALLERY_LABELS))
_STRICT_DINO_GALLERY_LABELS = frozenset(
    {
        "black bowl",
        "cookie box",
        "white bowl",
        "bowl",
        "plate",
        "ramekin",
        "red mug",
        "white mug",
        "yellow and white mug",
        "moka pot",
        "book",
        "frying pan",
        "wine bottle",
        "cream cheese box",
    }
)
_LARGE_FIXTURE_LABELS = frozenset(
    {
        "cabinet",
        "wooden cabinet",
        "cabinet shelf",
        "caddy",
        "drawer",
        "microwave",
        "rack",
        "shelf",
        "stove",
        "tray",
        "wine rack",
    }
)
_DINO_QUERY_ALIASES: Mapping[str, tuple[str, ...]] = {
    # The public fixture asset and DINO vocabulary use "flat stove", while
    # LIBERO's instruction says simply "stove".  In a top-down kitchen view
    # the frozen detector often names the same compact single-burner fixture
    # a "single burner electric hot plate"; keep that visually specific
    # synonym in the shared query expansion and retain the existing RGB-D
    # shape/gallery gates downstream.  The longer phrase also avoids promoting
    # an unrelated handled frying pan that the generic phrase "hot plate" can
    # weakly match in the same view.
    "stove": (
        "flat stove",
        "kitchen cooktop",
        "stovetop",
        "single burner electric hot plate",
    ),
    "black bowl": ("black ceramic bowl",),
    "white bowl": ("white ceramic bowl",),
    "bowl": ("ceramic bowl", "small bowl"),
    "plate": ("white dinner plate",),
    "ramekin": ("white ceramic ramekin", "small ceramic ramekin"),
    "cookie box": ("box of cookies", "cookie package"),
    "red mug": ("red coffee mug", "red cup"),
    "white mug": ("white porcelain mug", "plain porcelain coffee cup"),
    "yellow and white mug": (
        "yellow patterned coffee cup",
        "two tone yellow cup",
    ),
    "moka pot": ("metal espresso maker", "italian coffee maker"),
    "book": ("hardcover book", "closed book", "standing book"),
    "frying pan": ("skillet", "fry pan"),
    "tray": ("wooden serving tray", "serving tray"),
    "caddy": ("desk caddy", "wooden desk organizer"),
    "wine bottle": ("green wine bottle", "glass wine bottle"),
    "wine rack": ("wooden wine rack", "wine bottle rack"),
    "rack": ("small wooden rack", "dish rack"),
    "shelf": ("wooden shelf", "small wooden shelf"),
    "cabinet shelf": ("wooden two layer shelf", "open cabinet shelf"),
    "microwave": ("microwave oven",),
    "cabinet": ("wooden drawer cabinet",),
    "wooden cabinet": ("wooden drawer cabinet", "cabinet with drawers"),
    "cream cheese box": ("cream cheese package", "cream cheese"),
    "drawer": ("wooden cabinet drawer",),
}


def _neutral_gallery_queries(queries: Sequence[str]) -> list[str]:
    """Expand a neutral classification request to a meaningful label set.

    Gallery probabilities are conditional on ``allowed_labels``.  Asking for
    only ``black bowl`` therefore assigns every component a vacuous bowl
    probability of one, including a visible ramekin.  Whenever a Spatial
    gallery class is requested, score all four public Spatial assets together
    so argmax agreement is genuine.  Object-suite requests retain their full
    eleven-way gallery for the same reason.
    """

    normalized = list(dict.fromkeys(normalize_label(query) for query in queries))
    # DINO-gated labels are removed from the neutral request first.  Otherwise
    # a large fixture such as ``microwave`` would be the only allowed gallery
    # class, causing every tiny tabletop component to be completed to a
    # microwave-sized obstacle before the detector even runs.
    expanded = {
        query for query in normalized if query not in _DINO_GALLERY_LABELS
    }
    if _uses_object_gallery(normalized):
        expanded.update(LIBERO_OBJECT_LABELS)
        expanded.add("basket")
    if any(query in _SPATIAL_GALLERY_LABELS for query in normalized):
        expanded.update(_SPATIAL_GALLERY_LABELS)
    if expanded == set(normalized):
        return normalized
    return sorted(expanded)


@dataclass(frozen=True)
class _DINO2DProvenance:
    """One raw same-camera DINO box and its optional RGB-D fit.

    ``instance`` remains ``None`` when depth extraction or the static size
    gate rejects the box.  Keeping those raw fixture boxes is intentional: a
    large container can be visually grounded even when its depth crop is too
    mixed to fit a reliable 3-D OBB.
    """

    canonical_query: str
    detection: BoxDetection
    frame_width: int
    frame_height: int
    instance: ObjectInstance | None
    surface_points_world: np.ndarray | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.frame_width <= 0 or self.frame_height <= 0:
            raise ValueError("DINO provenance frame dimensions must be positive")
        object.__setattr__(self, "canonical_query", normalize_label(self.canonical_query))
        points = None
        if self.surface_points_world is not None:
            points = np.asarray(self.surface_points_world, dtype=np.float64)
            if (
                self.instance is None
                or points.ndim != 2
                or points.shape[1:] != (3,)
                or len(points) < 8
                or not np.all(np.isfinite(points))
            ):
                raise ValueError(
                    "DINO surface points require an accepted instance and a "
                    "finite Nx3 RGB-D component"
                )
            points = points.copy()
        object.__setattr__(self, "surface_points_world", points)


def _expanded_dino_queries(queries: Sequence[str]) -> tuple[list[str], dict[str, str]]:
    """Expand frozen text aliases and return alias->canonical provenance."""

    expanded: list[str] = []
    canonical_for: dict[str, str] = {}
    for raw in queries:
        canonical = normalize_label(raw)
        aliases = list(_DINO_QUERY_ALIASES.get(canonical, ()))
        # The task compiler intentionally keeps only the relational anchor
        # (for example ``top drawer``).  Grounding-DINO produces materially
        # tighter fixture boxes when the containing noun is present.  Expand
        # every bare drawer level in the same language-only way; the resulting
        # box must still pass the sensor geometry gate below.
        if "drawer" in canonical and "cabinet" not in canonical:
            aliases.append(f"{canonical} of the wooden cabinet")
        for query in (canonical, *aliases):
            normalized = normalize_label(query)
            if normalized not in canonical_for:
                expanded.append(normalized)
                canonical_for[normalized] = canonical
    return expanded, canonical_for


def _requires_all_view_spatial_search(queries: Sequence[str]) -> bool:
    """Small manipulanda must be grounded in both calibrated RGB-D views.

    A single false agent-view proposal otherwise suppresses the wrist query
    before the independent gallery gate can reject it.  Frozen per-episode
    caching keeps this two-view search to the first localization attempt.
    """

    return any(
        normalize_label(query) in _STRICT_DINO_GALLERY_LABELS
        for query in queries
    )


def _all_view_spatial_queries(queries: Sequence[str]) -> tuple[str, ...]:
    """Return only labels that need independent grounding in every view.

    ``_dino_instances`` accepts compound prompts.  Treating the old
    all-view decision as one batch-wide boolean meant that asking for a small
    bowl also re-grounded a large static stove in the moving wrist camera.
    Keep the existing strict-label policy, but apply it to each canonical
    query rather than to every label that happens to share its prompt.
    """

    return tuple(
        dict.fromkeys(
            normalized
            for query in queries
            if ((normalized := normalize_label(query)) in _STRICT_DINO_GALLERY_LABELS
                or "drawer" in normalized.split())
        )
    )


def _gallery_planar_size_agrees(
    observed_extents_m: np.ndarray,
    expected_planar_m: float,
    *,
    minimum_ratio: float = 0.45,
    maximum_ratio: float = 2.6,
    maximum_absolute_m: float = 0.16,
) -> bool:
    """Reject tiny edge fragments and oversized supports before semantics."""

    observed_planar = float(np.max(np.asarray(observed_extents_m)[:2]))
    expected = float(expected_planar_m)
    return bool(
        minimum_ratio * expected <= observed_planar
        <= max(maximum_absolute_m, maximum_ratio * expected)
    )


def _gallery_label_candidates(
    instances: Sequence[ObjectInstance],
    query: str,
    *,
    minimum_score: float = 0.04,
    relative_score: float = 0.72,
    require_argmax: bool = False,
) -> list[ObjectInstance]:
    """Return static-gallery candidates whose best label agrees with ``query``.

    Spatial contains several similarly sized round objects.  Grounding-DINO
    reliably proposes every black bowl but can also call the white ramekin a
    bowl, while its text query for ``ramekin`` is unreliable.  This gate uses
    only the visible RGB-D component and the frozen public asset gallery; it
    never consults simulator ids, object poses, or task predicates.
    """

    normalized = normalize_label(query)
    candidates: list[ObjectInstance] = []
    for instance in instances:
        score = instance.score_for(normalized)
        top_score = max(instance.label_scores.values(), default=0.0)
        if (
            score >= minimum_score
            and score >= relative_score * top_score
            and (not require_argmax or score >= top_score - 1e-12)
        ):
            candidates.append(instance)
    return candidates


@dataclass(frozen=True)
class _GalleryFamilyEvidence:
    """Independent multi-class evidence for one DINO-grounded RGB-D crop."""

    prior: GalleryQueryPrior
    scores: Mapping[str, float]
    top_label: str
    best_accepted_label: str
    accepted_score: float
    shape_similarity: float


def _gallery_family_evidence(
    instance: ObjectInstance,
    query: str,
    classifier: TextureSizeGallery | Any | None,
) -> _GalleryFamilyEvidence | None:
    """Compare a crop with every member of its meaningful public-asset family.

    A partial gallery is rejected instead of silently turning the comparison
    into a one-class probability of one.  The caller still owns the policy for
    strict manipulanda versus partially visible fixtures.
    """

    normalized = normalize_label(query)
    prior = GALLERY_QUERY_PRIORS.get(normalized)
    prototypes = (
        getattr(classifier, "prototypes", {})
        if classifier is not None
        else {}
    )
    if prior is None or instance.color_histogram is None:
        return None
    if not set(prior.family) <= set(prototypes):
        return None
    try:
        classification = classifier.classify(instance, allowed_labels=prior.family)
    except (KeyError, ValueError):
        return None
    scores = {
        normalize_label(label): float(score)
        for label, score in classification.scores.items()
    }
    if set(scores) != set(prior.family):
        return None
    top_label = max(scores, key=scores.__getitem__)
    best_accepted = max(prior.accepted, key=lambda label: scores[label])
    prototype = prototypes[best_accepted]
    shape = float(
        classifier._shape_similarity(
            instance.observed.extents_m,
            prototype.dimensions_xyz_m,
        )
    )
    return _GalleryFamilyEvidence(
        prior=prior,
        scores=scores,
        top_label=top_label,
        best_accepted_label=best_accepted,
        accepted_score=float(sum(scores[label] for label in prior.accepted)),
        shape_similarity=shape,
    )


def _family_evidence_accepts(
    evidence: _GalleryFamilyEvidence,
    observed_extents_m: np.ndarray,
    classifier: TextureSizeGallery | Any,
    *,
    minimum_score: float = 0.16,
    relative_score: float = 0.72,
    minimum_shape_similarity: float = 0.30,
) -> bool:
    """Conservative semantic/size check after (never before) DINO grounding."""

    top_score = max(evidence.scores.values(), default=0.0)
    prototype = classifier.prototypes[evidence.best_accepted_label]
    expected_planar = float(np.max(prototype.dimensions_xyz_m[:2]))
    return bool(
        evidence.accepted_score >= minimum_score
        and evidence.accepted_score >= relative_score * top_score
        and evidence.shape_similarity >= minimum_shape_similarity
        and _gallery_planar_size_agrees(
            observed_extents_m,
            expected_planar,
            minimum_ratio=0.30,
            maximum_ratio=2.5,
            maximum_absolute_m=max(0.16, 1.25 * expected_planar),
        )
    )


def _gallery_prototype_for_query(
    query: str,
    instance: ObjectInstance,
    classifier: TextureSizeGallery | Any | None,
) -> tuple[Any | None, _GalleryFamilyEvidence | None]:
    """Choose the best valid public asset variant for a grounded query."""

    if classifier is None:
        return None, None
    normalized = normalize_label(query)
    evidence = _gallery_family_evidence(instance, normalized, classifier)
    if evidence is not None:
        return classifier.prototypes[evidence.best_accepted_label], evidence
    prior = GALLERY_QUERY_PRIORS.get(normalized)
    prototype_label = prior.primary if prior is not None else normalized
    return getattr(classifier, "prototypes", {}).get(prototype_label), None


def _spatial_dino_gallery_gate(
    query: str,
    detections: Sequence[ObjectInstance],
    neutral_instances: Sequence[ObjectInstance],
    *,
    workspace_min: np.ndarray,
    workspace_max: np.ndarray,
    classifier: TextureSizeGallery | None = None,
    association_radius_m: float = 0.09,
    allow_bowl_cavity_merge: bool = False,
) -> list[ObjectInstance]:
    """Reject DINO class confusions with an independent RGB-D asset prior.

    DINO geometry and connected-component geometry have different partial-view
    centre biases, so association is deliberately looser than cross-camera
    fusion.  The radius is still smaller than the normal repeated-bowl spacing
    in Spatial.  An unmatched detection is retained only for labels outside the
    static Spatial gallery; callers may explicitly bypass this gate after an
    object has left the support surface.
    """

    normalized = normalize_label(query)
    lower = np.asarray(workspace_min, dtype=np.float64)
    upper = np.asarray(workspace_max, dtype=np.float64)
    in_workspace = [
        item
        for item in detections
        if np.all(item.center_world >= lower) and np.all(item.center_world <= upper)
    ]
    if normalized not in _DINO_GALLERY_LABELS:
        return in_workspace
    if normalized not in _SPATIAL_GALLERY_LABELS:
        # Large receptacles/appliances are frequently visible as one face or
        # rim rather than a complete 3-D component.  Their public-asset prior
        # is therefore applied as confidence/size evidence in
        # ``_dino_instances`` and ``_fixture_size_gate``; hard family argmax
        # here would delete correctly grounded partial fixtures.
        if normalized in _LARGE_FIXTURE_LABELS:
            return in_workspace
        if normalized not in _STRICT_DINO_GALLERY_LABELS:
            return in_workspace
        accepted: list[ObjectInstance] = []
        for detection in in_workspace:
            evidence = _gallery_family_evidence(detection, normalized, classifier)
            if evidence is not None and _family_evidence_accepts(
                evidence,
                detection.observed.extents_m,
                classifier,
            ):
                accepted.append(detection)
                continue
            # A separately segmented support-surface crop may be cleaner than
            # a DINO box.  It still has to carry a complete multi-class family
            # score vector and agree spatially with the grounded detection.
            for neutral in neutral_instances:
                if (
                    np.linalg.norm(
                        detection.center_world[:2] - neutral.center_world[:2]
                    )
                    > association_radius_m
                ):
                    continue
                support_evidence = _gallery_family_evidence(
                    neutral,
                    normalized,
                    classifier,
                )
                if support_evidence is not None and _family_evidence_accepts(
                    support_evidence,
                    neutral.observed.extents_m,
                    classifier,
                ):
                    accepted.append(detection)
                    break
        return accepted
    require_argmax = normalized in {"black bowl", "ramekin"}
    gallery_candidates = _gallery_label_candidates(
        neutral_instances,
        normalized,
        require_argmax=require_argmax,
    )
    accepted: list[ObjectInstance] = []
    for detection in in_workspace:
        prototype = getattr(classifier, "prototypes", {}).get(normalized)
        if normalized == "black bowl" and prototype is not None:
            # Shape completion can turn a large dark cabinet crop into a
            # canonical-sized bowl. Rank selectors must check the measured
            # footprint before that completion, or FRONT can select the
            # cabinet instead of one of the three actual bowls.
            diameter = float(np.max(prototype.dimensions_xyz_m[:2]))
            measured_xy = (detection.observed.bounds_max_world
                           - detection.observed.bounds_min_world)[:2]
            shallow_cavity_merge = bool(
                allow_bowl_cavity_merge
                and detection.observed.bounds_max_world[2] - detection.observed.bounds_min_world[2]
                <= 1.60 * float(prototype.dimensions_xyz_m[2])
            )
            if (max(measured_xy) > 1.60 * diameter and min(measured_xy) > 1.25 * diameter
                    and not shallow_cavity_merge):
                continue
        support_agrees = any(
            np.linalg.norm(detection.center_world[:2] - candidate.center_world[:2])
            <= association_radius_m
            for candidate in gallery_candidates
        )
        # Spatial also places bowls on a cookie box, stove, cabinet, or inside
        # an open drawer.  Such instances are correctly absent from the
        # tabletop support components.  Classify the DINO crop itself against
        # the same frozen gallery so elevation never becomes an implicit task
        # or coordinate prior.
        crop_agrees = False
        if classifier is not None and detection.color_histogram is not None:
            classification = classifier.classify(
                detection,
                allowed_labels=sorted(_SPATIAL_GALLERY_LABELS),
            )
            crop_score = classification.scores.get(normalized, 0.0)
            crop_top_score = max(classification.scores.values(), default=0.0)
            crop_agrees = (
                crop_score >= 0.04
                and crop_score >= 0.72 * crop_top_score
                and (not require_argmax or crop_score >= crop_top_score - 1e-12)
            )
        if support_agrees or crop_agrees:
            accepted.append(detection)
    return accepted


def _independent_spatial_gallery_score(
    instance: ObjectInstance,
    query: str,
    classifier: TextureSizeGallery | Any | None,
    *,
    minimum_confidence: float = 0.55,
    minimum_shape_similarity: float = 0.52,
) -> float | None:
    """Validate a DINO crop against the complete, independent asset gallery."""

    normalized = normalize_label(query)
    prototypes = getattr(classifier, "prototypes", {}) if classifier is not None else {}
    prototype = prototypes.get(normalized)
    if (
        normalized not in _SPATIAL_GALLERY_LABELS
        or prototype is None
        or instance.color_histogram is None
    ):
        return None
    try:
        classification = classifier.classify(
            instance,
            allowed_labels=sorted(_SPATIAL_GALLERY_LABELS),
        )
    except (KeyError, ValueError):
        # A partial gallery is not independent evidence for an argmax.
        return None
    scores = {
        normalize_label(label): float(score)
        for label, score in classification.scores.items()
    }
    if set(scores) != set(_SPATIAL_GALLERY_LABELS):
        return None
    top_label = max(scores, key=scores.__getitem__)
    score = scores.get(normalized, 0.0)
    shape = classifier._shape_similarity(
        instance.observed.extents_m,
        prototype.dimensions_xyz_m,
    )
    expected_planar = float(np.max(prototype.dimensions_xyz_m[:2]))
    observed_planar = float(np.max(instance.observed.extents_m[:2]))
    if not 0.45 * expected_planar <= observed_planar <= 1.9 * expected_planar:
        return None
    if (
        top_label != normalized
        or score < minimum_confidence
        or shape < minimum_shape_similarity
    ):
        return None
    return score


def _strict_in_source_from_2d(
    provenance: Sequence[_DINO2DProvenance],
    subject: str,
    reference: str,
    classifier: TextureSizeGallery | Any | None,
    *,
    workspace_min: np.ndarray,
    workspace_max: np.ndarray,
    minimum_fixture_source_area_ratio: float = 6.0,
    maximum_fixture_frame_area_fraction: float = 0.45,
    minimum_source_detection_score: float = 0.25,
    minimum_fixture_detection_score: float = 0.25,
    source_cluster_radius_m: float = 0.045,
) -> ObjectInstance | None:
    """Recover only the source of an ``IN`` relation from nested 2-D boxes.

    This is deliberately weaker than fabricating a 3-D fixture: the returned
    object has an accepted RGB-D fit and a four-way gallery argmax, while the
    container contributes only a same-camera semantic box.  The caller must
    leave its 3-D selector reference unset.  Full-view fixture boxes and weak
    overlaps, low-score boxes, and multiple distinct nested sources are
    rejected, and normal 3-D relation resolution (including its same-proposal
    gate) always runs first.
    """

    normalized_subject = normalize_label(subject)
    normalized_reference = normalize_label(reference)
    if (
        normalized_subject not in _SPATIAL_GALLERY_LABELS
        or not any(token in normalized_reference for token in _IN_FIXTURE_TOKENS)
        or minimum_fixture_source_area_ratio <= 1.0
        or not 0.0 < maximum_fixture_frame_area_fraction < 1.0
        or not 0.0 <= minimum_source_detection_score <= 1.0
        or not 0.0 <= minimum_fixture_detection_score <= 1.0
        or source_cluster_radius_m <= 0.0
    ):
        return None
    lower = np.asarray(workspace_min, dtype=np.float64)
    upper = np.asarray(workspace_max, dtype=np.float64)
    ranked: list[tuple[float, float, float, str, ObjectInstance]] = []
    sources = [
        item
        for item in provenance
        if item.canonical_query == normalized_subject
        and item.instance is not None
        and float(item.detection.score) >= minimum_source_detection_score
    ]
    fixtures = [
        item
        for item in provenance
        if item.canonical_query == normalized_reference
        and float(item.detection.score) >= minimum_fixture_detection_score
    ]
    for source in sources:
        instance = source.instance
        assert instance is not None
        if not (np.all(instance.center_world >= lower) and np.all(instance.center_world <= upper)):
            continue
        gallery_score = _independent_spatial_gallery_score(
            instance,
            normalized_subject,
            classifier,
        )
        if gallery_score is None:
            continue
        source_box = source.detection.xyxy
        source_area = float(
            (source_box[2] - source_box[0]) * (source_box[3] - source_box[1])
        )
        for fixture in fixtures:
            if (
                fixture.detection.camera_name != source.detection.camera_name
                or fixture.frame_width != source.frame_width
                or fixture.frame_height != source.frame_height
            ):
                continue
            fixture_box = fixture.detection.xyxy
            fully_contains = bool(
                source_box[0] >= fixture_box[0]
                and source_box[1] >= fixture_box[1]
                and source_box[2] <= fixture_box[2]
                and source_box[3] <= fixture_box[3]
            )
            if not fully_contains:
                continue
            fixture_area = float(
                (fixture_box[2] - fixture_box[0])
                * (fixture_box[3] - fixture_box[1])
            )
            frame_area = float(fixture.frame_width * fixture.frame_height)
            area_ratio = fixture_area / max(source_area, 1e-9)
            if (
                area_ratio < minimum_fixture_source_area_ratio
                or fixture_area / frame_area > maximum_fixture_frame_area_fraction
            ):
                continue
            ranked.append(
                (
                    -gallery_score,
                    -float(source.detection.score),
                    -float(fixture.detection.score),
                    instance.instance_id,
                    instance,
                )
            )
    if not ranked:
        return None
    # A 2-D fixture box is deliberately not promoted to a fabricated 3-D OBB.
    # Consequently it is not strong enough to choose between two spatially
    # distinct, independently gallery-confirmed sources.  Cluster duplicate
    # DINO proposals in measured 3-D and accept the fallback only when every
    # valid nested pair refers to one physical source.
    source_clusters = fuse_instances_3d(
        tuple(item[4] for item in ranked),
        centre_radius_m=source_cluster_radius_m,
    )
    if len(source_clusters) != 1:
        return None
    return min(ranked, key=lambda item: item[:4])[4]


def _relabel_spatial_crop_proposals(
    proposals: Sequence[ObjectInstance],
    classifier: TextureSizeGallery | None,
    *,
    workspace_min: np.ndarray,
    workspace_max: np.ndarray,
    minimum_confidence: float = 0.55,
    minimum_shape_similarity: float = 0.52,
) -> list[ObjectInstance]:
    """Relabel any semantic proposal from its measured crop and static asset.

    Open-vocabulary proposal text is deliberately not treated as the final
    class.  In Spatial, DINO sometimes calls a black bowl in an open drawer
    ``top drawer`` or ``wooden cabinet``.  A high-confidence gallery match can
    recover it, but only inside the calibrated workspace and only when the raw
    depth dimensions agree with the public collision asset.
    """

    if classifier is None:
        return []
    lower = np.asarray(workspace_min, dtype=np.float64)
    upper = np.asarray(workspace_max, dtype=np.float64)
    relabelled: list[ObjectInstance] = []
    for proposal in proposals:
        if proposal.color_histogram is None or not (
            np.all(proposal.center_world >= lower) and np.all(proposal.center_world <= upper)
        ):
            continue
        classification = classifier.classify(
            proposal,
            allowed_labels=sorted(_SPATIAL_GALLERY_LABELS),
        )
        label = normalize_label(classification.label)
        confidence = float(classification.scores.get(label, 0.0))
        prototype = classifier.prototypes[label]
        shape = classifier._shape_similarity(
            proposal.observed.extents_m,
            prototype.dimensions_xyz_m,
        )
        expected_planar = float(np.max(prototype.dimensions_xyz_m[:2]))
        observed_planar = float(np.max(proposal.observed.extents_m[:2]))
        size_agrees = 0.45 * expected_planar <= observed_planar <= 1.9 * expected_planar
        if (
            confidence < minimum_confidence
            or shape < minimum_shape_similarity
            or not size_agrees
        ):
            continue
        dimensions = classifier.orient_dimensions(
            prototype.dimensions_xyz_m,
            proposal.observed.extents_m,
        )
        grasp = proposal.center_world.copy()
        grasp[2] += dimensions[2] / 2.0
        relabelled.append(
            proposal.with_semantics(
                {label: confidence},
                grasp_point_world=grasp,
                extents_m=dimensions,
                confidence=confidence,
            )
        )
    return list(fuse_instances_3d(relabelled))


def _drawer_handle_consistent_candidates(
    query: str, candidates: Sequence[ObjectInstance], frames: Sequence[RGBDFrame]
) -> list[ObjectInstance]:
    """Disambiguate drawer apertures from the cabinet top using visible bars."""
    from anchor.common import CameraCalibration, CameraFrame
    from anchor.contact.detectors import DrawerHandleDetector

    levels = [level for level in ("top", "middle", "bottom") if level in query.split()]
    levels = levels or ["top", "middle", "bottom"]
    detector = DrawerHandleDetector()
    handles = []
    for frame in frames:
        camera = CameraFrame(frame.rgb, frame.depth_m, CameraCalibration(
            frame.name, frame.width, frame.height, frame.intrinsics,
            frame.world_from_camera, frame.observation_v_flipped))
        for level in levels:
            # A lone bar/roof pair can supply a contact target, but does not
            # establish which side of a partial aperture its front edge is.
            # Use the stronger multi-handle geometry to reject cavity boxes.
            handle = detector._detect_frame_global_triplet(
                camera, level, allow_roof_fallback=False,
            )
            if handle is not None:
                handles.append(handle)
        if handles:
            break
    if not handles:
        return list(candidates)
    accepted = []
    for instance in candidates:
        observed = instance.observed
        for handle in handles:
            center_delta = observed.center_world - handle.point_world
            half_along_normal = .5 * float(
                np.abs(handle.outward_world @ observed.axes_world) @ observed.extents_m)
            front_delta = float(center_delta @ handle.outward_world) + half_along_normal
            if (observed.bounds_min_world[2] <= handle.point_world[2] + .005
                    and observed.bounds_max_world[2] >= handle.point_world[2] - .070
                    and abs(float(center_delta @ handle.feature_axis_world)) <= .14
                    and -.055 <= front_delta <= .070):
                accepted.append(instance)
                break
    return accepted


def _fixture_size_gate(query: str, candidates: Sequence[ObjectInstance]) -> list[ObjectInstance]:
    """Reject object-sized, full-view, and edge crops for fixture queries.

    Grounding-DINO often returns both a useful drawer opening and boxes spanning
    the complete cabinet or wrist image.  A minimum-only gate lets the latter
    contain almost every source candidate, making an ``IN`` selector meaningless.
    The limits below describe a visible drawer-like aperture (two substantial
    planar sides and a shallower vertical span), not a benchmark location.
    """

    normalized = normalize_label(query)
    if "drawer" in normalized:
        accepted: list[ObjectInstance] = []
        for item in candidates:
            planar = np.sort(np.asarray(item.observed.extents_m[:2], dtype=np.float64))
            vertical = float(
                item.observed.bounds_max_world[2]
                - item.observed.bounds_min_world[2]
            )
            if (
                planar[1] >= 0.14
                and planar[0] >= 0.080
                and planar[1] <= 0.45
                and float(np.prod(planar)) <= 0.14
                and vertical <= 0.16
            ):
                accepted.append(item)
        return accepted
    minimum_by_label = {
        # Lower bounds admit a single sensor-visible panel/rim.  Full public
        # collision AABBs range from 0.19 m (stove) to 0.434 m (caddy), so none
        # of these paths pass through the neutral 0.28-m tabletop-object cap.
        "cabinet": 0.11,
        "wooden cabinet": 0.11,
        "cabinet shelf": 0.10,
        "caddy": 0.10,
        "microwave": 0.12,
        "rack": 0.09,
        "shelf": 0.10,
        "stove": 0.10,
        "tray": 0.10,
        "wine rack": 0.10,
    }
    minimum_planar = minimum_by_label.get(normalized)
    if minimum_planar is None:
        if any(token in normalized for token in ("cabinet", "stove", "microwave", "rack", "shelf")):
            minimum_planar = 0.10
        else:
            return list(candidates)
    if minimum_planar <= 0.0:  # pragma: no cover - defensive static-table invariant
        return list(candidates)
    return [
        item
        for item in candidates
        if float(np.max(item.observed.extents_m[:2])) >= minimum_planar
    ]


def _exclude_small_reference_components(
    sources: Sequence[ObjectInstance],
    references: Mapping[str, ObjectInstance],
    subject: str,
) -> list[ObjectInstance]:
    """Prevent one physical crop from filling two distinct small-object roles.

    In ``black bowl NEXT_TO ramekin`` a ramekin crop can carry a weaker bowl
    hypothesis.  A zero-distance selector would otherwise choose that same
    component as both source and reference.  Fixture relations intentionally
    remain eligible for cross-query anchors (for example a bowl crop called
    ``top drawer``), so only distinct frozen-gallery object classes are
    excluded here.
    """

    normalized_subject = normalize_label(subject)
    small_references = [
        reference
        for label, reference in references.items()
        if normalize_label(label) in _SPATIAL_GALLERY_LABELS
        and normalize_label(label) != normalized_subject
    ]
    if not small_references:
        return list(sources)

    def same_component(source: ObjectInstance, reference: ObjectInstance) -> bool:
        intersection_min = np.maximum(
            source.observed.bounds_min_world,
            reference.observed.bounds_min_world,
        )
        intersection_max = np.minimum(
            source.observed.bounds_max_world,
            reference.observed.bounds_max_world,
        )
        intersection = np.maximum(intersection_max - intersection_min, 0.0)
        intersection_volume = float(np.prod(intersection))
        source_volume = float(np.prod(np.maximum(source.observed.extents_m, 1e-6)))
        reference_volume = float(np.prod(np.maximum(reference.observed.extents_m, 1e-6)))
        overlap = intersection_volume / max(min(source_volume, reference_volume), 1e-12)
        distance = float(np.linalg.norm(source.center_world - reference.center_world))
        return overlap >= 0.35 or distance <= 0.025

    return [
        source
        for source in sources
        if not any(same_component(source, reference) for reference in small_references)
    ]


def _same_geometric_proposal(
    source: ObjectInstance,
    reference: ObjectInstance,
) -> bool:
    """Identify a cross-query clone without rejecting genuinely stacked objects."""

    if float(np.linalg.norm(source.center_world - reference.center_world)) > 0.025:
        return False
    source_extent = np.maximum(source.observed.extents_m, 1e-6)
    reference_extent = np.maximum(reference.observed.extents_m, 1e-6)
    extent_similarity = np.minimum(source_extent, reference_extent) / np.maximum(
        source_extent,
        reference_extent,
    )
    # An exact DINO crop relabelled for two queries has the same measured
    # dimensions.  A bowl genuinely stacked on a small support can have a
    # nearby centre, but its round footprint and height differ substantially
    # from the thin cookie-box strip beneath it.
    return bool(np.all(extent_similarity >= 0.65))


def _support_relation_pair_metrics(
    source: ObjectInstance,
    reference: ObjectInstance,
    relation: str,
    *,
    planar_margin_m: float = 0.035,
    vertical_margin_m: float = 0.050,
) -> tuple[bool, float, float, float, float]:
    """Score one sensor-measured source/support pair for an ON/IN selector."""

    normalized_relation = normalize_label(relation)
    if normalized_relation not in {"on", "in"}:
        raise ValueError("joint support ranking requires an ON or IN relation")
    # Relation contact geometry must stay in the raw RGB-D OBB.  ``extents_m``
    # may carry a canonical gallery size and cross-proposal fusion takes a
    # component-wise maximum; with rotated/relabelled crops that can inflate a
    # 50-mm bowl into a 107-mm semantic cube.  The observed OBB is the direct
    # sensor measurement and therefore owns all continuous relation tests.
    local = (
        source.center_world - reference.center_world
    ) @ reference.observed.axes_world
    reference_half = reference.observed.extents_m / 2.0
    planar_limit = reference_half[:2] + planar_margin_m
    planar_valid = not np.any(np.abs(local[:2]) > planar_limit)
    normalized_planar = float(
        np.linalg.norm(local[:2] / np.maximum(reference_half[:2], 0.020))
    )
    source_half_z = float(
        np.abs(source.observed.axes_world[2])
        @ (source.observed.extents_m / 2.0)
    )
    reference_half_z = float(
        np.abs(reference.observed.axes_world[2])
        @ (reference.observed.extents_m / 2.0)
    )
    if normalized_relation == "in":
        # IN is containment, not merely overlap.  The previous expression
        # added the source half-height, which admitted an object resting above
        # a shallow drawer whenever its XY happened to be close to the box
        # centre.  Require the source envelope to fit inside the measured
        # vertical span, with the same tolerance for partial RGB-D surfaces.
        vertical_residual = max(
            abs(float(local[2]))
            + source_half_z
            - reference_half_z
            - vertical_margin_m,
            0.0,
        )
        vertical_valid = vertical_residual <= 1e-12
    else:
        source_bottom = float(source.center_world[2] - source_half_z)
        reference_top = float(reference.center_world[2] + reference_half_z)
        vertical_residual = abs(source_bottom - reference_top)
        vertical_valid = vertical_residual <= 0.080
    area = float(np.prod(reference.observed.extents_m[:2]))
    cost = vertical_residual + 0.020 * normalized_planar
    return (
        bool(planar_valid and vertical_valid),
        cost,
        vertical_residual,
        normalized_planar,
        area,
    )


def _resolve_support_relation_joint(
    sources: Sequence[ObjectInstance],
    references: Sequence[ObjectInstance],
    relation: str,
) -> tuple[ObjectInstance, ObjectInstance]:
    """Jointly choose source and support instead of preselecting a text box.

    Open-vocabulary confidence is proposal evidence, not object identity.  A
    sprawling false support box can have a higher DINO score than the partly
    occluded true support.  Rank every geometrically valid source/reference
    pair first, then use measured support area and confidence only as stable
    tie-breakers.  No simulator identity, task index, or evaluator predicate is
    available to this function.
    """

    ranked: list[
        tuple[
            float,
            float,
            float,
            float,
            str,
            str,
            ObjectInstance,
            ObjectInstance,
        ]
    ] = []
    for source in sources:
        for reference in references:
            if _same_geometric_proposal(source, reference):
                continue
            valid, cost, _vertical, _planar, area = _support_relation_pair_metrics(
                source,
                reference,
                relation,
            )
            if not valid:
                continue
            ranked.append(
                (
                    cost,
                    area,
                    -source.confidence,
                    -reference.confidence,
                    source.instance_id,
                    reference.instance_id,
                    source,
                    reference,
                )
            )
    if not ranked:
        raise LookupError("no sensor source/support pair satisfies selector")
    selected = min(ranked, key=lambda item: item[:6])
    return selected[6], selected[7]


def _ambiguous_fixture_fallback(
    query: str,
    raw_candidates: Sequence[ObjectInstance],
) -> list[ObjectInstance]:
    """Retain a bowl-sized semantic anchor only for a drawer relation.

    A compound DINO prompt can use the visible bowl crop for ``top drawer``
    while omitting the drawer box.  That same crop is useful as a relational
    anchor after gallery relabelling.  Treating it as a cabinet, stove, or rack
    would instead make one dark crop both the source and a fabricated large
    fixture, so those queries deliberately have no ambiguous fallback.
    """

    if "drawer" not in normalize_label(query) or not raw_candidates:
        return []
    # This fallback exists only for a tight bowl-sized crop whose text was
    # ambiguously assigned to the containing drawer.  Never use it to undo the
    # fixture gate for a rejected full-frame or long edge proposal.
    bowl_sized = [
        item
        for item in raw_candidates
        if 0.045 <= float(np.max(item.observed.extents_m[:2])) < 0.14
        and float(np.min(item.observed.extents_m[:2])) >= 0.035
        and float(
            item.observed.bounds_max_world[2]
            - item.observed.bounds_min_world[2]
        )
        <= 0.13
    ]
    if not bowl_sized:
        return []
    return [max(bowl_sized, key=lambda item: item.score_for(query))]


def _upright_stack_dimensions(
    classifier: TextureSizeGallery | Any | None,
    label: str,
) -> np.ndarray | None:
    """Return gravity-aligned dimensions for a known stackable asset.

    These four Spatial assets are placed on a support face: their thinnest
    public collision-box dimension is vertical.  Keeping this tiny helper
    separate avoids applying that semantic prior to arbitrary HOPE packages,
    whose XML axes may legitimately be permuted when placed upright.
    """

    normalized = normalize_label(label)
    prototypes = getattr(classifier, "prototypes", {}) if classifier is not None else {}
    prototype = prototypes.get(normalized)
    if normalized not in _STACKABLE_GALLERY_LABELS or prototype is None:
        return None
    raw = np.asarray(prototype.dimensions_xyz_m, dtype=np.float64)
    planar = sorted(raw, reverse=True)[:2]
    return np.array((planar[0], planar[1], float(np.min(raw))), dtype=np.float64)


def _low_profile_fixture_dimensions(
    label: str,
    prototype_dimensions_m: np.ndarray,
    *,
    maximum_thickness_ratio: float = 0.55,
) -> np.ndarray | None:
    """Return gravity-aligned dimensions for a low, planar public fixture.

    Public articulated-fixture XML is authored in its deployed gravity frame.
    A partial or contaminated RGB-D crop must therefore not rotate a stove or
    tray's long planar side into world Z.  Shape, rather than a benchmark
    coordinate, identifies these supports: their shortest side is at most
    55 percent of the next-shortest side.  Drawer labels denote an articulated
    cavity part whose measured elevation is meaningful, not a freestanding
    support resting on the dominant table, so they deliberately stay on the
    ordinary fixture path.
    """

    normalized = normalize_label(label)
    raw = np.asarray(prototype_dimensions_m, dtype=np.float64)
    if (
        normalized not in _LARGE_FIXTURE_LABELS
        or "drawer" in normalized
        or raw.shape != (3,)
        or not np.all(np.isfinite(raw))
        or np.any(raw <= 0.0)
        or not 0.0 < maximum_thickness_ratio < 1.0
    ):
        return None
    planar = np.sort(raw[:2])[::-1]
    height = float(raw[2])
    if height > float(planar[1]) or height / float(planar[1]) > maximum_thickness_ratio:
        return None
    # The observed upright OBB stores its major/minor planar axes first.  Put
    # the two public planar dimensions in that same order; this permits only
    # an XY exchange under scene yaw and fixes the shortest public axis to Z.
    return np.array((planar[0], planar[1], height), dtype=np.float64)


def _low_profile_fixture_observation_agrees(
    instance: ObjectInstance,
    gravity_dimensions_m: np.ndarray,
    *,
    table_height_m: float,
    maximum_base_gap_heights: float = 0.75,
    maximum_vertical_span_heights: float = 1.50,
    minimum_footprint_fraction: float = 0.45,
    maximum_major_footprint_fraction: float = 1.50,
) -> bool:
    """Validate a visible flat-fixture crop before static completion.

    Every tolerance is dimensionless and scales with the public collision
    prior.  The gate requires a table-supported base, a shallow visible span,
    and evidence for both footprint axes.  It rejects a high robot/cabinet
    cluster even when its longest side happens to resemble a stove, while
    retaining a partially occluded but genuinely two-dimensional support.
    """

    dimensions = np.asarray(gravity_dimensions_m, dtype=np.float64)
    if dimensions.shape != (3,) or np.any(dimensions <= 0.0):
        return False
    if min(
        maximum_base_gap_heights,
        maximum_vertical_span_heights,
        minimum_footprint_fraction,
        maximum_major_footprint_fraction,
    ) <= 0.0:
        raise ValueError("flat-fixture observation ratios must be positive")
    observed_planar = np.sort(
        np.asarray(instance.observed.extents_m[:2], dtype=np.float64)
    )[::-1]
    expected_planar = np.sort(dimensions[:2])[::-1]
    observed_bottom = float(instance.observed.bounds_min_world[2])
    observed_top = float(instance.observed.bounds_max_world[2])
    observed_vertical_span = observed_top - observed_bottom
    base_gap = abs(observed_bottom - float(table_height_m))
    prior_height = float(dimensions[2])
    return bool(
        base_gap <= maximum_base_gap_heights * prior_height
        and observed_vertical_span
        <= maximum_vertical_span_heights * prior_height
        and np.all(
            observed_planar
            >= minimum_footprint_fraction * expected_planar
        )
        and np.all(observed_planar
                   <= maximum_major_footprint_fraction * expected_planar)
    )


_STOVE_CONFUSION_FAMILY = ("frying pan", "tray", "stove", "plate")


def _stove_fixture_semantics_agree(
    instance: ObjectInstance,
    classifier: TextureSizeGallery | Any | None,
    *,
    minimum_stove_score: float = 0.16,
    relative_score: float = 0.50,
    minimum_pan_margin: float = 0.02,
) -> bool:
    """Reject a pan body that DINO also calls a compact stove.

    A stove's visible burner can legitimately look more plate-like than the
    complete public stove asset, so requiring a stove-family argmax would
    delete good partial surfaces.  Instead, compare the DINO-grounded RGB-D
    crop against the public low-profile confusion family and require stove to
    retain at least half the best score while strictly beating the frying
    pan. A visible circular burner can give the plate prototype roughly twice
    the stove score even when the surrounding RGB-D footprint is square.
    The preceding fixture gate still owns table support and footprint
    geometry; this gate adds appearance/shape identity without coordinates or
    simulator state.

    Minimal external galleries that do not contain the complete family retain
    the historical geometry-only behaviour.  Once all four public prototypes
    are present, missing or malformed crop evidence fails closed.
    """

    if min(minimum_stove_score, relative_score, minimum_pan_margin) <= 0.0:
        raise ValueError("stove semantic thresholds must be positive")
    if relative_score > 1.0:
        raise ValueError("relative stove score cannot exceed one")
    prototypes = getattr(classifier, "prototypes", {}) if classifier is not None else {}
    family = set(_STOVE_CONFUSION_FAMILY)
    if not family <= set(prototypes):
        return True
    if instance.color_histogram is None:
        return False
    try:
        classification = classifier.classify(
            instance,
            allowed_labels=_STOVE_CONFUSION_FAMILY,
        )
        scores = {
            normalize_label(label): float(score)
            for label, score in classification.scores.items()
        }
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
    if set(scores) != family:
        return False
    values = np.asarray(tuple(scores.values()), dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        return False
    total = float(np.sum(values))
    if total <= 0.0:
        return False
    normalized_scores = {label: score / total for label, score in scores.items()}
    stove_score = normalized_scores["stove"]
    pan_score = normalized_scores["frying pan"]
    top_score = max(normalized_scores.values())
    return bool(
        stove_score >= minimum_stove_score
        and stove_score >= relative_score * top_score
        and stove_score >= pan_score + minimum_pan_margin
    )


def _complete_low_profile_fixture_geometry(
    instance: ObjectInstance,
    gravity_dimensions_m: np.ndarray,
    *,
    table_height_m: float,
) -> ObjectInstance:
    """Anchor an accepted flat fixture to its measured dominant support.

    The crop owns XY and the visible support top. The full public collision
    box can include a knob or a rim above that support; using its full height
    as the placement plane leaves objects hovering and rejects correct ON
    relations after they settle. The public thickness remains an upper bound.
    """

    dimensions = np.array(gravity_dimensions_m, dtype=np.float64, copy=True)
    if dimensions.shape != (3,) or np.any(dimensions <= 0.0):
        raise ValueError("flat-fixture dimensions must be a positive 3-vector")
    measured_height = float(instance.observed.bounds_max_world[2] - table_height_m)
    if measured_height > 0.0:
        dimensions[2] = min(dimensions[2], measured_height)
    center = instance.center_world.copy()
    center[:2] = instance.observed.center_world[:2]
    center[2] = float(table_height_m) + dimensions[2] / 2.0
    grasp = center.copy()
    grasp[2] = float(table_height_m) + dimensions[2]
    return instance.with_semantics(
        instance.label_scores,
        center_world=center,
        grasp_point_world=grasp,
        extents_m=dimensions,
    )


def _complete_static_detection_geometry(
    instance: ObjectInstance,
    dimensions: np.ndarray,
    *,
    table_height_m: float,
    prefer_observed_xy: bool = False,
) -> ObjectInstance:
    """Complete a DINO crop without projecting elevated objects to the table.

    ``BoxGeometryExtractor`` normally intersects the crop-centre ray with the
    support plane.  That is useful on the table, but at cabinet height it also
    changes X/Y by tens of centimetres.  Elevated crops already contain metric
    RGB-D points, so their measured centroid owns X/Y.  A component materially
    taller than the static asset is treated as the visible envelope of a stack;
    the queried top object is aligned to the measured top surface.
    """

    completed_dimensions = np.asarray(dimensions, dtype=np.float64)
    lower = instance.observed.bounds_min_world
    upper = instance.observed.bounds_max_world
    observed_height = float(upper[2] - lower[2])
    object_height = float(completed_dimensions[2])
    elevated = float(lower[2]) > float(table_height_m) + 0.020
    stacked_envelope = observed_height >= 1.35 * object_height
    center = instance.center_world.copy()
    if prefer_observed_xy or elevated or stacked_envelope:
        center[:2] = instance.observed.center_world[:2]
    if stacked_envelope:
        center[2] = float(upper[2]) - object_height / 2.0
    else:
        support_height = float(lower[2]) if elevated else float(table_height_m)
        center[2] = support_height + object_height / 2.0
    grasp = center.copy()
    grasp[2] += object_height / 2.0
    return instance.with_semantics(
        instance.label_scores,
        center_world=center,
        grasp_point_world=grasp,
        extents_m=completed_dimensions,
    )


def _complete_occluded_support_pairs(
    sources: Sequence[ObjectInstance],
    references: Sequence[ObjectInstance],
    source_label: str,
    reference_label: str,
    classifier: TextureSizeGallery | Any | None,
    *,
    maximum_height_error_m: float = 0.012,
) -> tuple[list[ObjectInstance], list[ObjectInstance]]:
    """Conservatively split a measured stacked envelope into two known assets.

    A bowl can completely hide a smaller ramekin or cookie box in both top-down
    cameras.  In that case text grounding has no independent support crop, but
    RGB-D still measures an envelope whose vertical span is the sum of the two
    frozen public collision heights.  This is relation-ready amodal completion:
    no simulator id, task number, absolute position, or evaluator predicate is
    consulted.  Existing valid source/support pairs always take precedence.
    """

    result_sources = list(sources)
    result_references = list(references)
    source_dimensions = _upright_stack_dimensions(classifier, source_label)
    reference_dimensions = _upright_stack_dimensions(classifier, reference_label)
    if source_dimensions is None or reference_dimensions is None:
        return result_sources, result_references
    if normalize_label(source_label) == normalize_label(reference_label):
        return result_sources, result_references
    for source in result_sources:
        if any(
            not _same_geometric_proposal(source, reference)
            and _support_relation_pair_metrics(source, reference, "on")[0]
            for reference in result_references
        ):
            return result_sources, result_references

    expected_height = float(source_dimensions[2] + reference_dimensions[2])
    completed_sources: list[ObjectInstance] = []
    inferred_references: list[ObjectInstance] = []
    for source in result_sources:
        lower = source.observed.bounds_min_world
        upper = source.observed.bounds_max_world
        measured_height = float(upper[2] - lower[2])
        planar = np.sort(np.asarray(source.observed.extents_m[:2], dtype=np.float64))
        expected_planar = np.sort(source_dimensions[:2])
        planar_ratio = planar / np.maximum(expected_planar, 1e-6)
        height_error = abs(measured_height - expected_height)
        stack_evidence = (
            measured_height
            >= float(source_dimensions[2] + 0.45 * reference_dimensions[2])
            and height_error <= maximum_height_error_m
            and np.all((planar_ratio >= 0.55) & (planar_ratio <= 1.55))
        )
        if not stack_evidence:
            completed_sources.append(source)
            continue

        top = float(upper[2])
        source_center = source.center_world.copy()
        source_center[:2] = source.observed.center_world[:2]
        source_center[2] = top - float(source_dimensions[2]) / 2.0
        source_grasp = source_center.copy()
        source_grasp[2] = top
        completed_source = source.with_semantics(
            source.label_scores,
            center_world=source_center,
            grasp_point_world=source_grasp,
            extents_m=source_dimensions,
        )
        completed_sources.append(completed_source)

        reference_center = source_center.copy()
        reference_center[2] = (
            top
            - float(source_dimensions[2])
            - float(reference_dimensions[2]) / 2.0
        )
        reference_observed = ObjectInstance(
            instance_id=(
                f"amodal-{normalize_label(reference_label).replace(' ', '-')}-under-"
                f"{source.instance_id}"
            ),
            center_world=reference_center,
            grasp_point_world=reference_center
            + np.array((0.0, 0.0, reference_dimensions[2] / 2.0)),
            axes_world=np.eye(3, dtype=np.float64),
            extents_m=reference_dimensions,
            observed=type(source.observed)(
                center_world=reference_center,
                axes_world=np.eye(3, dtype=np.float64),
                extents_m=reference_dimensions,
                bounds_min_world=reference_center - reference_dimensions / 2.0,
                bounds_max_world=reference_center + reference_dimensions / 2.0,
            ),
            label_scores={normalize_label(reference_label): min(source.confidence, 0.75)},
            confidence=min(source.confidence, 0.75),
            point_count=source.point_count,
            # The RGB pixels belong to the visible composite/top object; do
            # not misrepresent them as an independently observed support crop.
            color_histogram=None,
        )
        inferred_references.append(reference_observed)
    return completed_sources, [*result_references, *inferred_references]


def _select_source_on_measured_low_profile_support(
    frames: Sequence[RGBDFrame],
    sources: Sequence[ObjectInstance],
    source_label: str,
    reference_label: str,
    classifier: TextureSizeGallery | Any | None,
    *,
    table_height_m: float,
    minimum_sector_count: int = 6,
    minimum_footprint_ratio: float = 0.55,
    maximum_footprint_ratio: float = 1.35,
    minimum_family_score: float = 0.65,
    minimum_family_margin: float = 0.15,
) -> tuple[ObjectInstance | None, dict[str, Any]]:
    """Bind one source to an occluded, measured low-profile support plane.

    A source can completely hide a stove burner from both top-down views.  In
    that case a literal fixture box is unavailable even though the dual RGB-D
    cloud directly measures the horizontal support surrounding the source.
    This fallback is deliberately narrower than amodal fixture completion:

    * it runs only for a public low-profile fixture family after normal joint
      source/reference resolution has failed;
    * it preserves the already detected source identity and never fabricates a
      fixture OBB; after acceptance, only source Z is re-completed from the
      measured plane and the frozen public source height;
    * the visible source must be above the dominant table and surrounded by one
      unambiguous, connected, gravity-aligned height mode whose two axes match
      the public fixture footprint; and
    * texture/shape classification must independently choose the requested
      fixture from every public low-profile alternative (currently stove and
      tray), so a one-class gallery cannot manufacture semantics.

    All continuous geometry comes from calibrated RGB-D.  Public assets are
    used only for scale-normalised gates and the frozen semantic gallery.
    """

    source_name = normalize_label(source_label)
    reference_name = normalize_label(reference_label)
    trace: dict[str, Any] = {
        "kind": "on_low_profile_support_plane",
        "source_label": source_name,
        "reference_label": reference_name,
        "result": "not_applicable",
        "candidates": [],
    }
    if not 1 <= minimum_sector_count <= 8:
        raise ValueError("minimum_sector_count must lie in [1, 8]")
    if not 0.0 < minimum_footprint_ratio <= maximum_footprint_ratio:
        raise ValueError("invalid low-profile footprint ratio interval")
    if not 0.0 <= minimum_family_score <= 1.0:
        raise ValueError("minimum_family_score must lie in [0, 1]")
    if not 0.0 <= minimum_family_margin <= 1.0:
        raise ValueError("minimum_family_margin must lie in [0, 1]")

    prototypes = getattr(classifier, "prototypes", {}) if classifier is not None else {}
    reference_prototype = prototypes.get(reference_name)
    classify_features = getattr(classifier, "classify_features", None)
    if reference_prototype is None or not callable(classify_features):
        trace["result"] = "reference_has_no_public_gallery_prototype"
        return None, trace
    reference_dimensions = _low_profile_fixture_dimensions(
        reference_name,
        np.asarray(reference_prototype.dimensions_xyz_m, dtype=np.float64),
    )
    if reference_dimensions is None:
        trace["result"] = "reference_is_not_low_profile"
        return None, trace

    low_profile_labels = sorted(
        label
        for label, prototype in prototypes.items()
        if _low_profile_fixture_dimensions(
            label,
            np.asarray(prototype.dimensions_xyz_m, dtype=np.float64),
        )
        is not None
    )
    trace["low_profile_family"] = low_profile_labels
    trace["reference_dimensions_m"] = reference_dimensions.tolist()
    if reference_name not in low_profile_labels or len(low_profile_labels) < 2:
        trace["result"] = "low_profile_gallery_is_not_independent"
        return None, trace
    if not sources:
        trace["result"] = "no_source_candidates"
        return None, trace

    cloud = fuse_rgbd_frames(frames, stride=1)
    points = cloud.points_world
    accepted: list[ObjectInstance] = []
    expected_footprint = np.sort(reference_dimensions[:2])[::-1]
    fixture_height = float(reference_dimensions[2])
    ambiguous_mode_sources: list[str] = []

    for source in sources:
        candidate_trace: dict[str, Any] = {
            "source_instance_id": source.instance_id,
            "accepted": False,
            "rejections": [],
            "support_modes": [],
        }
        rejections: list[str] = candidate_trace["rejections"]
        observed_bottom = float(source.observed.bounds_min_world[2])
        observed_top = float(source.observed.bounds_max_world[2])
        observed_height = observed_top - observed_bottom
        raw_support_gap = observed_bottom - float(table_height_m)
        vertical_axis = int(np.argmax(np.abs(source.axes_world[2, :])))
        source_height = float(source.extents_m[vertical_axis])
        source_planar_axes = tuple(index for index in range(3) if index != vertical_axis)
        source_planar = np.asarray(source.extents_m[list(source_planar_axes)])
        minimum_support_gap = 0.08 * fixture_height
        maximum_support_gap = 1.50 * fixture_height
        minimum_vertical_span_ratio = 0.75
        maximum_vertical_span_ratio = 1.35
        plane_tolerance = float(np.clip(0.08 * source_height, 0.0025, 0.0050))
        histogram_bin_size = float(
            np.clip(0.025 * source_height, 0.0010, 0.0015)
        )
        inner_radius = float(
            0.5 * np.max(source_planar) + 0.04 * np.min(reference_dimensions[:2])
        )
        outer_radius = float(0.5 * np.linalg.norm(reference_dimensions[:2]))
        search_lower = max(
            float(table_height_m) + minimum_support_gap,
            observed_top - maximum_vertical_span_ratio * source_height,
        )
        search_upper = min(
            float(table_height_m) + maximum_support_gap,
            observed_top - minimum_vertical_span_ratio * source_height,
            observed_bottom + plane_tolerance,
        )
        candidate_trace.update(
            {
                "observed_bottom_z_m": observed_bottom,
                "observed_top_z_m": observed_top,
                "observed_height_m": observed_height,
                "source_height_m": source_height,
                "support_gap_from_table_m": raw_support_gap,
                "allowed_support_gap_m": [minimum_support_gap, maximum_support_gap],
                "allowed_source_vertical_span_ratio": [
                    minimum_vertical_span_ratio,
                    maximum_vertical_span_ratio,
                ],
                "support_search_z_m": [search_lower, search_upper],
                "support_histogram_bin_size_m": histogram_bin_size,
                "plane_tolerance_m": plane_tolerance,
                "annulus_radius_m": [inner_radius, outer_radius],
            }
        )
        # Retain the coarse source gate: it cheaply excludes a table-sized
        # robot crop and a high wrist/arm crop before any mode can borrow a
        # real fixture plane elsewhere in the image.  A partially visible bowl
        # whose lower edge moves by centimetres still lies inside this broad,
        # fixture-scaled interval.
        if not minimum_support_gap <= raw_support_gap <= maximum_support_gap:
            rejections.append("source_bottom_not_on_raised_low_profile_support")
        if not 0.45 * source_height <= observed_height <= 1.80 * source_height:
            rejections.append("source_observed_height_implausible")
        if not inner_radius < outer_radius:
            rejections.append("source_footprint_leaves_no_support_annulus")
        if search_lower > search_upper:
            rejections.append("source_has_no_valid_support_search_band")
        if rejections:
            trace["candidates"].append(candidate_trace)
            continue

        delta_xy = points[:, :2] - source.center_world[:2]
        radius = np.linalg.norm(delta_xy, axis=1)
        radial_mask = (radius >= inner_radius) & (radius <= outer_radius)
        search_mask = (
            radial_mask
            & (points[:, 2] >= search_lower)
            & (points[:, 2] <= search_upper)
        )
        search_indices = np.flatnonzero(search_mask)
        candidate_trace["support_search_point_count"] = int(len(search_indices))
        if len(search_indices) < 24:
            rejections.append("support_annulus_too_sparse")
            trace["candidates"].append(candidate_trace)
            continue

        bin_count = max(
            1,
            int(np.ceil((search_upper - search_lower) / histogram_bin_size)),
        )
        histogram_edges = search_lower + histogram_bin_size * np.arange(
            bin_count + 1,
            dtype=np.float64,
        )
        histogram_edges[-1] = max(histogram_edges[-1], search_upper)
        histogram_counts, _ = np.histogram(
            points[search_indices, 2],
            bins=histogram_edges,
        )
        smoothed_counts = np.convolve(
            histogram_counts.astype(np.float64),
            np.array((0.25, 0.50, 0.25), dtype=np.float64),
            mode="same",
        )
        peak_candidates: list[tuple[float, int, float]] = []
        for index, score in enumerate(smoothed_counts):
            if histogram_counts[index] <= 0:
                continue
            left = smoothed_counts[index - 1] if index > 0 else -1.0
            right = (
                smoothed_counts[index + 1]
                if index + 1 < len(smoothed_counts)
                else -1.0
            )
            if score < left or score < right or (score == left == right):
                continue
            in_bin = search_indices[
                (points[search_indices, 2] >= histogram_edges[index])
                & (points[search_indices, 2] <= histogram_edges[index + 1])
            ]
            if len(in_bin) == 0:
                continue
            peak_candidates.append(
                (
                    float(score),
                    int(histogram_counts[index]),
                    float(np.median(points[in_bin, 2])),
                )
            )
        # Density ranks proposals but never decides semantics.  NMS only
        # removes adjacent histogram aliases; every separated local mode runs
        # through the complete geometric and public-gallery gate below.
        support_peaks: list[float] = []
        for _score, _count, peak_z in sorted(
            peak_candidates,
            key=lambda item: (item[0], item[1], -item[2]),
            reverse=True,
        ):
            if all(abs(peak_z - kept) > plane_tolerance for kept in support_peaks):
                support_peaks.append(peak_z)
        support_peaks.sort()
        candidate_trace["support_peak_heights_m"] = support_peaks
        if not support_peaks:
            rejections.append("support_annulus_has_no_height_mode")
            trace["candidates"].append(candidate_trace)
            continue

        voxel_size = float(
            np.clip(0.02 * np.min(reference_dimensions[:2]), 0.003, 0.005)
        )
        component_config = ComponentConfig(
            voxel_size_m=voxel_size,
            connectivity_radius_m=2.75 * voxel_size,
            min_points=24,
            min_voxels=8,
        )
        accepted_modes: list[tuple[float, int, int, dict[str, Any]]] = []
        mode_traces: list[dict[str, Any]] = candidate_trace["support_modes"]
        for peak_z in support_peaks:
            mode_trace: dict[str, Any] = {
                "support_peak_height_m": peak_z,
                "accepted": False,
                "rejections": [],
            }
            mode_rejections: list[str] = mode_trace["rejections"]
            annulus_indices = np.flatnonzero(
                radial_mask & (np.abs(points[:, 2] - peak_z) <= plane_tolerance)
            )
            mode_trace["annulus_point_count"] = int(len(annulus_indices))
            if len(annulus_indices) < 24:
                mode_rejections.append("support_annulus_too_sparse")
                mode_traces.append(mode_trace)
                continue

            annulus_points = points[annulus_indices]
            components = connected_components_3d(annulus_points, component_config)
            if not components:
                mode_rejections.append("support_annulus_has_no_connected_plane")
                mode_traces.append(mode_trace)
                continue

            def occupied_xy_voxels(indices: np.ndarray) -> int:
                xy_keys = np.floor(
                    annulus_points[indices, :2] / voxel_size
                ).astype(np.int64)
                return int(len(np.unique(xy_keys, axis=0)))

            ranked_components = sorted(
                components,
                key=lambda indices: (occupied_xy_voxels(indices), len(indices)),
                reverse=True,
            )
            component = ranked_components[0]
            component_voxels = occupied_xy_voxels(component)
            runner_up_voxels = (
                occupied_xy_voxels(ranked_components[1])
                if len(ranked_components) > 1
                else 0
            )
            mode_trace.update(
                {
                    "support_component_point_count": int(len(component)),
                    "support_component_xy_voxels": component_voxels,
                    "support_runner_up_xy_voxels": runner_up_voxels,
                    "support_voxel_size_m": voxel_size,
                }
            )
            if component_voxels < 24:
                mode_rejections.append("support_plane_has_too_few_occupied_voxels")
            if runner_up_voxels > 0.35 * component_voxels:
                mode_rejections.append("support_plane_has_competing_components")

            component_global = annulus_indices[component]
            plane_points = points[component_global]
            measured_support_height = float(np.median(plane_points[:, 2]))
            measured_support_gap = measured_support_height - float(table_height_m)
            vertical_span_ratio = (
                observed_top - measured_support_height
            ) / source_height
            mode_trace.update(
                {
                    "measured_support_height_m": measured_support_height,
                    "measured_support_gap_from_table_m": measured_support_gap,
                    "source_vertical_span_above_support_ratio": vertical_span_ratio,
                }
            )
            if not minimum_support_gap <= measured_support_gap <= maximum_support_gap:
                mode_rejections.append("measured_support_not_raised_above_table")
            if measured_support_height > observed_bottom + plane_tolerance:
                mode_rejections.append("measured_support_above_source_lower_surface")
            if not (
                minimum_vertical_span_ratio
                <= vertical_span_ratio
                <= maximum_vertical_span_ratio
            ):
                mode_rejections.append("source_vertical_span_above_support_implausible")

            centered = plane_points - np.mean(plane_points, axis=0)
            eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centered, rowvar=False))
            normal = eigenvectors[:, int(np.argmin(eigenvalues))]
            gravity_alignment = abs(float(normal[2]))
            plane_residual = float(np.sqrt(np.mean(np.square(centered @ normal))))
            mode_trace["plane_gravity_alignment"] = gravity_alignment
            mode_trace["plane_rms_residual_m"] = plane_residual
            if gravity_alignment < 0.985:
                mode_rejections.append("support_plane_normal_not_gravity_aligned")
            if plane_residual > max(0.0015, 0.05 * source_height):
                mode_rejections.append("support_plane_residual_too_large")

            plane_xy = plane_points[:, :2]
            centered_xy = plane_xy - np.mean(plane_xy, axis=0)
            _, planar_axes = np.linalg.eigh(np.cov(centered_xy, rowvar=False))
            planar_coordinates = centered_xy @ planar_axes
            lower, upper = np.quantile(
                planar_coordinates,
                (0.01, 0.99),
                axis=0,
            )
            observed_footprint = np.sort(upper - lower)[::-1]
            footprint_ratio = observed_footprint / expected_footprint
            mode_trace["support_footprint_m"] = observed_footprint.tolist()
            mode_trace["support_footprint_ratio"] = footprint_ratio.tolist()
            if np.any(footprint_ratio < minimum_footprint_ratio):
                mode_rejections.append("support_plane_footprint_too_small")
            if np.any(footprint_ratio > maximum_footprint_ratio):
                mode_rejections.append("support_plane_footprint_too_large")

            angles = np.arctan2(
                plane_xy[:, 1] - source.center_world[1],
                plane_xy[:, 0] - source.center_world[0],
            )
            sectors = np.floor((angles + np.pi) / (np.pi / 4.0)).astype(np.int64)
            sector_count = int(len(np.unique(np.clip(sectors, 0, 7))))
            mode_trace["support_sector_count"] = sector_count
            if sector_count < minimum_sector_count:
                mode_rejections.append("support_plane_does_not_surround_source")

            try:
                from scipy.spatial import ConvexHull, QhullError

                hull = ConvexHull(plane_xy)
                inside_hull = bool(
                    np.all(
                        hull.equations[:, :2] @ source.center_world[:2]
                        + hull.equations[:, 2]
                        <= voxel_size
                    )
                )
            except (ValueError, QhullError):
                inside_hull = False
            mode_trace["source_inside_support_hull"] = inside_hull
            if not inside_hull:
                mode_rejections.append("source_center_outside_support_hull")

            plane_thickness = max(2.0 * plane_residual, voxel_size)
            try:
                classification = classify_features(
                    hsv_histogram(cloud.colors_rgb[component_global]),
                    np.array(
                        (
                            observed_footprint[0],
                            observed_footprint[1],
                            plane_thickness,
                        ),
                        dtype=np.float64,
                    ),
                    allowed_labels=low_profile_labels,
                )
                family_scores = {
                    normalize_label(label): float(score)
                    for label, score in classification.scores.items()
                }
                mode_trace.update(
                    {
                        "support_family_scores": {
                            label: score if np.isfinite(score) else None
                            for label, score in family_scores.items()
                        },
                        "support_family_label": normalize_label(classification.label),
                    }
                )
                if set(family_scores) != set(low_profile_labels):
                    mode_rejections.append("support_gallery_score_vector_incomplete")
                elif not all(np.isfinite(score) for score in family_scores.values()):
                    mode_rejections.append("support_gallery_scores_not_finite")
                elif not all(0.0 <= score <= 1.0 for score in family_scores.values()):
                    mode_rejections.append("support_gallery_scores_out_of_range")
                elif not np.isclose(
                    sum(family_scores.values()), 1.0, rtol=1e-6, atol=1e-6
                ):
                    mode_rejections.append("support_gallery_scores_not_normalized")
                else:
                    requested_score = family_scores[reference_name]
                    runner_up_score = max(
                        score
                        for label, score in family_scores.items()
                        if label != reference_name
                    )
                    family_margin = requested_score - runner_up_score
                    mode_trace.update(
                        {
                            "support_family_score": requested_score,
                            "support_family_margin": family_margin,
                        }
                    )
                    if normalize_label(classification.label) != reference_name:
                        mode_rejections.append(
                            "support_gallery_selected_different_fixture"
                        )
                    if requested_score < minimum_family_score:
                        mode_rejections.append("support_gallery_score_too_low")
                    if family_margin < minimum_family_margin:
                        mode_rejections.append("support_gallery_margin_too_small")
            except (KeyError, ValueError):
                mode_rejections.append("support_gallery_classification_unavailable")

            if not mode_rejections:
                mode_trace["accepted"] = True
                accepted_modes.append(
                    (
                        measured_support_height,
                        component_voxels,
                        int(len(component)),
                        mode_trace,
                    )
                )
            mode_traces.append(mode_trace)

        if not accepted_modes:
            # Preserve the old flat candidate trace surface for result readers
            # while retaining every attempted mode under ``support_modes``.
            for mode_trace in mode_traces:
                for reason in mode_trace["rejections"]:
                    if reason not in rejections:
                        rejections.append(reason)
            if len(mode_traces) == 1:
                candidate_trace.update(
                    {
                        key: value
                        for key, value in mode_traces[0].items()
                        if key not in {"accepted", "rejections"}
                    }
                )
            trace["candidates"].append(candidate_trace)
            continue

        mode_clusters: list[list[tuple[float, int, int, dict[str, Any]]]] = []
        for mode in sorted(accepted_modes, key=lambda item: item[0]):
            if (
                not mode_clusters
                or mode[0] - mode_clusters[-1][-1][0] > plane_tolerance
            ):
                mode_clusters.append([mode])
            else:
                mode_clusters[-1].append(mode)
        representatives = [
            max(cluster, key=lambda item: (item[1], item[2], -abs(item[0])))
            for cluster in mode_clusters
        ]
        if len(representatives) != 1:
            rejections.append("ambiguous_multiple_support_planes")
            candidate_trace["accepted_support_plane_heights_m"] = [
                item[0] for item in representatives
            ]
            ambiguous_mode_sources.append(source.instance_id)
            trace["candidates"].append(candidate_trace)
            continue

        measured_support_height, _voxels, _point_count, selected_mode = representatives[0]
        candidate_trace.update(
            {
                key: value
                for key, value in selected_mode.items()
                if key not in {"accepted", "rejections"}
            }
        )
        candidate_trace["accepted"] = True
        completed_center = source.center_world.copy()
        completed_center[2] = measured_support_height + source_height / 2.0
        completed_grasp = source.grasp_point_world.copy()
        completed_grasp[2] = measured_support_height + source_height
        needs_reanchor = bool(
            abs(float(source.center_world[2]) - float(completed_center[2])) > 1e-6
            or abs(float(source.grasp_point_world[2]) - float(completed_grasp[2]))
            > 1e-6
        )
        candidate_trace.update(
            {
                "source_z_reanchored": needs_reanchor,
                "original_source_center_z_m": float(source.center_world[2]),
                "reanchored_source_center_z_m": float(completed_center[2]),
                "reanchored_source_grasp_z_m": float(completed_grasp[2]),
            }
        )
        selected_source = (
            source.with_semantics(
                source.label_scores,
                center_world=completed_center,
                grasp_point_world=completed_grasp,
            )
            if needs_reanchor
            else source
        )
        accepted.append(selected_source)
        trace["candidates"].append(candidate_trace)

    if ambiguous_mode_sources:
        trace["result"] = "ambiguous_multiple_support_planes"
        trace["ambiguous_source_instance_ids"] = ambiguous_mode_sources
        return None, trace
    if len(accepted) == 1:
        trace["result"] = "accepted_unique_source"
        trace["selected_source_instance_id"] = accepted[0].instance_id
        selected_traces = [
            item
            for item in trace["candidates"]
            if item.get("accepted")
            and item.get("source_instance_id") == accepted[0].instance_id
        ]
        if len(selected_traces) == 1:
            trace["measured_support_height_m"] = selected_traces[0][
                "measured_support_height_m"
            ]
        return accepted[0], trace
    if len(accepted) > 1:
        trace["result"] = "ambiguous_multiple_supported_sources"
        trace["accepted_source_instance_ids"] = [item.instance_id for item in accepted]
        return None, trace
    trace["result"] = "no_source_passed_support_plane_gates"
    return None, trace


def _recover_sources_above_fixture(
    frames: Sequence[RGBDFrame],
    references: Sequence[ObjectInstance],
    subject: str,
    classifier: TextureSizeGallery | None,
    *,
    workspace_min: np.ndarray,
    workspace_max: np.ndarray,
    minimum_reference_planar_m: float = 0.15,
) -> list[ObjectInstance]:
    """Recover a gallery object from RGB-D points above a detected fixture.

    A bowl on a tall cabinet may be missed by the literal ``black bowl`` text
    query, while the cabinet box still supplies a measured top support plane.
    This function searches only the thin prism immediately above that measured
    plane and accepts components through the same frozen texture/size gallery.
    The relation comes from language; all continuous geometry comes from RGB-D.
    """

    if minimum_reference_planar_m <= 0:
        raise ValueError("minimum reference planar size must be positive")
    normalized = normalize_label(subject)
    if classifier is None or normalized not in classifier.prototypes or not references:
        return []
    prototype = classifier.prototypes[normalized]
    expected_planar = float(np.max(prototype.dimensions_xyz_m[:2]))
    cloud = fuse_rgbd_frames(frames, stride=1)
    lower_workspace = np.asarray(workspace_min, dtype=np.float64)
    upper_workspace = np.asarray(workspace_max, dtype=np.float64)
    component_config = ComponentConfig(
        voxel_size_m=0.003,
        connectivity_radius_m=0.010,
        min_points=10,
        min_voxels=2,
    )
    recovered: list[ObjectInstance] = []
    for reference_index, reference in enumerate(references):
        if (
            float(np.max(reference.observed.extents_m[:2]))
            < minimum_reference_planar_m
        ):
            continue
        reference_lower = reference.observed.bounds_min_world
        reference_upper = reference.observed.bounds_max_world
        support_height = float(reference_upper[2])
        maximum_height = max(0.09, 1.8 * float(np.max(prototype.dimensions_xyz_m)))
        points = cloud.points_world
        mask = (
            (points[:, 0] >= reference_lower[0] - 0.04)
            & (points[:, 0] <= reference_upper[0] + 0.04)
            & (points[:, 1] >= reference_lower[1] - 0.04)
            & (points[:, 1] <= reference_upper[1] + 0.04)
            & (points[:, 2] >= support_height + 0.001)
            & (points[:, 2] <= support_height + maximum_height)
        )
        cropped = cloud.subset(mask)
        if len(cropped.points_world) < component_config.min_points:
            continue
        for component_index, indices in enumerate(
            connected_components_3d(cropped.points_world, component_config)
        ):
            component = cropped.subset(indices)
            observed = fit_observed_geometry(
                component.points_world,
                support_height_m=support_height,
                force_support_plane=True,
            )
            observed_planar = float(np.max(observed.extents_m[:2]))
            if not 0.45 * expected_planar <= observed_planar <= 1.9 * expected_planar:
                continue
            histogram = hsv_histogram(component.colors_rgb)
            classification = classifier.classify_features(
                histogram,
                observed.extents_m,
                allowed_labels=sorted(_SPATIAL_GALLERY_LABELS),
            )
            score = float(classification.scores.get(normalized, 0.0))
            top_score = max(classification.scores.values(), default=0.0)
            shape = classifier._shape_similarity(
                observed.extents_m,
                prototype.dimensions_xyz_m,
            )
            if score < 0.55 or score < 0.72 * top_score or shape < 0.52:
                continue
            dimensions = classifier.orient_dimensions(
                prototype.dimensions_xyz_m,
                observed.extents_m,
            )
            center = observed.center_world.copy()
            center[2] = support_height + dimensions[2] / 2.0
            if not (np.all(center >= lower_workspace) and np.all(center <= upper_workspace)):
                continue
            grasp = center.copy()
            grasp[2] += dimensions[2] / 2.0
            recovered.append(
                ObjectInstance(
                    instance_id=f"supported-{normalized.replace(' ', '-')}-{reference_index:02d}-{component_index:02d}",
                    center_world=center,
                    grasp_point_world=grasp,
                    axes_world=observed.axes_world,
                    extents_m=dimensions,
                    observed=observed,
                    label_scores={normalized: score},
                    confidence=score,
                    point_count=len(component.points_world),
                    color_histogram=histogram,
                )
            )
    return list(fuse_instances_3d(recovered))


def coerce_rgbd_frame(
    frame: Any,
    *,
    name: str | None = None,
    timestamp_s: float | None = None,
    capture_id: str | None = None,
) -> RGBDFrame:
    """Convert neutral, ANCHOR, Planning, or equivalent duck-typed RGB-D frames."""

    if isinstance(frame, RGBDFrame):
        if name is None and timestamp_s is None and capture_id is None:
            return frame
        return replace(
            frame,
            name=frame.name if name is None else name,
            timestamp_s=frame.timestamp_s if timestamp_s is None else timestamp_s,
            capture_id=frame.capture_id if capture_id is None else capture_id,
        )
    calibration = getattr(frame, "calibration", None)
    intrinsics = getattr(frame, "intrinsics", None)
    if intrinsics is None and calibration is not None:
        intrinsics = calibration.intrinsic
    if all(hasattr(intrinsics, field) for field in ("fx", "fy", "cx", "cy")):
        matrix = np.array(
            [[intrinsics.fx, 0.0, intrinsics.cx], [0.0, intrinsics.fy, intrinsics.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    else:
        matrix = np.asarray(intrinsics, dtype=np.float64)
    world_from_camera = getattr(frame, "world_from_camera", None)
    if world_from_camera is None and calibration is not None:
        world_from_camera = calibration.T_world_camera
    if hasattr(world_from_camera, "matrix"):
        world_from_camera = world_from_camera.matrix
    camera_name = (
        name
        or getattr(frame, "camera_name", None)
        or getattr(frame, "name", None)
        or getattr(calibration, "name", None)
    )
    stamp = timestamp_s if timestamp_s is not None else getattr(frame, "timestamp_s", 0.0)
    commitment = (
        capture_id
        if capture_id is not None
        else getattr(frame, "capture_id", "")
    )
    return RGBDFrame(
        name=str(camera_name),
        rgb=np.asarray(frame.rgb),
        depth_m=np.asarray(frame.depth_m),
        intrinsics=matrix,
        world_from_camera=np.asarray(world_from_camera),
        timestamp_s=float(stamp),
        observation_v_flipped=bool(
            getattr(frame, "observation_v_flipped", False)
            or getattr(calibration, "observation_v_flipped", False)
        ),
        capture_id=commitment,
    )


def _dino_instances(
    frames: Sequence[RGBDFrame],
    queries: Sequence[str],
    detector: FrozenBoxDetector | None,
    extractor: BoxGeometryExtractor | None,
    table_height_m: float,
    *,
    gallery: TextureSizeGallery | None = None,
    search_all_frames: bool | Sequence[str] = False,
    provenance_only_secondary_queries: Sequence[str] = (),
    provenance_sink: Callable[[_DINO2DProvenance], None] | None = None,
) -> list[ObjectInstance]:
    if detector is None or extractor is None:
        return []
    fixtures = (
        "cabinet",
        "drawer",
        "stove",
        "microwave",
        "rack",
        "shelf",
        "caddy",
        "tray",
    )
    normalized_queries = list(dict.fromkeys(normalize_label(query) for query in queries))
    if not normalized_queries:
        return []
    if isinstance(search_all_frames, bool):
        all_view_queries = (
            set(normalized_queries) if search_all_frames else set()
        )
    else:
        all_view_queries = {
            normalize_label(query) for query in search_all_frames
        }
        unknown = all_view_queries - set(normalized_queries)
        if unknown:
            raise ValueError(
                "all-view queries must belong to the DINO request: "
                f"{sorted(unknown)}"
            )
    provenance_only_queries = {
        normalize_label(query) for query in provenance_only_secondary_queries
    }
    unknown_provenance = provenance_only_queries - set(normalized_queries)
    if unknown_provenance:
        raise ValueError(
            "provenance-only queries must belong to the DINO request: "
            f"{sorted(unknown_provenance)}"
        )
    detector_queries, canonical_for = _expanded_dino_queries(normalized_queries)
    primary = [frame for frame in frames if "agent" in frame.name] or [frames[0]]
    primary_detections = list(detector.detect(primary[0], detector_queries))
    searched_queries = {primary[0].name: set(normalized_queries)}
    # Keep the ordinary detector stream byte-for-byte independent of the
    # provenance-only stream below.  Grounding-DINO uses one compound prompt
    # per call, so merely appending a fixture phrase to an existing wrist
    # request can change the ordinary bowl boxes produced by that forward.
    ordinary_detections = list(primary_detections)
    found = {
        canonical_for.get(normalize_label(detection.query), normalize_label(detection.query))
        for detection in primary_detections
    }
    missing = [query for query in normalized_queries if query not in found]
    if missing or all_view_queries:
        for frame in frames:
            if frame is primary[0]:
                continue
            secondary_canonical = [
                query
                for query in normalized_queries
                if query in all_view_queries or query in missing
            ]
            if not secondary_canonical:
                break
            secondary_queries, _ = _expanded_dino_queries(
                secondary_canonical
            )
            secondary_detections = list(
                detector.detect(frame, secondary_queries)
            )
            searched_queries[frame.name] = set(secondary_canonical)
            ordinary_detections.extend(secondary_detections)
            found.update(
                canonical_for.get(normalize_label(detection.query), normalize_label(detection.query))
                for detection in secondary_detections
            )
            missing = [query for query in missing if query not in found]
            if not all_view_queries and not missing:
                break

    # Perform a genuinely separate detector forward for the extra fixture
    # evidence.  These calls happen only after the complete ordinary stream,
    # so neither their prompt tokens nor detector-side mutable state can
    # perturb any pre-existing primary/missing/all-view call.  Tag occurrences
    # structurally rather than by object id: an adapter may legally return the
    # same immutable detection object from the ordinary and provenance calls.
    provenance_only_detections: list[BoxDetection] = []
    if provenance_only_queries:
        provenance_canonical = [
            query
            for query in normalized_queries
            if query in provenance_only_queries
        ]
        provenance_detector_queries, _ = _expanded_dino_queries(
            provenance_canonical
        )
        for frame in frames:
            if frame is primary[0]:
                continue
            provenance_only_detections.extend(
                detector.detect(frame, provenance_detector_queries)
            )
    results: list[ObjectInstance] = []
    for query in normalized_queries:
        ordinary_query_detections = [
            detection
            for detection in ordinary_detections
            if canonical_for.get(normalize_label(detection.query), normalize_label(detection.query))
            == query
        ]
        provenance_query_detections = [
            detection
            for detection in provenance_only_detections
            if canonical_for.get(normalize_label(detection.query), normalize_label(detection.query))
            == query
        ]
        result_rank = 0
        provenance_rank = 0
        ranked_occurrences = (
            *(
                (detection, True)
                for detection in sorted(
                    ordinary_query_detections,
                    key=lambda item: item.score,
                    reverse=True,
                )
            ),
            *(
                (detection, False)
                for detection in sorted(
                    provenance_query_detections,
                    key=lambda item: item.score,
                    reverse=True,
                )
            ),
        )
        for detection, contributes_result in ranked_occurrences:
            if contributes_result:
                rank = result_rank
                result_rank += 1
                instance_suffix = f"{rank:02d}"
            else:
                rank = provenance_rank
                provenance_rank += 1
                camera_token = normalize_label(detection.camera_name).replace(
                    " ", "-"
                )
                instance_suffix = f"provenance-{camera_token}-{rank:02d}"
            frame = next(item for item in frames if item.name == detection.camera_name)
            extracted: ObjectInstance | None = None
            extracted_points: np.ndarray | None = None

            def emit_provenance() -> None:
                if provenance_sink is not None:
                    provenance_sink(
                        _DINO2DProvenance(
                            canonical_query=query,
                            detection=detection,
                            frame_width=frame.width,
                            frame_height=frame.height,
                            instance=extracted,
                            surface_points_world=(
                                extracted_points if extracted is not None else None
                            ),
                        )
                    )

            above_support = not any(token in normalize_label(query) for token in fixtures)
            try:
                extraction_kwargs = {
                    "support_height_m": table_height_m if above_support else None,
                    "above_support_only": above_support,
                    "instance_id": (
                        f"dino-{normalize_label(query).replace(' ', '-')}-"
                        f"{instance_suffix}"
                    ),
                }
                extract_with_cloud = getattr(extractor, "extract_with_cloud", None)
                if callable(extract_with_cloud):
                    _, instance, cloud = extract_with_cloud(
                        frame,
                        detection,
                        **extraction_kwargs,
                    )
                    extracted_points = np.asarray(
                        cloud.points_world,
                        dtype=np.float64,
                    ).copy()
                else:
                    # Test doubles and external adapters retain the historic
                    # two-value protocol.  They cannot claim raw surface
                    # evidence, so pan affordance inference will fail closed.
                    _, instance = extractor.extract(
                        frame,
                        detection,
                        **extraction_kwargs,
                    )
            except (ValueError, RuntimeError):
                emit_provenance()
                continue
            # The wrist camera is rigidly mounted on the robot. Its immediate
            # housing/hand neighbourhood also attracts book and fixture
            # false positives. A held object's centre is beyond this compact
            # mounting envelope; use the calibrated camera position for this
            # test rather than a world-coordinate or task-specific mask.
            if any(
                "wrist" in camera.name
                and np.linalg.norm(
                    instance.observed.center_world
                    - camera.world_from_camera[:3, 3]
                ) < 0.085
                for camera in frames
            ):
                emit_provenance()
                continue
            prototype, family_evidence = _gallery_prototype_for_query(
                query,
                instance,
                gallery,
            )
            if prototype is not None:
                if query == "moka pot":
                    # Thin stove faces attract the same metal-appliance
                    # prompt. They cannot supply the vertical body of the
                    # public pot, even in its shortest resting orientation.
                    observed_height = float(instance.observed.bounds_max_world[2]
                                            - instance.observed.bounds_min_world[2])
                    if observed_height < .80 * min(prototype.dimensions_xyz_m):
                        emit_provenance()
                        continue
                flat_fixture_dimensions = _low_profile_fixture_dimensions(
                    query,
                    prototype.dimensions_xyz_m,
                )
                if query == "caddy":
                    # The public fixture is authored standing on its base;
                    # unlike movable packages its Z cannot be exchanged with
                    # the long horizontal axis by a partial wrist crop.
                    raw = np.asarray(prototype.dimensions_xyz_m)
                    flat_fixture_dimensions = np.array(
                        (*sorted(raw[:2], reverse=True), raw[2]), dtype=np.float64
                    )
                if flat_fixture_dimensions is not None:
                    if not _low_profile_fixture_observation_agrees(
                        instance,
                        flat_fixture_dimensions,
                        table_height_m=table_height_m,
                    ):
                        emit_provenance()
                        continue
                    if query == "stove" and not _stove_fixture_semantics_agree(
                        instance,
                        gallery,
                    ):
                        emit_provenance()
                        continue
                    oriented_dimensions = flat_fixture_dimensions
                else:
                    oriented_dimensions = gallery.orient_dimensions(
                        prototype.dimensions_xyz_m,
                        instance.observed.extents_m,
                    )
                shape = gallery._shape_similarity(
                    instance.observed.extents_m,
                    prototype.dimensions_xyz_m,
                )
                expected_planar = float(np.max(oriented_dimensions[:2]))
                size_kwargs: dict[str, float] = {}
                if query in _LARGE_FIXTURE_LABELS:
                    # A DINO fixture box can expose just an aperture, front
                    # panel, or rim.  Admit such measured partial geometry and
                    # complete it from the public collision AABB; this branch
                    # never runs through the tabletop component's 28-cm cap.
                    size_kwargs = {
                        "minimum_ratio": 0.16,
                        "maximum_ratio": 2.8,
                        "maximum_absolute_m": max(0.30, 1.5 * expected_planar),
                    }
                if not _gallery_planar_size_agrees(
                    instance.observed.extents_m,
                    expected_planar,
                    **size_kwargs,
                ):
                    emit_provenance()
                    continue
                family_score = (
                    family_evidence.accepted_score
                    if family_evidence is not None
                    else 0.0
                )
                if query in _SPATIAL_GALLERY_LABELS:
                    adjusted = float(
                        np.clip(detection.score * (0.20 + 0.80 * shape), 0.0, 1.0)
                    )
                else:
                    # Texture is compared against a complete semantic family,
                    # then combined with DINO and collision-shape evidence.
                    # It can lower confidence but can never create a proposal.
                    adjusted = float(
                        np.clip(
                            detection.score
                            * (0.25 + 0.55 * shape + 0.20 * family_score),
                            0.0,
                            1.0,
                        )
                    )
                upright_dimensions = _upright_stack_dimensions(gallery, query)
                dimensions = (
                    upright_dimensions
                    if upright_dimensions is not None
                    else np.array(
                        [
                            *sorted(oriented_dimensions[:2], reverse=True),
                            oriented_dimensions[2],
                        ],
                        dtype=np.float64,
                    )
                )
                instance = instance.with_semantics(
                    {query: adjusted},
                    confidence=adjusted,
                )
                if flat_fixture_dimensions is not None:
                    instance = _complete_low_profile_fixture_geometry(
                        instance,
                        dimensions,
                        table_height_m=table_height_m,
                    )
                elif upright_dimensions is not None:
                    instance = _complete_static_detection_geometry(
                        instance,
                        dimensions,
                        table_height_m=table_height_m,
                        # A plate is a broad, shallow target whose DINO box
                        # centre ray is easily biased by crop asymmetry.  Its
                        # segmented RGB-D points directly measure planar
                        # position and remain substantially more accurate in
                        # the unobstructed pre-pick view.  Height/dimensions
                        # still come from the frozen public asset gallery.
                        prefer_observed_xy=query == "plate",
                    )
                else:
                    local_support = table_height_m
                    elevated = bool(
                        instance.observed.bounds_min_world[2]
                        > table_height_m + 0.020
                    )
                    if elevated:
                        local_support = float(instance.observed.bounds_min_world[2])
                    center = instance.center_world.copy()
                    # ``BoxGeometryExtractor`` projects the crop-centre ray to
                    # the dominant table when extracting ordinary objects.
                    # That XY is appropriate for a table-supported crop, but
                    # an elevated object lies earlier on the same ray and can
                    # otherwise be displaced by tens of centimetres.  Its
                    # segmented RGB-D component directly measures XY; retain
                    # the existing local-support Z and public-size completion.
                    if elevated:
                        center[:2] = instance.observed.center_world[:2]
                    center[2] = local_support + dimensions[2] / 2.0
                    grasp = center.copy()
                    grasp[2] = local_support + dimensions[2]
                    instance = instance.with_semantics(
                        instance.label_scores,
                        center_world=center,
                        grasp_point_world=grasp,
                        extents_m=dimensions,
                    )
            elif normalize_label(detection.query) != query:
                # Preserve the detector box and its text provenance internally,
                # but expose the instruction's canonical entity label to the
                # selector and fixture size gate.
                instance = instance.with_semantics(
                    {query: float(detection.score)},
                    confidence=float(detection.score),
                )
            extracted = instance
            emit_provenance()
            if contributes_result:
                results.append(instance)
    # A raw primary box is not a localized entity. The primary detector can
    # call the robot or a plate a stove, then all its crops fail RGB-D/gallery
    # validation. In that case the unused wrist view still needs a chance to
    # localize the fixture. Keep valid primary geometry stable; only retry
    # missing accepted labels, with the same geometry and semantic checks.
    for frame in frames:
        if frame is primary[0]:
            continue
        accepted = {query for query in normalized_queries
                    if any(item.score_for(query) > 0.0 for item in results)}
        retry_queries = [query for query in normalized_queries
                         if query not in accepted
                         and query not in searched_queries.get(frame.name, set())]
        if retry_queries:
            results.extend(_dino_instances(
                (frame,), retry_queries, detector, extractor, table_height_m,
                gallery=gallery, provenance_sink=provenance_sink,
            ))
    fused_results = list(fuse_instances_3d(results))
    for query in normalized_queries:
        if "drawer" not in query.split():
            continue
        queried = [item for item in fused_results if item.score_for(query) > 0]
        if not queried:
            continue
        consistent = _drawer_handle_consistent_candidates(query, queried, frames)
        accepted_ids = {id(item) for item in consistent}
        fused_results = [item for item in fused_results
                         if item.score_for(query) <= 0 or id(item) in accepted_ids]
    return fused_results


class AnchorScenePerceptionAdapter:
    """Implements ``anchor.perception.ScenePerception`` without oracle data."""

    def __init__(
        self,
        backend: SensorOnlyScenePerception,
        *,
        box_detector: FrozenBoxDetector | None = None,
        box_extractor: BoxGeometryExtractor | None = None,
        minimum_label_score: float = 0.04,
    ) -> None:
        self.backend = backend
        self.box_detector = box_detector
        self.box_extractor = box_extractor
        self.minimum_label_score = float(minimum_label_score)
        self._dino_cache: dict[str, list[ObjectInstance]] = {}
        self._dino_raw_cache: dict[str, list[ObjectInstance]] = {}
        # Empty detector results are observation-scoped, not episode facts.
        # Keep them only long enough to deduplicate repeated queries made for
        # the same immutable policy observation; a later RGB-D frame gets one
        # genuine retry.  Non-empty static fixture geometry remains cached as
        # before so successful grounding does not jitter across a carry.
        self._dino_cache_timestamp: dict[str, float] = {}
        self._dino_gate_context: dict[str, bool] = {}
        self._dino_provenance: dict[str, list[_DINO2DProvenance]] = {}
        self._tracks: dict[str, np.ndarray] = {}
        self.last_selector_reference: ObjectInstance | None = None
        self._selector_diagnostics: list[dict[str, Any]] = []
        self._observation_sequence = 0
        self._last_observation: Any | None = None
        self._last_observation_timestamp_s = 0.0

    @property
    def selector_diagnostics(self) -> tuple[dict[str, Any], ...]:
        """Append-only sensor diagnostics for relation fallbacks this episode."""

        return tuple(getattr(self, "_selector_diagnostics", ()))

    def reset(self) -> None:
        """Clear episode-local, sensor-derived detector and identity tracks."""

        self._dino_cache.clear()
        getattr(self, "_dino_raw_cache", {}).clear()
        getattr(self, "_dino_cache_timestamp", {}).clear()
        getattr(self, "_dino_gate_context", {}).clear()
        getattr(self, "_dino_provenance", {}).clear()
        self._tracks.clear()
        self.last_selector_reference = None
        getattr(self, "_selector_diagnostics", []).clear()
        self._observation_sequence = 0
        self._last_observation = None
        self._last_observation_timestamp_s = 0.0

    def _capture_timestamp_s(self, observation: Any) -> float:
        """Return a deterministic episode-local token for this policy value.

        ``SensorObservation`` intentionally contains no clock or step counter.
        The immutable adapter value is recreated for each policy action, so
        identity lets all perception calls for that value share one internal
        provenance token without accepting environment timing as input.
        """

        if observation is not getattr(self, "_last_observation", None):
            sequence = int(getattr(self, "_observation_sequence", 0))
            self._last_observation = observation
            self._last_observation_timestamp_s = float(sequence)
            self._observation_sequence = sequence + 1
        return float(getattr(self, "_last_observation_timestamp_s", 0.0))

    def begin_observation(self, observation: Any, sequence: int) -> None:
        """Bind all perception queries made by one controller act."""

        if sequence < 0:
            raise ValueError("observation sequence must be non-negative")
        self._last_observation = observation
        self._last_observation_timestamp_s = float(sequence)
        self._observation_sequence = max(
            int(getattr(self, "_observation_sequence", 0)),
            sequence + 1,
        )

    def invalidate_dynamic(self, query: str) -> None:
        """Discard cached geometry after a grasp or release changes the scene."""

        normalized = normalize_label(query)
        self._dino_cache.pop(normalized, None)
        getattr(self, "_dino_raw_cache", {}).pop(normalized, None)
        getattr(self, "_dino_cache_timestamp", {}).pop(normalized, None)
        getattr(self, "_dino_gate_context", {}).pop(normalized, None)
        self._dino_provenance.pop(normalized, None)
        self._tracks.pop(normalized, None)

    def _candidates(self, observation: Any, queries: Sequence[str]):
        timestamp_s = self._capture_timestamp_s(observation)
        frames = [
            coerce_rgbd_frame(frame, name=name, timestamp_s=timestamp_s)
            for name, frame in observation.cameras.items()
        ]
        normalized_queries = [normalize_label(query) for query in queries]
        cavity_context = any("drawer" in query.split() for query in normalized_queries)
        object_mode = _uses_object_gallery(normalized_queries)
        gallery_queries = _neutral_gallery_queries(normalized_queries)
        neutral = self.backend.observe(frames, requested_labels=gallery_queries)
        neutral_instances = list(fuse_instances_3d(neutral.instances))
        if object_mode:
            neutral_instances = [
                instance
                for instance in neutral_instances
                if instance.observed.bounds_min_world[2] - neutral.table_height_m <= 0.025
            ]
        by_query: dict[str, list[ObjectInstance]] = {}
        missing: list[str] = []
        force_dino = _DINO_GALLERY_LABELS
        for raw_query in normalized_queries:
            query = normalize_label(raw_query)
            candidates = []
            for instance in _gallery_label_candidates(
                neutral_instances,
                query,
                minimum_score=self.minimum_label_score,
                require_argmax=query in {"black bowl", "ramekin"},
            ):
                prototype = (
                    self.backend.classifier.prototypes.get(query)
                    if self.backend.classifier is not None
                    else None
                )
                if (
                    prototype is not None
                    and self.backend.classifier._shape_similarity(
                        instance.observed.extents_m,
                        prototype.dimensions_xyz_m,
                    )
                    < 0.30
                ):
                    continue
                candidates.append(instance)
            if candidates and query not in force_dino:
                by_query[query] = candidates
            else:
                missing.append(query)
        cache_timestamps = getattr(self, "_dino_cache_timestamp", {})
        self._dino_cache_timestamp = cache_timestamps
        gate_context = getattr(self, "_dino_gate_context", {})
        self._dino_gate_context = gate_context
        raw_cache = getattr(self, "_dino_raw_cache", {})
        self._dino_raw_cache = raw_cache
        if ("black bowl" in missing and "black bowl" in self._dino_cache
                and "black bowl" in raw_cache
                and cache_timestamps.get("black bowl") == timestamp_s
                and gate_context.get("black bowl", False) != cavity_context):
            # The same RGB-D capture already has the detector proposal. A
            # relation changes its acceptance criteria, not its text caption
            # or measured geometry. Reuse raw proposals before filtering.
            self._dino_cache["black bowl"] = _spatial_dino_gallery_gate(
                "black bowl", raw_cache["black bowl"], neutral_instances,
                workspace_min=self.backend.config.workspace_min,
                workspace_max=self.backend.config.workspace_max,
                classifier=self.backend.classifier,
                allow_bowl_cavity_merge=cavity_context,
            )
            gate_context["black bowl"] = cavity_context
        uncached = [
            query
            for query in missing
            if query not in self._dino_cache
            or (query == "black bowl" and gate_context.get(query, False) != cavity_context)
            or (
                not self._dino_cache[query]
                and cache_timestamps.get(query) != timestamp_s
            )
        ]
        if (cavity_context and "black bowl" in uncached
                and "black bowl" in self._dino_cache
                and not gate_context.get("black bowl", False)):
            # A context change must rebuild the compound proposal. Ordinary
            # dynamic reacquisition preserves the cached drawer and original
            # bowl-only refresh, so it does not change the geometry merely by
            # expanding the caption after a failed grasp.
            uncached.extend(query for query in normalized_queries
                            if "drawer" in query.split() and query not in uncached)
        for query in uncached:
            self._dino_provenance.pop(query, None)
        dino = (
            _dino_instances(
                frames,
                uncached,
                self.box_detector,
                self.box_extractor,
                neutral.table_height_m,
                gallery=self.backend.classifier,
                search_all_frames=_all_view_spatial_queries(uncached),
                provenance_sink=lambda item: self._dino_provenance.setdefault(
                    item.canonical_query, []
                ).append(item),
            )
            if uncached
            else []
        )
        relabelled = _relabel_spatial_crop_proposals(
            dino,
            self.backend.classifier,
            workspace_min=self.backend.config.workspace_min,
            workspace_max=self.backend.config.workspace_max,
        )
        for query in uncached:
            query_detections = [item for item in dino if item.score_for(query) > 0]
            query_detections.extend(
                item for item in relabelled if item.score_for(query) > 0
            )
            raw_query_detections = list(fuse_instances_3d(query_detections))
            query_detections = _fixture_size_gate(query, raw_query_detections)
            if not query_detections:
                # With a compound ``black bowl. top drawer.`` prompt DINO can
                # tightly ground the visible bowl as both phrases and omit the
                # full drawer.  Keep the strongest measured proposal as an
                # ambiguous relation anchor; cross-crop relabelling supplies
                # the same proposal as a source, so selector geometry still
                # disambiguates without inventing a fixture pose.
                query_detections = _ambiguous_fixture_fallback(
                    query,
                    raw_query_detections,
                )
            if query == "black bowl":
                raw_cache[query] = list(query_detections)
            self._dino_cache[query] = _spatial_dino_gallery_gate(
                query,
                query_detections,
                neutral_instances,
                workspace_min=self.backend.config.workspace_min,
                workspace_max=self.backend.config.workspace_max,
                classifier=self.backend.classifier,
                allow_bowl_cavity_merge=cavity_context,
            )
            if query in LIBERO_OBJECT_LABELS:
                from .object_identity import ObjectAliasDetector, object_identity_candidates

                accepted, identity = object_identity_candidates(
                    query, self._dino_cache[query], self.backend.classifier,
                )
                if identity is not None:
                    self._selector_diagnostics.append(identity)
                    if (not accepted and self.box_detector is not None
                            and self.box_extractor is not None):
                        retry = _dino_instances(
                            frames, [query], ObjectAliasDetector(self.box_detector),
                            self.box_extractor, neutral.table_height_m,
                            gallery=self.backend.classifier, search_all_frames=True,
                            provenance_sink=lambda item: self._dino_provenance.setdefault(
                                item.canonical_query, []
                            ).append(item),
                        )
                        retry = _spatial_dino_gallery_gate(
                            query, list(fuse_instances_3d(retry)), neutral_instances,
                            workspace_min=self.backend.config.workspace_min,
                            workspace_max=self.backend.config.workspace_max,
                            classifier=self.backend.classifier,
                        )
                        accepted, retry_identity = object_identity_candidates(
                            query, retry, self.backend.classifier,
                        )
                        if retry_identity is not None:
                            retry_identity['search'] = 'package_aliases_all_cameras'
                            self._selector_diagnostics.append(retry_identity)
                    from .object_top_completion import complete_visible_flat_top

                    completed = []
                    for instance in accepted:
                        item, top_evidence = complete_visible_flat_top(
                            instance, self._fresh_surface_points(instance, query),
                        )
                        completed.append(item)
                        if top_evidence is not None:
                            self._selector_diagnostics.append(top_evidence)
                    self._dino_cache[query] = completed
            cache_timestamps[query] = timestamp_s
            gate_context[query] = cavity_context
        for query in missing:
            candidates = self._dino_cache.get(query, [])
            if candidates:
                by_query[query] = candidates
            elif query in _SPATIAL_GALLERY_LABELS:
                # Frozen-model text grounding can miss small reference objects
                # such as the white ramekin.  A high-confidence static asset
                # match is an independent, sensor-only fallback.
                fallback = _gallery_label_candidates(
                    neutral_instances,
                    query,
                    minimum_score=self.minimum_label_score,
                    require_argmax=query in {"black bowl", "ramekin"},
                )
                if fallback:
                    by_query[query] = fallback
        return neutral, by_query

    def _fresh_surface_points(
        self,
        instance: ObjectInstance,
        label: str,
    ) -> np.ndarray | None:
        """Return current-frame DINO RGB-D points for one ANCHOR object.

        Fused aliases and calibrated views may contribute more than one box.
        Keep only the closest accepted component per camera and concatenate
        the measured world points.  A cached object without matching current
        provenance deliberately returns ``None``; an OBB is not expanded into
        an invented surface cloud.
        """

        rows = getattr(self, "_dino_provenance", {}).get(
            normalize_label(label), ()
        )
        candidates: list[tuple[float, _DINO2DProvenance]] = []
        observed_center = np.asarray(
            instance.observed.center_world,
            dtype=np.float64,
        )
        maximum_distance = min(
            0.060,
            max(0.030, 0.25 * float(np.max(instance.observed.extents_m[:2]))),
        )
        for row in rows:
            if row.instance is None or row.surface_points_world is None:
                continue
            distance = float(
                np.linalg.norm(
                    row.instance.observed.center_world - observed_center
                )
            )
            if distance <= maximum_distance:
                candidates.append((distance, row))
        if not candidates:
            return None

        by_camera: dict[str, tuple[float, _DINO2DProvenance]] = {}
        for distance, row in candidates:
            camera = row.detection.camera_name
            previous = by_camera.get(camera)
            rank = (
                distance,
                -float(row.detection.score),
                -len(row.surface_points_world),
            )
            if previous is None:
                by_camera[camera] = (distance, row)
                continue
            previous_distance, previous_row = previous
            previous_rank = (
                previous_distance,
                -float(previous_row.detection.score),
                -len(previous_row.surface_points_world),
            )
            if rank < previous_rank:
                by_camera[camera] = (distance, row)
        points = np.concatenate(
            [
                row.surface_points_world
                for _, row in sorted(
                    by_camera.values(),
                    key=lambda item: item[1].detection.camera_name,
                )
            ],
            axis=0,
        )
        return np.asarray(points, dtype=np.float64).copy()

    @staticmethod
    def _snapshot(
        timestamp_s: float,
        selected: Mapping[str, ObjectInstance],
        *,
        surface_points_by_instance_id: Mapping[str, np.ndarray] | None = None,
    ):
        from anchor.manipulation.models import SceneObject, SceneSnapshot

        surfaces = surface_points_by_instance_id or {}
        objects = {}
        for query, instance in selected.items():
            center = instance.center_world.copy()
            # Preserve the final fused RGB-D component centroid for broad,
            # shallow placement plates.  Fusion deliberately keeps a semantic
            # completion centre and an observed union-AABB centre; for plates
            # the latter is the directly measured planar target, whereas a
            # crop-centre ray can remain biased after multi-view fusion.
            if normalize_label(query) == "plate":
                center[:2] = instance.observed.center_world[:2]
            half_world = np.abs(instance.axes_world) @ (instance.extents_m / 2.0)
            objects[query] = SceneObject(
                name=query,
                centroid_world=center,
                axes_world=instance.axes_world,
                extents_m=instance.extents_m,
                bounds_min_world=center - half_world,
                bounds_max_world=center + half_world,
                confidence=max(instance.score_for(query), instance.confidence),
                point_count=instance.point_count,
                surface_points_world=surfaces.get(instance.instance_id),
            )
        return SceneSnapshot(float(timestamp_s), objects)

    def _snapshot_with_surface(
        self,
        timestamp_s: float,
        selected: Mapping[str, ObjectInstance],
    ):
        from .prior_consistency import query_supported_completion

        corrected = {}
        for query, instance in selected.items():
            corrected[query], repair = query_supported_completion(
                query, instance, getattr(getattr(self, "backend", None), "classifier", None),
            )
            if repair is not None:
                self._selector_diagnostics.append(repair)
        selected = corrected
        surfaces: dict[str, np.ndarray] = {}
        for query, instance in selected.items():
            points = self._fresh_surface_points(instance, query)
            if points is not None:
                surfaces[instance.instance_id] = points
        return self._snapshot(
            timestamp_s,
            selected,
            surface_points_by_instance_id=surfaces,
        )

    def _identity_selection_score(self, query: str, instance: ObjectInstance) -> float:
        score = float(instance.score_for(query))
        if normalize_label(query) in {"red mug", "white mug", "yellow and white mug"}:
            evidence = _gallery_family_evidence(
                instance, query, getattr(self.backend, "classifier", None),
            )
            if evidence is not None:
                # A support crop can legitimately admit an ambiguous proposal
                # through the geometric gate. At identity selection, retain
                # its own appearance evidence: a marginal text-score lead
                # must not override a visibly different coloured mug.
                score *= evidence.accepted_score / max(evidence.scores.values())
        return score

    def observe(self, observation: Any, queries: Sequence[str]):
        self.last_selector_reference = None
        timestamp_s = self._capture_timestamp_s(observation)
        _, candidates = self._candidates(observation, queries)
        selected: dict[str, ObjectInstance] = {}
        for query, instances in candidates.items():
            previous = self._tracks.get(query)
            nearby = (
                [item for item in instances if np.linalg.norm(item.center_world - previous) <= 0.10]
                if previous is not None
                else []
            )
            choice = (
                min(nearby, key=lambda item: np.linalg.norm(item.center_world - previous))
                if nearby
                else max(instances, key=lambda item: self._identity_selection_score(query, item))
            )
            selected[query] = choice
            self._tracks[query] = choice.center_world.copy()
        return self._snapshot_with_surface(timestamp_s, selected)

    def _view_rank_coordinates(
        self,
        observation: Any,
        candidates: Sequence[ObjectInstance],
    ) -> dict[int, np.ndarray]:
        """Return sensor-frame lateral/depth coordinates for rank words.

        LIBERO's world X/Y axes are not named left/right in instruction
        space.  The fixed agent camera provides the observable convention:
        image-horizontal orders LEFT/RIGHT and optical depth orders
        FRONT/BACK.  This also remains correct if a scene is translated or
        the calibrated camera pose changes.  The fallback is the equivalent
        public LIBERO world-frame convention and exists only for minimal
        adapter tests that omit camera buffers.
        """

        cameras = getattr(observation, "cameras", {})
        ranked_frames: list[tuple[int, str, Any]] = []
        if hasattr(cameras, "items"):
            for name, raw_frame in cameras.items():
                normalized = normalize_label(str(name))
                # Wrist extrinsics move with the robot and therefore cannot
                # define stable language left/right/front/back semantics.
                if "agent" not in normalized:
                    continue
                ranked_frames.append((0, normalized, raw_frame))
        for _priority, name, raw_frame in sorted(
            ranked_frames,
            key=lambda item: (item[0], item[1]),
        ):
            try:
                frame = coerce_rgbd_frame(
                    raw_frame,
                    name=name,
                    timestamp_s=self._capture_timestamp_s(observation),
                )
            except (AttributeError, TypeError, ValueError):
                continue
            rotation = frame.world_from_camera[:3, :3]
            translation = frame.world_from_camera[:3, 3]
            camera_points = np.stack(
                [candidate.center_world - translation for candidate in candidates]
            ) @ rotation
            if not np.all(np.isfinite(camera_points)) or np.any(
                camera_points[:, 2] <= 1e-5
            ):
                continue
            return {
                id(candidate): np.array(
                    (
                        camera_points[index, 0] / camera_points[index, 2],
                        camera_points[index, 2],
                    ),
                    dtype=np.float64,
                )
                for index, candidate in enumerate(candidates)
            }

        # Agent-view calibration in the public LIBERO frame maps image-right
        # to +world-Y and optical-near/front to +world-X.
        return {
            id(candidate): np.array(
                (candidate.center_world[1], -candidate.center_world[0]),
                dtype=np.float64,
            )
            for candidate in candidates
        }

    def observe_selected(self, observation: Any, subject: str, selector: Any):
        """Resolve a compiled language selector against measured multi-instances."""

        self.last_selector_reference = None
        timestamp_s = self._capture_timestamp_s(observation)
        references = tuple(normalize_label(item) for item in selector.references)
        subject = normalize_label(subject)
        neutral, by_query = self._candidates(observation, (subject, *references))
        source_candidates = by_query.get(subject, [])
        relation = getattr(selector.relation, "value", str(selector.relation))
        normalized_relation = normalize_label(relation)
        if normalized_relation in {"left", "right", "front", "back", "middle"}:
            required = 3 if normalized_relation == "middle" else 2
            if len(source_candidates) < required:
                # A ranked phrase is meaningful only relative to other visible
                # instances.  Never turn "right plate" into an arbitrary
                # singleton merely because the other proposal was missed.
                return self._snapshot_with_surface(timestamp_s, {})
            rank_coordinates = self._view_rank_coordinates(
                observation,
                source_candidates,
            )
            if normalized_relation == "left":
                selected = min(
                    source_candidates,
                    key=lambda item: (
                        float(rank_coordinates[id(item)][0]),
                        -float(item.score_for(subject)),
                        item.instance_id,
                    ),
                )
            elif normalized_relation == "right":
                selected = min(
                    source_candidates,
                    key=lambda item: (
                        -float(rank_coordinates[id(item)][0]),
                        -float(item.score_for(subject)),
                        item.instance_id,
                    ),
                )
            elif normalized_relation == "front":
                selected = min(
                    source_candidates,
                    key=lambda item: (
                        float(rank_coordinates[id(item)][1]),
                        -float(item.score_for(subject)),
                        item.instance_id,
                    ),
                )
            elif normalized_relation == "back":
                selected = min(
                    source_candidates,
                    key=lambda item: (
                        -float(rank_coordinates[id(item)][1]),
                        -float(item.score_for(subject)),
                        item.instance_id,
                    ),
                )
            else:
                centres = np.stack(
                    [rank_coordinates[id(item)] for item in source_candidates]
                )
                median_xy = np.median(centres, axis=0)
                selected = min(
                    source_candidates,
                    key=lambda item: (
                        float(
                            np.linalg.norm(
                                rank_coordinates[id(item)] - median_xy
                            )
                        ),
                        -float(item.score_for(subject)),
                        item.instance_id,
                    ),
                )
            self._tracks[subject] = selected.center_world.copy()
            return self._snapshot_with_surface(timestamp_s, {subject: selected})
        if normalize_label(relation) == "on" and references:
            frames = [
                coerce_rgbd_frame(frame, name=name, timestamp_s=timestamp_s)
                for name, frame in observation.cameras.items()
            ]
            recovered = _recover_sources_above_fixture(
                frames,
                tuple(
                    candidate
                    for label in references
                    for candidate in by_query.get(label, ())
                ),
                subject,
                self.backend.classifier,
                workspace_min=self.backend.config.workspace_min,
                workspace_max=self.backend.config.workspace_max,
            )
            if recovered:
                source_candidates = list(
                    fuse_instances_3d((*source_candidates, *recovered))
                )
        if normalize_label(relation) in {"on", "in"} and len(references) == 1:
            reference_candidates = by_query.get(references[0], [])
            if normalize_label(relation) == "on":
                source_candidates, reference_candidates = (
                    _complete_occluded_support_pairs(
                        source_candidates,
                        reference_candidates,
                        subject,
                        references[0],
                        self.backend.classifier,
                    )
                )

            def low_profile_on_fallback():
                if normalize_label(relation) != "on":
                    return None
                frames = [
                    coerce_rgbd_frame(
                        frame,
                        name=name,
                        timestamp_s=timestamp_s,
                    )
                    for name, frame in observation.cameras.items()
                ]
                fallback, diagnostic = _select_source_on_measured_low_profile_support(
                    frames,
                    source_candidates,
                    subject,
                    references[0],
                    self.backend.classifier,
                    table_height_m=neutral.table_height_m,
                )
                history = getattr(self, "_selector_diagnostics", None)
                if history is None:
                    history = []
                    self._selector_diagnostics = history
                history.append(diagnostic)
                return fallback

            def strict_in_fallback():
                if normalize_label(relation) != "in":
                    return None
                provenance_by_query = getattr(self, "_dino_provenance", {})
                provenance = (
                    *provenance_by_query.get(subject, ()),
                    *provenance_by_query.get(references[0], ()),
                )
                fallback = _strict_in_source_from_2d(
                    provenance,
                    subject,
                    references[0],
                    self.backend.classifier,
                    workspace_min=self.backend.config.workspace_min,
                    workspace_max=self.backend.config.workspace_max,
                )
                if fallback is not None and reference_candidates:
                    # A clean source recovered from its own 2-D crop still
                    # needs the independently measured cavity for approach
                    # planning. Reapply the same 3-D containment test; do not
                    # discard an available support merely because the first
                    # fused source proposal was rejected as oversized.
                    try:
                        _, reference = _resolve_support_relation_joint(
                            (fallback,), reference_candidates, "in",
                        )
                    except LookupError:
                        pass
                    else:
                        self.last_selector_reference = reference
                return fallback

            if not source_candidates or not reference_candidates:
                fallback = strict_in_fallback()
                if fallback is None:
                    fallback = low_profile_on_fallback()
                if fallback is None:
                    return self._snapshot_with_surface(timestamp_s, {})
                self._tracks[subject] = fallback.center_world.copy()
                return self._snapshot_with_surface(timestamp_s, {subject: fallback})
            try:
                selected, selected_reference = _resolve_support_relation_joint(
                    source_candidates,
                    reference_candidates,
                    relation,
                )
            except LookupError:
                fallback = strict_in_fallback()
                if fallback is None:
                    fallback = low_profile_on_fallback()
                if fallback is None:
                    return self._snapshot_with_surface(timestamp_s, {})
                self._tracks[subject] = fallback.center_world.copy()
                return self._snapshot_with_surface(timestamp_s, {subject: fallback})
            self._tracks[subject] = selected.center_world.copy()
            self.last_selector_reference = selected_reference
            return self._snapshot_with_surface(timestamp_s, {subject: selected})
        selected_references: dict[str, ObjectInstance] = {}
        for label in references:
            candidates = by_query.get(label, [])
            if not candidates:
                return self._snapshot_with_surface(timestamp_s, {})
            selected_references[label] = max(
                candidates,
                key=lambda item: item.score_for(label),
            )
        source_candidates = _exclude_small_reference_components(
            source_candidates,
            selected_references,
            subject,
        )
        if not source_candidates:
            return self._snapshot_with_surface(timestamp_s, {})
        selected = resolve_geometric_selector(
            source_candidates,
            relation,
            selected_references,
            table_center_xy=(
                self.backend.config.workspace_min[:2] + self.backend.config.workspace_max[:2]
            )
            / 2.0,
        )
        self._tracks[subject] = selected.center_world.copy()
        return self._snapshot_with_surface(timestamp_s, {subject: selected})

    def refine_carried_mug(self, observation: Any, source):
        """Refine one identified held cup from fresh crop points and both depths."""
        if source.name != "yellow and white mug":
            return None
        from .carried_mug import fit_carried_yellow_white_mug
        from ..manipulation.models import SceneObject

        # For this non-Spatial mug label, bound records the current surface
        # provenance while preserving nearest-crop identity selection.
        fresh = self.observe_nearest_bound(
            observation, source.name, source.centroid_world,
        ).objects.get(source.name)
        if fresh is None:
            return None
        frames = [coerce_rgbd_frame(frame, name=name)
                  for name, frame in observation.cameras.items()]
        try:
            fit = fit_carried_yellow_white_mug(frames, fresh)
        except LookupError:
            return None
        if np.linalg.norm(fit.center_world-source.centroid_world) > .08:
            return None
        half = np.abs(fit.axes_world) @ (fit.extents_m / 2)
        refined = SceneObject(
            source.name, fit.center_world, fit.axes_world, fit.extents_m,
            fit.center_world-half, fit.center_world+half,
            source.confidence, fresh.point_count,
        )
        return refined, dict(matched_pixels=fit.matched_pixels,
                             free_space_conflicts=fit.free_space_conflicts,
                             rendered_pixels=fit.rendered_pixels)

    def refine_carried_flat(self, observation: Any, source, robot):
        """Refine one identified held flat package from fresh crop points and both depths."""
        if source.name != "chocolate pudding":
            return None
        from .carried_flat import fit_carried_flat_package
        from ..manipulation.models import SceneObject

        # For this known flat-package label, bound records the current surface
        # provenance while preserving nearest-crop identity selection.
        fresh = self.observe_nearest_bound(
            observation, source.name, source.centroid_world,
        ).objects.get(source.name)
        if fresh is None:
            return None
        frames = [coerce_rgbd_frame(frame, name=name)
                  for name, frame in observation.cameras.items()]
        try:
            fit = fit_carried_flat_package(frames, fresh, robot)
        except LookupError:
            return None
        if np.linalg.norm(fit.center_world-source.centroid_world) > .08:
            return None
        half = np.abs(fit.axes_world) @ (fit.extents_m / 2)
        refined = SceneObject(
            source.name, fit.center_world, fit.axes_world, fit.extents_m,
            fit.center_world-half, fit.center_world+half,
            source.confidence, fresh.point_count,
        )
        return refined, dict(matched_pixels=fit.matched_pixels,
                             free_space_conflicts=fit.free_space_conflicts,
                             rendered_pixels=fit.rendered_pixels)

    def observe_nearest(self, observation: Any, subject: str, point_world: np.ndarray):
        """Reobserve a held object with a deliberately relaxed crop identity.

        A carried object can be absent from support-surface components and its
        crop can include the hand.  This path therefore keeps the historical
        requested-prototype size gate; controller-side held-object mechanics
        decide whether the resulting centre is plausible.
        """

        return self._observe_nearest(
            observation,
            subject,
            point_world,
            require_independent_identity=False,
        )

    def observe_nearest_bound(
        self,
        observation: Any,
        subject: str,
        point_world: np.ndarray,
    ):
        """Reacquire a selector-bound instance after a released grasp retry.

        Unlike held-object reobservation, an ON/IN retry occurs after opening
        the hand and must not turn a nearby ramekin into the selected black
        bowl.  Require an independent argmax over all four frozen Spatial
        gallery classes before nearest-neighbour association.
        """

        return self._observe_nearest(
            observation,
            subject,
            point_world,
            require_independent_identity=True,
        )

    def _observe_nearest(
        self,
        observation: Any,
        subject: str,
        point_world: np.ndarray,
        *,
        require_independent_identity: bool,
    ):
        """Shared RGB-D nearest-neighbour implementation."""

        timestamp_s = self._capture_timestamp_s(observation)
        subject = normalize_label(subject)
        self._dino_cache.pop(subject, None)
        getattr(self, "_dino_raw_cache", {}).pop(subject, None)
        frames = [
            coerce_rgbd_frame(frame, name=name, timestamp_s=timestamp_s)
            for name, frame in observation.cameras.items()
        ]
        neutral = self.backend.observe(frames, requested_labels=(subject,))
        # Once grasped, the object is intentionally rejected by the tabletop
        # support-gap filter.  Requiring a support-surface gallery association
        # here would therefore discard the correct detection.  Re-ground both
        # allowed RGB-D views and select by measured distance to the supplied
        # centre anchor; the frozen gallery shape prior is still applied in
        # _dino_instances.  The bound retry path adds its independent class
        # argmax below, while held reobservation intentionally remains relaxed.
        if require_independent_identity:
            self._dino_provenance = getattr(self, "_dino_provenance", {})
            self._dino_provenance.pop(subject, None)
        candidates = _dino_instances(
            frames,
            (subject,),
            self.box_detector,
            self.box_extractor,
            neutral.table_height_m,
            gallery=self.backend.classifier,
            search_all_frames=True,
            provenance_sink=(lambda item: self._dino_provenance.setdefault(
                item.canonical_query, []).append(item)) if require_independent_identity else None,
        )
        lower = self.backend.config.workspace_min
        upper = self.backend.config.workspace_max
        candidates = [
            item
            for item in candidates
            if item.score_for(subject) > 0
            and np.all(item.center_world >= lower)
            and np.all(item.center_world <= upper)
        ]
        if require_independent_identity and subject in _SPATIAL_GALLERY_LABELS:
            candidates = [
                item
                for item in candidates
                if _independent_spatial_gallery_score(
                    item,
                    subject,
                    self.backend.classifier,
                )
                is not None
            ]
        if not candidates:
            # Gallery fallback still provides an observable failure rather
            # than inventing a held pose when the object is fully occluded.
            _, by_query = self._candidates(observation, (subject,))
            candidates = by_query.get(subject, [])
            if require_independent_identity and subject in _SPATIAL_GALLERY_LABELS:
                candidates = [
                    item
                    for item in candidates
                    if _independent_spatial_gallery_score(
                        item,
                        subject,
                        self.backend.classifier,
                    )
                    is not None
                ]
        if not candidates:
            return self._snapshot_with_surface(timestamp_s, {})
        point = np.asarray(point_world, dtype=np.float64)
        selected = min(candidates, key=lambda item: np.linalg.norm(item.center_world - point))
        self._tracks[subject] = selected.center_world.copy()
        return self._snapshot_with_surface(timestamp_s, {subject: selected})


class PlanningSceneEstimatorAdapter:
    """Implements Planning's provider-style ``SceneEstimator`` protocol."""

    def __init__(
        self,
        frame_provider: Callable[[], Sequence[Any]],
        backend: SensorOnlyScenePerception,
        *,
        box_detector: FrozenBoxDetector | None = None,
        box_extractor: BoxGeometryExtractor | None = None,
    ) -> None:
        self.frame_provider = frame_provider
        self.backend = backend
        self.box_detector = box_detector
        self.box_extractor = box_extractor
        self._tracks: dict[str, dict[str, ObjectInstance]] = {}
        self._next_track_id: dict[str, int] = {}
        self._dino_provenance: dict[str, list[_DINO2DProvenance]] = {}
        self._frame_visible_track_ids: set[str] = set()
        self._last_visible_instance_ids: frozenset[str] = frozenset()
        self.last_selector_view_hint: dict[str, object] | None = None
        self._selector_diagnostics: list[dict[str, Any]] = []

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id.clear()
        self._dino_provenance.clear()
        self._frame_visible_track_ids.clear()
        self._last_visible_instance_ids = frozenset()
        self.last_selector_view_hint = None
        self._selector_diagnostics.clear()

    @property
    def selector_diagnostics(self) -> tuple[dict[str, Any], ...]:
        """Append-only sensor evidence audit for the current adapter episode."""

        return tuple(dict(item) for item in self._selector_diagnostics)

    @property
    def visible_instance_ids(self) -> frozenset[str]:
        """Entity ids backed by an RGB-D component in the latest frame.

        Requested tracks are deliberately retained through short occlusions,
        but those cached entities are not fresh visual evidence.  Consumers
        such as the outer stable tracker use this set to distinguish an
        observed component from an occlusion prior.
        """

        return self._last_visible_instance_ids

    def _tracked_instances(
        self,
        label: str,
        candidates: Sequence[ObjectInstance],
    ) -> list[ObjectInstance]:
        """Nearest-neighbour data association with persistent semantic ids."""

        previous = dict(self._tracks.get(label, {}))
        current = list(candidates)
        associated: dict[str, ObjectInstance] = {}
        if len(previous) == 1 and current:
            # Object/Goal normally has one source per requested label.  Keep the
            # bound id even after a large grasp-induced displacement.
            track_id, old = next(iter(previous.items()))
            chosen_index = min(
                range(len(current)),
                key=lambda index: np.linalg.norm(
                    current[index].center_world - old.center_world
                ),
            )
            associated[track_id] = current.pop(chosen_index)
            previous.pop(track_id)
        else:
            pairs = sorted(
                (
                    float(np.linalg.norm(item.center_world - old.center_world)),
                    track_id,
                    index,
                )
                for track_id, old in previous.items()
                for index, item in enumerate(current)
            )
            used_tracks: set[str] = set()
            used_current: set[int] = set()
            for distance, track_id, index in pairs:
                if distance > 0.18 or track_id in used_tracks or index in used_current:
                    continue
                associated[track_id] = current[index]
                used_tracks.add(track_id)
                used_current.add(index)
            previous = {key: value for key, value in previous.items() if key not in used_tracks}
            current = [item for index, item in enumerate(current) if index not in used_current]
        fresh_track_ids = set(associated)
        # Retain unmatched requested tracks through gripper occlusion.  Planning
        # uses their extent and grasp binding during lift and transfer.
        associated.update(previous)
        for item in current:
            counter = self._next_track_id.get(label, 0)
            self._next_track_id[label] = counter + 1
            track_id = f"tracked-{label.replace(' ', '-')}-{counter:02d}"
            associated[track_id] = item
            fresh_track_ids.add(track_id)
        self._frame_visible_track_ids.update(fresh_track_ids)
        tracked = {
            track_id: self._clone_track(track_id, label, item)
            for track_id, item in associated.items()
        }
        self._tracks[label] = tracked
        return list(tracked.values())

    @staticmethod
    def _clone_track(track_id: str, label: str, item: ObjectInstance) -> ObjectInstance:
        center = item.center_world.copy()
        grasp = item.grasp_point_world.copy()
        if normalize_label(label) == "plate":
            center[:2] = item.observed.center_world[:2]
            grasp[:2] = item.observed.center_world[:2]
        return ObjectInstance(
            instance_id=track_id,
            center_world=center,
            grasp_point_world=grasp,
            axes_world=item.axes_world,
            extents_m=item.extents_m,
            observed=item.observed,
            label_scores={label: max(item.score_for(label), item.confidence)},
            confidence=item.confidence,
            point_count=item.point_count,
            color_histogram=item.color_histogram,
        )

    def _fresh_surface_points(
        self,
        instance: ObjectInstance,
        label: str,
    ) -> np.ndarray | None:
        """Return current-frame RGB-D points for one accepted DINO entity.

        DINO aliases and the two calibrated cameras can produce several boxes
        for one object.  Select at most one accepted component per camera by
        observed 3-D proximity, then fuse their public world points.  Cached
        tracks and legacy extractor doubles carry no such evidence and return
        ``None`` rather than an invented OBB surface.
        """

        per_camera = self._fresh_surface_points_by_camera(instance, label)
        if not per_camera:
            return None
        return np.concatenate([points for _, points in per_camera], axis=0)

    def _fresh_surface_points_by_camera(
        self,
        instance: ObjectInstance,
        label: str,
    ) -> tuple[tuple[str, np.ndarray], ...]:
        """Return current raw DINO RGB-D support without losing camera origin."""

        if (
            instance.instance_id.startswith("tracked-")
            and instance.instance_id not in self._frame_visible_track_ids
        ):
            return ()
        rows = self._dino_provenance.get(normalize_label(label), ())
        candidates: list[tuple[float, _DINO2DProvenance]] = []
        observed_center = np.asarray(
            instance.observed.center_world,
            dtype=np.float64,
        )
        maximum_distance = min(
            0.060,
            max(0.030, 0.25 * float(np.max(instance.observed.extents_m[:2]))),
        )
        for row in rows:
            if row.instance is None or row.surface_points_world is None:
                continue
            distance = float(
                np.linalg.norm(
                    row.instance.observed.center_world - observed_center
                )
            )
            if distance <= maximum_distance:
                candidates.append((distance, row))
        if not candidates:
            return ()

        by_camera: dict[str, tuple[float, _DINO2DProvenance]] = {}
        for distance, row in candidates:
            camera = row.detection.camera_name
            previous = by_camera.get(camera)
            rank = (
                distance,
                -float(row.detection.score),
                -len(row.surface_points_world),
            )
            if previous is None:
                by_camera[camera] = (distance, row)
                continue
            previous_distance, previous_row = previous
            previous_rank = (
                previous_distance,
                -float(previous_row.detection.score),
                -len(previous_row.surface_points_world),
            )
            if rank < previous_rank:
                by_camera[camera] = (distance, row)
        return tuple(
            (
                row.detection.camera_name,
                np.asarray(row.surface_points_world, dtype=np.float64).copy(),
            )
            for _, row in sorted(
                by_camera.values(),
                key=lambda item: item[1].detection.camera_name,
            )
        )

    def observe(self, requested_labels: Sequence[str]):
        return self._observe(requested_labels, source_reference=None)

    def observe_source_selector(
        self,
        requested_labels: Sequence[str],
        source_reference: Any,
    ):
        """Observe with one language-derived source selector for this frame.

        The explicit call prevents a plain label list from authorising an ON
        relation or changing object geometry.  No selector context is retained
        after the synchronous observation.
        """

        return self._observe(requested_labels, source_reference=source_reference)

    def _observe(
        self,
        requested_labels: Sequence[str],
        *,
        source_reference: Any | None,
    ):
        from anchor.planning.perception import (
            SceneEntity,
            SceneEstimate,
            SupportRelationEvidence,
            TargetRegion,
        )
        from anchor.planning.schema import EntityRef, Relation
        from anchor.planning.sdf import BoxSDF, CompositeSDF, EmptySDF

        # A hint is valid only for this exact observation.  Clear it before any
        # detector/backend work so exceptions cannot leak the previous frame.
        self.last_selector_view_hint = None
        self._frame_visible_track_ids.clear()
        self._last_visible_instance_ids = frozenset()
        frames = [coerce_rgbd_frame(frame) for frame in self.frame_provider()]
        normalized_requested = [normalize_label(query) for query in requested_labels]
        support_relation_request: tuple[str, str] | None = None
        if source_reference is not None:
            if not isinstance(source_reference, EntityRef):
                raise TypeError("source selector observation requires an EntityRef")
            if source_reference.role != "source":
                raise ValueError("source selector observation requires source role")
            selector = source_reference.selector
            if selector is not None and selector.relation is Relation.ON:
                if len(selector.references) != 1:
                    raise ValueError(
                        "ON source selector observation requires exactly one reference"
                    )
                source_label = normalize_label(source_reference.label)
                reference_label = normalize_label(selector.references[0])
                if (
                    source_label not in normalized_requested
                    or reference_label not in normalized_requested
                ):
                    raise ValueError(
                        "source selector labels must be present in requested_labels"
                    )
                support_relation_request = (source_label, reference_label)
        gallery_queries = _neutral_gallery_queries(normalized_requested)
        neutral = self.backend.observe(frames, requested_labels=gallery_queries)
        supported = set(self.backend.classifier.prototypes) if self.backend.classifier is not None else set()
        force_dino = _DINO_GALLERY_LABELS
        # A package inside a receptacle can lose its tabletop connected
        # component while remaining visible from the wrist.  Supported
        # gallery vocabulary alone must not suppress the detector fallback.
        unsegmented_labels = {
            label
            for label in normalized_requested
            if label in supported
            and label not in force_dino
            and self._tracks.get(label)
            and not any(
                instance.score_for(label)
                >= max(0.04, 0.72 * max(instance.label_scores.values(), default=0.0))
                and any(
                    np.linalg.norm(instance.center_world - previous.center_world) <= .09
                    for previous in self._tracks[label].values()
                )
                for instance in neutral.instances
            )
        }
        missing = [
            label
            for label in requested_labels
            if normalize_label(label) not in supported
            or normalize_label(label) in force_dino
            or normalize_label(label) in unsegmented_labels
        ]
        self._dino_provenance.clear()
        dino = _dino_instances(
            frames,
            missing,
            self.box_detector,
            self.box_extractor,
            neutral.table_height_m,
            gallery=self.backend.classifier,
            search_all_frames=tuple(dict.fromkeys((
                *_all_view_spatial_queries(missing),
                *(label for label in missing if normalize_label(label) == "basket"),
                *(label for label in normalized_requested if label in unsegmented_labels),
            ))),
            # Planning's narrowly scoped released-hand self-field proof needs
            # independent raw RGB-D support from both calibrated views.  The
            # extra drawer/cabinet crops are provenance only: they cannot add
            # an entity, obstacle, or fusion vote to the ordinary scene.
            provenance_only_secondary_queries=tuple(
                query
                for query in missing
                if {"drawer", "cabinet"}
                & set(normalize_label(query).split())
            ),
            provenance_sink=lambda item: self._dino_provenance.setdefault(
                item.canonical_query, []
            ).append(item),
        )
        relabelled = _relabel_spatial_crop_proposals(
            dino,
            self.backend.classifier,
            workspace_min=self.backend.config.workspace_min,
            workspace_max=self.backend.config.workspace_max,
        )
        # A relabelled item is a second semantic hypothesis for an existing
        # measured DINO crop, not another physical observation.  Keep those
        # hypotheses available for requested-label tracking, while the raw
        # physical pool below contains each detector crop only once.
        semantic_hypotheses = [*dino, *relabelled]

        # Strict nested 2-D evidence may identify one measured source even
        # when a partial fixture crop has no trustworthy 3-D OBB.  Preserve it
        # only as an active-view hint: no fixture geometry is created here and
        # the normal Planning resolver remains responsible for accepting a
        # later, fresh 3-D source/reference pair.  Multiple matching
        # source/reference hypotheses are deliberately treated as ambiguous.
        provenance = tuple(
            item
            for rows in self._dino_provenance.values()
            for item in rows
        )
        strict_hints: list[dict[str, object]] = []
        for source_label in normalized_requested:
            if source_label not in _SPATIAL_GALLERY_LABELS:
                continue
            for reference_label in normalized_requested:
                if not any(
                    token in reference_label for token in _IN_FIXTURE_TOKENS
                ):
                    continue
                source = _strict_in_source_from_2d(
                    provenance,
                    source_label,
                    reference_label,
                    self.backend.classifier,
                    workspace_min=self.backend.config.workspace_min,
                    workspace_max=self.backend.config.workspace_max,
                )
                if source is None:
                    continue
                strict_hints.append(
                    {
                        "relation": "in",
                        "source_label": source_label,
                        "reference_label": reference_label,
                        "source_center_world": source.center_world.copy(),
                        "timestamp_s": float(neutral.timestamp_s),
                        "capture_id": neutral.capture_id,
                    }
                )
        if len(strict_hints) == 1:
            self.last_selector_view_hint = strict_hints[0]

        spatial_by_label: dict[str, list[ObjectInstance]] = {}
        for label in normalized_requested:
            raw = list(
                fuse_instances_3d(
                    item
                    for item in semantic_hypotheses
                    if item.score_for(label) > 0
                )
            )
            gated = _fixture_size_gate(label, raw)
            if not gated:
                gated = _ambiguous_fixture_fallback(label, raw)
            gated = _spatial_dino_gallery_gate(
                label,
                gated,
                neutral.instances,
                workspace_min=self.backend.config.workspace_min,
                workspace_max=self.backend.config.workspace_max,
                classifier=self.backend.classifier,
            )
            if not gated and label in force_dino:
                # Mirror ANCHOR's sensor-only fallback for small references:
                # DINO may miss the white ramekin even though the frozen asset
                # gallery has a strong, shape-consistent RGB-D component.
                # These instances are still cloned into label-specific tracks
                # below; generic neutral entities are never exposed directly.
                prototype = (
                    self.backend.classifier.prototypes.get(label)
                    if self.backend.classifier is not None
                    else None
                )
                fallback = _gallery_label_candidates(
                    neutral.instances,
                    label,
                    minimum_score=0.04,
                    require_argmax=label in {"black bowl", "ramekin"},
                )
                if prototype is not None:
                    fallback = [
                        instance
                        for instance in fallback
                        if self.backend.classifier._shape_similarity(
                            instance.observed.extents_m,
                            prototype.dimensions_xyz_m,
                        )
                        >= 0.30
                    ]
                gated = fallback
            spatial_by_label[label] = gated

        # Complete a fully occluded small support only when a requested source
        # crop carries the measured two-layer height signature.  This happens
        # before top-plane recovery so a distant texture false-positive cannot
        # seed a self-consistent but unrelated source/support pair.
        small_support_labels = {"cookie box", "ramekin"}
        if "black bowl" in normalized_requested:
            for support_label in normalized_requested:
                if support_label not in small_support_labels:
                    continue
                completed_sources, completed_supports = (
                    _complete_occluded_support_pairs(
                        spatial_by_label.get("black bowl", ()),
                        spatial_by_label.get(support_label, ()),
                        "black bowl",
                        support_label,
                        self.backend.classifier,
                    )
                )
                spatial_by_label["black bowl"] = completed_sources
                spatial_by_label[support_label] = completed_supports

        # Planning requests source-selector references alongside source/target.
        # Recover a missed gallery source from the measured top plane of any
        # true-sized requested fixture, using the same sensor-only helper as B.
        fixture_tokens = ("cabinet", "drawer", "stove", "microwave", "rack")
        large_references = tuple(
            candidate
            for label in normalized_requested
            if any(token in label for token in fixture_tokens)
            for candidate in spatial_by_label.get(label, ())
        )
        # Cookie boxes and ramekins are legitimate language-selected support
        # surfaces, but unlike cabinets they must pass their label-specific
        # frozen gallery score before opening a small top-plane search prism.
        # Plate is intentionally excluded: it is the placement target in these
        # tasks, not evidence for which black bowl the source selector names.
        small_references = tuple(
            candidate
            for label in normalized_requested
            if label in small_support_labels
            for candidate in spatial_by_label.get(label, ())
            if candidate.score_for(label) >= 0.55
        )
        for source_label in normalized_requested:
            if source_label != "black bowl":
                continue
            recovered: list[ObjectInstance] = []
            if large_references:
                recovered.extend(
                    _recover_sources_above_fixture(
                        frames,
                        large_references,
                        source_label,
                        self.backend.classifier,
                        workspace_min=self.backend.config.workspace_min,
                        workspace_max=self.backend.config.workspace_max,
                    )
                )
            if small_references:
                recovered.extend(
                    _recover_sources_above_fixture(
                        frames,
                        small_references,
                        source_label,
                        self.backend.classifier,
                        workspace_min=self.backend.config.workspace_min,
                        workspace_max=self.backend.config.workspace_max,
                        minimum_reference_planar_m=0.060,
                    )
                )
            if recovered:
                spatial_by_label[source_label] = list(
                    fuse_instances_3d(
                        (*spatial_by_label.get(source_label, ()), *recovered)
                    )
                )

        # Only independently measured components may become generic entities
        # or obstacle fields.  Relabelled hypotheses share a crop and
        # provenance with ``dino`` and must never duplicate its physical SDF.
        all_instances = [*neutral.instances, *dino]
        tracked_requested: list[ObjectInstance] = []
        for label in normalized_requested:
            if label in force_dino:
                candidates = spatial_by_label.get(label, [])
            else:
                candidates = [
                    instance
                    for instance in neutral.instances
                    if instance.score_for(label)
                    >= max(
                        0.04,
                        0.72 * max(instance.label_scores.values(), default=0.0),
                    )
                ]
                candidates = list(
                    fuse_instances_3d(
                        (*candidates, *spatial_by_label.get(label, ()))
                    )
                )
            tracked_requested.extend(self._tracked_instances(label, candidates))

        # A shallow source can fully occlude a low-profile support such as a
        # stove in both top-down crops.  Planning must not invent the missing
        # fixture OBB: when no fresh reference track exists, retain a typed,
        # observation-scoped proof on the already visible source instead.  The
        # downstream resolver consumes it only after its ordinary 3-D
        # source/reference path has failed.  All geometry here comes from this
        # exact dual RGB-D frame; the public gallery supplies only frozen shape
        # scale and an independent stove-vs-tray semantic comparison.
        support_evidence: list[SupportRelationEvidence] = []
        fresh_ids = set(self._frame_visible_track_ids)
        tracked_by_label = {
            label: [
                item
                for item in tracked_requested
                if normalize_label(item.label) == label
                and item.instance_id in fresh_ids
            ]
            for label in normalized_requested
        }
        if support_relation_request is not None:
            source_label, reference_label = support_relation_request
            prototypes = (
                self.backend.classifier.prototypes
                if self.backend.classifier is not None
                else {}
            )
            prototype = prototypes.get(reference_label)
            is_low_profile = bool(
                prototype is not None
                and _low_profile_fixture_dimensions(
                    reference_label,
                    np.asarray(prototype.dimensions_xyz_m, dtype=np.float64),
                )
                is not None
            )
            # A freshly measured reference entity leaves semantic resolution to
            # the normal joint-pair path.  This also avoids paying for a dense
            # support-plane search on ordinary, unobscured observations.
            if is_low_profile and not tracked_by_label.get(reference_label):
                sources = tracked_by_label.get(source_label, ())
                if sources:
                    selected, diagnostic = (
                        _select_source_on_measured_low_profile_support(
                            frames,
                            sources,
                            source_label,
                            reference_label,
                            self.backend.classifier,
                            table_height_m=neutral.table_height_m,
                        )
                    )
                    diagnostic = {
                        **diagnostic,
                        "route": "c",
                        "timestamp_s": float(neutral.timestamp_s),
                    }
                    self._selector_diagnostics.append(diagnostic)
                    if selected is not None:
                        try:
                            support_height = float(
                                diagnostic["measured_support_height_m"]
                            )
                        except (KeyError, TypeError, ValueError):
                            # The helper's accepted result and its measured plane
                            # must remain atomic; never emit identity-only evidence.
                            selected = None
                        matching_indices = (
                            [
                                index
                                for index, item in enumerate(tracked_requested)
                                if item.instance_id == selected.instance_id
                                and normalize_label(item.label) == source_label
                            ]
                            if selected is not None
                            else []
                        )
                        if selected is not None and len(matching_indices) == 1:
                            tracked_requested[matching_indices[0]] = selected
                            self._tracks[source_label][selected.instance_id] = selected
                            support_evidence.append(
                                SupportRelationEvidence(
                                    timestamp_s=float(neutral.timestamp_s),
                                    relation=Relation.ON,
                                    source_label=source_label,
                                    reference_label=reference_label,
                                    source_instance_id=selected.instance_id,
                                    support_point_world=np.array(
                                        (
                                            selected.center_world[0],
                                            selected.center_world[1],
                                            support_height,
                                        ),
                                        dtype=np.float64,
                                    ),
                                    capture_id=neutral.capture_id,
                                    camera_capture_ids=neutral.camera_capture_ids,
                                )
                            )
        # Do not duplicate the sensor component underlying a tracked entity.
        generic_instances = [
            instance
            for instance in all_instances
            if not any(
                (
                    instance.label == tracked.label
                    and np.linalg.norm(instance.center_world - tracked.center_world) < 0.045
                )
                or np.linalg.norm(instance.center_world - tracked.center_world) < 0.025
                for tracked in tracked_requested
            )
        ]
        entity_instances = tracked_requested + generic_instances
        self._last_visible_instance_ids = frozenset(
            (*self._frame_visible_track_ids, *(item.instance_id for item in generic_instances))
        )
        entities = []
        requested_set = {normalize_label(item) for item in requested_labels}
        fixed_view = next(
            (
                frame
                for frame in frames
                if getattr(frame, "camera_name", "") == "agentview"
            ),
            frames[0],
        )
        view_origin = np.asarray(
            fixed_view.world_from_camera[:3, 3], dtype=np.float64
        )
        view_right_axis = np.asarray(
            fixed_view.world_from_camera[:3, 0], dtype=np.float64
        )
        view_forward_axis = np.asarray(
            fixed_view.world_from_camera[:3, 2], dtype=np.float64
        )
        for instance in entity_instances:
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = instance.axes_world
            pose[:3, 3] = instance.center_world
            label = instance.label
            surface_points_world = self._fresh_surface_points(instance, label)
            world_height = float(
                np.abs(instance.axes_world[2, :]) @ instance.extents_m
            )
            region = TargetRegion(
                center=instance.center_world,
                axes=instance.axes_world,
                half_extents=instance.extents_m / 2.0,
                surface_normal=np.array([0.0, 0.0, 1.0]),
            )
            entities.append(
                SceneEntity(
                    instance_id=instance.instance_id,
                    label=label,
                    pose=pose,
                    extent=instance.extents_m,
                    confidence=instance.confidence,
                    keypoints={
                        "centroid": instance.center_world,
                        "top": instance.grasp_point_world,
                        "grasp": instance.grasp_point_world,
                        "bottom": instance.center_world
                        - np.array([0.0, 0.0, world_height / 2.0]),
                        # Public camera calibration fixes front/back and
                        # left/right semantics for target-local subregions.
                        "view_origin": view_origin,
                        "view_right_axis": view_right_axis,
                        "view_forward_axis": view_forward_axis,
                    },
                    region=region,
                    surface_points_world=surface_points_world,
                )
            )
        requested_instances = tracked_requested
        requested_identity = {id(instance) for instance in requested_instances}
        fields = []
        for instance in generic_instances:
            if id(instance) in requested_identity:
                continue
            too_close = False
            for requested in requested_instances:
                extent_clearance = 0.5 * float(
                    np.max(instance.extents_m[:2]) + np.max(requested.extents_m[:2])
                )
                if np.linalg.norm(instance.center_world - requested.center_world) < max(
                    0.08, extent_clearance
                ):
                    too_close = True
                    break
            if not too_close:
                camera_surfaces = self._fresh_surface_points_by_camera(
                    instance, instance.label
                )
                surface_points = (
                    np.concatenate(
                        [points for _, points in camera_surfaces], axis=0
                    )
                    if camera_surfaces
                    else None
                )
                fields.append(
                    BoxSDF(
                        instance.center_world,
                        instance.extents_m / 2.0,
                        instance.axes_world,
                        source_instance_id=instance.instance_id,
                        source_label=normalize_label(instance.label),
                        surface_points_world=surface_points,
                        surface_points_by_camera=camera_surfaces,
                    )
                )
        obstacle_sdf = CompositeSDF(fields) if fields else EmptySDF()
        return SceneEstimate(
            timestamp_s=neutral.timestamp_s,
            entities=tuple(entities),
            obstacle_sdf=obstacle_sdf,
            workspace_min=self.backend.config.workspace_min,
            workspace_max=self.backend.config.workspace_max,
            scene_floor_z=neutral.table_height_m,
            support_relation_evidence=tuple(support_evidence),
            capture_id=neutral.capture_id,
            camera_capture_ids=neutral.camera_capture_ids,
        )


def make_default_perception(
    *,
    asset_root: str = "/home/kzoacn/.cache/libero/assets",
) -> SensorOnlyScenePerception:
    """Construct the CPU geometry + static Object-gallery backend."""

    from .gallery import TextureSizeGallery

    return SensorOnlyScenePerception(classifier=TextureSizeGallery.from_libero_assets(asset_root))
