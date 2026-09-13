"""Sensor-derived cross-camera instance fusion and geometric selectors."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

from .schema import ObjectInstance, ObservedGeometry, normalize_label


def fuse_instances_3d(
    instances: Sequence[ObjectInstance],
    *,
    centre_radius_m: float = 0.045,
) -> tuple[ObjectInstance, ...]:
    """Fuse same-label boxes reconstructed independently from multiple cameras.

    This is 3-D clustering, not image NMS: detections may have no image overlap
    across agent and wrist cameras.  The 4.5 cm default stays below the normal
    spacing between distinct LIBERO objects while covering partial-view centre
    errors.
    """

    if centre_radius_m <= 0:
        raise ValueError("centre_radius_m must be positive")
    by_label: dict[str, list[ObjectInstance]] = defaultdict(list)
    for instance in instances:
        by_label[instance.label].append(instance)
    fused: list[ObjectInstance] = []
    for label, group in by_label.items():
        unused = set(range(len(group)))
        clusters: list[list[int]] = []
        while unused:
            seed = unused.pop()
            cluster = {seed}
            frontier = [seed]
            while frontier:
                current = frontier.pop()
                neighbours = [
                    index
                    for index in unused
                    if np.linalg.norm(
                        group[index].center_world - group[current].center_world
                    )
                    <= centre_radius_m
                ]
                for index in neighbours:
                    unused.remove(index)
                    cluster.add(index)
                    frontier.append(index)
            clusters.append(sorted(cluster))
        for cluster_index, indices in enumerate(clusters):
            members = [group[index] for index in indices]
            fused.append(_merge_members(label, cluster_index, members))
    fused.sort(key=lambda item: (item.label, item.center_world[0], item.center_world[1]))
    return tuple(fused)


def _merge_members(label: str, index: int, members: Sequence[ObjectInstance]) -> ObjectInstance:
    if len(members) == 1:
        member = members[0]
        return ObjectInstance(
            instance_id=f"fused-{label.replace(' ', '-')}-{index:02d}",
            center_world=member.center_world,
            grasp_point_world=member.grasp_point_world,
            axes_world=member.axes_world,
            extents_m=member.extents_m,
            observed=member.observed,
            label_scores=member.label_scores,
            confidence=member.confidence,
            point_count=member.point_count,
            color_histogram=member.color_histogram,
        )
    weights = np.array(
        [max(member.confidence, 0.05) * np.sqrt(member.point_count) for member in members],
        dtype=np.float64,
    )
    weights /= weights.sum()
    centre = np.sum([weight * member.center_world for weight, member in zip(weights, members)], axis=0)
    grasp = np.sum(
        [weight * member.grasp_point_world for weight, member in zip(weights, members)], axis=0
    )
    extents = np.max([member.extents_m for member in members], axis=0)
    representative = max(members, key=lambda member: member.confidence * np.sqrt(member.point_count))
    if all(np.allclose(np.sort(member.extents_m), np.sort(representative.extents_m),
                       atol=.002, rtol=.03) for member in members):
        # These views carry the same rigid size prior in different axis
        # permutations. Componentwise maxima duplicate the longest axis and
        # invent a larger object. Keep dimensions with their chosen frame.
        extents = representative.extents_m.copy()
    observed_lower = np.min([member.observed.bounds_min_world for member in members], axis=0)
    observed_upper = np.max([member.observed.bounds_max_world for member in members], axis=0)
    observed = ObservedGeometry(
        center_world=(observed_lower + observed_upper) / 2.0,
        axes_world=representative.observed.axes_world,
        extents_m=np.maximum(observed_upper - observed_lower, 0.002),
        bounds_min_world=observed_lower,
        bounds_max_world=observed_upper,
    )
    labels = set().union(*(member.label_scores for member in members))
    scores = {name: max(member.label_scores.get(name, 0.0) for member in members) for name in labels}
    # Scores are independent detector confidences, so they need not sum to one.
    histogram = None
    available = [(weight, member.color_histogram) for weight, member in zip(weights, members) if member.color_histogram is not None]
    if available:
        histogram = sum(weight * value for weight, value in available)
        histogram /= histogram.sum()
    return ObjectInstance(
        instance_id=f"fused-{label.replace(' ', '-')}-{index:02d}",
        center_world=centre,
        grasp_point_world=grasp,
        axes_world=representative.axes_world,
        extents_m=extents,
        observed=observed,
        label_scores=scores,
        confidence=max(member.confidence for member in members),
        point_count=sum(member.point_count for member in members),
        color_histogram=histogram,
    )


def resolve_geometric_selector(
    candidates: Sequence[ObjectInstance],
    relation: str,
    references: Mapping[str, ObjectInstance],
    *,
    table_center_xy: np.ndarray = np.zeros(2),
) -> ObjectInstance:
    """Resolve Route-B-style BETWEEN/NEXT_TO/CENTER/ON/IN from measured 3-D."""

    if not candidates:
        raise LookupError("selector has no source candidates")
    normalized_relation = normalize_label(relation)
    if normalized_relation == "center":
        centre = np.asarray(table_center_xy, dtype=np.float64)
        return min(candidates, key=lambda item: np.linalg.norm(item.center_world[:2] - centre))
    refs = list(references.values())
    small_labels = {"black bowl", "plate", "ramekin", "cookie box"}
    small_references = [
        reference
        for reference in refs
        if normalize_label(reference.label) in small_labels
    ]
    if small_references:
        distinct: list[ObjectInstance] = []
        for candidate in candidates:
            overlaps_reference = False
            for reference in small_references:
                if normalize_label(candidate.label) == normalize_label(reference.label):
                    continue
                intersection_min = np.maximum(
                    candidate.observed.bounds_min_world,
                    reference.observed.bounds_min_world,
                )
                intersection_max = np.minimum(
                    candidate.observed.bounds_max_world,
                    reference.observed.bounds_max_world,
                )
                intersection_volume = float(
                    np.prod(np.maximum(intersection_max - intersection_min, 0.0))
                )
                candidate_volume = float(
                    np.prod(np.maximum(candidate.observed.extents_m, 1e-6))
                )
                reference_volume = float(
                    np.prod(np.maximum(reference.observed.extents_m, 1e-6))
                )
                overlap = intersection_volume / max(
                    min(candidate_volume, reference_volume),
                    1e-12,
                )
                distance = float(
                    np.linalg.norm(candidate.center_world - reference.center_world)
                )
                if overlap >= 0.35 or distance <= 0.025:
                    overlaps_reference = True
                    break
            if not overlaps_reference:
                distinct.append(candidate)
        if not distinct:
            raise LookupError("selector source candidates alias a small-object reference")
        candidates = distinct
    if normalized_relation == "between":
        if len(refs) != 2:
            raise ValueError("between selector requires two references")
        target = (refs[0].center_world + refs[1].center_world) / 2.0
        return min(candidates, key=lambda item: np.linalg.norm(item.center_world[:2] - target[:2]))
    if len(refs) != 1:
        raise ValueError(f"{normalized_relation} selector requires one reference")
    reference = refs[0]
    if normalized_relation == "next to":
        return min(
            candidates,
            key=lambda item: np.linalg.norm(item.center_world[:2] - reference.center_world[:2]),
        )
    if normalized_relation in {"on", "in"}:
        def relation_cost(item: ObjectInstance) -> float:
            local = (item.center_world - reference.center_world) @ reference.axes_world
            planar_outside = np.maximum(
                np.abs(local[:2]) - reference.extents_m[:2] / 2.0,
                0.0,
            )
            if normalized_relation == "on":
                expected_z = reference.center_world[2] + reference.extents_m[2] / 2.0
                vertical = abs((item.center_world[2] - item.extents_m[2] / 2.0) - expected_z)
            else:
                vertical = max(abs(local[2]) - reference.extents_m[2] / 2.0, 0.0)
            return float(np.linalg.norm(planar_outside) + vertical)

        return min(candidates, key=relation_cost)
    raise ValueError(f"unsupported geometric selector: {relation!r}")
