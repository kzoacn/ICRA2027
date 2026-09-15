"""Check a selected package's completed dimensions against its own visible body."""
import numpy as np
from .gallery import LIBERO_OBJECT_LABELS
from .schema import normalize_label


def query_supported_completion(query, instance, classifier):
    """Repair a too-short competing prior after the component has been selected.

    A near-tied colour classification can supply one package's dimensions to
    a component subsequently selected under another language label. Require
    the raw RGB-D height and planar coverage to support the selected label's
    public shape before replacing that inconsistent completion. Identity,
    horizontal position, appearance scores and observed geometry stay fixed.
    """
    query = normalize_label(query)
    prototypes = getattr(classifier, 'prototypes', {})
    if query not in LIBERO_OBJECT_LABELS or query not in prototypes or instance.point_count < 100:
        return instance, None
    observed = np.asarray(instance.observed.extents_m, dtype=float)
    completed = np.asarray(instance.extents_m, dtype=float)
    observed_height = float(observed[2])
    if completed[2] >= .80 * observed_height or observed_height - completed[2] < .020:
        return instance, None
    maximum = max(instance.label_scores.values(), default=0.)
    if maximum <= 0 or instance.score_for(query) < .90 * maximum:
        return instance, None
    raw = classifier.orient_dimensions(prototypes[query].dimensions_xyz_m, observed)
    expected = np.array([*sorted(raw[:2], reverse=True), raw[2]], dtype=float)
    planar_coverage = np.sort(observed[:2])[::-1] / expected[:2]
    if (not .85 <= observed_height / expected[2] <= 1.15
            or np.any(planar_coverage < .65) or np.any(planar_coverage > 1.60)):
        return instance, None
    center = instance.center_world.copy()
    bottom = float(instance.observed.bounds_min_world[2])
    center[2] = bottom + expected[2] / 2
    grasp = instance.grasp_point_world.copy(); grasp[2] = bottom + expected[2]
    corrected = instance.with_semantics(instance.label_scores, center_world=center,
        grasp_point_world=grasp, extents_m=expected, confidence=instance.confidence)
    return corrected, dict(kind='query_supported_shape_completion', query=query,
        observed_height_m=observed_height, old_extents_m=completed.tolist(),
        selected_extents_m=expected.tolist(), relative_label_score=float(instance.score_for(query)/maximum))
