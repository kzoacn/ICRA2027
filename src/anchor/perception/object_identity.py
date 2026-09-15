"""Independent package identity evidence for detector fallback proposals."""
from dataclasses import dataclass, replace
from .gallery import LIBERO_OBJECT_LABELS
from .schema import normalize_label

OBJECT_IDENTITY_FAMILY = tuple(sorted(LIBERO_OBJECT_LABELS | {'basket'}))
OBJECT_SEARCH_ALIASES = {
    'alphabet soup': ('alphabet soup can', 'soup can'),
    'cream cheese': ('cream cheese box', 'cream cheese package'),
    'salad dressing': ('salad dressing bottle',),
    'bbq sauce': ('barbecue sauce bottle', 'bbq sauce bottle'),
    'ketchup': ('ketchup bottle',),
    'tomato sauce': ('tomato sauce can',),
    'butter': ('butter box', 'butter package'),
    'milk': ('milk carton',),
    'chocolate pudding': ('chocolate pudding box', 'pudding package'),
    'orange juice': ('orange juice carton',),
}


def object_identity_candidates(query, instances, classifier):
    """Keep detector crops supported by their own complete gallery comparison.

    Text detection confidence and an independent appearance probability have
    different meanings. Compare every crop's observed colour and geometry
    against the same full family before retaining its language identity.
    Existing neutral-gallery selection happens before this fallback helper.
    A missing full gallery preserves the legacy adapter's optional backend.
    """
    query = normalize_label(query)
    if (query not in LIBERO_OBJECT_LABELS
            or not set(OBJECT_IDENTITY_FAMILY) <= set(getattr(classifier, 'prototypes', {}))):
        return list(instances), None
    accepted, evidence = [], []
    for instance in instances:
        if instance.color_histogram is None:
            evidence.append(dict(instance_id=instance.instance_id, reason='missing_rgb_histogram'))
            continue
        result = classifier.classify(instance, allowed_labels=OBJECT_IDENTITY_FAMILY)
        keep = result.label == query
        evidence.append(dict(instance_id=instance.instance_id,
            top_label=result.label, query_score=float(result.scores[query]), accepted=keep))
        if keep:
            # Keep the detector's measured geometry and query-specific physical
            # completion. Ranking now uses comparable full-family probabilities.
            accepted.append(instance.with_semantics(dict(result.scores)))
    return accepted, dict(kind='object_detector_identity', query=query,
        input_count=len(instances), accepted_count=len(accepted), candidates=evidence)


@dataclass
class ObjectAliasDetector:
    """Expand package nouns only for an explicit failed-identity search."""
    detector: object

    def detect(self, frame, queries):
        expanded, canonical = [], {}
        for raw in queries:
            query = normalize_label(raw)
            for label in (query, *OBJECT_SEARCH_ALIASES.get(query, ())):
                if label not in canonical:
                    canonical[label] = query
                    expanded.append(label)
        return tuple(replace(item, query=canonical.get(normalize_label(item.query), item.query))
                     for item in self.detector.detect(frame, expanded))
