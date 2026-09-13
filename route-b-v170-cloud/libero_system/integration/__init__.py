"""Real LIBERO integration with capability-safe lazy public exports.

Importing the package itself must not import the evaluator.  In particular,
the formal Route C child imports policy-side integration submodules inside a
restricted process and must never acquire evaluator or simulator modules as
an accidental side effect of package initialisation.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, tuple[str, str]] = {
    "AnalyticTopDownGraspProvider": (".adapters", "AnalyticTopDownGraspProvider"),
    "ContactAwareRouteCController": (".adapters", "ContactAwareRouteCController"),
    "LabelCheckedGeometryVerifier": (".adapters", "LabelCheckedGeometryVerifier"),
    "LabelCheckedGoalSynthesizer": (".adapters", "LabelCheckedGoalSynthesizer"),
    "LiberoRouteCRobot": (".adapters", "LiberoRouteCRobot"),
    "OSCWaypointConfig": (".adapters", "OSCWaypointConfig"),
    "RouteBPolicy": (".adapters", "RouteBPolicy"),
    "SensorBoundGeometryVerifier": (".adapters", "SensorBoundGeometryVerifier"),
    "SensorBoundGoalSynthesizer": (".adapters", "SensorBoundGoalSynthesizer"),
    "StableSceneEstimator": (".adapters", "StableSceneEstimator"),
    "to_route_b_observation": (".adapters", "to_route_b_observation"),
    "PerceptionBundle": (".components", "PerceptionBundle"),
    "PerceptionFactoryContext": (".components", "PerceptionFactoryContext"),
    "build_perception_bundle": (".components", "build_perception_bundle"),
    "EvaluationConfig": (".config", "EvaluationConfig"),
    "parse_config": (".config", "parse_config"),
    "CavityFrame": (".cavity", "CavityFrame"),
    "CavityGeometryError": (".cavity", "CavityGeometryError"),
    "align_panda_finger_axis": (".cavity", "align_panda_finger_axis"),
    "infer_cavity_frame": (".cavity", "infer_cavity_frame"),
    "EvaluationDriver": (".evaluator", "EvaluationDriver"),
    "evaluate": (".evaluator", "evaluate"),
    "GraspCloudAdapterConfig": (".grasp_cloud", "GraspCloudAdapterConfig"),
    "GraspCloudHTTPConfig": (".grasp_cloud", "GraspCloudHTTPConfig"),
    "GraspCloudInference": (".grasp_cloud", "GraspCloudInference"),
    "GraspCloudJobClient": (".grasp_cloud", "GraspCloudJobClient"),
    "GraspCloudResultProvider": (".grasp_cloud", "GraspCloudResultProvider"),
    "EpisodeKey": (".results", "EpisodeKey"),
    "EpisodeRecord": (".results", "EpisodeRecord"),
    "JsonlResultStore": (".results", "JsonlResultStore"),
    "episode_schedule": (".results", "episode_schedule"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))


__all__ = sorted(_EXPORTS)
