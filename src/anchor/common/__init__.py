"""Capability-safe lazy public interfaces shared by Routes B and C.

Importing :mod:`anchor.common` is intentionally inert.  In particular,
the formal Planning worker may import observation, action, and grasp DTOs without
also importing the evaluator-facing environment adapter or result logger.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, tuple[str, str]] = {
    "backproject_depth": (".camera_geometry", "backproject_depth"),
    "metric_depth_from_normalized": (
        ".camera_geometry",
        "metric_depth_from_normalized",
    ),
    "project_world_points": (".camera_geometry", "project_world_points"),
    "quaternion_xyzw_to_matrix": (
        ".camera_geometry",
        "quaternion_xyzw_to_matrix",
    ),
    "transform_points": (".camera_geometry", "transform_points"),
    "DEFAULT_MAX_STEPS": (".env_adapter", "DEFAULT_MAX_STEPS"),
    "UNIFIED_EVALUATION_HORIZON": (
        ".env_adapter",
        "UNIFIED_EVALUATION_HORIZON",
    ),
    "EnvironmentStep": (".env_adapter", "EnvironmentStep"),
    "EvaluationSignal": (".env_adapter", "EvaluationSignal"),
    "LiberoEnvAdapter": (".env_adapter", "LiberoEnvAdapter"),
    "LiberoEnvConfig": (".env_adapter", "LiberoEnvConfig"),
    "TaskMetadata": (".env_adapter", "TaskMetadata"),
    "configure_runtime_environment": (
        ".env_adapter",
        "configure_runtime_environment",
    ),
    "GraspAttemptEvent": (".grasp_journal", "GraspAttemptEvent"),
    "GraspAttemptJournal": (".grasp_journal", "GraspAttemptJournal"),
    "GraspEvidenceCategory": (".grasp_journal", "GraspEvidenceCategory"),
    "GraspOutcome": (".grasp_journal", "GraspOutcome"),
    "GraspReason": (".grasp_journal", "GraspReason"),
    "JawBehavior": (".grasp_journal", "JawBehavior"),
    "PendingGraspEngagement": (".grasp_journal", "PendingGraspEngagement"),
    "CameraCalibration": (".observation", "CameraCalibration"),
    "CameraFrame": (".observation", "CameraFrame"),
    "Proprioception": (".observation", "Proprioception"),
    "RobotObservation": (".observation", "RobotObservation"),
    "OSCAction": (".policy", "OSCAction"),
    "Policy": (".policy", "Policy"),
    "PolicyDecision": (".policy", "PolicyDecision"),
    "PolicyTask": (".policy", "PolicyTask"),
    "EpisodeResult": (".runner", "EpisodeResult"),
    "EpisodeRunner": (".runner", "EpisodeRunner"),
    "JsonlEpisodeLogger": (".runner", "JsonlEpisodeLogger"),
    "RunnerConfig": (".runner", "RunnerConfig"),
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
