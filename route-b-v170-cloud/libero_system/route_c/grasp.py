"""Bind external 6-DoF grasp proposals to a sensor-derived constraint graph."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .perception import EntityResolver, SceneEntity, SceneEstimate
from .schema import ConstraintGraph


FloatArray = NDArray[np.float64]


class GraspBindingError(RuntimeError):
    pass


def _pose(value: object) -> FloatArray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError("grasp pose must be finite 4x4")
    if not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
        raise ValueError("grasp pose has an invalid homogeneous row")
    if not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=2e-3):
        raise ValueError("grasp rotation is not orthonormal")
    return pose.copy()


@dataclass(frozen=True)
class GraspCandidate:
    candidate_id: str
    object_id: str
    world_from_ee: FloatArray
    approach_world: FloatArray
    score: float
    clearance_m: float
    gripper_width_m: float

    def __post_init__(self) -> None:
        approach = np.asarray(self.approach_world, dtype=np.float64)
        if approach.shape != (3,) or not np.all(np.isfinite(approach)):
            raise ValueError("approach_world must be finite xyz")
        norm = float(np.linalg.norm(approach))
        if norm < 1e-8:
            raise ValueError("approach_world must be non-zero")
        if not self.candidate_id or not self.object_id:
            raise ValueError("candidate_id and object_id are required")
        if not 0.0 <= float(self.score) <= 1.0:
            raise ValueError("grasp score must be in [0, 1]")
        if self.clearance_m < 0 or self.gripper_width_m <= 0:
            raise ValueError("clearance must be non-negative and gripper width positive")
        object.__setattr__(self, "world_from_ee", _pose(self.world_from_ee))
        object.__setattr__(self, "approach_world", approach / norm)


@runtime_checkable
class GraspProvider(Protocol):
    """Adapter point for GraspGenX, an analytic grasp sampler, or another model."""

    def propose(self, scene: SceneEstimate, object_id: str) -> Sequence[GraspCandidate]: ...


@dataclass(frozen=True)
class GraspBinding:
    candidate: GraspCandidate
    object_from_ee: FloatArray
    rank_score: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_from_ee", _pose(self.object_from_ee))


@dataclass(frozen=True)
class BoundConstraintGraph:
    graph: ConstraintGraph
    source_id: str
    target_id: str
    source_label: str
    target_label: str
    grasp_options: tuple[GraspBinding, ...]
    active_grasp_index: int = 0

    def __post_init__(self) -> None:
        if not self.grasp_options:
            raise GraspBindingError("bound graph has no grasp options")
        if not 0 <= self.active_grasp_index < len(self.grasp_options):
            raise GraspBindingError("active grasp index is out of range")

    @property
    def grasp(self) -> GraspBinding:
        return self.grasp_options[self.active_grasp_index]

    def next_grasp(self) -> "BoundConstraintGraph":
        if self.active_grasp_index + 1 >= len(self.grasp_options):
            raise GraspBindingError("no alternate grasp remains")
        return replace(self, active_grasp_index=self.active_grasp_index + 1)


class GraspBinder:
    def __init__(self, resolver: EntityResolver | None = None, max_options: int = 8) -> None:
        if max_options < 1:
            raise ValueError("max_options must be positive")
        self.resolver = resolver or EntityResolver()
        self.max_options = int(max_options)

    def bind(
        self,
        graph: ConstraintGraph,
        scene: SceneEstimate,
        candidates: Sequence[GraspCandidate],
    ) -> BoundConstraintGraph:
        source = self.resolver.resolve(graph.source, scene)
        target = self.resolver.resolve(graph.target, scene)
        bindings: list[GraspBinding] = []
        for candidate in candidates:
            if candidate.object_id != source.instance_id:
                continue
            position = candidate.world_from_ee[:3, 3]
            reach_violation = np.maximum(scene.workspace_min - position, 0.0) + np.maximum(
                position - scene.workspace_max, 0.0
            )
            if np.any(reach_violation > 0.05):
                continue
            sdf_clearance = float(scene.obstacle_sdf.distance(position))
            measured_clearance = min(candidate.clearance_m, sdf_clearance)
            clearance_bonus = 0.2 * np.tanh(max(measured_clearance, 0.0) / 0.025)
            reach_penalty = 20.0 * float(np.linalg.norm(reach_violation))
            rank_score = float(candidate.score + clearance_bonus - reach_penalty)
            object_from_ee = np.linalg.inv(source.pose) @ candidate.world_from_ee
            bindings.append(GraspBinding(candidate, object_from_ee, rank_score))
        if not bindings:
            raise GraspBindingError(f"no usable grasp candidate for visual entity {source.instance_id!r}")
        bindings.sort(key=lambda x: x.rank_score, reverse=True)
        return BoundConstraintGraph(
            graph=graph,
            source_id=source.instance_id,
            target_id=target.instance_id,
            source_label=source.label,
            target_label=target.label,
            grasp_options=tuple(bindings[: self.max_options]),
        )


def current_binding_for_entity(binding: GraspBinding, source: SceneEntity) -> FloatArray:
    """Re-anchor a grasp after a fresh visual pose estimate."""

    return source.pose @ binding.object_from_ee

