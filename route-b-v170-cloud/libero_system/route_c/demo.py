"""Run a dependency-light Route C continuous-optimisation smoke demo."""

from __future__ import annotations

import json

import numpy as np

from .compiler import TemplateConstraintCompiler
from .optimizer import MotionRequest, OptimizerConfig, TrajectoryOptimizer
from .perception import SceneEstimate
from .schema import Phase
from .sdf import SphereSDF


def _pose(x: float, y: float, z: float) -> np.ndarray:
    result = np.eye(4)
    result[:3, 3] = (x, y, z)
    return result


def main() -> None:
    graph = TemplateConstraintCompiler().compile(
        "Pick the akita black bowl next to the ramekin and place it on the plate"
    )
    scene = SceneEstimate(
        timestamp_s=0.0,
        entities=(),
        obstacle_sdf=SphereSDF(np.array([0.0, 0.0, 0.22]), 0.07),
        workspace_min=np.array([-0.5, -0.5, 0.05]),
        workspace_max=np.array([0.5, 0.5, 0.7]),
        scene_floor_z=0.0,
    )
    request = MotionRequest(
        phase=Phase.TRANSFER,
        start_pose=_pose(-0.30, 0.0, 0.22),
        goal_pose=_pose(0.30, 0.0, 0.22),
        clearance_m=0.02,
        tool_radius_m=0.015,
    )
    trajectory = TrajectoryOptimizer(
        OptimizerConfig(num_waypoints=14, max_iterations=100)
    ).optimise(request, scene)
    print(
        json.dumps(
            {
                "compiled_goal": graph.goal_relation.value,
                "source": graph.source.label,
                "source_selector": graph.source.selector.to_dict() if graph.source.selector else None,
                "target": graph.target.label,
                "trajectory_feasible": trajectory.feasible,
                "waypoints": len(trajectory.poses),
                "min_clearance_m": trajectory.min_clearance_m,
                "objective": trajectory.objective,
                "status": trajectory.status,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

