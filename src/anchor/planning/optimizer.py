"""Warm-started continuous trajectory optimisation and receding-horizon wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation, Slerp

from .perception import SceneEstimate
from .schema import Phase


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class InfeasibleClearanceDiagnostic:
    """Structured provenance for one infeasible clearance constraint.

    The human-readable optimiser message is deliberately not an API. Contact
    recovery code may inspect this immutable record instead, so changing or
    spoofing log text cannot authorize a motion.
    """

    phase: Phase
    reach_ok: bool
    height_ok: bool
    clearance_ok: bool
    clearance_limit_m: float
    tool_radius_m: float
    minimum_clearance_m: float
    raw_sdf_m: float
    field_index: int
    source_instance_id: str | None
    source_label: str | None
    sample_index: int
    sample_count: int
    nearest_point_world_m: tuple[float, float, float]
    field_center_world_m: tuple[float, float, float] | None
    field_half_extents_m: tuple[float, float, float] | None
    is_start: bool
    is_end: bool


class OptimisationError(RuntimeError):
    """Trajectory optimisation failed, with optional machine-readable facts."""

    def __init__(
        self,
        detail: str,
        *,
        clearance_diagnostic: InfeasibleClearanceDiagnostic | None = None,
    ) -> None:
        super().__init__(detail)
        self.clearance_diagnostic = clearance_diagnostic


def _pose(value: object, name: str) -> FloatArray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError(f"{name} must be finite 4x4")
    if not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
        raise ValueError(f"{name} has invalid homogeneous row")
    if not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=2e-3):
        raise ValueError(f"{name} rotation is not orthonormal")
    return pose.copy()


@dataclass(frozen=True)
class CostWeights:
    obstacle: float = 100.0
    smoothness: float = 2.0
    path_length: float = 0.35
    reachability: float = 120.0
    safe_height: float = 25.0

    def __post_init__(self) -> None:
        if any(not np.isfinite(x) or x < 0 for x in vars(self).values()):
            raise ValueError("cost weights must be finite and non-negative")


@dataclass(frozen=True)
class MotionRequest:
    phase: Phase
    start_pose: FloatArray
    goal_pose: FloatArray
    clearance_m: float = 0.025
    tool_radius_m: float = 0.025
    workspace_margin_m: float = 0.01
    min_height_m: float | None = None
    weights: CostWeights = field(default_factory=CostWeights)

    def __post_init__(self) -> None:
        object.__setattr__(self, "start_pose", _pose(self.start_pose, "start_pose"))
        object.__setattr__(self, "goal_pose", _pose(self.goal_pose, "goal_pose"))
        for name in ("clearance_m", "tool_radius_m", "workspace_margin_m"):
            value = float(getattr(self, name))
            if value < 0 or not np.isfinite(value):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.min_height_m is not None and not np.isfinite(self.min_height_m):
            raise ValueError("min_height_m must be finite")


@dataclass(frozen=True)
class Trajectory:
    phase: Phase
    poses: FloatArray
    objective: float
    min_clearance_m: float
    feasible: bool
    reach_ok: bool
    height_ok: bool
    clearance_ok: bool
    iterations: int
    status: str
    cost_terms: Mapping[str, float]

    def __post_init__(self) -> None:
        poses = np.asarray(self.poses, dtype=np.float64)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 2:
            raise ValueError("trajectory poses must be Nx4x4")
        if not np.all(np.isfinite(poses)):
            raise ValueError("trajectory contains non-finite values")
        object.__setattr__(self, "poses", poses.copy())
        object.__setattr__(self, "cost_terms", dict(self.cost_terms))


@runtime_checkable
class ReachabilityEvaluator(Protocol):
    """Optional adapter for an IK/collision backend such as cuRobo."""

    def cost(self, positions_world: FloatArray) -> FloatArray: ...


class WorkspaceReachability:
    """Dependency-free conservative reachability surrogate."""

    def __init__(self, lower: FloatArray, upper: FloatArray, margin: float = 0.0) -> None:
        self.lower = np.asarray(lower, dtype=np.float64) + margin
        self.upper = np.asarray(upper, dtype=np.float64) - margin
        if self.lower.shape != (3,) or self.upper.shape != (3,) or np.any(self.lower >= self.upper):
            raise ValueError("invalid workspace reachability bounds")

    def cost(self, positions_world: FloatArray) -> FloatArray:
        points = np.asarray(positions_world, dtype=np.float64)
        lower = np.maximum(self.lower - points, 0.0)
        upper = np.maximum(points - self.upper, 0.0)
        return np.sum((lower + upper) ** 2, axis=-1)


@dataclass(frozen=True)
class OptimizerConfig:
    num_waypoints: int = 14
    segment_samples: int = 3
    max_iterations: int = 90
    feasibility_tolerance_m: float = 0.004
    arch_height_m: float = 0.10
    gradient_tolerance: float = 1e-6
    # This is a deployment-calibrated Panda OSC motion envelope, not the
    # RGB-D crop used by perception.  Keeping the two concepts separate lets
    # the end effector safely move above a visible support without making
    # benchmark/suite metadata part of the policy.
    motion_workspace_min_m: tuple[float, float, float] = (-0.55, -0.45, -0.10)
    motion_workspace_max_m: tuple[float, float, float] = (0.45, 0.65, 1.43)

    def __post_init__(self) -> None:
        if self.num_waypoints < 4 or self.segment_samples < 1 or self.max_iterations < 1:
            raise ValueError("invalid optimiser iteration/waypoint configuration")
        lower = np.asarray(self.motion_workspace_min_m, dtype=np.float64)
        upper = np.asarray(self.motion_workspace_max_m, dtype=np.float64)
        if (
            lower.shape != (3,)
            or upper.shape != (3,)
            or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(lower >= upper)
        ):
            raise ValueError("invalid calibrated motion workspace")


class TrajectoryOptimizer:
    """Optimise translation; interpolate orientation geodesically on SO(3).

    Translation is the dominant degree of freedom for LIBERO pick/place.  The
    strict pose interface leaves room for a cuRobo-backed full joint-space
    ``ReachabilityEvaluator`` without changing the controller.
    """

    def __init__(
        self,
        config: OptimizerConfig | None = None,
        reachability: ReachabilityEvaluator | None = None,
    ) -> None:
        self.config = config or OptimizerConfig()
        self.reachability = reachability

    def optimise(
        self,
        request: MotionRequest,
        scene: SceneEstimate,
        warm_start: Trajectory | None = None,
    ) -> Trajectory:
        count = self.config.num_waypoints
        start = request.start_pose[:3, 3]
        goal = request.goal_pose[:3, 3]
        initial = self._initial_path(request, scene, warm_start)
        if count == 2:  # guarded by config, retained for clarity
            positions = np.stack((start, goal))
            return self._make_trajectory(request, scene, positions, 0, "fixed")

        lower, upper = self.motion_bounds(request.workspace_margin_m)
        if np.any(lower >= upper):
            raise OptimisationError("workspace is empty after applying margin")
        interior_lower = lower.copy()
        if request.min_height_m is not None:
            if request.min_height_m > upper[2]:
                raise OptimisationError(
                    "minimum height exceeds the motion workspace upper-z bound "
                    f"({request.min_height_m:.4f} > {upper[2]:.4f} m)"
                )
            # Safe height is a physical constraint, rather than merely an
            # objective preference.  The fixed start and goal are handled by
            # the feasibility check below; every optimisable waypoint gets a
            # hard L-BFGS-B lower bound.  Linear dense interpolation then
            # cannot dip between two height-safe waypoints.
            interior_lower[2] = max(interior_lower[2], request.min_height_m)
        interior_lowers = np.broadcast_to(interior_lower, (count - 2, 3)).copy()
        interior_uppers = np.broadcast_to(upper, (count - 2, 3)).copy()
        if request.min_height_m is not None and start[2] < request.min_height_m:
            # The sole below-floor segment is a vertical recovery.  Without
            # fixing x/y here, a long transfer would also travel roughly one
            # waypoint of horizontal distance before regaining safe height.
            interior_lowers[0, :2] = start[:2]
            interior_uppers[0, :2] = start[:2]
        bounds = [
            (float(interior_lowers[index, axis]), float(interior_uppers[index, axis]))
            for index in range(count - 2)
            for axis in range(3)
        ]
        x0 = np.clip(initial[1:-1], interior_lowers, interior_uppers).reshape(-1)

        def unpack(x: FloatArray) -> FloatArray:
            return np.vstack((start, np.asarray(x).reshape(count - 2, 3), goal))

        def objective(x: FloatArray) -> float:
            terms = self._cost_terms(unpack(x), request, scene)
            return float(sum(terms.values()))

        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": self.config.max_iterations,
                "ftol": 1e-10,
                "gtol": self.config.gradient_tolerance,
                "maxls": 35,
            },
        )
        positions = unpack(result.x)
        status = "converged" if result.success else f"stopped:{result.message}"
        return self._make_trajectory(request, scene, positions, int(result.nit), status)

    # Alias with US spelling for adapters that use ``optimize``.
    optimize = optimise

    def _initial_path(
        self, request: MotionRequest, scene: SceneEstimate, warm_start: Trajectory | None
    ) -> FloatArray:
        count = self.config.num_waypoints
        start, goal = request.start_pose[:3, 3], request.goal_pose[:3, 3]
        if warm_start is not None:
            old = warm_start.poses[:, :3, 3]
            old_t = np.linspace(0.0, 1.0, len(old))
            new_t = np.linspace(0.0, 1.0, count)
            path = np.column_stack([np.interp(new_t, old_t, old[:, j]) for j in range(3)])
            path += np.linspace(start - path[0], goal - path[-1], count)
            path[0], path[-1] = start, goal
            return path

        alpha = np.linspace(0.0, 1.0, count)
        line = start[None, :] * (1.0 - alpha[:, None]) + goal[None, :] * alpha[:, None]
        envelope = np.sin(np.pi * alpha)[:, None]
        candidates = [line]
        for direction in (
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([-1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, -1.0, 0.0]),
        ):
            candidates.append(line + envelope * direction * self.config.arch_height_m)
        return min(candidates, key=lambda path: sum(self._cost_terms(path, request, scene).values()))

    def _dense(self, positions: FloatArray) -> FloatArray:
        samples = self.config.segment_samples
        if samples == 1:
            return positions
        chunks = []
        alpha = np.linspace(0.0, 1.0, samples, endpoint=False)
        for start, end in zip(positions[:-1], positions[1:]):
            chunks.append(start[None, :] * (1.0 - alpha[:, None]) + end[None, :] * alpha[:, None])
        chunks.append(positions[-1:])
        return np.concatenate(chunks, axis=0)

    def _height_lower_profile(
        self, positions: FloatArray, min_height_m: float
    ) -> FloatArray:
        """Return the hard lower-z profile for the densely sampled path.

        A measured replan start can already be slightly below a phase's
        frozen safe height because the low-level controller did not track the
        previous plan exactly.  That fixed state cannot be made feasible by
        the optimiser.  In that case only, allow the first segment to recover
        monotonically from the measured height to the requested floor.  The
        first optimisable waypoint and the entire remaining path must still
        be at or above ``min_height_m``.  This admits recovery without hiding
        continued or renewed descent.
        """

        dense_count = (
            len(positions)
            if self.config.segment_samples == 1
            else (len(positions) - 1) * self.config.segment_samples + 1
        )
        lower = np.full(dense_count, float(min_height_m), dtype=np.float64)
        start_height = float(positions[0, 2])
        if start_height < min_height_m:
            first_segment_samples = min(self.config.segment_samples, dense_count - 1)
            lower[: first_segment_samples + 1] = np.linspace(
                start_height,
                float(min_height_m),
                first_segment_samples + 1,
            )
        return lower

    def _cost_terms(
        self, positions: FloatArray, request: MotionRequest, scene: SceneEstimate
    ) -> dict[str, float]:
        dense = self._dense(positions)
        required = request.clearance_m + request.tool_radius_m
        distances = np.asarray(scene.obstacle_sdf.distance(dense), dtype=np.float64)
        collision_hinge = np.maximum(required - distances, 0.0)
        delta = np.diff(positions, axis=0)
        acceleration = np.diff(positions, n=2, axis=0)
        motion_lower, motion_upper = self.motion_bounds(0.0)
        reachability = self.reachability or WorkspaceReachability(
            motion_lower, motion_upper, request.workspace_margin_m
        )
        reach_cost = np.asarray(reachability.cost(dense), dtype=np.float64)
        height_cost = 0.0
        if request.min_height_m is not None:
            height_lower = self._height_lower_profile(positions, request.min_height_m)
            height_cost = request.weights.safe_height * float(
                np.sum(np.maximum(height_lower - dense[:, 2], 0.0) ** 2)
            )
        return {
            "obstacle": request.weights.obstacle * float(np.sum(collision_hinge**2)),
            "smoothness": request.weights.smoothness * float(np.sum(acceleration**2)),
            "path_length": request.weights.path_length * float(np.sum(delta**2)),
            "reachability": request.weights.reachability * float(np.sum(reach_cost)),
            "safe_height": height_cost,
        }

    def _make_trajectory(
        self,
        request: MotionRequest,
        scene: SceneEstimate,
        positions: FloatArray,
        iterations: int,
        status: str,
    ) -> Trajectory:
        rotations = Rotation.from_matrix(
            np.stack((request.start_pose[:3, :3], request.goal_pose[:3, :3]), axis=0)
        )
        interpolation = Slerp([0.0, 1.0], rotations)(np.linspace(0.0, 1.0, len(positions)))
        poses = np.repeat(np.eye(4)[None, :, :], len(positions), axis=0)
        poses[:, :3, :3] = interpolation.as_matrix()
        poses[:, :3, 3] = positions
        dense = self._dense(positions)
        min_clearance = float(np.min(scene.obstacle_sdf.distance(dense))) - request.tool_radius_m
        lower, upper = self.motion_bounds(request.workspace_margin_m)
        reach_ok = bool(np.all(dense >= lower - 1e-8) and np.all(dense <= upper + 1e-8))
        height_ok = True
        if request.min_height_m is not None:
            height_lower = self._height_lower_profile(positions, request.min_height_m)
            height_ok = bool(np.all(dense[:, 2] >= height_lower - 1e-8))
        clearance_ok = min_clearance >= request.clearance_m - self.config.feasibility_tolerance_m
        terms = self._cost_terms(positions, request, scene)
        return Trajectory(
            phase=request.phase,
            poses=poses,
            objective=float(sum(terms.values())),
            min_clearance_m=min_clearance,
            feasible=reach_ok and height_ok and clearance_ok,
            reach_ok=reach_ok,
            height_ok=height_ok,
            clearance_ok=clearance_ok,
            iterations=iterations,
            status=status,
            cost_terms=terms,
        )

    def motion_bounds(self, margin_m: float) -> tuple[FloatArray, FloatArray]:
        """Return the static robot motion envelope after a request margin."""

        margin = float(margin_m)
        lower = np.asarray(self.config.motion_workspace_min_m, dtype=np.float64) + margin
        upper = np.asarray(self.config.motion_workspace_max_m, dtype=np.float64) - margin
        return lower, upper


@dataclass(frozen=True)
class MPCChunk:
    trajectory: Trajectory
    poses: FloatArray
    reached_goal: bool


class RecedingHorizonOptimizer:
    """Warm-start and expose only a short safe prefix on every replan."""

    def __init__(self, optimizer: TrajectoryOptimizer, horizon_waypoints: int = 4) -> None:
        if horizon_waypoints < 2:
            raise ValueError("horizon_waypoints must be at least two")
        self.optimizer = optimizer
        self.horizon_waypoints = int(horizon_waypoints)
        self._warm_start: Trajectory | None = None
        self._phase: Phase | None = None

    def reset(self) -> None:
        self._warm_start = None
        self._phase = None

    @staticmethod
    def _clearance_diagnostic(
        request: MotionRequest,
        scene: SceneEstimate,
        dense: FloatArray,
        trajectory: Trajectory,
    ) -> tuple[str, InfeasibleClearanceDiagnostic]:
        """Describe and structure the raw-SDF argmin without changing semantics."""

        field = scene.obstacle_sdf
        diagnostic_provider = getattr(field, "nearest_field_diagnostic", None)
        if callable(diagnostic_provider):
            diagnostic = diagnostic_provider(dense)
            raw_distance = float(diagnostic.raw_distance_m)
            field_index = int(diagnostic.field_index)
            sample_index = int(diagnostic.sample_index)
            nearest_point = diagnostic.nearest_point_world_m
            source_instance_id = diagnostic.source_instance_id
            source_label = diagnostic.source_label
            field_center = diagnostic.field_center_world_m
            field_half_extents = diagnostic.field_half_extents_m
        else:
            raw_distances = np.asarray(field.distance(dense), dtype=np.float64)
            sample_index = int(np.argmin(raw_distances))
            raw_distance = float(raw_distances[sample_index])
            field_index = 0
            nearest_point = tuple(float(value) for value in dense[sample_index])
            source_instance_id = getattr(field, "source_instance_id", None)
            source_label = getattr(field, "source_label", None)
            center = getattr(field, "center", None)
            half_extents = getattr(field, "half_extents", None)
            field_center = (
                tuple(float(value) for value in np.asarray(center, dtype=np.float64))
                if center is not None
                else None
            )
            field_half_extents = (
                tuple(
                    float(value)
                    for value in np.asarray(half_extents, dtype=np.float64)
                )
                if half_extents is not None
                else None
            )

        is_start = sample_index == 0
        is_end = sample_index == len(dense) - 1
        sample_location = "start" if is_start else "end" if is_end else "interior"
        field_id_text = source_instance_id if source_instance_id is not None else "unknown"
        field_label_text = source_label if source_label is not None else "unknown"
        point_text = "[" + ", ".join(f"{value:.4f}" for value in nearest_point) + "]"
        field_center_text = (
            "[" + ", ".join(f"{value:.4f}" for value in field_center) + "]"
            if field_center is not None
            else "unknown"
        )
        field_half_extents_text = (
            "["
            + ", ".join(f"{value:.4f}" for value in field_half_extents)
            + "]"
            if field_half_extents is not None
            else "unknown"
        )
        detail = (
            f"raw_sdf={raw_distance:.4f} m, field_index={field_index}, "
            f"field_id={field_id_text}, field_label={field_label_text}, "
            f"sample_index={sample_index}, point_xyz={point_text} m, "
            f"field_center={field_center_text} m, "
            f"field_half_extents={field_half_extents_text} m, "
            f"sample_location={sample_location}, "
            f"is_start={is_start}, is_end={is_end}"
        )
        diagnostic = InfeasibleClearanceDiagnostic(
            phase=request.phase,
            reach_ok=bool(trajectory.reach_ok),
            height_ok=bool(trajectory.height_ok),
            clearance_ok=bool(trajectory.clearance_ok),
            clearance_limit_m=float(request.clearance_m),
            tool_radius_m=float(request.tool_radius_m),
            minimum_clearance_m=float(trajectory.min_clearance_m),
            raw_sdf_m=raw_distance,
            field_index=field_index,
            source_instance_id=source_instance_id,
            source_label=source_label,
            sample_index=sample_index,
            sample_count=len(dense),
            nearest_point_world_m=tuple(float(value) for value in nearest_point),
            field_center_world_m=field_center,
            field_half_extents_m=field_half_extents,
            is_start=is_start,
            is_end=is_end,
        )
        return detail, diagnostic

    def replan(self, request: MotionRequest, scene: SceneEstimate) -> MPCChunk:
        warm = self._warm_start if self._phase == request.phase else None
        trajectory = self.optimizer.optimise(request, scene, warm_start=warm)
        if not trajectory.feasible:
            self._warm_start = None
            self._phase = request.phase
            failures: list[str] = []
            clearance_diagnostic: InfeasibleClearanceDiagnostic | None = None
            dense = self.optimizer._dense(trajectory.poses[:, :3, 3])
            if not trajectory.reach_ok:
                lower, upper = self.optimizer.motion_bounds(request.workspace_margin_m)
                below = np.max(lower[None, :] - dense, axis=0)
                above = np.max(dense - upper[None, :], axis=0)
                axis = ("x", "y", "z")
                for index, value in enumerate(below):
                    if value > 1e-8:
                        failures.append(
                            f"workspace_lower_{axis[index]} violation={value:.4f} m "
                            f"(minimum={np.min(dense[:, index]):.4f}, "
                            f"limit={lower[index]:.4f})"
                        )
                for index, value in enumerate(above):
                    if value > 1e-8:
                        failures.append(
                            f"workspace_upper_{axis[index]} violation={value:.4f} m "
                            f"(maximum={np.max(dense[:, index]):.4f}, "
                            f"limit={upper[index]:.4f})"
                        )
                if not failures:
                    failures.append("workspace reachability check failed")
            if not trajectory.height_ok:
                assert request.min_height_m is not None
                positions = trajectory.poses[:, :3, 3]
                height_lower = self.optimizer._height_lower_profile(
                    positions, request.min_height_m
                )
                height_deficit = height_lower - dense[:, 2]
                worst_height_index = int(np.argmax(height_deficit))
                failures.append(
                    "safe_height violation="
                    f"{height_deficit[worst_height_index]:.4f} m "
                    f"(height={dense[worst_height_index, 2]:.4f}, "
                    f"limit={height_lower[worst_height_index]:.4f}, "
                    f"sample_index={worst_height_index})"
                )
            if not trajectory.clearance_ok:
                clearance_detail, clearance_diagnostic = self._clearance_diagnostic(
                    request, scene, dense, trajectory
                )
                failures.append(
                    "clearance violation="
                    f"{request.clearance_m - trajectory.min_clearance_m:.4f} m "
                    f"(minimum={trajectory.min_clearance_m:.4f}, "
                    f"limit={request.clearance_m:.4f}); "
                    + clearance_detail
                )
            raise OptimisationError(
                f"{request.phase.value} has no feasible trajectory: "
                + "; ".join(failures)
                + "; checks "
                f"reach={trajectory.reach_ok}, height={trajectory.height_ok}, "
                f"clearance={trajectory.clearance_ok}; min clearance "
                f"{trajectory.min_clearance_m:.4f} m",
                clearance_diagnostic=clearance_diagnostic,
            )
        self._warm_start = trajectory
        self._phase = request.phase
        stop = min(self.horizon_waypoints, len(trajectory.poses))
        poses = trajectory.poses[:stop]
        reached = stop == len(trajectory.poses)
        return MPCChunk(trajectory=trajectory, poses=poses, reached_goal=reached)
