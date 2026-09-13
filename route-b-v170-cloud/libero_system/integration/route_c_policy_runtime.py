"""Route C policy runtime with no evaluator or simulator capabilities.

This module assembles Route C from language, sanitized sensor callbacks, a
bounded OSC action callback, and frozen perception deployment objects.  It
does not import the LIBERO environment adapter, evaluator score latch, result
store, campaign metadata, or benchmark identifiers, which lets the formal
runner execute it in a separate process with a narrow serialized channel.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from types import SimpleNamespace
from typing import Any, Callable

from libero_system.common import OSCAction, RobotObservation
from libero_system.integration.adapters import (
    AnalyticTopDownGraspProvider,
    ContactAwareRouteCController,
    LiberoRouteCContactTargetProvider,
    LiberoRouteCRobot,
    PolicyStepBudgetExhausted,
    SensorBoundGeometryVerifier,
    SensorBoundGoalSynthesizer,
    SensorFormedStackHandler,
    SensorRankedTargetHandler,
    StableSceneEstimator,
    WorkspaceCenterEntityResolver,
)
from libero_system.integration.components import PerceptionBundle
from libero_system.integration.policy_ipc import camera_content_sha256
from libero_system.integration.policy_result import RouteCPolicyExecution
from libero_system.perception import coerce_rgbd_frame
from libero_system.route_c import (
    AtomicGoalGraphCompiler,
    EntityGraspModeSelector,
    GraspBinder,
    MPCContactGoalExecutor,
    OptimizerConfig,
    RecedingHorizonOptimizer,
    RouteCControllerConfig,
    RouteCSequentialCoordinator,
    TrajectoryOptimizer,
)


SensorProvider = Callable[[], RobotObservation]
ActionExecutor = Callable[[OSCAction], RobotObservation]
_EPISODE_RUNTIME_SCHEMA = "libero-route-c-episode-runtime.v1"
_RUNTIME_ASSEMBLY_COUNT = 0


def _runtime_state_sha256(value: dict[str, int | bool]) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(
        b"libero-route-c-runtime-initial-state.v1\0" + encoded
    ).hexdigest()


def _assemble_route_c_episode_runtime(
    *,
    bundle: PerceptionBundle | Any,
    frames: Callable[[], tuple[Any, ...]],
    current_observation: SensorProvider,
    execute_action: ActionExecutor,
    step_budget: int,
) -> SimpleNamespace:
    """Build every mutable policy component anew for exactly one episode.

    The model bundle is intentionally worker-scoped and immutable.  All
    controllers, selectors, caches, journals, and robot counters are created
    here and are episode-local.  The returned receipt commits the actually
    constructed component identities and their observed empty initial state.
    """

    global _RUNTIME_ASSEMBLY_COUNT

    observer = StableSceneEstimator(
        bundle.for_route_c(frames),
        ee_pose_provider=lambda: current_observation().proprio.T_world_ee,
    )
    robot = LiberoRouteCRobot(
        current_observation,
        execute_action,
        step_budget=step_budget,
    )
    optimizer = TrajectoryOptimizer(
        OptimizerConfig(num_waypoints=10, segment_samples=2, max_iterations=70)
    )
    goals = SensorBoundGoalSynthesizer(
        track_updater=observer.update_track,
        track_invalidator=observer.invalidate_sensor_cache,
        track_discarder=observer.discard_track,
    )
    grasp_mode_selector = EntityGraspModeSelector()
    grasp_provider = AnalyticTopDownGraspProvider(
        robot.current_ee_pose,
        grasp_mode_selector=grasp_mode_selector,
    )
    resolver = WorkspaceCenterEntityResolver(
        selector_rejection_sink=observer.invalidate_sensor_cache,
        visible_instance_ids_provider=lambda: observer.visible_instance_ids,
    )
    formed_stack_handler = SensorFormedStackHandler(observer, resolver)
    ranked_target_handler = SensorRankedTargetHandler(observer, resolver)
    graph_compiler = AtomicGoalGraphCompiler()
    motion_mpc = RecedingHorizonOptimizer(optimizer, horizon_waypoints=10)
    controller = ContactAwareRouteCController(
        compiler=graph_compiler,
        observer=observer,
        grasp_provider=grasp_provider,
        robot=robot,
        mpc=motion_mpc,
        binder=GraspBinder(resolver=resolver),
        goals=goals,
        verifier=SensorBoundGeometryVerifier(goals, observer),
        grasp_mode_selector=grasp_mode_selector,
        config=RouteCControllerConfig(
            max_task_attempts=4,
            max_phase_attempts=2,
            max_mpc_replans=12,
            position_tolerance_m=0.003,
        ),
    )
    contact_provider = LiberoRouteCContactTargetProvider(
        current_observation,
        observer,
    )
    contact_mpc = RecedingHorizonOptimizer(optimizer, horizon_waypoints=10)
    contact_executor = MPCContactGoalExecutor(
        observer,
        robot,
        contact_mpc,
        contact_provider,
    )
    coordinator = RouteCSequentialCoordinator(
        controller,
        graph_compiler,
        contact_executor,
        between_goals=observer.invalidate_sensor_cache,
        source_consumed=resolver.exclude_source,
        formed_stack_handler=formed_stack_handler,
        ranked_target_handler=ranked_target_handler,
    )
    components = {
        "contact_executor": contact_executor,
        "contact_mpc": contact_mpc,
        "contact_provider": contact_provider,
        "controller": controller,
        "coordinator": coordinator,
        "formed_stack_handler": formed_stack_handler,
        "goals": goals,
        "graph_compiler": graph_compiler,
        "grasp_mode_selector": grasp_mode_selector,
        "grasp_provider": grasp_provider,
        "motion_mpc": motion_mpc,
        "observer": observer,
        "optimizer": optimizer,
        "ranked_target_handler": ranked_target_handler,
        "resolver": resolver,
        "robot": robot,
    }
    initial_state: dict[str, int | bool] = {
        "contact_trace_entries": len(contact_provider.last_contact_trace),
        "controller_grasp_events": len(controller.grasp_attempt_events),
        "controller_pending_grasp": controller.pending_grasp_engagement is not None,
        "observer_obstacle_filter_entries": len(observer.obstacle_filter_history),
        "observer_selector_diagnostics": len(observer.selector_diagnostics),
        "resolver_resolution_history": len(resolver.resolution_history),
        "robot_grasp_checks": len(robot.grasp_checks),
        "robot_phase_trace_entries": len(robot.phase_trace),
        "robot_steps_executed": robot.steps_executed,
    }
    expected_initial_state: dict[str, int | bool] = {
        "contact_trace_entries": 0,
        "controller_grasp_events": 0,
        "controller_pending_grasp": False,
        "observer_obstacle_filter_entries": 0,
        "observer_selector_diagnostics": 0,
        "resolver_resolution_history": 0,
        "robot_grasp_checks": 0,
        "robot_phase_trace_entries": 0,
        "robot_steps_executed": 0,
    }
    if initial_state != expected_initial_state:
        raise RuntimeError("new Route C episode runtime did not start empty")
    _RUNTIME_ASSEMBLY_COUNT += 1
    assembly_nonce = secrets.token_hex(32)
    identity = hashlib.sha256(
        b"libero-route-c-runtime-components.v1\0"
        + bytes.fromhex(assembly_nonce)
    )
    for name, component in sorted(components.items()):
        encoded = name.encode("utf-8")
        identity.update(len(encoded).to_bytes(4, "big"))
        identity.update(encoded)
        identity.update(str(id(component)).encode("ascii"))
        identity.update(b"\0")
    return SimpleNamespace(
        **components,
        runtime_receipt={
            "schema": _EPISODE_RUNTIME_SCHEMA,
            "assembly_nonce": assembly_nonce,
            "assembly_count": _RUNTIME_ASSEMBLY_COUNT,
            "component_count": len(components),
            "component_identity_sha256": identity.hexdigest(),
            "initial_state": initial_state,
            "initial_state_sha256": _runtime_state_sha256(initial_state),
        },
    )


def _completed_runtime_receipt(
    assembly: SimpleNamespace,
    *,
    final_action_count: int,
) -> dict[str, Any]:
    receipt = dict(assembly.runtime_receipt)
    receipt["final_action_count"] = final_action_count
    return receipt


def _trace(
    result: Any,
    robot: LiberoRouteCRobot,
    grasp_provider: AnalyticTopDownGraspProvider,
    resolver: WorkspaceCenterEntityResolver,
    observer: StableSceneEstimator,
    contact_provider: LiberoRouteCContactTargetProvider,
) -> dict[str, Any]:
    attempts = [
        {
            "task_attempt": item.task_attempt,
            "phase": item.phase.value,
            "attempt": item.attempt,
            "success": item.success,
            "detail": item.detail,
        }
        for item in result.attempts
    ]
    verification = None
    if result.verification is not None:
        verification = {
            "success": result.verification.success,
            "relation": result.verification.relation.value,
            "position_error_m": result.verification.position_error_m,
            "detail": result.verification.detail,
        }
    planned_goals = [
        {
            "index": item.index,
            "goal": item.goal.to_dict(),
            "success": item.success,
            "failure": item.failure,
            "attempt_count": len(item.attempts),
        }
        for item in getattr(result, "goals", ())
    ]
    return {
        "controller_success": result.success,
        "controller_failure": result.failure,
        "evaluator_terminated": False,
        "attempts": attempts,
        "verification": verification,
        "planned_goals": planned_goals,
        "grasp_mode": robot.active_grasp_mode.value,
        "mpc_phase_chunks": list(robot.phase_trace),
        "grasp_checks": list(robot.grasp_checks),
        "grasp_proposal": dict(grasp_provider.last_proposal_trace),
        "selector_resolution": dict(resolver.last_resolution_trace),
        "selector_resolution_history": list(resolver.resolution_history),
        "selector_diagnostics": list(observer.selector_diagnostics),
        "obstacle_filter": list(observer.obstacle_filter_history),
        "contact_target": dict(contact_provider.last_contact_trace),
        "optimizer_path_executed": bool(robot.phase_trace),
        # Formal verification requires a populated sensor-content commitment
        # and a constant legacy timestamp; no evaluator step/time value enters
        # this runtime.
        "freshness_authority": "sha256_of_whitelisted_sensor_content",
    }


def run_route_c_policy_episode(
    *,
    bundle: PerceptionBundle | Any,
    instruction: str,
    current_observation: SensorProvider,
    execute_action: ActionExecutor,
    step_budget: int,
) -> RouteCPolicyExecution:
    """Execute one Route C episode from capability-limited arguments only."""

    if type(instruction) is not str or not instruction.strip():
        raise ValueError("Route C instruction must be a non-empty native string")
    if type(step_budget) is not int or step_budget < 1:
        raise ValueError("Route C step_budget must be a positive native integer")
    if not callable(current_observation) or not callable(execute_action):
        raise TypeError("Route C sensor and action capabilities must be callable")

    def frames():
        current = current_observation()
        if type(current) is not RobotObservation:
            raise TypeError("Route C sensor capability returned a non-whitelist DTO")
        return tuple(
            coerce_rgbd_frame(
                frame,
                name=name,
                timestamp_s=0.0,
                capture_id=camera_content_sha256(current, name),
            )
            for name, frame in current.cameras.items()
        )

    assembly = _assemble_route_c_episode_runtime(
        bundle=bundle,
        frames=frames,
        current_observation=current_observation,
        execute_action=execute_action,
        step_budget=step_budget,
    )
    try:
        result = assembly.coordinator.run(instruction)
    except PolicyStepBudgetExhausted as exc:
        # Fixed-horizon exhaustion is a valid policy outcome, not a process
        # or evaluator failure.  Preserve issued grasp state so the parent can
        # close any pending engagement exactly once in its formal audit.
        result = SimpleNamespace(
            success=False,
            failure=str(exc) or "fixed Route C policy horizon exhausted",
            attempts=(),
            verification=None,
            goals=(),
            grasp_attempt_events=assembly.controller.grasp_attempt_events,
        )
    trace = _trace(
        result,
        assembly.robot,
        assembly.grasp_provider,
        assembly.resolver,
        assembly.observer,
        assembly.contact_provider,
    )
    trace["episode_runtime"] = _completed_runtime_receipt(
        assembly,
        final_action_count=assembly.robot.steps_executed,
    )
    return RouteCPolicyExecution(
        success=bool(result.success),
        failure=result.failure,
        steps_executed=assembly.robot.steps_executed,
        trace=trace,
        grasp_attempt_events=tuple(result.grasp_attempt_events),
        pending_grasp_engagement=getattr(
            assembly.controller,
            "pending_grasp_engagement",
            None,
        ),
    )


def run_route_c_policy_test_episode(
    *,
    bundle: PerceptionBundle | Any,
    instruction: str,
    current_observation: SensorProvider,
    execute_action: ActionExecutor,
    step_budget: int,
) -> RouteCPolicyExecution:
    """Exercise the production assembly/reset boundary with one cheap action.

    This hook is compiled into the fixed read-only policy projection.  It
    cannot be supplied through ``PYTHONPATH`` and deliberately constructs the
    same mutable Route C runtime as a normal episode before issuing one hold.
    """

    if type(instruction) is not str or not instruction.strip():
        raise ValueError("Route C test instruction must be non-empty")
    if type(step_budget) is not int or step_budget < 1:
        raise ValueError("Route C test step_budget must be positive")

    def frames():
        current = current_observation()
        if type(current) is not RobotObservation:
            raise TypeError("Route C sensor capability returned a non-whitelist DTO")
        return tuple(
            coerce_rgbd_frame(
                frame,
                name=name,
                timestamp_s=0.0,
                capture_id=camera_content_sha256(current, name),
            )
            for name, frame in current.cameras.items()
        )

    assembly = _assemble_route_c_episode_runtime(
        bundle=bundle,
        frames=frames,
        current_observation=current_observation,
        execute_action=execute_action,
        step_budget=step_budget,
    )
    feedback = assembly.robot.capture_fresh_sensor_frame_at_pose(-1.0)
    if not feedback.accepted or assembly.robot.steps_executed != 1:
        raise RuntimeError("production-shaped Route C test action was rejected")
    return RouteCPolicyExecution(
        success=True,
        failure=None,
        steps_executed=1,
        trace={
            "test_policy": "production_assembly_one_hold_action",
            "episode_local_action_index": assembly.robot.steps_executed,
            "freshness_authority": (
                "both_camera_rgbd_calibration_sha256_separate_from_proprioception"
            ),
            "episode_runtime": _completed_runtime_receipt(
                assembly,
                final_action_count=assembly.robot.steps_executed,
            ),
        },
        grasp_attempt_events=(),
        pending_grasp_engagement=None,
    )


__all__ = [
    "RouteCPolicyExecution",
    "run_route_c_policy_episode",
    "run_route_c_policy_test_episode",
]
