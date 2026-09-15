"""End-to-end LIBERO evaluator for the two sensor-only routes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Protocol
import uuid

from anchor.common import (
    EvaluationSignal,
    GraspAttemptEvent,
    GraspOutcome,
    LiberoEnvAdapter,
    LiberoEnvConfig,
    OSCAction,
    PendingGraspEngagement,
    PolicyTask,
    RobotObservation,
)
from anchor.integration.adapters import (
    AnalyticTopDownGraspProvider,
    ContactAwarePlanningController,
    LiberoPlanningContactTargetProvider,
    LiberoPlanningRobot,
    AnchorPolicy,
    SensorBoundGeometryVerifier,
    SensorBoundGoalSynthesizer,
    SensorFormedStackHandler,
    SensorRankedTargetHandler,
    StableSceneEstimator,
    WorkspaceCenterEntityResolver,
)
from anchor.integration.components import (
    PerceptionBundle,
    PerceptionFactoryContext,
    build_perception_bundle,
)
from anchor.integration.campaign_plan import orchestration_source_sha256
from anchor.integration.config import EvaluationConfig, source_tree_sha256
from anchor.integration.formal_provenance import (
    LIFECYCLE_SCHEMA,
    FormalProvenanceError,
    load_shard_provenance,
    opaque_policy_token_for_attempt,
    validate_provenance_for_config,
)
from anchor.integration.planning_process import (
    FormalPlanningPolicyProcess,
    validate_process_boundary_attestation,
)
from anchor.integration.grasp_audit import (
    AuditRoute,
    CommandedJawBehavior,
    GraspAuditTrail,
    GraspEvidenceSource,
    GraspReasonCode,
    ObservedGraspMode,
    build_grasp_attempt_counts,
    build_grasp_audit_report,
    validate_grasp_attempt_counts,
    validate_grasp_audit_report,
)
from anchor.integration.results import (
    EpisodeKey,
    EpisodeRecord,
    JsonlResultStore,
    episode_schedule,
)
from anchor.integration.policy_boundary_audit import (
    PolicyBoundaryAuditor,
    validate_policy_boundary_audit,
)
from anchor.integration.video import DualViewVideoRecorder
from anchor.integration.video_audit import audit_dual_view_video
from anchor.perception import coerce_rgbd_frame
from anchor.manipulation import AnchorController
from anchor.planning import (
    AtomicGoalGraphCompiler,
    EntityGraspModeSelector,
    GraspBinder,
    MPCContactGoalExecutor,
    OptimizerConfig,
    RecedingHorizonOptimizer,
    PlanningControllerConfig,
    PlanningSequentialCoordinator,
    TrajectoryOptimizer,
)


class EnvironmentFactory(Protocol):
    def __call__(self, config: LiberoEnvConfig) -> LiberoEnvAdapter: ...


def prepare_runtime(config: EvaluationConfig) -> None:
    """Apply renderer and config settings before LIBERO imports robosuite."""

    import random
    import numpy as np
    import torch

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    config_file = config.libero_config_path / "config.yaml"
    if not config_file.is_file():
        raise FileNotFoundError(
            f"LIBERO configuration is missing: {config_file}. "
            "Run the repository setup first or pass --libero-config-path."
        )
    os.environ["LIBERO_CONFIG_PATH"] = str(config.libero_config_path)
    os.environ["MUJOCO_GL"] = config.render_backend
    os.environ["PYOPENGL_PLATFORM"] = config.render_backend
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/libero-routes-numba")
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/libero-routes-matplotlib")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _episode_seed(config: EvaluationConfig, key: EpisodeKey) -> int:
    # VLA comparison protocol: fixed environment seed for every official init.
    return config.seed


def _opaque_policy_episode_id(attempt_id: str | None = None) -> str:
    """Return fresh, non-semantic entropy for one policy episode.

    A hash of ``EpisodeKey`` is enumerable over LIBERO's small schedule and is
    therefore not opaque.  Formal evaluation reuses the already-random
    attempt UUID; direct test/development calls receive an independent UUID.
    """

    if attempt_id is None:
        attempt_id = uuid.uuid4().hex
    try:
        return opaque_policy_token_for_attempt(attempt_id)
    except FormalProvenanceError as exc:
        raise ValueError("attempt_id must be a canonical UUIDv4 hex string") from exc


def _video_path(config: EvaluationConfig, key: EpisodeKey) -> Path:
    method = "anchor" if key.route == "b" else "planning"
    name = f"{method}_{key.suite}_task-{key.task_id:02d}_ep-{key.episode_index:04d}.mp4"
    return config.video_dir / name


def _recorder(config: EvaluationConfig, key: EpisodeKey) -> DualViewVideoRecorder | None:
    if not config.record_video:
        return None
    return DualViewVideoRecorder(
        _video_path(config, key),
        fps=config.video_fps,
        stride=config.video_stride,
    )


@dataclass(slots=True)
class _ExternalStickyScore:
    """Evaluator-owned score latch that cannot affect route control flow."""

    success_observed: bool = False
    step_signals_observed: int = 0
    final_checks: int = 0

    def observe(self, signal: EvaluationSignal) -> None:
        self.step_signals_observed += 1
        self.success_observed = bool(self.success_observed or signal.success)

    def finalize(self, environment: LiberoEnvAdapter) -> bool:
        # A reset state or policy-requested zero-action stop can already
        # satisfy a benchmark predicate.  This final evaluator-side query is
        # score-only and its result is never returned to a route.
        self.final_checks += 1
        final_success = bool(environment.check_success())
        self.success_observed = bool(self.success_observed or final_success)
        return self.success_observed

    def audit(self) -> dict[str, Any]:
        zero_delivery = {
            "reward": 0,
            "success": 0,
            "done": 0,
            "terminated": 0,
            "truncated": 0,
        }
        return {
            "scoring": "external_sticky_any_success",
            "step_signals_observed_externally": self.step_signals_observed,
            "final_evaluator_checks": self.final_checks,
            "delivered_to_policy": dict(zero_delivery),
            "delivered_to_controller": dict(zero_delivery),
            "evaluator_derived_clock_or_step_inputs_to_policy_or_controller": 0,
            "evaluator_driven_early_stops": 0,
        }


@dataclass(slots=True)
class _SensorOnlyActionDispatcher:
    """Execute Planning actions while returning observations and nothing else."""

    environment: LiberoEnvAdapter
    recorder: DualViewVideoRecorder | None
    score: _ExternalStickyScore
    _generation: int = field(default=0, init=False)

    @property
    def sensor_generation(self) -> int:
        """Monotonic frame generation without exposing simulator clocks."""

        return self._generation

    def __call__(self, action: OSCAction) -> RobotObservation:
        self._generation += 1
        step = self.environment.step(action)
        if self.recorder:
            self.recorder.add(step.observation)
        self.score.observe(step.evaluation)
        return step.observation


@dataclass(frozen=True, slots=True)
class _FormalRunContext:
    provenance: dict[str, Any]
    provenance_sha256: str

    @property
    def expected_source_sha256(self) -> str:
        return str(self.provenance["locks"]["source_tree_sha256"])

    @property
    def expected_orchestration_source_sha256(self) -> str:
        return str(self.provenance["locks"]["orchestration_source_sha256"])

    def check_source(self) -> str:
        current = source_tree_sha256()
        if current != self.expected_source_sha256:
            raise FormalProvenanceError(
                "source tree changed during formal evaluation: "
                f"expected={self.expected_source_sha256}, current={current}"
            )
        current_orchestration = orchestration_source_sha256(
            source_tree_digest=current
        )
        if current_orchestration != self.expected_orchestration_source_sha256:
            raise FormalProvenanceError(
                "formal orchestration source changed during evaluation: "
                f"expected={self.expected_orchestration_source_sha256}, "
                f"current={current_orchestration}"
            )
        return current


def _commit_policy_boundary_audit(
    auditor: PolicyBoundaryAuditor | None,
    environment: LiberoEnvAdapter,
    *,
    steps: int,
    route_trace: dict[str, Any],
) -> None:
    """Validate and attach a formal audit without reading evaluator signals."""

    if auditor is None:
        return
    # This is the sole evaluator-owned value consumed by the audit.  Only the
    # integer action count and its fixed source label are recorded; no reward,
    # termination, or success object is inspected here.
    auditor.record_evaluator_step_count(environment.episode_steps)
    report = auditor.to_report()
    validate_policy_boundary_audit(report)
    evaluator_steps = report["evaluator_step_count"]["value"]
    if evaluator_steps != steps:
        raise RuntimeError(
            "formal policy boundary evaluator step count differs from row steps: "
            f"audit={evaluator_steps}, row={steps}"
        )
    route_trace["policy_boundary_audit"] = report


def _commit_grasp_audit(
    *,
    route: AuditRoute,
    episode_id: str,
    completed_events: tuple[GraspAttemptEvent, ...],
    pending_engagement: PendingGraspEngagement | None,
    fixed_horizon_exhausted: bool,
    route_trace: dict[str, Any],
) -> None:
    """Atomically attach a formal audit of issued mechanical engagements.

    A proposal is deliberately not an attempt.  Only route-local completed
    engagement events are converted.  ANCHOR's immutable pending snapshot is
    terminally represented here when, and only when, the evaluator's fixed
    action horizon has ended the episode.
    """

    if "grasp_audit" in route_trace or "grasp_attempt_counts" in route_trace:
        raise RuntimeError("formal grasp audit fields already exist")
    trail = GraspAuditTrail(route, episode_id)
    for expected_index, event in enumerate(completed_events, start=1):
        if not isinstance(event, GraspAttemptEvent):
            raise TypeError(
                "completed grasp journal entries must be GraspAttemptEvent"
            )
        if event.attempt_index != expected_index:
            raise RuntimeError(
                "route-local grasp attempt indices must be contiguous: "
                f"expected {expected_index}, got {event.attempt_index}"
            )
        if event.accepted != (event.outcome is GraspOutcome.ACCEPTED):
            raise RuntimeError("route-local grasp outcome disagrees with accepted")
        trail.append(
            target_text=event.source_text,
            target_class=event.source_class,
            grasp_mode=ObservedGraspMode(event.grasp_mode),
            jaw_behavior=CommandedJawBehavior(event.jaw_behavior.value),
            evidence_source=GraspEvidenceSource(event.evidence_source.value),
            accepted=event.accepted,
            reason_code=GraspReasonCode(event.reason.value),
        )

    pending_count = 0
    if pending_engagement is not None:
        if not isinstance(pending_engagement, PendingGraspEngagement):
            raise TypeError("pending grasp must be an immutable engagement snapshot")
        if route is not AuditRoute.B:
            raise RuntimeError("Planning cannot persist a pending grasp engagement")
        if not fixed_horizon_exhausted:
            raise RuntimeError(
                "pending ANCHOR grasp may only be closed by fixed-horizon exhaustion"
            )
        if pending_engagement.attempt_index != len(completed_events) + 1:
            raise RuntimeError(
                "pending grasp attempt index must immediately follow completed events"
            )
        trail.append(
            target_text=pending_engagement.source_text,
            target_class=pending_engagement.source_class,
            grasp_mode=ObservedGraspMode(pending_engagement.grasp_mode),
            jaw_behavior=CommandedJawBehavior(
                pending_engagement.jaw_behavior.value
            ),
            evidence_source=GraspEvidenceSource.CONTROLLER_EXECUTION,
            accepted=False,
            reason_code=GraspReasonCode.STEP_BUDGET_EXHAUSTED,
        )
        pending_count = 1

    report = build_grasp_audit_report(route, episode_id, trail.records)
    counts = build_grasp_attempt_counts(
        completed=len(completed_events), pending=pending_count
    )
    # Validate the exact persisted forms before mutating the trace.  Thus an
    # invalid black-bowl mode or malformed journal cannot leave a partial row.
    validated_report = validate_grasp_audit_report(
        report,
        expected_route=route,
        expected_episode_id=episode_id,
    )
    validated_counts = validate_grasp_attempt_counts(
        counts,
        validated_report["records"],
        route=route,
    )
    route_trace["grasp_audit"] = validated_report
    route_trace["grasp_attempt_counts"] = validated_counts


def _json_normalized(value: Any) -> Any:
    """Match the representation persisted by ``json.dump`` (tuples become lists)."""

    return json.loads(json.dumps(value, sort_keys=True))


def _validate_resume_provenance(
    config: EvaluationConfig,
    store: JsonlResultStore,
    schedule: tuple[EpisodeKey, ...],
) -> None:
    """Refuse to mix episodes produced by different evaluation settings."""

    try:
        previous = json.loads(config.summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            "--resume requires the existing, valid summary.json written with the JSONL"
        ) from exc
    old_config = previous.get("run_config")
    if not isinstance(old_config, dict):
        raise ValueError("--resume requires summary.json.run_config")
    new_config = _json_normalized(config.serializable())
    ignored = {"resume"}
    keys = (set(old_config) | set(new_config)) - ignored
    mismatches = [
        key
        for key in sorted(keys)
        if _json_normalized(old_config.get(key)) != _json_normalized(new_config.get(key))
    ]
    if mismatches:
        detail = ", ".join(mismatches)
        raise ValueError(
            f"--resume configuration differs from the existing run: {detail}; "
            "choose a new --run-name"
        )
    expected_ids = {key.id for key in schedule}
    unexpected = sorted(store.completed_ids() - expected_ids)
    if unexpected:
        raise ValueError(
            "--resume JSONL contains episodes outside the configured schedule: "
            + ", ".join(unexpected[:3])
        )


def _load_formal_context(
    config: EvaluationConfig, schedule: tuple[EpisodeKey, ...]
) -> _FormalRunContext | None:
    if not config.formal:
        return None
    assert config.formal_provenance_path is not None
    assert config.formal_provenance_sha256 is not None
    payload, digest = load_shard_provenance(
        config.formal_provenance_path,
        expected_sha256=config.formal_provenance_sha256,
    )
    validate_provenance_for_config(
        payload,
        config.serializable(),
        run_directory=config.trace_path.parent,
    )
    expected_ids = [key.id for key in schedule]
    if payload.get("expected_episode_ids") != expected_ids:
        raise FormalProvenanceError(
            "formal schedule differs from shard provenance"
        )
    context = _FormalRunContext(payload, digest)
    context.check_source()
    return context


def _planning_trace(
    result: Any,
    robot: LiberoPlanningRobot,
    grasp_provider: AnalyticTopDownGraspProvider | None = None,
    resolver: WorkspaceCenterEntityResolver | None = None,
    observer: StableSceneEstimator | None = None,
    contact_provider: LiberoPlanningContactTargetProvider | None = None,
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
    ] if result is not None else []
    verification = None
    if result is not None and result.verification is not None:
        verification = {
            "success": result.verification.success,
            "relation": result.verification.relation.value,
            "position_error_m": result.verification.position_error_m,
            "detail": result.verification.detail,
        }
    planned_goals = []
    if result is not None:
        for item in getattr(result, "goals", ()):
            planned_goals.append(
                {
                    "index": item.index,
                    "goal": item.goal.to_dict(),
                    "success": item.success,
                    "failure": item.failure,
                    "attempt_count": len(item.attempts),
                }
            )
    return {
        "controller_success": result.success if result is not None else None,
        "controller_failure": result.failure if result is not None else None,
        "evaluator_terminated": result is None,
        "attempts": attempts,
        "verification": verification,
        "planned_goals": planned_goals,
        "grasp_mode": robot.active_grasp_mode.value,
        "mpc_phase_chunks": list(robot.phase_trace),
        "grasp_checks": list(robot.grasp_checks),
        "grasp_proposal": (
            dict(grasp_provider.last_proposal_trace)
            if grasp_provider is not None
            else {}
        ),
        "selector_resolution": (
            dict(resolver.last_resolution_trace)
            if resolver is not None
            else {}
        ),
        "selector_resolution_history": (
            list(resolver.resolution_history)
            if resolver is not None
            else []
        ),
        "selector_diagnostics": (
            list(observer.selector_diagnostics)
            if observer is not None
            else []
        ),
        "obstacle_filter": (
            list(observer.obstacle_filter_history)
            if observer is not None
            else []
        ),
        "contact_target": (
            dict(contact_provider.last_contact_trace)
            if contact_provider is not None
            else {}
        ),
        "optimizer_path_executed": bool(robot.phase_trace),
    }


@dataclass
class EvaluationDriver:
    config: EvaluationConfig
    environment_factory: EnvironmentFactory = LiberoEnvAdapter
    perception_bundle: PerceptionBundle | Any | None = None

    def run(self) -> dict[str, Any]:
        prepare_runtime(self.config)
        schedule = episode_schedule(
            self.config.route,
            self.config.suite,
            self.config.task_ids,
            self.config.episodes_per_task,
            self.config.init_state_start,
        )
        formal_context = _load_formal_context(self.config, schedule)
        if formal_context is not None:
            formal_root = self.config.trace_path.parent.resolve()
            expected_provenance = (formal_root / "provenance.json").resolve()
            actual_provenance = Path(
                str(self.config.formal_provenance_path)
            ).expanduser().resolve()
            if actual_provenance != expected_provenance:
                raise FormalProvenanceError(
                    "formal campaign provenance must be the canonical "
                    "run-directory provenance.json"
                )
            unexpected = [
                entry.name
                for entry in formal_root.iterdir()
                if entry.resolve() != expected_provenance
            ]
            if unexpected:
                raise FileExistsError(
                    "fresh formal run directory may initially contain only "
                    f"provenance.json; found={sorted(unexpected)}"
                )
        store = JsonlResultStore(
            self.config.trace_path,
            self.config.summary_path,
            formal_provenance_sha256=(
                formal_context.provenance_sha256
                if formal_context is not None
                else None
            ),
            expected_episode_ids=(key.id for key in schedule),
            source_check=(
                formal_context.check_source
                if formal_context is not None
                else None
            ),
        )
        has_run_artifacts = self.config.trace_path.exists() or self.config.summary_path.exists()
        if formal_context is not None and self.config.resume:
            raise FormalProvenanceError(
                "formal campaign episodes are fresh-only and cannot use --resume"
            )
        if formal_context is not None and has_run_artifacts:
            raise FileExistsError(
                f"fresh formal run directory already contains result artifacts: "
                f"{self.config.output_dir / self.config.run_name}"
            )
        if has_run_artifacts and not self.config.resume:
            raise FileExistsError(
                f"{self.config.output_dir / self.config.run_name} already contains run artifacts; "
                "pass --resume or choose another --run-name"
            )
        if has_run_artifacts and self.config.resume:
            _validate_resume_provenance(self.config, store, schedule)
        existing = store.completed_ids()
        if not has_run_artifacts:
            # Persist the effective configuration before the first expensive
            # episode.  Formal runs are one-shot; diagnostic runs may still
            # use this summary for their explicitly requested resume.
            store.write_summary(run_config=self.config.serializable())
        pending = [key for key in schedule if key.id not in existing]
        if not pending:
            if formal_context is not None:
                formal_context.check_source()
            return store.write_summary(run_config=self.config.serializable())

        planning_process: FormalPlanningPolicyProcess | None = None
        owns_bundle = False
        bundle: PerceptionBundle | Any | None = None
        if formal_context is not None and self.config.route == "c":
            if self.perception_bundle is not None:
                raise FormalProvenanceError(
                    "formal Planning forbids an in-process perception bundle"
                )
            if self.config.perception_factory is not None:
                raise FormalProvenanceError(
                    "formal Planning forbids a custom in-process perception factory"
                )
        use_planning_process = bool(
            self.config.route == "c"
            and self.perception_bundle is None
            and self.config.perception_factory is None
        )
        if use_planning_process:
            planning_process = FormalPlanningPolicyProcess(
                PerceptionFactoryContext.from_evaluation_config(self.config)
            )
        else:
            owns_bundle = self.perception_bundle is None
            bundle = self.perception_bundle or build_perception_bundle(self.config)
        try:
            anchor_policy = None
            if self.config.route == "b":
                assert bundle is not None
                anchor_policy = AnchorPolicy(AnchorController(bundle.for_anchor()))
            for ordinal, key in enumerate(pending, start=1):
                if formal_context is not None:
                    formal_context.check_source()
                record = self._run_one(
                    key,
                    bundle,
                    anchor_policy,
                    formal_context=formal_context,
                    planning_process=planning_process,
                )
                if formal_context is not None:
                    formal_context.check_source()
                store.append(record)
                if formal_context is not None:
                    formal_context.check_source()
                summary = store.write_summary(run_config=self.config.serializable())
                if formal_context is not None:
                    formal_context.check_source()
                total = summary["overall"]["episodes"]
                successes = summary["overall"]["successes"]
                print(
                    f"[{ordinal}/{len(pending)}] {key.id} "
                    f"success={record.evaluator_success} cumulative={successes}/{total}",
                    flush=True,
                )
        finally:
            if planning_process is not None:
                planning_process.close()
            if owns_bundle:
                close = getattr(bundle, "close", None)
                if callable(close):
                    close()
        if formal_context is not None:
            formal_context.check_source()
        return store.write_summary(run_config=self.config.serializable())

    def _make_environment(self, key: EpisodeKey) -> LiberoEnvAdapter:
        return self.environment_factory(
            LiberoEnvConfig(
                suite_name=self.config.suite,
                task_id=key.task_id,
                init_state_index=key.episode_index,
                width=self.config.image_size,
                height=self.config.image_size,
                max_steps=self.config.max_steps,
                seed=_episode_seed(self.config, key),
            )
        )

    def _run_one(
        self,
        key: EpisodeKey,
        bundle: Any,
        anchor_policy: AnchorPolicy | None,
        *,
        formal_context: _FormalRunContext | None = None,
        planning_process: FormalPlanningPolicyProcess | Any | None = None,
    ) -> EpisodeRecord:
        if self.config.route == "c" and self.config.formal:
            if planning_process is None:
                raise FormalProvenanceError(
                    "formal Planning requires its spawned serialized policy process"
                )
        elif self.config.route != "c" and planning_process is not None:
            raise ValueError(
                "a formal Planning policy process cannot be used by this run"
            )
        environment = self._make_environment(key)
        recorder = _recorder(self.config, key)
        instruction = f"LIBERO {self.config.suite} task {key.task_id}"
        started = time.monotonic()
        started_at_unix_ns = time.time_ns()
        attempt_id = uuid.uuid4().hex
        score = _ExternalStickyScore()
        try:
            metadata = environment.task_metadata
            instruction = metadata.instruction
            # Construct the exact two-field whitelist envelope once at the
            # evaluator boundary.  Planning intentionally forwards only its
            # language field into the coordinator; the opaque id remains an
            # audit/correlation token and carries no benchmark identity.
            policy_task = PolicyTask(
                instruction, _opaque_policy_episode_id(attempt_id)
            )
            boundary_auditor = (
                PolicyBoundaryAuditor(
                    self.config.route,
                    expected_height=self.config.image_size,
                    expected_width=self.config.image_size,
                )
                if self.config.formal or planning_process is not None
                else None
            )
            if self.config.route == "b":
                assert anchor_policy is not None
                payload = self._run_b(
                    environment,
                    anchor_policy,
                    recorder,
                    key,
                    instruction,
                    score,
                    policy_task=policy_task,
                    boundary_auditor=boundary_auditor,
                )
            else:
                payload = self._run_c(
                    environment,
                    bundle,
                    recorder,
                    instruction,
                    score,
                    policy_task=policy_task,
                    boundary_auditor=boundary_auditor,
                    planning_process=planning_process,
                )
            elapsed = time.monotonic() - started
            video_paths = recorder.close() if recorder else {}
            lifecycle = None
            provenance_sha256 = None
            if self.config.formal:
                if formal_context is None:
                    raise FormalProvenanceError(
                        "formal episode requires its loaded execution provenance"
                    )
                if not isinstance(video_paths.get("dual"), str):
                    raise RuntimeError(
                        "formal episode did not finalize its dual-view video"
                    )
                video_path = Path(video_paths["dual"]).expanduser().resolve()
                video_paths = {**video_paths, "dual": str(video_path)}
                video_content_audit = audit_dual_view_video(
                    video_path,
                    expected_frames=(
                        int(payload["steps"]) // self.config.video_stride + 1
                    ),
                    expected_height=self.config.image_size,
                    expected_half_width=self.config.image_size,
                )
                if video_content_audit.get("formal_pass") is not True:
                    reasons = video_content_audit.get("reasons")
                    raise RuntimeError(
                        "formal dual-view video content audit failed: "
                        f"{reasons!r}"
                    )
                status = str(payload["policy_status"])
                stop_reason = status if status in {"succeeded", "failed", "timeout"} else None
                if stop_reason is None:
                    raise RuntimeError(
                        f"formal episode has invalid policy status {status!r}"
                    )
                ended_at_unix_ns = max(time.time_ns(), started_at_unix_ns + 1)
                execution = formal_context.provenance["execution"]
                lifecycle = {
                    "schema": LIFECYCLE_SCHEMA,
                    "state": "completed",
                    "attempt_id": attempt_id,
                    "execution_nonce": execution["execution_nonce"],
                    "shard_execution_nonce": execution[
                        "shard_execution_nonce"
                    ],
                    "started_at_unix_ns": started_at_unix_ns,
                    "ended_at_unix_ns": ended_at_unix_ns,
                    "reset_completed": True,
                    "score_finalized": True,
                    "video_finalized": True,
                    "stop_owner": (
                        "fixed_horizon" if status == "timeout" else "policy"
                    ),
                    "stop_reason": stop_reason,
                    "action_steps": int(payload["steps"]),
                    # This exact, JSON-native report commits the closed video
                    # bytes and decoded content before the episode row can be
                    # appended.  The strict verifier independently recomputes
                    # it and requires byte-for-byte JSON equality.
                    "video_content_audit": video_content_audit,
                }
                provenance_sha256 = self.config.formal_provenance_sha256
            return EpisodeRecord(
                key=key,
                instruction=instruction,
                evaluator_success=payload["evaluator_success"],
                policy_status=payload["policy_status"],
                steps=payload["steps"],
                elapsed_s=elapsed,
                seed=_episode_seed(self.config, key),
                failure=payload.get("failure"),
                video_paths=video_paths,
                route_trace=payload.get("route_trace", {}),
                provenance_sha256=provenance_sha256,
                lifecycle=lifecycle,
            )
        except Exception as exc:
            exception_video_paths = recorder.close() if recorder else {}
            if self.config.formal:
                raise
            success = False
            exception_steps = int(getattr(environment, "episode_steps", 0))
            try:
                success = score.finalize(environment)
            except Exception:
                success = score.success_observed
            return EpisodeRecord(
                key=key,
                instruction=instruction,
                evaluator_success=success,
                policy_status="exception",
                steps=exception_steps,
                elapsed_s=time.monotonic() - started,
                seed=_episode_seed(self.config, key),
                failure=f"{type(exc).__name__}: {exc}",
                video_paths=exception_video_paths,
                route_trace={
                    "exception_type": type(exc).__name__,
                    "evaluator_isolation": score.audit(),
                },
            )
        finally:
            environment.close()

    def _run_b(
        self,
        environment: LiberoEnvAdapter,
        policy: AnchorPolicy,
        recorder: DualViewVideoRecorder | None,
        key: EpisodeKey,
        instruction: str,
        score: _ExternalStickyScore | None = None,
        *,
        policy_task: PolicyTask | None = None,
        boundary_auditor: PolicyBoundaryAuditor | None = None,
    ) -> dict[str, Any]:
        observation = environment.reset()
        task = policy_task or PolicyTask(instruction, _opaque_policy_episode_id())
        if boundary_auditor is not None:
            boundary_auditor.record_reset(task)
        policy.reset(task)
        if recorder:
            recorder.add(observation)
        score = score or _ExternalStickyScore()
        steps = 0
        phase_transitions: list[dict[str, Any]] = []
        last_phase: str | None = None
        while steps < environment.max_steps:
            decision = policy.act(observation)
            if boundary_auditor is not None:
                sequence_token = policy.observation_sequence_token
                if sequence_token is None:
                    raise RuntimeError(
                        "ANCHOR did not expose the controller-owned observation token"
                    )
                boundary_auditor.record_act(
                    observation,
                    act_index=boundary_auditor.act_delivery_count,
                    internal_sequence=sequence_token,
                )
            route_decision = policy.last_decision
            if route_decision is not None and route_decision.phase != last_phase:
                phase_transitions.append(
                    {
                        "step": steps,
                        "skill_index": route_decision.skill_index,
                        "phase": route_decision.phase,
                    }
                )
                last_phase = route_decision.phase
            if decision.request_stop:
                break
            env_step = environment.step(decision.action)
            steps += 1
            observation = env_step.observation
            score.observe(env_step.evaluation)
            if recorder:
                recorder.add(observation)
        evaluator_success = score.finalize(environment)
        final = policy.last_decision
        status = final.status.value if final is not None else "no_decision"
        failure = final.message if final is not None and status == "failed" else None
        if steps >= environment.max_steps and status == "running":
            status = "timeout"
            failure = f"policy remained running at the {environment.max_steps}-step limit"
        trace = {
            "status": status,
            "skill_index": final.skill_index if final is not None else None,
            "phase": final.phase if final is not None else None,
            "message": final.message if final is not None else None,
            "phase_transitions": phase_transitions,
            "grasp_mode": policy.controller.grasp_mode.value,
            "grasp_target_attempts": list(policy.grasp_target_attempts),
            "grasp_verifications": list(policy.grasp_verifications),
            "placement_target_attempts": list(policy.placement_target_attempts),
            "selector_diagnostics": list(policy.selector_diagnostics),
            "evaluator_isolation": score.audit(),
        }
        drawer_contact_proof = getattr(policy, "drawer_contact_proof", None)
        if isinstance(drawer_contact_proof, Mapping):
            trace["drawer_contact_proof"] = dict(drawer_contact_proof)
        if boundary_auditor is not None:
            _commit_grasp_audit(
                route=AuditRoute.B,
                episode_id=task.episode_id,
                completed_events=tuple(
                    getattr(policy, "grasp_attempt_events", ())
                ),
                pending_engagement=getattr(
                    policy, "pending_grasp_engagement", None
                ),
                fixed_horizon_exhausted=(
                    status == "timeout" and steps >= environment.max_steps
                ),
                route_trace=trace,
            )
        _commit_policy_boundary_audit(
            boundary_auditor,
            environment,
            steps=steps,
            route_trace=trace,
        )
        return {
            "evaluator_success": evaluator_success,
            "policy_status": status,
            "failure": failure,
            "steps": steps,
            "route_trace": trace,
        }

    def _run_c(
        self,
        environment: LiberoEnvAdapter,
        bundle: Any,
        recorder: DualViewVideoRecorder | None,
        instruction: str,
        score: _ExternalStickyScore | None = None,
        *,
        policy_task: PolicyTask | None = None,
        boundary_auditor: PolicyBoundaryAuditor | None = None,
        planning_process: FormalPlanningPolicyProcess | Any | None = None,
    ) -> dict[str, Any]:
        observation = environment.reset()
        if recorder:
            recorder.add(observation)
        score = score or _ExternalStickyScore()
        if boundary_auditor is not None:
            if policy_task is None:
                raise RuntimeError("formal Planning requires the policy task envelope")
            # This audits the evaluator-to-route whitelist envelope before any
            # potentially eager sensor consumer is constructed.  Only the
            # instruction is subsequently forwarded into the coordinator.
            policy_instruction = (
                boundary_auditor.adapt_planning_task_to_instruction(policy_task)
            )
        else:
            policy_instruction = instruction

        if planning_process is not None:
            if boundary_auditor is None or policy_task is None:
                raise RuntimeError(
                    "isolated Planning execution requires a formal boundary audit"
                )
            execute_isolated_action = _SensorOnlyActionDispatcher(
                environment,
                recorder,
                score,
            )

            def record_serialized_observation(
                delivered: RobotObservation,
                action_count: int,
            ) -> None:
                # ``action_count`` is parent-side audit bookkeeping only.  It
                # is absent from the serialized observation and never enters
                # the policy process.
                boundary_auditor.record_act(
                    delivered,
                    act_index=action_count,
                    internal_sequence=action_count,
                )

            result = planning_process.run_episode(
                instruction=policy_instruction,
                initial_observation=observation,
                step_budget=environment.max_steps,
                action_handler=execute_isolated_action,
                observation_delivery=record_serialized_observation,
            )
            # Bind the audit to the completed live child/episode, not merely
            # to a startup declaration captured before any policy traffic.
            attestation = validate_process_boundary_attestation(
                planning_process.boundary_attestation()
            )
            episode_receipt = attestation.get("latest_episode_receipt")
            if type(episode_receipt) is not dict:
                raise RuntimeError(
                    "isolated Planning returned no completed episode receipt"
                )
            environment_steps = environment.episode_steps
            if (
                type(environment_steps) is not int
                or environment_steps < 0
                or episode_receipt.get("action_count") != environment_steps
                or episode_receipt.get("observation_count")
                != environment_steps + 1
                or result.steps_executed != environment_steps
            ):
                raise RuntimeError(
                    "isolated Planning runtime receipt does not match the "
                    "environment action/observation lifecycle"
                )
            boundary_auditor.record_planning_process_boundary(attestation)
            evaluator_success = score.finalize(environment)
            trace = dict(result.trace)
            trace["evaluator_isolation"] = score.audit()
            trace["planning_process_boundary"] = attestation
            fixed_horizon_exhausted = bool(
                not result.success
                and result.steps_executed >= environment.max_steps
            )
            _commit_grasp_audit(
                route=AuditRoute.C,
                episode_id=policy_task.episode_id,
                completed_events=tuple(result.grasp_attempt_events),
                pending_engagement=result.pending_grasp_engagement,
                fixed_horizon_exhausted=fixed_horizon_exhausted,
                route_trace=trace,
            )
            _commit_policy_boundary_audit(
                boundary_auditor,
                environment,
                steps=result.steps_executed,
                route_trace=trace,
            )
            return {
                "evaluator_success": evaluator_success,
                "policy_status": (
                    "succeeded"
                    if result.success
                    else "timeout"
                    if fixed_horizon_exhausted
                    else "failed"
                ),
                "failure": result.failure,
                "steps": result.steps_executed,
                "route_trace": trace,
            }

        def current_sensor_observation() -> RobotObservation:
            current = environment.current_observation
            if boundary_auditor is not None:
                generation = execute_action.sensor_generation
                # Every real Planning callback is recorded, including repeated
                # reads of the same immutable snapshot within one generation.
                boundary_auditor.record_act(
                    current,
                    act_index=generation,
                    internal_sequence=generation,
                )
            return current

        execute_action = _SensorOnlyActionDispatcher(environment, recorder, score)

        def frames():
            current = current_sensor_observation()
            return tuple(
                coerce_rgbd_frame(
                    frame,
                    name=name,
                    timestamp_s=float(execute_action.sensor_generation),
                )
                for name, frame in current.cameras.items()
            )

        observer = StableSceneEstimator(
            bundle.for_planning(frames),
            ee_pose_provider=lambda: (
                current_sensor_observation().proprio.T_world_ee
            ),
        )
        robot = LiberoPlanningRobot(
            current_sensor_observation,
            execute_action,
            step_budget=environment.max_steps,
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
        controller = ContactAwarePlanningController(
            compiler=graph_compiler,
            observer=observer,
            grasp_provider=grasp_provider,
            robot=robot,
            # Object scenes are sparse: execute the complete continuously
            # optimised path (transfer still tracks its safe midpoint) instead
            # of spending 12 replans on a geometric prefix tail.
            mpc=motion_mpc,
            binder=GraspBinder(resolver=resolver),
            goals=goals,
            verifier=SensorBoundGeometryVerifier(goals, observer),
            grasp_mode_selector=grasp_mode_selector,
            config=PlanningControllerConfig(
                # A failed load/lift retention check invalidates one analytic
                # sensor grasp.  A fourth attempt exposes the independently
                # sensor-derived fresh-axis/deep-inset recovery profile after
                # retaining the three established cavity profiles.  Failed
                # profiles receive no SDF bonus, so the fourth attempt cannot
                # silently repeat one of them.
                max_task_attempts=4,
                max_phase_attempts=2,
                max_mpc_replans=12,
                position_tolerance_m=0.003,
            ),
        )
        contact_provider = LiberoPlanningContactTargetProvider(
            current_sensor_observation,
            observer,
        )
        contact_executor = MPCContactGoalExecutor(
            observer,
            robot,
            RecedingHorizonOptimizer(optimizer, horizon_waypoints=10),
            contact_provider,
        )
        coordinator = PlanningSequentialCoordinator(
            controller,
            graph_compiler,
            contact_executor,
            between_goals=observer.invalidate_sensor_cache,
            source_consumed=resolver.exclude_source,
            formed_stack_handler=formed_stack_handler,
            ranked_target_handler=ranked_target_handler,
        )
        result = coordinator.run(policy_instruction)
        # Scoring remains outside PlanningController.  Success seen after any
        # action is retained even if later policy-side verification or retreat
        # subsequently disturbs the benchmark predicate.
        evaluator_success = score.finalize(environment)
        trace = _planning_trace(
            result,
            robot,
            grasp_provider,
            resolver,
            observer,
            contact_provider,
        )
        trace["evaluator_isolation"] = score.audit()
        if boundary_auditor is not None:
            assert policy_task is not None
            _commit_grasp_audit(
                route=AuditRoute.C,
                episode_id=policy_task.episode_id,
                completed_events=tuple(result.grasp_attempt_events),
                pending_engagement=getattr(
                    controller, "pending_grasp_engagement", None
                ),
                fixed_horizon_exhausted=False,
                route_trace=trace,
            )
        _commit_policy_boundary_audit(
            boundary_auditor,
            environment,
            steps=robot.steps_executed,
            route_trace=trace,
        )
        return {
            "evaluator_success": evaluator_success,
            "policy_status": "succeeded" if result.success else "failed",
            "failure": result.failure,
            "steps": robot.steps_executed,
            "route_trace": trace,
        }


def evaluate(config: EvaluationConfig) -> dict[str, Any]:
    return EvaluationDriver(config).run()


def format_summary(summary: dict[str, Any]) -> str:
    overall = summary["overall"]
    rate = overall["success_rate"]
    rate_text = "n/a" if rate is None else f"{100.0 * rate:.1f}%"
    return json.dumps(
        {
            "episodes": overall["episodes"],
            "successes": overall["successes"],
            "success_rate": rate_text,
            "elapsed_s": overall["elapsed_s"],
        },
        ensure_ascii=False,
    )
