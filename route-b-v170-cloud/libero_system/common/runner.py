"""Episode runner that keeps evaluation signals outside the policy boundary."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np

from .env_adapter import EnvironmentStep, LiberoEnvAdapter, TaskMetadata
from .policy import Policy


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    episode_id: str | None = None
    stop_on_policy_request: bool = True


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    episode_id: str
    success: bool
    terminated: bool
    truncated: bool
    policy_requested_stop: bool
    steps: int
    wall_time_s: float
    task: TaskMetadata


class EpisodeLogger(Protocol):
    def begin(self, episode_id: str, task: TaskMetadata) -> None: ...

    def step(self, record: Mapping[str, Any]) -> None: ...

    def end(self, result: EpisodeResult) -> None: ...


class NullEpisodeLogger:
    def begin(self, episode_id: str, task: TaskMetadata) -> None:  # noqa: ARG002
        pass

    def step(self, record: Mapping[str, Any]) -> None:  # noqa: ARG002
        pass

    def end(self, result: EpisodeResult) -> None:  # noqa: ARG002
        pass


class JsonlEpisodeLogger:
    """Compact telemetry logger.  RGB-D frames are not persisted by default."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self._step_file: Any | None = None
        self._episode_dir: Path | None = None

    def begin(self, episode_id: str, task: TaskMetadata) -> None:
        self._episode_dir = self.output_dir / episode_id
        self._episode_dir.mkdir(parents=True, exist_ok=False)
        (self._episode_dir / "metadata.json").write_text(
            json.dumps(asdict(task), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        self._step_file = (self._episode_dir / "steps.jsonl").open("w", encoding="utf-8")

    def step(self, record: Mapping[str, Any]) -> None:
        assert self._step_file is not None
        self._step_file.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")
        self._step_file.flush()

    def end(self, result: EpisodeResult) -> None:
        assert self._episode_dir is not None
        if self._step_file is not None:
            self._step_file.close()
            self._step_file = None
        payload = asdict(result)
        (self._episode_dir / "summary.json").write_text(
            json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


class EpisodeRunner:
    def __init__(
        self,
        environment: LiberoEnvAdapter,
        *,
        logger: EpisodeLogger | None = None,
        config: RunnerConfig | None = None,
    ):
        self.environment = environment
        self.logger = logger or NullEpisodeLogger()
        self.config = config or RunnerConfig()

    def run(self, policy: Policy) -> EpisodeResult:
        episode_id = self.config.episode_id or uuid.uuid4().hex[:12]
        task = self.environment.task_metadata
        self.logger.begin(episode_id, task)
        observation = self.environment.reset()
        policy.reset(task.policy_view(episode_id))
        started = time.monotonic()
        last_step: EnvironmentStep | None = None
        policy_stopped = False
        steps = 0
        while steps < self.environment.max_steps:
            decision_started = time.monotonic()
            decision = policy.act(observation)
            inference_s = time.monotonic() - decision_started
            if decision.request_stop and self.config.stop_on_policy_request:
                policy_stopped = True
                break
            last_step = self.environment.step(decision.action)
            steps += 1
            observation = last_step.observation
            self.logger.step(
                {
                    "step": steps,
                    "inference_s": inference_s,
                    "action": decision.action.values,
                    "gripper_width_m": observation.proprio.gripper_width_m,
                    "evaluation": asdict(last_step.evaluation),
                    "diagnostics": decision.diagnostics,
                }
            )
            if last_step.evaluation.terminated or last_step.evaluation.truncated:
                break
        evaluation = last_step.evaluation if last_step is not None else None
        result = EpisodeResult(
            episode_id=episode_id,
            success=bool(evaluation.success) if evaluation else False,
            terminated=bool(evaluation.terminated) if evaluation else False,
            truncated=bool(evaluation.truncated) if evaluation else False,
            policy_requested_stop=policy_stopped,
            steps=steps,
            wall_time_s=time.monotonic() - started,
            task=task,
        )
        self.logger.end(result)
        return result
