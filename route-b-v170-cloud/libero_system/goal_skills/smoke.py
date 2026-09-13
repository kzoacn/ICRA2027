"""Reproducible sensor-only LIBERO-Goal contact smoke runner."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from libero_system.common import LiberoEnvAdapter, LiberoEnvConfig
from libero_system.integration.video import DualViewVideoRecorder

from .controller import GoalContactPolicy


def _opaque_episode_id(task_id: int, seed: int) -> str:
    """Hash evaluator schedule metadata before it crosses the policy API."""

    structured = f"libero_goal:{int(task_id)}:{int(seed)}".encode("utf-8")
    return f"episode-{hashlib.sha256(structured).hexdigest()[:20]}"


def run(
    task_id: int,
    *,
    seed: int = 0,
    max_steps: int = 300,
    video_path: str | Path | None = None,
) -> dict[str, object]:
    config = LiberoEnvConfig(
        suite_name="libero_goal",
        task_id=task_id,
        width=256,
        height=256,
        seed=seed,
        max_steps=max_steps,
    )
    # The evaluator accumulator below is deliberately outside the policy API.
    # It is logged but never read to select an action, phase, or stop decision.
    with LiberoEnvAdapter(config) as environment:
        observation = environment.reset()
        policy = GoalContactPolicy()
        policy.reset(environment.task_metadata.policy_view(_opaque_episode_id(task_id, seed)))
        evaluator_success = False
        evaluator_terminated = False
        evaluator_truncated = False
        policy_requested_stop = False
        stop_reason = "max_steps"
        decision = None
        executed = 0
        recorder = DualViewVideoRecorder(video_path, fps=20) if video_path else None
        if recorder is not None:
            recorder.add(observation)
        try:
            for _ in range(environment.max_steps):
                # Policy acts before any evaluator signal for this step exists.
                decision = policy.act(observation)
                if decision.request_stop:
                    policy_requested_stop = True
                    stop_reason = "policy_request"
                    break
                result = environment.step(decision.action)
                executed += 1
                observation = result.observation
                if recorder is not None:
                    recorder.add(observation)
                evaluator_success = evaluator_success or result.evaluation.success
                evaluator_terminated = evaluator_terminated or result.evaluation.terminated
                evaluator_truncated = evaluator_truncated or result.evaluation.truncated
                # Standard episode termination is runner-side only.  It stops
                # the episode and is never passed back into policy.act().
                if result.evaluation.terminated:
                    stop_reason = "evaluator_terminated"
                    break
                if result.evaluation.truncated:
                    stop_reason = "evaluator_truncated"
                    break
        finally:
            videos = dict(recorder.close()) if recorder is not None else {}
        assert decision is not None
        return {
            "schema_version": 1,
            "run_kind": "formal_sensor_only",
            "suite": "libero_goal",
            "task_id": task_id,
            "instruction": environment.task_metadata.instruction,
            "seed": seed,
            "steps": executed,
            "policy_status": policy.status.value,
            "policy_phase": policy.phase,
            "policy_requested_stop": policy_requested_stop,
            "stop_reason": stop_reason,
            "evaluator": {
                "success": evaluator_success,
                "terminated": evaluator_terminated,
                "truncated": evaluator_truncated,
            },
            "policy_diagnostics": dict(decision.diagnostics),
            "policy_inputs": [
                "language",
                "opaque_episode_id",
                "agentview_rgbd_calibration",
                "wrist_rgbd_calibration",
                "proprioception",
                "wrench",
            ],
            "forbidden_policy_inputs": [
                "simulator object/body pose",
                "simulator joint state outside robot proprioception",
                "simulator contacts",
                "simulator segmentation",
                "BDDL predicates",
                "evaluator reward/success/termination",
            ],
            "integrity": (
                "evaluator signals are inspected only by the runner after an action; "
                "they can terminate the episode but cannot select any policy action"
            ),
            "videos": videos,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--video", type=Path, help="Optional dual agentview+wrist MP4")
    parser.add_argument("--output", type=Path, help="Optional JSON result path")
    arguments = parser.parse_args()
    payload = run(
        arguments.task_id,
        seed=arguments.seed,
        max_steps=arguments.max_steps,
        video_path=arguments.video,
    )
    rendered = json.dumps(payload, indent=2) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
