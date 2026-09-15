# Original 400-episode parallel evaluation

This evaluation completed with **347/400 (86.75%)**. The original scheduling history, time estimates, and operating instructions are retained below. See the [Route B improvement records](../../archive/experiments/baseline/README.md) for subsequent improvements and a fresh full reevaluation.

Original run: `smolvla_scale_400_parallel_20260913`

The evaluation follows the sample count in [SmolVLA Section 4.1](https://arxiv.org/html/2506.01844v1): Spatial, Object, Goal, and Long, with 10 tasks per suite and official initial-state indices 0–9 per task, totaling 400 episodes. It retains the deployed Route B controller, seed=7, two 256×256 RGB-D streams, action budgets of 220/280/300/520, and the original external success-scoring rule. Only the sample count is aligned; the controller, input modalities, and other experimental conditions remain those of Route B.

The background job started at 2026-09-13 15:09:18 UTC with 4 independent processes, then expanded to 8. Each process evaluates the 10 initial states of one task and automatically takes the next task on completion. Progress updates every 10 seconds, and the job runs detached from the launching terminal.

Measurements at 2026-09-13 15:14 UTC showed 34/400 completed episodes: 24 fresh and 10 reused. Stable throughput with 8 processes was approximately 427 episodes/hour, directly extrapolating to about 51 minutes remaining. Allowing for complex tasks not yet covered, failed episodes, and an uneven final workload, the suggested allowance was **1–2 hours**. Measured peak usage was approximately 16.5 GiB of GPU memory and 34.6 GiB of container memory, with average CPU utilization equivalent to 18.2 cores. The basis for the estimate is saved in `runtime/jobs/smolvla_scale_400_parallel_20260913/time-estimate.json`; the status file records actual progress.

The original 2,000-episode campaign was stopped, retaining its 18 completed results. This campaign reused the 10 results for Spatial task 0, initial states 0–9, and ran another 390 fresh episodes. Reuse eligibility was determined in advance from the initial-state list. A snapshot and provenance were saved after confirming the controller fingerprint and evaluation-affecting configuration matched; records were not selected by success or failure. Videos for reused records remain in the original campaign directory and must be retained with them.

```bash
cd /root/route-b-upload/route-b-v170-cloud

# Inspect progress, active processes, and success rates by suite.
cat runtime/jobs/smolvla_scale_400_parallel_20260913/status.json

# Follow the dispatcher log.
tail -f runtime/jobs/smolvla_scale_400_parallel_20260913/runner.log
```

Main files:

- Overall progress and task summaries: `outputs/smolvla_scale_400_parallel_20260913/summary.json`
- Fixed 400-episode manifest and effective configuration: `outputs/smolvla_scale_400_parallel_20260913/manifest.json`
- Fresh episode records and videos: `outputs/smolvla_scale_400_parallel_20260913/shards/<suite>_task<id>/`
- Reused records and provenance: `outputs/smolvla_scale_400_parallel_20260913/reused/spatial_task00/`
- Per-process logs: `runtime/jobs/smolvla_scale_400_parallel_20260913/logs/`
- Scheduling and throughput history: `runtime/jobs/smolvla_scale_400_parallel_20260913/progress.jsonl`

On completion, the dispatcher automatically generates the merged `outputs/smolvla_scale_400_parallel_20260913/episodes.jsonl` and `runtime/jobs/smolvla_scale_400_parallel_20260913/final-validation.json`. Validation checks complete coverage of all 400 initializations, absence of duplicates, configuration consistency, video existence, agreement of summary counts, and deployed source checksums. The expected state is `completed`; a state of `completed_with_exceptions`, `failed`, or `runner_error` requires reviewing the logs and validation report. Failed episodes remain in the statistics and are not automatically retried to select better outcomes.

If the job is externally interrupted, confirm that the old manager and evaluation subprocesses have stopped before resuming:

```bash
cd /root/route-b-upload/route-b-v170-cloud
nohup .venv/bin/python runtime/jobs/smolvla_scale_400_parallel_20260913/runner.py --resume \
  > runtime/jobs/smolvla_scale_400_parallel_20260913/runner-resume.log 2>&1 < /dev/null &
```

Resuming preserves completed episodes and continues with unfinished initial states. The manager uses a file lock to prevent duplicate launches of the same dispatcher. The simple remaining-time estimate in the status file extrapolates from mean throughput over completed fresh episodes. Initial loading, changes in concurrency, and differences in task difficulty all affect this estimate, so it is not a guarantee.
