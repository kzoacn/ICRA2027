# Original 2,000-episode campaign

Run name: `full_2000_20260913`

This campaign was stopped at the user's request on 2026-09-13, retaining 18 completed
episodes. Work then switched to the [400-episode parallel evaluation](./RUN_400_PARALLEL.md),
which reused 10 results from this campaign matching official initial-state indices 0–9.
The following instructions document the original run.

It was launched in the background at 2026-09-13 14:45:45 UTC. The status file and
running processes were the authoritative sources for progress.

The planned evaluation covered Spatial, Object, Goal, and Long: 10 tasks per suite,
official initial-state indices 0–49 per task, and 2,000 episodes in total. It used an
RTX 5090, seed=7, and two 256×256 cameras, with video recording enabled. The suite
action budgets were 220, 280, 300, and 520 steps, respectively. Execution was sequential,
using the same controller source as the deployment validation.

```bash
cd /root/route-b-upload/route-b-v170-cloud

# Inspect run status, updated every 15 seconds.
cat runtime/jobs/full_2000_20260913/status.json

# Follow episode completion records.
tail -f logs/full_2000_20260913.log

# Check whether the evaluation process is still running.
ps -p "$(cat runtime/jobs/full_2000_20260913/campaign.pid)" -o pid,etime,pcpu,pmem,args
```

Run directories:

- `outputs/full_2000_20260913_spatial/`
- `outputs/full_2000_20260913_object/`
- `outputs/full_2000_20260913_goal/`
- `outputs/full_2000_20260913_long/`

Each directory incrementally stores `episodes.jsonl`, `summary.json`, and `videos/`.
On completion, the campaign generates `outputs/full_2000_20260913_campaign.json`.
The background manager also writes `runtime/jobs/full_2000_20260913/final-validation.json`,
checking initial-state coverage, duplicate records, configuration, summary counts,
video files, and source checksums, and counting exceptions.

The status field `success_rate_so_far` is the interim success rate over completed
samples. Final results require a `completed` state and inspection of the full summary.
A `failed` or `completed_with_exceptions` state requires reviewing the logs and validation report.

The job was detached from the launching terminal. To resume an interrupted evaluation,
first confirm that the original evaluation process has stopped, then use the same configuration:

```bash
cd /root/route-b-upload/route-b-v170-cloud
nohup .venv/bin/python runtime/jobs/full_2000_20260913/runner.py --resume \
  > runtime/jobs/full_2000_20260913/runner-resume.log 2>&1 < /dev/null &
```

Resuming preserves completed records and skips initial states already evaluated.
The background manager uses a file lock to prevent duplicate launches. After a process
exits or the host restarts, the status file may reflect the last state before exit;
inspect the processes as well to determine the actual state.
