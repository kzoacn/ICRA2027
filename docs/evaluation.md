# LIBERO evaluation

The current complete local evaluation records **376/400 (94.0%)**. Its source,
environment, launch and acceptance records are in
[experiments/libero400/](../experiments/libero400/README.md).

## Protocol

| Setting | Value |
| --- | --- |
| Suites | Spatial, Object, Goal, Long; ten tasks each |
| Initial states | Official indices 0–9 for every task |
| Seed | 7 |
| Action caps | Spatial 220, Object 280, Goal 300, Long 520 |
| Observations | Two calibrated 256 × 256 RGB-D views and robot proprioception |
| Score | Official success predicate reached at any point before stopping |
| Stopping | Controller status or the suite action cap |

The official predicate is checked externally and is not supplied to the controller.
All unsuccessful episodes remain in the aggregate.

## Execute and validate

After [setup](setup.md), run from the repository root:

```bash
python3 scripts/run_batch.py --all --label libero400 --workers 4
```

The runner copies `src/anchor/` into an immutable per-run snapshot and launches
one process per task. Initial states remain in order within each task. Records,
configuration, source manifests and videos are retained in
`runtime/evaluations/libero400/`. The runner merges the task records into
`episodes.jsonl` and writes `final-validation.json`, checking coverage,
configuration, source consistency and runtime failures. Inspect this validation
and `status.json` before treating a run as complete. Retain an environment record
measured for that run and source snapshot alongside the results.

## Records

- [Current measured result](../experiments/libero400/README.md)
- [Published protocol review](evaluation-protocol.md)
- [Source and data conventions](repository.md)
- [Earlier development records](../archive/experiments/)

The current score is computed from the 400 new episodes in
[experiments/libero400/episodes.jsonl](../experiments/libero400/episodes.jsonl).
[Compressed original records](../experiments/libero400/raw_episodes.jsonl.gz),
[provenance](../experiments/libero400/provenance.json),
[statistics](../experiments/libero400/statistics.json) and
[per-task results](../experiments/libero400/per_task.csv) accompany that projection.
Earlier data are retained in
[full400-reference/](../archive/experiments/full400-reference/) and
[task-updates-reference/](../archive/experiments/task-updates-reference/).
