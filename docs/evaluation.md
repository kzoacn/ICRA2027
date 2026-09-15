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

## Execute and import

After [setup](setup.md), run from the repository root:

```bash
python3 scripts/run_batch.py --all --label libero400 --workers 4
```

The runner copies `src/anchor/` into an immutable per-run snapshot and launches
one process per task. Initial states remain in order within each task. Records,
configuration, source manifests and videos are retained in
`runtime/evaluations/libero400/`. Incomplete coverage and runtime failures are
rejected by final import validation.

To generate paper data from a completed full run:

```bash
python3 paper/scripts/capture_results.py --batch-dir runtime/evaluations/libero400 --environment /path/to/measured-environment.json
make -C paper figures
make -C paper check
```

Supply an environment record measured for that run and source snapshot. Regenerate
rollout figures from the same batch when updating the manuscript; see the
[paper build guide](../paper/README.md). The importer supports both the current
source layout and archived full-run layouts.

## Records

- [Current measured result](../experiments/libero400/README.md)
- [Published protocol review](evaluation-protocol.md)
- [Source and data conventions](repository.md)
- [Earlier development records](../archive/experiments/)

The paper retains earlier data under `paper/data/full400_reference/` and
`paper/data/task_updates_reference/`. The current score is computed from the
400 new episodes in `paper/data/episodes.jsonl`.
