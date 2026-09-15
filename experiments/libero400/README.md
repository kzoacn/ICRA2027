# Local frozen-controller evaluation

The complete local run `full400_local_20260914T153307Z` records **376/400 (94%)** with one frozen controller. All 400 episodes are freshly executed.

Spatial: **93/100** / Object: **98/100** / Goal: **98/100** / Long: **87/100**.

The dispatcher wall time is **62.7 minutes**. Coverage, source/configuration consistency, record integrity, video presence, and runtime-exception checks passed. See [the acceptance record](accepted_result.json).

The requested run evaluates all 40 tasks on official initial-state indices 0-9,
with seed 7 and Spatial/Object/Goal/Long action budgets 220/280/300/520.
Every episode is newly executed with the controller source
`7bdcb97a7b847debeb0bd47c951455934701f81d7b561ad097fe5b41269de256`.
The evaluator uses the existing external ever-success rule and policy-controlled
stopping. Failures remain in the results.

The local host is Ubuntu 24.04 under WSL2 with an RTX 3090. Python is 3.12.14,
PyTorch is 2.11.0+cu130, and the core simulator and runtime versions match the
pinned project requirements. The run started with four workers, then increased
to six after memory and GPU usage were checked. This changes scheduling only.

- [Environment](environment.json)
- [Concurrency history](concurrency.json)
- [Simulator installation hash checks](simulator-installation-audit.json)
- [Local detector parameter audit](model_audit.json)
- [Launch record](run.json)
- [Recorded completion status](../../runtime/route_b_90/full400_local_20260914T153307Z/status.json)
- [Frozen source and task manifest](../../runtime/route_b_90/full400_local_20260914T153307Z/manifest.json)

Before the full run, the rendering check validated both 256-by-256 camera views
and all 40 instruction compilations. All 585 asset files and eight detector
files passed the pinned resource hashes. A separate one-episode execution check
completed Object 00 / initial state 00 successfully; that record is not reused
in this full run.

To reuse this host's isolated deployment with the current repository source:

    python3 scripts/run_batch.py --label local400_reproduction --all --workers 6 --deployed-root /home/kzoacn/.local/share/anchor-icra2027

The original Conda environment was read and verified; the OpenCV version needed
by this project was installed in the separate deployment environment. Model
weights and assets were copied to the deployment's own resource directory.

To reproduce the validated import, write it into a staging directory:

    python3 paper/scripts/capture_results.py --batch-dir runtime/route_b_90/full400_local_20260914T153307Z --environment experiments/libero400/environment.json --output-dir runtime/local_evaluation/paper-stage/data

The rollout generator chooses the first externally successful Object 00 episode
and the first externally unsuccessful Goal 03 episode when available. If a
preferred task has no matching outcome, it uses sorted task/initial-state order.
This rule selects illustrations only; all 400 records determine the reported score.

The exact evaluated source is preserved in [full400-source.zip](../../archive/evaluated-source/full400-source.zip).
Current code lives in `src/anchor/`; see the [name and path mapping](../../docs/repository.md).
Frozen JSON records retain their original paths and source hashes.
