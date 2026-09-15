# Long 09 targeted improvement

Objective: at least 7/10 external successes for "put the yellow and white mug in the microwave and close it".

The accepted result is **8/10**, up from **0/10** in the original full evaluation.
`long09_final_02` completed all ten initializations with one frozen controller
source, `da1e0b8603977522e81cd9b4fd51837c10345a11566f89744018bb689230618a`.
Coverage, configuration, source, record-integrity, video-presence, and runtime-exception
checks passed. The complete targeted batch took 771.5 seconds.

- Successful initial states: 0, 1, 2, 4, 5, 6, 7, 8.
- Initial state 3 failed during grasp approach; initial state 9 failed during cavity-entry staging.
- Initial state 7 reached external success but later failed the internal panel-angle check.
  Its external label follows the unchanged ever-success rule; the internal failure is preserved.
- [Acceptance and per-initial-state results](accepted_result.json)
- [Exact accepted source-file hashes](accepted_source_files.json)

The user requested a single-task retest, with no new full-suite campaign. Every accepted result must cover all ten official initial states, indices 0–9, with seed 7, the original 520-action budget, and the unchanged external ever-success evaluator. Development attempts remain separate from the final ten-episode retest.

The original full-400 result remains 360/400, including Long 09 at 0/10. An updated overall statistic combines its other 390 historical episodes with ten freshly evaluated Long 09 episodes and must be labeled as results from two controller versions. It is not a new full-400 evaluation.

The combined record count is **368/400 (92.0%)**: Spatial 93/100, Object 98/100,
Goal 91/100, and Long 86/100. The other 39 tasks were not rerun. The original
400-episode record set is preserved in [full400-reference/](../full400-reference/).
The targeted initializations also informed development, so this follow-up is not held out.

Development uses immutable snapshots under `runtime/route_b_90/long09_*`, with complete raw records, videos, manifests, source hashes, and validation reports. The close skill fits the visible door panel in a measured appliance hinge frame and uses a compact top-down face pusher. Mug release requires a measured 78-mm jaw opening and a settling dwell before withdrawing from the cavity. Its inputs are calibrated RGB-D and robot measurements; it does not receive task predicates or simulator object/door joint states.

Run `python3 experiments/long09/update_record.py` to refresh the complete development index. Each candidate's outcomes remain separate; unsuccessful attempts are retained. The first complete targeted retest, `long09_final_01`, achieved 6/10 before the mug-release repair.

The original full evaluation remains preserved separately from task retests.
The later 374/400 combined record set, incorporating both this microwave retest
and the drawer retest, is retained in
[task-updates-reference/](../task-updates-reference/) with its original records
and provenance.
