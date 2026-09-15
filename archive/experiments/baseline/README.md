# Route B improvement campaign

Objective: at least 360 external successes in a fresh, fixed-protocol 400-episode run.
Baseline: commit fa398a2, controller source SHA256
67c93be1df7da7572ecbb8faff53d01a47aea42ec36affb07ac54ba36e59edba,
347/400 successes. The baseline deployment and its original records are preserved.

The protocol remains 40 tasks, initial-state indices 0–9, seed 7,
Spatial/Object/Goal/Long budgets 220/280/300/520, two calibrated RGB-D views,
and external ever-success scoring.

Run a candidate batch from the repository root:

    python3 experiments/route_b_90/run_batch.py --label candidate_01 --tasks goal:3 spatial:4 long:3 --workers 3

The runner freezes the current source in an ignored runtime directory before
launching workers. Each batch retains the exact source, manifest, logs, videos,
original episode records and validation result. Modifying the working copy
during a batch cannot affect the already frozen candidate.

Final acceptance uses --all --workers 8, with all 400 episodes freshly evaluated
against one source snapshot. Only validated measured outcomes are accepted.

## Diagnostic observations

- Goal 03: post-drawer movement to the bowl's pregrasp lasts 32–161 steps.
  Several failures retain substantial wrist tilt or lateral error; this movement
  consumes much of the 300-action budget.
- Spatial 04: unsuccessful cavity grasps and extra viewpoint/retry stages consume
  the 220-action budget.
- Long 03: preplacement and descent stalls; one preplacement residual is about
  9 mm, while other failures show larger contact-related displacement.
- Long 08: placement-verification failures and controller completion without
  external success require checking the actual stove support and object positions.
- Long 09: microwave localization, placement and door-approach failures.

These are observations and hypotheses; candidate results determine which changes
are retained.

## Development checkpoint

The accepted complete replay is **360/400 (90.0%)**, documented below.
The earlier development results in this section came from different immutable
candidates; acceptance uses the separate fresh 400-episode replay.

- cavity_near_02: Spatial 04 improved from 5/10 to 7/10. Wider measured
  near-side clearance allows two extra direct nominal rim grasps.
- drawer_contact_01: Goal 00 improved from 7/10 to 10/10. A single equivalent
  wrist recovery also handles low-force translation plateaus.
- Goal 01 remained 10/10 with the new visible-burner localizer.
- stove_disk_01 and stove_disk_02: Long 08 improved from 4/10 to 7/10.
  The complete visible burner disk supplies the support centre and height;
  the second version caches that unoccluded geometry across placements.
- drawer_placement_02: Long 03 improved from 6/10 to 7/10, while Goal 03
  fell from 3/10 to 2/10. The subsequent candidate confines carry-frame and
  convergence changes to the bottom drawer. The complete replay confirms
  Long 03 at 7/10 and preserves Goal 03 at 3/10.
- Post-drawer detours, initial wrist-frame changes, and direct active-view
  near approaches have not demonstrated a retained improvement. Their records
  are preserved; the current candidate restores the baseline free-space pick.
- Microwave insertion now reaches door manipulation in some development
  episodes, but no microwave success has yet been confirmed. Door feature
  association, wrist staging, and retained contact still need improvement.
- Basket placement checks identified colliding sequential placements and
  packages carried away after opening the gripper. The complete replay
  confirms Object 07 and Long 00 at 10/10; Long 01 and Long 07 retain 9/10
  and 10/10 respectively.

The microwave development fixture contains two raw boundary RGB-D surface
clouds from the first observation of Long 09, init 0: a flat table false positive
and the actual appliance. It contains no simulator object poses or evaluator
state. The localizer requires vertical control-panel/side-wall structure and a
measured roof height before applying physical wall dimensions from the public
asset. The captured partially occluded roof is also tested under rigid scene
transforms. The burner tests reject partial disks rather than recentering on
an occluded visible fragment.

Run check_geometry.py with the deployed Python environment. Run update_record.py
to refresh development_record.json from preserved batch outputs. This index
includes unsuccessful and interrupted attempts and never merges candidate
scores into an acceptance claim.

## Accepted complete replay

`full400_candidate_01` completed 400 new episodes with one frozen policy source:
`c954f817fb7dd2b046a961a13a9acc9f7f2f0ad87da97eba1f6c0da995eff4e6`.
Its policy files match commit `5705b19`; later documentation or runner edits do
not enter the snapshot. All 400 episodes are new, with the protocol above.
The dispatcher started with two workers while development jobs finished,
then increased to eight. Priority affects task scheduling only; every task
retains initial-state order 0–9 and its independent reset stream.

The outcome is 360/400: Spatial 93/100, Object 98/100, Goal 91/100,
and Long 78/100. Thirteen previously unsuccessful initializations succeed;
no previously successful initialization regresses. All coverage, source,
configuration, video-presence and runtime-exception checks passed.
See [accepted_result.json](accepted_result.json) for the exact comparison.

| Task | Baseline | Complete replay |
|---|---:|---:|
| Spatial 04: bowl from top drawer | 5/10 | 7/10 |
| Object 07: milk into basket | 8/10 | 10/10 |
| Goal 00: open middle drawer | 7/10 | 10/10 |
| Long 00: soup and tomato sauce into basket | 8/10 | 10/10 |
| Long 03: bowl into bottom drawer and close | 6/10 | 7/10 |
| Long 08: both moka pots onto stove | 4/10 | 7/10 |

Other task outcomes are unchanged. Microwave Long 09 is 0/10 in this complete replay.
The subsequent [Long 09 follow-up](../microwave/README.md) updates that task only;
its new records are distinct from the historical 390.
This original full evaluation is preserved in
[full400-reference/](../full400-reference/), including the episode projection,
compressed original records and provenance. The later combined task-retest
records are preserved in [task-updates-reference/](../task-updates-reference/).
Development snapshots, logs and videos retain their original server locations.
