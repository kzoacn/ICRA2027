# Goal 03 targeted improvement

Objective: at least 7/10 external successes for "open the top drawer and put the bowl inside".
The original full campaign records 3/10 successes. Acceptance requires a fresh,
complete replay of official initial states 0–9 with one frozen controller,
seed 7, the original 300-action budget, and external ever-success scoring.

The original failures include three pregrasp timeouts and four episodes that
reach bowl placement but exhaust the total budget. Post-drawer motion to the
bowl takes 32–161 actions. A replay of initial state 4 records oscillating
Cartesian motion and encounters the Panda's wrist and elbow joint limits
before reaching the pregrasp pose. These samples contain only public robot
pose and joint measurements and the controller's geometric target.

Development records are retained under `runtime/route_b_90/goal03_*`.
Each batch saves its source snapshot, configuration, raw episode records,
video recordings, and validation. Run `python3 experiments/goal03/update_record.py`
to refresh the index of all attempts. Candidate outcomes are kept separately.
The targeted initial states inform development; the retest is not held out.

## Accepted result

`goal03_final_02` records **9/10** external successes, compared
with **3/10** in the original campaign. All ten official states were replayed
in order under one frozen source snapshot. The original budget and scoring
were retained. [The acceptance record](accepted_result.json) includes each
initial state's outcome and source-line hash.

The revision repairs 7 previous failures and regresses
1 previous success. Every final-run outcome is retained.
Median pregrasp-stage actions change from 113 to 43.5 across all ten episodes, including timed-out stages. The two related Long 03
regression episodes remain externally successful (2/2); they are not a full
task reevaluation and are not substituted into the combined dataset.

The historical combined dataset contains **374/400 (93.5%)** from 380 original records, the ten
Long 09 retest records, and the ten Goal 03 retest records. These span three
controller versions. The [combined dataset](../task-updates-reference/) and
[original full campaign](../full400-reference/) remain preserved separately.

| Initial state | Original success | Updated success | Updated actions |
| --- | --- | --- | --- |
| 00 | yes | no | 272 |
| 01 | yes | yes | 282 |
| 02 | no | yes | 272 |
| 03 | no | yes | 278 |
| 04 | no | yes | 295 |
| 05 | no | yes | 273 |
| 06 | no | yes | 277 |
| 07 | no | yes | 280 |
| 08 | yes | yes | 271 |
| 09 | no | yes | 286 |

## Controller revision

The frozen revision:

- Checks both equivalent jaw frames along a continuous public-model IK path,
  preserving the rim point, physical jaw line, and vertical grasp approach.
  An alternate needs at least 0.15 rad joint clearance and an improvement of
  more than 0.10 rad when the preferred path is feasible. This is a local
  reachability heuristic, not collision checking.
- Bounds post-handle pregrasp rotation to a normalized vector norm of 0.70.
  Translation is bounded to 0.30 during large turns and 0.75 after alignment.
  The original 3-mm pregrasp acceptance gate is retained.
- Uses the observed drawer floor, its outward normal, and the held bowl's
  footprint to move clear of the front panel before lifting. It preserves
  wrist orientation and height during that short outward move. Subsequent
  post-handle bowl transport uses a translation norm bound of 0.85.

These changes read calibrated RGB-D geometry and robot proprioception, with
public Panda kinematics. They do not read task IDs, initial-state indices,
simulator object state, rewards, or the evaluator's success predicate.
The detector checkpoint and neural parameter count are unchanged.

The unrestricted drawer carry limit regressed a bottom-drawer case that starts
with an already open drawer. The revision therefore applies that limit only
after the controller has opened a drawer. An additional high pregrasp traverse
did not resolve the observed approach collision and was discarded. Each
attempt, including interrupted candidates and related-task checks, is indexed.

Geometric invariants can be checked without the simulator:

    /path/to/venv/bin/python experiments/goal03/check_geometry.py

This checks payload clearance, unchanged grasp geometry, rejection of needless
clearance moves, and consistent decisions after yaw and translation changes.

To reproduce the complete targeted run from the repository root:

    python3 experiments/route_b_90/run_batch.py --label goal03_reproduction --tasks goal:3 --episodes 10 --workers 1
