# Manuscript evidence map

This file is an author-facing audit aid; it is not supplementary material included in the submission PDF.
Implementation paths below are relative to ../route-b-v170-cloud.
Paths beginning with paper/, experiments/, or runtime/ are relative to the repository root.
The current manuscript reports the local full evaluation: 376/400 (94%) with one frozen controller. Historical development rows below are identified as history and do not supply current episode results.
The manuscript is a system study of the existing implementation, not a claim of a new general learning algorithm.

| Claim or reported quantity | Evidence | Scope |
|---|---|---|
| Observation contains two RGB-D cameras, calibration and robot measurements | libero_system/common/observation.py: RobotObservation, CameraFrame, Proprioception; common/env_adapter.py: _sanitize_observation | Simulated, calibrated data, including force/torque; not RGB-only sensing |
| Grasp/place policy uses a narrower sensor projection | libero_system/integration/adapters.py: to_route_b_observation | Pose, joint position and gripper width; broader contact observation remains available |
| Deterministic language grammar and capability check | libero_system/route_b/task_compiler.py: TaskCompiler, route_b_execution_issue | Known language forms and allow-listed entities; no language-model resolver configured |
| Frozen detector with offline asset gallery | libero_system/integration/components.py: build_perception_bundle_from_context; perception/grounding_dino.py | The detector is pretrained; the action controller is manually engineered |
| 172,249,090 unique neural parameters; FP32 weights occupy 689,359,096 bytes | paper/data/model_audit.json; paper/scripts/audit_model.py; resources.lock.json | All vision and text parameters in the actual deployment detector are counted once; every checkpoint file is verified against the resource lock |
| No robot-demonstration action-policy training and one shared detector | libero_system/integration/components.py: PerceptionBundle, build_perception_bundle_from_context; route_b/task_compiler.py: TaskCompiler; route_b/controller.py; goal_skills/controller.py | Geometric action generation and manual skills; pretrained perception and asset knowledge are retained, and worker processes can each load a detector |
| VLA model-size and policy-training comparison | paper/data/published_comparisons.json: sources.*.model_size; paper/scripts/build_comparison_table.py; paper/generated/model_table.tex | SmolVLA 0.45B and the reported 7B OpenVLA/OpenVLA-OFT class; source-rounded sizes, not fresh checkpoint counts or measured speed/memory ratios |
| Asset-derived texture and collision-size priors | libero_system/perception/gallery.py: TextureSizeGallery, collision_aabb_dimensions | Public asset knowledge is explicitly part of the system |
| Visible burner and microwave physical geometry in the improved controller | libero_system/route_b/stove_support.py; microwave_interior.py; integration/adapters.py: RouteBMicrowaveDoorDetector; experiments/route_b_90/check_geometry.py | Calibrated visible surfaces plus declared physical dimensions; no scene object pose or task-region coordinates |
| Basket slots and open-palm withdrawal in the improved controller | libero_system/route_b/controller.py: _shared_support_slot_offset, _act_place; release_geometry.py: open_jaw_peel_pose | Bounded hand-written geometric proposals, evaluated as part of the complete controller |
| Gallery weights and temperature | TextureSizeGallery.__init__, classify_features | Color 0.4, shape 0.6, temperature 0.13; additional class-dependent gates |
| Weighted multi-view fusion | libero_system/perception/fusion.py: fuse_instances_3d, _merge_members | Default center threshold 0.045 m; confidence times square-root point count |
| Clipped Cartesian servo | libero_system/route_b/controller.py: _cartesian_action, ControllerConfig | Defaults 0.05 m and 0.25 rad; some branches use specialized scales |
| Carried-object displacement affects placement | Same file: _set_place_targets, _refresh_rim_held_offset_for_place | Equation in paper summarizes a geometric relation with additional branch-specific corrections |
| Sequential manipulation/contact dispatch | libero_system/integration/adapters.py: RouteBPolicy | Includes specialized hand-written logic; no global task-motion optimization is claimed |
| Geometric state across skills | route_b/controller.py: _held_offset_world, _set_place_targets; integration/adapters.py: _drawer_anchors, _prepare_manipulation_segment; route_b/stove_support.py | Table II summarizes accepted scene estimates, carried offsets, fixture anchors and phase state; it does not introduce an additional learned representation |
| Evaluated grasp modes | route_b/controller.py: EntityGraspModeSelector; integration/evaluator.py: _make_b_scene_model; paper/data/raw_episodes.jsonl.gz | Default configuration selects pinch or closing rim pinch. All 459 logged grasp-target attempts use these modes (286 pinch, 173 rim_pinch); interior expansion is not configured in this run |
| Contact evidence uses robot pose/progress/force | libero_system/goal_skills/controller.py: _contact_reached, _move | Operational guards; no formal contact safety or correctness proof |
| Official initial-state indices and reset-stream handling | libero_system/common/env_adapter.py: reset | Seed 7; reset stream is restored/replayed, not equivalent to every benchmark implementation |
| External ever-success and controller stopping | libero_system/integration/evaluator.py: _ExternalStickyScore, EvaluationDriver._run_b | Success can precede a later internal failure; final-only outcome is not separately recorded |
| Software-boundary counters | _ExternalStickyScore.audit | Several zero counters are constructed by implementation; not independent process attestation |
| 400 episodes and declared record provenance | paper/data/provenance.json; paper/scripts/capture_results.py | Exact reuse/fresh counts, official IDs, seeds, budgets, source hashes and validation are preserved in the dataset provenance |
| Success, runtime and status tables | paper/data/episodes.jsonl; paper/data/raw_episodes.jsonl.gz; paper/scripts/analyze_results.py | 376/400 from one frozen controller in the local full evaluation; all 400 episodes are fresh, and projections match original record hashes |
| Microwave release and closure | libero_system/route_b/microwave_close.py; microwave_interior.py: observed_door_panel; controller.py: _shelf_release_command, _act_place | Current full-run controller uses RGB-D panel fitting, hinge-frame motion and measured jaw clearance; no object/door joint-state input |
| Historical: Inspectable execution and targeted revision without changing neural weights | experiments/long09/README.md; experiments/long09/development_record.json; experiments/long09/accepted_result.json; experiments/long09/accepted_source_files.json; paper/data/model_audit.json | Long 09 improves from 0/10 to 8/10 with the combined release/close revision; this is a development case, not a measured reduction in debugging effort relative to a VLA |
| Post-handle wrist motion and drawer-front bowl clearance | libero_system/route_b/drawer_transfer.py; controller.py: _motion, _set_pick_targets, _set_place_targets | Public-model IK and observed drawer/bowl geometry in the current full-run controller; a local reachability heuristic without collision checking |
| Historical: Whole-task replacement and historical preservation | paper/scripts/update_task_results.py; paper/data/full400_reference/; paper/scripts/check_paper.py | Complete initial-state coverage, unchanged scoring, per-record source hashes, twenty replaced and 380 retained episodes; not a new full-suite evaluation |
| Timing environment | experiments/local400/environment.json; experiments/local400/concurrency.json; paper/data/environment.json | One RTX 3090 under WSL2; four then six workers. Timings describe this local campaign and include execution overhead |
| Published policy comparison | paper/data/published_comparisons.json; paper/scripts/build_comparison_table.py | Five cited literature rows and one ANCHOR row measured in the local full evaluation; protocols differ |
| Full-run source and coverage | paper/data/provenance.json; experiments/local400/accepted_result.json; paper/scripts/check_paper.py | 400 unique task/initial-state pairs from one immutable controller source, with per-record campaign and controller hashes |
| Historical: Revision-case outcome details | experiments/long09/accepted_result.json; experiments/long09/README.md; experiments/goal03/accepted_result.json | One externally successful Long 09 episode later fails its internal panel-angle check; the regressed Goal 03 initial state 00 terminates at a pregrasp timeout; these are existing records, not new experiments |
| Failure groups | failure_group in paper/scripts/analyze_results.py | Message-based terminal categories, not independently verified physical causes |
| Conditional completion and stopping analysis | paper/scripts/analyze_results.py; paper/data/statistics.json: stopping_analysis; paper/scripts/check_paper.py | 341/347 local completions have external ever-success; all 24 external failures stop before the global cap, with at least 13 unused actions; all 11 global timeouts have external success. No changed-budget experiment or causal claim follows from these summaries |
| Failure concentration | paper/data/episodes.jsonl; paper/data/statistics.json: stopping_analysis | Long accounts for 13/24 external failures (54.2%) and 100/400 episodes. This is descriptive suite-level concentration, not an isolated effect of task length |
| Per-task examples in text | paper/generated/numbers.tex: TopDrawerSuccess, DrawerPickSuccess, BottomDrawerSuccess, MokaSuccess, MicrowaveSuccess | Each macro is independently checked against the frozen episode projection by paper/scripts/check_paper.py |
| First-page overview | paper/figures/architecture.tex; paper/generated/rollout_metadata.tex; paper/data/figure_provenance.json | Editable geometry schematic with camera insets selected from the current local Object 00 rollout |
| Qualitative rollouts | paper/data/figure_provenance.json; paper/generated/rollout_metadata.tex | Current local full-run Object 00 / init 00 success (136 actions) and Goal 03 / init 00 phase-timeout failure (272 actions); original video, record and frame hashes are preserved |
| Frame-to-phase alignment and failure timeline | integration/evaluator.py: _run_b; integration/video.py: DualViewVideoRecorder.add; paper/data/figure_provenance.json | Reset is frame index 0; stride is 2 actions. Phase labels use the last transition at or before frame index times stride. Goal 03 / init 00 enters move_pregrasp at 111, fails at 272, and leaves 28 of 300 actions unused; the 161-action interval is read from the trace |
| Historical 85.9% over 2000 episodes | provenance/source.json only | Excluded from current measured results because raw historical records are unavailable in the uploaded package |

## Primary references checked

See [CITATION_AUDIT.md](../../docs/paper/citation-audit.md) for the 2026-09-14 bibliography,
claim, source-table, and pinned-code checks, including publication-year and
author-order differences between source records.

| BibTeX key | Primary source | Use in manuscript |
|---|---|---|
| libero | [LIBERO, NeurIPS 2023](https://proceedings.nips.cc/paper_files/paper/2023/hash/8c3c666820ea055a77726d66fc7d447f-Abstract-Datasets_and_Benchmarks.html) | Benchmark origin and research setting; published metadata |
| openvla | [OpenVLA v3, Appendix E, Table 12](https://arxiv.org/html/2406.09246v3) | Learned action-policy context; the authors' measured Diffusion Policy, Octo and OpenVLA results |
| smolvla | [SmolVLA v1, Sections 4.1/4.3 and Table 2](https://arxiv.org/html/2506.01844v1) | Compact VLA context, 10 trials per task, and the 0.45B simulation result |
| openvlaoft | [OpenVLA-OFT v2, Section V and Tables I/IV](https://arxiv.org/html/2502.19645v2); [RSS publication](https://www.roboticsproceedings.org/rss21/p017.html) | Full OpenVLA-OFT result with extra inputs and filtered training demonstrations; numerical source remains v2 |
| tamp | [Integrated Task and Motion Planning, Annual Reviews](https://www.annualreviews.org/content/journals/10.1146/annurev-control-091420-084139) | Discrete/continuous planning context; published in 2021 |
| saycan | [SayCan published PDF](https://proceedings.mlr.press/v205/ichter23a/ichter23a.pdf) | Language and skill affordances; author order follows the PDF |
| codepolicies | [Code as Policies, ICRA 2023](https://doi.org/10.1109/ICRA48891.2023.10160591); [authors' v4](https://arxiv.org/abs/2209.07753v4) | Language-generated policy programs and published metadata |
| voxposer | [VoxPoser, CoRL 2023 proceedings](https://proceedings.mlr.press/v229/huang23b.html) | Spatial value-map representation; formal publication metadata |
| kpam | [kPAM published chapter](https://link.springer.com/chapter/10.1007/978-3-030-95459-8_9); [authors' v2](https://arxiv.org/abs/1903.06684v2) | Semantic keypoints and geometric goal specification; ISRR 2019 proceedings published in 2022 |
| rekep | [ReKep, PMLR 270](https://proceedings.mlr.press/v270/huang25g.html) | Relational keypoint constraints; CoRL 2024 proceedings published in 2025 |
| groundingdino | [ECCV PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06319.pdf); [Springer record](https://link.springer.com/chapter/10.1007/978-3-031-72970-6_3) | Frozen open-set region detector; Springer citation year 2025 |
| robosuite | [robosuite v3](https://arxiv.org/abs/2009.12293v3) | Framework attribution and v3 metadata; deployment version is documented separately |
| mujoco | [MuJoCo author-hosted paper](https://www.roboti.us/lab/papers/TodorovIROS12.pdf); [IEEE record](https://doi.org/10.1109/IROS.2012.6386109) | Physics-engine attribution |
| openvlacode | [Evaluation loop](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/experiments/robot/libero/run_libero_eval.py), [environment helper](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/experiments/robot/libero/libero_utils.py) | Pinned numerical horizons, environment seed, and early stopping |

The comparison table transcribes published numerical means; these policies were not rerun in this task.
Source-reported averages are preserved, and available standard errors remain in the source-data JSON.
Protocol differences are described next to the comparison and in its caption.
The reference descriptions summarize the cited primary sources; figures and prose from those works are not copied.
