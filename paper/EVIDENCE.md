# Manuscript evidence map

This file is an author-facing audit aid; it is not supplementary material included in the submission PDF.
Implementation paths below are relative to ../route-b-v170-cloud.
Paths beginning with paper/, experiments/, or runtime/ are relative to the repository root.
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
| Contact evidence uses robot pose/progress/force | libero_system/goal_skills/controller.py: _contact_reached, _move | Operational guards; no formal contact safety or correctness proof |
| Official initial-state indices and reset-stream handling | libero_system/common/env_adapter.py: reset | Seed 7; reset stream is restored/replayed, not equivalent to every benchmark implementation |
| External ever-success and controller stopping | libero_system/integration/evaluator.py: _ExternalStickyScore, EvaluationDriver._run_b | Success can precede a later internal failure; final-only outcome is not separately recorded |
| Software-boundary counters | _ExternalStickyScore.audit | Several zero counters are constructed by implementation; not independent process attestation |
| 400 episodes and declared record provenance | paper/data/provenance.json; paper/scripts/capture_results.py | Exact reuse/fresh counts, official IDs, seeds, budgets, source hashes and validation are preserved in the dataset provenance |
| Success, runtime and status tables | paper/data/episodes.jsonl; paper/data/raw_episodes.jsonl.gz; paper/scripts/analyze_results.py | 368/400 combined records from two controller versions: 390 historical episodes plus ten new Long 09 episodes (8/10); projected fields and original lines are checked against each other |
| Targeted microwave release and closure | libero_system/route_b/microwave_close.py; microwave_interior.py: observed_door_panel; controller.py: _shelf_release_command, _act_place; experiments/long09/ | RGB-D door-panel fitting, exterior face pushing, and measured jaw clearance; evaluated only on Long 09 |
| Inspectable execution and targeted revision without changing neural weights | experiments/long09/README.md; experiments/long09/development_record.json; experiments/long09/accepted_result.json; experiments/long09/accepted_source_files.json; paper/data/model_audit.json | Long 09 improves from 0/10 to 8/10 with the combined release/close revision; this is a development case, not a measured reduction in debugging effort relative to a VLA |
| Whole-task replacement and historical preservation | paper/scripts/update_task_results.py; paper/data/full400_reference/; paper/scripts/check_paper.py | Complete initial-state coverage, unchanged scoring, per-record source hashes, ten replaced and 390 retained episodes; not a new full-suite evaluation |
| Timing environments | experiments/long09/environment.json; paper/data/provenance.json | Original full run on RTX 5090 and targeted retest on RTX 4090; combined runtime statistics are descriptive |
| Published policy comparison | paper/data/published_comparisons.json; paper/scripts/build_comparison_table.py | Five externally reported mean rows with source versions, table numbers and PDF hashes; the ANCHOR row is computed from its own episode records |
| Failure groups | failure_group in paper/scripts/analyze_results.py | Message-based terminal categories, not independently verified physical causes |
| Per-task examples in text | paper/generated/numbers.tex: TopDrawerSuccess, DrawerPickSuccess, BottomDrawerSuccess, MokaSuccess, MicrowaveSuccess | Each macro is independently checked against the frozen episode projection by paper/scripts/check_paper.py |
| First-page overview | paper/figures/architecture.tex; paper/figures/recorded/; paper/data/figure_provenance.json | Editable vector data-flow and carried-object schematic with actual Object 00 frames 0 and 45; schematic geometry is not a measured reconstruction |
| Qualitative rollouts | paper/data/figure_provenance.json | Selected recorded episodes, video hashes and exact frame indices; fixed-camera crop is flipped vertically for upright display, preserving left/right; exported image hashes are recorded |
| Historical 85.9% over 2000 episodes | provenance/source.json only | Excluded from current measured results because raw historical records are unavailable in the uploaded package |

## Primary references checked

| BibTeX key | Primary source | Use in manuscript |
|---|---|---|
| libero | [LIBERO paper](https://arxiv.org/abs/2306.03310) | Benchmark origin and research setting |
| openvla | [OpenVLA v3, Appendix E, Table 12](https://arxiv.org/html/2406.09246v3) | Learned action-policy context; the authors' measured Diffusion Policy, Octo and OpenVLA results |
| smolvla | [SmolVLA v1, Sections 4.1/4.3 and Table 2](https://arxiv.org/html/2506.01844v1) | Compact VLA context, 10 trials per task, and the 0.45B simulation result |
| openvlaoft | [OpenVLA-OFT v2, Section V and Table I](https://arxiv.org/html/2502.19645v2) | Full OpenVLA-OFT result with extra inputs and filtered training demonstrations |
| tamp | [Integrated Task and Motion Planning](https://arxiv.org/abs/2010.01083) | Discrete/continuous planning context |
| saycan | [SayCan paper](https://arxiv.org/abs/2204.01691) | Language and skill affordances |
| codepolicies | [Code as Policies paper](https://arxiv.org/abs/2209.07753) | Language-generated policy programs |
| voxposer | [VoxPoser paper](https://arxiv.org/abs/2307.05973) | Spatial value-map representation |
| rekep | [ReKep paper](https://arxiv.org/abs/2409.01652) | Relational keypoint constraints |
| groundingdino | [ECCV 2024 paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06319.pdf) | Frozen open-set region detector |
| robosuite | [robosuite paper](https://arxiv.org/abs/2009.12293) | Simulation framework, v3 bibliographic metadata |
| mujoco | [MuJoCo paper](https://homes.cs.washington.edu/~todorov/papers/TodorovIROS12.pdf) | Physics engine |
| openvlacode | [Evaluation loop](https://github.com/openvla/openvla/blob/main/experiments/robot/libero/run_libero_eval.py), [environment helper](https://github.com/openvla/openvla/blob/main/experiments/robot/libero/libero_utils.py) | Numerical horizons and distinct environment seed |

The comparison table transcribes published numerical means; these policies were not rerun in this task.
Source-reported averages are preserved, and available standard errors remain in the source-data JSON.
Protocol differences are described next to the comparison and in its caption.
The reference descriptions summarize the cited primary sources; figures and prose from those works are not copied.
