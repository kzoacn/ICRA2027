# Manuscript evidence map

This file is an author-facing audit aid; it is not supplementary material included in the submission PDF.
Paths below are relative to ../route-b-v170-cloud unless stated otherwise.
The manuscript is a system study of the existing implementation, not a claim of a new general learning algorithm.

| Claim or reported quantity | Evidence | Scope |
|---|---|---|
| Observation contains two RGB-D cameras, calibration and robot measurements | libero_system/common/observation.py: RobotObservation, CameraFrame, Proprioception; common/env_adapter.py: _sanitize_observation | Simulated, calibrated data, including force/torque; not RGB-only sensing |
| Grasp/place policy uses a narrower sensor projection | libero_system/integration/adapters.py: to_route_b_observation | Pose, joint position and gripper width; broader contact observation remains available |
| Deterministic language grammar and capability check | libero_system/route_b/task_compiler.py: TaskCompiler, route_b_execution_issue | Known language forms and allow-listed entities; no language-model resolver configured |
| Frozen detector with offline asset gallery | libero_system/integration/components.py: build_perception_bundle_from_context; perception/grounding_dino.py | The detector is pretrained; the action controller is manually engineered |
| Asset-derived texture and collision-size priors | libero_system/perception/gallery.py: TextureSizeGallery, collision_aabb_dimensions | Public asset knowledge is explicitly part of the system |
| Gallery weights and temperature | TextureSizeGallery.__init__, classify_features | Color 0.4, shape 0.6, temperature 0.13; additional class-dependent gates |
| Weighted multi-view fusion | libero_system/perception/fusion.py: fuse_instances_3d, _merge_members | Default center threshold 0.045 m; confidence times square-root point count |
| Clipped Cartesian servo | libero_system/route_b/controller.py: _cartesian_action, ControllerConfig | Defaults 0.05 m and 0.25 rad; some branches use specialized scales |
| Carried-object displacement affects placement | Same file: _set_place_targets, _refresh_rim_held_offset_for_place | Equation in paper summarizes a geometric relation with additional branch-specific corrections |
| Sequential manipulation/contact dispatch | libero_system/integration/adapters.py: RouteBPolicy | Includes specialized hand-written logic; no global task-motion optimization is claimed |
| Contact evidence uses robot pose/progress/force | libero_system/goal_skills/controller.py: _contact_reached, _move | Operational guards; no formal contact safety or correctness proof |
| Official initial-state indices and reset-stream handling | libero_system/common/env_adapter.py: reset | Seed 7; reset stream is restored/replayed, not equivalent to every benchmark implementation |
| External ever-success and controller stopping | libero_system/integration/evaluator.py: _ExternalStickyScore, EvaluationDriver._run_b | Success can precede a later internal failure; final-only outcome is not separately recorded |
| Software-boundary counters | _ExternalStickyScore.audit | Several zero counters are constructed by implementation; not independent process attestation |
| 400 episodes and ten reused records | runtime/jobs/smolvla_scale_400_parallel_20260913/manifest.json; paper/data/provenance.json | Reuse selected by official IDs and compatible configuration before this campaign |
| Success, runtime and status tables | paper/data/episodes.jsonl; paper/scripts/analyze_results.py | Derived directly from frozen record fields; elapsed times exclude reused records |
| Published policy comparison | paper/data/published_comparisons.json; paper/scripts/build_comparison_table.py | Five externally reported mean rows with source versions, table numbers and PDF hashes; the Route B row is computed from its own episode records |
| Failure groups | failure_group in paper/scripts/analyze_results.py | Message-based terminal categories, not independently verified physical causes |
| Per-task examples in text | Goal 03: 3/10; Spatial 04: 5/10; Long 03: 6/10; Long 08: 4/10; Long 09: 0/10 | Counts and Long 09 stopping-reason breakdown checked by paper/scripts/check_paper.py |
| Qualitative rollouts | paper/data/figure_provenance.json | Selected recorded episodes, with video hashes and exact frame indices |
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
