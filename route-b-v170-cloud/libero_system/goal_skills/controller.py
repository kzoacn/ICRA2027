"""Closed-loop Route-B-style controller for LIBERO-Goal contact skills."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Protocol

import numpy as np
from scipy.spatial.transform import Rotation

from libero_system.common import (
    OSCAction,
    PolicyDecision,
    PolicyTask,
    RobotObservation,
    backproject_depth,
)

from .compiler import GoalTaskCompiler
from .detectors import DrawerHandleDetector, PlateFrontDetector, StoveKnobDetector
from .schema import (
    ContactTarget,
    DrawerEpisodeAnchor,
    GoalExecutorStatus,
    GoalSkillKind,
    GoalSkillPlan,
    GoalSkillStep,
    PushTarget,
)


class MicrowaveDoorDetector(Protocol):
    """RGB-D microwave handle/door target supplied by the route adapter."""

    def reset(self) -> None: ...

    def detect(
        self,
        observation: RobotObservation,
        kind: GoalSkillKind,
    ) -> ContactTarget: ...

    def track(
        self,
        observation: RobotObservation,
        reference: ContactTarget,
        kind: GoalSkillKind,
    ) -> ContactTarget: ...


@dataclass(frozen=True, slots=True)
class GoalControllerConfig:
    translation_scale_m: float = 0.05
    rotation_scale_rad: float = 0.5
    position_tolerance_m: float = 0.009
    rotation_tolerance_rad: float = 0.12
    precontact_clearance_m: float = 0.075
    drawer_precontact_clearance_m: float = 0.055
    drawer_lateral_clearance_m: float = 0.050
    drawer_safe_height_m: float = 0.140
    drawer_retreat_clearance_m: float = 0.080
    drawer_pull_distance_m: float = 0.170
    drawer_visual_displacement_m: float = 0.055
    # A blocked width immediately after closing can be a transient one-pad
    # wedge.  Before the full pull, move the public TCP through a short loaded
    # chord, keep closing briefly at that endpoint, and require the measured
    # jaw width to settle to a retained two-pad aperture.  This mirrors the
    # task-independent load proof used by Route C and does not infer contact
    # from simulator bodies or joints.
    drawer_load_proof_distance_m: float = 0.012
    drawer_load_proof_position_tolerance_m: float = 0.004
    drawer_load_proof_min_width_m: float = 0.008
    drawer_load_proof_max_settle_change_m: float = 0.003
    drawer_load_proof_min_ticks: int = 5
    drawer_load_proof_settle_ticks: int = 5
    drawer_load_proof_max_ticks: int = 24
    # An open drawer can reach its physical rail stop a few centimetres
    # before the deliberately conservative Cartesian pull target.  Treat
    # that only as a typed transition out of ``pull``: it needs a late,
    # aligned Cartesian plateau, retained two-pad width, and a sustained
    # wrench-magnitude rise.  Completion still needs fresh RGB-D after the
    # fingers release, so none of these proprioceptive cues is a task-success
    # signal on its own.
    drawer_pull_stop_max_residual_m: float = 0.030
    drawer_pull_stop_max_orthogonal_m: float = 0.020
    drawer_pull_stop_width_epsilon_m: float = 0.0006
    drawer_pull_stop_width_ticks: int = 4
    drawer_pull_stop_force_ticks: int = 4
    # A released top/bottom pinch can leave a horizontal handle between the
    # open fingers.  Freeze the visible handle interval from public RGB-D,
    # slide beyond its nearer/clearer long-axis endpoint plus a physical finger
    # margin, and only then retreat from the drawer.  Every segment is proven
    # by public EE pose.  When the local RGB-D interval is unreliable, two
    # short opposite-axis probes are the only fallback; neither permits a lift.
    drawer_rail_stop_axis_span_min_m: float = 0.025
    drawer_rail_stop_axis_span_max_m: float = 0.180
    drawer_rail_stop_axis_normal_band_m: float = 0.012
    drawer_rail_stop_axis_vertical_band_m: float = 0.018
    drawer_rail_stop_axis_endpoint_quantile: float = 0.05
    drawer_rail_stop_axis_endpoint_margin_m: float = 0.020
    drawer_rail_stop_axis_clearance_preference_m: float = 0.015
    # Every Cartesian rail-stop axis command is a probe-sized micro-segment.
    # A 25-mm recovery remains a *budget*, never one unobserved command.
    drawer_rail_stop_axis_segment_m: float = 0.010
    drawer_rail_stop_axis_probe_m: float = 0.010
    drawer_rail_stop_axis_progress_tolerance_m: float = 0.004
    drawer_rail_stop_axis_max_cross_drift_m: float = 0.008
    # The released jaw aperture is public proprioception.  Half that aperture
    # plus a fixed Panda fingertip/palm overhang gives a conservative lateral
    # body envelope; reject, rather than clip, an envelope above the typed
    # physical cap.  A short normal-direction probe must then demonstrate
    # actual public EE progress before the ordinary retreat is authorised.
    drawer_rail_stop_body_padding_m: float = 0.015
    drawer_rail_stop_body_margin_max_m: float = 0.060
    drawer_rail_stop_axis_retry_increment_m: float = 0.025
    drawer_rail_stop_axis_retry_max_m: float = 0.075
    drawer_rail_stop_axis_total_max_m: float = 0.260
    drawer_rail_stop_outward_probe_m: float = 0.010
    drawer_rail_stop_outward_progress_tolerance_m: float = 0.002
    drawer_rail_stop_outward_max_cross_drift_m: float = 0.004
    drawer_rail_stop_outward_max_rotation_rad: float = 0.040
    # A rail-stop release remains close to the handle even after the first
    # 10-mm normal probe.  Accumulate a full measured normal retreat in
    # probe-sized segments before allowing the vertical safety motion.
    drawer_rail_stop_normal_retreat_m: float = 0.075
    drawer_rail_stop_axis_min_points: int = 12
    drawer_rail_stop_axis_segment_max_ticks: int = 18
    drawer_rail_stop_axis_probe_max_ticks: int = 12
    drawer_rail_stop_outward_probe_max_ticks: int = 12
    # A blocked normal probe can end at a transient compliance peak.  Issue a
    # bounded open-gripper hold before freezing the next rail-axis origin so a
    # one-sample recoil cannot deterministically trip the retry's 2-mm reverse
    # gate.  The first returned sample has its own 3-mm recoil cap; after that
    # sample the settle anchor is immutable and the original 2-mm gate applies.
    drawer_rail_stop_axis_retry_settle_stability_m: float = 0.00035
    drawer_rail_stop_axis_retry_settle_warmup_reverse_max_m: float = 0.003
    drawer_rail_stop_axis_retry_settle_stable_ticks: int = 2
    drawer_rail_stop_axis_retry_settle_max_ticks: int = 6
    drawer_closed_handle_offset_m: float = 0.085
    drawer_close_min_travel_m: float = 0.040
    drawer_close_max_travel_m: float = 0.180
    drawer_close_overshoot_m: float = 0.020
    microwave_travel_m: float = 0.140
    microwave_visual_displacement_m: float = 0.035
    microwave_visual_min_confidence: float = 0.45
    microwave_retreat_clearance_m: float = 0.090
    drawer_grasp_z_offset_m: float = -0.008
    drawer_preshape_command: float = 0.0
    drawer_preshape_width_m: float = 0.045
    drawer_preshape_hysteresis_m: float = 0.007
    drawer_close_pusher_width_m: float = 0.014
    drawer_close_pusher_hysteresis_m: float = 0.003
    drawer_close_tool_radius_m: float = 0.018
    drawer_close_slot_clearance_m: float = 0.008
    drawer_close_front_support_tolerance_m: float = 0.040
    drawer_obstacle_radius_m: float = 0.075
    drawer_obstacle_below_handle_m: float = 0.105
    drawer_waypoint_tolerance_m: float = 0.018
    drawer_close_waypoint_tolerance_m: float = 0.032
    drawer_safe_waypoint_tolerance_m: float = 0.050
    drawer_descent_tolerance_m: float = 0.036
    drawer_close_descent_stage_tolerance_m: float = 0.042
    staged_translation_step_m: float = 0.055
    staged_waypoint_tolerance_m: float = 0.032
    contact_retreat_tolerance_m: float = 0.038
    drawer_blocked_min_width_m: float = 0.005
    drawer_blocked_max_width_m: float = 0.060
    drawer_release_width_m: float = 0.065
    drawer_retry_grasp_z_delta_m: float = 0.016
    drawer_max_attempts: int = 2
    # A free-space drawer waypoint can become kinematically unreachable when
    # the wrist arrives at the lower joint-7 limit, even though the same
    # two-pad contact frame has an exactly equivalent pi-yaw representation.
    # Detect only a public Cartesian plateau with low force, lift clear of the
    # fixture, change to that equivalent frame, and resume the original
    # sensor target.  The recovery is bounded to once per language skill.
    drawer_wrist_recovery_stall_ticks: int = 18
    drawer_wrist_recovery_lift_m: float = 0.120
    drawer_wrist_recovery_min_rotation_error_rad: float = 0.16
    knob_turn_rad: float = 1.82
    knob_visual_min_angle_rad: float = 0.35
    knob_track_radius_m: float = 0.060
    push_contact_z_offset_m: float = 0.008
    push_rim_radius_fraction: float = 0.82
    push_safe_height_m: float = 0.120
    push_goal_overshoot_m: float = 0.035
    push_retry_overshoot_m: float = 0.025
    push_retry_rim_inset_fraction: float = 0.16
    push_retry_contact_depth_m: float = 0.006
    push_safe_waypoint_tolerance_m: float = 0.045
    push_contact_tolerance_m: float = 0.045
    # The red-rim track supplies an explicit metric target.  Require the
    # refreshed plate centre to enter a tight neighbourhood of that target;
    # a fixed displacement alone can stop a few centimetres outside a narrow
    # stove-front region.  A third bounded recontact handles the remaining
    # short correction without extending successful first/second attempts.
    push_goal_tolerance_m: float = 0.015
    push_max_attempts: int = 3
    push_release_width_m: float = 0.065
    microwave_safe_height_m: float = 0.100
    microwave_safe_position_tolerance_m: float = 0.050
    microwave_safe_rotation_tolerance_rad: float = 0.28
    microwave_close_outer_clearance_m: float = 0.155
    microwave_closed_handle_surface_inset_m: float = 0.010
    microwave_open_edge_surface_inset_m: float = 0.006
    microwave_waypoint_tolerance_m: float = 0.032
    # Route B's hinge is reconstructed from a closed-handle point plus a
    # detector OBB, whose depth axis can retain centimetre-scale bias.  This
    # tolerance advances only intermediate hinge waypoints; every step still
    # requires a retained two-pad width and completion still requires fresh
    # visual displacement.
    microwave_arc_waypoint_tolerance_m: float = 0.028
    microwave_arc_segment_rad: float = 0.05
    microwave_arc_timeout_ticks: int = 360
    microwave_open_angle_rad: float = 1.23
    # A 10-mm rolling target leaves only a 0.2 normalised OSC translation
    # command.  Real LIBERO microwave traces show that this opens the latch
    # but then equilibrates against hinge friction for hundreds of ticks.
    # Keep the command sensor-relative and bounded, while retaining enough
    # Cartesian load to continue the measured hinge motion.
    microwave_open_tangent_step_m: float = 0.030
    microwave_open_tangent_max_step_m: float = 0.038
    microwave_open_high_angle_rad: float = 0.82
    microwave_open_high_angle_tangent_step_m: float = 0.016
    microwave_open_high_angle_tangent_max_step_m: float = 0.020
    microwave_open_tangent_ramp_start_ticks: int = 4
    microwave_open_tangent_ramp_per_tick_m: float = 0.002
    # A closed-front handle remains a two-pad grasp for roughly 0.5 rad in the
    # public LIBERO geometry.  Release before that grasp becomes one-sided,
    # reacquire the moving vertical edge from fresh RGB-D, and continue in a
    # new wrist frame.  The hinge-radius and local-anchor gates prevent a
    # fixed appliance edge from being accepted as the continuation.
    microwave_open_regrasp_segment_rad: float = 0.43
    microwave_open_continuation_segment_rad: float = 0.18
    # A Cartesian servo can keep making milliradian progress while sitting on
    # the same joint-space branch, so the ordinary no-progress watchdog never
    # fires.  Bound each retained-grasp continuation and reacquire the same
    # moving RGB-D edge before that slow crawl consumes the episode budget.
    microwave_open_segment_max_ticks: int = 42
    microwave_open_high_angle_segment_max_ticks: int = 24
    microwave_open_regrasp_precontact_clearance_m: float = 0.022
    microwave_open_regrasp_engage_ticks: int = 5
    # Once the door is sufficiently oblique, pulling the vertical edge from
    # the appliance-front side becomes a one-sided pinch.  Route around the
    # sensor-observed free edge and finish with a closed-finger push from the
    # back side of the same hinge-gated edge.
    microwave_open_push_transition_rad: float = 0.82
    microwave_open_push_edge_clearance_m: float = 0.070
    microwave_open_push_backside_clearance_m: float = 0.055
    microwave_open_push_lift_m: float = 0.040
    microwave_open_push_precontact_m: float = 0.030
    microwave_open_push_tangent_step_m: float = 0.055
    microwave_open_push_tangent_max_step_m: float = 0.055
    microwave_open_push_tangent_ramp_start_ticks: int = 4
    microwave_open_push_tangent_ramp_per_tick_m: float = 0.003
    microwave_open_push_contact_inset_m: float = 0.006
    microwave_open_push_radius_tolerance_m: float = 0.020
    microwave_open_push_radius_correction_max_m: float = 0.012
    microwave_open_push_reverse_tolerance_rad: float = 0.020
    microwave_open_push_release_radial_escape_m: float = 0.060
    microwave_open_push_final_exit_m: float = 0.030
    microwave_open_push_waypoint_tolerance_m: float = 0.012
    microwave_open_push_compact_width_m: float = 0.014
    microwave_open_push_preshape_ticks: int = 4
    microwave_open_push_segment_max_ticks: int = 48
    microwave_open_push_continuation_segment_max_ticks: int = 80
    # A protruding handle can momentarily spread an otherwise compact
    # back-side pusher.  Never keep moving tangentially above the 14-mm gate:
    # allow two stationary close commands, then release and reacquire through
    # fresh RGB-D.  Repeated compact-loss cycles are independently bounded.
    microwave_open_push_recompact_max_ticks: int = 2
    microwave_open_push_max_compact_recoveries: int = 2
    # A continuation can make a few milliradians of public hinge progress per
    # tick and therefore evade the one-sample stall watchdog for all 80 ticks.
    # Use a bounded window and reserve part of that same fixed contact horizon
    # for a fresh RGB-D reacquisition whenever its optimistic rate cannot
    # finish the remaining public opening angle.
    microwave_open_push_progress_window_ticks: int = 12
    microwave_open_push_reacquire_reserve_ticks: int = 12
    microwave_open_regrasp_retreat_m: float = 0.065
    microwave_open_regrasp_lift_m: float = 0.025
    microwave_open_release_escape_m: float = 0.040
    microwave_open_release_escape_lift_m: float = 0.015
    microwave_open_regrasp_anchor_radius_m: float = 0.080
    microwave_open_regrasp_radius_tolerance_m: float = 0.035
    microwave_open_release_anchor_width_m: float = 0.030
    microwave_open_stall_progress_epsilon_rad: float = 0.002
    microwave_open_stall_min_segment_progress_rad: float = 0.06
    microwave_open_stall_ticks: int = 18
    microwave_open_max_regrasps: int = 10
    microwave_open_rebase_arc_rad: float = 0.38
    microwave_open_min_chunk_progress_rad: float = 0.15
    microwave_open_max_rebases: int = 5
    microwave_close_overshoot_rad: float = 0.08
    # A fully closed door edge can remain pinched between one finger and the
    # hand even while the public gripper command is opening.  After the normal
    # release dwell, roll the frozen Cartesian target only along the
    # sensor-derived closed-door free-space normal.  This small, bounded
    # motion merely unseats the edge; the ordinary 90-mm retreat and strict
    # fresh RGB-D terminal verifier remain separate and mandatory.
    microwave_close_release_unseat_step_m: float = 0.002
    microwave_close_release_unseat_max_m: float = 0.018
    # Before a retained edge pinch reaches the appliance frame, switch to a
    # compact exterior-face pusher.  All geometry below is expressed in the
    # frozen RGB-D hinge frame; no benchmark joint or task identity is used.
    microwave_close_push_transition_remaining_rad: float = 0.24
    microwave_close_push_release_radial_m: float = 0.060
    microwave_close_push_release_outward_m: float = 0.040
    microwave_close_push_release_lift_m: float = 0.020
    microwave_close_push_contact_inset_m: float = 0.035
    microwave_close_push_precontact_m: float = 0.030
    # The Panda grip site can still be several centimetres from the sensed
    # door plane when the closed fingertips first carry load.  Admit that
    # contact only inside a short, sensor-framed precontact corridor and only
    # after both a force-magnitude rise and repeated low Cartesian progress.
    # This is a contact transition, never a completion signal.
    microwave_close_push_precontact_contact_residual_m: float = 0.060
    microwave_close_push_precontact_progress_epsilon_m: float = 0.003
    microwave_close_push_precontact_stall_ticks: int = 2
    microwave_close_push_compact_width_m: float = 0.014
    microwave_close_push_tangent_step_m: float = 0.030
    microwave_close_push_radius_tolerance_m: float = 0.022
    microwave_close_push_radius_correction_max_m: float = 0.010
    microwave_close_push_reverse_tolerance_rad: float = 0.020
    microwave_close_push_goal_tolerance_rad: float = 0.035
    # Reaching the frozen slot angle is a geometric cue, not proof that the
    # door joint has followed the hand.  Continue with a bounded face-normal
    # load and accept completion of the push only from the public Cartesian,
    # jaw-width, and force plateaus in ``_microwave_close_mechanical_stop``.
    microwave_close_push_terminal_step_m: float = 0.010
    microwave_close_push_terminal_max_displacement_m: float = 0.065
    microwave_close_push_terminal_max_extra_angle_rad: float = 0.32
    microwave_close_push_terminal_bound_tolerance_m: float = 0.004
    microwave_close_push_exit_radial_m: float = 0.070
    microwave_close_push_exit_outward_m: float = 0.040
    microwave_close_push_exit_lift_m: float = 0.025
    microwave_close_push_waypoint_tolerance_m: float = 0.015
    microwave_close_verify_slot_tolerance_m: float = 0.060
    microwave_close_push_preshape_ticks: int = 4
    microwave_close_push_terminal_max_ticks: int = 48
    microwave_wrist_corotation_limit_rad: float = 0.58
    microwave_hinge_radius_range_m: tuple[float, float] = (0.10, 0.60)
    microwave_mechanical_stop_min_angle_rad: float = 0.35
    microwave_mechanical_stop_residual_m: float = 0.10
    microwave_force_plateau_epsilon_n: float = 1.5
    microwave_force_plateau_ticks: int = 5
    microwave_width_plateau_epsilon_m: float = 0.0006
    microwave_width_plateau_ticks: int = 5
    microwave_rotation_tolerance_rad: float = 0.24
    engage_ticks: int = 8
    release_ticks: int = 5
    contact_min_ticks: int = 8
    contact_stall_ticks: int = 7
    progress_epsilon_m: float = 0.00035
    # Contact can leave a Cartesian residual because the sensed fixture blocks
    # the commanded grip-site pose.  ``robot0_eef_pos`` is already the MuJoCo
    # grip site between the fingers; no fingertip offset is applied here.
    contact_position_tolerance_m: float = 0.120
    drawer_contact_tolerance_m: float = 0.032
    contact_force_delta_n: float = 5.0
    phase_timeout_ticks: int = 150
    max_detection_misses: int = 4

    def __post_init__(self) -> None:
        # Python comparisons against NaN are false, so range checks alone can
        # silently turn a safety ceiling into a fail-open gate.  Validate all
        # numeric configuration at the single construction boundary before
        # evaluating the more specific range and ordering rules below.
        for parameter in fields(self):
            value = getattr(self, parameter.name)
            default = parameter.default
            if isinstance(default, float):
                if (
                    isinstance(value, (bool, np.bool_))
                    or not isinstance(
                        value,
                        (int, float, np.integer, np.floating),
                    )
                    or not np.isfinite(float(value))
                ):
                    raise ValueError(
                        f"{parameter.name} must be a finite non-boolean number"
                    )
            elif type(default) is int:
                if type(value) is not int or value <= 0:
                    raise ValueError(
                        f"{parameter.name} must be a strictly positive integer"
                    )
            elif isinstance(default, tuple):
                if (
                    not isinstance(value, tuple)
                    or len(value) != len(default)
                    or any(
                        isinstance(item, (bool, np.bool_))
                        or not isinstance(
                            item,
                            (int, float, np.integer, np.floating),
                        )
                        or not np.isfinite(float(item))
                        for item in value
                    )
                ):
                    raise ValueError(
                        f"{parameter.name} must contain finite non-boolean numbers"
                    )
        if any(
            value <= 0
            for value in (
                self.translation_scale_m,
                self.rotation_scale_rad,
                self.position_tolerance_m,
                self.rotation_tolerance_rad,
                self.precontact_clearance_m,
                self.drawer_precontact_clearance_m,
                self.drawer_lateral_clearance_m,
                self.drawer_safe_height_m,
                self.drawer_retreat_clearance_m,
                self.drawer_pull_distance_m,
                self.drawer_visual_displacement_m,
                self.drawer_load_proof_distance_m,
                self.drawer_load_proof_position_tolerance_m,
                self.drawer_load_proof_min_width_m,
                self.drawer_load_proof_max_settle_change_m,
                self.drawer_pull_stop_max_residual_m,
                self.drawer_pull_stop_max_orthogonal_m,
                self.drawer_pull_stop_width_epsilon_m,
                self.drawer_rail_stop_axis_span_min_m,
                self.drawer_rail_stop_axis_span_max_m,
                self.drawer_rail_stop_axis_normal_band_m,
                self.drawer_rail_stop_axis_vertical_band_m,
                self.drawer_rail_stop_axis_endpoint_quantile,
                self.drawer_rail_stop_axis_endpoint_margin_m,
                self.drawer_rail_stop_axis_clearance_preference_m,
                self.drawer_rail_stop_axis_segment_m,
                self.drawer_rail_stop_axis_probe_m,
                self.drawer_rail_stop_axis_progress_tolerance_m,
                self.drawer_rail_stop_axis_max_cross_drift_m,
                self.drawer_rail_stop_body_padding_m,
                self.drawer_rail_stop_body_margin_max_m,
                self.drawer_rail_stop_axis_retry_increment_m,
                self.drawer_rail_stop_axis_retry_max_m,
                self.drawer_rail_stop_axis_total_max_m,
                self.drawer_rail_stop_outward_probe_m,
                self.drawer_rail_stop_outward_progress_tolerance_m,
                self.drawer_rail_stop_outward_max_cross_drift_m,
                self.drawer_rail_stop_outward_max_rotation_rad,
                self.drawer_rail_stop_normal_retreat_m,
                self.drawer_rail_stop_axis_retry_settle_stability_m,
                self.drawer_rail_stop_axis_retry_settle_warmup_reverse_max_m,
                self.drawer_closed_handle_offset_m,
                self.drawer_close_min_travel_m,
                self.drawer_close_max_travel_m,
                self.drawer_close_overshoot_m,
                self.microwave_travel_m,
                self.microwave_visual_displacement_m,
                self.microwave_retreat_clearance_m,
                self.drawer_waypoint_tolerance_m,
                self.drawer_close_waypoint_tolerance_m,
                self.drawer_safe_waypoint_tolerance_m,
                self.drawer_descent_tolerance_m,
                self.drawer_close_descent_stage_tolerance_m,
                self.staged_translation_step_m,
                self.staged_waypoint_tolerance_m,
                self.contact_retreat_tolerance_m,
                self.drawer_preshape_width_m,
                self.drawer_preshape_hysteresis_m,
                self.drawer_close_pusher_width_m,
                self.drawer_close_pusher_hysteresis_m,
                self.drawer_close_tool_radius_m,
                self.drawer_close_slot_clearance_m,
                self.drawer_close_front_support_tolerance_m,
                self.drawer_obstacle_radius_m,
                self.drawer_obstacle_below_handle_m,
                self.drawer_release_width_m,
                self.drawer_retry_grasp_z_delta_m,
                self.drawer_wrist_recovery_lift_m,
                self.drawer_wrist_recovery_min_rotation_error_rad,
                self.knob_turn_rad,
                self.knob_visual_min_angle_rad,
                self.knob_track_radius_m,
                self.push_rim_radius_fraction,
                self.push_safe_height_m,
                self.push_goal_overshoot_m,
                self.push_retry_overshoot_m,
                self.push_retry_rim_inset_fraction,
                self.push_retry_contact_depth_m,
                self.push_safe_waypoint_tolerance_m,
                self.push_contact_tolerance_m,
                self.push_goal_tolerance_m,
                self.push_release_width_m,
            self.microwave_safe_height_m,
            self.microwave_safe_position_tolerance_m,
            self.microwave_safe_rotation_tolerance_rad,
            self.microwave_close_outer_clearance_m,
            self.microwave_closed_handle_surface_inset_m,
            self.microwave_open_edge_surface_inset_m,
                self.microwave_waypoint_tolerance_m,
                self.microwave_arc_waypoint_tolerance_m,
            self.microwave_arc_segment_rad,
            self.microwave_open_angle_rad,
            self.microwave_open_tangent_step_m,
            self.microwave_open_tangent_max_step_m,
            self.microwave_open_tangent_ramp_per_tick_m,
            self.microwave_open_high_angle_rad,
            self.microwave_open_high_angle_tangent_step_m,
            self.microwave_open_high_angle_tangent_max_step_m,
            self.microwave_open_regrasp_segment_rad,
            self.microwave_open_continuation_segment_rad,
            self.microwave_open_regrasp_precontact_clearance_m,
            self.microwave_open_push_transition_rad,
            self.microwave_open_push_edge_clearance_m,
            self.microwave_open_push_backside_clearance_m,
            self.microwave_open_push_lift_m,
            self.microwave_open_push_precontact_m,
            self.microwave_open_push_tangent_step_m,
            self.microwave_open_push_tangent_max_step_m,
            self.microwave_open_push_tangent_ramp_per_tick_m,
            self.microwave_open_push_contact_inset_m,
            self.microwave_open_push_radius_tolerance_m,
            self.microwave_open_push_radius_correction_max_m,
            self.microwave_open_push_reverse_tolerance_rad,
            self.microwave_open_push_release_radial_escape_m,
            self.microwave_open_push_final_exit_m,
            self.microwave_open_push_waypoint_tolerance_m,
            self.microwave_open_push_compact_width_m,
            self.microwave_open_regrasp_retreat_m,
            self.microwave_open_regrasp_lift_m,
            self.microwave_open_release_escape_m,
            self.microwave_open_release_escape_lift_m,
            self.microwave_open_regrasp_anchor_radius_m,
            self.microwave_open_regrasp_radius_tolerance_m,
            self.microwave_open_release_anchor_width_m,
            self.microwave_open_stall_progress_epsilon_rad,
            self.microwave_open_stall_min_segment_progress_rad,
            self.microwave_open_rebase_arc_rad,
            self.microwave_open_min_chunk_progress_rad,
            self.microwave_close_overshoot_rad,
            self.microwave_close_release_unseat_step_m,
            self.microwave_close_release_unseat_max_m,
            self.microwave_close_push_transition_remaining_rad,
            self.microwave_close_push_release_radial_m,
            self.microwave_close_push_release_outward_m,
            self.microwave_close_push_release_lift_m,
            self.microwave_close_push_contact_inset_m,
            self.microwave_close_push_precontact_m,
            self.microwave_close_push_precontact_contact_residual_m,
            self.microwave_close_push_precontact_progress_epsilon_m,
            self.microwave_close_push_compact_width_m,
            self.microwave_close_push_tangent_step_m,
            self.microwave_close_push_radius_tolerance_m,
            self.microwave_close_push_radius_correction_max_m,
            self.microwave_close_push_reverse_tolerance_rad,
            self.microwave_close_push_goal_tolerance_rad,
            self.microwave_close_push_terminal_step_m,
            self.microwave_close_push_terminal_max_displacement_m,
            self.microwave_close_push_terminal_max_extra_angle_rad,
            self.microwave_close_push_terminal_bound_tolerance_m,
            self.microwave_close_push_exit_radial_m,
            self.microwave_close_push_exit_outward_m,
            self.microwave_close_push_exit_lift_m,
            self.microwave_close_push_waypoint_tolerance_m,
            self.microwave_close_verify_slot_tolerance_m,
            self.microwave_wrist_corotation_limit_rad,
                self.microwave_mechanical_stop_min_angle_rad,
                self.microwave_mechanical_stop_residual_m,
                self.microwave_force_plateau_epsilon_n,
                self.microwave_width_plateau_epsilon_m,
                self.microwave_rotation_tolerance_rad,
                self.progress_epsilon_m,
                self.contact_position_tolerance_m,
                self.drawer_contact_tolerance_m,
                self.contact_force_delta_n,
            )
        ):
            raise ValueError("controller distances, scales, and thresholds must be positive")
        if any(
            value <= 0
            for value in (
                self.engage_ticks,
                self.release_ticks,
                self.contact_min_ticks,
                self.contact_stall_ticks,
                self.phase_timeout_ticks,
                self.max_detection_misses,
                self.drawer_max_attempts,
                self.drawer_load_proof_min_ticks,
                self.drawer_load_proof_settle_ticks,
                self.drawer_load_proof_max_ticks,
                self.drawer_pull_stop_width_ticks,
                self.drawer_pull_stop_force_ticks,
                self.drawer_rail_stop_axis_min_points,
                self.drawer_rail_stop_axis_segment_max_ticks,
                self.drawer_rail_stop_axis_probe_max_ticks,
                self.drawer_rail_stop_outward_probe_max_ticks,
                self.drawer_rail_stop_axis_retry_settle_stable_ticks,
                self.drawer_rail_stop_axis_retry_settle_max_ticks,
                self.drawer_wrist_recovery_stall_ticks,
                self.push_max_attempts,
                self.microwave_open_max_rebases,
                self.microwave_open_max_regrasps,
                self.microwave_open_stall_ticks,
                self.microwave_open_segment_max_ticks,
                self.microwave_open_high_angle_segment_max_ticks,
                self.microwave_open_regrasp_engage_ticks,
                self.microwave_open_push_preshape_ticks,
                self.microwave_open_push_segment_max_ticks,
                self.microwave_open_push_continuation_segment_max_ticks,
                self.microwave_open_push_recompact_max_ticks,
                self.microwave_open_push_max_compact_recoveries,
                self.microwave_open_push_progress_window_ticks,
                self.microwave_open_push_reacquire_reserve_ticks,
                self.microwave_open_push_tangent_ramp_start_ticks,
                self.microwave_open_tangent_ramp_start_ticks,
                self.microwave_close_push_preshape_ticks,
                self.microwave_close_push_precontact_stall_ticks,
                self.microwave_close_push_terminal_max_ticks,
                self.microwave_width_plateau_ticks,
                self.microwave_force_plateau_ticks,
                self.microwave_arc_timeout_ticks,
            )
        ):
            raise ValueError("controller tick limits must be positive")
        if not (
            0.0 < self.drawer_blocked_min_width_m
            < self.drawer_blocked_max_width_m
        ):
            raise ValueError("drawer blocked-width bounds must be ordered and positive")
        if self.drawer_pull_stop_max_residual_m >= self.drawer_pull_distance_m:
            raise ValueError("drawer pull-stop residual must be below the commanded pull")
        if not (
            self.drawer_rail_stop_axis_span_min_m
            < self.drawer_rail_stop_axis_span_max_m
        ):
            raise ValueError(
                "drawer rail-stop observable axis-span bounds must be ordered"
            )
        if not 0.0 < self.drawer_rail_stop_axis_endpoint_quantile < 0.5:
            raise ValueError("drawer rail-stop endpoint quantile must lie in (0, 0.5)")
        if self.drawer_rail_stop_axis_progress_tolerance_m >= min(
            self.drawer_rail_stop_axis_endpoint_margin_m,
            self.drawer_rail_stop_axis_segment_m,
            self.drawer_rail_stop_axis_probe_m,
        ):
            raise ValueError(
                "drawer rail-stop axis tolerance must be below margin, segment, and probe"
            )
        if not (
            self.drawer_rail_stop_axis_endpoint_margin_m
            < self.drawer_rail_stop_body_margin_max_m
        ):
            raise ValueError(
                "drawer rail-stop body cap must exceed the endpoint margin"
            )
        if (
            0.5 * self.drawer_release_width_m
            + self.drawer_rail_stop_body_padding_m
            > self.drawer_rail_stop_body_margin_max_m
        ):
            raise ValueError(
                "drawer released-width body envelope exceeds its physical cap"
            )
        if (
            self.drawer_rail_stop_axis_retry_increment_m
            > self.drawer_rail_stop_axis_retry_max_m
        ):
            raise ValueError(
                "drawer rail-stop retry increment must not exceed its retry bound"
            )
        if self.drawer_rail_stop_axis_retry_increment_m > 0.025:
            raise ValueError(
                "drawer rail-stop retry increment must be at most 0.025 m"
            )
        if self.drawer_rail_stop_axis_retry_max_m > 0.075:
            raise ValueError(
                "drawer rail-stop retry bound must be at most 0.075 m"
            )
        if self.drawer_rail_stop_axis_total_max_m > 0.260:
            raise ValueError(
                "drawer rail-stop total axis bound must be at most 0.260 m"
            )
        if (
            self.drawer_rail_stop_axis_span_max_m
            + self.drawer_rail_stop_body_margin_max_m
            > self.drawer_rail_stop_axis_total_max_m
        ):
            raise ValueError(
                "drawer rail-stop total bound cannot cover the typed handle envelope"
            )
        if (
            self.drawer_rail_stop_outward_progress_tolerance_m
            >= self.drawer_rail_stop_outward_probe_m
        ):
            raise ValueError(
                "drawer outward-probe tolerance must be below the probe distance"
            )
        if self.drawer_rail_stop_outward_probe_m > 0.010:
            raise ValueError(
                "drawer rail-stop outward probe must be at most 0.010 m"
            )
        if self.drawer_rail_stop_axis_segment_m > 0.010:
            raise ValueError(
                "drawer rail-stop axis segment must be at most 0.010 m"
            )
        if self.drawer_rail_stop_axis_probe_m > 0.010:
            raise ValueError(
                "drawer rail-stop axis probe must be at most 0.010 m"
            )
        if self.drawer_rail_stop_normal_retreat_m < 0.075:
            raise ValueError(
                "drawer rail-stop normal retreat must be at least 0.075 m"
            )
        if self.drawer_rail_stop_normal_retreat_m > 0.075:
            raise ValueError(
                "drawer rail-stop normal retreat must be at most 0.075 m"
            )
        if self.drawer_rail_stop_outward_max_cross_drift_m > 0.004:
            raise ValueError(
                "drawer rail-stop cumulative cross drift must be at most 0.004 m"
            )
        if self.drawer_rail_stop_outward_max_rotation_rad > 0.040:
            raise ValueError(
                "drawer rail-stop cumulative rotation must be at most 0.040 rad"
            )
        if self.drawer_rail_stop_outward_probe_max_ticks > 12:
            raise ValueError(
                "drawer rail-stop normal segment timeout must be at most 12 ticks"
            )
        if self.drawer_rail_stop_axis_retry_settle_stability_m > 0.00035:
            raise ValueError(
                "drawer rail-stop retry settle stability must be at most 0.00035 m"
            )
        if self.drawer_rail_stop_axis_retry_settle_warmup_reverse_max_m > 0.003:
            raise ValueError(
                "drawer rail-stop retry settle warm-up reverse bound must be at "
                "most 0.003 m"
            )
        if self.drawer_rail_stop_axis_retry_settle_stable_ticks != 2:
            raise ValueError(
                "drawer rail-stop retry settle proof must use exactly 2 stable ticks"
            )
        if self.drawer_rail_stop_axis_retry_settle_max_ticks > 6:
            raise ValueError(
                "drawer rail-stop retry settle must be at most 6 ticks"
            )
        if (
            self.drawer_rail_stop_axis_retry_settle_stable_ticks
            >= self.drawer_rail_stop_axis_retry_settle_max_ticks
        ):
            raise ValueError(
                "drawer rail-stop retry settle stable ticks must be below its timeout"
            )
        if not (
            self.drawer_rail_stop_outward_probe_m
            <= self.drawer_rail_stop_normal_retreat_m
            <= self.drawer_retreat_clearance_m
        ):
            raise ValueError(
                "drawer rail-stop normal retreat must contain a probe and fit "
                "inside the ordinary retreat clearance"
            )
        if not (
            self.drawer_load_proof_position_tolerance_m
            < self.drawer_load_proof_distance_m
            < self.drawer_pull_distance_m
        ):
            raise ValueError(
                "drawer load-proof tolerance, chord, and full pull must be ordered"
            )
        if not (
            self.drawer_blocked_min_width_m
            < self.drawer_load_proof_min_width_m
            < self.drawer_blocked_max_width_m
        ):
            raise ValueError(
                "drawer load-proof width must lie inside the blocked-width interval"
            )
        if self.drawer_load_proof_min_ticks >= self.drawer_load_proof_max_ticks:
            raise ValueError("drawer load-proof minimum ticks must be below its timeout")
        if not (
            self.drawer_blocked_min_width_m
            < self.microwave_open_release_anchor_width_m
            < self.drawer_release_width_m
        ):
            raise ValueError(
                "microwave release-anchor width must lie inside the blocked/released interval"
            )
        if self.microwave_open_tangent_max_step_m < self.microwave_open_tangent_step_m:
            raise ValueError("microwave tangent maximum must not be below its base step")
        if (
            self.microwave_open_push_continuation_segment_max_ticks
            < self.microwave_open_push_segment_max_ticks
        ):
            raise ValueError(
                "microwave back-side continuation must not be shorter than its first segment"
            )
        if (
            self.microwave_open_push_progress_window_ticks
            + self.microwave_open_push_reacquire_reserve_ticks
            >= self.microwave_open_push_continuation_segment_max_ticks
        ):
            raise ValueError(
                "microwave progress window and reacquisition reserve must fit the continuation"
            )
        if (
            self.microwave_open_high_angle_tangent_max_step_m
            < self.microwave_open_high_angle_tangent_step_m
        ):
            raise ValueError(
                "microwave high-angle tangent maximum must not be below its base step"
            )
        if (
            self.microwave_close_release_unseat_step_m
            > self.microwave_close_release_unseat_max_m
        ):
            raise ValueError(
                "microwave close-release unseat step must not exceed its bound"
            )
        if (
            self.microwave_close_release_unseat_max_m
            > self.microwave_retreat_clearance_m
        ):
            raise ValueError(
                "microwave close-release unseat bound must not exceed the formal retreat"
            )
        if (
            self.microwave_close_push_contact_inset_m
            >= self.microwave_close_push_release_radial_m
            or self.microwave_close_push_contact_inset_m
            >= self.microwave_close_push_exit_radial_m
        ):
            raise ValueError(
                "microwave close pusher must clear farther radially than its face inset"
            )
        if not (
            self.drawer_blocked_min_width_m
            < self.microwave_close_push_compact_width_m
            < self.drawer_release_width_m
        ):
            raise ValueError(
                "microwave close pusher width must lie inside the blocked/released interval"
            )
        if (
            self.microwave_close_push_goal_tolerance_rad
            >= self.microwave_close_push_transition_remaining_rad
        ):
            raise ValueError(
                "microwave close-push goal tolerance must be below its transition angle"
            )
        if (
            self.microwave_close_push_precontact_progress_epsilon_m
            >= self.microwave_close_push_precontact_contact_residual_m
        ):
            raise ValueError(
                "microwave close-push progress epsilon must be below its "
                "early-contact residual bound"
            )
        if (
            self.microwave_close_push_terminal_step_m
            > self.microwave_close_push_terminal_max_displacement_m
        ):
            raise ValueError(
                "microwave close-push terminal step must fit inside its displacement bound"
            )
        if (
            self.microwave_open_push_tangent_max_step_m
            < self.microwave_open_push_tangent_step_m
        ):
            raise ValueError(
                "microwave push tangent maximum must not be below its base step"
            )
        if not -1.0 <= self.drawer_preshape_command <= 1.0:
            raise ValueError("drawer preshape command must lie in [-1, 1]")
        if (
            self.drawer_close_pusher_width_m
            >= self.drawer_preshape_width_m - self.drawer_preshape_hysteresis_m
        ):
            raise ValueError("drawer close pusher must be narrower than the preshape band")
        if self.drawer_close_pusher_hysteresis_m >= self.drawer_close_pusher_width_m:
            raise ValueError("drawer close pusher hysteresis must be below its target width")
        if self.push_rim_radius_fraction > 1.0:
            raise ValueError("push rim radius fraction cannot exceed 1")
        if (
            self.push_retry_rim_inset_fraction * (self.push_max_attempts - 1)
            >= self.push_rim_radius_fraction
        ):
            raise ValueError("push retry rim inset must retain a positive radius")
        if self.drawer_close_min_travel_m > self.drawer_close_max_travel_m:
            raise ValueError("drawer close travel bounds must be ordered")
        if self.microwave_close_outer_clearance_m <= self.precontact_clearance_m:
            raise ValueError("microwave close outer clearance must exceed precontact clearance")
        if not 0.0 < self.microwave_visual_min_confidence <= 1.0:
            raise ValueError("microwave visual confidence must lie in (0, 1]")
        hinge_min, hinge_max = self.microwave_hinge_radius_range_m
        if not 0.0 < hinge_min < hinge_max:
            raise ValueError("microwave hinge-radius bounds must be ordered and positive")


@dataclass(frozen=True, slots=True)
class _DrawerRailStopNormalCreditEvidence:
    """Validated same-observation normal evidence awaiting an accepted action."""

    global_normal_m: float
    ledger_after_m: float
    credit_increment_m: float
    source: str


class GoalContactPolicy:
    """Common-policy implementation for drawer, knob, and plate-push tasks.

    The object accepts exactly ``PolicyTask`` language and ``RobotObservation``
    sensor data.  Evaluator reward/success cannot enter this API.
    """

    def __init__(
        self,
        *,
        compiler: GoalTaskCompiler | None = None,
        drawer_detector: DrawerHandleDetector | None = None,
        knob_detector: StoveKnobDetector | None = None,
        plate_detector: PlateFrontDetector | None = None,
        microwave_detector: MicrowaveDoorDetector | None = None,
        config: GoalControllerConfig | None = None,
    ) -> None:
        self.compiler = compiler or GoalTaskCompiler()
        self.drawer_detector = drawer_detector or DrawerHandleDetector()
        self.knob_detector = knob_detector or StoveKnobDetector()
        self.plate_detector = plate_detector or PlateFrontDetector(self.knob_detector)
        self.microwave_detector = microwave_detector
        self.config = config or GoalControllerConfig()
        self._plan: GoalSkillPlan | None = None
        self._step_index = 0
        self._status = GoalExecutorStatus.IDLE
        self._phase = "idle"
        self._phase_ticks = 0
        self._detection_misses = 0
        self._message = ""
        self._target: ContactTarget | None = None
        self._push: PushTarget | None = None
        self._base_rotation = np.eye(3)
        self._motion_position = np.zeros(3)
        self._motion_rotation = np.eye(3)
        self._initial_feature_point = np.zeros(3)
        self._initial_feature_axis = np.array([1.0, 0.0, 0.0])
        self._force_baseline = np.zeros(3)
        self._previous_error: float | None = None
        self._stall_ticks = 0
        self._manipulation_progress_m = 0.0
        self._manipulation_start_position = np.zeros(3)
        self._drawer_load_proof_progress_m = 0.0
        self._drawer_load_proof_loaded_width_m: float | None = None
        self._drawer_load_proof_settled_width_m: float | None = None
        self._drawer_load_proof_passed = False
        self._drawer_load_proof_failure = ""
        self._drawer_pull_grasp_lost = False
        self._drawer_pull_previous_width_m: float | None = None
        self._drawer_pull_width_plateau_ticks = 0
        self._drawer_pull_force_ticks = 0
        self._drawer_pull_mechanical_stop_observed = False
        self._drawer_rail_stop_visible_axis_bounds_m = np.full(2, np.nan)
        self._drawer_rail_stop_axis_world = np.array([1.0, 0.0, 0.0])
        self._drawer_rail_stop_slide_direction = np.array([1.0, 0.0, 0.0])
        self._drawer_rail_stop_slide_segment_start_position = np.zeros(3)
        self._drawer_rail_stop_slide_segment_distance_m = 0.0
        self._drawer_rail_stop_slide_remaining_m = 0.0
        self._drawer_rail_stop_axis_total_progress_m = 0.0
        self._drawer_rail_stop_axis_segment_progress_m = 0.0
        self._drawer_rail_stop_axis_segment_peak_progress_m = 0.0
        self._drawer_rail_stop_axis_segment_max_regression_m = 0.0
        self._drawer_rail_stop_axis_max_observed_regression_m = 0.0
        self._drawer_rail_stop_axis_partial_progress_m = 0.0
        self._drawer_rail_stop_axis_partial_segment_count = 0
        self._drawer_rail_stop_axis_chain_active = False
        self._drawer_rail_stop_axis_chain_origin = np.zeros(3)
        self._drawer_rail_stop_axis_chain_rotation = np.eye(3)
        self._drawer_rail_stop_axis_chain_direction = np.array([1.0, 0.0, 0.0])
        self._drawer_rail_stop_axis_chain_total_start_m = 0.0
        self._drawer_rail_stop_axis_chain_progress_m = 0.0
        self._drawer_rail_stop_axis_chain_cross_drift_m = 0.0
        self._drawer_rail_stop_axis_chain_rotation_error_rad = 0.0
        self._drawer_rail_stop_axis_chain_local_normal_m = 0.0
        self._drawer_rail_stop_axis_chain_third_axis_drift_m = 0.0
        self._drawer_rail_stop_axis_chain_global_normal_progress_m = 0.0
        self._drawer_rail_stop_axis_chain_global_normal_start_m = 0.0
        self._drawer_rail_stop_axis_chain_last_verified_global_normal_m = 0.0
        self._drawer_rail_stop_axis_chain_pending_baseline_normal_credit_m = 0.0
        self._drawer_rail_stop_axis_chain_baseline_normal_segment_cap_m = 0.0
        self._drawer_rail_stop_axis_chain_normal_credit_evidence: (
            _DrawerRailStopNormalCreditEvidence | None
        ) = None
        self._drawer_rail_stop_probe_origin = np.zeros(3)
        self._drawer_rail_stop_probe_index = 0
        self._drawer_rail_stop_release_width_m = 0.0
        self._drawer_rail_stop_current_public_width_m = 0.0
        self._drawer_rail_stop_body_margin_m = 0.0
        self._drawer_rail_stop_axis_retry_used_m = 0.0
        self._drawer_rail_stop_axis_retry_require_full_distance = False
        self._drawer_rail_stop_required_axis_scalar_m = float("nan")
        self._drawer_rail_stop_outward_probe_origin = np.zeros(3)
        self._drawer_rail_stop_outward_probe_distance_m = 0.0
        self._drawer_rail_stop_outward_probe_progress_m = 0.0
        self._drawer_rail_stop_outward_axis_drift_m = 0.0
        self._drawer_rail_stop_outward_residual_drift_m = 0.0
        self._drawer_rail_stop_outward_net_clearance_m = float("-inf")
        self._drawer_rail_stop_axis_retry_settle_transition_position = np.zeros(3)
        self._drawer_rail_stop_axis_retry_settle_transition_rotation = np.eye(3)
        self._drawer_rail_stop_axis_retry_settle_origin = np.zeros(3)
        self._drawer_rail_stop_axis_retry_settle_start_global_normal_m = 0.0
        self._drawer_rail_stop_axis_retry_settle_previous_global_normal_m = 0.0
        self._drawer_rail_stop_axis_retry_settle_delta_normal_m = 0.0
        self._drawer_rail_stop_axis_retry_settle_local_normal_m = 0.0
        self._drawer_rail_stop_axis_retry_settle_transition_normal_m = 0.0
        self._drawer_rail_stop_axis_retry_settle_rail_drift_m = 0.0
        self._drawer_rail_stop_axis_retry_settle_third_axis_drift_m = 0.0
        self._drawer_rail_stop_axis_retry_settle_rotation_error_rad = 0.0
        self._drawer_rail_stop_axis_retry_settle_stable_count = 0
        self._drawer_rail_stop_axis_retry_settle_anchor_frozen = False
        self._drawer_rail_stop_axis_retry_settle_requested_distance_m = 0.0
        self._drawer_rail_stop_axis_retry_settle_probe_axis_drift_m = 0.0
        self._drawer_rail_stop_axis_retry_settle_required_axis_correction_m = 0.0
        self._drawer_rail_stop_axis_retry_settle_entry_net_clearance_m = 0.0
        self._drawer_rail_stop_axis_retry_settle_entry_baseline_credit_cap_m = 0.0
        self._drawer_rail_stop_axis_retry_settle_require_full_distance = False
        self._drawer_rail_stop_axis_retry_settle_message = ""
        self._drawer_rail_stop_axis_retry_settle_failure_message = ""
        self._drawer_rail_stop_normal_retreat_progress_m = 0.0
        self._drawer_rail_stop_normal_segment_origin = np.zeros(3)
        self._drawer_rail_stop_normal_segment_distance_m = 0.0
        self._drawer_rail_stop_normal_segment_progress_m = 0.0
        self._drawer_rail_stop_normal_segment_cross_drift_m = 0.0
        self._drawer_rail_stop_normal_chain_active = False
        self._drawer_rail_stop_normal_chain_origin = np.zeros(3)
        self._drawer_rail_stop_normal_chain_rotation = np.eye(3)
        self._drawer_rail_stop_normal_chain_direction = np.array([0.0, 1.0, 0.0])
        self._drawer_rail_stop_normal_chain_axis = np.array([1.0, 0.0, 0.0])
        self._drawer_rail_stop_normal_chain_progress_m = 0.0
        self._drawer_rail_stop_normal_chain_cross_drift_m = 0.0
        self._drawer_rail_stop_normal_chain_rotation_error_rad = 0.0
        self._drawer_rail_stop_normal_credit_source = "none"
        self._drawer_rail_stop_normal_credit_increment_m = 0.0
        self._push_contact_position = np.zeros(3)
        self._drawer_precontact_position = np.zeros(3)
        self._drawer_close_contact_position = np.zeros(3)
        self._drawer_pull_direction = np.array([0.0, 1.0, 0.0])
        self._drawer_command_distance_m = 0.0
        self._drawer_closed_target_position = np.zeros(3)
        self._drawer_close_push_count = 0
        self._drawer_attempt = 0
        self._drawer_lateral_offset = np.zeros(3)
        self._drawer_anchor: DrawerEpisodeAnchor | None = None
        self._drawer_close_has_prior_anchor = False
        self._drawer_wrist_recovery_used = False
        self._drawer_free_space_best_error_m = float("inf")
        self._drawer_free_space_stall_ticks = 0
        self._drawer_wrist_recovery_return_position = np.zeros(3)
        self._drawer_wrist_recovery_rotation = np.eye(3)
        self._microwave_initial_point = np.zeros(3)
        self._microwave_direction = np.zeros(3)
        self._microwave_precontact_position = np.zeros(3)
        self._microwave_outer_position = np.zeros(3)
        self._microwave_contact_rotation = np.eye(3)
        self._microwave_use_arc = False
        self._microwave_hinge_position = np.zeros(3)
        self._microwave_rotation_axis = np.array([0.0, 0.0, 1.0])
        self._microwave_arc_angle_rad = 0.0
        self._microwave_arc_start_position = np.zeros(3)
        self._microwave_arc_start_rotation = np.eye(3)
        self._microwave_arc_chunk_angle_rad = 0.0
        self._microwave_arc_completed_angle_rad = 0.0
        self._microwave_arc_rebases = 0
        self._microwave_open_regrasps = 0
        self._microwave_open_segment_start_angle_rad = 0.0
        self._microwave_open_segment_start_rotation = np.eye(3)
        self._microwave_open_regrasp_anchor = np.zeros(3)
        self._microwave_open_regrasp_outward = np.array((0.0, 1.0, 0.0))
        self._microwave_initial_outward = np.array((0.0, 1.0, 0.0))
        self._microwave_open_best_angle_rad = 0.0
        self._microwave_open_angle_stall_ticks = 0
        self._microwave_open_release_escape_active = False
        self._microwave_open_push_mode = False
        self._microwave_open_push_tangent = np.array((0.0, 1.0, 0.0))
        self._microwave_open_push_contact_point = np.zeros(3)
        self._microwave_open_push_precontact_position = np.zeros(3)
        self._microwave_open_push_segment_start_angle_rad = 0.0
        self._microwave_open_push_segments_completed = 0
        self._microwave_open_push_recompact_ticks = 0
        self._microwave_open_push_compact_recoveries = 0
        self._microwave_open_push_recompact_pending = False
        self._microwave_open_push_last_compact_angle_rad = 0.0
        self._microwave_open_push_compact_loss_baseline_angle_rad = 0.0
        self._microwave_open_push_progress_samples: list[
            tuple[int, float]
        ] = []
        self._microwave_open_push_finalizing = False
        self._microwave_arc_segments = 0
        self._microwave_arc_index = 0
        self._microwave_previous_width_m: float | None = None
        self._microwave_width_plateau_ticks = 0
        self._microwave_previous_force_n: np.ndarray | None = None
        self._microwave_force_plateau_ticks = 0
        self._microwave_mechanical_stop_observed = False
        self._microwave_close_release_pose_frozen = False
        self._microwave_release_position = np.zeros(3)
        self._microwave_release_rotation = np.eye(3)
        self._microwave_close_release_outward = np.array((0.0, 1.0, 0.0))
        self._microwave_close_release_unseat_distance_m = 0.0
        self._microwave_close_retreat_outward = np.array((0.0, 1.0, 0.0))
        self._microwave_closed_slot_position = np.zeros(3)
        self._microwave_close_target_angle_rad = 0.0
        self._microwave_close_push_mode = False
        self._microwave_close_push_release_anchor = np.zeros(3)
        self._microwave_close_push_radial = np.array((1.0, 0.0, 0.0))
        self._microwave_close_push_outward = np.array((0.0, 1.0, 0.0))
        self._microwave_close_push_contact_point = np.zeros(3)
        self._microwave_close_push_precontact_position = np.zeros(3)
        self._microwave_close_push_radius_m = 0.0
        self._microwave_close_push_best_angle_rad = 0.0
        self._microwave_close_push_start_angle_rad = 0.0
        self._microwave_close_push_terminal_anchor = np.zeros(3)
        self._microwave_close_push_terminal_start_angle_rad = 0.0
        self._microwave_close_push_terminal_face_normal = np.array(
            (0.0, -1.0, 0.0)
        )
        self._microwave_close_push_exit_anchor = np.zeros(3)
        self._microwave_close_push_mechanical_stop = False
        self._knob_turn_target_rotation = np.eye(3)
        self._knob_rotation_progress_rad = 0.0
        self._push_attempt = 0
        self._staged_destination_position = np.zeros(3)

    @property
    def status(self) -> GoalExecutorStatus:
        return self._status

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def drawer_anchor(self) -> DrawerEpisodeAnchor | None:
        """Return the sensor-only handle identity captured in this episode."""

        return self._drawer_anchor

    def seed_drawer_anchor(self, anchor: DrawerEpisodeAnchor) -> None:
        """Import an anchor when consecutive sensor-only phases use new policies."""

        if self._status is not GoalExecutorStatus.RUNNING or self._phase != "detect":
            raise RuntimeError("drawer anchor must be seeded immediately after reset")
        assert self._plan is not None
        step = self._plan.steps[self._step_index]
        if step.kind not in {
            GoalSkillKind.OPEN_DRAWER,
            GoalSkillKind.CLOSE_DRAWER,
        }:
            raise ValueError("current skill does not manipulate a drawer")
        if step.level != anchor.level:
            raise ValueError("drawer anchor level does not match the current skill")
        self._drawer_anchor = anchor

    def clear_episode_state(self) -> None:
        """Clear all episode-local contact state without compiling a task."""

        self._plan = None
        self._step_index = 0
        self._status = GoalExecutorStatus.IDLE
        self._phase = "idle"
        self._phase_ticks = 0
        self._detection_misses = 0
        self._message = ""
        self._target = None
        self._push = None
        self._base_rotation = np.eye(3)
        self._motion_position = np.zeros(3)
        self._motion_rotation = np.eye(3)
        self._initial_feature_point = np.zeros(3)
        self._initial_feature_axis = np.array([1.0, 0.0, 0.0])
        self._force_baseline = np.zeros(3)
        self._previous_error = None
        self._stall_ticks = 0
        self._manipulation_progress_m = 0.0
        self._manipulation_start_position = np.zeros(3)
        self._drawer_load_proof_progress_m = 0.0
        self._drawer_load_proof_loaded_width_m = None
        self._drawer_load_proof_settled_width_m = None
        self._drawer_load_proof_passed = False
        self._drawer_load_proof_failure = ""
        self._drawer_pull_grasp_lost = False
        self._drawer_pull_previous_width_m = None
        self._drawer_pull_width_plateau_ticks = 0
        self._drawer_pull_force_ticks = 0
        self._drawer_pull_mechanical_stop_observed = False
        self._drawer_rail_stop_visible_axis_bounds_m = np.full(2, np.nan)
        self._drawer_rail_stop_axis_world = np.array([1.0, 0.0, 0.0])
        self._drawer_rail_stop_slide_direction = np.array([1.0, 0.0, 0.0])
        self._drawer_rail_stop_slide_segment_start_position = np.zeros(3)
        self._drawer_rail_stop_slide_segment_distance_m = 0.0
        self._drawer_rail_stop_slide_remaining_m = 0.0
        self._drawer_rail_stop_axis_total_progress_m = 0.0
        self._drawer_rail_stop_axis_segment_progress_m = 0.0
        self._drawer_rail_stop_axis_segment_peak_progress_m = 0.0
        self._drawer_rail_stop_axis_segment_max_regression_m = 0.0
        self._drawer_rail_stop_axis_max_observed_regression_m = 0.0
        self._drawer_rail_stop_axis_partial_progress_m = 0.0
        self._drawer_rail_stop_axis_partial_segment_count = 0
        self._drawer_rail_stop_axis_chain_active = False
        self._drawer_rail_stop_axis_chain_origin = np.zeros(3)
        self._drawer_rail_stop_axis_chain_rotation = np.eye(3)
        self._drawer_rail_stop_axis_chain_direction = np.array([1.0, 0.0, 0.0])
        self._drawer_rail_stop_axis_chain_total_start_m = 0.0
        self._drawer_rail_stop_axis_chain_progress_m = 0.0
        self._drawer_rail_stop_axis_chain_cross_drift_m = 0.0
        self._drawer_rail_stop_axis_chain_rotation_error_rad = 0.0
        self._drawer_rail_stop_axis_chain_local_normal_m = 0.0
        self._drawer_rail_stop_axis_chain_third_axis_drift_m = 0.0
        self._drawer_rail_stop_axis_chain_global_normal_progress_m = 0.0
        self._drawer_rail_stop_axis_chain_global_normal_start_m = 0.0
        self._drawer_rail_stop_axis_chain_last_verified_global_normal_m = 0.0
        self._drawer_rail_stop_axis_chain_pending_baseline_normal_credit_m = 0.0
        self._drawer_rail_stop_axis_chain_baseline_normal_segment_cap_m = 0.0
        self._drawer_rail_stop_axis_chain_normal_credit_evidence = None
        self._drawer_rail_stop_probe_origin = np.zeros(3)
        self._drawer_rail_stop_probe_index = 0
        self._drawer_rail_stop_release_width_m = 0.0
        self._drawer_rail_stop_current_public_width_m = 0.0
        self._drawer_rail_stop_body_margin_m = 0.0
        self._drawer_rail_stop_axis_retry_used_m = 0.0
        self._drawer_rail_stop_axis_retry_require_full_distance = False
        self._drawer_rail_stop_required_axis_scalar_m = float("nan")
        self._drawer_rail_stop_outward_probe_origin = np.zeros(3)
        self._drawer_rail_stop_outward_probe_distance_m = 0.0
        self._drawer_rail_stop_outward_probe_progress_m = 0.0
        self._drawer_rail_stop_outward_axis_drift_m = 0.0
        self._drawer_rail_stop_outward_residual_drift_m = 0.0
        self._drawer_rail_stop_outward_net_clearance_m = float("-inf")
        self._drawer_rail_stop_axis_retry_settle_transition_position = np.zeros(3)
        self._drawer_rail_stop_axis_retry_settle_transition_rotation = np.eye(3)
        self._drawer_rail_stop_axis_retry_settle_origin = np.zeros(3)
        self._drawer_rail_stop_axis_retry_settle_start_global_normal_m = 0.0
        self._drawer_rail_stop_axis_retry_settle_previous_global_normal_m = 0.0
        self._drawer_rail_stop_axis_retry_settle_delta_normal_m = 0.0
        self._drawer_rail_stop_axis_retry_settle_local_normal_m = 0.0
        self._drawer_rail_stop_axis_retry_settle_transition_normal_m = 0.0
        self._drawer_rail_stop_axis_retry_settle_rail_drift_m = 0.0
        self._drawer_rail_stop_axis_retry_settle_third_axis_drift_m = 0.0
        self._drawer_rail_stop_axis_retry_settle_rotation_error_rad = 0.0
        self._drawer_rail_stop_axis_retry_settle_stable_count = 0
        self._drawer_rail_stop_axis_retry_settle_anchor_frozen = False
        self._drawer_rail_stop_axis_retry_settle_requested_distance_m = 0.0
        self._drawer_rail_stop_axis_retry_settle_probe_axis_drift_m = 0.0
        self._drawer_rail_stop_axis_retry_settle_required_axis_correction_m = 0.0
        self._drawer_rail_stop_axis_retry_settle_entry_net_clearance_m = 0.0
        self._drawer_rail_stop_axis_retry_settle_entry_baseline_credit_cap_m = 0.0
        self._drawer_rail_stop_axis_retry_settle_require_full_distance = False
        self._drawer_rail_stop_axis_retry_settle_message = ""
        self._drawer_rail_stop_axis_retry_settle_failure_message = ""
        self._drawer_rail_stop_normal_retreat_progress_m = 0.0
        self._drawer_rail_stop_normal_segment_origin = np.zeros(3)
        self._drawer_rail_stop_normal_segment_distance_m = 0.0
        self._drawer_rail_stop_normal_segment_progress_m = 0.0
        self._drawer_rail_stop_normal_segment_cross_drift_m = 0.0
        self._drawer_rail_stop_normal_chain_active = False
        self._drawer_rail_stop_normal_chain_origin = np.zeros(3)
        self._drawer_rail_stop_normal_chain_rotation = np.eye(3)
        self._drawer_rail_stop_normal_chain_direction = np.array([0.0, 1.0, 0.0])
        self._drawer_rail_stop_normal_chain_axis = np.array([1.0, 0.0, 0.0])
        self._drawer_rail_stop_normal_chain_progress_m = 0.0
        self._drawer_rail_stop_normal_chain_cross_drift_m = 0.0
        self._drawer_rail_stop_normal_chain_rotation_error_rad = 0.0
        self._drawer_rail_stop_normal_credit_source = "none"
        self._drawer_rail_stop_normal_credit_increment_m = 0.0
        self._drawer_precontact_position = np.zeros(3)
        self._drawer_close_contact_position = np.zeros(3)
        self._drawer_pull_direction = np.array([0.0, 1.0, 0.0])
        self._drawer_command_distance_m = 0.0
        self._drawer_closed_target_position = np.zeros(3)
        self._drawer_close_push_count = 0
        self._drawer_attempt = 0
        self._drawer_lateral_offset = np.zeros(3)
        self._drawer_anchor = None
        self._drawer_close_has_prior_anchor = False
        self._drawer_wrist_recovery_used = False
        self._drawer_free_space_best_error_m = float("inf")
        self._drawer_free_space_stall_ticks = 0
        self._drawer_wrist_recovery_return_position = np.zeros(3)
        self._drawer_wrist_recovery_rotation = np.eye(3)
        self._push_contact_position = np.zeros(3)
        self._microwave_initial_point = np.zeros(3)
        self._microwave_direction = np.zeros(3)
        self._microwave_precontact_position = np.zeros(3)
        self._microwave_outer_position = np.zeros(3)
        self._microwave_contact_rotation = np.eye(3)
        self._microwave_use_arc = False
        self._microwave_hinge_position = np.zeros(3)
        self._microwave_rotation_axis = np.array([0.0, 0.0, 1.0])
        self._microwave_arc_angle_rad = 0.0
        self._microwave_arc_start_position = np.zeros(3)
        self._microwave_arc_start_rotation = np.eye(3)
        self._microwave_arc_chunk_angle_rad = 0.0
        self._microwave_arc_completed_angle_rad = 0.0
        self._microwave_arc_rebases = 0
        self._microwave_open_regrasps = 0
        self._microwave_open_segment_start_angle_rad = 0.0
        self._microwave_open_segment_start_rotation = np.eye(3)
        self._microwave_open_regrasp_anchor = np.zeros(3)
        self._microwave_open_regrasp_outward = np.array((0.0, 1.0, 0.0))
        self._microwave_initial_outward = np.array((0.0, 1.0, 0.0))
        self._microwave_open_best_angle_rad = 0.0
        self._microwave_open_angle_stall_ticks = 0
        self._microwave_open_release_escape_active = False
        self._microwave_open_push_mode = False
        self._microwave_open_push_tangent = np.array((0.0, 1.0, 0.0))
        self._microwave_open_push_contact_point = np.zeros(3)
        self._microwave_open_push_precontact_position = np.zeros(3)
        self._microwave_open_push_segment_start_angle_rad = 0.0
        self._microwave_open_push_segments_completed = 0
        self._microwave_open_push_recompact_ticks = 0
        self._microwave_open_push_compact_recoveries = 0
        self._microwave_open_push_recompact_pending = False
        self._microwave_open_push_last_compact_angle_rad = 0.0
        self._microwave_open_push_compact_loss_baseline_angle_rad = 0.0
        self._microwave_open_push_progress_samples = []
        self._microwave_open_push_finalizing = False
        self._microwave_arc_segments = 0
        self._microwave_arc_index = 0
        self._microwave_previous_width_m = None
        self._microwave_width_plateau_ticks = 0
        self._microwave_previous_force_n = None
        self._microwave_force_plateau_ticks = 0
        self._microwave_mechanical_stop_observed = False
        self._microwave_close_release_pose_frozen = False
        self._microwave_release_position = np.zeros(3)
        self._microwave_release_rotation = np.eye(3)
        self._microwave_close_release_outward = np.array((0.0, 1.0, 0.0))
        self._microwave_close_release_unseat_distance_m = 0.0
        self._microwave_close_retreat_outward = np.array((0.0, 1.0, 0.0))
        self._microwave_closed_slot_position = np.zeros(3)
        self._microwave_close_target_angle_rad = 0.0
        self._microwave_close_push_mode = False
        self._microwave_close_push_release_anchor = np.zeros(3)
        self._microwave_close_push_radial = np.array((1.0, 0.0, 0.0))
        self._microwave_close_push_outward = np.array((0.0, 1.0, 0.0))
        self._microwave_close_push_contact_point = np.zeros(3)
        self._microwave_close_push_precontact_position = np.zeros(3)
        self._microwave_close_push_radius_m = 0.0
        self._microwave_close_push_best_angle_rad = 0.0
        self._microwave_close_push_start_angle_rad = 0.0
        self._microwave_close_push_terminal_anchor = np.zeros(3)
        self._microwave_close_push_terminal_start_angle_rad = 0.0
        self._microwave_close_push_terminal_face_normal = np.array(
            (0.0, -1.0, 0.0)
        )
        self._microwave_close_push_exit_anchor = np.zeros(3)
        self._microwave_close_push_mechanical_stop = False
        self._knob_turn_target_rotation = np.eye(3)
        self._knob_rotation_progress_rad = 0.0
        self._push_attempt = 0
        self._staged_destination_position = np.zeros(3)

        if self.microwave_detector is not None:
            reset_microwave = getattr(type(self.microwave_detector), "reset", None)
            if callable(reset_microwave):
                reset_microwave(self.microwave_detector)

    def activate_after_episode_clear(self, task: PolicyTask) -> None:
        """Activate a real contact task after one explicit neutral clear."""

        if (
            self._plan is not None
            or self._status is not GoalExecutorStatus.IDLE
            or self._phase != "idle"
            or self._phase_ticks != 0
        ):
            raise RuntimeError(
                "contact-policy activation requires freshly cleared state"
            )
        self._plan = type(self.compiler).compile(
            self.compiler,
            task.instruction,
        )
        self._status = GoalExecutorStatus.RUNNING
        self._set_phase("detect")

    def reset(self, task: PolicyTask) -> None:
        """Clear once, then activate the supplied real contact task."""

        self.clear_episode_state()
        self.activate_after_episode_clear(task)

    def act(self, observation: RobotObservation) -> PolicyDecision:
        # These fields describe credit issued by this one public observation.
        # Reset once at the outer policy boundary so nested phase transitions in
        # the same tick cannot erase a credit event from the trace.
        self._drawer_rail_stop_normal_credit_source = "none"
        self._drawer_rail_stop_normal_credit_increment_m = 0.0
        self._drawer_rail_stop_axis_chain_normal_credit_evidence = None
        if self._plan is None or self._status is GoalExecutorStatus.IDLE:
            return self._decision(OSCAction.hold(-1.0), "policy has not been reset", stop=True)
        if self._status in {
            GoalExecutorStatus.SUCCEEDED,
            GoalExecutorStatus.FAILED,
            GoalExecutorStatus.HANDOFF,
        }:
            return self._decision(OSCAction.hold(-1.0), self._message, stop=True)
        step = self._plan.steps[self._step_index]
        phase_timeout_ticks = (
            self.config.microwave_arc_timeout_ticks
            if self._phase == "microwave_move_door" and self._microwave_use_arc
            else self.config.phase_timeout_ticks
        )
        if self._phase_ticks > phase_timeout_ticks:
            return self._handle_phase_timeout(step, observation)
        if step.kind is GoalSkillKind.ROUTE_B_PLACE_IN:
            self._status = GoalExecutorStatus.HANDOFF
            self._phase = "route_b_handoff"
            self._message = f"handoff to Route B: put {step.subject} in {step.target}"
            return self._decision(OSCAction.hold(-1.0), self._message, stop=True)
        if step.kind is GoalSkillKind.OPEN_DRAWER:
            return self._act_drawer(step, observation)
        if step.kind is GoalSkillKind.CLOSE_DRAWER:
            return self._act_close_drawer(step, observation)
        if step.kind is GoalSkillKind.TURN_KNOB:
            return self._act_knob(step, observation)
        if step.kind is GoalSkillKind.PUSH_OBJECT:
            return self._act_push(observation)
        if step.kind in {
            GoalSkillKind.OPEN_MICROWAVE,
            GoalSkillKind.CLOSE_MICROWAVE,
        }:
            return self._act_microwave(step, observation)
        return self._fail(f"unsupported skill {step.kind.value}")

    def _act_drawer(self, step: GoalSkillStep, observation: RobotObservation) -> PolicyDecision:
        if self._phase == "detect":
            try:
                assert step.level is not None
                reference = self._drawer_reference(step.level)
                if reference is None:
                    self._target = self.drawer_detector.detect(observation, step.level)
                    self._drawer_anchor = DrawerEpisodeAnchor(step.level, self._target)
                else:
                    self._target = self.drawer_detector.track(
                        observation,
                        reference,
                        step.level,
                    )
            except (LookupError, ValueError) as exc:
                return self._detection_miss(str(exc), gripper=-1.0)
            self._capture_target(observation)
            visible_axis_bounds = self._observe_drawer_handle_axis_bounds(
                observation,
                self._target,
            )
            self._drawer_rail_stop_visible_axis_bounds_m = (
                np.full(2, np.nan)
                if visible_axis_bounds is None
                else visible_axis_bounds
            )
            drawer_point = self._target.point_world.copy()
            drawer_point[2] += self.config.drawer_grasp_z_offset_m
            # The upper handles rule out a straight top-down path to the
            # middle / lower levels, while a low head-on path sweeps the palm
            # through objects in front of the cabinet.  Approach around that
            # clutter: first move high to the side of the handle, descend at
            # the side, translate beside the drawer front, then make only the
            # final short head-on contact motion.
            local_z = -self._target.outward_world
            local_y = np.array([0.0, 0.0, 1.0])
            local_x = np.cross(local_y, local_z)
            local_x /= np.linalg.norm(local_x)
            self._motion_rotation = np.column_stack((local_x, local_y, local_z))
            self._drawer_pull_direction = self._axis_aligned_fixture_direction(
                self._target.outward_world
            )
            precontact = (
                drawer_point
                + self._target.outward_world * self.config.drawer_precontact_clearance_m
            )
            lateral = self._select_drawer_lateral(observation, precontact, drawer_point)
            self._drawer_lateral_offset = lateral.copy()
            self._drawer_precontact_position = precontact.copy()
            self._motion_position = precontact + lateral
            self._motion_position[2] += self.config.drawer_safe_height_m
            self._set_phase("move_safe")

        assert self._target is not None
        if self._phase == "move_safe":
            if self._position_reached(
                observation,
                self._motion_position,
                tolerance_m=self.config.drawer_waypoint_tolerance_m,
            ):
                self._set_phase("preshape")
            else:
                return self._move(observation, -1.0)
        if self._phase == "preshape":
            width_error = abs(
                observation.proprio.gripper_width_m - self.config.drawer_preshape_width_m
            )
            if width_error <= self.config.drawer_preshape_hysteresis_m:
                destination = self._motion_position.copy()
                destination[2] -= self.config.drawer_safe_height_m
                self._start_staged_translation(
                    observation,
                    destination,
                    "descend_lateral",
                )
            else:
                return self._move(
                    observation,
                    self._drawer_preshape_gripper(observation),
                )
        if self._phase == "descend_lateral":
            if self._staged_translation_reached(
                observation,
                tolerance_m=self.config.drawer_descent_tolerance_m,
            ):
                self._motion_position = self._drawer_precontact_position.copy()
                self._drawer_free_space_best_error_m = float("inf")
                self._drawer_free_space_stall_ticks = 0
                self._set_phase("move_precontact")
            else:
                return self._move(observation, self._drawer_preshape_gripper(observation))
        if self._phase == "move_precontact":
            if self._position_reached(
                observation,
                self._motion_position,
                tolerance_m=self.config.drawer_waypoint_tolerance_m,
            ):
                self._motion_position = self._target.point_world.copy()
                self._motion_position[2] += self.config.drawer_grasp_z_offset_m
                self._set_phase("approach_contact")
            else:
                if self._drawer_free_space_wrist_stalled(observation):
                    return self._start_drawer_wrist_recovery(observation)
                return self._move(observation, self._drawer_preshape_gripper(observation))
        if self._phase == "drawer_wrist_recovery_lift":
            if self._position_reached(
                observation,
                self._motion_position,
                tolerance_m=self.config.drawer_safe_waypoint_tolerance_m,
            ):
                self._motion_position = (
                    observation.proprio.ee_position_world.copy()
                )
                self._motion_rotation = (
                    self._drawer_wrist_recovery_rotation.copy()
                )
                self._set_phase("drawer_wrist_recovery_rotate")
            else:
                return self._move(
                    observation,
                    self._drawer_preshape_gripper(observation),
                    "lifting clear before an equivalent drawer wrist reorientation",
                )
        if self._phase == "drawer_wrist_recovery_rotate":
            rotation_aligned = (
                self._rotation_error(
                    observation.proprio.T_world_ee[:3, :3],
                    self._motion_rotation,
                )
                <= self.config.microwave_safe_rotation_tolerance_rad
            )
            if rotation_aligned and self._position_reached(
                observation,
                self._motion_position,
                tolerance_m=self.config.drawer_safe_waypoint_tolerance_m,
            ):
                self._start_staged_translation(
                    observation,
                    self._drawer_wrist_recovery_return_position,
                    "drawer_wrist_recovery_return",
                )
            else:
                return self._move(
                    observation,
                    self._drawer_preshape_gripper(observation),
                    "using the pi-yaw-equivalent two-pad frame at safe height",
                )
        if self._phase == "drawer_wrist_recovery_return":
            if self._staged_translation_reached(
                observation,
                tolerance_m=self.config.drawer_waypoint_tolerance_m,
            ):
                self._motion_position = self._target.point_world.copy()
                self._motion_position[2] += self.config.drawer_grasp_z_offset_m
                self._set_phase("approach_contact")
            else:
                return self._move(
                    observation,
                    self._drawer_preshape_gripper(observation),
                    "resuming the frozen RGB-D drawer target after wrist recovery",
                )
        if self._phase == "approach_contact":
            if self._position_reached(observation, self._motion_position) or self._contact_reached(
                observation, tolerance_m=self.config.drawer_contact_tolerance_m
            ):
                self._set_phase("engage")
            else:
                return self._move(observation, self._drawer_preshape_gripper(observation))
        if self._phase == "engage":
            if self._phase_ticks >= self.config.engage_ticks:
                width = observation.proprio.gripper_width_m
                if not (
                    self.config.drawer_blocked_min_width_m
                    <= width
                    <= self.config.drawer_blocked_max_width_m
                ):
                    return self._fail(
                        "drawer handle pinch was not confirmed by measured gripper width"
                    )
                self._manipulation_start_position = observation.proprio.ee_position_world.copy()
                self._motion_position = (
                    self._manipulation_start_position
                    + self._drawer_pull_direction
                    * self.config.drawer_load_proof_distance_m
                )
                self._force_baseline = observation.proprio.ee_force_sensor.copy()
                self._drawer_load_proof_progress_m = 0.0
                self._drawer_load_proof_loaded_width_m = None
                self._drawer_load_proof_settled_width_m = None
                self._drawer_load_proof_passed = False
                self._drawer_load_proof_failure = ""
                self._drawer_pull_grasp_lost = False
                self._drawer_pull_previous_width_m = None
                self._drawer_pull_width_plateau_ticks = 0
                self._drawer_pull_force_ticks = 0
                self._drawer_pull_mechanical_stop_observed = False
                self._set_phase("drawer_load_proof")
            else:
                return self._move(observation, 1.0)
        if self._phase == "drawer_load_proof":
            current = observation.proprio.ee_position_world
            progress = float(
                np.dot(
                    current - self._manipulation_start_position,
                    self._drawer_pull_direction,
                )
            )
            self._drawer_load_proof_progress_m = max(
                self._drawer_load_proof_progress_m,
                progress,
            )
            self._manipulation_progress_m = max(
                self._manipulation_progress_m,
                progress,
            )
            error = float(np.linalg.norm(self._motion_position - current))
            if (
                self._previous_error is None
                or self._previous_error - error > self.config.progress_epsilon_m
            ):
                self._stall_ticks = 0
            else:
                self._stall_ticks += 1
            self._previous_error = error
            width = observation.proprio.gripper_width_m
            if not (
                self.config.drawer_blocked_min_width_m
                <= width
                <= self.config.drawer_blocked_max_width_m
            ):
                return self._release_rejected_drawer_grasp(
                    "drawer load proof lost its measured blocked width "
                    f"({width:.4f} m)"
                )
            force_rise = (
                float(np.linalg.norm(observation.proprio.ee_force_sensor))
                - float(np.linalg.norm(self._force_baseline))
            )
            minimum_progress = (
                self.config.drawer_load_proof_distance_m
                - self.config.drawer_load_proof_position_tolerance_m
            )
            loaded_pose_reached = bool(
                self._phase_ticks >= self.config.drawer_load_proof_min_ticks
                and progress >= minimum_progress
                and error <= self.config.drawer_load_proof_position_tolerance_m
            )
            if loaded_pose_reached:
                self._drawer_load_proof_loaded_width_m = width
                self._set_phase("drawer_load_settle")
            elif (
                self._phase_ticks >= self.config.drawer_load_proof_max_ticks
                or (
                    self._stall_ticks >= self.config.contact_stall_ticks
                    and force_rise >= self.config.contact_force_delta_n
                    and progress < minimum_progress
                )
            ):
                return self._release_rejected_drawer_grasp(
                    "drawer handle did not follow the public TCP through the "
                    f"{self.config.drawer_load_proof_distance_m:.3f}-m load proof "
                    f"(measured {self._drawer_load_proof_progress_m:.4f} m)"
                )
            else:
                return self._move(
                    observation,
                    1.0,
                    "testing the drawer pinch with a bounded loaded pull",
                )
        if self._phase == "drawer_load_settle":
            if self._phase_ticks >= self.config.drawer_load_proof_settle_ticks:
                assert self._drawer_load_proof_loaded_width_m is not None
                settled_width = observation.proprio.gripper_width_m
                self._drawer_load_proof_settled_width_m = settled_width
                settle_change = abs(
                    settled_width - self._drawer_load_proof_loaded_width_m
                )
                position_error = float(
                    np.linalg.norm(
                        self._motion_position
                        - observation.proprio.ee_position_world
                    )
                )
                rotation_error = self._rotation_error(
                    observation.proprio.T_world_ee[:3, :3],
                    self._motion_rotation,
                )
                if not (
                    self.config.drawer_load_proof_min_width_m
                    <= settled_width
                    <= self.config.drawer_blocked_max_width_m
                    and settle_change
                    <= self.config.drawer_load_proof_max_settle_change_m
                    and position_error
                    <= self.config.drawer_load_proof_position_tolerance_m
                    and rotation_error <= self.config.rotation_tolerance_rad
                ):
                    return self._release_rejected_drawer_grasp(
                        "drawer load proof did not retain a settled two-pad pinch "
                        f"({self._drawer_load_proof_loaded_width_m:.4f}->"
                        f"{settled_width:.4f} m; pose residual "
                        f"{position_error:.4f} m/{rotation_error:.4f} rad)"
                    )
                self._start_drawer_full_pull(observation)
            else:
                return self._tick(
                    OSCAction.hold(1.0),
                    "settling the measured two-pad drawer pinch",
                )
        if self._phase == "pull":
            current = observation.proprio.ee_position_world
            width = observation.proprio.gripper_width_m
            if not (
                self.config.drawer_load_proof_min_width_m
                <= width
                <= self.config.drawer_blocked_max_width_m
            ):
                self._drawer_pull_grasp_lost = True
                return self._release_rejected_drawer_grasp(
                    "load-proven drawer pinch was lost during the full pull "
                    f"({width:.4f} m)"
                )
            self._manipulation_progress_m = max(
                self._manipulation_progress_m,
                float(
                    np.dot(
                        current - self._manipulation_start_position,
                        self._drawer_pull_direction,
                    )
                ),
            )
            ordinary_pull_complete = self._position_reached(
                observation,
                self._motion_position,
            ) or (
                self._manipulation_progress_m >= self.config.drawer_pull_distance_m - 0.018
            )
            if ordinary_pull_complete:
                self._set_phase("release")
            elif self._drawer_pull_mechanical_stop(observation):
                # This is permission to release a handle at its rail stop,
                # not semantic task completion.  ``verify_visual`` below
                # therefore requires a fresh RGB-D displacement and cannot
                # use its ordinary occlusion fallback for this path.
                self._drawer_pull_mechanical_stop_observed = True
                self._set_phase("release")
            else:
                return self._move(observation, 1.0)
        if self._phase == "release":
            if (
                self._phase_ticks >= self.config.release_ticks
                and observation.proprio.gripper_width_m >= self.config.drawer_release_width_m
            ):
                self._motion_rotation = (
                    observation.proprio.T_world_ee[:3, :3].copy()
                )
                self._motion_position = observation.proprio.ee_position_world.copy()
                if self._drawer_pull_mechanical_stop_observed:
                    # The horizontal handle can remain between the now-open
                    # top/bottom fingers.  Clear it along its frozen public
                    # RGB-D long axis before any normal retreat or lift.
                    return self._start_drawer_rail_stop_axis_clearance(
                        observation
                    )
                else:
                    self._motion_position += (
                        self._drawer_pull_direction
                        * self.config.drawer_retreat_clearance_m
                    )
                    self._set_phase("retreat_outward")
            else:
                return self._tick(OSCAction.hold(-1.0))
        if self._phase == "drawer_rail_stop_axis_probe":
            return self._act_drawer_rail_stop_axis_probe(observation)
        if self._phase == "drawer_rail_stop_axis_slide":
            return self._act_drawer_rail_stop_axis_slide(observation)
        if self._phase == "drawer_rail_stop_outward_probe":
            return self._act_drawer_rail_stop_outward_probe(observation)
        if self._phase == "drawer_rail_stop_axis_retry_settle":
            return self._act_drawer_rail_stop_axis_retry_settle(observation)
        if self._phase == "retreat_outward":
            if self._drawer_pull_mechanical_stop_observed:
                return self._act_drawer_rail_stop_normal_segment(
                    observation
                )
            if self._position_reached(
                observation,
                self._motion_position,
                tolerance_m=max(
                    self.config.drawer_waypoint_tolerance_m,
                    self.config.contact_retreat_tolerance_m,
                ),
            ):
                self._motion_position[2] += self.config.drawer_safe_height_m
                self._set_phase("retreat_up")
            else:
                return self._move(observation, -1.0)
        if self._phase == "retreat_up":
            if self._position_reached(
                observation,
                self._motion_position,
                tolerance_m=max(
                    self.config.drawer_waypoint_tolerance_m,
                    self.config.contact_retreat_tolerance_m,
                ),
            ):
                self._set_phase("verify_visual")
            else:
                return self._move(observation, -1.0)
        if self._phase == "verify_visual":
            visual_displacement = 0.0
            visual_available = False
            refreshed: ContactTarget | None = None
            try:
                refreshed = self._track_drawer(observation, step.level or "middle")
                visual_available = True
                visual_displacement = float(
                    np.dot(
                        refreshed.point_world - self._initial_feature_point,
                        self._drawer_pull_direction,
                    )
                )
            except (LookupError, ValueError):
                pass
            if visual_displacement >= self.config.drawer_visual_displacement_m:
                return self._complete("drawer opening verified from RGB-D handle displacement")
            # Occlusion fallback is still sensor-only: actual Cartesian progress
            # plus a completed close/open cycle, never commanded progress alone.
            if (
                not visual_available
                and not self._drawer_pull_mechanical_stop_observed
                and not self._drawer_pull_grasp_lost
                and self._manipulation_progress_m >= self.config.drawer_visual_displacement_m
            ):
                return self._complete("drawer opening verified from measured pull displacement")
            if self._drawer_pull_mechanical_stop_observed and not visual_available:
                return self._fail(
                    "drawer rail-stop release lacked fresh RGB-D displacement verification"
                )
            if self._drawer_pull_grasp_lost and not visual_available:
                return self._fail(
                    "lost drawer pinch lacked fresh RGB-D displacement verification"
                )
            if (
                refreshed is not None
                and self._drawer_attempt + 1 < self.config.drawer_max_attempts
            ):
                return self._retry_open_drawer(observation, refreshed)
            return self._fail("drawer did not exhibit sufficient sensor-measured displacement")
        return self._fail(f"invalid drawer phase {self._phase!r}")

    def _act_close_drawer(
        self,
        step: GoalSkillStep,
        observation: RobotObservation,
    ) -> PolicyDecision:
        """Push an observed open drawer inward using only sensor geometry."""

        if self._phase == "detect":
            try:
                assert step.level is not None
                reference = self._drawer_reference(step.level)
                self._drawer_close_has_prior_anchor = reference is not None
                if reference is None:
                    self._target = self.drawer_detector.detect(observation, step.level)
                    self._drawer_anchor = DrawerEpisodeAnchor(step.level, self._target)
                else:
                    self._target = self.drawer_detector.track(
                        observation,
                        reference,
                        step.level,
                    )
            except (LookupError, ValueError) as exc:
                return self._detection_miss(str(exc), gripper=-1.0)
            self._capture_target(observation)
            # Closing pushes the observed front/handle face and does not need
            # the below-handle seating offset used by the retained pinch in
            # OPEN_DRAWER.  Lowering this face contact needlessly consumes the
            # RGB-D clearance above countertop clutter.
            drawer_point = self._target.point_world.copy()
            self._drawer_pull_direction = self._axis_aligned_fixture_direction(
                self._target.outward_world
            )
            # Closing uses the fingers as a compact face pusher rather than a
            # retained horizontal handle pinch.  Keep the reachable downward
            # tool axis, but yaw the jaw-width axis along the observed handle.
            # Both fingertips then meet the same drawer-front plane; leaving
            # the reset yaw can put one finger behind the visible face when
            # the reset jaws happen to span the drawer normal.
            self._motion_rotation = self._drawer_close_pusher_rotation(step.level)

            # The dark cabinet component supplies a sensor-derived fixture
            # center.  Its handle-to-center extension grows as a drawer opens;
            # subtract the static front-face clearance to estimate how far to
            # push, then keep the command inside plausible drawer travel.
            if self._drawer_close_has_prior_anchor:
                assert reference is not None
                estimated_travel = float(
                    np.dot(
                        drawer_point - reference.point_world,
                        self._drawer_pull_direction,
                    )
                )
            else:
                extension = float(
                    np.dot(
                        drawer_point - self._target.fixture_center_world,
                        self._drawer_pull_direction,
                    )
                )
                estimated_travel = extension - self.config.drawer_closed_handle_offset_m
            if estimated_travel <= 0.5 * self.config.drawer_close_min_travel_m:
                return self._complete("drawer already appears closed in RGB-D")
            self._drawer_command_distance_m = float(
                np.clip(
                    estimated_travel,
                    self.config.drawer_close_min_travel_m,
                    self.config.drawer_close_max_travel_m,
                )
            )
            self._drawer_closed_target_position = (
                drawer_point - self._drawer_pull_direction * estimated_travel
            )
            self._drawer_close_push_count = 1
            # Pick a contact along the sensed horizontal handle/front span,
            # rather than unconditionally returning to its centre.  Foreground
            # clutter can block the centre even though a nearby point on the
            # same moving drawer front has a clear RGB-D approach corridor.
            try:
                contact_offset = self._select_drawer_close_contact_offset(
                    observation,
                    drawer_point,
                )
            except (LookupError, ValueError) as exc:
                return self._detection_miss(str(exc), gripper=-1.0)
            self._drawer_close_contact_position = drawer_point + contact_offset
            precontact = (
                self._drawer_close_contact_position
                + self._target.outward_world * self.config.drawer_precontact_clearance_m
            )
            self._drawer_lateral_offset = contact_offset.copy()
            self._drawer_precontact_position = precontact.copy()
            self._motion_position = precontact.copy()
            self._motion_position[2] += self.config.drawer_safe_height_m
            self._set_phase("close_move_safe")

        assert self._target is not None
        if self._phase == "close_move_safe":
            if self._position_reached(
                observation,
                self._motion_position,
                tolerance_m=max(
                    self.config.drawer_close_waypoint_tolerance_m,
                    self.config.drawer_safe_waypoint_tolerance_m,
                ),
            ):
                self._set_phase("close_preshape")
            else:
                return self._move(observation, -1.0)
        if self._phase == "close_preshape":
            width_error = abs(
                observation.proprio.gripper_width_m
                - self.config.drawer_preshape_width_m
            )
            if width_error <= self.config.drawer_preshape_hysteresis_m:
                # Closing is a push rather than a retained pinch.  Freeze the
                # safely reached wrist frame before descending.  Keep a
                # measured narrow preshape during the side descent instead of
                # fully closing the fingers through cabinet geometry.
                self._motion_rotation = (
                    observation.proprio.T_world_ee[:3, :3].copy()
                )
                destination = self._motion_position.copy()
                destination[2] -= self.config.drawer_safe_height_m
                self._start_staged_translation(
                    observation,
                    destination,
                    "close_descend_lateral",
                )
            else:
                return self._move(
                    observation,
                    self._drawer_preshape_gripper(observation),
                )
        if self._phase == "close_descend_lateral":
            if self._staged_translation_reached(
                observation,
                tolerance_m=self.config.drawer_descent_tolerance_m,
                intermediate_tolerance_m=(
                    self.config.drawer_close_descent_stage_tolerance_m
                ),
            ):
                self._motion_position = self._drawer_precontact_position.copy()
                self._set_phase("close_move_precontact")
            else:
                return self._move(
                    observation,
                    self._drawer_preshape_gripper(observation),
                )
        if self._phase == "close_move_precontact":
            if self._position_reached(
                observation,
                self._motion_position,
                tolerance_m=self.config.drawer_close_waypoint_tolerance_m,
            ):
                self._motion_position = self._drawer_close_contact_position.copy()
                self._set_phase("close_approach_contact")
            else:
                return self._move(
                    observation,
                    self._drawer_preshape_gripper(observation),
                )
        if self._phase == "close_approach_contact":
            if self._position_reached(observation, self._motion_position) or self._contact_reached(
                observation, tolerance_m=self.config.drawer_contact_tolerance_m
            ):
                self._set_phase("close_pusher_close")
            else:
                return self._move(
                    observation,
                    self._drawer_preshape_gripper(observation),
                )
        if self._phase == "close_pusher_close":
            # Only after the face has been reached may the fingers close into
            # the compact pusher.  The measured jaw width, not elapsed command
            # time alone, gates the inward motion.
            if (
                observation.proprio.gripper_width_m
                <= self.config.drawer_close_pusher_width_m
            ):
                self._manipulation_start_position = (
                    observation.proprio.ee_position_world.copy()
                )
                # A reacquired push has its own measured origin. Retaining
                # the previous push's progress can instantly satisfy the
                # smaller remaining travel and skip every retry motion.
                self._manipulation_progress_m = 0.0
                self._motion_position = (
                    self._manipulation_start_position
                    - self._drawer_pull_direction * (
                        self._drawer_command_distance_m + self.config.drawer_close_overshoot_m
                    )
                )
                self._set_phase("close_push")
            else:
                return self._move(observation, 1.0)
        if self._phase == "close_push":
            progress = float(
                np.dot(
                    self._manipulation_start_position
                    - observation.proprio.ee_position_world,
                    self._drawer_pull_direction,
                )
            )
            self._manipulation_progress_m = max(self._manipulation_progress_m, progress)
            reached = self._position_reached(observation, self._motion_position) or (
                self._manipulation_progress_m >= (
                    self._drawer_command_distance_m + self.config.drawer_close_overshoot_m - 0.003
                )
            )
            blocked_closed = (
                self._manipulation_progress_m >= self.config.drawer_visual_displacement_m
                and self._contact_reached(observation)
                # Sliding friction raises force throughout an ordinary push.
                # A closed-end stop additionally needs observed stalled motion.
                and self._stall_ticks >= self.config.contact_stall_ticks
            )
            if reached or blocked_closed:
                self._set_phase("close_release")
            else:
                # Servo around the measured compact width.  A neutral command
                # does not brake the physical fingers; after an initial close
                # they otherwise continue to collapse and can hook a bracket.
                return self._move(
                    observation,
                    self._drawer_pusher_gripper(observation),
                )
        if self._phase == "close_release":
            if (
                self._phase_ticks >= self.config.release_ticks
                and observation.proprio.gripper_width_m >= self.config.drawer_release_width_m
            ):
                self._motion_rotation = (
                    observation.proprio.T_world_ee[:3, :3].copy()
                )
                self._motion_position = observation.proprio.ee_position_world.copy()
                self._motion_position += (
                    self._drawer_pull_direction * self.config.drawer_retreat_clearance_m
                )
                self._set_phase("close_retreat_outward")
            else:
                return self._tick(OSCAction.hold(-1.0))
        if self._phase == "close_retreat_outward":
            if self._position_reached(
                observation,
                self._motion_position,
                tolerance_m=max(
                    self.config.drawer_close_waypoint_tolerance_m,
                    self.config.contact_retreat_tolerance_m,
                ),
            ):
                self._motion_position[2] += self.config.drawer_safe_height_m
                self._set_phase("close_retreat_up")
            else:
                return self._move(observation, -1.0)
        if self._phase == "close_retreat_up":
            if self._position_reached(
                observation,
                self._motion_position,
                tolerance_m=max(
                    self.config.drawer_close_waypoint_tolerance_m,
                    self.config.contact_retreat_tolerance_m,
                ),
            ):
                self._set_phase("close_verify_visual")
            else:
                return self._move(observation, -1.0)
        if self._phase == "close_verify_visual":
            visual_displacement = 0.0
            refreshed = None
            try:
                refreshed = self._track_drawer(observation, step.level or "middle")
                visual_displacement = float(
                    np.dot(
                        self._initial_feature_point - refreshed.point_world,
                        self._drawer_pull_direction,
                    )
                )
            except (LookupError, ValueError):
                pass
            if refreshed is not None:
                remaining = float(np.dot(
                    refreshed.point_world - self._drawer_closed_target_position,
                    self._drawer_pull_direction,
                ))
                if remaining <= 0.004:
                    return self._complete("drawer reached its RGB-D closed-handle position")
                if self._drawer_close_push_count < 3 and visual_displacement >= 0.008:
                    # Sliding friction can produce a temporary plateau before
                    # the drawer reaches its closed position. Reacquire the
                    # visible front and push only its measured remaining travel.
                    self._drawer_close_push_count += 1
                    self._target = refreshed
                    self._initial_feature_point = refreshed.point_world.copy()
                    self._drawer_command_distance_m = min(
                        remaining, self.config.drawer_close_max_travel_m
                    )
                    try:
                        offset = self._select_drawer_close_contact_offset(
                            observation, refreshed.point_world
                        )
                    except (LookupError, ValueError) as exc:
                        return self._fail(str(exc))
                    self._drawer_close_contact_position = refreshed.point_world + offset
                    self._drawer_lateral_offset = offset.copy()
                    self._drawer_precontact_position = (
                        self._drawer_close_contact_position
                        + self._drawer_pull_direction * self.config.drawer_precontact_clearance_m
                    )
                    self._motion_position = self._drawer_precontact_position.copy()
                    self._motion_position[2] += self.config.drawer_safe_height_m
                    self._motion_rotation = self._drawer_close_pusher_rotation(step.level)
                    self._set_phase("close_move_safe")
                    return self._move(observation, -1.0)
            return self._fail("drawer did not exhibit sufficient inward sensor-measured displacement")
        return self._fail(f"invalid close-drawer phase {self._phase!r}")

    def _act_knob(
        self,
        step: GoalSkillStep,
        observation: RobotObservation,
    ) -> PolicyDecision:
        if self._phase == "detect":
            try:
                self._target = self.knob_detector.detect(observation)
            except (LookupError, ValueError) as exc:
                return self._detection_miss(str(exc), gripper=-1.0)
            self._capture_target(observation)
            self._motion_position = self._target.point_world + np.array(
                [0.0, 0.0, self.config.precontact_clearance_m]
            )
            self._set_phase("move_precontact")
        assert self._target is not None
        if self._phase == "move_precontact":
            if self._position_reached(observation, self._motion_position):
                self._motion_position = self._target.point_world.copy()
                self._set_phase("approach_contact")
            else:
                return self._move(observation, -1.0)
        if self._phase == "approach_contact":
            if self._position_reached(observation, self._motion_position) or self._contact_reached(observation):
                self._set_phase("engage")
            else:
                return self._move(observation, -1.0)
        if self._phase == "engage":
            if self._phase_ticks >= self.config.engage_ticks:
                turn = Rotation.from_rotvec(
                    self._target.axis_world
                    * self.config.knob_turn_rad
                    * step.turn_direction
                )
                self._knob_turn_target_rotation = (
                    turn.as_matrix() @ self._base_rotation
                )
                self._motion_rotation = self._knob_turn_target_rotation.copy()
                self._set_phase("turn")
            else:
                return self._tick(OSCAction.hold(1.0))
        if self._phase == "turn":
            angle = self._rotation_error(
                observation.proprio.T_world_ee[:3, :3],
                self._knob_turn_target_rotation,
            )
            self._knob_rotation_progress_rad = max(
                self._knob_rotation_progress_rad,
                self.config.knob_turn_rad - angle,
            )
            if angle <= self.config.rotation_tolerance_rad:
                self._set_phase("release")
            else:
                return self._move(observation, 1.0)
        if self._phase == "release":
            if self._phase_ticks >= self.config.release_ticks:
                self._motion_rotation = (
                    observation.proprio.T_world_ee[:3, :3].copy()
                )
                self._motion_position = observation.proprio.ee_position_world.copy()
                self._motion_position[2] += self.config.precontact_clearance_m
                self._set_phase("retreat")
            else:
                return self._tick(OSCAction.hold(-1.0))
        if self._phase == "retreat":
            if self._position_reached(
                observation,
                self._motion_position,
                tolerance_m=self.config.contact_retreat_tolerance_m,
            ):
                self._set_phase("verify_visual")
            else:
                return self._move(observation, -1.0)
        if self._phase == "verify_visual":
            visual_angle = self._fresh_knob_visual_angle(observation)
            if (
                visual_angle >= self.config.knob_visual_min_angle_rad
                or self._knob_rotation_progress_rad
                >= self.config.knob_turn_rad - 0.2
            ):
                return self._complete("stove turn verified from RGB-D/proprioceptive rotation")
            return self._fail("stove knob did not exhibit sufficient sensor-measured rotation")
        return self._fail(f"invalid knob phase {self._phase!r}")

    def _act_push(self, observation: RobotObservation) -> PolicyDecision:
        if self._phase == "detect":
            try:
                self._push = self.plate_detector.detect(observation)
            except (LookupError, ValueError) as exc:
                return self._detection_miss(str(exc), gripper=-1.0)
            self._push_attempt = 0
            self._initial_feature_point = self._push.object_center_world.copy()
            self._prepare_push_attempt(observation)
        assert self._push is not None
        if self._phase == "move_safe":
            if self._position_reached(
                observation,
                self._motion_position,
                tolerance_m=self.config.push_safe_waypoint_tolerance_m,
            ):
                self._motion_position = self._push_contact_position.copy()
                self._motion_position[2] += self.config.precontact_clearance_m
                self._set_phase("move_precontact")
            else:
                return self._move(observation, -1.0)
        if self._phase == "move_precontact":
            if self._position_reached(observation, self._motion_position):
                self._motion_position[2] -= self.config.precontact_clearance_m
                self._set_phase("approach_contact")
            else:
                return self._move(observation, -1.0)
        if self._phase == "approach_contact":
            if self._position_reached(
                observation,
                self._motion_position,
            ) or self._contact_reached(
                observation,
                tolerance_m=self.config.push_contact_tolerance_m,
            ):
                self._set_phase("engage_rim")
            else:
                return self._move(observation, -1.0)
        if self._phase == "engage_rim":
            if self._phase_ticks >= self.config.engage_ticks:
                width = observation.proprio.gripper_width_m
                if not (
                    self.config.drawer_blocked_min_width_m
                    <= width
                    <= self.config.drawer_blocked_max_width_m
                ):
                    return self._fail(
                        "plate rim pinch was not confirmed by measured gripper width"
                    )
                self._manipulation_start_position = observation.proprio.ee_position_world.copy()
                # Retries are re-anchored from fresh RGB-D.  Keep the
                # proven second-attempt contact profile for later retries;
                # compounding the inset/depth made a third attempt close
                # underneath a thin plate rim.
                retry_profile = min(self._push_attempt, 1)
                rim_fraction = (
                    self.config.push_rim_radius_fraction
                    - retry_profile
                    * self.config.push_retry_rim_inset_fraction
                )
                rim_offset = (
                    self._push.direction_world
                    * self._push.object_radius_m
                    * rim_fraction
                )
                # Keep the reachable leading-rim path.  A sensor-confirmed
                # retry seats the fingers slightly farther inside the thin
                # observed rim instead of repeating an empty edge closure.
                self._motion_position = (
                    self._push.target_center_world
                    + rim_offset
                    + self._push.direction_world
                    * (
                        self.config.push_goal_overshoot_m
                        + retry_profile * self.config.push_retry_overshoot_m
                    )
                )
                self._motion_position[2] = self._manipulation_start_position[2]
                self._set_phase("drag_rim")
            else:
                return self._move(observation, 1.0)
        if self._phase == "drag_rim":
            progress = float(
                np.dot(
                    observation.proprio.ee_position_world - self._manipulation_start_position,
                    self._push.direction_world,
                )
            )
            self._manipulation_progress_m = max(self._manipulation_progress_m, progress)
            if self._position_reached(observation, self._motion_position):
                self._set_phase("release_rim")
            else:
                return self._move(observation, 1.0)
        if self._phase == "release_rim":
            if (
                self._phase_ticks >= self.config.release_ticks
                and observation.proprio.gripper_width_m >= self.config.push_release_width_m
            ):
                self._motion_rotation = (
                    observation.proprio.T_world_ee[:3, :3].copy()
                )
                self._motion_position = observation.proprio.ee_position_world.copy()
                self._motion_position[2] += self.config.push_safe_height_m
                self._set_phase("retreat_rim")
            else:
                return self._tick(OSCAction.hold(-1.0))
        if self._phase == "retreat_rim":
            if self._position_reached(
                observation,
                self._motion_position,
                tolerance_m=self.config.contact_retreat_tolerance_m,
            ):
                self._set_phase("verify_visual")
            else:
                return self._move(observation, -1.0)
        if self._phase == "verify_visual":
            try:
                refreshed = self.plate_detector.track(observation, self._push)
            except (LookupError, ValueError) as exc:
                return self._fail(f"plate visual verification failed: {exc}")
            goal_error = float(
                np.linalg.norm(
                    (refreshed.object_center_world - self._push.target_center_world)[:2]
                )
            )
            displacement = float(
                np.dot(
                    refreshed.object_center_world - self._initial_feature_point,
                    self._push.direction_world,
                )
            )
            if goal_error <= self.config.push_goal_tolerance_m:
                return self._complete(
                    "plate push verified from fresh RGB-D target proximity "
                    f"({goal_error:.4f} m; displacement={displacement:.4f} m)"
                )
            if self._push_attempt + 1 < self.config.push_max_attempts:
                self._push_attempt += 1
                self._push = refreshed
                self._prepare_push_attempt(observation)
                return self._move(
                    observation,
                    -1.0,
                    "plate remained outside target; retrying from fresh RGB-D",
                )
            return self._fail("plate remained outside the sensor-derived stove-front region")
        return self._fail(f"invalid push phase {self._phase!r}")

    def _act_microwave(
        self,
        step: GoalSkillStep,
        observation: RobotObservation,
    ) -> PolicyDecision:
        """Pull or push a visually grounded microwave handle in closed loop."""

        if self.microwave_detector is None:
            return self._fail("no sensor-only microwave detector is configured")
        if self._phase == "detect":
            try:
                self._target = self.microwave_detector.detect(
                    observation,
                    step.kind,
                )
            except (LookupError, ValueError) as exc:
                return self._detection_miss(str(exc), gripper=-1.0)
            # RGB-D reconstructs the first visible surface.  Center the
            # grip-site on the public 20-mm handle diameter (closed state) or
            # 12-mm door-edge thickness (open state), so opposing pads land
            # on opposite sides instead of closing together in front of it.
            surface_inset = (
                self.config.microwave_closed_handle_surface_inset_m
                if step.kind is GoalSkillKind.OPEN_MICROWAVE
                else self.config.microwave_open_edge_surface_inset_m
            )
            corrected_point = (
                self._target.point_world
                - self._target.outward_world * surface_inset
            )
            self._target = ContactTarget(
                self._target.kind,
                corrected_point,
                self._target.axis_world,
                self._target.outward_world,
                self._target.fixture_center_world,
                self._target.feature_axis_world,
                self._target.confidence,
                self._target.source_cameras,
            )
            self._capture_target(observation)
            outward = self._axis_aligned_fixture_direction(
                self._target.outward_world
            )
            self._microwave_initial_outward = outward.copy()
            if step.kind is GoalSkillKind.OPEN_MICROWAVE:
                self._microwave_direction = outward
                rotary = self._microwave_rotary_geometry(initial_is_open=False)
                if rotary is not None:
                    hinge, rotation_axis, _closed_slot = rotary
                    start_radius = self._target.point_world - hinge
                    start_radius -= rotation_axis * float(
                        np.dot(start_radius, rotation_axis)
                    )
                    radius = float(np.linalg.norm(start_radius))
                    radius_min, radius_max = self.config.microwave_hinge_radius_range_m
                    if not radius_min <= radius <= radius_max:
                        return self._fail(
                            "microwave RGB-D hinge geometry had an implausible radius"
                        )
                    tangent = np.cross(rotation_axis, start_radius)
                    tangent_alignment = float(np.dot(tangent, outward))
                    if abs(tangent_alignment) < 0.05 * radius:
                        return self._fail(
                            "microwave pull direction was ambiguous relative to the RGB-D hinge"
                        )
                    direction_sign = 1.0 if tangent_alignment > 0.0 else -1.0
                    self._microwave_hinge_position = hinge
                    self._microwave_rotation_axis = rotation_axis
                    self._microwave_arc_angle_rad = (
                        direction_sign * self.config.microwave_open_angle_rad
                    )
                    endpoint = hinge + Rotation.from_rotvec(
                        rotation_axis * self._microwave_arc_angle_rad
                    ).apply(start_radius)
                    opening_chord = endpoint - self._target.point_world
                    chord_norm = float(np.linalg.norm(opening_chord))
                    if chord_norm < self.config.microwave_visual_displacement_m:
                        return self._fail(
                            "microwave RGB-D hinge arc was too short to verify"
                        )
                    self._microwave_direction = opening_chord / chord_norm
                    self._microwave_use_arc = True
            else:
                rotary = self._microwave_rotary_geometry(initial_is_open=True)
                if rotary is None:
                    # Minimal detector implementations may expose only the
                    # contact target.  Retain a conservative linear fallback,
                    # but the Route-B RGB-D provider supplies full OBB-derived
                    # articulation geometry and therefore takes the arc path.
                    closing_chord = (
                        self._target.fixture_center_world
                        - self._target.point_world
                    )
                    closing_chord[2] = 0.0
                    chord_norm = float(np.linalg.norm(closing_chord))
                    self._microwave_direction = (
                        closing_chord / chord_norm
                        if chord_norm >= self.config.precontact_clearance_m
                        else -outward
                    )
                    self._microwave_close_retreat_outward = (
                        -self._microwave_direction
                    )
                else:
                    hinge, rotation_axis, closed_slot = rotary
                    start_radius = self._target.point_world - hinge
                    start_radius -= rotation_axis * float(
                        np.dot(start_radius, rotation_axis)
                    )
                    goal_radius = closed_slot - hinge
                    goal_radius -= rotation_axis * float(
                        np.dot(goal_radius, rotation_axis)
                    )
                    start_norm = float(np.linalg.norm(start_radius))
                    goal_norm = float(np.linalg.norm(goal_radius))
                    radius_min, radius_max = self.config.microwave_hinge_radius_range_m
                    if not (
                        radius_min <= start_norm <= radius_max
                        and radius_min <= goal_norm <= radius_max
                    ):
                        return self._fail(
                            "microwave RGB-D hinge geometry had an implausible radius"
                        )
                    signed_angle = float(
                        np.arctan2(
                            np.dot(
                                rotation_axis,
                                np.cross(start_radius, goal_radius),
                            ),
                            np.dot(start_radius, goal_radius),
                        )
                    )
                    if abs(signed_angle) < 0.10:
                        tangent = np.cross(rotation_axis, start_radius)
                        tangent_sign = float(np.dot(tangent, -outward))
                        signed_angle = (1.0 if tangent_sign >= 0.0 else -1.0) * 0.10
                    self._microwave_hinge_position = hinge
                    self._microwave_rotation_axis = rotation_axis
                    self._microwave_closed_slot_position = closed_slot.copy()
                    self._microwave_close_target_angle_rad = signed_angle
                    closed_normal = np.cross(rotation_axis, goal_radius)
                    closed_normal_norm = float(np.linalg.norm(closed_normal))
                    if closed_normal_norm < 1e-8:
                        return self._fail(
                            "microwave closed door had a degenerate retreat normal"
                        )
                    closed_normal /= closed_normal_norm
                    # The reset-time RGB-D contact outward vector signs which
                    # side of the closed panel faces free space.  The panel
                    # normal itself comes from the sensor-derived closed
                    # hinge radius, not from the open edge orientation.
                    if float(np.dot(closed_normal, outward)) < 0.0:
                        closed_normal *= -1.0
                    self._microwave_close_retreat_outward = closed_normal
                    self._microwave_arc_angle_rad = signed_angle + np.sign(
                        signed_angle
                    ) * self.config.microwave_close_overshoot_rad
                    endpoint = hinge + Rotation.from_rotvec(
                        rotation_axis * self._microwave_arc_angle_rad
                    ).apply(start_radius)
                    closing_chord = endpoint - self._target.point_world
                    chord_norm = float(np.linalg.norm(closing_chord))
                    if chord_norm < self.config.microwave_visual_displacement_m:
                        return self._fail(
                            "microwave RGB-D hinge arc was too short to verify"
                        )
                    self._microwave_direction = closing_chord / chord_norm
                    self._microwave_use_arc = True
            self._microwave_initial_point = self._target.point_world.copy()
            self._microwave_contact_rotation = self._microwave_grasp_rotation(
                self._target,
                self._base_rotation,
            )
            self._microwave_precontact_position = (
                self._target.point_world
                + outward * self.config.precontact_clearance_m
            )
            self._motion_rotation = self._base_rotation.copy()
            if step.kind is GoalSkillKind.CLOSE_MICROWAVE:
                # An open door exposes a vertical leading edge.  Descending
                # above that edge drives the reset-time palm through the door
                # or appliance top.  First move farther outward along the
                # sensed edge normal, align the horizontal contact frame at
                # height, descend in free space, then approach the edge only
                # along that normal.
                self._microwave_outer_position = (
                    self._target.point_world
                    + outward * self.config.microwave_close_outer_clearance_m
                )
                safe = self._microwave_outer_position.copy()
                safe[2] = max(
                    observation.proprio.ee_position_world[2],
                    safe[2] + self.config.microwave_safe_height_m,
                )
                self._start_staged_translation(
                    observation,
                    safe,
                    "microwave_close_move_outer_safe",
                )
            else:
                # A closed front is safely approached by first translating
                # above the handle in the reset-time wrist frame, then
                # descending and aligning locally.
                safe = self._microwave_precontact_position.copy()
                safe[2] = max(
                    observation.proprio.ee_position_world[2],
                    safe[2] + self.config.microwave_safe_height_m,
                )
                self._start_staged_translation(
                    observation,
                    safe,
                    "microwave_move_safe",
                )

        assert self._target is not None
        if self._phase == "microwave_close_move_outer_safe":
            if self._staged_translation_reached(
                observation,
                tolerance_m=self.config.microwave_waypoint_tolerance_m,
            ):
                self._motion_position = self._staged_destination_position.copy()
                self._motion_rotation = self._microwave_contact_rotation.copy()
                self._set_phase("microwave_close_align_outer_safe")
            else:
                return self._move(observation, -1.0)
        if self._phase == "microwave_close_align_outer_safe":
            aligned = (
                self._position_reached(
                    observation,
                    self._motion_position,
                    tolerance_m=self.config.microwave_safe_position_tolerance_m,
                )
                and self._rotation_error(
                    observation.proprio.T_world_ee[:3, :3],
                    self._motion_rotation,
                )
                <= self.config.microwave_safe_rotation_tolerance_rad
            )
            if aligned:
                self._start_staged_translation(
                    observation,
                    self._microwave_outer_position,
                    "microwave_close_descend_outer",
                )
            else:
                return self._move(observation, -1.0)
        if self._phase == "microwave_close_descend_outer":
            if self._staged_translation_reached(
                observation,
                tolerance_m=self.config.microwave_waypoint_tolerance_m,
            ):
                self._start_staged_translation(
                    observation,
                    self._microwave_precontact_position,
                    "microwave_close_move_precontact",
                )
            else:
                return self._move(observation, -1.0)
        if self._phase == "microwave_close_move_precontact":
            if self._staged_translation_reached(
                observation,
                tolerance_m=self.config.microwave_waypoint_tolerance_m,
            ):
                self._motion_position = self._target.point_world.copy()
                self._set_phase("microwave_approach")
            else:
                return self._move(observation, -1.0)

        if self._phase == "microwave_move_safe":
            if self._staged_translation_reached(
                observation,
                tolerance_m=self.config.microwave_waypoint_tolerance_m,
            ):
                self._motion_position = self._staged_destination_position.copy()
                self._motion_rotation = self._microwave_contact_rotation.copy()
                self._set_phase("microwave_align_safe")
            else:
                return self._move(observation, -1.0)
        if self._phase == "microwave_align_safe":
            aligned = (
                self._position_reached(
                    observation,
                    self._motion_position,
                    tolerance_m=self.config.microwave_safe_position_tolerance_m,
                )
                and self._rotation_error(
                    observation.proprio.T_world_ee[:3, :3],
                    self._motion_rotation,
                )
                <= self.config.microwave_safe_rotation_tolerance_rad
            )
            if aligned:
                self._start_staged_translation(
                    observation,
                    self._microwave_precontact_position,
                    "microwave_descend_precontact",
                )
            else:
                return self._move(observation, -1.0)
        if self._phase == "microwave_descend_precontact":
            if self._staged_translation_reached(
                observation,
                tolerance_m=self.config.microwave_waypoint_tolerance_m,
            ):
                self._motion_position = self._target.point_world.copy()
                self._set_phase("microwave_approach")
            else:
                return self._move(observation, -1.0)
        if self._phase == "microwave_align_precontact":
            aligned = (
                self._position_reached(
                    observation,
                    self._motion_position,
                    tolerance_m=self.config.microwave_waypoint_tolerance_m,
                )
                and self._rotation_error(
                    observation.proprio.T_world_ee[:3, :3],
                    self._motion_rotation,
                )
                <= self.config.microwave_rotation_tolerance_rad
            )
            if aligned:
                self._motion_position = self._target.point_world.copy()
                self._set_phase("microwave_approach")
            else:
                return self._move(observation, -1.0)
        if self._phase == "microwave_approach":
            if self._position_reached(
                observation,
                self._motion_position,
            ) or self._contact_reached(observation):
                self._set_phase("microwave_engage")
            else:
                return self._move(observation, -1.0)
        if self._phase == "microwave_engage":
            if self._phase_ticks >= self.config.engage_ticks:
                width = observation.proprio.gripper_width_m
                if not (
                    self.config.drawer_blocked_min_width_m
                    <= width
                    <= self.config.drawer_blocked_max_width_m
                ):
                    return self._fail(
                        "microwave handle pinch was not confirmed by measured width"
                    )
                self._manipulation_start_position = (
                    observation.proprio.ee_position_world.copy()
                )
                if self._microwave_use_arc:
                    radial = (
                        self._manipulation_start_position
                        - self._microwave_hinge_position
                    )
                    radial -= self._microwave_rotation_axis * float(
                        np.dot(radial, self._microwave_rotation_axis)
                    )
                    radius = float(np.linalg.norm(radial))
                    radius_min, radius_max = self.config.microwave_hinge_radius_range_m
                    if not radius_min <= radius <= radius_max:
                        return self._fail(
                            "microwave grasp had an implausible sensor-derived hinge radius"
                        )
                    self._microwave_arc_start_position = (
                        self._manipulation_start_position.copy()
                    )
                    self._microwave_arc_start_rotation = (
                        observation.proprio.T_world_ee[:3, :3].copy()
                    )
                    if step.kind is GoalSkillKind.OPEN_MICROWAVE:
                        self._microwave_open_segment_start_angle_rad = 0.0
                        self._microwave_open_segment_start_rotation = (
                            self._microwave_arc_start_rotation.copy()
                        )
                        self._microwave_open_best_angle_rad = 0.0
                        self._microwave_open_angle_stall_ticks = 0
                    self._microwave_arc_completed_angle_rad = 0.0
                    self._microwave_arc_rebases = 0
                    self._microwave_arc_chunk_angle_rad = (
                        np.sign(self._microwave_arc_angle_rad)
                        * min(
                            abs(self._microwave_arc_angle_rad),
                            self.config.microwave_open_rebase_arc_rad,
                        )
                        if step.kind is GoalSkillKind.OPEN_MICROWAVE
                        else self._microwave_arc_angle_rad
                    )
                    self._microwave_arc_segments = max(
                        1,
                        int(
                            np.ceil(
                                abs(self._microwave_arc_chunk_angle_rad)
                                / self.config.microwave_arc_segment_rad
                            )
                        ),
                    )
                    self._microwave_arc_index = 1
                    self._set_microwave_arc_waypoint()
                else:
                    self._motion_position = (
                        self._manipulation_start_position
                        + self._microwave_direction * self.config.microwave_travel_m
                    )
                self._force_baseline = observation.proprio.ee_force_sensor.copy()
                self._microwave_previous_width_m = (
                    observation.proprio.gripper_width_m
                )
                self._microwave_width_plateau_ticks = 0
                self._microwave_previous_force_n = (
                    observation.proprio.ee_force_sensor.copy()
                )
                self._microwave_force_plateau_ticks = 0
                self._set_phase("microwave_move_door")
            else:
                return self._tick(OSCAction.hold(1.0))
        if self._phase == "microwave_open_segment_release":
            # While the measured aperture is still near the retained two-pad
            # width, the public grip-site remains the best contact-centre
            # observation.  Freeze the last such sample before the fingers
            # become visibly open; subsequent free-space drift must not move
            # the propagated handle anchor.
            if (
                not self._microwave_open_release_escape_active
                and
                observation.proprio.gripper_width_m
                <= self.config.microwave_open_release_anchor_width_m
            ):
                self._microwave_open_regrasp_anchor = (
                    observation.proprio.ee_position_world.copy()
                )
            release_ready = (
                self._phase_ticks >= self.config.release_ticks
                and observation.proprio.gripper_width_m
                >= self.config.drawer_release_width_m
            )
            if release_ready:
                if self._microwave_open_push_finalizing:
                    exit_progress = float(
                        np.linalg.norm(
                            observation.proprio.ee_position_world
                            - self._microwave_open_regrasp_anchor
                        )
                    )
                    if (
                        exit_progress
                        >= self.config.microwave_open_push_final_exit_m
                    ):
                        self._set_phase("microwave_verify")
                        return self._tick(
                            OSCAction.hold(-1.0),
                            "released back-side pusher cleared the observed door edge",
                        )
                    return self._move(observation, -1.0)
                lift_axis = self._microwave_rotation_axis.copy()
                if lift_axis[2] < 0.0:
                    lift_axis *= -1.0
                if self._microwave_open_push_mode:
                    radial = (
                        self._microwave_open_regrasp_anchor
                        - self._microwave_hinge_position
                    )
                    radial -= self._microwave_rotation_axis * float(
                        np.dot(radial, self._microwave_rotation_axis)
                    )
                    radial_norm = float(np.linalg.norm(radial))
                    if radial_norm < 1e-8:
                        return self._fail(
                            "microwave free edge could not define a push clearance"
                        )
                    radial /= radial_norm
                    self._motion_position = (
                        self._microwave_open_regrasp_anchor
                        + radial * self.config.microwave_open_push_edge_clearance_m
                        + lift_axis * self.config.microwave_open_push_lift_m
                    )
                    self._motion_rotation = (
                        observation.proprio.T_world_ee[:3, :3].copy()
                    )
                    self._set_phase("microwave_open_push_clear_edge")
                    return self._move(
                        observation,
                        -1.0,
                        "routing outside the observed microwave free edge",
                    )
                self._motion_position = (
                    observation.proprio.ee_position_world
                    + self._microwave_open_regrasp_outward
                    * self.config.microwave_open_regrasp_retreat_m
                    + lift_axis * self.config.microwave_open_regrasp_lift_m
                )
                self._motion_rotation = (
                    observation.proprio.T_world_ee[:3, :3].copy()
                )
                self._set_phase("microwave_open_segment_retreat")
                return self._move(observation, -1.0)
            if (
                self._phase_ticks >= self.config.release_ticks
                and not self._microwave_open_release_escape_active
            ):
                lift_axis = self._microwave_rotation_axis.copy()
                if lift_axis[2] < 0.0:
                    lift_axis *= -1.0
                self._motion_position = (
                    observation.proprio.ee_position_world
                    + self._microwave_open_regrasp_outward
                    * self.config.microwave_open_release_escape_m
                    + lift_axis * self.config.microwave_open_release_escape_lift_m
                )
                if self._microwave_open_push_mode:
                    escape_radial = (
                        observation.proprio.ee_position_world
                        - self._microwave_hinge_position
                    )
                    escape_radial -= self._microwave_rotation_axis * float(
                        np.dot(escape_radial, self._microwave_rotation_axis)
                    )
                    escape_radial /= max(
                        float(np.linalg.norm(escape_radial)),
                        1e-12,
                    )
                    self._motion_position += (
                        escape_radial
                        * self.config.microwave_open_push_release_radial_escape_m
                    )
                self._motion_rotation = (
                    observation.proprio.T_world_ee[:3, :3].copy()
                )
                self._microwave_open_release_escape_active = True
            if self._microwave_open_release_escape_active:
                return self._move(observation, -1.0)
            return self._tick(OSCAction.hold(-1.0))
        if self._phase == "microwave_open_push_clear_edge":
            if not self._position_reached(
                observation,
                self._motion_position,
                tolerance_m=self.config.microwave_open_push_waypoint_tolerance_m,
            ):
                return self._move(observation, -1.0)
            radial = (
                self._microwave_open_regrasp_anchor
                - self._microwave_hinge_position
            )
            radial -= self._microwave_rotation_axis * float(
                np.dot(radial, self._microwave_rotation_axis)
            )
            radial /= max(float(np.linalg.norm(radial)), 1e-12)
            try:
                tangent = self._microwave_opening_tangent(
                    self._microwave_open_regrasp_anchor
                )
            except ValueError as exc:
                return self._fail(str(exc))
            self._microwave_open_push_tangent = tangent
            lift_axis = self._microwave_rotation_axis.copy()
            if lift_axis[2] < 0.0:
                lift_axis *= -1.0
            self._motion_position = (
                self._microwave_open_regrasp_anchor
                + radial * self.config.microwave_open_push_edge_clearance_m
                - tangent * self.config.microwave_open_push_backside_clearance_m
                + lift_axis * self.config.microwave_open_push_lift_m
            )
            # Keep the already feasible retained-grasp orientation while
            # translating around the free edge.  Reversing the tool z-axis by
            # roughly pi at this radius can drive the OSC arm far from W2;
            # the compact closed fingers provide a valid side pusher without
            # that joint-space excursion.
            self._motion_rotation = observation.proprio.T_world_ee[:3, :3].copy()
            self._set_phase("microwave_open_push_backside_safe")
            return self._move(
                observation,
                -1.0,
                "moving around the free edge to its back side",
            )
        if self._phase == "microwave_open_push_backside_safe":
            anchor_radial = (
                self._microwave_open_regrasp_anchor
                - self._microwave_hinge_position
            )
            anchor_radial -= self._microwave_rotation_axis * float(
                np.dot(anchor_radial, self._microwave_rotation_axis)
            )
            anchor_radial /= max(float(np.linalg.norm(anchor_radial)), 1e-12)
            routed_delta = (
                observation.proprio.ee_position_world
                - self._microwave_open_regrasp_anchor
            )
            radial_clearance = float(np.dot(routed_delta, anchor_radial))
            backside_clearance = -float(
                np.dot(routed_delta, self._microwave_open_push_tangent)
            )
            topology_reached = (
                radial_clearance
                >= self.config.microwave_open_push_edge_clearance_m
                - self.config.microwave_open_push_waypoint_tolerance_m
                and backside_clearance
                >= self.config.microwave_open_push_precontact_m
                - self.config.microwave_open_push_waypoint_tolerance_m
            )
            aligned = (
                (
                    self._position_reached(
                        observation,
                        self._motion_position,
                        tolerance_m=self.config.microwave_open_push_waypoint_tolerance_m,
                    )
                    or topology_reached
                )
                and self._rotation_error(
                    observation.proprio.T_world_ee[:3, :3],
                    self._motion_rotation,
                )
                <= self.config.microwave_safe_rotation_tolerance_rad
            )
            if not aligned:
                return self._move(observation, -1.0)
            try:
                self._target = self._microwave_reacquire_open_edge(observation)
            except (LookupError, ValueError) as exc:
                return self._detection_miss(str(exc), gripper=-1.0)
            self._detection_misses = 0
            try:
                tangent = self._microwave_opening_tangent(
                    self._target.point_world
                )
            except ValueError as exc:
                return self._fail(str(exc))
            self._microwave_open_push_tangent = tangent
            contact_radial = (
                self._target.point_world - self._microwave_hinge_position
            )
            contact_radial -= self._microwave_rotation_axis * float(
                np.dot(contact_radial, self._microwave_rotation_axis)
            )
            contact_radial /= max(float(np.linalg.norm(contact_radial)), 1e-12)
            self._microwave_open_push_contact_point = (
                self._target.point_world
                - contact_radial * self.config.microwave_open_push_contact_inset_m
            )
            self._motion_rotation = observation.proprio.T_world_ee[:3, :3].copy()
            self._microwave_open_push_precontact_position = (
                self._microwave_open_push_contact_point
                - tangent * self.config.microwave_open_push_precontact_m
            )
            # Close the fingers while still outside the free-edge radius.
            # Closing at W3 can accidentally capture the edge and turns the
            # intended compressive pusher into another fragile pinch.
            self._set_phase("microwave_open_push_preshape")
            return self._tick(
                OSCAction.hold(1.0),
                "fresh RGB-D edge retained; compacting outside its free radius",
            )
        if self._phase == "microwave_open_push_precontact":
            rotation_aligned = (
                self._rotation_error(
                    observation.proprio.T_world_ee[:3, :3],
                    self._motion_rotation,
                )
                <= self.config.microwave_safe_rotation_tolerance_rad
            )
            aligned = (
                self._position_reached(
                    observation,
                    self._motion_position,
                    tolerance_m=self.config.microwave_open_push_waypoint_tolerance_m,
                )
                and rotation_aligned
            )
            compact_contact = (
                rotation_aligned
                and observation.proprio.gripper_width_m
                <= self.config.microwave_open_push_compact_width_m
                and self._contact_reached(observation)
            )
            if not aligned and not compact_contact:
                return self._move(observation, 1.0)
            if compact_contact:
                measured_angle = self._microwave_measured_chunk_angle(
                    observation.proprio.ee_position_world
                )
                self._microwave_open_best_angle_rad = (
                    np.sign(self._microwave_arc_angle_rad) * measured_angle
                )
                self._microwave_open_push_segment_start_angle_rad = (
                    self._microwave_open_best_angle_rad
                )
                self._microwave_open_angle_stall_ticks = 0
                self._set_phase("microwave_open_push_door")
                self._microwave_open_push_progress_samples = [
                    (0, self._microwave_open_best_angle_rad)
                ]
                tangent = self._microwave_opening_tangent(
                    observation.proprio.ee_position_world
                )
                self._motion_position = (
                    observation.proprio.ee_position_world
                    + tangent * self.config.microwave_open_push_tangent_step_m
                )
                return self._move(
                    observation,
                    1.0,
                    "compact back-side contact started the bounded door push",
                )
            self._motion_position = self._microwave_open_push_contact_point.copy()
            self._force_baseline = observation.proprio.ee_force_sensor.copy()
            self._set_phase("microwave_open_push_approach")
            return self._move(observation, 1.0)
        if self._phase == "microwave_open_push_preshape":
            if (
                self._phase_ticks < self.config.microwave_open_push_preshape_ticks
                or observation.proprio.gripper_width_m
                > self.config.microwave_open_push_compact_width_m
            ):
                return self._tick(OSCAction.hold(1.0))
            self._motion_position = (
                self._microwave_open_push_precontact_position.copy()
            )
            self._force_baseline = observation.proprio.ee_force_sensor.copy()
            self._set_phase("microwave_open_push_precontact")
            return self._move(
                observation,
                1.0,
                "compact pusher entering behind the observed free edge",
            )
        if self._phase == "microwave_open_push_approach":
            if not (
                self._position_reached(observation, self._motion_position)
                or self._contact_reached(observation)
            ):
                return self._move(observation, 1.0)
            self._microwave_open_best_angle_rad = (
                np.sign(self._microwave_arc_angle_rad)
                * self._microwave_measured_chunk_angle(
                    observation.proprio.ee_position_world
                )
            )
            self._microwave_open_push_segment_start_angle_rad = (
                self._microwave_open_best_angle_rad
            )
            self._microwave_open_angle_stall_ticks = 0
            self._microwave_open_push_recompact_ticks = 0
            self._microwave_open_push_recompact_pending = False
            self._microwave_open_push_last_compact_angle_rad = (
                self._microwave_open_best_angle_rad
            )
            self._microwave_open_push_compact_loss_baseline_angle_rad = (
                self._microwave_open_best_angle_rad
            )
            self._set_phase("microwave_open_push_door")
            self._microwave_open_push_progress_samples = [
                (0, self._microwave_open_best_angle_rad)
            ]
        if self._phase == "microwave_open_push_door":
            if (
                observation.proprio.gripper_width_m
                > self.config.microwave_open_push_compact_width_m
            ):
                if self._microwave_open_push_recompact_ticks == 0:
                    # Freeze the last angle measured while the pusher was
                    # compact.  A later narrow width alone cannot certify that
                    # the same back-side contact topology was recovered.
                    self._microwave_open_push_compact_loss_baseline_angle_rad = (
                        self._microwave_open_push_last_compact_angle_rad
                    )
                    self._microwave_open_push_recompact_pending = True
                self._microwave_open_push_recompact_ticks += 1
                if (
                    self._microwave_open_push_recompact_ticks
                    <= self.config.microwave_open_push_recompact_max_ticks
                ):
                    # The previous target still contains a tangential push.
                    # Emit an explicit zero-Cartesian hold instead: while the
                    # measured aperture exceeds 14 mm, only a bounded CLOSE
                    # attempt may be applied.  Phase ticks continue, so this
                    # recovery cannot extend the fixed 48/80 contact horizon.
                    return self._tick(
                        OSCAction.hold(1.0),
                        "wide back-side pusher holding still to recompact",
                    )
                return self._release_microwave_open_compact_loss(
                    observation,
                    "persistent wide pusher releasing for fresh RGB-D reacquisition",
                )
            measured_angle = self._microwave_measured_chunk_angle(
                observation.proprio.ee_position_world
            )
            direction_sign = np.sign(self._microwave_arc_angle_rad)
            aligned_angle = direction_sign * measured_angle
            current_radial = (
                observation.proprio.ee_position_world
                - self._microwave_hinge_position
            )
            current_radial -= self._microwave_rotation_axis * float(
                np.dot(current_radial, self._microwave_rotation_axis)
            )
            origin_radial = (
                self._microwave_arc_start_position
                - self._microwave_hinge_position
            )
            origin_radial -= self._microwave_rotation_axis * float(
                np.dot(origin_radial, self._microwave_rotation_axis)
            )
            radius_error = abs(
                float(np.linalg.norm(current_radial))
                - float(np.linalg.norm(origin_radial))
            )
            if radius_error > self.config.microwave_open_push_radius_tolerance_m:
                return self._fail(
                    "back-side microwave pusher left the frozen RGB-D hinge radius "
                    f"(error={radius_error:.4f} m)"
                )
            if self._microwave_open_push_recompact_pending:
                baseline = (
                    self._microwave_open_push_compact_loss_baseline_angle_rad
                )
                if (
                    aligned_angle
                    < self._microwave_open_best_angle_rad
                    - self.config.microwave_open_push_reverse_tolerance_rad
                ):
                    return self._release_microwave_open_compact_loss(
                        observation,
                        "recompacted back-side pusher reversed its public hinge angle; "
                        "releasing for fresh RGB-D reacquisition",
                    )
                if (
                    aligned_angle
                    < baseline
                    + self.config.microwave_open_stall_progress_epsilon_rad
                ):
                    return self._release_microwave_open_compact_loss(
                        observation,
                        "recompacted back-side pusher had no positive public hinge progress; "
                        "releasing for fresh RGB-D reacquisition",
                    )
                # The public angle advanced while Cartesian motion was frozen.
                # Discard the old target; the ordinary path below recomputes a
                # new tangent from the current EE/hinge geometry.
                self._microwave_open_push_recompact_pending = False
                self._microwave_open_push_progress_samples = [
                    (self._phase_ticks, aligned_angle)
                ]
            self._microwave_open_push_recompact_ticks = 0
            if (
                aligned_angle
                < self._microwave_open_best_angle_rad
                - self.config.microwave_open_push_reverse_tolerance_rad
            ):
                return self._fail(
                    "back-side microwave push reversed the sensor-derived opening angle"
                )
            self._microwave_open_push_last_compact_angle_rad = aligned_angle
            opening_threshold = self.config.microwave_open_angle_rad - 0.05
            if aligned_angle >= opening_threshold:
                tangent = self._microwave_opening_tangent(
                    observation.proprio.ee_position_world
                )
                self._target = ContactTarget(
                    GoalSkillKind.OPEN_MICROWAVE,
                    observation.proprio.ee_position_world.copy(),
                    self._target.axis_world,
                    tangent,
                    self._target.fixture_center_world,
                    self._target.feature_axis_world,
                    self._target.confidence,
                    self._target.source_cameras,
                )
                self._microwave_open_push_finalizing = True
                self._microwave_open_push_progress_samples = []
                self._microwave_open_regrasp_anchor = (
                    observation.proprio.ee_position_world.copy()
                )
                self._microwave_open_regrasp_outward = -tangent
                final_radial = (
                    observation.proprio.ee_position_world
                    - self._microwave_hinge_position
                )
                final_radial -= self._microwave_rotation_axis * float(
                    np.dot(final_radial, self._microwave_rotation_axis)
                )
                final_radial /= max(float(np.linalg.norm(final_radial)), 1e-12)
                lift_axis = self._microwave_rotation_axis.copy()
                if lift_axis[2] < 0.0:
                    lift_axis *= -1.0
                self._motion_position = (
                    observation.proprio.ee_position_world
                    - tangent * self.config.microwave_open_release_escape_m
                    + final_radial
                    * self.config.microwave_open_push_release_radial_escape_m
                    + lift_axis * self.config.microwave_open_release_escape_lift_m
                )
                self._motion_rotation = (
                    observation.proprio.T_world_ee[:3, :3].copy()
                )
                self._microwave_open_release_escape_active = True
                self._set_phase("microwave_open_segment_release")
                return self._move(
                    observation,
                    -1.0,
                    "opening threshold reached; releasing while clearing the free edge",
                )
            push_segment_tick_limit = (
                self.config.microwave_open_push_segment_max_ticks
                if self._microwave_open_push_segments_completed == 0
                else self.config.microwave_open_push_continuation_segment_max_ticks
            )
            # TCP motion around the frozen hinge is a safety proxy, not fresh
            # evidence that the door followed the pusher.  Keep the fixed
            # 48/80 hard caps, and additionally ask whether a bounded window
            # of public hinge progress can plausibly finish the remaining
            # angle while leaving a reserve for fresh RGB-D reacquisition.
            if (
                aligned_angle
                >= self._microwave_open_best_angle_rad
                + self.config.microwave_open_stall_progress_epsilon_rad
            ):
                self._microwave_open_best_angle_rad = aligned_angle
                self._microwave_open_angle_stall_ticks = 0
            else:
                self._microwave_open_angle_stall_ticks += 1
            push_segment_progress = max(
                0.0,
                self._microwave_open_best_angle_rad
                - self._microwave_open_push_segment_start_angle_rad,
            )
            if (
                self._microwave_open_angle_stall_ticks
                >= self.config.microwave_open_stall_ticks
            ):
                force_delta = float(
                    np.linalg.norm(
                        observation.proprio.ee_force_sensor
                        - self._force_baseline
                    )
                )
                return self._fail(
                    "back-side microwave edge push made no measured hinge progress "
                    f"(angle={aligned_angle:.4f} rad; "
                    f"best={self._microwave_open_best_angle_rad:.4f} rad; "
                    f"force_delta={force_delta:.2f} N; "
                    f"position_error={np.linalg.norm(self._motion_position - observation.proprio.ee_position_world):.4f} m)"
                )

            now = int(self._phase_ticks)
            sample = (now, self._microwave_open_best_angle_rad)
            if (
                self._microwave_open_push_progress_samples
                and self._microwave_open_push_progress_samples[-1][0] == now
            ):
                self._microwave_open_push_progress_samples[-1] = sample
            else:
                self._microwave_open_push_progress_samples.append(sample)
            window_ticks = self.config.microwave_open_push_progress_window_ticks
            cutoff = now - window_ticks
            self._microwave_open_push_progress_samples = [
                item
                for item in self._microwave_open_push_progress_samples
                if item[0] >= cutoff
            ][-(window_ticks + 1) :]
            first_tick, first_angle = (
                self._microwave_open_push_progress_samples[0]
            )
            sample_span = now - first_tick
            if (
                self._microwave_open_push_segments_completed > 0
                and push_segment_progress
                >= self.config.microwave_open_stall_min_segment_progress_rad
                and sample_span >= window_ticks
            ):
                window_gain = max(
                    0.0,
                    self._microwave_open_best_angle_rad - first_angle,
                )
                optimistic_rate = (
                    window_gain
                    + self.config.microwave_open_stall_progress_epsilon_rad
                ) / float(sample_span)
                ticks_left = max(0, push_segment_tick_limit - now)
                usable_ticks = max(
                    0,
                    ticks_left
                    - self.config.microwave_open_push_reacquire_reserve_ticks,
                )
                remaining_angle = max(
                    0.0,
                    opening_threshold - self._microwave_open_best_angle_rad,
                )
                optimistic_remaining_progress = optimistic_rate * usable_ticks
                if (
                    optimistic_remaining_progress
                    + self.config.microwave_open_stall_progress_epsilon_rad
                    < remaining_angle
                ):
                    return self._release_microwave_open_push_segment(
                        observation,
                        "slow public hinge window cannot finish the remaining "
                        "bounded push budget; releasing for fresh RGB-D reacquisition "
                        f"(gain={window_gain:.4f} rad/{sample_span} ticks; "
                        f"remaining={remaining_angle:.4f} rad; usable={usable_ticks})",
                    )

            if self._phase_ticks >= push_segment_tick_limit:
                if (
                    push_segment_progress
                    < self.config.microwave_open_stall_min_segment_progress_rad
                ):
                    return self._fail(
                        "bounded back-side microwave push ended without sufficient "
                        f"public hinge progress ({push_segment_progress:.4f} rad)"
                    )
                return self._release_microwave_open_push_segment(
                    observation,
                    "releasing after a bounded positive back-side push segment",
                )

            tangent = self._microwave_opening_tangent(
                observation.proprio.ee_position_world
            )
            push_ramp_ticks = max(
                0,
                self._microwave_open_angle_stall_ticks
                - self.config.microwave_open_push_tangent_ramp_start_ticks,
            )
            push_step = min(
                self.config.microwave_open_push_tangent_max_step_m,
                self.config.microwave_open_push_tangent_step_m
                + push_ramp_ticks
                * self.config.microwave_open_push_tangent_ramp_per_tick_m,
            )
            current_radius = float(np.linalg.norm(current_radial))
            origin_radius = float(np.linalg.norm(origin_radial))
            radial_unit = current_radial / max(current_radius, 1e-12)
            radius_correction = float(
                np.clip(
                    origin_radius - current_radius,
                    -self.config.microwave_open_push_radius_correction_max_m,
                    self.config.microwave_open_push_radius_correction_max_m,
                )
            )
            self._motion_position = (
                observation.proprio.ee_position_world
                + tangent * push_step
                + radial_unit * radius_correction
            )
            return self._move(
                observation,
                1.0,
                "closed-finger back-side push along the sensor-derived hinge tangent",
            )
        if self._phase == "microwave_open_segment_retreat":
            if not self._position_reached(
                observation,
                self._motion_position,
                tolerance_m=self.config.contact_retreat_tolerance_m,
            ):
                return self._move(observation, -1.0)
            self._set_phase("microwave_open_segment_reacquire")
        if self._phase == "microwave_open_segment_reacquire":
            try:
                self._target = self._microwave_reacquire_open_edge(observation)
            except (LookupError, ValueError) as exc:
                return self._detection_miss(str(exc), gripper=-1.0)
            self._detection_misses = 0
            self._microwave_contact_rotation = self._microwave_grasp_rotation(
                self._target,
                observation.proprio.T_world_ee[:3, :3],
            )
            self._microwave_precontact_position = (
                self._target.point_world
                + self._target.outward_world
                * self.config.microwave_open_regrasp_precontact_clearance_m
            )
            self._motion_position = self._microwave_precontact_position.copy()
            self._motion_rotation = self._microwave_contact_rotation.copy()
            self._force_baseline = observation.proprio.ee_force_sensor.copy()
            self._set_phase("microwave_open_segment_precontact")
            return self._move(
                observation,
                -1.0,
                "fresh RGB-D moving edge passed local hinge gates",
            )
        if self._phase == "microwave_open_segment_precontact":
            aligned = (
                self._position_reached(
                    observation,
                    self._motion_position,
                    tolerance_m=self.config.microwave_waypoint_tolerance_m,
                )
                and self._rotation_error(
                    observation.proprio.T_world_ee[:3, :3],
                    self._motion_rotation,
                )
                <= self.config.microwave_safe_rotation_tolerance_rad
            )
            if not aligned:
                return self._move(observation, -1.0)
            self._motion_position = self._target.point_world.copy()
            self._set_phase("microwave_open_segment_approach")
        if self._phase == "microwave_open_segment_approach":
            if not (
                self._position_reached(observation, self._motion_position)
                or self._contact_reached(observation)
            ):
                return self._move(observation, -1.0)
            self._set_phase("microwave_open_segment_engage")
        if self._phase == "microwave_open_segment_engage":
            if self._phase_ticks < self.config.microwave_open_regrasp_engage_ticks:
                return self._move(observation, 1.0)
            width = observation.proprio.gripper_width_m
            if not (
                self.config.drawer_blocked_min_width_m
                <= width
                <= self.config.drawer_blocked_max_width_m
            ):
                return self._fail(
                    "fresh microwave edge pinch was not confirmed by measured width"
                )
            measured_angle = self._microwave_measured_chunk_angle(
                observation.proprio.ee_position_world
            )
            self._microwave_open_segment_start_angle_rad = (
                np.sign(self._microwave_arc_angle_rad) * measured_angle
            )
            self._microwave_open_segment_start_rotation = (
                observation.proprio.T_world_ee[:3, :3].copy()
            )
            self._microwave_open_best_angle_rad = (
                self._microwave_open_segment_start_angle_rad
            )
            self._microwave_open_angle_stall_ticks = 0
            self._microwave_previous_width_m = width
            self._microwave_width_plateau_ticks = 0
            self._microwave_previous_force_n = (
                observation.proprio.ee_force_sensor.copy()
            )
            self._microwave_force_plateau_ticks = 0
            self._set_phase("microwave_move_door")
        if self._phase.startswith("microwave_close_push_"):
            return self._act_microwave_close_face_pusher(step, observation)
        if self._phase == "microwave_move_door":
            progress = float(
                np.dot(
                    observation.proprio.ee_position_world
                    - self._manipulation_start_position,
                    self._microwave_direction,
                )
            )
            self._manipulation_progress_m = max(
                self._manipulation_progress_m,
                progress,
            )
            if self._microwave_use_arc:
                width = observation.proprio.gripper_width_m
                if not (
                    self.config.drawer_blocked_min_width_m
                    <= width
                    <= self.config.drawer_blocked_max_width_m
                ):
                    measured_angle = self._microwave_measured_chunk_angle(
                        observation.proprio.ee_position_world
                    )
                    aligned_angle = (
                        np.sign(self._microwave_arc_angle_rad) * measured_angle
                    )
                    return self._fail(
                        "microwave handle grasp was lost during the RGB-D hinge arc "
                        f"(width={width:.4f} m; waypoint="
                        f"{self._microwave_arc_index}/{self._microwave_arc_segments}; "
                        f"measured_open_angle={aligned_angle:.4f} rad)"
                    )
                if step.kind is GoalSkillKind.OPEN_MICROWAVE:
                    measured_angle = self._microwave_measured_chunk_angle(
                        observation.proprio.ee_position_world
                    )
                    direction_sign = np.sign(self._microwave_arc_angle_rad)
                    aligned_angle = direction_sign * measured_angle
                    self._microwave_arc_completed_angle_rad = measured_angle
                    if (
                        aligned_angle
                        >= self._microwave_open_best_angle_rad
                        + self.config.microwave_open_stall_progress_epsilon_rad
                    ):
                        self._microwave_open_best_angle_rad = aligned_angle
                        self._microwave_open_angle_stall_ticks = 0
                    else:
                        self._microwave_open_angle_stall_ticks += 1
                    best_segment_progress = max(
                        0.0,
                        self._microwave_open_best_angle_rad
                        - self._microwave_open_segment_start_angle_rad,
                    )
                    stalled_after_progress = (
                        self._microwave_open_angle_stall_ticks
                        >= self.config.microwave_open_stall_ticks
                        and best_segment_progress
                        >= self.config.microwave_open_stall_min_segment_progress_rad
                    )
                    bounded_segment_complete = (
                        self._microwave_open_regrasps > 0
                        and self._phase_ticks
                        >= (
                            self.config.microwave_open_high_angle_segment_max_ticks
                            if aligned_angle
                            >= self.config.microwave_open_high_angle_rad
                            else self.config.microwave_open_segment_max_ticks
                        )
                        and best_segment_progress
                        >= self.config.microwave_open_stall_min_segment_progress_rad
                    )
                    segment_limit = (
                        self.config.microwave_open_regrasp_segment_rad
                        if self._microwave_open_regrasps == 0
                        else self.config.microwave_open_continuation_segment_rad
                    )
                    switch_to_push = (
                        aligned_angle
                        >= self.config.microwave_open_push_transition_rad
                    )
                    if aligned_angle >= self.config.microwave_open_angle_rad - 0.05:
                        self._set_phase("microwave_release")
                    elif (
                        switch_to_push
                        or aligned_angle
                        - self._microwave_open_segment_start_angle_rad
                        >= segment_limit
                        or stalled_after_progress
                        or bounded_segment_complete
                    ):
                        if (
                            not switch_to_push
                            and
                            self._microwave_open_regrasps
                            >= self.config.microwave_open_max_regrasps
                        ):
                            return self._fail(
                                "microwave open arc exhausted its bounded RGB-D regrasp budget "
                                f"(measured_open_angle={aligned_angle:.4f} rad; "
                                f"best_open_angle={self._microwave_open_best_angle_rad:.4f} rad)"
                            )
                        if switch_to_push:
                            self._microwave_open_push_mode = True
                        else:
                            self._microwave_open_regrasps += 1
                        self._microwave_open_release_escape_active = False
                        self._microwave_open_regrasp_anchor = (
                            observation.proprio.ee_position_world.copy()
                        )
                        rotated_outward = Rotation.from_rotvec(
                            self._microwave_rotation_axis * measured_angle
                        ).apply(self._microwave_initial_outward)
                        self._microwave_open_regrasp_outward = (
                            self._axis_aligned_fixture_direction(rotated_outward)
                        )
                        self._set_phase("microwave_open_segment_release")
                        return self._tick(
                            OSCAction.hold(-1.0),
                            (
                                "releasing to route around the observed free edge for a back-side push"
                                if switch_to_push
                                else (
                                    "releasing after a measured hinge-motion plateau"
                                    if stalled_after_progress
                                    else (
                                        "releasing after the bounded retained-grasp continuation"
                                        if bounded_segment_complete
                                        else "releasing before the retained handle grasp becomes one-sided"
                                    )
                                )
                            ),
                        )
                    else:
                        current = observation.proprio.ee_position_world
                        radial = current - self._microwave_hinge_position
                        radial -= self._microwave_rotation_axis * float(
                            np.dot(radial, self._microwave_rotation_axis)
                        )
                        radius = float(np.linalg.norm(radial))
                        radius_min, radius_max = (
                            self.config.microwave_hinge_radius_range_m
                        )
                        if not radius_min <= radius <= radius_max:
                            return self._fail(
                                "microwave retained handle left the sensor-derived hinge radius"
                            )
                        tangent = np.cross(
                            self._microwave_rotation_axis,
                            radial,
                        )
                        tangent *= direction_sign / max(
                            float(np.linalg.norm(tangent)),
                            1e-12,
                        )
                        ramp_ticks = max(
                            0,
                            self._microwave_open_angle_stall_ticks
                            - self.config.microwave_open_tangent_ramp_start_ticks,
                        )
                        high_angle = (
                            aligned_angle
                            >= self.config.microwave_open_high_angle_rad
                        )
                        base_tangent_step = (
                            self.config.microwave_open_high_angle_tangent_step_m
                            if high_angle
                            else self.config.microwave_open_tangent_step_m
                        )
                        max_tangent_step = (
                            self.config.microwave_open_high_angle_tangent_max_step_m
                            if high_angle
                            else self.config.microwave_open_tangent_max_step_m
                        )
                        tangent_step = min(
                            max_tangent_step,
                            base_tangent_step
                            + ramp_ticks
                            * self.config.microwave_open_tangent_ramp_per_tick_m,
                        )
                        self._motion_position = current + tangent * tangent_step
                        segment_angle = max(
                            0.0,
                            aligned_angle
                            - self._microwave_open_segment_start_angle_rad,
                        )
                        wrist_angle = direction_sign * min(
                            segment_angle,
                            self.config.microwave_wrist_corotation_limit_rad,
                        )
                        self._motion_rotation = (
                            Rotation.from_rotvec(
                                self._microwave_rotation_axis * wrist_angle
                            ).as_matrix()
                            @ self._microwave_open_segment_start_rotation
                        )
                        return self._move(observation, 1.0)
                else:
                    measured_angle = self._microwave_measured_chunk_angle(
                        observation.proprio.ee_position_world
                    )
                    direction_sign = np.sign(self._microwave_arc_angle_rad)
                    aligned_angle = direction_sign * measured_angle
                    target_angle = abs(self._microwave_close_target_angle_rad)
                    remaining_to_slot = target_angle - aligned_angle
                    if (
                        target_angle > 1e-8
                        and not self._microwave_close_push_mode
                        and aligned_angle >= 0.0
                        and remaining_to_slot
                        <= self.config.microwave_close_push_transition_remaining_rad
                    ):
                        return self._start_microwave_close_face_push(
                            observation,
                            aligned_angle=aligned_angle,
                        )
                    waypoint_reached = (
                        self._position_reached(
                            observation,
                            self._motion_position,
                            tolerance_m=self.config.microwave_arc_waypoint_tolerance_m,
                        )
                        and self._rotation_error(
                            observation.proprio.T_world_ee[:3, :3],
                            self._motion_rotation,
                        )
                        <= self.config.microwave_rotation_tolerance_rad
                    )
                if (
                    step.kind is GoalSkillKind.CLOSE_MICROWAVE
                    and waypoint_reached
                ):
                    if self._microwave_arc_index >= self._microwave_arc_segments:
                        if step.kind is GoalSkillKind.OPEN_MICROWAVE:
                            measured_angle = self._microwave_measured_chunk_angle(
                                observation.proprio.ee_position_world
                            )
                            aligned_progress = (
                                np.sign(self._microwave_arc_angle_rad)
                                * measured_angle
                            )
                            if (
                                aligned_progress
                                < self.config.microwave_open_min_chunk_progress_rad
                            ):
                                return self._fail(
                                    "microwave open arc chunk lacked measured "
                                    "sensor-hinge progress"
                                )
                            self._microwave_arc_completed_angle_rad += measured_angle
                            remaining = max(
                                0.0,
                                abs(self._microwave_arc_angle_rad)
                                - abs(self._microwave_arc_completed_angle_rad),
                            )
                            if remaining <= 0.08:
                                self._set_phase("microwave_release")
                            elif (
                                self._microwave_arc_rebases
                                >= self.config.microwave_open_max_rebases
                            ):
                                return self._fail(
                                    "microwave remained short of the measured open arc "
                                    "after bounded rebases"
                                )
                            else:
                                self._microwave_arc_rebases += 1
                                self._microwave_arc_start_position = (
                                    observation.proprio.ee_position_world.copy()
                                )
                                self._microwave_arc_chunk_angle_rad = (
                                    np.sign(self._microwave_arc_angle_rad)
                                    * min(
                                        remaining,
                                        self.config.microwave_open_rebase_arc_rad,
                                    )
                                )
                                self._microwave_arc_segments = max(
                                    1,
                                    int(
                                        np.ceil(
                                            abs(self._microwave_arc_chunk_angle_rad)
                                            / self.config.microwave_arc_segment_rad
                                        )
                                    ),
                                )
                                self._microwave_arc_index = 1
                                self._set_microwave_arc_waypoint()
                                self._phase_ticks = 0
                                self._previous_error = None
                                self._stall_ticks = 0
                                return self._move(observation, 1.0)
                        else:
                            if abs(self._microwave_close_target_angle_rad) > 1e-8:
                                return self._fail(
                                    "microwave retained pinch bypassed the required close face-pusher transition"
                                )
                            self._start_microwave_close_release(
                                observation,
                                mechanical_stop=False,
                            )
                    else:
                        self._microwave_arc_index += 1
                        self._set_microwave_arc_waypoint()
                        self._previous_error = None
                        self._stall_ticks = 0
                        self._microwave_previous_width_m = (
                            observation.proprio.gripper_width_m
                        )
                        self._microwave_width_plateau_ticks = 0
                        self._microwave_previous_force_n = (
                            observation.proprio.ee_force_sensor.copy()
                        )
                        self._microwave_force_plateau_ticks = 0
                        return self._move(observation, 1.0)
                elif (
                    step.kind is GoalSkillKind.CLOSE_MICROWAVE
                    and self._microwave_close_mechanical_stop(observation)
                ):
                    if abs(self._microwave_close_target_angle_rad) > 1e-8:
                        return self._fail(
                            "microwave retained pinch stopped before the required close face-pusher transition"
                        )
                    # Compatibility for minimal non-articulated detector
                    # adapters only.  The production RGB-D CLOSE path binds a
                    # frozen slot and must switch to the compact face pusher.
                    self._start_microwave_close_release(
                        observation,
                        mechanical_stop=True,
                    )
                elif step.kind is GoalSkillKind.CLOSE_MICROWAVE:
                    return self._move(observation, 1.0)
            elif self._position_reached(
                observation,
                self._motion_position,
            ) or (
                self._manipulation_progress_m
                >= self.config.microwave_travel_m - 0.018
            ):
                if step.kind is GoalSkillKind.CLOSE_MICROWAVE:
                    self._start_microwave_close_release(
                        observation,
                        mechanical_stop=False,
                    )
                else:
                    self._set_phase("microwave_release")
            else:
                return self._move(observation, 1.0)
        if self._phase == "microwave_release":
            if (
                self._phase_ticks >= self.config.release_ticks
                and observation.proprio.gripper_width_m
                >= self.config.drawer_release_width_m
            ):
                if (
                    step.kind is GoalSkillKind.CLOSE_MICROWAVE
                    and self._microwave_close_release_pose_frozen
                ):
                    self._start_microwave_close_retreat(observation)
                else:
                    self._motion_rotation = (
                        observation.proprio.T_world_ee[:3, :3].copy()
                    )
                    self._motion_position = (
                        observation.proprio.ee_position_world.copy()
                        + self._target.outward_world
                        * self.config.microwave_retreat_clearance_m
                    )
                    self._motion_position[2] += self.config.precontact_clearance_m
                    self._set_phase("microwave_retreat")
            else:
                if (
                    step.kind is GoalSkillKind.CLOSE_MICROWAVE
                    and self._microwave_close_release_pose_frozen
                ):
                    self._motion_position = self._microwave_release_position.copy()
                    self._motion_rotation = self._microwave_release_rotation.copy()
                    if self._phase_ticks >= self.config.release_ticks:
                        # The public jaws are still blocked after the normal
                        # in-place dwell.  Enter a distinct, bounded unseat
                        # phase; only CLOSE plus the measured width can take
                        # this edge-release path.
                        self._set_phase("microwave_close_release_unseat")
                    else:
                        return self._move(
                            observation,
                            -1.0,
                            "opening fingers while holding the measured microwave stop pose",
                        )
                else:
                    return self._tick(OSCAction.hold(-1.0))
        if self._phase == "microwave_close_release_unseat":
            if (
                step.kind is not GoalSkillKind.CLOSE_MICROWAVE
                or not self._microwave_close_release_pose_frozen
            ):
                return self._fail(
                    "microwave close-release unseat lacked a frozen CLOSE pose"
                )

            outward = self._microwave_close_release_outward
            outward_norm = float(np.linalg.norm(outward))
            if not np.all(np.isfinite(outward)) or not np.isclose(
                outward_norm,
                1.0,
                atol=1e-6,
            ):
                return self._fail(
                    "microwave close-release unseat had an invalid sensor normal"
                )

            maximum = self.config.microwave_close_release_unseat_max_m
            step_distance = self.config.microwave_close_release_unseat_step_m
            commanded = float(
                self._microwave_close_release_unseat_distance_m
            )
            if (
                not np.isfinite(commanded)
                or commanded < -1e-9
                or commanded > maximum + 1e-9
            ):
                return self._fail(
                    "microwave close-release unseat command reversed or exceeded its bound"
                )

            measured_delta = (
                observation.proprio.ee_position_world
                - self._microwave_release_position
            )
            measured_signed = float(np.dot(measured_delta, outward))
            # Sensor noise can move the measured grip site by a fraction of
            # one rolling increment.  A full-step excursion behind the frozen
            # stop pose or beyond the 18-mm envelope is a hard safety failure.
            if measured_signed < -step_distance - 1e-9:
                return self._fail(
                    "microwave close-release unseat moved opposite the sensor free-space normal"
                )
            if measured_signed > maximum + step_distance + 1e-9:
                return self._fail(
                    "microwave close-release unseat exceeded its sensor-space bound"
                )

            if (
                observation.proprio.gripper_width_m
                >= self.config.drawer_release_width_m
            ):
                self._start_microwave_close_retreat(observation)
            else:
                self._motion_rotation = self._microwave_release_rotation.copy()
                if commanded >= maximum - 1e-9:
                    self._motion_position = (
                        self._microwave_release_position
                        + outward * maximum
                    )
                    if self._position_reached(
                        observation,
                        self._motion_position,
                        tolerance_m=step_distance,
                    ):
                        return self._fail(
                            "microwave fingers remained blocked after the bounded close-release unseat"
                        )
                    return self._move(
                        observation,
                        -1.0,
                        "opening fingers at the bounded microwave close-release unseat limit",
                    )
                next_distance = min(maximum, commanded + step_distance)
                if (
                    next_distance <= commanded
                    or next_distance - commanded > step_distance + 1e-9
                ):
                    return self._fail(
                        "microwave close-release unseat command was not bounded and monotonic"
                    )
                self._microwave_close_release_unseat_distance_m = next_distance
                self._motion_position = (
                    self._microwave_release_position
                    + outward * next_distance
                )
                return self._move(
                    observation,
                    -1.0,
                    "opening fingers while monotonically unseating the microwave edge",
                )
        if self._phase == "microwave_close_retreat_outward":
            if self._position_reached(
                observation,
                self._motion_position,
                tolerance_m=self.config.contact_retreat_tolerance_m,
            ):
                self._motion_position = (
                    observation.proprio.ee_position_world.copy()
                )
                self._motion_position[2] += self.config.precontact_clearance_m
                self._set_phase("microwave_close_retreat_up")
            else:
                return self._move(observation, -1.0)
        if self._phase == "microwave_close_retreat_up":
            if self._position_reached(
                observation,
                self._motion_position,
                tolerance_m=self.config.contact_retreat_tolerance_m,
            ):
                self._set_phase("microwave_verify")
            else:
                return self._move(observation, -1.0)
        if self._phase == "microwave_retreat":
            if self._position_reached(
                observation,
                self._motion_position,
                tolerance_m=self.config.contact_retreat_tolerance_m,
            ):
                self._set_phase("microwave_verify")
            else:
                return self._move(observation, -1.0)
        if self._phase == "microwave_verify":
            visual_displacement = 0.0
            visual_available = False
            closed_slot_consistent = True
            refreshed: ContactTarget | None = None
            try:
                refreshed = self.microwave_detector.track(
                    observation,
                    self._target,
                    step.kind,
                )
                visual_available = True
                visual_displacement = abs(
                    float(
                        np.dot(
                            refreshed.point_world - self._microwave_initial_point,
                            self._microwave_direction,
                        )
                    )
                )
                if (
                    step.kind is GoalSkillKind.CLOSE_MICROWAVE
                    and self._microwave_use_arc
                    and abs(self._microwave_close_target_angle_rad) > 1e-8
                ):
                    closed_slot_consistent = bool(
                        np.linalg.norm(
                            refreshed.point_world
                            - self._microwave_closed_slot_position
                        )
                        <= self.config.microwave_close_verify_slot_tolerance_m
                    )
            except (LookupError, ValueError):
                pass
            if (
                refreshed is not None
                and refreshed.confidence >= self.config.microwave_visual_min_confidence
                and visual_displacement
                >= self.config.microwave_visual_displacement_m
                and closed_slot_consistent
            ):
                return self._complete(
                    "microwave door motion verified from fresh RGB-D"
                )
            if (
                step.kind is GoalSkillKind.OPEN_MICROWAVE
                and not visual_available
                and self._manipulation_progress_m
                >= self.config.microwave_visual_displacement_m
            ):
                return self._complete(
                    "microwave door motion verified from measured retained-handle travel"
                )
            return self._fail(
                "microwave door lacked sufficient sensor-measured displacement"
            )
        return self._fail(f"invalid microwave phase {self._phase!r}")

    def _handle_phase_timeout(
        self,
        step: GoalSkillStep,
        observation: RobotObservation,
    ) -> PolicyDecision:
        """Resolve a blocked phase without treating a command as evidence.

        Manipulation phases may advance only when the normal Cartesian gate
        or a fresh visual semantic gate passes.  Retreats are safety motions;
        if one is mechanically constrained after the fixture has already
        changed state, a fresh RGB-D verification may finish the skill.
        """

        if self._phase == "turn":
            visual_angle = self._fresh_knob_visual_angle(observation)
            if visual_angle >= self.config.knob_visual_min_angle_rad:
                self._set_phase("release")
                return self._tick(
                    OSCAction.hold(-1.0),
                    "stove turn visually verified after constrained rotation",
                )

        retreat_phases = {
            "retreat_outward",
            "retreat_up",
            "close_retreat_outward",
            "close_retreat_up",
            "retreat",
            "retreat_rim",
            "microwave_retreat",
            "microwave_close_retreat_outward",
            "microwave_close_retreat_up",
        }
        if self._phase in retreat_phases:
            verified = self._fresh_semantic_transition(step, observation)
            if verified is not None:
                return self._complete(
                    f"{verified}; constrained retreat ended after fresh RGB-D verification"
                )
        position_error = float(
            np.linalg.norm(
                observation.proprio.ee_position_world - self._motion_position
            )
        )
        rotation_error = self._rotation_error(
            observation.proprio.T_world_ee[:3, :3],
            self._motion_rotation,
        )
        return self._fail(
            f"phase {self._phase!r} timed out "
            f"(position_error={position_error:.4f} m; "
            f"rotation_error={rotation_error:.4f} rad; "
            "current="
            f"{np.round(observation.proprio.ee_position_world, 4).tolist()}; "
            f"target={np.round(self._motion_position, 4).tolist()})"
        )

    def _fresh_semantic_transition(
        self,
        step: GoalSkillStep,
        observation: RobotObservation,
    ) -> str | None:
        """Return a message only for a freshly observed semantic transition."""

        if step.kind in {GoalSkillKind.OPEN_DRAWER, GoalSkillKind.CLOSE_DRAWER}:
            try:
                refreshed = self._track_drawer(
                    observation,
                    step.level or "middle",
                )
            except (LookupError, ValueError):
                return None
            delta = (
                refreshed.point_world - self._initial_feature_point
                if step.kind is GoalSkillKind.OPEN_DRAWER
                else self._initial_feature_point - refreshed.point_world
            )
            displacement = float(np.dot(delta, self._drawer_pull_direction))
            if displacement >= self.config.drawer_visual_displacement_m:
                action = "opening" if step.kind is GoalSkillKind.OPEN_DRAWER else "closing"
                return f"drawer {action} verified from fresh RGB-D handle displacement"
            return None

        if step.kind is GoalSkillKind.TURN_KNOB:
            if (
                self._fresh_knob_visual_angle(observation)
                >= self.config.knob_visual_min_angle_rad
            ):
                return "stove turn verified from fresh RGB-D feature rotation"
            return None

        if step.kind is GoalSkillKind.PUSH_OBJECT and self._push is not None:
            try:
                refreshed = self.plate_detector.track(observation, self._push)
            except (LookupError, ValueError):
                return None
            goal_error = float(
                np.linalg.norm(
                    (
                        refreshed.object_center_world
                        - self._push.target_center_world
                    )[:2]
                )
            )
            displacement = float(
                np.dot(
                    refreshed.object_center_world - self._initial_feature_point,
                    self._push.direction_world,
                )
            )
            if (
                goal_error <= max(0.045, 0.75 * self._push.object_radius_m)
                or displacement >= 0.18
            ):
                return "plate push verified from fresh RGB-D displacement"
            return None

        if (
            step.kind
            in {GoalSkillKind.OPEN_MICROWAVE, GoalSkillKind.CLOSE_MICROWAVE}
            and self.microwave_detector is not None
            and self._target is not None
        ):
            try:
                refreshed = self.microwave_detector.track(
                    observation,
                    self._target,
                    step.kind,
                )
            except (LookupError, ValueError):
                return None
            displacement = abs(
                float(
                    np.dot(
                        refreshed.point_world - self._microwave_initial_point,
                        self._microwave_direction,
                    )
                )
            )
            if (
                refreshed.confidence
                >= self.config.microwave_visual_min_confidence
                and displacement
                >= self.config.microwave_visual_displacement_m
            ):
                return "microwave door motion verified from fresh RGB-D"
        return None

    def _fresh_knob_visual_angle(
        self,
        observation: RobotObservation,
    ) -> float:
        try:
            refreshed = self.knob_detector.detect(observation)
        except (LookupError, ValueError):
            return 0.0
        if (
            np.linalg.norm(
                refreshed.point_world - self._initial_feature_point
            )
            > self.config.knob_track_radius_m
        ):
            return 0.0
        dot = abs(
            float(
                np.dot(
                    refreshed.feature_axis_world,
                    self._initial_feature_axis,
                )
            )
        )
        return float(np.arccos(np.clip(dot, 0.0, 1.0)))

    def _start_drawer_full_pull(
        self,
        observation: RobotObservation,
    ) -> None:
        """Commit to the full pull only after a settled public load proof."""

        self._drawer_load_proof_passed = True
        self._drawer_load_proof_failure = ""
        self._motion_position = (
            self._manipulation_start_position
            + self._drawer_pull_direction * self.config.drawer_pull_distance_m
        )
        # Rebase after the small loaded chord.  A later rail stop must show a
        # new wrench-magnitude rise, not merely the force that proved the
        # initial handle seat.
        self._force_baseline = observation.proprio.ee_force_sensor.copy()
        self._drawer_pull_previous_width_m = None
        self._drawer_pull_width_plateau_ticks = 0
        self._drawer_pull_force_ticks = 0
        self._drawer_pull_mechanical_stop_observed = False
        self._set_phase("pull")

    def _release_rejected_drawer_grasp(
        self,
        reason: str,
    ) -> PolicyDecision:
        """Release an unproven/lost grasp before ordinary visual retry logic."""

        self._drawer_load_proof_failure = reason
        self._drawer_pull_mechanical_stop_observed = False
        self._set_phase("release")
        return self._tick(OSCAction.hold(-1.0), reason)

    def _retry_open_drawer(
        self,
        observation: RobotObservation,
        refreshed: ContactTarget,
    ) -> PolicyDecision:
        """Retry a visible, unchanged handle with diversified sensor geometry."""

        self._drawer_attempt += 1
        self._target = refreshed
        self._base_rotation = observation.proprio.T_world_ee[:3, :3].copy()
        drawer_point = refreshed.point_world.copy()
        drawer_point[2] += (
            self.config.drawer_grasp_z_offset_m
            + self._drawer_attempt * self.config.drawer_retry_grasp_z_delta_m
        )
        local_z = -refreshed.outward_world
        local_y = np.array([0.0, 0.0, 1.0])
        local_x = np.cross(local_y, local_z)
        local_x /= max(float(np.linalg.norm(local_x)), 1e-12)
        self._motion_rotation = np.column_stack((local_x, local_y, local_z))
        self._drawer_pull_direction = self._axis_aligned_fixture_direction(
            refreshed.outward_world
        )
        precontact = (
            drawer_point
            + refreshed.outward_world * self.config.drawer_precontact_clearance_m
        )
        lateral = -self._drawer_lateral_offset
        if float(np.linalg.norm(lateral)) < 1e-9:
            lateral = self._select_drawer_lateral(
                observation,
                precontact,
                drawer_point,
            )
        self._drawer_lateral_offset = lateral.copy()
        self._drawer_precontact_position = precontact.copy()
        self._motion_position = precontact + lateral
        self._motion_position[2] += self.config.drawer_safe_height_m
        self._force_baseline = observation.proprio.ee_force_sensor.copy()
        self._manipulation_progress_m = 0.0
        self._drawer_load_proof_progress_m = 0.0
        self._drawer_load_proof_loaded_width_m = None
        self._drawer_load_proof_settled_width_m = None
        self._drawer_load_proof_passed = False
        self._drawer_load_proof_failure = ""
        self._drawer_pull_grasp_lost = False
        self._drawer_pull_previous_width_m = None
        self._drawer_pull_width_plateau_ticks = 0
        self._drawer_pull_force_ticks = 0
        self._drawer_pull_mechanical_stop_observed = False
        visible_axis_bounds = self._observe_drawer_handle_axis_bounds(
            observation,
            refreshed,
        )
        self._drawer_rail_stop_visible_axis_bounds_m = (
            np.full(2, np.nan)
            if visible_axis_bounds is None
            else visible_axis_bounds
        )
        self._reset_drawer_rail_stop_motion_state()
        self._set_phase("move_safe")
        return self._move(
            observation,
            -1.0,
            "drawer unchanged in fresh RGB-D; retrying alternate handle contact",
        )

    def _prepare_push_attempt(self, observation: RobotObservation) -> None:
        assert self._push is not None
        direction = self._push.direction_world
        # Keep the currently reachable top-down wrist frame.  A retry seats
        # the same closed-finger rim pinch slightly inward and lower instead
        # of repeating an empty thin-edge closure.
        self._base_rotation = observation.proprio.T_world_ee[:3, :3].copy()
        self._motion_rotation = self._base_rotation.copy()
        retry_profile = min(self._push_attempt, 1)
        rim_fraction = (
            self.config.push_rim_radius_fraction
            - retry_profile * self.config.push_retry_rim_inset_fraction
        )
        rim_offset = (
            direction
            * self._push.object_radius_m
            * rim_fraction
        )
        contact = self._push.object_center_world + rim_offset
        contact[2] += (
            self.config.push_contact_z_offset_m
            - retry_profile * self.config.push_retry_contact_depth_m
        )
        self._push_contact_position = contact.copy()
        safe = contact.copy()
        safe[2] = max(
            observation.proprio.ee_position_world[2],
            contact[2] + self.config.push_safe_height_m,
        )
        self._force_baseline = observation.proprio.ee_force_sensor.copy()
        self._manipulation_progress_m = 0.0
        self._motion_position = safe
        self._set_phase("move_safe")

    def _start_staged_translation(
        self,
        observation: RobotObservation,
        destination: np.ndarray,
        phase: str,
    ) -> None:
        self._staged_destination_position = np.array(destination, copy=True)
        self._motion_position = self._next_translation_stage(
            observation.proprio.ee_position_world,
            self._staged_destination_position,
        )
        self._set_phase(phase)

    def _staged_translation_reached(
        self,
        observation: RobotObservation,
        *,
        tolerance_m: float,
        intermediate_tolerance_m: float | None = None,
    ) -> bool:
        final_stage = bool(
            np.linalg.norm(
                self._motion_position - self._staged_destination_position
            )
            <= 1e-9
        )
        # A short operational-space command can retain a steady Cartesian
        # residual near joint limits even while making useful progress.  Let
        # an intermediate waypoint advance within a bounded, contact-specific
        # tolerance; the final destination still uses the caller's tolerance
        # and every manipulation still requires its fresh visual verifier.
        if final_stage:
            stage_tolerance = tolerance_m
        elif intermediate_tolerance_m is None:
            stage_tolerance = min(
                tolerance_m,
                self.config.staged_waypoint_tolerance_m,
            )
        else:
            stage_tolerance = intermediate_tolerance_m
        if not self._position_reached(
            observation,
            self._motion_position,
            tolerance_m=stage_tolerance,
        ):
            return False
        remaining = float(
            np.linalg.norm(
                self._staged_destination_position
                - observation.proprio.ee_position_world
            )
        )
        if remaining <= tolerance_m:
            return True
        self._motion_position = self._next_translation_stage(
            observation.proprio.ee_position_world,
            self._staged_destination_position,
        )
        return False

    def _next_translation_stage(
        self,
        current: np.ndarray,
        destination: np.ndarray,
    ) -> np.ndarray:
        delta = destination - current
        distance = float(np.linalg.norm(delta))
        if distance <= self.config.staged_translation_step_m:
            return np.array(destination, copy=True)
        return np.array(
            current
            + delta / max(distance, 1e-12)
            * self.config.staged_translation_step_m,
            copy=True,
        )

    def _capture_target(self, observation: RobotObservation) -> None:
        assert self._target is not None
        self._base_rotation = observation.proprio.T_world_ee[:3, :3].copy()
        self._motion_rotation = self._base_rotation.copy()
        self._initial_feature_point = self._target.point_world.copy()
        self._initial_feature_axis = self._target.feature_axis_world.copy()
        self._force_baseline = observation.proprio.ee_force_sensor.copy()
        self._detection_misses = 0

    def _drawer_reference(self, level: str) -> ContactTarget | None:
        anchor = self._drawer_anchor
        if anchor is None or anchor.level != level:
            return None
        return anchor.target

    def _track_drawer(
        self,
        observation: RobotObservation,
        level: str,
    ) -> ContactTarget:
        reference = self._drawer_reference(level)
        if reference is None:
            return self.drawer_detector.detect(observation, level)
        return self.drawer_detector.track(observation, reference, level)

    def _drawer_preshape_gripper(self, observation: RobotObservation) -> float:
        width = observation.proprio.gripper_width_m
        lower = self.config.drawer_preshape_width_m - self.config.drawer_preshape_hysteresis_m
        upper = self.config.drawer_preshape_width_m + self.config.drawer_preshape_hysteresis_m
        if width < lower:
            return -1.0
        if width > upper:
            return 1.0
        return self.config.drawer_preshape_command

    def _drawer_pull_mechanical_stop(
        self,
        observation: RobotObservation,
    ) -> bool:
        """Recognise a late drawer rail stop from typed public proprioception.

        A Cartesian plateau alone is ambiguous: it can also be a kinematic
        wrist limit or a lost handle.  This gate is consequently restricted
        to the final 30 mm of an open-drawer pull and combines an aligned
        pose, low cross-axis drift, a stable retained pinch, and a sustained
        positive wrench-magnitude rise from the post-grasp baseline.  It only
        advances to release; fresh RGB-D remains mandatory for completion.
        """

        if self._phase != "pull":
            return False

        current = observation.proprio.ee_position_world
        target_error = float(np.linalg.norm(self._motion_position - current))
        if (
            self._previous_error is None
            or self._previous_error - target_error > self.config.progress_epsilon_m
        ):
            self._stall_ticks = 0
        else:
            self._stall_ticks += 1
        self._previous_error = target_error

        width = observation.proprio.gripper_width_m
        if (
            self._drawer_pull_previous_width_m is not None
            and abs(width - self._drawer_pull_previous_width_m)
            <= self.config.drawer_pull_stop_width_epsilon_m
        ):
            self._drawer_pull_width_plateau_ticks += 1
        else:
            self._drawer_pull_width_plateau_ticks = 0
        self._drawer_pull_previous_width_m = width

        # The wrist wrench is expressed in the rotating tool frame.  A
        # positive difference of norms is coordinate-invariant, whereas a
        # vector difference would fabricate load when only the frame turns.
        force_rise = (
            float(np.linalg.norm(observation.proprio.ee_force_sensor))
            - float(np.linalg.norm(self._force_baseline))
        )
        if force_rise >= self.config.contact_force_delta_n:
            self._drawer_pull_force_ticks += 1
        else:
            self._drawer_pull_force_ticks = 0

        direction = np.asarray(self._drawer_pull_direction, dtype=np.float64)
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        displacement = current - self._manipulation_start_position
        measured_progress = float(np.dot(displacement, direction))
        orthogonal_error = float(
            np.linalg.norm(displacement - direction * measured_progress)
        )
        rotation_error = self._rotation_error(
            observation.proprio.T_world_ee[:3, :3],
            self._motion_rotation,
        )

        return bool(
            self._phase_ticks >= self.config.contact_min_ticks
            and measured_progress
            >= self.config.drawer_pull_distance_m
            - self.config.drawer_pull_stop_max_residual_m
            and target_error <= self.config.drawer_pull_stop_max_residual_m
            and orthogonal_error <= self.config.drawer_pull_stop_max_orthogonal_m
            and rotation_error <= self.config.rotation_tolerance_rad
            and self._stall_ticks >= self.config.contact_stall_ticks
            and self._drawer_pull_width_plateau_ticks
            >= self.config.drawer_pull_stop_width_ticks
            and self._drawer_pull_force_ticks
            >= self.config.drawer_pull_stop_force_ticks
            and self.config.drawer_blocked_min_width_m
            <= width
            <= self.config.drawer_blocked_max_width_m
        )

    def _drawer_free_space_wrist_stalled(
        self,
        observation: RobotObservation,
    ) -> bool:
        """Detect a low-force Cartesian plateau before drawer contact.

        This gate is intentionally restricted to the known free-space
        ``move_precontact`` phase.  It consumes only public end-effector pose
        and wrench observations and can request at most one recovery for the
        current language skill.  A contact or a mostly aligned wrist never
        takes this path.
        """

        if self._drawer_wrist_recovery_used or self._phase != "move_precontact":
            return False
        error = float(
            np.linalg.norm(
                self._motion_position
                - observation.proprio.ee_position_world
            )
        )
        if (
            not np.isfinite(self._drawer_free_space_best_error_m)
            or error
            <= self._drawer_free_space_best_error_m
            - self.config.progress_epsilon_m
        ):
            self._drawer_free_space_best_error_m = error
            self._drawer_free_space_stall_ticks = 0
        else:
            self._drawer_free_space_stall_ticks += 1

        rotation_error = self._rotation_error(
            observation.proprio.T_world_ee[:3, :3],
            self._motion_rotation,
        )
        # MuJoCo exposes the wrist force in the rotating sensor frame.  Direct
        # vector subtraction therefore reports roughly twice the tool weight
        # after a quarter-turn even in empty space.  Magnitude is invariant to
        # that frame rotation, while a genuine load large enough to forbid a
        # free-space wrist recovery still changes it and fails this gate.
        force_delta = abs(
            float(np.linalg.norm(observation.proprio.ee_force_sensor))
            - float(np.linalg.norm(self._force_baseline))
        )
        return bool(
            self._phase_ticks >= self.config.drawer_wrist_recovery_stall_ticks
            and self._drawer_free_space_stall_ticks
            >= self.config.drawer_wrist_recovery_stall_ticks
            and rotation_error
            >= self.config.drawer_wrist_recovery_min_rotation_error_rad
            and force_delta < self.config.contact_force_delta_n
        )

    def _start_drawer_wrist_recovery(
        self,
        observation: RobotObservation,
    ) -> PolicyDecision:
        """Lift and select the exact pi-yaw equivalent of a drawer grip."""

        self._drawer_wrist_recovery_used = True
        self._drawer_wrist_recovery_return_position = (
            self._motion_position.copy()
        )
        # Flipping local X and Y preserves the same tool-Z and the same
        # unoriented jaw-width line.  It changes no sensor target or contact
        # geometry, but gives the redundant arm a different wrist solution.
        equivalent = self._motion_rotation @ np.diag((-1.0, -1.0, 1.0))
        self._drawer_wrist_recovery_rotation = equivalent
        self._motion_rotation = (
            observation.proprio.T_world_ee[:3, :3].copy()
        )
        self._motion_position = observation.proprio.ee_position_world.copy()
        self._motion_position[2] = max(
            self._motion_position[2] + self.config.drawer_wrist_recovery_lift_m,
            self._drawer_wrist_recovery_return_position[2]
            + self.config.drawer_safe_height_m,
        )
        self._set_phase("drawer_wrist_recovery_lift")
        return self._move(
            observation,
            self._drawer_preshape_gripper(observation),
            "free-space drawer waypoint stalled; lifting for an equivalent wrist frame",
        )

    def _drawer_pusher_gripper(self, observation: RobotObservation) -> float:
        """Hold a compact measured pusher without collapsing the fingers."""

        width = observation.proprio.gripper_width_m
        lower = (
            self.config.drawer_close_pusher_width_m
            - self.config.drawer_close_pusher_hysteresis_m
        )
        upper = (
            self.config.drawer_close_pusher_width_m
            + self.config.drawer_close_pusher_hysteresis_m
        )
        if width < lower:
            return -1.0
        if width > upper:
            return 1.0
        return self.config.drawer_preshape_command

    def _drawer_close_pusher_rotation(self, level: str | None) -> np.ndarray:
        """Incline a lower-front pusher to keep the palm outside upper handles.

        The fingertip position remains the observed drawer front. Tilting the
        tool toward the inward push direction puts the palm behind that point,
        while keeping a downward component for the existing high approach.
        """
        vertical = self._drawer_close_vertical_rotation()
        if level not in {"middle", "bottom"}:
            return vertical
        assert self._target is not None
        inward = -self._axis_aligned_fixture_direction(self._target.outward_world)
        tool_z = vertical[:, 2] + inward
        tool_z /= max(float(np.linalg.norm(tool_z)), 1e-12)
        jaw = vertical[:, 1] - tool_z * float(vertical[:, 1] @ tool_z)
        jaw /= max(float(np.linalg.norm(jaw)), 1e-12)
        return np.column_stack((np.cross(jaw, tool_z), jaw, tool_z))

    def _drawer_close_vertical_rotation(self) -> np.ndarray:
        """Keep a downward wrist while aligning both fingertips to the front.

        The frozen handle axis is an RGB-D feature.  Projecting it into the
        reset tool plane changes only wrist yaw and avoids commanding a low
        Cartesian wrist flip beside the cabinet.
        """

        assert self._target is not None
        tool_z = np.asarray(self._base_rotation[:, 2], dtype=np.float64).copy()
        tool_z /= max(float(np.linalg.norm(tool_z)), 1e-12)
        jaw = np.asarray(
            self._target.feature_axis_world,
            dtype=np.float64,
        ).copy()
        jaw -= tool_z * float(np.dot(jaw, tool_z))
        jaw_norm = float(np.linalg.norm(jaw))
        if not np.isfinite(jaw_norm) or jaw_norm < 1e-8:
            raise ValueError("drawer handle axis cannot define a vertical pusher yaw")
        jaw /= jaw_norm
        if float(np.dot(jaw, self._base_rotation[:, 1])) < 0.0:
            jaw *= -1.0
        tool_x = np.cross(jaw, tool_z)
        tool_x /= max(float(np.linalg.norm(tool_x)), 1e-12)
        jaw = np.cross(tool_z, tool_x)
        return np.column_stack((tool_x, jaw, tool_z))

    def _microwave_grasp_rotation(
        self,
        target: ContactTarget,
        reference_rotation: np.ndarray,
    ) -> np.ndarray:
        """Build the nearest two-pad frame around a sensed vertical feature."""

        outward = self._axis_aligned_fixture_direction(target.outward_world)
        local_z = -outward
        feature = np.asarray(target.feature_axis_world, dtype=np.float64).copy()
        feature -= local_z * float(np.dot(feature, local_z))
        feature_norm = float(np.linalg.norm(feature))
        if feature_norm < 1e-8:
            raise ValueError("microwave feature cannot define a two-pad grasp frame")
        feature /= feature_norm
        # Panda local Y is the jaw-width axis.  It must cross, rather than run
        # along, the vertical handle/edge so the two pads land on opposite
        # surfaces.
        local_x = feature
        local_y = np.cross(local_z, local_x)
        local_y /= max(float(np.linalg.norm(local_y)), 1e-12)
        contact = np.column_stack((local_x, local_y, local_z))
        mirrored = contact @ np.diag((-1.0, -1.0, 1.0))
        reference = np.asarray(reference_rotation, dtype=np.float64)
        return min(
            (contact, mirrored),
            key=lambda candidate: self._rotation_error(reference, candidate),
        ).copy()

    def _microwave_reacquire_open_edge(
        self,
        observation: RobotObservation,
    ) -> ContactTarget:
        """Reassociate the moving vertical edge inside a frozen local gate.

        The production Route-B wrapper exposes the original RGB-D appliance
        OBB and its reusable handle detector.  Prefer that detector's local
        anchor plus frozen hinge-circle mode.  Minimal test/detector adapters
        can fall back to their ordinary fresh track, but receive the same
        episode-anchor distance check.
        """

        assert self._target is not None
        detector = self.microwave_detector
        if detector is None:
            raise LookupError("no microwave detector is configured")
        fixture = getattr(detector, "_fixture_geometry", None)
        reference_ee = getattr(detector, "_reference_ee_position_world", None)
        handle_detector = getattr(detector, "handle_detector", None)
        detected: ContactTarget | None = None
        used_local_gate = False
        if (
            fixture is not None
            and reference_ee is not None
            and handle_detector is not None
        ):
            detect = getattr(handle_detector, "detect", None)
            if callable(detect):
                initial_radius = self._microwave_arc_start_position - self._microwave_hinge_position
                initial_radius -= self._microwave_rotation_axis * float(
                    np.dot(initial_radius, self._microwave_rotation_axis)
                )
                frozen_radius = float(np.linalg.norm(initial_radius))
                try:
                    detected = detect(
                        observation,
                        *fixture,
                        True,
                        reference_ee_position_world=reference_ee,
                        local_anchor_world=self._microwave_open_regrasp_anchor,
                        local_anchor_radius_m=self.config.microwave_open_regrasp_anchor_radius_m,
                        frozen_hinge_world=self._microwave_hinge_position,
                        frozen_rotation_axis_world=self._microwave_rotation_axis,
                        frozen_radius_m=frozen_radius,
                        frozen_radius_tolerance_m=(
                            self.config.microwave_open_regrasp_radius_tolerance_m
                        ),
                    )
                    used_local_gate = True
                except TypeError:
                    # Compatibility is intentionally narrow: only an older
                    # detector signature may use the ordinary fresh tracker.
                    detected = None
        if detected is None:
            track = getattr(detector, "track", None)
            if not callable(track):
                raise LookupError("microwave detector cannot refresh an open edge")
            detected = track(
                observation,
                self._target,
                GoalSkillKind.OPEN_MICROWAVE,
            )

        anchor_error = float(
            np.linalg.norm(detected.point_world - self._microwave_open_regrasp_anchor)
        )
        if anchor_error > self.config.microwave_open_regrasp_anchor_radius_m:
            raise LookupError("fresh microwave edge left its predicted local anchor")
        radial = detected.point_world - self._microwave_hinge_position
        radial -= self._microwave_rotation_axis * float(
            np.dot(radial, self._microwave_rotation_axis)
        )
        origin_radial = self._microwave_arc_start_position - self._microwave_hinge_position
        origin_radial -= self._microwave_rotation_axis * float(
            np.dot(origin_radial, self._microwave_rotation_axis)
        )
        radius_error = abs(
            float(np.linalg.norm(radial)) - float(np.linalg.norm(origin_radial))
        )
        if radius_error > self.config.microwave_open_regrasp_radius_tolerance_m:
            raise LookupError("fresh microwave edge left the frozen hinge circle")
        if detected.confidence < self.config.microwave_visual_min_confidence:
            raise LookupError("fresh microwave edge confidence was insufficient")

        # The fresh image proves that the same moving vertical edge remains in
        # the local hinge gate.  Its visible-surface median can lag the true
        # grip centre under self-occlusion, whereas the last retained two-pad
        # grip-site is a direct public proprioceptive contact observation.
        # Project that frozen contact back onto the same sensor-derived hinge
        # circle; do not turn the image median into an unmeasured 3-D offset.
        anchor_radial = (
            self._microwave_open_regrasp_anchor
            - self._microwave_hinge_position
        )
        anchor_height = float(
            np.dot(anchor_radial, self._microwave_rotation_axis)
        )
        anchor_radial -= self._microwave_rotation_axis * anchor_height
        anchor_norm = float(np.linalg.norm(anchor_radial))
        origin_norm = float(np.linalg.norm(origin_radial))
        if anchor_norm < 1e-8:
            raise LookupError("propagated microwave contact anchor was degenerate")
        propagated_point = (
            self._microwave_hinge_position
            + anchor_radial * (origin_norm / anchor_norm)
            + self._microwave_rotation_axis * anchor_height
        )
        # The vertical handle is cylindrical, so a later grasp need not inherit
        # the door's accumulated yaw.  Re-enter from the frozen appliance-front
        # side observed at reset; each segment then spends only its bounded
        # incremental wrist rotation instead of driving the arm back into the
        # same absolute-yaw limit.  The rotated door normal remains the retreat
        # direction above, where it provides collision clearance.
        outward = self._microwave_initial_outward.copy()
        append_diagnostic = getattr(detector, "_append_selector_diagnostic", None)
        if callable(append_diagnostic):
            append_diagnostic(
                {
                    "kind": "route_b_microwave_local_reacquisition",
                    "point_world_m": detected.point_world.tolist(),
                    "predicted_anchor_world_m": (
                        self._microwave_open_regrasp_anchor.tolist()
                    ),
                    "anchor_error_m": anchor_error,
                    "frozen_radius_error_m": radius_error,
                    "propagated_contact_world_m": propagated_point.tolist(),
                    "contact_position_source": (
                        "last_retained_grip_site_projected_to_rgbd_hinge_circle"
                    ),
                    "local_hinge_gate_used": used_local_gate,
                    "source_cameras": list(detected.source_cameras),
                }
            )
        return ContactTarget(
            GoalSkillKind.OPEN_MICROWAVE,
            propagated_point,
            detected.axis_world,
            outward,
            detected.fixture_center_world,
            detected.feature_axis_world,
            detected.confidence,
            detected.source_cameras,
        )

    def _microwave_rotary_geometry(
        self,
        *,
        initial_is_open: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Read optional articulation geometry reconstructed from RGB-D.

        Route B wraps the reusable microwave feature detector and freezes its
        appliance OBB plus reset-time EE sign.  Accept either a detector that
        exposes articulation geometry directly or that standard wrapper.  No
        simulator joint/body state is available through either path.
        """

        detector = self.microwave_detector
        if detector is None:
            return None
        fixture = getattr(detector, "_fixture_geometry", None)
        reference_ee = getattr(
            detector,
            "_reference_ee_position_world",
            None,
        )
        providers = (
            detector,
            getattr(detector, "handle_detector", None),
        )
        for provider in providers:
            geometry = getattr(provider, "articulation_geometry", None)
            if not callable(geometry):
                continue
            values = None
            state_aware_geometry = False
            attempts: list[tuple[tuple[object, ...], dict[str, object]]] = []
            if fixture is not None and reference_ee is not None:
                if self._target is not None:
                    attempts.append(
                        (
                            (reference_ee, *fixture),
                            {
                                "observed_handle_world": self._target.point_world,
                                "initial_is_open": initial_is_open,
                            },
                        )
                    )
                    attempts.append(
                        (
                            (reference_ee, *fixture),
                            {"observed_handle_world": self._target.point_world},
                        )
                    )
                # Keep compatibility with older sensor providers that expose
                # the same frozen RGB-D fixture but predate the optional
                # observed-handle argument.
                attempts.append(((reference_ee, *fixture), {}))
            attempts.append(((), {}))
            for args, kwargs in attempts:
                try:
                    values = geometry(*args, **kwargs)
                    state_aware_geometry = "initial_is_open" in kwargs
                    break
                except (TypeError, ValueError, LookupError):
                    continue
            if values is None:
                continue
            if not isinstance(values, tuple) or len(values) != 3:
                continue
            hinge, axis, closed_slot = (
                np.asarray(value, dtype=np.float64) for value in values
            )
            if (
                hinge.shape != (3,)
                or axis.shape != (3,)
                or closed_slot.shape != (3,)
                or not np.all(np.isfinite(hinge))
                or not np.all(np.isfinite(axis))
                or not np.all(np.isfinite(closed_slot))
            ):
                continue
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm < 1e-8:
                continue
            axis = axis / axis_norm

            # An appliance OBB has two long-side/front corners that are both
            # geometrically valid hinge hypotheses.  Reset-time EE signing can
            # choose the wrong one when the arm starts diagonally from an open
            # door.  For the standard Route-B wrapper, mirror the provider's
            # hypothesis across the sensed OBB long axis and retain the
            # feasible candidate closest to the *currently visible* door
            # edge.  This disambiguation uses only the frozen RGB-D body OBB
            # and the current RGB-D contact feature; it does not assume a task
            # instance, joint pose, or simulator object identity.
            candidates = [(hinge.copy(), closed_slot.copy())]
            if (
                not state_aware_geometry
                and fixture is not None
                and self._target is not None
            ):
                center, axes, half_extents = (
                    np.asarray(value, dtype=np.float64) for value in fixture
                )
                if (
                    center.shape == (3,)
                    and axes.shape == (3, 3)
                    and half_extents.shape == (3,)
                    and np.all(np.isfinite(center))
                    and np.all(np.isfinite(axes))
                    and np.all(np.isfinite(half_extents))
                ):
                    vertical_index = int(np.argmax(np.abs(axes.T @ axis)))
                    horizontal_indices = [
                        index for index in range(3) if index != vertical_index
                    ]
                    if horizontal_indices:
                        long_index = max(
                            horizontal_indices,
                            key=lambda index: float(half_extents[index]),
                        )
                        long_axis = axes[:, long_index].copy()
                        long_axis -= axis * float(np.dot(long_axis, axis))
                        long_norm = float(np.linalg.norm(long_axis))
                        if long_norm >= 1e-8:
                            long_axis /= long_norm
                            mirrored_hinge = hinge - 2.0 * long_axis * float(
                                np.dot(hinge - center, long_axis)
                            )
                            mirrored_slot = closed_slot - 2.0 * long_axis * float(
                                np.dot(closed_slot - center, long_axis)
                            )
                            candidates.append((mirrored_hinge, mirrored_slot))

            observed = (
                None
                if self._target is None
                else np.asarray(self._target.point_world, dtype=np.float64)
            )
            radius_min, radius_max = self.config.microwave_hinge_radius_range_m
            feasible: list[tuple[float, np.ndarray, np.ndarray]] = []
            for candidate_hinge, candidate_slot in candidates:
                start_radius = (
                    candidate_slot - candidate_hinge
                    if observed is None
                    else observed - candidate_hinge
                )
                start_radius -= axis * float(np.dot(start_radius, axis))
                goal_radius = candidate_slot - candidate_hinge
                goal_radius -= axis * float(np.dot(goal_radius, axis))
                start_norm = float(np.linalg.norm(start_radius))
                goal_norm = float(np.linalg.norm(goal_radius))
                if (
                    np.all(np.isfinite(candidate_hinge))
                    and np.all(np.isfinite(candidate_slot))
                    and radius_min <= start_norm <= radius_max
                    and radius_min <= goal_norm <= radius_max
                ):
                    feasible.append(
                        (start_norm, candidate_hinge.copy(), candidate_slot.copy())
                    )
            if feasible:
                _, hinge, closed_slot = min(feasible, key=lambda item: item[0])
            append_diagnostic = getattr(
                detector,
                "_append_selector_diagnostic",
                None,
            )
            if callable(append_diagnostic):
                append_diagnostic(
                    {
                        "kind": "route_b_microwave_articulation_geometry",
                        "hinge_world_m": hinge.tolist(),
                        "closed_slot_world_m": closed_slot.tolist(),
                        "rotation_axis_world": axis.tolist(),
                        "observed_handle_world_m": (
                            None if observed is None else observed.tolist()
                        ),
                        "candidate_count": len(candidates),
                        "state_aware_geometry": state_aware_geometry,
                        "initial_door_state": (
                            "open" if initial_is_open else "closed"
                        ),
                    }
                )
            return hinge.copy(), axis.copy(), closed_slot.copy()
        return None

    def _set_microwave_arc_waypoint(self) -> None:
        """Set one bounded hinge-arc waypoint from frozen public geometry."""

        fraction = self._microwave_arc_index / self._microwave_arc_segments
        partial_angle = self._microwave_arc_chunk_angle_rad * fraction
        rotation = Rotation.from_rotvec(
            self._microwave_rotation_axis * partial_angle
        ).as_matrix()
        self._motion_position = (
            self._microwave_hinge_position
            + rotation
            @ (
                self._microwave_arc_start_position
                - self._microwave_hinge_position
            )
        )
        cumulative_angle = (
            self._microwave_arc_completed_angle_rad + partial_angle
        )
        wrist_angle = np.sign(cumulative_angle) * min(
            abs(cumulative_angle),
            self.config.microwave_wrist_corotation_limit_rad,
        )
        wrist_rotation = Rotation.from_rotvec(
            self._microwave_rotation_axis * wrist_angle
        ).as_matrix()
        self._motion_rotation = (
            wrist_rotation @ self._microwave_arc_start_rotation
        )

    def _microwave_measured_chunk_angle(self, current_position: np.ndarray) -> float:
        """Measure signed public EE progress around the frozen RGB-D hinge."""

        start_radius = (
            self._microwave_arc_start_position
            - self._microwave_hinge_position
        )
        current_radius = (
            np.asarray(current_position, dtype=np.float64)
            - self._microwave_hinge_position
        )
        for radius in (start_radius, current_radius):
            radius -= self._microwave_rotation_axis * float(
                np.dot(radius, self._microwave_rotation_axis)
            )
        return float(
            np.arctan2(
                np.dot(
                    self._microwave_rotation_axis,
                    np.cross(start_radius, current_radius),
                ),
                np.dot(start_radius, current_radius),
            )
        )

    def _microwave_opening_tangent(self, position: np.ndarray) -> np.ndarray:
        """Return the commanded opening tangent at a public 3-D contact point."""

        radial = (
            np.asarray(position, dtype=np.float64)
            - self._microwave_hinge_position
        )
        radial -= self._microwave_rotation_axis * float(
            np.dot(radial, self._microwave_rotation_axis)
        )
        tangent = np.cross(self._microwave_rotation_axis, radial)
        tangent *= np.sign(self._microwave_arc_angle_rad)
        norm = float(np.linalg.norm(tangent))
        if norm < 1e-8:
            raise ValueError("microwave contact cannot define an opening tangent")
        return tangent / norm

    def _release_microwave_open_compact_loss(
        self,
        observation: RobotObservation,
        message: str,
    ) -> PolicyDecision:
        """Bound a lost compact pusher and reacquire it from fresh public RGB-D."""

        self._microwave_open_push_recompact_ticks = 0
        self._microwave_open_push_recompact_pending = False
        self._microwave_open_push_progress_samples = []
        if (
            self._microwave_open_push_compact_recoveries
            >= self.config.microwave_open_push_max_compact_recoveries
        ):
            return self._fail(
                "back-side microwave compact-loss recovery budget exhausted"
            )
        try:
            tangent = self._microwave_opening_tangent(
                observation.proprio.ee_position_world
            )
        except ValueError as exc:
            return self._fail(str(exc))
        self._microwave_open_push_compact_recoveries += 1
        self._microwave_open_regrasp_anchor = (
            observation.proprio.ee_position_world.copy()
        )
        self._microwave_open_regrasp_outward = -tangent
        self._microwave_open_release_escape_active = False
        self._microwave_open_push_finalizing = False
        self._set_phase("microwave_open_segment_release")
        return self._tick(OSCAction.hold(-1.0), message)

    def _release_microwave_open_push_segment(
        self,
        observation: RobotObservation,
        message: str,
    ) -> PolicyDecision:
        """Safely leave one useful bounded pusher segment for fresh RGB-D."""

        try:
            tangent = self._microwave_opening_tangent(
                observation.proprio.ee_position_world
            )
        except ValueError as exc:
            return self._fail(str(exc))
        self._microwave_open_push_segments_completed += 1
        self._microwave_open_regrasp_anchor = (
            observation.proprio.ee_position_world.copy()
        )
        self._microwave_open_regrasp_outward = -tangent
        self._microwave_open_release_escape_active = False
        self._microwave_open_push_finalizing = False
        self._microwave_open_push_progress_samples = []
        self._set_phase("microwave_open_segment_release")
        return self._tick(OSCAction.hold(-1.0), message)

    def _microwave_close_push_frame(
        self,
        position: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Return radial, exterior normal, and closing tangent from RGB-D geometry."""

        radial = (
            np.asarray(position, dtype=np.float64)
            - self._microwave_hinge_position
        )
        radial -= self._microwave_rotation_axis * float(
            np.dot(radial, self._microwave_rotation_axis)
        )
        radius = float(np.linalg.norm(radial))
        if not np.isfinite(radius) or radius < 1e-8:
            raise ValueError("microwave close pusher had a degenerate hinge radius")
        radial /= radius
        direction_sign = float(np.sign(self._microwave_arc_angle_rad))
        if direction_sign == 0.0:
            raise ValueError("microwave close pusher lacked a signed hinge arc")
        closing_tangent = np.cross(self._microwave_rotation_axis, radial)
        closing_tangent *= direction_sign
        tangent_norm = float(np.linalg.norm(closing_tangent))
        if tangent_norm < 1e-8:
            raise ValueError("microwave close pusher had a degenerate tangent")
        closing_tangent /= tangent_norm
        outward = -closing_tangent
        frozen_outward = self._microwave_close_retreat_outward.copy()
        frozen_outward -= self._microwave_rotation_axis * float(
            np.dot(frozen_outward, self._microwave_rotation_axis)
        )
        frozen_norm = float(np.linalg.norm(frozen_outward))
        if frozen_norm < 1e-8:
            raise ValueError("microwave close pusher lacked a frozen exterior sign")
        frozen_outward /= frozen_norm
        if float(np.dot(outward, frozen_outward)) <= 0.0:
            raise ValueError(
                "microwave close pusher exterior normal contradicted the frozen RGB-D sign"
            )
        return radial, outward, closing_tangent, radius

    def _start_microwave_close_face_push(
        self,
        observation: RobotObservation,
        *,
        aligned_angle: float,
    ) -> PolicyDecision:
        """Release the retained edge before it enters the appliance frame."""

        try:
            radial, outward, _closing, _radius = (
                self._microwave_close_push_frame(
                    observation.proprio.ee_position_world
                )
            )
        except ValueError as exc:
            return self._fail(str(exc))
        target_angle = abs(self._microwave_close_target_angle_rad)
        remaining = target_angle - aligned_angle
        if (
            remaining < -self.config.microwave_close_push_goal_tolerance_rad
            or remaining
            > self.config.microwave_close_push_transition_remaining_rad + 1e-9
        ):
            return self._fail(
                "microwave close pusher transition left its sensor-angle window"
            )
        self._microwave_close_push_mode = True
        self._microwave_close_push_release_anchor = (
            observation.proprio.ee_position_world.copy()
        )
        # The local reassociation gate is centred on the last public retained
        # edge pose; fresh RGB-D, not this proprioceptive anchor, must still
        # supply the moving edge before a face contact can be emitted.
        self._microwave_open_regrasp_anchor = (
            self._microwave_close_push_release_anchor.copy()
        )
        self._microwave_close_push_radial = radial
        self._microwave_close_push_outward = outward
        self._microwave_close_push_best_angle_rad = aligned_angle
        self._motion_position = self._microwave_close_push_release_anchor.copy()
        self._motion_rotation = observation.proprio.T_world_ee[:3, :3].copy()
        self._set_phase("microwave_close_push_release")
        return self._tick(
            OSCAction.hold(-1.0),
            "releasing the retained microwave edge before entering the front frame",
        )

    def _prepare_microwave_close_face_contact(
        self,
        observation: RobotObservation,
    ) -> PolicyDecision:
        """Use a fresh locally associated edge to stage an exterior-face pusher."""

        try:
            self._target = self._microwave_reacquire_open_edge(observation)
            radial, outward, _closing, radius = (
                self._microwave_close_push_frame(self._target.point_world)
            )
        except (LookupError, ValueError) as exc:
            return self._detection_miss(str(exc), gripper=-1.0)
        if float(np.dot(outward, self._microwave_close_push_outward)) <= 0.0:
            return self._fail(
                "fresh microwave edge reversed the frozen exterior half-space"
            )
        pusher_radius = radius - self.config.microwave_close_push_contact_inset_m
        radius_min, _radius_max = self.config.microwave_hinge_radius_range_m
        if pusher_radius <= radius_min:
            return self._fail(
                "microwave close face inset left no safe hinge radius"
            )
        self._detection_misses = 0
        self._microwave_close_push_radial = radial
        self._microwave_close_push_outward = outward
        self._microwave_close_push_contact_point = (
            self._target.point_world
            - radial * self.config.microwave_close_push_contact_inset_m
        )
        self._microwave_close_push_precontact_position = (
            self._microwave_close_push_contact_point
            + outward * self.config.microwave_close_push_precontact_m
        )
        self._microwave_close_push_radius_m = pusher_radius
        self._motion_rotation = observation.proprio.T_world_ee[:3, :3].copy()
        self._set_phase("microwave_close_push_preshape")
        return self._tick(
            OSCAction.hold(1.0),
            "fresh RGB-D edge retained; compacting outside the microwave face",
        )

    def _microwave_close_push_precontact_loaded(
        self,
        observation: RobotObservation,
    ) -> float | None:
        """Return the public EE hinge radius for a typed early face contact.

        The sensed point denotes the door surface, whereas LIBERO reports the
        Panda grip-site pose between the fingers.  A compact fingertip can
        therefore meet the door while the grip site retains a bounded
        Cartesian residual.  Treat that as contact only after repeated low
        progress and a rotation-invariant force-magnitude rise.  The current
        EE radius is returned because it, rather than the surface-point
        radius, is the circle that the Cartesian pusher must subsequently
        preserve.  No simulator contact, joint, or task predicate is read.
        """

        if self._phase != "microwave_close_push_precontact":
            return None
        error = float(
            np.linalg.norm(
                self._motion_position
                - observation.proprio.ee_position_world
            )
        )
        if (
            self._previous_error is None
            or self._previous_error - error
            > self.config.microwave_close_push_precontact_progress_epsilon_m
        ):
            self._stall_ticks = 0
        else:
            self._stall_ticks += 1
        self._previous_error = error

        if (
            error
            > self.config.microwave_close_push_precontact_contact_residual_m
            or self._stall_ticks
            < self.config.microwave_close_push_precontact_stall_ticks
            or self._rotation_error(
                observation.proprio.T_world_ee[:3, :3],
                self._motion_rotation,
            )
            > self.config.microwave_safe_rotation_tolerance_rad
        ):
            return None
        baseline_force = float(np.linalg.norm(self._force_baseline))
        current_force = float(
            np.linalg.norm(observation.proprio.ee_force_sensor)
        )
        if current_force - baseline_force < self.config.contact_force_delta_n:
            return None
        try:
            _radial, _outward, _closing, radius = (
                self._microwave_close_push_frame(
                    observation.proprio.ee_position_world
                )
            )
        except ValueError:
            return None
        if (
            abs(radius - self._microwave_close_push_radius_m)
            > self.config.microwave_close_push_precontact_contact_residual_m
        ):
            return None
        return radius

    def _enter_microwave_close_push_door(
        self,
        observation: RobotObservation,
        *,
        ee_radius_m: float | None = None,
    ) -> None:
        """Initialize the exterior-face tangent controller after contact."""

        if ee_radius_m is not None:
            self._microwave_close_push_radius_m = float(ee_radius_m)
        measured = self._microwave_measured_chunk_angle(
            observation.proprio.ee_position_world
        )
        aligned = np.sign(self._microwave_arc_angle_rad) * measured
        self._microwave_close_push_best_angle_rad = aligned
        self._microwave_close_push_start_angle_rad = aligned
        self._manipulation_start_position = (
            observation.proprio.ee_position_world.copy()
        )
        self._manipulation_progress_m = 0.0
        self._microwave_previous_width_m = observation.proprio.gripper_width_m
        self._microwave_width_plateau_ticks = 0
        self._microwave_previous_force_n = (
            observation.proprio.ee_force_sensor.copy()
        )
        self._microwave_force_plateau_ticks = 0
        self._set_phase("microwave_close_push_door")

    def _start_microwave_close_push_terminal_load(
        self,
        observation: RobotObservation,
        *,
        aligned_angle: float,
    ) -> PolicyDecision:
        """Continue past the slot proxy until a public mechanical stop is seen.

        The EE angle can lead the physical door while pushing its exterior
        surface.  Crossing the frozen closed-slot angle therefore changes the
        command from a rolling tangent to a bounded face-normal load; it does
        not authorise release or success.
        """

        if (
            observation.proprio.gripper_width_m
            > self.config.microwave_close_push_compact_width_m
        ):
            return self._fail(
                "microwave close face pusher opened before terminal loading"
            )
        try:
            _radial, _outward, closing, _radius = (
                self._microwave_close_push_frame(
                    observation.proprio.ee_position_world
                )
            )
        except ValueError as exc:
            return self._fail(str(exc))
        face_normal = -self._microwave_close_push_outward.copy()
        face_normal -= self._microwave_rotation_axis * float(
            np.dot(face_normal, self._microwave_rotation_axis)
        )
        face_norm = float(np.linalg.norm(face_normal))
        if face_norm < 1e-8:
            return self._fail(
                "microwave close terminal load lacked a frozen face normal"
            )
        face_normal /= face_norm
        if float(np.dot(face_normal, closing)) <= 0.0:
            return self._fail(
                "microwave close terminal face normal opposed the closing tangent"
            )

        self._microwave_close_push_terminal_anchor = (
            observation.proprio.ee_position_world.copy()
        )
        self._microwave_close_push_terminal_start_angle_rad = aligned_angle
        self._microwave_close_push_terminal_face_normal = face_normal
        self._motion_position = (
            self._microwave_close_push_terminal_anchor
            + closing * self.config.microwave_close_push_terminal_step_m
        )
        self._motion_rotation = observation.proprio.T_world_ee[:3, :3].copy()
        self._set_phase("microwave_close_push_terminal_load")
        # A stop must be newly established against the bounded terminal
        # command, rather than inherited from a moving tangent waypoint.
        self._microwave_previous_width_m = observation.proprio.gripper_width_m
        self._microwave_width_plateau_ticks = 0
        self._microwave_previous_force_n = (
            observation.proprio.ee_force_sensor.copy()
        )
        self._microwave_force_plateau_ticks = 0
        return self._move(
            observation,
            1.0,
            "slot-angle cue reached; applying bounded microwave face-normal load",
        )

    def _start_microwave_close_push_exit(
        self,
        observation: RobotObservation,
        *,
        mechanical_stop: bool,
    ) -> PolicyDecision:
        """Keep fingers compact and slide beyond the free edge before release."""

        if not mechanical_stop:
            return self._fail(
                "microwave close pusher exit lacked a public mechanical stop"
            )
        try:
            radial, outward, _closing, _radius = (
                self._microwave_close_push_frame(
                    observation.proprio.ee_position_world
                )
            )
        except ValueError as exc:
            return self._fail(str(exc))
        self._microwave_close_push_mechanical_stop = bool(mechanical_stop)
        self._microwave_close_push_radial = radial
        self._microwave_close_push_outward = outward
        self._microwave_close_push_exit_anchor = (
            observation.proprio.ee_position_world.copy()
        )
        lift_axis = self._microwave_rotation_axis.copy()
        if lift_axis[2] < 0.0:
            lift_axis *= -1.0
        self._motion_position = (
            self._microwave_close_push_exit_anchor
            + radial * self.config.microwave_close_push_exit_radial_m
            + lift_axis * self.config.microwave_close_push_exit_lift_m
        )
        self._motion_rotation = observation.proprio.T_world_ee[:3, :3].copy()
        self._set_phase("microwave_close_push_radial_exit")
        return self._move(
            observation,
            1.0,
            "closed microwave face pusher sliding beyond the free edge",
        )

    def _act_microwave_close_face_pusher(
        self,
        step: GoalSkillStep,
        observation: RobotObservation,
    ) -> PolicyDecision:
        """Run the sensor-only compact face-pusher and collision-safe exit."""

        if (
            step.kind is not GoalSkillKind.CLOSE_MICROWAVE
            or not self._microwave_close_push_mode
        ):
            return self._fail("microwave close-pusher phase lacked a CLOSE binding")

        if self._phase == "microwave_close_push_release":
            if self._phase_ticks < self.config.release_ticks:
                return self._tick(OSCAction.hold(-1.0))
            lift_axis = self._microwave_rotation_axis.copy()
            if lift_axis[2] < 0.0:
                lift_axis *= -1.0
            self._motion_position = (
                self._microwave_close_push_release_anchor
                + self._microwave_close_push_radial
                * self.config.microwave_close_push_release_radial_m
                + self._microwave_close_push_outward
                * self.config.microwave_close_push_release_outward_m
                + lift_axis * self.config.microwave_close_push_release_lift_m
            )
            self._set_phase("microwave_close_push_clear_edge")
            return self._move(
                observation,
                -1.0,
                "opening while clearing the microwave free edge in its sensor frame",
            )

        if self._phase == "microwave_close_push_clear_edge":
            delta = (
                observation.proprio.ee_position_world
                - self._microwave_close_push_release_anchor
            )
            radial_progress = float(
                np.dot(delta, self._microwave_close_push_radial)
            )
            outward_progress = float(
                np.dot(delta, self._microwave_close_push_outward)
            )
            topology_clear = bool(
                radial_progress
                >= self.config.microwave_close_push_release_radial_m
                - self.config.microwave_close_push_waypoint_tolerance_m
                and outward_progress
                >= self.config.microwave_close_push_release_outward_m
                - self.config.microwave_close_push_waypoint_tolerance_m
            )
            pose_clear = bool(
                self._position_reached(
                    observation,
                    self._motion_position,
                    tolerance_m=self.config.microwave_close_push_waypoint_tolerance_m,
                )
                and self._rotation_error(
                    observation.proprio.T_world_ee[:3, :3],
                    self._motion_rotation,
                )
                <= self.config.microwave_safe_rotation_tolerance_rad
            )
            if not (topology_clear and pose_clear):
                return self._move(observation, -1.0)
            if (
                observation.proprio.gripper_width_m
                < self.config.drawer_release_width_m
            ):
                return self._move(
                    observation,
                    -1.0,
                    "waiting for public jaw release outside the microwave edge",
                )
            return self._prepare_microwave_close_face_contact(observation)

        if self._phase == "microwave_close_push_preshape":
            if (
                self._phase_ticks < self.config.microwave_close_push_preshape_ticks
                or observation.proprio.gripper_width_m
                > self.config.microwave_close_push_compact_width_m
            ):
                return self._tick(OSCAction.hold(1.0))
            self._motion_position = (
                self._microwave_close_push_precontact_position.copy()
            )
            # Freeze a free-space, same-orientation force magnitude immediately
            # before the compact hand starts its exterior approach.  It is the
            # only baseline accepted by the typed early-contact gate below.
            self._force_baseline = observation.proprio.ee_force_sensor.copy()
            self._set_phase("microwave_close_push_precontact")
            return self._move(
                observation,
                1.0,
                "compact pusher approaching the fresh microwave exterior face",
            )

        if self._phase == "microwave_close_push_precontact":
            if (
                observation.proprio.gripper_width_m
                > self.config.microwave_close_push_compact_width_m
            ):
                return self._fail(
                    "microwave face pusher opened before exterior precontact"
                )
            early_contact_radius = (
                self._microwave_close_push_precontact_loaded(observation)
            )
            if early_contact_radius is not None:
                self._enter_microwave_close_push_door(
                    observation,
                    ee_radius_m=early_contact_radius,
                )
            else:
                aligned = bool(
                    self._position_reached(
                        observation,
                        self._motion_position,
                        tolerance_m=self.config.microwave_close_push_waypoint_tolerance_m,
                    )
                    and self._rotation_error(
                        observation.proprio.T_world_ee[:3, :3],
                        self._motion_rotation,
                    )
                    <= self.config.microwave_safe_rotation_tolerance_rad
                )
                if not aligned:
                    return self._move(observation, 1.0)
                self._motion_position = self._microwave_close_push_contact_point.copy()
                self._force_baseline = observation.proprio.ee_force_sensor.copy()
                self._set_phase("microwave_close_push_approach")
                return self._move(observation, 1.0)

        if self._phase == "microwave_close_push_approach":
            if (
                observation.proprio.gripper_width_m
                > self.config.microwave_close_push_compact_width_m
            ):
                return self._fail(
                    "microwave exterior approach opened the compact pusher"
                )
            if not (
                self._position_reached(observation, self._motion_position)
                or self._contact_reached(observation)
            ):
                return self._move(observation, 1.0)
            self._enter_microwave_close_push_door(observation)

        if self._phase == "microwave_close_push_door":
            if (
                observation.proprio.gripper_width_m
                > self.config.microwave_close_push_compact_width_m
            ):
                return self._fail("microwave close face pusher lost its compact width")
            try:
                radial, _outward, closing, radius = (
                    self._microwave_close_push_frame(
                        observation.proprio.ee_position_world
                    )
                )
            except ValueError as exc:
                return self._fail(str(exc))
            radius_error = abs(radius - self._microwave_close_push_radius_m)
            if radius_error > self.config.microwave_close_push_radius_tolerance_m:
                return self._fail(
                    "microwave close face pusher left its fresh RGB-D hinge radius "
                    f"(error={radius_error:.4f} m)"
                )
            measured = self._microwave_measured_chunk_angle(
                observation.proprio.ee_position_world
            )
            aligned = np.sign(self._microwave_arc_angle_rad) * measured
            if (
                aligned
                < self._microwave_close_push_best_angle_rad
                - self.config.microwave_close_push_reverse_tolerance_rad
            ):
                return self._fail(
                    "microwave close face push reversed its sensor-derived hinge angle"
                )
            self._microwave_close_push_best_angle_rad = max(
                self._microwave_close_push_best_angle_rad,
                aligned,
            )
            self._manipulation_progress_m = max(
                self._manipulation_progress_m,
                self._microwave_close_push_radius_m
                * max(
                    0.0,
                    self._microwave_close_push_best_angle_rad
                    - self._microwave_close_push_start_angle_rad,
                ),
            )
            target_angle = abs(self._microwave_close_target_angle_rad)
            at_sensor_goal = bool(
                self._microwave_close_push_best_angle_rad
                >= target_angle - self.config.microwave_close_push_goal_tolerance_rad
            )
            mechanical_stop = self._microwave_close_mechanical_stop(observation)
            if at_sensor_goal:
                return self._start_microwave_close_push_terminal_load(
                    observation,
                    aligned_angle=self._microwave_close_push_best_angle_rad,
                )
            if mechanical_stop:
                return self._start_microwave_close_push_exit(
                    observation,
                    mechanical_stop=True,
                )
            radius_correction = float(
                np.clip(
                    self._microwave_close_push_radius_m - radius,
                    -self.config.microwave_close_push_radius_correction_max_m,
                    self.config.microwave_close_push_radius_correction_max_m,
                )
            )
            self._motion_position = (
                observation.proprio.ee_position_world
                + closing * self.config.microwave_close_push_tangent_step_m
                + radial * radius_correction
            )
            return self._move(
                observation,
                1.0,
                "compact exterior-face pusher closing along the RGB-D hinge tangent",
            )

        if self._phase == "microwave_close_push_terminal_load":
            if (
                observation.proprio.gripper_width_m
                > self.config.microwave_close_push_compact_width_m
            ):
                return self._fail(
                    "microwave close terminal load lost its compact pusher width"
                )
            try:
                radial, _outward, closing, radius = (
                    self._microwave_close_push_frame(
                        observation.proprio.ee_position_world
                    )
                )
            except ValueError as exc:
                return self._fail(str(exc))
            radius_error = abs(radius - self._microwave_close_push_radius_m)
            if radius_error > self.config.microwave_close_push_radius_tolerance_m:
                return self._fail(
                    "microwave close terminal load left its fresh RGB-D hinge radius "
                    f"(error={radius_error:.4f} m)"
                )

            measured = self._microwave_measured_chunk_angle(
                observation.proprio.ee_position_world
            )
            aligned = np.sign(self._microwave_arc_angle_rad) * measured
            if (
                aligned
                < self._microwave_close_push_best_angle_rad
                - self.config.microwave_close_push_reverse_tolerance_rad
            ):
                return self._fail(
                    "microwave close terminal load reversed its sensor-derived hinge angle"
                )
            self._microwave_close_push_best_angle_rad = max(
                self._microwave_close_push_best_angle_rad,
                aligned,
            )
            extra_angle = max(
                0.0,
                self._microwave_close_push_best_angle_rad
                - self._microwave_close_push_terminal_start_angle_rad,
            )
            displacement = float(
                np.linalg.norm(
                    observation.proprio.ee_position_world
                    - self._microwave_close_push_terminal_anchor
                )
            )
            if (
                extra_angle
                > self.config.microwave_close_push_terminal_max_extra_angle_rad
                + 1e-9
            ):
                return self._fail(
                    "microwave close terminal load exceeded its sensor-angle bound"
                )
            if (
                displacement
                > self.config.microwave_close_push_terminal_max_displacement_m
                + self.config.microwave_close_push_terminal_bound_tolerance_m
            ):
                return self._fail(
                    "microwave close terminal load exceeded its Cartesian bound"
                )
            if (
                self._phase_ticks
                >= self.config.microwave_close_push_terminal_max_ticks
            ):
                return self._fail(
                    "microwave close terminal load exhausted its public stop timeout"
                )
            if (
                extra_angle
                >= self.config.microwave_close_push_terminal_max_extra_angle_rad
                - 1e-9
                or displacement
                >= self.config.microwave_close_push_terminal_max_displacement_m
                - 1e-9
            ):
                return self._fail(
                    "microwave close terminal load exhausted its bounded travel without a public stop"
                )
            self._manipulation_progress_m = max(
                self._manipulation_progress_m,
                self._microwave_close_push_radius_m
                * max(
                    0.0,
                    self._microwave_close_push_best_angle_rad
                    - self._microwave_close_push_start_angle_rad,
                ),
            )
            if self._microwave_close_mechanical_stop(observation):
                return self._start_microwave_close_push_exit(
                    observation,
                    mechanical_stop=True,
                )

            blend = max(
                extra_angle
                / self.config.microwave_close_push_terminal_max_extra_angle_rad,
                displacement
                / self.config.microwave_close_push_terminal_max_displacement_m,
                self._phase_ticks
                / self.config.microwave_close_push_terminal_max_ticks,
            )
            blend = float(np.clip(blend, 0.0, 1.0))
            load_direction = (
                (1.0 - blend) * closing
                + blend * self._microwave_close_push_terminal_face_normal
            )
            load_norm = float(np.linalg.norm(load_direction))
            if load_norm < 1e-8:
                return self._fail(
                    "microwave close terminal tangent/normal blend was degenerate"
                )
            load_direction /= load_norm
            if float(np.dot(load_direction, closing)) <= 0.0:
                return self._fail(
                    "microwave close terminal command reversed the closing tangent"
                )
            radius_correction = float(
                np.clip(
                    self._microwave_close_push_radius_m - radius,
                    -self.config.microwave_close_push_radius_correction_max_m,
                    self.config.microwave_close_push_radius_correction_max_m,
                )
            )
            step_distance = min(
                self.config.microwave_close_push_terminal_step_m,
                self.config.microwave_close_push_terminal_max_displacement_m
                - displacement,
            )
            candidate = (
                observation.proprio.ee_position_world
                + load_direction * step_distance
                + radial * radius_correction
            )
            candidate_delta = (
                candidate - self._microwave_close_push_terminal_anchor
            )
            candidate_distance = float(np.linalg.norm(candidate_delta))
            if (
                candidate_distance
                > self.config.microwave_close_push_terminal_max_displacement_m
            ):
                candidate = (
                    self._microwave_close_push_terminal_anchor
                    + candidate_delta
                    * (
                        self.config.microwave_close_push_terminal_max_displacement_m
                        / max(candidate_distance, 1e-12)
                    )
                )

            candidate_measured = self._microwave_measured_chunk_angle(candidate)
            candidate_aligned = (
                np.sign(self._microwave_arc_angle_rad) * candidate_measured
            )
            maximum_angle = (
                self._microwave_close_push_terminal_start_angle_rad
                + self.config.microwave_close_push_terminal_max_extra_angle_rad
            )
            if candidate_aligned > maximum_angle:
                angular_step = candidate_aligned - aligned
                if angular_step <= 1e-9:
                    return self._fail(
                        "microwave close terminal angle bound could not be enforced"
                    )
                scale = float(
                    np.clip(
                        (maximum_angle - aligned) / angular_step,
                        0.0,
                        1.0,
                    )
                )
                candidate = (
                    observation.proprio.ee_position_world
                    + (candidate - observation.proprio.ee_position_world) * scale
                )
            if (
                np.linalg.norm(candidate - observation.proprio.ee_position_world)
                <= 1e-6
            ):
                return self._fail(
                    "microwave close terminal load reached its command bound without a public stop"
                )
            self._motion_position = candidate
            return self._move(
                observation,
                1.0,
                "bounded terminal microwave load awaiting a public mechanical stop",
            )

        if self._phase == "microwave_close_push_radial_exit":
            if (
                observation.proprio.gripper_width_m
                > self.config.microwave_close_push_compact_width_m
            ):
                return self._fail(
                    "microwave close pusher opened before clearing the free edge"
                )
            delta = (
                observation.proprio.ee_position_world
                - self._microwave_close_push_exit_anchor
            )
            radial_progress = float(
                np.dot(delta, self._microwave_close_push_radial)
            )
            if radial_progress < -self.config.microwave_close_push_waypoint_tolerance_m:
                return self._fail("microwave close pusher exit reversed toward the hinge")
            if not self._position_reached(
                observation,
                self._motion_position,
                tolerance_m=self.config.microwave_close_push_waypoint_tolerance_m,
            ):
                return self._move(observation, 1.0)
            if (
                radial_progress
                < self.config.microwave_close_push_exit_radial_m
                - self.config.microwave_close_push_waypoint_tolerance_m
            ):
                return self._fail("microwave close pusher did not clear the free-edge radius")
            self._microwave_close_push_exit_anchor = (
                observation.proprio.ee_position_world.copy()
            )
            lift_axis = self._microwave_rotation_axis.copy()
            if lift_axis[2] < 0.0:
                lift_axis *= -1.0
            self._motion_position = (
                self._microwave_close_push_exit_anchor
                + self._microwave_close_push_outward
                * self.config.microwave_close_push_exit_outward_m
                + lift_axis * self.config.microwave_close_push_exit_lift_m
            )
            self._set_phase("microwave_close_push_outward_exit")
            return self._move(
                observation,
                1.0,
                "closed microwave pusher moving into the exterior free space",
            )

        if self._phase == "microwave_close_push_outward_exit":
            if (
                observation.proprio.gripper_width_m
                > self.config.microwave_close_push_compact_width_m
            ):
                return self._fail(
                    "microwave close pusher opened before entering exterior free space"
                )
            delta = (
                observation.proprio.ee_position_world
                - self._microwave_close_push_exit_anchor
            )
            outward_progress = float(
                np.dot(delta, self._microwave_close_push_outward)
            )
            if outward_progress < -self.config.microwave_close_push_waypoint_tolerance_m:
                return self._fail("microwave close pusher exit reversed toward the door")
            if not self._position_reached(
                observation,
                self._motion_position,
                tolerance_m=self.config.microwave_close_push_waypoint_tolerance_m,
            ):
                return self._move(observation, 1.0)
            if (
                outward_progress
                < self.config.microwave_close_push_exit_outward_m
                - self.config.microwave_close_push_waypoint_tolerance_m
            ):
                return self._fail("microwave close pusher did not enter exterior free space")
            self._motion_position = observation.proprio.ee_position_world.copy()
            self._motion_rotation = observation.proprio.T_world_ee[:3, :3].copy()
            self._set_phase("microwave_close_push_safe_release")
            return self._tick(
                OSCAction.hold(-1.0),
                "free-edge exit complete; opening fingers only in safe space",
            )

        if self._phase == "microwave_close_push_safe_release":
            if (
                self._phase_ticks >= self.config.release_ticks
                and observation.proprio.gripper_width_m
                >= self.config.drawer_release_width_m
            ):
                self._set_phase("microwave_verify")
                return self._tick(
                    OSCAction.hold(-1.0),
                    "safe jaw release complete; requesting strict fresh RGB-D close verification",
                )
            return self._move(observation, -1.0)

        return self._fail(f"invalid microwave close-pusher phase {self._phase!r}")

    def _start_microwave_close_release(
        self,
        observation: RobotObservation,
        *,
        mechanical_stop: bool,
    ) -> None:
        """Freeze the public close-completion pose before opening the jaws.

        Both an attained final arc/linear waypoint and a typed mechanical stop
        mean that the next command must release at the *measured* EE pose.
        Keeping the mechanical-stop bit separate preserves its diagnostic
        meaning while the close-release bit controls the common safe path.
        """

        self._microwave_mechanical_stop_observed = bool(mechanical_stop)
        self._microwave_close_release_pose_frozen = True
        self._microwave_release_position = (
            observation.proprio.ee_position_world.copy()
        )
        self._microwave_release_rotation = (
            observation.proprio.T_world_ee[:3, :3].copy()
        )
        release_outward = self._microwave_close_retreat_outward.copy()
        release_outward -= self._microwave_rotation_axis * float(
            np.dot(release_outward, self._microwave_rotation_axis)
        )
        release_outward_norm = float(np.linalg.norm(release_outward))
        if release_outward_norm >= 1e-8:
            release_outward /= release_outward_norm
        else:
            # Leave an invalid vector explicit.  The unseat safety gate will
            # fail rather than inventing a world direction if the frozen
            # sensor geometry cannot define closed-door free space.
            release_outward = np.zeros(3)
        self._microwave_close_release_outward = release_outward
        self._microwave_close_release_unseat_distance_m = 0.0
        # A nominal endpoint can lie beyond the door stop.  Fixed-world pose
        # servoing resists rebound while the fingers visibly open; a zero
        # incremental hold would simply follow the pushed-back hand.
        self._motion_position = self._microwave_release_position.copy()
        self._motion_rotation = self._microwave_release_rotation.copy()
        self._set_phase("microwave_release")

    def _start_microwave_close_retreat(
        self,
        observation: RobotObservation,
    ) -> None:
        """Begin the ordinary post-close retreat after public jaw release."""

        self._motion_rotation = observation.proprio.T_world_ee[:3, :3].copy()
        self._microwave_close_retreat_outward = (
            self._fresh_microwave_closed_normal(observation)
        )
        self._motion_position = (
            observation.proprio.ee_position_world.copy()
            + self._microwave_close_retreat_outward
            * self.config.microwave_retreat_clearance_m
        )
        self._set_phase("microwave_close_retreat_outward")

    def _fresh_microwave_closed_normal(
        self,
        observation: RobotObservation,
    ) -> np.ndarray:
        """Return a signed sensor-only normal for post-close retreat.

        A fresh high-confidence closed-handle observation is preferred after
        the fingers have visibly opened.  If the hand still occludes that
        feature, retain the reset-time RGB-D outward sign.  This observation
        is used only for a bounded safety retreat; the later ``microwave_verify``
        phase still performs its own mandatory fresh displacement/confidence
        check before completing the task.
        """

        fallback = self._microwave_close_retreat_outward.copy()
        fallback -= self._microwave_rotation_axis * float(
            np.dot(fallback, self._microwave_rotation_axis)
        )
        fallback_norm = float(np.linalg.norm(fallback))
        if fallback_norm < 1e-8:
            fallback = self._target.outward_world.copy()  # type: ignore[union-attr]
            fallback -= self._microwave_rotation_axis * float(
                np.dot(fallback, self._microwave_rotation_axis)
            )
            fallback_norm = float(np.linalg.norm(fallback))
        fallback /= max(fallback_norm, 1e-12)

        try:
            assert self.microwave_detector is not None
            assert self._target is not None
            refreshed = self.microwave_detector.track(
                observation,
                self._target,
                GoalSkillKind.CLOSE_MICROWAVE,
            )
        except (LookupError, ValueError):
            return fallback
        if refreshed.confidence < self.config.microwave_visual_min_confidence:
            return fallback
        candidate = refreshed.outward_world.copy()
        candidate -= self._microwave_rotation_axis * float(
            np.dot(candidate, self._microwave_rotation_axis)
        )
        candidate_norm = float(np.linalg.norm(candidate))
        if candidate_norm < 1e-8:
            return fallback
        candidate /= candidate_norm
        if float(np.dot(candidate, fallback)) < 0.0:
            candidate *= -1.0
        return candidate

    def _microwave_close_mechanical_stop(
        self,
        observation: RobotObservation,
    ) -> bool:
        """Recognise a typed stop candidate from three public plateaus.

        This only exits the commanded arc.  A fresh RGB-D door displacement
        remains mandatory after release and retreat before success.
        """

        error = float(
            np.linalg.norm(
                self._motion_position - observation.proprio.ee_position_world
            )
        )
        if (
            self._previous_error is None
            or self._previous_error - error > self.config.progress_epsilon_m
        ):
            self._stall_ticks = 0
        else:
            self._stall_ticks += 1
        self._previous_error = error

        width = observation.proprio.gripper_width_m
        if (
            self._microwave_previous_width_m is not None
            and abs(width - self._microwave_previous_width_m)
            <= self.config.microwave_width_plateau_epsilon_m
        ):
            self._microwave_width_plateau_ticks += 1
        else:
            self._microwave_width_plateau_ticks = 0
        self._microwave_previous_width_m = width

        force_delta = float(
            np.linalg.norm(
                observation.proprio.ee_force_sensor - self._force_baseline
            )
        )
        force = observation.proprio.ee_force_sensor
        if (
            self._microwave_previous_force_n is not None
            and np.linalg.norm(force - self._microwave_previous_force_n)
            <= self.config.microwave_force_plateau_epsilon_n
        ):
            self._microwave_force_plateau_ticks += 1
        else:
            self._microwave_force_plateau_ticks = 0
        self._microwave_previous_force_n = force.copy()

        start_radius = (
            self._microwave_arc_start_position - self._microwave_hinge_position
        )
        start_radius -= self._microwave_rotation_axis * float(
            np.dot(start_radius, self._microwave_rotation_axis)
        )
        current_radius = (
            observation.proprio.ee_position_world - self._microwave_hinge_position
        )
        current_radius -= self._microwave_rotation_axis * float(
            np.dot(current_radius, self._microwave_rotation_axis)
        )
        measured_angle = abs(
            float(
                np.arctan2(
                    np.dot(
                        self._microwave_rotation_axis,
                        np.cross(start_radius, current_radius),
                    ),
                    np.dot(start_radius, current_radius),
                )
            )
        )
        return bool(
            self._phase_ticks >= self.config.contact_min_ticks
            and measured_angle
            >= self.config.microwave_mechanical_stop_min_angle_rad
            and self._manipulation_progress_m
            >= self.config.microwave_visual_displacement_m
            and error <= self.config.microwave_mechanical_stop_residual_m
            and self._stall_ticks >= self.config.contact_stall_ticks
            and self._microwave_force_plateau_ticks
            >= self.config.microwave_force_plateau_ticks
            and self._microwave_width_plateau_ticks
            >= self.config.microwave_width_plateau_ticks
            and self.config.drawer_blocked_min_width_m
            <= width
            <= self.config.drawer_blocked_max_width_m
            and force_delta >= self.config.contact_force_delta_n
        )

    def _select_drawer_close_contact_offset(
        self,
        observation: RobotObservation,
        drawer_point: np.ndarray,
    ) -> np.ndarray:
        """Choose a clear point on the sensed movable drawer-front span.

        The handle centre can be hidden behind foreground clutter.  Score a
        small symmetric set along its measured long axis using only the two
        calibrated depth clouds.  The corridor ends 20 mm before the surface
        so the intended drawer itself remains a contact target rather than an
        obstacle.  Clearance applies to the compact pusher's swept half-width,
        not merely its Cartesian centreline.  Prefer the nearest clear slot,
        not the globally emptiest point outside the moving front.
        """

        assert self._target is not None
        axis = np.asarray(self._target.feature_axis_world, dtype=np.float64)
        axis[2] = 0.0
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        lateral_offsets_m = (
            0.0,
            -0.045,
            0.045,
            -0.070,
            0.070,
            -0.085,
            0.085,
            -0.100,
            0.100,
        )
        # Keep the detector's visible handle/front height.  Raising a lower
        # drawer contact can silently place the pusher on the next drawer.
        height_offsets_m = (0.0,)
        clouds = [
            backproject_depth(frame, stride=4, world=True)
            for frame in observation.cameras.values()
        ]
        cloud = np.concatenate(clouds, axis=0)
        # Arm pixels close to the reset EE are not fixture clutter and can
        # otherwise make every candidate appear occupied.
        robot_distance = np.linalg.norm(
            cloud - observation.proprio.ee_position_world[None, :],
            axis=1,
        )
        cloud = cloud[robot_distance >= 0.075]
        scored: list[tuple[float, float, float, np.ndarray]] = []
        for height in height_offsets_m:
            for scalar in lateral_offsets_m:
                offset = axis * scalar
                offset[2] += height
                contact = drawer_point + offset
                precontact = (
                    contact
                    + self._target.outward_world
                    * self.config.drawer_precontact_clearance_m
                )
                safe = precontact.copy()
                safe[2] += self.config.drawer_safe_height_m
                free_end = (
                    contact
                    + self._target.outward_world * 0.020
                )
                vertical = np.linspace(safe, precontact, 9)
                approach = np.linspace(precontact, free_end, 8)
                samples = np.concatenate((vertical, approach[1:]), axis=0)
                if len(cloud) == 0:
                    clearance = float("inf")
                    support_distance = float("inf")
                else:
                    distances = np.linalg.norm(
                        samples[:, None, :] - cloud[None, :, :],
                        axis=2,
                    )
                    clearance = float(np.min(distances))
                    support_distance = float(
                        np.min(np.linalg.norm(cloud - contact[None, :], axis=1))
                    )
                scored.append(
                    (
                        clearance,
                        support_distance,
                        float(np.linalg.norm(offset)),
                        offset.copy(),
                    )
                )

        # Jaw aperture is not the outer tool envelope: even closed fingers
        # retain their own thickness.  Apply a conservative physical swept
        # radius plus the explicit obstacle margin.  A centreline can otherwise
        # appear 1--2 cm clear while one finger is already inside foreground
        # clutter.
        required_clearance = (
            self.config.drawer_close_tool_radius_m
            + self.config.drawer_close_slot_clearance_m
        )
        safe = [
            item
            for item in scored
            if item[0] >= required_clearance
            and item[1] <= self.config.drawer_close_front_support_tolerance_m
        ]
        if not safe:
            raise LookupError(
                "no supported movable drawer-front slot has bounded RGB-D clearance"
            )
        return min(safe, key=lambda item: (item[2], -item[0]))[3]

    @staticmethod
    def _axis_aligned_fixture_direction(direction: np.ndarray) -> np.ndarray:
        """Retain the calibrated RGB-D fixture normal as a unit XY vector.

        Fixtures need not be Manhattan-aligned in the world frame.  Snapping
        an 8-degree sensed drawer yaw to a cardinal axis creates a sustained
        sideways load in its prismatic guide, so the measured normal is the
        safer contact direction.
        """

        planar = np.asarray(direction, dtype=np.float64).copy()
        planar[2] = 0.0
        norm = float(np.linalg.norm(planar))
        if norm < 1e-8:
            raise ValueError("fixture direction has no horizontal component")
        return planar / norm

    def _select_drawer_lateral(
        self,
        observation: RobotObservation,
        precontact: np.ndarray,
        drawer_point: np.ndarray,
    ) -> np.ndarray:
        """Choose the less occupied side of a handle from the two RGB-D clouds."""

        axis = self._target.feature_axis_world  # type: ignore[union-attr]
        offsets = (
            -axis * self.config.drawer_lateral_clearance_m,
            axis * self.config.drawer_lateral_clearance_m,
        )
        point_groups = [
            backproject_depth(frame, stride=4, world=True)
            for frame in observation.cameras.values()
        ]
        cloud = np.concatenate(point_groups, axis=0)
        z_low = drawer_point[2] - self.config.drawer_obstacle_below_handle_m
        z_high = drawer_point[2] + 0.09
        cloud = cloud[(cloud[:, 2] >= z_low) & (cloud[:, 2] <= z_high)]
        current = observation.proprio.ee_position_world

        def score(offset: np.ndarray) -> tuple[int, float]:
            candidate = precontact + offset
            radial = np.linalg.norm(cloud[:, :2] - candidate[:2], axis=1)
            occupied = int(np.count_nonzero(radial <= self.config.drawer_obstacle_radius_m))
            travel = float(np.linalg.norm(candidate - current))
            return occupied, travel

        return min(offsets, key=score).copy()

    def _drawer_handle_axis(self, target: ContactTarget) -> np.ndarray:
        """Return the public handle long axis orthogonal to the drawer normal."""

        outward = self._axis_aligned_fixture_direction(target.outward_world)
        axis = np.asarray(target.feature_axis_world, dtype=np.float64).copy()
        axis[2] = 0.0
        axis -= outward * float(np.dot(axis, outward))
        norm = float(np.linalg.norm(axis))
        if norm < 1e-6:
            raise ValueError("drawer handle feature axis has no reliable planar component")
        axis /= norm
        if float(np.dot(axis, target.feature_axis_world)) < 0.0:
            axis *= -1.0
        return axis

    def _observe_drawer_handle_axis_bounds(
        self,
        observation: RobotObservation,
        target: ContactTarget,
    ) -> np.ndarray | None:
        """Freeze robust visible handle endpoints from calibrated public depth.

        The detector already supplies a handle point, front normal, long axis,
        and source-camera identity.  This local metric band only recovers the
        missing long-axis interval; it cannot introduce an object identity,
        simulator pose, segmentation mask, or task-coordinate prior.  A broad
        drawer face or one-sided/short fragment fails the typed span gates and
        falls back to the bounded bidirectional probe.
        """

        try:
            axis = self._drawer_handle_axis(target)
            outward = self._axis_aligned_fixture_direction(target.outward_world)
        except ValueError:
            return None
        clouds: list[np.ndarray] = []
        for camera_name in target.source_cameras:
            frame = observation.cameras.get(camera_name)
            if frame is None:
                continue
            try:
                cloud = backproject_depth(frame, stride=2, world=True)
            except (TypeError, ValueError):
                continue
            if len(cloud):
                clouds.append(cloud)
        if not clouds:
            return None
        cloud = np.concatenate(clouds, axis=0)
        finite = np.all(np.isfinite(cloud), axis=1)
        delta = cloud[finite] - target.point_world[None, :]
        if not len(delta):
            return None
        axial = delta @ axis
        normal = delta @ outward
        vertical = delta[:, 2]
        maximum_half_interval = (
            0.5 * self.config.drawer_rail_stop_axis_span_max_m
            + self.config.drawer_rail_stop_axis_endpoint_margin_m
        )
        local = (
            (np.abs(axial) <= maximum_half_interval)
            & (
                np.abs(normal)
                <= self.config.drawer_rail_stop_axis_normal_band_m
            )
            & (
                np.abs(vertical)
                <= self.config.drawer_rail_stop_axis_vertical_band_m
            )
        )
        coordinates = axial[local]
        if len(coordinates) < self.config.drawer_rail_stop_axis_min_points:
            return None
        quantile = self.config.drawer_rail_stop_axis_endpoint_quantile
        lower, upper = np.quantile(coordinates, (quantile, 1.0 - quantile))
        span = float(upper - lower)
        tolerance = self.config.drawer_rail_stop_axis_progress_tolerance_m
        if not (
            self.config.drawer_rail_stop_axis_span_min_m
            <= span
            <= self.config.drawer_rail_stop_axis_span_max_m
            and lower <= -tolerance
            and upper >= tolerance
        ):
            return None
        return np.array((float(lower), float(upper)), dtype=np.float64)

    def _reset_drawer_rail_stop_motion_state(self) -> None:
        self._drawer_rail_stop_axis_world = np.array([1.0, 0.0, 0.0])
        self._drawer_rail_stop_slide_direction = np.array([1.0, 0.0, 0.0])
        self._drawer_rail_stop_slide_segment_start_position = np.zeros(3)
        self._drawer_rail_stop_slide_segment_distance_m = 0.0
        self._drawer_rail_stop_slide_remaining_m = 0.0
        self._drawer_rail_stop_axis_total_progress_m = 0.0
        self._drawer_rail_stop_axis_segment_progress_m = 0.0
        self._drawer_rail_stop_axis_segment_peak_progress_m = 0.0
        self._drawer_rail_stop_axis_segment_max_regression_m = 0.0
        self._drawer_rail_stop_axis_max_observed_regression_m = 0.0
        self._drawer_rail_stop_axis_partial_progress_m = 0.0
        self._drawer_rail_stop_axis_partial_segment_count = 0
        self._drawer_rail_stop_axis_chain_active = False
        self._drawer_rail_stop_axis_chain_origin = np.zeros(3)
        self._drawer_rail_stop_axis_chain_rotation = np.eye(3)
        self._drawer_rail_stop_axis_chain_direction = np.array([1.0, 0.0, 0.0])
        self._drawer_rail_stop_axis_chain_total_start_m = 0.0
        self._drawer_rail_stop_axis_chain_progress_m = 0.0
        self._drawer_rail_stop_axis_chain_cross_drift_m = 0.0
        self._drawer_rail_stop_axis_chain_rotation_error_rad = 0.0
        self._drawer_rail_stop_axis_chain_local_normal_m = 0.0
        self._drawer_rail_stop_axis_chain_third_axis_drift_m = 0.0
        self._drawer_rail_stop_axis_chain_global_normal_progress_m = 0.0
        self._drawer_rail_stop_axis_chain_global_normal_start_m = 0.0
        self._drawer_rail_stop_axis_chain_last_verified_global_normal_m = 0.0
        self._drawer_rail_stop_axis_chain_pending_baseline_normal_credit_m = 0.0
        self._drawer_rail_stop_axis_chain_baseline_normal_segment_cap_m = 0.0
        self._drawer_rail_stop_axis_chain_normal_credit_evidence = None
        self._drawer_rail_stop_probe_origin = np.zeros(3)
        self._drawer_rail_stop_probe_index = 0
        self._drawer_rail_stop_release_width_m = 0.0
        self._drawer_rail_stop_current_public_width_m = 0.0
        self._drawer_rail_stop_body_margin_m = 0.0
        self._drawer_rail_stop_axis_retry_used_m = 0.0
        self._drawer_rail_stop_axis_retry_require_full_distance = False
        self._drawer_rail_stop_required_axis_scalar_m = float("nan")
        self._drawer_rail_stop_outward_probe_origin = np.zeros(3)
        self._drawer_rail_stop_outward_probe_distance_m = 0.0
        self._drawer_rail_stop_outward_probe_progress_m = 0.0
        self._drawer_rail_stop_outward_axis_drift_m = 0.0
        self._drawer_rail_stop_outward_residual_drift_m = 0.0
        self._drawer_rail_stop_outward_net_clearance_m = float("-inf")
        self._drawer_rail_stop_axis_retry_settle_transition_position = np.zeros(3)
        self._drawer_rail_stop_axis_retry_settle_transition_rotation = np.eye(3)
        self._drawer_rail_stop_axis_retry_settle_origin = np.zeros(3)
        self._drawer_rail_stop_axis_retry_settle_start_global_normal_m = 0.0
        self._drawer_rail_stop_axis_retry_settle_previous_global_normal_m = 0.0
        self._drawer_rail_stop_axis_retry_settle_delta_normal_m = 0.0
        self._drawer_rail_stop_axis_retry_settle_local_normal_m = 0.0
        self._drawer_rail_stop_axis_retry_settle_transition_normal_m = 0.0
        self._drawer_rail_stop_axis_retry_settle_rail_drift_m = 0.0
        self._drawer_rail_stop_axis_retry_settle_third_axis_drift_m = 0.0
        self._drawer_rail_stop_axis_retry_settle_rotation_error_rad = 0.0
        self._drawer_rail_stop_axis_retry_settle_stable_count = 0
        self._drawer_rail_stop_axis_retry_settle_anchor_frozen = False
        self._drawer_rail_stop_axis_retry_settle_requested_distance_m = 0.0
        self._drawer_rail_stop_axis_retry_settle_probe_axis_drift_m = 0.0
        self._drawer_rail_stop_axis_retry_settle_required_axis_correction_m = 0.0
        self._drawer_rail_stop_axis_retry_settle_entry_net_clearance_m = 0.0
        self._drawer_rail_stop_axis_retry_settle_entry_baseline_credit_cap_m = 0.0
        self._drawer_rail_stop_axis_retry_settle_require_full_distance = False
        self._drawer_rail_stop_axis_retry_settle_message = ""
        self._drawer_rail_stop_axis_retry_settle_failure_message = ""
        self._drawer_rail_stop_normal_retreat_progress_m = 0.0
        self._drawer_rail_stop_normal_segment_origin = np.zeros(3)
        self._drawer_rail_stop_normal_segment_distance_m = 0.0
        self._drawer_rail_stop_normal_segment_progress_m = 0.0
        self._drawer_rail_stop_normal_segment_cross_drift_m = 0.0
        self._drawer_rail_stop_normal_chain_active = False
        self._drawer_rail_stop_normal_chain_origin = np.zeros(3)
        self._drawer_rail_stop_normal_chain_rotation = np.eye(3)
        self._drawer_rail_stop_normal_chain_direction = np.array([0.0, 1.0, 0.0])
        self._drawer_rail_stop_normal_chain_axis = np.array([1.0, 0.0, 0.0])
        self._drawer_rail_stop_normal_chain_progress_m = 0.0
        self._drawer_rail_stop_normal_chain_cross_drift_m = 0.0
        self._drawer_rail_stop_normal_chain_rotation_error_rad = 0.0
        self._drawer_rail_stop_normal_credit_source = "none"
        self._drawer_rail_stop_normal_credit_increment_m = 0.0

    def _drawer_rail_stop_nonfinite_failure(
        self,
        context: str,
        **values: object,
    ) -> PolicyDecision | None:
        """Fail closed before non-finite rail-stop state can issue an action."""

        # The dataclass is frozen, but checking again here makes this boundary
        # robust to corrupted deserialisation or deliberate low-level mutation
        # in addition to ordinary constructor validation.
        for parameter in fields(self.config):
            value = getattr(self.config, parameter.name)
            default = parameter.default
            if (
                isinstance(value, (bool, np.bool_))
                or (type(default) is int and type(value) is not int)
            ):
                return self._fail(
                    "drawer rail-stop runtime configuration had invalid "
                    f"numeric {parameter.name}"
                )
            try:
                raw_array = np.asarray(value)
                if raw_array.dtype.kind not in "iuf":
                    raise TypeError
                array = np.asarray(raw_array, dtype=np.float64)
            except (TypeError, ValueError):
                return self._fail(
                    "drawer rail-stop runtime configuration had invalid "
                    f"numeric {parameter.name}"
                )
            if not np.all(np.isfinite(array)):
                return self._fail(
                    "drawer rail-stop runtime configuration had non-finite "
                    f"{parameter.name}"
                )
        for name, value in values.items():
            try:
                if isinstance(value, (bool, np.bool_)):
                    raise TypeError
                raw_array = np.asarray(value)
                if raw_array.dtype.kind not in "iuf":
                    raise TypeError
                array = np.asarray(raw_array, dtype=np.float64)
            except (TypeError, ValueError):
                return self._fail(
                    f"drawer rail-stop {context} had invalid numeric {name}"
                )
            if not np.all(np.isfinite(array)):
                return self._fail(
                    f"drawer rail-stop {context} had non-finite {name}"
                )
        return None

    def _drawer_rail_stop_exact_scalar_failure(
        self,
        context: str,
        *,
        float_values: tuple[tuple[str, object], ...] = (),
        int_values: tuple[tuple[str, object], ...] = (),
        bool_values: tuple[tuple[str, object], ...] = (),
    ) -> PolicyDecision | None:
        """Reject scalar state whose runtime type no longer matches its schema.

        Public poses, directions, and rotations intentionally remain NumPy
        arrays.  Budget, timeout, and proof state is different: accepting a
        NumPy scalar (or an ``int`` in a float slot) lets later arithmetic wash
        a corrupted value back into an apparently valid builtin.  Validate
        those internal scalar slots before any state mutation or action.
        """

        for name, value in float_values:
            if type(value) is not float or not np.isfinite(value):
                return self._fail(
                    f"drawer rail-stop {context} had invalid builtin float {name}"
                )
        for name, value in int_values:
            if type(value) is not int:
                return self._fail(
                    f"drawer rail-stop {context} had invalid builtin int {name}"
                )
        for name, value in bool_values:
            if type(value) is not bool:
                return self._fail(
                    f"drawer rail-stop {context} had invalid builtin bool {name}"
                )
        return None

    def _freeze_drawer_rail_stop_axis_chain(
        self,
        observation: RobotObservation,
    ) -> PolicyDecision | None:
        """Freeze one contiguous same-direction rail-axis recovery chain."""

        direction = np.asarray(
            self._drawer_rail_stop_slide_direction,
            dtype=np.float64,
        ).copy()
        finite_failure = self._drawer_rail_stop_nonfinite_failure(
            "handle-axis chain freeze",
            public_pose=observation.proprio.T_world_ee,
            direction=direction,
            axis_total_progress=self._drawer_rail_stop_axis_total_progress_m,
        )
        if finite_failure is not None:
            return finite_failure
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm < 1e-8:
            return self._fail(
                "drawer rail-stop handle-axis chain direction was degenerate"
            )
        direction /= direction_norm
        global_normal_start = 0.0
        pending_baseline_normal_credit = 0.0
        baseline_normal_segment_cap = 0.0
        if self._drawer_rail_stop_normal_chain_active:
            normal = np.asarray(
                self._drawer_rail_stop_normal_chain_direction,
                dtype=np.float64,
            )
            finite_failure = self._drawer_rail_stop_nonfinite_failure(
                "handle-axis chain normal baseline",
                normal_chain_origin=self._drawer_rail_stop_normal_chain_origin,
                outward_normal=normal,
                normal_ledger=self._drawer_rail_stop_normal_retreat_progress_m,
            )
            if finite_failure is not None:
                return finite_failure
            global_normal_start = float(
                np.dot(
                    observation.proprio.ee_position_world
                    - self._drawer_rail_stop_normal_chain_origin,
                    normal,
                )
            )
            pending_baseline_normal_credit = max(
                0.0,
                global_normal_start
                - self._drawer_rail_stop_normal_retreat_progress_m,
            )
            if pending_baseline_normal_credit <= 1e-12:
                pending_baseline_normal_credit = 0.0
            if self._phase == "drawer_rail_stop_outward_probe":
                baseline_normal_segment_cap = (
                    self._drawer_rail_stop_outward_probe_distance_m
                )
            elif self._phase == "drawer_rail_stop_axis_retry_settle":
                # The settle phase is a zero-command continuation of the
                # immediately preceding bounded outward probe.  Only a gap
                # already present and bounded at settle entry can be staged;
                # positive zero-command drift during the hold is never
                # retroactively relabelled as normal-retreat credit.
                baseline_normal_segment_cap = (
                    self._drawer_rail_stop_outward_probe_distance_m
                )
                entry_credit_cap = (
                    self._drawer_rail_stop_axis_retry_settle_entry_baseline_credit_cap_m
                )
                entry_net_gap = max(
                    0.0,
                    self._drawer_rail_stop_axis_retry_settle_start_global_normal_m
                    - self._drawer_rail_stop_normal_retreat_progress_m,
                )
                finite_failure = self._drawer_rail_stop_nonfinite_failure(
                    "handle-axis chain settle-entry baseline proof",
                    entry_credit_cap=entry_credit_cap,
                    entry_net_gap=entry_net_gap,
                    settle_start_global_normal=(
                        self._drawer_rail_stop_axis_retry_settle_start_global_normal_m
                    ),
                )
                if finite_failure is not None:
                    return finite_failure
                if (
                    entry_credit_cap < 0.0
                    or entry_credit_cap
                    > baseline_normal_segment_cap + 1e-12
                    or entry_credit_cap > entry_net_gap + 1e-12
                ):
                    return self._fail(
                        "drawer rail-stop axis retry settle-entry baseline proof "
                        "was inconsistent"
                    )
                pending_baseline_normal_credit = min(
                    pending_baseline_normal_credit,
                    entry_credit_cap,
                )
            elif self._phase == "retreat_outward":
                baseline_normal_segment_cap = (
                    self._drawer_rail_stop_normal_segment_distance_m
                )
            finite_failure = self._drawer_rail_stop_nonfinite_failure(
                "handle-axis chain baseline segment proof",
                baseline_normal_segment_cap=baseline_normal_segment_cap,
                pending_baseline_normal_credit=pending_baseline_normal_credit,
            )
            if finite_failure is not None:
                return finite_failure
            if (
                baseline_normal_segment_cap > 0.010 + 1e-12
                or pending_baseline_normal_credit
                > baseline_normal_segment_cap + 1e-12
            ):
                return self._fail(
                    "drawer rail-stop axis retry baseline normal gap exceeded "
                    "its preceding bounded public normal segment"
                )
        self._drawer_rail_stop_axis_chain_active = True
        self._drawer_rail_stop_axis_chain_origin = (
            observation.proprio.ee_position_world.copy()
        )
        self._drawer_rail_stop_axis_chain_rotation = (
            observation.proprio.T_world_ee[:3, :3].copy()
        )
        self._motion_rotation = self._drawer_rail_stop_axis_chain_rotation.copy()
        self._drawer_rail_stop_axis_chain_direction = direction
        self._drawer_rail_stop_axis_chain_total_start_m = float(
            self._drawer_rail_stop_axis_total_progress_m
        )
        self._drawer_rail_stop_axis_chain_progress_m = 0.0
        self._drawer_rail_stop_axis_chain_cross_drift_m = 0.0
        self._drawer_rail_stop_axis_chain_rotation_error_rad = 0.0
        self._drawer_rail_stop_axis_chain_local_normal_m = 0.0
        self._drawer_rail_stop_axis_chain_third_axis_drift_m = 0.0
        self._drawer_rail_stop_axis_chain_global_normal_progress_m = float(
            global_normal_start
        )
        self._drawer_rail_stop_axis_chain_global_normal_start_m = float(
            global_normal_start
        )
        self._drawer_rail_stop_axis_chain_last_verified_global_normal_m = float(
            global_normal_start
        )
        self._drawer_rail_stop_axis_chain_pending_baseline_normal_credit_m = float(
            pending_baseline_normal_credit
        )
        self._drawer_rail_stop_axis_chain_baseline_normal_segment_cap_m = float(
            baseline_normal_segment_cap
        )
        return None

    def _drawer_rail_stop_axis_chain_safety_failure(
        self,
        observation: RobotObservation,
    ) -> PolicyDecision | None:
        """Validate one public retry pose and stage, but never issue, credit."""

        # A candidate belongs only to this exact public observation.  It is
        # committed by the caller after every remaining command/segment gate
        # has passed; a later failure therefore cannot leave evidence behind.
        self._drawer_rail_stop_axis_chain_normal_credit_evidence = None

        if not self._drawer_rail_stop_axis_chain_active:
            return self._fail(
                "drawer rail-stop handle-axis motion lacked a frozen chain anchor"
            )
        finite_failure = self._drawer_rail_stop_nonfinite_failure(
            "handle-axis chain",
            public_pose=observation.proprio.T_world_ee,
            chain_origin=self._drawer_rail_stop_axis_chain_origin,
            chain_rotation=self._drawer_rail_stop_axis_chain_rotation,
            chain_direction=self._drawer_rail_stop_axis_chain_direction,
            current_direction=self._drawer_rail_stop_slide_direction,
            chain_total_start=self._drawer_rail_stop_axis_chain_total_start_m,
            axis_total_progress=self._drawer_rail_stop_axis_total_progress_m,
            last_verified_global_normal=(
                self._drawer_rail_stop_axis_chain_last_verified_global_normal_m
            ),
            pending_baseline_normal_credit=(
                self._drawer_rail_stop_axis_chain_pending_baseline_normal_credit_m
            ),
            baseline_normal_segment_cap=(
                self._drawer_rail_stop_axis_chain_baseline_normal_segment_cap_m
            ),
            local_normal=self._drawer_rail_stop_axis_chain_local_normal_m,
            third_axis_drift=(
                self._drawer_rail_stop_axis_chain_third_axis_drift_m
            ),
            global_normal_progress=(
                self._drawer_rail_stop_axis_chain_global_normal_progress_m
            ),
            global_normal_start=(
                self._drawer_rail_stop_axis_chain_global_normal_start_m
            ),
            normal_chain_progress=self._drawer_rail_stop_normal_chain_progress_m,
            normal_chain_cross_drift=(
                self._drawer_rail_stop_normal_chain_cross_drift_m
            ),
            normal_chain_rotation_error=(
                self._drawer_rail_stop_normal_chain_rotation_error_rad
            ),
            normal_credit=self._drawer_rail_stop_normal_retreat_progress_m,
            normal_credit_increment=self._drawer_rail_stop_normal_credit_increment_m,
        )
        if finite_failure is not None:
            return finite_failure
        direction = self._drawer_rail_stop_axis_chain_direction
        current_direction = np.asarray(
            self._drawer_rail_stop_slide_direction,
            dtype=np.float64,
        ).copy()
        current_direction /= max(float(np.linalg.norm(current_direction)), 1e-12)
        if float(np.dot(direction, current_direction)) < 1.0 - 1e-6:
            return self._fail(
                "drawer rail-stop handle-axis direction changed inside its frozen chain"
            )
        displacement = (
            observation.proprio.ee_position_world
            - self._drawer_rail_stop_axis_chain_origin
        )
        progress = float(np.dot(displacement, direction))
        cross_drift = float(
            np.linalg.norm(displacement - direction * progress)
        )
        rotation_error = self._rotation_error(
            observation.proprio.T_world_ee[:3, :3],
            self._drawer_rail_stop_axis_chain_rotation,
        )
        self._drawer_rail_stop_axis_chain_progress_m = progress
        self._drawer_rail_stop_axis_chain_cross_drift_m = cross_drift
        self._drawer_rail_stop_axis_chain_rotation_error_rad = rotation_error
        tolerance = self.config.drawer_rail_stop_axis_progress_tolerance_m
        if not self._drawer_rail_stop_normal_chain_active:
            # Initial handle clearance has no independently frozen drawer-normal
            # corridor.  Preserve its original strict all-orthogonal cumulative
            # gate: no component outside the selected rail may exceed 4 mm.
            if cross_drift > self.config.drawer_rail_stop_outward_max_cross_drift_m:
                return self._fail(
                    "drawer rail-stop handle-axis chain exceeded its cumulative public "
                    f"cross-axis drift gate ({cross_drift:.4f} m)"
                )
        else:
            # A later rail retry happens inside the already frozen normal-retreat
            # corridor.  Decompose the *same public pose* relative to the frozen
            # axis-chain origin.  Beneficial outward compliance is not third-axis
            # drift, but it is independently bounded and can never earn rail
            # credit.  Only a fully gated increase in global public normal pose
            # may enter the separate normal ledger below.
            normal = np.asarray(
                self._drawer_rail_stop_normal_chain_direction,
                dtype=np.float64,
            ).copy()
            normal_axis = np.asarray(
                self._drawer_rail_stop_normal_chain_axis,
                dtype=np.float64,
            ).copy()
            finite_failure = self._drawer_rail_stop_nonfinite_failure(
                "handle-axis/normal-chain decomposition",
                normal_chain_origin=self._drawer_rail_stop_normal_chain_origin,
                normal_chain_rotation=self._drawer_rail_stop_normal_chain_rotation,
                outward_normal=normal,
                normal_chain_axis=normal_axis,
                normal_credit=self._drawer_rail_stop_normal_retreat_progress_m,
            )
            if finite_failure is not None:
                return finite_failure
            direction_norm = float(np.linalg.norm(direction))
            normal_norm = float(np.linalg.norm(normal))
            normal_axis_norm = float(np.linalg.norm(normal_axis))
            if (
                abs(direction_norm - 1.0) > 1e-6
                or abs(normal_norm - 1.0) > 1e-6
                or abs(normal_axis_norm - 1.0) > 1e-6
                or abs(float(np.dot(direction, normal))) > 1e-6
                or abs(float(np.dot(normal_axis, normal))) > 1e-6
                or abs(abs(float(np.dot(direction, normal_axis))) - 1.0) > 1e-6
            ):
                return self._fail(
                    "drawer rail-stop handle-axis retry lacked a consistent frozen "
                    "rail/normal basis"
                )
            local_normal = float(np.dot(displacement, normal))
            third_axis_residual = (
                displacement
                - direction * progress
                - normal * local_normal
            )
            third_axis_drift = float(np.linalg.norm(third_axis_residual))
            self._drawer_rail_stop_axis_chain_local_normal_m = local_normal
            self._drawer_rail_stop_axis_chain_third_axis_drift_m = third_axis_drift

            # Keep the full normal chain live during rail retries.  This updates
            # its global progress/residual/rotation from this exact observation
            # and applies every original gate.  Any stale-credit clamp or
            # positive increment remains staged until the caller also accepts
            # the later segment/command gates.
            normal_chain_unsafe = (
                self._drawer_rail_stop_normal_chain_safety_failure(
                    observation,
                    commit_credit_clamp=False,
                )
            )
            self._drawer_rail_stop_axis_chain_global_normal_progress_m = (
                self._drawer_rail_stop_normal_chain_progress_m
            )
            if normal_chain_unsafe is not None:
                return normal_chain_unsafe
            normal_tolerance = (
                self.config.drawer_rail_stop_outward_progress_tolerance_m
            )
            if local_normal < -normal_tolerance - 1e-12:
                return self._fail(
                    "drawer rail-stop handle-axis retry reversed its local frozen "
                    f"drawer-normal progress ({local_normal:.4f} m)"
                )
            if local_normal > 0.010 + 1e-12:
                return self._fail(
                    "drawer rail-stop handle-axis retry exceeded its 0.010 m local "
                    f"positive drawer-normal bound ({local_normal:.4f} m)"
                )
            if (
                third_axis_drift
                > self.config.drawer_rail_stop_outward_max_cross_drift_m + 1e-12
            ):
                return self._fail(
                    "drawer rail-stop handle-axis retry exceeded its cumulative public "
                    f"third-axis drift gate ({third_axis_drift:.4f} m)"
                )
        if rotation_error > self.config.drawer_rail_stop_outward_max_rotation_rad:
            return self._fail(
                "drawer rail-stop handle-axis chain lost its frozen public wrist "
                f"pose ({rotation_error:.4f} rad)"
            )
        if progress < -tolerance:
            return self._fail(
                "drawer rail-stop handle-axis chain reversed its cumulative public "
                f"EE progress ({progress:.4f} m)"
            )
        if self._drawer_rail_stop_normal_chain_active:
            # Stage only newly observed, fully validated global normal
            # displacement.  The frozen-rail projection remains the sole source
            # of axis credit, repeated poses cannot double count, and the hard
            # 75-mm public-normal cap cannot be exceeded.
            global_normal = (
                self._drawer_rail_stop_axis_chain_global_normal_progress_m
            )
            previous_global = (
                self._drawer_rail_stop_axis_chain_last_verified_global_normal_m
            )
            ledger = min(
                self._drawer_rail_stop_normal_retreat_progress_m,
                max(0.0, global_normal),
                0.075,
            )
            pending_baseline = (
                self._drawer_rail_stop_axis_chain_pending_baseline_normal_credit_m
            )
            new_global = max(0.0, global_normal - previous_global)
            net_public_room = max(0.0, max(0.0, global_normal) - ledger)
            hard_room = max(0.0, 0.075 - ledger)
            baseline_credit = min(
                pending_baseline,
                net_public_room,
                hard_room,
            )
            ledger_after_baseline = ledger + baseline_credit
            incremental_public_room = max(
                0.0,
                max(0.0, global_normal) - ledger_after_baseline,
            )
            incremental_hard_room = max(
                0.0,
                0.075 - ledger_after_baseline,
            )
            incremental_credit = min(
                new_global,
                incremental_public_room,
                incremental_hard_room,
            )
            normal_credit = baseline_credit + incremental_credit
            if normal_credit <= 1e-12:
                normal_credit = 0.0
            finite_failure = self._drawer_rail_stop_nonfinite_failure(
                "handle-axis verified normal credit",
                global_normal=global_normal,
                previous_global=previous_global,
                normal_ledger=ledger,
                pending_baseline=pending_baseline,
                new_global=new_global,
                net_public_room=net_public_room,
                hard_room=hard_room,
                baseline_credit=baseline_credit,
                ledger_after_baseline=ledger_after_baseline,
                incremental_public_room=incremental_public_room,
                incremental_hard_room=incremental_hard_room,
                incremental_credit=incremental_credit,
                normal_credit=normal_credit,
            )
            if finite_failure is not None:
                return finite_failure
            if baseline_credit > 0.0 and incremental_credit > 0.0:
                credit_source = "axis_retry_baseline_and_public_pose"
            elif baseline_credit > 0.0:
                credit_source = "axis_retry_baseline_public_pose"
            elif incremental_credit > 0.0:
                credit_source = "axis_retry_public_pose"
            else:
                credit_source = "none"
            self._drawer_rail_stop_axis_chain_normal_credit_evidence = (
                _DrawerRailStopNormalCreditEvidence(
                    global_normal_m=float(global_normal),
                    ledger_after_m=float(ledger + normal_credit),
                    credit_increment_m=float(normal_credit),
                    source=credit_source,
                )
            )
        return None

    def _commit_drawer_rail_stop_axis_normal_credit(
        self,
    ) -> PolicyDecision | None:
        """Commit same-tick public-pose evidence after every later gate passes."""

        evidence = self._drawer_rail_stop_axis_chain_normal_credit_evidence
        if evidence is None:
            return None
        if type(evidence) is not _DrawerRailStopNormalCreditEvidence:
            return self._fail(
                "drawer rail-stop handle-axis normal-credit evidence had invalid type"
            )
        typed_failure = self._drawer_rail_stop_exact_scalar_failure(
            "handle-axis normal-credit commit",
            float_values=(
                ("global_normal", evidence.global_normal_m),
                ("ledger_after", evidence.ledger_after_m),
                ("credit_increment", evidence.credit_increment_m),
            ),
        )
        if typed_failure is not None:
            return typed_failure
        if type(evidence.source) is not str:
            return self._fail(
                "drawer rail-stop handle-axis normal-credit evidence had invalid source"
            )
        finite_failure = self._drawer_rail_stop_nonfinite_failure(
            "handle-axis normal-credit commit",
            global_normal=evidence.global_normal_m,
            ledger_after=evidence.ledger_after_m,
            credit_increment=evidence.credit_increment_m,
        )
        if finite_failure is not None:
            return finite_failure
        valid_sources = {
            "none",
            "axis_retry_public_pose",
            "axis_retry_baseline_public_pose",
            "axis_retry_baseline_and_public_pose",
        }
        if (
            evidence.source not in valid_sources
            or evidence.ledger_after_m < 0.0
            or evidence.ledger_after_m > 0.075 + 1e-12
            or evidence.credit_increment_m < 0.0
            or (
                evidence.credit_increment_m > 0.0
                and evidence.source == "none"
            )
            or (
                evidence.credit_increment_m == 0.0
                and evidence.source != "none"
            )
        ):
            return self._fail(
                "drawer rail-stop handle-axis normal-credit evidence was invalid"
            )
        self._drawer_rail_stop_normal_retreat_progress_m = float(
            evidence.ledger_after_m
        )
        if evidence.credit_increment_m > 0.0:
            self._drawer_rail_stop_normal_credit_source = evidence.source
            self._drawer_rail_stop_normal_credit_increment_m = float(
                evidence.credit_increment_m
            )
        self._drawer_rail_stop_axis_chain_pending_baseline_normal_credit_m = 0.0
        self._drawer_rail_stop_axis_chain_last_verified_global_normal_m = float(
            evidence.global_normal_m
        )
        self._drawer_rail_stop_axis_chain_normal_credit_evidence = None
        return None

    def _finalize_drawer_rail_stop_axis_decision(
        self,
        decision: PolicyDecision,
    ) -> PolicyDecision:
        """Attach staged credit only to an action that has passed every gate."""

        if decision.request_stop:
            self._drawer_rail_stop_axis_chain_normal_credit_evidence = None
            return decision
        commit_failure = self._commit_drawer_rail_stop_axis_normal_credit()
        if commit_failure is not None:
            return commit_failure
        # ``decision`` was already ticked by the action builder.  Rebuild only
        # its diagnostics after the evidence commit; never tick twice.
        message = str(decision.diagnostics.get("message", ""))
        return self._decision(decision.action, message)

    def _snapshot_drawer_rail_stop_axis_retry_state(self) -> dict[str, object]:
        """Capture every retry/evidence field a nested command may mutate."""

        names = (
            "_drawer_rail_stop_axis_world",
            "_drawer_rail_stop_slide_direction",
            "_drawer_rail_stop_axis_retry_used_m",
            "_drawer_rail_stop_axis_retry_require_full_distance",
            "_drawer_rail_stop_slide_remaining_m",
            "_drawer_rail_stop_axis_total_progress_m",
            "_drawer_rail_stop_axis_partial_progress_m",
            "_drawer_rail_stop_axis_partial_segment_count",
            "_drawer_rail_stop_slide_segment_start_position",
            "_drawer_rail_stop_slide_segment_distance_m",
            "_drawer_rail_stop_axis_segment_progress_m",
            "_drawer_rail_stop_axis_segment_peak_progress_m",
            "_drawer_rail_stop_axis_segment_max_regression_m",
            "_drawer_rail_stop_axis_max_observed_regression_m",
            "_drawer_rail_stop_axis_chain_active",
            "_drawer_rail_stop_axis_chain_origin",
            "_drawer_rail_stop_axis_chain_rotation",
            "_drawer_rail_stop_axis_chain_direction",
            "_drawer_rail_stop_axis_chain_total_start_m",
            "_drawer_rail_stop_axis_chain_progress_m",
            "_drawer_rail_stop_axis_chain_cross_drift_m",
            "_drawer_rail_stop_axis_chain_rotation_error_rad",
            "_drawer_rail_stop_axis_chain_local_normal_m",
            "_drawer_rail_stop_axis_chain_third_axis_drift_m",
            "_drawer_rail_stop_axis_chain_global_normal_progress_m",
            "_drawer_rail_stop_axis_chain_global_normal_start_m",
            "_drawer_rail_stop_axis_chain_last_verified_global_normal_m",
            "_drawer_rail_stop_axis_chain_pending_baseline_normal_credit_m",
            "_drawer_rail_stop_axis_chain_baseline_normal_segment_cap_m",
            "_drawer_rail_stop_axis_chain_normal_credit_evidence",
            "_drawer_rail_stop_probe_origin",
            "_drawer_rail_stop_probe_index",
            "_drawer_rail_stop_release_width_m",
            "_drawer_rail_stop_current_public_width_m",
            "_drawer_rail_stop_body_margin_m",
            "_drawer_rail_stop_required_axis_scalar_m",
            "_drawer_rail_stop_outward_probe_origin",
            "_drawer_rail_stop_outward_probe_distance_m",
            "_drawer_rail_stop_outward_probe_progress_m",
            "_drawer_rail_stop_outward_axis_drift_m",
            "_drawer_rail_stop_outward_residual_drift_m",
            "_drawer_rail_stop_outward_net_clearance_m",
            "_drawer_rail_stop_axis_retry_settle_transition_position",
            "_drawer_rail_stop_axis_retry_settle_transition_rotation",
            "_drawer_rail_stop_axis_retry_settle_origin",
            "_drawer_rail_stop_axis_retry_settle_start_global_normal_m",
            "_drawer_rail_stop_axis_retry_settle_previous_global_normal_m",
            "_drawer_rail_stop_axis_retry_settle_delta_normal_m",
            "_drawer_rail_stop_axis_retry_settle_local_normal_m",
            "_drawer_rail_stop_axis_retry_settle_transition_normal_m",
            "_drawer_rail_stop_axis_retry_settle_rail_drift_m",
            "_drawer_rail_stop_axis_retry_settle_third_axis_drift_m",
            "_drawer_rail_stop_axis_retry_settle_rotation_error_rad",
            "_drawer_rail_stop_axis_retry_settle_stable_count",
            "_drawer_rail_stop_axis_retry_settle_anchor_frozen",
            "_drawer_rail_stop_axis_retry_settle_requested_distance_m",
            "_drawer_rail_stop_axis_retry_settle_probe_axis_drift_m",
            "_drawer_rail_stop_axis_retry_settle_required_axis_correction_m",
            "_drawer_rail_stop_axis_retry_settle_entry_net_clearance_m",
            "_drawer_rail_stop_axis_retry_settle_entry_baseline_credit_cap_m",
            "_drawer_rail_stop_axis_retry_settle_require_full_distance",
            "_drawer_rail_stop_axis_retry_settle_message",
            "_drawer_rail_stop_axis_retry_settle_failure_message",
            "_drawer_rail_stop_normal_segment_origin",
            "_drawer_rail_stop_normal_segment_distance_m",
            "_drawer_rail_stop_normal_segment_progress_m",
            "_drawer_rail_stop_normal_segment_cross_drift_m",
            "_drawer_rail_stop_normal_chain_active",
            "_drawer_rail_stop_normal_chain_origin",
            "_drawer_rail_stop_normal_chain_rotation",
            "_drawer_rail_stop_normal_chain_direction",
            "_drawer_rail_stop_normal_chain_axis",
            "_drawer_rail_stop_normal_chain_progress_m",
            "_drawer_rail_stop_normal_chain_cross_drift_m",
            "_drawer_rail_stop_normal_chain_rotation_error_rad",
            "_drawer_rail_stop_normal_retreat_progress_m",
            "_drawer_rail_stop_normal_credit_source",
            "_drawer_rail_stop_normal_credit_increment_m",
            "_motion_position",
            "_motion_rotation",
        )
        snapshot: dict[str, object] = {}
        for name in names:
            value = getattr(self, name)
            snapshot[name] = value.copy() if isinstance(value, np.ndarray) else value
        return snapshot

    def _restore_drawer_rail_stop_axis_retry_state(
        self,
        snapshot: dict[str, object],
    ) -> None:
        for name, value in snapshot.items():
            setattr(
                self,
                name,
                value.copy() if isinstance(value, np.ndarray) else value,
            )

    def _rollback_failed_drawer_rail_stop_downstream(
        self,
        snapshot: dict[str, object],
        decision: PolicyDecision,
        fallback_message: str,
    ) -> PolicyDecision:
        """Undo accounting/proof changes when a nested transition fails.

        Callers take the snapshot only after recording the current public
        measurement.  Restoring it therefore retains that measurement while
        removing credit, budget, chain, and motion state that depended on a
        downstream action which was never authorised.
        """

        if not decision.request_stop:
            return decision
        failure_message = str(
            decision.diagnostics.get("message", fallback_message)
        )
        self._restore_drawer_rail_stop_axis_retry_state(snapshot)
        return self._fail(failure_message)

    def _drawer_rail_stop_axis_chain_credit(
        self,
        requested_m: float,
    ) -> float:
        """Return only new net progress relative to the frozen chain anchor."""

        already_credited = max(
            0.0,
            self._drawer_rail_stop_axis_total_progress_m
            - self._drawer_rail_stop_axis_chain_total_start_m,
        )
        new_net_progress = max(
            0.0,
            self._drawer_rail_stop_axis_chain_progress_m - already_credited,
        )
        return min(max(0.0, requested_m), new_net_progress)

    def _freeze_drawer_rail_stop_normal_chain(
        self,
        observation: RobotObservation,
        direction: np.ndarray,
    ) -> PolicyDecision | None:
        """Freeze the pose and orthogonal rail axis for the full 75-mm retreat."""

        normal = np.asarray(direction, dtype=np.float64).copy()
        finite_failure = self._drawer_rail_stop_nonfinite_failure(
            "normal-chain freeze",
            public_pose=observation.proprio.T_world_ee,
            normal_direction=normal,
            rail_axis=self._drawer_rail_stop_axis_world,
        )
        if finite_failure is not None:
            return finite_failure
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm < 1e-8:
            return self._fail(
                "drawer rail-stop normal retreat direction was degenerate"
            )
        normal /= normal_norm
        rail_axis = np.asarray(
            self._drawer_rail_stop_axis_world,
            dtype=np.float64,
        ).copy()
        rail_axis -= normal * float(np.dot(rail_axis, normal))
        rail_norm = float(np.linalg.norm(rail_axis))
        if rail_norm < 1e-8:
            return self._fail(
                "drawer rail-stop normal retreat lacked an independent frozen rail axis"
            )
        rail_axis /= rail_norm
        self._drawer_rail_stop_normal_chain_active = True
        self._drawer_rail_stop_normal_chain_origin = (
            observation.proprio.ee_position_world.copy()
        )
        self._drawer_rail_stop_normal_chain_rotation = (
            observation.proprio.T_world_ee[:3, :3].copy()
        )
        self._drawer_rail_stop_normal_chain_direction = normal
        self._drawer_rail_stop_normal_chain_axis = rail_axis
        self._drawer_rail_stop_normal_chain_progress_m = 0.0
        self._drawer_rail_stop_normal_chain_cross_drift_m = 0.0
        self._drawer_rail_stop_normal_chain_rotation_error_rad = 0.0
        return None

    def _drawer_rail_stop_normal_chain_safety_failure(
        self,
        observation: RobotObservation,
        *,
        commit_credit_clamp: bool = True,
        require_active: bool = True,
    ) -> PolicyDecision | None:
        """Validate the full retreat against its first public pose."""

        if require_active and not self._drawer_rail_stop_normal_chain_active:
            return self._fail(
                "drawer rail-stop normal retreat lacked a frozen chain anchor"
            )
        finite_failure = self._drawer_rail_stop_nonfinite_failure(
            "normal chain",
            public_pose=observation.proprio.T_world_ee,
            chain_origin=self._drawer_rail_stop_normal_chain_origin,
            chain_rotation=self._drawer_rail_stop_normal_chain_rotation,
            normal_direction=self._drawer_rail_stop_normal_chain_direction,
            rail_axis=self._drawer_rail_stop_normal_chain_axis,
            normal_credit=self._drawer_rail_stop_normal_retreat_progress_m,
        )
        if finite_failure is not None:
            return finite_failure
        normal = self._drawer_rail_stop_normal_chain_direction
        rail_axis = self._drawer_rail_stop_normal_chain_axis
        normal_norm = float(np.linalg.norm(normal))
        rail_axis_norm = float(np.linalg.norm(rail_axis))
        if (
            abs(normal_norm - 1.0) > 1e-6
            or abs(rail_axis_norm - 1.0) > 1e-6
            or abs(float(np.dot(normal, rail_axis))) > 1e-6
        ):
            return self._fail(
                "drawer rail-stop normal retreat lacked a consistent frozen "
                "normal/rail basis"
            )
        displacement = (
            observation.proprio.ee_position_world
            - self._drawer_rail_stop_normal_chain_origin
        )
        progress = float(np.dot(displacement, normal))
        rail_progress = float(np.dot(displacement, rail_axis))
        residual = displacement - normal * progress - rail_axis * rail_progress
        cross_drift = float(np.linalg.norm(residual))
        rotation_error = self._rotation_error(
            observation.proprio.T_world_ee[:3, :3],
            self._drawer_rail_stop_normal_chain_rotation,
        )
        self._drawer_rail_stop_normal_chain_progress_m = progress
        self._drawer_rail_stop_normal_chain_cross_drift_m = cross_drift
        self._drawer_rail_stop_normal_chain_rotation_error_rad = rotation_error
        tolerance = self.config.drawer_rail_stop_outward_progress_tolerance_m
        if progress < -tolerance - 1e-12:
            return self._fail(
                "drawer rail-stop normal retreat reversed its cumulative public "
                f"EE progress ({progress:.4f} m)"
            )
        if progress > 0.075 + 1e-12:
            return self._fail(
                "drawer rail-stop normal retreat exceeded its frozen 0.075 m "
                f"global public progress bound ({progress:.4f} m)"
            )
        if (
            cross_drift
            > self.config.drawer_rail_stop_outward_max_cross_drift_m + 1e-12
        ):
            return self._fail(
                "drawer rail-stop normal retreat exceeded its cumulative public "
                f"cross-axis drift gate ({cross_drift:.4f} m)"
            )
        if rotation_error > self.config.drawer_rail_stop_outward_max_rotation_rad:
            return self._fail(
                "drawer rail-stop normal retreat lost its first frozen public "
                f"wrist pose ({rotation_error:.4f} rad)"
            )
        # A separately validated rail-axis recovery may carry a small normal
        # compliance displacement.  Never retain more accepted retreat credit
        # than the current net displacement from the original normal anchor.
        if commit_credit_clamp:
            self._drawer_rail_stop_normal_retreat_progress_m = float(
                min(
                    self._drawer_rail_stop_normal_retreat_progress_m,
                    max(0.0, progress),
                    self.config.drawer_rail_stop_normal_retreat_m,
                )
            )
        return None

    def _extend_drawer_rail_stop_for_public_width(
        self,
        observation: RobotObservation,
        *,
        schedule_axis_clearance: bool = True,
    ) -> PolicyDecision | None:
        """Extend lateral clearance if the publicly measured jaw opens wider."""

        finite_failure = self._drawer_rail_stop_nonfinite_failure(
            "public released-width update",
            public_pose=observation.proprio.T_world_ee,
            released_width=observation.proprio.gripper_width_m,
            frozen_release_width=self._drawer_rail_stop_release_width_m,
            current_public_width=self._drawer_rail_stop_current_public_width_m,
            body_margin=self._drawer_rail_stop_body_margin_m,
            slide_remaining=self._drawer_rail_stop_slide_remaining_m,
            axis_total_progress=self._drawer_rail_stop_axis_total_progress_m,
            retry_used=self._drawer_rail_stop_axis_retry_used_m,
            normal_credit=self._drawer_rail_stop_normal_retreat_progress_m,
        )
        if finite_failure is not None:
            return finite_failure
        released_width = float(observation.proprio.gripper_width_m)
        self._drawer_rail_stop_current_public_width_m = released_width
        if released_width < self.config.drawer_release_width_m:
            return self._fail(
                "drawer rail-stop clearance lost its public released-width proof"
            )
        body_margin = float(
            max(
                self.config.drawer_rail_stop_axis_endpoint_margin_m,
                0.5 * released_width
                + self.config.drawer_rail_stop_body_padding_m,
            )
        )
        if body_margin > self.config.drawer_rail_stop_body_margin_max_m:
            return self._fail(
                "drawer released-width body envelope exceeded its typed physical cap "
                f"({body_margin:.4f} m)"
            )
        self._drawer_rail_stop_release_width_m = float(
            max(
                self._drawer_rail_stop_release_width_m,
                released_width,
            )
        )
        margin_growth = float(
            body_margin - self._drawer_rail_stop_body_margin_m
        )
        if margin_growth > 0.0:
            self._drawer_rail_stop_body_margin_m = float(body_margin)
            if np.isfinite(self._drawer_rail_stop_required_axis_scalar_m):
                direction_sign = float(
                    np.dot(
                        self._drawer_rail_stop_slide_direction,
                        self._drawer_rail_stop_axis_world,
                    )
                )
                self._drawer_rail_stop_required_axis_scalar_m = float(
                    self._drawer_rail_stop_required_axis_scalar_m
                    + direction_sign * margin_growth
                )
            # The endpoint command already carries one pose-tolerance
            # allowance.  Larger aperture growth needs additional measured
            # same-axis travel.  During an outward probe the simultaneously
            # observed selected-axis drift is evaluated first, so do not
            # blindly schedule travel that may already have occurred.
            if (
                schedule_axis_clearance
                and margin_growth
                > self.config.drawer_rail_stop_axis_progress_tolerance_m
            ):
                self._drawer_rail_stop_slide_remaining_m = float(
                    self._drawer_rail_stop_slide_remaining_m + margin_growth
                )
        return None

    def _start_drawer_rail_stop_axis_clearance(
        self,
        observation: RobotObservation,
    ) -> PolicyDecision:
        """Start endpoint clearance or a finite two-direction axis probe."""

        assert self._target is not None
        finite_failure = self._drawer_rail_stop_nonfinite_failure(
            "axis-clearance start",
            public_pose=observation.proprio.T_world_ee,
            target_point=self._target.point_world,
            target_feature_axis=self._target.feature_axis_world,
            lateral_offset=self._drawer_lateral_offset,
            released_width=observation.proprio.gripper_width_m,
        )
        if finite_failure is not None:
            return finite_failure
        try:
            axis = self._drawer_handle_axis(self._target)
        except ValueError as exc:
            return self._fail(str(exc))
        current = observation.proprio.ee_position_world
        self._reset_drawer_rail_stop_motion_state()
        self._drawer_rail_stop_axis_world = axis.copy()
        released_width = float(observation.proprio.gripper_width_m)
        self._drawer_rail_stop_current_public_width_m = released_width
        if released_width < self.config.drawer_release_width_m:
            return self._fail(
                "drawer rail-stop clearance lost its public released-width proof"
            )
        body_margin = float(
            max(
                self.config.drawer_rail_stop_axis_endpoint_margin_m,
                0.5 * released_width
                + self.config.drawer_rail_stop_body_padding_m,
            )
        )
        if body_margin > self.config.drawer_rail_stop_body_margin_max_m:
            return self._fail(
                "drawer released-width body envelope exceeded its typed physical cap "
                f"({body_margin:.4f} m)"
            )
        self._drawer_rail_stop_release_width_m = released_width
        self._drawer_rail_stop_body_margin_m = float(body_margin)
        bounds = np.asarray(
            self._drawer_rail_stop_visible_axis_bounds_m,
            dtype=np.float64,
        )
        if bounds.shape == (2,) and np.all(np.isfinite(bounds)):
            current_scalar = float(
                np.dot(current - self._target.point_world, axis)
            )
            # Compensate the final public pose tolerance in the commanded
            # endpoint.  Otherwise several accepted, slightly short segments
            # could leave the fingers inside the intended physical margin.
            # Actual progress is accumulated below, so this single terminal
            # allowance guarantees that the complete public-width-derived
            # finger/palm envelope, not just the TCP, is beyond the visible
            # endpoint even when the last target is reached at tolerance.
            clear_sign = float(np.sign(np.dot(self._drawer_lateral_offset, axis)))
            choices: list[tuple[float, float, float, float]] = []
            for visible_endpoint, direction_sign in (
                (float(bounds[0]), -1.0),
                (float(bounds[1]), 1.0),
            ):
                required_scalar = (
                    visible_endpoint + direction_sign * body_margin
                )
                commanded_scalar = (
                    required_scalar
                    + direction_sign
                    * self.config.drawer_rail_stop_axis_progress_tolerance_m
                )
                distance = max(
                    0.0,
                    direction_sign * (commanded_scalar - current_scalar),
                )
                clearance_bonus = (
                    self.config.drawer_rail_stop_axis_clearance_preference_m
                    if clear_sign != 0.0 and direction_sign == clear_sign
                    else 0.0
                )
                choices.append(
                    (
                        max(0.0, distance - clearance_bonus),
                        distance,
                        direction_sign,
                        required_scalar,
                    )
                )
            _score, distance, direction_sign, required_scalar = min(
                choices,
                key=lambda item: (item[0], item[1]),
            )
            if distance > self.config.drawer_rail_stop_axis_total_max_m:
                return self._fail(
                    "drawer handle endpoint required clearance beyond the typed "
                    f"axis bound ({distance:.4f} m)"
                )
            self._drawer_rail_stop_slide_direction = axis * direction_sign
            self._drawer_rail_stop_required_axis_scalar_m = float(required_scalar)
            self._drawer_rail_stop_slide_remaining_m = float(distance)
            return self._command_next_drawer_rail_stop_axis_segment(
                observation,
                "sliding an open gripper past the public RGB-D handle endpoint",
            )

        preferred_sign = float(np.sign(np.dot(self._drawer_lateral_offset, axis)))
        if preferred_sign == 0.0:
            preferred_sign = 1.0
        self._drawer_rail_stop_probe_origin = current.copy()
        self._drawer_rail_stop_probe_index = 0
        self._drawer_rail_stop_slide_direction = axis * preferred_sign
        self._drawer_rail_stop_slide_segment_start_position = current.copy()
        self._drawer_rail_stop_slide_segment_distance_m = float(
            self.config.drawer_rail_stop_axis_probe_m
        )
        freeze_failure = self._freeze_drawer_rail_stop_axis_chain(observation)
        if freeze_failure is not None:
            return freeze_failure
        self._motion_position = (
            current
            + self._drawer_rail_stop_slide_direction
            * self.config.drawer_rail_stop_axis_probe_m
        )
        self._set_phase("drawer_rail_stop_axis_probe")
        return self._move(
            observation,
            -1.0,
            "probing the clearer public handle-axis direction with open fingers",
        )

    def _drawer_rail_stop_axis_segment_measurements(
        self,
        observation: RobotObservation,
    ) -> tuple[float, float, float, float]:
        current = observation.proprio.ee_position_world
        displacement = (
            current - self._drawer_rail_stop_slide_segment_start_position
        )
        direction = self._drawer_rail_stop_slide_direction
        signed_progress = float(np.dot(displacement, direction))
        cross_axis_drift = float(
            np.linalg.norm(displacement - direction * signed_progress)
        )
        rotation_error = self._rotation_error(
            observation.proprio.T_world_ee[:3, :3],
            self._motion_rotation,
        )
        target_error = float(np.linalg.norm(self._motion_position - current))
        return signed_progress, cross_axis_drift, rotation_error, target_error

    def _drawer_rail_stop_axis_safety_failure(
        self,
        *,
        cross_axis_drift: float,
        rotation_error: float,
    ) -> PolicyDecision | None:
        gated_cross_drift = cross_axis_drift
        if self._drawer_rail_stop_normal_chain_active:
            # The active frozen normal corridor has already bounded local normal
            # travel to 10 mm and true cumulative third-axis travel to 4 mm.
            # Do not reinterpret safe drawer-normal compliance as the older
            # undifferentiated 8-mm per-segment cross error.
            gated_cross_drift = (
                self._drawer_rail_stop_axis_chain_third_axis_drift_m
            )
        if (
            gated_cross_drift
            > self.config.drawer_rail_stop_axis_max_cross_drift_m
        ):
            return self._fail(
                "drawer rail-stop handle-axis clearance exceeded its public "
                f"cross-axis drift gate ({gated_cross_drift:.4f} m)"
            )
        if rotation_error > self.config.rotation_tolerance_rad:
            return self._fail(
                "drawer rail-stop handle-axis clearance lost its frozen wrist "
                f"pose ({rotation_error:.4f} rad)"
            )
        return None

    def _act_drawer_rail_stop_axis_probe(
        self,
        observation: RobotObservation,
    ) -> PolicyDecision:
        finite_failure = self._drawer_rail_stop_nonfinite_failure(
            "axis probe",
            public_pose=observation.proprio.T_world_ee,
            motion_position=self._motion_position,
            motion_rotation=self._motion_rotation,
            segment_origin=self._drawer_rail_stop_slide_segment_start_position,
            segment_distance=self._drawer_rail_stop_slide_segment_distance_m,
            slide_direction=self._drawer_rail_stop_slide_direction,
            axis_total_progress=self._drawer_rail_stop_axis_total_progress_m,
        )
        if finite_failure is not None:
            return finite_failure
        width_failure = self._extend_drawer_rail_stop_for_public_width(
            observation,
            schedule_axis_clearance=False,
        )
        if width_failure is not None:
            return width_failure
        cumulative_unsafe = self._drawer_rail_stop_axis_chain_safety_failure(
            observation
        )
        if cumulative_unsafe is not None:
            return cumulative_unsafe
        progress, drift, rotation_error, target_error = (
            self._drawer_rail_stop_axis_segment_measurements(observation)
        )
        unsafe = self._drawer_rail_stop_axis_safety_failure(
            cross_axis_drift=drift,
            rotation_error=rotation_error,
        )
        if unsafe is not None:
            return unsafe
        tolerance = self.config.drawer_rail_stop_axis_progress_tolerance_m
        if progress < -tolerance:
            return self._fail(
                "drawer rail-stop handle-axis probe reversed its public EE "
                f"progress ({progress:.4f} m)"
            )
        if progress > self.config.drawer_rail_stop_axis_probe_m + 1e-12:
            return self._fail(
                "drawer rail-stop handle-axis probe exceeded its commanded "
                f"micro-segment ({progress:.4f} m)"
            )
        if (
            progress
            >= self.config.drawer_rail_stop_axis_probe_m - tolerance
            and target_error <= tolerance
        ):
            completed = float(
                self._drawer_rail_stop_axis_chain_credit(progress)
            )
            self._drawer_rail_stop_axis_total_progress_m = float(
                self._drawer_rail_stop_axis_total_progress_m + completed
            )
            direction_sign = float(
                np.sign(
                    np.dot(
                        self._drawer_rail_stop_slide_direction,
                        self._drawer_rail_stop_axis_world,
                    )
                )
            )
            assert self._target is not None
            current_scalar = float(
                np.dot(
                    observation.proprio.ee_position_world - self._target.point_world,
                    self._drawer_rail_stop_axis_world,
                )
            )
            required_endpoint = float(
                direction_sign
                * (
                    0.5 * self.config.drawer_rail_stop_axis_span_max_m
                    + self._drawer_rail_stop_body_margin_m
                )
            )
            commanded_endpoint = float(
                required_endpoint
                + direction_sign
                * self.config.drawer_rail_stop_axis_progress_tolerance_m
            )
            self._drawer_rail_stop_required_axis_scalar_m = float(
                required_endpoint
            )
            remaining = float(
                direction_sign * (commanded_endpoint - current_scalar)
            )
            self._drawer_rail_stop_slide_remaining_m = float(
                max(0.0, remaining)
            )
            return self._command_next_drawer_rail_stop_axis_segment(
                observation,
                "axis probe advanced; clearing the conservative public handle span",
            )
        if self._phase_ticks >= self.config.drawer_rail_stop_axis_probe_max_ticks:
            if self._drawer_rail_stop_probe_index == 0:
                completed = float(
                    self._drawer_rail_stop_axis_chain_credit(progress)
                )
                if (
                    self._drawer_rail_stop_axis_total_progress_m + completed
                    > self.config.drawer_rail_stop_axis_total_max_m + 1e-12
                ):
                    return self._fail(
                        "drawer rail-stop handle-axis probe exceeded its typed "
                        "total axis bound"
                    )
                self._drawer_rail_stop_axis_total_progress_m = float(
                    self._drawer_rail_stop_axis_total_progress_m + completed
                )
                self._drawer_rail_stop_probe_index = 1
                self._drawer_rail_stop_slide_direction *= -1.0
                self._drawer_rail_stop_slide_segment_start_position = (
                    observation.proprio.ee_position_world.copy()
                )
                self._motion_position = (
                    observation.proprio.ee_position_world
                    + self._drawer_rail_stop_slide_direction
                    * self.config.drawer_rail_stop_axis_probe_m
                )
                freeze_failure = self._freeze_drawer_rail_stop_axis_chain(
                    observation
                )
                if freeze_failure is not None:
                    return freeze_failure
                self._set_phase("drawer_rail_stop_axis_probe")
                decision = self._move(
                    observation,
                    -1.0,
                    "first handle-axis probe blocked; trying the bounded opposite probe",
                )
                return self._finalize_drawer_rail_stop_axis_decision(decision)
            return self._fail(
                "both public handle-axis probes lacked signed EE progress "
                f"(last={progress:.4f} m; drift={drift:.4f} m; "
                f"rotation_error={rotation_error:.4f} rad)"
            )
        decision = self._move(
            observation,
            -1.0,
            "testing signed public EE progress along the handle axis",
        )
        return self._finalize_drawer_rail_stop_axis_decision(decision)

    def _command_next_drawer_rail_stop_axis_segment(
        self,
        observation: RobotObservation,
        message: str,
    ) -> PolicyDecision:
        typed_failure = self._drawer_rail_stop_exact_scalar_failure(
            "axis command",
            float_values=(
                ("slide_remaining", self._drawer_rail_stop_slide_remaining_m),
                ("axis_total_progress", self._drawer_rail_stop_axis_total_progress_m),
                ("retry_used", self._drawer_rail_stop_axis_retry_used_m),
            ),
            bool_values=(
                (
                    "require_full_distance",
                    self._drawer_rail_stop_axis_retry_require_full_distance,
                ),
                ("axis_chain_active", self._drawer_rail_stop_axis_chain_active),
                ("normal_chain_active", self._drawer_rail_stop_normal_chain_active),
            ),
        )
        if typed_failure is not None:
            return typed_failure
        finite_failure = self._drawer_rail_stop_nonfinite_failure(
            "axis command",
            public_pose=observation.proprio.T_world_ee,
            slide_direction=self._drawer_rail_stop_slide_direction,
            slide_remaining=self._drawer_rail_stop_slide_remaining_m,
            axis_total_progress=self._drawer_rail_stop_axis_total_progress_m,
            retry_used=self._drawer_rail_stop_axis_retry_used_m,
        )
        if finite_failure is not None:
            return finite_failure
        require_full = self._drawer_rail_stop_axis_retry_require_full_distance
        if type(require_full) is not bool:
            return self._fail(
                "drawer rail-stop axis command had invalid full-distance state"
            )
        current = observation.proprio.ee_position_world
        tolerance = self.config.drawer_rail_stop_axis_progress_tolerance_m
        completion_tolerance = (
            self.config.progress_epsilon_m if require_full else tolerance
        )
        if (
            self._drawer_rail_stop_axis_total_progress_m
            + self._drawer_rail_stop_slide_remaining_m
            > self.config.drawer_rail_stop_axis_total_max_m + 1e-12
        ):
            return self._fail(
                "drawer rail-stop clearance exceeded its typed total axis bound"
            )
        if self._drawer_rail_stop_slide_remaining_m <= completion_tolerance:
            if self._drawer_rail_stop_axis_chain_active:
                cumulative_unsafe = (
                    self._drawer_rail_stop_axis_chain_safety_failure(observation)
                )
                if cumulative_unsafe is not None:
                    return cumulative_unsafe
            transaction_before = (
                self._snapshot_drawer_rail_stop_axis_retry_state()
            )
            commit_failure = self._commit_drawer_rail_stop_axis_normal_credit()
            if commit_failure is not None:
                return commit_failure
            self._drawer_rail_stop_axis_chain_active = False
            self._drawer_rail_stop_axis_retry_require_full_distance = False
            decision = self._start_drawer_rail_stop_outward_probe(observation)
            if decision.request_stop:
                failure_message_from_decision = str(
                    decision.diagnostics.get(
                        "message",
                        "drawer rail-stop axis completion failed",
                    )
                )
                self._restore_drawer_rail_stop_axis_retry_state(
                    transaction_before
                )
                return self._fail(failure_message_from_decision)
            return decision
        if not self._drawer_rail_stop_axis_chain_active:
            freeze_failure = self._freeze_drawer_rail_stop_axis_chain(
                observation
            )
            if freeze_failure is not None:
                return freeze_failure
        # Keep every contiguous rail-axis recovery on the line frozen at its
        # first public pose.  Rebasing each target from ``current`` alone
        # preserves small OSC tracking errors and can turn individually safe
        # micro-segments into unsafe cumulative drift.  Correct that measured
        # orthogonal error in the next bounded command without relaxing the
        # frozen-chain gate.
        cumulative_unsafe = self._drawer_rail_stop_axis_chain_safety_failure(
            observation
        )
        if cumulative_unsafe is not None:
            return cumulative_unsafe
        segment_cap = float(
            min(
                self.config.drawer_rail_stop_axis_segment_m,
                0.010,
            )
        )
        frozen_direction = self._drawer_rail_stop_axis_chain_direction.copy()
        chain_displacement = (
            current - self._drawer_rail_stop_axis_chain_origin
        )
        chain_scalar = float(np.dot(chain_displacement, frozen_direction))
        if self._drawer_rail_stop_normal_chain_active:
            # During a post-retreat rail retry, retain bounded beneficial
            # outward displacement.  Correct only true third-axis error and any
            # harmful (negative) local normal displacement.  The unchanged
            # Cartesian cap still covers rail travel plus every correction.
            frozen_normal = (
                self._drawer_rail_stop_normal_chain_direction.copy()
            )
            local_normal = float(np.dot(chain_displacement, frozen_normal))
            third_axis_error = (
                chain_displacement
                - frozen_direction * chain_scalar
                - frozen_normal * local_normal
            )
            cross_correction = (
                third_axis_error
                + frozen_normal * min(0.0, local_normal)
            )
        else:
            cross_correction = (
                chain_displacement - frozen_direction * chain_scalar
            )
        cross_correction_m = float(np.linalg.norm(cross_correction))
        forward_room_squared = (
            segment_cap * segment_cap
            - cross_correction_m * cross_correction_m
        )
        finite_failure = self._drawer_rail_stop_nonfinite_failure(
            "axis centreline correction",
            segment_cap=segment_cap,
            frozen_direction=frozen_direction,
            chain_displacement=chain_displacement,
            chain_scalar=chain_scalar,
            cross_correction=cross_correction,
            cross_correction_m=cross_correction_m,
            forward_room_squared=forward_room_squared,
        )
        if finite_failure is not None:
            return finite_failure
        if forward_room_squared < 0.0:
            return self._fail(
                "drawer rail-stop handle-axis centreline correction exceeded "
                "its Cartesian micro-segment cap"
            )
        forward_room = float(np.sqrt(forward_room_squared))
        distance = float(
            min(
                self._drawer_rail_stop_slide_remaining_m,
                forward_room,
            )
        )
        finite_failure = self._drawer_rail_stop_nonfinite_failure(
            "axis centreline forward room",
            forward_room=forward_room,
            commanded_forward_distance=distance,
        )
        if finite_failure is not None:
            return finite_failure
        if distance <= completion_tolerance:
            return self._fail(
                "drawer rail-stop handle-axis centreline correction left "
                "insufficient bounded forward progress"
            )
        self._drawer_rail_stop_slide_segment_start_position = current.copy()
        self._drawer_rail_stop_slide_segment_distance_m = float(distance)
        self._drawer_rail_stop_axis_segment_progress_m = 0.0
        self._drawer_rail_stop_axis_segment_peak_progress_m = 0.0
        self._drawer_rail_stop_axis_segment_max_regression_m = 0.0
        self._motion_position = (
            current
            + frozen_direction * distance
            - cross_correction
        )
        self._motion_rotation = (
            self._drawer_rail_stop_axis_chain_rotation.copy()
        )
        command_displacement = self._motion_position - current
        command_distance = float(np.linalg.norm(command_displacement))
        finite_failure = self._drawer_rail_stop_nonfinite_failure(
            "axis command target",
            motion_position=self._motion_position,
            motion_rotation=self._motion_rotation,
            commanded_forward_distance=distance,
            command_displacement=command_displacement,
            commanded_distance=command_distance,
        )
        if finite_failure is not None:
            return finite_failure
        if command_distance > segment_cap + 1e-12:
            return self._fail(
                "drawer rail-stop handle-axis centreline correction exceeded "
                f"its Cartesian micro-segment cap ({command_distance:.4f} m)"
            )
        self._set_phase("drawer_rail_stop_axis_slide")
        decision = self._move(observation, -1.0, message)
        return self._finalize_drawer_rail_stop_axis_decision(decision)

    def _retarget_active_drawer_rail_stop_axis_segment(
        self,
        observation: RobotObservation,
    ) -> PolicyDecision | None:
        """Keep every retry tick on the safe rail/normal corridor within 10 mm."""

        if not self._drawer_rail_stop_normal_chain_active:
            return None
        current = observation.proprio.ee_position_world
        rail = self._drawer_rail_stop_axis_chain_direction
        normal = self._drawer_rail_stop_normal_chain_direction
        chain_displacement = (
            current - self._drawer_rail_stop_axis_chain_origin
        )
        rail_progress = float(np.dot(chain_displacement, rail))
        local_normal = float(np.dot(chain_displacement, normal))
        third_axis_error = (
            chain_displacement
            - rail * rail_progress
            - normal * local_normal
        )
        correction = (
            third_axis_error
            + normal * min(0.0, local_normal)
        )
        correction_m = float(np.linalg.norm(correction))
        segment_cap = float(
            min(
                self.config.drawer_rail_stop_axis_segment_m,
                0.010,
            )
        )
        forward_room_squared = (
            segment_cap * segment_cap - correction_m * correction_m
        )
        segment_displacement = (
            current - self._drawer_rail_stop_slide_segment_start_position
        )
        segment_progress = float(
            np.dot(segment_displacement, self._drawer_rail_stop_slide_direction)
        )
        remaining_forward = float(
            max(
                0.0,
                self._drawer_rail_stop_slide_segment_distance_m - segment_progress,
            )
        )
        finite_failure = self._drawer_rail_stop_nonfinite_failure(
            "active axis retry target",
            current=current,
            frozen_rail=rail,
            frozen_normal=normal,
            chain_displacement=chain_displacement,
            rail_progress=rail_progress,
            local_normal=local_normal,
            third_axis_error=third_axis_error,
            correction=correction,
            correction_m=correction_m,
            segment_cap=segment_cap,
            forward_room_squared=forward_room_squared,
            segment_displacement=segment_displacement,
            segment_progress=segment_progress,
            remaining_forward=remaining_forward,
        )
        if finite_failure is not None:
            return finite_failure
        if forward_room_squared < 0.0:
            return self._fail(
                "drawer rail-stop active retry correction exceeded its Cartesian "
                "micro-segment cap"
            )
        forward = float(
            min(
                remaining_forward,
                float(np.sqrt(forward_room_squared)),
            )
        )
        self._motion_position = current + rail * forward - correction
        self._motion_rotation = self._drawer_rail_stop_axis_chain_rotation.copy()
        command = self._motion_position - current
        command_m = float(np.linalg.norm(command))
        finite_failure = self._drawer_rail_stop_nonfinite_failure(
            "active axis retry bounded command",
            forward=forward,
            motion_position=self._motion_position,
            motion_rotation=self._motion_rotation,
            command=command,
            command_m=command_m,
        )
        if finite_failure is not None:
            return finite_failure
        if command_m > segment_cap + 1e-12:
            return self._fail(
                "drawer rail-stop active retry exceeded its Cartesian "
                f"micro-segment cap ({command_m:.4f} m)"
            )
        return None

    def _act_drawer_rail_stop_axis_slide(
        self,
        observation: RobotObservation,
    ) -> PolicyDecision:
        require_full = self._drawer_rail_stop_axis_retry_require_full_distance
        if type(require_full) is not bool:
            return self._fail(
                "drawer rail-stop axis slide had invalid full-distance state"
            )
        typed_failure = self._drawer_rail_stop_exact_scalar_failure(
            "axis slide",
            float_values=(
                (
                    "segment_distance",
                    self._drawer_rail_stop_slide_segment_distance_m,
                ),
                ("slide_remaining", self._drawer_rail_stop_slide_remaining_m),
                ("axis_total_progress", self._drawer_rail_stop_axis_total_progress_m),
                (
                    "segment_progress",
                    self._drawer_rail_stop_axis_segment_progress_m,
                ),
                (
                    "segment_peak_progress",
                    self._drawer_rail_stop_axis_segment_peak_progress_m,
                ),
                (
                    "segment_max_regression",
                    self._drawer_rail_stop_axis_segment_max_regression_m,
                ),
                (
                    "axis_max_observed_regression",
                    self._drawer_rail_stop_axis_max_observed_regression_m,
                ),
                (
                    "axis_partial_progress",
                    self._drawer_rail_stop_axis_partial_progress_m,
                ),
                (
                    "axis_chain_total_start",
                    self._drawer_rail_stop_axis_chain_total_start_m,
                ),
                ("axis_chain_progress", self._drawer_rail_stop_axis_chain_progress_m),
                (
                    "axis_chain_cross_drift",
                    self._drawer_rail_stop_axis_chain_cross_drift_m,
                ),
                (
                    "axis_chain_rotation_error",
                    self._drawer_rail_stop_axis_chain_rotation_error_rad,
                ),
                (
                    "axis_chain_local_normal",
                    self._drawer_rail_stop_axis_chain_local_normal_m,
                ),
                (
                    "axis_chain_third_axis_drift",
                    self._drawer_rail_stop_axis_chain_third_axis_drift_m,
                ),
                (
                    "axis_chain_global_normal_progress",
                    self._drawer_rail_stop_axis_chain_global_normal_progress_m,
                ),
                (
                    "axis_chain_global_normal_start",
                    self._drawer_rail_stop_axis_chain_global_normal_start_m,
                ),
                (
                    "axis_chain_last_verified_global_normal",
                    self._drawer_rail_stop_axis_chain_last_verified_global_normal_m,
                ),
                (
                    "axis_chain_pending_baseline_normal_credit",
                    self._drawer_rail_stop_axis_chain_pending_baseline_normal_credit_m,
                ),
                (
                    "axis_chain_baseline_normal_segment_cap",
                    self._drawer_rail_stop_axis_chain_baseline_normal_segment_cap_m,
                ),
                ("release_width", self._drawer_rail_stop_release_width_m),
                (
                    "current_public_width",
                    self._drawer_rail_stop_current_public_width_m,
                ),
                ("body_margin", self._drawer_rail_stop_body_margin_m),
                ("retry_used", self._drawer_rail_stop_axis_retry_used_m),
                (
                    "required_axis_scalar",
                    self._drawer_rail_stop_required_axis_scalar_m,
                ),
                (
                    "normal_retreat_progress",
                    self._drawer_rail_stop_normal_retreat_progress_m,
                ),
                (
                    "normal_chain_progress",
                    self._drawer_rail_stop_normal_chain_progress_m,
                ),
                (
                    "normal_chain_cross_drift",
                    self._drawer_rail_stop_normal_chain_cross_drift_m,
                ),
                (
                    "normal_chain_rotation_error",
                    self._drawer_rail_stop_normal_chain_rotation_error_rad,
                ),
                (
                    "normal_credit_increment",
                    self._drawer_rail_stop_normal_credit_increment_m,
                ),
            ),
            int_values=(
                ("phase_ticks", self._phase_ticks),
                (
                    "axis_partial_segment_count",
                    self._drawer_rail_stop_axis_partial_segment_count,
                ),
            ),
            bool_values=(
                ("require_full_distance", require_full),
                ("axis_chain_active", self._drawer_rail_stop_axis_chain_active),
                ("normal_chain_active", self._drawer_rail_stop_normal_chain_active),
            ),
        )
        if typed_failure is not None:
            return typed_failure
        finite_failure = self._drawer_rail_stop_nonfinite_failure(
            "axis slide",
            public_pose=observation.proprio.T_world_ee,
            motion_position=self._motion_position,
            motion_rotation=self._motion_rotation,
            segment_origin=self._drawer_rail_stop_slide_segment_start_position,
            segment_distance=self._drawer_rail_stop_slide_segment_distance_m,
            slide_remaining=self._drawer_rail_stop_slide_remaining_m,
            slide_direction=self._drawer_rail_stop_slide_direction,
            axis_total_progress=self._drawer_rail_stop_axis_total_progress_m,
            retry_used=self._drawer_rail_stop_axis_retry_used_m,
        )
        if finite_failure is not None:
            return finite_failure
        width_failure = self._extend_drawer_rail_stop_for_public_width(
            observation
        )
        if width_failure is not None:
            return width_failure
        cumulative_unsafe = self._drawer_rail_stop_axis_chain_safety_failure(
            observation
        )
        if cumulative_unsafe is not None:
            return cumulative_unsafe
        retarget_failure = self._retarget_active_drawer_rail_stop_axis_segment(
            observation
        )
        if retarget_failure is not None:
            return retarget_failure
        progress, drift, rotation_error, target_error = (
            self._drawer_rail_stop_axis_segment_measurements(observation)
        )
        self._drawer_rail_stop_axis_segment_progress_m = float(progress)
        previous_peak = self._drawer_rail_stop_axis_segment_peak_progress_m
        self._drawer_rail_stop_axis_segment_peak_progress_m = float(
            max(
                previous_peak,
                progress,
            )
        )
        regression = float(
            max(
                0.0,
                self._drawer_rail_stop_axis_segment_peak_progress_m - progress,
            )
        )
        self._drawer_rail_stop_axis_segment_max_regression_m = float(
            max(
                self._drawer_rail_stop_axis_segment_max_regression_m,
                regression,
            )
        )
        self._drawer_rail_stop_axis_max_observed_regression_m = float(
            max(
                self._drawer_rail_stop_axis_max_observed_regression_m,
                regression,
            )
        )
        unsafe = self._drawer_rail_stop_axis_safety_failure(
            cross_axis_drift=drift,
            rotation_error=rotation_error,
        )
        if unsafe is not None:
            return unsafe
        tolerance = self.config.drawer_rail_stop_axis_progress_tolerance_m
        completion_tolerance = (
            self.config.progress_epsilon_m if require_full else tolerance
        )
        if progress < -tolerance:
            return self._fail(
                "drawer rail-stop handle-axis segment reversed its public EE "
                f"progress ({progress:.4f} m)"
            )
        if (
            progress
            > self._drawer_rail_stop_slide_segment_distance_m
            + 1e-12
        ):
            return self._fail(
                "drawer rail-stop handle-axis segment exceeded its commanded "
                f"micro-segment ({progress:.4f} m)"
            )
        if (
            progress
            >= (
                self._drawer_rail_stop_slide_segment_distance_m
                - completion_tolerance
            )
            and target_error <= completion_tolerance
        ):
            # Consume only measured signed public-EE travel.  Treating a
            # within-tolerance endpoint as the commanded micro-segment on every
            # segment would accumulate the residual and could authorize the
            # outward/lift transition before the fingers actually cleared the
            # visible handle endpoint.
            transaction_before = (
                self._snapshot_drawer_rail_stop_axis_retry_state()
            )
            completed = float(
                min(
                    self._drawer_rail_stop_axis_chain_credit(progress),
                    self._drawer_rail_stop_slide_remaining_m,
                )
            )
            self._drawer_rail_stop_axis_total_progress_m = float(
                self._drawer_rail_stop_axis_total_progress_m + completed
            )
            self._drawer_rail_stop_slide_remaining_m = float(
                max(
                    0.0,
                    self._drawer_rail_stop_slide_remaining_m - completed,
                )
            )
            decision = self._command_next_drawer_rail_stop_axis_segment(
                observation,
                "continuing the segmented public handle-axis clearance",
            )
            if decision.request_stop:
                failure_message_from_decision = str(
                    decision.diagnostics.get(
                        "message",
                        "drawer rail-stop axis continuation failed",
                    )
                )
                self._restore_drawer_rail_stop_axis_retry_state(
                    transaction_before
                )
                return self._fail(failure_message_from_decision)
            return decision
        if self._phase_ticks >= self.config.drawer_rail_stop_axis_segment_max_ticks:
            # A compliant contact can make safe, monotonic progress without
            # reaching the endpoint tolerance before this bounded dwell ends.
            # Preserve only the public-EE travel actually observed, then rebase
            # the next <=10-mm command at the current public pose.  The retry
            # allowance was reserved when the recovery was scheduled, so this
            # continuation must not charge it again.
            monotonic = (
                self._drawer_rail_stop_axis_segment_max_regression_m
                <= self.config.progress_epsilon_m
            )
            if progress > completion_tolerance and monotonic:
                transaction_before = (
                    self._snapshot_drawer_rail_stop_axis_retry_state()
                )
                completed = float(
                    min(
                        self._drawer_rail_stop_axis_chain_credit(progress),
                        self._drawer_rail_stop_slide_remaining_m,
                    )
                )
                if (
                    self._drawer_rail_stop_axis_total_progress_m + completed
                    > self.config.drawer_rail_stop_axis_total_max_m + 1e-12
                ):
                    return self._fail(
                        "drawer rail-stop partial handle-axis progress exceeded "
                        "its typed total axis bound"
                    )
                self._drawer_rail_stop_axis_total_progress_m = float(
                    self._drawer_rail_stop_axis_total_progress_m + completed
                )
                self._drawer_rail_stop_slide_remaining_m = float(
                    max(
                        0.0,
                        self._drawer_rail_stop_slide_remaining_m - completed,
                    )
                )
                self._drawer_rail_stop_axis_partial_progress_m = float(
                    self._drawer_rail_stop_axis_partial_progress_m + completed
                )
                self._drawer_rail_stop_axis_partial_segment_count += 1
                decision = self._command_next_drawer_rail_stop_axis_segment(
                    observation,
                    "continuing only the measured residual of a safe partial "
                    "handle-axis micro-segment",
                )
                if decision.request_stop:
                    failure_message_from_decision = str(
                        decision.diagnostics.get(
                            "message",
                            "drawer rail-stop partial axis continuation failed",
                        )
                    )
                    self._restore_drawer_rail_stop_axis_retry_state(
                        transaction_before
                    )
                    return self._fail(failure_message_from_decision)
                return decision
            return self._fail(
                "drawer rail-stop handle-axis segment lacked signed public EE "
                f"progress (progress={progress:.4f} m; "
                f"target_error={target_error:.4f} m; drift={drift:.4f} m; "
                f"rotation_error={rotation_error:.4f} rad; "
                "monotonic="
                f"{monotonic}; max_regression="
                f"{self._drawer_rail_stop_axis_segment_max_regression_m:.4f} m)"
            )
        decision = self._move(
            observation,
            -1.0,
            "sliding open fingers along the frozen public handle axis",
        )
        return self._finalize_drawer_rail_stop_axis_decision(decision)

    def _start_drawer_rail_stop_outward_probe(
        self,
        observation: RobotObservation,
    ) -> PolicyDecision:
        """Require measured free-space progress normal to the drawer front."""

        finite_failure = self._drawer_rail_stop_nonfinite_failure(
            "normal-probe start",
            public_pose=observation.proprio.T_world_ee,
            pull_direction=self._drawer_pull_direction,
            slide_direction=self._drawer_rail_stop_slide_direction,
            axis_world=self._drawer_rail_stop_axis_world,
            slide_remaining=self._drawer_rail_stop_slide_remaining_m,
            axis_total_progress=self._drawer_rail_stop_axis_total_progress_m,
            retry_used=self._drawer_rail_stop_axis_retry_used_m,
            required_axis_scalar=self._drawer_rail_stop_required_axis_scalar_m,
        )
        if finite_failure is not None:
            return finite_failure
        width_failure = self._extend_drawer_rail_stop_for_public_width(
            observation
        )
        if width_failure is not None:
            return width_failure
        current = observation.proprio.ee_position_world
        direction = np.asarray(self._drawer_pull_direction, dtype=np.float64).copy()
        norm = float(np.linalg.norm(direction))
        if norm < 1e-8:
            return self._fail("drawer rail-stop outward direction was degenerate")
        direction /= norm
        if not self._drawer_rail_stop_normal_chain_active:
            freeze_failure = self._freeze_drawer_rail_stop_normal_chain(
                observation,
                direction,
            )
            if freeze_failure is not None:
                return freeze_failure
        cumulative_unsafe = self._drawer_rail_stop_normal_chain_safety_failure(
            observation
        )
        if cumulative_unsafe is not None:
            return cumulative_unsafe
        if (
            self._drawer_rail_stop_slide_remaining_m
            > self.config.drawer_rail_stop_axis_progress_tolerance_m
        ):
            return self._command_next_drawer_rail_stop_axis_segment(
                observation,
                "public released width enlarged the bounded handle-axis envelope",
            )
        remaining = max(
            0.0,
            self.config.drawer_rail_stop_normal_retreat_m
            - self._drawer_rail_stop_normal_retreat_progress_m,
        )
        if remaining <= 1e-12:
            return self._start_drawer_rail_stop_normal_segment(observation)
        probe_distance = float(
            min(
                self.config.drawer_rail_stop_outward_probe_m,
                remaining,
            )
        )
        self._drawer_rail_stop_outward_probe_origin = current.copy()
        self._drawer_rail_stop_outward_probe_distance_m = float(probe_distance)
        self._drawer_rail_stop_outward_probe_progress_m = 0.0
        self._motion_position = (
            current
            + direction * probe_distance
        )
        self._motion_rotation = (
            self._drawer_rail_stop_normal_chain_rotation.copy()
        )
        self._set_phase("drawer_rail_stop_outward_probe")
        return self._move(
            observation,
            -1.0,
            "probing the public drawer-normal retreat corridor after handle clearance",
        )

    def _start_drawer_rail_stop_axis_retry_settle(
        self,
        observation: RobotObservation,
        requested_distance_m: float,
        *,
        uncredited_probe_normal_m: float,
        probe_axis_drift_m: float,
        require_full_distance: bool,
        message: str,
        failure_message: str,
    ) -> PolicyDecision:
        """Hold open once before freezing a post-probe rail-retry origin.

        Removing the final normal command can release a short compliance peak.
        The transition emits exactly one zero-Cartesian OPEN hold; the first
        returned public pose then becomes an immutable settle anchor.  No pose
        observed here is rail, retry, or normal-retreat credit.
        """

        typed_failure = self._drawer_rail_stop_exact_scalar_failure(
            "axis-retry settle start",
            float_values=(
                ("requested_distance", requested_distance_m),
                ("uncredited_probe_normal", uncredited_probe_normal_m),
                ("probe_axis_drift", probe_axis_drift_m),
                ("normal_credit", self._drawer_rail_stop_normal_retreat_progress_m),
                ("axis_total_progress", self._drawer_rail_stop_axis_total_progress_m),
                ("retry_used", self._drawer_rail_stop_axis_retry_used_m),
                ("release_width", self._drawer_rail_stop_release_width_m),
                (
                    "current_public_width",
                    self._drawer_rail_stop_current_public_width_m,
                ),
                ("body_margin", self._drawer_rail_stop_body_margin_m),
                ("slide_remaining", self._drawer_rail_stop_slide_remaining_m),
                (
                    "required_axis_scalar",
                    self._drawer_rail_stop_required_axis_scalar_m,
                ),
                (
                    "outward_probe_distance",
                    self._drawer_rail_stop_outward_probe_distance_m,
                ),
            ),
            bool_values=(
                ("require_full_distance", require_full_distance),
                ("normal_chain_active", self._drawer_rail_stop_normal_chain_active),
                ("axis_chain_active", self._drawer_rail_stop_axis_chain_active),
            ),
        )
        if typed_failure is not None:
            return typed_failure
        finite_failure = self._drawer_rail_stop_nonfinite_failure(
            "axis-retry settle start",
            public_pose=observation.proprio.T_world_ee,
            requested_distance=requested_distance_m,
            uncredited_probe_normal=uncredited_probe_normal_m,
            probe_axis_drift=probe_axis_drift_m,
            outward_probe_origin=self._drawer_rail_stop_outward_probe_origin,
            slide_direction=self._drawer_rail_stop_slide_direction,
            normal_chain_origin=self._drawer_rail_stop_normal_chain_origin,
            normal_chain_rotation=self._drawer_rail_stop_normal_chain_rotation,
            normal_chain_direction=self._drawer_rail_stop_normal_chain_direction,
            normal_chain_axis=self._drawer_rail_stop_normal_chain_axis,
            normal_credit=self._drawer_rail_stop_normal_retreat_progress_m,
            axis_total_progress=self._drawer_rail_stop_axis_total_progress_m,
            retry_used=self._drawer_rail_stop_axis_retry_used_m,
        )
        if finite_failure is not None:
            return finite_failure
        if type(message) is not str or type(failure_message) is not str:
            return self._fail(
                "drawer rail-stop axis-retry settle had invalid typed request"
            )
        if not self._drawer_rail_stop_normal_chain_active:
            return self._fail(
                "drawer rail-stop axis-retry settle lacked a frozen normal chain"
            )
        if self._drawer_rail_stop_axis_chain_active:
            return self._fail(
                "drawer rail-stop axis-retry settle cannot rebase an active rail chain"
            )
        width_failure = self._extend_drawer_rail_stop_for_public_width(
            observation,
            schedule_axis_clearance=False,
        )
        if width_failure is not None:
            return width_failure
        normal_unsafe = self._drawer_rail_stop_normal_chain_safety_failure(
            observation,
            commit_credit_clamp=False,
        )
        if normal_unsafe is not None:
            return normal_unsafe
        net_clearance = self._drawer_rail_stop_net_axis_clearance(
            observation.proprio.ee_position_world
        )
        if net_clearance is None:
            return self._fail(
                "drawer rail-stop axis-retry settle lacked a frozen RGB-D "
                "endpoint bound"
            )

        current = observation.proprio.ee_position_world
        selected = np.asarray(
            self._drawer_rail_stop_slide_direction,
            dtype=np.float64,
        ).copy()
        selected_norm = float(np.linalg.norm(selected))
        if selected_norm < 1e-8:
            return self._fail(
                "drawer rail-stop axis-retry settle lacked a selected rail axis"
            )
        selected /= selected_norm
        measured_probe_axis_drift = float(
            np.dot(
                current - self._drawer_rail_stop_outward_probe_origin,
                selected,
            )
        )
        if abs(measured_probe_axis_drift - probe_axis_drift_m) > 1e-9:
            return self._fail(
                "drawer rail-stop axis-retry settle probe-axis evidence did not "
                "match its adjacent public probe"
            )
        required_axis_correction = float(
            -net_clearance
            + self.config.drawer_rail_stop_axis_progress_tolerance_m
            if net_clearance < 0.0
            else 0.0
        )
        retry_room = float(
            max(
                0.0,
                self.config.drawer_rail_stop_axis_retry_max_m
                - self._drawer_rail_stop_axis_retry_used_m,
            )
        )
        total_room = float(
            max(
                0.0,
                self.config.drawer_rail_stop_axis_total_max_m
                - self._drawer_rail_stop_axis_total_progress_m,
            )
        )
        available_axis = float(min(retry_room, total_room))
        finite_failure = self._drawer_rail_stop_nonfinite_failure(
            "axis-retry settle endpoint correction",
            measured_probe_axis_drift=measured_probe_axis_drift,
            required_axis_correction=required_axis_correction,
            retry_room=retry_room,
            total_room=total_room,
            available_axis=available_axis,
        )
        if finite_failure is not None:
            return finite_failure
        if required_axis_correction > available_axis + 1e-12:
            return self._fail(
                "drawer rail-stop axis-retry settle required endpoint correction "
                "beyond the remaining bounded axis budget"
            )
        settled_requested_distance = float(
            required_axis_correction
            if required_axis_correction > 0.0
            else requested_distance_m
        )
        settled_require_full_distance = bool(
            True if required_axis_correction > 0.0 else require_full_distance
        )
        current_net_gap = float(
            max(
                0.0,
                self._drawer_rail_stop_normal_chain_progress_m
                - self._drawer_rail_stop_normal_retreat_progress_m,
            )
        )
        entry_baseline_credit_cap = float(
            min(
                max(0.0, uncredited_probe_normal_m),
                current_net_gap,
            )
        )
        finite_failure = self._drawer_rail_stop_nonfinite_failure(
            "axis-retry settle entry baseline",
            uncredited_probe_normal=uncredited_probe_normal_m,
            current_net_gap=current_net_gap,
            entry_baseline_credit_cap=entry_baseline_credit_cap,
            preceding_outward_segment_cap=(
                self._drawer_rail_stop_outward_probe_distance_m
            ),
        )
        if finite_failure is not None:
            return finite_failure
        if (
            uncredited_probe_normal_m < 0.0
            or uncredited_probe_normal_m
            > self._drawer_rail_stop_outward_probe_distance_m + 1e-12
            or entry_baseline_credit_cap
            > self._drawer_rail_stop_outward_probe_distance_m + 1e-12
            or self._drawer_rail_stop_outward_probe_distance_m > 0.010 + 1e-12
        ):
            return self._fail(
                "drawer rail-stop axis-retry settle entry normal gap exceeded "
                "its preceding bounded public probe"
            )
        self._drawer_rail_stop_outward_net_clearance_m = float(net_clearance)
        self._drawer_rail_stop_axis_retry_settle_transition_position = (
            current.copy()
        )
        self._drawer_rail_stop_axis_retry_settle_transition_rotation = (
            observation.proprio.T_world_ee[:3, :3].copy()
        )
        self._drawer_rail_stop_axis_retry_settle_origin = np.zeros(3)
        self._drawer_rail_stop_axis_retry_settle_start_global_normal_m = (
            float(self._drawer_rail_stop_normal_chain_progress_m)
        )
        self._drawer_rail_stop_axis_retry_settle_previous_global_normal_m = (
            float(self._drawer_rail_stop_normal_chain_progress_m)
        )
        self._drawer_rail_stop_axis_retry_settle_delta_normal_m = 0.0
        self._drawer_rail_stop_axis_retry_settle_local_normal_m = 0.0
        self._drawer_rail_stop_axis_retry_settle_transition_normal_m = 0.0
        self._drawer_rail_stop_axis_retry_settle_rail_drift_m = 0.0
        self._drawer_rail_stop_axis_retry_settle_third_axis_drift_m = 0.0
        self._drawer_rail_stop_axis_retry_settle_rotation_error_rad = 0.0
        self._drawer_rail_stop_axis_retry_settle_stable_count = 0
        self._drawer_rail_stop_axis_retry_settle_anchor_frozen = False
        self._drawer_rail_stop_axis_retry_settle_requested_distance_m = (
            float(settled_requested_distance)
        )
        self._drawer_rail_stop_axis_retry_settle_probe_axis_drift_m = (
            float(measured_probe_axis_drift)
        )
        self._drawer_rail_stop_axis_retry_settle_required_axis_correction_m = (
            float(required_axis_correction)
        )
        self._drawer_rail_stop_axis_retry_settle_entry_net_clearance_m = (
            float(net_clearance)
        )
        self._drawer_rail_stop_axis_retry_settle_entry_baseline_credit_cap_m = (
            float(entry_baseline_credit_cap)
        )
        self._drawer_rail_stop_axis_retry_settle_require_full_distance = bool(
            settled_require_full_distance
        )
        self._drawer_rail_stop_axis_retry_settle_message = message
        self._drawer_rail_stop_axis_retry_settle_failure_message = failure_message
        self._motion_position = current.copy()
        self._motion_rotation = (
            observation.proprio.T_world_ee[:3, :3].copy()
        )
        # Settling is a zero-Cartesian observation phase, not an active normal
        # or rail motion chain.  Preserve the frozen basis/anchor values above,
        # but make both activity flags unambiguously false before the first
        # OPEN hold is emitted.  The normal flag is restored transactionally
        # only when a proven-stable endpoint leaves this phase.
        self._drawer_rail_stop_normal_chain_active = False
        self._drawer_rail_stop_axis_chain_active = False
        self._set_phase("drawer_rail_stop_axis_retry_settle")
        return self._tick(
            OSCAction.hold(-1.0),
            "holding the released gripper for bounded public normal-pose settling",
        )

    def _finish_drawer_rail_stop_axis_retry_settle(
        self,
        observation: RobotObservation,
    ) -> PolicyDecision:
        """Leave the bounded hold through the existing retry transaction."""

        requested = self._drawer_rail_stop_axis_retry_settle_requested_distance_m
        require_full = (
            self._drawer_rail_stop_axis_retry_settle_require_full_distance
        )
        message = self._drawer_rail_stop_axis_retry_settle_message
        failure_message = (
            self._drawer_rail_stop_axis_retry_settle_failure_message
        )
        typed_failure = self._drawer_rail_stop_exact_scalar_failure(
            "axis-retry settle completion",
            float_values=(("requested_distance", requested),),
            int_values=((
                "stable_count",
                self._drawer_rail_stop_axis_retry_settle_stable_count,
            ),),
            bool_values=(("require_full_distance", require_full),),
        )
        if typed_failure is not None:
            return typed_failure
        if type(message) is not str or type(failure_message) is not str:
            return self._fail(
                "drawer rail-stop axis-retry settle had invalid pending request"
            )
        finite_failure = self._drawer_rail_stop_nonfinite_failure(
            "axis-retry settle completion",
            public_pose=observation.proprio.T_world_ee,
            requested_distance=requested,
            stable_count=self._drawer_rail_stop_axis_retry_settle_stable_count,
            probe_axis_drift=(
                self._drawer_rail_stop_axis_retry_settle_probe_axis_drift_m
            ),
            required_axis_correction=(
                self._drawer_rail_stop_axis_retry_settle_required_axis_correction_m
            ),
            entry_net_clearance=(
                self._drawer_rail_stop_axis_retry_settle_entry_net_clearance_m
            ),
            entry_baseline_credit_cap=(
                self._drawer_rail_stop_axis_retry_settle_entry_baseline_credit_cap_m
            ),
            normal_credit=self._drawer_rail_stop_normal_retreat_progress_m,
            axis_total_progress=self._drawer_rail_stop_axis_total_progress_m,
            retry_used=self._drawer_rail_stop_axis_retry_used_m,
        )
        if finite_failure is not None:
            return finite_failure
        if (
            self._drawer_rail_stop_normal_chain_active
            or self._drawer_rail_stop_axis_chain_active
        ):
            return self._fail(
                "drawer rail-stop axis-retry settle completion had an active "
                "motion chain"
            )
        transaction_before = self._snapshot_drawer_rail_stop_axis_retry_state()
        self._drawer_rail_stop_normal_chain_active = True
        decision = self._schedule_drawer_rail_stop_axis_retry(
            observation,
            requested,
            require_full_distance=require_full,
            message=message,
            failure_message=failure_message,
        )
        return self._rollback_failed_drawer_rail_stop_downstream(
            transaction_before,
            decision,
            failure_message,
        )

    def _act_drawer_rail_stop_axis_retry_settle(
        self,
        observation: RobotObservation,
    ) -> PolicyDecision:
        """Bound one compliance recoil, then prove a stable public normal pose."""

        anchor_frozen = self._drawer_rail_stop_axis_retry_settle_anchor_frozen
        require_full = (
            self._drawer_rail_stop_axis_retry_settle_require_full_distance
        )
        stable_count = self._drawer_rail_stop_axis_retry_settle_stable_count
        phase_ticks = self._phase_ticks
        typed_failure = self._drawer_rail_stop_exact_scalar_failure(
            "axis-retry settle",
            float_values=(
                (
                    "start_global_normal",
                    self._drawer_rail_stop_axis_retry_settle_start_global_normal_m,
                ),
                (
                    "previous_global_normal",
                    self._drawer_rail_stop_axis_retry_settle_previous_global_normal_m,
                ),
                ("delta_normal", self._drawer_rail_stop_axis_retry_settle_delta_normal_m),
                ("local_normal", self._drawer_rail_stop_axis_retry_settle_local_normal_m),
                (
                    "transition_normal",
                    self._drawer_rail_stop_axis_retry_settle_transition_normal_m,
                ),
                ("rail_drift", self._drawer_rail_stop_axis_retry_settle_rail_drift_m),
                (
                    "third_axis_drift",
                    self._drawer_rail_stop_axis_retry_settle_third_axis_drift_m,
                ),
                (
                    "rotation_error",
                    self._drawer_rail_stop_axis_retry_settle_rotation_error_rad,
                ),
                (
                    "requested_distance",
                    self._drawer_rail_stop_axis_retry_settle_requested_distance_m,
                ),
                (
                    "probe_axis_drift",
                    self._drawer_rail_stop_axis_retry_settle_probe_axis_drift_m,
                ),
                (
                    "required_axis_correction",
                    self._drawer_rail_stop_axis_retry_settle_required_axis_correction_m,
                ),
                (
                    "entry_net_clearance",
                    self._drawer_rail_stop_axis_retry_settle_entry_net_clearance_m,
                ),
                (
                    "entry_baseline_credit_cap",
                    self._drawer_rail_stop_axis_retry_settle_entry_baseline_credit_cap_m,
                ),
                (
                    "normal_retreat_progress",
                    self._drawer_rail_stop_normal_retreat_progress_m,
                ),
                (
                    "axis_total_progress",
                    self._drawer_rail_stop_axis_total_progress_m,
                ),
                ("axis_retry_used", self._drawer_rail_stop_axis_retry_used_m),
                ("release_width", self._drawer_rail_stop_release_width_m),
                (
                    "current_public_width",
                    self._drawer_rail_stop_current_public_width_m,
                ),
                ("body_margin", self._drawer_rail_stop_body_margin_m),
                ("slide_remaining", self._drawer_rail_stop_slide_remaining_m),
                (
                    "required_axis_scalar",
                    self._drawer_rail_stop_required_axis_scalar_m,
                ),
                (
                    "outward_probe_distance",
                    self._drawer_rail_stop_outward_probe_distance_m,
                ),
            ),
            int_values=(
                ("stable_count", stable_count),
                ("phase_ticks", phase_ticks),
            ),
            bool_values=(
                ("anchor_frozen", anchor_frozen),
                ("require_full_distance", require_full),
                ("normal_chain_active", self._drawer_rail_stop_normal_chain_active),
                ("axis_chain_active", self._drawer_rail_stop_axis_chain_active),
            ),
        )
        if typed_failure is not None:
            return typed_failure
        if (
            stable_count < 0
            or stable_count
            >= self.config.drawer_rail_stop_axis_retry_settle_stable_ticks
            or phase_ticks < 1
            or phase_ticks
            > self.config.drawer_rail_stop_axis_retry_settle_max_ticks
            or type(
                self._drawer_rail_stop_axis_retry_settle_message,
            ) is not str
            or type(
                self._drawer_rail_stop_axis_retry_settle_failure_message,
            ) is not str
            or self._drawer_rail_stop_normal_chain_active
            or self._drawer_rail_stop_axis_chain_active
        ):
            return self._fail(
                "drawer rail-stop axis-retry settle had invalid typed state"
            )
        finite_failure = self._drawer_rail_stop_nonfinite_failure(
            "axis-retry settle",
            public_pose=observation.proprio.T_world_ee,
            transition_position=(
                self._drawer_rail_stop_axis_retry_settle_transition_position
            ),
            transition_rotation=(
                self._drawer_rail_stop_axis_retry_settle_transition_rotation
            ),
            settle_origin=self._drawer_rail_stop_axis_retry_settle_origin,
            start_global_normal=(
                self._drawer_rail_stop_axis_retry_settle_start_global_normal_m
            ),
            previous_global_normal=(
                self._drawer_rail_stop_axis_retry_settle_previous_global_normal_m
            ),
            delta_normal=self._drawer_rail_stop_axis_retry_settle_delta_normal_m,
            local_normal=self._drawer_rail_stop_axis_retry_settle_local_normal_m,
            transition_normal=(
                self._drawer_rail_stop_axis_retry_settle_transition_normal_m
            ),
            rail_drift=self._drawer_rail_stop_axis_retry_settle_rail_drift_m,
            third_axis_drift=(
                self._drawer_rail_stop_axis_retry_settle_third_axis_drift_m
            ),
            rotation_error=(
                self._drawer_rail_stop_axis_retry_settle_rotation_error_rad
            ),
            stable_count=stable_count,
            phase_ticks=phase_ticks,
            requested_distance=(
                self._drawer_rail_stop_axis_retry_settle_requested_distance_m
            ),
            probe_axis_drift=(
                self._drawer_rail_stop_axis_retry_settle_probe_axis_drift_m
            ),
            required_axis_correction=(
                self._drawer_rail_stop_axis_retry_settle_required_axis_correction_m
            ),
            entry_net_clearance=(
                self._drawer_rail_stop_axis_retry_settle_entry_net_clearance_m
            ),
            entry_baseline_credit_cap=(
                self._drawer_rail_stop_axis_retry_settle_entry_baseline_credit_cap_m
            ),
            normal_credit=self._drawer_rail_stop_normal_retreat_progress_m,
            axis_total_progress=self._drawer_rail_stop_axis_total_progress_m,
            retry_used=self._drawer_rail_stop_axis_retry_used_m,
        )
        if finite_failure is not None:
            return finite_failure
        width_failure = self._extend_drawer_rail_stop_for_public_width(
            observation,
            schedule_axis_clearance=False,
        )
        if width_failure is not None:
            return width_failure
        normal_unsafe = self._drawer_rail_stop_normal_chain_safety_failure(
            observation,
            commit_credit_clamp=False,
            require_active=False,
        )
        if normal_unsafe is not None:
            return normal_unsafe

        current = observation.proprio.ee_position_world
        normal = self._drawer_rail_stop_normal_chain_direction
        rail = self._drawer_rail_stop_normal_chain_axis
        reference = (
            self._drawer_rail_stop_axis_retry_settle_origin
            if anchor_frozen
            else self._drawer_rail_stop_axis_retry_settle_transition_position
        )
        displacement = current - reference
        transition_displacement = (
            current
            - self._drawer_rail_stop_axis_retry_settle_transition_position
        )
        local_normal = float(np.dot(displacement, normal))
        transition_normal = float(np.dot(transition_displacement, normal))
        local_rail_progress = float(np.dot(displacement, rail))
        # Rail displacement is never credit during a zero-command hold.  Keep
        # it cumulative from the transition pose even after the one-shot
        # normal-recoil anchor freezes, so the warm-up cannot rebase it away.
        rail_progress = float(np.dot(transition_displacement, rail))
        local_third_axis_drift = float(
            np.linalg.norm(
                displacement
                - normal * local_normal
                - rail * local_rail_progress
            )
        )
        transition_third_axis_drift = float(
            np.linalg.norm(
                transition_displacement
                - normal * transition_normal
                - rail * rail_progress
            )
        )
        rotation_error = self._rotation_error(
            observation.proprio.T_world_ee[:3, :3],
            self._drawer_rail_stop_axis_retry_settle_transition_rotation,
        )
        global_normal = float(self._drawer_rail_stop_normal_chain_progress_m)
        delta_normal = float(
            global_normal
            - self._drawer_rail_stop_axis_retry_settle_previous_global_normal_m
        )
        net_clearance = self._drawer_rail_stop_net_axis_clearance(current)
        finite_failure = self._drawer_rail_stop_nonfinite_failure(
            "axis-retry settle measurement",
            current=current,
            normal=normal,
            rail=rail,
            displacement=displacement,
            transition_displacement=transition_displacement,
            local_normal=local_normal,
            transition_normal=transition_normal,
            local_rail_progress=local_rail_progress,
            rail_progress=rail_progress,
            local_third_axis_drift=local_third_axis_drift,
            transition_third_axis_drift=transition_third_axis_drift,
            rotation_error=rotation_error,
            global_normal=global_normal,
            delta_normal=delta_normal,
            net_clearance=(float("nan") if net_clearance is None else net_clearance),
        )
        if finite_failure is not None:
            return finite_failure
        assert net_clearance is not None
        instantaneous_axis_correction = float(
            -net_clearance
            + self.config.drawer_rail_stop_axis_progress_tolerance_m
            if net_clearance < 0.0
            else 0.0
        )
        required_axis_correction = float(
            max(
                self._drawer_rail_stop_axis_retry_settle_required_axis_correction_m,
                instantaneous_axis_correction,
            )
        )
        retry_room = max(
            0.0,
            self.config.drawer_rail_stop_axis_retry_max_m
            - self._drawer_rail_stop_axis_retry_used_m,
        )
        total_room = max(
            0.0,
            self.config.drawer_rail_stop_axis_total_max_m
            - self._drawer_rail_stop_axis_total_progress_m,
        )
        available_axis = min(retry_room, total_room)
        finite_failure = self._drawer_rail_stop_nonfinite_failure(
            "axis-retry settle live endpoint correction",
            instantaneous_axis_correction=instantaneous_axis_correction,
            required_axis_correction=required_axis_correction,
            retry_room=retry_room,
            total_room=total_room,
            available_axis=available_axis,
        )
        if finite_failure is not None:
            return finite_failure
        if required_axis_correction > available_axis + 1e-12:
            return self._fail(
                "drawer rail-stop axis-retry settle required endpoint correction "
                "beyond the remaining bounded axis budget"
            )
        if required_axis_correction > 0.0:
            self._drawer_rail_stop_axis_retry_settle_requested_distance_m = (
                float(required_axis_correction)
            )
            self._drawer_rail_stop_axis_retry_settle_require_full_distance = True
            self._drawer_rail_stop_axis_retry_settle_required_axis_correction_m = (
                float(required_axis_correction)
            )
        self._drawer_rail_stop_axis_retry_settle_delta_normal_m = float(delta_normal)
        self._drawer_rail_stop_axis_retry_settle_local_normal_m = float(local_normal)
        self._drawer_rail_stop_axis_retry_settle_transition_normal_m = (
            float(transition_normal)
        )
        self._drawer_rail_stop_axis_retry_settle_rail_drift_m = float(rail_progress)
        self._drawer_rail_stop_axis_retry_settle_third_axis_drift_m = (
            float(transition_third_axis_drift)
        )
        self._drawer_rail_stop_axis_retry_settle_rotation_error_rad = (
            float(rotation_error)
        )
        self._drawer_rail_stop_outward_net_clearance_m = float(net_clearance)

        if not anchor_frozen:
            warmup_reverse = (
                self.config.drawer_rail_stop_axis_retry_settle_warmup_reverse_max_m
            )
            if transition_normal < -warmup_reverse - 1e-12:
                return self._fail(
                    "drawer rail-stop axis-retry settle warm-up exceeded its "
                    f"public reverse bound ({transition_normal:.4f} m)"
                )
        else:
            reverse_tolerance = (
                self.config.drawer_rail_stop_outward_progress_tolerance_m
            )
            if local_normal < -reverse_tolerance - 1e-12:
                return self._fail(
                    "drawer rail-stop axis-retry settle reversed its immutable "
                    f"public normal anchor ({local_normal:.4f} m)"
                )
        if transition_normal > 0.010 + 1e-12:
            return self._fail(
                "drawer rail-stop axis-retry settle exceeded its transition-"
                "cumulative 0.010 m positive normal bound "
                f"({transition_normal:.4f} m)"
            )
        if (
            abs(rail_progress)
            > self.config.drawer_rail_stop_axis_progress_tolerance_m + 1e-12
        ):
            return self._fail(
                "drawer rail-stop axis-retry settle exceeded its uncredited "
                f"public rail-drift gate ({rail_progress:.4f} m)"
            )
        if (
            transition_third_axis_drift
            > self.config.drawer_rail_stop_outward_max_cross_drift_m + 1e-12
        ):
            return self._fail(
                "drawer rail-stop axis-retry settle exceeded its transition-"
                "cumulative public third-axis drift gate "
                f"({transition_third_axis_drift:.4f} m)"
            )
        if rotation_error > self.config.drawer_rail_stop_outward_max_rotation_rad:
            return self._fail(
                "drawer rail-stop axis-retry settle lost its frozen public "
                f"wrist pose ({rotation_error:.4f} rad)"
            )

        if (
            not anchor_frozen
            and self._phase_ticks
            >= self.config.drawer_rail_stop_axis_retry_settle_max_ticks
        ):
            return self._fail(
                "drawer rail-stop axis-retry settle reached its bounded timeout "
                "before its one-shot public recoil anchor"
            )
        if not anchor_frozen:
            # Exactly one public sample may absorb the bounded compliance
            # recoil caused by replacing the probe command with a zero-motion
            # hold.  This anchor is never updated again.
            self._drawer_rail_stop_axis_retry_settle_origin = current.copy()
            self._drawer_rail_stop_axis_retry_settle_previous_global_normal_m = (
                float(global_normal)
            )
            self._drawer_rail_stop_axis_retry_settle_stable_count = 0
            self._drawer_rail_stop_axis_retry_settle_anchor_frozen = True
            return self._tick(
                OSCAction.hold(-1.0),
                "froze the one-shot public recoil sample; proving normal-pose stability",
            )

        stability = (
            self.config.drawer_rail_stop_axis_retry_settle_stability_m
        )
        if abs(delta_normal) <= stability + 1e-12:
            stable_count = int(stable_count + 1)
        else:
            stable_count = 0
        self._drawer_rail_stop_axis_retry_settle_stable_count = int(stable_count)
        self._drawer_rail_stop_axis_retry_settle_previous_global_normal_m = (
            float(global_normal)
        )
        if (
            stable_count
            >= self.config.drawer_rail_stop_axis_retry_settle_stable_ticks
        ):
            return self._finish_drawer_rail_stop_axis_retry_settle(observation)
        if (
            self._phase_ticks
            >= self.config.drawer_rail_stop_axis_retry_settle_max_ticks
        ):
            return self._fail(
                "drawer rail-stop axis-retry settle reached its bounded timeout "
                "without two consecutive stable public normal poses"
            )
        return self._tick(
            OSCAction.hold(-1.0),
            "holding OPEN until the bounded public drawer-normal pose settles",
        )

    def _schedule_drawer_rail_stop_axis_retry(
        self,
        observation: RobotObservation,
        requested_distance_m: float,
        *,
        require_full_distance: bool,
        message: str,
        failure_message: str,
    ) -> PolicyDecision:
        """Continue only the frozen selected handle-axis direction within caps."""

        if type(require_full_distance) is not bool:
            return self._fail(
                "drawer rail-stop axis-retry schedule had invalid full-distance request"
            )
        typed_failure = self._drawer_rail_stop_exact_scalar_failure(
            "axis-retry schedule",
            float_values=(
                ("requested_distance", requested_distance_m),
                ("retry_used", self._drawer_rail_stop_axis_retry_used_m),
                ("axis_total_progress", self._drawer_rail_stop_axis_total_progress_m),
                (
                    "normal_retreat_progress",
                    self._drawer_rail_stop_normal_retreat_progress_m,
                ),
            ),
            bool_values=(
                ("axis_chain_active", self._drawer_rail_stop_axis_chain_active),
                ("normal_chain_active", self._drawer_rail_stop_normal_chain_active),
            ),
        )
        if typed_failure is not None:
            return typed_failure
        finite_failure = self._drawer_rail_stop_nonfinite_failure(
            "axis-retry schedule",
            public_pose=observation.proprio.T_world_ee,
            requested_distance=requested_distance_m,
            retry_used=self._drawer_rail_stop_axis_retry_used_m,
            axis_total_progress=self._drawer_rail_stop_axis_total_progress_m,
            slide_direction=self._drawer_rail_stop_slide_direction,
        )
        if finite_failure is not None:
            return finite_failure
        retry_room = float(
            max(
                0.0,
                self.config.drawer_rail_stop_axis_retry_max_m
                - self._drawer_rail_stop_axis_retry_used_m,
            )
        )
        total_room = float(
            max(
                0.0,
                self.config.drawer_rail_stop_axis_total_max_m
                - self._drawer_rail_stop_axis_total_progress_m,
            )
        )
        available = float(min(retry_room, total_room))
        finite_failure = self._drawer_rail_stop_nonfinite_failure(
            "axis-retry budget",
            retry_room=retry_room,
            total_room=total_room,
            available=available,
        )
        if finite_failure is not None:
            return finite_failure
        axis_tolerance = self.config.drawer_rail_stop_axis_progress_tolerance_m
        if (
            require_full_distance
            and requested_distance_m > available + 1e-12
        ):
            return self._fail(failure_message)
        increment = float(min(requested_distance_m, available))
        finite_failure = self._drawer_rail_stop_nonfinite_failure(
            "axis-retry increment",
            increment=increment,
        )
        if finite_failure is not None:
            return finite_failure
        if increment <= axis_tolerance:
            return self._fail(failure_message)
        transaction_before = self._snapshot_drawer_rail_stop_axis_retry_state()
        self._drawer_rail_stop_axis_retry_used_m = float(
            self._drawer_rail_stop_axis_retry_used_m + increment
        )
        self._drawer_rail_stop_slide_remaining_m = float(increment)
        # ``require_full_distance`` is used only by endpoint-debt repair.  Keep
        # it live for every public micro-segment so the ordinary 4-mm endpoint
        # tolerance cannot silently erase the final portion of that debt.  The
        # existing 0.35-mm progress epsilon is the stricter completion gate.
        self._drawer_rail_stop_axis_retry_require_full_distance = bool(
            require_full_distance
        )
        decision = self._command_next_drawer_rail_stop_axis_segment(
            observation,
            message,
        )
        if decision.request_stop:
            # Reservation is not evidence.  A new decomposition/basis/command
            # gate can still fail inside the command builder, in which case the
            # retry ledger must remain byte-for-byte unchanged.
            failure_message_from_decision = str(
                decision.diagnostics.get("message", failure_message)
            )
            self._restore_drawer_rail_stop_axis_retry_state(transaction_before)
            # The first failure decision was constructed before rollback.  Build
            # the terminal diagnostic once more so it cannot expose transient
            # pending credit, retry reservation, or a newly frozen chain.
            return self._fail(failure_message_from_decision)
        return decision

    def _drawer_rail_stop_net_axis_clearance(
        self,
        position: np.ndarray,
    ) -> float | None:
        """Return signed tool-body clearance beyond the frozen RGB-D endpoint."""

        if (
            self._target is None
            or not np.all(np.isfinite(position))
            or not np.all(
                np.isfinite(self._drawer_rail_stop_slide_direction)
            )
            or not np.all(np.isfinite(self._drawer_rail_stop_axis_world))
            or not np.all(np.isfinite(self._target.point_world))
            or not np.isfinite(
                self._drawer_rail_stop_required_axis_scalar_m
            )
        ):
            return None
        selected = np.asarray(
            self._drawer_rail_stop_slide_direction,
            dtype=np.float64,
        ).copy()
        selected_norm = float(np.linalg.norm(selected))
        if selected_norm < 1e-8:
            return None
        selected /= selected_norm
        selected_sign = float(
            np.dot(selected, self._drawer_rail_stop_axis_world)
        )
        if abs(abs(selected_sign) - 1.0) > 1e-6:
            return None
        current_scalar = float(
            np.dot(
                position - self._target.point_world,
                self._drawer_rail_stop_axis_world,
            )
        )
        return selected_sign * (
            current_scalar
            - self._drawer_rail_stop_required_axis_scalar_m
        )

    def _credit_drawer_rail_stop_normal_progress(
        self,
        signed_progress_m: float,
        segment_distance_m: float,
    ) -> None:
        """Credit only measured, accepted normal travel, never its command."""

        remaining = max(
            0.0,
            self.config.drawer_rail_stop_normal_retreat_m
            - self._drawer_rail_stop_normal_retreat_progress_m,
        )
        requested_credit = min(
            max(0.0, signed_progress_m),
            max(0.0, segment_distance_m),
            remaining,
        )
        available_net_progress = max(
            0.0,
            self._drawer_rail_stop_normal_chain_progress_m
            - self._drawer_rail_stop_normal_retreat_progress_m,
        )
        credit = float(min(requested_credit, available_net_progress))
        self._drawer_rail_stop_normal_retreat_progress_m = float(
            self._drawer_rail_stop_normal_retreat_progress_m + credit
        )
        if credit > 0.0:
            self._drawer_rail_stop_normal_credit_source = (
                "normal_retreat_public_pose"
            )
            self._drawer_rail_stop_normal_credit_increment_m = float(credit)

    def _start_drawer_rail_stop_normal_segment(
        self,
        observation: RobotObservation,
    ) -> PolicyDecision:
        """Command one open-gripper, public-pose normal retreat segment."""

        finite_failure = self._drawer_rail_stop_nonfinite_failure(
            "normal-segment start",
            public_pose=observation.proprio.T_world_ee,
            pull_direction=self._drawer_pull_direction,
            slide_direction=self._drawer_rail_stop_slide_direction,
            motion_rotation=self._motion_rotation,
            normal_credit=self._drawer_rail_stop_normal_retreat_progress_m,
            slide_remaining=self._drawer_rail_stop_slide_remaining_m,
            axis_total_progress=self._drawer_rail_stop_axis_total_progress_m,
            retry_used=self._drawer_rail_stop_axis_retry_used_m,
        )
        if finite_failure is not None:
            return finite_failure
        width_failure = self._extend_drawer_rail_stop_for_public_width(
            observation
        )
        if width_failure is not None:
            return width_failure
        if (
            self._drawer_rail_stop_slide_remaining_m
            > self.config.drawer_rail_stop_axis_progress_tolerance_m
        ):
            return self._command_next_drawer_rail_stop_axis_segment(
                observation,
                "public released width enlarged the bounded handle-axis envelope",
            )
        current = observation.proprio.ee_position_world
        direction = np.asarray(
            self._drawer_pull_direction,
            dtype=np.float64,
        ).copy()
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm < 1e-8:
            return self._fail(
                "drawer rail-stop normal retreat direction was degenerate"
            )
        direction /= direction_norm
        cumulative_unsafe = self._drawer_rail_stop_normal_chain_safety_failure(
            observation
        )
        if cumulative_unsafe is not None:
            return cumulative_unsafe
        net_clearance = self._drawer_rail_stop_net_axis_clearance(current)
        if net_clearance is None:
            return self._fail(
                "drawer rail-stop retreat lacked a frozen RGB-D endpoint bound"
            )
        self._drawer_rail_stop_outward_net_clearance_m = float(net_clearance)
        if net_clearance < 0.0:
            correction = float(
                -net_clearance
                + self.config.drawer_rail_stop_axis_progress_tolerance_m
            )
            return self._schedule_drawer_rail_stop_axis_retry(
                observation,
                correction,
                require_full_distance=True,
                message=(
                    "normal retreat lost endpoint clearance; correcting only "
                    "the frozen selected handle axis"
                ),
                failure_message=(
                    "drawer rail-stop normal retreat lost its frozen RGB-D "
                    "endpoint clearance beyond the remaining axis budget"
                ),
            )
        remaining = max(
            0.0,
            self.config.drawer_rail_stop_normal_retreat_m
            - self._drawer_rail_stop_normal_retreat_progress_m,
        )
        if remaining <= 1e-12:
            self._motion_position = current.copy()
            self._motion_position[2] += self.config.drawer_safe_height_m
            self._motion_rotation = observation.proprio.T_world_ee[
                :3, :3
            ].copy()
            self._set_phase("retreat_up")
            return self._move(
                observation,
                -1.0,
                "full measured rail-stop normal retreat complete; lifting clear",
            )
        segment_distance = float(
            min(
                self.config.drawer_rail_stop_outward_probe_m,
                remaining,
            )
        )
        self._drawer_rail_stop_normal_segment_origin = current.copy()
        self._drawer_rail_stop_normal_segment_distance_m = float(segment_distance)
        self._drawer_rail_stop_normal_segment_progress_m = 0.0
        self._drawer_rail_stop_normal_segment_cross_drift_m = 0.0
        self._motion_position = current + direction * segment_distance
        self._motion_rotation = (
            self._drawer_rail_stop_normal_chain_rotation.copy()
        )
        finite_failure = self._drawer_rail_stop_nonfinite_failure(
            "normal-segment target",
            motion_position=self._motion_position,
            motion_rotation=self._motion_rotation,
            segment_distance=segment_distance,
        )
        if finite_failure is not None:
            return finite_failure
        self._set_phase("retreat_outward")
        return self._move(
            observation,
            -1.0,
            "continuing the measured segmented rail-stop normal retreat",
        )

    def _act_drawer_rail_stop_normal_segment(
        self,
        observation: RobotObservation,
    ) -> PolicyDecision:
        """Validate and accumulate one bounded normal retreat segment."""

        finite_failure = self._drawer_rail_stop_nonfinite_failure(
            "normal segment",
            public_pose=observation.proprio.T_world_ee,
            pull_direction=self._drawer_pull_direction,
            slide_direction=self._drawer_rail_stop_slide_direction,
            motion_position=self._motion_position,
            motion_rotation=self._motion_rotation,
            segment_origin=self._drawer_rail_stop_normal_segment_origin,
            segment_distance=self._drawer_rail_stop_normal_segment_distance_m,
            normal_credit=self._drawer_rail_stop_normal_retreat_progress_m,
            axis_total_progress=self._drawer_rail_stop_axis_total_progress_m,
            retry_used=self._drawer_rail_stop_axis_retry_used_m,
        )
        if finite_failure is not None:
            return finite_failure
        width_failure = self._extend_drawer_rail_stop_for_public_width(
            observation,
            schedule_axis_clearance=False,
        )
        if width_failure is not None:
            return width_failure
        direction = np.asarray(
            self._drawer_pull_direction,
            dtype=np.float64,
        ).copy()
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm < 1e-8:
            return self._fail(
                "drawer rail-stop normal retreat direction was degenerate"
            )
        direction /= direction_norm
        current = observation.proprio.ee_position_world
        displacement = (
            current - self._drawer_rail_stop_normal_segment_origin
        )
        signed_progress = float(np.dot(displacement, direction))
        cross_drift = float(
            np.linalg.norm(displacement - direction * signed_progress)
        )
        rotation_error = self._rotation_error(
            observation.proprio.T_world_ee[:3, :3],
            self._motion_rotation,
        )
        cumulative_unsafe = self._drawer_rail_stop_normal_chain_safety_failure(
            observation
        )
        if cumulative_unsafe is not None:
            return cumulative_unsafe
        self._drawer_rail_stop_normal_segment_progress_m = max(
            0.0,
            signed_progress,
        )
        self._drawer_rail_stop_normal_segment_cross_drift_m = cross_drift
        tolerance = self.config.drawer_rail_stop_outward_progress_tolerance_m
        if signed_progress < -tolerance:
            return self._fail(
                "drawer rail-stop normal retreat segment reversed its public "
                f"EE progress ({signed_progress:.4f} m)"
            )
        if (
            signed_progress
            > self._drawer_rail_stop_normal_segment_distance_m + 1e-12
        ):
            return self._fail(
                "drawer rail-stop normal retreat segment exceeded its typed "
                f"displacement cap ({signed_progress:.4f} m)"
            )
        if (
            cross_drift
            > self.config.drawer_rail_stop_outward_max_cross_drift_m
        ):
            return self._fail(
                "drawer rail-stop normal retreat segment exceeded its public "
                f"cross-axis drift gate ({cross_drift:.4f} m)"
            )
        if rotation_error > self.config.drawer_rail_stop_outward_max_rotation_rad:
            return self._fail(
                "drawer rail-stop normal retreat segment lost its frozen "
                f"public wrist pose ({rotation_error:.4f} rad)"
            )
        net_clearance = self._drawer_rail_stop_net_axis_clearance(current)
        if net_clearance is None:
            return self._fail(
                "drawer rail-stop retreat lacked a frozen RGB-D endpoint bound"
            )
        self._drawer_rail_stop_outward_net_clearance_m = float(net_clearance)

        def account_beneficial_axis_travel() -> PolicyDecision | None:
            selected = self._drawer_rail_stop_slide_direction
            selected_axis_travel = float(
                max(
                    0.0,
                    float(np.dot(displacement, selected)),
                )
            )
            if (
                self._drawer_rail_stop_axis_total_progress_m
                + selected_axis_travel
                > self.config.drawer_rail_stop_axis_total_max_m + 1e-12
            ):
                return self._fail(
                    "drawer rail-stop normal retreat exceeded its typed total "
                    "axis bound"
                )
            self._drawer_rail_stop_axis_total_progress_m = float(
                self._drawer_rail_stop_axis_total_progress_m
                + selected_axis_travel
            )
            return None

        if net_clearance < 0.0:
            transaction_before = self._snapshot_drawer_rail_stop_axis_retry_state()
            unsafe_axis = account_beneficial_axis_travel()
            if unsafe_axis is not None:
                return self._rollback_failed_drawer_rail_stop_downstream(
                    transaction_before,
                    unsafe_axis,
                    "drawer rail-stop normal segment axis accounting failed",
                )
            self._credit_drawer_rail_stop_normal_progress(
                signed_progress,
                self._drawer_rail_stop_normal_segment_distance_m,
            )
            correction = float(
                -net_clearance
                + self.config.drawer_rail_stop_axis_progress_tolerance_m
            )
            decision = self._schedule_drawer_rail_stop_axis_retry(
                observation,
                correction,
                require_full_distance=True,
                message=(
                    "normal segment lost endpoint clearance; correcting only "
                    "the frozen selected handle axis"
                ),
                failure_message=(
                    "drawer rail-stop normal segment lost its frozen RGB-D "
                    "endpoint clearance beyond the remaining axis budget"
                ),
            )
            return self._rollback_failed_drawer_rail_stop_downstream(
                transaction_before,
                decision,
                "drawer rail-stop normal segment endpoint correction failed",
            )

        required_progress = max(
            self._drawer_rail_stop_normal_segment_distance_m - tolerance,
            min(
                self._drawer_rail_stop_normal_segment_distance_m,
                1e-6,
            ),
        )
        if signed_progress >= required_progress:
            transaction_before = self._snapshot_drawer_rail_stop_axis_retry_state()
            unsafe_axis = account_beneficial_axis_travel()
            if unsafe_axis is not None:
                return self._rollback_failed_drawer_rail_stop_downstream(
                    transaction_before,
                    unsafe_axis,
                    "drawer rail-stop completed normal segment accounting failed",
                )
            self._credit_drawer_rail_stop_normal_progress(
                signed_progress,
                self._drawer_rail_stop_normal_segment_distance_m,
            )
            decision = self._start_drawer_rail_stop_normal_segment(observation)
            return self._rollback_failed_drawer_rail_stop_downstream(
                transaction_before,
                decision,
                "drawer rail-stop next normal segment failed",
            )
        if (
            self._phase_ticks
            >= self.config.drawer_rail_stop_outward_probe_max_ticks
        ):
            transaction_before = self._snapshot_drawer_rail_stop_axis_retry_state()
            unsafe_axis = account_beneficial_axis_travel()
            if unsafe_axis is not None:
                return self._rollback_failed_drawer_rail_stop_downstream(
                    transaction_before,
                    unsafe_axis,
                    "drawer rail-stop blocked normal segment accounting failed",
                )
            self._credit_drawer_rail_stop_normal_progress(
                signed_progress,
                self._drawer_rail_stop_normal_segment_distance_m,
            )
            decision = self._schedule_drawer_rail_stop_axis_retry(
                observation,
                float(self.config.drawer_rail_stop_axis_retry_increment_m),
                require_full_distance=False,
                message=(
                    "normal retreat segment blocked; extending only the "
                    "selected public handle axis"
                ),
                failure_message=(
                    "drawer rail-stop normal retreat remained blocked after "
                    "the bounded handle-axis clearance "
                    f"(progress={signed_progress:.4f} m; "
                    f"cross={cross_drift:.4f} m; "
                    f"rotation={rotation_error:.4f} rad)"
                ),
            )
            return self._rollback_failed_drawer_rail_stop_downstream(
                transaction_before,
                decision,
                "drawer rail-stop blocked normal segment retry failed",
            )
        return self._move(
            observation,
            -1.0,
            "requiring signed public EE progress in the normal retreat segment",
        )

    def _act_drawer_rail_stop_outward_probe(
        self,
        observation: RobotObservation,
    ) -> PolicyDecision:
        """Accept a rail-stop retreat only after bounded public EE progress."""

        finite_failure = self._drawer_rail_stop_nonfinite_failure(
            "outward probe",
            public_pose=observation.proprio.T_world_ee,
            pull_direction=self._drawer_pull_direction,
            slide_direction=self._drawer_rail_stop_slide_direction,
            motion_position=self._motion_position,
            motion_rotation=self._motion_rotation,
            probe_origin=self._drawer_rail_stop_outward_probe_origin,
            probe_distance=self._drawer_rail_stop_outward_probe_distance_m,
            slide_remaining=self._drawer_rail_stop_slide_remaining_m,
            normal_credit=self._drawer_rail_stop_normal_retreat_progress_m,
            axis_total_progress=self._drawer_rail_stop_axis_total_progress_m,
            retry_used=self._drawer_rail_stop_axis_retry_used_m,
            required_axis_scalar=self._drawer_rail_stop_required_axis_scalar_m,
        )
        if finite_failure is not None:
            return finite_failure
        width_failure = self._extend_drawer_rail_stop_for_public_width(
            observation,
            schedule_axis_clearance=False,
        )
        if width_failure is not None:
            return width_failure
        cumulative_unsafe = self._drawer_rail_stop_normal_chain_safety_failure(
            observation
        )
        if cumulative_unsafe is not None:
            return cumulative_unsafe
        if (
            self._drawer_rail_stop_slide_remaining_m
            > self.config.drawer_rail_stop_axis_progress_tolerance_m
        ):
            return self._command_next_drawer_rail_stop_axis_segment(
                observation,
                "wider public release required more same-axis body clearance",
            )
        direction = np.asarray(self._drawer_pull_direction, dtype=np.float64).copy()
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        selected_axis = np.asarray(
            self._drawer_rail_stop_slide_direction,
            dtype=np.float64,
        ).copy()
        selected_axis /= max(float(np.linalg.norm(selected_axis)), 1e-12)
        current = observation.proprio.ee_position_world
        displacement = current - self._drawer_rail_stop_outward_probe_origin
        signed_progress = float(np.dot(displacement, direction))
        self._drawer_rail_stop_outward_probe_progress_m = float(
            max(
                0.0,
                signed_progress,
            )
        )
        selected_axis_drift = float(np.dot(displacement, selected_axis))
        residual_drift = float(
            np.linalg.norm(
                displacement
                - direction * signed_progress
                - selected_axis * selected_axis_drift
            )
        )
        self._drawer_rail_stop_outward_axis_drift_m = float(selected_axis_drift)
        self._drawer_rail_stop_outward_residual_drift_m = float(residual_drift)
        rotation_error = self._rotation_error(
            observation.proprio.T_world_ee[:3, :3],
            self._motion_rotation,
        )
        tolerance = self.config.drawer_rail_stop_outward_progress_tolerance_m
        assert self._target is not None
        if not np.isfinite(self._drawer_rail_stop_required_axis_scalar_m):
            return self._fail(
                "drawer rail-stop outward probe lacked a frozen RGB-D endpoint bound"
            )
        current_axis_scalar = float(
            np.dot(
                current - self._target.point_world,
                self._drawer_rail_stop_axis_world,
            )
        )
        selected_sign = float(
            np.dot(selected_axis, self._drawer_rail_stop_axis_world)
        )
        net_clearance = float(
            selected_sign
            * (
                current_axis_scalar
                - self._drawer_rail_stop_required_axis_scalar_m
            )
        )
        self._drawer_rail_stop_outward_net_clearance_m = float(net_clearance)
        if signed_progress < -tolerance:
            return self._fail(
                "drawer rail-stop outward probe reversed its public EE progress "
                f"({signed_progress:.4f} m)"
            )
        if (
            signed_progress
            > self._drawer_rail_stop_outward_probe_distance_m + 1e-12
        ):
            return self._fail(
                "drawer rail-stop outward probe exceeded its typed displacement cap "
                f"({signed_progress:.4f} m)"
            )
        if (
            residual_drift
            > self.config.drawer_rail_stop_outward_max_cross_drift_m
        ):
            return self._fail(
                "drawer rail-stop outward probe exceeded its vertical/unknown "
                f"orthogonal drift gate ({residual_drift:.4f} m)"
            )
        if rotation_error > self.config.drawer_rail_stop_outward_max_rotation_rad:
            return self._fail(
                "drawer rail-stop outward probe lost its frozen wrist pose "
                f"({rotation_error:.4f} rad)"
            )
        if (
            signed_progress
            >= self._drawer_rail_stop_outward_probe_distance_m - tolerance
        ):
            beneficial_axis_travel = float(max(0.0, selected_axis_drift))
            if (
                self._drawer_rail_stop_axis_total_progress_m
                + beneficial_axis_travel
                > self.config.drawer_rail_stop_axis_total_max_m + 1e-12
            ):
                return self._fail(
                    "drawer rail-stop outward probe exceeded its typed total axis bound"
                )
            transaction_before = self._snapshot_drawer_rail_stop_axis_retry_state()
            self._drawer_rail_stop_axis_total_progress_m = float(
                self._drawer_rail_stop_axis_total_progress_m
                + beneficial_axis_travel
            )
            self._credit_drawer_rail_stop_normal_progress(
                signed_progress,
                self._drawer_rail_stop_outward_probe_distance_m,
            )
            if net_clearance < 0.0:
                correction = float(
                    -net_clearance
                    + self.config.drawer_rail_stop_axis_progress_tolerance_m
                )
                decision = self._schedule_drawer_rail_stop_axis_retry(
                    observation,
                    correction,
                    require_full_distance=True,
                    message=(
                        "outward probe lost endpoint clearance; correcting only "
                        "the frozen selected handle axis"
                    ),
                    failure_message=(
                        "drawer rail-stop outward probe lost its frozen RGB-D "
                        "endpoint clearance beyond the remaining axis budget"
                    ),
                )
                return self._rollback_failed_drawer_rail_stop_downstream(
                    transaction_before,
                    decision,
                    "drawer rail-stop outward endpoint correction failed",
                )
            decision = self._start_drawer_rail_stop_normal_segment(observation)
            return self._rollback_failed_drawer_rail_stop_downstream(
                transaction_before,
                decision,
                "drawer rail-stop normal retreat start failed",
            )
        if self._phase_ticks >= self.config.drawer_rail_stop_outward_probe_max_ticks:
            beneficial_axis_travel = float(max(0.0, selected_axis_drift))
            if (
                self._drawer_rail_stop_axis_total_progress_m
                + beneficial_axis_travel
                > self.config.drawer_rail_stop_axis_total_max_m + 1e-12
            ):
                return self._fail(
                    "drawer rail-stop outward probe exceeded its typed total axis bound"
                )
            transaction_before = self._snapshot_drawer_rail_stop_axis_retry_state()
            self._drawer_rail_stop_axis_total_progress_m = float(
                self._drawer_rail_stop_axis_total_progress_m
                + beneficial_axis_travel
            )
            accepted_probe_normal = float(
                min(
                    max(0.0, signed_progress),
                    self._drawer_rail_stop_outward_probe_distance_m,
                )
            )
            normal_credit_before = (
                self._drawer_rail_stop_normal_retreat_progress_m
            )
            self._credit_drawer_rail_stop_normal_progress(
                signed_progress,
                self._drawer_rail_stop_outward_probe_distance_m,
            )
            credited_probe_normal = float(
                max(
                    0.0,
                    self._drawer_rail_stop_normal_retreat_progress_m
                    - normal_credit_before,
                )
            )
            uncredited_probe_normal = float(
                max(
                    0.0,
                    accepted_probe_normal - credited_probe_normal,
                )
            )
            finite_failure = self._drawer_rail_stop_nonfinite_failure(
                "outward probe settle attribution",
                accepted_probe_normal=accepted_probe_normal,
                normal_credit_before=normal_credit_before,
                credited_probe_normal=credited_probe_normal,
                uncredited_probe_normal=uncredited_probe_normal,
            )
            if finite_failure is not None:
                return self._rollback_failed_drawer_rail_stop_downstream(
                    transaction_before,
                    finite_failure,
                    "drawer rail-stop outward settle attribution failed",
                )
            if credited_probe_normal > accepted_probe_normal + 1e-12:
                failure = self._fail(
                    "drawer rail-stop outward probe credit exceeded its current "
                    "signed public progress"
                )
                return self._rollback_failed_drawer_rail_stop_downstream(
                    transaction_before,
                    failure,
                    "drawer rail-stop outward settle attribution failed",
                )
            decision = self._start_drawer_rail_stop_axis_retry_settle(
                observation,
                float(self.config.drawer_rail_stop_axis_retry_increment_m),
                uncredited_probe_normal_m=uncredited_probe_normal,
                probe_axis_drift_m=selected_axis_drift,
                require_full_distance=False,
                message=(
                    "outward probe blocked; extending only the selected public "
                    "handle axis"
                ),
                failure_message=(
                    "drawer rail-stop outward probe remained blocked after the "
                    "bounded handle-axis clearance "
                    f"(progress={signed_progress:.4f} m; "
                    f"axis={selected_axis_drift:.4f} m; "
                    f"residual={residual_drift:.4f} m; "
                    f"rotation={rotation_error:.4f} rad)"
                ),
            )
            return self._rollback_failed_drawer_rail_stop_downstream(
                transaction_before,
                decision,
                "drawer rail-stop outward retry settle start failed",
            )
        return self._move(
            observation,
            -1.0,
            "requiring signed public EE progress in the drawer-normal micro-probe",
        )

    def _move(
        self,
        observation: RobotObservation,
        gripper: float,
        message: str = "",
    ) -> PolicyDecision:
        current = observation.proprio.T_world_ee
        if not (
            np.all(np.isfinite(current))
            and np.all(np.isfinite(self._motion_position))
            and np.all(np.isfinite(self._motion_rotation))
            and np.isfinite(self.config.translation_scale_m)
            and np.isfinite(self.config.rotation_scale_rad)
            and np.isfinite(gripper)
        ):
            return self._fail(
                "controller motion state was non-finite; refusing to issue an action"
            )
        translation = (self._motion_position - current[:3, 3]) / self.config.translation_scale_m
        relative = self._motion_rotation @ current[:3, :3].T
        rotation = Rotation.from_matrix(relative).as_rotvec() / self.config.rotation_scale_rad
        return self._tick(
            OSCAction.from_array(
                np.concatenate((translation, rotation, (gripper,))),
                clip=True,
            ),
            message,
        )

    def _position_reached(
        self,
        observation: RobotObservation,
        target: np.ndarray,
        *,
        tolerance_m: float | None = None,
    ) -> bool:
        tolerance = self.config.position_tolerance_m if tolerance_m is None else tolerance_m
        return bool(
            np.linalg.norm(target - observation.proprio.ee_position_world)
            <= tolerance
        )

    @staticmethod
    def _rotation_error(current: np.ndarray, target: np.ndarray) -> float:
        return float(Rotation.from_matrix(target @ current.T).magnitude())

    def _contact_reached(
        self,
        observation: RobotObservation,
        *,
        tolerance_m: float | None = None,
    ) -> bool:
        error = float(np.linalg.norm(self._motion_position - observation.proprio.ee_position_world))
        if self._previous_error is None or self._previous_error - error > self.config.progress_epsilon_m:
            self._stall_ticks = 0
        else:
            self._stall_ticks += 1
        self._previous_error = error
        force_delta = float(
            np.linalg.norm(observation.proprio.ee_force_sensor - self._force_baseline)
        )
        tolerance = self.config.contact_position_tolerance_m if tolerance_m is None else tolerance_m
        return bool(
            self._phase_ticks >= self.config.contact_min_ticks
            and error <= tolerance
            and (
                self._stall_ticks >= self.config.contact_stall_ticks
                or force_delta >= self.config.contact_force_delta_n
            )
        )

    def _set_phase(self, phase: str) -> None:
        self._phase = phase
        self._phase_ticks = 0
        self._previous_error = None
        self._stall_ticks = 0

    def _tick(self, action: OSCAction, message: str = "") -> PolicyDecision:
        decision = self._decision(action, message)
        self._phase_ticks += 1
        return decision

    def _detection_miss(self, message: str, *, gripper: float) -> PolicyDecision:
        self._detection_misses += 1
        if self._detection_misses >= self.config.max_detection_misses:
            return self._fail(message)
        return self._tick(OSCAction.hold(gripper), message)

    def _complete(self, message: str) -> PolicyDecision:
        assert self._plan is not None
        self._step_index += 1
        if self._step_index < len(self._plan.steps):
            self._manipulation_progress_m = 0.0
            self._drawer_attempt = 0
            self._drawer_load_proof_progress_m = 0.0
            self._drawer_load_proof_loaded_width_m = None
            self._drawer_load_proof_settled_width_m = None
            self._drawer_load_proof_passed = False
            self._drawer_load_proof_failure = ""
            self._drawer_pull_grasp_lost = False
            self._drawer_pull_previous_width_m = None
            self._drawer_pull_width_plateau_ticks = 0
            self._drawer_pull_force_ticks = 0
            self._drawer_pull_mechanical_stop_observed = False
            self._drawer_rail_stop_visible_axis_bounds_m = np.full(2, np.nan)
            self._reset_drawer_rail_stop_motion_state()
            self._drawer_wrist_recovery_used = False
            self._drawer_free_space_best_error_m = float("inf")
            self._drawer_free_space_stall_ticks = 0
            self._push_attempt = 0
            self._knob_rotation_progress_rad = 0.0
            self._microwave_open_push_recompact_ticks = 0
            self._microwave_open_push_compact_recoveries = 0
            self._microwave_open_push_recompact_pending = False
            self._microwave_open_push_last_compact_angle_rad = 0.0
            self._microwave_open_push_compact_loss_baseline_angle_rad = 0.0
            self._microwave_open_push_progress_samples = []
            next_step = self._plan.steps[self._step_index]
            if next_step.kind is GoalSkillKind.ROUTE_B_PLACE_IN:
                self._status = GoalExecutorStatus.HANDOFF
                self._phase = "route_b_handoff"
                self._message = (
                    f"{message}; handoff to Route B: put {next_step.subject} in {next_step.target}"
                )
                return self._decision(OSCAction.hold(-1.0), self._message, stop=True)
            self._set_phase("detect")
            return self._decision(OSCAction.hold(-1.0), message)
        self._status = GoalExecutorStatus.SUCCEEDED
        self._phase = "done"
        self._message = message
        return self._decision(OSCAction.hold(-1.0), message, stop=True)

    def _fail(self, message: str) -> PolicyDecision:
        # A failed observation may never leave a staged public-pose credit that
        # a later internal call could accidentally commit.
        self._drawer_rail_stop_axis_chain_normal_credit_evidence = None
        self._status = GoalExecutorStatus.FAILED
        self._phase = "failed"
        self._message = message
        return self._decision(OSCAction.hold(-1.0), message, stop=True)

    def _decision(self, action: OSCAction, message: str = "", *, stop: bool = False) -> PolicyDecision:
        diagnostics = {
            "route": "b_goal_contact",
            "status": self._status.value,
            "skill_index": self._step_index,
            "phase": self._phase,
            "message": message,
        }
        if self._target is not None:
            diagnostics["target_confidence"] = self._target.confidence
            diagnostics["source_cameras"] = self._target.source_cameras
        elif self._push is not None:
            diagnostics["target_confidence"] = self._push.confidence
            diagnostics["source_cameras"] = self._push.source_cameras
        if (
            self._plan is not None
            and self._step_index < len(self._plan.steps)
            and self._plan.steps[self._step_index].kind
            is GoalSkillKind.OPEN_DRAWER
        ):
            # Every value is derived from public RGB-D/proprioception and is
            # retained in the final episode row to make failed contact seats
            # distinguishable from late rail stops.
            diagnostics["drawer_contact_proof"] = {
                "attempt": self._drawer_attempt,
                "progress_m": self._drawer_load_proof_progress_m,
                "loaded_width_m": self._drawer_load_proof_loaded_width_m,
                "settled_width_m": self._drawer_load_proof_settled_width_m,
                "passed": self._drawer_load_proof_passed,
                "failure": self._drawer_load_proof_failure or None,
                "pull_grasp_lost": self._drawer_pull_grasp_lost,
                "rail_stop_observed": (
                    self._drawer_pull_mechanical_stop_observed
                ),
                "rail_stop_visible_axis_bounds_m": (
                    self._drawer_rail_stop_visible_axis_bounds_m.tolist()
                    if np.all(
                        np.isfinite(
                            self._drawer_rail_stop_visible_axis_bounds_m
                        )
                    )
                    else None
                ),
                "rail_stop_axis_world": (
                    self._drawer_rail_stop_axis_world.tolist()
                ),
                "rail_stop_axis_progress_m": (
                    self._drawer_rail_stop_axis_total_progress_m
                ),
                "rail_stop_axis_slide_remaining_m": (
                    self._drawer_rail_stop_slide_remaining_m
                ),
                "rail_stop_axis_segment_distance_m": (
                    self._drawer_rail_stop_slide_segment_distance_m
                ),
                "rail_stop_axis_segment_progress_m": (
                    self._drawer_rail_stop_axis_segment_progress_m
                ),
                "rail_stop_axis_segment_peak_progress_m": (
                    self._drawer_rail_stop_axis_segment_peak_progress_m
                ),
                "rail_stop_axis_segment_max_regression_m": (
                    self._drawer_rail_stop_axis_segment_max_regression_m
                ),
                "rail_stop_axis_max_observed_regression_m": (
                    self._drawer_rail_stop_axis_max_observed_regression_m
                ),
                "rail_stop_axis_partial_progress_m": (
                    self._drawer_rail_stop_axis_partial_progress_m
                ),
                "rail_stop_axis_partial_segment_count": (
                    self._drawer_rail_stop_axis_partial_segment_count
                ),
                "rail_stop_axis_chain_progress_m": (
                    self._drawer_rail_stop_axis_chain_progress_m
                ),
                "rail_stop_axis_chain_cross_drift_m": (
                    self._drawer_rail_stop_axis_chain_cross_drift_m
                ),
                "rail_stop_axis_chain_rotation_error_rad": (
                    self._drawer_rail_stop_axis_chain_rotation_error_rad
                ),
                "rail_stop_axis_chain_local_normal_m": (
                    self._drawer_rail_stop_axis_chain_local_normal_m
                ),
                "rail_stop_axis_chain_third_axis_drift_m": (
                    self._drawer_rail_stop_axis_chain_third_axis_drift_m
                ),
                "rail_stop_axis_chain_global_normal_progress_m": (
                    self._drawer_rail_stop_axis_chain_global_normal_progress_m
                ),
                "rail_stop_axis_chain_pending_baseline_normal_credit_m": (
                    self._drawer_rail_stop_axis_chain_pending_baseline_normal_credit_m
                ),
                "rail_stop_axis_chain_baseline_normal_segment_cap_m": (
                    self._drawer_rail_stop_axis_chain_baseline_normal_segment_cap_m
                ),
                "rail_stop_normal_credit_source": (
                    self._drawer_rail_stop_normal_credit_source
                ),
                "rail_stop_normal_credit_increment_m": (
                    self._drawer_rail_stop_normal_credit_increment_m
                ),
                "rail_stop_axis_probe_index": self._drawer_rail_stop_probe_index,
                "rail_stop_release_width_m": (
                    self._drawer_rail_stop_release_width_m
                ),
                "rail_stop_current_public_width_m": (
                    self._drawer_rail_stop_current_public_width_m
                ),
                "rail_stop_body_margin_m": (
                    self._drawer_rail_stop_body_margin_m
                ),
                "rail_stop_axis_retry_used_m": (
                    self._drawer_rail_stop_axis_retry_used_m
                ),
                "rail_stop_axis_retry_require_full_distance": (
                    self._drawer_rail_stop_axis_retry_require_full_distance
                ),
                "rail_stop_outward_probe_progress_m": (
                    self._drawer_rail_stop_outward_probe_progress_m
                ),
                "rail_stop_required_axis_scalar_m": (
                    self._drawer_rail_stop_required_axis_scalar_m
                    if np.isfinite(
                        self._drawer_rail_stop_required_axis_scalar_m
                    )
                    else None
                ),
                "rail_stop_outward_axis_drift_m": (
                    self._drawer_rail_stop_outward_axis_drift_m
                ),
                "rail_stop_outward_residual_drift_m": (
                    self._drawer_rail_stop_outward_residual_drift_m
                ),
                "rail_stop_outward_net_clearance_m": (
                    self._drawer_rail_stop_outward_net_clearance_m
                    if np.isfinite(
                        self._drawer_rail_stop_outward_net_clearance_m
                    )
                    else None
                ),
                "rail_stop_axis_retry_settle_phase_ticks": (
                    self._phase_ticks
                    if self._phase == "drawer_rail_stop_axis_retry_settle"
                    else 0
                ),
                "rail_stop_axis_retry_settle_anchor_frozen": (
                    self._drawer_rail_stop_axis_retry_settle_anchor_frozen
                ),
                "rail_stop_axis_retry_settle_start_global_normal_m": (
                    self._drawer_rail_stop_axis_retry_settle_start_global_normal_m
                ),
                "rail_stop_axis_retry_settle_delta_normal_m": (
                    self._drawer_rail_stop_axis_retry_settle_delta_normal_m
                ),
                "rail_stop_axis_retry_settle_local_normal_m": (
                    self._drawer_rail_stop_axis_retry_settle_local_normal_m
                ),
                "rail_stop_axis_retry_settle_transition_normal_m": (
                    self._drawer_rail_stop_axis_retry_settle_transition_normal_m
                ),
                "rail_stop_axis_retry_settle_rail_drift_m": (
                    self._drawer_rail_stop_axis_retry_settle_rail_drift_m
                ),
                "rail_stop_axis_retry_settle_third_axis_drift_m": (
                    self._drawer_rail_stop_axis_retry_settle_third_axis_drift_m
                ),
                "rail_stop_axis_retry_settle_rotation_error_rad": (
                    self._drawer_rail_stop_axis_retry_settle_rotation_error_rad
                ),
                "rail_stop_axis_retry_settle_stable_count": (
                    self._drawer_rail_stop_axis_retry_settle_stable_count
                ),
                "rail_stop_axis_retry_settle_requested_distance_m": (
                    self._drawer_rail_stop_axis_retry_settle_requested_distance_m
                ),
                "rail_stop_axis_retry_settle_probe_axis_drift_m": (
                    self._drawer_rail_stop_axis_retry_settle_probe_axis_drift_m
                ),
                "rail_stop_axis_retry_settle_required_axis_correction_m": (
                    self._drawer_rail_stop_axis_retry_settle_required_axis_correction_m
                ),
                "rail_stop_axis_retry_settle_entry_net_clearance_m": (
                    self._drawer_rail_stop_axis_retry_settle_entry_net_clearance_m
                ),
                "rail_stop_axis_retry_settle_entry_baseline_credit_cap_m": (
                    self._drawer_rail_stop_axis_retry_settle_entry_baseline_credit_cap_m
                ),
                "rail_stop_normal_retreat_progress_m": (
                    self._drawer_rail_stop_normal_retreat_progress_m
                ),
                "rail_stop_normal_segment_distance_m": (
                    self._drawer_rail_stop_normal_segment_distance_m
                ),
                "rail_stop_normal_segment_progress_m": (
                    self._drawer_rail_stop_normal_segment_progress_m
                ),
                "rail_stop_normal_segment_cross_drift_m": (
                    self._drawer_rail_stop_normal_segment_cross_drift_m
                ),
                "rail_stop_normal_chain_progress_m": (
                    self._drawer_rail_stop_normal_chain_progress_m
                ),
                "rail_stop_normal_chain_cross_drift_m": (
                    self._drawer_rail_stop_normal_chain_cross_drift_m
                ),
                "rail_stop_normal_chain_rotation_error_rad": (
                    self._drawer_rail_stop_normal_chain_rotation_error_rad
                ),
            }
        return PolicyDecision(action, request_stop=stop, diagnostics=diagnostics)

    def close(self) -> None:
        return None
