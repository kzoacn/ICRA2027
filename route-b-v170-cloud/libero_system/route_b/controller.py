"""Closed-loop neuro-symbolic Pick/Place controller for LIBERO.

The controller emits normalized relative OSC_POSE commands.  Cartesian motion
is closed around measured end-effector state; task state is closed around
sensor-derived scene snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np

from ..common.grasp_journal import (
    GraspAttemptEvent,
    GraspAttemptJournal,
    GraspEvidenceCategory,
    GraspReason,
    JawBehavior,
    PendingGraspEngagement,
)
from ..common.pan_affordance import (
    PanAffordanceError,
    PanHandleAffordance,
    PanHandleSlot,
    infer_pan_handle_affordance,
)
from .models import (
    ControlDecision,
    ExecutorStatus,
    Pose,
    SceneObject,
    SceneSnapshot,
    SensorObservation,
)
from .perception import ScenePerception
from .task_compiler import (
    EXECUTABLE_GRASP_PLACE_SKILLS,
    EntityBinding,
    EntitySelector,
    SelectorRelation,
    SkillKind,
    SkillStep,
    TaskCompiler,
    TaskSpec,
)


GRIPPER_OPEN = -1.0
GRIPPER_CLOSE = 1.0


_PAN_HANDLE_LABELS = frozenset({"frying pan", "fry pan", "skillet"})


# These placement phrases denote a visible *part* of a fixture, rather than a
# direction in the world frame.  The query text comes solely from the compiled
# instruction.  Execution accepts a target only when the open-vocabulary RGB-D
# proposal is a smaller region contained by an independently detected fixture;
# there is deliberately no whole-fixture or world-axis fallback.
_SEMANTIC_PLACEMENT_REGIONS: dict[
    tuple[str, str], tuple[tuple[str, ...], tuple[str, ...]]
] = {
    ("caddy", "front_compartment"): (
        ("front compartment of the caddy",),
        ("caddy",),
    ),
    ("caddy", "back_compartment"): (
        ("back compartment of the caddy",),
        ("caddy",),
    ),
    ("caddy", "left_compartment"): (
        ("left compartment of the caddy",),
        ("caddy",),
    ),
    ("caddy", "right_compartment"): (
        ("right compartment of the caddy",),
        ("caddy",),
    ),
    ("cabinet shelf", "upper_shelf"): (
        ("upper shelf inside the cabinet", "upper compartment of the cabinet"),
        ("cabinet", "wooden cabinet"),
    ),
    ("cabinet shelf", "lower_shelf"): (
        ("lower shelf inside the cabinet", "lower compartment of the cabinet"),
        ("cabinet", "wooden cabinet"),
    ),
    ("microwave", ""): (
        ("heating compartment inside the microwave", "inside of the microwave"),
        ("microwave",),
    ),
}


class GraspMode(str, Enum):
    """How the fingers mechanically retain an object."""

    PINCH = "pinch"
    RIM_PINCH = "rim_pinch"
    EXPAND = "expand"


@dataclass(frozen=True, slots=True)
class _PendingGraspEngagement:
    source_text: str
    source_class: str
    grasp_mode: str
    jaw_behavior: JawBehavior


@dataclass(frozen=True)
class ControllerConfig:
    """Geometry and normalized-action calibration for the LIBERO OSC controller."""

    position_action_scale_m: float = 0.05
    rotation_action_scale_rad: float = 0.25
    position_tolerance_m: float = 0.008
    rotation_tolerance_rad: float = 0.12
    pregrasp_height_m: float = 0.10
    preplace_height_m: float = 0.10
    # Legacy upper bound from the original tall-can calibration.
    grasp_z_offset_m: float = 0.035
    # A single centre-relative offset only works for the tall cans used during
    # the original calibration.  For a flat package it leaves the entire
    # 16-mm-high finger pad above the object.  Put the EE slightly below the
    # RGB-D top surface instead, while retaining ``grasp_z_offset_m`` as a
    # conservative upper bound for unusually tall / partially observed items.
    pinch_grasp_top_inset_m: float = 0.006
    # Offline MuJoCo contact audit of the free-space black bowl showed that a
    # 6-mm inset only caught the polygonal top lip with the finger collision
    # shells: no pad contact formed and width decayed throughout the lift.  A
    # sensor-top-minus-10-mm plane seated both pads while retaining the bowl.
    # Keep this separate from ordinary pinch calibration and derive it only
    # from the RGB-D completed top surface.
    free_space_rim_grasp_top_inset_m: float = 0.010
    # Sensor-derived lower bound relative to the observed support/bottom.  It
    # keeps the open Panda fingertip geometry clear of the table for packages
    # thinner than twice ``pinch_grasp_top_inset_m``.
    pinch_ee_min_above_support_m: float = 0.012
    # The public Akita black-bowl collision AABB is about 107 mm across.  At
    # the calibrated finger-pad height, the sloped wall's balanced grip centre
    # is about 3 mm inward from that measured outer radius.
    rim_pinch_radial_inset_m: float = 0.003
    rim_pinch_min_radius_m: float = 0.025
    rim_pinch_max_radius_m: float = 0.060
    grasp_position_tolerance_m: float = 0.003
    grasp_contact_position_tolerance_m: float = 0.012
    # An exterior bowl can stop the open Panda hand about 25--45 mm before
    # the commanded grip-site height because the fingertip collision shell
    # reaches the lip first.  A single such stop is ambiguous with a fixture
    # collision and therefore still triggers the open antipodal recovery.
    # Only two sensor-bound, exactly antipodal stops on the same world-Z plane
    # may authorize closing on the second physical rim.
    free_space_rim_bilateral_radial_tolerance_m: float = 0.012
    free_space_rim_bilateral_tangential_tolerance_m: float = 0.003
    free_space_rim_bilateral_min_vertical_clearance_m: float = 0.025
    free_space_rim_bilateral_max_vertical_clearance_m: float = 0.045
    free_space_rim_bilateral_seat_plane_tolerance_m: float = 0.008
    # Inside a measured cavity an open fingertip can touch a drawer side a few
    # millimetres before the grip site reaches the requested rim height.  That
    # is a usable seating contact only when XY and wrist orientation are
    # already precise; a larger arbitrary 3-D tolerance would also accept a
    # hand stopped on the cabinet roof.
    cavity_rim_contact_xy_tolerance_m: float = 0.005
    cavity_rim_contact_vertical_tolerance_m: float = 0.008
    # An 80-mm-open Panda finger can bottom out on a drawer floor before the
    # grip site reaches a bowl rim.  At the already-reached elevated pregrasp,
    # use measured finger width to pre-shape a still-open aperture, then
    # bang-bang around that width during descent.  The final CLOSE phase
    # remains the only phase allowed to claim a grasp.
    cavity_rim_preshape_target_width_m: float = 0.020
    cavity_rim_preshape_min_width_m: float = 0.010
    cavity_rim_preshape_max_width_m: float = 0.024
    # A bilateral exterior-rim stall is only evidence that both sides of the
    # sensed bowl were reachable.  Before one final precise descent, form a
    # narrow *closing* aperture at the high waypoint.  This state is separate
    # from the cavity preshape so a drawer retry cannot authorize it.
    free_space_rim_preshape_target_width_m: float = 0.020
    free_space_rim_preshape_min_width_m: float = 0.010
    free_space_rim_preshape_max_width_m: float = 0.024
    # Public Panda geometry: the flat finger-pad working surface spans these
    # tool-local Z offsets from the public grip site.  A final exterior-rim
    # close is allowed only when the fresh RGB-D top plane crosses this band.
    panda_pad_local_z_min_from_grip_site_m: float = -0.0116
    panda_pad_local_z_max_from_grip_site_m: float = 0.0044
    # A cavity pick needs a distinct operational high waypoint.  Fifty
    # millimetres is only the blocked-width proof distance; the carried rim
    # must continue vertically to this 90-mm clearance before any XY motion.
    cavity_pregrasp_height_m: float = 0.090
    # The high traverse is intentionally admitted by a typed, signed gate.
    # The Panda may remain a few millimetres above the requested high pose
    # under fixture load, but must already be laterally aligned and level.
    cavity_pregrasp_planar_tolerance_m: float = 0.008
    cavity_pregrasp_max_height_overshoot_m: float = 0.050
    cavity_pregrasp_rotation_tolerance_rad: float = 0.10
    # Use the Route-C-matched cavity dynamics only while the current attempt
    # has a valid sensor OBB frame.  Free-space picks and placement retain the
    # generic controller calibration.
    cavity_rotation_action_scale_rad: float = 0.50
    cavity_grasp_position_tolerance_m: float = 0.0025
    # Once the narrow hand clears the fixture lip, use the independently
    # calibrated cavity rim plane.  This remains measured from the RGB-D
    # completed top surface and is clamped above the sensed support plane.
    cavity_rim_grasp_top_inset_m: float = 0.010
    # Retried cavity grasps retain the independently calibrated top-minus-10
    # sensor plane.  Candidate diversity comes from the other measured rim
    # side, rather than changing both height and side at once.
    cavity_rim_retry_grasp_top_inset_m: float = 0.010
    # A polygonal / faceted rim can turn exact OBB-axis alignment into a
    # one-pad wedge.  Apply one small, sign-stable yaw to both physical rim
    # sides so their grasp frames remain antipodal and directly comparable.
    cavity_rim_yaw_offset_rad: float = 0.20943951023931956
    # The far-side pregrasp has to pass the visible fixture lip before the
    # width-controlled descent.  Raise only the sensor-selected outside finger
    # while traversing to that high waypoint, then level at the same XYZ before
    # descent.  The final grasp/seat/proof orientation remains horizontal.
    cavity_rim_outer_finger_lift_rad: float = 0.17453292519943295
    cavity_rim_min_final_wall_clearance_m: float = 0.025
    # Prefer a direct robot-facing grasp when that side itself has enough
    # observed wall clearance. A roomy far side need not be reached first,
    # and an additional camera view does not invalidate measured near space.
    cavity_rim_roomy_near_clearance_m: float = 0.040
    cavity_rim_roomy_near_top_inset_m: float = 0.006
    # A strict 2-D IN fallback can identify the correct bowl while refusing to
    # invent a drawer OBB from insufficient depth.  Before giving up, move the
    # empty hand to at most two collision-safe, constant-height RGB-D views and
    # rerun the original relational selector.  No grasp target is emitted
    # until the fresh reference passes the existing full 3-D cavity gate.
    cavity_active_view_clearance_m: float = 0.180
    cavity_active_view_offset_m: float = 0.060
    # Constant-height viewing is an operational high waypoint, not a grasp
    # pose.  The real arm settles a few millimetres below/aside the exact
    # Cartesian request while remaining safely above the sensed fixture.  Use
    # a signed component gate here; it authorizes only a fresh RGB-D query.
    cavity_active_view_planar_tolerance_m: float = 0.010
    cavity_active_view_max_target_shortfall_m: float = 0.012
    # A high RGB-D observation may settle just outside the primary component
    # gate without contacting anything.  Admit a deliberately narrow second
    # envelope only after the existing proprioceptive progress window has
    # plateaued and the measured Cartesian span stays bounded.  This gate can
    # authorize a fresh relational observation; it never authorizes descent.
    cavity_active_view_plateau_planar_tolerance_m: float = 0.011
    cavity_active_view_plateau_max_target_shortfall_m: float = 0.0125
    cavity_active_view_plateau_cartesian_span_m: float = 0.004
    # Treat the first unobstructed RGB-D centre as the retry prior, but admit a
    # small radial innovation after a proof-lift failure.  The failed contact
    # can physically nudge a bowl; accepting the complete post-contact crop
    # shift (8--9 mm in the motivating trace) over-corrects, while freezing it
    # entirely leaves only one finger body on the rim.  This bounded update is
    # applied only to a cavity retry and its sign comes from fresh RGB-D.
    cavity_rim_retry_radial_update_m: float = 0.004
    cavity_rim_retry_tangential_update_m: float = 0.003
    # With the pads closed on a sloped rim, a short support-plane pull toward
    # the already selected outside rim direction seats both pads before lift.
    # The direction is the RGB-D cavity/rim proposal; distance and duration
    # stay deliberately small and the original blocked-width proof remains
    # authoritative.
    cavity_rim_seat_pull_m: float = 0.008
    cavity_rim_seat_pull_ticks: int = 5
    cavity_rim_seat_pull_min_progress_m: float = 0.007
    cavity_rim_seat_pull_max_orthogonal_error_m: float = 0.002
    cavity_rim_seat_pull_max_vertical_error_m: float = 0.002
    # A rigid fixture may stop the requested radial pull before 7 mm while
    # the bowl settles more deeply between the pads.  Admit that second,
    # proprioceptive completion only after measurable outward progress has
    # plateaued and blocked width has increased; proof lift remains mandatory.
    cavity_rim_seat_contact_min_progress_m: float = 0.001
    cavity_rim_seat_contact_progress_span_m: float = 0.0003
    cavity_rim_seat_contact_min_width_gain_m: float = 0.0004
    cavity_rim_seat_contact_max_vertical_error_m: float = 0.004
    cavity_rim_seat_contact_max_rotation_error_rad: float = 0.10
    cavity_rim_seat_contact_window_ticks: int = 4
    cavity_rim_seat_pull_max_ticks: int = 15
    # A transient blocked width at CLOSE is not evidence of a retained bowl.
    # Require it to remain blocked through a meaningful vertical proof lift.
    cavity_rim_proof_lift_m: float = 0.050
    # After proof, finish the same-XY vertical escape.  This decomposed gate
    # does not confuse a harmless vertical servo residual with lateral
    # clearance, and retains explicit rotation / blocked-width evidence.
    cavity_operational_planar_tolerance_m: float = 0.008
    cavity_operational_target_z_gap_m: float = 0.008
    cavity_operational_rotation_tolerance_rad: float = 0.10
    # An explicitly injected expansion grasp may descend into an open object
    # whose rim stops the compliant EE before the Cartesian centre is exact.
    # Keep the tighter production pinch tolerance above and scope this
    # allowance to the opt-in EXPAND mode only.
    expand_contact_position_tolerance_m: float = 0.025
    # The object is static before contact.  Once the hand occludes it, a fresh
    # connected component can slide onto a finger or neighbouring package.
    # Only accept refreshes consistent with the first sensor measurement.
    pick_refresh_max_shift_m: float = 0.008
    # Relational retry association is intentionally tighter than held-object
    # tracking: released same-label distractors 46--86 mm away must not replace
    # the source selected by the initial ON/IN language grounding.
    selector_retry_reacquire_radius_m: float = 0.045
    # A pre-contact exterior-rim retry is also bound to its first sensor
    # component.  Centre proximity alone can still admit a neighbouring or
    # fused same-label crop, so require its axis-order-independent extents to
    # remain within this bounded scale envelope.
    free_space_rim_retry_min_extent_scale: float = 0.65
    free_space_rim_retry_max_extent_scale: float = 1.50
    max_grasp_retries: int = 2
    grasp_retry_release_ticks: int = 8
    grasp_retry_height_step_m: float = 0.004
    # RGB-D SceneObject.centroid_world is the generic sensor target for an
    # explicitly injected expansion mode.  The production selector does not
    # assign this mode to any entity.
    expand_grasp_z_offset_m: float = 0.0
    # Finger motion is not instantaneous in robosuite.  Keep the hand
    # stationary until an explicitly requested expansion is nearly complete.
    expand_engage_hold_ticks: int = 12
    # The real Panda simulation is still about 41 mm open after the generic
    # six-tick close dwell.  A rim grasp must reach the thin-wall contact band
    # before retreat starts; otherwise the rising hand pulls the pads above
    # the rim while they are still closing.  Free-space calibration reaches
    # the 2--6 mm band after about 15 close commands.
    rim_pinch_engage_hold_ticks: int = 15
    release_clearance_m: float = 0.006
    container_transfer_clearance_m: float = 0.025
    close_hold_ticks: int = 6
    open_hold_ticks: int = 5
    release_settle_retry_ticks: int = 15
    visual_refresh_ticks: int = 5
    max_perception_misses: int = 3
    phase_timeout_ticks: int = 160
    holding_distance_m: float = 0.065
    gripper_open_width_m: float = 0.07
    gripper_blocked_min_width_m: float = 0.005
    # A bowl wall is thinner than ordinary packages.  The free-space Panda
    # close measured about 2.35 mm at the normal verify time; require a
    # slightly wider sensor reading to accept a rim pinch.
    rim_pinch_blocked_min_width_m: float = 0.003
    # A rim crop can include a support directly below the bowl (for example a
    # ramekin), making an otherwise squat object look as tall as it is wide.
    # The mechanical rim-pinch contract gives us a sensor-only shape prior:
    # cap the half-height used for PLACE_ON to this fraction of the measured
    # planar diameter.  A clean LIBERO bowl (50.5 mm high, 107.2 mm wide) is
    # unaffected, while a fused 107-mm-tall crop no longer causes a high drop.
    rim_pinch_place_half_height_ratio: float = 0.25
    # A width only just above the calibrated empty-close signature denotes a
    # marginal rim hold.  Limit Cartesian translation while carrying such an
    # object so acceleration cannot turn a weak edge contact into a swing.
    rim_pinch_weak_width_margin_m: float = 0.001
    weak_grasp_translation_action_limit: float = 0.35
    # A thin-wall hold just above the empty-close signature is mechanically
    # compliant: during transfer the bowl can rotate until its centre of mass
    # hangs below the wrist.  In that measured marginal regime the original
    # rim-radius XY offset is no longer a valid rigid transform.  Plan the
    # release beneath the wrist while retaining the observed vertical offset.
    marginal_rim_placement_xy_offset_scale: float = 0.0
    # A post-retreat RGB-D centre must sit observably below the grip site
    # before it can replace the initial rigid rim offset.  This rejects the
    # common hand-centred crop (about 10 mm below the EE) while retaining both
    # upright and pendulum-like bowl observations.
    rim_held_visual_min_below_ee_m: float = 0.012
    rim_held_visual_min_distance_ratio: float = 0.35
    rim_held_visual_max_prediction_shift_ratio: float = 0.65
    rim_held_visual_max_orthogonal_ratio: float = 0.25
    rim_preplace_settle_ticks: int = 4
    # Opening a Panda hand from the thin-wall contact band takes longer than
    # the generic release dwell.  Keep the wrist still until both fingers are
    # clear and the released object has settled on its support.
    rim_pinch_release_hold_ticks: int = 12
    gripper_expand_min_width_m: float = 0.02
    relation_xy_margin_m: float = 0.025
    relation_z_tolerance_m: float = 0.035
    # A fused two-bowl crop remaining at its original support is not evidence
    # that the closed fingers retained the stack.  Require fresh visible group
    # geometry to rise by this much during the second PICK proof.
    formed_stack_proof_lift_m: float = 0.035
    # Language-selected subregions are placed as far toward the requested
    # side as the measured source/target AABBs safely allow.  Relative goals
    # use a small visible gap rather than a benchmark-coordinate waypoint.
    placement_region_fraction: float = 0.72
    relative_object_gap_m: float = 0.025
    contact_position_tolerance_m: float = 0.075
    # A carried rim-grasped bowl can meet its support before the release EE
    # pose is exact.  The legacy 75-mm isotropic contact allowance is far too
    # broad for this case: it can accept a sideways cabinet collision.  Keep a
    # separate, directional sensor/proprio gate for thin-wall placement.
    rim_place_contact_xy_tolerance_m: float = 0.015
    rim_place_contact_vertical_tolerance_m: float = 0.025
    # Scalar distance can look flat while the hand slides tangentially around
    # the target.  Require the measured Cartesian path itself to settle.
    rim_place_contact_cartesian_span_m: float = 0.008
    contact_progress_epsilon_m: float = 0.0004
    contact_stall_ticks: int = 8
    contact_min_ticks: int = 10

    def __post_init__(self) -> None:
        positive_floats = (
            self.position_action_scale_m,
            self.rotation_action_scale_rad,
            self.position_tolerance_m,
            self.rotation_tolerance_rad,
            self.pregrasp_height_m,
            self.preplace_height_m,
            self.container_transfer_clearance_m,
            self.holding_distance_m,
            self.gripper_open_width_m,
            self.gripper_blocked_min_width_m,
            self.gripper_expand_min_width_m,
            self.contact_position_tolerance_m,
            self.contact_progress_epsilon_m,
            self.relative_object_gap_m,
            self.pinch_grasp_top_inset_m,
            self.free_space_rim_grasp_top_inset_m,
            self.pinch_ee_min_above_support_m,
            self.rim_pinch_radial_inset_m,
            self.rim_pinch_min_radius_m,
            self.rim_pinch_max_radius_m,
            self.rim_pinch_blocked_min_width_m,
            self.rim_pinch_place_half_height_ratio,
            self.rim_pinch_weak_width_margin_m,
            self.weak_grasp_translation_action_limit,
            self.grasp_position_tolerance_m,
            self.grasp_contact_position_tolerance_m,
            self.free_space_rim_bilateral_radial_tolerance_m,
            self.free_space_rim_bilateral_tangential_tolerance_m,
            self.free_space_rim_bilateral_min_vertical_clearance_m,
            self.free_space_rim_bilateral_max_vertical_clearance_m,
            self.free_space_rim_bilateral_seat_plane_tolerance_m,
            self.cavity_rim_contact_xy_tolerance_m,
            self.cavity_rim_contact_vertical_tolerance_m,
            self.cavity_rim_preshape_target_width_m,
            self.cavity_rim_preshape_min_width_m,
            self.cavity_rim_preshape_max_width_m,
            self.free_space_rim_preshape_target_width_m,
            self.free_space_rim_preshape_min_width_m,
            self.free_space_rim_preshape_max_width_m,
            self.cavity_pregrasp_height_m,
            self.cavity_pregrasp_planar_tolerance_m,
            self.cavity_pregrasp_max_height_overshoot_m,
            self.cavity_pregrasp_rotation_tolerance_rad,
            self.cavity_rotation_action_scale_rad,
            self.cavity_grasp_position_tolerance_m,
            self.cavity_rim_grasp_top_inset_m,
            self.cavity_rim_retry_grasp_top_inset_m,
            self.cavity_rim_yaw_offset_rad,
            self.cavity_rim_outer_finger_lift_rad,
            self.cavity_rim_min_final_wall_clearance_m,
            self.cavity_rim_roomy_near_clearance_m,
            self.cavity_rim_roomy_near_top_inset_m,
            self.cavity_active_view_clearance_m,
            self.cavity_active_view_offset_m,
            self.cavity_active_view_planar_tolerance_m,
            self.cavity_active_view_max_target_shortfall_m,
            self.cavity_active_view_plateau_planar_tolerance_m,
            self.cavity_active_view_plateau_max_target_shortfall_m,
            self.cavity_active_view_plateau_cartesian_span_m,
            self.cavity_rim_retry_radial_update_m,
            self.cavity_rim_retry_tangential_update_m,
            self.cavity_rim_seat_pull_m,
            self.cavity_rim_seat_pull_min_progress_m,
            self.cavity_rim_seat_pull_max_orthogonal_error_m,
            self.cavity_rim_seat_pull_max_vertical_error_m,
            self.cavity_rim_seat_contact_min_progress_m,
            self.cavity_rim_seat_contact_progress_span_m,
            self.cavity_rim_seat_contact_min_width_gain_m,
            self.cavity_rim_seat_contact_max_vertical_error_m,
            self.cavity_rim_seat_contact_max_rotation_error_rad,
            self.cavity_rim_proof_lift_m,
            self.cavity_operational_planar_tolerance_m,
            self.cavity_operational_target_z_gap_m,
            self.cavity_operational_rotation_tolerance_rad,
            self.rim_place_contact_xy_tolerance_m,
            self.rim_place_contact_vertical_tolerance_m,
            self.rim_place_contact_cartesian_span_m,
            self.expand_contact_position_tolerance_m,
            self.pick_refresh_max_shift_m,
            self.selector_retry_reacquire_radius_m,
            self.free_space_rim_retry_min_extent_scale,
            self.free_space_rim_retry_max_extent_scale,
            self.grasp_retry_height_step_m,
            self.rim_held_visual_min_below_ee_m,
            self.rim_held_visual_min_distance_ratio,
            self.rim_held_visual_max_prediction_shift_ratio,
            self.rim_held_visual_max_orthogonal_ratio,
            self.formed_stack_proof_lift_m,
        )
        if any(value <= 0 for value in positive_floats):
            raise ValueError("controller scales, tolerances, and clearances must be positive")
        if not 0.0 < self.placement_region_fraction <= 1.0:
            raise ValueError("placement_region_fraction must lie in (0, 1]")
        if (
            self.cavity_active_view_plateau_max_target_shortfall_m
            >= self.cavity_active_view_clearance_m
        ):
            raise ValueError(
                "active-view target shortfall must remain below its sensor clearance"
            )
        if (
            self.cavity_active_view_plateau_planar_tolerance_m
            < self.cavity_active_view_planar_tolerance_m
            or self.cavity_active_view_plateau_max_target_shortfall_m
            < self.cavity_active_view_max_target_shortfall_m
        ):
            raise ValueError(
                "active-view plateau envelope must contain the primary envelope"
            )
        if self.expand_grasp_z_offset_m < 0:
            raise ValueError("expand grasp z offset must be non-negative")
        if not (
            self.free_space_rim_retry_min_extent_scale <= 1.0
            <= self.free_space_rim_retry_max_extent_scale
        ):
            raise ValueError(
                "free-space rim retry extent scales must contain unity"
            )
        if self.rim_pinch_min_radius_m >= self.rim_pinch_max_radius_m:
            raise ValueError("rim-pinch radius bounds must be ordered")
        if (
            self.free_space_rim_bilateral_min_vertical_clearance_m
            >= self.free_space_rim_bilateral_max_vertical_clearance_m
        ):
            raise ValueError(
                "free-space bilateral rim vertical-clearance bounds must be ordered"
            )
        if self.rim_pinch_blocked_min_width_m >= self.gripper_open_width_m:
            raise ValueError("rim-pinch blocked width must be below open width")
        if not (
            self.rim_pinch_blocked_min_width_m
            < self.cavity_rim_preshape_min_width_m
            < self.cavity_rim_preshape_target_width_m
            < self.cavity_rim_preshape_max_width_m
            < self.gripper_open_width_m
        ):
            raise ValueError(
                "cavity rim preshape widths must be ordered above the grasp "
                "verification threshold and below open width"
            )
        if not (
            self.rim_pinch_blocked_min_width_m
            < self.free_space_rim_preshape_min_width_m
            < self.free_space_rim_preshape_target_width_m
            < self.free_space_rim_preshape_max_width_m
            < self.gripper_open_width_m
        ):
            raise ValueError(
                "free-space rim preshape widths must be ordered above the "
                "grasp verification threshold and below open width"
            )
        if not (
            np.isfinite(self.panda_pad_local_z_min_from_grip_site_m)
            and np.isfinite(self.panda_pad_local_z_max_from_grip_site_m)
            and self.panda_pad_local_z_min_from_grip_site_m < 0.0
            < self.panda_pad_local_z_max_from_grip_site_m
        ):
            raise ValueError("Panda pad work band must be finite and straddle grip site")
        if (
            self.cavity_rim_grasp_top_inset_m
            < self.free_space_rim_grasp_top_inset_m
        ):
            raise ValueError(
                "cavity rim top inset must not be shallower than free-space inset"
            )
        if self.cavity_rim_retry_grasp_top_inset_m > self.cavity_rim_grasp_top_inset_m:
            raise ValueError("cavity rim retry must not be deeper than its first candidate")
        if self.cavity_rim_yaw_offset_rad >= np.pi / 4.0:
            raise ValueError("cavity rim yaw diversity must stay local to the OBB axis")
        if self.cavity_rim_outer_finger_lift_rad > np.deg2rad(25.0):
            raise ValueError("cavity outer-finger lift must be at most 25 degrees")
        if (
            self.cavity_rim_roomy_near_top_inset_m
            > self.cavity_rim_grasp_top_inset_m
        ):
            raise ValueError(
                "roomy near rim plane must not be deeper than the constrained "
                "cavity rim plane"
            )
        if self.cavity_rim_seat_pull_m >= self.cavity_rim_min_final_wall_clearance_m:
            raise ValueError("cavity rim seat pull must remain inside wall clearance")
        if self.cavity_rim_seat_pull_min_progress_m > self.cavity_rim_seat_pull_m:
            raise ValueError("cavity rim seat progress cannot exceed its target distance")
        if self.cavity_rim_proof_lift_m >= self.cavity_pregrasp_height_m:
            raise ValueError("cavity rim proof lift must be below cavity pregrasp height")
        if (
            self.cavity_operational_target_z_gap_m
            >= self.cavity_pregrasp_height_m
        ):
            raise ValueError(
                "cavity operational z gap must be below cavity pregrasp height"
            )
        if self.weak_grasp_translation_action_limit > 1.0:
            raise ValueError("weak-grasp translation action limit cannot exceed 1")
        if any(
            ratio > 1.0
            for ratio in (
                self.rim_held_visual_min_distance_ratio,
                self.rim_held_visual_max_prediction_shift_ratio,
                self.rim_held_visual_max_orthogonal_ratio,
            )
        ):
            raise ValueError("rim-held visual ratios cannot exceed 1")
        if not 0.0 <= self.marginal_rim_placement_xy_offset_scale <= 1.0:
            raise ValueError("marginal rim placement XY offset scale must be in [0, 1]")
        positive_ints = (
            self.close_hold_ticks,
            self.open_hold_ticks,
            self.release_settle_retry_ticks,
            self.visual_refresh_ticks,
            self.max_perception_misses,
            self.phase_timeout_ticks,
            self.contact_stall_ticks,
            self.contact_min_ticks,
            self.grasp_retry_release_ticks,
            self.expand_engage_hold_ticks,
            self.rim_pinch_engage_hold_ticks,
            self.rim_pinch_release_hold_ticks,
            self.rim_preplace_settle_ticks,
            self.cavity_rim_seat_pull_ticks,
            self.cavity_rim_seat_contact_window_ticks,
            self.cavity_rim_seat_pull_max_ticks,
        )
        if any(value <= 0 for value in positive_ints):
            raise ValueError("controller tick limits must be positive")
        if self.cavity_rim_seat_pull_ticks > self.cavity_rim_seat_pull_max_ticks:
            raise ValueError("cavity rim seat-pull tick budget must be ordered")
        if self.max_grasp_retries < 0:
            raise ValueError("max_grasp_retries cannot be negative")


class PlacementPlanner(Protocol):
    def plan(
        self,
        kind: SkillKind,
        source: SceneObject,
        destination: SceneObject,
        *,
        held_offset_world: np.ndarray,
        ee_rotation_world: np.ndarray,
        config: ControllerConfig,
        relation: str | None = None,
        support_z_m: float | None = None,
        semantic_direction_world: np.ndarray | None = None,
    ) -> tuple[Pose, Pose]:
        """Return ``(release_ee_pose, preplace_ee_pose)``."""


class GraspModeSelector(Protocol):
    def select(self, step: SkillStep) -> GraspMode:
        ...


class EntityGraspModeSelector:
    """Use a closing rim pinch for bowl/black-bowl sources and pinch otherwise.

    Internal expansion remains available only by explicit injection.  The
    exact ``bowl`` and ``black bowl`` mappings are auditable semantic special
    cases: LIBERO sometimes omits the black colour from the instruction, but
    both mappings still use a sensor-derived rim target and close the fingers.
    Explicitly named bowls of another colour are unaffected.
    """

    def __init__(
        self,
        expansion_entities: tuple[str, ...] = (),
        *,
        rim_pinch_entities: tuple[str, ...] = ("black bowl", "bowl"),
    ) -> None:
        protected_expansion_entities = frozenset({"black bowl", "bowl"})
        protected_rim_entities = frozenset({"black bowl", "bowl"})
        self.expansion_entities = frozenset(
            name.strip().lower() for name in expansion_entities
        )
        self.rim_pinch_entities = protected_rim_entities | frozenset(
            name.strip().lower() for name in rim_pinch_entities
        )
        overlap = self.expansion_entities & protected_expansion_entities
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(
                "grasp-mode overlap: closed-finger rim entities cannot use "
                "internal expansion: "
                f"{names}"
            )
        overlap = self.expansion_entities & self.rim_pinch_entities
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(f"grasp-mode entity sets overlap: {names}")

    def select(self, step: SkillStep) -> GraspMode:
        subject = step.subject.strip().lower()
        if subject in self.expansion_entities:
            return GraspMode.EXPAND
        if subject in self.rim_pinch_entities:
            return GraspMode.RIM_PINCH
        return GraspMode.PINCH


class AABBPlacementPlanner:
    """Conservative top-down placement proposal from RGB-D axis-aligned bounds."""

    def plan(
        self,
        kind: SkillKind,
        source: SceneObject,
        destination: SceneObject,
        *,
        held_offset_world: np.ndarray,
        ee_rotation_world: np.ndarray,
        config: ControllerConfig,
        relation: str | None = None,
        support_z_m: float | None = None,
        semantic_direction_world: np.ndarray | None = None,
    ) -> tuple[Pose, Pose]:
        source_half_height = max(source.height_m * 0.5, 0.005)
        desired_object_center = destination.centroid_world.copy()
        source_half_world = (
            source.bounds_max_world - source.bounds_min_world
        ) / 2.0
        target_half_world = (
            destination.bounds_max_world - destination.bounds_min_world
        ) / 2.0

        if kind in {SkillKind.PLACE_ON, SkillKind.STACK}:
            desired_object_center[2] = (
                destination.bounds_max_world[2]
                + source_half_height
                + config.release_clearance_m
            )
        elif kind is SkillKind.PLACE_IN:
            desired_object_center[2] = (
                destination.bounds_min_world[2]
                + source_half_height
                + config.release_clearance_m
            )
        elif kind is SkillKind.PLACE_RELATIVE:
            if relation in {"left_of", "right_of", "front_of"}:
                if semantic_direction_world is None:
                    semantic_direction_world = {
                        "left_of": np.array((0.0, -1.0, 0.0)),
                        "right_of": np.array((0.0, 1.0, 0.0)),
                        "front_of": np.array((1.0, 0.0, 0.0)),
                    }[relation]
                direction = np.asarray(
                    semantic_direction_world,
                    dtype=np.float64,
                ).copy()
                direction[2] = 0.0
                norm = float(np.linalg.norm(direction))
                if norm < 1e-8:
                    raise ValueError("relative placement direction is degenerate")
                direction /= norm
                source_radius = float(np.dot(np.abs(direction), source_half_world))
                target_radius = float(np.dot(np.abs(direction), target_half_world))
                desired_object_center[:2] = (
                    destination.centroid_world[:2]
                    + direction[:2]
                    * (
                        source_radius
                        + target_radius
                        + config.relative_object_gap_m
                    )
                )
            elif relation == "under":
                # XY centre is already the sensor-derived fixture centre.
                pass
            else:
                raise ValueError(f"unsupported relative relation: {relation!r}")
            if support_z_m is None:
                raise ValueError("relative placement requires a measured support height")
            desired_object_center[2] = (
                float(support_z_m)
                + source_half_height
                + config.release_clearance_m
            )
        else:
            raise ValueError(f"unsupported placement kind: {kind.value}")

        region = relation
        if region in {
            "front",
            "back",
            "front_compartment",
            "back_compartment",
            "left_compartment",
            "right_compartment",
        }:
            if semantic_direction_world is None:
                semantic_direction_world = {
                    "front": np.array((1.0, 0.0, 0.0)),
                    "back": np.array((-1.0, 0.0, 0.0)),
                    "front_compartment": np.array((1.0, 0.0, 0.0)),
                    "back_compartment": np.array((-1.0, 0.0, 0.0)),
                    "left_compartment": np.array((0.0, -1.0, 0.0)),
                    "right_compartment": np.array((0.0, 1.0, 0.0)),
                }[region]
            direction = np.asarray(
                semantic_direction_world,
                dtype=np.float64,
            ).copy()
            direction[2] = 0.0
            norm = float(np.linalg.norm(direction))
            if norm < 1e-8:
                raise ValueError("placement-region direction is degenerate")
            direction /= norm
            source_radius = float(np.dot(np.abs(direction), source_half_world))
            target_radius = float(np.dot(np.abs(direction), target_half_world))
            available = max(
                target_radius
                - source_radius
                - config.release_clearance_m,
                0.0,
            )
            desired_object_center[:2] += (
                direction[:2]
                * config.placement_region_fraction
                * available
            )
        release_position = desired_object_center - held_offset_world
        release = Pose(release_position, ee_rotation_world)
        preplace_position = release_position.copy()
        preplace_position[2] += config.preplace_height_m
        if kind is SkillKind.PLACE_IN:
            # During the lateral transfer the carried object's bottom must be
            # above the receptacle rim.  Merely adding a fixed offset to the
            # low in-container release pose lets the package strike and push
            # LIBERO's light basket before the descent begins.
            rim_safe_ee_z = (
                destination.bounds_max_world[2]
                + source_half_height
                + config.container_transfer_clearance_m
                - held_offset_world[2]
            )
            preplace_position[2] = max(preplace_position[2], rim_safe_ee_z)
        return release, Pose(preplace_position, ee_rotation_world)


class SensorRelationVerifier:
    """Visual/geometric termination checks with no simulator predicates."""

    def holding(
        self,
        snapshot: SceneSnapshot,
        observation: SensorObservation,
        object_name: str,
        grasp_mode: GraspMode,
        config: ControllerConfig,
    ) -> bool:
        detected = snapshot.objects.get(object_name)
        if grasp_mode is not GraspMode.EXPAND:
            blocked_min_width = (
                config.rim_pinch_blocked_min_width_m
                if grasp_mode is GraspMode.RIM_PINCH
                else config.gripper_blocked_min_width_m
            )
            gripper_engaged = (
                blocked_min_width
                <= observation.robot.gripper_width_m
                < config.gripper_open_width_m
            )
        else:
            gripper_engaged = (
                observation.robot.gripper_width_m > config.gripper_expand_min_width_m
            )
        if grasp_mode is not GraspMode.EXPAND:
            # A lifted package is often occluded by the fingers and is also
            # deliberately removed by the tabletop support-gap filter.  Other
            # table packages can then be returned as a far false match.  The
            # gripper stopping above its calibrated empty width is the
            # authoritative robot-side contact measurement for a centre or
            # rim pinch.  The latter uses its own thin-wall lower threshold.
            return bool(gripper_engaged)
        # An opt-in expansion grasp has a proprioceptive blocked-width
        # signature too: after a free OPEN command the Panda reaches about
        # 77 mm, whereas an object retained between the fingers remains below
        # the calibrated ``gripper_open_width_m`` threshold after the retreat.  This is the
        # same robot-side evidence used for PINCH, with the inequality reversed.
        # It also avoids a false negative when the wrist/hand occludes most of
        # the lifted object and the detector returns a biased crop centre.
        expansion_loaded = bool(
            config.gripper_expand_min_width_m
            < observation.robot.gripper_width_m
            < config.gripper_open_width_m
        )
        if expansion_loaded:
            return True
        if detected is None:
            return False
        distance = float(
            np.linalg.norm(detected.centroid_world - observation.robot.ee_pose.position)
        )
        return gripper_engaged and distance <= config.holding_distance_m

    def placed(
        self,
        kind: SkillKind,
        snapshot: SceneSnapshot,
        source_name: str,
        destination_name: str,
        config: ControllerConfig,
        relation: str | None = None,
        semantic_direction_world: np.ndarray | None = None,
    ) -> bool:
        source = snapshot.objects.get(source_name)
        destination = snapshot.objects.get(destination_name)
        if source is None or destination is None:
            return False
        if kind is SkillKind.PLACE_RELATIVE:
            if relation in {"left_of", "right_of", "front_of"}:
                if semantic_direction_world is None:
                    semantic_direction_world = {
                        "left_of": np.array((0.0, -1.0, 0.0)),
                        "right_of": np.array((0.0, 1.0, 0.0)),
                        "front_of": np.array((1.0, 0.0, 0.0)),
                    }[relation]
                direction = np.asarray(
                    semantic_direction_world,
                    dtype=np.float64,
                ).copy()
                direction[2] = 0.0
                norm = float(np.linalg.norm(direction))
                if norm < 1e-8:
                    return False
                direction /= norm
                source_half = (
                    source.bounds_max_world - source.bounds_min_world
                ) / 2.0
                target_half = (
                    destination.bounds_max_world - destination.bounds_min_world
                ) / 2.0
                source_radius = float(np.dot(np.abs(direction), source_half))
                target_radius = float(np.dot(np.abs(direction), target_half))
                separation = float(
                    np.dot(
                        source.centroid_world - destination.centroid_world,
                        direction,
                    )
                )
                return bool(
                    separation
                    >= source_radius
                    + target_radius
                    - config.relation_xy_margin_m
                )
            if relation == "under":
                within_xy = np.all(
                    source.centroid_world[:2]
                    >= destination.bounds_min_world[:2] - config.relation_xy_margin_m
                ) and np.all(
                    source.centroid_world[:2]
                    <= destination.bounds_max_world[:2] + config.relation_xy_margin_m
                )
                return bool(
                    within_xy
                    and source.bounds_max_world[2]
                    <= destination.bounds_min_world[2] + config.relation_z_tolerance_m
                )
            return False

        xy = source.centroid_world[:2]
        within_xy = np.all(
            xy >= destination.bounds_min_world[:2] - config.relation_xy_margin_m
        ) and np.all(xy <= destination.bounds_max_world[:2] + config.relation_xy_margin_m)
        if not within_xy:
            return False
        if relation in {
            "front",
            "back",
            "front_compartment",
            "back_compartment",
            "left_compartment",
            "right_compartment",
        }:
            if semantic_direction_world is None:
                semantic_direction_world = {
                    "front": np.array((1.0, 0.0, 0.0)),
                    "back": np.array((-1.0, 0.0, 0.0)),
                    "front_compartment": np.array((1.0, 0.0, 0.0)),
                    "back_compartment": np.array((-1.0, 0.0, 0.0)),
                    "left_compartment": np.array((0.0, -1.0, 0.0)),
                    "right_compartment": np.array((0.0, 1.0, 0.0)),
                }[relation]
            direction = np.asarray(
                semantic_direction_world,
                dtype=np.float64,
            ).copy()
            direction[2] = 0.0
            norm = float(np.linalg.norm(direction))
            if norm < 1e-8:
                return False
            direction /= norm
            if float(
                np.dot(
                    source.centroid_world - destination.centroid_world,
                    direction,
                )
            ) < 0.0:
                return False
        if kind in {SkillKind.PLACE_ON, SkillKind.STACK}:
            source_bottom = source.bounds_min_world[2]
            vertical_error = source_bottom - destination.bounds_max_world[2]
            return abs(float(vertical_error)) <= config.relation_z_tolerance_m
        if kind is SkillKind.PLACE_IN:
            source_bottom = source.bounds_min_world[2]
            return bool(
                source_bottom >= destination.bounds_min_world[2] - config.relation_z_tolerance_m
                and source.centroid_world[2]
                <= destination.bounds_max_world[2] + config.relation_z_tolerance_m
            )
        return False

    def stacked(
        self,
        detected: SceneObject | None,
        source_before: SceneObject | None,
        destination_before: SceneObject | None,
        config: ControllerConfig,
    ) -> bool:
        """Verify a same-label stack from fresh RGB-D height growth.

        Same-label bowls may fuse into one post-release component, so ordinary
        source/target dictionary keys cannot express both.  A successful stack
        must instead produce new visible geometry over the frozen destination
        footprint whose top rises materially above the destination's original
        top.  An unchanged bottom bowl cannot pass this gate.
        """

        if detected is None or source_before is None or destination_before is None:
            return False
        within_xy = np.all(
            detected.centroid_world[:2]
            >= destination_before.bounds_min_world[:2] - config.relation_xy_margin_m
        ) and np.all(
            detected.centroid_world[:2]
            <= destination_before.bounds_max_world[:2] + config.relation_xy_margin_m
        )
        minimum_growth = max(0.40 * source_before.height_m, 0.012)
        height_growth = (
            detected.bounds_max_world[2]
            - destination_before.bounds_max_world[2]
        )
        return bool(
            within_xy
            and height_growth >= minimum_growth
            and self.formed_stack_shape(
                detected,
                source_before,
                destination_before,
            )
        )

    @staticmethod
    def formed_stack_shape(
        detected: SceneObject | None,
        top_before: SceneObject | None,
        base_before: SceneObject | None,
    ) -> bool:
        """Recognize retention of two stacked members without a world anchor."""

        if detected is None or top_before is None or base_before is None:
            return False
        minimum_height = base_before.height_m + max(
            0.40 * top_before.height_m,
            0.012,
        )
        maximum_height = 1.8 * (base_before.height_m + top_before.height_m)
        detected_size = detected.bounds_max_world - detected.bounds_min_world
        top_size = top_before.bounds_max_world - top_before.bounds_min_world
        base_size = base_before.bounds_max_world - base_before.bounds_min_world
        member_planar_max = max(
            float(np.max(top_size[:2])),
            float(np.max(base_size[:2])),
            1e-6,
        )
        detected_planar_max = float(np.max(detected_size[:2]))
        detected_planar_min = float(np.min(detected_size[:2]))
        member_planar_min = max(
            min(float(np.min(top_size[:2])), float(np.min(base_size[:2]))),
            1e-6,
        )
        return bool(
            minimum_height <= detected.height_m <= maximum_height
            and detected_planar_max <= 1.8 * member_planar_max
            and detected_planar_min >= 0.45 * member_planar_min
        )

    def formed_stack_inside(
        self,
        detected: SceneObject | None,
        destination: SceneObject | None,
        top_before: SceneObject | None,
        base_before: SceneObject | None,
        config: ControllerConfig,
    ) -> bool:
        """Verify the retained two-member shape was released inside a receptacle."""

        if destination is None or not self.formed_stack_shape(
            detected,
            top_before,
            base_before,
        ):
            return False
        assert detected is not None
        xy = detected.centroid_world[:2]
        within_xy = bool(
            np.all(
                xy
                >= destination.bounds_min_world[:2]
                - config.relation_xy_margin_m
            )
            and np.all(
                xy
                <= destination.bounds_max_world[:2]
                + config.relation_xy_margin_m
            )
        )
        stack_bottom = float(detected.bounds_min_world[2])
        bottom_inside = bool(
            destination.bounds_min_world[2] - config.relation_z_tolerance_m
            <= stack_bottom
            <= destination.bounds_max_world[2] + config.relation_z_tolerance_m
        )
        return within_xy and bottom_inside


class RouteBController:
    """Execute a validated TaskSpec as visual closed-loop Cartesian skills."""

    def __init__(
        self,
        perception: ScenePerception,
        *,
        compiler: TaskCompiler | None = None,
        placement_planner: PlacementPlanner | None = None,
        grasp_mode_selector: GraspModeSelector | None = None,
        verifier: SensorRelationVerifier | None = None,
        config: ControllerConfig | None = None,
    ) -> None:
        self.perception = perception
        self.compiler = compiler or TaskCompiler()
        self.placement_planner = placement_planner or AABBPlacementPlanner()
        self.grasp_mode_selector = grasp_mode_selector or EntityGraspModeSelector()
        self.verifier = verifier or SensorRelationVerifier()
        self.config = config or ControllerConfig()
        self._spec: TaskSpec | None = None
        self._status = ExecutorStatus.IDLE
        self._skill_index = 0
        self._phase = "idle"
        self._phase_ticks = 0
        self._perception_misses = 0
        self._message = ""
        self._motion_pose: Pose | None = None
        self._secondary_pose: Pose | None = None
        self._transfer_clearance_pose: Pose | None = None
        self._transfer_clearance_z_m: float | None = None
        self._rim_transfer_rotation_reference: np.ndarray | None = None
        self._held_subject: str | None = None
        self._release_width_target_m: float | None = None
        self._release_settle_retry_used = False
        self._release_recenter_pose: Pose | None = None
        self._release_peel_pose: Pose | None = None
        self._held_offset_world = np.zeros(3, dtype=np.float64)
        self._held_offset_source = "none"
        self._held_geometry: SceneObject | None = None
        self._held_alternate_book_geometry: SceneObject | None = None
        # Preserve the cavity-specific OSC rotation calibration through the
        # carry and placement skill.  ``_active_cavity_rim`` is attempt-local
        # and is intentionally cleared when PICK completes.
        self._held_from_cavity_rim = False
        self._place_destination_geometry: SceneObject | None = None
        self._place_rotation_world: np.ndarray | None = None
        self._shelf_entry_pose: Pose | None = None
        self._place_rotation_delta_world: np.ndarray | None = None
        self._place_expected_settled_center_world: np.ndarray | None = None
        self._place_destination_grounding: str | None = None
        self._prefetched_place_destination: SceneObject | None = None
        self._prefetched_place_target: str | None = None
        self._prefetched_place_selector: EntitySelector | None = None
        self._prefetched_place_relation: str | None = None
        self._prefetched_place_grounding: str | None = None
        self._prefetched_place_destinations: dict[
            int,
            tuple[
                str,
                EntitySelector | None,
                str | None,
                SceneObject,
                str | None,
            ],
        ] = {}
        # Repeated-label commands (for example both moka pots) bind the next
        # still-unmoved instance in the same unobstructed RGB-D view.  The
        # later pick reacquires only near this sensor anchor, preventing the
        # already placed first instance from being selected again.
        self._prefetched_pick_sources: dict[int, SceneObject] = {}
        self._optional_pick_source_anchors: set[int] = set()
        # A continued stack task may move the verified pair only after three
        # independent sensor gates: post-stack height growth, post-lift shape
        # retention, and post-placement shape/location.  The signature keeps
        # the language-bound top/base identities tied to that evidence.
        self._formed_stack_geometry: SceneObject | None = None
        self._formed_stack_top_geometry: SceneObject | None = None
        self._formed_stack_base_geometry: SceneObject | None = None
        self._formed_stack_signature: tuple[
            str, EntitySelector | None, EntitySelector | None
        ] | None = None
        self._active_carry_stack = False
        self._place_destination_observation_stage = "none"
        self._grasp_is_marginal = False
        self._rim_preplace_visual_refreshed = False
        self._visual_place_correction_active = False
        self._pick_source_geometry: SceneObject | None = None
        self._pick_initial_geometry: SceneObject | None = None
        self._drawer_episode_anchors: dict[str, object] = {}
        self._flat_transfer_waypoints: list[Pose] = []
        self._stove_support_geometry = None
        # Optional sensor-only OBB of the fixture used to resolve an IN
        # selector.  Production perception exposes this as
        # ``last_selector_reference``; legacy/test adapters need not do so.
        # Freezing it with the first unobstructed source crop keeps retries
        # deterministic while still allowing a fresh acquisition after a
        # rejected grasp.
        self._pick_cavity_reference: object | None = None
        self._pick_anchor_center_world: np.ndarray | None = None
        self._pick_rotation_anchor_world: np.ndarray | None = None
        self._pick_rotation_anchor_source: str | None = None
        self._pick_start_pose_world: Pose | None = None
        # The sequential wrapper freezes the episode's first public
        # proprioceptive pose before any contact skill can rotate the wrist.
        # Store its position and canonical, world-vertical top-down rotation
        # here; segment reset deliberately clears both until the wrapper
        # re-injects the same episode pose through
        # ``freeze_episode_reset_pose``.  The position is used only to choose
        # which of two antipodal free-space rim points is physically nearer
        # the reset approach.  A preceding drawer skill must not reverse that
        # choice by leaving the wrist on the other side of the bowl.
        self._episode_reset_position_world: np.ndarray | None = None
        self._episode_reset_top_down_rotation_world: np.ndarray | None = None
        self._episode_reset_tool_z_world_up_dot: float | None = None
        self._episode_reset_local_y_planar_norm: float | None = None
        self._pick_selector_continuity = False
        self._pick_selector_reacquire_radius_m: float | None = None
        # A frying pan is a compound body-plus-handle shape whose whole RGB-D
        # centroid is not an antipodal grasp.  These fields retain only
        # current-episode sensor geometry: one solid handle slot per attempt,
        # failed slot identities across the bounded retry, and the observed
        # pan-body reference used to carry the body (not the crop centroid) to
        # its destination.  No task id or simulator state participates.
        self._active_pan_handle_slot_id: str | None = None
        self._failed_pan_handle_slot_ids: set[str] = set()
        self._pan_body_reference_world: np.ndarray | None = None
        # Preserve deliberate-overhead-view provenance so the first grasp
        # keeps the approach-aligned physical rim rather than applying the
        # optional roomy-side switch calibrated for an initial view.  Once
        # accepted, that high view proceeds directly to this far pregrasp.
        self._cavity_reference_from_active_view = False
        # A strict-2D relation is useful identity evidence but is not enough
        # geometry for a cavity motion.  These attempt-local fields implement
        # the safe vertical-then-horizontal active-view reacquisition.
        self._cavity_active_view_source: SceneObject | None = None
        self._cavity_active_view_safe_z_m: float | None = None
        self._cavity_active_view_vertical_pose: Pose | None = None
        self._cavity_active_view_pose: Pose | None = None
        self._cavity_active_view_index = 0
        self._cavity_active_view_trace: dict[str, object] | None = None
        # Freeze the first sensor-selected physical rim direction.  A retry is
        # selected against this vector, rather than against a re-ordered yawed
        # axis whose nominal "second" side can still lie on the same rim.
        self._cavity_primary_radial_world: np.ndarray | None = None
        # A selector reference may exist even when its OBB fails the geometric
        # cavity gate.  Track whether the *current attempt* actually built a
        # cavity rim frame so free-space fallback never inherits cavity-only
        # pre-shape, contact, seat, proof, or retry behaviour.
        self._active_cavity_rim = False
        # A drawer OBB has two unoriented horizontal PCA axes.  Its longer
        # observed side is only the first hypothesis: perspective and an open
        # drawer can make depth appear longer than lateral width.  This index
        # enumerates both axes and their high/low-clearance rim sides.
        self._cavity_candidate_index = 0
        self._cavity_switch_pose: Pose | None = None
        # A free-space rim can still be hidden immediately behind a moved
        # fixture edge (for example, an opened drawer handle).  A blocked
        # vertical descent must first return to a public-proprio high pose
        # with open fingers before the antipodal physical rim may be bound.
        self._free_rim_retract_pose: Pose | None = None
        self._free_rim_retry_terminal_failure: str | None = None
        self._free_rim_retry_anchor_geometry: SceneObject | None = None
        self._free_rim_antipodal_reacquisition_required = False
        self._free_rim_preshape_reseat_required = False
        self._free_rim_gripper_preshaped = False
        self._free_rim_reseat_reacquisition: dict[str, object] | None = None
        # A bilateral open-finger probe can move a free bowl twice.  After
        # the second probe and its verified OPEN high retract, freeze the
        # already identity-bound attempt-one centre as a one-shot association
        # anchor for the third RGB-D crop.  The attempt-zero centre/geometry
        # remains the immutable identity, size, and global-distance anchor.
        self._free_rim_reseat_association_anchor_world: np.ndarray | None = None
        self._free_rim_reseat_association_capture_sequence: int | None = None
        self._free_rim_bilateral_completion_sequence: int | None = None
        self._free_rim_open_high_retract_completion_sequence: int | None = None
        self._cavity_level_pose: Pose | None = None
        self._cavity_seat_pose: Pose | None = None
        self._cavity_gripper_preshaped = False
        self._cavity_lift_start_z_m: float | None = None
        self._cavity_retry_terminal_failure: str | None = None
        self._previous_position_error_m: float | None = None
        self._contact_stall_ticks = 0
        self._contact_error_history: list[float] = []
        self._contact_position_history: list[np.ndarray] = []
        self._grasp_mode = GraspMode.PINCH
        self._grasp_retry_index = 0
        self._grasp_target_attempts: list[dict[str, object]] = []
        self._grasp_verifications: list[dict[str, object]] = []
        self._grasp_attempt_journal = GraspAttemptJournal()
        self._pending_grasp_engagement: _PendingGraspEngagement | None = None
        self._placement_target_attempts: list[dict[str, object]] = []
        self._observation_sequence = 0
        self._observation_timestamp_s = 0.0

    @property
    def task_spec(self) -> TaskSpec | None:
        return self._spec

    @property
    def status(self) -> ExecutorStatus:
        return self._status

    @property
    def grasp_mode(self) -> GraspMode:
        return self._grasp_mode

    @property
    def grasp_target_attempts(self) -> tuple[dict[str, object], ...]:
        """JSON-safe, sensor-derived grasp targets for result provenance."""

        return tuple(dict(item) for item in self._grasp_target_attempts)

    @property
    def grasp_verifications(self) -> tuple[dict[str, object], ...]:
        """Measured finger-width decisions, never simulator contacts."""

        return tuple(dict(item) for item in self._grasp_verifications)

    @property
    def grasp_attempt_events(self) -> tuple[GraspAttemptEvent, ...]:
        """Completed physical jaw engagements in this controller run."""

        return self._grasp_attempt_journal.records

    @property
    def pending_grasp_engagement(self) -> PendingGraspEngagement | None:
        """Issued jaw engagement awaiting a sensor-derived terminal outcome."""

        pending = self._pending_grasp_engagement
        if pending is None:
            return None
        return PendingGraspEngagement(
            attempt_index=len(self._grasp_attempt_journal.records) + 1,
            source_text=pending.source_text,
            source_class=pending.source_class,
            grasp_mode=pending.grasp_mode,
            jaw_behavior=pending.jaw_behavior,
        )

    @property
    def placement_target_attempts(self) -> tuple[dict[str, object], ...]:
        """JSON-safe sensor/proprio placement compensation provenance."""

        return tuple(dict(item) for item in self._placement_target_attempts)

    def freeze_episode_reset_pose(self, reset_pose: Pose) -> None:
        """Freeze a sensor-only canonical top-down frame for this segment.

        Panda's fingers close along tool-local Y.  Project the episode-reset
        local-Y axis onto the support plane, and use only the reset tool-Z
        hemisphere to choose world-up versus world-down.  This removes tilt
        acquired during an earlier drawer/door skill without inventing a
        world-frame yaw or using task, simulator, or evaluator state.
        """

        if self._episode_reset_top_down_rotation_world is not None:
            return
        reset_rotation = np.asarray(reset_pose.rotation, dtype=np.float64)
        world_up = np.array((0.0, 0.0, 1.0), dtype=np.float64)
        jaw_axis = reset_rotation[:, 1].copy()
        jaw_axis -= world_up * float(np.dot(jaw_axis, world_up))
        planar_norm = float(np.linalg.norm(jaw_axis))
        # Keep the same fail-closed planarity threshold enforced immediately
        # before rim target construction.  A vertical/ambiguous jaw axis must
        # never be converted into an arbitrary grasp yaw.
        if planar_norm < 0.5:
            raise ValueError(
                "episode-reset Panda local-Y finger axis is not sufficiently planar"
            )
        jaw_axis /= planar_norm
        tool_z_dot = float(np.dot(reset_rotation[:, 2], world_up))
        tool_z = (1.0 if tool_z_dot >= 0.0 else -1.0) * world_up
        tool_x = np.cross(jaw_axis, tool_z)
        tool_x /= float(np.linalg.norm(tool_x))
        jaw_axis = np.cross(tool_z, tool_x)
        jaw_axis /= float(np.linalg.norm(jaw_axis))
        canonical = np.column_stack((tool_x, jaw_axis, tool_z))
        # Reuse the public boundary type's rigid-rotation validation rather
        # than admitting an approximately orthogonal frame into OSC targets.
        canonical = Pose(np.zeros(3, dtype=np.float64), canonical).rotation
        self._episode_reset_position_world = np.asarray(
            reset_pose.position, dtype=np.float64
        ).copy()
        self._episode_reset_top_down_rotation_world = canonical.copy()
        self._episode_reset_tool_z_world_up_dot = tool_z_dot
        self._episode_reset_local_y_planar_norm = planar_norm

    def clear_episode_state(self) -> None:
        """Clear every episode-local controller and perception value.

        This operation deliberately performs no task compilation or semantic
        activation.  The Route-B dispatcher uses it to clear an inactive
        manipulation delegate without priming that delegate with a synthetic
        task.
        """

        reset_perception = getattr(type(self.perception), "reset", None)
        if callable(reset_perception):
            reset_perception(self.perception)
        self._spec = None
        self._status = ExecutorStatus.IDLE
        self._skill_index = 0
        self._phase = "idle"
        self._phase_ticks = 0
        self._perception_misses = 0
        self._message = ""
        self._motion_pose = None
        self._secondary_pose = None
        self._transfer_clearance_pose = None
        self._transfer_clearance_z_m = None
        self._rim_transfer_rotation_reference = None
        self._held_subject = None
        self._release_width_target_m = None
        self._release_settle_retry_used = False
        self._release_recenter_pose = None
        self._release_peel_pose = None
        self._held_offset_world = np.zeros(3, dtype=np.float64)
        self._held_offset_source = "none"
        self._held_geometry = None
        self._held_alternate_book_geometry = None
        self._held_from_cavity_rim = False
        self._place_destination_geometry = None
        self._place_rotation_world = None
        self._shelf_entry_pose = None
        self._place_rotation_delta_world = None
        self._place_expected_settled_center_world = None
        self._place_destination_grounding = None
        self._prefetched_place_destination = None
        self._prefetched_place_target = None
        self._prefetched_place_selector = None
        self._prefetched_place_relation = None
        self._prefetched_place_grounding = None
        self._prefetched_place_destinations = {}
        self._prefetched_pick_sources = {}
        self._optional_pick_source_anchors = set()
        self._formed_stack_geometry = None
        self._formed_stack_top_geometry = None
        self._formed_stack_base_geometry = None
        self._formed_stack_signature = None
        self._active_carry_stack = False
        self._place_destination_observation_stage = "none"
        self._grasp_is_marginal = False
        self._rim_preplace_visual_refreshed = False
        self._visual_place_correction_active = False
        self._pick_source_geometry = None
        self._pick_initial_geometry = None
        self._drawer_episode_anchors.clear()
        self._flat_transfer_waypoints = []
        self._stove_support_geometry = None
        self._pick_cavity_reference = None
        self._pick_anchor_center_world = None
        self._pick_rotation_anchor_world = None
        self._pick_rotation_anchor_source = None
        self._pick_start_pose_world = None
        self._episode_reset_position_world = None
        self._episode_reset_top_down_rotation_world = None
        self._episode_reset_tool_z_world_up_dot = None
        self._episode_reset_local_y_planar_norm = None
        self._pick_selector_continuity = False
        self._pick_selector_reacquire_radius_m = None
        self._active_pan_handle_slot_id = None
        self._failed_pan_handle_slot_ids = set()
        self._pan_body_reference_world = None
        self._cavity_reference_from_active_view = False
        self._clear_cavity_active_view_state()
        self._cavity_primary_radial_world = None
        self._active_cavity_rim = False
        self._cavity_candidate_index = 0
        self._cavity_switch_pose = None
        self._free_rim_retract_pose = None
        self._free_rim_retry_terminal_failure = None
        self._free_rim_retry_anchor_geometry = None
        self._free_rim_antipodal_reacquisition_required = False
        self._free_rim_preshape_reseat_required = False
        self._free_rim_gripper_preshaped = False
        self._free_rim_reseat_reacquisition = None
        self._free_rim_reseat_association_anchor_world = None
        self._free_rim_reseat_association_capture_sequence = None
        self._free_rim_bilateral_completion_sequence = None
        self._free_rim_open_high_retract_completion_sequence = None
        self._cavity_level_pose = None
        self._cavity_seat_pose = None
        self._cavity_gripper_preshaped = False
        self._cavity_lift_start_z_m = None
        self._cavity_retry_terminal_failure = None
        self._previous_position_error_m = None
        self._contact_stall_ticks = 0
        self._contact_error_history = []
        self._contact_position_history = []
        self._grasp_mode = GraspMode.PINCH
        self._grasp_retry_index = 0
        self._grasp_target_attempts = []
        self._grasp_verifications = []
        self._grasp_attempt_journal.reset()
        self._pending_grasp_engagement = None
        self._placement_target_attempts = []
        self._observation_sequence = 0
        self._observation_timestamp_s = 0.0
    def activate_after_episode_clear(self, task: str | TaskSpec) -> TaskSpec:
        """Activate a real task after one explicit neutral episode clear."""

        if (
            self._spec is not None
            or self._status is not ExecutorStatus.IDLE
            or self._phase != "idle"
            or self._phase_ticks != 0
        ):
            raise RuntimeError(
                "Route B controller activation requires freshly cleared state"
            )
        spec = (
            type(self.compiler).compile(self.compiler, task)
            if isinstance(task, str)
            else task
        )
        self._spec = spec
        self._status = ExecutorStatus.RUNNING
        self._phase = "acquire"
        return spec

    def reset(self, task: str | TaskSpec) -> TaskSpec:
        """Clear once, then activate the supplied real task."""

        self.clear_episode_state()
        return self.activate_after_episode_clear(task)

    def act(self, observation: SensorObservation) -> ControlDecision:
        # This is an episode-local provenance token owned by the controller,
        # not an environment clock or a field in the policy observation.
        self._observation_timestamp_s = float(self._observation_sequence)
        self._observation_sequence += 1
        begin_observation = getattr(self.perception, "begin_observation", None)
        if callable(begin_observation):
            begin_observation(observation, int(self._observation_timestamp_s))
        if self._spec is None or self._status is ExecutorStatus.IDLE:
            return self._decision(self._hold_action(GRIPPER_OPEN), "controller has not been reset")
        if self._status in {ExecutorStatus.SUCCEEDED, ExecutorStatus.FAILED}:
            return self._decision(
                self._hold_action(self._current_hold_command()), self._message
            )
        if self._skill_index >= len(self._spec.steps):
            self._status = ExecutorStatus.SUCCEEDED
            self._phase = "done"
            self._message = "task skill graph completed"
            return self._decision(
                self._hold_action(self._current_hold_command()), self._message
            )

        step = self._spec.steps[self._skill_index]
        if step.kind not in EXECUTABLE_GRASP_PLACE_SKILLS:
            return self._fail(
                f"skill {step.kind.value!r} is compiled but not implemented "
                "by the grasp/place executor"
            )
        if self._phase_ticks > self.config.phase_timeout_ticks:
            target = self._motion_pose
            if self._phase in {"move_pregrasp", "move_preplace", "retreat"}:
                target = self._secondary_pose
            elif self._phase == "move_transfer_clearance":
                target = self._transfer_clearance_pose
            records = (self._grasp_target_attempts if step.kind is SkillKind.PICK
                       else self._placement_target_attempts)
            if target is not None and records:
                records[-1]["timeout_motion"] = {
                    "phase": self._phase,
                    "current_world_m": observation.robot.ee_pose.position.tolist(),
                    "target_world_m": target.position.tolist(),
                    "current_rotation_world": observation.robot.ee_pose.rotation.tolist(),
                    "target_rotation_world": target.rotation.tolist(),
                    "rotation_error_rad": float(np.linalg.norm(self._rotation_vector(
                        target.rotation @ observation.robot.ee_pose.rotation.T))),
                    "gripper_width_m": float(observation.robot.gripper_width_m),
                }
            return self._fail(f"phase {self._phase!r} timed out")

        if step.kind is SkillKind.PICK:
            return self._act_pick(step, observation)
        return self._act_place(step, observation)

    def _act_pick(self, step: SkillStep, observation: SensorObservation) -> ControlDecision:
        if self._phase in {
            "cavity_active_view_vertical",
            "cavity_active_view_traverse",
        }:
            return self._act_cavity_active_view(step, observation)

        if self._phase == "free_rim_retract":
            retract_target = self._free_rim_retract_pose
            if self._free_space_rim_high_retract_reached(
                observation.robot.ee_pose,
                retract_target,
                observation.robot.gripper_width_m,
            ):
                if self._free_rim_preshape_reseat_required:
                    # Arm exactly one contact-conditioned association hop.
                    # Attempt one was itself accepted against immutable P0;
                    # only a proven bilateral OPEN outcome followed by this
                    # measured open/high retract may promote its centre P1 to
                    # the query anchor for the third RGB-D frame.
                    attempt1 = (
                        self._grasp_target_attempts[-1]
                        if self._grasp_target_attempts
                        else None
                    )
                    bilateral = (
                        attempt1.get("free_space_rim_bilateral_contact")
                        if isinstance(attempt1, dict)
                        else None
                    )
                    bilateral_completion = (
                        bilateral.get("completion")
                        if isinstance(bilateral, dict)
                        else None
                    )
                    bilateral_authorization = (
                        bilateral.get("authorization")
                        if isinstance(bilateral, dict)
                        else None
                    )
                    retract_trace = (
                        attempt1.get("free_space_rim_approach_retract")
                        if isinstance(attempt1, dict)
                        else None
                    )
                    retract_completion = (
                        retract_trace.get("completion")
                        if isinstance(retract_trace, dict)
                        else None
                    )
                    retract_authorization = (
                        retract_trace.get("authorization")
                        if isinstance(retract_trace, dict)
                        else None
                    )
                    try:
                        association_anchor = np.asarray(
                            attempt1.get("sensor_center_world_m")
                            if isinstance(attempt1, dict)
                            else None,
                            dtype=np.float64,
                        )
                    except (TypeError, ValueError):
                        association_anchor = np.empty(0, dtype=np.float64)
                    association_capture = (
                        attempt1.get("capture_sequence")
                        if isinstance(attempt1, dict)
                        else None
                    )
                    attempt0 = (
                        self._grasp_target_attempts[-2]
                        if len(self._grasp_target_attempts) >= 2
                        else None
                    )
                    attempt0_capture = (
                        attempt0.get("capture_sequence")
                        if isinstance(attempt0, dict)
                        else None
                    )
                    first_fallback = (
                        attempt0.get("free_space_rim_approach_fallback")
                        if isinstance(attempt0, dict)
                        else None
                    )
                    first_fallback_capture = (
                        first_fallback.get("capture_sequence")
                        if isinstance(first_fallback, dict)
                        else None
                    )
                    first_records = (
                        attempt0.get("free_space_rim_retry_reacquisition")
                        if isinstance(attempt0, dict)
                        else None
                    )
                    first_record: dict[str, object] | None = None
                    if isinstance(first_records, list):
                        for item in reversed(first_records):
                            if (
                                isinstance(item, dict)
                                and item.get("capture_sequence")
                                == association_capture
                            ):
                                first_record = item
                                break

                    def arm_vector(value: object) -> np.ndarray | None:
                        try:
                            result = np.asarray(value, dtype=np.float64)
                        except (TypeError, ValueError):
                            return None
                        if result.shape != (3,) or not np.all(np.isfinite(result)):
                            return None
                        return result

                    initial_anchor = arm_vector(
                        attempt0.get("sensor_center_world_m")
                        if isinstance(attempt0, dict)
                        else None
                    )
                    initial_extents = arm_vector(
                        attempt0.get("sensor_extents_m")
                        if isinstance(attempt0, dict)
                        else None
                    )
                    attempt1_extents = arm_vector(
                        attempt1.get("sensor_extents_m")
                        if isinstance(attempt1, dict)
                        else None
                    )
                    first_candidate = arm_vector(
                        first_record.get("candidate_center_world_m")
                        if isinstance(first_record, dict)
                        else None
                    )
                    first_anchor = arm_vector(
                        first_record.get("anchor_center_world_m")
                        if isinstance(first_record, dict)
                        else None
                    )
                    first_anchor_extents = arm_vector(
                        first_record.get("anchor_extents_sorted_m")
                        if isinstance(first_record, dict)
                        else None
                    )
                    first_candidate_extents = arm_vector(
                        first_record.get("candidate_extents_sorted_m")
                        if isinstance(first_record, dict)
                        else None
                    )
                    first_scales = arm_vector(
                        first_record.get("candidate_extent_scales")
                        if isinstance(first_record, dict)
                        else None
                    )
                    first_distance = (
                        float(np.linalg.norm(association_anchor - initial_anchor))
                        if association_anchor.shape == (3,)
                        and initial_anchor is not None
                        else None
                    )
                    first_recorded_distance = (
                        first_record.get("nearest_bound_distance_m")
                        if isinstance(first_record, dict)
                        else None
                    )
                    first_recorded_radius = (
                        first_record.get("anchor_radius_m")
                        if isinstance(first_record, dict)
                        else None
                    )
                    first_binding_ready = bool(
                        isinstance(attempt0, dict)
                        and type(attempt0.get("attempt")) is int
                        and attempt0.get("attempt") == 0
                        and attempt0.get("rim_strategy") == "free_space"
                        and attempt0.get("active_cavity_rim") is False
                        and isinstance(first_record, dict)
                        and type(first_record.get("attempt")) is int
                        and first_record.get("attempt") == 1
                        and type(first_record.get("capture_sequence")) is int
                        and first_record.get("capture_sequence")
                        == association_capture
                        and type(attempt0_capture) is int
                        and type(first_fallback_capture) is int
                        and type(association_capture) is int
                        and 0 <= attempt0_capture
                        < first_fallback_capture
                        < association_capture
                        and first_record.get("nearest_bound_result") == "accepted"
                        and first_record.get("selected_source")
                        == "fresh_nearest_bound_attempt0_anchor"
                        and first_record.get("size_continuous") is True
                        and initial_anchor is not None
                        and first_anchor is not None
                        and np.allclose(first_anchor, initial_anchor, rtol=0.0, atol=1e-9)
                        and first_candidate is not None
                        and np.allclose(first_candidate, association_anchor, rtol=0.0, atol=1e-9)
                        and first_distance is not None
                        and isinstance(first_recorded_distance, (int, float))
                        and np.isfinite(first_recorded_distance)
                        and abs(float(first_recorded_distance) - first_distance)
                        <= 1e-9
                        and isinstance(first_recorded_radius, (int, float))
                        and np.isfinite(first_recorded_radius)
                        and self._pick_selector_reacquire_radius_m is not None
                        and abs(
                            float(first_recorded_radius)
                            - self._pick_selector_reacquire_radius_m
                        )
                        <= 1e-9
                        and first_distance <= float(first_recorded_radius) + 1e-12
                        and initial_extents is not None
                        and attempt1_extents is not None
                        and first_anchor_extents is not None
                        and first_candidate_extents is not None
                        and first_scales is not None
                        and np.allclose(
                            first_anchor_extents,
                            np.sort(initial_extents),
                            rtol=0.0,
                            atol=1e-9,
                        )
                        and np.allclose(
                            first_candidate_extents,
                            np.sort(attempt1_extents),
                            rtol=0.0,
                            atol=1e-9,
                        )
                        and np.allclose(
                            first_scales,
                            first_candidate_extents / first_anchor_extents,
                            rtol=0.0,
                            atol=1e-9,
                        )
                        and np.all(
                            first_scales
                            >= self.config.free_space_rim_retry_min_extent_scale
                            - 1e-12
                        )
                        and np.all(
                            first_scales
                            <= self.config.free_space_rim_retry_max_extent_scale
                            + 1e-12
                        )
                    )
                    bilateral_sequence = (
                        bilateral.get("bilateral_completion_sequence")
                        if isinstance(bilateral, dict)
                        else None
                    )
                    nested_bilateral_sequence = (
                        bilateral_completion.get(
                            "bilateral_completion_sequence"
                        )
                        if isinstance(bilateral_completion, dict)
                        else None
                    )
                    bilateral_fresh_sequence = (
                        bilateral_completion.get(
                            "fresh_reacquisition_capture_sequence"
                        )
                        if isinstance(bilateral_completion, dict)
                        else None
                    )
                    retract_sequence = int(self._observation_timestamp_s)
                    association_ready = bool(
                        first_binding_ready
                        and isinstance(attempt1, dict)
                        and type(attempt1.get("attempt")) is int
                        and attempt1.get("attempt") == 1
                        and attempt1.get("rim_strategy") == "free_space"
                        and attempt1.get("active_cavity_rim") is False
                        and association_anchor.shape == (3,)
                        and np.all(np.isfinite(association_anchor))
                        and type(association_capture) is int
                        and type(bilateral_sequence) is int
                        and type(nested_bilateral_sequence) is int
                        and type(bilateral_fresh_sequence) is int
                        and type(self._free_rim_bilateral_completion_sequence)
                        is int
                        and bilateral_sequence
                        == self._free_rim_bilateral_completion_sequence
                        and association_capture
                        < bilateral_sequence
                        < retract_sequence
                        and isinstance(bilateral_completion, dict)
                        and bilateral_completion.get(
                            "preshape_reseat_authorized"
                        )
                        is True
                        and bilateral_completion.get("close_authorized") is False
                        and bilateral_completion.get(
                            "bilateral_completion_sequence"
                        )
                        == bilateral_sequence
                        and nested_bilateral_sequence == bilateral_sequence
                        and bilateral_fresh_sequence == association_capture
                        and isinstance(bilateral_authorization, dict)
                        and bilateral_authorization.get("close") is False
                        and bilateral_authorization.get("task_success") is False
                        and bilateral_authorization.get(
                            "open_high_retract_then_fresh_preshape_reseat"
                        )
                        is True
                        and isinstance(retract_completion, dict)
                        and retract_completion.get("safe_retract_reached") is True
                        and retract_completion.get("reacquire_authorized") is True
                        and isinstance(retract_authorization, dict)
                        and retract_authorization.get("close") is False
                        and retract_authorization.get("task_success") is False
                        and retract_authorization.get(
                            "fresh_antipodal_reacquisition"
                        )
                        is True
                    )
                    if association_ready:
                        self._free_rim_reseat_association_anchor_world = (
                            association_anchor.copy()
                        )
                        self._free_rim_reseat_association_capture_sequence = (
                            association_capture
                        )
                        self._free_rim_open_high_retract_completion_sequence = (
                            retract_sequence
                        )
                        assert isinstance(retract_trace, dict)
                        assert isinstance(retract_completion, dict)
                        retract_trace["open_high_retract_completion_sequence"] = (
                            retract_sequence
                        )
                        retract_completion[
                            "open_high_retract_completion_sequence"
                        ] = retract_sequence
                    else:
                        self._free_rim_retry_terminal_failure = (
                            "free-space rim reseat lacks a typed bilateral/open-high "
                            "association chain"
                        )
                if self._free_rim_retry_terminal_failure is not None:
                    return self._fail(self._free_rim_retry_terminal_failure)
                invalidate = getattr(self.perception, "invalidate_dynamic", None)
                if callable(invalidate):
                    invalidate(step.subject)
                self._pick_source_geometry = None
                self._motion_pose = None
                self._secondary_pose = None
                self._free_rim_retract_pose = None
                self._perception_misses = 0
                self._set_phase("acquire")
                return self._tick_decision(
                    self._hold_action(GRIPPER_OPEN),
                    "reacquiring the antipodal physical rim after typed "
                    "open-gripper high retraction",
                )
            if retract_target is None:
                return self._fail(
                    "free-space rim retry lacks a public-proprio high target"
                )
            return self._motion(retract_target, observation, GRIPPER_OPEN)

        if self._phase == "retry_release":
            if self._phase_ticks >= self.config.grasp_retry_release_ticks:
                invalidate = getattr(self.perception, "invalidate_dynamic", None)
                if callable(invalidate):
                    invalidate(step.subject)
                self._pick_source_geometry = None
                # ON/IN selects one physical instance.  Preserve that first
                # sensor association and its fixture OBB across empty-grasp
                # retries; otherwise a fresh global selector can jump to a
                # different same-label object.  Unqualified picks retain the
                # historical full reacquisition behaviour.
                if not self._pick_selector_continuity:
                    self._pick_cavity_reference = None
                    self._cavity_reference_from_active_view = False
                    self._pick_anchor_center_world = None
                    self._pick_selector_reacquire_radius_m = None
                self._motion_pose = None
                self._secondary_pose = None
                self._cavity_switch_pose = None
                self._cavity_level_pose = None
                self._cavity_seat_pose = None
                self._cavity_gripper_preshaped = False
                self._cavity_lift_start_z_m = None
                self._perception_misses = 0
                self._set_phase("acquire")
            else:
                return self._tick_decision(
                    self._hold_action(self._released_gripper_command()),
                    f"releasing before sensor-only grasp retry {self._grasp_retry_index}",
                )

        if self._phase == "cavity_retry_retract":
            retract_target = self._secondary_pose
            if self._cavity_retry_high_retract_reached(
                observation.robot.ee_pose,
                retract_target,
                observation.robot.gripper_width_m,
            ):
                if self._cavity_retry_terminal_failure is not None:
                    return self._fail(self._cavity_retry_terminal_failure)
                invalidate = getattr(self.perception, "invalidate_dynamic", None)
                if callable(invalidate):
                    invalidate(step.subject)
                self._pick_source_geometry = None
                self._motion_pose = None
                self._secondary_pose = None
                self._perception_misses = 0
                self._cavity_level_pose = None
                self._cavity_seat_pose = None
                self._set_phase("acquire")
                return self._tick_decision(
                    self._hold_action(GRIPPER_OPEN),
                    "reacquiring after typed open-gripper cavity retraction",
                )
            if retract_target is None:
                return self._fail(
                    "cavity retry retraction lacks a sensor-derived high target"
                )
            return self._motion(
                retract_target,
                observation,
                GRIPPER_OPEN,
            )

        if self._phase == "acquire":
            snapshot = self._observe_pick_subject(step, observation)
            source = snapshot.objects.get(step.subject)
            if source is None:
                qualifier = (
                    " near the initially selected instance"
                    if self._pick_selector_continuity
                    and self._grasp_retry_index > 0
                    else ""
                )
                return self._perception_miss(
                    f"cannot localize pick object {step.subject!r}{qualifier}",
                    gripper=self._released_gripper_command(),
                )
            try:
                self._grasp_mode = self._sensor_refined_grasp_mode(step, source)
            except ValueError as exc:
                return self._fail(str(exc))
            self._perception_misses = 0
            source = self._stabilize_cavity_retry_source(source)
            self._pick_source_geometry = source
            if self._pick_rotation_anchor_world is None:
                # PCA axes have no sign.  Freeze the first unobstructed wrist
                # frame so every cavity hypothesis chooses the same equivalent
                # yaw hemisphere; canonicalizing from the previous candidate
                # can flip an orthogonal axis by pi into a joint-limited pose.
                if self._episode_reset_top_down_rotation_world is not None:
                    self._pick_rotation_anchor_world = (
                        self._episode_reset_top_down_rotation_world.copy()
                    )
                    self._pick_rotation_anchor_source = (
                        "episode_reset_top_down_from_public_proprioception"
                    )
                else:
                    self._pick_rotation_anchor_world = (
                        observation.robot.ee_pose.rotation.copy()
                    )
                    self._pick_rotation_anchor_source = (
                        "first_pick_public_proprioception"
                    )
            if self._pick_start_pose_world is None:
                # The canonical rotation is episode-global, but active-view
                # approach geometry must still start at this pick's live,
                # post-retreat public proprioceptive position.
                self._pick_start_pose_world = observation.robot.ee_pose
            first_selector_binding = self._pick_anchor_center_world is None
            if first_selector_binding:
                self._pick_selector_continuity = self._selector_binds_pick_identity(step)
                self._pick_anchor_center_world = source.centroid_world.copy()
                if self._pick_selector_continuity:
                    self._pick_selector_reacquire_radius_m = (
                        self._selector_reacquire_radius(source)
                    )
            elif (
                not self._pick_selector_continuity
                and not self._free_rim_antipodal_reacquisition_required
            ):
                self._pick_anchor_center_world = source.centroid_world.copy()
            if self._pick_cavity_reference is None:
                self._pick_cavity_reference = self._selector_cavity_reference(step)
            self._prefetch_next_place_destination(observation)
            self._prefetch_future_ranked_place_destinations(observation)
            self._prefetch_future_repeated_pick(step, observation)
            if (
                self._grasp_mode is GraspMode.RIM_PINCH
                and self._pick_cavity_reference is None
                and self._selector_requires_cavity_reference(step)
            ):
                return self._begin_cavity_active_view(
                    step,
                    source,
                    observation,
                )
            try:
                self._set_pick_targets(source, observation)
            except ValueError as exc:
                return self._fail(
                    f"invalid sensor grasp geometry for {step.subject!r}: {exc}"
                )
            self._set_phase("move_pregrasp")

        elif (
            self._phase in {"move_pregrasp", "approach"}
            and self._should_refresh_visual()
            and self._grasp_mode is GraspMode.PINCH
            and not self._is_pan_handle_subject(step.subject)
        ):
            # During an ordinary centre-pinch approach, partial hand pixels
            # can shift the component centre.  Accept only small updates
            # relative to the first unobstructed sensor anchor.  Rim and
            # expansion targets freeze their first unobstructed geometry:
            # even a small centre drift can move a narrow rim between the
            # wrong pair of finger surfaces.  Retries deliberately invalidate
            # and reacquire either target.
            snapshot = self._observe_pick_subject(step, observation)
            source = snapshot.objects.get(step.subject)
            if source is not None and self._pick_refresh_is_consistent(source):
                self._pick_source_geometry = source
                self._set_pick_targets(source, observation, retain_phase=True)

        if self._phase == "cavity_retract":
            assert self._cavity_switch_pose is not None
            # Reverse a blocked approach to its original pregrasp with the
            # hand fully released.  The old candidate's partial-width state
            # must not issue a CLOSE while escaping the fixture.  A freshly
            # acquired candidate starts its own explicit ``cavity_preshape``
            # phase only after reaching its new high pregrasp.
            if self._cavity_approach_high_retract_reached(
                observation.robot.ee_pose,
                self._cavity_switch_pose,
                observation.robot.gripper_width_m,
            ):
                if self._cavity_retry_terminal_failure is not None:
                    return self._fail(self._cavity_retry_terminal_failure)
                # Re-observe after the full vertical escape.  A rejected wedge
                # can nudge the bowl, so the second local-yaw candidate must
                # not silently reuse stale pre-contact geometry.
                invalidate = getattr(self.perception, "invalidate_dynamic", None)
                if callable(invalidate):
                    invalidate(step.subject)
                self._pick_source_geometry = None
                self._motion_pose = None
                self._secondary_pose = None
                self._cavity_switch_pose = None
                self._cavity_level_pose = None
                self._cavity_seat_pose = None
                self._perception_misses = 0
                self._grasp_retry_index += 1
                self._set_phase("acquire")
                return self._tick_decision(
                    self._hold_action(GRIPPER_OPEN),
                    "reacquiring selected cavity yaw candidate after typed "
                    "open-gripper retraction",
                )
            else:
                return self._motion(
                    self._cavity_switch_pose,
                    observation,
                    GRIPPER_OPEN,
                )

        if self._phase == "move_pregrasp":
            assert self._secondary_pose is not None
            while self._flat_transfer_waypoints:
                waypoint = self._flat_transfer_waypoints[0]
                if self._pose_reached(observation.robot.ee_pose, waypoint):
                    self._flat_transfer_waypoints.pop(0)
                else:
                    return self._motion(waypoint, observation, self._released_gripper_command())
            # A loose pregrasp acceptance leaves several millimetres of
            # lateral / angular error to be corrected during the vertical
            # descent.  On flat packages that cross-coupling makes one open
            # finger touch the table and the OSC controller stalls sideways.
            pregrasp_reached = (
                self._cavity_high_pregrasp_reached(
                    observation.robot.ee_pose,
                    self._secondary_pose,
                )
                if self._grasp_mode is GraspMode.RIM_PINCH
                and self._active_cavity_rim
                else self._pose_reached(
                    observation.robot.ee_pose,
                    self._secondary_pose,
                    position_tolerance_m=self.config.grasp_position_tolerance_m,
                )
            )
            if pregrasp_reached:
                    self._grasp_target_attempts[-1]["post_contact_staging_residual_m"] = float(
                        np.linalg.norm(observation.robot.ee_pose.position - self._secondary_pose.position))
            if pregrasp_reached:
                pinch_aperture = self._flat_pinch_approach_aperture()
                if (
                    pinch_aperture is not None
                    and observation.robot.gripper_width_m > pinch_aperture + 0.003
                ):
                    return self._motion(
                        self._secondary_pose, observation, GRIPPER_CLOSE
                    )
                if (
                    self._grasp_mode is GraspMode.RIM_PINCH
                    and self._active_cavity_rim
                ):
                    self._record_cavity_preshape_start(
                        observation.robot.gripper_width_m
                    )
                    self._set_phase("cavity_preshape")
                elif (
                    self._grasp_mode is GraspMode.RIM_PINCH
                    and not self._active_cavity_rim
                    and self._free_rim_preshape_reseat_required
                ):
                    self._record_free_space_rim_preshape_start(
                        observation.robot.gripper_width_m
                    )
                    self._set_phase("free_rim_preshape")
                else:
                    self._set_phase("approach")
            else:
                gripper_command = self._released_gripper_command()
                if (
                    self._grasp_mode is GraspMode.RIM_PINCH
                    and self._active_cavity_rim
                    and self._cavity_gripper_preshaped
                ):
                    gripper_command = self._cavity_rim_preshape_command(
                        observation.robot.gripper_width_m
                    )
                pregrasp_target = self._secondary_pose
                current_position = observation.robot.ee_pose.position
                if (
                    self._grasp_mode is GraspMode.PINCH
                    and self._skill_index > 0
                    and current_position[2] > pregrasp_target.position[2] + .025
                    and np.linalg.norm(current_position[:2] - pregrasp_target.position[:2]) > .012
                ):
                    # Leave a raised receptacle horizontally before lowering
                    # toward the next tabletop object in a compound command.
                    high_position = pregrasp_target.position.copy()
                    high_position[2] = current_position[2]
                    pregrasp_target = Pose(high_position, pregrasp_target.rotation)
                return self._motion(pregrasp_target, observation, gripper_command)

        if self._phase == "cavity_preshape":
            assert self._secondary_pose is not None
            preshape_command = self._cavity_rim_preshape_command(
                observation.robot.gripper_width_m
            )
            self._record_cavity_preshape_sample(
                "preshape",
                observation.robot.gripper_width_m,
                preshape_command,
            )
            if self._cavity_rim_preshape_ready(
                observation.robot.gripper_width_m
            ):
                self._cavity_gripper_preshaped = True
                self._record_cavity_preshape_completion(
                    observation.robot.gripper_width_m
                )
                if self._cavity_level_pose is not None:
                    if self._grasp_target_attempts:
                        trace = self._grasp_target_attempts[-1].get(
                            "cavity_tilted_pregrasp"
                        )
                        if isinstance(trace, dict):
                            trace["level_transition"] = (
                                "combined_with_vertical_grasp_descent"
                            )
                            trace["level_transition_width_m"] = float(
                                observation.robot.gripper_width_m
                            )
                    # The tilted pose is only the collision-free high
                    # waypoint.  Mirror Route C's proven mechanics: command
                    # the level low grasp pose while descending, rather than
                    # trying to rotate in place against the fixture.  Retain
                    # the level high pose for proof-lift / safe retraction.
                    self._secondary_pose = self._cavity_level_pose
                    self._cavity_level_pose = None
                self._set_phase("approach")
            else:
                return self._motion(
                    self._secondary_pose,
                    observation,
                    preshape_command,
                )

        if self._phase == "free_rim_preshape":
            assert self._secondary_pose is not None
            preshape_command = self._free_space_rim_preshape_command(
                observation.robot.gripper_width_m
            )
            self._record_free_space_rim_preshape_sample(
                "preshape",
                observation.robot.gripper_width_m,
                preshape_command,
            )
            if self._free_space_rim_preshape_ready(
                observation.robot.gripper_width_m
            ):
                self._free_rim_gripper_preshaped = True
                self._record_free_space_rim_preshape_completion(
                    observation.robot.gripper_width_m
                )
                self._set_phase("approach")
            else:
                return self._motion(
                    self._secondary_pose,
                    observation,
                    preshape_command,
                )

        if self._phase == "approach":
            assert self._motion_pose is not None
            pose_reached = self._pose_reached(
                observation.robot.ee_pose,
                self._motion_pose,
                position_tolerance_m=(
                    self.config.cavity_grasp_position_tolerance_m
                    if self._grasp_mode is GraspMode.RIM_PINCH
                    and self._active_cavity_rim
                    else self.config.grasp_position_tolerance_m
                ),
            )
            generic_contact_reached = self._contact_reached(
                observation.robot.ee_pose,
                self._motion_pose,
                tolerance_m=(
                    self.config.expand_contact_position_tolerance_m
                    if self._grasp_mode is GraspMode.EXPAND
                    else self.config.grasp_contact_position_tolerance_m
                ),
            )
            # Still call the generic checker above so its proprioceptive
            # progress window is populated, but never let its isotropic 12-mm
            # gate close a cavity or exterior rim grasp.  A one-finger / wall
            # collision can have a small total residual while remaining off
            # the commanded bilateral seat.
            is_cavity_rim = bool(
                self._grasp_mode is GraspMode.RIM_PINCH
                and self._active_cavity_rim
            )
            is_exterior_rim = bool(
                self._grasp_mode is GraspMode.RIM_PINCH
                and not self._active_cavity_rim
            )
            exterior_rim_width_open = bool(
                np.isfinite(observation.robot.gripper_width_m)
                and observation.robot.gripper_width_m
                >= self.config.gripper_open_width_m
            )
            # An exterior bowl rim is a thin, two-sided contact affordance.
            # The generic isotropic stall gate cannot distinguish one finger
            # touching the lip from both pads straddling it.  An exact
            # Cartesian target may still close normally; every non-exact
            # exterior-rim stall must instead pass the typed open-retract /
            # fresh-antipode path below (and, on the second side, the strict
            # bilateral seat gate).
            contact_reached = bool(
                generic_contact_reached
                and not is_cavity_rim
                and not is_exterior_rim
            )
            pose_reached_for_close = bool(
                pose_reached
                and (
                    not is_exterior_rim
                    or (
                        exterior_rim_width_open
                        and not self._free_rim_preshape_reseat_required
                    )
                )
            )
            cavity_contact_reached = self._cavity_rim_contact_reached(
                observation.robot.ee_pose,
                self._motion_pose,
            )
            bilateral_free_rim_contact_reached = (
                self._free_space_rim_bilateral_contact_reached(
                    observation.robot.ee_pose,
                    self._motion_pose,
                    observation.robot.gripper_width_m,
                )
            )
            preshaped_free_rim_close_reached = (
                self._free_space_rim_preshape_close_reached(
                    observation.robot.ee_pose,
                    self._motion_pose,
                    observation.robot.gripper_width_m,
                )
            )
            if bilateral_free_rim_contact_reached:
                # Two open-hand stalls can only authorize a clean restart.
                # They are not pad/rim contact and must never issue CLOSE.
                assert self._secondary_pose is not None
                self._free_rim_preshape_reseat_required = True
                self._free_rim_gripper_preshaped = False
                self._free_rim_reseat_reacquisition = None
                self._free_rim_reseat_association_anchor_world = None
                self._free_rim_reseat_association_capture_sequence = None
                self._free_rim_open_high_retract_completion_sequence = None
                bilateral_completion_sequence = int(
                    self._observation_timestamp_s
                )
                self._free_rim_bilateral_completion_sequence = (
                    bilateral_completion_sequence
                )
                retract_position = observation.robot.ee_pose.position.copy()
                retract_position[2] = max(
                    float(retract_position[2]),
                    float(self._secondary_pose.position[2]),
                )
                self._free_rim_retract_pose = Pose(
                    retract_position,
                    observation.robot.ee_pose.rotation.copy(),
                )
                bilateral_trace = self._grasp_target_attempts[-1].get(
                    "free_space_rim_bilateral_contact"
                )
                if isinstance(bilateral_trace, dict):
                    bilateral_trace["bilateral_completion_sequence"] = (
                        bilateral_completion_sequence
                    )
                    completion = bilateral_trace.get("completion")
                    if isinstance(completion, dict):
                        completion["bilateral_completion_sequence"] = (
                            bilateral_completion_sequence
                        )
                    bilateral_trace["retract_start_world_m"] = (
                        observation.robot.ee_pose.position.tolist()
                    )
                    bilateral_trace["retract_target_world_m"] = (
                        self._free_rim_retract_pose.position.tolist()
                    )
                self._set_phase("free_rim_retract")
                return self._motion(
                    self._free_rim_retract_pose,
                    observation,
                    GRIPPER_OPEN,
                )
            if (
                pose_reached_for_close
                or contact_reached
                or cavity_contact_reached
                or preshaped_free_rim_close_reached
            ):
                if is_cavity_rim:
                    self._record_cavity_contact_completion(
                        observation.robot.ee_pose,
                        self._motion_pose,
                        observation.robot.gripper_width_m,
                        reason=(
                            "pose_tolerance"
                            if pose_reached
                            else "directional_proprioceptive_stall"
                        ),
                    )
                self._set_phase("close")
            elif self._free_space_rim_preshape_reseat_should_fail(
                observation.robot.ee_pose,
                self._motion_pose,
                observation.robot.gripper_width_m,
            ):
                assert self._secondary_pose is not None
                self._free_rim_retry_terminal_failure = (
                    "fresh preshaped free-space rim re-entry remained blocked "
                    "outside the exact Panda pad work band"
                )
                retract_position = observation.robot.ee_pose.position.copy()
                retract_position[2] = max(
                    float(retract_position[2]),
                    float(self._secondary_pose.position[2]),
                )
                self._free_rim_retract_pose = Pose(
                    retract_position,
                    observation.robot.ee_pose.rotation.copy(),
                )
                self._set_phase("free_rim_retract")
                return self._motion(
                    self._free_rim_retract_pose,
                    observation,
                    GRIPPER_OPEN,
                )
            elif self._cavity_approach_should_fallback(
                observation.robot.ee_pose,
                self._motion_pose,
            ):
                self._record_cavity_approach_fallback(
                    observation.robot.ee_pose,
                    self._motion_pose,
                    observation.robot.gripper_width_m,
                )
                if (
                    self._cavity_candidate_index
                    >= self._maximum_cavity_candidate_index()
                ):
                    fallback = self._grasp_target_attempts[-1].get(
                        "cavity_approach_fallback"
                    )
                    if isinstance(fallback, dict):
                        fallback["next_candidate_index"] = None
                        fallback["recovery"] = (
                            "stop_after_local_yaw_candidates_exhausted"
                        )
                    self._cavity_retry_terminal_failure = (
                        "width-controlled cavity rim approach remained blocked "
                        "before safe contact"
                    )
                else:
                    self._cavity_candidate_index = (
                        self._next_cavity_candidate_after_approach_block()
                    )
                assert self._secondary_pose is not None
                # Escape along world Z from the measured blocked pose.  A
                # simultaneous return to the old high-pose XY / wrist
                # orientation couples translation and rotation in OSC and can
                # settle outside the otherwise strict 8-mm reacquisition
                # gate.  Neither lateral motion nor wrist reorientation is
                # needed before a fresh RGB-D observation, so freeze both
                # public proprioceptive components and borrow only the
                # sensor-derived pregrasp height.
                old_high_pose = self._secondary_pose
                retract_position = old_high_pose.position.copy()
                retract_position[:2] = observation.robot.ee_pose.position[:2]
                self._cavity_switch_pose = Pose(
                    retract_position,
                    observation.robot.ee_pose.rotation.copy(),
                )
                # The partial aperture belonged to the rejected approach.
                # Clear it before the first retract tick so it cannot leak
                # into either the escape or the next candidate's pregrasp.
                self._cavity_gripper_preshaped = False
                if self._grasp_target_attempts:
                    fallback = self._grasp_target_attempts[-1].get(
                        "cavity_approach_fallback"
                    )
                    if isinstance(fallback, dict):
                        fallback["retract_strategy"] = (
                            "measured_xy_and_rotation_world_z_to_sensor_high"
                        )
                        fallback["retract_start_world_m"] = (
                            observation.robot.ee_pose.position.tolist()
                        )
                        fallback["old_high_world_m"] = (
                            old_high_pose.position.tolist()
                        )
                        fallback["retract_target_world_m"] = (
                            self._cavity_switch_pose.position.tolist()
                        )
                        fallback["retreat_pregrasp_z_m"] = float(
                            self._cavity_switch_pose.position[2]
                        )
                self._set_phase("cavity_retract")
                return self._motion(
                    self._cavity_switch_pose,
                    observation,
                    self._released_gripper_command(),
                )
            elif self._free_space_rim_approach_should_fallback(
                observation.robot.ee_pose,
                self._motion_pose,
                observation.robot.gripper_width_m,
            ):
                assert self._secondary_pose is not None
                residual = (
                    self._motion_pose.position
                    - observation.robot.ee_pose.position
                )
                rotation_error = float(
                    np.linalg.norm(
                        self._rotation_vector(
                            self._motion_pose.rotation
                            @ observation.robot.ee_pose.rotation.T
                        )
                    )
                )
                can_try_antipodal = bool(
                    self._grasp_retry_index == 0
                    and self.config.max_grasp_retries >= 1
                    and self._pick_source_geometry is not None
                    and self._pick_anchor_center_world is not None
                )
                next_retry_index = 1 if can_try_antipodal else None
                fallback_trace = {
                    "strategy": (
                        "proprio_stall_open_vertical_retract_then_fresh_antipodal_rim"
                    ),
                    "residual_world_m": residual.tolist(),
                    "planar_residual_m": float(np.linalg.norm(residual[:2])),
                    "vertical_clearance_above_target_m": -float(residual[2]),
                    "rotation_error_rad": rotation_error,
                    "gripper_width_m": float(
                        observation.robot.gripper_width_m
                    ),
                    "capture_sequence": int(
                        self._observation_timestamp_s
                    ),
                    "contact_window_net_progress_m": (
                        self._contact_window_net_progress_m()
                    ),
                    "contact_cartesian_span_m": (
                        self._contact_cartesian_span_m()
                    ),
                    "contact_cartesian_span_tolerance_m": float(
                        self.config.rim_place_contact_cartesian_span_m
                    ),
                    "next_retry_index": next_retry_index,
                    "authorization": {
                        "close": False,
                        "task_success": False,
                        "fresh_antipodal_reacquisition": can_try_antipodal,
                    },
                }
                if self._grasp_target_attempts:
                    self._grasp_target_attempts[-1][
                        "free_space_rim_approach_fallback"
                    ] = fallback_trace
                if can_try_antipodal:
                    assert self._pick_source_geometry is not None
                    self._grasp_retry_index = 1
                    self._free_rim_retry_terminal_failure = None
                    self._free_rim_retry_anchor_geometry = (
                        self._pick_source_geometry
                    )
                    self._free_rim_reseat_association_anchor_world = None
                    self._free_rim_reseat_association_capture_sequence = None
                    self._free_rim_bilateral_completion_sequence = None
                    self._free_rim_open_high_retract_completion_sequence = None
                    self._pick_selector_reacquire_radius_m = (
                        self._selector_reacquire_radius(
                            self._free_rim_retry_anchor_geometry
                        )
                    )
                    self._free_rim_antipodal_reacquisition_required = True
                else:
                    self._free_rim_retry_terminal_failure = (
                        "both sensor-bound free-space physical rim approaches "
                        "remained blocked before contact"
                    )
                retract_position = observation.robot.ee_pose.position.copy()
                retract_position[2] = max(
                    float(retract_position[2]),
                    float(self._secondary_pose.position[2]),
                )
                self._free_rim_retract_pose = Pose(
                    retract_position,
                    observation.robot.ee_pose.rotation.copy(),
                )
                fallback_trace["retract_start_world_m"] = (
                    observation.robot.ee_pose.position.tolist()
                )
                fallback_trace["retract_target_world_m"] = (
                    self._free_rim_retract_pose.position.tolist()
                )
                self._set_phase("free_rim_retract")
                return self._motion(
                    self._free_rim_retract_pose,
                    observation,
                    GRIPPER_OPEN,
                )
            else:
                gripper_command = self._released_gripper_command()
                if is_cavity_rim:
                    gripper_command = self._cavity_rim_preshape_command(
                        observation.robot.gripper_width_m
                    )
                    self._record_cavity_preshape_sample(
                        "approach",
                        observation.robot.gripper_width_m,
                        gripper_command,
                    )
                elif (
                    is_exterior_rim
                    and self._free_rim_preshape_reseat_required
                ):
                    gripper_command = self._free_space_rim_preshape_command(
                        observation.robot.gripper_width_m
                    )
                    self._record_free_space_rim_preshape_sample(
                        "approach",
                        observation.robot.gripper_width_m,
                        gripper_command,
                    )
                else:
                    pinch_aperture = self._flat_pinch_approach_aperture()
                    if pinch_aperture is not None:
                        gripper_command = (
                            GRIPPER_CLOSE
                            if observation.robot.gripper_width_m > pinch_aperture
                            else GRIPPER_OPEN
                        )
                return self._motion(
                    self._motion_pose, observation, gripper_command
                )

        if self._phase == "close":
            # Entering CLOSE is the first actual engagement command.  The
            # pending guard makes a multi-tick close/open dwell one mechanical
            # attempt, rather than one event per control tick.
            self._begin_grasp_engagement(step)
            engage_ticks = (
                self.config.expand_engage_hold_ticks
                if self._grasp_mode is GraspMode.EXPAND
                else (
                    self.config.rim_pinch_engage_hold_ticks
                    if self._grasp_mode is GraspMode.RIM_PINCH
                    else self.config.close_hold_ticks
                )
            )
            pinch_settled = True
            if self._grasp_mode is GraspMode.PINCH and self._grasp_target_attempts:
                widths = self._grasp_target_attempts[-1].setdefault("close_width_samples_m", [])
                if self._phase_ticks > 0:
                    widths.append(float(observation.robot.gripper_width_m))
                recent = widths[-3:]
                pinch_settled = len(recent) >= 2 and max(recent) - min(recent) <= 0.001
            if self._phase_ticks >= engage_ticks and (
                pinch_settled or self._phase_ticks >= engage_ticks + 12
            ):
                if (
                    self._grasp_mode is GraspMode.RIM_PINCH
                    and self._active_cavity_rim
                    and self._grasp_target_attempts
                ):
                    self._grasp_target_attempts[-1][
                        "cavity_close_completion_width_m"
                    ] = float(observation.robot.gripper_width_m)
                    if (
                        self._grasp_target_attempts[-1].get(
                            "cavity_selected_side_profile"
                        )
                        == "roomy_near_nominal"
                    ):
                        return self._start_roomy_near_direct_proof(
                            observation
                        )
                    if not self._start_cavity_seat_pull(observation):
                        return self._fail(
                            "cavity rim seat pull lacks a valid sensor radial axis"
                        )
                    self._set_phase("cavity_seat_pull")
                else:
                    self._set_phase("retreat")
            else:
                if self._grasp_mode is GraspMode.RIM_PINCH:
                    assert self._motion_pose is not None
                    # A rim/contact transition may occur before the Cartesian
                    # pose is exact.  Keep the same sensor target active while
                    # the fingers close so compliance seats the pads on the
                    # edge instead of freezing the wrist at first contact.
                    return self._motion(
                        self._motion_pose,
                        observation,
                        self._engaged_gripper_command(),
                    )
                return self._tick_decision(
                    self._hold_action(self._engaged_gripper_command())
                )

        if self._phase == "cavity_seat_pull":
            assert self._cavity_seat_pose is not None
            self._record_cavity_seat_pull_sample(
                observation.robot.ee_pose,
                observation.robot.gripper_width_m,
            )
            geometric_seat_reached = self._cavity_seat_pull_reached(
                observation.robot.ee_pose
            )
            contact_seat_reached = self._cavity_seat_contact_reached(
                observation.robot.ee_pose,
                observation.robot.gripper_width_m,
            )
            seat_reached = geometric_seat_reached or contact_seat_reached
            invalid_width = bool(
                observation.robot.gripper_width_m
                < self.config.rim_pinch_blocked_min_width_m
                or observation.robot.gripper_width_m
                >= self.config.gripper_open_width_m
            )
            budget_exhausted = bool(
                self._phase_ticks >= self.config.cavity_rim_seat_pull_max_ticks
            )
            if invalid_width or (budget_exhausted and not seat_reached):
                self._finish_grasp_engagement(
                    accepted=False,
                    reason=(
                        GraspReason.RETENTION_REJECTED
                        if invalid_width
                        else GraspReason.EXECUTION_STALLED
                    ),
                    evidence_source=(
                        GraspEvidenceCategory.PROPRIOCEPTION
                        if invalid_width
                        else GraspEvidenceCategory.CONTROLLER_EXECUTION
                    ),
                )
                self._complete_cavity_seat_pull(
                    observation.robot.ee_pose,
                    observation.robot.gripper_width_m,
                    outcome="rejected",
                    reason=(
                        "proprioceptive_blocked_width_gate_failed"
                        if invalid_width
                        else "measured_pose_gate_not_reached_within_budget"
                    ),
                )
                if (
                    self._cavity_candidate_index
                    >= self._maximum_cavity_candidate_index()
                ):
                    self._cavity_retry_terminal_failure = (
                        "cavity rim seat pull failed its sensor/proprioceptive gate"
                    )
                else:
                    self._cavity_candidate_index += 1
                    self._grasp_retry_index += 1
                # First return vertically from the measured seated/partially
                # seated XY.  A diagonal open-gripper sweep near the fixture
                # would defeat the purpose of the bounded support-plane move.
                assert self._secondary_pose is not None
                retract_position = self._secondary_pose.position.copy()
                retract_position[:2] = observation.robot.ee_pose.position[:2]
                self._secondary_pose = Pose(
                    retract_position,
                    self._secondary_pose.rotation,
                )
                self._cavity_gripper_preshaped = False
                self._cavity_lift_start_z_m = None
                self._cavity_seat_pose = None
                self._set_phase("cavity_retry_retract")
                return self._motion(
                    self._secondary_pose,
                    observation,
                    self._released_gripper_command(),
                )
            if (
                self._phase_ticks >= self.config.cavity_rim_seat_pull_ticks
                and seat_reached
            ):
                self._complete_cavity_seat_pull(
                    observation.robot.ee_pose,
                    observation.robot.gripper_width_m,
                    outcome="accepted",
                    reason=(
                        "measured_pose_gate_reached"
                        if geometric_seat_reached
                        else "proprio_width_gain_contact_seated"
                    ),
                )
                self._cavity_lift_start_z_m = float(
                    observation.robot.ee_pose.position[2]
                )
                self._start_cavity_lift_proof(
                    observation.robot.gripper_width_m
                )
                # Proof lift vertically from the *measured* seated XY.  Moving
                # back toward the nominal pregrasp while lifting would undo
                # the support-plane seating action we just established.
                assert self._secondary_pose is not None
                proof_position = self._secondary_pose.position.copy()
                proof_position[:2] = observation.robot.ee_pose.position[:2]
                self._secondary_pose = Pose(
                    proof_position,
                    self._secondary_pose.rotation,
                )
                self._set_phase("retreat")
            else:
                return self._motion(
                    self._cavity_seat_pose,
                    observation,
                    self._engaged_gripper_command(),
                )

        if self._phase == "retreat":
            assert self._secondary_pose is not None
            cavity_operational_reached: bool | None = None
            if (
                self._grasp_mode is GraspMode.RIM_PINCH
                and self._active_cavity_rim
                and self._cavity_lift_start_z_m is not None
            ):
                lift_failed = self._record_cavity_lift_proof_sample(
                    observation.robot.ee_pose,
                    observation.robot.gripper_width_m,
                )
                if lift_failed:
                    self._finish_grasp_engagement(
                        accepted=False,
                        reason=GraspReason.RETENTION_REJECTED,
                        evidence_source=GraspEvidenceCategory.PROPRIOCEPTION,
                    )
                    if (
                        self._cavity_candidate_index
                        >= self._maximum_cavity_candidate_index()
                    ):
                        self._cavity_retry_terminal_failure = (
                            "cavity rim grasp lost blocked width during proof lift"
                        )
                    else:
                        self._cavity_candidate_index += 1
                        self._grasp_retry_index += 1
                    self._cavity_gripper_preshaped = False
                    self._cavity_lift_start_z_m = None
                    self._cavity_seat_pose = None
                    self._set_phase("cavity_retry_retract")
                    return self._motion(
                        self._secondary_pose,
                        observation,
                        self._released_gripper_command(),
                    )
                cavity_operational_reached = (
                    self._cavity_operational_clearance_reached(
                        observation.robot.ee_pose,
                        self._secondary_pose,
                        observation.robot.gripper_width_m,
                    )
                )
            retreat_reached = (
                cavity_operational_reached
                if cavity_operational_reached is not None
                else self._pose_reached(
                    observation.robot.ee_pose,
                    self._secondary_pose,
                )
            )
            if retreat_reached:
                self._set_phase("verify_grasp")
            else:
                return self._motion(
                    self._secondary_pose, observation, self._engaged_gripper_command()
                )

        if self._phase == "verify_grasp":
            nearest = getattr(self.perception, "observe_nearest", None)
            nearest_point = observation.robot.ee_pose.position
            if (
                self._grasp_mode is GraspMode.RIM_PINCH
                and self._pick_source_geometry is not None
                and self._motion_pose is not None
            ):
                nearest_point = (
                    observation.robot.ee_pose.position
                    + self._pick_source_geometry.centroid_world
                    - self._motion_pose.position
                )
            snapshot = (
                nearest(observation, step.subject, nearest_point)
                if callable(nearest)
                else self.perception.observe(observation, (step.subject,))
            )
            grasp_verified = self.verifier.holding(
                snapshot, observation, step.subject, self._grasp_mode, self.config
            )
            detected_source = snapshot.objects.get(step.subject)
            formed_stack_retained: bool | None = None
            formed_stack_lift_m: float | None = None
            if self._active_carry_stack:
                formed_stack_lift_m = (
                    float(
                        detected_source.centroid_world[2]
                        - self._formed_stack_geometry.centroid_world[2]
                    )
                    if detected_source is not None
                    and self._formed_stack_geometry is not None
                    else None
                )
                formed_stack_retained = bool(
                    self.verifier.formed_stack_shape(
                        detected_source,
                        self._formed_stack_top_geometry,
                        self._formed_stack_base_geometry,
                    )
                    and formed_stack_lift_m is not None
                    and formed_stack_lift_m
                    >= self.config.formed_stack_proof_lift_m
                )
                grasp_verified = bool(grasp_verified and formed_stack_retained)
            verification_trace: dict[str, object] = {
                "attempt": self._grasp_retry_index,
                "mode": self._grasp_mode.value,
                "gripper_width_m": float(observation.robot.gripper_width_m),
                "accepted": bool(grasp_verified),
                **(
                    {
                        "formed_stack_retained": bool(formed_stack_retained),
                        "formed_stack_visual_lift_m": formed_stack_lift_m,
                        "formed_stack_required_lift_m": float(
                            self.config.formed_stack_proof_lift_m
                        ),
                    }
                    if formed_stack_retained is not None
                    else {}
                ),
                "marginal": bool(
                    grasp_verified
                    and self._grasp_mode is GraspMode.RIM_PINCH
                    and observation.robot.gripper_width_m
                    <= self.config.rim_pinch_blocked_min_width_m
                    + self.config.rim_pinch_weak_width_margin_m
                ),
            }
            if self._active_pan_handle_slot_id is not None:
                verification_trace["pan_handle_slot_id"] = (
                    self._active_pan_handle_slot_id
                )
            self._grasp_verifications.append(verification_trace)
            if not grasp_verified:
                if self._active_pan_handle_slot_id is not None:
                    # The ordinary retry already opens at the proof-lift pose,
                    # invalidates the dynamic crop, and returns through a high
                    # pregrasp.  Remember only the failed sensor slot so the
                    # next fresh cloud selects a physically distinct handle
                    # section instead of changing Z at the same empty point.
                    self._failed_pan_handle_slot_ids.add(
                        self._active_pan_handle_slot_id
                    )
                self._finish_grasp_engagement(
                    accepted=False,
                    reason=GraspReason.RETENTION_REJECTED,
                    evidence_source=(
                        GraspEvidenceCategory.RGBD_AND_PROPRIOCEPTION
                    ),
                )
                exhausted_free_space_rims = bool(
                    self._grasp_mode is GraspMode.RIM_PINCH
                    and not self._active_cavity_rim
                    and self._grasp_retry_index >= 1
                )
                if exhausted_free_space_rims:
                    # Exterior rim selection is deliberately a two-member
                    # physical set: reset-bearing near, then its antipode.
                    # The generic retry budget may be larger because it also
                    # serves centre pinches and handles, but retry index 2
                    # would map the modulo-two rim target back to the already
                    # attempted near edge.  Release at the public proof-lift
                    # pose (or continue vertically to its sensor high pose)
                    # and terminate only after the open/high gate passes.
                    retract_position = observation.robot.ee_pose.position.copy()
                    if self._secondary_pose is not None:
                        retract_position[2] = max(
                            float(retract_position[2]),
                            float(self._secondary_pose.position[2]),
                        )
                    self._free_rim_retract_pose = Pose(
                        retract_position,
                        observation.robot.ee_pose.rotation.copy(),
                    )
                    self._free_rim_retry_terminal_failure = (
                        "both sensor-bound free-space physical rim grasps "
                        "failed sensor retention verification"
                    )
                    verification_trace["next_retry_index"] = None
                    verification_trace["recovery"] = (
                        "open_high_retract_then_fail_after_antipodal_retention_rejection"
                    )
                    self._set_phase("free_rim_retract")
                    return self._motion(
                        self._free_rim_retract_pose,
                        observation,
                        GRIPPER_OPEN,
                    )
                if self._grasp_retry_index < self.config.max_grasp_retries:
                    self._grasp_retry_index += 1
                    if self._active_cavity_rim:
                        self._cavity_candidate_index = min(
                            self._cavity_candidate_index + 1,
                            self._maximum_cavity_candidate_index(),
                        )
                    self._set_phase("retry_release")
                    return self._tick_decision(
                        self._hold_action(self._released_gripper_command()),
                        f"grasp rejected; starting sensor-only retry "
                        f"{self._grasp_retry_index}/{self.config.max_grasp_retries}",
                    )
                return self._fail(f"sensor verification rejected grasp of {step.subject!r}")
            self._finish_grasp_engagement(
                accepted=True,
                reason=GraspReason.ACCEPTED,
                evidence_source=GraspEvidenceCategory.RGBD_AND_PROPRIOCEPTION,
            )
            if self._grasp_mode is GraspMode.RIM_PINCH:
                detected_is_near, rim_visual_trace = self._rim_held_visual_plausibility(
                    detected_source,
                    observation,
                )
                rim_visual_trace["rim_visual_query_point_world_m"] = (
                    nearest_point.tolist()
                )
                verification_trace.update(rim_visual_trace)
            else:
                detected_is_near = bool(
                    detected_source is not None
                    and np.linalg.norm(
                        detected_source.centroid_world
                        - observation.robot.ee_pose.position
                    )
                    <= self.config.holding_distance_m
                )
            source = (
                detected_source
                if self._active_carry_stack and formed_stack_retained
                else (
                    detected_source
                    if detected_is_near
                    else self._pick_source_geometry
                )
            )
            if source is None or self._motion_pose is None:
                return self._fail(f"lost sensor geometry for grasped {step.subject!r}")
            self._held_subject = step.subject
            self._held_from_cavity_rim = bool(self._active_cavity_rim)
            verification_trace["held_from_cavity_rim"] = (
                self._held_from_cavity_rim
            )
            pan_body_reference = self._pan_body_reference_world
            if (
                self._active_pan_handle_slot_id is not None
                and pan_body_reference is not None
            ):
                # The handle target is intentionally far from the pan body's
                # centre.  Carry the independently inferred wide-body centre
                # relative to that target so PLACE_ON aligns the load-bearing
                # pan body with the support instead of aligning the asymmetric
                # whole-crop centroid.  Z remains the completed RGB-D object
                # centre because the surface estimator's body Z is a visible
                # surface statistic rather than a volume centre.
                self._held_offset_world = (
                    pan_body_reference - self._motion_pose.position
                )
            elif self._active_carry_stack and formed_stack_retained:
                assert detected_source is not None
                self._held_offset_world = (
                    detected_source.centroid_world
                    - observation.robot.ee_pose.position
                )
            elif detected_is_near and self._grasp_mode is GraspMode.RIM_PINCH:
                self._held_offset_world = np.asarray(
                    verification_trace["rim_visual_constrained_offset_world_m"],
                    dtype=np.float64,
                )
            else:
                self._held_offset_world = (
                    source.centroid_world - observation.robot.ee_pose.position
                    if detected_is_near
                    else source.centroid_world - self._motion_pose.position
                )
            self._held_offset_source = (
                "rgbd_pan_body_center_to_handle"
                if self._active_pan_handle_slot_id is not None
                and pan_body_reference is not None
                else (
                    "rgbd_verified_formed_stack"
                    if self._active_carry_stack and formed_stack_retained
                    else (
                        "rgbd_nearest_rim"
                        if detected_is_near
                        and self._grasp_mode is GraspMode.RIM_PINCH
                        else (
                            "rgbd_nearest"
                            if detected_is_near
                            else "initial_unoccluded_grasp_geometry"
                        )
                    )
                )
            )
            if (
                self._grasp_mode is GraspMode.PINCH
                and self._pick_initial_geometry is not None
                and detected_is_near
                and self._active_pan_handle_slot_id is None
                and not self._active_carry_stack
            ):
                from .rigid_shape import compact_center_pinch_offset

                rigid_offset = compact_center_pinch_offset(
                    self._pick_initial_geometry, self._motion_pose,
                    observation.robot.ee_pose, observation.robot.gripper_width_m,
                )
                if rigid_offset is not None and np.linalg.norm(
                    rigid_offset - self._held_offset_world
                ) > .015:
                    verification_trace["compact_center_pinch_geometry"] = {
                        "visual_offset_world_m": self._held_offset_world.tolist(),
                        "rigid_offset_world_m": rigid_offset.tolist(),
                        "source_center_world_m": self._pick_initial_geometry.centroid_world.tolist(),
                        "grasp_target_world_m": self._motion_pose.position.tolist(),
                        "measured_width_m": float(observation.robot.gripper_width_m),
                    }
                    self._held_offset_world = rigid_offset
                    self._held_offset_source = "precontact_compact_center_pinch_geometry"
            if (self._grasp_mode is GraspMode.PINCH and detected_is_near
                    and source.name == "yellow and white mug"):
                refine = getattr(self.perception, "refine_carried_mug", None)
                refined = refine(observation, source) if callable(refine) else None
                if refined is not None:
                    source, fit_detail = refined
                    verification_trace["current_public_mug_mesh"] = {
                        **fit_detail,
                        "previous_offset_world_m": self._held_offset_world.tolist(),
                        "center_world_m": source.centroid_world.tolist(),
                        "axes_world": source.axes_world.tolist(),
                    }
                    self._held_offset_world = source.centroid_world - observation.robot.ee_pose.position
                    self._held_offset_source = "current_rgbd_public_mug_mesh"
            if (self._grasp_mode is GraspMode.PINCH and detected_is_near
                    and source.name == "chocolate pudding"):
                refine = getattr(self.perception, "refine_carried_flat", None)
                refined = refine(observation, source, observation.robot) if callable(refine) else None
                if refined is not None:
                    source, fit_detail = refined
                    verification_trace["current_public_flat_mesh"] = {
                        **fit_detail,
                        "previous_offset_world_m": self._held_offset_world.tolist(),
                        "center_world_m": source.centroid_world.tolist(),
                        "axes_world": source.axes_world.tolist(),
                    }
                    self._held_offset_world = source.centroid_world - observation.robot.ee_pose.position
                    self._held_offset_source = "current_rgbd_public_flat_mesh"
            if pan_body_reference is not None:
                verification_trace["pan_body_reference_world_m"] = (
                    pan_body_reference.tolist()
                )
            verification_trace["held_offset_source"] = self._held_offset_source
            verification_trace["held_offset_world_m"] = self._held_offset_world.tolist()
            self._held_alternate_book_geometry = None
            if (step.subject == "book" and self._grasp_mode is GraspMode.PINCH
                    and self._pick_source_geometry is not None):
                from .rigid_shape import propagate_grasped_shape

                self._held_alternate_book_geometry = propagate_grasped_shape(
                    self._pick_source_geometry, source,
                    self._motion_pose.rotation, observation.robot.ee_pose.rotation,
                )
            self._held_geometry = source
            self._grasp_is_marginal = bool(
                self._grasp_mode is GraspMode.RIM_PINCH
                and observation.robot.gripper_width_m
                <= self.config.rim_pinch_blocked_min_width_m
                + self.config.rim_pinch_weak_width_margin_m
            )
            return self._complete_skill("grasp verified from RGB-D and gripper state")

        return self._fail(f"invalid pick phase {self._phase!r}")

    def _observe_pick_subject(
        self,
        step: SkillStep,
        observation: SensorObservation,
    ) -> SceneSnapshot:
        """Use selector-aware perception when the adapter supports it.

        Legacy and test perception implementations keep the original two-arg
        protocol.  The production sensor adapter consumes the compiled
        BETWEEN/NEXT_TO/CENTER/ON/IN qualifier here.  Once ON/IN has selected
        a physical instance, retries use nearest-neighbour association around
        that frozen sensor centre instead of rerunning the global selector.
        A typed free-space antipodal-rim retry applies the same principle even
        without a language selector, additionally preserving attempt zero's
        sensor size as identity evidence.
        """

        if step.relation == "carry_stack":
            return self._observe_formed_stack_for_pick(step, observation)

        if self._free_rim_antipodal_reacquisition_required:
            return self._observe_free_space_rim_retry_subject(
                step,
                observation,
            )

        prefetched = self._prefetched_pick_sources.get(self._skill_index)
        if prefetched is not None:
            if (self._skill_index in self._optional_pick_source_anchors
                    and self.grasp_mode_selector.select(step) is GraspMode.RIM_PINCH):
                from ..perception.adapters import coerce_rgbd_frame
                from ..perception.geometry import backproject_frame
                from .anchored_rim import anchored_rim_from_points

                clouds = [backproject_frame(coerce_rgbd_frame(frame, name=name), stride=1).points_world
                          for name, frame in observation.cameras.items()]
                try:
                    fresh = anchored_rim_from_points(np.concatenate(clouds), prefetched)
                except ValueError:
                    pass
                else:
                    diagnostics = getattr(self.perception, "_selector_diagnostics", None)
                    if isinstance(diagnostics, list):
                        diagnostics.append({
                            "kind": "precontact_identity_fresh_rim_reacquisition",
                            "anchor_center_world_m": prefetched.centroid_world.tolist(),
                            "fresh_center_world_m": fresh.centroid_world.tolist(),
                            "fresh_extents_m": fresh.extents_m.tolist(),
                            "fresh_point_count": fresh.point_count,
                        })
                    return SceneSnapshot(self._observation_timestamp_s, {step.subject: fresh})
            def missing_prefetched_source() -> SceneSnapshot:
                if self._skill_index in self._optional_pick_source_anchors:
                    # Precontact geometry is a search hint. If it cannot be
                    # confirmed now, use ordinary fresh semantic grounding.
                    # Repeated-object identity anchors remain mandatory.
                    return self._observe_entity(observation, step.subject, step.selector)
                return SceneSnapshot(self._observation_timestamp_s, {})

            nearest = getattr(self.perception, "observe_nearest_bound", None)
            if not callable(nearest):
                nearest = getattr(self.perception, "observe_nearest", None)
            if not callable(nearest):
                return missing_prefetched_source()
            try:
                snapshot = nearest(
                    observation,
                    step.subject,
                    prefetched.centroid_world,
                )
            except LookupError:
                return missing_prefetched_source()
            candidate = snapshot.objects.get(step.subject)
            if candidate is None:
                return missing_prefetched_source()
            radius = self._selector_reacquire_radius(prefetched)
            if float(
                np.linalg.norm(candidate.centroid_world - prefetched.centroid_world)
            ) > radius:
                return missing_prefetched_source()
            return snapshot

        if (
            self._grasp_retry_index > 0
            and self._pick_selector_continuity
            and self._pick_anchor_center_world is not None
        ):
            retry_trace: dict[str, object] = {
                "attempt": int(self._grasp_retry_index),
                "anchor_center_world_m": self._pick_anchor_center_world.tolist(),
                "anchor_radius_m": (
                    float(self._pick_selector_reacquire_radius_m)
                    if self._pick_selector_reacquire_radius_m is not None
                    else None
                ),
            }
            nearest = getattr(self.perception, "observe_nearest_bound", None)
            if not callable(nearest):
                nearest = getattr(self.perception, "observe_nearest", None)
            if not callable(nearest):
                snapshot = SceneSnapshot(self._observation_timestamp_s, {})
                retry_trace["nearest_bound_result"] = "unavailable"
            else:
                try:
                    snapshot = nearest(
                        observation,
                        step.subject,
                        self._pick_anchor_center_world,
                    )
                except LookupError as exc:
                    snapshot = SceneSnapshot(self._observation_timestamp_s, {})
                    retry_trace["nearest_bound_result"] = (
                        f"lookup_error:{type(exc).__name__}"
                    )
            candidate = snapshot.objects.get(step.subject)
            radius = self._pick_selector_reacquire_radius_m
            nearest_distance = (
                float(
                    np.linalg.norm(
                        candidate.centroid_world - self._pick_anchor_center_world
                    )
                )
                if candidate is not None
                else None
            )
            retry_trace["nearest_bound_distance_m"] = nearest_distance
            if (
                candidate is not None
                and radius is not None
                and nearest_distance is not None
                and nearest_distance <= radius
            ):
                retry_trace["nearest_bound_result"] = "accepted"
                retry_trace["selected_source"] = "nearest_bound"
                self._record_selector_retry_reacquisition(retry_trace)
                return snapshot
            if "nearest_bound_result" not in retry_trace:
                retry_trace["nearest_bound_result"] = (
                    "missing" if candidate is None else "outside_initial_anchor_gate"
                )

            # The four-class nearest gallery can conservatively reject a
            # partly occluded bowl.  Re-run the original IN/ON relation as a
            # stricter fallback, then apply the *same* frozen 45-mm anchor
            # gate.  This restores relational context without widening or
            # replacing the initially selected physical instance.
            selected = getattr(self.perception, "observe_selected", None)
            relational = SceneSnapshot(self._observation_timestamp_s, {})
            if step.selector is not None and callable(selected):
                try:
                    relational = selected(
                        observation,
                        step.subject,
                        step.selector,
                    )
                except LookupError as exc:
                    retry_trace["relational_fallback_result"] = (
                        f"lookup_error:{type(exc).__name__}"
                    )
            else:
                retry_trace["relational_fallback_result"] = "unavailable"
            relational_candidate = relational.objects.get(step.subject)
            relational_distance = (
                float(
                    np.linalg.norm(
                        relational_candidate.centroid_world
                        - self._pick_anchor_center_world
                    )
                )
                if relational_candidate is not None
                else None
            )
            retry_trace["relational_fallback_distance_m"] = relational_distance
            if (
                relational_candidate is not None
                and radius is not None
                and relational_distance is not None
                and relational_distance <= radius
            ):
                retry_trace["relational_fallback_result"] = "accepted"
                retry_trace["selected_source"] = "fresh_relational_selector"
                self._record_selector_retry_reacquisition(retry_trace)
                return relational
            if "relational_fallback_result" not in retry_trace:
                retry_trace["relational_fallback_result"] = (
                    "missing"
                    if relational_candidate is None
                    else "outside_initial_anchor_gate"
                )
            retry_trace["selected_source"] = "none"
            self._record_selector_retry_reacquisition(retry_trace)
            return SceneSnapshot(self._observation_timestamp_s, {})

        selected = getattr(self.perception, "observe_selected", None)
        if step.selector is not None and callable(selected):
            return selected(observation, step.subject, step.selector)
        return self.perception.observe(observation, (step.subject,))

    def _observe_free_space_rim_retry_subject(
        self,
        step: SkillStep,
        observation: SensorObservation,
    ) -> SceneSnapshot:
        """Freshly rebind the antipodal rim to attempt zero's RGB-D object.

        This path intentionally has no global-label or relaxed-nearest
        fallback.  A missing or discontinuous bound crop leaves the already
        retracted hand high and open; the ordinary finite perception-miss
        budget then fails closed without ever emitting a jump target.
        """

        anchor_geometry = self._free_rim_retry_anchor_geometry
        anchor_center = self._pick_anchor_center_world
        radius = self._pick_selector_reacquire_radius_m
        final_reseat = self._free_rim_preshape_reseat_required
        association_anchor = (
            self._free_rim_reseat_association_anchor_world
            if final_reseat
            else anchor_center
        )
        association_capture = (
            self._free_rim_reseat_association_capture_sequence
        )
        bilateral_sequence = self._free_rim_bilateral_completion_sequence
        retract_sequence = (
            self._free_rim_open_high_retract_completion_sequence
        )
        capture_sequence = int(self._observation_timestamp_s)
        cumulative_limit = (
            2.0 * float(radius) if radius is not None else None
        )
        trace: dict[str, object] = {
            "attempt": int(self._grasp_retry_index),
            "capture_sequence": capture_sequence,
            "anchor_center_world_m": (
                anchor_center.tolist() if anchor_center is not None else None
            ),
            "identity_anchor_center_world_m": (
                anchor_center.tolist() if anchor_center is not None else None
            ),
            "association_anchor_center_world_m": (
                association_anchor.tolist()
                if association_anchor is not None
                else None
            ),
            "association_anchor_capture_sequence": association_capture,
            "bilateral_completion_sequence": bilateral_sequence,
            "open_high_retract_completion_sequence": retract_sequence,
            "anchor_radius_m": float(radius) if radius is not None else None,
            "initial_anchor_cumulative_limit_m": cumulative_limit,
            "nearest_bound_result": "unavailable",
            "selected_source": "none",
        }
        if (
            self._grasp_mode is not GraspMode.RIM_PINCH
            or self._active_cavity_rim
            or self._grasp_retry_index != 1
            or anchor_geometry is None
            or anchor_center is None
            or radius is None
        ):
            trace["nearest_bound_result"] = "invalid_typed_retry_state"
            self._record_free_space_rim_retry_reacquisition(trace)
            return SceneSnapshot(self._observation_timestamp_s, {})

        if final_reseat:
            association_state_valid = bool(
                association_anchor is not None
                and association_anchor.shape == (3,)
                and np.all(np.isfinite(association_anchor))
                and type(association_capture) is int
                and type(bilateral_sequence) is int
                and type(retract_sequence) is int
                and association_capture
                < bilateral_sequence
                < retract_sequence
                < capture_sequence
                and self._free_rim_reseat_reacquisition is None
            )
            if not association_state_valid:
                trace["nearest_bound_result"] = (
                    "final_reseat_already_consumed"
                    if self._free_rim_reseat_reacquisition is not None
                    else "invalid_typed_reseat_association_state"
                )
                self._record_free_space_rim_retry_reacquisition(trace)
                return SceneSnapshot(self._observation_timestamp_s, {})
        assert association_anchor is not None
        assert cumulative_limit is not None

        nearest = getattr(self.perception, "observe_nearest_bound", None)
        if not callable(nearest):
            self._record_free_space_rim_retry_reacquisition(trace)
            return SceneSnapshot(self._observation_timestamp_s, {})
        try:
            snapshot = nearest(
                observation,
                step.subject,
                association_anchor,
            )
        except LookupError as exc:
            trace["nearest_bound_result"] = f"lookup_error:{type(exc).__name__}"
            self._record_free_space_rim_retry_reacquisition(trace)
            return SceneSnapshot(self._observation_timestamp_s, {})

        candidate = snapshot.objects.get(step.subject)
        if candidate is None:
            trace["nearest_bound_result"] = "missing"
            self._record_free_space_rim_retry_reacquisition(trace)
            return SceneSnapshot(self._observation_timestamp_s, {})

        association_leg_distance = float(
            np.linalg.norm(candidate.centroid_world - association_anchor)
        )
        initial_anchor_cumulative_distance = float(
            np.linalg.norm(candidate.centroid_world - anchor_center)
        )
        association_anchor_initial_distance = float(
            np.linalg.norm(association_anchor - anchor_center)
        )
        anchor_extents = np.sort(
            np.asarray(anchor_geometry.extents_m, dtype=np.float64)
        )
        candidate_extents = np.sort(
            np.asarray(candidate.extents_m, dtype=np.float64)
        )
        extent_scales = candidate_extents / anchor_extents
        size_continuous = bool(
            np.all(np.isfinite(extent_scales))
            and np.all(
                extent_scales
                >= self.config.free_space_rim_retry_min_extent_scale
            )
            and np.all(
                extent_scales
                <= self.config.free_space_rim_retry_max_extent_scale
            )
        )
        trace.update(
            {
                "nearest_bound_distance_m": association_leg_distance,
                "association_leg_distance_m": association_leg_distance,
                "association_anchor_initial_distance_m": (
                    association_anchor_initial_distance
                ),
                "initial_anchor_cumulative_distance_m": (
                    initial_anchor_cumulative_distance
                ),
                "candidate_center_world_m": candidate.centroid_world.tolist(),
                "anchor_extents_sorted_m": anchor_extents.tolist(),
                "candidate_extents_sorted_m": candidate_extents.tolist(),
                "candidate_extent_scales": extent_scales.tolist(),
                "extent_scale_gate": [
                    float(self.config.free_space_rim_retry_min_extent_scale),
                    float(self.config.free_space_rim_retry_max_extent_scale),
                ],
                "size_continuous": size_continuous,
            }
        )
        if association_anchor_initial_distance > radius:
            trace["nearest_bound_result"] = (
                "association_anchor_outside_initial_anchor_gate"
            )
        elif association_leg_distance > radius:
            trace["nearest_bound_result"] = (
                "outside_association_anchor_gate"
                if final_reseat
                else "outside_initial_anchor_gate"
            )
        elif initial_anchor_cumulative_distance > cumulative_limit:
            trace["nearest_bound_result"] = "outside_initial_anchor_global_cap"
        elif not size_continuous:
            trace["nearest_bound_result"] = "outside_initial_size_gate"
        else:
            trace["nearest_bound_result"] = "accepted"
            trace["selected_source"] = (
                "fresh_nearest_bound_attempt1_association_anchor"
                if final_reseat
                else "fresh_nearest_bound_attempt0_anchor"
            )
            if final_reseat:
                # Preserve the exact fresh binding consumed by the final
                # preshape/reseat attempt.  The close gate revalidates every
                # field against the attempt emitted from this same frame.
                self._free_rim_reseat_reacquisition = dict(trace)
            self._record_free_space_rim_retry_reacquisition(trace)
            return snapshot

        self._record_free_space_rim_retry_reacquisition(trace)
        return SceneSnapshot(self._observation_timestamp_s, {})

    def _record_free_space_rim_retry_reacquisition(
        self,
        trace: dict[str, object],
    ) -> None:
        if not self._grasp_target_attempts:
            return
        attempts = self._grasp_target_attempts[-1].setdefault(
            "free_space_rim_retry_reacquisition",
            [],
        )
        if isinstance(attempts, list):
            attempts.append(trace)

    def _observe_formed_stack_for_pick(
        self,
        step: SkillStep,
        observation: SensorObservation,
    ) -> SceneSnapshot:
        """Reacquire a previously verified pair and expose its base rim.

        A rank selector cannot be rerun after stacking because two same-label
        detections may now fuse.  The anchor and member geometry here were
        created by the immediately preceding fresh post-stack observation.
        The current RGB-D crop must remain near that anchor and retain the
        measured two-member height/footprint signature before a grasp target
        is emitted.
        """

        signature = self._formed_stack_signature
        formed = self._formed_stack_geometry
        top = self._formed_stack_top_geometry
        base = self._formed_stack_base_geometry
        if signature is None or formed is None or top is None or base is None:
            return SceneSnapshot(self._observation_timestamp_s, {})
        label, top_selector, base_selector = signature
        if (
            step.subject != label
            or step.selector != base_selector
            or step.companions != (EntityBinding(label, top_selector),)
        ):
            return SceneSnapshot(self._observation_timestamp_s, {})
        nearest = getattr(self.perception, "observe_nearest", None)
        if not callable(nearest):
            return SceneSnapshot(self._observation_timestamp_s, {})
        try:
            snapshot = nearest(
                observation,
                step.subject,
                formed.centroid_world,
            )
        except LookupError:
            return SceneSnapshot(self._observation_timestamp_s, {})
        detected = snapshot.objects.get(step.subject)
        anchor_radius = max(
            0.060,
            0.75
            * float(
                np.max(
                    formed.bounds_max_world[:2]
                    - formed.bounds_min_world[:2]
                )
            ),
        )
        if (
            detected is None
            or float(
                np.linalg.norm(
                    detected.centroid_world - formed.centroid_world
                )
            )
            > anchor_radius
            or not self.verifier.formed_stack_shape(detected, top, base)
        ):
            return SceneSnapshot(self._observation_timestamp_s, {})

        # Target the verified lower member's rim, not the fused component's
        # upper rim.  This is what mechanically retains the lower bowl while
        # the upper one remains supported by it.  XY and support height come
        # from the fresh group crop; size/orientation come from the frozen
        # pre-stack base observation.
        center = np.array(
            (
                detected.centroid_world[0],
                detected.centroid_world[1],
                detected.bounds_min_world[2] + 0.5 * base.height_m,
            ),
            dtype=np.float64,
        )
        half_world = np.abs(base.axes_world) @ (base.extents_m / 2.0)
        base_grip = SceneObject(
            name=step.subject,
            centroid_world=center,
            axes_world=base.axes_world,
            extents_m=base.extents_m,
            bounds_min_world=center - half_world,
            bounds_max_world=center + half_world,
            confidence=min(detected.confidence, base.confidence),
            point_count=detected.point_count,
        )
        self._formed_stack_geometry = detected
        self._active_carry_stack = True
        return SceneSnapshot(self._observation_timestamp_s, {step.subject: base_grip})

    def _record_selector_retry_reacquisition(
        self,
        trace: dict[str, object],
    ) -> None:
        if not self._grasp_target_attempts:
            return
        attempts = self._grasp_target_attempts[-1].setdefault(
            "selector_retry_reacquisition",
            [],
        )
        if isinstance(attempts, list):
            attempts.append(trace)

    def _stabilize_cavity_retry_source(self, source: SceneObject) -> SceneObject:
        """Keep retry radial placement tied to the first unobstructed RGB-D centre.

        A post-contact bowl crop can move several millimetres even when the
        physical bowl mostly fell back into place.  That shift is amplified
        directly into a missed rim target.  Treat the initial centre as the
        prior and admit only small, independently bounded radial/tangential
        innovations.  Their signs still come entirely from fresh RGB-D.
        """

        if (
            self._grasp_retry_index <= 0
            or not self._pick_selector_continuity
            or not self._active_cavity_rim
            or self._pick_anchor_center_world is None
            or not self._grasp_target_attempts
        ):
            return source
        base_axis_value = self._grasp_target_attempts[-1].get(
            "cavity_base_width_axis_world"
        )
        try:
            base_axis = np.asarray(base_axis_value, dtype=np.float64)[:2]
        except (TypeError, ValueError):
            return source
        norm = float(np.linalg.norm(base_axis))
        if norm < 0.8:
            return source
        base_axis /= norm
        tangent = np.array((-base_axis[1], base_axis[0]), dtype=np.float64)
        raw_delta = source.centroid_world[:2] - self._pick_anchor_center_world[:2]
        raw_radial = float(np.dot(raw_delta, base_axis))
        raw_tangential = float(np.dot(raw_delta, tangent))
        retained_radial = float(
            np.clip(
                raw_radial,
                -self.config.cavity_rim_retry_radial_update_m,
                self.config.cavity_rim_retry_radial_update_m,
            )
        )
        retained_tangential = float(
            np.clip(
                raw_tangential,
                -self.config.cavity_rim_retry_tangential_update_m,
                self.config.cavity_rim_retry_tangential_update_m,
            )
        )
        stabilized_center = source.centroid_world.copy()
        stabilized_center[:2] = (
            self._pick_anchor_center_world[:2]
            + retained_radial * base_axis
            + retained_tangential * tangent
        )
        translation = stabilized_center - source.centroid_world
        retry_traces = self._grasp_target_attempts[-1].get(
            "selector_retry_reacquisition"
        )
        if isinstance(retry_traces, list) and retry_traces:
            retry_traces[-1]["geometry_stabilization"] = {
                "raw_center_world_m": source.centroid_world.tolist(),
                "stabilized_center_world_m": stabilized_center.tolist(),
                "raw_offset_from_initial_world_m": (
                    source.centroid_world - self._pick_anchor_center_world
                ).tolist(),
                "base_radial_axis_world": [
                    float(base_axis[0]),
                    float(base_axis[1]),
                    0.0,
                ],
                "raw_radial_component_m": raw_radial,
                "retained_radial_component_m": retained_radial,
                "dropped_radial_component_m": raw_radial - retained_radial,
                "radial_update_limit_m": float(
                    self.config.cavity_rim_retry_radial_update_m
                ),
                "raw_tangential_component_m": raw_tangential,
                "retained_tangential_component_m": retained_tangential,
                "tangential_update_limit_m": float(
                    self.config.cavity_rim_retry_tangential_update_m
                ),
                "strategy": (
                    "initial_rgbd_prior_bounded_radial_and_tangential_update"
                ),
            }
        return SceneObject(
            name=source.name,
            centroid_world=stabilized_center,
            axes_world=source.axes_world,
            extents_m=source.extents_m,
            bounds_min_world=source.bounds_min_world + translation,
            bounds_max_world=source.bounds_max_world + translation,
            confidence=source.confidence,
            point_count=source.point_count,
        )

    @staticmethod
    def _selector_binds_pick_identity(step: SkillStep) -> bool:
        selector = step.selector
        if selector is None:
            return False
        relation = getattr(selector.relation, "value", str(selector.relation))
        return str(relation).strip().lower() in {"in", "on"}

    def _selector_reacquire_radius(self, source: SceneObject) -> float:
        """Sensor-size-scaled association radius for ON/IN grasp retries."""

        vertical_axis = int(np.argmax(np.abs(source.axes_world[2, :])))
        planar_axes = tuple(index for index in range(3) if index != vertical_axis)
        object_radius = 0.5 * float(np.max(source.extents_m[list(planar_axes)]))
        return min(
            self.config.selector_retry_reacquire_radius_m,
            max(
                self.config.holding_distance_m,
                object_radius + self.config.pick_refresh_max_shift_m,
            ),
        )

    def _selector_cavity_reference(self, step: SkillStep) -> object | None:
        """Return the measured fixture OBB for an object selected inside it.

        This hook is deliberately optional so the controller remains
        compatible with the small Route-B perception protocol.  The compiled
        selector supplies only language semantics (``IN top drawer``); all
        pose, axis, and clearance values still come from RGB-D perception.
        ``ON cabinet`` is excluded because a support surface does not form a
        cavity around the hand.
        """

        selector = step.selector
        if selector is None:
            return None
        relation = getattr(selector.relation, "value", str(selector.relation))
        if str(relation).strip().lower() != "in" or len(selector.references) != 1:
            return None
        reference_name = selector.references[0].strip().lower()
        if not any(token in reference_name for token in ("drawer", "cabinet")):
            return None
        return getattr(self.perception, "last_selector_reference", None)

    @staticmethod
    def _selector_requires_cavity_reference(step: SkillStep) -> bool:
        """Whether language requires an IN-fixture cavity grasp.

        This is only a semantic guard deciding whether missing geometry may
        fall back to free space.  It contributes no pose or benchmark state;
        every motion axis and clearance still has to come from RGB-D/proprio.
        """

        selector = step.selector
        if selector is None:
            return False
        relation = getattr(selector.relation, "value", str(selector.relation))
        if str(relation).strip().lower() != "in" or len(selector.references) != 1:
            return False
        reference_name = selector.references[0].strip().lower()
        return any(token in reference_name for token in ("drawer", "cabinet"))

    def _clear_cavity_active_view_state(self) -> None:
        self._cavity_active_view_source = None
        self._cavity_active_view_safe_z_m = None
        self._cavity_active_view_vertical_pose = None
        self._cavity_active_view_pose = None
        self._cavity_active_view_index = 0
        self._cavity_active_view_trace = None

    @staticmethod
    def _sensor_geometry_trace(value: object | None) -> dict[str, object] | None:
        """Return a JSON-safe summary of sensor geometry, if available."""

        if value is None:
            return None
        center_value = getattr(value, "center_world", None)
        if center_value is None:
            center_value = getattr(value, "centroid_world", None)
        fields = {
            "center_world_m": center_value,
            "axes_world": getattr(value, "axes_world", None),
            "extents_m": getattr(value, "extents_m", None),
            "bounds_min_world_m": getattr(value, "bounds_min_world", None),
            "bounds_max_world_m": getattr(value, "bounds_max_world", None),
        }
        trace: dict[str, object] = {
            "name": str(
                getattr(value, "name", getattr(value, "label", "sensor_object"))
            )
        }
        for name, raw in fields.items():
            try:
                array = np.asarray(raw, dtype=np.float64)
            except (TypeError, ValueError):
                continue
            if array.size > 0 and np.all(np.isfinite(array)):
                trace[name] = array.tolist()
        observed = getattr(value, "observed", None)
        if observed is not None and observed is not value:
            observed_trace = RouteBController._sensor_geometry_trace(observed)
            if observed_trace is not None:
                trace["observed"] = observed_trace
        return trace

    def _begin_cavity_active_view(
        self,
        step: SkillStep,
        source: SceneObject,
        observation: SensorObservation,
    ) -> ControlDecision:
        """Start a collision-safe active view without emitting a grasp pose."""

        current = observation.robot.ee_pose
        safe_z = max(
            float(current.position[2]),
            float(source.bounds_max_world[2])
            + self.config.cavity_active_view_clearance_m,
        )
        vertical_position = current.position.copy()
        vertical_position[2] = safe_z
        vertical_pose = Pose(vertical_position, current.rotation)
        overhead_position = vertical_position.copy()
        overhead_position[:2] = source.centroid_world[:2]
        overhead_pose = Pose(overhead_position, current.rotation)
        self._cavity_active_view_source = source
        self._cavity_active_view_safe_z_m = safe_z
        self._cavity_active_view_vertical_pose = vertical_pose
        self._cavity_active_view_pose = overhead_pose
        self._cavity_active_view_index = 0
        self._pick_cavity_reference = None
        self._cavity_reference_from_active_view = False
        self._active_cavity_rim = False
        self._motion_pose = None
        self._secondary_pose = None
        self._cavity_active_view_trace = {
            "strategy": (
                "strict_2d_in_then_vertical_clearance_and_constant_height_rgbd"
            ),
            "selector_relation": "in",
            "selector_references": list(step.selector.references)
            if step.selector is not None
            else [],
            "initial_source_geometry": self._sensor_geometry_trace(source),
            "identity_anchor_world_m": (
                self._pick_anchor_center_world.tolist()
                if self._pick_anchor_center_world is not None
                else None
            ),
            "identity_anchor_radius_m": (
                float(self._pick_selector_reacquire_radius_m)
                if self._pick_selector_reacquire_radius_m is not None
                else None
            ),
            "safe_high_z_m": safe_z,
            "clearance_above_sensor_top_m": float(
                safe_z - source.bounds_max_world[2]
            ),
            "vertical_target_world_m": vertical_position.tolist(),
            "vertical_xy_locked_world_m": current.position[:2].tolist(),
            "max_rgbd_views": 2,
            "views": [
                {
                    "index": 0,
                    "kind": "source_overhead",
                    "target_world_m": overhead_position.tolist(),
                    "constant_height_z_m": safe_z,
                }
            ],
            "outcome": "in_progress",
        }
        self._set_phase("cavity_active_view_vertical")
        return self._motion(
            vertical_pose,
            observation,
            self._released_gripper_command(),
        )

    def _cavity_active_view_offset_pose(self) -> Pose | None:
        source = self._cavity_active_view_source
        safe_z = self._cavity_active_view_safe_z_m
        start = self._pick_start_pose_world
        rotation = self._pick_rotation_anchor_world
        if source is None or safe_z is None or rotation is None:
            return None
        direction = (
            start.position[:2] - source.centroid_world[:2]
            if start is not None
            else np.zeros(2, dtype=np.float64)
        )
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-6:
            # Proprio supplies a deterministic fallback when the wrist started
            # exactly above the source.  Select the most planar wrist axis;
            # no world-frame direction or task-specific constant is assumed.
            planar_axes = (rotation[:2, 0], rotation[:2, 1])
            direction = np.asarray(
                max(planar_axes, key=lambda axis: float(np.linalg.norm(axis))),
                dtype=np.float64,
            )
            norm = float(np.linalg.norm(direction))
        if norm <= 1e-6:
            return None
        direction /= norm
        position = source.centroid_world.copy()
        position[:2] += self.config.cavity_active_view_offset_m * direction
        position[2] = safe_z
        return Pose(position, rotation)

    def _cavity_active_view_traverse_reached(
        self,
        current: Pose,
        target: Pose,
    ) -> tuple[bool, dict[str, object]]:
        """Typed high-view completion gate using only RGB-D and proprio.

        Unlike a grasp waypoint, a safe camera view does not need 3-mm
        isotropic convergence.  It must remain close in the support plane,
        no more than a bounded amount below the requested height, level, and
        independently above the first sensed source top.  Passing this gate
        only permits fresh perception; it never admits a grasp target.
        """

        source = self._cavity_active_view_source
        residual = target.position - current.position
        planar_error = float(np.linalg.norm(residual[:2]))
        signed_target_shortfall = float(residual[2])
        target_shortfall = max(signed_target_shortfall, 0.0)
        rotation_error = float(
            np.linalg.norm(
                self._rotation_vector(target.rotation @ current.rotation.T)
            )
        )
        source_top = (
            float(source.bounds_max_world[2]) if source is not None else None
        )
        minimum_safe_z = (
            source_top
            + self.config.cavity_active_view_clearance_m
            - self.config.cavity_active_view_max_target_shortfall_m
            if source_top is not None
            else None
        )
        plateau_minimum_safe_z = (
            source_top
            + self.config.cavity_active_view_clearance_m
            - self.config.cavity_active_view_plateau_max_target_shortfall_m
            if source_top is not None
            else None
        )
        boundary_epsilon = 1e-12
        primary_safe_height = bool(
            minimum_safe_z is not None
            and float(current.position[2]) + boundary_epsilon >= minimum_safe_z
        )
        plateau_safe_height = bool(
            plateau_minimum_safe_z is not None
            and float(current.position[2]) + boundary_epsilon
            >= plateau_minimum_safe_z
        )

        position_error = float(np.linalg.norm(residual))
        self._contact_error_history.append(position_error)
        self._contact_position_history.append(current.position.copy())
        window_size = self.config.contact_stall_ticks + 1
        if len(self._contact_error_history) > window_size:
            self._contact_error_history.pop(0)
        if len(self._contact_position_history) > window_size:
            self._contact_position_history.pop(0)
        window_stalled = self._contact_window_stalled()
        cartesian_span = self._contact_cartesian_span_m()

        primary_passed = bool(
            source is not None
            and planar_error
            <= self.config.cavity_active_view_planar_tolerance_m
            + boundary_epsilon
            and target_shortfall
            <= self.config.cavity_active_view_max_target_shortfall_m
            + boundary_epsilon
            and rotation_error
            <= self.config.rotation_tolerance_rad + boundary_epsilon
            and primary_safe_height
        )
        plateau_passed = bool(
            not primary_passed
            and source is not None
            and self._phase_ticks >= self.config.contact_min_ticks
            and window_stalled
            and cartesian_span
            <= self.config.cavity_active_view_plateau_cartesian_span_m
            + boundary_epsilon
            and planar_error
            <= self.config.cavity_active_view_plateau_planar_tolerance_m
            + boundary_epsilon
            and target_shortfall
            <= self.config.cavity_active_view_plateau_max_target_shortfall_m
            + boundary_epsilon
            and rotation_error
            <= self.config.rotation_tolerance_rad + boundary_epsilon
            and plateau_safe_height
        )
        passed = primary_passed or plateau_passed
        safe_height = primary_safe_height or (
            plateau_passed and plateau_safe_height
        )
        return passed, {
            "strategy": "componentwise_safe_high_rgbd_observation_gate",
            "completion_kind": (
                "primary_component_gate"
                if primary_passed
                else "bounded_proprioceptive_plateau_gate"
                if plateau_passed
                else "none"
            ),
            "goal_minus_current_world_m": residual.tolist(),
            "position_error_m": position_error,
            "planar_error_m": planar_error,
            "planar_tolerance_m": float(
                self.config.cavity_active_view_planar_tolerance_m
            ),
            "signed_target_shortfall_m": signed_target_shortfall,
            "target_shortfall_m": target_shortfall,
            "maximum_target_shortfall_m": float(
                self.config.cavity_active_view_max_target_shortfall_m
            ),
            "rotation_error_rad": rotation_error,
            "rotation_tolerance_rad": float(self.config.rotation_tolerance_rad),
            "source_sensor_top_z_m": source_top,
            "minimum_safe_observation_z_m": minimum_safe_z,
            "plateau_minimum_safe_observation_z_m": (
                plateau_minimum_safe_z
            ),
            "current_z_m": float(current.position[2]),
            "safe_height_passed": safe_height,
            "primary_gate_passed": primary_passed,
            "plateau_gate": {
                "passed": plateau_passed,
                "planar_tolerance_m": float(
                    self.config.cavity_active_view_plateau_planar_tolerance_m
                ),
                "maximum_target_shortfall_m": float(
                    self.config.cavity_active_view_plateau_max_target_shortfall_m
                ),
                "window_ticks": int(self.config.contact_stall_ticks),
                "window_stalled": window_stalled,
                "maximum_cartesian_span_m": float(
                    self.config.cavity_active_view_plateau_cartesian_span_m
                ),
                "cartesian_span_m": (
                    cartesian_span if np.isfinite(cartesian_span) else None
                ),
                "safe_height_passed": plateau_safe_height,
            },
            "passed": passed,
            "authorization": "fresh_observe_selected_only",
        }

    def _act_cavity_active_view(
        self,
        step: SkillStep,
        observation: SensorObservation,
    ) -> ControlDecision:
        """Execute vertical-first views and accept only a fresh valid 3-D OBB."""

        vertical_pose = self._cavity_active_view_vertical_pose
        view_pose = self._cavity_active_view_pose
        trace = self._cavity_active_view_trace
        if vertical_pose is None or view_pose is None or trace is None:
            return self._fail("cavity active-view state is incomplete")
        if self._phase == "cavity_active_view_vertical":
            if not self._pose_reached(
                observation.robot.ee_pose,
                vertical_pose,
                position_tolerance_m=self.config.grasp_position_tolerance_m,
            ):
                return self._motion(
                    vertical_pose,
                    observation,
                    self._released_gripper_command(),
                )
            trace["vertical_completion_world_m"] = (
                observation.robot.ee_pose.position.tolist()
            )
            trace["vertical_completion_xy_error_m"] = float(
                np.linalg.norm(
                    observation.robot.ee_pose.position[:2]
                    - vertical_pose.position[:2]
                )
            )
            self._set_phase("cavity_active_view_traverse")
            return self._motion(
                view_pose,
                observation,
                self._released_gripper_command(),
            )

        views = trace.get("views")
        view_trace = (
            views[-1]
            if isinstance(views, list) and views and isinstance(views[-1], dict)
            else None
        )
        if view_trace is None:
            return self._fail("cavity active-view trace is incomplete")
        view_reached, completion_gate = (
            self._cavity_active_view_traverse_reached(
                observation.robot.ee_pose,
                view_pose,
            )
        )
        view_trace["high_observation_completion_gate"] = completion_gate
        if not view_reached:
            return self._motion(
                view_pose,
                observation,
                self._released_gripper_command(),
            )

        view_trace["measured_pose_world_m"] = (
            observation.robot.ee_pose.position.tolist()
        )
        view_trace["measured_height_error_m"] = float(
            observation.robot.ee_pose.position[2] - view_pose.position[2]
        )

        invalidated_queries = [step.subject]
        if step.selector is not None:
            invalidated_queries.extend(step.selector.references)
        invalidated_queries = list(dict.fromkeys(invalidated_queries))
        invalidate = getattr(self.perception, "invalidate_dynamic", None)
        if callable(invalidate):
            for query in invalidated_queries:
                invalidate(query)
        view_trace["invalidated_queries"] = invalidated_queries
        self._pick_cavity_reference = None

        selected = getattr(self.perception, "observe_selected", None)
        snapshot = SceneSnapshot(self._observation_timestamp_s, {})
        selection_error: str | None = None
        if step.selector is not None and callable(selected):
            try:
                snapshot = selected(observation, step.subject, step.selector)
            except LookupError as exc:
                selection_error = f"lookup_error:{type(exc).__name__}"
        else:
            selection_error = "observe_selected_unavailable"
        fresh_source = snapshot.objects.get(step.subject)
        fresh_reference = self._selector_cavity_reference(step)
        view_trace["selection_error"] = selection_error
        view_trace["fresh_source_geometry"] = self._sensor_geometry_trace(
            fresh_source
        )
        view_trace["fresh_reference_geometry"] = self._sensor_geometry_trace(
            fresh_reference
        )
        anchor_distance: float | None = None
        if fresh_source is not None and self._pick_anchor_center_world is not None:
            anchor_distance = float(
                np.linalg.norm(
                    fresh_source.centroid_world - self._pick_anchor_center_world
                )
            )
        view_trace["fresh_source_anchor_distance_m"] = anchor_distance
        view_trace["fresh_source_anchor_radius_m"] = (
            float(self._pick_selector_reacquire_radius_m)
            if self._pick_selector_reacquire_radius_m is not None
            else None
        )

        rejection: str | None = None
        if fresh_source is None:
            rejection = "fresh_relational_source_missing"
        elif (
            anchor_distance is None
            or self._pick_selector_reacquire_radius_m is None
            or anchor_distance > self._pick_selector_reacquire_radius_m
        ):
            rejection = "fresh_source_outside_initial_identity_anchor"
        elif fresh_reference is None:
            rejection = "fresh_relation_still_lacks_3d_reference"
        else:
            self._pick_cavity_reference = fresh_reference
            self._cavity_reference_from_active_view = True
            preview_rotation = (
                self._pick_rotation_anchor_world
                if self._pick_rotation_anchor_world is not None
                else observation.robot.ee_pose.rotation
            )
            try:
                _, _, preview = self._rim_pinch_target(
                    fresh_source,
                    observation,
                    preview_rotation,
                )
            except ValueError as exc:
                rejection = f"fresh_reference_failed_3d_gate:{exc}"
            else:
                if preview.get("rim_strategy") != "cavity_width_axis":
                    rejection = "fresh_reference_failed_existing_3d_cavity_gate"
        view_trace["existing_3d_cavity_gate_passed"] = rejection is None
        view_trace["rejection_reason"] = rejection

        attempt_count = len(self._grasp_target_attempts)
        primary_before_target = (
            self._cavity_primary_radial_world.copy()
            if self._cavity_primary_radial_world is not None
            else None
        )
        if rejection is None and fresh_source is not None:
            self._pick_source_geometry = self._stabilize_cavity_retry_source(
                fresh_source
            )
            try:
                self._set_pick_targets(self._pick_source_geometry, observation)
            except ValueError as exc:
                rejection = f"fresh_reference_failed_target_gate:{exc}"
            else:
                if not self._active_cavity_rim:
                    rejection = "fresh_reference_would_fall_back_to_free_space"
        view_trace["fresh_target_gate_passed"] = rejection is None
        if rejection is None:
            assert self._secondary_pose is not None
            attempt = self._grasp_target_attempts[-1]
            trace["outcome"] = "fresh_3d_cavity_reference_accepted"
            trace["accepted_view_index"] = int(self._cavity_active_view_index)
            trace["post_view_egress"] = {
                "strategy": "direct_to_approach_aligned_far_high_pregrasp",
                "start_world_m": observation.robot.ee_pose.position.tolist(),
                "target_world_m": self._secondary_pose.position.tolist(),
                "selected_side_profile": attempt.get(
                    "cavity_selected_side_profile"
                ),
                "reference_origin": attempt.get("cavity_reference_origin"),
            }
            attempt["cavity_active_view"] = trace
            pregrasp_pose = self._secondary_pose
            self._clear_cavity_active_view_state()
            self._set_phase("move_pregrasp")
            return self._motion(
                pregrasp_pose,
                observation,
                self._released_gripper_command(),
            )

        # Never preserve an invalid relation reference or the preview's target.
        self._pick_cavity_reference = None
        self._cavity_reference_from_active_view = False
        self._active_cavity_rim = False
        self._motion_pose = None
        self._secondary_pose = None
        del self._grasp_target_attempts[attempt_count:]
        self._cavity_primary_radial_world = primary_before_target
        view_trace["rejection_reason"] = rejection
        if self._cavity_active_view_index == 0:
            offset_pose = self._cavity_active_view_offset_pose()
            if offset_pose is not None:
                self._cavity_active_view_index = 1
                self._cavity_active_view_pose = offset_pose
                assert isinstance(views, list)
                views.append(
                    {
                        "index": 1,
                        "kind": "offset_toward_initial_proprio_view",
                        "target_world_m": offset_pose.position.tolist(),
                        "constant_height_z_m": float(offset_pose.position[2]),
                    }
                )
                self._set_phase("cavity_active_view_traverse")
                return self._motion(
                    offset_pose,
                    observation,
                    self._released_gripper_command(),
                )

        trace["outcome"] = "failed_safe_high_no_valid_3d_cavity_reference"
        trace["accepted_view_index"] = None
        trace["terminal_pose_world_m"] = (
            observation.robot.ee_pose.position.tolist()
        )
        source = self._cavity_active_view_source
        failure_attempt: dict[str, object] = {
            "attempt": self._grasp_retry_index,
            "mode": self._grasp_mode.value,
            "active_cavity_rim": False,
            "rim_strategy": "cavity_active_view_failed",
            "cavity_active_view": trace,
        }
        if source is not None:
            failure_attempt.update(
                {
                    "sensor_center_world_m": source.centroid_world.tolist(),
                    "sensor_extents_m": source.extents_m.tolist(),
                }
            )
        self._grasp_target_attempts.append(failure_attempt)
        self._cavity_active_view_source = None
        self._cavity_active_view_safe_z_m = None
        self._cavity_active_view_vertical_pose = None
        self._cavity_active_view_pose = None
        self._cavity_active_view_trace = None
        return self._fail(
            "strict IN selector did not yield a valid RGB-D cavity OBB after "
            "two safe high active views; free-space descent is forbidden"
        )

    def _prefetch_next_place_destination(
        self,
        observation: SensorObservation,
    ) -> None:
        """Freeze the next support from the unobstructed pre-pick RGB-D view."""

        if self._spec is None or self._skill_index + 1 >= len(self._spec.steps):
            return
        next_index = self._skill_index + 1
        next_step = self._spec.steps[next_index]
        if (
            next_step.kind
            not in {
                SkillKind.PLACE_ON,
                SkillKind.PLACE_IN,
                SkillKind.PLACE_RELATIVE,
                SkillKind.STACK,
            }
            or next_step.target is None
        ):
            return
        if next_index in self._prefetched_place_destinations:
            return
        if (
            self._prefetched_place_destination is not None
            and self._prefetched_place_target == next_step.target
            and self._prefetched_place_selector == next_step.target_selector
            and self._prefetched_place_relation == next_step.relation
        ):
            return
        snapshot, grounding = self._observe_place_destination(
            next_step,
            observation,
        )
        destination = snapshot.objects.get(next_step.target)
        if destination is None:
            return
        self._prefetched_place_destination = destination
        self._prefetched_place_target = next_step.target
        self._prefetched_place_selector = next_step.target_selector
        self._prefetched_place_relation = next_step.relation
        self._prefetched_place_grounding = grounding
        self._prefetched_place_destinations[next_index] = (
            next_step.target,
            next_step.target_selector,
            next_step.relation,
            destination,
            grounding,
        )

    def _prefetch_future_ranked_place_destinations(
        self,
        observation: SensorObservation,
    ) -> None:
        """Bind all future ranked supports before earlier placements occlude them."""

        if self._spec is None:
            return
        placement_kinds = {
            SkillKind.PLACE_ON,
            SkillKind.PLACE_IN,
            SkillKind.PLACE_RELATIVE,
            SkillKind.STACK,
        }
        for index in range(self._skill_index + 1, len(self._spec.steps)):
            step = self._spec.steps[index]
            if (
                step.kind not in placement_kinds
                or step.target is None
                or step.target_selector is None
                or index in self._prefetched_place_destinations
            ):
                continue
            snapshot, grounding = self._observe_place_destination(
                step,
                observation,
            )
            destination = snapshot.objects.get(step.target)
            if destination is None:
                continue
            self._prefetched_place_destinations[index] = (
                step.target,
                step.target_selector,
                step.relation,
                destination,
                grounding,
            )

    def _prefetched_destination_for(
        self,
        step: SkillStep,
    ) -> tuple[SceneObject, str | None] | None:
        prefetched = self._prefetched_place_destinations.get(self._skill_index)
        if prefetched is not None:
            target, selector, relation, destination, grounding = prefetched
            if (
                target == step.target
                and selector == step.target_selector
                and relation == step.relation
            ):
                return destination, grounding
        if (
            self._prefetched_place_destination is not None
            and self._prefetched_place_target == step.target
            and self._prefetched_place_selector == step.target_selector
            and self._prefetched_place_relation == step.relation
        ):
            return (
                self._prefetched_place_destination,
                self._prefetched_place_grounding,
            )
        return None

    def _prefetch_future_repeated_pick(
        self,
        current_step: SkillStep,
        observation: SensorObservation,
    ) -> None:
        """Bind the next independent source before moving the current object.

        Distinct language labels can share a visual category (for example two
        differently coloured mugs).  Preserve the initial instance association
        so the second pick cannot relabel the already placed first object.
        A source used as an intervening placement target is not stationary and
        must instead be grounded after that interaction.
        """

        if self._spec is None:
            return
        for index in range(self._skill_index + 1, len(self._spec.steps)):
            future = self._spec.steps[index]
            if future.kind is not SkillKind.PICK:
                continue
            if index in self._prefetched_pick_sources:
                return
            repeated_ranked = (
                future.subject == current_step.subject
                and future.selector is not None
            )
            independent_distinct = (
                future.subject != current_step.subject
                and future.subject.split()[-1] == current_step.subject.split()[-1]
                and future.selector is None
                and future.relation != "carry_stack"
                and not any(
                    earlier.target == future.subject
                    for earlier in self._spec.steps[self._skill_index + 1:index]
                )
            )
            if not (repeated_ranked or independent_distinct):
                return
            snapshot = self._observe_entity(
                observation,
                future.subject,
                future.selector,
            )
            selected = snapshot.objects.get(future.subject)
            current_source = self._pick_source_geometry
            if selected is not None and independent_distinct:
                if current_source is None or float(np.linalg.norm(
                    selected.centroid_world - current_source.centroid_world
                )) <= max(
                    self._selector_reacquire_radius(selected),
                    self._selector_reacquire_radius(current_source),
                ):
                    # An initial cross-label alias is not identity evidence.
                    return
            if selected is not None:
                self._prefetched_pick_sources[index] = selected
            return

    def _observe_entity(
        self,
        observation: SensorObservation,
        label: str,
        selector: EntitySelector | None,
    ) -> SceneSnapshot:
        """Resolve one language binding from current RGB-D geometry."""

        selected = getattr(self.perception, "observe_selected", None)
        if selector is not None and callable(selected):
            return selected(observation, label, selector)
        if selector is not None:
            # A legacy perception adapter cannot prove an instance selector.
            # Returning an empty snapshot makes the ordinary miss budget fail
            # closed rather than silently accepting an arbitrary singleton.
            return SceneSnapshot(self._observation_timestamp_s, {})
        return self.perception.observe(observation, (label,))

    def _observe_place_destination(
        self,
        step: SkillStep,
        observation: SensorObservation,
    ) -> tuple[SceneSnapshot, str | None]:
        """Ground a support or a named fixture subregion from current RGB-D.

        Caddy compartment words are expressed in the caddy's local frame, so
        applying them as world-X/Y offsets is incorrect when the fixture is
        yawed.  Likewise, the cabinet shelf is an internal support rather than
        the cabinet roof.  For these phrases we query the named visible region
        directly and require it to be a contained subregion of an independently
        grounded fixture.  Missing or whole-fixture detections remain misses.
        """

        assert step.target is not None
        if step.kind is SkillKind.PLACE_IN and step.target == "microwave":
            from .microwave_interior import observed_microwave_cavity

            reference = (self._episode_reset_position_world
                         if self._episode_reset_position_world is not None
                         else observation.robot.ee_pose.position)
            try:
                cavity, trace = observed_microwave_cavity(self.perception, observation, reference)
            except (LookupError, ValueError) as exc:
                trace = {"rejection": str(exc)}
            else:
                getattr(self.perception, "_selector_diagnostics", []).append({"kind": "microwave_cavity_grounding", **trace})
                return (SceneSnapshot(self._observation_timestamp_s, {step.target: cavity}),
                        "sensor-local microwave cavity")
            getattr(self.perception, "_selector_diagnostics", []).append({"kind": "microwave_cavity_grounding", **trace})
        if step.kind is SkillKind.PLACE_IN and "drawer" in step.target.split():
            snapshot = self._observe_entity(observation, step.target, step.target_selector)
            fixture = snapshot.objects.get(step.target)
            if fixture is not None:
                from ..perception.adapters import coerce_rgbd_frame
                from ..perception.geometry import backproject_frame
                from .drawer_floor import drawer_floor_region

                clouds = [backproject_frame(coerce_rgbd_frame(frame, name=name), stride=2).points_world
                          for name, frame in observation.cameras.items()]
                try:
                    floor = drawer_floor_region(np.concatenate(clouds), fixture)
                except ValueError:
                    pass
                else:
                    source = self._pick_source_geometry
                    if source is not None:
                        from .drawer_interior import observed_drawer_interior

                        try:
                            contained = observed_drawer_interior(
                                observation, step.target, floor, source,
                                reference=next((anchor.target for level, anchor
                                    in self._drawer_episode_anchors.items()
                                    if level in step.target.split()), None),
                            )
                        except (LookupError, ValueError):
                            pass
                        else:
                            floor = contained
                    return (SceneSnapshot(self._observation_timestamp_s, {step.target: floor}),
                            "sensor-local open drawer floor")
            if self._pick_source_geometry is not None and step.target_selector is None:
                from .drawer_interior import observed_handle_drawer_interior

                try:
                    floor = observed_handle_drawer_interior(
                        observation, step.target, self._pick_source_geometry, preferred=fixture,
                        reference=next((anchor.target for level, anchor in self._drawer_episode_anchors.items()
                                        if level in step.target.split()), None),
                    )
                except (LookupError, ValueError):
                    pass
                else:
                    return (SceneSnapshot(self._observation_timestamp_s, {step.target: floor}),
                            "sensor-local open drawer floor")
            return snapshot, None
        if (step.kind is SkillKind.PLACE_ON and step.subject == "wine bottle"
                and step.target in {"rack", "wine rack"} and step.target_selector is None):
            from .slatted_rack import slatted_rack_surface

            try:
                rack = slatted_rack_surface(observation)
            except (LookupError, ValueError):
                pass
            else:
                return (SceneSnapshot(self._observation_timestamp_s, {step.target: rack}),
                        "sensor-local slatted rack")
        if (step.target == "cabinet shelf"
                and (step.subject == "book" or self._is_pan_handle_subject(step.subject))
                and step.relation in {"upper_shelf", "lower_shelf"}):
            from .fixture_surfaces import shelf_region_from_points

            fixtures = self.perception.observe(
                observation, ("cabinet", "wooden cabinet", "cabinet shelf")
            )
            for fixture in sorted(fixtures.objects.values(), key=lambda item: -item.point_count):
                if fixture.surface_points_world is None:
                    continue
                try:
                    bay = shelf_region_from_points(
                        fixture.surface_points_world, step.relation,
                        observation.robot.ee_pose.position,
                    )
                except ValueError:
                    if (not self._is_pan_handle_subject(step.subject)
                            or fixture.name != "cabinet shelf"
                            or self._pick_source_geometry is None):
                        continue
                    from .partial_shelf import partial_shelf_region_from_points

                    try:
                        bay = partial_shelf_region_from_points(
                            fixture.surface_points_world, step.relation,
                            observation.robot.ee_pose.position,
                            float(self._pick_source_geometry.bounds_min_world[2]),
                        )
                    except ValueError:
                        continue
                return (SceneSnapshot(self._observation_timestamp_s, {step.target: bay}),
                        f"sensor-local open shelf {step.relation}")
        if step.target == "cabinet" and step.kind is SkillKind.PLACE_ON:
            from .fixture_surfaces import cabinet_top_surface

            try:
                roof = cabinet_top_surface(observation)
            except (LookupError, ValueError):
                pass
            else:
                return (
                    SceneSnapshot(self._observation_timestamp_s, {"cabinet": roof}),
                    "sensor-local cabinet top",
                )
        region_spec = _SEMANTIC_PLACEMENT_REGIONS.get(
            (step.target, step.relation or "")
        )
        if step.target == "microwave" and step.kind is not SkillKind.PLACE_IN:
            region_spec = None
        if region_spec is None:
            return (
                self._observe_entity(
                    observation,
                    step.target,
                    step.target_selector,
                ),
                None,
            )
        # A ranked target selector and a named fixture subregion would require
        # joint instance/part association.  No official command needs both;
        # refuse instead of binding the part to an arbitrary repeated fixture.
        if step.target_selector is not None:
            return SceneSnapshot(self._observation_timestamp_s, {}), None

        if step.target == "caddy":
            fixture_snapshot = self.perception.observe(observation, ("caddy",))
            fixture = fixture_snapshot.objects.get("caddy")
            if fixture is not None:
                region = self._caddy_local_compartment(
                    fixture, step.relation or "", observation
                )
                if region is not None:
                    return (
                        SceneSnapshot(self._observation_timestamp_s, {"caddy": region}),
                        f"sensor-local caddy {step.relation}",
                    )

        region_queries, fixture_queries = region_spec
        snapshot = self.perception.observe(
            observation,
            (*region_queries, *fixture_queries),
        )
        regions = [
            (query, snapshot.objects[query])
            for query in region_queries
            if query in snapshot.objects
        ]
        fixtures = [
            snapshot.objects[query]
            for query in fixture_queries
            if query in snapshot.objects
        ]
        for query, region in sorted(
            regions,
            key=lambda item: (-float(item[1].confidence), item[0]),
        ):
            if not any(
                self._semantic_region_is_contained(region, fixture)
                for fixture in fixtures
            ):
                continue
            rebound = SceneObject(
                name=step.target,
                centroid_world=region.centroid_world,
                axes_world=region.axes_world,
                extents_m=region.extents_m,
                bounds_min_world=region.bounds_min_world,
                bounds_max_world=region.bounds_max_world,
                confidence=region.confidence,
                point_count=region.point_count,
            )
            return (
                SceneSnapshot(self._observation_timestamp_s, {step.target: rebound}),
                query,
            )
        return SceneSnapshot(self._observation_timestamp_s, {}), None

    @staticmethod
    def _caddy_local_compartment(
        fixture: SceneObject,
        relation: str,
        observation: SensorObservation,
    ) -> SceneObject | None:
        """Ground the four physical pockets in a calibrated fixture frame.

        The elongated desk-caddy family has two end pockets and a divided
        middle third. Its fixed construction supplies ratios; current RGB-D
        supplies position, yaw, size and the viewing side.
        """
        name = relation.removesuffix("_compartment")
        if name not in {"front", "back", "left", "right"}:
            return None
        vertical = int(np.argmax(np.abs(fixture.axes_world[2])))
        if abs(fixture.axes_world[2, vertical]) < 0.95:
            return None
        horizontal = sorted(
            (i for i in range(3) if i != vertical),
            key=lambda i: fixture.extents_m[i],
        )
        short_index, long_index = horizontal
        short, long = fixture.extents_m[horizontal]
        if not (0.10 <= short <= 0.25 and 2.0 <= long / short <= 3.5):
            return None
        frames = [
            frame for key, frame in getattr(observation, "cameras", {}).items()
            if "agent" in key.lower()
        ]
        if not frames:
            return None
        view = frames[0].world_from_camera.position - fixture.centroid_world
        view[2] = 0.0
        front = fixture.axes_world[:, short_index].copy()
        front[2] = 0.0
        front /= np.linalg.norm(front)
        if abs(float(np.dot(front, view))) < 0.03:
            return None
        if np.dot(front, view) < 0:
            front *= -1.0
        up = np.array((0.0, 0.0, 1.0))
        left = np.cross(front, up)
        center = fixture.centroid_world.copy()
        height = float(fixture.extents_m[vertical])
        if name in {"front", "back"}:
            center += front * (0.22 * short) * (1 if name == "front" else -1)
            size = np.array((0.36 * short, 0.29 * long, 0.90 * height))
        else:
            center += left * (0.325 * long) * (1 if name == "left" else -1)
            size = np.array((0.80 * short, 0.29 * long, 0.90 * height))
        if name == "front":
            size[2] = 0.45 * height
        # Every pocket shares the visible base; the front pocket has a lower
        # wall, so both its centre and height are reduced together.
        center[2] = fixture.bounds_min_world[2] + 0.04 * height + size[2] / 2
        axes = np.column_stack((front, -left, up))
        half_world = np.abs(axes) @ (size / 2)
        return SceneObject(
            name="caddy", centroid_world=center, axes_world=axes,
            extents_m=size, bounds_min_world=center - half_world,
            bounds_max_world=center + half_world,
            confidence=fixture.confidence, point_count=fixture.point_count,
        )

    @staticmethod
    def _semantic_region_is_contained(
        region: SceneObject,
        fixture: SceneObject,
    ) -> bool:
        """Reject a detector that merely repeats the whole fixture box."""

        margin_m = 0.04
        center_inside = bool(
            np.all(region.centroid_world >= fixture.bounds_min_world - margin_m)
            and np.all(region.centroid_world <= fixture.bounds_max_world + margin_m)
        )
        if not center_inside:
            return False
        region_size = np.maximum(
            region.bounds_max_world - region.bounds_min_world,
            1e-6,
        )
        fixture_size = np.maximum(
            fixture.bounds_max_world - fixture.bounds_min_world,
            1e-6,
        )
        # A shelf may span almost the full cabinet width, so require smaller
        # footprint area rather than demanding every individual axis shrink.
        footprint_ratio = float(
            np.prod(region_size[:2]) / np.prod(fixture_size[:2])
        )
        volume_ratio = float(np.prod(region_size) / np.prod(fixture_size))
        return bool(footprint_ratio <= 0.80 and volume_ratio <= 0.65)

    @staticmethod
    def _semantic_relation_direction_world(
        relation: str | None,
        observation: SensorObservation,
    ) -> np.ndarray | None:
        """Resolve language directions from calibrated agent-view axes."""

        family = {
            "left_of": "left",
            "right_of": "right",
            "front_of": "front",
            "left_compartment": "left",
            "right_compartment": "right",
            "front_compartment": "front",
            "back_compartment": "back",
            "front": "front",
            "back": "back",
        }.get(relation or "")
        if family is None:
            return None

        frames = sorted(
            (
                item
                for item in getattr(observation, "cameras", {}).items()
                if "agent" in item[0].lower()
            ),
            key=lambda item: item[0],
        )
        base_direction: np.ndarray | None = None
        if frames:
            rotation = frames[0][1].world_from_camera.rotation
            if family in {"left", "right"}:
                # Camera +X is image-right in the calibrated pinhole frame.
                base_direction = rotation[:, 0].copy()
                if family == "left":
                    base_direction *= -1.0
            else:
                # Optical +Z points from the agent camera into the scene;
                # tabletop FRONT points back toward the observing agent.
                base_direction = -rotation[:, 2].copy()
                if family == "back":
                    base_direction *= -1.0
        if base_direction is None:
            base_direction = {
                "left": np.array((0.0, -1.0, 0.0)),
                "right": np.array((0.0, 1.0, 0.0)),
                "front": np.array((1.0, 0.0, 0.0)),
                "back": np.array((-1.0, 0.0, 0.0)),
            }[family]
        base_direction[2] = 0.0
        norm = float(np.linalg.norm(base_direction))
        if norm < 1e-8:
            return None
        return base_direction / norm

    def _rim_held_visual_plausibility(
        self,
        detected: SceneObject | None,
        observation: SensorObservation,
        *,
        expected_offset_world: np.ndarray | None = None,
    ) -> tuple[bool, dict[str, object]]:
        """Gate a post-retreat visual centre before updating a rim offset.

        A real bowl centre remains roughly one radius from its rim contact,
        whether the bowl stays upright or swings below the wrist.  A detector
        crop dominated by the hand instead lands very close to the EE.  This
        sensor/proprio gate distinguishes those cases without contacts or
        simulator object state.
        """

        trace: dict[str, object] = {"rim_visual_offset_accepted": False}
        if detected is None:
            trace["rim_visual_offset_reason"] = "no_rgbd_candidate"
            return False, trace
        if self._pick_source_geometry is None:
            trace["rim_visual_offset_reason"] = "missing_initial_rim_geometry"
            return False, trace
        vertical_axis = int(
            np.argmax(np.abs(self._pick_source_geometry.axes_world[2, :]))
        )
        planar_axes = tuple(index for index in range(3) if index != vertical_axis)
        planar_diameter = float(
            np.min(self._pick_source_geometry.extents_m[list(planar_axes)])
        )
        min_distance = (
            planar_diameter * self.config.rim_held_visual_min_distance_ratio
        )
        max_prediction_shift = (
            planar_diameter
            * self.config.rim_held_visual_max_prediction_shift_ratio
        )
        max_orthogonal_residual = (
            planar_diameter * self.config.rim_held_visual_max_orthogonal_ratio
        )
        offset = detected.centroid_world - observation.robot.ee_pose.position
        distance = float(np.linalg.norm(offset))
        below_ee = float(-offset[2])
        trace.update(
            {
                "rim_visual_candidate_offset_world_m": offset.tolist(),
                "rim_visual_candidate_distance_m": distance,
                "rim_visual_candidate_below_ee_m": below_ee,
                "rim_visual_min_candidate_distance_m": min_distance,
                "rim_visual_max_prediction_shift_m": max_prediction_shift,
                "rim_visual_max_orthogonal_residual_m": max_orthogonal_residual,
            }
        )
        if distance > self.config.holding_distance_m:
            trace["rim_visual_offset_reason"] = "candidate_too_far_from_ee"
            return False, trace
        if below_ee < self.config.rim_held_visual_min_below_ee_m:
            trace["rim_visual_offset_reason"] = "candidate_too_close_to_hand_center"
            return False, trace
        if distance < min_distance:
            trace["rim_visual_offset_reason"] = "candidate_too_close_to_rim_contact"
            return False, trace
        if expected_offset_world is None:
            if self._motion_pose is None:
                trace["rim_visual_offset_reason"] = "missing_initial_rim_geometry"
                return False, trace
            expected_offset_world = (
                self._pick_source_geometry.centroid_world - self._motion_pose.position
            )
        predicted_center = (
            observation.robot.ee_pose.position
            + np.asarray(expected_offset_world, dtype=np.float64)
        )
        expected_offset = np.asarray(expected_offset_world, dtype=np.float64)
        expected_horizontal_norm = float(np.linalg.norm(expected_offset[:2]))
        constrained_offset = offset.copy()
        if expected_horizontal_norm > 1e-6:
            radial_axis_xy = expected_offset[:2] / expected_horizontal_norm
            radial_component = float(np.dot(offset[:2], radial_axis_xy))
            constrained_offset[:2] = radial_axis_xy * radial_component
            dropped_component = offset - constrained_offset
            orthogonal_residual = float(np.linalg.norm(dropped_component[:2]))
            trace.update(
                {
                    "rim_visual_radial_axis_world": [
                        float(radial_axis_xy[0]),
                        float(radial_axis_xy[1]),
                        0.0,
                    ],
                    "rim_visual_orthogonal_residual_m": orthogonal_residual,
                    "rim_visual_dropped_orthogonal_world_m": (
                        dropped_component.tolist()
                    ),
                }
            )
            if orthogonal_residual > max_orthogonal_residual:
                trace["rim_visual_offset_reason"] = (
                    "candidate_too_far_from_rim_radial_plane"
                )
                return False, trace
        else:
            trace["rim_visual_offset_reason"] = "missing_rim_radial_axis"
            return False, trace
        trace["rim_visual_constrained_offset_world_m"] = constrained_offset.tolist()
        prediction_shift = float(
            np.linalg.norm(detected.centroid_world - predicted_center)
        )
        trace["rim_visual_prediction_shift_m"] = prediction_shift
        if prediction_shift > max_prediction_shift:
            trace["rim_visual_offset_reason"] = "candidate_inconsistent_with_rim_arc"
            return False, trace

        trace["rim_visual_offset_accepted"] = True
        trace["rim_visual_offset_reason"] = "rgbd_center_passed_rim_mechanics_gate"
        return True, trace

    def _refresh_rim_held_offset_for_place(
        self,
        step: SkillStep,
        observation: SensorObservation,
    ) -> bool:
        """Reobserve a settled carried bowl and close placement around its COM."""

        nearest = getattr(self.perception, "observe_nearest", None)
        nearest_point = (
            observation.robot.ee_pose.position + self._held_offset_world
        )
        snapshot = (
            nearest(observation, step.subject, nearest_point)
            if callable(nearest)
            else self.perception.observe(observation, (step.subject,))
        )
        detected = snapshot.objects.get(step.subject)
        accepted, trace = self._rim_held_visual_plausibility(
            detected,
            observation,
            expected_offset_world=self._held_offset_world,
        )
        trace["stage"] = "settled_preplace"
        trace["rim_visual_query_point_world_m"] = nearest_point.tolist()
        if not accepted and self._grasp_is_marginal and self._pick_initial_geometry is not None:
            from .carried_rim import fit_carried_rim
            from ..perception.adapters import coerce_rgbd_frame
            from ..perception.geometry import backproject_frame

            try:
                clouds = [backproject_frame(coerce_rgbd_frame(frame, name=name), stride=2).points_world
                          for name, frame in observation.cameras.items()]
                fitted = fit_carried_rim(
                    np.concatenate(clouds), self._pick_initial_geometry,
                    observation.robot.ee_pose, observation.robot.gripper_width_m,
                    self._held_offset_world,
                )
                fit_accepted, fit_trace = self._rim_held_visual_plausibility(
                    fitted, observation, expected_offset_world=self._held_offset_world,
                )
                trace["current_carried_rim_fit"] = {
                    "center_world_m": fitted.centroid_world.tolist(),
                    "extents_m": fitted.extents_m.tolist(),
                    "point_count": fitted.point_count,
                    "mechanics": fit_trace,
                }
                if fit_accepted:
                    trace["rejected_crop_reason"] = trace.get("rim_visual_offset_reason")
                    trace.update(fit_trace)
                    trace["rim_geometry_source"] = "current_rgbd_arc_without_public_hand"
                    detected = fitted
                    accepted = True
            except ValueError as error:
                trace["current_carried_rim_fit"] = {"rejected": str(error)}
        if accepted and detected is not None:
            destination = self._place_destination_geometry
            if destination is None:
                accepted = False
                trace["rim_visual_offset_accepted"] = False
                trace["rim_visual_offset_reason"] = "missing_frozen_place_geometry"
            else:
                self._held_offset_world = np.asarray(
                    trace["rim_visual_constrained_offset_world_m"],
                    dtype=np.float64,
                )
                self._held_offset_source = "rgbd_nearest_rim_preplace"
                self._held_geometry = detected
                self._set_place_targets(
                    step,
                    detected,
                    destination,
                    observation,
                    retain_phase=True,
                )
                trace["updated_held_offset_world_m"] = self._held_offset_world.tolist()
                assert self._motion_pose is not None
                trace["updated_release_ee_target_world_m"] = (
                    self._motion_pose.position.tolist()
                )
        if self._placement_target_attempts:
            self._placement_target_attempts[-1]["pre_release_visual_correction"] = trace
            self._placement_target_attempts[-1]["held_offset_source_after_correction"] = (
                self._held_offset_source
            )
        return accepted

    def _act_place(self, step: SkillStep, observation: SensorObservation) -> ControlDecision:
        assert step.target is not None
        self._update_rim_transfer_offset(observation)
        # Releasing is intentional: after the OPEN phase, retreat and visual
        # verification must continue even though the symbolic held slot is empty.
        if self._phase not in {"retreat", "shelf_retreat_up", "verify_place", "settle_after_release", "recenter_release", "peel_release", "release_clearance"} and self._held_subject != step.subject:
            return self._fail(
                f"cannot place {step.subject!r}: current held object is {self._held_subject!r}"
            )

        if self._phase == "acquire":
            self._release_settle_retry_used = False
            self._release_recenter_pose = None
            self._release_peel_pose = None
            source = self._tracked_held_source(observation)
            prefetched_destination = self._prefetched_destination_for(step)
            if prefetched_destination is not None:
                destination, destination_grounding = prefetched_destination
                self._place_destination_observation_stage = (
                    "pre_contact_unoccluded"
                    if destination_grounding == "precontact_rgbd_stove"
                    else "pre_pick_unoccluded"
                )
            else:
                snapshot, destination_grounding = self._observe_place_destination(
                    step,
                    observation,
                )
                destination = snapshot.objects.get(step.target)
                self._place_destination_observation_stage = "place_acquire"
            if source is None or destination is None:
                missing = [
                    name
                    for name, value in ((step.subject, source), (step.target, destination))
                    if value is None
                ]
                return self._perception_miss(
                    f"cannot localize placement entities: {', '.join(missing)}",
                    gripper=self._engaged_gripper_command(),
                )
            self._perception_misses = 0
            self._set_place_targets(
                step,
                source,
                destination,
                observation,
                destination_grounding=destination_grounding,
            )
            self._set_phase("move_transfer_clearance")

        elif self._phase in {
            "move_transfer_clearance",
            "move_preplace",
            "descend",
        } and self._should_refresh_visual():
            source = self._tracked_held_source(observation)
            prefetched_destination = self._prefetched_destination_for(step)
            if prefetched_destination is not None:
                # Preserve the unobstructed pre-pick support anchor.  A fresh
                # target box under the carried object/hand is less reliable;
                # the held source still advances from proprioception below.
                destination = self._place_destination_geometry
                destination_grounding = self._place_destination_grounding
            else:
                snapshot, destination_grounding = self._observe_place_destination(
                    step,
                    observation,
                )
                destination = snapshot.objects.get(step.target)
            if source is not None and destination is not None:
                self._set_place_targets(
                    step,
                    source,
                    destination,
                    observation,
                    retain_phase=True,
                    destination_grounding=destination_grounding,
                )

        if self._phase == "move_transfer_clearance":
            assert self._transfer_clearance_pose is not None
            if self._pose_reached(
                observation.robot.ee_pose,
                self._transfer_clearance_pose,
            ):
                self._set_phase("move_preplace")
            else:
                return self._motion(
                    self._transfer_clearance_pose,
                    observation,
                    self._engaged_gripper_command(),
                )

        if self._phase == "move_preplace":
            assert self._secondary_pose is not None
            bottom_drawer_staging = (step.target == "bottom drawer"
                and self._place_destination_grounding == "sensor-local open drawer floor"
                and self._phase_ticks >= 35)
            if self._pose_reached(observation.robot.ee_pose, self._secondary_pose,
                    position_tolerance_m=.012 if bottom_drawer_staging else None):
                if (
                    self._grasp_mode is GraspMode.RIM_PINCH
                    and not self._rim_preplace_visual_refreshed
                ):
                    self._set_phase("settle_preplace")
                    return self._tick_decision(
                        self._hold_action(self._engaged_gripper_command()),
                        "settling rim hold before RGB-D placement correction",
                    )
                self._visual_place_correction_active = False
                self._set_phase("lower_to_shelf_entry" if self._shelf_entry_pose is not None else "descend")
            else:
                return self._motion(
                    self._secondary_pose, observation, self._engaged_gripper_command()
                )

        if self._phase == "lower_to_shelf_entry":
            assert self._shelf_entry_pose is not None
            if self._pose_reached(observation.robot.ee_pose, self._shelf_entry_pose):
                self._set_phase("descend")
            else:
                return self._motion(self._shelf_entry_pose, observation, self._engaged_gripper_command())

        if self._phase == "settle_preplace":
            if self._phase_ticks < self.config.rim_preplace_settle_ticks:
                return self._tick_decision(
                    self._hold_action(self._engaged_gripper_command()),
                    "settling rim hold before RGB-D placement correction",
                )
            self._rim_preplace_visual_refreshed = True
            corrected = self._refresh_rim_held_offset_for_place(step, observation)
            if corrected:
                assert self._secondary_pose is not None
                if not self._pose_reached(
                    observation.robot.ee_pose,
                    self._secondary_pose,
                ):
                    self._visual_place_correction_active = True
                    self._set_phase("move_preplace")
                    return self._motion(
                        self._secondary_pose,
                        observation,
                        self._engaged_gripper_command(),
                    )
            self._visual_place_correction_active = False
            self._set_phase("descend")

        if self._phase == "descend":
            assert self._motion_pose is not None
            pose_reached = self._pose_reached(
                observation.robot.ee_pose,
                self._motion_pose,
                position_tolerance_m=(.010 if step.target == "bottom drawer"
                    and self._place_destination_grounding == "sensor-local open drawer floor" else None),
            )
            generic_contact_reached = self._contact_reached(
                observation.robot.ee_pose,
                self._motion_pose,
                tolerance_m=self.config.contact_position_tolerance_m,
            )
            rim_place_contact_reached = self._rim_place_contact_reached(
                observation.robot.ee_pose,
                self._motion_pose,
                observation.robot.gripper_width_m,
            )
            # Exact pose convergence remains a valid release condition.  A
            # stalled RIM_PINCH descent, however, must pass the directional
            # thin-wall gate rather than the legacy 75-mm all-direction gate.
            contact_reached = (
                rim_place_contact_reached
                if self._grasp_mode is GraspMode.RIM_PINCH
                else generic_contact_reached
            )
            if self._shelf_entry_pose is not None:
                # A blocked insertion outside the bay is not a release pose.
                contact_reached = False
                if (self._place_destination_grounding == "sensor-local microwave cavity"
                        and self._contact_window_stalled()
                        and self._contact_cartesian_span_m() <= .005):
                    cavity = self._place_destination_geometry
                    predicted = observation.robot.ee_pose.position + self._side_cavity_payload_offset
                    local = cavity.axes_world.T @ (predicted - cavity.centroid_world)
                    half = self._side_cavity_payload_half
                    clearance = cavity.extents_m / 2 - abs(local) - half
                    residual = self._motion_pose.position - observation.robot.ee_pose.position
                    inward = -cavity.axes_world[:, 1]
                    shortfall = float(residual @ inward)
                    transverse = float(np.linalg.norm(residual - shortfall*inward))
                    rotation_error = float(np.linalg.norm(self._rotation_vector(
                        self._motion_pose.rotation @ observation.robot.ee_pose.rotation.T)))
                    contact_reached = bool(
                        np.all(clearance >= .003)
                        and 0. <= shortfall <= .035 and transverse <= .010
                        and rotation_error <= .060
                        and .003 <= observation.robot.gripper_width_m <= .060
                    )
                    self._placement_target_attempts[-1]["cavity_contained_insertion"] = {
                        "accepted": contact_reached, "payload_clearance_m": clearance.tolist(),
                        "inward_shortfall_m": shortfall, "transverse_error_m": transverse,
                        "predicted_payload_center_world_m": predicted.tolist(),
                    }
            if pose_reached or contact_reached:
                if rim_place_contact_reached and not pose_reached:
                    self._record_rim_place_contact_completion(
                        observation.robot.ee_pose,
                        self._motion_pose,
                        observation.robot.gripper_width_m,
                    )
                # A fixed dwell can expire while a wide body is still pinched.
                # Establish measured jaw clearance before translating away.
                self._release_width_target_m = (
                    min(self.config.gripper_open_width_m,
                        float(observation.robot.gripper_width_m) + 0.010)
                    if self._grasp_mode is GraspMode.PINCH
                    and self._shelf_entry_pose is None else None
                )
                if (self._grasp_mode is GraspMode.PINCH
                        and step.kind is SkillKind.PLACE_IN
                        and step.target in {"basket", "bowl"}):
                    # Clear the solid payload before withdrawing. Ten mm
                    # beyond the engaged width can leave a tilted package
                    # supported on a finger even after an internal check.
                    self._release_width_target_m = .078
                self._set_phase("open")
            else:
                return self._motion(
                    self._motion_pose, observation, self._engaged_gripper_command()
                )

        if self._phase == "open":
            release_hold_ticks = (
                self.config.rim_pinch_release_hold_ticks
                if self._grasp_mode is GraspMode.RIM_PINCH
                else self.config.open_hold_ticks
            )
            shelf_released = (self._shelf_entry_pose is None
                              or observation.robot.gripper_width_m >= 0.038)
            if self._release_width_target_m == .078:
                release_hold_ticks = max(release_hold_ticks, 15)
            jaw_released = (self._release_width_target_m is None
                            or observation.robot.gripper_width_m >= self._release_width_target_m)
            if self._placement_target_attempts and self._release_width_target_m is not None:
                self._placement_target_attempts[-1]["release_aperture"] = {
                    "required_width_m": self._release_width_target_m,
                    "measured_width_m": float(observation.robot.gripper_width_m),
                    "ready": bool(jaw_released),
                }
            if self._phase_ticks >= release_hold_ticks and shelf_released and jaw_released:
                peel = None
                if step.kind is SkillKind.PLACE_IN and step.target == "basket":
                    from .release_geometry import open_jaw_peel_pose

                    peel = open_jaw_peel_pose(self._tracked_held_source(observation), observation.robot)
                self._held_subject = None
                self._held_from_cavity_rim = False
                invalidate = getattr(self.perception, "invalidate_dynamic", None)
                if callable(invalidate):
                    invalidate(step.subject)
                self._set_phase("retreat")
                if peel is not None:
                    self._release_peel_pose = peel
                    self._placement_target_attempts[-1]["open_palm_clearance_rotation_world"] = peel.rotation.tolist()
                    self._set_phase("release_clearance")
            else:
                command = (self._shelf_release_command(observation)
                           if self._shelf_entry_pose is not None else self._released_gripper_command())
                return self._tick_decision(self._hold_action(command))

        if self._phase == "release_clearance":
            if self._phase_ticks >= 30 or self._pose_reached(observation.robot.ee_pose, self._release_peel_pose):
                self._set_phase("retreat")
            else:
                return self._motion(self._release_peel_pose, observation, self._released_gripper_command())

        if self._phase == "retreat" and self._shelf_entry_pose is not None:
            if self._pose_reached(observation.robot.ee_pose, self._shelf_entry_pose):
                self._set_phase("shelf_retreat_up")
            else:
                return self._motion(self._shelf_entry_pose, observation,
                                    self._shelf_release_command(observation))

        if self._phase == "shelf_retreat_up":
            assert self._secondary_pose is not None
            if self._pose_reached(observation.robot.ee_pose, self._secondary_pose):
                self._set_phase("verify_place")
            else:
                return self._motion(self._secondary_pose, observation, self._released_gripper_command())

        if self._phase == "retreat":
            assert self._secondary_pose is not None
            # Placement is already released. A high cabinet can put the last
            # centimetres of the old preplace waypoint near the arm's reach
            # limit. Once the open hand has cleared the release site, proceed
            # to fresh visual verification without requiring its old yaw.
            released_clearance = (
                self._motion_pose is not None
                and self._grasp_mode in {GraspMode.PINCH, GraspMode.RIM_PINCH}
                and observation.robot.gripper_width_m >= 0.060
                and observation.robot.ee_pose.position[2] >= max(
                    self._motion_pose.position[2] + 0.75 * self.config.preplace_height_m,
                    self._secondary_pose.position[2] - 0.025,
                )
                and np.linalg.norm(
                    observation.robot.ee_pose.position[:2] - self._secondary_pose.position[:2]
                ) <= 0.025
            )
            if self._pose_reached(observation.robot.ee_pose, self._secondary_pose) or released_clearance:
                self._set_phase("verify_place")
            else:
                return self._motion(
                    self._secondary_pose, observation, self._released_gripper_command()
                )

        if self._phase == "recenter_release":
            assert self._release_recenter_pose is not None
            if np.linalg.norm(
                observation.robot.ee_pose.position - self._release_recenter_pose.position
            ) <= .003:
                self._set_phase("settle_after_release")
            elif self._phase_ticks >= 30:
                from .release_geometry import open_jaw_peel_pose

                nearest = getattr(self.perception, "observe_nearest", None)
                near_hand = None
                if callable(nearest):
                    try:
                        near_hand = nearest(observation, step.subject,
                            observation.robot.ee_pose.position + self._held_offset_world).objects.get(step.subject)
                    except LookupError:
                        pass
                self._release_peel_pose = open_jaw_peel_pose(near_hand, observation.robot)
                if self._release_peel_pose is not None:
                    if self._placement_target_attempts:
                        self._placement_target_attempts[-1]["release_peel_rotation_world"] = (
                            self._release_peel_pose.rotation.tolist())
                    self._set_phase("peel_release")
                else:
                    self._set_phase("settle_after_release")
            else:
                return self._motion(
                    self._release_recenter_pose, observation, self._released_gripper_command(),
                )

        if self._phase == "peel_release":
            assert self._release_peel_pose is not None
            if self._phase_ticks >= 30 or self._pose_reached(
                    observation.robot.ee_pose, self._release_peel_pose):
                self._set_phase("settle_after_release")
            else:
                return self._motion(self._release_peel_pose, observation,
                                    self._released_gripper_command())

        if self._phase == "settle_after_release":
            if self._phase_ticks < self.config.release_settle_retry_ticks:
                return self._tick_decision(self._hold_action(self._released_gripper_command()))
            invalidate = getattr(self.perception, "invalidate_dynamic", None)
            if callable(invalidate):
                invalidate(step.subject)
            self._set_phase("verify_place")

        if self._phase == "verify_place":
            if step.kind is SkillKind.STACK:
                nearest = getattr(self.perception, "observe_nearest", None)
                expected_center = (
                    self._motion_pose.position + self._held_offset_world
                    if self._motion_pose is not None
                    else observation.robot.ee_pose.position
                )
                stacked_snapshot = (
                    nearest(observation, step.subject, expected_center)
                    if callable(nearest)
                    else self.perception.observe(observation, (step.subject,))
                )
                detected_stack = stacked_snapshot.objects.get(step.subject)
                if not self.verifier.stacked(
                    detected_stack,
                    self._pick_source_geometry,
                    self._place_destination_geometry,
                    self.config,
                ):
                    return self._fail(
                        "sensor verification rejected stack height/footprint growth"
                    )
                assert detected_stack is not None
                assert self._pick_source_geometry is not None
                assert self._place_destination_geometry is not None
                self._formed_stack_geometry = detected_stack
                self._formed_stack_top_geometry = self._pick_source_geometry
                self._formed_stack_base_geometry = self._place_destination_geometry
                self._formed_stack_signature = (
                    step.subject,
                    step.selector,
                    step.target_selector,
                )
                return self._complete_skill(
                    "stack verified from fresh RGB-D height growth"
                )
            if self._active_carry_stack and step.relation == "carry_stack":
                nearest = getattr(self.perception, "observe_nearest", None)
                if not callable(nearest):
                    return self._fail(
                        "formed-stack placement requires nearest RGB-D reacquisition"
                    )
                expected_center = (
                    self._motion_pose.position + self._held_offset_world
                    if self._motion_pose is not None
                    else observation.robot.ee_pose.position
                )
                try:
                    group_snapshot = nearest(
                        observation,
                        step.subject,
                        expected_center,
                    )
                except LookupError:
                    group_snapshot = SceneSnapshot(self._observation_timestamp_s, {})
                group = group_snapshot.objects.get(step.subject)
                if not self.verifier.formed_stack_inside(
                    group,
                    self._place_destination_geometry,
                    self._formed_stack_top_geometry,
                    self._formed_stack_base_geometry,
                    self.config,
                ):
                    return self._fail(
                        "sensor verification rejected retained stack inside receptacle"
                    )
                if self._placement_target_attempts:
                    self._placement_target_attempts[-1][
                        "formed_stack_final_verified"
                    ] = True
                self._formed_stack_geometry = None
                self._formed_stack_top_geometry = None
                self._formed_stack_base_geometry = None
                self._formed_stack_signature = None
                self._active_carry_stack = False
                return self._complete_skill(
                    "formed stack verified inside receptacle from fresh RGB-D"
                )
            # Target rank qualifiers must remain active during visual
            # verification.  Querying only its label can jump from the
            # requested left/right support to a same-label distractor.
            # A released object can be above the tabletop and partly hidden
            # inside a receptacle. Reacquire around its last sensor-estimated
            # release position across both cameras; the general tabletop query
            # can otherwise select an untouched, similar-looking distractor.
            reacquire = getattr(self.perception, "observe_nearest_bound", None)
            if callable(reacquire) and self._motion_pose is not None:
                source_snapshot = reacquire(
                    observation,
                    step.subject,
                    self._place_expected_settled_center_world
                    if self._place_expected_settled_center_world is not None
                    else self._motion_pose.position + (
                        self._held_offset_world
                        if self._place_rotation_delta_world is None
                        else self._place_rotation_delta_world @ self._held_offset_world
                    ),
                )
            else:
                source_snapshot = self.perception.observe(observation, (step.subject,))
            if (
                self._place_destination_grounding is not None
                and self._place_destination_geometry is not None
            ):
                destination_snapshot = SceneSnapshot(
                    self._observation_timestamp_s,
                    {step.target: self._place_destination_geometry},
                )
            elif (
                step.target == "basket"
                and step.target_selector is None
                and self._place_destination_geometry is not None
                and callable(reacquire)
            ):
                destination_snapshot = reacquire(
                    observation, step.target,
                    self._place_destination_geometry.centroid_world,
                )
            else:
                destination_snapshot, _ = self._observe_place_destination(
                    step,
                    observation,
                )
            snapshot = SceneSnapshot(
                self._observation_timestamp_s,
                {
                    **source_snapshot.objects,
                    **destination_snapshot.objects,
                },
            )
            effective_relation = (
                None
                if self._place_destination_grounding is not None
                else step.relation
            )
            verified = self.verifier.placed(
                step.kind,
                snapshot,
                step.subject,
                step.target,
                self.config,
                effective_relation,
                self._semantic_relation_direction_world(
                    effective_relation,
                    observation,
                ),
            )
            if self._placement_target_attempts:
                self._placement_target_attempts[-1]["visual_verification"] = {
                    "success": bool(verified),
                    "objects": {
                        name: {
                            "center_world": obj.centroid_world.tolist(),
                            "bounds_min_world": obj.bounds_min_world.tolist(),
                            "bounds_max_world": obj.bounds_max_world.tolist(),
                            "confidence": float(obj.confidence),
                        }
                        for name, obj in snapshot.objects.items()
                    },
                }
            if not verified:
                if not self._release_settle_retry_used:
                    # A fresh crop can still show a falling or partly released
                    # object at the first retreat frame. Keep opening at the
                    # reached retreat pose before one fresh relation check.
                    # Successful checks retain their original trajectory.
                    self._release_settle_retry_used = True
                    nearest = getattr(self.perception, "observe_nearest", None)
                    if self._grasp_mode is GraspMode.PINCH and callable(nearest):
                        from .release_geometry import open_jaw_recenter_pose, open_jaw_peel_pose

                        try:
                            near_hand = nearest(
                                observation, step.subject,
                                observation.robot.ee_pose.position + self._held_offset_world,
                            ).objects.get(step.subject)
                        except LookupError:
                            near_hand = None
                        self._release_recenter_pose = open_jaw_recenter_pose(
                            near_hand, observation.robot,
                        )
                        # Rotate at the reached retreat position before a
                        # translational withdrawal can drag the released body
                        # onto the receptacle rim.
                        self._release_peel_pose = open_jaw_peel_pose(
                            near_hand, observation.robot,
                        )
                    if self._placement_target_attempts:
                        self._placement_target_attempts[-1]["release_settle_retry"] = {
                            "dwell_ticks": self.config.release_settle_retry_ticks,
                            "initial_verification": self._placement_target_attempts[-1]["visual_verification"],
                            "open_jaw_recenter_target_world_m": (
                                self._release_recenter_pose.position.tolist()
                                if self._release_recenter_pose is not None else None
                            ),
                        }
                    if self._release_peel_pose is not None:
                        if self._placement_target_attempts:
                            self._placement_target_attempts[-1]["direct_release_peel_rotation_world"] = (
                                self._release_peel_pose.rotation.tolist())
                        self._set_phase("peel_release")
                        return self._motion(self._release_peel_pose, observation,
                                            self._released_gripper_command())
                    if self._release_recenter_pose is not None:
                        self._set_phase("recenter_release")
                        return self._motion(
                            self._release_recenter_pose, observation, self._released_gripper_command(),
                        )
                    self._set_phase("settle_after_release")
                    return self._tick_decision(
                        self._hold_action(self._released_gripper_command()),
                        "allowing the released object to settle before a fresh RGB-D relation check",
                    )
                return self._fail(
                    f"sensor verification rejected {step.kind.value}({step.subject}, {step.target})"
                )
            return self._complete_skill("placement relation verified from RGB-D")

        return self._fail(f"invalid placement phase {self._phase!r}")

    def _update_rim_transfer_offset(self, observation: SensorObservation) -> None:
        if self._rim_transfer_rotation_reference is None or self._held_subject is None:
            return
        current = observation.robot.ee_pose.rotation
        delta = current @ self._rim_transfer_rotation_reference.T
        self._held_offset_world = delta @ self._held_offset_world
        self._rim_transfer_rotation_reference = current.copy()
        # A circular bowl's dimensions are invariant to this upright yaw.
        if self._held_geometry is not None:
            old = self._held_geometry
            axes = delta @ old.axes_world
            center = observation.robot.ee_pose.position + self._held_offset_world
            half = np.abs(axes) @ (old.extents_m / 2)
            self._held_geometry = SceneObject(old.name, center, axes, old.extents_m,
                center - half, center + half, old.confidence, old.point_count)

    def _tracked_held_source(self, observation: SensorObservation) -> SceneObject | None:
        """Rigidly propagate sensor-estimated geometry while the object is occluded."""

        if self._held_geometry is None or self._held_subject is None:
            return None
        previous = self._held_geometry
        center = observation.robot.ee_pose.position + self._held_offset_world
        translation = center - previous.centroid_world
        return SceneObject(
            name=self._held_subject,
            centroid_world=center,
            axes_world=previous.axes_world,
            extents_m=previous.extents_m,
            bounds_min_world=previous.bounds_min_world + translation,
            bounds_max_world=previous.bounds_max_world + translation,
            confidence=previous.confidence,
            point_count=previous.point_count,
        )

    def _set_pick_targets(
        self,
        source: SceneObject,
        observation: SensorObservation,
        *,
        retain_phase: bool = False,
    ) -> None:
        if not retain_phase:
            self._pick_initial_geometry = source
            self._flat_transfer_waypoints = []
        pan_handle_grasp = bool(
            self._grasp_mode is GraspMode.PINCH
            and self._is_pan_handle_subject(source.name)
        )
        mug_rim_grasp = bool(
            self._grasp_mode is GraspMode.PINCH
            and source.name.endswith("mug")
            and source.surface_points_world is not None
        )
        if mug_rim_grasp and retain_phase and self._grasp_target_attempts and (
            self._grasp_target_attempts[-1].get("mug_rim_surface_fit") is True
        ):
            # The rim is a local contact, not the object's crop centroid.
            # Keep that contact through one descent as the wrist occludes it.
            return
        if pan_handle_grasp and retain_phase:
            # A pan target is a solid local handle slot, not a crop centroid.
            # Freeze it throughout one approach; after a rejected proof lift
            # the normal retry path releases, retreats, invalidates DINO, and
            # supplies a fresh surface cloud for the next distinct slot.
            if self._motion_pose is None or self._secondary_pose is None:
                raise ValueError("pan handle refresh lacks a frozen grasp target")
            return
        # Keep one top-down wrist orientation for the whole pick.  Replacing
        # it with the instantaneous OSC orientation at each RGB-D refresh lets
        # small tracking errors accumulate into a tilted finger; the low side
        # then contacts the table roughly 30 mm before the desired thin-object
        # grasp plane.
        if retain_phase and self._motion_pose is not None:
            rotation = self._motion_pose.rotation
        elif self._pick_rotation_anchor_world is not None:
            rotation = self._pick_rotation_anchor_world
        else:
            rotation = observation.robot.ee_pose.rotation
        grasp_position = source.centroid_world.copy()
        target_details: dict[str, object] = {}
        self._active_cavity_rim = False
        self._active_pan_handle_slot_id = None
        self._pan_body_reference_world = None
        if self._grasp_mode is GraspMode.EXPAND:
            grasp_position[2] += self.config.expand_grasp_z_offset_m
        elif self._grasp_mode is GraspMode.RIM_PINCH:
            grasp_position, rotation, target_details = self._rim_pinch_target(
                source,
                observation,
                rotation,
            )
            self._active_cavity_rim = bool(
                target_details.get("rim_strategy") == "cavity_width_axis"
            )
            if self._active_cavity_rim:
                radial = np.asarray(
                    target_details.get("finger_axis_world"), dtype=np.float64
                )
                if radial.shape != (3,) or not np.all(np.isfinite(radial)):
                    raise ValueError("cavity rim target lacks a valid radial axis")
                radial_norm = float(np.linalg.norm(radial[:2]))
                if radial_norm < 0.8:
                    raise ValueError("cavity rim target lacks a valid radial axis")
                radial = radial / radial_norm
                if (
                    self._cavity_candidate_index == 0
                    and not retain_phase
                    and self._cavity_primary_radial_world is None
                ):
                    self._cavity_primary_radial_world = radial.copy()
                if self._cavity_primary_radial_world is not None:
                    target_details["cavity_primary_radial_world"] = (
                        self._cavity_primary_radial_world.tolist()
                    )
        elif pan_handle_grasp:
            grasp_position, rotation, target_details = self._pan_handle_target(
                source,
                observation,
                rotation,
            )
        else:
            grasp_position[2] = self._pinch_grasp_height(source)
            planar = np.sort(source.extents_m[:2])
            flat_package = bool(
                source.height_m <= 0.040 and planar[1] >= 0.055
                and planar[0] <= 0.055 and planar[1] >= 1.4 * planar[0]
            )
            if flat_package:
                grasp_position[2] += 0.004
            if not retain_phase:
                rotation = self._pinch_width_aligned_rotation(source, rotation)
                grasp_position, target_details = self._flat_pinch_clearance_target(
                    source, observation, grasp_position
                )
                if flat_package:
                    from ..perception.adapters import coerce_rgbd_frame
                    from ..perception.geometry import backproject_frame
                    from .gripper_clearance import choose_flat_pinch_pose

                    points = np.concatenate([
                        backproject_frame(coerce_rgbd_frame(frame, name=name), stride=3).points_world
                        for name, frame in observation.cameras.items()
                    ])
                    delta = points - source.centroid_world
                    object_local = delta @ source.axes_world
                    intended_contact = np.all(
                        np.abs(object_local) <= source.extents_m / 2.0 + 0.003, axis=1
                    )
                    local = ((np.linalg.norm(delta[:, :2], axis=1) < 0.20)
                             & (points[:, 2] > source.bounds_min_world[2] - 0.003)
                             & (points[:, 2] < source.bounds_max_world[2] + 0.27)
                             & ~intended_contact)
                    points = points[local]
                    if len(points) > 3500:
                        points = points[np.linspace(0, len(points) - 1, 3500, dtype=int)]
                    axes = np.argsort(source.extents_m[:2])
                    proposal, clearance_details = choose_flat_pinch_pose(
                        points, grasp_position, rotation,
                        source.axes_world[:, axes[1]], source.axes_world[:, axes[0]],
                        source.extents_m[axes[::-1]],
                    )
                    previous_offset = target_details.get("flat_pinch_clearance_offset_world_m")
                    unchanged = (np.allclose(proposal.position, grasp_position, atol=1e-12)
                                 and abs(clearance_details["flat_pinch_pitch_rad"] + np.deg2rad(20.)) < 1e-12)
                    if previous_offset is not None:
                        clearance_details["flat_pinch_clearance_offset_world_m"] = (
                            np.asarray(previous_offset) + proposal.position - grasp_position
                        ).tolist()
                    elif unchanged:
                        clearance_details.pop("flat_pinch_clearance_offset_world_m")
                    grasp_position, rotation = proposal.position, proposal.rotation
                    target_details.update(clearance_details)
                    from .flat_transfer import flat_pregrasp_transfer

                    high = grasp_position + np.array([0., 0., self.config.pregrasp_height_m])
                    self._flat_transfer_waypoints = flat_pregrasp_transfer(
                        points, observation.robot.ee_pose, Pose(high, rotation),
                        aperture=observation.robot.gripper_width_m,
                    )
                    if self._flat_transfer_waypoints:
                        target_details["flat_pregrasp_transfer"] = {
                            "strategy": "fixed_clearance_traverse_before_descent",
                            "waypoints_world_m": [pose.position.tolist() for pose in self._flat_transfer_waypoints],
                        }
            elif (
                self._motion_pose is not None
                and self._grasp_target_attempts
                and "flat_pinch_clearance_offset_world_m" in self._grasp_target_attempts[-1]
            ):
                grasp_position[:2] = self._motion_pose.position[:2]
        if mug_rim_grasp and not retain_phase:
            from .rim_geometry import fit_visible_upper_rim

            try:
                center_xy, radius, rim_top = fit_visible_upper_rim(
                    source.surface_points_world
                )
                radial = observation.robot.ee_pose.position[:2] - center_xy
                if np.linalg.norm(radial) < 0.020:
                    radial = rotation[:2, 1].copy()
                radial /= np.linalg.norm(radial)
                # Tool-local Y is Panda's closing axis. Put one open finger
                # inside the fitted rim and the other outside, then close.
                rotation = np.column_stack((
                    np.array((-radial[1], radial[0], 0.0)),
                    np.r_[radial, 0.0],
                    np.array((0.0, 0.0, -1.0)),
                ))
                grasp_position = np.r_[
                    center_xy + (radius - 0.002) * radial, rim_top - 0.006
                ]
                target_details.update({
                    "mug_rim_surface_fit": True,
                    "mug_rim_center_xy_m": center_xy.tolist(),
                    "mug_rim_radius_m": radius,
                    "mug_rim_top_z_m": rim_top,
                    "mug_rim_radial_xy": radial.tolist(),
                })
            except ValueError as exc:
                target_details["mug_rim_surface_fit"] = False
                target_details["mug_rim_fit_rejection"] = str(exc)
        self._motion_pose = Pose(grasp_position, rotation)
        pregrasp_position = grasp_position.copy()
        pregrasp_height_m = (
            self.config.cavity_pregrasp_height_m
            if self._active_cavity_rim
            else self.config.pregrasp_height_m
        )
        pregrasp_position[2] += pregrasp_height_m
        if (
            (self._grasp_mode is GraspMode.RIM_PINCH and not self._active_cavity_rim
             or self._grasp_mode is GraspMode.PINCH and not pan_handle_grasp
             and not self._flat_transfer_waypoints)
            and not retain_phase
            and observation.robot.joint_position is not None
        ):
            # A half turn about tool Z swaps the two fingers while retaining
            # the exact rim point, vertical approach and physical jaw line.
            # Use it only if public Panda IK rejects the preferred frame and
            # accepts the equivalent at both approach and contact waypoints.
            from ..common.panda_kinematics import reachable_equivalent_frame

            selected = reachable_equivalent_frame(
                observation.robot.joint_position,
                observation.robot.ee_pose.matrix,
                (pregrasp_position, grasp_position),
                rotation,
                rotation @ np.diag((-1.0, -1.0, 1.0)),
            )
            equivalent_trace_key = (
                "free_space_rim_equivalent_frame" if self._grasp_mode is GraspMode.RIM_PINCH
                else "pinch_equivalent_frame"
            )
            target_details[equivalent_trace_key] = {
                "strategy": "public_joint_ik_preserve_preferred_else_equivalent_yaw",
                "equivalent_selected": not np.allclose(selected, rotation),
                "selected_jaw_axis_world": selected[:, 1].tolist(),
                "waypoints_world_m": [pregrasp_position.tolist(), grasp_position.tolist()],
            }
            rotation = selected
            self._motion_pose = Pose(grasp_position, rotation)
        pregrasp_rotation = rotation
        self._cavity_level_pose = None
        if (
            self._active_cavity_rim
            and target_details.get("cavity_selected_side_profile") == "far"
        ):
            radial = np.asarray(
                target_details.get("finger_axis_world"), dtype=np.float64
            )
            pregrasp_rotation = self._cavity_outer_finger_lift_rotation(
                rotation,
                radial,
            )
            self._cavity_level_pose = Pose(pregrasp_position, rotation)
            target_details["cavity_tilted_pregrasp"] = {
                "strategy": "sensor_outside_finger_up_then_level_during_descent",
                "outer_finger_lift_rad": float(
                    self.config.cavity_rim_outer_finger_lift_rad
                ),
                "outside_radial_axis_world": radial.tolist(),
                "tilted_finger_axis_world": (
                    pregrasp_rotation[:, 1].tolist()
                ),
                "levelled_finger_axis_world": rotation[:, 1].tolist(),
                "levelled_proof_waypoint_world_m": pregrasp_position.tolist(),
            }
        self._secondary_pose = Pose(pregrasp_position, pregrasp_rotation)
        if not retain_phase:
            self._grasp_target_attempts.append(
                {
                    "attempt": self._grasp_retry_index,
                    "capture_sequence": int(
                        self._observation_timestamp_s
                    ),
                    "mode": self._grasp_mode.value,
                    "sensor_name": source.name,
                    "sensor_center_world_m": source.centroid_world.tolist(),
                    "sensor_extents_m": source.extents_m.tolist(),
                    "pick_rotation_anchor_source": (
                        self._pick_rotation_anchor_source
                    ),
                    **(
                        {
                            "episode_reset_tool_z_world_up_dot": float(
                                self._episode_reset_tool_z_world_up_dot
                            ),
                            "episode_reset_local_y_planar_norm": float(
                                self._episode_reset_local_y_planar_norm
                            ),
                        }
                        if self._pick_rotation_anchor_source
                        == "episode_reset_top_down_from_public_proprioception"
                        else {}
                    ),
                    "active_cavity_rim": bool(self._active_cavity_rim),
                    **(
                        {
                            "free_space_rim_preshape_reseat": True,
                            "free_space_rim_physical_approach_ordinal": 3,
                            "free_space_rim_reseat_reacquisition": dict(
                                self._free_rim_reseat_reacquisition
                            ),
                        }
                        if self._free_rim_preshape_reseat_required
                        and not self._active_cavity_rim
                        and self._free_rim_reseat_reacquisition is not None
                        else {}
                    ),
                    **(
                        {
                            "cavity_pregrasp_height_m": float(
                                pregrasp_height_m
                            )
                        }
                        if self._active_cavity_rim
                        else {}
                    ),
                    "target_world_m": grasp_position.tolist(),
                    "center_to_target_world_m": (
                        grasp_position - source.centroid_world
                    ).tolist(),
                    **target_details,
                }
            )
            self._phase_ticks = 0

    @staticmethod
    def _is_pan_handle_subject(subject: str) -> bool:
        normalized = " ".join(subject.lower().replace("_", " ").split())
        return normalized in _PAN_HANDLE_LABELS

    def _pan_handle_target(
        self,
        source: SceneObject,
        observation: SensorObservation,
        rotation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        """Infer one closed-finger grasp on a uniquely observed pan handle.

        The semantic label only selects this mechanical affordance family.
        Every pose value comes from the current DINO RGB-D surface and public
        proprioception.  Missing or ambiguous geometry fails closed and never
        falls back to the compound object's ungraspable whole-crop centroid.
        """

        points = source.surface_points_world
        if points is None:
            raise ValueError(
                "sensor-only frying-pan handle inference requires a fresh "
                "RGB-D surface component"
            )
        reference_pose = Pose(
            observation.robot.ee_pose.position.copy(),
            np.asarray(rotation, dtype=np.float64).copy(),
        ).matrix
        try:
            affordance: PanHandleAffordance = infer_pan_handle_affordance(
                points,
                reference_pose,
            )
        except (PanAffordanceError, ValueError) as exc:
            raise ValueError(
                f"sensor-only frying-pan handle inference failed: {exc}"
            ) from exc

        ordered_slots: tuple[PanHandleSlot, ...] = tuple(
            sorted(
                (
                    slot
                    for slot in affordance.slots
                    if slot.slot_id not in self._failed_pan_handle_slot_ids
                ),
                key=lambda slot: (
                    slot.longitudinal_fraction,
                    -slot.score,
                    slot.slot_id,
                ),
            )
        )
        if not ordered_slots:
            raise ValueError(
                "sensor-only frying-pan handle inference exhausted distinct "
                "solid slots"
            )
        slot = ordered_slots[0]
        level_rotation, tool_z_hemisphere = (
            self._cavity_world_vertical_grasp_rotation(
                rotation,
                affordance.jaw_axis_world[:2],
            )
        )
        target = slot.position_world.copy()
        target[2] = min(
            max(
                slot.local_top_z_m - self.config.pinch_grasp_top_inset_m,
                source.bounds_min_world[2]
                + self.config.pinch_ee_min_above_support_m,
            ),
            source.centroid_world[2] + self.config.grasp_z_offset_m,
        )

        # Use the inferred wide-body XY as the transported load reference,
        # while retaining the completed RGB-D object's volume-centre Z.
        body_reference = source.centroid_world.copy()
        body_reference[:2] = affordance.body_center_world[:2]
        self._active_pan_handle_slot_id = slot.slot_id
        self._pan_body_reference_world = body_reference
        return target, level_rotation, {
            "pan_handle_strategy": "rgbd_wide_body_unique_narrow_tail",
            "pan_surface_point_count": int(len(points)),
            "pan_handle_axis_world": affordance.handle_axis_world.tolist(),
            "pan_jaw_axis_world": affordance.jaw_axis_world.tolist(),
            "pan_aligned_local_y_world": level_rotation[:, 1].tolist(),
            "pan_tool_z_world": level_rotation[:, 2].tolist(),
            "pan_tool_z_hemisphere_sign": float(tool_z_hemisphere),
            "pan_body_center_world_m": affordance.body_center_world.tolist(),
            "pan_body_reference_world_m": body_reference.tolist(),
            "pan_handle_start_world_m": affordance.handle_start_world.tolist(),
            "pan_handle_end_world_m": affordance.handle_end_world.tolist(),
            "pan_body_diameter_m": float(affordance.body_diameter_m),
            "pan_handle_length_m": float(affordance.handle_length_m),
            "pan_median_handle_width_m": float(
                affordance.median_handle_width_m
            ),
            "pan_tail_bin_coverage": float(affordance.tail_bin_coverage),
            "pan_narrow_bin_fraction": float(affordance.narrow_bin_fraction),
            "pan_selected_slot_id": slot.slot_id,
            "pan_selected_slot_fraction": float(slot.longitudinal_fraction),
            "pan_selected_slot_local_top_z_m": float(slot.local_top_z_m),
            "pan_selected_slot_local_width_m": float(slot.local_width_m),
            "pan_selected_slot_center_gap_m": float(slot.center_gap_m),
            "pan_selected_slot_sensor_score": float(slot.score),
            "pan_failed_slot_ids": sorted(self._failed_pan_handle_slot_ids),
            "pan_slot_order_strategy": "proximal_low_load_arm_then_sensor_score",
            "orientation_source": (
                "rgbd_handle_axis_world_vertical_tool_z_with_proprio_hemisphere"
            ),
        }

    def _sensor_refined_grasp_mode(
        self,
        step: SkillStep,
        source: SceneObject,
    ) -> GraspMode:
        """Apply the non-expansion bowl contract using measured geometry.

        Explicit ``black bowl`` language always uses a closing rim pinch.
        Unqualified ``bowl`` is classified mechanically: a squat, rim-sized
        RGB-D OBB gets the same closing pinch, while an implausibly small test
        object remains an ordinary pinch.  Neither spelling may ever opt into
        the internal-expansion hook.
        """

        requested = self.grasp_mode_selector.select(step)
        subject = step.subject.strip().lower()
        if subject in {"black bowl", "bowl"} and requested is GraspMode.EXPAND:
            raise ValueError(
                f"internal expansion is forbidden for bowl entity {subject!r}"
            )
        if subject == "black bowl":
            return GraspMode.RIM_PINCH
        if subject != "bowl" or requested is GraspMode.RIM_PINCH:
            return requested

        vertical_axis = int(np.argmax(np.abs(source.axes_world[2, :])))
        planar_axes = tuple(index for index in range(3) if index != vertical_axis)
        planar_diameter = float(np.min(source.extents_m[list(planar_axes)]))
        height = float(source.extents_m[vertical_axis])
        radius = planar_diameter / 2.0 - self.config.rim_pinch_radial_inset_m
        bowl_like = bool(
            self.config.rim_pinch_min_radius_m
            <= radius
            <= self.config.rim_pinch_max_radius_m
            and height <= 0.90 * planar_diameter
        )
        return GraspMode.RIM_PINCH if bowl_like else GraspMode.PINCH

    @staticmethod
    def _cavity_world_vertical_grasp_rotation(
        current_rotation: np.ndarray,
        finger_axis_xy: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Level the final cavity grasp using only the measured finger axis.

        The initial wrist can carry a small roll/pitch bias.  Preserving that
        bias at a narrow drawer rim changes which finger shell touches first.
        Rebuild a proper frame whose local Y follows the sensor-selected rim
        radial axis and whose tool Z is exactly world vertical, retaining the
        observed tool-Z hemisphere so the hand never flips.
        """

        rotation = np.asarray(current_rotation, dtype=np.float64)
        finger_xy = np.asarray(finger_axis_xy, dtype=np.float64)
        if (
            rotation.shape != (3, 3)
            or finger_xy.shape != (2,)
            or not np.all(np.isfinite(rotation))
            or not np.all(np.isfinite(finger_xy))
        ):
            raise ValueError("cavity level rotation requires finite axes")
        finger_norm = float(np.linalg.norm(finger_xy))
        if finger_norm < 0.8:
            raise ValueError("cavity finger axis is not sufficiently planar")
        finger_y = np.array(
            (finger_xy[0] / finger_norm, finger_xy[1] / finger_norm, 0.0),
            dtype=np.float64,
        )
        world_up = np.array((0.0, 0.0, 1.0), dtype=np.float64)
        hemisphere_sign = (
            1.0 if float(np.dot(rotation[:, 2], world_up)) >= 0.0 else -1.0
        )
        tool_z = hemisphere_sign * world_up
        tool_x = np.cross(finger_y, tool_z)
        tool_x /= max(float(np.linalg.norm(tool_x)), 1e-12)
        finger_y = np.cross(tool_z, tool_x)
        finger_y /= max(float(np.linalg.norm(finger_y)), 1e-12)
        return np.column_stack((tool_x, finger_y, tool_z)), hemisphere_sign

    def _cavity_outer_finger_lift_rotation(
        self,
        level_rotation: np.ndarray,
        outside_radial_world: np.ndarray,
    ) -> np.ndarray:
        """Raise the sensor-selected outside finger for a far-side traverse."""

        rotation = np.asarray(level_rotation, dtype=np.float64)
        radial = np.asarray(outside_radial_world, dtype=np.float64).copy()
        if (
            rotation.shape != (3, 3)
            or radial.shape != (3,)
            or not np.all(np.isfinite(rotation))
            or not np.all(np.isfinite(radial))
        ):
            raise ValueError("cavity tilted pregrasp requires finite 3-D axes")
        world_up = np.array((0.0, 0.0, 1.0), dtype=np.float64)
        planar_finger = rotation[:, 1].copy()
        planar_finger -= world_up * float(np.dot(planar_finger, world_up))
        planar_norm = float(np.linalg.norm(planar_finger))
        radial[2] = 0.0
        radial_norm = float(np.linalg.norm(radial))
        if planar_norm < 0.8 or radial_norm < 0.8:
            raise ValueError("cavity tilted pregrasp axes are not sufficiently planar")
        planar_finger /= planar_norm
        radial /= radial_norm
        outside_sign = (
            1.0 if float(np.dot(radial, planar_finger)) >= 0.0 else -1.0
        )
        tilt = self.config.cavity_rim_outer_finger_lift_rad
        tilted_finger = (
            np.cos(tilt) * planar_finger
            + outside_sign * np.sin(tilt) * world_up
        )
        tilted_finger /= float(np.linalg.norm(tilted_finger))
        tool_x = rotation[:, 0].copy()
        tool_x -= tilted_finger * float(np.dot(tool_x, tilted_finger))
        tool_x_norm = float(np.linalg.norm(tool_x))
        if tool_x_norm < 0.8:
            raise ValueError("cavity tilted pregrasp tool axis is degenerate")
        tool_x /= tool_x_norm
        tool_z = np.cross(tool_x, tilted_finger)
        tool_z /= float(np.linalg.norm(tool_z))
        tilted_finger = np.cross(tool_z, tool_x)
        tilted_finger /= float(np.linalg.norm(tilted_finger))
        tilted_rotation = np.column_stack((tool_x, tilted_finger, tool_z))
        if outside_sign * float(tilted_rotation[2, 1]) <= 0.0:
            raise ValueError("cavity tilted pregrasp did not raise outside finger")
        return tilted_rotation

    @staticmethod
    def _pinch_width_aligned_rotation(source: SceneObject, rotation: np.ndarray) -> np.ndarray:
        """Close across the narrow side of an elongated observed footprint."""
        planar_axes = source.axes_world[:2] * source.extents_m[None, :]
        values, vectors = np.linalg.eigh(planar_axes @ planar_axes.T)
        spans = np.sqrt(np.maximum(values, 0.0))
        if spans[1] < 0.055 or spans[1] < 1.4 * max(spans[0], 1e-6):
            return rotation
        desired = vectors[:, 0]
        current = rotation[:2, 1]
        if np.dot(desired, current) < 0:
            desired = -desired
        yaw = np.arctan2(current[0] * desired[1] - current[1] * desired[0],
                         np.dot(current, desired))
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))) @ rotation

    def _flat_pinch_approach_aperture(self) -> float | None:
        """Leave clearance around a flat package without sweeping a full-open jaw.

        Set the aperture at the high waypoint, then maintain it through the
        descent. Nearby upright packages can otherwise be hit by the outside
        of an open finger before the intended low grasp reaches the table.
        """
        source = self._pick_source_geometry
        if (
            self._grasp_mode is not GraspMode.PINCH
            or source is None
            or self._motion_pose is None
            or self._is_pan_handle_subject(source.name)
        ):
            return None
        size = source.bounds_max_world - source.bounds_min_world
        jaw_axis = self._motion_pose.rotation[:, 1]
        width = float(np.sum(np.abs(jaw_axis @ source.axes_world) * source.extents_m))
        if size[2] > 0.040 or max(size[:2]) < 0.055 or width > 0.055:
            return None
        aperture = max(0.030, width + 0.014)
        if self._grasp_target_attempts:
            self._grasp_target_attempts[-1]["flat_pinch_approach_aperture_m"] = aperture
        return aperture

    @staticmethod
    def _flat_pinch_clearance_target(
        source: SceneObject,
        observation: SensorObservation,
        grasp_position: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """Move along a package's long face away from observed tall clutter."""
        size = source.bounds_max_world - source.bounds_min_world
        planar = np.sort(source.extents_m[:2])
        if size[2] > 0.040 or planar[1] < 0.055 or planar[0] > 0.055:
            return grasp_position, {}
        if planar[1] < 1.4 * planar[0]:
            return grasp_position, {}
        from ..perception.adapters import coerce_rgbd_frame
        from ..perception.geometry import backproject_frame

        clouds = [
            backproject_frame(coerce_rgbd_frame(frame, name=name), stride=2).points_world
            for name, frame in observation.cameras.items()
            if "agent" in name.lower()
        ]
        if not clouds:
            return grasp_position, {}
        points = np.concatenate(clouds)
        top = source.bounds_max_world[2]
        points = points[(points[:, 2] > top + 0.020) & (points[:, 2] < top + 0.160)]
        if len(points) < 20:
            return grasp_position, {}
        long_axis = source.axes_world[:, int(np.argmax(source.extents_m[:2]))]
        offset = min(0.030, 0.5 * planar[1] - 0.016)
        candidates = np.array([grasp_position, grasp_position + offset * long_axis,
                               grasp_position - offset * long_axis])
        distances = np.linalg.norm(points[None, :, :2] - candidates[:, None, :2], axis=2)
        scores = np.sum(np.maximum(0.0, 0.080 - distances), axis=1)
        best = int(np.argmin(scores))
        if best == 0 or np.count_nonzero(distances[0] < 0.080) < 20:
            return grasp_position, {}
        if scores[best] > 0.70 * scores[0]:
            return grasp_position, {}
        return candidates[best], {
            "flat_pinch_clearance_offset_world_m": (candidates[best] - grasp_position).tolist(),
            "flat_pinch_clutter_scores": scores.tolist(),
        }

    def _pinch_grasp_height(
        self,
        source: SceneObject,
        *,
        top_inset_m: float | None = None,
        apply_retry_offset: bool = True,
    ) -> float:
        if top_inset_m is None:
            top_inset_m = self.config.pinch_grasp_top_inset_m
        observed_size = source.bounds_max_world - source.bounds_min_world
        tall_narrow = observed_size[2] > 2.0 * max(observed_size[:2])
        if tall_narrow:
            # A bottle-like vertical object needs an upper-body/neck grasp.
            # The legacy centre+35 mm cap drove the palm into its top before
            # the fingertips reached the requested body height. Use its
            # measured top with a height-scaled inset instead.
            top_inset_m = max(top_inset_m, min(0.030, 0.15 * observed_size[2]))
        top_surface_target = source.bounds_max_world[2] - top_inset_m
        support_clearance_target = (
            source.bounds_min_world[2]
            + self.config.pinch_ee_min_above_support_m
        )
        legacy_upper_bound = source.centroid_world[2] + self.config.grasp_z_offset_m
        if tall_narrow or source.name == "moka pot":
            # A moka pot's broad lower body exceeds the finger opening.
            # Engage its narrower upper lid/knob instead of driving the open
            # fingertips down to the centre+35 mm body plane.
            legacy_upper_bound = top_surface_target
        base_target = max(top_surface_target, support_clearance_target)
        retry_target = base_target + (
            self._grasp_retry_height_offset_m() if apply_retry_offset else 0.0
        )
        return float(
            min(
                max(retry_target, support_clearance_target),
                legacy_upper_bound,
            )
        )

    def _rim_pinch_target(
        self,
        source: SceneObject,
        observation: SensorObservation,
        rotation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        """Place the Panda grip site over one sensed black-bowl rim.

        In Route B's raw robosuite EEF-body orientation, Panda's two pads lie
        and slide along tool-local Y (the grip-site frame itself differs by a
        fixed yaw).  Projecting that observable axis to the table and selecting
        a rim point on the same radial line makes one open finger descend
        inside the bowl and the other outside.  If the language selector also
        supplied a measured drawer/cabinet OBB, treat both horizontal axes as
        hypotheses: an open drawer and perspective can make its observed depth
        longer than its width.  Each axis tries the rim side with more wall
        clearance first.  Otherwise the original free-space proposal is
        preserved exactly.  The bowl is nearly circular, so its noisy RGB-D
        PCA yaw is deliberately not used as a grasp direction.
        """

        finger_axis_xy = np.array(rotation[:2, 1], dtype=np.float64, copy=True)
        axis_norm = float(np.linalg.norm(finger_axis_xy))
        if axis_norm < 0.5:
            raise ValueError("Panda local-Y finger axis is not sufficiently planar")
        finger_axis_xy /= axis_norm

        frozen_retry_anchor: SceneObject | None = None
        frozen_retry_center: np.ndarray | None = None
        frozen_attempt0_target: np.ndarray | None = None
        if (
            self._free_rim_antipodal_reacquisition_required
            and self._grasp_retry_index == 1
        ):
            frozen_retry_anchor = self._free_rim_retry_anchor_geometry
            frozen_retry_center = self._pick_anchor_center_world
            for attempt in reversed(self._grasp_target_attempts):
                if (
                    type(attempt.get("attempt")) is int
                    and attempt.get("attempt") == 0
                    and attempt.get("rim_strategy") == "free_space"
                ):
                    candidate_target = np.asarray(
                        attempt.get("target_world_m"), dtype=np.float64
                    )
                    if (
                        candidate_target.shape == (3,)
                        and np.all(np.isfinite(candidate_target))
                    ):
                        frozen_attempt0_target = candidate_target
                        break
            if (
                frozen_retry_anchor is None
                or frozen_retry_center is None
                or frozen_retry_center.shape != (3,)
                or not np.all(np.isfinite(frozen_retry_center))
                or frozen_attempt0_target is None
            ):
                raise ValueError(
                    "free-space antipodal retry lacks frozen attempt-zero geometry"
                )
            fresh_top_offset = float(
                source.bounds_max_world[2]
                - frozen_retry_anchor.bounds_max_world[2]
            )
            top_update_bound = self._pick_selector_reacquire_radius_m
            if (
                top_update_bound is None
                or not np.isfinite(top_update_bound)
                or abs(fresh_top_offset) > top_update_bound
            ):
                raise ValueError(
                    "free-space antipodal retry fresh top left its frozen "
                    "instance-continuity bound"
                )

        fresh_reseat_geometry = bool(
            frozen_retry_anchor is not None
            and self._free_rim_preshape_reseat_required
        )
        radius_source = (
            # Attempt one's open bilateral probe must remain an exact mirror
            # of attempt zero.  Once those two probes have authorized the
            # separate high-preshape/reseat cycle, however, its newly bound
            # RGB-D candidate owns the physical rim geometry.  The old anchor
            # remains only an identity/size/distance continuity reference.
            source
            if fresh_reseat_geometry
            else frozen_retry_anchor
            if frozen_retry_anchor is not None
            else source
        )
        vertical_axis = int(np.argmax(np.abs(radius_source.axes_world[2, :])))
        planar_axes = tuple(index for index in range(3) if index != vertical_axis)
        planar_diameter = float(
            np.min(radius_source.extents_m[list(planar_axes)])
        )
        radius = planar_diameter / 2.0 - self.config.rim_pinch_radial_inset_m
        if not self.config.rim_pinch_min_radius_m <= radius <= self.config.rim_pinch_max_radius_m:
            raise ValueError(
                "sensor-derived rim radius "
                f"{radius:.4f} m is outside calibrated bounds "
                f"[{self.config.rim_pinch_min_radius_m:.4f}, "
                f"{self.config.rim_pinch_max_radius_m:.4f}] m"
            )

        axis_preference_rank, side_rank = self._cavity_candidate_axis_side()
        cavity = self._cavity_rim_frame(
            source,
            rotation,
            finger_axis_xy,
            radius,
            observation.robot.ee_pose.position[:2],
            axis_preference_rank=axis_preference_rank,
        )
        details: dict[str, object]
        if cavity is None:
            side_reference_xy = observation.robot.ee_pose.position[:2]
            side_reference_source = "live_pick_public_proprioception"
            if self._episode_reset_position_world is not None:
                side_reference_xy = self._episode_reset_position_world[:2]
                side_reference_source = (
                    "episode_reset_public_proprioception"
                )
            center_to_ee_xy = (
                side_reference_xy - source.centroid_world[:2]
            )
            jaw_alignment_details: dict[str, object] = {
                "free_space_jaw_alignment_strategy": (
                    "preserve_frozen_or_live_public_jaw_axis"
                ),
            }
            # A reset-time EE bearing can be almost orthogonal to Panda's
            # frozen local-Y line.  Selecting the near sign by projection in
            # that case puts the contact on an unrelated physical side (the
            # post-drawer G03 failure approached the +Y wall although the
            # unobstructed reset approach was predominantly -X).  For a
            # genuinely separated, free-space bowl, align the *unoriented jaw
            # line* with the public reset-to-source bearing.  The sign of
            # local Y is chosen only to stay in the closest frozen yaw
            # hemisphere; ``near_sign`` below still selects the physical rim
            # facing the reset pose.  No object PCA yaw, task id, simulator
            # pose, or evaluator state participates in this construction.
            bearing_norm = float(np.linalg.norm(center_to_ee_xy))
            minimum_bearing_norm = max(2.0 * radius, 0.080)
            if (
                self._episode_reset_position_world is not None
                and np.all(np.isfinite(center_to_ee_xy))
                and bearing_norm >= minimum_bearing_norm
            ):
                reset_bearing_xy = center_to_ee_xy / bearing_norm
                aligned_finger_axis_xy = reset_bearing_xy.copy()
                frozen_alignment = float(
                    np.dot(aligned_finger_axis_xy, finger_axis_xy)
                )
                if frozen_alignment < -1e-6:
                    aligned_finger_axis_xy *= -1.0
                elif abs(frozen_alignment) <= 1e-6:
                    # At an exact 90-degree tie both local-Y signs describe
                    # the same two-pad line.  Canonicalize geometrically so
                    # equivalent reset-frame sign flips remain identical.
                    dominant = int(np.argmax(np.abs(aligned_finger_axis_xy)))
                    if aligned_finger_axis_xy[dominant] < 0.0:
                        aligned_finger_axis_xy *= -1.0
                rotation, tool_z_hemisphere_sign = (
                    self._cavity_world_vertical_grasp_rotation(
                        rotation,
                        aligned_finger_axis_xy,
                    )
                )
                finger_axis_xy = aligned_finger_axis_xy
                jaw_alignment_details = {
                    "free_space_jaw_alignment_strategy": (
                        "episode_reset_bearing_aligned_jaw_line"
                    ),
                    "free_space_reset_bearing_world": [
                        float(reset_bearing_xy[0]),
                        float(reset_bearing_xy[1]),
                        0.0,
                    ],
                    "free_space_aligned_jaw_axis_world": [
                        float(finger_axis_xy[0]),
                        float(finger_axis_xy[1]),
                        0.0,
                    ],
                    "free_space_reset_bearing_norm_m": bearing_norm,
                    "free_space_minimum_bearing_norm_m": minimum_bearing_norm,
                    "free_space_tool_z_hemisphere_sign": float(
                        tool_z_hemisphere_sign
                    ),
                }
            near_sign = (
                1.0
                if float(np.dot(center_to_ee_xy, finger_axis_xy)) >= 0.0
                else -1.0
            )
            # Try the unobstructed near rim first.  The first retry samples the
            # opposite rim as an independent geometric candidate.  Contact
            # audit established the same sensor-top-minus-10-mm plane on the
            # successful opposite side, so rim retries vary side/reobservation
            # without reintroducing the shallow lip-only grasp plane.
            use_near_side = self._grasp_retry_index != 1
            side_sign = near_sign if use_near_side else -near_sign
            rim_side = "near" if use_near_side else "opposite"
            details = {
                "rim_strategy": "free_space",
                "rim_height_strategy": "sensor_completed_top_surface",
                "rim_near_side_reference": side_reference_source,
                "rim_near_side_reference_world_m": [
                    float(side_reference_xy[0]),
                    float(side_reference_xy[1]),
                ],
                **jaw_alignment_details,
            }
        else:
            (
                finger_axis_xy,
                rotation,
                clearance_side_order,
                wall_clearances,
                yaw_adjustment,
                reference_name,
                axis_extent,
                planar_aspect,
                axis_extent_rank,
                approach_alignment,
                approach_direction,
            ) = cavity
            base_width_axis_xy = finger_axis_xy.copy()
            base_frame_rotation = rotation.copy()
            local_yaw_offset = self._cavity_candidate_yaw_offset_rad()
            yaw_cosine = float(np.cos(local_yaw_offset))
            yaw_sine = float(np.sin(local_yaw_offset))
            local_yaw_rotation = np.array(
                (
                    (yaw_cosine, -yaw_sine, 0.0),
                    (yaw_sine, yaw_cosine, 0.0),
                    (0.0, 0.0, 1.0),
                ),
                dtype=np.float64,
            )
            finger_axis_xy = (
                local_yaw_rotation[:2, :2] @ base_width_axis_xy
            )
            rotation = local_yaw_rotation @ rotation
            pre_level_rotation = rotation.copy()
            rotation, tool_z_hemisphere_sign = (
                self._cavity_world_vertical_grasp_rotation(
                    rotation,
                    finger_axis_xy,
                )
            )
            final_wall_clearances: dict[float, float | None] = {
                sign: self._cavity_radial_wall_clearance(
                    source,
                    sign * finger_axis_xy,
                    radius,
                )
                for sign in (1.0, -1.0)
            }
            approach_available = bool(
                float(np.linalg.norm(approach_direction)) > 0.8
            )
            far_side_sign = (
                1.0
                if float(np.dot(finger_axis_xy, approach_direction)) >= 0.0
                else -1.0
            )
            far_side_clearance = final_wall_clearances[far_side_sign]
            far_side_usable = bool(
                approach_available
                and far_side_clearance is not None
                and far_side_clearance
                >= self.config.cavity_rim_min_final_wall_clearance_m
            )
            if far_side_usable:
                candidate_side_order = (far_side_sign, -far_side_sign)
                side_ordering_reason = (
                    "approach_aligned_far_side_meets_final_clearance"
                )
            else:
                candidate_side_order = tuple(
                    sorted(
                        (1.0, -1.0),
                        key=lambda sign: (
                            -(
                                final_wall_clearances[sign]
                                if final_wall_clearances[sign] is not None
                                else -np.inf
                            ),
                            clearance_side_order.index(sign),
                        ),
                    )
                )
                side_ordering_reason = "maximum_final_wall_clearance_fallback"
            antipodal_retry_trace: dict[str, object] | None = None
            primary_radial = self._cavity_primary_radial_world
            if side_rank > 0 and primary_radial is not None:
                primary_xy = np.asarray(primary_radial, dtype=np.float64)[:2]
                primary_norm = float(np.linalg.norm(primary_xy))
                if primary_norm < 0.8 or not np.all(np.isfinite(primary_xy)):
                    raise ValueError("frozen primary cavity rim axis is invalid")
                primary_xy /= primary_norm
                retry_dots = {
                    sign: float(np.dot(sign * finger_axis_xy, primary_xy))
                    for sign in (1.0, -1.0)
                }
                side_sign = min(retry_dots, key=retry_dots.get)
                antipodal_dot = retry_dots[side_sign]
                maximum_antipodal_dot = float(
                    -np.cos(abs(self.config.cavity_rim_yaw_offset_rad)) + 1e-6
                )
                if antipodal_dot > maximum_antipodal_dot:
                    raise ValueError(
                        "retry cavity rim candidate is not antipodal to the "
                        "first sensor-selected physical rim"
                    )
                side_ordering_reason = "antipodal_to_primary_sensor_rim"
                antipodal_retry_trace = {
                    "strategy": "minimum_dot_to_frozen_primary_sensor_rim",
                    "primary_radial_world": [
                        float(primary_xy[0]),
                        float(primary_xy[1]),
                        0.0,
                    ],
                    "retry_radial_world": [
                        float(side_sign * finger_axis_xy[0]),
                        float(side_sign * finger_axis_xy[1]),
                        0.0,
                    ],
                    "dot": float(antipodal_dot),
                    "angle_rad": float(
                        np.arccos(np.clip(antipodal_dot, -1.0, 1.0))
                    ),
                    "maximum_allowed_dot": maximum_antipodal_dot,
                    "positive_axis_dot": retry_dots[1.0],
                    "negative_axis_dot": retry_dots[-1.0],
                }
            else:
                side_sign = candidate_side_order[side_rank]
            final_radial_xy = side_sign * finger_axis_xy
            final_wall_clearance = final_wall_clearances[side_sign]
            if (
                final_wall_clearance is None
                or final_wall_clearance
                < self.config.cavity_rim_min_final_wall_clearance_m
            ):
                raise ValueError(
                    "local-yaw rim candidate lacks measured cavity wall clearance"
                )
            rim_side = (
                "cavity_clearance_first"
                if side_rank == 0
                else "cavity_clearance_second"
            )
            selected_side_approach_dot = float(
                np.dot(final_radial_xy, approach_direction)
            )
            selected_side_profile = (
                "far"
                if approach_available and selected_side_approach_dot >= 0.0
                else "near"
                if approach_available
                else "clearance_only"
            )
            roomy_near_trace: dict[str, object] | None = None
            if (
                side_rank == 0
                and selected_side_profile == "far"
                and not self._cavity_reference_from_active_view
                and far_side_clearance is not None
            ):
                # A direct near approach avoids crossing the bowl and far
                # lip when its own clearance supports the nominal fingers.
                # Test that choice in the same measured OBB
                # frame, then rebuild the selected candidate on the untouched
                # base axis.  Both direction checks are explicit because PCA
                # axes are unoriented and may flip sign between observations.
                trigger_far_side_sign = far_side_sign
                trigger_far_clearance = float(far_side_clearance)
                trigger_other_side_sign = -trigger_far_side_sign
                trigger_other_clearance = final_wall_clearances[
                    trigger_other_side_sign
                ]
                trigger_other_radial = (
                    trigger_other_side_sign * finger_axis_xy
                )
                trigger_other_approach_dot = float(
                    np.dot(trigger_other_radial, approach_direction)
                )
                nominal_near_sign = (
                    1.0
                    if float(
                        np.dot(base_width_axis_xy, trigger_other_radial)
                    )
                    >= 0.0
                    else -1.0
                )
                nominal_near_radial = (
                    nominal_near_sign * base_width_axis_xy
                )
                nominal_near_approach_dot = float(
                    np.dot(nominal_near_radial, approach_direction)
                )
                nominal_final_wall_clearances: dict[
                    float, float | None
                ] = {
                    sign: self._cavity_radial_wall_clearance(
                        source,
                        sign * base_width_axis_xy,
                        radius,
                    )
                    for sign in (1.0, -1.0)
                }
                nominal_near_clearance = nominal_final_wall_clearances[
                    nominal_near_sign
                ]
                use_roomy_near = bool(
                    trigger_other_clearance is not None
                    and trigger_other_clearance
                    >= self.config.cavity_rim_roomy_near_clearance_m
                    and trigger_other_approach_dot < 0.0
                    and nominal_near_approach_dot < 0.0
                    and nominal_near_clearance is not None
                    and nominal_near_clearance
                    >= self.config.cavity_rim_roomy_near_clearance_m
                )
                if use_roomy_near:
                    trigger_other_clearance_value = float(
                        trigger_other_clearance
                    )
                    finger_axis_xy = base_width_axis_xy.copy()
                    pre_level_rotation = base_frame_rotation.copy()
                    rotation, tool_z_hemisphere_sign = (
                        self._cavity_world_vertical_grasp_rotation(
                            base_frame_rotation,
                            finger_axis_xy,
                        )
                    )
                    local_yaw_offset = 0.0
                    final_wall_clearances = nominal_final_wall_clearances
                    side_sign = nominal_near_sign
                    final_radial_xy = nominal_near_radial
                    final_wall_clearance = nominal_near_clearance
                    far_side_sign = -nominal_near_sign
                    far_side_clearance = final_wall_clearances[
                        far_side_sign
                    ]
                    far_side_usable = bool(
                        far_side_clearance is not None
                        and far_side_clearance
                        >= self.config.cavity_rim_min_final_wall_clearance_m
                    )
                    selected_side_approach_dot = nominal_near_approach_dot
                    selected_side_profile = "roomy_near_nominal"
                    side_ordering_reason = (
                        "robot_facing_nominal_side_has_measured_clearance"
                    )
                    rim_side = "roomy_near_nominal"
                    roomy_near_trace = {
                        "near_clearance_threshold_m": float(
                            self.config.cavity_rim_roomy_near_clearance_m
                        ),
                        "trigger_far_final_clearance_m": (
                            trigger_far_clearance
                        ),
                        "trigger_opposite_final_clearance_m": (
                            trigger_other_clearance_value
                        ),
                        "trigger_opposite_approach_dot": (
                            trigger_other_approach_dot
                        ),
                        "selected_nominal_approach_dot": (
                            nominal_near_approach_dot
                        ),
                        "selected_nominal_clearance_m": float(
                            nominal_near_clearance
                        ),
                        "yaw_offset_rad": 0.0,
                        "top_inset_m": float(
                            self.config.cavity_rim_roomy_near_top_inset_m
                        ),
                        "tilted_pregrasp": False,
                        "seat_actions": 0,
                    }
            if (side_rank == 0 and selected_side_profile == "far"
                    and final_wall_clearances[-far_side_sign] is not None
                    and final_wall_clearances[-far_side_sign]
                    >= self.config.cavity_rim_roomy_near_clearance_m):
                # A near rim with sufficient measured clearance can use the
                # established yawed/seat-verified grasp directly. Keep this
                # frame after an active view, where a nominal-axis switch can
                # create a different high wrist solution next to the fixture.
                side_sign = -far_side_sign
                final_radial_xy = side_sign * finger_axis_xy
                final_wall_clearance = final_wall_clearances[side_sign]
                selected_side_approach_dot = float(np.dot(final_radial_xy, approach_direction))
                selected_side_profile = "near"
                side_ordering_reason = "robot_facing_yawed_side_has_measured_clearance"
                rim_side = "cavity_near_clearance_first"
            raw_rim_radius = radius
            raw_final_wall_clearance = float(final_wall_clearance)
            details = {
                "rim_strategy": "cavity_width_axis",
                "cavity_reference": reference_name,
                "cavity_candidate_index": self._cavity_candidate_index,
                "cavity_axis_preference": (
                    "preferred" if axis_preference_rank == 0 else "alternate"
                ),
                "cavity_axis_rank": (
                    "major" if axis_extent_rank == 0 else "minor"
                ),
                "cavity_axis_extent_m": float(axis_extent),
                "cavity_planar_aspect_ratio": float(planar_aspect),
                "cavity_axis_initial_approach_alignment_abs": float(
                    approach_alignment
                ),
                "initial_ee_to_source_direction_world": [
                    float(approach_direction[0]),
                    float(approach_direction[1]),
                    0.0,
                ],
                "cavity_axis_selection_strategy": (
                    "max_alignment_with_initial_free_space_approach"
                ),
                "cavity_base_width_axis_world": [
                    float(base_width_axis_xy[0]),
                    float(base_width_axis_xy[1]),
                    0.0,
                ],
                "cavity_local_yaw_offset_rad": float(local_yaw_offset),
                "cavity_final_finger_axis_world": [
                    float(finger_axis_xy[0]),
                    float(finger_axis_xy[1]),
                    0.0,
                ],
                "cavity_level_rotation": {
                    "strategy": "world_vertical_tool_z_sensor_finger_axis",
                    "tool_z_hemisphere_sign": float(tool_z_hemisphere_sign),
                    "input_tool_z_world": pre_level_rotation[:, 2].tolist(),
                    "output_tool_z_world": rotation[:, 2].tolist(),
                },
                "cavity_gripper_preshape_target_width_m": float(
                    self.config.cavity_rim_preshape_target_width_m
                ),
                "cavity_gripper_preshape_entry_band_m": [
                    float(self.config.cavity_rim_preshape_min_width_m),
                    float(self.config.cavity_rim_preshape_max_width_m),
                ],
                "cavity_width_axis_world": [
                    float(finger_axis_xy[0]),
                    float(finger_axis_xy[1]),
                    0.0,
                ],
                "cavity_wall_clearance_m": float(final_wall_clearance),
                "cavity_final_wall_clearances_m": {
                    "positive_axis": (
                        float(final_wall_clearances[1.0])
                        if final_wall_clearances[1.0] is not None
                        else None
                    ),
                    "negative_axis": (
                        float(final_wall_clearances[-1.0])
                        if final_wall_clearances[-1.0] is not None
                        else None
                    ),
                },
                "cavity_far_side_sign": float(far_side_sign),
                "cavity_far_side_final_clearance_m": (
                    float(far_side_clearance)
                    if far_side_clearance is not None
                    else None
                ),
                "cavity_far_side_clearance_usable": far_side_usable,
                "cavity_selected_side_approach_dot": (
                    selected_side_approach_dot
                ),
                "cavity_selected_side_profile": selected_side_profile,
                "cavity_side_ordering_reason": side_ordering_reason,
                "cavity_reference_origin": (
                    "fresh_active_view_rgbd"
                    if self._cavity_reference_from_active_view
                    else "initial_rgbd"
                ),
                "cavity_roomy_near_policy": "enabled_when_measured_near_clearance_suffices",
                "cavity_raw_rim_radius_m": float(raw_rim_radius),
                "cavity_applied_rim_radius_m": float(radius),
                "cavity_raw_final_wall_clearance_m": float(
                    raw_final_wall_clearance
                ),
                "cavity_applied_final_wall_clearance_m": float(
                    final_wall_clearance
                ),
                # Retain explicit zero-valued provenance for old result
                # readers: the production target is the sensor-completed raw
                # rim radius, with no low-clearance inward correction.
                "cavity_far_low_clearance_radial_inset_m": 0.0,
                "cavity_far_low_clearance_inset_reason": (
                    "disabled_use_sensor_raw_rim_radius"
                ),
                "cavity_base_wall_clearance_m": float(
                    wall_clearances[side_sign]
                ),
                "cavity_wall_clearances_m": {
                    "positive_axis": float(wall_clearances[1.0]),
                    "negative_axis": float(wall_clearances[-1.0]),
                },
                "wrist_yaw_adjustment_rad": float(
                    yaw_adjustment + local_yaw_offset
                ),
                "cavity_base_wrist_yaw_adjustment_rad": float(yaw_adjustment),
                "rim_height_strategy": "sensor_completed_top_surface",
                **(
                    {"cavity_roomy_near_nominal": roomy_near_trace}
                    if roomy_near_trace is not None
                    else {}
                ),
                **(
                    {"cavity_antipodal_retry": antipodal_retry_trace}
                    if antipodal_retry_trace is not None
                    else {}
                ),
            }
        radial_world = np.array(
            (side_sign * finger_axis_xy[0], side_sign * finger_axis_xy[1], 0.0),
            dtype=np.float64,
        )
        target = source.centroid_world.copy()
        target[:2] += radial_world[:2] * radius
        if (
            frozen_retry_anchor is not None
            and frozen_retry_center is not None
            and frozen_attempt0_target is not None
        ):
            # The fresh bound crop proves that the same physical instance is
            # still visible.  The second, open-hand probe stays an exact
            # attempt-zero reflection; after that bilateral proof authorizes
            # a third acquisition, its fresh candidate owns the final centre,
            # size, and top used by the preshaped physical rim target.
            attempt0_radial_xy = (
                frozen_attempt0_target[:2] - frozen_retry_center[:2]
            )
            frozen_radius = float(np.linalg.norm(attempt0_radial_xy))
            if (
                not np.isfinite(frozen_radius)
                or abs(frozen_radius - radius)
                > self.config.grasp_position_tolerance_m
            ):
                raise ValueError(
                    "free-space antipodal retry lost its frozen rim radius"
                )
            if fresh_reseat_geometry:
                # The third acquisition is not merely a freshness token.  Its
                # centre and current size define the final rim point; otherwise
                # a moved/previously occluded bowl would still be approached at
                # attempt zero's stale reflected XY.
                target[:2] = (
                    source.centroid_world[:2] + radial_world[:2] * radius
                )
                target_center_source = (
                    "fresh_reseat_candidate_center_xy_and_top_z"
                )
            else:
                retry_radial_xy = -attempt0_radial_xy
                target[:2] = frozen_retry_center[:2] + retry_radial_xy
                radial_world = np.array(
                    (
                        retry_radial_xy[0] / frozen_radius,
                        retry_radial_xy[1] / frozen_radius,
                        0.0,
                    ),
                    dtype=np.float64,
                )
                target_center_source = (
                    "attempt0_frozen_anchor_xy_fresh_bound_top_z"
                )
            rotation, tool_z_hemisphere_sign = (
                self._cavity_world_vertical_grasp_rotation(
                    rotation,
                    radial_world[:2],
                )
            )
            fresh_center_offset = source.centroid_world - frozen_retry_center
            details.update(
                {
                    "target_center_source": target_center_source,
                    "fresh_center_offset_from_attempt0_anchor_m": (
                        fresh_center_offset.tolist()
                    ),
                    "frozen_attempt0_rim_target_world_m": (
                        frozen_attempt0_target.tolist()
                    ),
                    "frozen_attempt0_anchor_center_world_m": (
                        frozen_retry_center.tolist()
                    ),
                    "fresh_top_world_m": float(source.bounds_max_world[2]),
                    "fresh_top_offset_from_attempt0_m": float(
                        source.bounds_max_world[2]
                        - frozen_retry_anchor.bounds_max_world[2]
                    ),
                    "fresh_top_continuity_bound_m": float(
                        self._pick_selector_reacquire_radius_m
                    ),
                    "free_space_aligned_jaw_axis_world": (
                        radial_world.tolist()
                    ),
                    "free_space_tool_z_hemisphere_sign": float(
                        tool_z_hemisphere_sign
                    ),
                }
            )
        if cavity is not None:
            if details.get("cavity_selected_side_profile") == (
                "roomy_near_nominal"
            ):
                rim_top_inset_m = (
                    self.config.cavity_rim_roomy_near_top_inset_m
                )
            else:
                rim_top_inset_m = (
                    self.config.cavity_rim_grasp_top_inset_m
                    if self._cavity_candidate_index == 0
                    else self.config.cavity_rim_retry_grasp_top_inset_m
                )
        else:
            rim_top_inset_m = self.config.free_space_rim_grasp_top_inset_m
        target[2] = self._pinch_grasp_height(
            source,
            top_inset_m=rim_top_inset_m,
            # Candidate diversity for a rim pinch comes from the near/opposite
            # radial side (or cavity-local yaw), not an unproven height shift.
            # This also preserves the already successful opposite candidate at
            # exactly sensor top minus 10 mm.
            apply_retry_offset=False,
        )
        return target, rotation, {
            "rim_side": rim_side,
            "rim_radius_m": radius,
            "rim_height_from_sensor_top_m": float(
                target[2] - source.bounds_max_world[2]
            ),
            "finger_axis_world": radial_world.tolist(),
            **(
                {
                    "cavity_grasp_top_inset_m": float(
                        rim_top_inset_m
                    )
                }
                if cavity is not None
                else {
                    "free_space_rim_grasp_top_inset_m": float(
                        self.config.free_space_rim_grasp_top_inset_m
                    )
                }
            ),
            **details,
        }

    @staticmethod
    def _maximum_cavity_candidate_index() -> int:
        return 1

    def _cavity_candidate_axis_side(self) -> tuple[int, int]:
        # Keep the sensor-ranked fixture axis fixed, then enumerate the two
        # dynamically ordered physical sides.  The order itself is derived in
        # ``_rim_pinch_target`` from initial proprio bearing and final OBB
        # clearance after applying the local yaw.
        return (0, self._cavity_candidate_index)

    def _cavity_candidate_yaw_offset_rad(self) -> float:
        # Use the same local frame perturbation on both sides.  Flipping the
        # radial sign then produces genuinely antipodal grasp sites instead of
        # mixing a side change with a second yaw variable.
        return -self.config.cavity_rim_yaw_offset_rad

    def _next_cavity_candidate_after_approach_block(self) -> int:
        # Retained for adapters/tests that bypass the production pre-shape.
        # Production proof-lift retries use the antipodal rim candidate.
        return min(
            self._cavity_candidate_index + 1,
            self._maximum_cavity_candidate_index(),
        )

    def _cavity_rim_frame(
        self,
        source: SceneObject,
        rotation: np.ndarray,
        current_finger_axis_xy: np.ndarray,
        radius: float,
        ee_position_xy: np.ndarray,
        *,
        axis_preference_rank: int = 0,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        tuple[float, float],
        dict[float, float],
        float,
        str,
        float,
        float,
        int,
        float,
        np.ndarray,
    ] | None:
        """Build a fixture-width grasp frame from an optional measured OBB.

        PCA axes are unoriented, so the selected width axis is canonicalized
        against the current Panda local-Y direction.  This makes an equivalent
        sign flip of the RGB-D OBB produce the same wrist yaw, rim candidate,
        and clearance ordering.
        """

        reference = self._pick_cavity_reference
        if reference is None:
            return None
        center_value = getattr(reference, "center_world", None)
        if center_value is None:
            center_value = getattr(reference, "centroid_world", None)
        try:
            reference_center = np.asarray(center_value, dtype=np.float64)
            axes = np.asarray(getattr(reference, "axes_world"), dtype=np.float64)
            extents = np.asarray(getattr(reference, "extents_m"), dtype=np.float64)
        except (TypeError, ValueError, AttributeError):
            return None
        if (
            reference_center.shape != (3,)
            or axes.shape != (3, 3)
            or extents.shape != (3,)
            or not np.all(np.isfinite(reference_center))
            or not np.all(np.isfinite(axes))
            or not np.all(np.isfinite(extents))
            or not np.allclose(axes.T @ axes, np.eye(3), atol=2e-2)
        ):
            return None

        vertical_axis = int(np.argmax(np.abs(axes[2, :])))
        if abs(float(axes[2, vertical_axis])) < 0.8:
            return None
        planar_axes = [index for index in range(3) if index != vertical_axis]
        ranked_planar_axes = sorted(
            planar_axes,
            key=lambda index: (-float(extents[index]), index),
        )
        if axis_preference_rank not in (0, 1):
            return None
        normalized_planar_axes: dict[int, np.ndarray] = {}
        for index in planar_axes:
            candidate_axis = axes[:2, index].copy()
            candidate_norm = float(np.linalg.norm(candidate_axis))
            if candidate_norm < 0.8:
                return None
            normalized_planar_axes[index] = candidate_axis / candidate_norm

        start_pose = self._pick_start_pose_world
        approach_direction = np.zeros(2, dtype=np.float64)
        if start_pose is not None:
            approach_center = (
                self._pick_anchor_center_world
                if self._pick_anchor_center_world is not None
                else source.centroid_world
            )
            approach_direction = (
                approach_center[:2] - start_pose.position[:2]
            )
        approach_norm = float(np.linalg.norm(approach_direction))
        if approach_norm > 1e-6:
            approach_direction /= approach_norm
            preferred_planar_axes = sorted(
                planar_axes,
                key=lambda index: (
                    -abs(
                        float(
                            np.dot(
                                normalized_planar_axes[index],
                                approach_direction,
                            )
                        )
                    ),
                    -float(extents[index]),
                    index,
                ),
            )
        else:
            # Legacy/test adapters that bypass acquire may lack the initial
            # proprio pose.  Preserve the old extent ordering exactly.
            preferred_planar_axes = ranked_planar_axes
        width_axis_index = preferred_planar_axes[axis_preference_rank]
        axis_extent_rank = ranked_planar_axes.index(width_axis_index)
        width_axis_xy = axes[:2, width_axis_index].copy()
        horizontal_norm = float(np.linalg.norm(width_axis_xy))
        width_extent = float(extents[width_axis_index])
        planar_aspect = float(
            extents[ranked_planar_axes[0]]
            / max(float(extents[ranked_planar_axes[1]]), 1e-12)
        )
        # Reject bowl-sized ambiguous relation anchors and strongly tilted or
        # degenerate fixture crops.  Falling back is safer than rotating the
        # hand from an unreliable PCA direction.
        if horizontal_norm < 0.8 or width_extent < 0.14:
            return None
        width_axis_xy /= horizontal_norm
        alignment = float(np.dot(width_axis_xy, current_finger_axis_xy))
        if alignment < -1e-6:
            width_axis_xy *= -1.0
        elif abs(alignment) <= 1e-6:
            dominant = int(np.argmax(np.abs(width_axis_xy)))
            if width_axis_xy[dominant] < 0.0:
                width_axis_xy *= -1.0

        half_width = width_extent * 0.5
        source_coordinate = float(
            np.dot(source.centroid_world[:2] - reference_center[:2], width_axis_xy)
        )
        if half_width <= radius + 0.005 or abs(source_coordinate) > half_width + 0.025:
            return None
        wall_clearances = {
            sign: half_width - radius - sign * source_coordinate
            for sign in (1.0, -1.0)
        }
        center_to_ee = np.asarray(ee_position_xy, dtype=np.float64) - source.centroid_world[:2]
        # Wall clearance is authoritative.  EE proximity is only a stable
        # tie-break for a centred bowl.
        near_sign = 1.0 if float(np.dot(center_to_ee, width_axis_xy)) >= 0.0 else -1.0
        side_order = tuple(
            sorted(
                (1.0, -1.0),
                key=lambda sign: (
                    -wall_clearances[sign],
                    0 if sign == near_sign else 1,
                ),
            )
        )

        cross_z = (
            current_finger_axis_xy[0] * width_axis_xy[1]
            - current_finger_axis_xy[1] * width_axis_xy[0]
        )
        yaw_adjustment = float(
            np.arctan2(cross_z, np.dot(current_finger_axis_xy, width_axis_xy))
        )
        cosine = float(np.cos(yaw_adjustment))
        sine = float(np.sin(yaw_adjustment))
        yaw_rotation = np.array(
            ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
        aligned_rotation = yaw_rotation @ rotation
        label = getattr(reference, "name", None)
        if label is None:
            label = getattr(reference, "label", "fixture")
        return (
            width_axis_xy,
            aligned_rotation,
            (float(side_order[0]), float(side_order[1])),
            wall_clearances,
            yaw_adjustment,
            str(label),
            width_extent,
            planar_aspect,
            axis_extent_rank,
            abs(float(np.dot(width_axis_xy, approach_direction))),
            approach_direction.copy(),
        )

    def _cavity_radial_wall_clearance(
        self,
        source: SceneObject,
        radial_xy: np.ndarray,
        radius: float,
    ) -> float | None:
        """Ray-cast a local-yaw rim proposal against the sensed fixture OBB."""

        reference = self._pick_cavity_reference
        if reference is None:
            return None
        center_value = getattr(reference, "center_world", None)
        if center_value is None:
            center_value = getattr(reference, "centroid_world", None)
        try:
            center = np.asarray(center_value, dtype=np.float64)
            axes = np.asarray(getattr(reference, "axes_world"), dtype=np.float64)
            extents = np.asarray(getattr(reference, "extents_m"), dtype=np.float64)
        except (TypeError, ValueError, AttributeError):
            return None
        if center.shape != (3,) or axes.shape != (3, 3) or extents.shape != (3,):
            return None
        vertical_axis = int(np.argmax(np.abs(axes[2, :])))
        ray_distances: list[float] = []
        source_delta = source.centroid_world[:2] - center[:2]
        for index in range(3):
            if index == vertical_axis:
                continue
            axis_xy = axes[:2, index].copy()
            norm = float(np.linalg.norm(axis_xy))
            if norm < 0.8:
                return None
            axis_xy /= norm
            direction_component = float(np.dot(radial_xy, axis_xy))
            if abs(direction_component) < 1e-8:
                continue
            source_coordinate = float(np.dot(source_delta, axis_xy))
            boundary = float(extents[index]) * 0.5 * np.sign(direction_component)
            ray_distance = (boundary - source_coordinate) / direction_component
            if ray_distance >= 0.0:
                ray_distances.append(float(ray_distance))
        if not ray_distances:
            return None
        return float(min(ray_distances) - radius)

    def _set_place_targets(
        self,
        step: SkillStep,
        source: SceneObject,
        destination: SceneObject,
        observation: SensorObservation,
        *,
        retain_phase: bool = False,
        destination_grounding: str | None = None,
    ) -> None:
        stove_support_trace = None
        if not retain_phase and step.kind is SkillKind.PLACE_ON and step.target == "stove":
            from .stove_support import observed_stove_support

            if self._stove_support_geometry is not None:
                destination = self._stove_support_geometry
                stove_support_trace = {"strategy": "reuse_unoccluded_rgbd_burner_geometry"}
            else:
                try:
                    destination, stove_support_trace = observed_stove_support(observation, destination)
                except (LookupError, ValueError):
                    pass
                else:
                    if (stove_support_trace is not None and stove_support_trace.get("strategy")
                            == "rgbd_visible_burner_disk_plus_public_plate_geometry"):
                        self._stove_support_geometry = destination
        if not retain_phase or self._place_destination_geometry is None:
            # The first place observation is made before the hand moves over
            # the destination and is therefore the least occluded RGB-D view.
            # Freeze that measured support geometry; later refreshes still
            # update the rigidly propagated held source, but cannot drag the
            # landing point onto a finger / carried-object crop.
            self._place_destination_geometry = destination
            self._place_destination_grounding = destination_grounding
        else:
            destination = self._place_destination_geometry
            destination_grounding = self._place_destination_grounding
        placement_source = self._placement_source_geometry(step.kind, source)
        circular_relative_footprint = False
        if (step.kind is SkillKind.PLACE_RELATIVE and source.name.endswith("mug")
                and self._grasp_target_attempts
                and self._grasp_target_attempts[-1].get("mug_rim_surface_fit") is True
                and self._grasp_target_attempts[-1].get("sensor_name") == source.name):
            # A mug's circular body does not grow when the grasp yaw changes.
            # A fresh crop's handle/palm can inflate the AABB and move a nearby
            # placement away from its intended neighbour. Use the measured rim
            # footprint for body-to-body spacing, keeping the existing gap.
            radius = float(self._grasp_target_attempts[-1]["mug_rim_radius_m"])
            half = np.array((radius, radius, source.height_m / 2.0))
            placement_source = SceneObject(
                source.name, source.centroid_world, np.eye(3), 2.0 * half,
                source.centroid_world - half, source.centroid_world + half,
                source.confidence, source.point_count,
            )
            circular_relative_footprint = True
            if destination.name.endswith("mug") and destination.surface_points_world is not None:
                from .rim_geometry import fit_visible_upper_rim

                try:
                    target_xy, target_radius, _ = fit_visible_upper_rim(destination.surface_points_world)
                except ValueError:
                    pass
                else:
                    center = destination.centroid_world.copy()
                    center[:2] = target_xy
                    half = np.array((target_radius, target_radius, destination.height_m / 2.0))
                    destination = SceneObject(
                        destination.name, center, np.eye(3), 2.0 * half,
                        center - half, center + half,
                        destination.confidence, destination.point_count,
                    )
        planning_held_offset = self._held_offset_world.copy()
        side_shelf = bool(destination_grounding and destination_grounding.startswith("sensor-local open shelf "))
        side_cavity = destination_grounding == "sensor-local microwave cavity"
        slatted_rack = destination_grounding == "sensor-local slatted rack"
        if not retain_phase:
            self._place_rotation_world = None
            self._place_rotation_delta_world = None
            self._shelf_entry_pose = None
            if (step.target == "bottom drawer"
                    and destination_grounding == "sensor-local open drawer floor"
                    and self._grasp_mode is GraspMode.RIM_PINCH):
                # Keep the carry wrist fixed in the lower cabinet opening;
                # refreshed contact poses can otherwise accumulate tilt.
                self._place_rotation_world = observation.robot.ee_pose.rotation.copy()
            if side_cavity:
                vertical = int(np.argmax(abs(placement_source.axes_world[2])))
                planar = [i for i in range(3) if i != vertical]
                long_axis = max(planar, key=lambda i: placement_source.extents_m[i])
                up = placement_source.axes_world[:, vertical].copy()
                if up[2] < 0:
                    up *= -1
                width = placement_source.axes_world[:, long_axis]
                source_frame = np.column_stack((width, np.cross(up, width), up))
                outward = destination.axes_world[:, 1]
                candidates = []
                for sign in (1., -1.):
                    width = sign * destination.axes_world[:, 0]
                    target_frame = np.column_stack((width, np.cross(outward, width), outward))
                    delta = target_frame @ source_frame.T
                    candidates.append((float(np.linalg.norm(self._rotation_vector(delta))), delta))
                self._place_rotation_delta_world = min(candidates, key=lambda item: item[0])[1]
                self._place_rotation_world = (
                    self._place_rotation_delta_world @ observation.robot.ee_pose.rotation
                )
            if slatted_rack:
                from .slatted_rack import slatted_bottle_rotation

                self._place_rotation_world = slatted_bottle_rotation(
                    placement_source, destination, observation.robot,
                    self._held_offset_world, self.config.preplace_height_m,
                )
                self._place_rotation_delta_world = (
                    self._place_rotation_world @ observation.robot.ee_pose.rotation.T
                )
            if side_shelf and not self._is_pan_handle_subject(step.subject):
                inward = -destination.axes_world[:, 1]
                jaw = np.array((0.0, 0.0, -1.0))
                self._place_rotation_world = np.column_stack((np.cross(jaw, inward), jaw, inward))
                self._place_rotation_delta_world = (
                    self._place_rotation_world @ observation.robot.ee_pose.rotation.T
                )
            if destination_grounding and destination_grounding.startswith("sensor-local caddy"):
                source_planar = np.sort(source.extents_m[:2])
                target_planar = np.sort(destination.extents_m[:2])
                if source_planar[1] >= 1.4 * source_planar[0] and target_planar[1] >= 1.4 * target_planar[0]:
                    long_source = source.axes_world[:2, int(np.argmax(source.extents_m[:2]))]
                    long_target = destination.axes_world[:2, int(np.argmax(destination.extents_m[:2]))]
                    if np.dot(long_source, long_target) < 0:
                        long_target = -long_target
                    yaw = np.arctan2(
                        long_source[0] * long_target[1] - long_source[1] * long_target[0],
                        np.dot(long_source, long_target),
                    )
                    cosine, sine = np.cos(yaw), np.sin(yaw)
                    delta = np.array(((cosine, -sine, 0.), (sine, cosine, 0.), (0., 0., 1.)))
                    self._place_rotation_delta_world = delta
                    self._place_rotation_world = delta @ observation.robot.ee_pose.rotation
                if step.kind is SkillKind.PLACE_IN and step.subject == "book":
                    from .fitting_orientation import compartment_yaw_rotation

                    preferred_delta = (self._place_rotation_delta_world
                                       if self._place_rotation_delta_world is not None else np.eye(3))
                    delta = compartment_yaw_rotation(
                        placement_source, destination, preferred_delta,
                        alternate_sources=((self._held_alternate_book_geometry,)
                            if self._held_alternate_book_geometry is not None else ()),
                    )
                    if not np.allclose(delta, preferred_delta):
                        self._place_rotation_delta_world = delta
                        self._place_rotation_world = delta @ observation.robot.ee_pose.rotation
        if (self._place_rotation_delta_world is not None
                or self._rim_transfer_rotation_reference is not None):
            delta = (self._place_rotation_world @ observation.robot.ee_pose.rotation.T
                     if self._rim_transfer_rotation_reference is not None
                     else self._place_rotation_delta_world)
            planning_held_offset = delta @ planning_held_offset
            axes = delta @ placement_source.axes_world
            half = np.abs(axes) @ (placement_source.extents_m / 2)
            placement_source = SceneObject(
                name=placement_source.name, centroid_world=placement_source.centroid_world,
                axes_world=axes, extents_m=placement_source.extents_m,
                bounds_min_world=placement_source.centroid_world - half,
                bounds_max_world=placement_source.centroid_world + half,
                confidence=placement_source.confidence, point_count=placement_source.point_count,
            )
        marginal_rim_compensation = bool(
            self._grasp_mode is GraspMode.RIM_PINCH
            and self._grasp_is_marginal
            and not self._held_offset_source.startswith("rgbd_nearest_rim")
        )
        if marginal_rim_compensation:
            planning_held_offset[:2] *= (
                self.config.marginal_rim_placement_xy_offset_scale
            )
        effective_relation = (
            None
            if destination_grounding is not None
            or (
                self._active_carry_stack
                and step.relation == "carry_stack"
            )
            else step.relation
        )
        semantic_direction = self._semantic_relation_direction_world(
            effective_relation,
            observation,
        )
        motion_pose, planned_preplace = self.placement_planner.plan(
            step.kind,
            placement_source,
            destination,
            held_offset_world=planning_held_offset,
            ee_rotation_world=(self._place_rotation_world if self._place_rotation_world is not None
                               else observation.robot.ee_pose.rotation),
            config=self.config,
            # A named compartment/shelf has already been localized as its own
            # RGB-D geometry.  Plan to that measured region centre; applying
            # the original word again as a world-frame offset would rotate it
            # incorrectly for a yawed fixture.
            relation=effective_relation,
            support_z_m=(
                float(self._pick_source_geometry.bounds_min_world[2])
                if self._pick_source_geometry is not None
                else None
            ),
            semantic_direction_world=semantic_direction,
        )
        drawer_landing_trace = None
        if (not retain_phase and step.kind is SkillKind.PLACE_IN
                and self._grasp_mode is GraspMode.RIM_PINCH
                and destination_grounding == "sensor-local open drawer floor"):
            from .drawer_landing import observed_drawer_landing

            try:
                selected, drawer_landing_trace = observed_drawer_landing(
                    observation, placement_source, destination, motion_pose,
                )
            except (LookupError, ValueError):
                pass
            else:
                if drawer_landing_trace is not None:
                    shift = selected.centroid_world - destination.centroid_world
                    destination = selected
                    self._place_destination_geometry = selected
                    motion_pose = Pose(motion_pose.position + shift, motion_pose.rotation)
                    planned_preplace = Pose(planned_preplace.position + shift, planned_preplace.rotation)
        shared_support_offset = self._shared_support_slot_offset(step, placement_source, destination)
        if np.linalg.norm(shared_support_offset) > 0.0:
            motion_pose = Pose(motion_pose.position + shared_support_offset, motion_pose.rotation)
            planned_preplace = Pose(planned_preplace.position + shared_support_offset, planned_preplace.rotation)
        top_open_compartment = bool(
            destination_grounding
            and destination_grounding.startswith("sensor-local caddy ")
        )
        release_above_narrow_opening = bool(
            step.kind is SkillKind.PLACE_IN
            and not side_shelf
            and not side_cavity
            and self._grasp_mode is GraspMode.PINCH
            and (
                step.target == "basket"
                or (
                    placement_source.height_m >= 0.080
                    and (
                        top_open_compartment
                        or min(destination.extents_m[:2])
                        <= min(placement_source.extents_m[:2]) + 0.035
                    )
                )
            )
        )
        if release_above_narrow_opening:
            # Top-open caddy pockets permit gravity seating even when their
            # footprint is wider than the payload. Width alone does not give
            # the fingers and wrist clearance for a deep insertion between
            # partition walls. Keep the hand above the measured opening.
            release = motion_pose.position.copy()
            release[2] = max(release[2], destination.bounds_max_world[2]
                             + placement_source.height_m / 2
                             - planning_held_offset[2] + 0.025)
            motion_pose = Pose(release, motion_pose.rotation)
            high = planned_preplace.position.copy()
            high[2] = max(high[2], release[2] + 0.035)
            planned_preplace = Pose(high, planned_preplace.rotation)
            self._place_expected_settled_center_world = release + planning_held_offset
            self._place_expected_settled_center_world[2] = (
                destination.bounds_min_world[2] + placement_source.height_m / 2
                + self.config.release_clearance_m
            )
        if side_shelf or side_cavity:
            release = motion_pose.position.copy()
            release[2] += 0.010
            motion_pose = Pose(release, motion_pose.rotation)
            outward = destination.axes_world[:, 1]
            payload_depth = float(np.abs(outward @ placement_source.axes_world) @ placement_source.extents_m)
            entry = release + outward * (destination.extents_m[1] / 2.0 + payload_depth / 2.0 + 0.035)
            self._shelf_entry_pose = Pose(entry, motion_pose.rotation)
            high_entry = entry.copy()
            high_entry[2] = max(planned_preplace.position[2], destination.bounds_max_world[2] + 0.080)
            if side_cavity:
                high_entry[2] = max(entry[2] + .080, destination.bounds_max_world[2] + .080)
            planned_preplace = Pose(high_entry, motion_pose.rotation)
            if side_cavity:
                self._side_cavity_payload_half = (
                    abs(destination.axes_world.T @ placement_source.axes_world)
                    @ (placement_source.extents_m / 2)
                )
                self._side_cavity_payload_offset = planning_held_offset.copy()
        if slatted_rack:
            normal = destination.axes_world[:, 2]
            payload_half = float(np.abs(normal @ placement_source.axes_world) @ placement_source.extents_m) / 2.0
            body_center = destination.centroid_world + normal * (payload_half + .008)
            release = body_center - planning_held_offset
            motion_pose = Pose(release, motion_pose.rotation)
            high = release.copy();high[2] += self.config.preplace_height_m
            planned_preplace = Pose(high, motion_pose.rotation)
        rim_transfer_trace = None
        if (not retain_phase and step.kind is SkillKind.PLACE_ON
                and self._grasp_mode is GraspMode.RIM_PINCH
                and "bowl" in step.subject and not self._active_carry_stack
                and not self._held_from_cavity_rim
                and not side_shelf and not top_open_compartment and not slatted_rack
                and self._place_rotation_world is None):
            from .rim_transfer_frame import observed_rim_transfer_frame

            selected = observed_rim_transfer_frame(observation, placement_source, motion_pose,
                max(float(observation.robot.ee_pose.position[2]), float(planned_preplace.position[2])),
                planning_held_offset)
            if selected is not None:
                delta, rim_transfer_trace = selected
                rotated_offset = delta @ planning_held_offset
                shift = planning_held_offset - rotated_offset
                motion_pose = Pose(motion_pose.position + shift, delta @ motion_pose.rotation)
                planned_preplace = Pose(planned_preplace.position + shift, delta @ planned_preplace.rotation)
                planning_held_offset = rotated_offset
                self._place_rotation_world = motion_pose.rotation.copy()
                self._rim_transfer_rotation_reference = observation.robot.ee_pose.rotation.copy()
        self._motion_pose = motion_pose
        if self._rim_transfer_rotation_reference is not None:
            self._place_expected_settled_center_world = motion_pose.position + planning_held_offset
        if not retain_phase:
            self._placement_target_attempts.append(
                {
                    "kind": step.kind.value,
                    "grasp_mode": self._grasp_mode.value,
                    "marginal_rim_compensation": marginal_rim_compensation,
                    "held_offset_source": self._held_offset_source,
                    "measured_held_offset_world_m": self._held_offset_world.tolist(),
                    "planning_held_offset_world_m": planning_held_offset.tolist(),
                    "destination_sensor_center_world_m": (
                        destination.centroid_world.tolist()
                    ),
                    "destination_observation_stage": (
                        self._place_destination_observation_stage
                    ),
                    "destination_grounding": destination_grounding or "whole_entity",
                    "release_above_narrow_opening": release_above_narrow_opening,
                    "top_open_compartment": top_open_compartment,
                    "side_shelf_insertion": side_shelf,
                    "side_cavity_insertion": side_cavity,
                    "circular_relative_footprint": circular_relative_footprint,
                    "slatted_rack_sideways_bottle": slatted_rack,
                    "shared_support_slot_offset_world_m": shared_support_offset.tolist(),
                    "expected_settled_center_world_m": (
                        self._place_expected_settled_center_world.tolist()
                        if self._place_expected_settled_center_world is not None else None
                    ),
                    "semantic_direction_world": (
                        semantic_direction.tolist()
                        if semantic_direction is not None
                        else None
                    ),
                    "formed_stack_carry": bool(
                        self._active_carry_stack
                        and step.relation == "carry_stack"
                    ),
                    "release_ee_target_world_m": motion_pose.position.tolist(),
                }
            )
        if rim_transfer_trace is not None and self._placement_target_attempts:
            self._placement_target_attempts[-1]["rim_transfer_frame"] = rim_transfer_trace
        if drawer_landing_trace is not None and self._placement_target_attempts:
            self._placement_target_attempts[-1]["drawer_landing_clearance"] = drawer_landing_trace
        if stove_support_trace is not None and self._placement_target_attempts:
            self._placement_target_attempts[-1]["stove_support_correction"] = stove_support_trace
        if not retain_phase or self._transfer_clearance_z_m is None:
            # Do not leave a tall support along a source->destination diagonal:
            # first move vertically at the measured current XY, then translate
            # at a constant sensor/proprio-derived clearance height, and only
            # afterwards descend above the destination.
            clearance_z = max(
                float(observation.robot.ee_pose.position[2]),
                float(planned_preplace.position[2]),
            )
            if (
                self._grasp_mode is GraspMode.PINCH
                and self._held_subject is not None
                and observation.robot.gripper_width_m <= 0.040
                and placement_source.height_m >= 0.10
                and not side_cavity
            ):
                # Upright thin objects can pivot below their occluded RGB-D
                # centre during transport. Keep their lower edge clear of
                # fixture lips even before the final vertical descent.
                clearance_z += max(0.0, min(0.050, 1.38 - clearance_z))
            self._transfer_clearance_z_m = clearance_z
            vertical_position = observation.robot.ee_pose.position.copy()
            vertical_position[2] = clearance_z
            self._transfer_clearance_pose = Pose(
                vertical_position,
                observation.robot.ee_pose.rotation,
            )
        preplace_position = planned_preplace.position.copy()
        preplace_position[2] = self._transfer_clearance_z_m
        self._secondary_pose = Pose(preplace_position, planned_preplace.rotation)
        if not retain_phase:
            self._phase_ticks = 0

    def _shared_support_slot_offset(self, step, source, destination) -> np.ndarray:
        """Reserve separate landing centres for independently ranked objects.

        Without advance allocation, the first object occupies the support
        centre and the second is lowered into it. Use the same frozen support
        frame for every placement and space the bodies along its longer side.
        """
        zero = np.zeros(3)
        if (self._spec is not None and step.kind is SkillKind.PLACE_IN
                and step.target == "basket" and step.target_selector is None):
            slots = [i for i, other in enumerate(self._spec.steps)
                     if other.kind is SkillKind.PLACE_IN and other.target == "basket"
                     and other.target_selector is None]
            if len(slots) == 2 and self._skill_index in slots:
                vertical = int(np.argmax(abs(destination.axes_world[2])))
                planar = [i for i in range(3) if i != vertical]
                direction = destination.axes_world[:, planar].sum(axis=1)
                direction[2] = 0.
                if np.linalg.norm(direction) > 1.:
                    direction /= np.linalg.norm(direction)
                    # Two cans fit on opposite diagonals of the square
                    # basket floor. Keep both landing centres inside it.
                    return direction * .035 * (-1 if slots.index(self._skill_index) == 0 else 1)
        if self._spec is None or step.kind is not SkillKind.PLACE_ON or step.target_selector is not None:
            return zero
        slots = []
        selectors = []
        for index, other in enumerate(self._spec.steps):
            if (index == 0 or other.kind is not SkillKind.PLACE_ON or other.target != step.target
                    or other.subject != step.subject or other.target_selector is not None):
                continue
            pick = self._spec.steps[index - 1]
            if pick.kind is not SkillKind.PICK or pick.selector is None or pick.selector in selectors:
                continue
            slots.append(index);selectors.append(pick.selector)
        if len(slots) < 2 or self._skill_index not in slots:
            return zero
        vertical = int(np.argmax(np.abs(destination.axes_world[2])))
        planar = [axis for axis in range(3) if axis != vertical]
        long_axis = max(planar, key=lambda axis: destination.extents_m[axis])
        direction = destination.axes_world[:, long_axis].copy()
        if abs(direction[2]) > .1:
            return zero
        direction[2] = 0.0;direction /= np.linalg.norm(direction)
        source_vertical = int(np.argmax(np.abs(source.axes_world[2])))
        body_radius = min(source.extents_m[axis] for axis in range(3) if axis != source_vertical) / 2.0
        available = destination.extents_m[long_axis] / 2.0 - body_radius - .005
        if available <= 0.0:
            return zero
        spacing = min(2.0 * body_radius + self.config.relative_object_gap_m,
                      2.0 * available / (len(slots) - 1))
        return direction * spacing * (slots.index(self._skill_index) - (len(slots) - 1) / 2.0)

    @staticmethod
    def _shelf_release_command(observation: SensorObservation) -> float:
        # Release a thin book without opening both vertical fingers through
        # the shelf floor and ceiling; withdraw before fully opening the hand.
        return GRIPPER_CLOSE if observation.robot.gripper_width_m > 0.045 else GRIPPER_OPEN

    def _placement_source_geometry(
        self,
        kind: SkillKind,
        source: SceneObject,
    ) -> SceneObject:
        """Bound a support-fused rim object's release height from RGB-D shape."""

        if (
            kind is not SkillKind.PLACE_ON
            or self._grasp_mode is not GraspMode.RIM_PINCH
        ):
            return source
        vertical_axis = int(np.argmax(np.abs(source.axes_world[2, :])))
        planar_axes = tuple(index for index in range(3) if index != vertical_axis)
        planar_diameter = float(np.min(source.extents_m[list(planar_axes)]))
        bounded_half_height = max(
            planar_diameter * self.config.rim_pinch_place_half_height_ratio,
            0.005,
        )
        if source.height_m * 0.5 <= bounded_half_height:
            return source
        bounds_min = source.bounds_min_world.copy()
        bounds_max = source.bounds_max_world.copy()
        bounds_min[2] = source.centroid_world[2] - bounded_half_height
        bounds_max[2] = source.centroid_world[2] + bounded_half_height
        return SceneObject(
            name=source.name,
            centroid_world=source.centroid_world,
            axes_world=source.axes_world,
            extents_m=source.extents_m,
            bounds_min_world=bounds_min,
            bounds_max_world=bounds_max,
            confidence=source.confidence,
            point_count=source.point_count,
        )

    def _motion(
        self, target: Pose, observation: SensorObservation, gripper: float
    ) -> ControlDecision:
        action = self._cartesian_action(observation.robot.ee_pose, target, gripper)
        if self._held_subject is not None and (
            self._grasp_is_marginal
            or self._visual_place_correction_active
            or self._active_carry_stack
        ):
            limit = self.config.weak_grasp_translation_action_limit
            action[:3] = np.clip(action[:3], -limit, limit)
        if (
            self._held_subject is not None
            and self._grasp_mode is GraspMode.PINCH
            and gripper == GRIPPER_CLOSE
            and observation.robot.gripper_width_m <= 0.040
        ):
            # Narrow grasps have little pad contact area. A full diagonal OSC
            # command shed a correctly lifted book before reaching its target.
            # Bound the vector norm so diagonal carries receive the same
            # acceleration limit as axial ones.
            limit = self.config.weak_grasp_translation_action_limit
            norm = float(np.linalg.norm(action[:3]))
            if norm > limit:
                action[:3] *= limit / norm
        return self._tick_decision(action)

    def _cartesian_action(self, current: Pose, target: Pose, gripper: float) -> np.ndarray:
        position_delta = (target.position - current.position) / self.config.position_action_scale_m
        rotation_delta = self._rotation_vector(target.rotation @ current.rotation.T)
        rotation_scale = (
            self.config.cavity_rotation_action_scale_rad
            if self._active_cavity_rim or self._held_from_cavity_rim
            else self.config.rotation_action_scale_rad
        )
        rotation_delta /= rotation_scale
        action = np.concatenate((position_delta, rotation_delta, (gripper,)))
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def _pose_reached(
        self,
        current: Pose,
        target: Pose,
        *,
        position_tolerance_m: float | None = None,
    ) -> bool:
        position_error = np.linalg.norm(target.position - current.position)
        rotation_error = np.linalg.norm(self._rotation_vector(target.rotation @ current.rotation.T))
        tolerance = (
            self.config.position_tolerance_m
            if position_tolerance_m is None
            else float(position_tolerance_m)
        )
        return bool(
            position_error <= tolerance
            and rotation_error <= self.config.rotation_tolerance_rad
        )

    def _pick_refresh_is_consistent(self, source: SceneObject) -> bool:
        """Reject hand-occluded re-detections that jump away from the anchor.

        This is a sensor-only temporal gate.  It compares two RGB-D estimates
        made in the same episode and never consults simulator identity, object
        state, contacts, or task predicates.
        """

        anchor = self._pick_anchor_center_world
        if anchor is None:
            return True
        shift = float(np.linalg.norm(source.centroid_world - anchor))
        return shift <= self.config.pick_refresh_max_shift_m

    def _grasp_retry_height_offset_m(self) -> float:
        """Alternate lower/upper sensor-frame grasp planes across retries."""

        index = self._grasp_retry_index
        if index <= 0:
            return 0.0
        magnitude = ((index + 1) // 2) * self.config.grasp_retry_height_step_m
        return -magnitude if index % 2 else magnitude

    @staticmethod
    def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
        cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
        angle = float(np.arccos(cosine))
        if angle < 1e-8:
            return np.zeros(3, dtype=np.float64)
        sine = np.sin(angle)
        if abs(sine) < 1e-7:
            # Stable eigenvector fallback near pi.
            eigenvalues, eigenvectors = np.linalg.eig(rotation)
            axis = np.real(eigenvectors[:, np.argmin(np.abs(eigenvalues - 1.0))])
            axis /= max(np.linalg.norm(axis), 1e-12)
            return axis * angle
        axis = np.array(
            (
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ),
            dtype=np.float64,
        ) / (2.0 * sine)
        return axis * angle

    def _complete_skill(self, message: str) -> ControlDecision:
        completed_kind = (
            self._spec.steps[self._skill_index].kind
            if self._spec is not None and self._skill_index < len(self._spec.steps)
            else None
        )
        self._skill_index += 1
        self._phase = "acquire"
        self._phase_ticks = 0
        self._perception_misses = 0
        self._motion_pose = None
        self._secondary_pose = None
        self._cavity_switch_pose = None
        self._cavity_level_pose = None
        self._cavity_seat_pose = None
        self._cavity_gripper_preshaped = False
        self._cavity_lift_start_z_m = None
        self._cavity_retry_terminal_failure = None
        self._active_cavity_rim = False
        self._free_rim_retract_pose = None
        self._free_rim_retry_terminal_failure = None
        self._free_rim_retry_anchor_geometry = None
        self._free_rim_antipodal_reacquisition_required = False
        self._free_rim_preshape_reseat_required = False
        self._free_rim_gripper_preshaped = False
        self._free_rim_reseat_reacquisition = None
        self._free_rim_reseat_association_anchor_world = None
        self._free_rim_reseat_association_capture_sequence = None
        self._free_rim_bilateral_completion_sequence = None
        self._free_rim_open_high_retract_completion_sequence = None
        self._clear_cavity_active_view_state()
        self._cavity_primary_radial_world = None
        self._pick_rotation_anchor_world = None
        self._pick_rotation_anchor_source = None
        self._pick_start_pose_world = None
        self._cavity_reference_from_active_view = False
        self._transfer_clearance_pose = None
        self._transfer_clearance_z_m = None
        self._rim_transfer_rotation_reference = None
        self._place_destination_geometry = None
        self._place_rotation_world = None
        self._shelf_entry_pose = None
        self._place_rotation_delta_world = None
        self._place_expected_settled_center_world = None
        self._place_destination_grounding = None
        self._rim_preplace_visual_refreshed = False
        self._visual_place_correction_active = False
        if completed_kind in {
            SkillKind.PLACE_ON,
            SkillKind.PLACE_IN,
            SkillKind.PLACE_RELATIVE,
            SkillKind.STACK,
        }:
            # A following PICK starts a new mechanical attempt.  Retaining the
            # previous source anchor/retry counter can otherwise make the
            # second object in a compound command inherit the first object's
            # identity and grasp geometry.
            self._pick_source_geometry = None
            self._pick_cavity_reference = None
            self._pick_anchor_center_world = None
            self._pick_rotation_anchor_world = None
            self._pick_rotation_anchor_source = None
            self._pick_start_pose_world = None
            self._pick_selector_continuity = False
            self._pick_selector_reacquire_radius_m = None
            self._active_pan_handle_slot_id = None
            self._failed_pan_handle_slot_ids.clear()
            self._pan_body_reference_world = None
            self._held_geometry = None
            self._held_offset_world = np.zeros(3, dtype=np.float64)
            self._held_offset_source = "none"
            self._grasp_is_marginal = False
            self._grasp_retry_index = 0
        if self._spec is not None and self._skill_index >= len(self._spec.steps):
            self._status = ExecutorStatus.SUCCEEDED
            self._phase = "done"
            self._message = message
        return self._decision(
            self._hold_action(self._current_hold_command()), message
        )

    def _perception_miss(self, message: str, *, gripper: float) -> ControlDecision:
        self._perception_misses += 1
        if self._perception_misses >= self.config.max_perception_misses:
            return self._fail(message)
        return self._tick_decision(self._hold_action(gripper), message)

    def _should_refresh_visual(self) -> bool:
        return self._phase_ticks > 0 and self._phase_ticks % self.config.visual_refresh_ticks == 0

    def _set_phase(self, phase: str) -> None:
        self._phase = phase
        self._phase_ticks = 0
        self._previous_position_error_m = None
        self._contact_stall_ticks = 0
        self._contact_error_history = []
        self._contact_position_history = []

    def _contact_reached(self, current: Pose, target: Pose, *, tolerance_m: float) -> bool:
        """Accept a safe Cartesian contact when the commanded descent stalls.

        Container rims and object surfaces can make an OSC target physically
        unreachable.  Waiting for exact pose equality then deadlocks a valid
        manipulation.  This check uses only measured end-effector progress; it
        does not inspect simulator contacts or task predicates.
        """

        error = float(np.linalg.norm(target.position - current.position))
        previous = self._previous_position_error_m
        self._previous_position_error_m = error
        self._contact_error_history.append(error)
        self._contact_position_history.append(current.position.copy())
        window_size = self.config.contact_stall_ticks + 1
        if len(self._contact_error_history) > window_size:
            self._contact_error_history.pop(0)
        if len(self._contact_position_history) > window_size:
            self._contact_position_history.pop(0)
        # Net progress over a window is robust to sub-millimetre OSC jitter.
        # The old consecutive-step reset could run forever when a contacted EE
        # alternated between tiny improvements and rebounds.
        window_stalled = self._contact_window_stalled()
        if previous is not None and previous - error <= self.config.contact_progress_epsilon_m:
            self._contact_stall_ticks += 1
        else:
            self._contact_stall_ticks = 0
        return bool(
            self._phase_ticks >= self.config.contact_min_ticks
            and error <= tolerance_m
            and window_stalled
        )

    def _contact_window_stalled(self) -> bool:
        window_size = self.config.contact_stall_ticks + 1
        if len(self._contact_error_history) != window_size:
            return False
        net_progress = self._contact_error_history[0] - self._contact_error_history[-1]
        return bool(
            net_progress
            <= self.config.contact_progress_epsilon_m
            * self.config.contact_stall_ticks
        )

    def _contact_window_net_progress_m(self) -> float | None:
        """Return signed progress for a complete public-proprio window."""

        window_size = self.config.contact_stall_ticks + 1
        if len(self._contact_error_history) != window_size:
            return None
        net_progress = float(
            self._contact_error_history[0] - self._contact_error_history[-1]
        )
        return net_progress if np.isfinite(net_progress) else None

    def _contact_window_stably_stationary(
        self,
        *,
        max_cartesian_span_m: float,
    ) -> bool:
        """Reject scalar-distance stalls that are sliding or retreating.

        A complete error window must make a small, non-negative approach
        progress while the measured Cartesian path remains spatially local.
        This is stricter than ``_contact_window_stalled`` because a negative
        net change (moving away) and equal-radius tangential motion both look
        stalled to the legacy scalar-only check.
        """

        numeric_epsilon = 1e-12
        net_progress = self._contact_window_net_progress_m()
        cartesian_span = self._contact_cartesian_span_m()
        return bool(
            net_progress is not None
            and -numeric_epsilon <= net_progress
            <= self.config.contact_progress_epsilon_m
            * self.config.contact_stall_ticks
            + numeric_epsilon
            and np.isfinite(cartesian_span)
            and cartesian_span <= max_cartesian_span_m + numeric_epsilon
        )

    def _contact_cartesian_span_m(self) -> float:
        """Maximum measured EE displacement within the current stall window."""

        window_size = self.config.contact_stall_ticks + 1
        if len(self._contact_position_history) != window_size:
            return float("inf")
        positions = np.stack(self._contact_position_history, axis=0)
        pairwise = positions[:, None, :] - positions[None, :, :]
        return float(np.max(np.linalg.norm(pairwise, axis=2)))

    def _cavity_high_pregrasp_reached(self, current: Pose, target: Pose) -> bool:
        """Typed high-waypoint gate for a sensor-confirmed cavity rim pick."""

        if not self._active_cavity_rim:
            return False
        residual = target.position - current.position
        planar_error = float(np.linalg.norm(residual[:2]))
        height_overshoot = -float(residual[2])
        rotation_error = float(
            np.linalg.norm(
                self._rotation_vector(target.rotation @ current.rotation.T)
            )
        )
        accepted = bool(
            planar_error <= self.config.cavity_pregrasp_planar_tolerance_m
            and 0.0
            <= height_overshoot
            <= self.config.cavity_pregrasp_max_height_overshoot_m
            and rotation_error
            <= self.config.cavity_pregrasp_rotation_tolerance_rad
        )
        if self._grasp_target_attempts:
            trace = self._grasp_target_attempts[-1].setdefault(
                "cavity_high_pregrasp_gate",
                {
                    "planar_tolerance_m": float(
                        self.config.cavity_pregrasp_planar_tolerance_m
                    ),
                    "signed_height_overshoot_range_m": [
                        0.0,
                        float(
                            self.config.cavity_pregrasp_max_height_overshoot_m
                        ),
                    ],
                    "rotation_tolerance_rad": float(
                        self.config.cavity_pregrasp_rotation_tolerance_rad
                    ),
                },
            )
            measurement = {
                "residual_world_m": residual.tolist(),
                "planar_error_m": planar_error,
                "height_overshoot_m": height_overshoot,
                "rotation_error_rad": rotation_error,
                "accepted": accepted,
            }
            trace["last_measurement"] = measurement
            if accepted:
                trace["completion"] = dict(measurement)
        return accepted

    def _cavity_operational_clearance_reached(
        self,
        current: Pose,
        target: Pose,
        gripper_width_m: float,
    ) -> bool:
        """Finish proof only at the level, same-XY operational high pose."""

        if (
            not self._active_cavity_rim
            or self._cavity_lift_start_z_m is None
            or not self._grasp_target_attempts
        ):
            return False
        attempt = self._grasp_target_attempts[-1]
        proof = attempt.get("cavity_lift_proof")
        proof_passed = bool(
            isinstance(proof, dict) and proof.get("passed_required_lift") is True
        )
        residual = target.position - current.position
        planar_error = float(np.linalg.norm(residual[:2]))
        target_z_gap = float(residual[2])
        rotation_error = float(
            np.linalg.norm(
                self._rotation_vector(target.rotation @ current.rotation.T)
            )
        )
        width_valid = bool(
            self.config.rim_pinch_blocked_min_width_m
            <= gripper_width_m
            < self.config.gripper_open_width_m
        )
        accepted = bool(
            proof_passed
            and planar_error
            <= self.config.cavity_operational_planar_tolerance_m
            and 0.0
            <= target_z_gap
            <= self.config.cavity_operational_target_z_gap_m
            and rotation_error
            <= self.config.cavity_operational_rotation_tolerance_rad
            and width_valid
        )
        trace = attempt.setdefault(
            "cavity_operational_clearance",
            {
                "strategy": "proof_then_same_xy_vertical_fixture_escape",
                "proof_required_m": float(self.config.cavity_rim_proof_lift_m),
                "planar_tolerance_m": float(
                    self.config.cavity_operational_planar_tolerance_m
                ),
                "signed_target_z_gap_range_m": [
                    0.0,
                    float(self.config.cavity_operational_target_z_gap_m),
                ],
                "rotation_tolerance_rad": float(
                    self.config.cavity_operational_rotation_tolerance_rad
                ),
                "blocked_width_gate_m": [
                    float(self.config.rim_pinch_blocked_min_width_m),
                    float(self.config.gripper_open_width_m),
                ],
                "samples": [],
            },
        )
        measurement = {
            "actual_lift_m": float(current.position[2])
            - self._cavity_lift_start_z_m,
            "residual_world_m": residual.tolist(),
            "planar_error_m": planar_error,
            "target_z_gap_m": target_z_gap,
            "rotation_error_rad": rotation_error,
            "gripper_width_m": float(gripper_width_m),
            "proof_passed": proof_passed,
            "accepted": accepted,
        }
        samples = trace.get("samples")
        if isinstance(samples, list):
            samples.append(measurement)
        if accepted:
            trace["completion"] = dict(measurement)
        return accepted

    def _cavity_retry_high_retract_reached(
        self,
        current: Pose,
        target: Pose | None,
        gripper_width_m: float,
    ) -> bool:
        """Authorize proof/seat retry reacquisition through the typed gate."""

        return self._cavity_open_high_retract_reached(
            current,
            target,
            gripper_width_m,
            trace_key="cavity_retry_retract",
            strategy="typed_open_high_retract_before_sensor_reacquisition",
            authorization_key=(
                "clear_dynamic_cache_and_reacquire_next_sensor_candidate"
            ),
        )

    def _cavity_approach_high_retract_reached(
        self,
        current: Pose,
        target: Pose | None,
        gripper_width_m: float,
    ) -> bool:
        """Authorize reacquisition after a blocked low approach."""

        return self._cavity_open_high_retract_reached(
            current,
            target,
            gripper_width_m,
            trace_key="cavity_approach_retract",
            strategy="typed_open_high_retract_after_blocked_approach",
            authorization_key=(
                "clear_dynamic_cache_and_reacquire_selected_sensor_candidate"
            ),
        )

    def _free_space_rim_high_retract_reached(
        self,
        current: Pose,
        target: Pose | None,
        gripper_width_m: float,
    ) -> bool:
        """Authorize only a high, open-hand retry of the opposite rim.

        The target is built from the measured blocked pose, so this gate
        requires an essentially vertical escape.  Passing it can invalidate
        RGB-D and bind the antipodal exterior edge; it cannot close the hand
        or complete a task.
        """

        target_available = target is not None
        residual: np.ndarray | None = None
        planar_error: float | None = None
        vertical_error: float | None = None
        rotation_error: float | None = None
        if target is not None:
            residual = target.position - current.position
            planar_error = float(np.linalg.norm(residual[:2]))
            vertical_error = abs(float(residual[2]))
            rotation_error = float(
                np.linalg.norm(
                    self._rotation_vector(target.rotation @ current.rotation.T)
                )
            )
        safe_retract_reached = bool(
            self._grasp_mode is GraspMode.RIM_PINCH
            and not self._active_cavity_rim
            and target_available
            and planar_error is not None
            and planar_error <= self.config.grasp_position_tolerance_m
            and vertical_error is not None
            and vertical_error <= self.config.position_tolerance_m
            and rotation_error is not None
            and rotation_error <= self.config.rotation_tolerance_rad
            and np.isfinite(gripper_width_m)
            and gripper_width_m >= self.config.gripper_open_width_m
        )
        reacquire_authorized = bool(
            safe_retract_reached
            and self._free_rim_retry_terminal_failure is None
            and self._free_rim_antipodal_reacquisition_required
            and self._grasp_retry_index == 1
        )
        if self._grasp_target_attempts:
            trace = self._grasp_target_attempts[-1].setdefault(
                "free_space_rim_approach_retract",
                {
                    "strategy": "measured_xy_and_rotation_world_z_to_sensor_high",
                    "planar_tolerance_m": float(
                        self.config.grasp_position_tolerance_m
                    ),
                    "vertical_tolerance_m": float(
                        self.config.position_tolerance_m
                    ),
                    "rotation_tolerance_rad": float(
                        self.config.rotation_tolerance_rad
                    ),
                    "minimum_open_width_m": float(
                        self.config.gripper_open_width_m
                    ),
                    "samples": [],
                },
            )
            measurement = {
                "active_cavity_rim": bool(self._active_cavity_rim),
                "rim_pinch_mode": bool(
                    self._grasp_mode is GraspMode.RIM_PINCH
                ),
                "target_pose_available": target_available,
                "residual_world_m": (
                    residual.tolist() if residual is not None else None
                ),
                "planar_error_m": planar_error,
                "vertical_error_m": vertical_error,
                "rotation_error_rad": rotation_error,
                "gripper_width_m": float(gripper_width_m),
                "safe_retract_reached": safe_retract_reached,
                "reacquire_authorized": reacquire_authorized,
            }
            samples = trace.get("samples")
            if isinstance(samples, list):
                samples.append(measurement)
            trace["last_measurement"] = dict(measurement)
            trace["authorization"] = {
                "fresh_antipodal_reacquisition": reacquire_authorized,
                "close": False,
                "task_success": False,
            }
            if safe_retract_reached:
                trace["completion"] = dict(measurement)
        return safe_retract_reached

    def _cavity_open_high_retract_reached(
        self,
        current: Pose,
        target: Pose | None,
        gripper_width_m: float,
        *,
        trace_key: str,
        strategy: str,
        authorization_key: str,
    ) -> bool:
        """Authorize only a high, open-hand RGB-D cavity reacquisition.

        A rejected rim attempt does not need the isotropic 3-mm grasp-pose
        tolerance before the next sensor observation.  It does need to be
        measurably high, level, and laterally aligned with the sensor-derived
        escape waypoint, with the fingers fully open.  Passing this gate only
        authorizes cache invalidation and candidate reacquisition; it is not a
        grasp, close, or task-success condition.
        """

        if not self._grasp_target_attempts:
            return False
        attempt = self._grasp_target_attempts[-1]
        source_available = self._pick_source_geometry is not None
        target_available = target is not None
        typed_cavity_available = bool(
            self._grasp_mode is GraspMode.RIM_PINCH
            and self._active_cavity_rim
            and source_available
            and target_available
        )

        residual: np.ndarray | None = None
        planar_error: float | None = None
        target_z_gap: float | None = None
        rotation_error: float | None = None
        if target is not None:
            residual = target.position - current.position
            planar_error = float(np.linalg.norm(residual[:2]))
            target_z_gap = float(residual[2])
            rotation_error = float(
                np.linalg.norm(
                    self._rotation_vector(target.rotation @ current.rotation.T)
                )
            )

        # Rotation-vector recovery through arccos can place an exact closed
        # boundary a few ulps above its source angle.  The epsilon preserves
        # the specified inclusive physical limits without admitting a
        # meaningful amount of extra motion.
        numeric_epsilon = 1e-12
        planar_valid = bool(
            planar_error is not None
            and planar_error
            <= self.config.cavity_operational_planar_tolerance_m
            + numeric_epsilon
        )
        target_z_valid = bool(
            target_z_gap is not None
            # Being slightly above a sensor-derived high waypoint is safer
            # than being below it.  Reuse the already validated cavity-high
            # overshoot bound instead of deadlocking an otherwise exact,
            # fully open recovery on the sign of sub-millimetre OSC error.
            and -self.config.cavity_pregrasp_max_height_overshoot_m
            - numeric_epsilon
            <= target_z_gap
            <= self.config.cavity_operational_target_z_gap_m
            + numeric_epsilon
        )
        rotation_valid = bool(
            rotation_error is not None
            and rotation_error
            <= self.config.cavity_operational_rotation_tolerance_rad
            + numeric_epsilon
        )
        width_valid = bool(
            np.isfinite(gripper_width_m)
            and gripper_width_m >= self.config.gripper_open_width_m
        )
        authorized = bool(
            typed_cavity_available
            and planar_valid
            and target_z_valid
            and rotation_valid
            and width_valid
        )

        trace = attempt.setdefault(
            trace_key,
            {
                "strategy": strategy,
                "planar_tolerance_m": float(
                    self.config.cavity_operational_planar_tolerance_m
                ),
                "signed_target_z_gap_range_m": [
                    -float(
                        self.config.cavity_pregrasp_max_height_overshoot_m
                    ),
                    float(self.config.cavity_operational_target_z_gap_m),
                ],
                "rotation_tolerance_rad": float(
                    self.config.cavity_operational_rotation_tolerance_rad
                ),
                "minimum_open_width_m": float(
                    self.config.gripper_open_width_m
                ),
                "selected_candidate_index": int(self._cavity_candidate_index),
                "samples": [],
            },
        )
        measurement = {
            "active_cavity_rim": bool(self._active_cavity_rim),
            "rim_pinch_mode": bool(self._grasp_mode is GraspMode.RIM_PINCH),
            "source_sensor_geometry_available": source_available,
            "target_pose_available": target_available,
            "residual_world_m": residual.tolist() if residual is not None else None,
            "planar_error_m": planar_error,
            "target_z_gap_m": target_z_gap,
            "rotation_error_rad": rotation_error,
            "gripper_width_m": float(gripper_width_m),
            "planar_gate_passed": planar_valid,
            "signed_target_z_gate_passed": target_z_valid,
            "rotation_gate_passed": rotation_valid,
            "open_width_gate_passed": width_valid,
            "reacquire_authorized": authorized,
        }
        samples = trace.get("samples")
        if isinstance(samples, list):
            samples.append(measurement)
        trace["last_measurement"] = dict(measurement)
        trace["authorization"] = {
            authorization_key: authorized,
            "close": False,
            "task_success": False,
        }
        if authorized:
            trace["completion"] = dict(measurement)
        return authorized

    def _cavity_rim_contact_reached(self, current: Pose, target: Pose) -> bool:
        """Accept only a tightly aligned, vertically blocked cavity rim pose."""

        if (
            self._grasp_mode is not GraspMode.RIM_PINCH
            or not self._active_cavity_rim
            or self._phase_ticks < self.config.contact_min_ticks
            or not self._contact_window_stalled()
        ):
            return False
        residual = target.position - current.position
        planar_error = float(np.linalg.norm(residual[:2]))
        vertical_clearance = -float(residual[2])
        rotation_error = float(
            np.linalg.norm(self._rotation_vector(target.rotation @ current.rotation.T))
        )
        return bool(
            planar_error <= self.config.cavity_rim_contact_xy_tolerance_m
            and 0.0
            <= vertical_clearance
            <= self.config.cavity_rim_contact_vertical_tolerance_m
            and rotation_error <= self.config.rotation_tolerance_rad
        )

    def _cavity_rim_preshape_ready(self, gripper_width_m: float) -> bool:
        """Require a visibly open, drawer-safe aperture before rim descent."""

        return bool(
            self.config.cavity_rim_preshape_min_width_m
            <= gripper_width_m
            <= self.config.cavity_rim_preshape_max_width_m
        )

    def _cavity_rim_preshape_command(self, gripper_width_m: float) -> float:
        """Feedback command for the binary Panda hand around a partial width."""

        if gripper_width_m > self.config.cavity_rim_preshape_target_width_m:
            return GRIPPER_CLOSE
        return GRIPPER_OPEN

    def _free_space_rim_preshape_ready(self, gripper_width_m: float) -> bool:
        """Require the isolated exterior-rim partial closing aperture."""

        return bool(
            np.isfinite(gripper_width_m)
            and self.config.free_space_rim_preshape_min_width_m
            <= gripper_width_m
            <= self.config.free_space_rim_preshape_max_width_m
        )

    def _free_space_rim_preshape_command(self, gripper_width_m: float) -> float:
        """Binary closing feedback for the exterior-rim high preshape."""

        if (
            np.isfinite(gripper_width_m)
            and gripper_width_m
            > self.config.free_space_rim_preshape_target_width_m
        ):
            return GRIPPER_CLOSE
        return GRIPPER_OPEN

    def _free_space_rim_preshape_close_reached(
        self,
        current: Pose,
        target: Pose,
        gripper_width_m: float,
    ) -> bool:
        """Authorize final CLOSE only at a fresh sensor/Panda-pad rim seat."""

        if (
            self._grasp_mode is not GraspMode.RIM_PINCH
            or self._active_cavity_rim
            or not self._free_rim_preshape_reseat_required
            or not self._free_rim_gripper_preshaped
            or not self._grasp_target_attempts
        ):
            return False
        attempt = self._grasp_target_attempts[-1]
        record = attempt.get("free_space_rim_reseat_reacquisition")
        preshape = attempt.get("free_space_rim_preshape")
        source = self._pick_source_geometry
        anchor_geometry = self._free_rim_retry_anchor_geometry
        anchor = self._pick_anchor_center_world
        eps = 1e-12

        def vector(value: object) -> np.ndarray | None:
            try:
                candidate = np.asarray(value, dtype=np.float64)
            except (TypeError, ValueError):
                return None
            if candidate.shape != (3,) or not np.all(np.isfinite(candidate)):
                return None
            return candidate

        attempt_center = vector(attempt.get("sensor_center_world_m"))
        attempt_extents = vector(attempt.get("sensor_extents_m"))
        candidate_center = (
            vector(record.get("candidate_center_world_m"))
            if isinstance(record, dict)
            else None
        )
        candidate_extents = (
            vector(record.get("candidate_extents_sorted_m"))
            if isinstance(record, dict)
            else None
        )
        recorded_anchor = (
            vector(record.get("anchor_center_world_m"))
            if isinstance(record, dict)
            else None
        )
        recorded_identity_anchor = (
            vector(record.get("identity_anchor_center_world_m"))
            if isinstance(record, dict)
            else None
        )
        recorded_association_anchor = (
            vector(record.get("association_anchor_center_world_m"))
            if isinstance(record, dict)
            else None
        )
        recorded_anchor_extents = (
            vector(record.get("anchor_extents_sorted_m"))
            if isinstance(record, dict)
            else None
        )
        capture = record.get("capture_sequence") if isinstance(record, dict) else None
        association_capture = (
            record.get("association_anchor_capture_sequence")
            if isinstance(record, dict)
            else None
        )
        bilateral_sequence = (
            record.get("bilateral_completion_sequence")
            if isinstance(record, dict)
            else None
        )
        retract_sequence = (
            record.get("open_high_retract_completion_sequence")
            if isinstance(record, dict)
            else None
        )
        recorded_radius = (
            record.get("anchor_radius_m") if isinstance(record, dict) else None
        )
        recorded_leg_distance = (
            record.get("association_leg_distance_m")
            if isinstance(record, dict)
            else None
        )
        recorded_nearest_distance = (
            record.get("nearest_bound_distance_m")
            if isinstance(record, dict)
            else None
        )
        recorded_anchor_leg_distance = (
            record.get("association_anchor_initial_distance_m")
            if isinstance(record, dict)
            else None
        )
        recorded_cumulative_distance = (
            record.get("initial_anchor_cumulative_distance_m")
            if isinstance(record, dict)
            else None
        )
        recorded_cumulative_limit = (
            record.get("initial_anchor_cumulative_limit_m")
            if isinstance(record, dict)
            else None
        )
        recorded_scales = (
            vector(record.get("candidate_extent_scales"))
            if isinstance(record, dict)
            else None
        )
        actual_leg_distance = (
            float(np.linalg.norm(candidate_center - recorded_association_anchor))
            if candidate_center is not None
            and recorded_association_anchor is not None
            else None
        )
        actual_anchor_leg_distance = (
            float(np.linalg.norm(recorded_association_anchor - anchor))
            if recorded_association_anchor is not None and anchor is not None
            else None
        )
        actual_cumulative_distance = (
            float(np.linalg.norm(candidate_center - anchor))
            if candidate_center is not None and anchor is not None
            else None
        )
        actual_scales = (
            candidate_extents / recorded_anchor_extents
            if candidate_extents is not None
            and recorded_anchor_extents is not None
            and np.all(recorded_anchor_extents > 0.0)
            else None
        )
        normalized_name = (
            " ".join(source.name.lower().replace("_", " ").split())
            if source is not None
            else ""
        )

        attempt1 = (
            self._grasp_target_attempts[-2]
            if len(self._grasp_target_attempts) >= 3
            else None
        )
        attempt0 = (
            self._grasp_target_attempts[-3]
            if len(self._grasp_target_attempts) >= 3
            else None
        )
        attempt0_capture = (
            attempt0.get("capture_sequence")
            if isinstance(attempt0, dict)
            else None
        )
        first_fallback = (
            attempt0.get("free_space_rim_approach_fallback")
            if isinstance(attempt0, dict)
            else None
        )
        first_fallback_capture = (
            first_fallback.get("capture_sequence")
            if isinstance(first_fallback, dict)
            else None
        )
        first_records = (
            attempt0.get("free_space_rim_retry_reacquisition")
            if isinstance(attempt0, dict)
            else None
        )
        first_record: dict[str, object] | None = None
        if isinstance(first_records, list):
            for item in reversed(first_records):
                if (
                    isinstance(item, dict)
                    and item.get("capture_sequence") == association_capture
                ):
                    first_record = item
                    break
        first_record_capture = (
            first_record.get("capture_sequence")
            if isinstance(first_record, dict)
            else None
        )
        first_candidate_center = (
            vector(first_record.get("candidate_center_world_m"))
            if isinstance(first_record, dict)
            else None
        )
        first_recorded_anchor = (
            vector(first_record.get("anchor_center_world_m"))
            if isinstance(first_record, dict)
            else None
        )
        first_anchor_extents = (
            vector(first_record.get("anchor_extents_sorted_m"))
            if isinstance(first_record, dict)
            else None
        )
        first_candidate_extents = (
            vector(first_record.get("candidate_extents_sorted_m"))
            if isinstance(first_record, dict)
            else None
        )
        first_scales = (
            vector(first_record.get("candidate_extent_scales"))
            if isinstance(first_record, dict)
            else None
        )
        first_recorded_distance = (
            first_record.get("nearest_bound_distance_m")
            if isinstance(first_record, dict)
            else None
        )
        first_recorded_radius = (
            first_record.get("anchor_radius_m")
            if isinstance(first_record, dict)
            else None
        )
        attempt1_center = (
            vector(attempt1.get("sensor_center_world_m"))
            if isinstance(attempt1, dict)
            else None
        )
        attempt1_capture = (
            attempt1.get("capture_sequence")
            if isinstance(attempt1, dict)
            else None
        )
        attempt0_extents = (
            vector(attempt0.get("sensor_extents_m"))
            if isinstance(attempt0, dict)
            else None
        )
        attempt1_extents = (
            vector(attempt1.get("sensor_extents_m"))
            if isinstance(attempt1, dict)
            else None
        )
        first_fallback_authorization = (
            first_fallback.get("authorization")
            if isinstance(first_fallback, dict)
            else None
        )
        bilateral = (
            attempt1.get("free_space_rim_bilateral_contact")
            if isinstance(attempt1, dict)
            else None
        )
        bilateral_completion = (
            bilateral.get("completion")
            if isinstance(bilateral, dict)
            else None
        )
        bilateral_authorization = (
            bilateral.get("authorization")
            if isinstance(bilateral, dict)
            else None
        )
        bilateral_trace_sequence = (
            bilateral.get("bilateral_completion_sequence")
            if isinstance(bilateral, dict)
            else None
        )
        bilateral_completion_sequence = (
            bilateral_completion.get("bilateral_completion_sequence")
            if isinstance(bilateral_completion, dict)
            else None
        )
        bilateral_fresh_sequence = (
            bilateral_completion.get("fresh_reacquisition_capture_sequence")
            if isinstance(bilateral_completion, dict)
            else None
        )
        retract_trace = (
            attempt1.get("free_space_rim_approach_retract")
            if isinstance(attempt1, dict)
            else None
        )
        retract_completion = (
            retract_trace.get("completion")
            if isinstance(retract_trace, dict)
            else None
        )
        retract_authorization = (
            retract_trace.get("authorization")
            if isinstance(retract_trace, dict)
            else None
        )
        retract_trace_sequence = (
            retract_trace.get("open_high_retract_completion_sequence")
            if isinstance(retract_trace, dict)
            else None
        )
        retract_completion_sequence = (
            retract_completion.get("open_high_retract_completion_sequence")
            if isinstance(retract_completion, dict)
            else None
        )
        cumulative_limit = (
            2.0 * float(self._pick_selector_reacquire_radius_m)
            if self._pick_selector_reacquire_radius_m is not None
            else None
        )
        identity_size_continuous = bool(
            type(attempt.get("attempt")) is int
            and attempt.get("attempt") == 1
            and attempt.get("rim_strategy") == "free_space"
            and attempt.get("active_cavity_rim") is False
            and attempt.get("free_space_rim_preshape_reseat") is True
            and isinstance(record, dict)
            and record.get("nearest_bound_result") == "accepted"
            and record.get("selected_source")
            == "fresh_nearest_bound_attempt1_association_anchor"
            and type(record.get("attempt")) is int
            and record.get("attempt") == 1
            and record.get("size_continuous") is True
            and isinstance(attempt0, dict)
            and type(attempt0.get("attempt")) is int
            and attempt0.get("attempt") == 0
            and attempt0.get("rim_strategy") == "free_space"
            and attempt0.get("active_cavity_rim") is False
            and isinstance(attempt1, dict)
            and type(attempt1.get("attempt")) is int
            and attempt1.get("attempt") == 1
            and attempt1.get("rim_strategy") == "free_space"
            and attempt1.get("active_cavity_rim") is False
            and type(attempt0_capture) is int
            and type(first_fallback_capture) is int
            and type(first_record_capture) is int
            and type(attempt1_capture) is int
            and type(association_capture) is int
            and type(bilateral_sequence) is int
            and type(bilateral_trace_sequence) is int
            and type(bilateral_completion_sequence) is int
            and type(bilateral_fresh_sequence) is int
            and type(retract_sequence) is int
            and type(retract_trace_sequence) is int
            and type(retract_completion_sequence) is int
            and type(capture) is int
            and type(attempt.get("capture_sequence")) is int
            and type(self._free_rim_reseat_association_capture_sequence) is int
            and type(self._free_rim_bilateral_completion_sequence) is int
            and type(
                self._free_rim_open_high_retract_completion_sequence
            )
            is int
            and attempt.get("capture_sequence") == capture
            and 0 <= attempt0_capture
            < first_fallback_capture
            < association_capture
            < bilateral_sequence
            < retract_sequence
            < capture
            < self._observation_timestamp_s
            and source is not None
            and anchor_geometry is not None
            and anchor is not None
            and normalized_name in {"black bowl", "bowl"}
            and source.name == anchor_geometry.name
            and attempt.get("sensor_name") == source.name
            and attempt0.get("sensor_name") == anchor_geometry.name
            and attempt1.get("sensor_name") == anchor_geometry.name
            and attempt_center is not None
            and candidate_center is not None
            and recorded_anchor is not None
            and recorded_identity_anchor is not None
            and recorded_association_anchor is not None
            and np.allclose(source.centroid_world, attempt_center, rtol=0.0, atol=1e-9)
            and np.allclose(attempt_center, candidate_center, rtol=0.0, atol=1e-9)
            and np.allclose(recorded_anchor, anchor, rtol=0.0, atol=1e-9)
            and np.allclose(recorded_identity_anchor, anchor, rtol=0.0, atol=1e-9)
            and self._free_rim_reseat_association_anchor_world is not None
            and np.allclose(
                recorded_association_anchor,
                self._free_rim_reseat_association_anchor_world,
                rtol=0.0,
                atol=1e-9,
            )
            and self._free_rim_reseat_association_capture_sequence
            == association_capture
            and self._free_rim_bilateral_completion_sequence
            == bilateral_sequence
            and self._free_rim_open_high_retract_completion_sequence
            == retract_sequence
            and attempt1_center is not None
            and np.allclose(attempt1_center, recorded_association_anchor, rtol=0.0, atol=1e-9)
            and type(attempt1.get("capture_sequence")) is int
            and attempt1_capture == association_capture
            and isinstance(first_record, dict)
            and first_record.get("nearest_bound_result") == "accepted"
            and first_record.get("selected_source")
            == "fresh_nearest_bound_attempt0_anchor"
            and first_record.get("size_continuous") is True
            and type(first_record.get("attempt")) is int
            and first_record.get("attempt") == 1
            and first_record_capture == association_capture
            and isinstance(first_fallback, dict)
            and first_fallback.get("strategy")
            == "proprio_stall_open_vertical_retract_then_fresh_antipodal_rim"
            and type(first_fallback.get("next_retry_index")) is int
            and first_fallback.get("next_retry_index") == 1
            and isinstance(first_fallback_authorization, dict)
            and first_fallback_authorization.get("close") is False
            and first_fallback_authorization.get("task_success") is False
            and first_fallback_authorization.get(
                "fresh_antipodal_reacquisition"
            )
            is True
            and first_candidate_center is not None
            and np.allclose(first_candidate_center, recorded_association_anchor, rtol=0.0, atol=1e-9)
            and first_recorded_anchor is not None
            and np.allclose(first_recorded_anchor, anchor, rtol=0.0, atol=1e-9)
            and isinstance(first_recorded_radius, (int, float))
            and np.isfinite(first_recorded_radius)
            and isinstance(recorded_radius, (int, float))
            and np.isfinite(recorded_radius)
            and abs(float(first_recorded_radius) - float(recorded_radius)) <= 1e-9
            and isinstance(first_recorded_distance, (int, float))
            and np.isfinite(first_recorded_distance)
            and actual_anchor_leg_distance is not None
            and abs(
                float(first_recorded_distance) - actual_anchor_leg_distance
            )
            <= 1e-9
            and attempt0_extents is not None
            and attempt1_extents is not None
            and first_anchor_extents is not None
            and first_candidate_extents is not None
            and first_scales is not None
            and np.all(first_anchor_extents > 0.0)
            and np.allclose(
                first_anchor_extents,
                np.sort(attempt0_extents),
                rtol=0.0,
                atol=1e-9,
            )
            and np.allclose(
                first_candidate_extents,
                np.sort(attempt1_extents),
                rtol=0.0,
                atol=1e-9,
            )
            and np.allclose(
                first_scales,
                first_candidate_extents / first_anchor_extents,
                rtol=0.0,
                atol=1e-9,
            )
            and np.all(
                first_scales
                >= self.config.free_space_rim_retry_min_extent_scale - eps
            )
            and np.all(
                first_scales
                <= self.config.free_space_rim_retry_max_extent_scale + eps
            )
            and isinstance(bilateral, dict)
            and bilateral_trace_sequence == bilateral_sequence
            and isinstance(bilateral_completion, dict)
            and bilateral_completion_sequence == bilateral_sequence
            and bilateral_fresh_sequence == association_capture
            and bilateral_completion.get("preshape_reseat_authorized") is True
            and bilateral_completion.get("close_authorized") is False
            and isinstance(bilateral_authorization, dict)
            and bilateral_authorization.get("close") is False
            and bilateral_authorization.get("task_success") is False
            and bilateral_authorization.get(
                "open_high_retract_then_fresh_preshape_reseat"
            )
            is True
            and isinstance(retract_trace, dict)
            and retract_trace_sequence == retract_sequence
            and isinstance(retract_completion, dict)
            and retract_completion_sequence == retract_sequence
            and retract_completion.get("safe_retract_reached") is True
            and retract_completion.get("reacquire_authorized") is True
            and isinstance(retract_authorization, dict)
            and retract_authorization.get("close") is False
            and retract_authorization.get("task_success") is False
            and retract_authorization.get("fresh_antipodal_reacquisition") is True
            and attempt_extents is not None
            and candidate_extents is not None
            and recorded_anchor_extents is not None
            and np.allclose(np.sort(source.extents_m), candidate_extents, rtol=0.0, atol=1e-9)
            and np.allclose(np.sort(attempt_extents), candidate_extents, rtol=0.0, atol=1e-9)
            and np.allclose(np.sort(anchor_geometry.extents_m), recorded_anchor_extents, rtol=0.0, atol=1e-9)
            and isinstance(recorded_radius, (int, float))
            and np.isfinite(recorded_radius)
            and self._pick_selector_reacquire_radius_m is not None
            and abs(float(recorded_radius) - self._pick_selector_reacquire_radius_m)
            <= 1e-9
            and actual_leg_distance is not None
            and isinstance(recorded_leg_distance, (int, float))
            and np.isfinite(recorded_leg_distance)
            and abs(float(recorded_leg_distance) - actual_leg_distance) <= 1e-9
            and isinstance(recorded_nearest_distance, (int, float))
            and np.isfinite(recorded_nearest_distance)
            and abs(float(recorded_nearest_distance) - actual_leg_distance)
            <= 1e-9
            and actual_leg_distance <= float(recorded_radius) + eps
            and actual_anchor_leg_distance is not None
            and isinstance(recorded_anchor_leg_distance, (int, float))
            and np.isfinite(recorded_anchor_leg_distance)
            and abs(
                float(recorded_anchor_leg_distance)
                - actual_anchor_leg_distance
            )
            <= 1e-9
            and actual_anchor_leg_distance <= float(recorded_radius) + eps
            and actual_cumulative_distance is not None
            and isinstance(recorded_cumulative_distance, (int, float))
            and np.isfinite(recorded_cumulative_distance)
            and abs(
                float(recorded_cumulative_distance)
                - actual_cumulative_distance
            )
            <= 1e-9
            and cumulative_limit is not None
            and isinstance(recorded_cumulative_limit, (int, float))
            and np.isfinite(recorded_cumulative_limit)
            and abs(float(recorded_cumulative_limit) - cumulative_limit) <= 1e-9
            and actual_cumulative_distance <= cumulative_limit + eps
            and actual_scales is not None
            and recorded_scales is not None
            and np.allclose(recorded_scales, actual_scales, rtol=0.0, atol=1e-9)
            and np.all(
                actual_scales
                >= self.config.free_space_rim_retry_min_extent_scale - eps
            )
            and np.all(
                actual_scales
                <= self.config.free_space_rim_retry_max_extent_scale + eps
            )
        )
        preshape_completion = (
            preshape.get("completion") if isinstance(preshape, dict) else None
        )
        completed_width = (
            preshape_completion.get("gripper_width_m")
            if isinstance(preshape_completion, dict)
            else None
        )
        preshape_valid = bool(
            isinstance(completed_width, (int, float))
            and self._free_space_rim_preshape_ready(float(completed_width))
            and self._free_space_rim_preshape_ready(gripper_width_m)
        )

        target_trace = vector(attempt.get("target_world_m"))
        target_matches = bool(
            target_trace is not None
            and np.allclose(target.position, target_trace, rtol=0.0, atol=1e-9)
        )
        finger_axis = vector(attempt.get("finger_axis_world"))
        recorded_rim_radius = attempt.get("rim_radius_m")
        calculated_fresh_radius: float | None = None
        expected_fresh_target_z: float | None = None
        if source is not None:
            vertical_axis = int(np.argmax(np.abs(source.axes_world[2, :])))
            planar_axes = tuple(index for index in range(3) if index != vertical_axis)
            calculated_fresh_radius = float(
                np.min(source.extents_m[list(planar_axes)]) / 2.0
                - self.config.rim_pinch_radial_inset_m
            )
            expected_fresh_target_z = self._pinch_grasp_height(
                source,
                top_inset_m=self.config.free_space_rim_grasp_top_inset_m,
                apply_retry_offset=False,
            )
        fresh_top = attempt.get("fresh_top_world_m")
        fresh_target_geometry_bound = bool(
            attempt.get("target_center_source")
            == "fresh_reseat_candidate_center_xy_and_top_z"
            and target_trace is not None
            and candidate_center is not None
            and source is not None
            and finger_axis is not None
            and abs(float(finger_axis[2])) <= 1e-9
            and abs(float(np.linalg.norm(finger_axis[:2])) - 1.0) <= 1e-9
            and isinstance(recorded_rim_radius, (int, float))
            and np.isfinite(recorded_rim_radius)
            and calculated_fresh_radius is not None
            and abs(float(recorded_rim_radius) - calculated_fresh_radius) <= 1e-9
            and np.allclose(
                target_trace[:2],
                candidate_center[:2]
                + float(recorded_rim_radius) * finger_axis[:2],
                rtol=0.0,
                atol=1e-9,
            )
            and isinstance(fresh_top, (int, float))
            and np.isfinite(fresh_top)
            and abs(float(fresh_top) - float(source.bounds_max_world[2])) <= 1e-9
            and expected_fresh_target_z is not None
            and abs(float(target_trace[2]) - expected_fresh_target_z) <= 1e-9
        )
        residual = target.position - current.position
        radial_residual: float | None = None
        tangential_residual: float | None = None
        # The final contact frame belongs to this third RGB-D candidate.  The
        # attempt-zero anchor is deliberately absent from this pose residual;
        # it is retained above only for identity/size/distance continuity.
        if candidate_center is not None and target_trace is not None:
            radial = target_trace[:2] - candidate_center[:2]
            norm = float(np.linalg.norm(radial))
            if norm > 0.0:
                unit = radial / norm
                radial_residual = float(np.dot(residual[:2], unit))
                tangential_residual = float(
                    -residual[0] * unit[1] + residual[1] * unit[0]
                )
        rotation_error = float(
            np.linalg.norm(self._rotation_vector(target.rotation @ current.rotation.T))
        )
        tool_z_world_z = float(current.rotation[2, 2])
        pad_offsets = sorted(
            (
                self.config.panda_pad_local_z_min_from_grip_site_m
                * tool_z_world_z,
                self.config.panda_pad_local_z_max_from_grip_site_m
                * tool_z_world_z,
            )
        )
        top_from_grip: float | None = None
        if isinstance(fresh_top, (int, float)) and np.isfinite(fresh_top):
            top_from_grip = float(fresh_top) - float(current.position[2])
        pad_band_valid = bool(
            abs(tool_z_world_z) >= 0.95
            and top_from_grip is not None
            and pad_offsets[0] - eps <= top_from_grip <= pad_offsets[1] + eps
        )
        authorized = bool(
            identity_size_continuous
            and fresh_target_geometry_bound
            and preshape_valid
            and target_matches
            and radial_residual is not None
            and abs(radial_residual) <= self.config.grasp_position_tolerance_m + eps
            and tangential_residual is not None
            and abs(tangential_residual) <= self.config.grasp_position_tolerance_m + eps
            and rotation_error <= self.config.rotation_tolerance_rad + eps
            and pad_band_valid
        )
        trace = attempt.setdefault(
            "free_space_rim_preshape_close_gate",
            {
                "strategy": "fresh_exact_rim_target_with_public_panda_pad_band",
                "radial_tolerance_m": float(self.config.grasp_position_tolerance_m),
                "tangential_tolerance_m": float(self.config.grasp_position_tolerance_m),
                "preshape_width_band_m": [
                    float(self.config.free_space_rim_preshape_min_width_m),
                    float(self.config.free_space_rim_preshape_max_width_m),
                ],
                "panda_pad_local_z_band_from_grip_site_m": [
                    float(self.config.panda_pad_local_z_min_from_grip_site_m),
                    float(self.config.panda_pad_local_z_max_from_grip_site_m),
                ],
                "samples": [],
            },
        )
        measurement = {
            "fresh_identity_and_size_continuous": identity_size_continuous,
            "fresh_candidate_owns_rim_target_geometry": (
                fresh_target_geometry_bound
            ),
            "fresh_rim_center_world_m": (
                candidate_center.tolist()
                if candidate_center is not None
                else None
            ),
            "calculated_fresh_rim_radius_m": calculated_fresh_radius,
            "preshape_band_proved": preshape_valid,
            "current_target_matches_fresh_trace": target_matches,
            "radial_residual_m": radial_residual,
            "tangential_residual_m": tangential_residual,
            "rotation_error_rad": rotation_error,
            "gripper_width_m": float(gripper_width_m),
            "fresh_sensor_top_world_z_m": (
                float(fresh_top)
                if isinstance(fresh_top, (int, float))
                else None
            ),
            "sensor_top_from_grip_site_world_z_m": top_from_grip,
            "tool_z_world_z": tool_z_world_z,
            "projected_pad_world_z_band_from_grip_site_m": pad_offsets,
            "pad_work_band_gate_passed": pad_band_valid,
            "public_proprio_stalled_used_for_authorization": False,
            "close_authorized": authorized,
        }
        samples = trace.get("samples")
        if isinstance(samples, list):
            samples.append(measurement)
        trace["last_measurement"] = dict(measurement)
        trace["authorization"] = {
            "close": authorized,
            "task_success": False,
            "post_close_retention_and_lift_proof_required": True,
        }
        if authorized:
            trace["completion"] = dict(measurement)
        return authorized

    def _free_space_rim_preshape_reseat_should_fail(
        self,
        current: Pose,
        target: Pose,
        gripper_width_m: float,
    ) -> bool:
        """Fail closed after a stationary but non-authorized final re-entry."""

        del current, target
        return bool(
            self._grasp_mode is GraspMode.RIM_PINCH
            and not self._active_cavity_rim
            and self._free_rim_preshape_reseat_required
            and self._free_rim_gripper_preshaped
            and np.isfinite(gripper_width_m)
            and self._phase_ticks >= self.config.contact_min_ticks
            and self._contact_window_stably_stationary(
                max_cartesian_span_m=self.config.rim_place_contact_cartesian_span_m
            )
        )

    def _cavity_approach_should_fallback(self, current: Pose, target: Pose) -> bool:
        """Detect a blocked cavity hypothesis using proprioceptive progress only."""

        if (
            self._grasp_mode is not GraspMode.RIM_PINCH
            or not self._active_cavity_rim
            or self._phase_ticks < self.config.contact_min_ticks
            or not self._contact_window_stalled()
        ):
            return False
        residual = target.position - current.position
        rotation_error = float(
            np.linalg.norm(self._rotation_vector(target.rotation @ current.rotation.T))
        )
        return bool(
            float(np.linalg.norm(residual))
            > self.config.grasp_position_tolerance_m
            and rotation_error <= self.config.rotation_tolerance_rad
        )

    def _free_space_rim_approach_should_fallback(
        self,
        current: Pose,
        target: Pose,
        gripper_width_m: float,
    ) -> bool:
        """Detect an obstacle-blocked vertical approach to an exterior rim.

        This is deliberately not a contact-completion gate.  It only permits
        an open, vertical high retreat followed by a fresh antipodal proposal.
        The public EE must be stalled directly above the sensor-bound rim;
        lateral stalls, overshoot below the target, wrist error, or ongoing
        progress fail closed and keep the ordinary controller active.
        """

        if (
            self._grasp_mode is not GraspMode.RIM_PINCH
            or self._active_cavity_rim
            or self._phase_ticks < self.config.contact_min_ticks
            or not self._contact_window_stably_stationary(
                max_cartesian_span_m=(
                    self.config.rim_place_contact_cartesian_span_m
                )
            )
            or not np.isfinite(gripper_width_m)
            or gripper_width_m < self.config.gripper_open_width_m
        ):
            return False
        residual = target.position - current.position
        planar_error = float(np.linalg.norm(residual[:2]))
        vertical_clearance = -float(residual[2])
        rotation_error = float(
            np.linalg.norm(
                self._rotation_vector(target.rotation @ current.rotation.T)
            )
        )
        return bool(
            float(np.linalg.norm(residual))
            > self.config.grasp_position_tolerance_m
            and planar_error
            <= self.config.grasp_contact_position_tolerance_m
            and 0.0 <= vertical_clearance <= self.config.pregrasp_height_m
            and rotation_error <= self.config.rotation_tolerance_rad
        )

    def _free_space_rim_bilateral_contact_reached(
        self,
        current: Pose,
        target: Pose,
        gripper_width_m: float,
    ) -> bool:
        """Authorize only a high-retract/fresh-preshape recovery.

        Two open-hand stalls on antipodal sides establish reachability, not
        finger-pad contact.  The returned boolean is therefore consumed only
        by the OPEN vertical-retract branch.  It can never authorize CLOSE.
        """

        if (
            self._grasp_mode is not GraspMode.RIM_PINCH
            or self._active_cavity_rim
            or self._grasp_retry_index != 1
            or not self._grasp_target_attempts
        ):
            return False

        numeric_epsilon = 1e-12
        exact_geometry_tolerance_m = 1e-9

        def finite_vector(value: object, size: int) -> np.ndarray | None:
            try:
                vector = np.asarray(value, dtype=np.float64)
            except (TypeError, ValueError):
                return None
            if vector.shape != (size,) or not np.all(np.isfinite(vector)):
                return None
            return vector

        attempt1 = self._grasp_target_attempts[-1]
        attempt1_capture_sequence = attempt1.get("capture_sequence")
        typed_attempt1 = bool(
            type(attempt1.get("attempt")) is int
            and attempt1.get("attempt") == 1
            and attempt1.get("rim_strategy") == "free_space"
            and attempt1.get("active_cavity_rim") is False
        )
        attempt0: dict[str, object] | None = None
        if typed_attempt1 and len(self._grasp_target_attempts) >= 2:
            candidate = self._grasp_target_attempts[-2]
            if (
                type(candidate.get("attempt")) is int
                and candidate.get("attempt") == 0
                and candidate.get("rim_strategy") == "free_space"
                and candidate.get("active_cavity_rim") is False
            ):
                attempt0 = candidate

        target0 = (
            finite_vector(attempt0.get("target_world_m"), 3)
            if attempt0 is not None
            else None
        )
        attempt0_capture_sequence = (
            attempt0.get("capture_sequence")
            if attempt0 is not None
            else None
        )
        target1 = finite_vector(attempt1.get("target_world_m"), 3)
        anchor = finite_vector(
            attempt1.get("frozen_attempt0_anchor_center_world_m"), 3
        )
        frozen_target0 = finite_vector(
            attempt1.get("frozen_attempt0_rim_target_world_m"), 3
        )
        current_target_matches_trace = bool(
            target1 is not None
            and np.allclose(
                target.position,
                target1,
                rtol=0.0,
                atol=exact_geometry_tolerance_m,
            )
        )

        fallback = (
            attempt0.get("free_space_rim_approach_fallback")
            if attempt0 is not None
            else None
        )
        fallback_dict = fallback if isinstance(fallback, dict) else {}
        first_residual = finite_vector(fallback_dict.get("residual_world_m"), 3)
        first_seat = finite_vector(fallback_dict.get("retract_start_world_m"), 3)
        first_width = fallback_dict.get("gripper_width_m")
        first_rotation_error = fallback_dict.get("rotation_error_rad")
        first_capture_sequence = fallback_dict.get("capture_sequence")
        first_window_net_progress = fallback_dict.get(
            "contact_window_net_progress_m"
        )
        first_cartesian_span = fallback_dict.get(
            "contact_cartesian_span_m"
        )
        first_cartesian_span_tolerance = fallback_dict.get(
            "contact_cartesian_span_tolerance_m"
        )
        first_authorization = fallback_dict.get("authorization")
        first_fallback_open = bool(
            target0 is not None
            and first_residual is not None
            and first_seat is not None
            and np.allclose(
                first_seat,
                target0 - first_residual,
                rtol=0.0,
                atol=exact_geometry_tolerance_m,
            )
            and fallback_dict.get("strategy")
            == "proprio_stall_open_vertical_retract_then_fresh_antipodal_rim"
            and type(fallback_dict.get("next_retry_index")) is int
            and fallback_dict.get("next_retry_index") == 1
            and isinstance(first_authorization, dict)
            and first_authorization.get("close") is False
            and first_authorization.get("task_success") is False
            and first_authorization.get("fresh_antipodal_reacquisition") is True
            and isinstance(first_width, (int, float))
            and np.isfinite(first_width)
            and float(first_width) >= self.config.gripper_open_width_m
            and isinstance(first_rotation_error, (int, float))
            and np.isfinite(first_rotation_error)
            and float(first_rotation_error)
            <= self.config.rotation_tolerance_rad + numeric_epsilon
            and type(first_capture_sequence) is int
            and type(attempt0_capture_sequence) is int
            and 0 <= attempt0_capture_sequence < first_capture_sequence
            and isinstance(first_window_net_progress, (int, float))
            and np.isfinite(first_window_net_progress)
            and -numeric_epsilon <= float(first_window_net_progress)
            <= self.config.contact_progress_epsilon_m
            * self.config.contact_stall_ticks
            + numeric_epsilon
            and isinstance(first_cartesian_span, (int, float))
            and np.isfinite(first_cartesian_span)
            and float(first_cartesian_span)
            <= self.config.rim_place_contact_cartesian_span_m
            + numeric_epsilon
            and isinstance(first_cartesian_span_tolerance, (int, float))
            and np.isfinite(first_cartesian_span_tolerance)
            and abs(
                float(first_cartesian_span_tolerance)
                - self.config.rim_place_contact_cartesian_span_m
            )
            <= numeric_epsilon
        )

        reacquisition_records = (
            attempt0.get("free_space_rim_retry_reacquisition", [])
            if attempt0 is not None
            else []
        )
        fresh_record: dict[str, object] | None = None
        if isinstance(reacquisition_records, list):
            for candidate in reversed(reacquisition_records):
                if (
                    isinstance(candidate, dict)
                    and type(candidate.get("attempt")) is int
                    and candidate.get("attempt") == 1
                ):
                    fresh_record = candidate
                    break
        fresh_center = finite_vector(attempt1.get("sensor_center_world_m"), 3)
        fresh_extents = finite_vector(attempt1.get("sensor_extents_m"), 3)
        candidate_center = (
            finite_vector(fresh_record.get("candidate_center_world_m"), 3)
            if fresh_record is not None
            else None
        )
        recorded_anchor_center = (
            finite_vector(fresh_record.get("anchor_center_world_m"), 3)
            if fresh_record is not None
            else None
        )
        recorded_anchor_extents = (
            finite_vector(fresh_record.get("anchor_extents_sorted_m"), 3)
            if fresh_record is not None
            else None
        )
        candidate_extents = (
            finite_vector(fresh_record.get("candidate_extents_sorted_m"), 3)
            if fresh_record is not None
            else None
        )
        extent_scales = (
            finite_vector(fresh_record.get("candidate_extent_scales"), 3)
            if fresh_record is not None
            else None
        )
        capture_sequence = (
            fresh_record.get("capture_sequence")
            if fresh_record is not None
            else None
        )
        recorded_anchor_radius = (
            fresh_record.get("anchor_radius_m")
            if fresh_record is not None
            else None
        )
        recorded_nearest_distance = (
            fresh_record.get("nearest_bound_distance_m")
            if fresh_record is not None
            else None
        )
        attempt0_center = (
            finite_vector(attempt0.get("sensor_center_world_m"), 3)
            if attempt0 is not None
            else None
        )
        attempt0_extents = (
            finite_vector(attempt0.get("sensor_extents_m"), 3)
            if attempt0 is not None
            else None
        )
        fresh_top = attempt1.get("fresh_top_world_m")
        fresh_top_offset = attempt1.get("fresh_top_offset_from_attempt0_m")
        fresh_top_bound = attempt1.get("fresh_top_continuity_bound_m")
        fresh_center_offset = finite_vector(
            attempt1.get("fresh_center_offset_from_attempt0_anchor_m"), 3
        )
        expected_fresh_top = (
            float(target1[2]) + self.config.free_space_rim_grasp_top_inset_m
            if target1 is not None
            else None
        )
        expected_initial_top = (
            float(target0[2]) + self.config.free_space_rim_grasp_top_inset_m
            if target0 is not None
            else None
        )
        fresh_identity_size = bool(
            fresh_record is not None
            and fresh_record.get("nearest_bound_result") == "accepted"
            and fresh_record.get("selected_source")
            == "fresh_nearest_bound_attempt0_anchor"
            and fresh_record.get("size_continuous") is True
            and type(capture_sequence) is int
            and type(first_capture_sequence) is int
            and first_capture_sequence < capture_sequence
            < self._observation_timestamp_s
            and type(attempt1_capture_sequence) is int
            and attempt1_capture_sequence == capture_sequence
            and fresh_center is not None
            and candidate_center is not None
            and np.allclose(
                fresh_center,
                candidate_center,
                rtol=0.0,
                atol=exact_geometry_tolerance_m,
            )
            and anchor is not None
            and attempt0_center is not None
            and recorded_anchor_center is not None
            and np.allclose(
                anchor,
                attempt0_center,
                rtol=0.0,
                atol=exact_geometry_tolerance_m,
            )
            and np.allclose(
                recorded_anchor_center,
                anchor,
                rtol=0.0,
                atol=exact_geometry_tolerance_m,
            )
            and isinstance(recorded_anchor_radius, (int, float))
            and np.isfinite(recorded_anchor_radius)
            and float(recorded_anchor_radius) > 0.0
            and self._pick_selector_reacquire_radius_m is not None
            and abs(
                float(recorded_anchor_radius)
                - float(self._pick_selector_reacquire_radius_m)
            )
            <= exact_geometry_tolerance_m
            and isinstance(recorded_nearest_distance, (int, float))
            and np.isfinite(recorded_nearest_distance)
            and abs(
                float(recorded_nearest_distance)
                - float(np.linalg.norm(candidate_center - anchor))
            )
            <= exact_geometry_tolerance_m
            and float(recorded_nearest_distance)
            <= float(recorded_anchor_radius) + numeric_epsilon
            and fresh_extents is not None
            and candidate_extents is not None
            and np.allclose(
                np.sort(fresh_extents),
                candidate_extents,
                rtol=0.0,
                atol=exact_geometry_tolerance_m,
            )
            and attempt0_extents is not None
            and recorded_anchor_extents is not None
            and np.all(recorded_anchor_extents > 0.0)
            and np.allclose(
                recorded_anchor_extents,
                np.sort(attempt0_extents),
                rtol=0.0,
                atol=exact_geometry_tolerance_m,
            )
            and extent_scales is not None
            and np.allclose(
                extent_scales,
                candidate_extents / recorded_anchor_extents,
                rtol=0.0,
                atol=exact_geometry_tolerance_m,
            )
            and np.all(
                extent_scales
                >= self.config.free_space_rim_retry_min_extent_scale
                - numeric_epsilon
            )
            and np.all(
                extent_scales
                <= self.config.free_space_rim_retry_max_extent_scale
                + numeric_epsilon
            )
            and attempt1.get("target_center_source")
            == "attempt0_frozen_anchor_xy_fresh_bound_top_z"
            and fresh_center_offset is not None
            and np.allclose(
                fresh_center_offset,
                fresh_center - anchor,
                rtol=0.0,
                atol=exact_geometry_tolerance_m,
            )
            and isinstance(fresh_top, (int, float))
            and np.isfinite(fresh_top)
            and expected_fresh_top is not None
            and abs(float(fresh_top) - expected_fresh_top)
            <= exact_geometry_tolerance_m
            and isinstance(fresh_top_offset, (int, float))
            and np.isfinite(fresh_top_offset)
            and expected_initial_top is not None
            and abs(
                float(fresh_top_offset)
                - (float(fresh_top) - expected_initial_top)
            )
            <= exact_geometry_tolerance_m
            and isinstance(fresh_top_bound, (int, float))
            and np.isfinite(fresh_top_bound)
            and abs(
                float(fresh_top_bound) - float(recorded_anchor_radius)
            )
            <= exact_geometry_tolerance_m
            and abs(float(fresh_top_offset))
            <= float(fresh_top_bound) + numeric_epsilon
        )

        radial0: np.ndarray | None = None
        radial1: np.ndarray | None = None
        radius0: float | None = None
        radius1: float | None = None
        antipodal_dot: float | None = None
        mirror_error: float | None = None
        radius_delta: float | None = None
        exact_antipode = False
        if (
            anchor is not None
            and target0 is not None
            and target1 is not None
            and frozen_target0 is not None
            and np.allclose(
                frozen_target0,
                target0,
                rtol=0.0,
                atol=exact_geometry_tolerance_m,
            )
        ):
            radial0 = target0[:2] - anchor[:2]
            radial1 = target1[:2] - anchor[:2]
            radius0 = float(np.linalg.norm(radial0))
            radius1 = float(np.linalg.norm(radial1))
            if radius0 > 0.0 and radius1 > 0.0:
                antipodal_dot = float(np.dot(radial0, radial1) / (radius0 * radius1))
                mirror_error = float(
                    np.linalg.norm(
                        target1[:2] - (2.0 * anchor[:2] - target0[:2])
                    )
                )
                radius_delta = abs(radius1 - radius0)
                attempt0_radius = attempt0.get("rim_radius_m") if attempt0 else None
                attempt1_radius = attempt1.get("rim_radius_m")
                exact_antipode = bool(
                    mirror_error <= exact_geometry_tolerance_m
                    and radius_delta <= exact_geometry_tolerance_m
                    and antipodal_dot <= -1.0 + exact_geometry_tolerance_m
                    and isinstance(attempt0_radius, (int, float))
                    and isinstance(attempt1_radius, (int, float))
                    and abs(float(attempt0_radius) - radius0)
                    <= exact_geometry_tolerance_m
                    and abs(float(attempt1_radius) - radius1)
                    <= exact_geometry_tolerance_m
                )

        residual = target.position - current.position
        rotation_error = float(
            np.linalg.norm(
                self._rotation_vector(target.rotation @ current.rotation.T)
            )
        )
        vertical_clearance = -float(residual[2])
        radial_residual: float | None = None
        tangential_residual: float | None = None
        first_radial_residual: float | None = None
        first_tangential_residual: float | None = None
        if radial1 is not None and radius1 is not None and radius1 > 0.0:
            unit = radial1 / radius1
            radial_residual = float(np.dot(residual[:2], unit))
            tangential_residual = float(
                -residual[0] * unit[1] + residual[1] * unit[0]
            )
        if (
            radial0 is not None
            and radius0 is not None
            and radius0 > 0.0
            and first_residual is not None
        ):
            first_unit = radial0 / radius0
            first_radial_residual = float(np.dot(first_residual[:2], first_unit))
            first_tangential_residual = float(
                -first_residual[0] * first_unit[1]
                + first_residual[1] * first_unit[0]
            )
        first_vertical_clearance = (
            -float(first_residual[2]) if first_residual is not None else None
        )
        seat_plane_delta = (
            abs(float(current.position[2]) - float(first_seat[2]))
            if first_seat is not None
            else None
        )
        current_window_net_progress = self._contact_window_net_progress_m()
        current_cartesian_span = self._contact_cartesian_span_m()
        stalled = bool(
            self._phase_ticks >= self.config.contact_min_ticks
            and self._contact_window_stably_stationary(
                max_cartesian_span_m=(
                    self.config.rim_place_contact_cartesian_span_m
                )
            )
        )
        first_directional_seat_valid = bool(
            first_radial_residual is not None
            and abs(first_radial_residual)
            <= self.config.free_space_rim_bilateral_radial_tolerance_m
            + numeric_epsilon
            and first_tangential_residual is not None
            and abs(first_tangential_residual)
            <= self.config.free_space_rim_bilateral_tangential_tolerance_m
            + numeric_epsilon
            and first_vertical_clearance is not None
            and self.config.free_space_rim_bilateral_min_vertical_clearance_m
            - numeric_epsilon
            <= first_vertical_clearance
            <= self.config.free_space_rim_bilateral_max_vertical_clearance_m
            + numeric_epsilon
        )
        current_directional_seat_valid = bool(
            radial_residual is not None
            and abs(radial_residual)
            <= self.config.free_space_rim_bilateral_radial_tolerance_m
            + numeric_epsilon
            and tangential_residual is not None
            and abs(tangential_residual)
            <= self.config.free_space_rim_bilateral_tangential_tolerance_m
            + numeric_epsilon
            and self.config.free_space_rim_bilateral_min_vertical_clearance_m
            - numeric_epsilon
            <= vertical_clearance
            <= self.config.free_space_rim_bilateral_max_vertical_clearance_m
            + numeric_epsilon
        )
        width_valid = bool(
            np.isfinite(gripper_width_m)
            and gripper_width_m >= self.config.gripper_open_width_m
        )
        rotation_valid = bool(
            rotation_error <= self.config.rotation_tolerance_rad + numeric_epsilon
        )
        same_seat_plane = bool(
            seat_plane_delta is not None
            and seat_plane_delta
            <= self.config.free_space_rim_bilateral_seat_plane_tolerance_m
            + numeric_epsilon
        )
        authorized = bool(
            typed_attempt1
            and attempt0 is not None
            and current_target_matches_trace
            and first_fallback_open
            and fresh_identity_size
            and exact_antipode
            and stalled
            and first_directional_seat_valid
            and current_directional_seat_valid
            and width_valid
            and rotation_valid
            and same_seat_plane
        )

        trace = attempt1.setdefault(
            "free_space_rim_bilateral_contact",
            {
                "strategy": "two_open_antipodal_stalls_on_one_world_z_rim_plane",
                "radial_tolerance_m": float(
                    self.config.free_space_rim_bilateral_radial_tolerance_m
                ),
                "tangential_tolerance_m": float(
                    self.config.free_space_rim_bilateral_tangential_tolerance_m
                ),
                "vertical_clearance_range_m": [
                    float(
                        self.config.free_space_rim_bilateral_min_vertical_clearance_m
                    ),
                    float(
                        self.config.free_space_rim_bilateral_max_vertical_clearance_m
                    ),
                ],
                "seat_plane_tolerance_m": float(
                    self.config.free_space_rim_bilateral_seat_plane_tolerance_m
                ),
                "minimum_open_width_m": float(self.config.gripper_open_width_m),
                "rotation_tolerance_rad": float(
                    self.config.rotation_tolerance_rad
                ),
                "cartesian_span_tolerance_m": float(
                    self.config.rim_place_contact_cartesian_span_m
                ),
                "exact_geometry_tolerance_m": exact_geometry_tolerance_m,
                "samples": [],
            },
        )
        measurement = {
            "typed_second_free_space_rim_attempt": typed_attempt1,
            "attempt0_open_fallback_proved": first_fallback_open,
            "fresh_identity_and_size_proved": fresh_identity_size,
            "current_target_matches_trace": current_target_matches_trace,
            "exact_frozen_anchor_antipode_proved": exact_antipode,
            "antipodal_dot": antipodal_dot,
            "mirror_error_m": mirror_error,
            "rim_radius_delta_m": radius_delta,
            "fresh_reacquisition_capture_sequence": capture_sequence,
            "public_proprio_stalled": stalled,
            "contact_window_net_progress_m": current_window_net_progress,
            "contact_cartesian_span_m": current_cartesian_span,
            "residual_world_m": residual.tolist(),
            "radial_residual_m": radial_residual,
            "tangential_residual_m": tangential_residual,
            "vertical_clearance_above_target_m": vertical_clearance,
            "rotation_error_rad": rotation_error,
            "gripper_width_m": float(gripper_width_m),
            "attempt0_residual_world_m": (
                first_residual.tolist() if first_residual is not None else None
            ),
            "attempt0_radial_residual_m": first_radial_residual,
            "attempt0_tangential_residual_m": first_tangential_residual,
            "attempt0_vertical_clearance_above_target_m": (
                first_vertical_clearance
            ),
            "attempt0_seat_world_z_m": (
                float(first_seat[2]) if first_seat is not None else None
            ),
            "attempt1_seat_world_z_m": float(current.position[2]),
            "seat_plane_delta_m": seat_plane_delta,
            "first_directional_seat_valid": first_directional_seat_valid,
            "current_directional_seat_valid": current_directional_seat_valid,
            "open_width_gate_passed": width_valid,
            "rotation_gate_passed": rotation_valid,
            "same_seat_plane_gate_passed": same_seat_plane,
            "preshape_reseat_authorized": authorized,
            "close_authorized": False,
        }
        samples = trace.get("samples")
        if isinstance(samples, list):
            samples.append(measurement)
        trace["last_measurement"] = dict(measurement)
        trace["authorization"] = {
            "close": False,
            "task_success": False,
            "open_high_retract_then_fresh_preshape_reseat": authorized,
        }
        if authorized:
            trace["completion"] = dict(measurement)
        return authorized

    def _rim_place_contact_reached(
        self,
        current: Pose,
        target: Pose,
        gripper_width_m: float,
    ) -> bool:
        """Accept a downward support contact while a thin rim is still held.

        All inputs are robot proprioception.  A stable non-zero finger width
        proves the bowl has not already fallen, while signed Z and planar
        residuals distinguish support contact from a lateral obstacle.
        """

        if (
            self._grasp_mode is not GraspMode.RIM_PINCH
            or self._held_subject is None
            or self._phase_ticks < self.config.contact_min_ticks
            or not self._contact_window_stalled()
            or self._contact_cartesian_span_m()
            > self.config.rim_place_contact_cartesian_span_m
            or gripper_width_m < self.config.rim_pinch_blocked_min_width_m
            or gripper_width_m >= self.config.gripper_open_width_m
        ):
            return False
        residual = target.position - current.position
        planar_error = float(np.linalg.norm(residual[:2]))
        vertical_clearance = -float(residual[2])
        rotation_error = float(
            np.linalg.norm(self._rotation_vector(target.rotation @ current.rotation.T))
        )
        return bool(
            planar_error <= self.config.rim_place_contact_xy_tolerance_m
            and 0.0
            <= vertical_clearance
            <= self.config.rim_place_contact_vertical_tolerance_m
            and rotation_error <= self.config.rotation_tolerance_rad
        )

    def _record_cavity_preshape_start(self, gripper_width_m: float) -> None:
        if not self._grasp_target_attempts:
            return
        self._grasp_target_attempts[-1]["cavity_gripper_preshape"] = {
            "start_width_m": float(gripper_width_m),
            "target_width_m": float(
                self.config.cavity_rim_preshape_target_width_m
            ),
            "entry_band_m": [
                float(self.config.cavity_rim_preshape_min_width_m),
                float(self.config.cavity_rim_preshape_max_width_m),
            ],
            "feedback": "proprioceptive_binary_width_control",
        }

    def _record_cavity_preshape_completion(self, gripper_width_m: float) -> None:
        if not self._grasp_target_attempts:
            return
        trace = self._grasp_target_attempts[-1].get("cavity_gripper_preshape")
        if not isinstance(trace, dict):
            self._record_cavity_preshape_start(gripper_width_m)
            trace = self._grasp_target_attempts[-1]["cavity_gripper_preshape"]
        trace["completion_width_m"] = float(gripper_width_m)
        trace["completion_ticks"] = int(self._phase_ticks)
        trace["still_open"] = bool(
            gripper_width_m > self.config.rim_pinch_blocked_min_width_m
        )

    def _record_cavity_preshape_sample(
        self,
        stage: str,
        gripper_width_m: float,
        command: float,
    ) -> None:
        if not self._grasp_target_attempts:
            return
        trace = self._grasp_target_attempts[-1].get("cavity_gripper_preshape")
        if not isinstance(trace, dict):
            self._record_cavity_preshape_start(gripper_width_m)
            trace = self._grasp_target_attempts[-1]["cavity_gripper_preshape"]
        samples = trace.setdefault("width_feedback_samples", [])
        if isinstance(samples, list):
            samples.append(
                {
                    "stage": stage,
                    "phase_tick": int(self._phase_ticks),
                    "width_m": float(gripper_width_m),
                    "command": float(command),
                }
            )

    def _record_free_space_rim_preshape_start(
        self, gripper_width_m: float
    ) -> None:
        if not self._grasp_target_attempts:
            return
        self._grasp_target_attempts[-1]["free_space_rim_preshape"] = {
            "strategy": "high_waypoint_close_fingers_binary_width_feedback",
            "start_width_m": float(gripper_width_m),
            "target_width_m": float(
                self.config.free_space_rim_preshape_target_width_m
            ),
            "entry_band_m": [
                float(self.config.free_space_rim_preshape_min_width_m),
                float(self.config.free_space_rim_preshape_max_width_m),
            ],
            "allowed_jaw_behavior": JawBehavior.CLOSE_FINGERS.value,
            "samples": [],
        }

    def _record_free_space_rim_preshape_completion(
        self, gripper_width_m: float
    ) -> None:
        if not self._grasp_target_attempts:
            return
        trace = self._grasp_target_attempts[-1].get(
            "free_space_rim_preshape"
        )
        if not isinstance(trace, dict):
            self._record_free_space_rim_preshape_start(gripper_width_m)
            trace = self._grasp_target_attempts[-1][
                "free_space_rim_preshape"
            ]
        trace["completion"] = {
            "gripper_width_m": float(gripper_width_m),
            "phase_tick": int(self._phase_ticks),
            "still_open": bool(
                gripper_width_m > self.config.rim_pinch_blocked_min_width_m
            ),
        }

    def _record_free_space_rim_preshape_sample(
        self,
        stage: str,
        gripper_width_m: float,
        command: float,
    ) -> None:
        if not self._grasp_target_attempts:
            return
        trace = self._grasp_target_attempts[-1].get(
            "free_space_rim_preshape"
        )
        if not isinstance(trace, dict):
            self._record_free_space_rim_preshape_start(gripper_width_m)
            trace = self._grasp_target_attempts[-1][
                "free_space_rim_preshape"
            ]
        samples = trace.setdefault("samples", [])
        if isinstance(samples, list):
            samples.append(
                {
                    "stage": stage,
                    "phase_tick": int(self._phase_ticks),
                    "width_m": float(gripper_width_m),
                    "command": float(command),
                    "jaw_behavior": JawBehavior.CLOSE_FINGERS.value,
                }
            )

    def _start_cavity_seat_pull(self, observation: SensorObservation) -> bool:
        """Create a short closed-gripper pull along the sensed outside rim."""

        if not self._grasp_target_attempts or self._motion_pose is None:
            return False
        attempt = self._grasp_target_attempts[-1]
        try:
            radial = np.asarray(attempt["finger_axis_world"], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            return False
        if radial.shape != (3,) or not np.all(np.isfinite(radial)):
            return False
        radial[2] = 0.0
        radial_norm = float(np.linalg.norm(radial))
        if radial_norm < 0.8:
            return False
        radial /= radial_norm
        start = observation.robot.ee_pose.position.copy()
        target = start + self.config.cavity_rim_seat_pull_m * radial
        target[2] = start[2]
        self._cavity_seat_pose = Pose(target, self._motion_pose.rotation)
        wall_clearance = attempt.get("cavity_wall_clearance_m")
        remaining_clearance = (
            float(wall_clearance) - self.config.cavity_rim_seat_pull_m
            if isinstance(wall_clearance, (float, int))
            else None
        )
        attempt["cavity_seat_pull"] = {
            "strategy": "closed_gripper_sensor_rim_radial_support_plane_pull",
            "radial_direction_world": radial.tolist(),
            "commanded_distance_m": float(self.config.cavity_rim_seat_pull_m),
            "minimum_osc_ticks": int(self.config.cavity_rim_seat_pull_ticks),
            "maximum_osc_ticks": int(
                self.config.cavity_rim_seat_pull_max_ticks
            ),
            "minimum_radial_progress_m": float(
                self.config.cavity_rim_seat_pull_min_progress_m
            ),
            "maximum_orthogonal_error_m": float(
                self.config.cavity_rim_seat_pull_max_orthogonal_error_m
            ),
            "maximum_vertical_error_m": float(
                self.config.cavity_rim_seat_pull_max_vertical_error_m
            ),
            "contact_seat_gate": {
                "profile": "far",
                "enabled_for_attempt": bool(
                    attempt.get("cavity_selected_side_profile") == "far"
                ),
                "minimum_osc_ticks": int(
                    self.config.cavity_rim_seat_pull_ticks
                ),
                "minimum_radial_progress_m": float(
                    self.config.cavity_rim_seat_contact_min_progress_m
                ),
                "recent_radial_window_ticks": int(
                    self.config.cavity_rim_seat_contact_window_ticks
                ),
                "maximum_recent_radial_span_m": float(
                    self.config.cavity_rim_seat_contact_progress_span_m
                ),
                "minimum_blocked_width_gain_m": float(
                    self.config.cavity_rim_seat_contact_min_width_gain_m
                ),
                "maximum_orthogonal_error_m": float(
                    self.config.cavity_rim_seat_pull_max_orthogonal_error_m
                ),
                "maximum_vertical_error_m": float(
                    self.config.cavity_rim_seat_contact_max_vertical_error_m
                ),
                "maximum_rotation_error_rad": float(
                    self.config.cavity_rim_seat_contact_max_rotation_error_rad
                ),
            },
            "start_position_world_m": start.tolist(),
            "target_position_world_m": target.tolist(),
            "start_width_m": float(observation.robot.gripper_width_m),
            "blocked_width_gate_m": [
                float(self.config.rim_pinch_blocked_min_width_m),
                float(self.config.gripper_open_width_m),
            ],
            "measured_fixture_clearance_before_pull_m": (
                float(wall_clearance)
                if isinstance(wall_clearance, (float, int))
                else None
            ),
            "measured_fixture_clearance_after_pull_m": remaining_clearance,
            "samples": [],
        }
        return True

    def _start_roomy_near_direct_proof(
        self,
        observation: SensorObservation,
    ) -> ControlDecision:
        """Lift a roomy nominal near-rim grasp without a support-plane pull.

        The profile is selected only from measured cavity clearance.  Its
        unyawed, level near-side close already creates a stable two-pad hold;
        an outward seat motion would move it toward the fixture wall.  A hard
        blocked-width gate remains mandatory before the unchanged 50-mm proof
        and operational-clearance gates.
        """

        if not self._grasp_target_attempts or self._secondary_pose is None:
            return self._fail(
                "roomy cavity direct proof lacks a measured high waypoint"
            )
        attempt = self._grasp_target_attempts[-1]
        width = float(observation.robot.gripper_width_m)
        width_valid = bool(
            self.config.rim_pinch_blocked_min_width_m
            <= width
            < self.config.gripper_open_width_m
        )
        direct_trace = {
            "strategy": "roomy_near_nominal_direct_vertical_proof",
            "profile": "roomy_near_nominal",
            "entry_width_m": width,
            "blocked_width_gate_m": [
                float(self.config.rim_pinch_blocked_min_width_m),
                float(self.config.gripper_open_width_m),
            ],
            "accepted": width_valid,
            "seat_actions": 0,
        }
        attempt["cavity_roomy_near_direct_proof"] = direct_trace
        if not width_valid:
            self._finish_grasp_engagement(
                accepted=False,
                reason=GraspReason.RETENTION_REJECTED,
                evidence_source=GraspEvidenceCategory.PROPRIOCEPTION,
            )
            direct_trace["completion_reason"] = (
                "proprioceptive_blocked_width_gate_failed"
            )
            if (
                self._cavity_candidate_index
                >= self._maximum_cavity_candidate_index()
            ):
                self._cavity_retry_terminal_failure = (
                    "roomy cavity rim close failed its blocked-width gate"
                )
                direct_trace["next_candidate_index"] = None
            else:
                self._cavity_candidate_index += 1
                self._grasp_retry_index += 1
                direct_trace["next_candidate_index"] = int(
                    self._cavity_candidate_index
                )
            # Retract vertically from measured XY before any fresh RGB-D
            # retry.  This is the same fixture-safe recovery used by a failed
            # seat/proof attempt and never sweeps an opening hand sideways.
            retract_position = self._secondary_pose.position.copy()
            retract_position[:2] = observation.robot.ee_pose.position[:2]
            self._secondary_pose = Pose(
                retract_position,
                self._secondary_pose.rotation,
            )
            self._cavity_gripper_preshaped = False
            self._cavity_lift_start_z_m = None
            self._cavity_seat_pose = None
            self._set_phase("cavity_retry_retract")
            return self._motion(
                self._secondary_pose,
                observation,
                self._released_gripper_command(),
            )

        direct_trace["completion_reason"] = (
            "proprioceptive_blocked_width_gate_passed"
        )
        self._cavity_lift_start_z_m = float(
            observation.robot.ee_pose.position[2]
        )
        self._start_cavity_lift_proof(width)
        high_position = self._secondary_pose.position.copy()
        high_position[:2] = observation.robot.ee_pose.position[:2]
        self._secondary_pose = Pose(
            high_position,
            self._secondary_pose.rotation,
        )
        self._cavity_seat_pose = None
        self._set_phase("retreat")
        return self._motion(
            self._secondary_pose,
            observation,
            self._engaged_gripper_command(),
        )

    def _cavity_seat_pull_reached(self, current: Pose) -> bool:
        if not self._grasp_target_attempts or self._cavity_seat_pose is None:
            return False
        trace = self._grasp_target_attempts[-1].get("cavity_seat_pull")
        if not isinstance(trace, dict):
            return False
        try:
            start = np.asarray(trace["start_position_world_m"], dtype=np.float64)
            radial = np.asarray(trace["radial_direction_world"], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            return False
        displacement = current.position - start
        radial_progress = float(np.dot(displacement, radial))
        planar_orthogonal = displacement[:2] - radial_progress * radial[:2]
        orthogonal_error = float(np.linalg.norm(planar_orthogonal))
        vertical_error = abs(float(displacement[2]))
        rotation_error = float(
            np.linalg.norm(
                self._rotation_vector(
                    self._cavity_seat_pose.rotation @ current.rotation.T
                )
            )
        )
        return bool(
            radial_progress >= self.config.cavity_rim_seat_pull_min_progress_m
            and orthogonal_error
            <= self.config.cavity_rim_seat_pull_max_orthogonal_error_m
            and vertical_error <= self.config.cavity_rim_seat_pull_max_vertical_error_m
            and rotation_error <= self.config.rotation_tolerance_rad
        )

    def _cavity_seat_contact_reached(
        self,
        current: Pose,
        gripper_width_m: float,
    ) -> bool:
        """Accept a blocked radial pull only when proprioception proves seating.

        This is deliberately independent of simulator contacts.  A small
        outward move must plateau against the fixture while the two pads gain
        blocked width, and every sampled width must remain in the valid held
        interval.  The subsequent lift proof is still authoritative.
        """

        if (
            not self._grasp_target_attempts
            or self._grasp_target_attempts[-1].get(
                "cavity_selected_side_profile"
            )
            != "far"
            or self._phase_ticks < self.config.cavity_rim_seat_pull_ticks
            or self._cavity_seat_pose is None
        ):
            return False
        trace = self._grasp_target_attempts[-1].get("cavity_seat_pull")
        if not isinstance(trace, dict):
            return False
        try:
            start = np.asarray(trace["start_position_world_m"], dtype=np.float64)
            radial = np.asarray(trace["radial_direction_world"], dtype=np.float64)
            start_width = float(trace["start_width_m"])
        except (KeyError, TypeError, ValueError):
            return False
        displacement = current.position - start
        radial_progress = float(np.dot(displacement, radial))
        planar_orthogonal = displacement[:2] - radial_progress * radial[:2]
        orthogonal_error = float(np.linalg.norm(planar_orthogonal))
        vertical_error = abs(float(displacement[2]))
        rotation_error = float(
            np.linalg.norm(
                self._rotation_vector(
                    self._cavity_seat_pose.rotation @ current.rotation.T
                )
            )
        )
        samples = trace.get("samples")
        window = self.config.cavity_rim_seat_contact_window_ticks
        if not isinstance(samples, list) or len(samples) < window:
            return False
        recent = samples[-window:]
        try:
            recent_progress = [
                float(sample["radial_displacement_m"]) for sample in recent
            ]
            all_widths = [float(sample["width_m"]) for sample in samples]
        except (KeyError, TypeError, ValueError):
            return False
        recent_span = max(recent_progress) - min(recent_progress)
        all_widths_valid = all(
            self.config.rim_pinch_blocked_min_width_m
            <= width
            < self.config.gripper_open_width_m
            for width in all_widths
        )
        return bool(
            radial_progress
            >= self.config.cavity_rim_seat_contact_min_progress_m
            and recent_span
            <= self.config.cavity_rim_seat_contact_progress_span_m
            and gripper_width_m - start_width
            >= self.config.cavity_rim_seat_contact_min_width_gain_m
            and all_widths_valid
            and orthogonal_error
            <= self.config.cavity_rim_seat_pull_max_orthogonal_error_m
            and vertical_error
            <= self.config.cavity_rim_seat_contact_max_vertical_error_m
            and rotation_error
            <= self.config.cavity_rim_seat_contact_max_rotation_error_rad
        )

    def _record_cavity_seat_pull_sample(
        self,
        current: Pose,
        gripper_width_m: float,
    ) -> None:
        if not self._grasp_target_attempts:
            return
        trace = self._grasp_target_attempts[-1].get("cavity_seat_pull")
        if not isinstance(trace, dict):
            return
        start_value = trace.get("start_position_world_m")
        radial_value = trace.get("radial_direction_world")
        try:
            start = np.asarray(start_value, dtype=np.float64)
            radial = np.asarray(radial_value, dtype=np.float64)
        except (TypeError, ValueError):
            return
        displacement = current.position - start
        radial_progress = float(np.dot(displacement, radial))
        planar_orthogonal = displacement[:2] - radial_progress * radial[:2]
        rotation_error = (
            float(
                np.linalg.norm(
                    self._rotation_vector(
                        self._cavity_seat_pose.rotation @ current.rotation.T
                    )
                )
            )
            if self._cavity_seat_pose is not None
            else None
        )
        samples = trace.get("samples")
        if isinstance(samples, list):
            samples.append(
                {
                    "phase_tick": int(self._phase_ticks),
                    "position_world_m": current.position.tolist(),
                    "displacement_world_m": displacement.tolist(),
                    "radial_displacement_m": radial_progress,
                    "orthogonal_displacement_m": float(
                        np.linalg.norm(planar_orthogonal)
                    ),
                    "vertical_drift_m": float(displacement[2]),
                    "rotation_error_rad": rotation_error,
                    "width_m": float(gripper_width_m),
                }
            )

    def _complete_cavity_seat_pull(
        self,
        current: Pose,
        gripper_width_m: float,
        *,
        outcome: str,
        reason: str,
    ) -> None:
        if not self._grasp_target_attempts:
            return
        trace = self._grasp_target_attempts[-1].get("cavity_seat_pull")
        if not isinstance(trace, dict):
            return
        start = np.asarray(trace["start_position_world_m"], dtype=np.float64)
        radial = np.asarray(trace["radial_direction_world"], dtype=np.float64)
        displacement = current.position - start
        radial_progress = float(np.dot(displacement, radial))
        planar_orthogonal = displacement[:2] - radial_progress * radial[:2]
        trace["completion_ticks"] = int(self._phase_ticks)
        trace["outcome"] = outcome
        trace["completion_reason"] = reason
        trace["accepted_kind"] = reason if outcome == "accepted" else None
        trace["completion_position_world_m"] = current.position.tolist()
        trace["completion_width_m"] = float(gripper_width_m)
        trace["measured_displacement_world_m"] = displacement.tolist()
        trace["measured_radial_displacement_m"] = radial_progress
        trace["measured_orthogonal_displacement_m"] = float(
            np.linalg.norm(planar_orthogonal)
        )
        trace["measured_vertical_drift_m"] = float(displacement[2])
        trace["pose_gate_reached"] = bool(
            self._cavity_seat_pull_reached(current)
        )
        samples = trace.get("samples")
        window = self.config.cavity_rim_seat_contact_window_ticks
        if isinstance(samples, list) and len(samples) >= window:
            recent = samples[-window:]
            try:
                progress = [
                    float(sample["radial_displacement_m"])
                    for sample in recent
                ]
                trace["contact_seat_recent_radial_span_m"] = (
                    max(progress) - min(progress)
                )
            except (KeyError, TypeError, ValueError):
                pass
        trace["contact_seat_blocked_width_gain_m"] = float(
            gripper_width_m - float(trace["start_width_m"])
        )

    def _start_cavity_lift_proof(self, gripper_width_m: float) -> None:
        if not self._grasp_target_attempts:
            return
        self._grasp_target_attempts[-1]["cavity_lift_proof"] = {
            "required_lift_m": float(self.config.cavity_rim_proof_lift_m),
            "blocked_min_width_m": float(
                self.config.rim_pinch_blocked_min_width_m
            ),
            "start_width_m": float(gripper_width_m),
            "samples": [],
            "passed_required_lift": False,
        }

    def _record_cavity_lift_proof_sample(
        self,
        current: Pose,
        gripper_width_m: float,
    ) -> bool:
        """Return true as soon as proprioception proves the rim hold was lost."""

        if self._cavity_lift_start_z_m is None or not self._grasp_target_attempts:
            return False
        lift_m = max(
            0.0,
            float(current.position[2]) - self._cavity_lift_start_z_m,
        )
        trace = self._grasp_target_attempts[-1].get("cavity_lift_proof")
        if not isinstance(trace, dict):
            self._start_cavity_lift_proof(gripper_width_m)
            trace = self._grasp_target_attempts[-1]["cavity_lift_proof"]
        samples = trace.get("samples")
        if isinstance(samples, list):
            samples.append(
                {
                    "lift_m": lift_m,
                    "width_m": float(gripper_width_m),
                }
            )
        if lift_m >= self.config.cavity_rim_proof_lift_m:
            trace["passed_required_lift"] = True
        invalid_width = bool(
            gripper_width_m < self.config.rim_pinch_blocked_min_width_m
            or gripper_width_m >= self.config.gripper_open_width_m
        )
        if invalid_width:
            trace["lost_width_m"] = float(gripper_width_m)
            trace["lost_at_lift_m"] = lift_m
            trace["next_candidate_index"] = (
                self._cavity_candidate_index + 1
                if self._cavity_candidate_index
                < self._maximum_cavity_candidate_index()
                else None
            )
            trace["reason"] = "proprioceptive_blocked_width_gate_failed"
        return invalid_width

    def _record_cavity_contact_completion(
        self,
        current: Pose,
        target: Pose,
        gripper_width_m: float,
        *,
        reason: str,
    ) -> None:
        if not self._grasp_target_attempts:
            return
        residual = target.position - current.position
        self._grasp_target_attempts[-1]["cavity_contact_completion"] = {
            "goal_minus_current_world_m": residual.tolist(),
            "planar_error_m": float(np.linalg.norm(residual[:2])),
            "vertical_clearance_m": -float(residual[2]),
            "preshape_width_m": float(gripper_width_m),
            "reason": reason,
        }

    def _record_cavity_approach_fallback(
        self,
        current: Pose,
        target: Pose,
        gripper_width_m: float,
    ) -> None:
        if not self._grasp_target_attempts:
            return
        residual = target.position - current.position
        rotation_error = float(
            np.linalg.norm(self._rotation_vector(target.rotation @ current.rotation.T))
        )
        self._grasp_target_attempts[-1]["cavity_approach_fallback"] = {
            "goal_minus_current_world_m": residual.tolist(),
            "position_error_m": float(np.linalg.norm(residual)),
            "signed_target_shortfall_m": float(residual[2]),
            "rotation_error_rad": rotation_error,
            "preshape_width_m": float(gripper_width_m),
            "next_candidate_index": self._next_cavity_candidate_after_approach_block(),
            "reason": "proprioceptive_approach_stall_failed_typed_close_gates",
        }

    def _record_rim_place_contact_completion(
        self,
        current: Pose,
        target: Pose,
        gripper_width_m: float,
    ) -> None:
        if not self._placement_target_attempts:
            return
        residual = target.position - current.position
        rotation_error = float(
            np.linalg.norm(self._rotation_vector(target.rotation @ current.rotation.T))
        )
        self._placement_target_attempts[-1]["rim_place_contact_completion"] = {
            "goal_minus_current_world_m": residual.tolist(),
            "planar_error_m": float(np.linalg.norm(residual[:2])),
            "vertical_clearance_m": -float(residual[2]),
            "rotation_error_rad": rotation_error,
            "gripper_width_m": float(gripper_width_m),
            "reason": "directional_proprioceptive_support_stall",
        }

    def _tick_decision(self, action: np.ndarray, message: str = "") -> ControlDecision:
        decision = self._decision(action, message)
        self._phase_ticks += 1
        return decision

    def _decision(self, action: np.ndarray, message: str = "") -> ControlDecision:
        return ControlDecision(
            action=action,
            status=self._status,
            skill_index=self._skill_index,
            phase=self._phase,
            message=message,
        )

    def _fail(self, message: str) -> ControlDecision:
        # Only an already issued jaw engagement can be closed out here.  An
        # acquisition/pregrasp/approach failure has no pending engagement and
        # therefore creates no grasp event.
        self._finish_grasp_engagement(
            accepted=False,
            reason=GraspReason.EXECUTION_STALLED,
            evidence_source=GraspEvidenceCategory.CONTROLLER_EXECUTION,
        )
        self._status = ExecutorStatus.FAILED
        self._phase = "failed"
        self._message = message
        return self._decision(
            self._hold_action(self._current_hold_command()), message
        )

    @staticmethod
    def _hold_action(gripper: float) -> np.ndarray:
        action = np.zeros(7, dtype=np.float32)
        action[6] = gripper
        return action

    def _engaged_gripper_command(self) -> float:
        return GRIPPER_OPEN if self._grasp_mode is GraspMode.EXPAND else GRIPPER_CLOSE

    def _begin_grasp_engagement(self, step: SkillStep) -> None:
        if self._pending_grasp_engagement is not None:
            return
        jaw_behavior = (
            JawBehavior.OPEN_FINGERS_INTERIOR_BRACE
            if self._grasp_mode is GraspMode.EXPAND
            else JawBehavior.CLOSE_FINGERS
        )
        self._pending_grasp_engagement = _PendingGraspEngagement(
            source_text=step.subject,
            source_class=step.subject,
            grasp_mode=self._grasp_mode.value,
            jaw_behavior=jaw_behavior,
        )

    def _finish_grasp_engagement(
        self,
        *,
        accepted: bool,
        reason: GraspReason,
        evidence_source: GraspEvidenceCategory,
    ) -> None:
        pending = self._pending_grasp_engagement
        if pending is None:
            return
        self._grasp_attempt_journal.append(
            source_text=pending.source_text,
            source_class=pending.source_class,
            grasp_mode=pending.grasp_mode,
            jaw_behavior=pending.jaw_behavior,
            accepted=accepted,
            reason=reason,
            evidence_source=evidence_source,
        )
        self._pending_grasp_engagement = None

    def _released_gripper_command(self) -> float:
        return GRIPPER_CLOSE if self._grasp_mode is GraspMode.EXPAND else GRIPPER_OPEN

    def _current_hold_command(self) -> float:
        return (
            self._engaged_gripper_command()
            if self._held_subject is not None
            else self._released_gripper_command()
        )
