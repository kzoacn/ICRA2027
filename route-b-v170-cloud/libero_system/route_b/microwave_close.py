"""Close a sensor-grounded appliance with a compact top-down face pusher."""
from __future__ import annotations

import numpy as np

from .microwave_interior import observed_door_panel


class MicrowaveFaceCloser:
    """Finite-state contact motion using RGB-D and public robot measurements."""

    def __init__(self, observation, detector):
        self.detector = detector
        self.hinge, self.side, self.outward, self.angle, radial, normal = detector.panel_geometry(observation)
        self.initial_angle = self.angle
        self.radius = .200
        self.height = float(self.hinge[2]+.045)
        self.safe_height = max(float(observation.proprio.ee_position_world[2]), self.hinge[2]+.160)
        down = np.array((0., 0., -1.))
        preferred = np.column_stack((np.cross(radial, down), radial, down))
        from ..common.panda_kinematics import reachable_equivalent_frame

        outside = self.point(self.angle, .080)
        high = outside.copy()
        high[2] = self.safe_height
        self.rotation = reachable_equivalent_frame(
            observation.proprio.joint_position, observation.proprio.T_world_ee,
            (high, outside), preferred, preferred @ np.diag((-1., -1., 1.)))
        self.start_rotation = observation.proprio.T_world_ee[:3, :3].copy()
        self.lift = observation.proprio.ee_position_world.copy()
        self.lift[2] = self.safe_height
        self.outside = outside
        self.high = high
        self.sweep_angle = self.angle
        self.total_ticks = 0
        self.last_position = observation.proprio.ee_position_world.copy()
        self.stall_ticks = 0

    def point(self, angle, normal_offset):
        radial = np.cos(angle)*self.side-np.sin(angle)*self.outward
        normal = np.sin(angle)*self.side+np.cos(angle)*self.outward
        point = self.hinge+self.radius*radial+normal_offset*normal
        point[2] = self.height
        return point

    def move(self, policy, observation, point, rotation=None):
        policy._motion_position = point.copy()
        policy._motion_rotation = self.rotation.copy() if rotation is None else rotation.copy()
        if self.total_ticks % 10 == 0:
            self.detector._append_selector_diagnostic({
                'kind': 'microwave_face_motion', 'phase': policy._phase,
                'tick': self.total_ticks, 'sweep_angle_rad': self.sweep_angle,
                'ee_world_m': observation.proprio.ee_position_world.tolist(),
                'target_world_m': point.tolist(),
                'gripper_width_m': float(observation.proprio.gripper_width_m),
            })
        return policy._move(observation, 1.0)

    def act(self, policy, observation):
        self.total_ticks += 1
        if self.total_ticks > 270:
            return policy._fail('microwave face closure exhausted its local motion budget')
        ee = observation.proprio.ee_position_world
        self.stall_ticks = self.stall_ticks+1 if np.linalg.norm(ee-self.last_position)<.001 else 0
        self.last_position = ee.copy()
        phase = policy._phase
        if phase == 'detect':
            policy._set_phase('microwave_face_lift')
            phase = policy._phase
        if phase == 'microwave_face_lift':
            if np.linalg.norm(ee-self.lift)<.022:
                policy._set_phase('microwave_face_align')
            return self.move(policy, observation, self.lift, self.start_rotation)
        if phase == 'microwave_face_align':
            if policy._rotation_error(observation.proprio.T_world_ee[:3, :3], self.rotation)<.18:
                policy._set_phase('microwave_face_move_above')
            return self.move(policy, observation, self.lift)
        if phase == 'microwave_face_move_above':
            if np.linalg.norm(ee-self.high)<.022:
                policy._set_phase('microwave_face_descend')
            return self.move(policy, observation, self.high)
        if phase == 'microwave_face_descend':
            if np.linalg.norm(ee-self.outside)<.018:
                policy._set_phase('microwave_face_engage')
            return self.move(policy, observation, self.outside)
        if phase == 'microwave_face_engage':
            target = self.point(self.angle, .005)
            if np.linalg.norm(ee-target)<.030 or (self.stall_ticks>=5 and np.linalg.norm(ee-target)<.055):
                policy._set_phase('microwave_face_sweep')
            return self.move(policy, observation, target)
        if phase == 'microwave_face_sweep':
            target = self.point(self.sweep_angle, .005)
            if self.stall_ticks>=8 and policy._phase_ticks % 8==0:
                try:
                    angle, _, _, trace = observed_door_panel(
                        observation, self.hinge, self.side, self.outward)
                except LookupError:
                    pass
                else:
                    self.detector._append_selector_diagnostic({'kind':'microwave_panel_contact_fit', **trace})
                    if abs(angle)<.10:
                        self.outside = ee+self.outward*.110
                        policy._set_phase('microwave_face_withdraw')
                        return self.move(policy, observation, self.outside)
            if np.linalg.norm(ee-target)<.038:
                self.sweep_angle = min(.045, self.sweep_angle+.065)
            if self.sweep_angle>=.045 and np.linalg.norm(ee-target)<.045:
                policy._set_phase('microwave_face_seat')
            return self.move(policy, observation, self.point(self.sweep_angle, .005))
        if phase == 'microwave_face_seat':
            if policy._phase_ticks>=24:
                self.outside = ee+self.outward*.110
                policy._set_phase('microwave_face_withdraw')
            return self.move(policy, observation, self.point(.045, -.020))
        if phase == 'microwave_face_withdraw':
            if np.linalg.norm(ee-self.outside)<.020:
                self.high = ee.copy()
                self.high[2] = self.safe_height
                policy._set_phase('microwave_face_retreat')
            return self.move(policy, observation, self.outside)
        if phase == 'microwave_face_retreat':
            if np.linalg.norm(ee-self.high)<.025:
                policy._set_phase('microwave_face_verify')
            return self.move(policy, observation, self.high)
        if phase == 'microwave_face_verify':
            try:
                angle, _, _, trace = observed_door_panel(observation, self.hinge, self.side, self.outward)
            except LookupError as exc:
                return policy._fail(f'microwave final panel observation failed: {exc}')
            self.detector._append_selector_diagnostic({'kind': 'microwave_panel_final_fit', **trace})
            if abs(angle)<.10:
                return policy._complete(f'microwave closure verified from observed panel angle {angle:.4f} rad')
            return policy._fail(f'microwave panel remained open at observed angle {angle:.4f} rad')
        return policy._fail(f'invalid microwave face phase {phase}')
