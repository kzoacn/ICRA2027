"""Real LIBERO dual-RGB-D sensor and control-interface smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .camera_geometry import backproject_depth, project_world_points
from .env_adapter import LiberoEnvAdapter, LiberoEnvConfig, configure_runtime_environment
from .policy import OSCAction


def _camera_report(frame) -> dict[str, object]:
    center = (frame.calibration.height // 2, frame.calibration.width // 2)
    mask = np.zeros(frame.depth_m.shape, dtype=bool)
    mask[center] = True
    point_world = backproject_depth(frame, mask=mask, world=True)[0]
    pixel, camera_z = project_world_points(point_world[None], frame)
    target_uv = np.array([center[1], center[0]], dtype=np.float64)
    return {
        "rgb_shape": list(frame.rgb.shape),
        "rgb_dtype": str(frame.rgb.dtype),
        "depth_shape": list(frame.depth_m.shape),
        "depth_unit": "m",
        "depth_percentiles_m": np.percentile(frame.depth_m, [1, 50, 99]).tolist(),
        "K": frame.calibration.intrinsic.tolist(),
        "T_world_camera": frame.calibration.T_world_camera.tolist(),
        "observation_v_flipped": frame.calibration.observation_v_flipped,
        "center_roundtrip_error_px": float(np.linalg.norm(pixel[0] - target_uv)),
        "center_camera_z_m": float(camera_z[0]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state", type=int, default=0)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--artifact-dir", help="Optional directory for direct and EE-overlay PNG diagnostics")
    args = parser.parse_args(argv)
    configure_runtime_environment()
    config = LiberoEnvConfig(
        suite_name=args.suite,
        task_id=args.task_id,
        init_state_index=args.init_state,
        width=args.size,
        height=args.size,
    )
    with LiberoEnvAdapter(config) as environment:
        observation = environment.reset()
        initial_wrist_pose = observation.wrist.calibration.T_world_camera.copy()
        step = environment.step(OSCAction.hold(-1.0))
        artifacts: dict[str, str] = {}
        if args.artifact_dir:
            import imageio.v2 as imageio

            artifact_dir = Path(args.artifact_dir)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            direct_path = artifact_dir / "agentview_direct.png"
            overlay_path = artifact_dir / "agentview_ee_overlay.png"
            direct = step.observation.agentview.rgb
            overlay = np.array(direct, copy=True)
            ee_uv, ee_z = project_world_points(
                step.observation.proprio.ee_position_world[None], step.observation.agentview
            )
            u, v = np.rint(ee_uv[0]).astype(int)
            if ee_z[0] > 0 and 0 <= u < overlay.shape[1] and 0 <= v < overlay.shape[0]:
                radius = max(2, overlay.shape[0] // 80)
                overlay[max(0, v - radius) : min(overlay.shape[0], v + radius + 1), u] = [255, 0, 0]
                overlay[v, max(0, u - radius) : min(overlay.shape[1], u + radius + 1)] = [255, 0, 0]
            imageio.imwrite(direct_path, direct)
            imageio.imwrite(overlay_path, overlay)
            artifacts = {"direct": str(direct_path), "ee_overlay": str(overlay_path)}
        report = {
            "gate": "PASS",
            "task": {
                "suite": environment.task_metadata.suite_name,
                "task_id": environment.task_metadata.task_id,
                "instruction": environment.task_metadata.instruction,
                "init_state": environment.task_metadata.init_state_index,
            },
            "observation_whitelist": [
                "agentview RGB-D + calibration",
                "wrist RGB-D + calibration",
                "EE pose/wrench",
                "joint position/velocity",
                "gripper position/velocity/width",
            ],
            "forbidden_policy_inputs": ["object truth poses", "sim segmentation", "BDDL predicates"],
            "agentview": _camera_report(step.observation.agentview),
            "wrist": _camera_report(step.observation.wrist),
            "wrist_extrinsic_delta_frobenius": float(
                np.linalg.norm(step.observation.wrist.calibration.T_world_camera - initial_wrist_pose)
            ),
            "proprio": {
                "T_world_ee": step.observation.proprio.T_world_ee.tolist(),
                "gripper_width_m": step.observation.proprio.gripper_width_m,
                "joint_position_shape": list(step.observation.proprio.joint_position.shape),
                "wrench_shape": [
                    len(step.observation.proprio.ee_force_sensor),
                    len(step.observation.proprio.ee_torque_sensor),
                ],
            },
            "action": {
                "shape": list(OSCAction.hold(-1.0).values.shape),
                "range": [-1.0, 1.0],
                "controller": "relative OSC_POSE at 20 Hz",
                "gripper": "-1=open, +1=close",
            },
            "evaluation_only": {
                "success": step.evaluation.success,
                "terminated": step.evaluation.terminated,
                "truncated": step.evaluation.truncated,
            },
            "artifacts": artifacts,
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
