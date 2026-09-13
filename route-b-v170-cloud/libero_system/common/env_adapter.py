"""Auditable sensor-only adapter around LIBERO's OffScreenRenderEnv."""

from __future__ import annotations

import copy
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .camera_geometry import metric_depth_from_normalized, quaternion_xyzw_to_matrix
from .observation import CameraCalibration, CameraFrame, Proprioception, RobotObservation
from .policy import OSCAction, PolicyTask


UNIFIED_EVALUATION_HORIZON = 520

# Evaluator-only reset RNG checkpoints. LIBERO fixture body transforms are
# sampled on reset and are not restored by its qpos initial-state files.
# Recreating an environment must therefore restore the task's continuous
# reset stream, rather than reseeding every episode at the same first draw.
@dataclass(frozen=True)
class _TaskResetCheckpoint:
    rng_state: tuple
    property_initializers: Any


_TASK_RESET_RNG_STATES: dict[tuple[str, int, int, int], _TaskResetCheckpoint] = {}


def _capture_task_reset(env: Any) -> _TaskResetCheckpoint:
    # LIBERO appends articulation samplers during every hard reset. Their
    # history changes the number of random draws before fixture placement.
    # This is evaluator reset state, never an observation or a policy input.
    domain = getattr(env, "env", None)
    initializers = getattr(domain, "object_property_initializers", None)
    return _TaskResetCheckpoint(np.random.get_state(), copy.deepcopy(initializers))


def _restore_task_reset(env: Any, checkpoint: _TaskResetCheckpoint) -> None:
    if checkpoint.property_initializers is not None:
        env.env.object_property_initializers = copy.deepcopy(checkpoint.property_initializers)
    np.random.set_state(checkpoint.rng_state)


DEFAULT_MAX_STEPS = {
    # The policy-facing step budget must not reveal which benchmark suite is
    # running.  Formal Route B/C evaluation therefore uses one fixed horizon
    # for all five suites; custom developer runs may still provide an explicit
    # LiberoEnvConfig.max_steps value.
    "libero_spatial": UNIFIED_EVALUATION_HORIZON,
    "libero_object": UNIFIED_EVALUATION_HORIZON,
    "libero_goal": UNIFIED_EVALUATION_HORIZON,
    "libero_10": UNIFIED_EVALUATION_HORIZON,
    "libero_90": UNIFIED_EVALUATION_HORIZON,
}


def configure_runtime_environment() -> None:
    """Set safe defaults before MuJoCo / robosuite are imported."""
    workspace = Path(__file__).resolve().parents[2]
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(workspace / ".local" / "libero"))
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/libero-system-numba")
    # The managed WSL environment can expose robosuite's source through an
    # import path for which Numba has no cache locator.  Disable its handful of
    # transform JITs rather than making environment startup nondeterministic.
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/libero-system-matplotlib")
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    prefix_lib = str(Path(sys.prefix) / "lib")
    extra = "/usr/lib/wsl/lib" if Path("/usr/lib/wsl/lib").is_dir() else None
    entries = [prefix_lib, extra, *os.environ.get("LD_LIBRARY_PATH", "").split(":")]
    os.environ["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(item for item in entries if item))


@dataclass(frozen=True, slots=True)
class LiberoEnvConfig:
    suite_name: str = "libero_goal"
    task_id: int = 0
    init_state_index: int = 0
    width: int = 256
    height: int = 256
    control_frequency: int = 20
    settle_steps: int = 10
    max_steps: int | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.control_frequency <= 0:
            raise ValueError("image size and control_frequency must be positive")
        if self.task_id < 0 or self.init_state_index < 0 or self.settle_steps < 0:
            raise ValueError("task/init indices and settle_steps must be non-negative")


@dataclass(frozen=True, slots=True)
class TaskMetadata:
    suite_name: str
    task_id: int
    task_name: str
    instruction: str
    init_state_index: int

    def policy_view(self, episode_id: str) -> PolicyTask:
        return PolicyTask(instruction=self.instruction, episode_id=episode_id)


@dataclass(frozen=True, slots=True)
class EvaluationSignal:
    reward: float
    success: bool
    terminated: bool
    truncated: bool
    raw_done: bool


@dataclass(frozen=True, slots=True)
class EnvironmentStep:
    observation: RobotObservation
    evaluation: EvaluationSignal


class LiberoEnvAdapter:
    """Single-episode LIBERO adapter with a strict policy-observation whitelist."""

    _CAMERAS = {"agentview": "agentview", "wrist": "robot0_eye_in_hand"}

    def __init__(self, config: LiberoEnvConfig):
        self.config = config
        self._env: Any | None = None
        self._suite: Any | None = None
        self._task_metadata: TaskMetadata | None = None
        self._init_states: Any | None = None
        self._episode_steps = 0
        self._last_observation: RobotObservation | None = None

    @property
    def max_steps(self) -> int:
        return self.config.max_steps or DEFAULT_MAX_STEPS.get(
            self.config.suite_name, UNIFIED_EVALUATION_HORIZON
        )

    @property
    def episode_steps(self) -> int:
        """Evaluator-owned action count; never included in policy observations."""

        return self._episode_steps

    @property
    def task_metadata(self) -> TaskMetadata:
        self._ensure_env()
        assert self._task_metadata is not None
        return self._task_metadata

    @property
    def current_observation(self) -> RobotObservation:
        if self._last_observation is None:
            raise RuntimeError("reset() must be called before accessing current_observation")
        return self._last_observation

    def _ensure_env(self) -> None:
        if self._env is not None:
            return
        configure_runtime_environment()
        import torch
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        suites = benchmark.get_benchmark_dict()
        if self.config.suite_name not in suites:
            raise ValueError(f"unknown LIBERO suite {self.config.suite_name!r}")
        self._suite = suites[self.config.suite_name]()
        if not 0 <= self.config.task_id < len(self._suite.tasks):
            raise ValueError(f"task_id {self.config.task_id} is out of range")
        task = self._suite.get_task(self.config.task_id)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        init_path = Path(get_libero_path("init_states")) / task.problem_folder / Path(task.init_states_file).name
        self._init_states = torch.load(init_path, weights_only=False)
        if len(self._init_states) == 0:
            raise RuntimeError(f"no init states in {init_path}")
        selected = self.config.init_state_index
        if not 0 <= selected < len(self._init_states):
            raise IndexError(
                f"init_state_index {selected} is out of range for {init_path} "
                f"({len(self._init_states)} available states)"
            )
        self._task_metadata = TaskMetadata(
            suite_name=self.config.suite_name,
            task_id=self.config.task_id,
            task_name=task.name,
            instruction=task.language,
            init_state_index=selected,
        )
        self._env = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_names=list(self._CAMERAS.values()),
            camera_heights=self.config.height,
            camera_widths=self.config.width,
            camera_depths=True,
            camera_segmentations=None,
            controller="OSC_POSE",
            control_freq=self.config.control_frequency,
            horizon=self.max_steps + self.config.settle_steps,
            # This adapter, rather than a benchmark-returned done flag, owns
            # the fixed episode horizon.  In LIBERO's BDDL environments the
            # returned ``done`` is task success; allowing subsequent steps is
            # required to keep it from becoming controller feedback.  The
            # adapter and both route drivers still enforce ``max_steps``.
            ignore_done=True,
        )

    def reset(self) -> RobotObservation:
        self._ensure_env()
        assert self._env is not None and self._task_metadata is not None
        key = (self.config.suite_name, self.config.task_id, self.config.seed)
        index = self.config.init_state_index
        checkpoint = _TASK_RESET_RNG_STATES.get((*key, index))
        if checkpoint is None:
            # A shard may begin in the middle of the official 50-state list.
            # Replay only the missing resets to recover its stream position;
            # these are setup resets, never policy episodes or scored trials.
            self._env.seed(self.config.seed)
            for previous in range(index):
                self._env.reset()
                _TASK_RESET_RNG_STATES[(*key, previous + 1)] = _capture_task_reset(self._env)
        else:
            _restore_task_reset(self._env, checkpoint)
        self._env.reset()
        _TASK_RESET_RNG_STATES[(*key, index + 1)] = _capture_task_reset(self._env)
        raw = self._env.set_init_state(self._init_states[self._task_metadata.init_state_index])
        for robot in self._env.robots:
            robot.controller.use_delta = True
        for _ in range(self.config.settle_steps):
            raw, _, _, _ = self._env.step(OSCAction.hold(-1.0).values)
        self._episode_steps = 0
        self._last_observation = self._sanitize_observation(raw)
        return self._last_observation

    def step(self, action: OSCAction | np.ndarray) -> EnvironmentStep:
        if self._env is None or self._last_observation is None:
            raise RuntimeError("reset() must be called before step()")
        native = action if isinstance(action, OSCAction) else OSCAction.from_array(action)
        raw, reward, raw_done, _raw_info = self._env.step(native.values)
        self._episode_steps += 1
        success = bool(self._env.check_success())
        truncated = bool(self._episode_steps >= self.max_steps and not success)
        terminated = bool(success or raw_done)
        self._last_observation = self._sanitize_observation(raw)
        return EnvironmentStep(
            observation=self._last_observation,
            evaluation=EvaluationSignal(
                reward=float(reward),
                success=success,
                terminated=terminated,
                truncated=truncated,
                raw_done=bool(raw_done),
            ),
        )

    def check_success(self) -> bool:
        """Evaluation-only API.  Never pass this value into a policy or planner."""
        if self._env is None:
            raise RuntimeError("environment is not open")
        return bool(self._env.check_success())

    def _sanitize_observation(self, raw: dict[str, Any]) -> RobotObservation:
        assert self._env is not None
        from robosuite import macros
        from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix

        extent = float(self._env.sim.model.stat.extent)
        near_m = float(self._env.sim.model.vis.map.znear) * extent
        far_m = float(self._env.sim.model.vis.map.zfar) * extent
        observation_v_flipped = macros.IMAGE_CONVENTION == "opengl"
        cameras: dict[str, CameraFrame] = {}
        for public_name, sim_name in self._CAMERAS.items():
            rgb = np.asarray(raw[f"{sim_name}_image"], dtype=np.uint8)
            depth = np.asarray(raw[f"{sim_name}_depth"], dtype=np.float32).squeeze(-1)
            calibration = CameraCalibration(
                name=public_name,
                width=self.config.width,
                height=self.config.height,
                intrinsic=get_camera_intrinsic_matrix(
                    self._env.sim, sim_name, self.config.height, self.config.width
                ),
                T_world_camera=get_camera_extrinsic_matrix(self._env.sim, sim_name),
                observation_v_flipped=observation_v_flipped,
            )
            cameras[public_name] = CameraFrame(
                rgb=np.ascontiguousarray(rgb),
                depth_m=metric_depth_from_normalized(np.ascontiguousarray(depth), near_m, far_m),
                calibration=calibration,
            )

        ee_pos = np.asarray(raw["robot0_eef_pos"], dtype=np.float64)
        ee_quat = np.asarray(raw["robot0_eef_quat"], dtype=np.float64)
        T_world_ee = np.eye(4, dtype=np.float64)
        T_world_ee[:3, :3] = quaternion_xyzw_to_matrix(ee_quat)
        T_world_ee[:3, 3] = ee_pos
        gripper_qpos = np.asarray(raw["robot0_gripper_qpos"], dtype=np.float64)
        robot = self._env.robots[0]
        proprio = Proprioception(
            T_world_ee=T_world_ee,
            ee_quat_xyzw=ee_quat,
            joint_position=raw["robot0_joint_pos"],
            joint_velocity=raw["robot0_joint_vel"],
            gripper_qpos=gripper_qpos,
            gripper_qvel=raw["robot0_gripper_qvel"],
            gripper_width_m=float(abs(gripper_qpos[0] - gripper_qpos[1])),
            ee_force_sensor=robot.ee_force,
            ee_torque_sensor=robot.ee_torque,
        )
        return RobotObservation(
            cameras=cameras,
            proprio=proprio,
        )

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
            self._last_observation = None

    def __enter__(self) -> "LiberoEnvAdapter":
        self._ensure_env()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
