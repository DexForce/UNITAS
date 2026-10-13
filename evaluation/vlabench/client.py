#!/usr/bin/env python3
"""Evaluate UNITAS with the fixed VLABench track configurations."""

from __future__ import annotations

import argparse
import json
import hashlib
import random
import os
import traceback
from collections import deque
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from scipy.spatial.transform import Rotation

from unitas.inference.client import PolicyClient
from unitas.inference.observations import save_observation
from evaluation.results import record_protocol, write_json
from evaluation.vlabench.configs import CONFIG_DIR, TRACKS, load_configs
from evaluation.vlabench.aggregate import aggregate


os.environ.setdefault("MUJOCO_GL", "egl")

OFFICIAL_CAMERA_NAMES = (
    # LM4ManipDMEnv.get_observation() iterates MuJoCo camera IDs.  The
    # official scene registers the fixed cameras as right, left, front,
    # followed by the robot wrist camera.  The converted training HDF5 files
    # preserve this source mapping in each camera's source_cam_index attr.
    "right_camera",
    "left_camera",
    "front_camera",
    "wrist_camera",
)

# MuJoCo camera axes are +X right, +Y up, -Z forward.  VLABench's converted
# episodes use OpenCV camera axes (+X right, +Y down, +Z forward).
MUJOCO_CAMERA_TO_OPENCV = np.diag([1.0, -1.0, -1.0]).astype(np.float32)


def _matrix(position: Any, rotation: Any) -> np.ndarray:
    result = np.eye(4, dtype=np.float32)
    result[:3, :3] = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    result[:3, 3] = np.asarray(position, dtype=np.float32).reshape(3)
    return result


def _world_from_robot_base(env: Any) -> np.ndarray:
    body = env.physics.bind(env.robot.link_base)
    return _matrix(body.xpos, body.xmat)


def _observed_ee_world(observation: dict[str, Any]) -> np.ndarray:
    ee = np.asarray(observation["ee_state"], dtype=np.float32).reshape(-1)
    if ee.shape != (8,):
        raise ValueError(f"official VLABench ee_state must be [8], got {ee.shape}")
    # Official VLABench uses wxyz; SciPy uses xyzw.
    rotation = Rotation.from_quat(np.roll(ee[3:7], -1)).as_matrix()
    return _matrix(ee[:3], rotation)


def state_observation(env: Any, observation: dict[str, Any]) -> dict[str, Any]:
    arm_joints = np.asarray(observation["q_state"], dtype=np.float32).reshape(-1)
    finger_joints = np.asarray(
        [
            np.asarray(
                env.physics.bind(env.robot.mjcf_model.find("joint", f"finger_joint{index}")).qpos
            ).item()
            for index in (1, 2)
        ],
        dtype=np.float32,
    )
    return {
        # The official online observation exposes only the seven arm joints in
        # q_state.  Converted training episodes store those seven values plus
        # the two measured finger-joint positions.
        "joint_positions": np.concatenate((arm_joints, finger_joints)),
        "world_from_robot_base": _world_from_robot_base(env),
        "observed_ee_world": _observed_ee_world(observation),
    }


def model_observation(env: Any, observation: dict[str, Any]) -> dict[str, Any]:
    payload = state_observation(env, observation)
    rgbs = np.asarray(observation["rgb"])
    depths = np.asarray(observation["depth"])
    intrinsics = np.asarray(observation["instrinsic"])
    if len(rgbs) != len(OFFICIAL_CAMERA_NAMES):
        raise ValueError(f"expected {len(OFFICIAL_CAMERA_NAMES)} official cameras, got {len(rgbs)}")
    payload["cameras"] = {
        name: {
            "rgb": np.asarray(rgbs[index], dtype=np.uint8),
            "depth": np.asarray(depths[index], dtype=np.float32),
            "intrinsic": np.asarray(intrinsics[index], dtype=np.float32),
            # dm_env.get_camera_matrix() uses model.cam_mat0.  That is valid
            # for the three fixed cameras but stale for the wrist camera.
            # Training conversion used the dynamic MuJoCo camera pose.
            "camera_to_world": _matrix(
                env.physics.data.cam_xpos[index],
                np.asarray(env.physics.data.cam_xmat[index]).reshape(3, 3)
                @ MUJOCO_CAMERA_TO_OPENCV,
            ),
        }
        for index, name in enumerate(OFFICIAL_CAMERA_NAMES)
    }
    return payload


def _video_frame(observation: dict[str, Any]) -> np.ndarray:
    views = np.asarray(observation["rgb"], dtype=np.uint8)
    return np.vstack((np.hstack(views[:2]), np.hstack(views[2:4])))


class VLABenchRemotePolicy:
    """Queue action chunks while preserving every measured history state."""

    def __init__(self, host: str, port: int, timeout: float) -> None:
        self.client = PolicyClient(host, port, domain="vlabench", timeout=timeout)
        self.actions: deque[np.ndarray] = deque()
        self.pending_history: list[dict[str, Any]] = []
        self.has_executed_action = False
        self.last_replanned = False
        self.last_chunk_length = 0

    def reset(self) -> None:
        self.actions.clear()
        self.pending_history.clear()
        self.has_executed_action = False
        self.last_replanned = False
        self.last_chunk_length = 0
        self.client.reset()

    def predict(self, env: Any, observation: dict[str, Any], instruction: str) -> np.ndarray:
        current_state = state_observation(env, observation)
        if self.has_executed_action:
            self.pending_history.append(current_state)
        if not self.actions:
            chunk = self.client.act(
                model_observation(env, observation),
                instruction,
                self.pending_history,
            )
            self.pending_history = []
            if chunk.ndim != 2 or chunk.shape[1] != 8 or len(chunk) == 0:
                raise RuntimeError(f"server returned invalid VLABench chunk {chunk.shape}")
            self.actions.extend(row.copy() for row in chunk)
            self.last_replanned = True
            self.last_chunk_length = int(len(chunk))
        else:
            self.last_replanned = False
        self.has_executed_action = True
        return self.actions.popleft()


def _official_task_max_steps(task_name: str) -> int:
    from VLABench.configs import name2config
    from VLABench.utils.utils import find_key_by_value

    config_path = Path(os.environ["VLABENCH_ROOT"]) / "configs/task_config.json"
    task_configs = json.loads(config_path.read_text(encoding="utf-8"))
    config_name = find_key_by_value(name2config, task_name)
    return int(
        task_configs.get(config_name, {}).get("evaluation", {}).get("max_episode_length", 200)
    )


def _episode_path(output_dir: Path, task: str, episode_index: int) -> Path:
    return output_dir / task / f"episode_{episode_index:03d}.json"


def run_episode(
    args: argparse.Namespace,
    policy: VLABenchRemotePolicy,
    task: str,
    episode_index: int,
    episode_config: dict[str, Any],
) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    result_path = _episode_path(output_dir, task, episode_index)
    environment_seed = (
        args.seed + int(hashlib.sha256(task.encode()).hexdigest()[:8], 16) + episode_index
    ) % (2**32)
    if args.resume and result_path.is_file():
        try:
            result = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError):
            result = {}
        if (
            not result.get("error")
            and result.get("track") == args.track
            and result.get("task") == task
            and result.get("episode") == episode_index
            and result.get("environment_seed") == environment_seed
        ):
            return result
    random.seed(environment_seed)
    np.random.seed(environment_seed)

    from VLABench.envs import load_env
    from VLABench.utils.utils import euler_to_quaternion

    env = None
    frames: list[np.ndarray] = []
    trace: list[dict[str, Any]] = []
    max_steps = int(args.max_steps or _official_task_max_steps(task))
    instruction = ""
    policy_steps = 0
    ik_failures = 0
    result: dict[str, Any]
    try:
        env = load_env(
            task,
            episode_config=episode_config,
            random_init=False,
            run_mode="eval",
        )
        env.reset()
        policy.reset()
        instruction = str(env.task.get_instruction())
        success = False

        while policy_steps < max_steps and not success:
            observation = env.get_observation(require_pcd=False)
            if args.save_videos:
                frames.append(_video_frame(observation))
            if (
                policy_steps == 0
                and args.save_observation is not None
                and not args.save_observation.exists()
            ):
                save_observation(
                    args.save_observation,
                    domain="vlabench",
                    observation=model_observation(env, observation),
                    instruction=instruction,
                    history=[],
                )
            action_ee = policy.predict(env, observation, instruction)
            position = np.asarray(action_ee[:3], dtype=np.float64)
            euler = np.asarray(action_ee[3:6], dtype=np.float64)
            fingers = np.clip(np.asarray(action_ee[6:8], dtype=np.float64), 0.0, 0.04)
            ik_success, arm_target = env.robot.get_qpos_from_ee_pos(
                physics=env.physics,
                pos=position,
                quat=euler_to_quaternion(*euler),
            )
            ik_failures += int(not ik_success)
            joint_action = np.concatenate([arm_target, fingers])
            measured_before = np.asarray(observation["ee_state"], dtype=np.float64)
            substeps = 0

            for _ in range(int(args.max_substeps)):
                timestep = env.step(joint_action)
                substeps += 1
                if timestep.last():
                    success = True
                    break
                current = np.asarray(env.task.robot.get_qpos(env.physics)).reshape(-1)
                if np.max(np.abs(current - joint_action[:7])) < float(args.tolerance):
                    break
            success = bool(success or env.task.check_success(env.physics))
            if args.save_traces:
                measured_after = np.asarray(
                    env.get_observation(require_pcd=False)["ee_state"],
                    dtype=np.float64,
                )
                trace.append(
                    {
                        "policy_step": int(policy_steps),
                        "replanned": bool(policy.last_replanned),
                        "chunk_length": int(policy.last_chunk_length),
                        "command_world_ee": np.asarray(action_ee, dtype=np.float64).tolist(),
                        "measured_before_world_ee": measured_before.tolist(),
                        "measured_after_world_ee": measured_after.tolist(),
                        "arm_joint_target": np.asarray(arm_target, dtype=np.float64).tolist(),
                        "finger_target": fingers.tolist(),
                        "ik_success": bool(ik_success),
                        "substeps": int(substeps),
                        "success": bool(success),
                    }
                )
            policy_steps += 1

        result = {
            "track": args.track,
            "task": task,
            "episode": int(episode_index),
            "instruction": instruction,
            "success": bool(success),
            "progress_score": float(env.get_task_progress()),
            "intention_score": float(
                env.get_intention_score(threshold=float(args.intention_threshold))
            ),
            "policy_steps": int(policy_steps),
            "max_steps": int(max_steps),
            "max_substeps": int(args.max_substeps),
            "ik_failures": int(ik_failures),
            "error": None,
        }
    except Exception as error:
        traceback.print_exc()
        result = {
            "track": args.track,
            "task": task,
            "episode": int(episode_index),
            "instruction": instruction,
            "success": False,
            "progress_score": 0.0,
            "intention_score": 0.0,
            "policy_steps": int(policy_steps),
            "max_steps": int(max_steps),
            "max_substeps": int(args.max_substeps),
            "ik_failures": int(ik_failures),
            "error": f"{type(error).__name__}: {error}",
        }
    finally:
        if env is not None:
            env.close()

    result["environment_seed"] = environment_seed
    write_json(result_path, result)
    if args.save_traces:
        result_path.with_suffix(".trace.json").write_text(
            json.dumps(trace, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.save_videos and frames:
        video_path = result_path.with_suffix(".mp4")
        imageio.mimsave(video_path, frames, fps=int(args.video_fps), codec="libx264")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def _selected_tasks(all_tasks: list[str], selection: str) -> list[str]:
    if selection.strip().lower() == "all":
        return all_tasks
    selected = [item.strip() for item in selection.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(all_tasks))
    if unknown:
        raise ValueError(f"tasks are not in the selected track: {unknown}")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--vlabench-root", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    parser.add_argument("--track", choices=("all", *TRACKS), default="all")
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        help="Limit each task; defaults to all its fixed configurations",
    )
    parser.add_argument("--max-steps", type=int, help="Override the task's rollout limit")
    parser.add_argument("--max-substeps", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--intention-threshold", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=160484)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--save-traces", action="store_true")
    parser.add_argument("--save-observation", type=Path)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episode_start < 0 or (
        args.episodes_per_task is not None and args.episodes_per_task < 1
    ):
        raise ValueError("Episode range must be non-negative and non-empty")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("--shard-id must be in [0, --num-shards)")
    if (
        args.max_substeps < 1
        or args.tolerance <= 0
        or (args.max_steps is not None and args.max_steps < 1)
    ):
        raise ValueError("Control limits and tolerance must be positive")
    os.environ["VLABENCH_ROOT"] = str(args.vlabench_root.resolve())
    __import__("VLABench.tasks")
    __import__("VLABench.robots")
    configs = load_configs(args.config_dir)
    selected = TRACKS if args.track == "all" else (args.track,)
    jobs = []
    for track in selected:
        for task in _selected_tasks(list(configs[track]), args.tasks):
            count = len(configs[track][task])
            stop = (
                count
                if args.episodes_per_task is None
                else min(count, args.episode_start + args.episodes_per_task)
            )
            if args.episode_start >= count:
                raise ValueError(f"Episode start exceeds {track}/{task}'s {count} configurations")
            jobs.extend((track, task, index) for index in range(args.episode_start, stop))
    policy = VLABenchRemotePolicy(args.host, args.port, args.timeout)
    output = args.output_dir
    try:
        health = policy.client.health()
        if health.get("domain") != "vlabench":
            raise ValueError("The server must load a VLABench policy")
        protocol = {
            "server": health.get("protocol", {}),
            "environment_seed_base": args.seed,
            "environment_seed_formula": "(seed + int(sha256(task)[:8],16) + episode_index) % 2**32",
            "max_substeps": args.max_substeps,
            "tolerance": args.tolerance,
            "max_steps": args.max_steps,
            "intention_threshold": args.intention_threshold,
            "tracks": json.loads((CONFIG_DIR / "manifest.json").read_text())["tracks"],
        }
        record_protocol(output / "run_config.json", protocol)
        for track, task, index in jobs[args.shard_id :: args.num_shards]:
            episode_args = argparse.Namespace(**vars(args))
            episode_args.track = track
            episode_args.output_dir = output / track
            run_episode(episode_args, policy, task, index, configs[track][task][index])
    finally:
        policy.client.close()
    print(json.dumps(aggregate(output, configs), indent=2))


if __name__ == "__main__":
    main()
