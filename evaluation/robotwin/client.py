"""Evaluate the UNITAS HTTP server with an official RoboTwin 2.0 checkout."""

from __future__ import annotations
import argparse
import importlib
import json
import logging
from contextlib import ExitStack
import os
import random
import sys
from pathlib import Path
from typing import Any
import imageio.v2 as imageio
import numpy as np
import yaml
from unitas.inference.client import PolicyClient
from unitas.inference.observations import save_observation
from evaluation.results import write_json

ALL_TASKS = (
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
)


def _pose_matrix(pose: Any) -> np.ndarray:
    return np.asarray(pose.to_transformation_matrix(), dtype=np.float32)


def _robot_state(env: Any) -> dict[str, np.ndarray]:
    names: list[str] = []
    positions: list[float] = []
    seen: set[int] = set()
    for entity in (env.robot.left_entity, env.robot.right_entity):
        if id(entity) in seen:
            continue
        seen.add(id(entity))
        qpos = np.asarray(entity.get_qpos(), dtype=np.float32)
        for index, joint in enumerate(entity.get_active_joints()):
            names.append(str(joint.get_name()))
            positions.append(float(qpos[index]))
    return {
        "joint_names": np.asarray(names, dtype="S96"),
        "joint_positions": np.asarray(positions, dtype=np.float32),
        "base_pose_world": _pose_matrix(env.robot.left_entity.get_root_pose()),
    }


def _augment_observation(env: Any, observation: dict[str, Any]) -> dict[str, Any]:
    observation["endpose"]["left_tcp_pose"] = np.asarray(
        env.robot.get_left_tcp_pose(), dtype=np.float32
    )
    observation["endpose"]["right_tcp_pose"] = np.asarray(
        env.robot.get_right_tcp_pose(), dtype=np.float32
    )
    observation["robot_state"] = _robot_state(env)
    return observation


def state_observation(env: Any, observation: dict[str, Any]) -> dict[str, Any]:
    observation = _augment_observation(env, observation)
    return {
        "endpose": {key: np.asarray(value).copy() for key, value in observation["endpose"].items()},
        "robot_state": {
            key: np.asarray(value).copy() for key, value in observation["robot_state"].items()
        },
    }


def model_observation(env: Any, observation: dict[str, Any]) -> dict[str, Any]:
    payload = state_observation(env, observation)
    payload["observation"] = {}
    for name in ("head_camera", "left_camera", "right_camera"):
        camera = observation["observation"][name]
        payload["observation"][name] = {
            key: np.asarray(camera[key]).copy()
            for key in ("rgb", "depth", "intrinsic_cv", "extrinsic_cv")
        }
    return payload


def _load_environment(task_name: str, task_config: str):
    task_module = importlib.import_module(f"envs.{task_name}")
    task = getattr(task_module, task_name)()
    with open(f"task_config/{task_config}.yml", "r", encoding="utf-8") as handle:
        args = yaml.safe_load(handle)
    args["task_name"] = task_name
    args["task_config"] = task_config
    args["eval_mode"] = True
    args["render_freq"] = 0
    args["policy_name"] = "ActionFlowHTTP"
    args["ckpt_setting"] = "http"
    args["eval_video_log"] = False
    args["data_type"].update(rgb=True, depth=True, endpose=True, qpos=True)
    with open("task_config/_embodiment_config.yml", "r", encoding="utf-8") as handle:
        embodiment_types = yaml.safe_load(handle)
    with open("task_config/_camera_config.yml", "r", encoding="utf-8") as handle:
        camera_types = yaml.safe_load(handle)
    embodiment = args["embodiment"]
    robot_files = [embodiment_types[name]["file_path"] for name in embodiment[:2]]
    if len(embodiment) == 1:
        robot_files = robot_files * 2
        args["dual_arm_embodied"] = True
    else:
        args["embodiment_dis"] = embodiment[2]
        args["dual_arm_embodied"] = False
    args["embodiment_name"] = "+".join(embodiment[:2])
    args["left_robot_file"], args["right_robot_file"] = robot_files
    with open(Path(robot_files[0]) / "config.yml", "r", encoding="utf-8") as handle:
        args["left_embodiment_config"] = yaml.safe_load(handle)
    with open(Path(robot_files[1]) / "config.yml", "r", encoding="utf-8") as handle:
        args["right_embodiment_config"] = yaml.safe_load(handle)
    head = camera_types[args["camera"]["head_camera_type"]]
    args["head_camera_h"], args["head_camera_w"] = (head["h"], head["w"])
    return (task, args)


def _expert_valid(
    env: Any, setup_args: dict[str, Any], episode: int, seed: int
) -> tuple[bool, dict[str, Any] | None]:
    try:
        env.setup_demo(now_ep_num=episode, seed=seed, is_test=True, **setup_args)
        episode_info = env.play_once()
        return (bool(env.plan_success and env.check_success()), episode_info)
    finally:
        env.close_env()


def _episode_instruction(
    task_name: str,
    episode_info: dict[str, Any],
    instruction_type: str,
) -> str:
    """Use the official language protocol for every RoboTwin checkpoint."""
    info = episode_info.get("info") if isinstance(episode_info, dict) else None
    if not isinstance(info, dict):
        raise ValueError(f"expert episode for {task_name!r} did not return language metadata")
    from generate_episode_instructions import generate_episode_descriptions

    generated = generate_episode_descriptions(task_name, [info], 100)
    candidates = generated[0].get(instruction_type, []) if generated else []
    if not candidates:
        raise ValueError(f"RoboTwin generated no {instruction_type} instruction for {task_name!r}")
    return str(np.random.choice(candidates))


def _rollout(
    env: Any,
    client: PolicyClient,
    instruction: str,
    max_steps: int | None = None,
    observation_path: Path | None = None,
    video_writer=None,
) -> bool:
    env._update_render()
    env.actor_pose = True
    env._policy_native_failed = False
    observation = _augment_observation(env, env.get_obs())
    pending_history: list[dict[str, Any]] = []
    client.reset()
    step_limit = env.step_lim if max_steps is None else min(env.step_lim, int(max_steps))
    while env.take_action_cnt < step_limit and (not env.eval_success):
        payload = model_observation(env, observation)
        save_observation(
            observation_path,
            domain="robotwin",
            observation=payload,
            instruction=instruction,
            history=pending_history,
        )
        actions = client.act(payload, instruction, pending_history)
        pending_history = []
        for action in actions:
            if env.take_action_cnt >= step_limit or env.eval_success:
                break
            env.take_action(action, action_type="ee")
            observation = _augment_observation(env, env.get_obs())
            observation["endpose"]["left_endpose"] = action[:7].copy()
            observation["endpose"]["right_endpose"] = action[8:15].copy()
            pending_history.append(state_observation(env, observation))
            if video_writer is not None:
                video_writer.append_data(
                    np.asarray(observation["observation"]["head_camera"]["rgb"]).copy()
                )
            if env.check_success():
                env.eval_success = True
                break
    return bool(env.eval_success)


def evaluate_task(
    args: argparse.Namespace, task_name: str, client: PolicyClient
) -> list[dict[str, Any]]:
    from envs.utils.create_actor import UnStableError

    env, setup_args = _load_environment(task_name, args.task_config)
    results: list[dict[str, Any]] = []
    candidate_seed = (
        100000 * (1 + args.seed)
        if args.candidate_seed_start is None
        else int(args.candidate_seed_start)
    )
    output = Path(args.output_dir) / task_name
    output.mkdir(parents=True, exist_ok=True)
    episode = 0
    attempts = 0
    while episode < args.num_episodes:
        if attempts >= args.max_candidates_per_episode:
            raise RuntimeError(
                f"{task_name}: exhausted {attempts} candidate scenes for episode {episode}"
            )
        attempts += 1
        current_seed = candidate_seed
        candidate_seed += 1
        try:
            expert_valid, episode_info = _expert_valid(env, setup_args, episode, current_seed)
        except UnStableError as exc:
            logging.warning(
                "Skipping unstable expert scene task=%s seed=%s: %s", task_name, current_seed, exc
            )
            continue
        except Exception:
            logging.exception("Expert evaluation failed task=%s seed=%s", task_name, current_seed)
            raise
        if not expert_valid:
            logging.info("Skipping unsolved expert scene task=%s seed=%s", task_name, current_seed)
            continue
        stem = f"episode{episode:03d}_seed{current_seed}"
        with ExitStack() as resources:
            resources.callback(
                env.close_env, clear_cache=(episode + 2) % setup_args["clear_cache_freq"] == 0
            )
            try:
                env.setup_demo(now_ep_num=episode, seed=current_seed, is_test=True, **setup_args)
            except UnStableError as exc:
                logging.warning(
                    "Skipping unstable policy scene task=%s seed=%s: %s",
                    task_name,
                    current_seed,
                    exc,
                )
                continue
            instruction_type = args.instruction_type or (
                "unseen" if args.task_config == "demo_randomized" else "seen"
            )
            instruction = _episode_instruction(task_name, episode_info, instruction_type)
            env.set_instruction(instruction=instruction)
            writer = (
                resources.enter_context(
                    imageio.get_writer(
                        output / f"{stem}.mp4", format="FFMPEG", mode="I", fps=30, codec="libx264"
                    )
                )
                if args.save_videos
                else None
            )
            success = _rollout(
                env,
                client,
                instruction,
                max_steps=args.max_steps,
                observation_path=args.save_observation,
                video_writer=writer,
            )
            result = {
                "task": task_name,
                "episode": episode,
                "seed": current_seed,
                "instruction": instruction,
                "instruction_type": instruction_type,
                "protocol": "official",
                "success": success,
                "steps": int(env.take_action_cnt),
                "candidates": attempts,
            }
        if args.save_videos and (output / f"{stem}.mp4").exists():
            (output / f"{stem}.mp4").replace(output / f"{stem}_{int(success)}.mp4")
        write_json(output / f"{stem}_{int(success)}.json", result)
        results.append(result)
        episode += 1
        attempts = 0
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--task-name", default="adjust_bottle")
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--instruction-type", choices=["seen", "unseen"])
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0, help="0 uses the task step limit")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--candidate-seed-start",
        type=int,
        default=None,
        help="Start from this exact environment seed instead of deriving it from --seed",
    )
    parser.add_argument("--output-dir", default="outputs/robotwin_http_eval")
    parser.add_argument("--save-observation", type=Path)
    parser.add_argument("--max-candidates-per-episode", type=int, default=1000)
    parser.add_argument("--no-save-videos", dest="save_videos", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.max_candidates_per_episode <= 0 or args.num_episodes <= 0:
        raise ValueError("Episode and candidate limits must be positive")
    if args.max_steps == 0:
        args.max_steps = None
    root = args.robotwin_root.resolve()
    if not (root / "envs").is_dir():
        raise FileNotFoundError(f"not a RoboTwin checkout: {root}")
    args.output_dir = str(Path(args.output_dir).resolve())
    if args.save_observation:
        args.save_observation = args.save_observation.resolve()
    sys.path[:0] = [str(root), str(root / "script"), str(root / "description/utils")]
    os.chdir(root)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tasks = ALL_TASKS if args.task_name == "all" else (args.task_name,)
    with PolicyClient(args.host, args.port, domain="robotwin") as client:
        results = [result for task in tasks for result in evaluate_task(args, task, client)]
    successes = sum((result["success"] for result in results))
    summary = {
        "tasks": list(tasks),
        "task_config": args.task_config,
        "protocol": "official",
        "successes": successes,
        "episodes": len(results),
        "success_rate": successes / len(results) if results else 0,
    }
    write_json(Path(args.output_dir) / "summary.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
