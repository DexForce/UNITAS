"""Evaluate the UNITAS HTTP server with the official LIBERO environment."""

from __future__ import annotations
import argparse
import copy
from contextlib import ExitStack
import json
import random
import hashlib
from pathlib import Path
from typing import Any
import imageio.v2 as imageio
import numpy as np
from unitas.inference.client import PolicyClient
from unitas.inference.observations import save_observation
from evaluation.results import record_protocol, write_json, protocol_id
from evaluation.libero.environment import (
    CAMERAS,
    model_observation,
    state_observation,
    settle_then_enable_absolute_control,
)

X_VLA_MAX_STEPS = {
    "libero_spatial": 800,
    "libero_object": 800,
    "libero_goal": 800,
    "libero_10": 900,
    "libero_90": 800,
}


def create_environment(suite_name: str, task_id: int, episode_id: int, seed: int):
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.utils import get_libero_path

    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=256,
        camera_widths=256,
        camera_depths=True,
        render_gpu_device_id=0,
    )
    try:
        env.seed(seed + episode_id + 100)
        observation = env.reset()
        initial_states = suite.get_task_init_states(task_id)
        observation = env.set_init_state(initial_states[episode_id % len(initial_states)])
    except BaseException:
        env.close()
        raise
    return (env, observation, str(task.language), str(task.name))


def run_episode(args: argparse.Namespace, episode_id: int) -> dict[str, Any]:
    env, observation, instruction, task_name = create_environment(
        args.suite, args.task_id, episode_id, args.seed
    )
    pending_history: list[dict[str, Any]] = []
    steps = 0
    success = False
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{args.suite}_task{args.task_id:02d}_episode{episode_id:03d}"
    with ExitStack() as resources:
        resources.callback(env.close)
        client = resources.enter_context(PolicyClient(args.host, args.port, domain="libero"))
        if client.health().get("protocol") != args.protocol["server"]:
            raise ValueError("The server protocol changed during evaluation")
        client.reset()
        writer = (
            resources.enter_context(
                imageio.get_writer(
                    output / f"{stem}.mp4", format="FFMPEG", mode="I", fps=30, codec="libx264"
                )
            )
            if args.save_videos
            else None
        )
        observation, settle_eef_displacement_m = settle_then_enable_absolute_control(
            env, observation, args.settle_steps
        )
        while steps < args.max_steps and (not success):
            payload = model_observation(env, observation)
            save_observation(
                args.save_observation,
                domain="libero",
                observation=payload,
                instruction=instruction,
                history=pending_history,
            )
            actions = client.act(payload, instruction, pending_history)
            pending_history = []
            for action in actions:
                if steps >= args.max_steps:
                    break
                if writer is not None:
                    writer.append_data(np.asarray(observation["agentview_image"])[::-1].copy())
                observation, _reward, done, _info = env.step(action)
                pending_history.append(state_observation(env, observation))
                steps += 1
                success = bool(env.check_success())
                if done or success:
                    break
    result = {
        "suite": args.suite,
        "task_id": args.task_id,
        "task_name": task_name,
        "episode_id": episode_id,
        "instruction": instruction,
        "success": success,
        "steps": steps,
        "max_steps": int(args.max_steps),
        "settle_steps": int(args.settle_steps),
        "settle_controller_mode": "delta_then_absolute",
        "settle_eef_displacement_m": settle_eef_displacement_m,
        "protocol_id": protocol_id(args.protocol),
    }
    write_json(output / f"{stem}.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument(
        "--tasks",
        default=None,
        help="Task selection: 'all' or a comma-separated list of task ids; defaults to --task-id",
    )
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=None,
        help="Episodes for every selected task (suite-level evaluation)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Rollout horizon; defaults to the X-VLA value for the selected suite",
    )
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/libero_http_eval")
    parser.add_argument(
        "--no-save-videos",
        dest="save_videos",
        action="store_false",
        help="Skip MP4 recording; episode JSON metrics are still saved",
    )
    parser.set_defaults(save_videos=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse an existing valid episode JSON instead of rerunning it",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="Zero-based shard selected from the flattened task/episode job list",
    )
    parser.add_argument(
        "--num-shards", type=int, default=1, help="Number of disjoint episode shards"
    )
    parser.add_argument("--save-observation", type=Path)
    return parser.parse_args()


def episode_result_path(args: argparse.Namespace, task_id: int, episode_id: int) -> Path:
    stem = f"{args.suite}_task{task_id:02d}_episode{episode_id:03d}"
    return Path(args.output_dir) / f"{stem}.json"


def load_completed_result(
    args: argparse.Namespace, task_id: int, episode_id: int
) -> dict[str, Any] | None:
    if not args.resume:
        return None
    path = episode_result_path(args, task_id, episode_id)
    if not path.is_file():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    required = {"suite", "task_id", "episode_id", "success", "steps"}
    if not required.issubset(result):
        return None
    if not hasattr(args, "protocol") or result.get("protocol_id") != protocol_id(args.protocol):
        raise ValueError(f"Episode result has a missing or different evaluation protocol: {path}")
    if (
        not isinstance(result["success"], bool)
        or not isinstance(result["steps"], int)
        or result["steps"] < 0
    ):
        return None
    if (
        result["suite"] != args.suite
        or int(result["task_id"]) != int(task_id)
        or int(result["episode_id"]) != int(episode_id)
    ):
        return None
    return result


def selected_task_ids(args: argparse.Namespace) -> list[int]:
    if args.tasks is None:
        return [int(args.task_id)]
    if str(args.tasks).strip().lower() == "all":
        from libero.libero import benchmark

        suite = benchmark.get_benchmark_dict()[args.suite]()
        return list(range(int(suite.n_tasks)))
    task_ids = [int(value.strip()) for value in str(args.tasks).split(",") if value.strip()]
    if not task_ids:
        raise ValueError("--tasks must be 'all' or a non-empty comma-separated list")
    return task_ids


def selected_episode_jobs(
    task_ids: list[int], episode_start: int, episodes_per_task: int, shard_id: int, num_shards: int
) -> list[tuple[int, int]]:
    """Return a deterministic, disjoint modulo shard of task/episode pairs."""
    jobs = [
        (task_id, episode_id)
        for task_id in task_ids
        for episode_id in range(episode_start, episode_start + episodes_per_task)
    ]
    return [job for index, job in enumerate(jobs) if index % num_shards == shard_id]


def main() -> None:
    args = parse_args()
    import torch

    if args.num_shards <= 0 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("--shard-id must be in [0, --num-shards)")
    if args.max_steps is None:
        try:
            args.max_steps = X_VLA_MAX_STEPS[args.suite]
        except KeyError as exc:
            raise ValueError(
                f"No X-VLA rollout horizon is defined for suite {args.suite!r}; pass --max-steps explicitly"
            ) from exc
    if int(args.max_steps) <= 0:
        raise ValueError("--max-steps must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    task_ids = selected_task_ids(args)
    episodes_per_task = (
        int(args.num_episodes) if args.episodes_per_task is None else int(args.episodes_per_task)
    )
    if episodes_per_task <= 0:
        raise ValueError("episode count must be positive")
    jobs = selected_episode_jobs(
        task_ids,
        int(args.episode_start),
        episodes_per_task,
        int(args.shard_id),
        int(args.num_shards),
    )
    if not jobs:
        raise ValueError("this shard has no episodes; reduce --num-shards or select more episodes")
    with PolicyClient(args.host, args.port, domain="libero") as connection:
        health = connection.health()
    if health.get("domain") != "libero" or not health.get("protocol", {}).get("checkpoint_sha256"):
        raise ValueError("The server must identify the LIBERO checkpoint by SHA256")
    args.protocol = {
        "benchmark": "LIBERO",
        "suite": args.suite,
        "server": health["protocol"],
        "client_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": args.seed,
        "max_steps": args.max_steps,
        "settle_steps": args.settle_steps,
        "control": "delta_then_absolute",
        "cameras": list(CAMERAS),
        "resolution": [256, 256],
    }
    protocol_path = Path(args.output_dir) / f"{args.suite}_protocol.json"
    if not protocol_path.exists() and any(
        Path(args.output_dir).glob(f"{args.suite}_task*_episode*.json")
    ):
        raise ValueError("Output contains results without a protocol; use a new output directory")
    record_protocol(protocol_path, args.protocol)
    results = []
    for task_id in task_ids:
        episode_ids = [episode_id for job_task_id, episode_id in jobs if job_task_id == task_id]
        if not episode_ids:
            continue
        task_args = copy.copy(args)
        task_args.task_id = int(task_id)
        task_results = []
        for episode_id in episode_ids:
            result = load_completed_result(task_args, task_id, episode_id)
            if result is None:
                result = run_episode(task_args, episode_id)
            task_results.append(result)
        results.extend(task_results)
    success_rate = float(np.mean([result["success"] for result in results]))
    output = Path(args.output_dir)
    summary = {
        "suite": args.suite,
        "task_ids": task_ids,
        "episodes_per_task": episodes_per_task,
        "shard_id": int(args.shard_id),
        "num_shards": int(args.num_shards),
        "max_steps": int(args.max_steps),
        "settle_steps": int(args.settle_steps),
        "settle_controller_mode": "delta_then_absolute",
        "successes": int(sum((result["success"] for result in results))),
        "rollouts": len(results),
        "success_rate": success_rate,
    }
    summary_name = (
        f"{args.suite}_summary.json"
        if args.num_shards == 1
        else f"{args.suite}_summary_shard{args.shard_id:03d}.json"
    )
    summary["protocol_id"] = protocol_id(args.protocol)
    write_json(output / summary_name, summary)


if __name__ == "__main__":
    main()
