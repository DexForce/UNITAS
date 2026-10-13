#!/usr/bin/env python3
"""Run one resumable shard of the official LIBERO-Plus benchmark."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
import traceback
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np


from unitas.inference.client import PolicyClient
from unitas.inference.observations import save_observation
from evaluation.results import record_protocol, write_json, read_result, protocol_id
from evaluation.libero_plus.protocol import select_jobs
from evaluation.libero.environment import (
    CAMERAS,
    settle_then_enable_absolute_control,
    state_observation,
)


# Official LIBERO / LIBERO-Plus rollout limits, excluding the 10 settling steps.
OFFICIAL_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}


def _camera_observation(
    env: Any, observation: dict[str, Any], camera_name: str
) -> dict[str, np.ndarray]:
    """Return training-oriented RGB-D while preserving Plus RGB perturbations.

    LIBERO-Plus applies sensor noise in its environment wrapper after rendering.
    Calling ``sim.render`` again would silently bypass that perturbation, so RGB
    must come from the observation returned by ``reset`` / ``step``. Depth and
    calibration still describe the exact same current simulator state.
    """
    from robosuite.utils.camera_utils import (
        get_camera_extrinsic_matrix,
        get_camera_intrinsic_matrix,
        get_real_depth_map,
    )

    image_key = f"{camera_name}_image"
    depth_key = f"{camera_name}_depth"
    rgb = np.asarray(observation[image_key], dtype=np.uint8)[::-1].copy()
    if depth_key in observation:
        normalized_depth = np.asarray(observation[depth_key])[::-1].copy()
        if normalized_depth.ndim == 3 and normalized_depth.shape[-1] == 1:
            normalized_depth = normalized_depth[..., 0]
        if normalized_depth.ndim != 2:
            raise ValueError(f"unexpected {depth_key} shape: {normalized_depth.shape}")
    else:
        normalized_depth = env.sim.render(
            camera_name=camera_name,
            height=rgb.shape[0],
            width=rgb.shape[1],
            depth=True,
        )[1][::-1].copy()
    return {
        "rgb": rgb,
        "depth": get_real_depth_map(env.sim, normalized_depth).astype(np.float32),
        "intrinsic": get_camera_intrinsic_matrix(
            env.sim, camera_name, rgb.shape[0], rgb.shape[1]
        ).astype(np.float32),
        "camera_to_world": get_camera_extrinsic_matrix(env.sim, camera_name).astype(np.float32),
    }


def model_observation(env: Any, observation: dict[str, Any]) -> dict[str, Any]:
    payload = state_observation(env, observation)
    payload["cameras"] = {name: _camera_observation(env, observation, name) for name in CAMERAS}
    return payload


def _load_classification(libero_plus_root: Path) -> dict[str, list[dict[str, Any]]]:
    path = libero_plus_root / "libero/libero/benchmark/task_classification.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_suites(names: list[str]) -> dict[str, Any]:
    from libero.libero import benchmark

    suites: dict[str, Any] = {}
    # The fork prints every task id when constructing a suite (thousands of ids).
    # Suppress that non-diagnostic output so shard logs stay usable.
    with contextlib.redirect_stdout(io.StringIO()):
        for name in names:
            suites[name] = benchmark.get_benchmark_dict()[name]()
    return suites


def _create_environment(suite: Any, task_id: int, seed: int):
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.utils import get_libero_path

    task = suite.get_task(task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=256,
        camera_widths=256,
        camera_depths=True,
        render_gpu_device_id=0,
    )
    # Match the official evaluator: every task environment receives the same seed.
    env.seed(seed)
    observation = env.reset()
    initial_states = suite.get_task_init_states(task_id)
    observation = env.set_init_state(initial_states[0])
    return env, observation, task


def _result_path(output_dir: Path, suite_name: str, task_id: int) -> Path:
    return output_dir / "episodes" / suite_name / f"task_{task_id:04d}.json"


def run_task(
    args: argparse.Namespace,
    suite_name: str,
    suite: Any,
    task_id: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    result_path = _result_path(output_dir, suite_name, task_id)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    client = PolicyClient(args.host, args.port, domain="libero")
    env = None
    frames: list[np.ndarray] = []
    steps = 0
    success = False
    settle_displacement = None
    task_name = str(metadata["name"])
    instruction = ""
    try:
        env, observation, task = _create_environment(suite, task_id, args.seed)
        task_name = str(task.name)
        instruction = str(task.language)
        client.reset()
        observation, settle_displacement = settle_then_enable_absolute_control(
            env, observation, args.settle_steps
        )
        pending_history: list[dict[str, Any]] = []
        max_steps = OFFICIAL_MAX_STEPS[suite_name]
        while steps < max_steps and not success:
            payload = model_observation(env, observation)
            if steps == 0:
                save_observation(
                    args.save_observation,
                    domain="libero",
                    observation=payload,
                    instruction=instruction,
                    history=[],
                )
            actions = client.act(payload, instruction, pending_history)
            pending_history = []
            for action in actions:
                if steps >= max_steps:
                    break
                if args.save_videos:
                    frames.append(
                        np.asarray(observation["agentview_image"], dtype=np.uint8)[::-1].copy()
                    )
                observation, _reward, _done, _info = env.step(action)
                pending_history.append(state_observation(env, observation))
                steps += 1
                success = bool(env.check_success())
                if success:
                    break
        result = {
            "status": "ok",
            "suite": suite_name,
            "task_id": int(task_id),
            "official_task_id": int(task_id + 1),
            "task_name": task_name,
            "instruction": instruction,
            "category": str(metadata["category"]),
            "difficulty_level": metadata.get("difficulty_level"),
            "success": bool(success),
            "steps": int(steps),
            "max_steps": int(OFFICIAL_MAX_STEPS[suite_name]),
            "settle_steps": int(args.settle_steps),
            "settle_eef_displacement_m": settle_displacement,
            "rgb_source": "official_environment_observation",
            "depth_source": "ground_truth_simulator_depth",
            "control": "absolute_tcp_measured_history_execute_10",
        }
    except Exception as exc:
        result = {
            "status": "error",
            "suite": suite_name,
            "task_id": int(task_id),
            "official_task_id": int(task_id + 1),
            "task_name": task_name,
            "instruction": instruction,
            "category": str(metadata["category"]),
            "difficulty_level": metadata.get("difficulty_level"),
            "success": False,
            "steps": int(steps),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        client.close()
        if env is not None:
            env.close()

    result["protocol_id"] = args.protocol_id
    write_json(result_path, result)
    if args.save_videos and frames:
        video_path = result_path.with_suffix(".mp4")
        imageio.mimsave(video_path, frames, fps=30, codec="libx264")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--libero-plus-root", type=Path, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--suites",
        default="libero_spatial,libero_object,libero_goal,libero_10",
    )
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--save-observation", type=Path)
    parser.add_argument("--rerun-errors", action="store_true")
    parser.add_argument(
        "--task-ids",
        default=None,
        help="Optional comma-separated task ids for targeted smoke/debug runs",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=0,
        help="Limit this shard for smoke testing; zero runs the full shard",
    )
    parser.add_argument(
        "--sample-tasks",
        type=int,
        default=0,
        help="Deterministically sample this many jobs before sharding; zero disables sampling",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=160484,
        help="Independent seed used by --sample-tasks",
    )
    parser.add_argument(
        "--categories",
        default=None,
        help="Optional comma-separated exact task-classification categories",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("--shard-id must be in [0, --num-shards)")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    root = Path(args.libero_plus_root)
    classifications = _load_classification(root)
    suite_names = [name.strip() for name in args.suites.split(",") if name.strip()]
    with PolicyClient(args.host, args.port, domain="libero") as connection:
        health = connection.health()
    if health.get("domain") != "libero":
        raise ValueError("The server must load a LIBERO policy")
    suites = _load_suites(suite_names)
    for name, suite in suites.items():
        if len(classifications[name]) != int(suite.n_tasks):
            raise RuntimeError(f"Classification/task count mismatch for {name}")
    categories = (
        sorted({v.strip() for v in args.categories.split(",") if v.strip()})
        if args.categories is not None
        else None
    )
    task_ids = (
        sorted({int(v.strip()) for v in args.task_ids.split(",") if v.strip()})
        if args.task_ids is not None
        else None
    )
    jobs = select_jobs(
        classifications,
        suite_names,
        categories=categories,
        task_ids=task_ids,
        sample_tasks=args.sample_tasks,
        sample_seed=args.sample_seed,
        max_tasks=args.max_tasks,
        num_shards=args.num_shards,
    )
    protocol = {
        "server": health.get("protocol", {}),
        "benchmark": "LIBERO-Plus",
        "client_seed": args.seed,
        "settle_steps": args.settle_steps,
        "suites": suite_names,
        "max_steps": OFFICIAL_MAX_STEPS,
        "classification_sha256": protocol_id(classifications),
        "selection": {
            "categories": categories,
            "task_ids": task_ids,
            "sample_tasks": args.sample_tasks,
            "sample_seed": args.sample_seed,
            "max_tasks_per_shard": args.max_tasks,
            "num_shards": args.num_shards,
        },
        "jobs": [{"suite": name, "task_id": task_id} for name, task_id, _ in jobs],
    }
    from unitas.inference.runtime import file_sha256

    protocol["client_sha256"] = file_sha256(Path(__file__))
    args.protocol_id = record_protocol(
        Path(args.output_dir) / "run_config.json", protocol, result_glob="episodes/*/*.json"
    )
    shard_jobs = [job for index, job in enumerate(jobs) if index % args.num_shards == args.shard_id]

    completed = skipped = errors = successes = 0
    for suite_name, task_id, metadata in shard_jobs:
        path = _result_path(Path(args.output_dir), suite_name, task_id)
        if path.exists():
            old = read_result(path, args.protocol_id, suite=suite_name, task_id=task_id)
            if old.get("status") not in {"ok", "error"}:
                raise ValueError(f"Invalid episode status: {path}")
            if old["status"] == "ok" or not args.rerun_errors:
                skipped += 1
                continue
        result = run_task(args, suite_name, suites[suite_name], task_id, metadata)
        completed += 1
        errors += int(result["status"] == "error")
        successes += int(result["status"] == "ok" and result["success"])
        print(
            f"[shard {args.shard_id}/{args.num_shards}] completed={completed} "
            f"skipped={skipped} success={successes} errors={errors}",
            flush=True,
        )


if __name__ == "__main__":
    main()
