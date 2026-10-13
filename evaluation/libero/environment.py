"""Shared LIBERO simulator observations and absolute-control setup."""

from __future__ import annotations

from typing import Any
import numpy as np
from scipy.spatial.transform import Rotation

CAMERAS = ("agentview", "robot0_eye_in_hand")
SETTLE_ACTION = np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)


def _matrix(position: Any, rotation: Any) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    transform[:3, 3] = np.asarray(position, dtype=np.float32).reshape(3)
    return transform


def _body_matrix(env: Any, body_name: str) -> np.ndarray:
    body_id = int(env.sim.model.body_name2id(body_name))
    return _matrix(env.sim.data.body_xpos[body_id], env.sim.data.body_xmat[body_id])


def _robot_base_matrix(env: Any) -> np.ndarray:
    for name in ("robot0_link0", "robot0_base"):
        try:
            return _body_matrix(env, name)
        except (KeyError, ValueError):
            continue
    raise KeyError("LIBERO model has no robot0_link0 or robot0_base body")


def _gripper_root_matrix(env: Any) -> np.ndarray:
    robot = env.env.robots[0]
    names = (robot.gripper.root_body, "gripper0_right_gripper", "robot0_right_gripper")
    for name in names:
        try:
            return _body_matrix(env, name)
        except (KeyError, ValueError):
            continue
    raise KeyError("LIBERO model has no Panda gripper-root body")


def _observed_tcp_matrix(observation: dict[str, Any]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = Rotation.from_quat(
        np.asarray(observation["robot0_eef_quat"], dtype=np.float32)
    ).as_matrix()
    transform[:3, 3] = np.asarray(observation["robot0_eef_pos"], dtype=np.float32)
    return transform


def _controller_tcp_matrix(env: Any) -> np.ndarray:
    controller = env.env.robots[0].controller
    return _matrix(controller.ee_pos, controller.ee_ori_mat)


def _render_camera(env: Any, camera_name: str) -> dict[str, np.ndarray]:
    from robosuite.utils.camera_utils import (
        get_camera_extrinsic_matrix,
        get_camera_intrinsic_matrix,
        get_real_depth_map,
    )

    height = width = 256
    rgb, depth = env.sim.render(camera_name=camera_name, height=height, width=width, depth=True)
    return {
        "rgb": rgb[::-1].copy().astype(np.uint8),
        "depth": get_real_depth_map(env.sim, depth[::-1].copy()).astype(np.float32),
        "intrinsic": get_camera_intrinsic_matrix(env.sim, camera_name, height, width).astype(
            np.float32
        ),
        "camera_to_world": get_camera_extrinsic_matrix(env.sim, camera_name).astype(np.float32),
    }


def state_observation(env: Any, observation: dict[str, Any]) -> dict[str, Any]:
    return {
        "robot0_joint_pos": np.asarray(observation["robot0_joint_pos"], dtype=np.float32),
        "robot0_gripper_qpos": np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32),
        "world_from_robot_base": _robot_base_matrix(env),
        "observed_tcp_world": _observed_tcp_matrix(observation),
        "controller_tcp_world": _controller_tcp_matrix(env),
        "gripper_root_world": _gripper_root_matrix(env),
    }


def model_observation(env: Any, observation: dict[str, Any]) -> dict[str, Any]:
    payload = state_observation(env, observation)
    payload["cameras"] = {name: _render_camera(env, name) for name in CAMERAS}
    return payload


def settle_then_enable_absolute_control(
    env: Any, observation: dict[str, Any], settle_steps: int
) -> tuple[dict[str, Any], float]:
    """Settle with delta no-ops, then switch to absolute OSC like X-VLA."""
    controllers = [robot.controller for robot in env.env.robots]
    for controller in controllers:
        controller.use_delta = True
    initial_eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float64).copy()
    for _ in range(int(settle_steps)):
        observation, _reward, _done, _info = env.step(SETTLE_ACTION)
    settled_eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float64).copy()
    for controller in controllers:
        controller.use_delta = False
    displacement = float(np.linalg.norm(settled_eef - initial_eef))
    return (observation, displacement)
