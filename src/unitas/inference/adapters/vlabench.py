"""Online inputs and controls for VLABench evaluation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from unitas.models.config import (
    as_bool as _as_bool,
)
from unitas.models.config import (
    temporal_spec_for_domain,
)
from unitas.inference.preprocessing import camera_payloads, pack_observation, padded_history
from unitas.inference.robots.frames import EEF_POSE_CONVENTION
from unitas.inference.robots.transforms import as_homogeneous as _matrix
from unitas.inference.robots.transforms import matrix_to_pose_xyzw, pose_xyzw_to_matrix
from unitas.inference.robots.vlabench import (
    CAMERA_LAYOUT,
    VLABENCH_JOINT_NAMES,
    build_vlabench_robot_flows,
    measured_vlabench_hand_poses,
)

VLABENCH_RIGHT_ARM_INDEX = 1
VLABENCH_MODEL_FRAME = "robot_base_at_episode_frame_0"

# The official Franka MJCF trajectory stores an EE-site pose. Training rotates
# that pose by Rz(-pi/2) to obtain the canonical TCP orientation.
VLABENCH_EE_TO_TCP = np.asarray(
    [
        [0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class VLABenchHistoryFrame:
    joint_positions: np.ndarray
    tcp_pose_base: np.ndarray
    hand_pose_base: np.ndarray
    finger_open_m: float


class VLABenchOnlineBatchBuilder:
    """Prepare observed VLABench state for policy inference."""

    def __init__(self, args: Any, *, num_gripper_tokens: int, seed: int = 0) -> None:
        self.args = args
        self.temporal_spec = temporal_spec_for_domain(args, "vlabench")
        self.horizon = int(self.temporal_spec["prediction_horizon"])
        if int(self.temporal_spec["history_horizon"]) != self.horizon:
            raise ValueError("VLABench online history and prediction horizons must match")
        self.num_gripper_tokens = int(num_gripper_tokens)
        if self.num_gripper_tokens != 20:
            raise ValueError(
                "VLABench uses the fixed 20-point Panda surface asset; "
                f"checkpoint requests {self.num_gripper_tokens} tokens"
            )
        self.seed = int(seed)
        self.camera_layout = CAMERA_LAYOUT
        self.history: deque[VLABenchHistoryFrame] = deque(maxlen=self.horizon + 1)

    def reset(self) -> None:
        self.history.clear()

    def capture(self, observation: dict[str, Any]) -> VLABenchHistoryFrame:
        world_from_base = _matrix(observation["world_from_robot_base"])
        base_from_world = np.linalg.inv(world_from_base)
        ee_base = base_from_world @ _matrix(observation["observed_ee_world"])
        tcp_base = ee_base @ VLABENCH_EE_TO_TCP
        hand_base = measured_vlabench_hand_poses(ee_base[None].astype(np.float32))[0]
        joints = np.asarray(observation["joint_positions"], dtype=np.float32).reshape(-1)
        if joints.shape != (9,):
            raise ValueError(f"VLABench joint_positions must be [9], got {joints.shape}")
        return VLABenchHistoryFrame(
            joint_positions=joints.copy(),
            tcp_pose_base=matrix_to_pose_xyzw(tcp_base),
            hand_pose_base=hand_base.astype(np.float32),
            finger_open_m=float(np.clip(np.mean(joints[-2:]), 0.0, 0.04)),
        )

    def update_history(self, observation: dict[str, Any]) -> VLABenchHistoryFrame:
        frame = self.capture(observation)
        self.history.append(frame)
        return frame

    def build_batch(self, observation: dict[str, Any], instruction: str) -> dict[str, Any]:
        """Prepare the measured history and cameras consumed by the policy."""
        current = self.capture(observation)
        if not self.history:
            self.history.append(current)
        history, valid = padded_history(self.history, current, self.horizon)
        sample = {
            "joint_positions": np.stack([frame.joint_positions for frame in history]),
            "joint_names": VLABENCH_JOINT_NAMES,
            "gripper_root_pose": np.stack([frame.hand_pose_base for frame in history]),
        }
        robot = build_vlabench_robot_flows(sample, self.num_gripper_tokens)
        opening = np.asarray([frame.finger_open_m for frame in history], dtype=np.float32) / 0.04
        cameras = camera_payloads(observation, self.camera_layout)
        return pack_observation(
            robot_history=robot["robot_flows"],
            eef_history=np.stack([frame.tcp_pose_base for frame in history]),
            eef_open=opening,
            valid=valid,
            cameras=cameras,
            domain="vlabench",
            robot_type="franka_panda",
            instruction=instruction,
            convention=EEF_POSE_CONVENTION,
            temporal=self.temporal_spec,
            image_size=int(getattr(self.args, "action_flow_vlm_image_size", 224) or 224),
        )


def vlabench_predictions_to_actions(
    poses: Any,
    opens: Any,
    *,
    world_from_robot_base: Any,
    chunk_size: int,
) -> np.ndarray:
    """Model ``base TCP + open`` -> official ``world EE xyz/euler + fingers``."""
    poses = np.asarray(poses, dtype=np.float64)
    opens = np.asarray(opens, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (2, 7):
        raise ValueError(f"predicted EEF poses must be [H,2,7], got {poses.shape}")
    if opens.shape == poses.shape[:2]:
        opens = opens[..., None]
    if opens.shape != (poses.shape[0], 2, 1):
        raise ValueError(f"predicted EEF open must be [H,2,1], got {opens.shape}")
    count = min(int(chunk_size), len(poses))
    if count <= 0:
        raise ValueError("chunk_size must select at least one action")
    world_from_base = _matrix(world_from_robot_base)
    tcp_to_ee = np.linalg.inv(VLABENCH_EE_TO_TCP.astype(np.float64))
    actions = []
    for pose, open_probability in zip(
        poses[:count, VLABENCH_RIGHT_ARM_INDEX],
        opens[:count, VLABENCH_RIGHT_ARM_INDEX, 0],
    ):
        world_ee = world_from_base @ pose_xyzw_to_matrix(pose) @ tcp_to_ee
        euler = Rotation.from_matrix(world_ee[:3, :3]).as_euler("xyz")
        # Preserve the continuous 0..0.04 supervision used in trajectory[T,8].
        fingers = np.full(2, np.clip(open_probability, 0.0, 1.0) * 0.04)
        actions.append(np.concatenate([world_ee[:3, 3], euler, fingers]))
    return np.asarray(actions, dtype=np.float32)


def validate_vlabench_checkpoint_args(raw_args: Any) -> dict[str, Any]:
    args = SimpleNamespace(**raw_args) if isinstance(raw_args, dict) else raw_args
    if not _as_bool(getattr(args, "disable_center_shift", True)):
        raise ValueError(
            "checkpoint used an untracked center shift and is unsafe for absolute control"
        )
    spec = temporal_spec_for_domain(args, "vlabench")
    return {
        "coordinate_frame": VLABENCH_MODEL_FRAME,
        "prediction_horizon": int(spec["prediction_horizon"]),
        "control_frequency_hz": float(spec["gripper_flow_fps"]),
        "action_supervision": "trajectory[t]_absolute_robot_base_tcp+continuous_fingers",
    }
