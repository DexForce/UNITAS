"""Online inputs and controls for UNITAS LIBERO evaluation."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
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
from unitas.inference.robots.assets import resolve_robot_urdf
from unitas.inference.robots.frames import (
    EEF_POSE_CONVENTION,
    canonicalize_libero_observed_tcp_matrices,
)
from unitas.inference.robots.libero import (
    CAMERA_LAYOUT,
    LIBERO_JOINT_NAMES,
    build_libero_robot_flows,
)
from unitas.inference.robots.sampler import RobotSampler as TorchRobotSampler
from unitas.inference.robots.transforms import as_homogeneous as _as_homogeneous
from unitas.inference.robots.transforms import matrix_to_pose_xyzw, pose_xyzw_to_matrix

LIBERO_RIGHT_ARM_INDEX = 1
LIBERO_MODEL_EEF_CONVENTION = "robot_base_initial:xyz+quaternion_xyzw"
LIBERO_OSC_ABSOLUTE_ACTION_CONVENTION = "world:xyz+axis_angle+gripper(-1=open,+1=close)"


def matrix_to_libero_absolute_action(world_from_tcp: Any, gripper_control: float) -> np.ndarray:
    """Build the 7-D action accepted by absolute LIBERO OSC_POSE control."""
    transform = _as_homogeneous(world_from_tcp)
    axis_angle = Rotation.from_matrix(transform[:3, :3]).as_rotvec()
    return np.concatenate(
        (transform[:3, 3], axis_angle, np.asarray([gripper_control], dtype=np.float64))
    ).astype(np.float32)


def libero_gripper_control(open_probability: float, *, threshold: float = 0.5) -> float:
    """Map ActionFlow's ``1=open`` probability to LIBERO's signed command."""
    value = float(open_probability)
    if not np.isfinite(value):
        raise ValueError("gripper open probability must be finite")
    return -1.0 if value >= float(threshold) else 1.0


@dataclass(frozen=True)
class LiberoActionChunk:
    actions: np.ndarray
    model_tcp_robot_base: np.ndarray
    controller_tcp_world: np.ndarray
    open_probability: np.ndarray


class LiberoEEFActionConverter:
    """Convert UNITAS's right-arm EEF trajectory into LIBERO actions."""

    def __init__(
        self,
        *,
        open_threshold: float = 0.5,
        max_translation_step_m: float | None = 0.08,
        max_rotation_step_deg: float | None = 45.0,
        workspace_min: Sequence[float] | None = None,
        workspace_max: Sequence[float] | None = None,
    ) -> None:
        self.open_threshold = float(open_threshold)
        self.max_translation_step_m = (
            None if max_translation_step_m is None else float(max_translation_step_m)
        )
        self.max_rotation_step_rad = (
            None if max_rotation_step_deg is None else float(np.deg2rad(max_rotation_step_deg))
        )
        self.workspace_min = (
            None
            if workspace_min is None
            else np.asarray(workspace_min, dtype=np.float64).reshape(3)
        )
        self.workspace_max = (
            None
            if workspace_max is None
            else np.asarray(workspace_max, dtype=np.float64).reshape(3)
        )
        if (self.workspace_min is None) != (self.workspace_max is None):
            raise ValueError("workspace_min and workspace_max must be provided together")
        if self.workspace_min is not None and np.any(self.workspace_min >= self.workspace_max):
            raise ValueError("workspace_min must be smaller than workspace_max")
        self.last_safety_events: list[dict[str, Any]] = []

    def _is_safe(self, candidate: np.ndarray, previous: np.ndarray) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if not np.isfinite(candidate).all():
            reasons.append("non_finite")
            return (False, reasons)
        if self.workspace_min is not None and (
            np.any(candidate[:3, 3] < self.workspace_min)
            or np.any(candidate[:3, 3] > self.workspace_max)
        ):
            reasons.append("workspace")
        if self.max_translation_step_m is not None:
            distance = float(np.linalg.norm(candidate[:3, 3] - previous[:3, 3]))
            if distance > self.max_translation_step_m:
                reasons.append(f"translation_step_m={distance:.6f}")
        if self.max_rotation_step_rad is not None:
            relative = previous[:3, :3].T @ candidate[:3, :3]
            angle = float(Rotation.from_matrix(relative).magnitude())
            if angle > self.max_rotation_step_rad:
                reasons.append(f"rotation_step_deg={np.rad2deg(angle):.3f}")
        return (not reasons, reasons)

    def trajectory_to_actions(
        self,
        pred_eef_poses: Any,
        pred_eef_open: Any,
        *,
        world_from_robot_base: Any,
        current_tcp_world: Any,
        current_open_probability: float,
        chunk_size: int | None = None,
    ) -> LiberoActionChunk:
        """Convert model outputs ``[H,2,7]`` / ``[H,2,1]`` into ``[K,7]``."""
        poses = np.asarray(pred_eef_poses, dtype=np.float64)
        opens = np.asarray(pred_eef_open, dtype=np.float64)
        if poses.ndim != 3 or poses.shape[1:] != (2, 7):
            raise ValueError(f"pred_eef_poses must have shape [H,2,7], got {poses.shape}")
        if opens.shape == (poses.shape[0], 2):
            opens = opens[..., None]
        if opens.shape != (poses.shape[0], 2, 1):
            raise ValueError(f"pred_eef_open must have shape [H,2,1], got {opens.shape}")
        horizon = poses.shape[0] if chunk_size is None else min(poses.shape[0], int(chunk_size))
        if horizon <= 0:
            raise ValueError("chunk_size must select at least one predicted frame")
        base = _as_homogeneous(world_from_robot_base)
        previous = _as_homogeneous(current_tcp_world).copy()
        previous_gripper = libero_gripper_control(
            current_open_probability, threshold=self.open_threshold
        )
        self.last_safety_events = []
        actions = []
        world_poses = []
        selected_poses = poses[:horizon, LIBERO_RIGHT_ARM_INDEX]
        selected_opens = opens[:horizon, LIBERO_RIGHT_ARM_INDEX, 0]
        for frame_index, (pose, open_probability) in enumerate(zip(selected_poses, selected_opens)):
            try:
                candidate = base @ pose_xyzw_to_matrix(pose)
                safe, reasons = self._is_safe(candidate, previous)
            except ValueError as exc:
                candidate = previous.copy()
                safe, reasons = (False, [str(exc)])
            if safe:
                gripper = libero_gripper_control(open_probability, threshold=self.open_threshold)
                previous = candidate
                previous_gripper = gripper
            else:
                candidate = previous.copy()
                gripper = previous_gripper
                self.last_safety_events.append(
                    {"frame": int(frame_index), "reasons": reasons, "action": "hold"}
                )
            world_poses.append(candidate.astype(np.float32))
            actions.append(matrix_to_libero_absolute_action(candidate, gripper))
        return LiberoActionChunk(
            actions=np.stack(actions),
            model_tcp_robot_base=selected_poses.astype(np.float32),
            controller_tcp_world=np.stack(world_poses),
            open_probability=selected_opens.astype(np.float32),
        )


@dataclass(frozen=True)
class LiberoHistoryFrame:
    """Robot state stored for one closed-loop history timestep."""

    joint_positions: np.ndarray
    finger_open_m: float
    tcp_pose_base: np.ndarray
    gripper_root_pose_base: np.ndarray


class LiberoOnlineBatchBuilder:
    """Prepare measured LIBERO history and cameras for inference."""

    def __init__(
        self, args: Any, *, num_gripper_tokens: int, seed: int = 0, drop_depth: bool = False
    ) -> None:
        self.args = args
        self.temporal_spec = temporal_spec_for_domain(args, "libero")
        self.horizon = int(self.temporal_spec["prediction_horizon"])
        if int(self.temporal_spec["history_horizon"]) != self.horizon:
            raise ValueError("LIBERO online history and prediction horizons must match")
        self.seed = int(seed)
        self.drop_depth = bool(drop_depth)
        self.history: deque[LiberoHistoryFrame] = deque(maxlen=self.horizon + 1)
        self.robot_sampler = TorchRobotSampler(
            urdf_path=resolve_robot_urdf("libero"),
            gripper_only=True,
            device="cpu",
            link_whitelist=["panda_hand", "panda_leftfinger", "panda_rightfinger"],
        )
        self.num_gripper_tokens = int(num_gripper_tokens)
        if self.num_gripper_tokens <= 0:
            raise ValueError("num_gripper_tokens must be positive")

    def reset(self) -> None:
        self.history.clear()

    def capture(self, observation: dict[str, Any]) -> LiberoHistoryFrame:
        """Convert one client-provided simulator state into the training frame."""
        world_from_base = _as_homogeneous(observation["world_from_robot_base"])
        base_from_world = np.linalg.inv(world_from_base)
        tcp_base = base_from_world @ _as_homogeneous(observation["observed_tcp_world"])
        tcp_base = canonicalize_libero_observed_tcp_matrices(tcp_base)
        return LiberoHistoryFrame(
            joint_positions=np.asarray(observation["robot0_joint_pos"], dtype=np.float32).copy(),
            finger_open_m=float(
                np.mean(np.abs(np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32)))
            ),
            tcp_pose_base=matrix_to_pose_xyzw(tcp_base),
            gripper_root_pose_base=matrix_to_pose_xyzw(
                base_from_world @ _as_homogeneous(observation["gripper_root_world"])
            ),
        )

    def update_history(self, observation: dict[str, Any]) -> LiberoHistoryFrame:
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
            "joint_names": LIBERO_JOINT_NAMES,
            "right_gripper_open": np.asarray(
                [[frame.finger_open_m] for frame in history], dtype=np.float32
            ),
            "right_gripper_root_pose": np.stack(
                [frame.gripper_root_pose_base for frame in history]
            ),
        }
        robot = build_libero_robot_flows(
            sample, self.robot_sampler, self.num_gripper_tokens, seed=self.seed
        )
        opening = np.clip(sample["right_gripper_open"][:, 0] / 0.04, 0.0, 1.0)
        cameras = camera_payloads(observation, CAMERA_LAYOUT, drop_depth=self.drop_depth)
        return pack_observation(
            robot_history=robot["robot_flows"],
            eef_history=np.stack([frame.tcp_pose_base for frame in history]),
            eef_open=opening,
            valid=valid,
            cameras=cameras,
            domain="libero",
            robot_type="panda",
            instruction=instruction,
            convention=EEF_POSE_CONVENTION,
            temporal=self.temporal_spec,
            image_size=int(getattr(self.args, "action_flow_vlm_image_size", 224) or 224),
        )


def validate_libero_checkpoint_args(raw_args: Any) -> dict[str, Any]:
    """Reject checkpoint conventions that cannot be inverted online safely."""
    args = SimpleNamespace(**raw_args) if isinstance(raw_args, dict) else raw_args
    if not _as_bool(getattr(args, "disable_center_shift", True)):
        raise ValueError(
            "checkpoint used per-sample center_shift; the missing shift must be restored before LIBERO absolute control"
        )
    return {
        "model_eef_convention": LIBERO_MODEL_EEF_CONVENTION,
        "controller_action_convention": LIBERO_OSC_ABSOLUTE_ACTION_CONVENTION,
        "right_arm_index": LIBERO_RIGHT_ARM_INDEX,
        "prediction_horizon": int(getattr(args, "libero_action_flow_prediction_horizon", 30)),
        "control_frequency_hz": float(getattr(args, "libero_action_flow_gripper_flow_fps", 20.0)),
        "eef_training_target": "tcp_target_pose_robot_base+gripper_control",
    }
