"""RoboTwin server policy for the episode-trained UNITAS model."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from unitas.models.config import (
    as_bool as _as_bool,
)
from unitas.inference.preprocessing import resize_rgb_depth_intrinsic
from unitas.inference.robots.frames import (
    EEF_POSE_CONVENTION,
    canonicalize_robotwin_tcp_trajectories,
)
from unitas.inference.robots.robotwin import (
    ROBOTWIN_EMBODIMENT_SPECS,
    RobotWinLiteSampler,
    configure_robotwin_gripper_surface_points,
    robotwin_model_from_source,
)

CAMERA_LAYOUT = (("cam0", "head_camera"), ("cam1", "left_camera"), ("cam2", "right_camera"))


def _as_homogeneous(matrix: Any) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float32)
    if value.shape == (4, 4):
        return value.copy()
    if value.shape == (3, 4):
        output = np.eye(4, dtype=np.float32)
        output[:3] = value
        return output
    raise ValueError(f"expected [4,4] or [3,4] transform, got {value.shape}")


def _quat_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        raise ValueError("quaternion norm must be positive")
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _matrix_to_quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(r))
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        q = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(r)))
        if i == 0:
            s = np.sqrt(max(1 + r[0, 0] - r[1, 1] - r[2, 2], 1e-12)) * 2
            q = [
                (r[2, 1] - r[1, 2]) / s,
                0.25 * s,
                (r[0, 1] + r[1, 0]) / s,
                (r[0, 2] + r[2, 0]) / s,
            ]
        elif i == 1:
            s = np.sqrt(max(1 + r[1, 1] - r[0, 0] - r[2, 2], 1e-12)) * 2
            q = [
                (r[0, 2] - r[2, 0]) / s,
                (r[0, 1] + r[1, 0]) / s,
                0.25 * s,
                (r[1, 2] + r[2, 1]) / s,
            ]
        else:
            s = np.sqrt(max(1 + r[2, 2] - r[0, 0] - r[1, 1], 1e-12)) * 2
            q = [
                (r[1, 0] - r[0, 1]) / s,
                (r[0, 2] + r[2, 0]) / s,
                (r[1, 2] + r[2, 1]) / s,
                0.25 * s,
            ]
    q = np.asarray(q, dtype=np.float32)
    return q / np.linalg.norm(q)


def _pose_wxyz_to_matrix(pose: Any) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32).reshape(7)
    output = np.eye(4, dtype=np.float32)
    output[:3, :3] = _quat_wxyz_to_matrix(pose[3:])
    output[:3, 3] = pose[:3]
    return output


def _pose_xyzw_to_matrix(pose: Any) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32).reshape(7)
    return _pose_wxyz_to_matrix(np.r_[pose[:3], pose[6], pose[3:6]])


def _matrix_to_pose_xyzw(matrix: Any) -> np.ndarray:
    matrix = _as_homogeneous(matrix)
    q = _matrix_to_quat_wxyz(matrix[:3, :3])
    return np.r_[matrix[:3, 3], q[1:], q[0]].astype(np.float32)


def _matrix_to_pose_wxyz(matrix: Any) -> np.ndarray:
    matrix = _as_homogeneous(matrix)
    return np.r_[matrix[:3, 3], _matrix_to_quat_wxyz(matrix[:3, :3])].astype(np.float32)


def _rotation_angle_rad(first: np.ndarray, second: np.ndarray) -> float:
    relative = first[:3, :3].T @ second[:3, :3]
    return float(np.arccos(np.clip((np.trace(relative) - 1) * 0.5, -1.0, 1.0)))


def _depth_meters(camera: dict[str, Any]) -> np.ndarray:
    if "depth" in camera:
        depth = np.asarray(camera["depth"], dtype=np.float32)
        if depth.size and float(np.nanmax(depth)) > 20:
            depth = depth * 0.001
        return depth
    position = np.asarray(camera.get("position"), dtype=np.float32)
    if position.ndim == 3 and position.shape[-1] >= 3:
        return np.maximum(-position[..., 2], 0).astype(np.float32)
    raise KeyError("camera payload requires depth (or camera-space position)")


@dataclass
class GripperPointSet:
    xyz: np.ndarray
    exists: np.ndarray
    local_points_link: np.ndarray
    link_names: list[str]
    side_ids: np.ndarray
    world_from_robot: np.ndarray | None = None

    def validate(self, expected_points: int | None = None) -> None:
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3:
            raise ValueError(f"gripper xyz must be [N,3], got {self.xyz.shape}")
        n = int(self.xyz.shape[0])
        if self.exists.shape != (n,) or self.local_points_link.shape != (n, 3):
            raise ValueError("invalid gripper correspondence metadata shapes")
        if self.side_ids.shape != (n,) or len(self.link_names) != n:
            raise ValueError("invalid gripper side/link metadata")
        if expected_points is not None and n != int(expected_points):
            raise ValueError(f"expected {expected_points} gripper points, got {n}")


class RobotWinGripperPointProvider:
    """The same deterministic ALOHA gripper surface sampling used in training."""

    def __init__(
        self,
        *,
        num_gripper_tokens_per_arm: int = 100,
        seed: int = 0,
        device: str = "cpu",
        urdf_path: str | Path | None = None,
    ) -> None:
        self.num_gripper_tokens_per_arm = int(num_gripper_tokens_per_arm)
        spec = ROBOTWIN_EMBODIMENT_SPECS["aloha"]
        path = str(urdf_path or spec.urdf_path)
        self.left_sampler = RobotWinLiteSampler(
            path,
            device=device,
            link_whitelist=list(spec.left_gripper_links),
            robot_type="aloha",
            side="left",
        )
        self.right_sampler = RobotWinLiteSampler(
            path,
            device=device,
            link_whitelist=list(spec.right_gripper_links),
            robot_type="aloha",
            side="right",
        )
        configure_robotwin_gripper_surface_points(
            self.left_sampler,
            robot_type="aloha",
            side="left",
            num_points=self.num_gripper_tokens_per_arm,
            seed=seed,
        )
        configure_robotwin_gripper_surface_points(
            self.right_sampler,
            robot_type="aloha",
            side="right",
            num_points=self.num_gripper_tokens_per_arm,
            seed=seed,
        )
        local, names, sides = ([], [], [])
        for side_id, sampler in enumerate((self.left_sampler, self.right_sampler)):
            for name, points in sampler._presampled_points.items():
                local.append(np.asarray(points, dtype=np.float32))
                names.extend([name] * len(points))
                sides.extend([side_id] * len(points))
        self.local_points = np.concatenate(local, axis=0)
        self.link_names = names
        self.side_ids = np.asarray(sides, dtype=np.int64)

    @staticmethod
    def _joint_dict(
        sampler: RobotWinLiteSampler, observation: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        state = observation.get("robot_state")
        if not isinstance(state, dict):
            raise KeyError("RoboTwin observation requires robot_state")
        names = [
            x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)
            for x in state.get("joint_names", [])
        ]
        positions = np.asarray(state.get("joint_positions", []), dtype=np.float32).reshape(-1)
        if len(names) != len(positions):
            raise ValueError("robot_state joint_names/joint_positions length mismatch")
        lookup = dict(zip(names, positions))
        result = {
            name: torch.tensor([lookup[name]], dtype=sampler.dtype, device=sampler.device)
            for name in sampler.joint_names
            if name in lookup
        }
        if not result:
            raise KeyError("live joint names do not match the training ALOHA URDF")
        return result

    def __call__(self, observation: dict[str, Any]) -> GripperPointSet:
        left, _, _ = self.left_sampler.compute_points(
            self._joint_dict(self.left_sampler, observation)
        )
        right, _, _ = self.right_sampler.compute_points(
            self._joint_dict(self.right_sampler, observation)
        )
        xyz = np.concatenate((left[0].cpu().numpy(), right[0].cpu().numpy()), axis=0).astype(
            np.float32
        )
        return GripperPointSet(
            xyz=xyz,
            exists=np.ones(len(xyz), dtype=np.bool_),
            local_points_link=self.local_points.copy(),
            link_names=list(self.link_names),
            side_ids=self.side_ids.copy(),
            world_from_robot=_as_homogeneous(
                observation.get("robot_state", {}).get("base_pose_world", np.eye(4))
            ),
        )


@dataclass
class RobotWinHistoryFrame:
    points_source: np.ndarray
    tcp_world: np.ndarray
    gripper_open: np.ndarray
    world_from_source: np.ndarray
    joint_positions: np.ndarray

    def equivalent_state(self, other: "RobotWinHistoryFrame") -> bool:
        return np.array_equal(self.joint_positions, other.joint_positions)


class RobotWinOnlineBatchBuilder:
    """Construct the exact history/camera contract consumed by UNITAS."""

    def __init__(
        self,
        *,
        num_gripper_tokens_per_arm: int,
        prediction_horizon: int,
        history_horizon: int,
        max_scene_points: int = 1,
        camera_name: str = "head_camera",
        model_camera: str = "cam0",
        robot_point_provider: Callable[[dict], GripperPointSet] | None = None,
        seed: int = 0,
        urdf_path: str | Path | None = None,
        use_compact_scene: bool = False,
        max_points_per_part: int | None = None,
        grid_size: float = 0.005,
        coordinate_frame: str = "bimanual_midpoint",
        disable_center_shift: bool = True,
        image_size: int = 224,
        flow_fps: float = 15.0,
    ) -> None:
        del max_scene_points, camera_name, use_compact_scene, max_points_per_part, grid_size
        self.num_gripper_tokens_per_arm = int(num_gripper_tokens_per_arm)
        self.num_gripper_tokens = 2 * self.num_gripper_tokens_per_arm
        self.prediction_horizon = int(prediction_horizon)
        self.history_horizon = int(history_horizon)
        self.model_camera = str(model_camera)
        self.coordinate_frame = str(coordinate_frame).lower()
        self.disable_center_shift = _as_bool(disable_center_shift)
        self.image_size = int(image_size)
        self.flow_fps = float(flow_fps)
        if self.model_camera != "cam0":
            raise ValueError("episode-trained RoboTwin checkpoints require action_flow_camera=cam0")
        if not self.disable_center_shift:
            raise ValueError(
                "online UNITAS currently requires the checkpoint's disable_center_shift=true contract"
            )
        self.robot_point_provider = robot_point_provider or RobotWinGripperPointProvider(
            num_gripper_tokens_per_arm=self.num_gripper_tokens_per_arm,
            seed=seed,
            urdf_path=urdf_path,
        )
        self.history: deque[RobotWinHistoryFrame] = deque(maxlen=self.history_horizon + 1)
        self.last_point_set: GripperPointSet | None = None
        self.last_source_tcp: np.ndarray | None = None

    def reset(self) -> None:
        self.history.clear()
        self.last_point_set = None
        self.last_source_tcp = None

    @staticmethod
    def _world_from_source(observation: dict[str, Any]) -> np.ndarray:
        return _as_homogeneous(observation.get("robot_state", {}).get("base_pose_world", np.eye(4)))

    def current_gripper_points(self, observation: dict[str, Any]) -> GripperPointSet:
        point_set = self.robot_point_provider(observation)
        point_set.validate(self.num_gripper_tokens)
        self.last_point_set = point_set
        return point_set

    def _history_frame(
        self, observation: dict[str, Any], point_set: GripperPointSet | None = None
    ) -> RobotWinHistoryFrame:
        point_set = point_set or self.current_gripper_points(observation)
        state = observation.get("robot_state", {})
        endpose = observation.get("endpose", {})
        required = ("left_tcp_pose", "right_tcp_pose", "left_endpose", "right_endpose")
        missing = [key for key in required if key not in endpose]
        if missing:
            raise KeyError(f"RoboTwin endpose is missing {missing}")
        return RobotWinHistoryFrame(
            points_source=np.asarray(point_set.xyz, dtype=np.float32).copy(),
            tcp_world=np.stack(
                (
                    _pose_wxyz_to_matrix(endpose["left_tcp_pose"]),
                    _pose_wxyz_to_matrix(endpose["right_tcp_pose"]),
                ),
                axis=0,
            ),
            gripper_open=np.asarray(
                [[endpose.get("left_gripper", 0.0)], [endpose.get("right_gripper", 0.0)]],
                dtype=np.float32,
            ),
            world_from_source=self._world_from_source(observation),
            joint_positions=np.asarray(state.get("joint_positions", []), dtype=np.float32).reshape(
                -1
            ),
        )

    def _record_current(
        self,
        observation: dict[str, Any],
        point_set: GripperPointSet | None = None,
        *,
        force_append: bool = False,
    ) -> RobotWinHistoryFrame:
        frame = self._history_frame(observation, point_set)
        if force_append or not self.history or (not self.history[-1].equivalent_state(frame)):
            self.history.append(frame)
        else:
            self.history[-1] = frame
        return frame

    def update_history(self, observation: dict[str, Any]) -> GripperPointSet:
        points = self.current_gripper_points(observation)
        self._record_current(observation, points, force_append=True)
        return points

    def _padded_history(
        self, current: RobotWinHistoryFrame
    ) -> tuple[list[RobotWinHistoryFrame], np.ndarray]:
        frames = list(self.history)
        if not frames or not frames[-1].equivalent_state(current):
            frames.append(current)
        valid = [True] * len(frames)
        while len(frames) < self.history_horizon + 1:
            frames.insert(0, frames[0])
            valid.insert(0, False)
        return (
            frames[-(self.history_horizon + 1) :],
            np.asarray(valid[-(self.history_horizon + 1) :], dtype=np.bool_),
        )

    def _camera_arrays(
        self, observation: dict[str, Any], source_name: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        camera = observation.get("observation", {}).get(source_name)
        if not isinstance(camera, dict):
            raise KeyError(
                f"RoboTwin observation is missing observation/{source_name}; enable all three cameras"
            )
        rgb = np.asarray(camera["rgb"], dtype=np.uint8).copy()
        depth = _depth_meters(camera).copy()
        intrinsic = np.asarray(camera["intrinsic_cv"], dtype=np.float32).copy()
        depth = np.clip(depth * 1000, 0, 65535).astype(np.uint16).astype(np.float32) * 0.001
        extrinsic = _as_homogeneous(camera["extrinsic_cv"]) @ self._world_from_source(observation)
        rgb_t, depth_t, intrinsic_t = resize_rgb_depth_intrinsic(
            torch.from_numpy(rgb),
            torch.from_numpy(depth),
            torch.from_numpy(intrinsic),
            self.image_size,
        )
        return (rgb_t.numpy(), depth_t.numpy(), intrinsic_t.numpy(), extrinsic.astype(np.float32))

    @staticmethod
    def _source_pose_trajectory(
        frames: list[RobotWinHistoryFrame], current: RobotWinHistoryFrame, side: int
    ) -> np.ndarray:
        current_source_from_world = np.linalg.inv(current.world_from_source).astype(np.float32)
        return np.stack(
            [
                _matrix_to_pose_xyzw(current_source_from_world @ frame.tcp_world[side])
                for frame in frames
            ],
            axis=0,
        )

    def build_batch(self, observation: dict[str, Any], instruction: str) -> dict[str, Any]:
        current_points = self.current_gripper_points(observation)
        current = self._record_current(observation, current_points)
        frames, history_valid = self._padded_history(current)
        cameras = {name: self._camera_arrays(observation, source) for name, source in CAMERA_LAYOUT}
        left_raw = self._source_pose_trajectory(frames, current, 0)
        right_raw = self._source_pose_trajectory(frames, current, 1)
        left_current, left_history = canonicalize_robotwin_tcp_trajectories(
            left_raw[-1:], left_raw, cameras["cam1"][3]
        )
        right_current, right_history = canonicalize_robotwin_tcp_trajectories(
            right_raw[-1:], right_raw, cameras["cam2"][3]
        )
        source_tcp = np.stack((left_current[0], right_current[0]), axis=0).astype(np.float32)
        history_tcp = np.stack((left_history, right_history), axis=1).astype(np.float32)
        self.last_source_tcp = source_tcp.copy()
        current_source_from_world = np.linalg.inv(current.world_from_source).astype(np.float32)
        history_robot = []
        for frame in frames:
            source_from_frame_source = current_source_from_world @ frame.world_from_source
            points = (
                frame.points_source @ source_from_frame_source[:3, :3].T
                + source_from_frame_source[:3, 3]
            )
            history_robot.append(points.astype(np.float32))
        history_robot = np.stack(history_robot, axis=0)
        model_from_source = robotwin_model_from_source("aloha", self.coordinate_frame)
        rotation, translation = (model_from_source[:3, :3], model_from_source[:3, 3])
        history_robot = np.einsum("tnj,ij->tni", history_robot, rotation) + translation
        history_tcp_model = history_tcp.copy()
        for array in (history_tcp_model,):
            flat = array.reshape(-1, 7)
            for index in range(len(flat)):
                flat[index] = _matrix_to_pose_xyzw(
                    model_from_source @ _pose_xyzw_to_matrix(flat[index])
                )
        history_open = np.stack([frame.gripper_open for frame in frames], axis=0)
        history_robot_exists = np.broadcast_to(
            history_valid[:, None], (len(frames), self.num_gripper_tokens)
        ).copy()
        history_eef_exists = np.broadcast_to(history_valid[:, None], (len(frames), 2)).copy()
        batch: dict[str, Any] = {
            "__key__": ["robotwin-online"],
            "__domain__": ["robotwin"],
            "language_instruction": [str(instruction)],
            "robot_type": ["aloha"],
            "eef_pose_convention": [EEF_POSE_CONVENTION],
            "history_robot_flows": torch.from_numpy(history_robot[None].astype(np.float32)),
            "history_robot_exists": torch.from_numpy(history_robot_exists[None]),
            "history_eef_poses": torch.from_numpy(history_tcp_model[None]),
            "history_eef_open": torch.from_numpy(history_open[None]),
            "history_eef_exists": torch.from_numpy(history_eef_exists[None]),
            "history_eef_valid_mask": torch.from_numpy(history_valid[None]),
            "gripper_flow_fps": torch.tensor([self.flow_fps], dtype=torch.float32),
            "eef_flow_fps": torch.tensor([self.flow_fps], dtype=torch.float32),
            "gripper_flow_horizon": torch.tensor([self.prediction_horizon], dtype=torch.long),
            "eef_flow_horizon": torch.tensor([self.prediction_horizon], dtype=torch.long),
            "__model_from_source__": torch.from_numpy(model_from_source[None]),
            "__world_from_model__": torch.from_numpy(
                (current.world_from_source @ np.linalg.inv(model_from_source))[None].astype(
                    np.float32
                )
            ),
        }
        for model_name, (rgb, depth, intrinsic, extrinsic_source) in cameras.items():
            extrinsic_model = extrinsic_source @ np.linalg.inv(model_from_source).astype(np.float32)
            batch[f"{model_name}_initial_rgb"] = torch.from_numpy(rgb[None])
            batch[f"{model_name}_initial_depth"] = torch.from_numpy(depth[None])
            batch[f"{model_name}_intrinsic"] = torch.from_numpy(intrinsic[None])
            batch[f"{model_name}_extrinsic"] = torch.from_numpy(extrinsic_model[None])
            batch[f"{model_name}_exists"] = torch.ones((1,), dtype=torch.bool)
        return batch

    def eef_frame_parity_report(
        self,
        batch: dict[str, Any],
        observation: dict[str, Any],
        *,
        translation_tolerance_m: float = 1e-05,
        rotation_tolerance_deg: float = 0.05,
    ) -> dict[str, Any]:
        if self.last_source_tcp is None:
            raise RuntimeError("build_batch must run first")
        model_from_source = batch["__model_from_source__"][0].numpy()
        actual = batch["history_eef_poses"][0, -1].numpy()
        trans_errors, rot_errors = ([], [])
        for side in range(2):
            expected = model_from_source @ _pose_xyzw_to_matrix(self.last_source_tcp[side])
            got = _pose_xyzw_to_matrix(actual[side])
            trans_errors.append(float(np.linalg.norm(expected[:3, 3] - got[:3, 3])))
            rot_errors.append(float(np.rad2deg(_rotation_angle_rad(expected, got))))
        return {
            "passed": max(trans_errors) <= translation_tolerance_m
            and max(rot_errors) <= rotation_tolerance_deg,
            "eef_pose_convention": EEF_POSE_CONVENTION,
            "translation_error_m": trans_errors,
            "rotation_error_deg": rot_errors,
            "coordinate_frame": self.coordinate_frame,
        }


class DirectEEFActionConverter:
    """Map predicted canonical TCP objectives to RoboTwin planner EEF actions."""

    def __init__(
        self,
        *,
        max_translation_step_m: float | None = 0.08,
        max_rotation_step_deg: float | None = 45.0,
        workspace_min: Sequence[float] | None = None,
        workspace_max: Sequence[float] | None = None,
        gripper_binary_threshold: float | None = None,
    ) -> None:
        self.max_translation_step_m = (
            None if max_translation_step_m is None else float(max_translation_step_m)
        )
        self.max_rotation_step_rad = (
            None if max_rotation_step_deg is None else float(np.deg2rad(max_rotation_step_deg))
        )
        self.workspace_min = (
            None if workspace_min is None else np.asarray(workspace_min, dtype=np.float32)
        )
        self.workspace_max = (
            None if workspace_max is None else np.asarray(workspace_max, dtype=np.float32)
        )
        self.gripper_binary_threshold = (
            None if gripper_binary_threshold is None else float(gripper_binary_threshold)
        )
        if (self.workspace_min is None) != (self.workspace_max is None):
            raise ValueError("workspace_min/workspace_max must be supplied together")
        if self.gripper_binary_threshold is not None and (
            not 0.0 <= self.gripper_binary_threshold <= 1.0
        ):
            raise ValueError("gripper_binary_threshold must be in [0, 1]")
        self.last_safety_events: list[dict[str, Any]] = []
        self.last_calibration_report: dict[str, Any] | None = None

    @staticmethod
    def _planner_frames(observation: dict[str, Any]) -> list[np.ndarray]:
        endpose = observation["endpose"]
        return [
            _pose_wxyz_to_matrix(endpose["left_endpose"]),
            _pose_wxyz_to_matrix(endpose["right_endpose"]),
        ]

    def current_C_T_A(
        self,
        *,
        current_eef_poses_model: np.ndarray,
        world_from_model: np.ndarray,
        observation: dict[str, Any],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        planner = self._planner_frames(observation)
        transforms, translation_errors, rotation_errors = ([], [], [])
        for side in range(2):
            world_tcp = world_from_model @ _pose_xyzw_to_matrix(current_eef_poses_model[side])
            tcp_to_planner = np.linalg.inv(world_tcp).astype(np.float32) @ planner[side]
            reconstructed = world_tcp @ tcp_to_planner
            transforms.append(tcp_to_planner)
            translation_errors.append(
                float(np.linalg.norm(reconstructed[:3, 3] - planner[side][:3, 3]))
            )
            rotation_errors.append(
                float(np.rad2deg(_rotation_angle_rad(reconstructed, planner[side])))
            )
        self.last_calibration_report = {
            "passed": max(translation_errors) <= 1e-05 and max(rotation_errors) <= 0.05,
            "translation_error_m": translation_errors,
            "rotation_error_deg": rotation_errors,
            "C_T_A": [x.copy() for x in transforms],
        }
        return (transforms, planner)

    def _safe_pose(
        self, candidate: np.ndarray, previous: np.ndarray, *, frame_idx: int, side: str
    ) -> tuple[np.ndarray, bool]:
        reasons = []
        if not np.isfinite(candidate).all():
            reasons.append("non_finite")
        if (
            not reasons
            and self.workspace_min is not None
            and (
                np.any(candidate[:3, 3] < self.workspace_min)
                or np.any(candidate[:3, 3] > self.workspace_max)
            )
        ):
            reasons.append("workspace")
        if not reasons and self.max_translation_step_m is not None:
            step = float(np.linalg.norm(candidate[:3, 3] - previous[:3, 3]))
            if step > self.max_translation_step_m:
                reasons.append(f"translation_step={step:.6f}")
        if not reasons and self.max_rotation_step_rad is not None:
            step = _rotation_angle_rad(previous, candidate)
            if step > self.max_rotation_step_rad:
                reasons.append(f"rotation_step_deg={np.rad2deg(step):.3f}")
        if reasons:
            self.last_safety_events.append(
                {"frame": frame_idx, "side": side, "reasons": reasons, "action": "hold"}
            )
            return (previous.copy(), True)
        return (candidate, False)

    def trajectory_to_actions(
        self,
        pred_eef_poses: np.ndarray,
        pred_eef_open: np.ndarray,
        *,
        current_eef_poses_model: np.ndarray,
        world_from_model: np.ndarray,
        observation: dict[str, Any],
        chunk_size: int | None = None,
    ) -> list[np.ndarray]:
        poses = np.asarray(pred_eef_poses, dtype=np.float32)
        opens = np.asarray(pred_eef_open, dtype=np.float32)
        if opens.ndim == 2:
            opens = opens[..., None]
        if poses.ndim != 3 or poses.shape[1:] != (2, 7) or opens.shape != (len(poses), 2, 1):
            raise ValueError(
                f"invalid predicted EEF shapes: poses={poses.shape}, open={opens.shape}"
            )
        tcp_to_planner, previous = self.current_C_T_A(
            current_eef_poses_model=current_eef_poses_model,
            world_from_model=world_from_model,
            observation=observation,
        )
        horizon = min(len(poses), int(chunk_size)) if chunk_size is not None else len(poses)
        self.last_safety_events = []
        actions = []
        for frame_idx in range(horizon):
            arm_poses, arm_opens = ([], [])
            for side_idx, side in enumerate(("left", "right")):
                world_tcp = world_from_model @ _pose_xyzw_to_matrix(poses[frame_idx, side_idx])
                candidate, held = self._safe_pose(
                    world_tcp @ tcp_to_planner[side_idx],
                    previous[side_idx],
                    frame_idx=frame_idx,
                    side=side,
                )
                arm_poses.append(_matrix_to_pose_wxyz(candidate))
                if held:
                    open_command = float(observation["endpose"].get(f"{side}_gripper", 0.0))
                else:
                    open_command = float(np.clip(opens[frame_idx, side_idx, 0], 0, 1))
                    if self.gripper_binary_threshold is not None:
                        open_command = float(open_command >= self.gripper_binary_threshold)
                arm_opens.append(open_command)
                if not held:
                    previous[side_idx] = candidate
            actions.append(
                np.r_[arm_poses[0], arm_opens[0], arm_poses[1], arm_opens[1]].astype(np.float32)
            )
        return actions


__all__ = [
    "DirectEEFActionConverter",
    "GripperPointSet",
    "RobotWinGripperPointProvider",
    "RobotWinOnlineBatchBuilder",
]
