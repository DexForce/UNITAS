"""Canonical end-effector frame used by the online adapters."""

from __future__ import annotations

import numpy as np

import unitas.inference.robots.transforms as transform_utils

EEF_POSE_CONVENTION = "tcp_z_forward_x_opening_y_camera"
LIBERO_GRIPPER_ROOT_FROM_PANDA_HAND = np.asarray(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
)
LIBERO_OBSERVED_FROM_CANONICAL = np.asarray(
    [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
)


def canonicalize_libero_observed_tcp_matrices(matrices: np.ndarray) -> np.ndarray:
    """Express measured LIBERO TCP poses as +Z forward / +X opening."""
    poses = np.asarray(matrices, dtype=np.float32)
    if poses.shape[-2:] != (4, 4):
        raise ValueError(f"LIBERO TCP matrices must end in [4,4], got {poses.shape}")
    canonical = poses.copy()
    canonical[..., :3, :3] = canonical[..., :3, :3] @ LIBERO_OBSERVED_FROM_CANONICAL
    return canonical


def _pose_xyzw_to_matrices(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim < 2 or poses.shape[-1] != 7:
        raise ValueError(f"pose array must end in 7 channels, got {poses.shape}")
    return np.asarray(
        transform_utils.convert_pose_quat2mat(poses.reshape(-1, 7)), dtype=np.float32
    ).reshape(*poses.shape[:-1], 4, 4)


def _pose_matrices_to_xyzw(matrices: np.ndarray, output_shape: tuple[int, ...]) -> np.ndarray:
    return np.asarray(
        transform_utils.convert_pose_mat2quat(matrices.reshape(-1, 4, 4)), dtype=np.float32
    ).reshape(output_shape)


def canonicalize_robotwin_tcp_trajectories(
    future_poses: np.ndarray, history_poses: np.ndarray, camera_extrinsic: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert RoboTwin X-forward/Y-opening TCP poses to the shared convention."""
    future = _pose_xyzw_to_matrices(future_poses)
    history = _pose_xyzw_to_matrices(history_poses)
    extrinsic = np.asarray(camera_extrinsic, dtype=np.float32)
    if extrinsic.shape != (4, 4):
        raise ValueError(f"camera extrinsic must be [4,4], got {extrinsic.shape}")
    anchor = future[0]
    camera_position = np.linalg.inv(extrinsic)[:3, 3]
    camera_vector = camera_position - anchor[:3, 3]
    camera_side = float(np.dot(anchor[:3, 2], camera_vector))
    if abs(camera_side) <= 1e-06:
        raise ValueError(
            "cannot choose the canonical TCP axis sign because the wrist camera has no component along the candidate +Y axis"
        )
    sign = 1.0 if camera_side > 0.0 else -1.0
    old_from_canonical = np.asarray(
        [[0.0, 0.0, 1.0], [sign, 0.0, 0.0], [0.0, sign, 0.0]], dtype=np.float32
    )
    future[..., :3, :3] = future[..., :3, :3] @ old_from_canonical
    history[..., :3, :3] = history[..., :3, :3] @ old_from_canonical
    return (
        _pose_matrices_to_xyzw(future, np.asarray(future_poses).shape),
        _pose_matrices_to_xyzw(history, np.asarray(history_poses).shape),
    )


__all__ = [
    "EEF_POSE_CONVENTION",
    "LIBERO_GRIPPER_ROOT_FROM_PANDA_HAND",
    "LIBERO_OBSERVED_FROM_CANONICAL",
    "canonicalize_libero_observed_tcp_matrices",
    "canonicalize_robotwin_tcp_trajectories",
]
