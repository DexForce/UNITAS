"""LIBERO-specific simulation geometry helpers."""

from __future__ import annotations

import numpy as np
import torch

from unitas.inference.robots.frames import LIBERO_GRIPPER_ROOT_FROM_PANDA_HAND

LIBERO_JOINT_NAMES = tuple((f"robot0_joint{i}" for i in range(1, 8)))
PANDA_JOINT_NAMES = tuple((f"panda_joint{i}" for i in range(1, 8)))
PANDA_FINGER_JOINTS = ("panda_finger_joint1", "panda_finger_joint2")
PANDA_HAND_LINK = "panda_hand"


def poses_xyzw_to_matrices(poses, *, device, dtype) -> torch.Tensor:
    """Convert ``[x,y,z,qx,qy,qz,qw]`` poses to homogeneous matrices."""
    poses = torch.as_tensor(poses, device=device, dtype=dtype)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(f"right_gripper_pose must have shape [T,7], got {tuple(poses.shape)}")
    xyz, quat = (poses[:, :3], poses[:, 3:])
    quat = quat / torch.linalg.vector_norm(quat, dim=1, keepdim=True).clamp_min(1e-08)
    x, y, z, w = quat.unbind(1)
    rotation = torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=1,
    ).reshape(-1, 3, 3)
    matrices = torch.eye(4, device=device, dtype=dtype).repeat(len(poses), 1, 1)
    matrices[:, :3, :3] = rotation
    matrices[:, :3, 3] = xyz
    return matrices


def build_libero_joint_state(sample: dict, robot_sampler) -> dict[str, torch.Tensor]:
    """Map saved MuJoCo Panda state to the joint names in the visualization URDF."""
    positions = torch.as_tensor(sample["joint_positions"], dtype=torch.float32)
    if positions.ndim != 2 or positions.shape[1] != 7:
        raise ValueError(f"joint_positions must have shape [T,7], got {tuple(positions.shape)}")
    saved_names = tuple((str(name) for name in sample["joint_names"]))
    if saved_names != LIBERO_JOINT_NAMES:
        raise ValueError(
            f"LIBERO v5 must store joint_names in robot0_joint1..robot0_joint7 order; got {saved_names}"
        )
    required = set(PANDA_JOINT_NAMES + PANDA_FINGER_JOINTS)
    missing = required - set(robot_sampler.joint_names)
    if missing:
        raise KeyError(f"LIBERO Panda URDF is missing joints: {sorted(missing)}")
    state = {panda_name: positions[:, index] for index, panda_name in enumerate(PANDA_JOINT_NAMES)}
    finger_open = (
        torch.as_tensor(sample["right_gripper_open"], dtype=torch.float32)
        .reshape(-1)
        .clamp(0.0, 0.04)
    )
    state[PANDA_FINGER_JOINTS[0]] = finger_open
    state[PANDA_FINGER_JOINTS[1]] = finger_open
    return state


def build_libero_robot_flows(
    sample: dict, robot_sampler, num_robot_tokens: int, *, seed: int
) -> dict:
    """Build Panda surface trajectories aligned to the saved MuJoCo gripper root."""
    robot_sampler.presample(num_robot_tokens, gripper_filter="both", seed=seed)
    joint_state = build_libero_joint_state(sample, robot_sampler)
    points, colors, normals, links = robot_sampler.compute_points_with_link_transforms(
        joint_state, extra_link_names=[PANDA_HAND_LINK]
    )
    if PANDA_HAND_LINK not in links:
        raise KeyError(f"LIBERO Panda URDF has no {PANDA_HAND_LINK!r} FK link")
    saved_root = poses_xyzw_to_matrices(
        sample["right_gripper_root_pose"], device=points.device, dtype=points.dtype
    )
    urdf_hand = links[PANDA_HAND_LINK].to(device=points.device, dtype=points.dtype)
    if saved_root.shape[0] != points.shape[0]:
        raise ValueError(
            f"gripper-root/FK length mismatch: {saved_root.shape[0]} versus {points.shape[0]}"
        )
    root_from_urdf_hand = torch.eye(4, device=points.device, dtype=points.dtype)
    root_from_urdf_hand[:3, :3] = torch.as_tensor(
        LIBERO_GRIPPER_ROOT_FROM_PANDA_HAND, device=points.device, dtype=points.dtype
    )
    alignment = saved_root @ root_from_urdf_hand @ torch.linalg.inv(urdf_hand)
    points_h = torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)
    aligned_points = torch.einsum("tij,tnj->tni", alignment, points_h)[..., :3]
    aligned_normals = torch.einsum("tij,tnj->tni", alignment[:, :3, :3], normals)
    flows = aligned_points.detach().cpu().numpy().astype(np.float32)
    return {
        "robot_flows": flows,
        "robot_colors": colors.detach().cpu().numpy(),
        "robot_normals": aligned_normals.detach().cpu().numpy().astype(np.float32),
        "robot_exists": np.ones(flows.shape[:2], dtype=np.bool_),
        "__preserve_robot_token_order__": True,
    }


CAMERA_LAYOUT = (("cam0", "agentview"), ("cam1", "robot0_eye_in_hand"))
