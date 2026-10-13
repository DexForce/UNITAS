"""History and calibrated RGB-D preparation for one policy observation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from unitas.inference.robots.transforms import as_homogeneous


def padded_history(history, current, horizon: int):
    """Replace the latest measured state and left-pad missing observations."""
    frames = list(history) or [current]
    frames[-1] = current
    missing = max(horizon + 1 - len(frames), 0)
    valid = np.asarray([False] * missing + [True] * len(frames), dtype=np.bool_)
    return ([frames[0]] * missing + frames)[-(horizon + 1) :], valid[-(horizon + 1) :]


def camera_payloads(observation, layout, *, drop_depth=False):
    """Express camera extrinsics in the episode's robot-base frame."""
    base_from_world = np.linalg.inv(as_homogeneous(observation["world_from_robot_base"]))
    cameras = {}
    for model_name, source_name in layout:
        source = observation["cameras"][source_name]
        depth = np.asarray(source["depth"], dtype=np.float32).copy()
        camera_to_base = base_from_world @ as_homogeneous(source["camera_to_world"])
        cameras[model_name] = {
            "initial_rgb": np.asarray(source["rgb"], dtype=np.uint8).copy(),
            "initial_depth": np.zeros_like(depth) if drop_depth else depth,
            "intrinsic": np.asarray(source["intrinsic"], dtype=np.float32).copy(),
            "extrinsic": np.linalg.inv(camera_to_base).astype(np.float32),
        }
    return cameras


def pack_observation(
    *,
    robot_history,
    eef_history,
    eef_open,
    valid,
    cameras,
    domain,
    robot_type,
    instruction,
    convention,
    temporal,
    image_size,
):
    """Build a single-arm inference batch without future targets or loss fields."""
    time_steps, points = robot_history.shape[:2]
    if time_steps != temporal["history_horizon"] + 1 or valid.shape != (time_steps,):
        raise ValueError("History length does not match the checkpoint")
    poses = np.broadcast_to(
        np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32), (time_steps, 2, 7)
    ).copy()
    poses[:, 1] = eef_history
    opening = np.zeros((time_steps, 2, 1), dtype=np.float32)
    opening[:, 1, 0] = eef_open
    arm_valid = np.zeros((time_steps, 2), dtype=np.bool_)
    arm_valid[:, 1] = valid
    arrays = {
        "history_robot_flows": robot_history,
        "history_robot_exists": np.broadcast_to(valid[:, None], (time_steps, points)).copy(),
        "history_eef_poses": poses,
        "history_eef_open": opening,
        "history_eef_exists": arm_valid,
        "history_valid_mask": valid,
        "history_frame_indices": np.arange(-(time_steps - 1), 1, dtype=np.int64),
    }
    batch = {key: torch.from_numpy(value)[None] for key, value in arrays.items()}
    batch.update(
        __domain__=[domain],
        robot_type=[robot_type],
        language_instruction=[str(instruction)],
        eef_pose_convention=[convention],
    )
    for prefix in ("gripper", "eef", "scene"):
        is_scene = prefix == "scene"
        batch[f"{prefix}_flow_fps"] = torch.tensor(
            [temporal["scene_flow_fps" if is_scene else "gripper_flow_fps"]], dtype=torch.float32
        )
        batch[f"{prefix}_flow_horizon"] = torch.tensor(
            [temporal["scene_flow_horizon" if is_scene else "prediction_horizon"]], dtype=torch.long
        )
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    for index in range(3):
        name = f"cam{index}"
        camera = cameras.get(name)
        if camera is None:
            batch[f"{name}_initial_rgb"] = torch.zeros(
                1, image_size, image_size, 3, dtype=torch.uint8
            )
            batch[f"{name}_initial_depth"] = torch.zeros(1, image_size, image_size)
            # Preserve the old placeholder camera's resized intrinsic convention.
            source = next(reversed(cameras.values()))
            height, width = source["initial_depth"].shape
            intrinsic = torch.eye(3)
            intrinsic[0, 0] *= image_size / width
            intrinsic[1, 1] *= image_size / height
            batch[f"{name}_intrinsic"] = intrinsic[None]
            batch[f"{name}_extrinsic"] = torch.eye(4)[None]
        else:
            rgb, depth, intrinsic = resize_rgb_depth_intrinsic(
                torch.from_numpy(camera["initial_rgb"]),
                torch.from_numpy(camera["initial_depth"]),
                torch.from_numpy(camera["intrinsic"]),
                image_size,
            )
            batch[f"{name}_initial_rgb"] = rgb[None]
            batch[f"{name}_initial_depth"] = depth[None]
            batch[f"{name}_intrinsic"] = intrinsic[None]
            batch[f"{name}_extrinsic"] = torch.from_numpy(camera["extrinsic"])[None]
        batch[f"{name}_exists"] = torch.tensor([camera is not None], dtype=torch.bool)
    return batch


def resize_rgb_depth_intrinsic(
    rgb: torch.Tensor, depth: torch.Tensor, intrinsic: torch.Tensor, image_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resize camera maps to ``image_size`` and scale intrinsics."""
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"rgb must be [H,W,3], got {tuple(rgb.shape)}")
    if depth.ndim != 2:
        raise ValueError(f"depth must be [H,W], got {tuple(depth.shape)}")
    h, w = (int(rgb.shape[0]), int(rgb.shape[1]))
    if h == image_size and w == image_size:
        return (rgb, depth, intrinsic)
    rgb_f = rgb.permute(2, 0, 1).unsqueeze(0).float()
    rgb_r = F.interpolate(
        rgb_f, size=(image_size, image_size), mode="bilinear", align_corners=False
    )
    rgb_out = rgb_r.squeeze(0).permute(1, 2, 0).to(dtype=rgb.dtype).contiguous()
    if rgb.dtype == torch.uint8:
        rgb_out = rgb_out.round().clamp(0, 255).to(torch.uint8)
    depth_r = (
        F.interpolate(
            depth.unsqueeze(0).unsqueeze(0).float(), size=(image_size, image_size), mode="nearest"
        )
        .squeeze(0)
        .squeeze(0)
        .to(dtype=depth.dtype)
    )
    scale_x = float(image_size) / float(w)
    scale_y = float(image_size) / float(h)
    k = intrinsic.to(dtype=torch.float32).clone()
    k[0, 0] *= scale_x
    k[1, 1] *= scale_y
    k[0, 2] *= scale_x
    k[1, 2] *= scale_y
    return (rgb_out, depth_r, k.to(dtype=intrinsic.dtype))
