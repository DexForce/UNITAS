from __future__ import annotations

from pathlib import Path

import numpy as np

HISTORY_FIELDS = {
    "history_robot_flows",
    "history_robot_exists",
    "history_eef_poses",
    "history_eef_open",
    "history_eef_exists",
    "history_eef_valid_mask",
    "history_valid_mask",
    "history_frame_indices",
}
TEMPORAL_FIELDS = {
    "gripper_flow_fps",
    "gripper_flow_horizon",
    "eef_flow_fps",
    "eef_flow_horizon",
    "scene_flow_fps",
    "scene_flow_horizon",
}


def input_field(key: str) -> bool:
    prefix, _, suffix = key.partition("_")
    camera = prefix.startswith("cam") and prefix[3:].isdigit()
    return key in HISTORY_FIELDS | TEMPORAL_FIELDS or (
        camera and suffix in {"initial_rgb", "initial_depth", "intrinsic", "extrinsic", "exists"}
    )


def save_input(path: Path, batch, xyz0, colors, *, domain: str, instruction: str):
    arrays = {}
    for key, value in batch.items():
        if not input_field(key):
            continue
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        arrays["batch." + key] = np.asarray(value)
    arrays.update(
        xyz0=np.asarray(xyz0, np.float32),
        colors=np.asarray(colors, np.uint8),
        domain=np.asarray(domain),
        instruction=np.asarray(instruction),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_input(path: Path, device):
    import torch

    with np.load(path, allow_pickle=False) as data:
        domain = str(data["domain"].item())
        if domain not in {"libero", "robotwin", "vlabench"}:
            raise ValueError("Unsupported simulation domain")
        instruction = str(data["instruction"].item())
        xyz0 = np.asarray(data["xyz0"], np.float32)
        colors = np.asarray(data["colors"], np.uint8)
        batch = {}
        for name in data.files:
            if name in {"domain", "instruction", "xyz0", "colors"}:
                continue
            key = name.removeprefix("batch.")
            if not name.startswith("batch.") or not input_field(key):
                raise ValueError(f"Unsupported input field: {name}")
            value = torch.as_tensor(
                data[name].copy(), device="cpu" if key.endswith("_initial_rgb") else device
            )
            if key.endswith("_initial_rgb"):
                if value.ndim != 4:
                    raise ValueError("Camera RGB must have a single batch dimension")
                if value.shape[-1] != 3 and value.shape[1] == 3:
                    value = value.permute(0, 2, 3, 1).contiguous()
                if value.shape[-1] != 3:
                    raise ValueError("Camera RGB must have three color channels")
            batch[key] = value
    if xyz0.ndim != 2 or xyz0.shape[1] != 3 or len(xyz0) == 0:
        raise ValueError("xyz0 must have shape [N,3] with N > 0")
    if not np.isfinite(xyz0).all() or colors.shape != xyz0.shape:
        raise ValueError("Query points must be finite; colors must have shape [N,3]")
    required = HISTORY_FIELDS - {
        "history_eef_valid_mask",
        "history_valid_mask",
        "history_frame_indices",
    }
    missing = (required | TEMPORAL_FIELDS) - batch.keys()
    if missing:
        raise ValueError(f"Missing input fields: {sorted(missing)}")
    for key in required:
        if batch[key].shape[0] != 1:
            raise ValueError("Scene flow inference accepts one observation at a time")
        if batch[key].is_floating_point() and (not torch.isfinite(batch[key]).all()):
            raise ValueError(f"Non-finite input: {key}")
    batch["__dataset__"] = [domain]
    batch["__domain__"] = [domain]
    batch["language_instruction"] = [instruction]
    return (batch, xyz0, colors, domain, instruction)


def backproject_queries(batch, *, voxel_size: float = 0.004, max_points: int = 60000):
    clouds, colors = ([], [])

    def array(value):
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)

    for index in range(3):
        prefix = f"cam{index}"
        if f"{prefix}_initial_depth" not in batch:
            continue
        if not bool(array(batch.get(f"{prefix}_exists", [True])).reshape(-1)[0]):
            continue
        depth = array(batch[f"{prefix}_initial_depth"])[0].squeeze()
        rgb = array(batch[f"{prefix}_initial_rgb"])[0]
        if rgb.shape[0] == 3 and rgb.shape[-1] != 3:
            rgb = rgb.transpose(1, 2, 0)
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb * (255 if rgb.max() <= 1 else 1), 0, 255).astype(np.uint8)
        intrinsic = array(batch[f"{prefix}_intrinsic"])[0]
        extrinsic = array(batch[f"{prefix}_extrinsic"])[0]
        valid = np.isfinite(depth) & (depth > 0)
        v, u = np.nonzero(valid)
        z = depth[v, u]
        camera = np.stack(
            [
                (u + 0.5 - intrinsic[0, 2]) * z / intrinsic[0, 0],
                (v + 0.5 - intrinsic[1, 2]) * z / intrinsic[1, 1],
                z,
            ],
            axis=1,
        )
        model_from_camera = np.linalg.inv(extrinsic)
        clouds.append(camera @ model_from_camera[:3, :3].T + model_from_camera[:3, 3])
        colors.append(rgb[v, u])
    if not clouds or sum(map(len, clouds)) == 0:
        raise ValueError("No positive-depth query points")
    xyz, rgb = (np.concatenate(clouds), np.concatenate(colors))
    if voxel_size <= 0 or max_points < 1:
        raise ValueError("voxel_size and max_points must be positive")
    _, ids = np.unique(np.floor(xyz / voxel_size).astype(np.int64), axis=0, return_index=True)
    if len(ids) > max_points:
        ids = ids[np.random.default_rng(42).choice(len(ids), max_points, replace=False)]
    return (xyz[ids].astype(np.float32), rgb[ids].astype(np.uint8))
