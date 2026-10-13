"""Panda hand geometry used by VLABench policy observations."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from unitas.inference.robots.assets import asset_root

VLABENCH_GRIPPER_POINTS_PATH = (
    asset_root() / "vlabench_embodiments/franka_panda/gripper_surface_points_20.json"
)
VLABENCH_JOINT_NAMES = tuple(
    [f"joint{i}" for i in range(1, 8)] + ["finger_joint1", "finger_joint2"]
)

PANDA_HAND_LINK = "panda_hand"

VLABENCH_GRIPPER_LINKS = (
    PANDA_HAND_LINK,
    "panda_leftfinger",
    "panda_rightfinger",
)

VLABENCH_HAND_TO_EE = np.eye(4, dtype=np.float32)

VLABENCH_HAND_TO_EE[2, 3] = 0.093


def measured_vlabench_hand_poses(ee_pose_robot_base: np.ndarray) -> np.ndarray:
    """Recover the physical MJCF hand-link pose from the saved EE-site pose."""
    ee = np.asarray(ee_pose_robot_base, dtype=np.float32)
    return (ee @ np.linalg.inv(VLABENCH_HAND_TO_EE)[None]).astype(np.float32)


@lru_cache(maxsize=1)
def _load_vlabench_gripper_surface_points(
    asset_path: str,
) -> tuple[tuple[str, np.ndarray, np.ndarray], ...]:
    """Load fixed FPS samples stored in their parent URDF link frames."""
    path = Path(asset_path)
    if not path.is_file():
        raise FileNotFoundError(f"VLABench gripper-point asset is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema") != "vlabench_gripper_surface_points_v1":
        raise ValueError(f"Unsupported VLABench gripper-point schema in {path}")
    if payload.get("coordinate_frame") != "urdf_link":
        raise ValueError(f"VLABench points in {path} must use urdf_link coordinates")

    entries = []
    seen = set()
    for item in payload.get("links", []):
        link_name = str(item.get("link_name", ""))
        if link_name not in VLABENCH_GRIPPER_LINKS or link_name in seen:
            raise ValueError(f"Invalid or duplicate link {link_name!r} in {path}")
        points = np.asarray(item.get("points"), dtype=np.float32)
        normals = np.asarray(item.get("normals"), dtype=np.float32)
        if (
            points.ndim != 2
            or points.shape[1:] != (3,)
            or normals.shape != points.shape
            or not np.isfinite(points).all()
            or not np.isfinite(normals).all()
        ):
            raise ValueError(f"Invalid points/normals for {link_name!r} in {path}")
        entries.append((link_name, points, normals))
        seen.add(link_name)
    if tuple(link for link, _, _ in entries) != VLABENCH_GRIPPER_LINKS:
        raise ValueError(
            f"Expected ordered links {VLABENCH_GRIPPER_LINKS}, got "
            f"{tuple(link for link, _, _ in entries)} in {path}"
        )
    expected_count = int(payload.get("points_total", -1))
    if sum(len(points) for _, points, _ in entries) != expected_count:
        raise ValueError(f"Point count does not match metadata in {path}")
    return tuple(entries)


def build_vlabench_robot_flows(
    sample: dict,
    num_robot_tokens: int = 20,
) -> dict:
    """Transform fixed palm/finger samples using the measured hand pose."""
    if int(num_robot_tokens) <= 0:
        raise ValueError("num_robot_tokens must be positive")
    joints = np.asarray(sample["joint_positions"], dtype=np.float32)
    if joints.ndim != 2 or joints.shape[1] != 9:
        raise ValueError(f"joint_positions must be [T,9], got {joints.shape}")
    if tuple(sample["joint_names"]) != VLABENCH_JOINT_NAMES:
        raise ValueError(f"unexpected VLABench joint order: {sample['joint_names']}")
    measured_hand = np.asarray(sample["gripper_root_pose"], dtype=np.float32)
    if measured_hand.shape != (len(joints), 4, 4):
        raise ValueError(f"gripper_root_pose must be [T,4,4], got {measured_hand.shape}")
    entries = _load_vlabench_gripper_surface_points(str(VLABENCH_GRIPPER_POINTS_PATH))
    if sum(len(points) for _, points, _ in entries) != int(num_robot_tokens):
        raise ValueError(f"fixed VLABench asset does not contain {num_robot_tokens} points")

    local_points = []
    local_normals = []
    for link_name, points, normals in entries:
        points_t = np.broadcast_to(points[None], (len(joints), len(points), 3)).copy()
        if link_name == "panda_leftfinger":
            points_t[..., 1] += np.clip(joints[:, 7], 0.0, 0.04)[:, None]
            points_t[..., 2] += 0.0584
        elif link_name == "panda_rightfinger":
            points_t[..., 1] -= np.clip(joints[:, 8], 0.0, 0.04)[:, None]
            points_t[..., 2] += 0.0584
        local_points.append(points_t)
        local_normals.append(np.broadcast_to(normals[None], (len(joints), len(normals), 3)))
    local_points_array = np.concatenate(local_points, axis=1)
    local_normals_array = np.concatenate(local_normals, axis=1)
    points_h = np.concatenate(
        (
            local_points_array,
            np.ones((*local_points_array.shape[:2], 1), dtype=np.float32),
        ),
        axis=-1,
    )
    flows = np.einsum("tij,tnj->tni", measured_hand, points_h, optimize=True)[..., :3].astype(
        np.float32
    )
    aligned_normals = np.einsum(
        "tij,tnj->tni",
        measured_hand[:, :3, :3],
        local_normals_array,
        optimize=True,
    ).astype(np.float32)
    colors = np.zeros_like(flows, dtype=np.uint8)
    colors[..., 0] = 255
    colors[..., 2] = 255
    return {
        "robot_flows": flows,
        "robot_colors": colors,
        "robot_normals": aligned_normals,
        "robot_exists": np.ones(flows.shape[:2], dtype=np.bool_),
        "__preserve_robot_token_order__": True,
    }


CAMERA_LAYOUT = (("cam0", "front_camera"), ("cam1", "right_camera"), ("cam2", "wrist_camera"))
