# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class DecoderConfig:
    """Configuration for the gripper relative-flow action decoder."""

    num_gripper_tokens: int = 100
    position_n_freqs: int = 11
    position_scale: float = 1.0
    rope_position_scale: float = 0.001
    rope_theta: float = 10000.0
    position_encoding_mode: str = "additive_rope4d"
    flow_codec_dim: int = 128
    max_prediction_time: float = 5.0
    vlm_dim: int = 1024
    aux_visual_dim: int = 1024
    frustum_vision_dim: int | None = None
    frustum_premerge_vision_queries: bool = False
    visual_pos_merge_size: int = 32
    use_frustum_pe_routing: bool = True
    frustum_pe_num_depth_samples: int = 64
    frustum_pe_z_near: float = 0.1
    frustum_pe_z_far: float = 2.0
    frustum_pe_subpatch_valid_threshold: float = 1.0
    depth: int = 24
    num_heads: int = 16
    hidden_size: int = 1024
    mlp_ratio: float = 4.0
    time_dim: int = 32
    attn_type: str = "mot"
    mot_num_denoise_steps: int = 10
    mot_xyz_scale: float = 10.0
    mot_rotation_scale: float = 1.0
    mot_gripper_scale: float = 1.0
    num_denoise_steps: int = 10


def temporal_spec_for_domain(args, domain: str) -> dict:
    """Resolve the saved temporal contract for one released simulation domain."""
    prefix = next(
        (name for name in ("robotwin", "libero", "vlabench") if str(domain).startswith(name)), None
    )
    fields = {
        "prediction_horizon": (int, 10),
        "history_horizon": (int, 10),
        "gripper_flow_fps": (float, 7.0),
        "scene_flow_horizon": (int, 1),
        "scene_flow_fps": (float, 0.7),
    }
    vlabench_defaults = {
        "prediction_horizon": 15,
        "history_horizon": 15,
        "gripper_flow_fps": 10.0,
        "scene_flow_horizon": 15,
        "scene_flow_fps": 10.0,
    }
    result = {}
    for name, (convert, fallback) in fields.items():
        value = getattr(args, f"action_flow_{name}", fallback)
        if prefix:
            override = getattr(args, f"{prefix}_action_flow_{name}", None)
            if override is not None:
                value = override
            elif prefix == "vlabench":
                value = vlabench_defaults[name]
        result[name] = convert(value)
        if not math.isfinite(result[name]) or result[name] <= 0:
            raise ValueError(f"{domain} {name} must be positive, got {result[name]}")
    maximum = float(getattr(args, "action_flow_max_prediction_time", 2.0))
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("max_prediction_time must be finite and positive")
    if result["scene_flow_horizon"] / result["scene_flow_fps"] > maximum:
        raise ValueError(f"{domain} scene prediction time exceeds max_prediction_time")
    return result


def as_bool(value: object) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
        raise ValueError(f"invalid boolean value: {value!r}")
    return bool(value)


ROBOTWIN_ROBOT_TYPES = frozenset({"aloha", "arx_x5", "piper", "franka_panda", "ur5_wsg"})


def normalize_robotwin_robot_type(value) -> str:
    if value is None:
        return "aloha"
    if isinstance(value, np.ndarray):
        value = value.item() if value.shape == () else value.reshape(-1)[0]
    if isinstance(value, (bytes, bytearray, np.bytes_)):
        value = value.decode("utf-8")
    robot_type = str(value).strip().lower().replace("-", "_")
    aliases = {
        "": "aloha",
        "aloha": "aloha",
        "aloha_agilex": "aloha",
        "arx": "arx_x5",
        "arx5": "arx_x5",
        "arx_x5": "arx_x5",
        "x5": "arx_x5",
        "franka": "franka_panda",
        "franka_panda": "franka_panda",
        "panda": "franka_panda",
        "piper": "piper",
        "ur5": "ur5_wsg",
        "ur5_wsg": "ur5_wsg",
    }
    robot_type = aliases.get(robot_type, robot_type)
    if robot_type not in ROBOTWIN_ROBOT_TYPES:
        raise ValueError(
            f"Unsupported RobotWin robot_type={value!r}. Supported values: {sorted(ROBOTWIN_ROBOT_TYPES)}"
        )
    return robot_type
