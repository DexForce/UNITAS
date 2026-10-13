from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
from typing import Any


def load_model(
    checkpoint: Path,
    florence_dir: Path,
    device: str = "cuda:0",
    *,
    inference_kind: str = "scene_flow",
    seed: int = 160484,
):
    import torch

    from unitas.models.model import UnitasModel

    if not florence_dir.is_dir():
        raise FileNotFoundError("Set --florence-dir to a local Florence-2-large directory.")
    saved = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    raw = saved.get("args")
    if raw is None:
        raise ValueError("The checkpoint must contain model weights and saved model arguments.")
    args = argparse.Namespace(**copy.deepcopy(raw if isinstance(raw, dict) else vars(raw)))
    required = {
        "action_flow_attn_type",
        "action_flow_hidden_size",
        "action_flow_depth",
        "action_flow_num_heads",
        "action_flow_codec_dim",
        "action_flow_position_encoding",
        "action_flow_num_gripper_tokens",
    }
    missing = sorted(required - vars(args).keys())
    if missing:
        raise ValueError(f"Checkpoint is missing model configuration: {', '.join(missing)}")
    backend = str(getattr(args, "action_flow_vlm_backend", "florence2"))
    if backend not in {"florence", "florence2"}:
        raise ValueError("This release supports Florence-2 checkpoints.")
    target = torch.device(device)
    if inference_kind == "libero":
        configure_determinism(seed)
    overrides = {
        "distributed": False,
        "device": str(target),
        "action_flow_skip_codec_checkpoint_load": True,
        "action_flow_codec_checkpoint_path": None,
        "florence_model_name": str(florence_dir.resolve()),
        "florence_cache_dir": None,
        "florence_local_files_only": True,
        "florence_dtype": "fp32",
        "florence_attn_implementation": "auto",
        "action_flow_mot_state_conditioning": "zero_vector",
        "action_flow_context_conditioning": "zero_vector",
    }
    for key, value in overrides.items():
        setattr(args, key, value)
    validate_checkpoint_args(args, inference_kind)
    model = UnitasModel(args, device=target)
    args.florence_attn_implementation = model.florence_encoder.attn_implementation
    model.load_state_dict(saved["model"], strict=True)
    model.requires_grad_(False).to(target).eval()
    del saved
    return (model, args, target)


def configure_determinism(seed: int) -> None:
    """Set LIBERO's random seeds and deterministic CUDA options before model loading."""
    import os
    import random

    import numpy as np
    import torch

    workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if workspace not in {":4096:8", ":16:8"}:
        raise ValueError("Unsupported CUBLAS_WORKSPACE_CONFIG for deterministic inference")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def move_batch(batch: dict[str, Any], device):
    import torch

    return {
        key: value.to(device)
        if torch.is_tensor(value) and not key.endswith("_initial_rgb")
        else value
        for key, value in batch.items()
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_fingerprint() -> str:
    """Identify the installed inference code, independent of checkout location."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for package in ("models", "inference"):
        for path in sorted((root / package).rglob("*.py")):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def validate_checkpoint_args(args, domain: str) -> None:
    from unitas.models.config import as_bool, temporal_spec_for_domain

    domains = ("libero", "robotwin", "vlabench") if domain == "scene_flow" else (domain,)
    tokens = int(args.action_flow_num_gripper_tokens)
    if tokens <= 0:
        raise ValueError("Checkpoint gripper token count must be positive")
    for name in domains:
        spec = temporal_spec_for_domain(args, name)
        if spec["history_horizon"] != spec["prediction_horizon"]:
            raise ValueError(f"{name} requires matching history and prediction horizons")
        maximum = float(getattr(args, "action_flow_max_prediction_time", 2.0))
        if (
            max(spec["history_horizon"], spec["prediction_horizon"]) / spec["gripper_flow_fps"]
            > maximum
        ):
            raise ValueError(f"{name} gripper trajectory exceeds checkpoint time range")
        if name == "libero":
            from unitas.inference.adapters.libero import validate_libero_checkpoint_args

            validate_libero_checkpoint_args(args)
        elif name == "vlabench":
            from unitas.inference.adapters.vlabench import validate_vlabench_checkpoint_args

            validate_vlabench_checkpoint_args(args)
            if tokens != 20:
                raise ValueError("VLABench requires the fixed 20-point gripper template")
        elif name == "robotwin":
            if not as_bool(getattr(args, "disable_center_shift", False)):
                raise ValueError("RoboTwin requires disable_center_shift=true")
            if getattr(args, "robotwin_coordinate_frame", "source") not in {
                "source",
                "bimanual_midpoint",
            }:
                raise ValueError("Unsupported RoboTwin coordinate frame")
        else:
            raise ValueError(f"Unsupported inference domain: {name}")
