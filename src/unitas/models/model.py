from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch
import torch.nn as nn

from unitas.models.config import normalize_robotwin_robot_type
from unitas.models.config import DecoderConfig
from unitas.models.config import as_bool as _as_bool

from .decoder import UnitasDecoder
from .encoders.florence import Florence2VLMEncoder, image_hw_to_world_points, resolve_language_batch


_DOMAIN_EMBODIMENT_ROLES = {
    "libero": "a single-arm Franka Panda robot",
    "vlabench": "a single-arm Franka Panda robot",
}
_ROBOTWIN_EMBODIMENT_ROLES = {
    "aloha": "a dual-arm ALOHA robot",
    "arx_x5": "a dual-arm ARX-X5 robot",
    "piper": "a dual-arm Piper robot",
    "franka_panda": "a dual-arm Franka Panda robot",
    "ur5_wsg": "a dual-arm UR5 robot",
}


@dataclass
class EncodedObservation:
    """Encoded observation tokens for the action head."""

    vlm_tokens: torch.Tensor
    vlm_xyz: torch.Tensor
    vlm_mask: torch.Tensor
    vlm_xyz_mask: torch.Tensor
    vlm_depth_valid_mask: torch.Tensor
    vlm_intrinsic: torch.Tensor
    vlm_extrinsic: torch.Tensor
    vlm_vision_tokens: torch.Tensor
    aux_visual_tokens: torch.Tensor
    aux_frustum_vision_tokens: torch.Tensor
    aux_visual_xyz: torch.Tensor
    aux_visual_mask: torch.Tensor
    aux_visual_xyz_mask: torch.Tensor
    aux_visual_depth_valid_mask: torch.Tensor
    aux_visual_intrinsic: torch.Tensor
    aux_visual_extrinsic: torch.Tensor

    def decoder_kwargs(self) -> dict[str, torch.Tensor]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


def _batch_metadata_values(batch: dict, key: str, batch_size: int) -> list[Any]:
    value = batch.get(key)
    if value is None:
        return [None] * batch_size
    if isinstance(value, (list, tuple)):
        if len(value) != batch_size:
            raise ValueError(f"batch field {key!r} has {len(value)} entries, expected {batch_size}")
        return list(value)
    return [value] * batch_size


def _metadata_text(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    return "" if value is None else str(value).strip()


def _canonical_embodiment_roles(batch: dict, batch_size: int) -> list[str | None]:
    domains = _batch_metadata_values(batch, "__domain__", batch_size)
    robot_types = _batch_metadata_values(batch, "robot_type", batch_size)
    roles = []
    for domain_value, robot_type in zip(domains, robot_types):
        domain = _metadata_text(domain_value).lower()
        if domain.startswith("robotwin"):
            roles.append(_ROBOTWIN_EMBODIMENT_ROLES[normalize_robotwin_robot_type(robot_type)])
        elif domain.startswith("libero"):
            roles.append(_DOMAIN_EMBODIMENT_ROLES["libero"])
        elif domain.startswith("vlabench"):
            roles.append(_DOMAIN_EMBODIMENT_ROLES["vlabench"])
        else:
            roles.append(None)
    return roles


def _embodiment_descriptions(batch: dict, batch_size: int) -> list[str | None]:
    return [
        f"The robot embodiment is {role}" if role is not None else None
        for role in _canonical_embodiment_roles(batch, batch_size)
    ]


def _trajectory_prompt(
    instruction: str, embodiment: str | None, *, successful: bool, subject: str = "robot"
) -> str:
    sentences = []
    if embodiment:
        sentences.append(embodiment.rstrip("."))
    sentences.append(str(instruction).strip().rstrip("."))
    outcome = "successful" if successful else "failed"
    sentences.append(f"Predict the {outcome} {subject} trajectory")
    return ". ".join((sentence for sentence in sentences if sentence)) + "."


class UnitasModel(nn.Module):
    """Florence-2 encoder and MoT decoder for UNITAS inference."""

    def __init__(
        self,
        args,
        *,
        device: str | torch.device | None = None,
        action_flow_config: DecoderConfig | None = None,
        florence_encoder: Any | None = None,
    ) -> None:
        super().__init__()
        self.args = args
        self.device_hint = torch.device(device or getattr(args, "device", "cuda"))
        self.instruction = str(getattr(args, "action_flow_instruction", "pick up the object"))
        self.camera = str(getattr(args, "action_flow_camera", "cam0"))
        self.vlm_backend = str(getattr(args, "action_flow_vlm_backend", "florence2")).lower()
        if self.vlm_backend not in {"florence", "florence2"}:
            raise ValueError("This release supports Florence-2 checkpoints.")
        vlm_dim = int(getattr(args, "action_flow_vlm_dim", getattr(args, "vlm_dim", 1024)))
        vlm_num_image_tokens = int(getattr(args, "action_flow_vlm_num_image_tokens", 577))
        self.vlm_num_image_tokens = vlm_num_image_tokens
        cfg = action_flow_config or DecoderConfig(
            num_gripper_tokens=int(getattr(args, "action_flow_num_gripper_tokens", 100)),
            hidden_size=int(getattr(args, "action_flow_hidden_size", 512)),
            flow_codec_dim=int(getattr(args, "action_flow_codec_dim", 48)),
            depth=int(getattr(args, "action_flow_depth", 8)),
            num_heads=int(getattr(args, "action_flow_num_heads", 8)),
            mlp_ratio=float(getattr(args, "action_flow_mlp_ratio", 4.0)),
            num_denoise_steps=int(getattr(args, "action_flow_num_denoise_steps", 10)),
            mot_num_denoise_steps=int(getattr(args, "action_flow_mot_num_denoise_steps", 10)),
            aux_visual_dim=vlm_dim,
            vlm_dim=vlm_dim,
            frustum_vision_dim=None,
            frustum_premerge_vision_queries=False,
            visual_pos_merge_size=int(getattr(args, "action_flow_visual_pos_merge_size", 32)),
            use_frustum_pe_routing=_as_bool(
                getattr(args, "action_flow_use_frustum_pe_routing", True)
            ),
            frustum_pe_num_depth_samples=int(
                getattr(args, "action_flow_frustum_pe_num_depth_samples", 64)
            ),
            frustum_pe_z_near=float(getattr(args, "action_flow_frustum_pe_z_near", 0.1)),
            frustum_pe_z_far=float(getattr(args, "action_flow_frustum_pe_z_far", 2.0)),
            frustum_pe_subpatch_valid_threshold=float(
                getattr(args, "action_flow_frustum_pe_subpatch_valid_threshold", 1.0)
            ),
            time_dim=int(getattr(args, "action_flow_time_dim", 32)),
            max_prediction_time=float(getattr(args, "action_flow_max_prediction_time", 2.0)),
            position_encoding_mode=str(
                getattr(args, "action_flow_position_encoding", "additive_rope4d")
            ),
            rope_theta=float(getattr(args, "action_flow_rope_theta", 10000.0)),
            rope_position_scale=float(getattr(args, "action_flow_rope_position_scale", 0.001)),
            mot_xyz_scale=float(getattr(args, "action_flow_mot_xyz_scale", 10.0)),
            mot_rotation_scale=float(getattr(args, "action_flow_mot_rotation_scale", 1.0)),
            mot_gripper_scale=float(getattr(args, "action_flow_mot_gripper_scale", 1.0)),
            attn_type=str(getattr(args, "action_flow_attn_type", "mot")),
        )
        self.decoder = UnitasDecoder(cfg)
        self.florence_encoder = None
        self._init_vlm_encoders(florence_encoder)
        self.vlm_encoder = self.florence_encoder
        if self.vlm_encoder is not None:
            expected_vlm_dim = getattr(self.vlm_encoder, "hidden_size", None)
            if expected_vlm_dim is not None and int(expected_vlm_dim) != int(
                self.decoder.config.vlm_dim
            ):
                raise ValueError(
                    f"Checkpoint decoder expects VLM dimension {self.decoder.config.vlm_dim}, "
                    f"but the Florence encoder emits {expected_vlm_dim}."
                )
            expected_image_tokens = getattr(self.vlm_encoder, "expected_num_image_tokens", None)
            if expected_image_tokens is not None:
                self.vlm_num_image_tokens = int(expected_image_tokens)
            token_px = getattr(self.vlm_encoder, "token_pixel_size", None)
            merge = int(self.decoder.config.visual_pos_merge_size)
            if token_px is not None and int(token_px) != merge:
                raise ValueError(
                    f"visual_pos_merge_size={merge} must equal encoder token size={token_px} so PE patches match vision tokens."
                )
        self._configure_encoder_modules()

    @property
    def config(self) -> DecoderConfig:
        return self.decoder.config

    def _init_vlm_encoders(self, florence_encoder) -> None:
        args = self.args
        kwargs = dict(
            model_name=str(getattr(args, "florence_model_name", "microsoft/Florence-2-large")),
            device=self.device_hint,
            dtype=str(getattr(args, "florence_dtype", "fp32")),
            cache_dir=getattr(args, "florence_cache_dir", None),
            local_files_only=bool(getattr(args, "florence_local_files_only", False)),
            image_size=int(getattr(args, "action_flow_vlm_image_size", 768)),
            attn_implementation=str(getattr(args, "florence_attn_implementation", "auto")),
        )
        self.florence_encoder = florence_encoder or Florence2VLMEncoder(**kwargs)

    def _configure_encoder_modules(self) -> None:
        module = self.florence_encoder.model
        module.requires_grad_(False).eval()
        self.add_module("florence_model", module)

    def _decoder_device(self) -> torch.device:
        return next(self.decoder.parameters()).device

    @staticmethod
    def _camera_prefixes(batch: dict) -> list[str]:
        suffix = "_initial_rgb"
        return sorted((key[: -len(suffix)] for key in batch.keys() if key.endswith(suffix)))

    def _ordered_camera_prefixes(self, batch: dict) -> list[str]:
        """Primary camera first (``self.camera``), then remaining views in sorted order."""
        prefixes = self._camera_prefixes(batch)
        if not prefixes:
            raise KeyError("batch does not contain any *_initial_rgb camera fields")
        primary = self.camera if self.camera in prefixes else prefixes[0]
        return [primary] + [prefix for prefix in prefixes if prefix != primary]

    def _gather_multiview_camera_batch(
        self, batch: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
        """Stack cameras as ``[B, V, ...]`` with view0 = primary VLM camera."""
        prefixes = self._ordered_camera_prefixes(batch)
        B = int(batch["history_robot_flows"].shape[0])
        rgb_views: list[torch.Tensor] = []
        depth_views: list[torch.Tensor] = []
        intrinsic_views: list[torch.Tensor] = []
        extrinsic_views: list[torch.Tensor] = []
        exists_views: list[torch.Tensor] = []
        for v, prefix in enumerate(prefixes):
            required = [
                f"{prefix}_initial_rgb",
                f"{prefix}_initial_depth",
                f"{prefix}_intrinsic",
                f"{prefix}_extrinsic",
            ]
            has_keys = all((key in batch for key in required))
            if not has_keys:
                if v == 0:
                    raise KeyError(f"batch is missing primary camera payload {prefix!r}")
                ref_rgb = batch[f"{prefixes[0]}_initial_rgb"]
                ref_depth = batch[f"{prefixes[0]}_initial_depth"]
                ref_intrinsic = batch[f"{prefixes[0]}_intrinsic"]
                ref_extrinsic = batch[f"{prefixes[0]}_extrinsic"]
                rgb_value = torch.zeros_like(ref_rgb)
                depth_value = torch.zeros_like(ref_depth)
                intrinsic_value = torch.zeros_like(ref_intrinsic)
                extrinsic_value = torch.eye(
                    4, dtype=ref_extrinsic.dtype, device=ref_extrinsic.device
                ).expand_as(ref_extrinsic)
                valid = torch.zeros(B, dtype=torch.bool, device=ref_depth.device)
            else:
                rgb_value = batch[f"{prefix}_initial_rgb"]
                depth_value = batch[f"{prefix}_initial_depth"]
                intrinsic_value = batch[f"{prefix}_intrinsic"]
                extrinsic_value = batch[f"{prefix}_extrinsic"]
                raw_exists = batch.get(f"{prefix}_exists")
                if raw_exists is None:
                    valid = torch.ones(B, dtype=torch.bool, device=depth_value.device)
                elif torch.is_tensor(raw_exists):
                    valid = raw_exists.bool()
                else:
                    valid = torch.as_tensor(raw_exists, dtype=torch.bool, device=depth_value.device)
                rgb_mask = valid.to(rgb_value.device).view(B, *[1] * (rgb_value.ndim - 1))
                depth_mask = valid.view(B, *[1] * (depth_value.ndim - 1))
                intrinsic_mask = valid.view(B, *[1] * (intrinsic_value.ndim - 1))
                extrinsic_mask = valid.view(B, *[1] * (extrinsic_value.ndim - 1))
                rgb_value = torch.where(rgb_mask, rgb_value, torch.zeros_like(rgb_value))
                depth_value = torch.where(depth_mask, depth_value, torch.zeros_like(depth_value))
                intrinsic_value = torch.where(
                    intrinsic_mask, intrinsic_value, torch.zeros_like(intrinsic_value)
                )
                extrinsic_value = torch.where(
                    extrinsic_mask,
                    extrinsic_value,
                    torch.eye(
                        4, dtype=extrinsic_value.dtype, device=extrinsic_value.device
                    ).expand_as(extrinsic_value),
                )
            rgb_views.append(rgb_value)
            depth_views.append(depth_value)
            intrinsic_views.append(intrinsic_value)
            extrinsic_views.append(extrinsic_value)
            exists_views.append(valid)
        rgb = torch.stack(rgb_views, dim=1)
        depth = torch.stack(depth_views, dim=1)
        intrinsic = torch.stack(intrinsic_views, dim=1)
        extrinsic = torch.stack(extrinsic_views, dim=1)
        exists = torch.stack(exists_views, dim=1)
        return (rgb, depth, intrinsic, extrinsic, exists, prefixes)

    def _vision_token_mask(
        self,
        batch_size: int,
        token_length: int,
        num_visual_tokens: int,
        num_spatial: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Token-level mask: True on spatial vision tokens (prefix/language stay False)."""
        if num_visual_tokens == num_spatial:
            start = 0
        elif num_visual_tokens == num_spatial + 1:
            start = 1
        else:
            raise ValueError(
                f"Cannot align {num_visual_tokens} visual tokens with {num_spatial} spatial patches"
            )
        end = start + num_spatial
        mask = torch.zeros(batch_size, token_length, device=device, dtype=torch.bool)
        mask[:, start:end] = True
        return mask

    def _resolve_instructions(self, batch: dict, batch_size: int) -> list[str]:
        instructions = resolve_language_batch(batch, batch_size, self.instruction)
        descriptions = _embodiment_descriptions(batch, batch_size)
        return [
            _trajectory_prompt(instruction, embodiment, successful=True, subject="robot")
            for instruction, embodiment in zip(instructions, descriptions)
        ]

    def _encode_vlm_multiview(self, batch: dict) -> EncodedObservation:
        """Encode primary view (+text) and aux views through the VLM vision tower."""
        rgb, depth, intrinsic, extrinsic, image_mask, _ = self._gather_multiview_camera_batch(batch)
        depth_valid = depth > 0
        B, V = (int(rgb.shape[0]), int(rgb.shape[1]))
        instructions = self._resolve_instructions(batch, B)
        encoded = self.vlm_encoder.encode_images_and_text(rgb, instructions, image_mask=image_mask)
        device = self._decoder_device()
        vlm_tokens = encoded.vlm_features.detach().to(device)
        vlm_vision_tokens = encoded.image_features[:, 0].detach().to(device)
        aux_frustum_vision_tokens = encoded.image_features[:, 1:].detach().to(device)
        vlm_mask = encoded.attention_mask.detach().to(device).bool()
        Hp, Wp = (int(encoded.processed_image_hw[0]), int(encoded.processed_image_hw[1]))
        merge = int(self.decoder.config.visual_pos_merge_size)
        num_spatial = Hp // merge * (Wp // merge)
        vlm_xyz, vlm_depth_valid_mask = image_hw_to_world_points(
            depth[:, 0].to(device),
            intrinsic[:, 0].to(device),
            extrinsic[:, 0].to(device),
            processed_image_hw=encoded.processed_image_hw,
        )
        vlm_depth_valid_mask = vlm_depth_valid_mask & self._resize_depth_valid(
            depth_valid[:, 0], (Hp, Wp), device=device
        )
        vlm_xyz = vlm_xyz.detach().to(dtype=vlm_tokens.dtype)
        vlm_xyz_mask = self._vision_token_mask(
            B, int(vlm_tokens.shape[1]), int(encoded.num_image_tokens), num_spatial, device=device
        )
        src_hw = (int(depth.shape[-2]), int(depth.shape[-1]))
        vlm_intrinsic = self._scale_intrinsic_to_processed(
            intrinsic[:, 0].to(device), src_hw=src_hw, dst_hw=(Hp, Wp)
        )
        vlm_extrinsic = extrinsic[:, 0].to(device=device, dtype=torch.float32)
        aux = encoded.aux_visual_inputs
        has_any_aux_view = bool(V > 1 and image_mask[:, 1:].any().item())
        if int(aux.shape[1]) == 0 or not has_any_aux_view:
            aux_visual_tokens = vlm_tokens.new_zeros(B, 0, vlm_tokens.shape[-1])
            aux_visual_xyz = vlm_tokens.new_zeros(B, 0, Hp, Wp, 3)
            aux_visual_depth_valid_mask = torch.zeros(B, 0, Hp, Wp, device=device, dtype=torch.bool)
            aux_visual_xyz_mask = torch.zeros(B, 0, device=device, dtype=torch.bool)
            aux_visual_mask = aux_visual_xyz_mask
            aux_visual_intrinsic = vlm_tokens.new_zeros(B, 0, 3, 3)
            aux_visual_extrinsic = vlm_tokens.new_zeros(B, 0, 4, 4)
            aux_frustum_vision_tokens = aux_frustum_vision_tokens[:, :0]
        else:
            aux_visual_tokens = aux.detach().to(device)
            aux_xyz_parts, aux_depth_parts, aux_xyz_mask_parts, aux_tok_mask_parts = (
                [],
                [],
                [],
                [],
            )
            aux_intr_parts, aux_extr_parts = ([], [])
            num_tokens_per_view = int(encoded.image_features.shape[2])
            for view_idx in range(1, V):
                view_xyz, view_depth_valid = image_hw_to_world_points(
                    depth[:, view_idx].to(device),
                    intrinsic[:, view_idx].to(device),
                    extrinsic[:, view_idx].to(device),
                    processed_image_hw=encoded.processed_image_hw,
                )
                view_depth_valid = view_depth_valid & self._resize_depth_valid(
                    depth_valid[:, view_idx], (Hp, Wp), device=device
                )
                view_depth_valid = (
                    view_depth_valid & image_mask[:, view_idx].to(device=device)[:, None, None]
                )
                aux_xyz_parts.append(view_xyz.detach().to(dtype=vlm_tokens.dtype))
                aux_depth_parts.append(view_depth_valid)
                aux_xyz_mask_parts.append(
                    self._vision_token_mask(
                        B, num_tokens_per_view, num_tokens_per_view, num_spatial, device=device
                    )
                )
                aux_tok_mask_parts.append(
                    image_mask[:, view_idx].to(device)[:, None].expand(B, num_tokens_per_view)
                )
                aux_intr_parts.append(
                    self._scale_intrinsic_to_processed(
                        intrinsic[:, view_idx].to(device), src_hw=src_hw, dst_hw=(Hp, Wp)
                    )
                )
                aux_extr_parts.append(extrinsic[:, view_idx].to(device=device, dtype=torch.float32))
            aux_visual_xyz = torch.stack(aux_xyz_parts, dim=1)
            aux_visual_depth_valid_mask = torch.stack(aux_depth_parts, dim=1)
            aux_visual_xyz_mask = torch.cat(aux_xyz_mask_parts, dim=1)
            aux_visual_mask = torch.cat(aux_tok_mask_parts, dim=1)
            aux_visual_intrinsic = torch.stack(aux_intr_parts, dim=1)
            aux_visual_extrinsic = torch.stack(aux_extr_parts, dim=1)
        return EncodedObservation(
            vlm_tokens=vlm_tokens,
            vlm_xyz=vlm_xyz,
            vlm_mask=vlm_mask,
            vlm_xyz_mask=vlm_xyz_mask,
            vlm_depth_valid_mask=vlm_depth_valid_mask,
            vlm_intrinsic=vlm_intrinsic,
            vlm_extrinsic=vlm_extrinsic,
            vlm_vision_tokens=vlm_vision_tokens,
            aux_visual_tokens=aux_visual_tokens,
            aux_frustum_vision_tokens=aux_frustum_vision_tokens,
            aux_visual_xyz=aux_visual_xyz,
            aux_visual_mask=aux_visual_mask,
            aux_visual_xyz_mask=aux_visual_xyz_mask,
            aux_visual_depth_valid_mask=aux_visual_depth_valid_mask,
            aux_visual_intrinsic=aux_visual_intrinsic,
            aux_visual_extrinsic=aux_visual_extrinsic,
        )

    @staticmethod
    def _resize_depth_valid(
        depth_valid: torch.Tensor, processed_image_hw: tuple[int, int], *, device: torch.device
    ) -> torch.Tensor:
        """Nearest-resize ``depth > 0`` mask to VLM input resolution ``[B,H,W]``."""
        Hp, Wp = (int(processed_image_hw[0]), int(processed_image_hw[1]))
        mask = depth_valid.to(device=device, dtype=torch.bool)
        if tuple(mask.shape[-2:]) == (Hp, Wp):
            return mask
        return (
            torch.nn.functional.interpolate(
                mask.float().unsqueeze(1), size=(Hp, Wp), mode="nearest"
            )
            .squeeze(1)
            .bool()
        )

    @staticmethod
    def _scale_intrinsic_to_processed(
        intrinsic: torch.Tensor, *, src_hw: tuple[int, int], dst_hw: tuple[int, int]
    ) -> torch.Tensor:
        """Scale K from depth/rgb resolution to VLM ``processed_image_hw``."""
        H0, W0 = (int(src_hw[0]), int(src_hw[1]))
        Hp, Wp = (int(dst_hw[0]), int(dst_hw[1]))
        intr = intrinsic.float().clone()
        if (H0, W0) == (Hp, Wp):
            return intr
        sx = float(Wp) / float(W0)
        sy = float(Hp) / float(H0)
        intr[..., 0, 0] *= sx
        intr[..., 0, 2] *= sx
        intr[..., 1, 1] *= sy
        intr[..., 1, 2] *= sy
        return intr

    def _vlm_grad_enabled(self) -> bool:
        return False

    def encode_batch(self, batch: dict) -> EncodedObservation:
        context = torch.enable_grad() if self._vlm_grad_enabled() else torch.no_grad()
        with context:
            return self._encode_vlm_multiview(batch)

    def _pack_eef_flows(self, eef_poses: torch.Tensor, eef_open: torch.Tensor) -> torch.Tensor:
        """Measured poses (xyz+quat) and opening -> decoder [B,T,2,10] (xyz+rot6d+open)."""
        open_value = eef_open
        if open_value.ndim == 3:
            open_value = open_value.unsqueeze(-1)
        rot6d = UnitasDecoder._quat_xyzw_to_rot6d(eef_poses[..., 3:7])
        return torch.cat([eef_poses[..., :3], rot6d, open_value], dim=-1)

    @staticmethod
    def _replace_invalid_history_with_current(
        history: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        """Endpoint-pad invalid history slots without changing their mask."""
        if history.ndim < 3 or valid.shape != history.shape[:3]:
            raise ValueError(
                f"history/valid shape mismatch: {tuple(history.shape)}/{tuple(valid.shape)}"
            )
        expanded = valid.bool()
        while expanded.ndim < history.ndim:
            expanded = expanded.unsqueeze(-1)
        current = history[:, -1:].expand_as(history)
        return torch.where(expanded, history, current)

    def sampling_inputs(self, batch: dict) -> dict[str, Any]:
        """History-only kwargs for ``decoder.sample`` (no future GT trajectories)."""
        required = (
            "gripper_flow_fps",
            "gripper_flow_horizon",
            "eef_flow_fps",
            "eef_flow_horizon",
            "history_robot_flows",
            "history_eef_poses",
            "history_eef_open",
            "history_robot_exists",
            "history_eef_exists",
        )
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f"batch is missing fields required for history-only sampling: {missing}")
        device = batch["history_robot_flows"].device
        gripper_mask = batch["history_robot_exists"].bool()
        eef_mask = batch["history_eef_exists"].bool()
        history_robot_flows = self._replace_invalid_history_with_current(
            batch["history_robot_flows"], gripper_mask
        )
        history_eef_poses = self._replace_invalid_history_with_current(
            batch["history_eef_poses"], eef_mask
        )
        history_eef_open = self._replace_invalid_history_with_current(
            batch["history_eef_open"], eef_mask
        )
        history_eef_flows = self._pack_eef_flows(history_eef_poses, history_eef_open)
        kwargs: dict[str, Any] = {
            "history_gripper_flows": history_robot_flows,
            "history_eef_flows": history_eef_flows,
            "gripper_mask": gripper_mask,
            "eef_mask": eef_mask,
            "gripper_flow_fps": batch["gripper_flow_fps"].to(device=device),
            "eef_flow_fps": batch["eef_flow_fps"].to(device=device),
            "gripper_flow_horizon": batch["gripper_flow_horizon"].to(device=device),
            "eef_flow_horizon": batch["eef_flow_horizon"].to(device=device),
            "decode_target": "eef",
        }
        return kwargs
