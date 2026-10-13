from __future__ import annotations
import torch
import torch.nn as nn
from unitas.models.config import DecoderConfig
from .flow_codec import FlowChunkCodec
from .mot_action_head import MoTActionHead
from .position_encoding import (
    Chunked4DRotaryEmbedding,
    FrustumRoutedHighResPatch3DPositionEmbedding,
    HighResPatch3DPositionEmbedding,
    Unified3DPositionEmbeddingMLP,
)
from .transformer import AdaLNDiTBlock, timestep_embedding


class UnitasDecoder(nn.Module):
    """Rectified-flow decoder for gripper flow and split bimanual EEF pose flow."""

    def __init__(self, config: DecoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or DecoderConfig()
        cfg = self.config
        if cfg.attn_type != "mot":
            raise ValueError("This release supports MoT checkpoints.")
        self.vlm_proj = nn.Linear(cfg.vlm_dim, cfg.hidden_size)
        self.aux_visual_proj = nn.Linear(cfg.aux_visual_dim, cfg.hidden_size)
        self.flow_codec_dim = int(cfg.flow_codec_dim)
        self.flow_chunk_codec = FlowChunkCodec(self.flow_codec_dim, coord_dim=3)
        self.register_buffer("latent_ema_mean", torch.zeros(self.flow_codec_dim), persistent=True)
        self.register_buffer("latent_ema_std", torch.ones(self.flow_codec_dim), persistent=True)
        self.flow_latent_proj_in = nn.Linear(self.flow_codec_dim, cfg.hidden_size)
        self.flow_latent_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_size), nn.Linear(cfg.hidden_size, self.flow_codec_dim)
        )
        self.position_embedding_3d = Unified3DPositionEmbeddingMLP(
            in_channels=3,
            num_pos_feats=cfg.hidden_size,
            n_freqs=cfg.position_n_freqs,
            scale=cfg.position_scale,
        )
        if cfg.use_frustum_pe_routing:
            self.visual_position_embedding = FrustumRoutedHighResPatch3DPositionEmbedding(
                vision_dim=int(cfg.frustum_vision_dim or cfg.vlm_dim),
                in_channels=3,
                num_pos_feats=cfg.hidden_size,
                n_freqs=cfg.position_n_freqs,
                scale=cfg.position_scale,
                merge_size=cfg.visual_pos_merge_size,
                num_depth_samples=cfg.frustum_pe_num_depth_samples,
                z_near=cfg.frustum_pe_z_near,
                z_far=cfg.frustum_pe_z_far,
                subpatch_valid_threshold=cfg.frustum_pe_subpatch_valid_threshold,
                premerge_vision_queries=bool(cfg.frustum_premerge_vision_queries),
            )
        else:
            self.visual_position_embedding = HighResPatch3DPositionEmbedding(
                in_channels=3,
                num_pos_feats=cfg.hidden_size,
                n_freqs=cfg.position_n_freqs,
                scale=cfg.position_scale,
                merge_size=cfg.visual_pos_merge_size,
            )
        self.uses_additive_position = cfg.position_encoding_mode in {"additive", "additive_rope4d"}
        self.uses_rope4d_position = cfg.position_encoding_mode in {"rope4d", "additive_rope4d"}
        head_dim = cfg.hidden_size // cfg.num_heads
        if self.uses_rope4d_position and head_dim % 8 != 0:
            raise ValueError(
                f"4D RoPE requires per-head dimension divisible by 8 (got head_dim={head_dim} from hidden_size={cfg.hidden_size}, num_heads={cfg.num_heads})"
            )
        self.rope4d = Chunked4DRotaryEmbedding(head_dim // 4, theta=cfg.rope_theta)
        self.timestep_mlp = nn.Sequential(
            nn.Linear(cfg.time_dim, cfg.hidden_size),
            nn.SiLU(),
            nn.LayerNorm(cfg.hidden_size),
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
        )
        self.token_type_emb = nn.Embedding(11, cfg.hidden_size)
        self.blocks = nn.ModuleList(
            [
                AdaLNDiTBlock(
                    cfg.hidden_size,
                    cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    cond_dim=cfg.hidden_size,
                )
                for _ in range(cfg.depth)
            ]
        )
        self.mot_action_head = MoTActionHead(
            main_hidden_size=cfg.hidden_size,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            time_dim=cfg.time_dim,
            max_prediction_time=cfg.max_prediction_time,
        )

    @staticmethod
    def _rescale_latent_with_ema(
        latent: torch.Tensor, ema_mean: torch.Tensor, ema_std: torch.Tensor
    ) -> torch.Tensor:
        """Standardize codec latents for flow matching: (z - mean) / std."""
        mean = ema_mean.to(device=latent.device, dtype=latent.dtype).view(1, 1, -1)
        std = ema_std.to(device=latent.device, dtype=latent.dtype).clamp(min=1e-05).view(1, 1, -1)
        return (latent - mean) / std

    @staticmethod
    def _unscale_latent_with_ema(
        latent: torch.Tensor, ema_mean: torch.Tensor, ema_std: torch.Tensor
    ) -> torch.Tensor:
        """Invert FM-space latents back to codec space: z * std + mean."""
        mean = ema_mean.to(device=latent.device, dtype=latent.dtype).view(1, 1, -1)
        std = ema_std.to(device=latent.device, dtype=latent.dtype).clamp(min=1e-05).view(1, 1, -1)
        return latent * std + mean

    @staticmethod
    def _resolve_depth_valid_mask(
        depth_valid_mask: torch.Tensor | None, xyz: torch.Tensor
    ) -> torch.Tensor:
        """Dense depth validity at VLM resolution. ``None`` → all-False (no 3D PE)."""
        if xyz.ndim == 4:
            shape = xyz.shape[:3]
        elif xyz.ndim == 5:
            shape = xyz.shape[:4]
        else:
            raise ValueError(f"dense xyz must be 4D/5D, got {tuple(xyz.shape)}")
        if depth_valid_mask is None:
            return torch.zeros(shape, dtype=torch.bool, device=xyz.device)
        mask = depth_valid_mask.to(device=xyz.device, dtype=torch.bool)
        if tuple(mask.shape) != tuple(shape):
            raise ValueError(
                f"depth_valid_mask shape {tuple(mask.shape)} does not match xyz spatial {tuple(shape)}"
            )
        return mask

    @staticmethod
    def _gather_spatial_vision_feat(
        tokens: torch.Tensor, vision_mask: torch.Tensor
    ) -> torch.Tensor:
        """Pack vision-token features ``[B,N,D]`` from a padded token sequence."""
        B, _, D = tokens.shape
        counts = vision_mask.sum(dim=-1)
        if not torch.equal(counts, counts.new_full(counts.shape, int(counts[0].item()))):
            raise ValueError("all samples must share the same spatial vision token count")
        N = int(counts[0].item())
        return tokens[vision_mask.bool()].view(B, N, D)

    def _position_bias(
        self,
        xyz: torch.Tensor | None,
        xyz_mask: torch.Tensor | None,
        name: str,
        *,
        batch: int,
        length: int,
        device: torch.device,
        dtype: torch.dtype,
        depth_valid_mask: torch.Tensor | None = None,
        vision_feat: torch.Tensor | None = None,
        intrinsic: torch.Tensor | None = None,
        extrinsic: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Returns ``(pos_bias, xyz_for_rope)``; xyz_for_rope may be hybrid frustum xyz."""
        if xyz is None:
            return (None, None)
        xyz = xyz.to(device=device)
        if xyz.ndim == 5:
            if xyz_mask is None:
                raise ValueError(f"{name} dense aux xyz requires a token-level xyz_mask")
            if xyz.shape[1] == 0 or length == 0:
                zeros = xyz.new_zeros(batch, length, self.config.hidden_size, dtype=dtype)
                return (zeros, xyz)
            depth_mask = self._resolve_depth_valid_mask(depth_valid_mask, xyz)
            pos_dtype = next(self.visual_position_embedding.parameters()).dtype
            spatial_parts = []
            xyz_parts = []
            V = int(xyz.shape[1])
            tokens_per_view = int(xyz_mask.shape[1] // max(V, 1)) if V > 0 else 0
            for view_idx in range(V):
                view_feat = None
                view_K = None
                view_E = None
                if vision_feat is not None and intrinsic is not None and (extrinsic is not None):
                    if vision_feat.ndim == 4:
                        view_feat = vision_feat[:, view_idx]
                    else:
                        sl = slice(view_idx * tokens_per_view, (view_idx + 1) * tokens_per_view)
                        view_mask = xyz_mask[:, sl]
                        view_feat = self._gather_spatial_vision_feat(vision_feat[:, sl], view_mask)
                    view_K = intrinsic[:, view_idx]
                    view_E = extrinsic[:, view_idx]
                pos_b, xyz_h = self._dense_visual_position_bias(
                    xyz[:, view_idx].to(dtype=pos_dtype),
                    xyz_mask[:, view_idx * tokens_per_view : (view_idx + 1) * tokens_per_view]
                    if tokens_per_view > 0
                    else xyz_mask,
                    depth_valid_mask=depth_mask[:, view_idx],
                    dtype=dtype,
                    vision_feat=view_feat,
                    intrinsic=view_K,
                    extrinsic=view_E,
                    scatter=False,
                )
                spatial_parts.append(pos_b)
                xyz_parts.append(xyz_h if xyz_h is not None else xyz[:, view_idx])
            spatial = torch.cat(spatial_parts, dim=1)
            bias = self._scatter_spatial_to_vision_tokens(spatial, xyz_mask.to(device=device))
            return (bias, torch.stack(xyz_parts, dim=1))
        if xyz.ndim == 4:
            if xyz_mask is None:
                raise ValueError(f"{name} dense xyz requires a token-level xyz_mask")
            return self._dense_visual_position_bias(
                xyz,
                xyz_mask.to(device=device),
                depth_valid_mask=depth_valid_mask,
                dtype=dtype,
                vision_feat=vision_feat,
                intrinsic=intrinsic,
                extrinsic=extrinsic,
                scatter=True,
            )
        finite_mask = torch.isfinite(xyz).all(dim=-1)
        if xyz_mask is not None:
            finite_mask = finite_mask & xyz_mask.to(device=device, dtype=torch.bool)
        pos_dtype = next(self.position_embedding_3d.parameters()).dtype
        clean_xyz = torch.nan_to_num(xyz, nan=0.0, posinf=0.0, neginf=0.0).to(dtype=pos_dtype)
        pos = self.position_embedding_3d(clean_xyz).to(dtype=dtype)
        return (pos * finite_mask.unsqueeze(-1).to(dtype=pos.dtype), xyz)

    def _add_3d_position(
        self,
        hidden: torch.Tensor,
        xyz: torch.Tensor | None,
        xyz_mask: torch.Tensor | None,
        name: str,
        depth_valid_mask: torch.Tensor | None = None,
        vision_feat: torch.Tensor | None = None,
        intrinsic: torch.Tensor | None = None,
        extrinsic: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        pos, xyz_rope = self._position_bias(
            xyz,
            xyz_mask,
            name,
            batch=hidden.shape[0],
            length=hidden.shape[1],
            device=hidden.device,
            dtype=hidden.dtype,
            depth_valid_mask=depth_valid_mask,
            vision_feat=vision_feat,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
        )
        if pos is not None:
            hidden = hidden + pos
        return (hidden, xyz_rope)

    @staticmethod
    def _scatter_spatial_to_vision_tokens(
        spatial: torch.Tensor, vision_mask: torch.Tensor
    ) -> torch.Tensor:
        """Scatter [B,N,*] patch features onto token slots where ``vision_mask`` is True."""
        if vision_mask.ndim != 2:
            raise ValueError(f"vision_mask must have shape [B,L], got {tuple(vision_mask.shape)}")
        B, N = (int(spatial.shape[0]), int(spatial.shape[1]))
        L = int(vision_mask.shape[1])
        if int(vision_mask.shape[0]) != B:
            raise ValueError(
                f"vision_mask batch {vision_mask.shape[0]} does not match spatial batch {B}"
            )
        counts = vision_mask.sum(dim=-1)
        expected = counts.new_full(counts.shape, N)
        if not torch.equal(counts, expected):
            raise ValueError(
                f"vision_mask True count must equal spatial tokens N={N} for every sample, got {counts.tolist()}"
            )
        out = spatial.new_zeros((B, L, *spatial.shape[2:]))
        out[vision_mask.bool()] = spatial.reshape(B * N, *spatial.shape[2:])
        return out

    def _patch_centers_from_hw(self, xyz_hw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample one xyz per merge patch from the dense VLM-resolution grid."""
        if xyz_hw.ndim != 4 or xyz_hw.shape[-1] != 3:
            raise ValueError(f"xyz_hw must have shape [B,H,W,3], got {tuple(xyz_hw.shape)}")
        B, H, W, _ = xyz_hw.shape
        merge = int(self.config.visual_pos_merge_size)
        if H % merge != 0 or W % merge != 0:
            raise ValueError(
                f"H and W must be divisible by visual_pos_merge_size={merge}, got H={H}, W={W}"
            )
        cy = merge // 2
        cx = merge // 2
        centers_hw = xyz_hw[:, cy::merge, cx::merge, :]
        center_mask = torch.isfinite(centers_hw).all(dim=-1)
        center_mask = center_mask & (centers_hw.abs().sum(dim=-1) > 0)
        centers = centers_hw.reshape(B, -1, 3)
        return (centers, center_mask.reshape(B, -1))

    def _dense_visual_position_bias(
        self,
        xyz_hw: torch.Tensor,
        vision_mask: torch.Tensor,
        *,
        depth_valid_mask: torch.Tensor | None = None,
        dtype: torch.dtype,
        vision_feat: torch.Tensor | None = None,
        intrinsic: torch.Tensor | None = None,
        extrinsic: torch.Tensor | None = None,
        scatter: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """High-res / frustum PE. Returns ``(pos_or_scattered, xyz_hw_hybrid)``."""
        depth_mask = self._resolve_depth_valid_mask(depth_valid_mask, xyz_hw)
        pos_dtype = next(self.visual_position_embedding.parameters()).dtype
        xyz_in = xyz_hw.to(dtype=pos_dtype)
        pe = self.visual_position_embedding
        if isinstance(pe, FrustumRoutedHighResPatch3DPositionEmbedding):
            pos, _, extras = pe(
                xyz_in,
                depth_mask,
                vision_feat=vision_feat.to(dtype=pos_dtype) if vision_feat is not None else None,
                intrinsic=intrinsic,
                extrinsic=extrinsic,
            )
            xyz_hybrid = extras["xyz_routed_hw"].to(dtype=xyz_hw.dtype)
        else:
            pos, _ = pe(xyz_in, depth_mask)
            xyz_hybrid = xyz_hw
        pos = pos.to(dtype=dtype)
        if scatter:
            pos = self._scatter_spatial_to_vision_tokens(pos, vision_mask)
        return (pos, xyz_hybrid)

    def _dense_visual_token_xyz(
        self,
        xyz_hw: torch.Tensor,
        vision_mask: torch.Tensor,
        depth_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Patch-center xyz scattered onto vision token slots for RoPE."""
        centers, center_valid = self._patch_centers_from_hw(xyz_hw)
        depth_mask = self._resolve_depth_valid_mask(depth_valid_mask, xyz_hw)
        merge = int(self.config.visual_pos_merge_size)
        cy, cx = (merge // 2, merge // 2)
        depth_center = depth_mask[:, cy::merge, cx::merge].reshape(centers.shape[0], -1)
        center_valid = center_valid & depth_center
        centers = torch.where(
            center_valid.unsqueeze(-1), centers, torch.full_like(centers, float("nan"))
        )
        return self._scatter_spatial_to_vision_tokens(centers, vision_mask)

    def _normalize_xyz_for_position(self, xyz: torch.Tensor) -> torch.Tensor:
        """RoPE 路径专用：把米制 xyz 换算到 rope_position_scale 指定的单位（默认毫米）。"""
        return xyz / float(self.config.rope_position_scale)

    @staticmethod
    def _rotation_matrix_to_rot6d(matrix: torch.Tensor) -> torch.Tensor:
        if matrix.shape[-2:] != (3, 3):
            raise ValueError(
                f"rotation matrix must end with shape [3,3], got {tuple(matrix.shape)}"
            )
        return torch.cat([matrix[..., :, 0], matrix[..., :, 1]], dim=-1)

    @staticmethod
    def _rot6d_to_rotation_matrix(rot6d: torch.Tensor) -> torch.Tensor:
        """Gram-Schmidt project a 6D rotation prediction onto SO(3)."""
        if rot6d.shape[-1] != 6:
            raise ValueError(f"rot6d must end with 6 channels, got {tuple(rot6d.shape)}")
        value = rot6d.float()
        a1, a2 = (value[..., :3], value[..., 3:6])
        a1_norm = a1.norm(dim=-1, keepdim=True)
        fallback_x = torch.zeros_like(a1)
        fallback_x[..., 0] = 1.0
        b1 = torch.where(a1_norm > 1e-06, a1 / a1_norm.clamp(min=1e-06), fallback_x)
        a2_orthogonal = a2 - (a2 * b1).sum(dim=-1, keepdim=True) * b1
        a2_norm = a2_orthogonal.norm(dim=-1, keepdim=True)
        fallback_axis = torch.zeros_like(b1)
        use_x = b1[..., 0:1].abs() < 0.9
        fallback_axis[..., 0] = use_x.squeeze(-1).to(dtype=fallback_axis.dtype)
        fallback_axis[..., 1] = (~use_x).squeeze(-1).to(dtype=fallback_axis.dtype)
        fallback_b2 = fallback_axis - (fallback_axis * b1).sum(dim=-1, keepdim=True) * b1
        fallback_b2 = fallback_b2 / fallback_b2.norm(dim=-1, keepdim=True).clamp(min=1e-06)
        b2 = torch.where(a2_norm > 1e-06, a2_orthogonal / a2_norm.clamp(min=1e-06), fallback_b2)
        b3 = torch.linalg.cross(b1, b2, dim=-1)
        return torch.stack([b1, b2, b3], dim=-1)

    @staticmethod
    def _rotation_matrix_to_quat_xyzw(matrix: torch.Tensor) -> torch.Tensor:
        """Convert an SO(3) matrix to a normalized quaternion in xyzw order."""
        if matrix.shape[-2:] != (3, 3):
            raise ValueError(
                f"rotation matrix must end with shape [3,3], got {tuple(matrix.shape)}"
            )
        m00, m01, m02 = (matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2])
        m10, m11, m12 = (matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2])
        m20, m21, m22 = (matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2])
        q_abs = torch.sqrt(
            torch.clamp(
                torch.stack(
                    [
                        1.0 + m00 + m11 + m22,
                        1.0 + m00 - m11 - m22,
                        1.0 - m00 + m11 - m22,
                        1.0 - m00 - m11 + m22,
                    ],
                    dim=-1,
                ),
                min=0.0,
            )
        )
        quat_candidates = torch.stack(
            [
                torch.stack([q_abs[..., 0].square(), m21 - m12, m02 - m20, m10 - m01], dim=-1),
                torch.stack([m21 - m12, q_abs[..., 1].square(), m10 + m01, m02 + m20], dim=-1),
                torch.stack([m02 - m20, m10 + m01, q_abs[..., 2].square(), m12 + m21], dim=-1),
                torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3].square()], dim=-1),
            ],
            dim=-2,
        )
        quat_candidates = quat_candidates / (2.0 * q_abs[..., :, None].clamp(min=0.1))
        best = q_abs.argmax(dim=-1)
        gather_index = best[..., None, None].expand(*best.shape, 1, 4)
        quat_wxyz = quat_candidates.gather(-2, gather_index).squeeze(-2)
        quat_wxyz = quat_wxyz / quat_wxyz.norm(dim=-1, keepdim=True).clamp(min=1e-08)
        return torch.cat([quat_wxyz[..., 1:], quat_wxyz[..., :1]], dim=-1)

    def _eef_rot_to_relative_chunks(self, eef_rot6d: torch.Tensor, anchor_rot6d: torch.Tensor):
        """
        eef_rot6d: [B, T, 2, 6]
        anchor_rot6d: [B, 2, 6]
        """
        target_rotation = self._rot6d_to_rotation_matrix(eef_rot6d)
        anchor_rotation = self._rot6d_to_rotation_matrix(anchor_rot6d)
        relative_rotation = anchor_rotation[:, None].transpose(-1, -2) @ target_rotation
        relative_rot6d = self._rotation_matrix_to_rot6d(relative_rotation)
        identity_rot6d = self._rotation_matrix_to_rot6d(
            torch.eye(3, device=eef_rot6d.device, dtype=relative_rot6d.dtype)
        )
        rot6d_residual = relative_rot6d - identity_rot6d
        return rot6d_residual

    def _eef_relative_chunks_to_pose_open(
        self,
        xyz_chunk: torch.Tensor,
        rotgrip_chunk: torch.Tensor,
        xyz0: torch.Tensor,
        rotgrip0: torch.Tensor,
        *,
        open_is_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compose per-arm relative trajectories with their absolute frame-0 anchors."""
        if xyz_chunk.ndim != 4 or xyz_chunk.shape[-1] != 3:
            raise ValueError(f"xyz_chunk must have shape [B,2,H,3], got {tuple(xyz_chunk.shape)}")
        B, arms, H, _ = xyz_chunk.shape
        if arms != 2 or rotgrip_chunk.shape != (B, 2, H, 7):
            raise ValueError(
                f"rotgrip_chunk must have shape {(B, 2, H, 7)}, got {tuple(rotgrip_chunk.shape)}"
            )
        if xyz0.shape != (B, 2, 3) or rotgrip0.shape != (B, 2, 7):
            raise ValueError(
                f"EEF anchors must have shapes {(B, 2, 3)} and {(B, 2, 7)}, got {tuple(xyz0.shape)} and {tuple(rotgrip0.shape)}"
            )
        absolute_xyz = xyz0[:, :, None] + xyz_chunk
        identity_rot6d = self._rotation_matrix_to_rot6d(
            torch.eye(3, device=rotgrip_chunk.device, dtype=rotgrip_chunk.dtype)
        )
        relative_rotation = self._rot6d_to_rotation_matrix(rotgrip_chunk[..., :6] + identity_rot6d)
        anchor_rotation = self._rot6d_to_rotation_matrix(rotgrip0[..., :6])
        absolute_rotation = anchor_rotation[:, :, None] @ relative_rotation
        absolute_quat = self._rotation_matrix_to_quat_xyzw(absolute_rotation)
        open_state = rotgrip_chunk[..., 6:7]
        absolute_open = torch.sigmoid(open_state) if open_is_logits else open_state.clamp(0.0, 1.0)
        pose = (
            torch.cat([absolute_xyz.float(), absolute_quat], dim=-1)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        return (pose, absolute_open.permute(0, 2, 1, 3).contiguous().float())

    def _unpack_mot_action(
        self, action: torch.Tensor, eef_xyz0: torch.Tensor, eef_rotgrip0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert sampled Hx20 action tokens to the existing policy outputs."""
        if action.ndim != 3 or action.shape[-1] != 20:
            raise ValueError(f"MoT action must have shape [B,H,20], got {tuple(action.shape)}")
        batch, horizon, _ = action.shape
        per_arm = action.view(batch, horizon, 2, 10)
        xyz_flow = per_arm[..., :3] / float(self.config.mot_xyz_scale)
        rot_flow = per_arm[..., 3:9] / float(self.config.mot_rotation_scale)
        gripper = per_arm[..., 9:10] / float(self.config.mot_gripper_scale)
        rotgrip_flow = torch.cat([rot_flow, gripper], dim=-1)
        poses, open_state = self._eef_relative_chunks_to_pose_open(
            xyz_flow.permute(0, 2, 1, 3).contiguous(),
            rotgrip_flow.permute(0, 2, 1, 3).contiguous(),
            eef_xyz0,
            eef_rotgrip0,
            open_is_logits=False,
        )
        return (xyz_flow, rotgrip_flow, poses, open_state)

    def _encode_context_hidden(
        self,
        vlm_tokens: torch.Tensor,
        aux_visual_tokens: torch.Tensor,
        vlm_xyz: torch.Tensor,
        aux_visual_xyz: torch.Tensor,
        vlm_xyz_mask: torch.Tensor | None,
        aux_visual_xyz_mask: torch.Tensor | None,
        vlm_depth_valid_mask: torch.Tensor | None = None,
        aux_visual_depth_valid_mask: torch.Tensor | None = None,
        vlm_intrinsic: torch.Tensor | None = None,
        vlm_extrinsic: torch.Tensor | None = None,
        aux_visual_intrinsic: torch.Tensor | None = None,
        aux_visual_extrinsic: torch.Tensor | None = None,
        vlm_vision_tokens: torch.Tensor | None = None,
        aux_frustum_vision_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns projected context hiddens and (possibly frustum-filled) xyz grids."""
        vlm_hidden = self.vlm_proj(vlm_tokens)
        aux_visual_hidden = self.aux_visual_proj(aux_visual_tokens)
        vlm_xyz_out, aux_xyz_out = (vlm_xyz, aux_visual_xyz)
        if self.uses_additive_position:
            if vlm_vision_tokens is not None and vlm_vision_tokens.numel() > 0:
                vlm_feat = vlm_vision_tokens
            else:
                vlm_feat = (
                    None
                    if vlm_xyz_mask is None
                    else self._gather_spatial_vision_feat(vlm_tokens, vlm_xyz_mask)
                )
            vlm_hidden, vlm_xyz_out = self._add_3d_position(
                vlm_hidden,
                vlm_xyz,
                vlm_xyz_mask,
                "vlm_xyz",
                depth_valid_mask=vlm_depth_valid_mask,
                vision_feat=vlm_feat,
                intrinsic=vlm_intrinsic,
                extrinsic=vlm_extrinsic,
            )
            if vlm_xyz_out is None:
                vlm_xyz_out = vlm_xyz
            if int(aux_visual_hidden.shape[1]) > 0:
                aux_frustum_feat = (
                    aux_frustum_vision_tokens
                    if aux_frustum_vision_tokens is not None
                    and aux_frustum_vision_tokens.numel() > 0
                    else aux_visual_tokens
                )
                aux_visual_hidden, aux_xyz_out = self._add_3d_position(
                    aux_visual_hidden,
                    aux_visual_xyz,
                    aux_visual_xyz_mask,
                    "aux_visual_xyz",
                    depth_valid_mask=aux_visual_depth_valid_mask,
                    vision_feat=aux_frustum_feat,
                    intrinsic=aux_visual_intrinsic,
                    extrinsic=aux_visual_extrinsic,
                )
                if aux_xyz_out is None:
                    aux_xyz_out = aux_visual_xyz
        return (vlm_hidden, aux_visual_hidden, vlm_xyz_out, aux_xyz_out)

    def _gripper_latent_to_hidden(
        self,
        latent: torch.Tensor,
        gripper_xyz0: torch.Tensor | None,
        gripper_xyz_mask: torch.Tensor | None,
        token_type_idx: int = 1,
        position_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.flow_latent_proj_in(latent)
        if self.uses_additive_position:
            if position_bias is not None:
                hidden = hidden + position_bias
            elif gripper_xyz0 is not None:
                hidden, _ = self._add_3d_position(
                    hidden, gripper_xyz0, gripper_xyz_mask, "gripper_xyz0"
                )
        return hidden + self.token_type_emb.weight[int(token_type_idx)].view(1, 1, -1)

    def _scene_latent_to_hidden(
        self,
        latent: torch.Tensor,
        scene_xyz: torch.Tensor | None,
        scene_flow_mask: torch.Tensor | None,
        *,
        position_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.flow_latent_proj_in(latent)
        if self.uses_additive_position:
            if position_bias is not None:
                hidden = hidden + position_bias
            elif scene_xyz is not None:
                hidden, _ = self._add_3d_position(hidden, scene_xyz, scene_flow_mask, "scene_xyz")
        return hidden + self.token_type_emb.weight[2].view(1, 1, -1)

    def _spatial_rope_position_ids(
        self,
        xyz: torch.Tensor | None,
        mask: torch.Tensor | None,
        length: int,
        batch: int,
        modality_id: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        default_rope_index = torch.zeros(batch, length, 4, device=device, dtype=dtype)
        default_rope_index[..., -1] = float(modality_id)
        if xyz is None or mask is None:
            return default_rope_index
        xyz = xyz.to(device=device, dtype=dtype)
        finite_mask = torch.isfinite(xyz).all(dim=-1)
        finite_mask = finite_mask & mask.to(device=device, dtype=torch.bool)
        xyz_n = self._normalize_xyz_for_position(
            torch.nan_to_num(xyz, nan=0.0, posinf=0.0, neginf=0.0)
        )
        spatial = torch.cat(
            [xyz_n, torch.full((batch, length, 1), float(modality_id), device=device, dtype=dtype)],
            dim=-1,
        )
        return torch.where(finite_mask.unsqueeze(-1), spatial, default_rope_index)

    def _vlm_rope_position_ids(
        self,
        xyz: torch.Tensor | None,
        xyz_mask: torch.Tensor | None,
        length: int,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if xyz_mask is not None:
            image_len = xyz_mask.sum(dim=-1)
        else:
            xyz_mask = torch.zeros(batch, length, dtype=torch.bool, device=device)
            image_len = 0
        language_len = (length - image_len).clamp(min=0)
        default_rope_index = self._spatial_rope_position_ids(
            xyz=xyz,
            mask=xyz_mask,
            length=length,
            batch=batch,
            modality_id=0.0,
            device=device,
            dtype=dtype,
        )
        language_pos = (
            torch.arange(language_len.max(), device=device, dtype=dtype)[None, :, None].expand(
                batch, -1, -1
            )
            + 1.0
        )
        language_rope = torch.cat(
            [torch.zeros(batch, language_len.max(), 3, device=device, dtype=dtype), language_pos],
            dim=-1,
        )
        default_rope_index[~xyz_mask] = torch.cat(
            [language_rope[idx, :length, :] for idx, length in enumerate(language_len)], dim=0
        )
        return default_rope_index

    def build_sequence(
        self,
        vlm_tokens: torch.Tensor,
        aux_visual_tokens: torch.Tensor | None,
        vlm_mask: torch.Tensor,
        aux_visual_mask: torch.Tensor | None,
        history_gripper_tokens: torch.Tensor,
        history_gripper_mask: torch.Tensor,
        gripper_xt: torch.Tensor,
        gripper_mask: torch.Tensor,
        vlm_xyz: torch.Tensor,
        aux_visual_xyz: torch.Tensor | None,
        gripper_xyz0: torch.Tensor,
        vlm_xyz_mask: torch.Tensor,
        aux_visual_xyz_mask: torch.Tensor | None,
        gripper_xyz0_mask: torch.Tensor,
        gripper_x0: torch.Tensor,
        scene_flow_xt: torch.Tensor | None = None,
        scene_flow_mask: torch.Tensor | None = None,
        scene_xyz: torch.Tensor | None = None,
        vlm_depth_valid_mask: torch.Tensor | None = None,
        aux_visual_depth_valid_mask: torch.Tensor | None = None,
        vlm_intrinsic: torch.Tensor | None = None,
        vlm_extrinsic: torch.Tensor | None = None,
        aux_visual_intrinsic: torch.Tensor | None = None,
        aux_visual_extrinsic: torch.Tensor | None = None,
        vlm_vision_tokens: torch.Tensor | None = None,
        aux_frustum_vision_tokens: torch.Tensor | None = None,
        return_rope_position_ids: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, dict[str, slice]]
        | tuple[torch.Tensor, torch.Tensor, dict[str, slice], torch.Tensor]
    ):
        """Pack visual context, gripper history, noisy/clean gripper and scene tokens.

        EEF actions use a separate MoT head and do not add tokens to this sequence.
        """
        device = vlm_tokens.device
        B = int(vlm_tokens.shape[0])
        if aux_visual_tokens is None:
            aux_visual_tokens = vlm_tokens.new_zeros(B, 0, int(self.aux_visual_proj.in_features))
        U = int(aux_visual_tokens.shape[1])
        if U == 0:
            aux_visual_mask = torch.zeros(B, 0, device=device, dtype=torch.bool)
            aux_visual_xyz = vlm_tokens.new_zeros(B, 0, 3)
            aux_visual_xyz_mask = torch.zeros(B, 0, device=device, dtype=torch.bool)
        elif aux_visual_mask is None or aux_visual_xyz is None or aux_visual_xyz_mask is None:
            raise ValueError("non-empty aux_visual_tokens require aux masks and xyz")
        V = vlm_tokens.shape[1]
        N_history = history_gripper_tokens.shape[1]
        N_gripper = gripper_xt.shape[1]
        N_clean = gripper_x0.shape[1]
        N_sf = 0 if scene_flow_xt is None else scene_flow_xt.shape[1]
        if not N_history == N_gripper == N_clean:
            raise ValueError("History, noisy and clean gripper token counts must match")
        vlm_hidden, aux_visual_hidden, vlm_xyz, aux_visual_xyz = self._encode_context_hidden(
            vlm_tokens,
            aux_visual_tokens,
            vlm_xyz,
            aux_visual_xyz,
            vlm_xyz_mask,
            aux_visual_xyz_mask,
            vlm_depth_valid_mask=vlm_depth_valid_mask,
            aux_visual_depth_valid_mask=aux_visual_depth_valid_mask,
            vlm_intrinsic=vlm_intrinsic,
            vlm_extrinsic=vlm_extrinsic,
            aux_visual_intrinsic=aux_visual_intrinsic,
            aux_visual_extrinsic=aux_visual_extrinsic,
            vlm_vision_tokens=vlm_vision_tokens,
            aux_frustum_vision_tokens=aux_frustum_vision_tokens,
        )
        history_gripper_hidden = self._gripper_latent_to_hidden(
            history_gripper_tokens, gripper_xyz0, gripper_xyz0_mask, token_type_idx=0
        )
        gripper_hidden = self._gripper_latent_to_hidden(
            gripper_xt, gripper_xyz0, gripper_xyz0_mask, token_type_idx=1
        )
        gripper_clean_hidden = self._gripper_latent_to_hidden(
            gripper_x0, gripper_xyz0, gripper_xyz0_mask
        )
        gripper_clean_mask = gripper_mask
        if scene_flow_xt is None:
            scene_hidden = gripper_hidden.new_zeros(B, 0, self.config.hidden_size)
            scene_xyz = None
            scene_flow_mask = gripper_mask.new_zeros(B, 0)
        else:
            scene_hidden = self._scene_latent_to_hidden(scene_flow_xt, scene_xyz, scene_flow_mask)
        sequence = torch.cat(
            [
                vlm_hidden,
                aux_visual_hidden,
                history_gripper_hidden,
                gripper_hidden,
                gripper_clean_hidden,
                scene_hidden,
            ],
            dim=1,
        )
        sequence_mask = torch.cat(
            [
                vlm_mask,
                aux_visual_mask,
                history_gripper_mask,
                gripper_mask,
                gripper_clean_mask,
                scene_flow_mask,
            ],
            dim=1,
        )
        offset = V + U
        gripper_noisy_start = offset + N_history
        gripper_clean_start = gripper_noisy_start + N_gripper
        scene_start = gripper_clean_start + N_clean
        slices = {
            "vlm": slice(0, V),
            "aux_visual": slice(V, offset),
            "history_gripper": slice(offset, offset + N_history),
            "gripper_noisy": slice(gripper_noisy_start, gripper_noisy_start + N_gripper),
            "gripper_clean": slice(gripper_clean_start, gripper_clean_start + N_clean),
            "scene_flow": slice(scene_start, scene_start + N_sf),
        }
        if not return_rope_position_ids:
            return (sequence, sequence_mask, slices)
        rope_dtype = sequence.dtype
        if vlm_xyz.ndim == 4:
            if vlm_xyz_mask is None:
                raise ValueError("vlm_xyz_mask is required when vlm_xyz is a dense [B,H,W,3] grid")
            vlm_xyz_tok = self._dense_visual_token_xyz(
                vlm_xyz, vlm_xyz_mask, depth_valid_mask=vlm_depth_valid_mask
            )
            vlm_xyz_mask_tok = vlm_xyz_mask
        else:
            vlm_xyz_tok = vlm_xyz
            vlm_xyz_mask_tok = vlm_xyz_mask
        vlm_rope = self._vlm_rope_position_ids(
            xyz=vlm_xyz_tok,
            xyz_mask=vlm_xyz_mask_tok,
            length=V,
            batch=B,
            device=device,
            dtype=rope_dtype,
        )
        if U == 0:
            aux_xyz_tok = None
            aux_mask_tok = aux_visual_xyz_mask
        elif aux_visual_xyz.ndim == 5:
            if aux_visual_xyz_mask is None:
                raise ValueError(
                    "aux_visual_xyz_mask is required when aux_visual_xyz is dense [B,V,H,W,3]"
                )
            if aux_visual_xyz.shape[1] == 0:
                aux_xyz_tok = aux_visual_hidden.new_zeros(B, U, 3)
                aux_mask_tok = aux_visual_xyz_mask
            else:
                aux_depth = self._resolve_depth_valid_mask(
                    aux_visual_depth_valid_mask, aux_visual_xyz
                )
                merge = int(self.config.visual_pos_merge_size)
                cy, cx = (merge // 2, merge // 2)
                center_parts = []
                for view_idx in range(int(aux_visual_xyz.shape[1])):
                    centers, center_valid = self._patch_centers_from_hw(aux_visual_xyz[:, view_idx])
                    depth_center = aux_depth[:, view_idx, cy::merge, cx::merge].reshape(
                        centers.shape[0], -1
                    )
                    center_valid = center_valid & depth_center
                    centers = torch.where(
                        center_valid.unsqueeze(-1), centers, torch.full_like(centers, float("nan"))
                    )
                    center_parts.append(centers)
                aux_xyz_tok = self._scatter_spatial_to_vision_tokens(
                    torch.cat(center_parts, dim=1), aux_visual_xyz_mask
                )
                aux_mask_tok = aux_visual_xyz_mask
        else:
            aux_xyz_tok = aux_visual_xyz
            aux_mask_tok = aux_visual_xyz_mask
        aux_visual_rope = self._spatial_rope_position_ids(
            xyz=aux_xyz_tok,
            mask=aux_mask_tok,
            length=U,
            batch=B,
            modality_id=0.0,
            device=device,
            dtype=rope_dtype,
        )
        history_rope = self._spatial_rope_position_ids(
            xyz=gripper_xyz0,
            mask=gripper_xyz0_mask,
            length=N_history,
            batch=B,
            modality_id=-1.0,
            device=device,
            dtype=rope_dtype,
        )
        gripper_rope = self._spatial_rope_position_ids(
            xyz=gripper_xyz0,
            mask=gripper_xyz0_mask,
            length=N_gripper,
            batch=B,
            modality_id=-2.0,
            device=device,
            dtype=rope_dtype,
        )
        scene_rope = self._spatial_rope_position_ids(
            xyz=scene_xyz,
            mask=scene_flow_mask,
            length=N_sf,
            batch=B,
            modality_id=-3.0,
            device=device,
            dtype=rope_dtype,
        )
        gripper_clean_rope = gripper_rope[:, :N_clean]
        rope_position_ids = torch.cat(
            [
                vlm_rope,
                aux_visual_rope,
                history_rope,
                gripper_rope,
                gripper_clean_rope,
                scene_rope,
            ],
            dim=1,
        )
        rope_position_ids = rope_position_ids * sequence_mask.unsqueeze(-1).to(
            dtype=rope_position_ids.dtype
        )
        return (sequence, sequence_mask, slices, rope_position_ids)

    def forward(
        self,
        vlm_tokens: torch.Tensor,
        aux_visual_tokens: torch.Tensor | None,
        vlm_mask: torch.Tensor,
        aux_visual_mask: torch.Tensor | None,
        history_gripper_tokens: torch.Tensor,
        history_gripper_mask: torch.Tensor,
        gripper_xt: torch.Tensor,
        gripper_mask: torch.Tensor,
        vlm_xyz: torch.Tensor,
        aux_visual_xyz: torch.Tensor | None,
        gripper_xyz0: torch.Tensor,
        vlm_xyz_mask: torch.Tensor,
        aux_visual_xyz_mask: torch.Tensor | None,
        gripper_xyz0_mask: torch.Tensor,
        timestep: torch.Tensor,
        gripper_x0: torch.Tensor,
        scene_flow_xt: torch.Tensor | None = None,
        scene_flow_mask: torch.Tensor | None = None,
        scene_xyz: torch.Tensor | None = None,
        scene_timestep: torch.Tensor | None = None,
        vlm_depth_valid_mask: torch.Tensor | None = None,
        aux_visual_depth_valid_mask: torch.Tensor | None = None,
        vlm_intrinsic: torch.Tensor | None = None,
        vlm_extrinsic: torch.Tensor | None = None,
        aux_visual_intrinsic: torch.Tensor | None = None,
        aux_visual_extrinsic: torch.Tensor | None = None,
        vlm_vision_tokens: torch.Tensor | None = None,
        aux_frustum_vision_tokens: torch.Tensor | None = None,
        return_mot_kv: bool = False,
        return_scene_kv: bool = False,
        return_context_kv: bool = False,
        skip_clean: bool = False,
        skip_noisy: bool = False,
        context_kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        context_key_padding_mask: torch.Tensor | None = None,
        prefix_kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        prefix_key_padding_mask: torch.Tensor | None = None,
        _sequence_cache: dict | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Predict the rectified-flow vector field in gripper-flow state space."""
        cached = None if _sequence_cache is None else _sequence_cache.get("prepared")
        if cached is None:
            sequence_out = self.build_sequence(
                vlm_tokens=vlm_tokens,
                aux_visual_tokens=aux_visual_tokens,
                vlm_mask=vlm_mask,
                aux_visual_mask=aux_visual_mask,
                history_gripper_tokens=history_gripper_tokens,
                history_gripper_mask=history_gripper_mask,
                gripper_xt=gripper_xt,
                gripper_mask=gripper_mask,
                vlm_xyz=vlm_xyz,
                aux_visual_xyz=aux_visual_xyz,
                gripper_xyz0=gripper_xyz0,
                vlm_xyz_mask=vlm_xyz_mask,
                aux_visual_xyz_mask=aux_visual_xyz_mask,
                gripper_xyz0_mask=gripper_xyz0_mask,
                vlm_depth_valid_mask=vlm_depth_valid_mask,
                aux_visual_depth_valid_mask=aux_visual_depth_valid_mask,
                vlm_intrinsic=vlm_intrinsic,
                vlm_extrinsic=vlm_extrinsic,
                aux_visual_intrinsic=aux_visual_intrinsic,
                aux_visual_extrinsic=aux_visual_extrinsic,
                vlm_vision_tokens=vlm_vision_tokens,
                aux_frustum_vision_tokens=aux_frustum_vision_tokens,
                gripper_x0=gripper_x0,
                scene_flow_xt=scene_flow_xt,
                scene_flow_mask=scene_flow_mask,
                scene_xyz=scene_xyz,
                return_rope_position_ids=self.uses_rope4d_position,
            )
            if self.uses_rope4d_position:
                x, mask, slices, rope_position_ids = sequence_out
                position_embeddings = self.rope4d(rope_position_ids, dtype=x.dtype)
            else:
                x, mask, slices = sequence_out
                position_embeddings = None
            if _sequence_cache is not None:
                if self.training:
                    raise ValueError("Sequence caching is available only in evaluation mode")
                _sequence_cache["prepared"] = (x, mask, slices, position_embeddings)
        else:
            base, mask, slices, position_embeddings = cached
            x = base.clone()
            x[:, slices["gripper_noisy"]] = self._gripper_latent_to_hidden(
                gripper_xt, gripper_xyz0, gripper_xyz0_mask, token_type_idx=1
            )
            x[:, slices["gripper_clean"]] = self._gripper_latent_to_hidden(
                gripper_x0, gripper_xyz0, gripper_xyz0_mask
            )
            if scene_flow_xt is not None:
                x[:, slices["scene_flow"]] = self._scene_latent_to_hidden(
                    scene_flow_xt, scene_xyz, scene_flow_mask
                )
        timestep = timestep.to(device=x.device, dtype=x.dtype)
        if scene_timestep is None:
            scene_timestep = timestep
        else:
            scene_timestep = scene_timestep.to(device=x.device, dtype=x.dtype)
        timestep_cond = self.timestep_mlp(
            timestep_embedding(timestep * 1000.0, self.config.time_dim)
        )
        scene_timestep_cond = self.timestep_mlp(
            timestep_embedding(scene_timestep * 1000.0, self.config.time_dim)
        )
        clean_cond = self.timestep_mlp(
            timestep_embedding(torch.ones_like(timestep) * 1000.0, self.config.time_dim)
        )
        no_timestep_cond = torch.zeros_like(clean_cond)
        if sum((return_context_kv, return_mot_kv, return_scene_kv)) > 1:
            raise ValueError("KV return modes are mutually exclusive")
        if context_kv_cache is not None and len(context_kv_cache) != len(self.blocks):
            raise ValueError(
                f"context_kv_cache depth {len(context_kv_cache)} != {len(self.blocks)}"
            )
        context_slice = slice(slices["vlm"].start, slices["history_gripper"].stop)
        action_static_len = slices["history_gripper"].start - context_slice.start
        context_cond_table = torch.stack([no_timestep_cond, clean_cond], dim=1)
        context_cond_index = torch.zeros(
            context_slice.stop - context_slice.start, dtype=torch.long, device=x.device
        )
        noisy_slice = slices["gripper_noisy"]
        clean_slice = slices["gripper_clean"]
        scene_slice = slices["scene_flow"]
        mot_kv_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        mot_main_mask = None
        if prefix_kv_cache is not None:
            x_scene = x[:, scene_slice]
            mask_scene = mask[:, scene_slice]
            if position_embeddings is None:
                rope_scene = None
            else:
                cos, sin = position_embeddings
                rope_scene = (cos[:, scene_slice], sin[:, scene_slice])
            for block, (key, value) in zip(self.blocks, prefix_kv_cache):
                x_scene = block(
                    x_scene,
                    scene_timestep_cond,
                    key_padding_mask=mask_scene,
                    position_embeddings=rope_scene,
                    past_kv=(key, value),
                    past_key_padding_mask=prefix_key_padding_mask,
                )
            scene_vf = self.flow_latent_head(x_scene)
            gripper_vf = gripper_xt.new_zeros(gripper_xt.shape)
            return (gripper_vf, scene_vf)
        x_ctx = x[:, context_slice]
        x_noisy = x[:, noisy_slice]
        x_clean = x[:, clean_slice]
        x_scene = x[:, scene_slice]
        mask_ctx = (
            context_key_padding_mask
            if context_key_padding_mask is not None
            else mask[:, context_slice]
        )
        mask_noisy = mask[:, noisy_slice]
        mask_clean = mask[:, clean_slice]
        mask_scene = mask[:, scene_slice]
        mask_scene_pre = torch.cat([mask_ctx, mask_clean], dim=1)
        mask_action_pre = mask_ctx[:, :action_static_len]
        mot_main_mask = mask_action_pre if return_mot_kv else mask_scene_pre
        head_dim = self.config.hidden_size // self.config.num_heads
        if position_embeddings is None:
            rope_ctx = rope_noisy = rope_clean = rope_scene = None
        else:
            cos, sin = position_embeddings
            rope_ctx = (cos[:, context_slice], sin[:, context_slice])
            rope_noisy = (cos[:, noisy_slice], sin[:, noisy_slice])
            rope_clean = (cos[:, clean_slice], sin[:, clean_slice])
            rope_scene = (cos[:, scene_slice], sin[:, scene_slice])
        if return_context_kv:
            context_kv_out: list[tuple[torch.Tensor, torch.Tensor]] = []
            for block in self.blocks:
                x_ctx, layer_kv = block(
                    x_ctx,
                    context_cond_table,
                    key_padding_mask=mask_ctx,
                    position_embeddings=rope_ctx,
                    cond_index=context_cond_index,
                    return_kv=True,
                )
                context_kv_out.append(layer_kv)
            gripper_vf = gripper_xt.new_zeros(gripper_xt.shape)
            scene_vf = gripper_xt.new_zeros(gripper_xt.shape[0], 0, self.flow_codec_dim)
            return (gripper_vf, scene_vf, context_kv_out, mask_ctx)
        for start, block in enumerate(self.blocks):
            cached_ctx_kv = None if context_kv_cache is None else context_kv_cache[start]

            if cached_ctx_kv is not None:
                k_ctx, v_ctx = cached_ctx_kv
            elif x_ctx.shape[1] > 0:
                x_ctx, (k_ctx, v_ctx) = block(
                    x_ctx,
                    context_cond_table,
                    key_padding_mask=mask_ctx,
                    position_embeddings=rope_ctx,
                    cond_index=context_cond_index,
                    return_kv=True,
                )
            else:
                k_ctx = x_ctx.new_zeros(x_ctx.shape[0], self.config.num_heads, 0, head_dim)
                v_ctx = k_ctx
            if not skip_clean and x_clean.shape[1] > 0:
                x_clean, (k_clean, v_clean) = block(
                    x_clean,
                    clean_cond,
                    key_padding_mask=mask_clean,
                    position_embeddings=rope_clean,
                    past_kv=(k_ctx, v_ctx),
                    past_key_padding_mask=mask_ctx,
                    return_kv=True,
                )
            else:
                k_clean = x_clean.new_zeros(x_clean.shape[0], self.config.num_heads, 0, head_dim)
                v_clean = k_clean
            if not skip_noisy and x_noisy.shape[1] > 0:
                x_noisy, _ = block(
                    x_noisy,
                    timestep_cond,
                    key_padding_mask=mask_noisy,
                    position_embeddings=rope_noisy,
                    past_kv=(k_ctx, v_ctx),
                    past_key_padding_mask=mask_ctx,
                    return_kv=True,
                )
            scene_key = torch.cat([k_ctx, k_clean], dim=2)
            scene_value = torch.cat([v_ctx, v_clean], dim=2)
            if x_scene.shape[1] > 0:
                x_scene = block(
                    x_scene,
                    scene_timestep_cond,
                    key_padding_mask=mask_scene,
                    position_embeddings=rope_scene,
                    past_kv=(scene_key, scene_value),
                    past_key_padding_mask=mask_scene_pre,
                )
            action_key = k_ctx[:, :, :action_static_len]
            action_value = v_ctx[:, :, :action_static_len]
            if return_mot_kv:
                mot_kv_cache.append((action_key, action_value))
            elif return_scene_kv:
                mot_kv_cache.append((scene_key, scene_value))
        x = torch.cat([x_ctx, x_noisy, x_clean, x_scene], dim=1)
        gripper_vf = self.flow_latent_head(x[:, slices["gripper_noisy"]])
        scene_vf = self.flow_latent_head(x[:, slices["scene_flow"]])
        if return_mot_kv or return_scene_kv:
            return (gripper_vf, scene_vf, mot_kv_cache, mot_main_mask)
        return (gripper_vf, scene_vf)

    @staticmethod
    def _current_token_mask(mask: torch.Tensor) -> torch.Tensor:
        """Accept ``[B, N]`` or ``[B, T, N]``; for temporal masks use the last frame (current)."""
        if mask.ndim == 2:
            return mask.to(dtype=torch.bool)
        if mask.ndim == 3:
            return mask[:, -1].to(dtype=torch.bool)
        raise ValueError(f"token mask must have 2 or 3 dims, got {tuple(mask.shape)}")

    def _prepare_sample_gripper_from_history(
        self,
        history_gripper_flows: torch.Tensor,
        gripper_mask: torch.Tensor,
        gripper_flow_fps: torch.Tensor,
        gripper_flow_horizon: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Anchor at history last frame (= current pose). Returns xyz0, history_rel, mask, fps, horizon."""
        if history_gripper_flows.ndim != 4 or history_gripper_flows.shape[-1] != 3:
            raise ValueError(
                f"history_gripper_flows must be [B,T,N,3], got {tuple(history_gripper_flows.shape)}"
            )
        if history_gripper_flows.shape[1] < 2:
            raise ValueError("history_gripper_flows needs at least 2 timesteps (past + current)")
        token_mask = self._current_token_mask(gripper_mask)
        flow_dtype = history_gripper_flows.dtype
        history_gripper_flows = history_gripper_flows * token_mask[:, None, :, None].to(
            dtype=flow_dtype
        )
        xyz0 = history_gripper_flows[:, -1]
        history_relative_flow = xyz0[:, None] - history_gripper_flows[:, :-1]
        return (xyz0, history_relative_flow, token_mask, gripper_flow_fps, gripper_flow_horizon)

    def _prepare_sample_eef_from_history(
        self,
        history_eef_flows: torch.Tensor,
        eef_mask: torch.Tensor,
        eef_flow_fps: torch.Tensor,
        eef_flow_horizon: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Anchor at history last frame (= current EEF). Returns xyz0, rotgrip0, hist xyz/rotgrip rel, mask, fps, horizon."""
        if history_eef_flows.ndim != 4 or history_eef_flows.shape[2:] != (2, 10):
            raise ValueError(
                f"history_eef_flows must be [B,T,2,10], got {tuple(history_eef_flows.shape)}"
            )
        if history_eef_flows.shape[1] < 2:
            raise ValueError("history_eef_flows needs at least 2 timesteps (past + current)")
        token_mask = self._current_token_mask(eef_mask)
        history_xyz = history_eef_flows[..., :3]
        history_rotgrip = history_eef_flows[..., 3:]
        xyz0 = history_xyz[:, -1]
        rot0 = history_rotgrip[:, -1, :, :-1]
        grip0 = history_rotgrip[:, -1, :, -1:].clamp(0.0, 1.0)
        rotgrip0 = torch.cat([rot0, grip0], dim=-1)
        history_xyz_relative_flow = xyz0[:, None] - history_xyz[:, :-1]
        history_rot_relative_flow = self._eef_rot_to_relative_chunks(
            history_rotgrip[:, :-1, :, :-1], rot0
        )
        history_grip = history_rotgrip[:, :-1, :, -1:].clamp(0.0, 1.0)
        history_rotgrip_relative_flow = torch.cat([history_rot_relative_flow, history_grip], dim=-1)
        finite = (
            torch.isfinite(history_xyz_relative_flow).all(dim=-1).all(dim=1)
            & torch.isfinite(history_rotgrip_relative_flow).all(dim=-1).all(dim=1)
            & torch.isfinite(xyz0).all(dim=-1)
            & torch.isfinite(rot0).all(dim=-1)
            & torch.isfinite(grip0).all(dim=-1)
        )
        token_mask = token_mask & finite
        return (
            torch.nan_to_num(xyz0),
            torch.nan_to_num(rotgrip0),
            torch.nan_to_num(history_xyz_relative_flow),
            torch.nan_to_num(history_rotgrip_relative_flow),
            token_mask,
            eef_flow_fps,
            eef_flow_horizon,
        )

    def _prepare_sample_scene_xyz0(
        self,
        scene_xyz0: torch.Tensor,
        scene_mask: torch.Tensor,
        scene_flow_fps: torch.Tensor,
        scene_flow_horizon: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare already-sampled current scene points; no future GT needed."""
        if scene_xyz0.ndim != 3 or scene_xyz0.shape[-1] != 3:
            raise ValueError(f"scene_xyz0 must be [B,N,3], got {tuple(scene_xyz0.shape)}")
        B, Ns, _ = scene_xyz0.shape
        device = scene_xyz0.device
        token_mask = scene_mask.to(device=device, dtype=torch.bool)
        selected_indices = torch.arange(Ns, device=device).unsqueeze(0).expand(B, Ns).contiguous()
        xyz0 = scene_xyz0
        xyz0 = xyz0 * token_mask.unsqueeze(-1).to(dtype=xyz0.dtype)
        return (xyz0, token_mask, scene_flow_fps, scene_flow_horizon, selected_indices)

    def _encode_history_latent(
        self,
        relative_flow: torch.Tensor,
        fps: torch.Tensor,
        horizon: torch.Tensor,
    ) -> torch.Tensor:
        device_type = "cuda" if relative_flow.is_cuda else relative_flow.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            return self._rescale_latent_with_ema(
                self.flow_chunk_codec.encode(
                    relative_flow.float(), fps, self.config.max_prediction_time, horizon, "history"
                ),
                self.latent_ema_mean,
                self.latent_ema_std,
            )

    def _prepare_sample_latents(
        self,
        history_gripper_flows: torch.Tensor,
        gripper_mask: torch.Tensor,
        gripper_flow_fps: torch.Tensor,
        gripper_flow_horizon: torch.Tensor,
        history_eef_flows: torch.Tensor | None = None,
        eef_mask: torch.Tensor | None = None,
        eef_flow_fps: torch.Tensor | None = None,
        eef_flow_horizon: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Encode gripper history and prepare EEF anchors for the separate action head."""
        gripper_xyz0, history_gripper_rel, gripper_token_mask, g_fps, g_horizon = (
            self._prepare_sample_gripper_from_history(
                history_gripper_flows, gripper_mask, gripper_flow_fps, gripper_flow_horizon
            )
        )
        out: dict[str, torch.Tensor | None] = {
            "gripper_xyz0": gripper_xyz0,
            "gripper_token_mask": gripper_token_mask,
            "g_fps": g_fps,
            "g_horizon": g_horizon,
            "history_gripper_tokens": self._encode_history_latent(
                history_gripper_rel, g_fps, g_horizon
            ),
            "eef_xyz0": None,
            "eef_rotgrip0": None,
            "eef_token_mask": None,
            "e_fps": None,
            "e_horizon": None,
        }
        if history_eef_flows is None:
            return out
        (
            eef_xyz0,
            eef_rotgrip0,
            history_eef_xyz_rel,
            history_eef_rotgrip_rel,
            eef_token_mask,
            e_fps,
            e_horizon,
        ) = self._prepare_sample_eef_from_history(
            history_eef_flows, eef_mask, eef_flow_fps, eef_flow_horizon
        )
        out.update(
            eef_xyz0=eef_xyz0,
            eef_rotgrip0=eef_rotgrip0,
            eef_token_mask=eef_token_mask,
            e_fps=e_fps,
            e_horizon=e_horizon,
        )
        return out

    def _sample_forward(
        self,
        obs: dict[str, torch.Tensor],
        *,
        history_gripper_tokens: torch.Tensor,
        gripper_token_mask: torch.Tensor,
        gripper_xyz0: torch.Tensor,
        gripper_xt: torch.Tensor,
        gripper_x0: torch.Tensor,
        timestep: torch.Tensor,
        scene_flow_xt: torch.Tensor | None = None,
        scene_flow_mask: torch.Tensor | None = None,
        scene_xyz: torch.Tensor | None = None,
        return_mot_kv: bool = False,
        return_scene_kv: bool = False,
        return_context_kv: bool = False,
        skip_clean: bool = False,
        skip_noisy: bool = False,
        context_kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        context_key_padding_mask: torch.Tensor | None = None,
        prefix_kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        prefix_key_padding_mask: torch.Tensor | None = None,
    ):
        """Shared main-DiT call used by gripper / scene / MoT-prefill sampling."""
        return self(
            **obs,
            history_gripper_tokens=history_gripper_tokens,
            history_gripper_mask=gripper_token_mask,
            gripper_xt=gripper_xt,
            gripper_mask=gripper_token_mask,
            gripper_xyz0=gripper_xyz0,
            gripper_xyz0_mask=gripper_token_mask,
            timestep=timestep,
            gripper_x0=gripper_x0,
            scene_flow_xt=scene_flow_xt,
            scene_flow_mask=scene_flow_mask,
            scene_xyz=scene_xyz,
            return_mot_kv=return_mot_kv,
            return_scene_kv=return_scene_kv,
            return_context_kv=return_context_kv,
            skip_clean=skip_clean,
            skip_noisy=skip_noisy,
            context_kv_cache=context_kv_cache,
            context_key_padding_mask=context_key_padding_mask,
            prefix_kv_cache=prefix_kv_cache,
            prefix_key_padding_mask=prefix_key_padding_mask,
        )

    @torch.inference_mode()
    def _denoise_scene_flow_from_clean_gripper(
        self,
        obs: dict[str, torch.Tensor],
        *,
        history_gripper_tokens: torch.Tensor,
        gripper_token_mask: torch.Tensor,
        clean_gripper_latent: torch.Tensor,
        gripper_xyz0: torch.Tensor,
        scene_xyz: torch.Tensor,
        scene_token_mask: torch.Tensor,
        scene_selected_indices: torch.Tensor,
        scene_flow_fps: torch.Tensor,
        scene_flow_horizon: torch.Tensor,
        scene_steps: int,
        prefix_kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        prefix_key_padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Scene-flow ODE on finished clean gripper; MoT reuses context+clean KV."""
        B, Ns = (int(clean_gripper_latent.shape[0]), int(scene_xyz.shape[1]))
        device, dtype = (clean_gripper_latent.device, clean_gripper_latent.dtype)
        scene_latent = torch.randn(B, Ns, self.flow_codec_dim, device=device, dtype=dtype)
        gripper_kwargs = dict(
            history_gripper_tokens=history_gripper_tokens,
            gripper_token_mask=gripper_token_mask,
            gripper_xyz0=gripper_xyz0,
            gripper_xt=clean_gripper_latent,
            gripper_x0=clean_gripper_latent,
        )
        if prefix_kv_cache is None:
            _, _, context_kv_cache, context_key_padding_mask = self._sample_forward(
                obs,
                **gripper_kwargs,
                timestep=clean_gripper_latent.new_zeros(B),
                return_context_kv=True,
            )
            _, _, prefix_kv_cache, prefix_key_padding_mask = self._sample_forward(
                obs,
                **gripper_kwargs,
                timestep=clean_gripper_latent.new_zeros(B),
                context_kv_cache=context_kv_cache,
                context_key_padding_mask=context_key_padding_mask,
                skip_noisy=True,
                return_scene_kv=True,
            )
        obs = dict(obs, _sequence_cache={})
        dt_s = 1.0 / float(scene_steps)
        for step_idx in range(scene_steps):
            scene_out = self._sample_forward(
                obs,
                **gripper_kwargs,
                timestep=torch.full(
                    (B,), float(step_idx) / float(scene_steps), device=device, dtype=dtype
                ),
                scene_flow_xt=scene_latent,
                scene_flow_mask=scene_token_mask,
                scene_xyz=scene_xyz,
                prefix_kv_cache=prefix_kv_cache,
                prefix_key_padding_mask=prefix_key_padding_mask,
            )
            scene_latent = scene_latent + dt_s * scene_out[1]
        device_type = "cuda" if scene_latent.is_cuda else scene_latent.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            scene_latent_codec = self._unscale_latent_with_ema(
                scene_latent.float(), self.latent_ema_mean, self.latent_ema_std
            )
            pred_scene_flow = self.flow_chunk_codec.decode(
                scene_latent_codec,
                scene_flow_horizon,
                scene_flow_fps,
                self.config.max_prediction_time,
                "future",
            )
        return {
            "pred_scene_relative_flow": pred_scene_flow,
            "pred_scene_future_xyz": scene_xyz.float()[:, None] + pred_scene_flow,
            "scene_xyz0": scene_xyz,
            "scene_token_mask": scene_token_mask,
            "scene_selected_indices": scene_selected_indices,
            "pred_scene_latent": scene_latent_codec,
        }

    @torch.inference_mode()
    def prepare_scene_condition(
        self,
        *,
        history_gripper_flows,
        gripper_mask,
        gripper_flow_fps,
        gripper_flow_horizon,
        clean_gripper_latent,
        **obs,
    ) -> dict:
        """Prepare history and clean-gripper KV once for one dense scene prediction."""
        prepared = self._prepare_sample_latents(
            history_gripper_flows, gripper_mask, gripper_flow_fps, gripper_flow_horizon
        )
        obs = dict(obs, _sequence_cache={})
        kwargs = dict(
            history_gripper_tokens=prepared["history_gripper_tokens"],
            gripper_token_mask=prepared["gripper_token_mask"],
            gripper_xyz0=prepared["gripper_xyz0"],
            gripper_xt=clean_gripper_latent,
            gripper_x0=clean_gripper_latent,
            timestep=clean_gripper_latent.new_zeros(clean_gripper_latent.shape[0]),
        )
        _, _, context, context_mask = self._sample_forward(obs, **kwargs, return_context_kv=True)
        _, _, prefix, prefix_mask = self._sample_forward(
            obs,
            **kwargs,
            context_kv_cache=context,
            context_key_padding_mask=context_mask,
            skip_noisy=True,
            return_scene_kv=True,
        )
        obs.pop("_sequence_cache")
        return dict(
            prepared=prepared,
            obs=obs,
            prefix=prefix,
            prefix_mask=prefix_mask,
            clean_gripper_latent=clean_gripper_latent,
        )

    @torch.inference_mode()
    def sample_scene_flow_from_gripper(
        self,
        vlm_tokens: torch.Tensor,
        aux_visual_tokens: torch.Tensor,
        vlm_mask: torch.Tensor,
        aux_visual_mask: torch.Tensor,
        history_gripper_flows: torch.Tensor,
        gripper_mask: torch.Tensor,
        vlm_xyz: torch.Tensor,
        aux_visual_xyz: torch.Tensor,
        vlm_xyz_mask: torch.Tensor,
        aux_visual_xyz_mask: torch.Tensor,
        gripper_flow_fps: torch.Tensor,
        gripper_flow_horizon: torch.Tensor,
        scene_xyz0: torch.Tensor,
        scene_mask: torch.Tensor,
        scene_flow_fps: torch.Tensor,
        scene_flow_horizon: torch.Tensor,
        clean_gripper_latent: torch.Tensor | None = None,
        gripper_relative_flow: torch.Tensor | None = None,
        scene_steps: int | None = None,
        vlm_depth_valid_mask: torch.Tensor | None = None,
        aux_visual_depth_valid_mask: torch.Tensor | None = None,
        vlm_intrinsic: torch.Tensor | None = None,
        vlm_extrinsic: torch.Tensor | None = None,
        aux_visual_intrinsic: torch.Tensor | None = None,
        aux_visual_extrinsic: torch.Tensor | None = None,
        vlm_vision_tokens: torch.Tensor | None = None,
        aux_frustum_vision_tokens: torch.Tensor | None = None,
        condition: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        """Denoise scene flow from a finished gripper (FM latent or relative flow)."""
        prepared = (
            condition["prepared"]
            if condition is not None
            else self._prepare_sample_latents(
                history_gripper_flows, gripper_mask, gripper_flow_fps, gripper_flow_horizon
            )
        )
        if condition is not None:
            clean_gripper_latent = condition["clean_gripper_latent"]
        if clean_gripper_latent is None:
            device_type = (
                "cuda" if gripper_relative_flow.is_cuda else gripper_relative_flow.device.type
            )
            with torch.autocast(device_type=device_type, enabled=False):
                clean_gripper_latent = self._rescale_latent_with_ema(
                    self.flow_chunk_codec.encode(
                        gripper_relative_flow.float(),
                        prepared["g_fps"],
                        self.config.max_prediction_time,
                        prepared["g_horizon"],
                        "future",
                    ),
                    self.latent_ema_mean,
                    self.latent_ema_std,
                ) * prepared["gripper_token_mask"].unsqueeze(-1).to(
                    dtype=gripper_relative_flow.dtype
                )
        scene_xyz, scene_token_mask, s_fps, s_horizon, scene_selected_indices = (
            self._prepare_sample_scene_xyz0(
                scene_xyz0, scene_mask, scene_flow_fps, scene_flow_horizon
            )
        )
        obs = dict(
            vlm_tokens=vlm_tokens,
            aux_visual_tokens=aux_visual_tokens,
            vlm_mask=vlm_mask,
            aux_visual_mask=aux_visual_mask,
            vlm_xyz=vlm_xyz,
            aux_visual_xyz=aux_visual_xyz,
            vlm_xyz_mask=vlm_xyz_mask,
            aux_visual_xyz_mask=aux_visual_xyz_mask,
            vlm_depth_valid_mask=vlm_depth_valid_mask,
            aux_visual_depth_valid_mask=aux_visual_depth_valid_mask,
            vlm_intrinsic=vlm_intrinsic,
            vlm_extrinsic=vlm_extrinsic,
            aux_visual_intrinsic=aux_visual_intrinsic,
            aux_visual_extrinsic=aux_visual_extrinsic,
            vlm_vision_tokens=vlm_vision_tokens,
            aux_frustum_vision_tokens=aux_frustum_vision_tokens,
        )
        return self._denoise_scene_flow_from_clean_gripper(
            condition["obs"] if condition is not None else obs,
            history_gripper_tokens=prepared["history_gripper_tokens"],
            gripper_token_mask=prepared["gripper_token_mask"],
            clean_gripper_latent=clean_gripper_latent,
            gripper_xyz0=prepared["gripper_xyz0"],
            scene_xyz=scene_xyz,
            scene_token_mask=scene_token_mask,
            scene_selected_indices=scene_selected_indices,
            scene_flow_fps=s_fps,
            scene_flow_horizon=s_horizon,
            scene_steps=int(self.config.num_denoise_steps if scene_steps is None else scene_steps),
            prefix_kv_cache=condition["prefix"] if condition is not None else None,
            prefix_key_padding_mask=condition["prefix_mask"] if condition is not None else None,
        )

    @torch.inference_mode()
    def sample(
        self,
        vlm_tokens: torch.Tensor,
        aux_visual_tokens: torch.Tensor,
        vlm_mask: torch.Tensor,
        aux_visual_mask: torch.Tensor,
        history_gripper_flows: torch.Tensor,
        gripper_mask: torch.Tensor,
        vlm_xyz: torch.Tensor,
        aux_visual_xyz: torch.Tensor,
        vlm_xyz_mask: torch.Tensor,
        aux_visual_xyz_mask: torch.Tensor,
        gripper_flow_fps: torch.Tensor,
        gripper_flow_horizon: torch.Tensor,
        vlm_depth_valid_mask: torch.Tensor | None = None,
        aux_visual_depth_valid_mask: torch.Tensor | None = None,
        vlm_intrinsic: torch.Tensor | None = None,
        vlm_extrinsic: torch.Tensor | None = None,
        aux_visual_intrinsic: torch.Tensor | None = None,
        aux_visual_extrinsic: torch.Tensor | None = None,
        vlm_vision_tokens: torch.Tensor | None = None,
        aux_frustum_vision_tokens: torch.Tensor | None = None,
        *,
        decode_target: str = "eef",
        history_eef_flows: torch.Tensor | None = None,
        eef_mask: torch.Tensor | None = None,
        eef_flow_fps: torch.Tensor | None = None,
        eef_flow_horizon: torch.Tensor | None = None,
        mot_action_steps: int | None = None,
        scene_xyz0: torch.Tensor | None = None,
        scene_mask: torch.Tensor | None = None,
        scene_flow_fps: torch.Tensor | None = None,
        scene_flow_horizon: torch.Tensor | None = None,
        scene_steps: int | None = None,
        steps: int | None = None,
        sampling_generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """History-only sampling. ``decode_target``: ``eef`` / ``scene`` / ``both``."""
        decode_target = str(decode_target).lower().strip()
        needs_eef = decode_target in ("eef", "both")
        needs_scene = decode_target in ("scene", "both")
        if decode_target not in ("eef", "scene", "both"):
            raise ValueError(f"decode_target must be eef|scene|both, got {decode_target!r}")
        if sampling_generator is not None and needs_scene:
            raise ValueError("Explicit sampling_generator supports eef-only inference")

        def sample_noise(*shape, device, dtype):
            if sampling_generator is None:
                return torch.randn(*shape, device=device, dtype=dtype)
            return torch.randn(
                *shape,
                generator=sampling_generator,
                device=sampling_generator.device,
                dtype=torch.float32,
            ).to(device=device, dtype=dtype)

        steps = int(steps or self.config.num_denoise_steps)
        obs = dict(
            vlm_tokens=vlm_tokens,
            aux_visual_tokens=aux_visual_tokens,
            vlm_mask=vlm_mask,
            aux_visual_mask=aux_visual_mask,
            vlm_xyz=vlm_xyz,
            aux_visual_xyz=aux_visual_xyz,
            vlm_xyz_mask=vlm_xyz_mask,
            aux_visual_xyz_mask=aux_visual_xyz_mask,
            vlm_depth_valid_mask=vlm_depth_valid_mask,
            aux_visual_depth_valid_mask=aux_visual_depth_valid_mask,
            vlm_intrinsic=vlm_intrinsic,
            vlm_extrinsic=vlm_extrinsic,
            aux_visual_intrinsic=aux_visual_intrinsic,
            aux_visual_extrinsic=aux_visual_extrinsic,
            vlm_vision_tokens=vlm_vision_tokens,
            aux_frustum_vision_tokens=aux_frustum_vision_tokens,
        )
        obs["_sequence_cache"] = {}
        prepared = self._prepare_sample_latents(
            history_gripper_flows,
            gripper_mask,
            gripper_flow_fps,
            gripper_flow_horizon,
            history_eef_flows=history_eef_flows if needs_eef else None,
            eef_mask=eef_mask,
            eef_flow_fps=eef_flow_fps,
            eef_flow_horizon=eef_flow_horizon,
        )
        gripper_xyz0 = prepared["gripper_xyz0"]
        gripper_token_mask = prepared["gripper_token_mask"]
        history_gripper_tokens = prepared["history_gripper_tokens"]
        g_fps, g_horizon = (prepared["g_fps"], prepared["g_horizon"])
        eef_xyz0, eef_rotgrip0 = (prepared["eef_xyz0"], prepared["eef_rotgrip0"])
        eef_token_mask = prepared["eef_token_mask"]
        e_fps, e_horizon = (prepared["e_fps"], prepared["e_horizon"])
        if needs_scene:
            scene_xyz, scene_token_mask, s_fps, s_horizon, scene_selected_indices = (
                self._prepare_sample_scene_xyz0(
                    scene_xyz0, scene_mask, scene_flow_fps, scene_flow_horizon
                )
            )
            scene_steps = int(self.config.num_denoise_steps if scene_steps is None else scene_steps)
        B = int(history_gripper_flows.shape[0])
        device, dtype = (history_gripper_flows.device, history_gripper_flows.dtype)
        _sample_device_type = "cuda" if history_gripper_flows.is_cuda else device.type
        gripper_latent = sample_noise(
            B, gripper_xyz0.shape[1], self.flow_codec_dim, device=device, dtype=dtype
        )
        mot_action = field_valid = None
        if needs_eef:
            mot_action_steps = int(
                self.config.mot_num_denoise_steps if mot_action_steps is None else mot_action_steps
            )
            if mot_action_steps != steps:
                raise ValueError(
                    f"Joint MoT denoising requires mot_action_steps == steps, got {mot_action_steps} and {steps}"
                )
            action_horizon = int(e_horizon.max().item())
            field_valid = (
                (
                    (torch.arange(action_horizon, device=device)[None, :] < e_horizon[:, None])[
                        :, :, None, None
                    ]
                    & eef_token_mask[:, None, :, None]
                )
                .expand(-1, -1, -1, 10)
                .reshape(B, action_horizon, 20)
            )
            mot_action = sample_noise(
                B, action_horizon, 20, device=device, dtype=dtype
            ) * field_valid.to(dtype=dtype)
        gripper_base = dict(
            history_gripper_tokens=history_gripper_tokens,
            gripper_token_mask=gripper_token_mask,
            gripper_xyz0=gripper_xyz0,
        )
        context_kv_cache = context_key_padding_mask = None
        _, _, context_kv_cache, context_key_padding_mask = self._sample_forward(
            obs,
            **gripper_base,
            gripper_xt=gripper_latent,
            gripper_x0=gripper_latent,
            timestep=gripper_latent.new_zeros(B),
            return_context_kv=True,
        )
        dt = 1.0 / float(steps)
        for step_idx in range(steps):
            joint_timestep = torch.full(
                (B,), float(step_idx) / float(steps), device=device, dtype=dtype
            )
            main_out = self._sample_forward(
                obs,
                **gripper_base,
                gripper_xt=gripper_latent,
                gripper_x0=gripper_latent,
                timestep=joint_timestep,
                context_kv_cache=context_kv_cache,
                context_key_padding_mask=context_key_padding_mask,
                skip_clean=True,
                return_mot_kv=needs_eef,
            )
            gripper_vf = main_out[0]
            if needs_eef:
                _, _, mot_kv_cache, mot_main_mask = main_out
                action_vf = self.mot_action_head(
                    noisy_action=mot_action,
                    timestep=joint_timestep,
                    fps=e_fps,
                    horizon=e_horizon,
                    eef_xyz0=eef_xyz0,
                    eef_rotgrip0=eef_rotgrip0,
                    arm_mask=eef_token_mask,
                    main_kv_cache=mot_kv_cache,
                    main_mask=mot_main_mask,
                )
                mot_action = (mot_action + dt * action_vf) * field_valid.to(dtype=dtype)
            gripper_latent = gripper_latent + dt * gripper_vf
        with torch.autocast(device_type=_sample_device_type, enabled=False):
            gripper_latent_codec = self._unscale_latent_with_ema(
                gripper_latent.float(), self.latent_ema_mean, self.latent_ema_std
            )
            pred_gripper_flow = self.flow_chunk_codec.decode(
                gripper_latent_codec, g_horizon, g_fps, self.config.max_prediction_time, "future"
            )
        result = {
            "pred_relative_flow": pred_gripper_flow,
            "pred_future_xyz": gripper_xyz0.float()[:, None] + pred_gripper_flow,
            "gripper_xyz0": gripper_xyz0,
            "gripper_token_mask": gripper_token_mask,
            "pred_gripper_latent": gripper_latent_codec,
            "pred_gripper_latent_fm": gripper_latent,
        }
        prefix_kv_cache = prefix_key_padding_mask = None
        if needs_scene:
            _, _, prefix_kv_cache, prefix_key_padding_mask = self._sample_forward(
                obs,
                **gripper_base,
                gripper_xt=gripper_latent,
                gripper_x0=gripper_latent,
                timestep=gripper_latent.new_zeros(B),
                context_kv_cache=context_kv_cache,
                context_key_padding_mask=context_key_padding_mask,
                skip_noisy=True,
                return_scene_kv=True,
            )
        if needs_scene:
            result.update(
                self._denoise_scene_flow_from_clean_gripper(
                    obs,
                    history_gripper_tokens=history_gripper_tokens,
                    gripper_token_mask=gripper_token_mask,
                    clean_gripper_latent=gripper_latent,
                    gripper_xyz0=gripper_xyz0,
                    scene_xyz=scene_xyz,
                    scene_token_mask=scene_token_mask,
                    scene_selected_indices=scene_selected_indices,
                    scene_flow_fps=s_fps,
                    scene_flow_horizon=s_horizon,
                    scene_steps=scene_steps,
                    prefix_kv_cache=prefix_kv_cache,
                    prefix_key_padding_mask=prefix_key_padding_mask,
                )
            )
            if not needs_eef:
                return result
        xyz_flow, rotgrip_flow, poses, open_ = self._unpack_mot_action(
            mot_action.float(), eef_xyz0.float(), eef_rotgrip0.float()
        )
        result.update(
            pred_eef_xyz_flow=xyz_flow,
            pred_eef_rotgrip_flow=rotgrip_flow,
            pred_eef_poses=poses,
            pred_eef_open=open_,
            pred_mot_action=mot_action,
        )
        return result

    @staticmethod
    def _quat_xyzw_to_rot6d(quat: torch.Tensor) -> torch.Tensor:
        if quat.shape[-1] != 4:
            raise ValueError(f"quat must end with 4 channels, got {tuple(quat.shape)}")
        q = quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-08)
        x, y, z, w = q.unbind(dim=-1)
        xx, yy, zz = (x * x, y * y, z * z)
        xy, xz, yz = (x * y, x * z, y * z)
        wx, wy, wz = (w * x, w * y, w * z)
        col0 = torch.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy)], dim=-1)
        col1 = torch.stack([2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx)], dim=-1)
        return torch.cat([col0, col1], dim=-1)
