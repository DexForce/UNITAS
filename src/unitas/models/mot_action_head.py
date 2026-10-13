from __future__ import annotations
import math
from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import Mlp, timestep_embedding


class PhysicalTrajectoryTimeEmbedding(nn.Module):
    """Continuous physical-time FiLM for padded action trajectories."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        powers = torch.arange(self.hidden_size, dtype=torch.float32) / float(self.hidden_size)
        self.register_buffer("powers", powers, persistent=False)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, 2 * self.hidden_size),
            nn.SiLU(),
            nn.LayerNorm(2 * self.hidden_size),
            nn.Linear(2 * self.hidden_size, 2 * self.hidden_size),
        )

    def _basis(self, normalized_time: torch.Tensor) -> torch.Tensor:
        value = normalized_time.float().clamp(min=1e-06)
        basis = torch.sin(value[..., None].pow(self.powers[None, None, :]) * (math.pi / 2.0))
        return basis

    def forward(
        self,
        pad_horizon: int,
        fps: torch.Tensor,
        max_prediction_time: float,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if fps.ndim != 1:
            raise ValueError(f"fps must be [B], got {tuple(fps.shape)}")
        if max_prediction_time <= 0:
            raise ValueError("max_prediction_time must be positive")
        steps = torch.arange(1, pad_horizon + 1, device=device, dtype=torch.float32)
        fps_f = fps.to(device=device, dtype=torch.float32).clamp(min=1e-06)
        token_time = steps[None, :] / fps_f[:, None] / float(max_prediction_time)
        scale, shift = self.mlp(self._basis(token_time).to(dtype=dtype)).chunk(2, dim=-1)
        return (scale, shift)


class MoTActionBlock(nn.Module):
    """Update one clean-state token and a bidirectional noisy-action suffix."""

    def __init__(
        self, hidden_size: int, main_hidden_size: int, num_heads: int, mlp_ratio: float
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = int(hidden_size)
        self.main_hidden_size = int(main_hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.norm1 = nn.LayerNorm(self.hidden_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(self.hidden_size, elementwise_affine=False)
        self.qkv = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=True)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.main_k_proj = nn.Linear(self.main_hidden_size, self.hidden_size)
        self.main_v_proj = nn.Linear(self.main_hidden_size, self.hidden_size)
        self.attn_dropout = nn.Dropout(0.1)
        self.mlp = Mlp(
            in_features=self.hidden_size,
            hidden_features=int(self.hidden_size * mlp_ratio),
            drop=0.1,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(self.hidden_size, 6 * self.hidden_size)
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    @staticmethod
    def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1.0 + scale) + shift

    def _project_main_kv(
        self, k_main: torch.Tensor, v_main: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, main_heads, tokens, head_dim = k_main.shape
        main_width = main_heads * head_dim
        if main_width != self.main_hidden_size:
            raise ValueError(
                f"main KV width mismatch: expected {self.main_hidden_size}, got {main_width}"
            )
        k_main = self.main_k_proj(k_main.transpose(1, 2).reshape(batch, tokens, main_width))
        v_main = self.main_v_proj(v_main.transpose(1, 2).reshape(batch, tokens, main_width))
        k_main = k_main.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v_main = v_main.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        return (k_main, v_main)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        timestep_cond: torch.Tensor,
        main_kv: tuple[torch.Tensor, torch.Tensor],
        main_mask: torch.Tensor,
        state_mask: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=1)
        modulation = self.adaLN_modulation(timestep_cond)
        if modulation.ndim == 2:
            modulation = modulation.unsqueeze(1)
        elif modulation.shape[1] == 2 and x.shape[1] != 2:
            modulation = torch.cat(
                [modulation[:, :1], modulation[:, 1:2].expand(-1, action.shape[1], -1)], dim=1
            )
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)
        attn_input = self._modulate(self.norm1(x), shift_msa, scale_msa)
        batch, tokens, _ = attn_input.shape
        qkv = (
            self.qkv(attn_input)
            .view(batch, tokens, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q_state, q_action = (q[:, :, :1], q[:, :, 1:])
        k_state, k_action = (k[:, :, :1], k[:, :, 1:])
        v_state, v_action = (v[:, :, :1], v[:, :, 1:])
        k_main, v_main = self._project_main_kv(*main_kv)
        expected_prefix = (batch, self.num_heads, main_mask.shape[1], self.head_dim)
        if k_main.shape != expected_prefix or v_main.shape != expected_prefix:
            raise ValueError(
                f"main KV shape mismatch: expected {expected_prefix}, got {tuple(k_main.shape)} and {tuple(v_main.shape)}"
            )
        state_key_mask = torch.cat([main_mask, state_mask], dim=1).to(dtype=torch.bool)
        state_attn = F.scaled_dot_product_attention(
            q_state,
            torch.cat([k_main, k_state], dim=2),
            torch.cat([v_main, v_state], dim=2),
            attn_mask=state_key_mask[:, None, None, :],
            dropout_p=self.attn_dropout.p if self.training else 0.0,
        )
        action_key_mask = torch.cat([main_mask, state_mask, action_mask], dim=1).to(
            dtype=torch.bool
        )
        action_attn = F.scaled_dot_product_attention(
            q_action,
            torch.cat([k_main, k_state, k_action], dim=2),
            torch.cat([v_main, v_state, v_action], dim=2),
            attn_mask=action_key_mask[:, None, None, :],
            dropout_p=self.attn_dropout.p if self.training else 0.0,
        )
        attn_out = torch.cat([state_attn, action_attn], dim=2)
        attn_out = attn_out.transpose(1, 2).reshape(batch, tokens, self.hidden_size)
        x = x + gate_msa * self.out_proj(attn_out)
        x = x + gate_mlp * self.mlp(self._modulate(self.norm2(x), shift_mlp, scale_mlp))
        token_mask = torch.cat([state_mask, action_mask], dim=1)
        x = x * token_mask.unsqueeze(-1).to(dtype=x.dtype)
        return (x[:, :1], x[:, 1:])


class MoTActionHead(nn.Module):
    """Direct H-token action flow head with layer-aligned main-model memory."""

    action_dim = 20

    def __init__(
        self,
        main_hidden_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        time_dim: int,
        max_prediction_time: float,
    ) -> None:
        super().__init__()
        if main_hidden_size % 2 != 0:
            raise ValueError("MoT action hidden size requires an even main hidden size")
        if num_heads % 2 != 0:
            raise ValueError("MoT action head requires an even main num_heads")
        self.main_hidden_size = int(main_hidden_size)
        self.num_heads = int(num_heads) // 2
        self.head_dim = self.main_hidden_size // int(num_heads)
        self.hidden_size = self.num_heads * self.head_dim
        self.time_dim = int(time_dim)
        self.max_prediction_time = float(max_prediction_time)
        self.input_proj = nn.Linear(self.action_dim, self.hidden_size)
        self.state_projector = nn.Sequential(
            nn.Linear(self.action_dim, self.hidden_size),
            nn.SiLU(),
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.time_embedding = PhysicalTrajectoryTimeEmbedding(self.hidden_size)
        self.timestep_mlp = nn.Sequential(
            nn.Linear(self.time_dim, self.hidden_size),
            nn.SiLU(),
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.blocks = nn.ModuleList(
            [
                MoTActionBlock(
                    hidden_size=self.hidden_size,
                    main_hidden_size=self.main_hidden_size,
                    num_heads=self.num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(depth)
            ]
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(self.hidden_size), nn.Linear(self.hidden_size, self.action_dim)
        )

    def _pack_state(
        self, eef_xyz0: torch.Tensor, eef_rotgrip0: torch.Tensor, arm_mask: torch.Tensor
    ) -> torch.Tensor:
        if eef_xyz0.shape[:2] != eef_rotgrip0.shape[:2] or eef_xyz0.shape[1] != 2:
            raise ValueError("packed MoT state requires exactly two aligned arm slots")
        fields = torch.cat([torch.nan_to_num(eef_xyz0), torch.nan_to_num(eef_rotgrip0)], dim=-1)
        fields = fields * arm_mask.to(device=fields.device, dtype=fields.dtype).unsqueeze(-1)
        return fields.reshape(fields.shape[0], self.action_dim)

    def forward(
        self,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        fps: torch.Tensor,
        horizon: torch.Tensor,
        eef_xyz0: torch.Tensor,
        eef_rotgrip0: torch.Tensor,
        arm_mask: torch.Tensor,
        main_kv_cache: Sequence[tuple[torch.Tensor, torch.Tensor]],
        main_mask: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_action.ndim != 3 or noisy_action.shape[-1] != self.action_dim:
            raise ValueError(
                f"noisy_action must have shape [B,H,{self.action_dim}], got {tuple(noisy_action.shape)}"
            )
        if len(main_kv_cache) != len(self.blocks):
            raise ValueError(
                f"expected {len(self.blocks)} main KV layers, got {len(main_kv_cache)}"
            )
        batch, pad_horizon, _ = noisy_action.shape
        action_mask = (
            torch.arange(pad_horizon, device=noisy_action.device)[None, :]
            < horizon.to(device=noisy_action.device)[:, None]
        )
        action_mask = action_mask & arm_mask.to(device=noisy_action.device).any(dim=1, keepdim=True)
        action = self.input_proj(noisy_action)
        scale, shift = self.time_embedding(
            pad_horizon,
            fps,
            self.max_prediction_time,
            device=noisy_action.device,
            dtype=action.dtype,
        )
        action = action * (1.0 + scale) + shift
        action = action * action_mask.unsqueeze(-1).to(dtype=action.dtype)
        state_mask = arm_mask.to(device=noisy_action.device).any(dim=1, keepdim=True)
        state = self.state_projector(
            self._pack_state(eef_xyz0, eef_rotgrip0, arm_mask).to(dtype=action.dtype)
        )[:, None, :]
        state = state * state_mask.unsqueeze(-1).to(dtype=state.dtype)
        timestep = timestep.to(device=action.device, dtype=action.dtype)
        action_timestep_cond = self.timestep_mlp(
            timestep_embedding(timestep * 1000.0, self.time_dim)
        )
        state_timestep_cond = torch.zeros_like(action_timestep_cond)
        timestep_cond = torch.stack([state_timestep_cond, action_timestep_cond], dim=1)
        for block, main_kv in zip(self.blocks, main_kv_cache):
            state, action = block(
                state, action, timestep_cond, main_kv, main_mask, state_mask, action_mask
            )
        return self.output_head(action) * action_mask.unsqueeze(-1).to(dtype=action.dtype)
