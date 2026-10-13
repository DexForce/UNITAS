# Copyright 2025 2toINF (https://github.com/2toINF)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from .position_encoding import apply_chunked_rotary_pos_emb
import math
from functools import partial
from typing import Iterable, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_2tuple(x) -> Tuple:
    """Minimal replacement for timm.layers.to_2tuple."""
    if isinstance(x, Iterable) and (not isinstance(x, (str, bytes))):
        t = tuple(x)
        return (t[0], t[1]) if len(t) >= 2 else (t[0], t[0])
    return (x, x)


class Mlp(nn.Module):
    """
    MLP used in ViT-style blocks.

    Supports Linear or 1x1 Conv 'linear_layer' for token/channel mixing.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        norm_layer: type[nn.Module] | None = None,
        bias: bool | Tuple[bool, bool] = True,
        drop: float | Tuple[float, float] = 0.0,
        use_conv: bool = False,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = _to_2tuple(bias)
        drop_probs = _to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = nn.GELU(approximate="tanh")
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):
    """Multi-head attention with PyTorch SDPA and optional cached keys and values."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** (-0.5)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _dropout_p(self) -> float:
        return self.attn_drop.p if self.training else 0.0

    def _sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_mask = None if key_mask is None else key_mask[:, None, None, :]
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self._dropout_p()
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_kv: bool = False,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        past_key_padding_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Parameters"""
        if key_padding_mask is not None:
            if token_mask is not None:
                raise ValueError("Pass only one of key_padding_mask or token_mask")
            token_mask = key_padding_mask
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = (self.q_norm(q), self.k_norm(k))
        if position_embeddings is not None:
            q, k = apply_chunked_rotary_pos_emb(q, k, *position_embeddings)
        if token_mask is not None:
            if token_mask.shape != (B, T):
                raise ValueError(
                    f"token_mask must have shape {(B, T)}, got {tuple(token_mask.shape)}"
                )
            token_mask = token_mask.to(device=x.device, dtype=torch.bool)
        if past_kv is not None:
            pk, pv = past_kv
            k_cat = torch.cat([pk, k], dim=2)
            v_cat = torch.cat([pv, v], dim=2)
            key_mask = token_mask
            if past_key_padding_mask is not None:
                past_key_padding_mask = past_key_padding_mask.to(device=x.device, dtype=torch.bool)
                key_mask = (
                    past_key_padding_mask
                    if token_mask is None
                    else torch.cat([past_key_padding_mask, token_mask], dim=1)
                )
            x = self._sdpa(q, k_cat, v_cat, key_mask=key_mask)
        else:
            x = self._sdpa(q, k, v, key_mask=token_mask)
        x = x.transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_kv:
            return (x, (k, v))
        return x


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 100) -> torch.Tensor:
    """Create sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=t.dtype, device=t.device)
        / half
    )
    args = t[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class AdaLNDiTBlock(nn.Module):
    """DiT-style Transformer block with AdaLN-Zero conditioning."""

    def __init__(
        self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, cond_dim: int | None = None
    ) -> None:
        super().__init__()
        cond_dim = cond_dim or hidden_size
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1)
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), drop=0.1
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * hidden_size))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    @staticmethod
    def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1.0 + scale) + shift

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        cond_index: torch.Tensor | None = None,
        return_kv: bool = False,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        past_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if cond.ndim == 2:
            if cond.shape[0] != x.shape[0]:
                raise ValueError(
                    f"cond must have shape [B, C], got {tuple(cond.shape)} for x {tuple(x.shape)}"
                )
            cond = cond.unsqueeze(1)
        elif cond.ndim == 3:
            if cond_index is not None:
                if cond_index.shape != (x.shape[1],):
                    raise ValueError(
                        f"cond_index must have shape [{x.shape[1]}], got {tuple(cond_index.shape)}"
                    )
            elif cond.shape[:2] != x.shape[:2]:
                raise ValueError(
                    f"per-token cond must have shape {tuple(x.shape[:2])}+[C], got {tuple(cond.shape)}"
                )
        else:
            raise ValueError(f"cond must have 2 or 3 dims, got {tuple(cond.shape)}")
        modulation = self.adaLN_modulation(cond)
        if cond_index is not None:
            modulation = modulation[:, cond_index]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)
        attn_hidden = self._modulate(self.norm1(x), shift_msa, scale_msa)
        attn_result = self.attn(
            attn_hidden,
            key_padding_mask=key_padding_mask,
            position_embeddings=position_embeddings,
            return_kv=return_kv,
            past_kv=past_kv,
            past_key_padding_mask=past_key_padding_mask,
        )
        if return_kv:
            attn_out, kv = attn_result
        else:
            attn_out = attn_result
        x = x + gate_msa * attn_out
        x = x + gate_mlp * self.mlp(self._modulate(self.norm2(x), shift_mlp, scale_mlp))
        if return_kv:
            return (x, kv)
        return x
