from __future__ import annotations
import math
import torch
import torch.nn as nn


class SiLUMLP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mlp_ratio: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, int(in_channels * mlp_ratio)),
            nn.SiLU(),
            nn.LayerNorm(int(in_channels * mlp_ratio)),
            nn.Linear(int(in_channels * mlp_ratio), out_channels),
        )

    def forward(self, x):
        return self.mlp(x)


class FlowChunkCodec(nn.Module):
    """Shared continuous-time encoder/decoder for point-wise flow chunks."""

    def __init__(self, hidden_size: int, coord_dim: int = 3, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.coord_dim = int(coord_dim)
        self.input_proj = SiLUMLP(coord_dim, hidden_size, mlp_ratio)
        self.encoded_time_mlp = SiLUMLP(hidden_size, hidden_size * 2, mlp_ratio)
        self.decoded_time_mlp = SiLUMLP(hidden_size, hidden_size * 2, mlp_ratio)
        self.output_proj = SiLUMLP(hidden_size, coord_dim, mlp_ratio)

    @staticmethod
    def _time_valid_mask(
        pad_horizon: int, horizon: torch.Tensor, time_direction: str
    ) -> torch.Tensor:
        """Return ``[B, T]`` mask: future is right-padded, history is left-padded."""
        t = torch.arange(pad_horizon, device=horizon.device)
        if time_direction == "future":
            return t.unsqueeze(0) < horizon.unsqueeze(1)
        if time_direction == "history":
            return t.unsqueeze(0) >= pad_horizon - horizon.unsqueeze(1)
        raise ValueError(f"time_direction must be 'future' or 'history', got {time_direction!r}")

    def _broadcast_scale_shift(
        self, scale: torch.Tensor, shift: torch.Tensor, pad_horizon: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize scale/shift to ``[B_or_1, T, hidden]``."""
        if scale.shape != shift.shape:
            raise ValueError(
                f"scale/shift shape mismatch: {tuple(scale.shape)} vs {tuple(shift.shape)}"
            )
        if scale.ndim == 2:
            if scale.shape != (pad_horizon, self.hidden_size):
                raise ValueError(
                    f"scale/shift must have shape [T={pad_horizon}, hidden={self.hidden_size}], got {tuple(scale.shape)}"
                )
            return (scale.unsqueeze(0), shift.unsqueeze(0))
        if scale.ndim == 3:
            if scale.shape[-2:] != (pad_horizon, self.hidden_size):
                raise ValueError(
                    f"scale/shift must have shape [B, T={pad_horizon}, hidden={self.hidden_size}], got {tuple(scale.shape)}"
                )
            return (scale, shift)
        raise ValueError(f"scale/shift must be 2D or 3D, got {tuple(scale.shape)}")

    def time_basis(
        self,
        horizon: int,
        fps: torch.Tensor,
        max_time: float,
        device: torch.device,
        dtype: torch.dtype,
        time_direction: str = "future",
    ) -> torch.Tensor:
        pad_horizon = int(horizon)
        if time_direction == "future":
            time_index = torch.arange(1, pad_horizon + 1, device=device, dtype=torch.float32)
        elif time_direction == "history":
            time_index = torch.arange(-pad_horizon, 0, device=device, dtype=torch.float32)
        else:
            raise ValueError(
                f"time_direction must be 'future' or 'history', got {time_direction!r}"
            )
        powers = torch.arange(self.hidden_size, device=device, dtype=torch.float32) / float(
            self.hidden_size
        )
        bsz = fps.shape[0]
        normalized_time = (
            time_index.abs().view(1, pad_horizon) / fps.view(bsz, 1) / float(max_time)
        ).clamp(min=1e-06)
        time_sign = time_index.sign().view(1, pad_horizon, 1)
        basis = (
            torch.sin(normalized_time[:, :, None].pow(powers[None, None, :]) * math.pi / 2.0)
            * time_sign
        )
        return basis.to(dtype=dtype)

    def time_scale_shift(
        self,
        horizon: int,
        fps: float | torch.Tensor,
        max_time: float,
        device: torch.device,
        dtype: torch.dtype,
        time_direction: str = "future",
        encoded: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        basis = self.time_basis(
            horizon, fps, max_time, device=device, dtype=dtype, time_direction=time_direction
        )
        scale_shift = self.encoded_time_mlp(basis) if encoded else self.decoded_time_mlp(basis)
        scale, shift = scale_shift.chunk(2, dim=-1)
        return (scale, shift)

    def encode(
        self,
        flow_chunk: torch.Tensor,
        fps: torch.Tensor,
        max_time: float,
        horizon: torch.Tensor | None = None,
        time_direction: str = "future",
    ) -> torch.Tensor:
        """Encode a padded flow chunk."""
        device_type = "cuda" if flow_chunk.is_cuda else flow_chunk.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            flow_chunk_f = flow_chunk.float()
            fps_f = fps.float() if fps.is_floating_point() else fps
            batch, pad_horizon, _num_points, _channels = flow_chunk_f.shape
            scale, shift = self.time_scale_shift(
                pad_horizon,
                fps_f,
                max_time,
                device=flow_chunk_f.device,
                dtype=torch.float32,
                time_direction=time_direction,
                encoded=True,
            )
            scale, shift = self._broadcast_scale_shift(scale, shift, pad_horizon)
            value = self.input_proj(flow_chunk_f)
            value = value * (1.0 + scale[:, :, None, :])
            value = value + shift[:, :, None, :]
            valid = self._time_valid_mask(pad_horizon, horizon, time_direction).to(
                dtype=value.dtype
            )
            value = value * valid[:, :, None, None]
            denom = horizon.to(dtype=value.dtype).clamp(min=1).sqrt().view(batch, 1, 1)
            return value.sum(dim=1) / denom

    def decode(
        self,
        hidden: torch.Tensor,
        horizon: torch.Tensor,
        fps: torch.Tensor,
        max_time: float,
        time_direction: str = "future",
    ) -> torch.Tensor:
        """Decode a latent into a padded flow chunk."""
        device_type = "cuda" if hidden.is_cuda else hidden.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            hidden_f = hidden.float()
            fps_f = fps.float() if fps.is_floating_point() else fps
            pad_horizon = int(horizon.max().item())
            scale, shift = self.time_scale_shift(
                pad_horizon,
                fps_f,
                max_time,
                device=hidden_f.device,
                dtype=torch.float32,
                time_direction=time_direction,
                encoded=False,
            )
            scale, shift = self._broadcast_scale_shift(scale, shift, pad_horizon)
            chunk_hidden = hidden_f[:, None, :, :] * (1.0 + scale[:, :, None, :])
            chunk_hidden = chunk_hidden + shift[:, :, None, :]
            out = self.output_proj(chunk_hidden)
            valid = self._time_valid_mask(pad_horizon, horizon, time_direction).to(dtype=out.dtype)
            return out * valid[:, :, None, None]
