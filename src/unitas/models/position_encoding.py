from __future__ import annotations
import math
import torch
import torch.nn as nn


def _freq_bands(n_freqs: int, logscale: bool) -> torch.Tensor:
    if logscale:
        return 2 ** torch.linspace(0, n_freqs - 1, n_freqs)
    return torch.linspace(1, 2 ** (n_freqs - 1), n_freqs)


def _xavier_small_(module: nn.Module) -> None:
    for p in module.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p, gain=0.01)


def _fourier_xyz(
    xyz: torch.Tensor, freq_bands: torch.Tensor, *, scale: float, in_channels: int
) -> torch.Tensor:
    """``[B,N,3] -> [B,N,F]`` with ``xyz, sin(2^k xyz), cos(2^k xyz)``."""
    if xyz.ndim != 3 or xyz.shape[-1] != in_channels:
        raise ValueError(f"xyz must be [B,N,{in_channels}], got {tuple(xyz.shape)}")
    bands = freq_bands.to(device=xyz.device, dtype=xyz.dtype)
    xyz_n = xyz / scale
    xyz_freq = xyz_n.unsqueeze(-1) * bands
    return torch.cat(
        [xyz_n.unsqueeze(-1), torch.sin(xyz_freq), torch.cos(xyz_freq)], dim=-1
    ).reshape(*xyz.shape[:2], -1)


class Unified3DPositionEmbeddingMLP(nn.Module):
    """Per-token Fourier MLP PE: ``[B,N,3] -> [B,N,C]``."""

    def __init__(
        self,
        in_channels: int = 3,
        num_pos_feats: int = 512,
        n_freqs: int = 11,
        logscale: bool = True,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.scale = float(scale)
        self.freq_out_channels = self.in_channels * (2 * n_freqs + 1)
        self.register_buffer("freq_bands", _freq_bands(n_freqs, logscale), persistent=False)
        self.position_embedding_head = nn.Sequential(
            nn.Linear(self.freq_out_channels, num_pos_feats),
            nn.LayerNorm(num_pos_feats),
            nn.ReLU(),
            nn.Linear(num_pos_feats, num_pos_feats),
        )
        _xavier_small_(self)

    def frequency_encoding(self, xyz: torch.Tensor) -> torch.Tensor:
        return _fourier_xyz(xyz, self.freq_bands, scale=self.scale, in_channels=self.in_channels)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return self.position_embedding_head(self.frequency_encoding(xyz))


class HighResPatch3DPositionEmbedding(nn.Module):
    """Dense xyz Fourier PE with hierarchical 32→16×16 merge → patch PE.

    ``[B,H,W,3] -> freq -> [B,N,4,sub²·F] -> patch_merge -> avg4 -> head -> [B,N,C]``
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_pos_feats: int = 512,
        n_freqs: int = 11,
        logscale: bool = True,
        scale: float = 1.0,
        merge_size: int = 32,
    ) -> None:
        super().__init__()
        if merge_size % 2 != 0:
            raise ValueError(f"merge_size must be even, got {merge_size}")
        self.in_channels = int(in_channels)
        self.scale = float(scale)
        self.merge_size = int(merge_size)
        self.sub_merge_size = self.merge_size // 2
        self.num_pos_feats = int(num_pos_feats)
        self.freq_out_channels = self.in_channels * (2 * n_freqs + 1)
        self.register_buffer("freq_bands", _freq_bands(n_freqs, logscale), persistent=False)
        merge_in = self.sub_merge_size**2 * self.freq_out_channels
        self.patch_merge = nn.Sequential(
            nn.Linear(merge_in, self.num_pos_feats), nn.SiLU(), nn.LayerNorm(self.num_pos_feats)
        )
        self.position_embedding_head = nn.Sequential(
            nn.Linear(self.num_pos_feats, self.num_pos_feats),
            nn.LayerNorm(self.num_pos_feats),
            nn.ReLU(),
            nn.Linear(self.num_pos_feats, self.num_pos_feats),
        )
        _xavier_small_(self)

    def frequency_encoding(self, xyz: torch.Tensor) -> torch.Tensor:
        return _fourier_xyz(xyz, self.freq_bands, scale=self.scale, in_channels=self.in_channels)

    def encode_xyz_to_freq(self, xyz_hw: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = xyz_hw.shape
        clean = torch.nan_to_num(xyz_hw, nan=0.0, posinf=0.0, neginf=0.0)
        return self.frequency_encoding(clean.reshape(B, H * W, 3)).view(
            B, H, W, self.freq_out_channels
        )

    def reshape_freq_to_subpatches(self, freq_hw: torch.Tensor) -> torch.Tensor:
        """``[B,H,W,F] -> [B,N,4,sub²·F]``."""
        B, H, W, F = freq_hw.shape
        m, s = (self.merge_size, self.sub_merge_size)
        Hf, Wf = (H // m, W // m)
        N = Hf * Wf
        return (
            freq_hw.view(B, Hf, m, Wf, m, F)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(B, N, 2, s, 2, s, F)
            .permute(0, 1, 2, 4, 3, 5, 6)
            .contiguous()
            .view(B, N, 4, s * s * F)
        )

    def merge_subpatches(self, freq_hw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(sub_merged [B,N,4,C], pos [B,N,C])``."""
        sub = self.patch_merge(self.reshape_freq_to_subpatches(freq_hw))
        return (sub, self.position_embedding_head(sub.mean(dim=2)))

    def patch_center_mask(
        self, xyz_hw: torch.Tensor, xyz_mask_hw: torch.Tensor | None
    ) -> torch.Tensor:
        B, H, W, _ = xyz_hw.shape
        m = self.merge_size
        cy = cx = m // 2
        if xyz_mask_hw is None:
            mask = torch.isfinite(xyz_hw).all(dim=-1)
        else:
            mask = xyz_mask_hw.bool()
        return mask[:, cy::m, cx::m].reshape(B, H // m * (W // m))

    def subpatch_valid_fraction(self, xyz_mask_hw: torch.Tensor) -> torch.Tensor:
        """Valid-depth fraction per subpatch: ``[B,N,4]``."""
        B, H, W = xyz_mask_hw.shape
        m, s = (self.merge_size, self.sub_merge_size)
        Hf, Wf = (H // m, W // m)
        N = Hf * Wf
        return (
            xyz_mask_hw.float()
            .view(B, Hf, m, Wf, m)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(B, N, 2, s, 2, s)
            .permute(0, 1, 2, 4, 3, 5)
            .contiguous()
            .view(B, N, 4, s * s)
            .mean(dim=-1)
        )

    def forward(
        self, xyz_hw: torch.Tensor, xyz_mask_hw: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, pos = self.merge_subpatches(self.encode_xyz_to_freq(xyz_hw))
        patch_mask = self.patch_center_mask(xyz_hw, xyz_mask_hw)
        return (pos * patch_mask.unsqueeze(-1).to(dtype=pos.dtype), patch_mask)


class FrustumRoutedHighResPatch3DPositionEmbedding(nn.Module):
    """Depth PE + frustum soft-depth routing at subpatch level."""

    def __init__(
        self,
        *,
        vision_dim: int,
        in_channels: int = 3,
        num_pos_feats: int = 512,
        n_freqs: int = 11,
        logscale: bool = True,
        scale: float = 1.0,
        merge_size: int = 32,
        num_depth_samples: int = 64,
        z_near: float = 0.1,
        z_far: float = 2.0,
        attn_dim: int | None = None,
        subpatch_valid_threshold: float = 1.0,
        premerge_vision_queries: bool = False,
        backbone: HighResPatch3DPositionEmbedding | None = None,
    ) -> None:
        super().__init__()
        if not 0.0 < z_near < z_far:
            raise ValueError(f"need 0 < z_near < z_far, got {z_near}, {z_far}")
        self.backbone = backbone or HighResPatch3DPositionEmbedding(
            in_channels=in_channels,
            num_pos_feats=num_pos_feats,
            n_freqs=n_freqs,
            logscale=logscale,
            scale=scale,
            merge_size=merge_size,
        )
        self.vision_dim = int(vision_dim)
        self.attn_dim = int(attn_dim or self.backbone.num_pos_feats)
        self.premerge_vision_queries = bool(premerge_vision_queries)
        self.subpatch_valid_threshold = float(subpatch_valid_threshold)
        F = self.backbone.freq_out_channels
        C = self.backbone.num_pos_feats
        if self.premerge_vision_queries:
            self.subpatch_query_proj = nn.Linear(self.vision_dim, self.attn_dim)
            nn.init.xavier_uniform_(self.subpatch_query_proj.weight, gain=0.01)
            nn.init.zeros_(self.subpatch_query_proj.bias)
        else:
            self.query_expand = nn.Linear(self.vision_dim, 4 * self.attn_dim)
            nn.init.xavier_uniform_(self.query_expand.weight, gain=0.01)
            nn.init.zeros_(self.query_expand.bias)
        self.key_proj = nn.Linear(F, self.attn_dim, bias=False)
        self.value_proj = nn.Linear(F, self.attn_dim, bias=False)
        self.out_proj = nn.Linear(self.attn_dim, C)
        nn.init.xavier_uniform_(self.key_proj.weight, gain=0.01)
        nn.init.xavier_uniform_(self.value_proj.weight, gain=0.01)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.01)
        nn.init.zeros_(self.out_proj.bias)
        self.register_buffer(
            "depth_samples",
            torch.linspace(float(z_near), float(z_far), int(num_depth_samples)),
            persistent=False,
        )

    def _vision_queries(self, vision_feat: torch.Tensor, n_spatial: int) -> torch.Tensor:
        """Build ``[B,N,4,attn_dim]`` depth-route queries."""
        feat = vision_feat.reshape(vision_feat.shape[0], -1, vision_feat.shape[-1])
        B, n_tok, d = (int(feat.shape[0]), int(feat.shape[1]), int(feat.shape[2]))
        if d != self.vision_dim:
            raise ValueError(f"vision_feat dim {d} != Frustum vision_dim {self.vision_dim}")
        if self.premerge_vision_queries:
            if n_tok != n_spatial * 4:
                raise ValueError(
                    f"premerge_vision_queries=True expects {n_spatial * 4} tokens (N*4 subpatches), got {n_tok} from shape {tuple(vision_feat.shape)}"
                )
            return self.subpatch_query_proj(feat.view(B, n_spatial, 4, d))
        if n_tok == n_spatial + 1:
            feat = feat[:, 1:, :]
            n_tok = n_spatial
        if n_tok != n_spatial:
            raise ValueError(
                f"premerge_vision_queries=False expects {n_spatial} (or +1 CLS) tokens, got {n_tok} from shape {tuple(vision_feat.shape)}"
            )
        return self.query_expand(feat).view(B, n_spatial, 4, self.attn_dim)

    def _candidate_xyz(
        self,
        intrinsic: torch.Tensor,
        extrinsic: torch.Tensor,
        height: int,
        width: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Subpatch-center rays → ``(xyz_cam, xyz_world)`` each ``[B,N,4,K,3]``."""
        bb = self.backbone
        m, s = (bb.merge_size, bb.sub_merge_size)
        B, device = (intrinsic.shape[0], intrinsic.device)
        Hf, Wf = (height // m, width // m)
        N = Hf * Wf
        sy = torch.tensor([0, 0, 1, 1], device=device)
        sx = torch.tensor([0, 1, 0, 1], device=device)
        hf = torch.arange(Hf, device=device)[:, None, None]
        wf = torch.arange(Wf, device=device)[None, :, None]
        v = (hf * m + sy * s + 0.5 * s).expand(Hf, Wf, 4).reshape(1, N, 4).expand(B, -1, -1)
        u = (wf * m + sx * s + 0.5 * s).expand(Hf, Wf, 4).reshape(1, N, 4).expand(B, -1, -1)
        K = self.depth_samples.numel()
        z = self.depth_samples.to(device=device, dtype=torch.float32).view(1, 1, 1, K)
        fx = intrinsic[:, 0, 0].float().view(B, 1, 1, 1).clamp(min=1e-06)
        fy = intrinsic[:, 1, 1].float().view(B, 1, 1, 1).clamp(min=1e-06)
        cx = intrinsic[:, 0, 2].float().view(B, 1, 1, 1)
        cy = intrinsic[:, 1, 2].float().view(B, 1, 1, 1)
        x_cam = (u.unsqueeze(-1) - cx) / fx * z
        y_cam = (v.unsqueeze(-1) - cy) / fy * z
        xyz_cam = torch.stack([x_cam, y_cam, z.expand_as(x_cam)], dim=-1)
        cam_h = torch.cat([xyz_cam, torch.ones_like(xyz_cam[..., :1])], dim=-1)
        xyz_world = torch.matmul(
            cam_h.reshape(B, -1, 4), torch.linalg.inv(extrinsic.float()).transpose(-2, -1)
        ).view(B, N, 4, K, 4)[..., :3]
        return (xyz_cam.to(dtype=dtype), xyz_world.to(dtype=dtype))

    def _route(self, vision_feat: torch.Tensor, xyz_cam: torch.Tensor, xyz_world: torch.Tensor):
        """Q (vision) / K (cam) / V (world); returns subpatch features and depth weights."""
        B, N, _, K, _ = xyz_cam.shape
        q = self._vision_queries(vision_feat, N)
        with torch.no_grad():
            freq_cam = self.backbone.frequency_encoding(xyz_cam.reshape(B, N * 4 * K, 3).float())
            freq_world = self.backbone.frequency_encoding(
                xyz_world.reshape(B, N * 4 * K, 3).float()
            )
        F = self.backbone.freq_out_channels
        freq_cam = freq_cam.to(dtype=q.dtype).view(B, N, 4, K, F)
        freq_world = freq_world.to(dtype=q.dtype).view(B, N, 4, K, F)
        keys = self.key_proj(freq_cam)
        values = self.value_proj(freq_world)
        logits = (q.unsqueeze(-2) * keys).sum(-1) * (1.0 / math.sqrt(self.attn_dim))
        attn = torch.softmax(logits, dim=-1)
        sub_route = self.out_proj((attn.unsqueeze(-1) * values).sum(dim=3))
        return (sub_route, attn)

    def _broadcast(self, values_sub: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """``[B,N,4,C]`` constant-fill each 16×16 → ``[B,H,W,C]``."""
        B, N, _, C = values_sub.shape
        m, s = (self.backbone.merge_size, self.backbone.sub_merge_size)
        Hf, Wf = (height // m, width // m)
        grid = (
            values_sub.view(B, Hf, Wf, 2, 2, C)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .repeat_interleave(s, dim=2)
            .repeat_interleave(s, dim=4)
        )
        return grid.reshape(B, height, width, C)

    def forward(
        self,
        xyz_hw: torch.Tensor,
        xyz_mask_hw: torch.Tensor | None = None,
        *,
        vision_feat: torch.Tensor | None = None,
        intrinsic: torch.Tensor | None = None,
        extrinsic: torch.Tensor | None = None,
    ):
        if vision_feat is None or intrinsic is None or extrinsic is None:
            pos, patch_mask = self.backbone(xyz_hw, xyz_mask_hw)
            return (pos, patch_mask, {"xyz_routed_hw": xyz_hw})
        B, H, W, _ = xyz_hw.shape
        m = self.backbone.merge_size
        N = H // m * (W // m)
        depth_mask = (
            torch.zeros(B, H, W, device=xyz_hw.device, dtype=torch.bool)
            if xyz_mask_hw is None
            else xyz_mask_hw.bool()
        )
        sub_reliable = (
            self.backbone.subpatch_valid_fraction(depth_mask) >= self.subpatch_valid_threshold
        )
        sub_use_depth = sub_reliable
        depth_mask_for_xyz = depth_mask
        freq_depth = self.backbone.encode_xyz_to_freq(xyz_hw)
        sub_depth = self.backbone.patch_merge(self.backbone.reshape_freq_to_subpatches(freq_depth))
        xyz_cam, xyz_world = self._candidate_xyz(intrinsic, extrinsic, H, W, freq_depth.dtype)
        sub_route, attn = self._route(vision_feat, xyz_cam, xyz_world)
        sub_hybrid = torch.where(sub_use_depth.unsqueeze(-1), sub_depth, sub_route)
        pos = self.backbone.position_embedding_head(sub_hybrid.mean(dim=2))
        patch_mask = torch.ones(B, N, device=pos.device, dtype=torch.bool)
        xyz_route = self._broadcast((attn.unsqueeze(-1) * xyz_world).sum(dim=3), H, W)
        xyz_hybrid = torch.where(depth_mask_for_xyz.unsqueeze(-1), xyz_hw, xyz_route)
        extras: dict[str, torch.Tensor] = {"xyz_routed_hw": xyz_hybrid}
        return (pos, patch_mask, extras)


class Chunked4DRotaryEmbedding(nn.Module):
    """Chunked 4D RoPE over x/y/z/1D axes."""

    def __init__(self, axis_dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        if axis_dim % 2 != 0:
            raise ValueError("axis_dim must be even")
        self.axis_dim = int(axis_dim)
        inv_freq = 1.0 / float(theta) ** (
            torch.arange(0, axis_dim, 2, dtype=torch.float32) / axis_dim
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, position_ids: torch.Tensor, *, dtype: torch.dtype | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim != 3 or position_ids.shape[-1] != 4:
            raise ValueError(f"position_ids must be [B,S,4], got {tuple(position_ids.shape)}")
        inv = self.inv_freq.to(device=position_ids.device, dtype=torch.float32)
        freqs = position_ids.float().unsqueeze(-1) * inv.view(1, 1, 1, -1)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos, sin = (emb.cos(), emb.sin())
        if dtype is not None:
            cos, sin = (cos.to(dtype=dtype), sin.to(dtype=dtype))
        return (cos, sin)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension in RoPE pairs."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_chunked_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply chunked multi-axis RoPE to q/k.

    q/k have shape [B, H, T, D]. cos/sin have shape [B, T, A, Da],
    where D = A * Da. Each axis chunk is rotated independently.
    """
    if cos.shape != sin.shape:
        raise ValueError(
            f"cos and sin must have the same shape, got {tuple(cos.shape)} and {tuple(sin.shape)}"
        )
    if cos.ndim != 4:
        raise ValueError(f"cos/sin must have shape [B,T,A,Da], got {tuple(cos.shape)}")
    B, _, T, D = q.shape
    if k.shape != q.shape:
        raise ValueError(
            f"q and k must have the same shape, got {tuple(q.shape)} and {tuple(k.shape)}"
        )
    if cos.shape[0] != B or cos.shape[1] != T:
        raise ValueError(
            f"cos/sin batch/length must match q/k, got {tuple(cos.shape)} for q {tuple(q.shape)}"
        )
    axes = int(cos.shape[2])
    axis_dim = int(cos.shape[3])
    if D != axes * axis_dim:
        raise ValueError(f"q/k head_dim={D} must equal axes*axis_dim={axes * axis_dim}")
    if axis_dim % 2 != 0:
        raise ValueError(f"RoPE axis_dim must be even, got {axis_dim}")
    H = q.shape[1]
    q_axes = q.float().view(B, H, T, axes, axis_dim)
    k_axes = k.float().view(B, H, T, axes, axis_dim)
    cos_axes = cos.unsqueeze(1).float()
    sin_axes = sin.unsqueeze(1).float()
    q_out = q_axes * cos_axes + rotate_half(q_axes) * sin_axes
    k_out = k_axes * cos_axes + rotate_half(k_axes) * sin_axes
    return (
        q_out.reshape(B, H, T, D).to(dtype=q.dtype),
        k_out.reshape(B, H, T, D).to(dtype=k.dtype),
    )
