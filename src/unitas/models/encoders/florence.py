# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
import importlib.util
import logging
import os
import torch
from torch import nn
import torch.nn.functional as F
from typing import Mapping, Sequence
from .modeling_florence2 import Florence2ForConditionalGeneration
from .configuration_florence2 import Florence2Config

logger = logging.getLogger(__name__)


def _select_attention(device: torch.device) -> str:
    """Prefer the optional FlashAttention 2 backend when its CUDA kernel works."""
    if (
        device.type != "cuda"
        or torch.version.cuda is None
        or not torch.cuda.is_available()
        or torch.cuda.get_device_capability(device)[0] < 8
        or importlib.util.find_spec("flash_attn") is None
    ):
        return "sdpa"
    try:
        from flash_attn import flash_attn_func, flash_attn_varlen_func
        from flash_attn.bert_padding import pad_input, unpad_input
        from transformers.utils import is_flash_attn_2_available

        if not is_flash_attn_2_available():
            return "sdpa"
        # Test both kernels used by Florence, without advancing the sampling RNG.
        with torch.inference_mode():
            q = torch.zeros(1, 2, 1, 64, device=device, dtype=torch.bfloat16)
            flash_attn_func(q, q, q)
            mask = torch.ones(1, 2, device=device, dtype=torch.bool)
            unpadded, indices, lengths, maximum, *_ = unpad_input(q, mask)
            output = flash_attn_varlen_func(
                unpadded, unpadded, unpadded, lengths, lengths, maximum, maximum
            )
            pad_input(output, indices, 1, 2)
            torch.cuda.synchronize(device)
    except torch.OutOfMemoryError:
        raise
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        logger.warning("FlashAttention 2 is unavailable; using SDPA: %s", error)
        return "sdpa"
    return "flash_attention_2"


@dataclass(frozen=True)
class Florence2VLMEncoding:
    """Florence image-language features used by the action decoder."""

    vlm_features: torch.Tensor
    visual_tokens: torch.Tensor
    language_tokens: torch.Tensor
    image_features: torch.Tensor
    aux_visual_inputs: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    image_mask: torch.Tensor
    processed_image_hw: tuple[int, int]

    @property
    def num_image_tokens(self) -> int:
        return int(self.visual_tokens.shape[1])


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    normalized = str(name).lower()
    if normalized in {"auto", "none"}:
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float32
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32", "full"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype '{name}'")


def _image_tensor_to_pil_list(images: torch.Tensor) -> tuple[list[object], int, int]:
    from PIL import Image

    if images.ndim == 4:
        if images.shape[-1] != 3:
            raise ValueError(f"image tensor must be [B,H,W,3], got {tuple(images.shape)}")
        batch, views = (int(images.shape[0]), 1)
        flat = images
    elif images.ndim == 5:
        if images.shape[-1] != 3:
            raise ValueError(f"image tensor must be [B,V,H,W,3], got {tuple(images.shape)}")
        batch, views = (int(images.shape[0]), int(images.shape[1]))
        flat = images.reshape(batch * views, *images.shape[2:])
    else:
        raise ValueError(
            f"image tensor must be [B,H,W,3] or [B,V,H,W,3], got {tuple(images.shape)}"
        )
    flat_cpu = flat.detach().cpu()
    if flat_cpu.dtype != torch.uint8:
        max_val = float(flat_cpu.max().item()) if flat_cpu.numel() else 1.0
        if max_val <= 1.0 + 1e-05:
            flat_cpu = flat_cpu * 255.0
        flat_cpu = flat_cpu.clamp(0, 255).to(torch.uint8)
    pil_images = [Image.fromarray(img.numpy()).convert("RGB") for img in flat_cpu]
    return (pil_images, batch, views)


def _images_to_pil_list(images) -> tuple[list[object], int, int]:
    from PIL import Image

    if isinstance(images, Image.Image):
        return ([images.convert("RGB")], 1, 1)
    if isinstance(images, (list, tuple)) and images and isinstance(images[0], Image.Image):
        return ([img.convert("RGB") for img in images], len(images), 1)
    if torch.is_tensor(images):
        return _image_tensor_to_pil_list(images)
    raise TypeError("images must be a PIL image, a list of PIL images, or a torch tensor")


def resolve_language_batch(
    batch: Mapping[str, object],
    batch_size: int,
    fallback_instruction: str | Sequence[str] | None = None,
) -> list[str]:
    """Resolve language instructions from a dataloader batch or a CLI fallback."""
    for key in ("language_instruction", "instruction", "language", "task", "text", "prompt"):
        if key not in batch:
            continue
        value = batch[key]
        if isinstance(value, str):
            return [value] * batch_size
        if isinstance(value, (list, tuple)):
            texts = [str(item) for item in value]
            if len(texts) != batch_size:
                raise ValueError(
                    f"batch field {key!r} has {len(texts)} entries, expected {batch_size}"
                )
            return texts
    if fallback_instruction is None:
        return [""] * batch_size
    if isinstance(fallback_instruction, str):
        return [fallback_instruction] * batch_size
    texts = [str(item) for item in fallback_instruction]
    if len(texts) != batch_size:
        raise ValueError(f"fallback instruction has {len(texts)} entries, expected {batch_size}")
    return texts


class Florence2VLMEncoder:
    """X-VLA-style Florence2 image-language encoder.

    The core path intentionally mirrors X-VLA:

    1. encode valid images with ``Florence2ForConditionalGeneration._encode_image``;
    2. merge first-view image tokens before language token embeddings;
    3. run ``language_model.model.encoder`` with bidirectional self-attention;
    4. return conditioned first-view VLM tokens plus raw auxiliary-view tokens.
    """

    def __init__(
        self,
        model_name: str = "microsoft/Florence-2-large",
        device: str | torch.device = "cuda",
        dtype: str = "fp32",
        cache_dir: str | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        trust_remote_code: bool = True,
        attn_implementation: str = "auto",
        image_size: int = 768,
    ):
        from transformers import AutoImageProcessor, AutoProcessor

        self.model_name = model_name
        self.device = torch.device(device)
        self.dtype = _resolve_dtype(dtype, self.device)
        if attn_implementation not in {"auto", "sdpa", "flash_attention_2"}:
            raise ValueError(f"Unsupported attention backend: {attn_implementation}")
        if attn_implementation != "sdpa":
            attn_implementation = _select_attention(self.device)
        self.attn_implementation = attn_implementation
        logger.info("Florence attention: %s", attn_implementation)
        self.image_size = int(image_size)
        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}")
        processor_kwargs = {
            "trust_remote_code": trust_remote_code,
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
        }
        image_processor_kwargs = {
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
            "use_fast": False,
        }
        model_kwargs = {
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
            "attn_implementation": str(attn_implementation),
        }
        if revision is not None:
            processor_kwargs["revision"] = revision
            image_processor_kwargs["revision"] = revision
            model_kwargs["revision"] = revision
        if self.device.type == "cuda":
            model_kwargs["dtype"] = self.dtype
        florence_config = Florence2Config.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            **{"revision": revision} if revision is not None else {},
        )
        florence_config._attn_implementation = str(attn_implementation)
        florence_config.text_config._attn_implementation = str(attn_implementation)
        model_kwargs["config"] = florence_config
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase

        if not hasattr(PreTrainedTokenizerBase, "additional_special_tokens"):

            def _get_additional_special_tokens(tokenizer):
                return list(getattr(tokenizer, "_florence_additional_special_tokens", []))

            def _set_additional_special_tokens(tokenizer, tokens):
                tokenizer._florence_additional_special_tokens = list(tokens or [])

            PreTrainedTokenizerBase.additional_special_tokens = property(
                _get_additional_special_tokens, _set_additional_special_tokens
            )
        self.processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)
        self.processor.image_processor = AutoImageProcessor.from_pretrained(
            model_name, **image_processor_kwargs
        )
        text_config = florence_config.text_config
        tokenizer = self.processor.tokenizer
        tokenizer.add_special_tokens(
            {
                "bos_token": tokenizer.convert_ids_to_tokens(text_config.bos_token_id),
                "pad_token": tokenizer.convert_ids_to_tokens(text_config.pad_token_id),
                "eos_token": tokenizer.convert_ids_to_tokens(text_config.eos_token_id),
            }
        )
        self.model = Florence2ForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
        self._retie_language_input_embeddings(self.model)
        resolved_attn = getattr(
            self.model.config.text_config,
            "_attn_implementation",
            getattr(self.model.config, "_attn_implementation", None),
        )
        if resolved_attn != str(attn_implementation):
            raise RuntimeError(
                f"Florence-2 language attention backend mismatch: requested={attn_implementation!r}, resolved={resolved_attn!r}"
            )
        self._prune_unused_generation_modules(self.model)
        self.model.to(self.device)
        self.model.vision_tower.enable_checkpoint = False
        self.model.gradient_checkpointing_disable()
        self.model.requires_grad_(False).eval()

    @staticmethod
    def _retie_language_input_embeddings(model: nn.Module) -> None:
        """Restore Florence's shared token embedding after Transformers 5.x load.

        Some 5.x releases materialize ``encoder.embed_tokens`` as a separate,
        randomly initialized parameter even though the Florence checkpoint
        stores the pretrained table as ``language_model.model.shared``.  The
        ActionFlow path obtains token embeddings through ``get_input_embeddings``
        and then calls the encoder with ``inputs_embeds``, so the duplicate is
        unused but still inflates FSDP and optimizer state.  Rebind it to the
        pretrained shared module, matching Florence's original tied-weight
        structure and newer Transformers releases.
        """
        language_model = getattr(model, "language_model", None)
        inner_model = getattr(language_model, "model", None)
        shared = getattr(inner_model, "shared", None)
        encoder = getattr(inner_model, "encoder", None)
        if not isinstance(shared, nn.Module) or not isinstance(encoder, nn.Module):
            raise RuntimeError("Florence language model is missing shared/encoder embeddings")
        encoder.embed_tokens = shared

    @staticmethod
    def _prune_unused_generation_modules(model: nn.Module) -> None:
        """Drop generation-only Florence modules before moving the encoder to GPU."""
        language_model = getattr(model, "language_model", None)
        if not isinstance(language_model, nn.Module):
            return
        inner_model = getattr(language_model, "model", None)
        if isinstance(inner_model, nn.Module) and hasattr(inner_model, "decoder"):
            delattr(inner_model, "decoder")
        if hasattr(language_model, "lm_head"):
            delattr(language_model, "lm_head")
        if "final_logits_bias" in getattr(language_model, "_buffers", {}):
            del language_model._buffers["final_logits_bias"]

    def preprocess_images(self, images) -> tuple[torch.Tensor, tuple[int, int], int, int]:
        pil_images, batch, views = _images_to_pil_list(images)
        image_size = {"height": self.image_size, "width": self.image_size}
        pixel_values = self.processor.image_processor(
            pil_images, return_tensors="pt", size=image_size
        )["pixel_values"]
        processed_hw = (int(pixel_values.shape[-2]), int(pixel_values.shape[-1]))
        pixel_values = pixel_values.view(batch, views, *pixel_values.shape[1:]).contiguous()
        return (pixel_values, processed_hw, batch, views)

    def tokenize_instructions(
        self, instructions: str | Sequence[str], batch_size: int
    ) -> tuple[torch.LongTensor, torch.Tensor]:
        """不限制 language 长度：有多少 token 编码多少，batch 内按最长序列 padding。"""
        if isinstance(instructions, str):
            texts = [instructions] * batch_size
        else:
            texts = [str(item) for item in instructions]
        if len(texts) != batch_size:
            raise ValueError(f"expected {batch_size} instructions, got {len(texts)}")
        try:
            max_tokens = max(1, int(os.environ.get("ACTION_FLOW_FLORENCE_MAX_TEXT_TOKENS", "512")))
        except (TypeError, ValueError):
            max_tokens = 512
        encoded = self.processor.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_tokens,
            return_token_type_ids=False,
        )
        return (encoded["input_ids"], encoded["attention_mask"])

    @torch.no_grad()
    def encode_images_and_text(
        self,
        images,
        language_instruction: str | Sequence[str],
        *,
        image_mask: torch.Tensor | None = None,
    ) -> Florence2VLMEncoding:
        pixel_values, processed_hw, batch, views = self.preprocess_images(images)
        input_ids, text_attention_mask = self.tokenize_instructions(language_instruction, batch)
        if image_mask is None:
            image_mask = torch.ones(batch, views, dtype=torch.bool)
        return self.forward_vlm(
            input_ids=input_ids.to(self.device),
            pixel_values=pixel_values.to(self.device),
            image_mask=image_mask.to(self.device),
            text_attention_mask=text_attention_mask.to(self.device),
            processed_image_hw=processed_hw,
        )

    def forward_vlm(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.Tensor,
        image_mask: torch.Tensor | None = None,
        text_attention_mask: torch.Tensor | None = None,
        processed_image_hw: tuple[int, int] | None = None,
    ) -> Florence2VLMEncoding:
        """Encode text plus images exactly like X-VLA's ``forward_vlm``.

        Args:
            input_ids: ``[B, L]`` tokenized instruction ids.
            pixel_values: ``[B, V, C, H, W]`` preprocessed images. A 4D
                ``[B, C, H, W]`` tensor is accepted as a single-view shortcut.
            image_mask: ``[B, V]`` bool mask. Invalid views are zero-filled.
            text_attention_mask: ``[B, L]`` tokenizer attention mask; padding
                positions are excluded from encoder attention and propagated
                downstream. ``None`` falls back to all-ones.
        """
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [B,L], got {tuple(input_ids.shape)}")
        if text_attention_mask is not None and text_attention_mask.shape != input_ids.shape:
            raise ValueError(
                f"text_attention_mask must have shape {tuple(input_ids.shape)}, got {tuple(text_attention_mask.shape)}"
            )
        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(1)
        if pixel_values.ndim != 5:
            raise ValueError(
                f"pixel_values must have shape [B,V,C,H,W], got {tuple(pixel_values.shape)}"
            )
        if input_ids.shape[0] != pixel_values.shape[0]:
            raise ValueError("Batch size mismatch between input_ids and pixel_values")
        B, V = (int(pixel_values.shape[0]), int(pixel_values.shape[1]))
        if image_mask is None:
            image_mask = torch.ones(B, V, dtype=torch.bool, device=pixel_values.device)
        if image_mask.shape != (B, V):
            raise ValueError(f"image_mask must have shape {(B, V)}, got {tuple(image_mask.shape)}")
        input_ids = input_ids.to(self.device)
        pixel_values = pixel_values.to(self.device)
        if self.device.type == "cuda" and pixel_values.dtype.is_floating_point:
            pixel_values = pixel_values.to(self.dtype)
        image_mask = image_mask.to(self.device).bool()
        flat_mask = image_mask.reshape(-1)
        flat_images = pixel_values.flatten(0, 1)
        num_valid = int(flat_mask.sum().item())
        if num_valid == 0:
            raise ValueError("At least one image view must be valid per batch.")
        valid_images = flat_images[flat_mask]
        valid_feats = self.model._encode_image(valid_images)
        if valid_feats.ndim != 3:
            raise ValueError(
                f"Florence image encoder must return [B,N,D], got {tuple(valid_feats.shape)}"
            )
        N, D = (int(valid_feats.shape[1]), int(valid_feats.shape[2]))
        image_features = valid_feats.new_zeros((B * V, N, D))
        image_features[flat_mask] = valid_feats
        image_features = image_features.view(B, V, N, D)
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        merged_embeds, attention_mask = self.model._merge_input_ids_with_image_features(
            image_features[:, 0],
            inputs_embeds,
            task_prefix_attention_mask=text_attention_mask.to(self.device)
            if text_attention_mask is not None
            else None,
        )
        enc_out = self.model.language_model.model.encoder(
            attention_mask=attention_mask, inputs_embeds=merged_embeds
        )[0]
        language_tokens = enc_out[:, N : N + input_ids.shape[1]]
        aux_visual_inputs = image_features[:, 1:].reshape(B, -1, D)
        if processed_image_hw is None:
            processed_image_hw = (int(pixel_values.shape[-2]), int(pixel_values.shape[-1]))
        return Florence2VLMEncoding(
            vlm_features=enc_out,
            visual_tokens=enc_out[:, :N],
            language_tokens=language_tokens,
            image_features=image_features,
            aux_visual_inputs=aux_visual_inputs,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_mask=image_mask,
            processed_image_hw=processed_image_hw,
        )


def image_hw_to_world_points(
    depth: torch.Tensor,
    intrinsic: torch.Tensor,
    extrinsic: torch.Tensor,
    *,
    processed_image_hw: tuple[int, int],
    depth_min: float = 1e-06,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backproject every pixel at VLM input resolution to world-frame xyz.

    Depth is resized to ``processed_image_hw`` and intrinsics are scaled to match.
    Returns ``xyz`` ``[B,H,W,3]`` and ``valid`` ``[B,H,W]``.
    """
    if depth.ndim != 3:
        raise ValueError(f"depth must have shape (B,H,W), got {tuple(depth.shape)}")
    if intrinsic.ndim != 3 or intrinsic.shape[-2:] != (3, 3):
        raise ValueError(f"intrinsic must have shape (B,3,3), got {tuple(intrinsic.shape)}")
    if extrinsic.ndim != 3 or extrinsic.shape[-2:] != (4, 4):
        raise ValueError(f"extrinsic must have shape (B,4,4), got {tuple(extrinsic.shape)}")
    B, H0, W0 = depth.shape
    Hp, Wp = (int(processed_image_hw[0]), int(processed_image_hw[1]))
    if Hp <= 0 or Wp <= 0:
        raise ValueError(f"processed_image_hw must be positive, got {processed_image_hw}")
    if (H0, W0) != (Hp, Wp):
        depth_r = F.interpolate(depth.unsqueeze(1).float(), size=(Hp, Wp), mode="nearest").squeeze(
            1
        )
        scale_x = float(Wp) / float(W0)
        scale_y = float(Hp) / float(H0)
        intr = intrinsic.float().clone()
        intr[:, 0, 0] = intr[:, 0, 0] * scale_x
        intr[:, 0, 2] = intr[:, 0, 2] * scale_x
        intr[:, 1, 1] = intr[:, 1, 1] * scale_y
        intr[:, 1, 2] = intr[:, 1, 2] * scale_y
    else:
        depth_r = depth.float()
        intr = intrinsic.float()
    u = torch.arange(Wp, device=depth.device, dtype=torch.float32) + 0.5
    v = torch.arange(Hp, device=depth.device, dtype=torch.float32) + 0.5
    yy, xx = torch.meshgrid(v, u, indexing="ij")
    z = depth_r
    fx = intr[:, 0, 0].view(B, 1, 1)
    fy = intr[:, 1, 1].view(B, 1, 1)
    cx = intr[:, 0, 2].view(B, 1, 1)
    cy = intr[:, 1, 2].view(B, 1, 1)
    valid_intrinsic = (fx > 0.0) & (fy > 0.0)
    valid = torch.isfinite(z) & (z > float(depth_min)) & valid_intrinsic
    safe_fx = torch.where(fx > 0.0, fx, torch.ones_like(fx))
    safe_fy = torch.where(fy > 0.0, fy, torch.ones_like(fy))
    x_cam = (xx.view(1, Hp, Wp) - cx) / safe_fx * z
    y_cam = (yy.view(1, Hp, Wp) - cy) / safe_fy * z
    ones = torch.ones_like(z)
    cam_h = torch.stack([x_cam, y_cam, z, ones], dim=-1)
    cam_to_world = torch.linalg.inv(extrinsic.float())
    world_h = torch.matmul(cam_h.view(B, Hp * Wp, 4), cam_to_world.transpose(-2, -1))
    xyz = world_h[..., :3].view(B, Hp, Wp, 3)
    xyz = torch.where(valid.unsqueeze(-1), xyz, torch.zeros_like(xyz))
    return (xyz.to(dtype=depth.dtype), valid)
