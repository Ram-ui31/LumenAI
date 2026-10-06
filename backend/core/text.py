"""Prompt encoding with caching. T5 is loaded only on a cache miss and is evicted (as a heavy model)
before the FLUX transformer loads."""
from __future__ import annotations

import torch

from core.config import get_config
from core.device import dtype_name, get_device, get_dtype
from core.prompt_cache import Embeds, PromptCache


def encode(prompt: str, max_sequence_length: int) -> tuple[Embeds, bool]:
    """Return ({prompt_embeds, pooled_prompt_embeds}, cache_hit)."""
    cfg = get_config()
    repo = cfg["text_encoders"]["repo"]
    dtype = get_dtype(cfg.get("dtype", "bfloat16"))

    def _enc(p: str, L: int) -> Embeds:
        from core.model_manager import get_manager

        pipe = get_manager().get("text_encoders")
        with torch.no_grad():
            pe, pooled, _ = pipe.encode_prompt(
                prompt=p, prompt_2=None, device=get_device(), num_images_per_prompt=1, max_sequence_length=L
            )
        return {"prompt_embeds": pe, "pooled_prompt_embeds": pooled}

    return PromptCache().get_or_encode(prompt, repo, max_sequence_length, dtype_name(dtype), _enc)
