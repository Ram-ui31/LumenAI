"""Disk cache of text-encoder outputs (T5 + CLIP pooled), keyed by a hash of everything that affects them.

Encoding once and unloading T5 (~9 GB bf16) before the transformer loads is what keeps FLUX
within 24 GB unified memory.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import torch
from safetensors.torch import load_file, save_file

from core.config import project_path

Embeds = dict[str, torch.Tensor]   # {"prompt_embeds": [1,L,4096], "pooled_prompt_embeds": [1,768]}


def prompt_key(prompt: str, encoder_repo: str, max_sequence_length: int, dtype: str) -> str:
    blob = json.dumps(
        {"p": prompt, "repo": encoder_repo, "L": max_sequence_length, "dtype": dtype}, sort_keys=True
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


class PromptCache:
    """Stores embeddings as safetensors files under ``.cache/prompt_embeds``."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or (project_path("cache") / "prompt_embeds")
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        return self.root / f"{key}.safetensors"

    def get(self, key: str) -> Embeds | None:
        p = self.path(key)
        return load_file(str(p)) if p.exists() else None

    def put(self, key: str, embeds: Embeds) -> None:
        save_file({k: v.detach().to("cpu").contiguous() for k, v in embeds.items()}, str(self.path(key)))

    def get_or_encode(
        self,
        prompt: str,
        encoder_repo: str,
        max_sequence_length: int,
        dtype: str,
        encode_fn: Callable[[str, int], Embeds],
    ) -> tuple[Embeds, bool]:
        """Return (embeds, was_cached). ``encode_fn(prompt, max_len)`` is only called on a miss."""
        key = prompt_key(prompt, encoder_repo, max_sequence_length, dtype)
        hit = self.get(key)
        if hit is not None:
            return hit, True
        emb = encode_fn(prompt, max_sequence_length)
        self.put(key, emb)
        return emb, False
