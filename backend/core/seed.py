"""Seed handling. CPU generators are used everywhere: they are the most reproducible on MPS."""
from __future__ import annotations

import secrets

import torch

MAX_SEED = 2**32 - 1


def resolve_seed(seed: int | None) -> int:
    """Return ``seed`` if given (and valid), else a fresh random 32-bit seed."""
    if seed is None or seed < 0:
        return secrets.randbelow(MAX_SEED + 1)
    return int(seed) & MAX_SEED


def make_generator(seed: int) -> torch.Generator:
    """A CPU ``torch.Generator`` seeded with ``seed``."""
    return torch.Generator("cpu").manual_seed(int(seed))
