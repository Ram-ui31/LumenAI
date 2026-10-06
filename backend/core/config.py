"""Load config.yaml once and resolve project-relative paths."""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    """Return the parsed config.yaml. ``ADOBEPS_BACKEND`` env var overrides ``backend``."""
    path = Path(os.environ.get("ADOBEPS_CONFIG", PROJECT_ROOT / "config.yaml"))
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if os.environ.get("ADOBEPS_BACKEND"):
        cfg["backend"] = os.environ["ADOBEPS_BACKEND"]
    mem = cfg.setdefault("memory", {})
    if os.environ.get("ADOBEPS_QUANT"):
        mem["quant"] = os.environ["ADOBEPS_QUANT"]
    q = mem.get("quant")
    if q:  # one switch for both GGUF transformers: rewrite the quant suffix of each file name
        for sec in ("generation", "fill"):
            c = cfg[sec]
            c["gguf_file"] = re.sub(r"(Q\d_[A-Z0-9_]+|F16)(?=\.gguf$)", q, c["gguf_file"])
            c["quantization"] = f"gguf-{q}"
    return cfg


def project_path(key: str) -> Path:
    """Absolute path for an entry of ``config.paths``; the directory is created if missing."""
    p = Path(get_config()["paths"][key])
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p
