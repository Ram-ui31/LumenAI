"""Small vision-language model client: MLX on Mac (mlx-vlm), transformers on CUDA, stub for tests.

Only one entry point matters: :func:`ask_json(images, prompt)` → dict parsed from the model's JSON answer.
"""
from __future__ import annotations

import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

from core.config import get_config
from core.device import get_device

log = logging.getLogger("adobeps.vlm")


def vlm_backend() -> str:
    b = get_config()["backend"]
    if b == "stub":
        return "stub"
    want = get_config()["vlm"].get("backend", "auto")
    if want != "auto":
        return want
    return "mlx" if get_device() == "mps" else "transformers"


def load_vlm() -> Any:
    """Registered as heavy model 'vlm'. Returns (kind, model, processor, config)."""
    c = get_config()["vlm"]
    import os

    if os.environ.get("ADOBEPS_VLM"):          # e.g. ADOBEPS_VLM=mlx-community/Qwen3-VL-8B-Instruct-4bit
        c = {**c, "mlx_repo": os.environ["ADOBEPS_VLM"], "hf_repo": os.environ["ADOBEPS_VLM"]}
    if vlm_backend() == "mlx":
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        model, proc = load(c["mlx_repo"])
        return "mlx", model, proc, load_config(c["mlx_repo"])
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = torch.bfloat16 if get_device() == "cuda" else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(c["hf_repo"], torch_dtype=dtype).to(get_device()).eval()
    return "transformers", model, AutoProcessor.from_pretrained(c["hf_repo"]), None


def ask(images: list[Image.Image], prompt: str, max_tokens: int | None = None) -> str:
    """Raw text answer (greedy decoding). ``images`` may be empty (text-only question)."""
    from core.model_manager import get_manager

    kind, model, proc, cfg = get_manager().get("vlm")
    max_tokens = int(max_tokens or get_config()["vlm"]["max_tokens"])
    if kind == "mlx":
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        with tempfile.TemporaryDirectory() as d:
            paths = []
            for i, im in enumerate(images):
                p = Path(d) / f"img{i}.png"
                im.convert("RGB").save(p)
                paths.append(str(p))
            text = apply_chat_template(proc, cfg, prompt, num_images=len(paths))
            out = generate(model, proc, text, paths or None, max_tokens=max_tokens, temperature=0.0, verbose=False)
        return getattr(out, "text", out)
    import torch

    msgs = [{"role": "user", "content": [*({"type": "image", "image": im.convert("RGB")} for im in images),
                                          {"type": "text", "text": prompt}]}]
    inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                      return_tensors="pt").to(model.device)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    return proc.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def parse_json(text: str) -> dict[str, Any]:
    """First {...} object in ``text`` (tolerates ```json fences, trailing prose, single quotes, True/False)."""
    t = re.sub(r"```(?:json)?", "", text)
    start = t.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in VLM answer: {text[:200]!r}")
    depth = 0
    for i in range(start, len(t)):
        depth += {"{": 1, "}": -1}.get(t[i], 0)
        if depth == 0:
            blob = t[start:i + 1]
            break
    else:
        blob = t[start:] + "}"
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        fixed = re.sub(r"\bTrue\b", "true", re.sub(r"\bFalse\b", "false", blob.replace("'", '"')))
        fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
        return json.loads(fixed)


def ask_json(images: list[Image.Image], prompt: str, stub: dict[str, Any] | None = None) -> tuple[dict, str]:
    """(parsed answer, raw text). On a parse failure the raw text is kept and {} returned."""
    if vlm_backend() == "stub":
        return dict(stub or {}), json.dumps(stub or {})
    raw = ask(images, prompt)
    try:
        return parse_json(raw), raw
    except Exception as e:  # noqa: BLE001
        log.warning("VLM answer not parseable (%s): %s", e, raw[:300])
        return {}, raw


def yes_no(text: str) -> bool | None:
    """Parse a short yes/no answer (first yes/no word wins); None if neither appears."""
    m = re.search(r"\b(yes|no)\b", text.lower())
    return None if m is None else m.group(1) == "yes"


def ask_short(images: list[Image.Image], question: str, stub: str = "no") -> str:
    """A short free-text answer (yes/no, one word or a short list)."""
    if vlm_backend() == "stub":
        return stub
    return ask(images, question, max_tokens=60).strip()
