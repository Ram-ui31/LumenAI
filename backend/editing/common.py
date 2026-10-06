"""Shared editing plumbing: the fill backend, windowed (crop-and-scale) filling, LaMa, result saving."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image

from core.config import get_config
from core.device import MemoryTracker, dtype_name, get_device, get_dtype
from core.recipe import Recipe, new_output_stem, save_with_recipe
from core.seed import make_generator

log = logging.getLogger("adobeps.edit")
StepFn = Callable[[int, int], None]
Box = tuple[int, int, int, int]   # x0, y0, x1, y1 (exclusive)


@dataclass
class EditResult:
    image: Image.Image
    recipe: Recipe
    path: Path | None
    generated: np.ndarray | None = None       # raw model output (full canvas) before blending
    region: np.ndarray | None = None
    passes: list["EditResult"] = field(default_factory=list)


def fill_backend_name() -> str:
    b = get_config()["backend"]
    return "stub" if b == "stub" else "diffusers"   # mflux has no Fill model; editing always uses diffusers


def describe_fill() -> dict[str, Any]:
    if fill_backend_name() == "stub":
        return {"model_id": "stub-fill", "transformer_weights": None, "quantization": None, "dtype": "float32",
                "device": "cpu", "scheduler": "none", "backend": "stub"}
    c = get_config()["fill"]
    return {"model_id": c["repo"], "transformer_weights": f"{c['gguf_repo']}/{c['gguf_file']}",
            "quantization": c["quantization"], "dtype": dtype_name(get_dtype(get_config().get("dtype", "bfloat16"))),
            "device": get_device(), "scheduler": c["scheduler"], "backend": "diffusers"}


# ---- model calls -----------------------------------------------------------------------

def fill_region(image: Image.Image, region: np.ndarray, prompt: str, seed: int, steps: int, guidance: float,
                strength: float, latent_preserve: bool, tracker: MemoryTracker, on_step: StepFn | None = None) -> Image.Image:
    """FLUX.1-Fill (or stub) on an image whose dims are multiples of 16."""
    if fill_backend_name() == "stub":
        return _stub_fill(image, region, seed, strength)
    from core.model_manager import get_manager
    from core.text import encode
    from editing.latent_preserve import run_fill

    c = get_config()["fill"]
    emb, _ = encode(prompt, c["max_sequence_length"])
    tracker.sample()
    pipe = get_manager().get("flux_fill")

    def step(i: int, n: int) -> None:
        tracker.sample()
        if on_step:
            on_step(i, n)

    out = run_fill(pipe, image, region, emb["prompt_embeds"], emb["pooled_prompt_embeds"], seed=seed, steps=steps,
                   guidance_scale=guidance, strength=strength, latent_preserve=latent_preserve,
                   stop_frac=get_config()["editing"]["latent_preserve_stop_frac"],
                   max_sequence_length=c["max_sequence_length"], on_step=step)
    tracker.sample()
    return out


def _stub_fill(image: Image.Image, region: np.ndarray, seed: int, strength: float) -> Image.Image:
    arr = np.asarray(image.convert("RGB"))
    m = (region > 0).astype(np.uint8)
    filled = cv2.inpaint(arr, m, 5, cv2.INPAINT_TELEA)
    noise = make_generator(seed).manual_seed(seed)
    import torch

    n = (torch.rand(arr.shape, generator=noise).numpy() - 0.5) * 30
    out = filled.astype(np.float32) + n * m[..., None]
    if strength < 1:
        out = strength * out + (1 - strength) * arr
    # stub is not latent-preserving; mimic VAE drift outside the mask so blending is exercised
    out = out + (1 - m[..., None]) * 2.0
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def lama_inpaint(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Big-LaMa inpainting (uint8 HxWx3, mask HxW). Output outside the mask is left to the caller's blend."""
    if get_config()["backend"] == "stub":
        return cv2.inpaint(img, (mask > 0).astype(np.uint8), 5, cv2.INPAINT_TELEA)
    import torch

    from core.model_manager import get_manager

    model, dev = get_manager().get("lama")
    h, w = img.shape[:2]
    ph, pw = (-h) % 8, (-w) % 8
    im = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="symmetric").astype(np.float32) / 255
    mk = np.pad((mask > 0).astype(np.float32), ((0, ph), (0, pw)), mode="symmetric")
    ti = torch.from_numpy(im.transpose(2, 0, 1))[None].to(dev)
    tm = torch.from_numpy(mk)[None, None].to(dev)
    with torch.inference_mode():
        out = model(ti, tm)[0].permute(1, 2, 0).float().cpu().numpy()
    return np.clip(out[:h, :w] * 255, 0, 255).astype(np.uint8)


# ---- windows -----------------------------------------------------------------------------

def _round16_down(x: float) -> int:
    return max(16, int(x // 16) * 16)


def work_size(w: int, h: int, max_pixels: int, min_short: int = 512) -> tuple[int, int]:
    """Working resolution for a w x h crop: <= max_pixels, short side >= min_short when possible,
    dims multiples of 16."""
    scale = 1.0
    if min(w, h) < min_short:
        scale = min_short / min(w, h)
    if w * h * scale * scale > max_pixels:
        scale = math.sqrt(max_pixels / (w * h))
    return _round16_down(w * scale), _round16_down(h * scale)


def expand_box(box: Box, pad: int, W: int, H: int, min_size: int = 512) -> Box:
    """Grow ``box`` by ``pad`` px, then to at least ``min_size`` per side, clamped to the image."""
    x0, y0, x1, y1 = box
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad

    def grow(a: int, b: int, lim: int) -> tuple[int, int]:
        need = min(min_size, lim) - (b - a)
        if need > 0:
            a -= need // 2
            b += need - need // 2
        if a < 0:
            b, a = b - a, 0
        if b > lim:
            a, b = max(0, a - (b - lim)), lim
        return a, b

    x0, x1 = grow(x0, x1, W)
    y0, y1 = grow(y0, y1, H)
    return x0, y0, x1, y1


def fill_window(canvas: np.ndarray, region: np.ndarray, box: Box, **fill_kw: Any) -> tuple[np.ndarray, tuple[int, int]]:
    """Fill ``region`` inside ``box`` of ``canvas`` at a capped working size; paste back at full res.
    Returns (generated_full_canvas, (work_w, work_h))."""
    x0, y0, x1, y1 = box
    crop = canvas[y0:y1, x0:x1]
    rc = region[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    ww, wh = work_size(cw, ch, get_config()["fill"]["max_pixels"])
    work = Image.fromarray(crop).resize((ww, wh), Image.LANCZOS)
    wmask = cv2.resize(rc.astype(np.uint8), (ww, wh), interpolation=cv2.INTER_NEAREST)
    if (ww, wh) != (cw, ch):
        wmask = cv2.dilate(wmask, np.ones((3, 3), np.uint8))   # cover resampling at the edge
    out = fill_region(work, wmask, **fill_kw)
    back = np.asarray(out.convert("RGB").resize((cw, ch), Image.LANCZOS))
    full = canvas.copy()
    full[y0:y1, x0:x1] = back
    return full, (ww, wh)


# ---- saving ------------------------------------------------------------------------------

def save_edit(result_img: Image.Image, recipe: Recipe, source: Image.Image, mask: np.ndarray,
              generated: np.ndarray | None = None) -> Path:
    """Save output + recipe, plus ``_source.png``, ``_mask.png`` (and ``_raw.png``) next to it.
    The recipe records their paths so the edit can be replayed exactly."""
    stem = new_output_stem(recipe.task, recipe.seed)
    src_p, mask_p = stem.with_name(stem.name + "_source.png"), stem.with_name(stem.name + "_mask.png")
    source.convert("RGB").save(src_p)
    Image.fromarray(((mask > 0) * 255).astype(np.uint8)).save(mask_p)
    recipe.edit = {**(recipe.edit or {}), "source_path": str(src_p), "mask_path": str(mask_p)}
    if generated is not None:
        raw_p = stem.with_name(stem.name + "_raw.png")
        Image.fromarray(generated).save(raw_p)
        recipe.edit["raw_path"] = str(raw_p)
    return save_with_recipe(result_img, recipe, stem)
