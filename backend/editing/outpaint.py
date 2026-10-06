"""Generative outpainting with Innovation A + B. Large expansions run iteratively (one side / chunk per
pass), each pass saving its own recipe linked to the previous one via ``parent_recipe``."""
from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np
from PIL import Image

from core.config import get_config
from core.device import MemoryTracker
from core.recipe import Recipe, sha256_image
from core.seed import resolve_seed
from editing.blending import composite
from editing.common import Box, EditResult, StepFn, describe_fill, fill_window, save_edit
from editing.masking import dilate
from generation.generate import log_run

SIDES = ("top", "right", "bottom", "left")


def round_expansion(w: int, h: int, left: int, right: int, top: int, bottom: int) -> dict[str, int]:
    """Adjust amounts so the final canvas is a multiple of 16 on each axis. The remainder goes to an
    expanded side on that axis (right/bottom preferred); if an axis isn't expanded but isn't a multiple
    of 16, a thin strip is added at right/bottom (recorded in the recipe)."""
    amt = {"left": max(0, int(left)), "right": max(0, int(right)), "top": max(0, int(top)), "bottom": max(0, int(bottom))}
    for a, b, size in (("left", "right", w), ("top", "bottom", h)):
        extra = (-(size + amt[a] + amt[b])) % 16
        if extra:
            amt[b if amt[b] > 0 or amt[a] == 0 else a] += extra
    return amt


def prepare_canvas(img: np.ndarray, amt: dict[str, int], fill: str) -> tuple[np.ndarray, np.ndarray]:
    """Pad with edge-replication (optionally heavily blurred). Returns (canvas, hole) where hole = new area."""
    h, w = img.shape[:2]
    canvas = cv2.copyMakeBorder(img, amt["top"], amt["bottom"], amt["left"], amt["right"], cv2.BORDER_REPLICATE)
    hole = np.ones(canvas.shape[:2], np.uint8)
    hole[amt["top"]:amt["top"] + h, amt["left"]:amt["left"] + w] = 0
    if fill == "blur" and hole.any():
        sigma = max(8.0, 0.08 * max(canvas.shape[:2]))
        blurred = cv2.GaussianBlur(canvas, (0, 0), sigmaX=sigma)
        canvas = np.where(hole[..., None] > 0, blurred, canvas).astype(np.uint8)
    return canvas, hole


def pass_window(canvas_shape: tuple[int, int], amt: dict[str, int], context_px: int | None) -> Box:
    """Whole canvas for a single pass; for a one-side pass, the new strip plus ``context_px`` of image."""
    H, W = canvas_shape
    if context_px is None:
        return 0, 0, W, H
    if amt["top"]:
        return 0, 0, W, min(H, amt["top"] + context_px)
    if amt["bottom"]:
        return 0, max(0, H - amt["bottom"] - context_px), W, H
    if amt["left"]:
        return 0, 0, min(W, amt["left"] + context_px), H
    return max(0, W - amt["right"] - context_px), 0, W, H


def outpaint_pass(image: Image.Image, amt: dict[str, int], prompt: str, seed: int, overlap_px: int,
                  steps: int, guidance: float, latent_preserve: bool, blend: str, fill: str,
                  context_px: int | None, parent_recipe: str | None, save: bool,
                  on_step: StepFn | None = None) -> EditResult:
    cfg = get_config()
    e = cfg["editing"]
    src = image.convert("RGB")
    O = np.asarray(src)
    canvas, hole = prepare_canvas(O, amt, fill)
    # region = new area + overlap band into the original
    region = np.maximum(hole, dilate(hole, overlap_px))
    box = pass_window(canvas.shape[:2], amt, context_px)
    tracker, t0 = MemoryTracker(), time.perf_counter()
    G, work = fill_window(canvas, region, box, prompt=prompt, seed=seed, steps=steps, guidance=guidance, strength=1.0,
                          latent_preserve=latent_preserve, tracker=tracker, on_step=on_step)
    out = composite(canvas, G, hole, region, mode=blend, levels=e["pyramid_levels"], band_high_px=e["blend_band_high_px"],
                    band_low_px=e["blend_band_low_px"], feather_px=e["feather_px"])
    result = Image.fromarray(out)
    recipe = Recipe(
        task="outpaint", prompt=prompt, steps=steps, guidance_scale=guidance, width=work[0], height=work[1], seed=seed,
        max_sequence_length=cfg["fill"]["max_sequence_length"], parent_recipe=parent_recipe, **describe_fill(),
        edit={"source_image_sha256": sha256_image(src), "latent_preserve": latent_preserve, "blend": blend,
              "mask_dilate_px": 0, "outpaint": {**amt, "overlap_px": overlap_px, "fill": fill, "context_px": context_px},
              "window_box": list(box), "source_size": list(src.size), "canvas_size": [canvas.shape[1], canvas.shape[0]]},
    )
    recipe.stats = {"seconds": round(time.perf_counter() - t0, 2), **tracker.report()}
    path = None
    if save:
        path = save_edit(result, recipe, src, region, G)
        log_run(recipe, path)
    return EditResult(result, recipe, path, generated=G, region=region)


def plan_passes(w: int, h: int, amt: dict[str, int], max_pixels: int, chunk_px: int,
                iterative: bool | None) -> list[dict[str, int]]:
    """One pass if the whole canvas fits the working cap (and not forced iterative), else one side /
    chunk per pass in the order top, right, bottom, left."""
    W, H = w + amt["left"] + amt["right"], h + amt["top"] + amt["bottom"]
    if iterative is False or (iterative is None and W * H <= max_pixels):
        return [amt]
    passes = []
    for side in SIDES:
        left = amt[side]
        while left > 0:
            step = min(chunk_px, left)
            passes.append({s: (step if s == side else 0) for s in SIDES})
            left -= step
    return passes


def outpaint(image: Image.Image, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0,
             prompt: str | None = None, seed: int | None = None, overlap_px: int | None = None,
             steps: int | None = None, guidance_scale: float | None = None, latent_preserve: bool | None = None,
             blend: str | None = None, fill: str | None = None, iterative: bool | None = None,
             save: bool = True, parent_recipe: str | None = None, on_step: StepFn | None = None) -> EditResult:
    """Expand ``image`` by the given pixels per side. Returns the final result; ``.passes`` holds each pass."""
    cfg = get_config()
    oc, fc, e = cfg["outpaint"], cfg["fill"], cfg["editing"]
    prompt = oc["prompt"] if prompt is None else prompt
    seed = resolve_seed(seed)
    overlap_px = oc["overlap_px"] if overlap_px is None else int(overlap_px)
    steps = int(steps or fc["steps"])
    guidance = float(fc["guidance_scale"] if guidance_scale is None else guidance_scale)
    latent_preserve = e["latent_preserve"] if latent_preserve is None else bool(latent_preserve)
    blend = blend or e["blend"]
    fill = fill or oc["fill"]
    w, h = image.size
    amt = round_expansion(w, h, left, right, top, bottom)
    if not any(amt.values()):
        raise ValueError("Nothing to expand: set at least one side > 0.")
    plan = plan_passes(w, h, amt, fc["max_pixels"], oc["chunk_px"], iterative)
    current, parent, passes = image, parent_recipe, []
    for i, p in enumerate(plan):
        ctx = None if len(plan) == 1 else oc["context_px"]
        res = outpaint_pass(current, p, prompt, seed + i, overlap_px, steps, guidance, latent_preserve, blend, fill,
                            ctx, parent, save, on_step)
        passes.append(res)
        current = res.image
        parent = str(res.path.with_suffix(".json")) if res.path else parent
    final = passes[-1]
    final.passes = passes
    return final


def replay_kwargs(recipe: Recipe) -> dict[str, Any]:
    """Arguments to re-run ONE pass exactly (outpaint_pass signature)."""
    ed = recipe.edit or {}
    op = dict(ed["outpaint"])
    return dict(amt={s: op[s] for s in SIDES}, prompt=recipe.prompt, seed=recipe.seed, overlap_px=op["overlap_px"],
                steps=recipe.steps, guidance=recipe.guidance_scale, latent_preserve=ed["latent_preserve"],
                blend=ed["blend"], fill=op["fill"], context_px=op.get("context_px"))
