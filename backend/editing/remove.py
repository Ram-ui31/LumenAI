"""Object removal: lama | flux | lama+flux, with Innovation A (latent preserve) and B (Laplacian blend)."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
from PIL import Image

from core.config import get_config
from core.device import MemoryTracker
from core.recipe import Recipe, sha256_image
from core.seed import resolve_seed
from editing.blending import composite
from editing.common import EditResult, StepFn, describe_fill, expand_box, fill_window, lama_inpaint, save_edit
from editing.masking import dilate, mask_bbox, reshape, shadow_extend, to_binary
from generation.generate import log_run

MODES = ("lama", "flux", "lama+flux")


def removal_box(region: np.ndarray, W: int, H: int) -> tuple[int, int, int, int]:
    """Crop around the (dilated) mask with context, so FLUX sees enough surroundings at a sane size."""
    bb = mask_bbox(region)
    if bb is None:
        raise ValueError("Mask is empty: click the object (or paint it) first.")
    x0, y0, x1, y1 = bb
    pad = max(64, int(get_config()["editing"]["crop_context_frac"] * max(x1 - x0, y1 - y0)))
    return expand_box(bb, pad, W, H)


def remove_object(image: Image.Image, mask: np.ndarray, mode: str | None = None, seed: int | None = None,
                  dilate_px: int | None = None, latent_preserve: bool | None = None, blend: str | None = None,
                  prompt: str | None = None, steps: int | None = None, strength: float | None = None,
                  guidance_scale: float | None = None, mask_shape: str | None = None, shadow_aware: bool | None = None, save: bool = True, parent_recipe: str | None = None,
                  on_step: StepFn | None = None) -> EditResult:
    """Remove the masked object. ``mask`` is the object mask (it gets dilated by ``dilate_px``)."""
    cfg = get_config()
    e, rc, fc = cfg["editing"], cfg["remove"], cfg["fill"]
    mode = mode or rc["default_mode"]
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    dilate_px = int(rc.get("dilate_px", e["mask_dilate_px"])) if dilate_px is None else int(dilate_px)
    shadow_aware = bool(rc.get("shadow_aware", False)) if shadow_aware is None else bool(shadow_aware)
    latent_preserve = e["latent_preserve"] if latent_preserve is None else bool(latent_preserve)
    blend = blend or e["blend"]
    auto_bg = None
    if prompt is None and rc.get("auto_prompt") and mode != "lama":
        from editing.bg_prompt import describe_background

        prompt, auto_bg = describe_background(image, dilate(reshape(to_binary(mask), mask_shape or rc.get("mask_shape", "exact")), int(dilate_px)))
    prompt = rc["prompt"] if prompt is None else prompt
    steps = int(steps or fc["steps"])
    guidance = float(fc["guidance_scale"] if guidance_scale is None else guidance_scale)
    strength = float(rc["lama_flux_strength"] if strength is None else strength) if mode == "lama+flux" else 1.0
    seed = resolve_seed(seed)

    src = image.convert("RGB")
    O = np.asarray(src)
    H, W = O.shape[:2]
    obj = to_binary(mask)
    if obj.shape != (H, W):
        raise ValueError(f"mask shape {obj.shape} != image {(H, W)}")
    hole = shadow_extend(obj) if shadow_aware else obj   # include the cast shadow below the object
    mask_shape = mask_shape or rc.get("mask_shape", "exact")
    region = dilate(reshape(hole, mask_shape), dilate_px)

    tracker, t0 = MemoryTracker(), time.perf_counter()
    box, work = None, (W, H)
    if mode == "lama":
        G = lama_inpaint(O, region)
        desc = {"model_id": "big-lama", "transformer_weights": cfg["lama"]["file"], "quantization": None,
                "dtype": "float32", "device": describe_fill()["device"], "scheduler": "none",
                "backend": describe_fill()["backend"]}
        steps_used, guidance_used = 0, 0.0
    else:
        base = O
        if mode == "lama+flux":
            L = lama_inpaint(O, region)
            base = np.where(region[..., None] > 0, L, O).astype(np.uint8)
        box = removal_box(region, W, H)
        G, work = fill_window(base, region, box, prompt=prompt, seed=seed, steps=steps, guidance=guidance,
                              strength=strength, latent_preserve=latent_preserve, tracker=tracker, on_step=on_step)
        desc, steps_used, guidance_used = describe_fill(), steps, guidance

    out = composite(O, G, hole, region, mode=blend, levels=e["pyramid_levels"], band_high_px=e["blend_band_high_px"],
                    band_low_px=e["blend_band_low_px"], feather_px=e["feather_px"])
    result = Image.fromarray(out)
    recipe = Recipe(
        task="remove", prompt=prompt if mode != "lama" else "", steps=steps_used, guidance_scale=guidance_used,
        width=work[0], height=work[1], seed=seed, max_sequence_length=fc["max_sequence_length"],
        parent_recipe=parent_recipe, **desc,
        edit={"mode": mode, "source_image_sha256": sha256_image(src), "mask_dilate_px": dilate_px,
              "latent_preserve": latent_preserve if mode != "lama" else None, "blend": blend,
              "strength": strength, "mask_shape": mask_shape, "shadow_aware": shadow_aware, "crop_box": list(box) if box else None, "source_size": [W, H]},
    )
    recipe.stats = {"seconds": round(time.perf_counter() - t0, 2), **tracker.report()}
    if auto_bg:
        recipe.extra["auto_prompt"] = {"top_backgrounds": auto_bg}
    if cfg["critics"].get("verify_removal"):
        from critics.detector import verify_removal

        recipe.extra["removal_check"] = verify_removal(src, result, region)
    path = None
    if save:
        path = save_edit(result, recipe, src, obj, G)
        log_run(recipe, path)
    return EditResult(result, recipe, path, generated=G, region=region)


def replay_kwargs(recipe: Recipe) -> dict[str, Any]:
    ed = recipe.edit or {}
    return dict(mode=ed["mode"], seed=recipe.seed, dilate_px=ed["mask_dilate_px"], latent_preserve=ed.get("latent_preserve"),
                mask_shape=ed.get("mask_shape", "exact"), shadow_aware=ed.get("shadow_aware", False),
                blend=ed["blend"], prompt=recipe.prompt or None, steps=recipe.steps or None, strength=ed.get("strength"),
                guidance_scale=recipe.guidance_scale if recipe.steps else None)


def retry_removal(recipe_path: str, on_step: StepFn | None = None) -> EditResult:
    """The UI's "Retry": same source and mask, a new seed and ``retry_dilate_step`` px more dilation."""
    from pathlib import Path

    from core.recipe import load_recipe

    r = load_recipe(recipe_path)
    ed = r.edit or {}
    src = Image.open(ed["source_path"]).convert("RGB")
    mask = (np.asarray(Image.open(ed["mask_path"]).convert("L")) > 127).astype(np.uint8)
    kw = replay_kwargs(r)
    kw["seed"] = None
    kw["dilate_px"] = int(kw["dilate_px"]) + int(get_config()["critics"]["retry_dilate_step"])
    return remove_object(src, mask, parent_recipe=str(Path(recipe_path).with_suffix(".json")), on_step=on_step, **kw)
