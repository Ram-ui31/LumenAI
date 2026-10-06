"""Selection-aware removal with a self-check (the production removal path).

Why (v4 audit, 42 shadow scenes): most "shadow left behind" cases were really the TARGET left behind - RORD targets
include toy cars, animals, bikes, pots, and our instruction always said "remove the person" (v3 wording even told Qwen
to skip vehicles). Real leftover shadows after a successful removal were rare.

So:
  1. plan     - Qwen3-VL sees the photo AND the photo with the user's selection highlighted, and lists everything
                inside the selection (people, animals, objects, what they hold or ride) + describes the background.
  2. remove   - draft 512 px -> sharpen 768 px on a crop around the selection (v4 default: same quality as 1024,
                31 % faster, 12 GB).
  3. check    - GT-free: share of the selection Qwen barely changed (CIEDE2000 < 6). On 150 v2/v4 results this flagged
                19/19 failed removals with 0/131 false alarms at > 0.25.
  4. retry    - (a) new seed + firmer wording, (b) start from LaMa's fill and let Qwen add detail (LaMa always removes),
                (c) LaMa alone. The first attempt that passes is used.
  5. finish   - shadow trick (editing.shadow) + seam-smoothing composite.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

TARGET_Q = ("The second image is the first image with an area highlighted in red. That area marks what must be removed. "
            "List every person, animal and object inside the red area, including anything they hold, carry, push or ride. "
            "Answer only with a short comma-separated list of nouns, for example: person, bicycle")
SCENE_Q = ("In one short sentence, describe only the floor, walls and background surfaces around the red-highlighted area of "
           "the second image. Do not mention what is inside the red area.")
KEEP = "Keep this photo exactly the same, with natural, sharp, realistic detail."
CHECK_DE = 6.0
CHECK_MAX_UNCHANGED = 0.25


@dataclass
class RemovalResult:
    image: np.ndarray                 # final composite (uint8)
    raw: np.ndarray                   # the accepted generative output, full frame
    plan: dict
    attempts: list = field(default_factory=list)
    info: dict = field(default_factory=dict)


def highlight(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Photo with the selection tinted red and outlined (what the user selected, shown to the VLM)."""
    a = np.asarray(image.convert("RGB")).astype(np.float32)
    m = mask > 0
    a[m] = 0.45 * a[m] + 0.55 * np.array([230, 30, 30])
    edge = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)) > 0
    a[edge] = [255, 0, 0]
    return Image.fromarray(a.astype(np.uint8))


def clean_list(text: str) -> list[str]:
    t = re.sub(r"[\n\.]", ",", text.lower())
    items = [re.sub(r"^(a|an|the)\s+", "", x.strip(" -*:\"'")) for x in t.split(",")]
    out = []
    for x in items:
        if x and x not in ("nothing", "none", "n/a") and len(x) < 40 and x not in out:
            out.append(x)
    return out[:6] or ["person"]


def plan_removal(ask, image: Image.Image, mask: np.ndarray) -> dict:
    """``ask(images, question, max_new_tokens) -> str`` (Qwen3-VL). Returns targets, scene, instruction, firm_instruction."""
    hi = highlight(image, mask)
    targets = clean_list(ask([image, hi], TARGET_Q, 40))
    scene = ask([image, hi], SCENE_Q, 60).strip()
    things = ", ".join(targets[:-1]) + (" and " if len(targets) > 1 else "") + targets[-1]
    instr = (f"Remove the {things} from this photo, together with any shadow they cast. "
             f"Fill the space with what is behind them: {scene} Keep everything else in the photo exactly unchanged.")
    firm = (f"Completely erase the {things} from this photo. Nothing of them may remain: no body parts, no outline, "
            f"no shadow, no reflection. Show only the empty background: {scene} Keep everything else exactly unchanged.")
    return {"targets": targets, "scene": scene, "instruction": instr, "firm_instruction": firm}


def unchanged_share(original: np.ndarray, result: np.ndarray, mask: np.ndarray, de: float = CHECK_DE) -> float:
    """GT-free failure check: the share of the selection whose colour the edit barely changed."""
    from editing.recolour import delta_e2000, rgb_to_cielab

    m = mask > 0
    if not m.any():
        return 0.0
    if result.shape != original.shape:
        result = np.asarray(Image.fromarray(result).resize(original.shape[1::-1], Image.LANCZOS))
    o = cv2.GaussianBlur(original, (0, 0), 1.0)[m]
    r = cv2.GaussianBlur(result, (0, 0), 1.0)[m]
    return float((delta_e2000(rgb_to_cielab(o), rgb_to_cielab(r)) < de).mean())


def remove(pipe, Q, image: Image.Image, mask: np.ndarray, hole: np.ndarray, region: np.ndarray, seed: int = 7,
           steps: int = 40, plan: dict | None = None, lama: np.ndarray | None = None, max_attempts: int = 4,
           composite_fn=None) -> RemovalResult:
    """Full removal with self-check and fallbacks. ``mask`` = user selection, ``hole`` = selection + shadow-aware
    extension, ``region`` = hole dilated for blending (as produced by editing.remove / stage_lama).
    ``lama``: LaMa's full-frame fill (computed on demand if None and needed)."""
    from editing.masking import dilate
    from editing.shadow import shadow_from_edit

    t0 = time.perf_counter()
    O = np.asarray(image.convert("RGB"))
    H, W = O.shape[:2]
    if plan is None:
        plan = plan_removal(lambda ims, q, n: Q.ask(pipe, ims, q, max_new_tokens=n), image, mask)
    keep_prompt = f"{KEEP} {plan.get('scene', '')}".strip()
    x0, y0, x1, y1 = Q.crop_window(region)
    crop_img = Image.fromarray(O[y0:y1, x0:x1])
    fit_area = np.zeros((H, W), np.uint8)
    fit_area[y0:y1, x0:x1] = 1

    def paste(img: Image.Image) -> np.ndarray:
        full = O.copy()
        full[y0:y1, x0:x1] = np.asarray(img.convert("RGB").resize((x1 - x0, y1 - y0), Image.LANCZOS))
        return full

    def draft_refine(instr: str, sd: int):
        d = Q.edit(pipe, crop_img, instr, seed=sd, steps=steps, resolution=512)
        r = Q.refine(pipe, d.image, keep_prompt, seed=sd, strength=0.3, steps=steps, resolution=768, condition=d.image)
        return paste(r.image), d.seconds + r.seconds, max(d.peak_gb, r.peak_gb)

    def lama_start(sd: int):
        L = lama if lama is not None else _lama(O, region)
        lc = Image.fromarray(L[y0:y1, x0:x1])
        r = Q.refine(pipe, lc, keep_prompt, seed=sd, strength=0.6, steps=steps, resolution=768, condition=lc)
        return paste(r.image), r.seconds, r.peak_gb

    def lama_only(sd: int):
        L = lama if lama is not None else _lama(O, region)
        return L.copy(), 0.0, 0.0

    plan_list = [("draft-refine", lambda: draft_refine(plan["instruction"], seed)),
                 ("draft-refine firm, new seed", lambda: draft_refine(plan["firm_instruction"], seed + 101)),
                 ("LaMa start + Qwen detail", lambda: lama_start(seed)),
                 ("LaMa only", lambda: lama_only(seed))][:max_attempts]
    attempts, raw, chosen = [], None, None
    for name, fn in plan_list:
        try:
            out, sec, peak = fn()
        except Exception as e:  # noqa: BLE001
            attempts.append({"name": name, "error": repr(e)[:200]})
            continue
        u = unchanged_share(O, out, mask)
        ok = u <= CHECK_MAX_UNCHANGED or name == "LaMa only"
        attempts.append({"name": name, "unchanged": round(u, 3), "ok": bool(ok), "seconds": round(sec, 2),
                         "peak_gb": round(peak, 2)})
        raw = out
        if ok:
            chosen = name
            break
    if raw is None:
        raise RuntimeError(f"all removal attempts failed: {attempts}")
    S = shadow_from_edit(O, raw, (mask > 0).astype(np.uint8), hole, fit_area=fit_area)
    h2, r2 = np.maximum(hole, S), np.maximum(region, dilate(S, 16))
    if composite_fn is None:
        from editing.blending import composite_back as composite_fn
    final = composite_fn(O, Image.fromarray(raw), h2, r2)
    info = {"chosen": chosen or attempts[-1]["name"], "n_attempts": len(attempts), "shadow_px": int(S.sum()),
            "crop": [x0, y0, x1, y1], "seconds_total": round(time.perf_counter() - t0, 2)}
    return RemovalResult(final, raw, plan, attempts, info)


def _lama(O: np.ndarray, region: np.ndarray) -> np.ndarray:
    from editing.common import lama_inpaint

    return np.asarray(lama_inpaint(O, region))
