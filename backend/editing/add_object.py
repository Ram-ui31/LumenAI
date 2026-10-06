"""AI object suggester + adder.

suggest(): Qwen3-VL looks at the photo and proposes objects that would naturally fit, each with a box (where it
           should stand, at a realistic size) and a one-line reason.
add():     puts the object inside that box only. The edit area = box + a contact/shadow zone below it; outside it the
           photo is unchanged (seam-smoothing composite). Three placement methods:
             centre - crop the photo so the box is in the middle, ask Qwen to "add X in the centre"
             ghost  - paste a grey placeholder box where X should stand, ask Qwen to "replace the grey box with X"
             keep   - latent preservation: every step resets everything outside the edit area to the photo, so Qwen can
                      only create inside it (x = m x + (1-m)((1-s) z_photo + s eps))
check():   Qwen3-VL grounds X in the result; success = found, centre inside the edit area, IoU with the box > 0.25.
           Shadow: mean lightness in the strip under the object drops (contact shadow present).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

SUGGEST_Q = ("Suggest {n} different objects that could be added to this photo so that it still looks natural and "
             "realistic (things that typically belong in such a place). For each, give where it should stand as a box "
             "at a realistic size for this scene, on a visible surface, not covering important things. Reply with JSON "
             'only: [{{"object": "potted plant", "reason": "...", "bbox_2d": [x1, y1, x2, y2]}}, ...]')
GROUND_Q = ('Locate the {obj} in the image. Output JSON: [{{"bbox_2d": [x1, y1, x2, y2], "label": "{obj}"}}]. '
            "If there is none, output [].")


@dataclass
class Suggestion:
    object: str
    reason: str
    box: tuple[int, int, int, int]          # pixels


@dataclass
class AddResult:
    image: np.ndarray
    raw_crop: np.ndarray
    method: str
    check: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)


def _boxes(text: str) -> list[tuple[float, float, float, float]]:
    return [tuple(float(g) for g in m.groups()) for m in
            re.finditer(r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]", text)]


def to_px(b, size):
    w, h = size
    x0, y0, x1, y1 = b
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    return (int(np.clip(x0 * w / 1000, 0, w - 1)), int(np.clip(y0 * h / 1000, 0, h - 1)),
            int(np.clip(x1 * w / 1000, 1, w)), int(np.clip(y1 * h / 1000, 1, h)))


def parse_suggestions(text: str, size: tuple[int, int]) -> list[Suggestion]:
    out = []
    m = re.search(r"\[.*\]", text, re.S)
    try:
        items = json.loads(m.group(0)) if m else []
    except Exception:  # noqa: BLE001
        items = []
    for it in items if isinstance(items, list) else []:
        try:
            b = to_px(it["bbox_2d"], size)
        except Exception:  # noqa: BLE001
            continue
        w, h = b[2] - b[0], b[3] - b[1]
        if w < 12 or h < 12 or w * h > 0.5 * size[0] * size[1]:
            continue
        out.append(Suggestion(str(it.get("object", "object")).strip()[:40], str(it.get("reason", ""))[:160], b))
    return out


def suggest(ask, image: Image.Image, n: int = 5) -> list[Suggestion]:
    """``ask(images, question, max_new_tokens) -> str``."""
    return parse_suggestions(ask([image], SUGGEST_Q.format(n=n), 400), image.size)


def edit_area(box, size, below: float = 0.35, side: float = 0.12, above: float = 0.06) -> np.ndarray:
    """Box + a contact/shadow zone under it (objects cast shadows downward onto the surface they stand on)."""
    w, h = size
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    m = np.zeros((h, w), np.uint8)
    m[max(0, int(y0 - above * bh)):min(h, int(y1 + below * bh)), max(0, int(x0 - side * bw)):min(w, int(x1 + side * bw))] = 1
    return m


def _window(area: np.ndarray, size, grow: float = 1.2) -> tuple[int, int, int, int]:
    """Square-ish crop centred on the edit area, ~ (1 + 2*grow) x its size, inside the image."""
    ys, xs = np.nonzero(area)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    side = max(xs.max() - xs.min(), ys.max() - ys.min()) * (1 + 2 * grow)
    side = max(side, 256)
    w, h = size
    side = min(side, w, h)
    x0 = int(np.clip(cx - side / 2, 0, w - side))
    y0 = int(np.clip(cy - side / 2, 0, h - side))
    return x0, y0, int(x0 + side), int(y0 + side)


def _a(obj: str) -> str:
    return ("an " if obj[:1].lower() in "aeiou" else "a ") + obj


def add(pipe, Q, image: Image.Image, obj: str, box, method: str = "ghost", seed: int = 7, steps: int = 40,
        resolution: int = 768, composite_fn=None) -> AddResult:
    import torch

    t0 = time.perf_counter()
    O = np.asarray(image.convert("RGB"))
    H, W = O.shape[:2]
    area = edit_area(box, (W, H))
    x0, y0, x1, y1 = _window(area, (W, H))
    crop = O[y0:y1, x0:x1].copy()
    a_crop = area[y0:y1, x0:x1]
    bx = (box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0)
    instr_tail = (" It stands naturally on the surface at a realistic size, matching the photo's lighting, perspective and "
                  "colours, and casts a soft natural shadow onto the surface below it. Keep everything else exactly unchanged.")
    torch.cuda.reset_peak_memory_stats()
    if method == "centre":
        r = Q.edit(pipe, Image.fromarray(crop), f"Add {_a(obj)} in the centre of this photo." + instr_tail,
                   seed=seed, steps=steps, resolution=resolution)
    elif method == "ghost":
        g = crop.astype(np.float32)
        gx0, gy0, gx1, gy1 = bx
        g[gy0:gy1, gx0:gx1] = 0.25 * g[gy0:gy1, gx0:gx1] + 0.75 * 150
        cv2.rectangle(g, (gx0, gy0), (gx1 - 1, gy1 - 1), (90, 90, 90), 2)
        r = Q.edit(pipe, Image.fromarray(g.astype(np.uint8)),
                   f"Replace the grey box with {_a(obj)} of the same size and position. Remove the grey box completely; "
                   f"behind and around the {obj} show the original background." + instr_tail,
                   seed=seed, steps=steps, resolution=resolution)
    elif method == "keep":
        ci = Image.fromarray(crop)
        r = Q.refine(pipe, ci, f"Add {_a(obj)} in the marked area of this photo." + instr_tail, seed=seed, strength=0.9,
                     steps=steps, resolution=resolution, condition=ci, keep=(ci, a_crop))
    else:
        raise ValueError(method)
    raw = np.asarray(r.image.convert("RGB").resize((x1 - x0, y1 - y0), Image.LANCZOS))
    full = O.copy()
    full[y0:y1, x0:x1] = raw
    if composite_fn is None:
        from editing.blending import composite_back as composite_fn
    region = cv2.dilate(area, np.ones((9, 9), np.uint8))
    final = composite_fn(O, Image.fromarray(full), area, region)
    info = {"seconds": round(time.perf_counter() - t0, 2), "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "window": [x0, y0, x1, y1]}
    return AddResult(final, raw, method, {}, info)


def check(ask, original: np.ndarray, result: np.ndarray, obj: str, box) -> dict:
    """Did the object appear inside its box? + a contact-shadow measure. ``ask(images, q, n)``."""
    H, W = result.shape[:2]
    area = edit_area(box, (W, H))
    ys, xs = np.nonzero(area)
    x0, y0, x1, y1 = max(0, xs.min() - 40), max(0, ys.min() - 40), min(W, xs.max() + 40), min(H, ys.max() + 40)
    crop = Image.fromarray(result[y0:y1, x0:x1])
    bs = _boxes(ask([crop], GROUND_Q.format(obj=obj), 60))
    out = {"found": False, "iou": 0.0, "inside": False}
    if bs:
        gx0, gy0, gx1, gy1 = to_px(bs[0], crop.size)
        g = (gx0 + x0, gy0 + y0, gx1 + x0, gy1 + y0)
        ix = max(0, min(g[2], box[2]) - max(g[0], box[0])) * max(0, min(g[3], box[3]) - max(g[1], box[1]))
        u = (g[2] - g[0]) * (g[3] - g[1]) + (box[2] - box[0]) * (box[3] - box[1]) - ix
        cx, cy = (g[0] + g[2]) // 2, (g[1] + g[3]) // 2
        out = {"found": True, "iou": round(ix / max(u, 1), 3), "inside": bool(area[min(cy, H - 1), min(cx, W - 1)]),
               "found_box": list(map(int, g))}
    # contact shadow: strip directly under the box, inside the edit area
    bx0, by0, bx1, by1 = box
    bh = by1 - by0
    s0, s1 = by1, min(H, int(by1 + 0.3 * bh))
    if s1 > s0 + 2:
        L0 = cv2.cvtColor(original, cv2.COLOR_RGB2LAB)[s0:s1, bx0:bx1, 0].astype(float).mean()
        L1 = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)[s0:s1, bx0:bx1, 0].astype(float).mean()
        out["shadow_dL"] = round((L0 - L1) * 100 / 255, 2)       # CIE L units darker under the object
    # leftover placeholder grey (ghost method): flat, colourless ~150 grey that was not there before
    lab_r = cv2.cvtColor(result, cv2.COLOR_RGB2LAB).astype(float)
    lab_o = cv2.cvtColor(original, cv2.COLOR_RGB2LAB).astype(float)
    def greyish(lab):
        return (np.abs(lab[..., 0] - 157) < 14) & (np.abs(lab[..., 1] - 128) < 4) & (np.abs(lab[..., 2] - 128) < 4)
    bm = np.zeros((H, W), bool)
    bm[box[1]:box[3], box[0]:box[2]] = True
    out["grey_left"] = round(float((greyish(lab_r) & ~greyish(lab_o) & bm).sum() / max(bm.sum(), 1)), 3)
    out["success"] = bool(out["found"] and out["inside"] and out["iou"] > 0.25 and out["grey_left"] < 0.1)
    return out
