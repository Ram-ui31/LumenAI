"""Masks: SAM 2.1 click-to-mask, dilation/feathering, conversion helpers for the UI."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from PIL import Image

from core.config import get_config


# ---- basic ops ------------------------------------------------------------------------

def to_binary(mask: np.ndarray | Image.Image, thresh: int = 127) -> np.ndarray:
    """Any mask (PIL/array, bool/uint8/RGB(A)) -> uint8 {0,1} HxW."""
    m = np.asarray(mask)
    if m.ndim == 3:
        if m.shape[2] == 4:          # painted layer: any opacity counts
            return (m[..., 3] > 0).astype(np.uint8)
        m = m.max(axis=2)
    if m.dtype == bool:
        return m.astype(np.uint8)
    if m.max() <= 1:
        return (m > 0).astype(np.uint8)
    return (m > thresh).astype(np.uint8)


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    """Dilate a binary mask by ``px`` pixels (elliptical kernel)."""
    m = to_binary(mask)
    if px <= 0:
        return m
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(m, k)


def shadow_extend(mask: np.ndarray, frac: float = 0.25, spread: float = 0.15) -> np.ndarray:
    """Grow the mask downward and a little sideways at its base, where cast shadows and contact
    shadows usually lie (light from above). ``frac`` = extension below as a fraction of object height."""
    m = to_binary(mask)
    bb = mask_bbox(m)
    if bb is None:
        return m
    x0, y0, x1, y1 = bb
    h, w = y1 - y0, x1 - x0
    down = max(4, int(frac * h))
    side = max(2, int(spread * w))
    k = np.zeros((2 * down + 1, 2 * side + 1), np.uint8)
    k[: down + 1, :] = 1                # cv2.dilate looks at src(y + ky - down): rows above -> extends DOWN
    return cv2.dilate(m, k, anchor=(side, down)) | m


def reshape(mask: np.ndarray, shape: str) -> np.ndarray:
    """Hide the object's silhouette from the fill model: ``exact`` (unchanged), ``hull`` (convex hull of
    each blob), ``box`` (bounding box). FLUX Fill tends to redraw *something shaped like the hole*."""
    m = to_binary(mask)
    if shape == "exact" or not m.any():
        return m
    if shape == "box":
        x0, y0, x1, y1 = mask_bbox(m)
        out = np.zeros_like(m)
        out[y0:y1, x0:x1] = 1
        return out
    if shape == "hull":
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = np.zeros_like(m)
        for c in cnts:
            cv2.fillConvexPoly(out, cv2.convexHull(c), 1)
        return out
    raise ValueError(f"mask shape must be exact|hull|box, not {shape!r}")


def feather(mask: np.ndarray, px: float) -> np.ndarray:
    """Gaussian-feathered float mask in [0,1]."""
    m = to_binary(mask).astype(np.float32)
    if px <= 0:
        return m
    return np.clip(cv2.GaussianBlur(m, (0, 0), sigmaX=px / 2), 0, 1)


def mask_to_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray((to_binary(mask) * 255).astype(np.uint8), mode="L")


def overlay(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int] = (255, 40, 40),
            alpha: float = 0.5, points: Sequence[tuple[int, int, int]] = ()) -> Image.Image:
    """Image with a translucent coloured mask, outline and click points (label 1 green, 0 red)."""
    arr = np.asarray(image.convert("RGB")).copy()
    m = to_binary(mask).astype(bool) if mask is not None else np.zeros(arr.shape[:2], bool)
    arr[m] = (arr[m] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    if m.any():
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(arr, cnts, -1, color, 2)
    r = max(4, min(arr.shape[:2]) // 120)
    for x, y, lab in points:
        c = (40, 220, 40) if lab == 1 else (230, 40, 40)
        cv2.circle(arr, (int(x), int(y)), r + 2, (255, 255, 255), -1)
        cv2.circle(arr, (int(x), int(y)), r, c, -1)
    return Image.fromarray(arr)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1) exclusive, or None if empty."""
    ys, xs = np.nonzero(to_binary(mask))
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def editor_to_mask(value: dict[str, Any] | None, size: tuple[int, int]) -> np.ndarray | None:
    """Gradio ImageEditor value -> binary mask (any painted layer pixel = masked)."""
    if not value:
        return None
    layers = value.get("layers") or []
    w, h = size
    out = np.zeros((h, w), np.uint8)
    for layer in layers:
        if layer is None:
            continue
        arr = np.asarray(layer)
        if arr.ndim == 3 and arr.shape[2] == 4:
            m = arr[..., 3] > 0
        else:
            m = to_binary(arr).astype(bool)
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        out |= m.astype(np.uint8)
    return out


def mask_to_layer(mask: np.ndarray, color: tuple[int, int, int] = (255, 40, 40), alpha: int = 150) -> np.ndarray:
    """Binary mask -> RGBA layer for the ImageEditor brush touch-up."""
    m = to_binary(mask).astype(bool)
    layer = np.zeros((*m.shape, 4), np.uint8)
    layer[m] = (*color, alpha)
    return layer


# ---- SAM 2.1 ---------------------------------------------------------------------------

@dataclass
class MaskCandidate:
    mask: np.ndarray   # uint8 {0,1}
    score: float


def sam_masks(image: Image.Image, points: Sequence[tuple[int, int, int]],
              box: tuple[int, int, int, int] | None = None) -> list[MaskCandidate]:
    """Run SAM 2.1 with click points [(x, y, label)] (label 1 = on object, 0 = background) and/or a box
    (x0, y0, x1, y1). Returns candidates sorted by predicted IoU (best first)."""
    if not points and box is None:
        return []
    if get_config()["backend"] == "stub":
        if not points and box is not None:
            points = [((box[0] + box[2]) // 2, (box[1] + box[3]) // 2, 1)]
        return _stub_masks(image, points)
    from core.model_manager import get_manager

    model, proc = get_manager().get("sam")
    img = image.convert("RGB")
    kw: dict[str, Any] = {}
    if points:
        kw["input_points"] = [[[[int(x), int(y)] for x, y, _ in points]]]
        kw["input_labels"] = [[[int(l) for _, _, l in points]]]
    if box is not None:
        kw["input_boxes"] = [[[float(v) for v in box]]]
    inputs = proc(images=img, return_tensors="pt", **kw).to(model.device)
    with torch.no_grad():
        out = model(**inputs, multimask_output=True)
    masks = proc.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])[0]  # [objects, K, H, W]
    scores = out.iou_scores.cpu().float().reshape(-1).tolist()
    ms = masks.reshape(-1, *masks.shape[-2:]).numpy()
    cands = [MaskCandidate((ms[i] > 0).astype(np.uint8), float(scores[i])) for i in range(len(ms))]
    return sorted(cands, key=lambda c: -c.score)


def _stub_masks(image: Image.Image, points: Sequence[tuple[int, int, int]]) -> list[MaskCandidate]:
    """Colour flood-fill stand-in for SAM (tests / UI development without model downloads)."""
    arr = np.asarray(image.convert("RGB")).copy()
    h, w = arr.shape[:2]
    cands = []
    for tol in (12, 24, 40):
        acc = np.zeros((h, w), np.uint8)
        for x, y, lab in points:
            ff = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(arr.copy(), ff, (int(x), int(y)), (0, 0, 0), (tol,) * 3, (tol,) * 3,
                          cv2.FLOODFILL_MASK_ONLY | (255 << 8) | 8)
            m = ff[1:-1, 1:-1] > 0
            acc = (acc | m) if lab == 1 else (acc & ~m)
        cands.append(MaskCandidate(acc.astype(np.uint8), 1.0 - tol / 100))
    return cands
