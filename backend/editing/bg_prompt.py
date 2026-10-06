"""Automatic background prompt for removal.

FLUX Fill tends to put *something* into a hole when the prompt is generic ("empty background"): on the
Phase 5 samples it swapped a chair for a board and a bicycle for a sign. Describing what should be
there instead ("sandy beach", "cobblestone pavement") steers it to continue the surface. We pick the
description with CLIP zero-shot on the surroundings of the mask (the object itself is hidden first).
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from core.config import get_config
from editing.common import expand_box
from editing.masking import dilate, mask_bbox


def surroundings(image: Image.Image, region: np.ndarray) -> Image.Image:
    """Crop around the region with the object hidden (fast OpenCV inpaint), so CLIP sees only context."""
    arr = np.asarray(image.convert("RGB"))
    hidden = cv2.inpaint(arr, dilate(region, 4), 7, cv2.INPAINT_TELEA)
    x0, y0, x1, y1 = mask_bbox(region)
    pad = max(48, int(0.75 * max(x1 - x0, y1 - y0)))
    bx0, by0, bx1, by1 = expand_box((x0, y0, x1, y1), pad, arr.shape[1], arr.shape[0], min_size=224)
    return Image.fromarray(hidden[by0:by1, bx0:bx1])


def describe_background(image: Image.Image, region: np.ndarray) -> tuple[str, list[tuple[str, float]]]:
    """(prompt, ranked labels). Prompt = template filled with the top-k background labels."""
    from critics.aesthetic import clip_rank

    rc = get_config()["remove"]
    labels = rc["backgrounds"]
    ranked = clip_rank(surroundings(image, region), [f"a photo of {b}" for b in labels])
    ranked = [(t.removeprefix("a photo of "), round(s, 2)) for t, s in ranked]
    top = [t for t, _ in ranked[: int(rc.get("auto_prompt_top_k", 2))]]
    return rc["prompt_template"].format(bg=", ".join(top)), ranked[:5]
