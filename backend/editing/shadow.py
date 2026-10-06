"""Shadow finding from a generative model's own edit (the "let Qwen show us the shadow" trick).

Qwen, told to remove a person *including their shadow*, removes the shadow in its redrawn copy of the photo -
but our compositing only keeps pixels inside the object outline, so a shadow outside it comes back.
Here: (1) undo Qwen's global brightness/colour drift with a per-channel linear fit on pixels far from the object,
(2) shadow = pixels near the object that became clearly lighter, (3) keep only parts touching the outline.
Prototype verified on RORD I-210824 (wall shadow fully covered).
"""
from __future__ import annotations

import cv2
import numpy as np

from editing.masking import dilate


def _lightness(x: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    return cv2.GaussianBlur(lab[..., 0].astype(np.float32), (0, 0), 2)


def colour_match(raw: np.ndarray, original: np.ndarray, keep_out: np.ndarray, min_px: int = 1000) -> np.ndarray:
    """Per-channel a*raw+b fitted on pixels where keep_out == 0 (far from the edit). If there are too few such pixels
    (the object fills the frame), no correction is applied."""
    far = keep_out == 0
    if far.sum() < min_px:
        return raw.astype(np.float32)
    out = np.empty(raw.shape, np.float32)
    for c in range(3):
        a, b = np.polyfit(raw[..., c][far].astype(np.float64), original[..., c][far].astype(np.float64), 1)
        out[..., c] = a * raw[..., c] + b
    return np.clip(out, 0, 255)


def shadow_from_edit(original: np.ndarray, raw_edit: np.ndarray, object_mask: np.ndarray, hole: np.ndarray,
                     lift_threshold: float = 14.0, reach: float = 0.6, fit_area: np.ndarray | None = None) -> np.ndarray:
    """uint8 {0,1} shadow mask: near the object, outside ``hole``, made lighter by the edit, connected to the hole.
    ``fit_area`` (optional): where the edit may have drifted (e.g. the crop window Qwen redrew); the brightness
    correction is then fitted only on pixels inside it and away from the object."""
    m = (object_mask > 0).astype(np.uint8)
    h = int(m.any(1).sum()) or 1
    zone = dilate(m, max(8, int(reach * h))) > 0
    exclude = zone if fit_area is None else (zone | (fit_area == 0))
    Qc = colour_match(raw_edit, original, exclude.astype(np.uint8))
    lift = _lightness(Qc) - _lightness(original)
    cand = ((lift > lift_threshold) & zone & (hole == 0)).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    _, lab = cv2.connectedComponents(np.maximum(cand, dilate(hole, 6)))
    keep = np.isin(lab, np.unique(lab[hole > 0])) & (cand > 0)
    return cv2.morphologyEx(keep.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
