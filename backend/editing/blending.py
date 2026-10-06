"""Innovation B: frequency-aware (Laplacian-pyramid) compositing.

Inputs
  original  O  - source pixels (valid outside ``hole``)
  generated G  - model output, valid everywhere
  hole         - pixels whose original content must NOT be used (the object / the new outpaint area)
  region       - hole dilated (removal) or hole + overlap band (outpaint); the only pixels allowed to change

The blend weight is 1 deep inside ``region`` and ramps to 0 at its boundary. The ramp width depends on
the pyramid level: narrow for fine levels (edges and texture stay crisp), wide for coarse levels
(colour and lighting drift is spread smoothly). The "original" side of the blend is
``Y = where(hole, G, O)``, so object/placeholder pixels never leak in even through wide low-frequency
ramps. Finally ``output == O`` exactly outside ``region`` (enforced and unit-tested).
"""
from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

BlendMode = Literal["none", "feather", "laplacian"]


def _f32(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32)


def inner_distance(region: np.ndarray) -> np.ndarray:
    """Distance (px) from each pixel inside ``region`` to the nearest pixel outside it; 0 outside."""
    r = (region > 0).astype(np.uint8)
    if r.all():  # no outside pixels -> treat image border as outside
        r = np.pad(r, 1)
        return cv2.distanceTransform(r, cv2.DIST_L2, 5)[1:-1, 1:-1]
    return cv2.distanceTransform(r, cv2.DIST_L2, 5)


def ramp(region: np.ndarray, band_px: float) -> np.ndarray:
    """Weight map: 0 at/outside the region boundary, rising linearly to 1 over ``band_px`` inside."""
    d = inner_distance(region)
    return np.clip(d / max(band_px, 1e-3), 0.0, 1.0).astype(np.float32)


def _gauss_pyr(img: np.ndarray, levels: int) -> list[np.ndarray]:
    pyr = [img]
    for _ in range(levels - 1):
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr


def _lap_pyr(img: np.ndarray, levels: int) -> list[np.ndarray]:
    g = _gauss_pyr(img, levels)
    lap = []
    for i in range(levels - 1):
        up = cv2.pyrUp(g[i + 1], dstsize=(g[i].shape[1], g[i].shape[0]))
        lap.append(g[i] - up)
    lap.append(g[-1])
    return lap


def _collapse(lap: list[np.ndarray]) -> np.ndarray:
    img = lap[-1]
    for lvl in reversed(lap[:-1]):
        img = cv2.pyrUp(img, dstsize=(lvl.shape[1], lvl.shape[0])) + lvl
    return img


def _level_bands(levels: int, high_px: float, low_px: float) -> list[float]:
    """Geometric interpolation of ramp widths from finest to coarsest level."""
    if levels == 1:
        return [high_px]
    return [float(high_px * (low_px / high_px) ** (k / (levels - 1))) for k in range(levels)]


def max_levels(h: int, w: int, requested: int) -> int:
    return max(1, min(requested, int(np.log2(max(1, min(h, w)))) - 3))


def composite(
    original: np.ndarray,
    generated: np.ndarray,
    hole: np.ndarray,
    region: np.ndarray,
    mode: BlendMode = "laplacian",
    levels: int = 6,
    band_high_px: float = 3.0,
    band_low_px: float = 48.0,
    feather_px: float = 12.0,
) -> np.ndarray:
    """Blend ``generated`` into ``original`` (both HxWx3 uint8). Returns uint8.

    Guarantee: pixels where ``region == 0`` are bit-identical to ``original``.
    """
    assert original.shape == generated.shape, (original.shape, generated.shape)
    hole_b = hole > 0
    region_b = (region > 0) | hole_b
    O, G = _f32(original), _f32(generated)
    Y = np.where(hole_b[..., None], G, O)

    if mode == "none":
        out = np.where(region_b[..., None], G, O)
    elif mode == "feather":
        w = ramp(region_b, feather_px)[..., None]
        out = w * G + (1 - w) * Y
    elif mode == "laplacian":
        h, wd = original.shape[:2]
        n = max_levels(h, wd, levels)
        bands = _level_bands(n, band_high_px, band_low_px)
        LG, LY = _lap_pyr(G, n), _lap_pyr(Y, n)
        blended = []
        for k in range(n):
            # weight computed at full resolution with this level's band, then reduced to level k
            wk = ramp(region_b, bands[k])
            for _ in range(k):
                wk = cv2.pyrDown(wk)
            wk = wk[..., None]
            blended.append(wk * LG[k] + (1 - wk) * LY[k])
        out = _collapse(blended)
    else:
        raise ValueError(f"unknown blend mode {mode!r}")

    out = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    # hard invariant: untouched outside the region
    return np.where(region_b[..., None], out, original).astype(np.uint8)


def composite_back(O: np.ndarray, G, hole: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Put a generated image ``G`` (PIL) back into the original ``O`` with the configured Laplacian blend."""
    from core.config import get_config
    from PIL import Image

    e = get_config()["editing"]
    G = np.asarray(G.convert("RGB").resize((O.shape[1], O.shape[0]), Image.LANCZOS))
    return composite(O, G, hole, region, mode="laplacian", levels=e["pyramid_levels"],
                     band_high_px=e["blend_band_high_px"], band_low_px=e["blend_band_low_px"])
