"""Change the light source.

  qwen      Qwen-Image 2.1 instruction edit ("Relight this photo with ... light from the left ...").
  transfer  OUR HYBRID - light is the smooth part of a photo, detail is the sharp part. Image = albedo x shading, and
            shading varies slowly, so in linear RGB per channel:
                gain_c = G( Qwen_c ) / G( photo_c )      (G = large Gaussian, ~3 % of the image diagonal)
                out_c  = photo_c x gain_c
            The new light (direction, colour, brightness) comes from Qwen; every edge, face, letter and texture comes
            from the original photo - Qwen cannot redraw them. A guided (edge-aware) smoothing keeps the gain from
            bleeding across strong object boundaries.
Measured without ground truth: the lighting effect the instruction asks for (left/right brightness balance,
warmth, overall level) and how much fine detail of the original survives.
"""
from __future__ import annotations

import cv2
import numpy as np

LIGHTS = {
    "left": "Relight this photo so that a strong light source shines from the left side: the left side is brightly lit "
            "and the right side falls into shadow. Keep every object, person, colour and position exactly the same.",
    "right": "Relight this photo so that a strong light source shines from the right side: the right side is brightly "
             "lit and the left side falls into shadow. Keep every object, person, colour and position exactly the same.",
    "warm": "Relight this photo with warm golden sunset light. Keep every object, person and position exactly the same.",
    "dim": "Relight this photo as a dim evening scene lit only by a single soft lamp. Keep every object, person and "
           "position exactly the same.",
}


def _lin(x):
    x = x.astype(np.float32) / 255
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _srgb(x):
    x = np.clip(x, 0, 1)
    return (np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055) * 255).round().astype(np.uint8)


def _guided(I: np.ndarray, p: np.ndarray, r: int, eps: float = 1e-3) -> np.ndarray:
    """He et al. guided filter (grey guide I, input p)."""
    box = lambda a: cv2.boxFilter(a, -1, (r, r))  # noqa: E731
    mI, mp = box(I), box(p)
    a = (box(I * p) - mI * mp) / (box(I * I) - mI * mI + eps)
    b = mp - a * mI
    return box(a) * I + box(b)


def transfer(photo: np.ndarray, qwen: np.ndarray, sigma_frac: float = 0.03, max_gain: float = 6.0) -> np.ndarray:
    """Take the illumination of ``qwen`` and the detail of ``photo`` (both uint8 RGB, same size)."""
    if qwen.shape != photo.shape:
        qwen = cv2.resize(qwen, (photo.shape[1], photo.shape[0]), interpolation=cv2.INTER_AREA)
    P, Qn = _lin(photo), _lin(qwen)
    s = sigma_frac * float(np.hypot(*photo.shape[:2]))
    guide = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    r = max(3, int(s) | 1)
    out = np.empty_like(P)
    for c in range(3):
        num = cv2.GaussianBlur(Qn[..., c], (0, 0), s) + 1e-3
        den = cv2.GaussianBlur(P[..., c], (0, 0), s) + 1e-3
        g = np.clip(num / den, 1 / max_gain, max_gain)
        g = np.clip(_guided(guide, g.astype(np.float32), r), 1 / max_gain, max_gain)
        out[..., c] = P[..., c] * g
    return _srgb(out)


def qwen_relight(pipe, Q, photo: np.ndarray, key: str, seed: int = 7, steps: int = 40, resolution: int = 1024):
    from PIL import Image

    r = Q.edit(pipe, Image.fromarray(photo), LIGHTS[key], seed=seed, steps=steps, resolution=resolution)
    out = np.asarray(r.image.convert("RGB").resize((photo.shape[1], photo.shape[0]), Image.LANCZOS))
    return out, {"seconds": r.seconds, "peak_gb": r.peak_gb}


# ----------------------------------------------------------------------------- measurements


def _lab(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2LAB).astype(np.float32)


def light_effect(before: np.ndarray, after: np.ndarray, key: str) -> float:
    """Signed effect in the requested direction (bigger = stronger, CIE-ish units).
    left/right: change in (L_left_third - L_right_third); warm: change in mean b*; dim: drop in mean L."""
    a, b = _lab(before), _lab(after)
    L0, L1 = a[..., 0] * 100 / 255, b[..., 0] * 100 / 255
    w = L0.shape[1] // 3
    if key in ("left", "right"):
        d0 = L0[:, :w].mean() - L0[:, -w:].mean()
        d1 = L1[:, :w].mean() - L1[:, -w:].mean()
        return float((d1 - d0) * (1 if key == "left" else -1))
    if key == "warm":
        return float((b[..., 2] - a[..., 2]).mean())
    if key == "dim":
        return float(L0.mean() - L1.mean())
    raise ValueError(key)


def detail_kept(before: np.ndarray, after: np.ndarray) -> float:
    """Correlation of local contrast-normalised fine detail (high-pass of log-luminance): 1 = every edge/texture kept,
    regardless of how the light changed."""
    def hp(x):
        y = np.log(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).astype(np.float32) + 4)
        d = y - cv2.GaussianBlur(y, (0, 0), 3)
        return d / (np.sqrt(cv2.GaussianBlur(d * d, (0, 0), 6)) + 1e-3)
    if after.shape != before.shape:
        after = cv2.resize(after, (before.shape[1], before.shape[0]), interpolation=cv2.INTER_AREA)
    a, b = hp(before).ravel(), hp(after).ravel()
    return float(np.corrcoef(a, b)[0, 1])
