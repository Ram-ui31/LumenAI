"""Motion-deblur ("I captured this while running, make it correct").

Physics: a motion-blurred photo is the sharp photo convolved with a blur kernel k (the path the camera moved during
the exposure):   B = k * S (+ noise).
  estimate_kernel   classic cepstrum trick for straight-line motion: blur of length L along angle theta puts periodic
                    zeros in |F(B)|, which show up as a negative peak at distance L along theta in the cepstrum
                    c = F^-1( log |F(B)| ).
  wiener            S ~ F^-1( conj(K) F(B) / (|K|^2 + 1/SNR) )   (no AI)
  qwen              Qwen-Image 2.1 instruction edit (the user's own words, or a precise instruction)
  consistent        OUR HYBRID - "Qwen proposes, physics checks": start from Qwen's sharp guess and run a few
                    Richardson-Lucy iterations,  S <- S * ( k^T * ( B / (k * S) ) ),  which pull the guess towards
                    agreeing with what the camera recorded (re-blurring it must give back B) while keeping Qwen's
                    detail where the data allow. Invented detail that contradicts the blurred photo is pushed out.
"""
from __future__ import annotations

import cv2
import numpy as np


def line_kernel(length: int, angle_deg: float, size: int | None = None) -> np.ndarray:
    size = size or (length + 2 + (length + 2) % 2 + 1)
    k = np.zeros((size, size), np.float32)
    c = size // 2
    dx, dy = np.cos(np.radians(angle_deg)) * (length - 1) / 2, np.sin(np.radians(angle_deg)) * (length - 1) / 2
    cv2.line(k, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1.0, 1, cv2.LINE_AA)
    return k / k.sum()


def shake_kernel(length: int = 21, seed: int = 0, size: int = 41) -> np.ndarray:
    """Hand-shake: a smooth random walk (more realistic than a straight line)."""
    rng = np.random.default_rng(seed)
    pts = np.cumsum(rng.normal(0, 1, (length * 4, 2)), 0)
    pts -= pts.mean(0)
    pts *= (length / 2) / max(np.abs(pts).max(), 1e-6)
    k = np.zeros((size, size), np.float32)
    c = size // 2
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        cv2.line(k, (int(c + x0), int(c + y0)), (int(c + x1), int(c + y1)), 1.0, 1, cv2.LINE_AA)
    return k / k.sum()


def blur(img: np.ndarray, k: np.ndarray, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    out = cv2.filter2D(img.astype(np.float32), -1, k, borderType=cv2.BORDER_REFLECT)
    if noise > 0:
        out += np.random.default_rng(seed).normal(0, noise, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def estimate_kernel(B: np.ndarray, min_len: int = 5, max_len: int = 60) -> tuple[np.ndarray, dict]:
    """Straight-line motion kernel from the cepstrum of the blurred photo."""
    g = cv2.cvtColor(B, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255 if B.ndim == 3 else B.astype(np.float32) / 255
    h, w = g.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    F = np.fft.fft2((g - g.mean()) * win)
    cep = np.real(np.fft.ifft2(np.log(np.abs(F) + 1e-3)))
    cep = np.fft.fftshift(cep)
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - cy, xx - cx)
    ring = (r >= min_len) & (r <= max_len)
    c2 = np.where(ring, cep, np.inf)
    iy, ix = np.unravel_index(np.argmin(c2), c2.shape)
    L = float(np.hypot(iy - cy, ix - cx))
    ang = float(np.degrees(np.arctan2(iy - cy, ix - cx)))
    return line_kernel(max(3, int(round(L))), ang), {"length": round(L, 1), "angle": round(ang % 180, 1)}


def _psf2otf(k: np.ndarray, shape) -> np.ndarray:
    pad = np.zeros(shape, np.float32)
    kh, kw = k.shape
    pad[:kh, :kw] = k
    pad = np.roll(pad, (-(kh // 2), -(kw // 2)), (0, 1))
    return np.fft.fft2(pad)


def wiener(B: np.ndarray, k: np.ndarray, nsr: float = 0.01) -> np.ndarray:
    pad = max(k.shape)
    Bp = cv2.copyMakeBorder(B, pad, pad, pad, pad, cv2.BORDER_REFLECT).astype(np.float32)
    K = _psf2otf(k, Bp.shape[:2])
    Wf = np.conj(K) / (np.abs(K) ** 2 + nsr)
    out = np.stack([np.real(np.fft.ifft2(np.fft.fft2(Bp[..., c]) * Wf)) for c in range(Bp.shape[2])], -1)
    return np.clip(out[pad:-pad, pad:-pad], 0, 255).astype(np.uint8)


def richardson_lucy(B: np.ndarray, k: np.ndarray, init: np.ndarray | None = None, iters: int = 15) -> np.ndarray:
    """Data-consistency refinement: multiplicative RL updates from ``init`` (Qwen's guess) towards k*S = B."""
    Bf = B.astype(np.float32) + 1.0
    S = (init if init is not None else B).astype(np.float32) + 1.0
    kf = k[::-1, ::-1]
    for _ in range(iters):
        est = cv2.filter2D(S, -1, k, borderType=cv2.BORDER_REFLECT)
        ratio = Bf / np.maximum(est, 1e-3)
        S = S * cv2.filter2D(ratio, -1, kf, borderType=cv2.BORDER_REFLECT)
        S = np.clip(S, 1.0, 256.0)
    return np.clip(S - 1.0, 0, 255).astype(np.uint8)


USER_PROMPT = "I captured this image while running, make it correct."
SPEC_PROMPT = ("Remove the motion blur from this photo. Make it sharp and in focus, as if taken with a fast shutter "
               "speed. Keep every object, person, colour, text and position exactly the same.")


def qwen_deblur(pipe, Q, B: np.ndarray, prompt: str, seed: int = 7, steps: int = 40, resolution: int = 1024):
    from PIL import Image

    r = Q.edit(pipe, Image.fromarray(B), prompt, seed=seed, steps=steps, resolution=resolution)
    return np.asarray(r.image.convert("RGB").resize((B.shape[1], B.shape[0]), Image.LANCZOS)), {"seconds": r.seconds,
                                                                                             "peak_gb": r.peak_gb}


def sharpness(img: np.ndarray) -> float:
    """Variance of the Laplacian (higher = sharper)."""
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())
