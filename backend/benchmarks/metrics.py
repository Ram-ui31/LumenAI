"""Image metrics for the removal benchmark. All take uint8 HxWx3 arrays.

* psnr / ssim (own implementation, Gaussian-window SSIM of Wang et al. 2004, on luminance)
* lpips (AlexNet, ``lpips`` package) — perceptual distance, lower is better
* masked variants: inside the edit region vs ground truth, outside it vs the source
* clip_i: CLIP image-image cosine between the result's edit crop and the ground-truth crop
"""
from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np


def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    """PSNR in dB over ``mask`` (all pixels if None). Identical pixels → inf."""
    d = (a.astype(np.float64) - b.astype(np.float64)) ** 2
    if mask is not None:
        m = mask.astype(bool)
        if not m.any():
            return float("nan")
        d = d[m]
    mse = float(d.mean())
    return float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def _lum(x: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).astype(np.float64) if x.ndim == 3 else x.astype(np.float64)


def ssim_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x, y = _lum(a), _lum(b)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda z: cv2.GaussianBlur(z, (11, 11), 1.5)  # noqa: E731
    mx, my = blur(x), blur(y)
    sxx, syy, sxy = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
    return ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))


def ssim(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    s = ssim_map(a, b)
    return float(s[mask.astype(bool)].mean() if mask is not None else s.mean())


@lru_cache(maxsize=1)
def _lpips_net():
    import lpips  # noqa: PLC0415

    return lpips.LPIPS(net="alex", verbose=False).eval()


def lpips_dist(a: np.ndarray, b: np.ndarray) -> float:
    import torch

    t = lambda z: torch.from_numpy(z.astype(np.float32) / 127.5 - 1).permute(2, 0, 1)[None]  # noqa: E731
    with torch.no_grad():
        return float(_lpips_net()(t(a), t(b)).item())


def crop_box(mask: np.ndarray, pad: float = 0.25) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    px, py = int((x1 - x0) * pad) + 8, int((y1 - y0) * pad) + 8
    return max(0, x0 - px), max(0, y0 - py), min(w, x1 + px), min(h, y1 + py)


def lpips_crop(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """LPIPS on the padded bounding box of the region (a whole-image LPIPS dilutes small edits)."""
    x0, y0, x1, y1 = crop_box(mask)
    ca, cb = a[y0:y1, x0:x1], b[y0:y1, x0:x1]
    if min(ca.shape[:2]) < 32:   # AlexNet needs a minimum size
        ca = cv2.resize(ca, (64, 64))
        cb = cv2.resize(cb, (64, 64))
    return lpips_dist(ca, cb)


def seam(result: np.ndarray, gt: np.ndarray, region: np.ndarray, width: int = 3) -> float:
    """Mean absolute gradient-magnitude difference to the ground truth along the region boundary:
    a visible seam adds edges the real photo does not have. Lower is better."""
    r = region.astype(np.uint8)
    ring = cv2.dilate(r, np.ones((2 * width + 1,) * 2, np.uint8)) - cv2.erode(r, np.ones((2 * width + 1,) * 2, np.uint8))
    g = lambda z: np.hypot(cv2.Sobel(_lum(z), cv2.CV_64F, 1, 0), cv2.Sobel(_lum(z), cv2.CV_64F, 0, 1))  # noqa: E731
    return float(np.abs(g(result) - g(gt))[ring > 0].mean())


def clip_i(result: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """CLIP ViT-L/14 image-image cosine x100 between the edit crops of result and ground truth."""
    import torch
    from PIL import Image

    from core.config import get_config
    from core.model_manager import get_manager

    if get_config()["backend"] == "stub":   # tests: cheap stand-in (100 = identical crops)
        x0, y0, x1, y1 = crop_box(mask)
        return float(100 - np.abs(result[y0:y1, x0:x1].astype(float) - gt[y0:y1, x0:x1]).mean())
    model, proc, _, dev = get_manager().get("critic")
    x0, y0, x1, y1 = crop_box(mask)
    ims = [Image.fromarray(result[y0:y1, x0:x1]), Image.fromarray(gt[y0:y1, x0:x1])]
    with torch.no_grad():
        f = model.get_image_features(pixel_values=proc(images=ims, return_tensors="pt")["pixel_values"].to(dev, model.dtype))
        f = (f.pooler_output if hasattr(f, "pooler_output") else f).float()
        f = f / f.norm(dim=-1, keepdim=True)
    return float(100 * (f[0] @ f[1]).item())


def bootstrap_ci(values: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """(mean, low, high) 95 % bootstrap interval; inf/nan values are dropped."""
    v = np.array([x for x in values if np.isfinite(x)], dtype=np.float64)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rs = np.random.RandomState(seed)
    means = v[rs.randint(0, len(v), (n, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
