"""Phase 8: metrics and the benchmark runner (stub backend, synthetic scenes, LPIPS stubbed)."""
import math

import numpy as np
import pytest
from PIL import Image

from benchmarks import metrics as M


def _img(seed, h=96, w=128):
    return (np.random.RandomState(seed).rand(h, w, 3) * 255).astype(np.uint8)


def test_psnr_ssim_seam_basics():
    a, b = _img(0), _img(1)
    m = np.zeros(a.shape[:2], np.uint8)
    m[20:60, 30:90] = 1
    assert M.psnr(a, a) == math.inf and M.psnr(a, b) < 15
    assert M.ssim(a, a) == pytest.approx(1.0) and M.ssim(a, b) < 0.2
    assert M.seam(a, a, m) == 0.0 and M.seam(b, a, m) > 0
    c = a.copy()
    c[m > 0] = b[m > 0]
    assert M.psnr(c, a, 1 - m) == math.inf and M.psnr(c, a, m) < 15


def test_bootstrap_ci_drops_inf():
    mean, lo, hi = M.bootstrap_ci([1.0, 2.0, 3.0, float("inf")])
    assert mean == pytest.approx(2.0) and lo <= mean <= hi
