import numpy as np
import pytest
from PIL import Image

from editing.blending import composite
from editing.common import expand_box, work_size
from editing.masking import dilate, editor_to_mask, feather, mask_bbox, mask_to_layer, to_binary
from editing.outpaint import plan_passes, prepare_canvas, round_expansion


def _imgs(h=200, w=260, seed=0):
    rs = np.random.RandomState(seed)
    O = (rs.rand(h, w, 3) * 255).astype(np.uint8)
    G = (rs.rand(h, w, 3) * 255).astype(np.uint8)
    hole = np.zeros((h, w), np.uint8)
    hole[60:120, 80:170] = 1
    return O, G, hole


# ---- masks ---------------------------------------------------------------------------

def test_to_binary_variants():
    m = np.zeros((10, 10), np.uint8)
    m[2:4, 2:4] = 255
    assert to_binary(m).sum() == 4
    assert to_binary(m > 0).sum() == 4
    rgba = np.zeros((10, 10, 4), np.uint8)
    rgba[5, 5, 3] = 10
    assert to_binary(rgba).sum() == 1


def test_dilate_grows_and_contains():
    m = np.zeros((50, 50), np.uint8)
    m[20:30, 20:30] = 1
    d = dilate(m, 8)
    assert d.sum() > m.sum() and (d[m > 0] == 1).all()
    assert mask_bbox(d) == (12, 12, 38, 38)
    assert dilate(m, 0).sum() == m.sum()


def test_feather_range():
    m = np.zeros((40, 40), np.uint8)
    m[10:30, 10:30] = 1
    f = feather(m, 6)
    assert f.min() >= 0 and f.max() <= 1 and 0 < f[10, 20] < 1


def test_editor_layer_roundtrip():
    m = np.zeros((30, 40), np.uint8)
    m[5:10, 5:20] = 1
    val = {"background": np.zeros((30, 40, 3), np.uint8), "layers": [mask_to_layer(m)], "composite": None}
    assert np.array_equal(editor_to_mask(val, (40, 30)), m)


# ---- blending invariant (Innovation B) ---------------------------------------------------

@pytest.mark.parametrize("mode", ["none", "feather", "laplacian"])
def test_outside_region_is_bit_identical(mode):
    O, G, hole = _imgs()
    region = dilate(hole, 8)
    out = composite(O, G, hole, region, mode=mode)
    assert out.dtype == np.uint8 and out.shape == O.shape
    assert np.array_equal(out[region == 0], O[region == 0])


def test_laplacian_uses_generated_inside_hole():
    O, G, hole = _imgs()
    region = dilate(hole, 8)
    out = composite(O, G, hole, region, mode="laplacian", band_low_px=48)
    # deep inside the hole the result should be much closer to G than to O
    core = np.zeros_like(hole)
    core[80:100, 110:140] = 1
    dG = np.abs(out.astype(int) - G.astype(int))[core > 0].mean()
    dO = np.abs(out.astype(int) - O.astype(int))[core > 0].mean()
    assert dG < 5 < dO


def test_object_pixels_never_leak():
    """Hole pixels of O (the object) must not influence the result: change them, output is unchanged."""
    O, G, hole = _imgs()
    region = dilate(hole, 8)
    O2 = O.copy()
    O2[hole > 0] = 255 - O2[hole > 0]
    a = composite(O, G, hole, region, mode="laplacian")
    b = composite(O2, G, hole, region, mode="laplacian")
    assert np.array_equal(a, b)


def test_laplacian_smooth_seam_vs_none():
    """With a constant colour offset (VAE drift), laplacian leaves a smaller step at the region edge."""
    h, w = 128, 128
    O = np.full((h, w, 3), 100, np.uint8)
    G = np.full((h, w, 3), 130, np.uint8)
    hole = np.zeros((h, w), np.uint8)
    hole[40:88, 40:88] = 1
    region = dilate(hole, 8)

    def max_step(img):
        return np.abs(np.diff(img[64, :, 0].astype(int))).max()

    assert max_step(composite(O, G, hole, region, mode="laplacian")) < max_step(composite(O, G, hole, region, mode="none"))


# ---- windows / outpaint geometry -----------------------------------------------------------

def test_work_size_caps_and_multiples():
    for w, h in [(4000, 3000), (300, 200), (1280, 768), (1024, 1024)]:
        ww, wh = work_size(w, h, 1_200_000)
        assert ww % 16 == 0 and wh % 16 == 0 and ww * wh <= 1_200_000


def test_expand_box_clamps():
    assert expand_box((10, 10, 20, 20), 64, 100, 80, min_size=512) == (0, 0, 100, 80)
    x0, y0, x1, y1 = expand_box((900, 900, 950, 950), 64, 2000, 2000, min_size=512)
    assert x1 - x0 >= 512 and 0 <= x0 and x1 <= 2000


def test_round_expansion_multiple_of_16():
    amt = round_expansion(1000, 1024, 0, 250, 256, 0)
    assert (1000 + amt["left"] + amt["right"]) % 16 == 0 and amt["right"] >= 250
    assert (1024 + amt["top"] + amt["bottom"]) % 16 == 0


def test_prepare_canvas_keeps_original():
    img = (np.random.RandomState(1).rand(64, 80, 3) * 255).astype(np.uint8)
    amt = {"left": 16, "right": 32, "top": 0, "bottom": 16}
    canvas, hole = prepare_canvas(img, amt, "blur")
    assert canvas.shape == (80, 128, 3)
    assert np.array_equal(canvas[0:64, 16:96], img) and hole[0:64, 16:96].sum() == 0
    assert hole.sum() == 80 * 128 - 64 * 80


def test_plan_passes_iterative_for_large():
    amt = {"left": 0, "right": 256, "top": 256, "bottom": 0}
    assert len(plan_passes(1024, 1024, amt, 2_000_000, 256, None)) == 1
    plan = plan_passes(1024, 1024, amt, 1_200_000, 256, None)
    assert [p for p in plan if p["top"]] and [p for p in plan if p["right"]] and len(plan) == 2
    assert len(plan_passes(1024, 1024, {"left": 0, "right": 600, "top": 0, "bottom": 0}, 1_200_000, 256, None)) == 3
