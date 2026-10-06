"""End-to-end with the stub backend, plus Innovation A through a tiny real FluxFillPipeline."""
import json

import numpy as np
import pytest
import torch
from PIL import Image

from core.recipe import load_recipe


def test_generate_regenerate_pixel_identical_and_variation():
    from editing.replay import rerun
    from generation.generate import generate, tweak, variation

    a = generate("a red fox in snow, golden hour", 42, width=256, height=256)
    assert a.path.exists() and a.path.with_suffix(".json").exists()
    b = rerun(a.path.with_suffix(".json"))
    assert a.recipe.extra["output_sha256"] == b.recipe.extra["output_sha256"]
    v = variation(a.path.with_suffix(".json"))
    assert v.recipe.seed != 42 and v.recipe.parent_recipe.endswith(".json")
    assert v.recipe.extra["output_sha256"] != a.recipe.extra["output_sha256"]
    t = tweak(a.path.with_suffix(".json"), prompt="a red fox in snow, blue hour")
    assert t.recipe.seed == 42 and t.recipe.prompt.endswith("blue hour")


@pytest.mark.parametrize("mode", ["lama", "flux", "lama+flux"])
def test_remove_modes_keep_outside_pixels(mode):
    from editing.masking import dilate
    from editing.remove import remove_object
    from editing.replay import rerun

    img = Image.fromarray((np.random.RandomState(0).rand(300, 400, 3) * 255).astype(np.uint8))
    m = np.zeros((300, 400), np.uint8)
    m[100:180, 150:260] = 1
    r = remove_object(img, m, mode=mode, seed=1)
    region = r.region
    O, out = np.asarray(img), np.asarray(r.image)
    assert np.array_equal(out[region == 0], O[region == 0])
    rec = load_recipe(r.path)
    assert rec.task == "remove" and rec.edit["mode"] == mode and rec.edit["source_image_sha256"]
    again = rerun(r.path.with_suffix(".json"))
    assert again.recipe.extra["output_sha256"] == r.recipe.extra["output_sha256"]


def test_outpaint_single_and_iterative_chain():
    from editing.outpaint import outpaint

    img = Image.fromarray((np.random.RandomState(2).rand(256, 256, 3) * 255).astype(np.uint8))
    r = outpaint(img, right=64, top=64, seed=3)
    assert r.image.size == (320, 320) and len(r.passes) == 1
    # original pixels untouched outside the overlap band
    out, O = np.asarray(r.image), np.asarray(img)
    ov = r.recipe.edit["outpaint"]["overlap_px"]
    assert np.array_equal(out[64 + ov:, :256 - ov], O[ov:, :256 - ov])

    r2 = outpaint(img, right=64, top=64, seed=3, iterative=True)
    assert r2.image.size == (320, 320) and len(r2.passes) == 2
    p1, p2 = r2.passes
    assert load_recipe(p2.path).parent_recipe == str(p1.path.with_suffix(".json"))


# ---- Innovation A through the real diffusers FluxFillPipeline (tiny random weights) ----------

@pytest.fixture(scope="module")
def tiny():
    pytest.importorskip("diffusers")
    from tests.tiny_flux import tiny_embeds, tiny_fill_pipeline

    return tiny_fill_pipeline(), *tiny_embeds()


def _case():
    img = Image.fromarray((np.random.RandomState(0).rand(64, 64, 3) * 255).astype(np.uint8))
    m = np.zeros((64, 64), np.uint8)
    m[16:40, 16:40] = 1
    return img, m


@pytest.mark.parametrize("strength", [1.0, 0.6])
def test_latent_preserve_exact_outside(tiny, strength):
    from editing.latent_preserve import run_fill

    pipe, pe, pooled = tiny
    img, m = _case()
    lat, st = run_fill(pipe, img, m, pe, pooled, seed=3, steps=6, guidance_scale=30, strength=strength,
                       latent_preserve=True, stop_frac=0.0, max_sequence_length=8, output_type="latent")
    assert ((lat - st.z0).abs() * (1 - st.m)).max().item() == 0.0
    lat_off, _ = run_fill(pipe, img, m, pe, pooled, seed=3, steps=6, guidance_scale=30, strength=strength,
                          latent_preserve=False, max_sequence_length=8, output_type="latent")
    assert ((lat_off - st.z0).abs() * (1 - st.m)).max().item() > 0.0


def test_latent_preserve_deterministic_and_mask_packing(tiny):
    from diffusers import FluxFillPipeline

    from editing.latent_preserve import mask_to_latent, pack, run_fill, unpack

    pipe, pe, pooled = tiny
    img, m = _case()
    a, _ = run_fill(pipe, img, m, pe, pooled, seed=5, steps=4, guidance_scale=30, max_sequence_length=8, output_type="latent")
    b, _ = run_fill(pipe, img, m, pe, pooled, seed=5, steps=4, guidance_scale=30, max_sequence_length=8, output_type="latent")
    assert torch.equal(a, b)
    x = torch.randn(1, 4, 8, 8)
    assert torch.equal(pack(x), FluxFillPipeline._pack_latents(x, 1, 4, 8, 8))
    assert torch.equal(unpack(pack(x), 8, 8), x)
    lm = unpack(mask_to_latent(m, 8, 8, 4), 8, 8)[0, 0]
    assert lm[2:5, 2:5].eq(1).all() and lm[0, 0] == 0 and lm[6, 6] == 0   # 16:40 px -> cells 2..4
