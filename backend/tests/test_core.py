import numpy as np
import torch
from PIL import Image

from core.device import get_device
from core.prompt_cache import PromptCache
from core.recipe import Recipe, load_recipe, save_with_recipe, sha256_image
from core.seed import make_generator, resolve_seed


def test_get_device_is_known():
    assert get_device() in {"cpu", "mps", "cuda"}


def test_recipe_roundtrip_json_and_png(tmp_path):
    r = Recipe(task="remove", prompt="p", seed=123, edit={"mask_dilate_px": 8, "outpaint": {"left": 0}},
               parent_recipe="x.json")
    r.extra["custom"] = 5
    assert Recipe.from_json(r.to_json()).to_dict() == r.to_dict()
    img = Image.fromarray(np.full((32, 48, 3), 100, np.uint8))
    png = save_with_recipe(img, r, tmp_path / "out")
    from_png, from_json = load_recipe(png), load_recipe(png.with_suffix(".json"))
    assert from_png.to_dict() == from_json.to_dict()
    assert from_png.seed == 123 and from_png.edit["mask_dilate_px"] == 8
    assert from_png.extra["output_sha256"] == sha256_image(Image.open(png))


def test_recipe_unknown_keys_preserved():
    d = Recipe(seed=1).to_dict()
    d["future_field"] = {"a": 1}
    assert Recipe.from_dict(d).extra["future_field"] == {"a": 1}


def test_seed_resolution_and_determinism():
    assert resolve_seed(42) == 42
    s = resolve_seed(None)
    assert 0 <= s < 2**32
    a = torch.randn(4, generator=make_generator(7))
    b = torch.randn(4, generator=make_generator(7))
    assert torch.equal(a, b)
    assert not torch.equal(a, torch.randn(4, generator=make_generator(8)))


def test_prompt_cache_hit(tmp_path):
    calls = []

    def enc(p, L):
        calls.append(p)
        return {"prompt_embeds": torch.ones(1, L, 4), "pooled_prompt_embeds": torch.zeros(1, 3)}

    c = PromptCache(tmp_path)
    e1, hit1 = c.get_or_encode("fox", "repo", 8, "bfloat16", enc)
    e2, hit2 = c.get_or_encode("fox", "repo", 8, "bfloat16", enc)
    _, hit3 = c.get_or_encode("fox", "repo", 16, "bfloat16", enc)
    assert (hit1, hit2, hit3) == (False, True, False)
    assert torch.equal(e1["prompt_embeds"], e2["prompt_embeds"]) and len(calls) == 2
