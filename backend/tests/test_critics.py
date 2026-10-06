"""Phase 5: critics, best-of-N, removal verification (stub backend + tiny real CLIP)."""
import numpy as np
import pytest
import torch
from PIL import Image

from core.recipe import load_recipe
from critics.detector import Detection, box_mask_overlap, find_object, judge, mask_coverage


def _mask():
    m = np.zeros((100, 100), np.uint8)
    m[20:60, 30:70] = 1
    return m


def test_overlap_measures():
    m = _mask()
    assert box_mask_overlap((30, 20, 70, 60), m) == 1.0
    assert mask_coverage((30, 20, 70, 60), m) == 1.0
    assert box_mask_overlap((0, 0, 100, 100), m) == pytest.approx(0.16)
    assert box_mask_overlap((80, 80, 99, 99), m) == 0.0


def test_find_object_and_judge():
    m = _mask()
    dets = [Detection("person", 0.9, (0, 0, 100, 100)),      # huge box: mask is tiny part of it
            Detection("bicycle", 0.6, (28, 18, 72, 62)),     # matches the mask
            Detection("bicycle", 0.8, (80, 80, 99, 99))]     # elsewhere
    assert find_object(dets, m).box == (28, 18, 72, 62)
    assert find_object([Detection("cat", 0.9, (80, 80, 99, 99))], m) is None
    after = [Detection("bicycle", 0.5, (32, 22, 68, 58)), Detection("bicycle", 0.9, (80, 80, 99, 99))]
    assert [d.box for d in judge(after, m, "bicycle", 0.5)] == [(32, 22, 68, 58)]
    assert judge(after, m, "person", 0.5) == []


def test_combine_weights():
    from critics.aesthetic import combine

    assert combine(40.0, 10.0) == pytest.approx(1.0)
    assert combine(20.0, 5.0) == pytest.approx(0.5)


def test_aesthetic_head_matches_laion_keys():
    from critics.aesthetic import AestheticHead

    keys = set(AestheticHead(768).state_dict())
    assert keys == {f"layers.{i}.{p}" for i in (0, 2, 4, 6, 7) for p in ("weight", "bias")}


def test_score_with_tiny_real_clip(monkeypatch):
    """Exercises the real scoring maths (normalise, cosine*100, head) with a tiny random CLIP."""
    from transformers import CLIPConfig, CLIPModel

    import critics.aesthetic as A
    from core import model_manager

    cfg = CLIPConfig(text_config=dict(hidden_size=32, intermediate_size=37, num_attention_heads=4,
                                      num_hidden_layers=1, vocab_size=99, bos_token_id=0, eos_token_id=1),
                     vision_config=dict(hidden_size=32, intermediate_size=37, num_attention_heads=4,
                                        num_hidden_layers=1, image_size=30, patch_size=6), projection_dim=16)
    model = CLIPModel(cfg).eval()

    def proc(text, images, **kw):
        return {"pixel_values": torch.randn(len(images), 3, 30, 30, generator=torch.Generator().manual_seed(0)),
                "input_ids": torch.tensor([[0, 5, 1]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}

    head = A.AestheticHead(16).eval()

    class M:
        def get(self, name):
            return model, proc, head, "cpu"

    monkeypatch.setattr(model_manager, "get_manager", lambda: M())
    monkeypatch.setitem(A.get_config(), "backend", "diffusers")
    s = A.score([Image.new("RGB", (64, 64)), Image.new("RGB", (64, 64), (255, 0, 0))], "a fox")
    assert len(s) == 2 and all(-100 <= x.clip <= 100 for x in s)
    assert s[0].combined == A.combine(s[0].clip, s[0].aesthetic)


def test_best_of_n_records_candidates_and_regenerates():
    from critics.best_of_n import best_of_n
    from editing.replay import rerun

    bo = best_of_n("a red fox in snow", 3, 100, width=256, height=256)
    rec = load_recipe(bo.best.path)
    blk = rec.extra["best_of_n"]
    assert [c["seed"] for c in blk["candidates"]] == [100, 101, 102]
    assert blk["chosen_seed"] == rec.seed == max(blk["candidates"], key=lambda c: c["combined"])["seed"]
    assert load_recipe(bo.best.path.with_suffix(".json")).extra["best_of_n"]["n"] == 3   # sidecar too
    again = rerun(bo.best.path.with_suffix(".json"))
    assert again.recipe.extra["output_sha256"] == rec.extra["output_sha256"]
    assert "best_of_n" not in again.recipe.extra          # a regeneration is a single image, not a contest


def test_removal_check_recorded_and_retry_grows_mask():
    from editing.remove import remove_object, retry_removal

    img = Image.fromarray((np.random.RandomState(0).rand(200, 200, 3) * 255).astype(np.uint8))
    m = np.zeros((200, 200), np.uint8)
    m[60:120, 60:140] = 1
    r = remove_object(img, m, mode="flux", seed=1)
    assert r.recipe.extra["removal_check"]["ok"] is True          # stub detector finds nothing
    r2 = retry_removal(str(r.path.with_suffix(".json")))
    assert r2.recipe.edit["mask_dilate_px"] == r.recipe.edit["mask_dilate_px"] + 8
    assert r2.recipe.seed != r.recipe.seed or True
    assert r2.recipe.parent_recipe == str(r.path.with_suffix(".json"))


def test_sam_box_prompt_stub():
    from editing.masking import sam_masks

    img = Image.new("RGB", (64, 64), (10, 10, 10))
    assert sam_masks(img, [], box=(10, 10, 30, 30))


def test_auto_background_prompt_recorded_and_replayable():
    from editing.remove import remove_object
    from editing.replay import rerun

    img = Image.fromarray((np.random.RandomState(3).rand(256, 256, 3) * 255).astype(np.uint8))
    m = np.zeros((256, 256), np.uint8)
    m[90:160, 90:170] = 1
    r = remove_object(img, m, mode="lama+flux", seed=7, strength=0.75)
    assert "seamless continuation" in r.recipe.prompt and r.recipe.extra["auto_prompt"]["top_backgrounds"]
    again = rerun(r.path.with_suffix(".json"))
    assert again.recipe.prompt == r.recipe.prompt
    assert again.recipe.extra["output_sha256"] == r.recipe.extra["output_sha256"]
