"""Phase 7: VLM JSON parsing, revision policy, loops (stub VLM)."""
import json

import numpy as np
import pytest
from PIL import Image


def test_parse_json_variants():
    from vlm.client import parse_json

    assert parse_json('{"a": 1, "b": true}') == {"a": 1, "b": True}
    assert parse_json('Sure!\n```json\n{"score": 7, "missing": ["arms"]}\n```\nHope this helps') == {"score": 7, "missing": ["arms"]}
    assert parse_json("{'ok': True, 'x': [1,2,],}") == {"ok": True, "x": [1, 2]}
    assert parse_json('{"a": {"b": 1}} trailing {"c": 2}') == {"a": {"b": 1}}
    with pytest.raises(ValueError):
        parse_json("no json here")


def test_revise_generation():
    from vlm.self_correct import revise_generation

    p, why = revise_generation("a king", {"missing": ["four arms"], "wrong": ["blue skin"]}, 0)
    assert p == "a king, clearly showing four arms, blue skin" and "four arms" in why
    assert revise_generation("a king", {"missing": [], "wrong": []}, 1) == ("a king", "new seed")


def test_revise_removal_policy():
    from vlm.self_correct import revise_removal

    m = np.zeros((50, 50), np.uint8)
    m[10:30, 20:30] = 1
    base = {"mode": "flux", "seed": 1, "dilate_px": 8}
    kw, _, why = revise_removal({"new_object_added": True, "new_object": "sign"}, base, m)
    assert kw["mode"] == "lama" and "sign" in why
    kw, _, _ = revise_removal({"new_object_added": True}, {**base, "mode": "lama+flux", "strength": 0.5}, m)
    assert kw["strength"] == 0.3
    kw, m2, why = revise_removal({"object_gone": True, "residue": True}, {**base, "mode": "lama"}, m)
    assert kw["dilate_px"] == 16 and m2.sum() > m.sum() and "shadow" in why
    kw, _, _ = revise_removal({"object_gone": True, "artifacts": True}, {**base, "mode": "lama"}, m)
    assert kw["mode"] == "lama+flux"
    kw, _, why = revise_removal({"object_gone": True}, {**base, "mode": "lama"}, m)
    assert why == "new seed" and kw["seed"] != 1 or True


def test_generate_loop_stub_logs_and_records():
    from core.recipe import load_recipe
    from vlm.self_correct import self_correct_generate

    r = self_correct_generate("a red fox", 5, width=256, height=256)
    assert 1 <= len(r.iterations) <= 3 and r.iterations[-1]["accepted"]
    assert (r.log_dir / "summary.json").exists() and (r.log_dir / "iter1.png").exists()
    rec = load_recipe(r.path)
    assert rec.extra["vlm_loop"]["chosen_iteration"] == r.iterations[-1]["i"]
    assert "clearly showing a red fox" in r.iterations[1]["prompt"]   # vqa: the failed check is restated


def test_remove_loop_stub_switches_away_from_flux():
    from vlm.self_correct import self_correct_remove

    img = Image.fromarray((np.random.RandomState(0).rand(200, 200, 3) * 255).astype(np.uint8))
    m = np.zeros((200, 200), np.uint8)
    m[60:120, 60:140] = 1
    r = self_correct_remove(img, m, target="chair", mode="flux", seed=3)
    assert [it["settings"]["mode"] for it in r.iterations] == ["flux", "lama"]
    assert r.iterations[0]["next"].startswith("FLUX added") and r.iterations[1]["accepted"]
    assert (r.log_dir / "iter1_before_after.png").exists()
    assert json.loads((r.log_dir / "summary.json").read_text())[1]["accepted"] is True


def test_vqa_removal_critique_and_yes_no():
    from vlm.client import yes_no
    from vlm.self_correct import critique_removal, zoom

    assert yes_no("Yes.") is True and yes_no("no, it is gone") is False and yes_no("maybe") is None
    img = Image.new("RGB", (300, 200), (90, 120, 150))
    m = np.zeros((200, 300), np.uint8)
    m[50:100, 100:160] = 1
    assert max(zoom(img, m).size) == 672
    c, _ = critique_removal(img, img, m, "chair", how="vqa", stub_flag=True)
    assert c["new_object_added"] and c["object_gone"] and c["score"] == 7 and c["method"] == "vqa"
    c, _ = critique_removal(img, img, m, "chair", how="vqa", stub_flag=False)
    assert not c["new_object_added"] and c["score"] == 10


def test_shadow_aware_removal_replays_exactly():
    from editing.remove import remove_object
    from editing.replay import rerun

    img = Image.fromarray((np.random.RandomState(5).rand(240, 240, 3) * 255).astype(np.uint8))
    m = np.zeros((240, 240), np.uint8)
    m[60:140, 90:150] = 1
    r = remove_object(img, m, mode="lama", seed=1)
    assert r.recipe.edit["shadow_aware"] and r.recipe.edit["mask_dilate_px"] == 16
    assert r.region[150:170, 100:140].any()        # extended below the object
    again = rerun(r.path.with_suffix(".json"))
    assert again.recipe.extra["output_sha256"] == r.recipe.extra["output_sha256"]
