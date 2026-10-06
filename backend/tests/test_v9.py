"""v9 benchmark plumbing on CPU: watermark maths, guard parsing, LoRA side mapping, mesh holes, Gemini judge parsing,
and that the report renders from partial results without crashing."""
import json

import numpy as np
import pytest
from PIL import Image

from benchmarks import run_v9 as V
from benchmarks import v9_cases as C


@pytest.fixture()
def v9dir(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "OUT", tmp_path / "v9")
    monkeypatch.setattr(V, "IN", tmp_path / "v9" / "inputs")
    (tmp_path / "v9" / "inputs").mkdir(parents=True)
    return tmp_path / "v9"


def photo(n=512, seed=0):
    rng = np.random.default_rng(seed)
    a = cv_smooth(rng.integers(0, 255, (n, n, 3)).astype(np.uint8))
    return a


def cv_smooth(a):
    import cv2

    return cv2.GaussianBlur(a, (0, 0), 3)


def test_guard_cases_balanced_and_unique():
    rows = C.guard_cases()
    assert len(rows) == 60 and len({r["id"] for r in rows}) == 60
    assert sum(r["label"] == "block" for r in rows) == 30


def test_parsers():
    assert V.parse_singguard("<think>..</think><answer>unsafe: Sexual Content Risk</answer>") == "block"
    assert V.parse_singguard("<answer>safe</answer>") == "allow"
    assert V.parse_allow_block(" BLOCK.") == "block" and V.parse_allow_block("allow") == "allow"
    assert V.parse_allow_block("hmm") == "unclear"


def test_watermark_roundtrip_and_attacks():
    img = photo()
    marked, decode = V.wm_embed_decode(img, "dwtDct")
    assert V.wm_bit_acc(decode(marked)) > 0.95
    att = V.wm_attacks(marked)
    assert set(att) >= {"none", "jpeg50", "resize50%", "crop10%"} and all(a.shape == img.shape for a in att.values())
    assert V.wm_bit_acc(None) == 0.0


def test_part_watermark_without_gpu(v9dir):
    (V.IN / "sr").mkdir()
    Image.fromarray(photo()).save(V.IN / "sr" / "s1_hr.png")
    V.jsave(V.IN / "manifest.json", {"sr": ["s1"]})
    V.part_watermark(None)
    d = json.loads((v9dir / "watermark" / "results.json").read_text())
    row = d["s1_hr"]["dwtDct"]
    assert row["none"] > 0.95 and row["psnr"] > 30 and "ai_edit" not in row


def test_angle_side_mapping_picks_lower_error():
    d = {"a#front-left": {"q2511_angles": {"v": 3, "fid@-45": {"geo_dE": 30}, "fid@+45": {"geo_dE": 10}}},
         "a#front-right": {"q2511_angles": {"v": 3, "fid@-45": {"geo_dE": 9}, "fid@+45": {"geo_dE": 31}}}}
    assert V.angle_side_mapping(d) == {-45.0: "front-right", 45.0: "front-left"}


def test_mesh_stats_counts_holes(tmp_path):
    import trimesh

    box = trimesh.creation.box()
    box.export(tmp_path / "ok.glb")
    holed = trimesh.Trimesh(box.vertices, box.faces[:-2])
    holed.export(tmp_path / "hole.glb")
    a, b = V.mesh_stats(tmp_path / "ok.glb"), V.mesh_stats(tmp_path / "hole.glb")
    assert a["watertight"] and a["boundary_edges"] == 0
    assert not b["watertight"] and b["boundary_edges"] > 0


def test_judge_maps_shuffled_letters_back(monkeypatch):
    J = V.Judge()
    J.gemini_ok = True
    seen = {}

    def fake(parts):
        labels = [p["text"] for p in parts if isinstance(p, dict) and "text" in p and p["text"].startswith("Candidate")]
        seen["labels"] = labels
        return {"A": {"follow": 9, "quality": "7"}, "B": {"follow": 2, "quality": 11}}
    monkeypatch.setattr(J, "gemini", fake)
    im = Image.new("RGB", (8, 8))
    r = J.score("intro", [im], {"ours": im, "other": im}, {"follow": "x", "quality": "y"}, "case1")
    assert r["_by"] == "gemini" and set(r) == {"ours", "other", "_by"}
    assert sorted([r["ours"]["follow"], r["other"]["follow"]]) == [2, 9] and 10 in (r["ours"]["quality"], r["other"]["quality"])


def test_report_renders_with_partial_results(v9dir):
    V.jsave(V.IN / "manifest.json", {"guard": C.guard_cases()[:3], "angles": [], "3d": [], "sr": [], "gedit": [],
                                     "t2i": [], "product": [], "pose": [], "removal": []})
    R = V.Results("guard")
    R.put("a00", "ours_qwen3vl", {"pred": "allow", "seconds": 0.4, "extra_gb": 0})
    V.jsave(v9dir / "judge.json", {"gedit": {"k": {"ours": {"follow": 8, "preserve": 6, "quality": 9}, "_by": "gemini"}}})
    V.part_report()
    md = (v9dir / "summary.md").read_text()
    assert "Guardrails" in md and "| ours_qwen3vl | 0/1 |" in md and "GEdit" in md
