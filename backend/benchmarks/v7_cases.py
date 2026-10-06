"""v7 test cases, built deterministically from files already on the volume (light dependencies: PIL + json only), so
the prompt-enhancer app (its own image) and the Qwen app produce the identical list.

Only the cases where Qwen misread us are re-tested (user's words for deblur, relight 'dim lamp', pose with a reference
photo) plus plain text-to-image prompts - features already accepted are not re-run.
"""
from __future__ import annotations

import json
from pathlib import Path

T2I_PROMPTS = [
    "a corgi reading a newspaper in a cafe", "rainy street in Tokyo at night", "a cozy reading nook",
    "product photo of a red sneaker", "an old fisherman portrait", "a futuristic kitchen", "mountain lake at sunrise",
    "a birthday cake with candles shaped like the number 7", "a poster that says 'Summer Sale 50% off'",
    "a cat astronaut", "a bowl of ramen, top view", "a medieval castle in fog", "children playing football in a park",
    "a minimalist living room with a green sofa", "a glass of orange juice on a wooden table, morning light",
    "a bicycle leaning on a brick wall", "a robot gardener watering tomatoes", "a busy Indian street market",
    "a watercolor painting of a lighthouse", "two friends laughing on a rooftop at sunset",
]
RELIGHT_USER = {"dim": "make it look like evening, lit only by a lamp",
                "left": "make the light come from the left"}
DEBLUR_USER = "I captured this image while running, make it correct."
POSE_USER = "Make the person in <image1> stand in exactly the same pose as the person in <image2>. Keep the person from <image1>."


def _rord(root: Path) -> Path:
    """RORD frames: <volume>/cache/... on Modal, <project>/.cache/... locally."""
    for c in ("cache", ".cache"):
        if (root / c / "eval" / "rord50_768").exists():
            return root / c / "eval" / "rord50_768"
    return root / "cache" / "eval" / "rord50_768"


def relight_photos(root: Path) -> list[tuple[str, Path]]:
    """Same photos as v6 relight: 6 empty scenes, 3 people, 3 products."""
    names = sorted(p.name[:-len("_input.png")] for p in _rord(root).glob("*_input.png"))
    items = [("scene_" + n, _rord(root) / f"{n}_gt.png") for n in names[::5][:6]]
    items += [("person_" + n, _rord(root) / f"{n}_input.png") for n in names[2::7][:3]]
    rec = root / "outputs" / "eval" / "recolour" / "inputs"
    objs = [p for p in sorted(rec.glob("P0*.png")) if not p.name.endswith(("_obj.png", "_part.png"))][:3]
    items += [("obj_" + p.stem, p) for p in objs]
    return items


def edit_cases(root: Path) -> list[dict]:
    """JSONL rows for the edit prompt enhancer: id, prompt, input_images (absolute paths)."""
    out = []
    dres = root / "outputs" / "eval" / "v6_deblur" / "results.json"
    if dres.exists():
        for key in sorted(json.loads(dres.read_text())):
            img = root / "outputs" / "eval" / "v6_deblur" / "img" / f"{key}_blurred.png"
            if img.exists():
                out.append({"id": f"deblur|{key}", "prompt": DEBLUR_USER, "input_images": [str(img)], "task_type": "deblur"})
    for name, path in relight_photos(root):
        for k, txt in RELIGHT_USER.items():
            out.append({"id": f"relight|{name}@{k}", "prompt": txt, "input_images": [str(path)], "task_type": "relight"})
    pres = root / "outputs" / "eval" / "v6_pose" / "results.json"
    if pres.exists():
        for key in sorted(json.loads(pres.read_text())):
            s, t = key.split("__to__")
            out.append({"id": f"pose|{key}", "prompt": POSE_USER,
                        "input_images": [str(_rord(root) / f"{s}_input.png"), str(_rord(root) / f"{t}_input.png")],
                        "task_type": "pose"})
    return out


def t2i_cases() -> list[dict]:
    return [{"id": f"t2i|{i:02d}", "prompt": p} for i, p in enumerate(T2I_PROMPTS)]


def load_rewrites(root: Path) -> dict[str, dict]:
    """id -> enhancer output record (positive_prompt, wh_ratio, ratio_follow, parse_ok, thinking)."""
    d = {}
    for f in ("pe_edit.jsonl", "pe_t2i.jsonl"):
        p = root / "outputs" / "eval" / "v7_pe" / f
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    d[r["id"]] = r
    return d
