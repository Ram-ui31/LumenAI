#!/usr/bin/env python
"""v5 object -> 3-D: Hunyuan3D-2 (mini shape + paint texture) on product photos and people/objects from RORD.

For each input: RGBA cut-out (our mask) -> GLB -> 8-view turntable. Reports time and peak memory for shape and texture,
face count, and a Qwen3-VL-free sanity number: silhouette IoU between the front turntable view and the input cut-out
(does the model's front view match the photo's outline?).
Runs in its own Modal image (CUDA build tools for the texture rasterizer).
  python -m benchmarks.run_v5_object3d --n_obj 8 --n_scene 4
Outputs: outputs/eval/v5_object3d/{summary.md, results.json, glb/<name>.glb, turntable/<name>.jpg}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from core.config import project_path  # noqa: E402


def inputs(n_obj: int, n_scene: int):
    items = []
    rec = project_path("outputs") / "eval" / "recolour" / "inputs"
    for p in sorted(rec.glob("P*_obj.png"))[:n_obj]:
        name = p.name[:-8]
        items.append((name, Image.open(rec / f"{name}.png").convert("RGB"), (np.asarray(Image.open(p)) > 0).astype(np.uint8)))
    try:
        from benchmarks.data import load

        for s in [s for s in load(50, 768) if s.mask.sum() > 15000][:n_scene]:
            items.append(("rord_" + s.name, s.image.convert("RGB"), s.mask))
    except Exception as e:  # noqa: BLE001
        print("RORD inputs skipped:", e, flush=True)
    return items


def completed_inputs():
    """v7 object completion results (RGBA, hidden parts filled in by Qwen) -> (name, RGB on white, mask)."""
    items = []
    for p in sorted((project_path("outputs") / "eval" / "v7_amodal").glob("rord_*.png")):
        rgba = np.asarray(Image.open(p).convert("RGBA"))
        a = rgba[..., 3:4].astype(np.float32) / 255
        rgb = (rgba[..., :3] * a + 255 * (1 - a)).round().astype(np.uint8)
        items.append((p.stem + "_completed", Image.fromarray(rgb), (rgba[..., 3] > 127).astype(np.uint8)))
    return items


def silhouette_iou(view: Image.Image, cut: Image.Image) -> float:
    v = (np.asarray(view.convert("RGB")) != 245).any(-1)
    a = np.asarray(cut)[..., 3] > 0

    def box(m):
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            return np.zeros((128, 128), bool)
        c = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        return np.asarray(Image.fromarray(c.astype(np.uint8) * 255).resize((128, 128))) > 127
    bv, ba = box(v), box(a)
    return float((bv & ba).sum() / max((bv | ba).sum(), 1))


def main(argv=None) -> int:
    import torch

    from editing import object3d as O3

    ap = argparse.ArgumentParser()
    ap.add_argument("--n_obj", type=int, default=8)
    ap.add_argument("--n_scene", type=int, default=4)
    ap.add_argument("--no_texture", action="store_true")
    ap.add_argument("--completed", action="store_true", help="v8: the v7 completed cut-outs -> outputs/eval/v8_object3d")
    a = ap.parse_args(argv)
    out = project_path("outputs") / "eval" / ("v8_object3d" if a.completed else "v5_object3d")
    for d in ("glb", "turntable", "cutout"):
        (out / d).mkdir(parents=True, exist_ok=True)
    rp = out / "results.json"
    res = json.loads(rp.read_text()) if rp.exists() else {}
    items = completed_inputs() if a.completed else inputs(a.n_obj, a.n_scene)
    for name, img, mask in items:
        if name in res and "error" not in res[name]:
            continue
        t0 = time.perf_counter()
        try:
            cut = O3.cutout(img, mask)
            cut.save(out / "cutout" / f"{name}.png")
            mesh, info = O3.make_mesh(cut, texture=not a.no_texture)
            mesh.export(out / "glb" / f"{name}.glb")
            views = O3.turntable(mesh, views=8, size=256)
            strip = Image.new("RGB", (256 * 9, 256), (245, 245, 245))
            strip.paste(cut.convert("RGBA").resize((256, 256)), (0, 0), cut.convert("RGBA").resize((256, 256)))
            for i, v in enumerate(views):
                strip.paste(v, (256 * (i + 1), 0))
            ImageDraw.Draw(strip).text((4, 4), name, fill="black")
            strip.save(out / "turntable" / f"{name}.jpg", quality=88)
            res[name] = {**info, "total_s": round(time.perf_counter() - t0, 1),
                         "front_iou": round(max(silhouette_iou(views[0], cut), silhouette_iou(views[4], cut)), 3)}
        except Exception as e:  # noqa: BLE001
            import traceback

            msg = traceback.format_exc()
            print(msg, flush=True)
            with open(out / "errors.txt", "a") as f:
                f.write(f"=== {name}\n{msg}\n")
            res[name] = {"error": repr(e)[:300]}
            torch.cuda.empty_cache()
        rp.write_text(json.dumps(res, indent=1))
        print(name, res[name], flush=True)
    ok = {k: v for k, v in res.items() if "error" not in v}
    L = ["# v5 object -> 3-D (Hunyuan3D-2mini shape + Hunyuan3D-2 paint)", "",
         f"{len(ok)}/{len(res)} objects converted.", "",
         "| object | shape s | shape peak GB | texture s | texture peak GB | faces (raw -> textured) | front silhouette IoU |", "|---|---|---|---|---|---|---|"]
    for k, v in ok.items():
        L.append(f"| {k} | {v.get('shape_s', '–')} | {v.get('shape_peak_gb', '–')} | {v.get('texture_s', '–')} | "
                 f"{v.get('texture_peak_gb', '–')} | {v.get('faces', '–')} -> {v.get('faces_textured', '–')} | {v.get('front_iou', '–')} |")
    for k, v in res.items():
        if "error" in v:
            L.append(f"- FAILED {k}: {v['error'][:160]}")
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
