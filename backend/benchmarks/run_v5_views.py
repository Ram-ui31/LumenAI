#!/usr/bin/env python
"""v5 novel views: MoGe-2 depth -> re-projection -> four ways to fill what was hidden.

Inputs: 8 RORD photos (people in rooms/streets) + 8 product photos from the recolour benchmark (objects on plain
backgrounds). Views: camera orbited yaw -20 and +20 degrees around the subject.
Methods: warp (holes filled by OpenCV, reference), lama, keep (Qwen + latent preservation, ours), anyangle (Qwen +
AnyAngle LoRA), text (Qwen told to rotate the camera).
No ground truth exists for unseen angles, so we report:
  geo_dE   colour error (CIEDE2000) vs the re-projection on pixels the original camera saw - does the method keep the
           geometry? (lower = more faithful; ~0 for warp/lama/keep by construction, the interesting ones are the others)
  judge    Qwen3-VL verdict on (original, result): "new_view" / "same_view" / "broken"
  seconds, peak GB;  grids/<name>.jpg for visual review.
  python -m benchmarks.run_v5_views
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from benchmarks.run_qwen_check import SEED, free  # noqa: E402
from core.config import project_path  # noqa: E402

YAWS = [-20.0, 20.0]
METHODS = ["warp", "lama", "keep", "anyangle", "text"]
JUDGE_Q = ("Image 1 is a photo. Image 2 is supposed to show the SAME scene photographed from a camera that moved about "
           "{deg} degrees around the main subject. Answer with exactly one word: new_view (same scene and objects, "
           "clearly different camera angle, looks realistic), same_view (essentially the same angle as image 1), or "
           "broken (distorted, wrong objects, or unrealistic).")


def inputs(n_scene: int, n_obj: int):
    from benchmarks.data import load

    items = []
    for s in [s for s in load(50, 768) if s.mask.sum() > 8000][:n_scene]:
        items.append(("scene_" + s.name, s.image.convert("RGB"), s.mask))
    rec = project_path("outputs") / "eval" / "recolour" / "inputs"
    for p in sorted(rec.glob("P*_obj.png"))[:n_obj]:
        name = p.name[:-8]
        img = Image.open(rec / f"{name}.png").convert("RGB")
        items.append(("obj_" + name, img, (np.asarray(Image.open(p)) > 0).astype(np.uint8)))
    return items


def judge(Q, pipe, a: Image.Image, b: Image.Image, deg: float) -> str:
    t = Q.ask(pipe, [a, b], JUDGE_Q.format(deg=int(abs(deg))), max_new_tokens=6).strip().lower()
    for k in ("new_view", "same_view", "broken"):
        if k in t or k.replace("_", " ") in t:
            return k
    return "unclear:" + t[:20]


def main(argv=None) -> int:
    import torch

    from editing import qwen as Q
    from editing import views3d as V

    ap = argparse.ArgumentParser()
    ap.add_argument("--n_scene", type=int, default=8)
    ap.add_argument("--n_obj", type=int, default=8)
    a = ap.parse_args(argv)
    out = project_path("outputs") / "eval" / "v5_views"
    for d in ["grids"] + METHODS + ["render"]:
        (out / d).mkdir(parents=True, exist_ok=True)
    items = inputs(a.n_scene, a.n_obj)
    res_p = out / "results.json"
    res = json.loads(res_p.read_text()) if res_p.exists() else {}

    def log_err(label):
        import traceback

        msg = traceback.format_exc()
        print(msg, flush=True)
        with open(out / "errors.txt", "a") as f:
            f.write(f"=== {label}\n{msg}\n")

    # 1) geometry + renders (MoGe on the GPU, then freed)
    geos = {}
    t0 = time.perf_counter()
    for name, img, mask in items:
        try:
            torch.cuda.reset_peak_memory_stats()
            g = V.estimate_geometry(img)
            geos[name] = g
            for yaw in YAWS:
                key = f"{name}@{yaw:+.0f}"
                r, valid, _ = V.render_view(g, yaw=yaw, subject=mask)
                holes = V.hole_mask(valid)
                Image.fromarray(r).save(out / "render" / f"{key}.png")
                Image.fromarray(holes * 255).save(out / "render" / f"{key}_holes.png")
                Image.fromarray(valid.astype(np.uint8) * 255).save(out / "render" / f"{key}_valid.png")
                res.setdefault(key, {})["render"] = {"hole_share": float(holes.mean()),
                                                     "moge_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
        except Exception:  # noqa: BLE001
            log_err(f"geometry {name}")
    V.unload_moge()
    free()
    print(f"geometry for {len(geos)} images in {time.perf_counter() - t0:.0f}s", flush=True)

    pipe = Q.load_pipe("nf4")
    Q.enable_memory_savers(pipe)
    for name, img, mask in items:
        for yaw in YAWS:
            key = f"{name}@{yaw:+.0f}"
            if not (out / "render" / f"{key}.png").exists():
                continue
            r = np.asarray(Image.open(out / "render" / f"{key}.png").convert("RGB"))
            holes = (np.asarray(Image.open(out / "render" / f"{key}_holes.png")) > 0).astype(np.uint8)
            valid = np.asarray(Image.open(out / "render" / f"{key}_valid.png")) > 0
            todo = [m for m in ("warp", "lama", "keep", "text") if m not in res.get(key, {})]
            lama = None
            for m in todo:
                try:
                    t1 = time.perf_counter()
                    torch.cuda.reset_peak_memory_stats()
                    if m == "warp":
                        o, info = cv2.inpaint(r, holes, 5, cv2.INPAINT_TELEA), {}
                    elif m == "lama":
                        lama = V.fill_lama(r, holes)
                        o, info = lama, {}
                    elif m == "keep":
                        lama = lama if lama is not None else V.fill_lama(r, holes)
                        o, info = V.fill_keep(pipe, Q, img, lama, holes, seed=SEED)
                    else:
                        o, info = V.fill_text(pipe, Q, img, yaw, seed=SEED)
                    Image.fromarray(o).save(out / m / f"{key}.png")
                    res.setdefault(key, {})[m] = {"seconds": round(time.perf_counter() - t1, 2),
                                   "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                                   **V.warp_fidelity(o, r, valid),
                                   **{k: v for k, v in info.items() if isinstance(v, (int, float, str, bool))}}
                except Exception:  # noqa: BLE001
                    log_err(f"{m} {key}")
            res_p.write_text(json.dumps(res, indent=1))
        print(f"{name} done", flush=True)

    # AnyAngle LoRA last (it changes the transformer while active)
    try:
        how = V.load_anyangle(pipe)
        print("AnyAngle loaded:", how, flush=True)
        res["_anyangle_load"] = how
        for name, img, mask in items:
            for yaw in YAWS:
                key = f"{name}@{yaw:+.0f}"
                if key not in res or "anyangle" in res[key] or not (out / "lama" / f"{key}.png").exists():
                    continue
                try:
                    r = np.asarray(Image.open(out / "render" / f"{key}.png").convert("RGB"))
                    valid = np.asarray(Image.open(out / "render" / f"{key}_valid.png")) > 0
                    lama = np.asarray(Image.open(out / "lama" / f"{key}.png").convert("RGB"))
                    o, info = V.fill_anyangle(pipe, Q, img, lama, seed=SEED)
                    Image.fromarray(o).save(out / "anyangle" / f"{key}.png")
                    res[key]["anyangle"] = {**info, **V.warp_fidelity(o, r, valid)}
                except Exception:  # noqa: BLE001
                    log_err(f"anyangle {key}")
            res_p.write_text(json.dumps(res, indent=1))
    except Exception:  # noqa: BLE001
        log_err("anyangle load")
        res["_anyangle_load"] = "failed"

    # judge with Qwen3-VL (LoRA off)
    try:
        V.set_anyangle(pipe, False)
    except Exception:  # noqa: BLE001
        pass
    for name, img, mask in items:
        for yaw in YAWS:
            key = f"{name}@{yaw:+.0f}"
            for m in METHODS:
                p = out / m / f"{key}.png"
                if p.exists() and m in res.get(key, {}) and "judge" not in res[key][m]:
                    try:
                        res[key][m]["judge"] = judge(Q, pipe, img, Image.open(p).convert("RGB"), yaw)
                    except Exception:  # noqa: BLE001
                        log_err(f"judge {m} {key}")
        res_p.write_text(json.dumps(res, indent=1))
        grid(out, name, img)
    md = summarize(res)
    (out / "summary.md").write_text(md)
    print(md)
    return 0


def grid(out: Path, name: str, img: Image.Image):
    th = 200
    rows = []
    for yaw in YAWS:
        key = f"{name}@{yaw:+.0f}"
        row = [("original", img)] + [("render", out / "render" / f"{key}.png")] + [(m, out / m / f"{key}.png") for m in METHODS]
        tiles = []
        for lab, src in row:
            try:
                im = src if isinstance(src, Image.Image) else Image.open(src).convert("RGB")
            except Exception:  # noqa: BLE001
                im = Image.new("RGB", (th, th), (60, 60, 60))
            im = im.resize((int(im.width * th / im.height), th))
            ImageDraw.Draw(im).rectangle([0, 0, im.width, 13], fill="black")
            ImageDraw.Draw(im).text((3, 1), f"{lab} {yaw:+.0f}°" if lab != "original" else lab, fill="white")
            tiles.append(im)
        rows.append(tiles)
    W = max(sum(t.width for t in r) for r in rows)
    G = Image.new("RGB", (W, th * len(rows)), "white")
    for i, r in enumerate(rows):
        x = 0
        for t in r:
            G.paste(t, (x, i * th))
            x += t.width
    G.save(out / "grids" / f"{name}.jpg", quality=85)


def summarize(res: dict) -> str:
    keys = [k for k in res if not k.startswith("_")]
    L = ["# v5 novel views (MoGe-2 re-projection + hole filling)", "",
         f"{len(keys)} views ({len({k.split('@')[0] for k in keys})} photos x {len(YAWS)} angles). "
         f"AnyAngle LoRA: {res.get('_anyangle_load', 'not run')}", "",
         "| method | geo dE ↓ (on seen pixels) | judge: new view | same view | broken | s | peak GB |", "|---|---|---|---|---|---|---|"]
    for m in METHODS:
        rs = [res[k][m] for k in keys if m in res[k]]
        if not rs:
            continue
        de = [r["geo_dE"] for r in rs if r.get("geo_dE") is not None]
        j = [r.get("judge", "") for r in rs]
        sec = [r["seconds"] for r in rs if r.get("seconds") is not None]
        pk = [r["peak_gb"] for r in rs if r.get("peak_gb") is not None]
        L.append(f"| {m} | {np.mean(de):.2f} | {j.count('new_view')}/{len(j)} | {j.count('same_view')} | {j.count('broken')} | "
                 f"{np.mean(sec):.1f} | {max(pk) if pk else 0:.1f} |")
    holes = [res[k]["render"]["hole_share"] for k in keys if "render" in res[k]]
    if holes:
        L += ["", f"Share of each new view the original camera never saw (holes): mean {np.mean(holes):.2f}"]
    for kind in ("scene", "obj"):
        L += ["", f"### {kind} photos: judge 'new_view' rate"]
        for m in METHODS:
            j = [res[k][m].get("judge") for k in keys if k.startswith(kind) and m in res[k]]
            if j:
                L.append(f"- {m}: {j.count('new_view')}/{len(j)}")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
