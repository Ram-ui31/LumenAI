#!/usr/bin/env python
"""v6: can Qwen-Image 2.1 re-pose people, relight photos and remove motion blur?  (one GPU, ~1.5 h)

  pose     10 RORD people, each re-posed into another RORD person's pose; target given as stick figure / photo /
           words. PCK@0.2 between the detected pose in the result and the target (baseline: the unchanged source),
           Qwen3-VL identity check ("same person and clothes?").
  relight  12 photos x 4 lights (left, right, warm sunset, dim lamp). Qwen edit vs our transfer (Qwen's light, the
           photo's detail). Light effect in the requested direction, fine-detail kept, Qwen3-VL "lit as asked?".
  deblur   16 sharp photos blurred by us (straight motion 21 px at a random angle, and a hand-shake path), so the
           sharp answer is known. Wiener (estimated kernel), Qwen with the user's words, Qwen with a precise
           instruction, ours = Qwen + physics check (Richardson-Lucy from Qwen's guess with the estimated kernel), and
           the same with the true kernel (oracle). PSNR / SSIM / LPIPS vs the sharp photo, sharpness.
  python -m benchmarks.run_v6 [--parts pose,relight,deblur]
Outputs: outputs/eval/v6_{pose,relight,deblur}/{summary.md, results.json, images, grids}
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
from benchmarks import metrics as M  # noqa: E402

EVAL = None


def _out(name):
    d = project_path("outputs") / "eval" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _err(out, label):
    import traceback

    msg = traceback.format_exc()
    print(msg, flush=True)
    with open(out / "errors.txt", "a") as f:
        f.write(f"=== {label}\n{msg}\n")


def _strip(paths_labels, h=220):
    tiles = []
    for lab, src in paths_labels:
        try:
            im = src if isinstance(src, Image.Image) else (Image.fromarray(src) if isinstance(src, np.ndarray) else Image.open(src).convert("RGB"))
        except Exception:  # noqa: BLE001
            im = Image.new("RGB", (h, h), (60, 60, 60))
        im = im.convert("RGB").resize((max(1, int(im.width * h / im.height)), h))
        ImageDraw.Draw(im).rectangle([0, 0, im.width, 13], fill="black")
        ImageDraw.Draw(im).text((3, 1), lab[:40], fill="white")
        tiles.append(im)
    G = Image.new("RGB", (sum(t.width for t in tiles), h), "white")
    x = 0
    for t in tiles:
        G.paste(t, (x, 0))
        x += t.width
    return G


def yes(Q, pipe, ims, q) -> bool:
    return Q.ask(pipe, ims, q + " Answer yes or no.", max_new_tokens=3).strip().lower().startswith("yes")


# ----------------------------------------------------------------------------- pose


def part_pose(Q, pipe, n: int = 10):
    from editing import pose as P
    from benchmarks.data import load

    out = _out("v6_pose")
    (out / "img").mkdir(exist_ok=True)
    rp = out / "results.json"
    res = json.loads(rp.read_text()) if rp.exists() else {}
    people = []
    for s in load(50, 768):
        if len(people) >= n + 3:
            break
        try:
            kp = P.keypoints(s.image)
        except Exception:  # noqa: BLE001
            _err(out, f"detect {s.name}")
            continue
        if kp is not None and kp[:, 2].sum() >= 12:
            people.append((s.name, s.image.convert("RGB"), kp))
    print(f"[pose] {len(people)} people with >= 12 joints", flush=True)
    for i in range(min(n, len(people))):
        sn, simg, skp = people[i]
        tn, timg, tkp = people[(i + 3) % len(people)]
        key = f"{sn}__to__{tn}"
        if key in res:
            continue
        r = {"baseline_pck": P.pck(skp, tkp)}
        tgt = P.retarget(tkp, skp)
        skel = P.draw_skeleton(tgt, simg.size)
        skel.save(out / "img" / f"{key}_skeleton.png")
        words = Q.ask(pipe, [timg], P.DESCRIBE_Q, max_new_tokens=80).strip()
        r["pose_words"] = words
        for m in ("skeleton", "photo", "words"):
            try:
                o, info = P.repose(pipe, Q, simg, m, target_img=skel if m == "skeleton" else timg, pose_words=words, seed=SEED)
                o.save(out / "img" / f"{key}_{m}.png")
                okp = P.keypoints(o)
                r[m] = {**info, "pck": P.pck(okp, tkp) if okp is not None else 0.0,
                        "same_person": yes(Q, pipe, [simg, o], "Are image 1 and image 2 the same person wearing the same clothes?")}
            except Exception:  # noqa: BLE001
                _err(out, f"{m} {key}")
        res[key] = r
        rp.write_text(json.dumps(res, indent=1))
        _strip([("source", simg), ("target pose", timg), ("skeleton", skel)] +
               [(m, out / "img" / f"{key}_{m}.png") for m in ("skeleton", "photo", "words")]).save(out / f"grid_{i:02d}.jpg", quality=85)
    L = ["# v6 pose transfer (Qwen-Image 2.1)", "", f"{len(res)} source -> target pairs. PCK@0.2 (pose match, 1 = perfect).", "",
         "| method | PCK ↑ | same person ↑ | s |", "|---|---|---|---|",
         f"| unchanged source (baseline) | {np.mean([v['baseline_pck'] for v in res.values() if v.get('baseline_pck') is not None]):.2f} | – | – |"]
    for m in ("skeleton", "photo", "words"):
        rs = [v[m] for v in res.values() if m in v]
        if rs:
            L.append(f"| {m} | {np.mean([x['pck'] for x in rs]):.2f} | {sum(x['same_person'] for x in rs)}/{len(rs)} | "
                     f"{np.mean([x['seconds'] for x in rs]):.1f} |")
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


# ----------------------------------------------------------------------------- relight


def _photos(n_scene=6, n_obj=3, n_people=3):
    from benchmarks.data import load

    S = load(50, 768)
    items = [("scene_" + s.name, np.asarray(s.gt.convert("RGB"))) for s in S[::5][:n_scene]]
    items += [("person_" + s.name, np.asarray(s.image.convert("RGB"))) for s in S[2::7][:n_people]]
    rec = project_path("outputs") / "eval" / "recolour" / "inputs"
    for p in sorted(rec.glob("P0*.png")):
        if not p.name.endswith("_obj.png") and not p.name.endswith("_part.png") and len([i for i in items if i[0].startswith("obj")]) < n_obj:
            items.append(("obj_" + p.stem, np.asarray(Image.open(p).convert("RGB"))))
    return items


def part_relight(Q, pipe):
    from editing import relight as R

    out = _out("v6_relight")
    (out / "img").mkdir(exist_ok=True)
    rp = out / "results.json"
    res = json.loads(rp.read_text()) if rp.exists() else {}
    desc = {"left": "lit by a strong light from the left", "right": "lit by a strong light from the right",
            "warm": "lit by warm golden sunset light", "dim": "a dim evening scene lit by a single lamp"}
    for name, photo in _photos():
        row = [("photo", photo)]
        for key in R.LIGHTS:
            k = f"{name}@{key}"
            try:
                if k not in res:
                    q, info = R.qwen_relight(pipe, Q, photo, key, seed=SEED)
                    t = R.transfer(photo, q)
                    Image.fromarray(q).save(out / "img" / f"{k}_qwen.png")
                    Image.fromarray(t).save(out / "img" / f"{k}_transfer.png")
                    res[k] = {}
                    for m, im in (("qwen", q), ("transfer", t)):
                        res[k][m] = {"effect": R.light_effect(photo, im, key), "detail": R.detail_kept(photo, im),
                                     "judge": yes(Q, pipe, [Image.fromarray(photo), Image.fromarray(im)],
                                                  f"Is image 2 the same scene as image 1 but {desc[key]}?")}
                    res[k]["qwen"].update(info)
                    rp.write_text(json.dumps(res, indent=1))
                row += [(f"{key} qwen", out / "img" / f"{k}_qwen.png"), (f"{key} transfer", out / "img" / f"{k}_transfer.png")]
            except Exception:  # noqa: BLE001
                _err(out, k)
        _strip(row, 180).save(out / f"grid_{name}.jpg", quality=85)
    L = ["# v6 relighting", "", "effect = change in the requested direction (left/right: L(left third) - L(right third); warm: b*; "
         "dim: drop in L). detail = fine-detail correlation with the photo (1 = all kept).", "",
         "| light | method | effect ↑ | detail kept ↑ | judge: lit as asked ↑ |", "|---|---|---|---|---|"]
    for key in R.LIGHTS:
        for m in ("qwen", "transfer"):
            rs = [v[m] for k, v in res.items() if k.endswith("@" + key) and m in v]
            if rs:
                L.append(f"| {key} | {m} | {np.mean([x['effect'] for x in rs]):.1f} | {np.mean([x['detail'] for x in rs]):.3f} | "
                         f"{sum(x['judge'] for x in rs)}/{len(rs)} |")
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


# ----------------------------------------------------------------------------- deblur


def part_deblur(Q, pipe):
    from editing import deblur as D

    out = _out("v6_deblur")
    (out / "img").mkdir(exist_ok=True)
    rp = out / "results.json"
    res = json.loads(rp.read_text()) if rp.exists() else {}
    items = _photos(n_scene=8, n_obj=4, n_people=4)
    for i, (name, S) in enumerate(items):
        for kind in ("line", "shake"):
            k = D.line_kernel(21, (37 * i + 11) % 180) if kind == "line" else D.shake_kernel(21, seed=i)
            key = f"{name}@{kind}"
            if key in res:
                continue
            try:
                B = D.blur(S, k, noise=1.0, seed=i)
                ke, kinfo = D.estimate_kernel(B)
                outs = {"blurred": B, "wiener": D.wiener(B, ke, 0.01)}
                t0 = time.perf_counter()
                outs["qwen_user"], _ = D.qwen_deblur(pipe, Q, B, D.USER_PROMPT, seed=SEED)
                outs["qwen_spec"], info = D.qwen_deblur(pipe, Q, B, D.SPEC_PROMPT, seed=SEED)
                outs["ours"] = D.richardson_lucy(B, ke, init=outs["qwen_spec"], iters=15)
                outs["ours_true_k"] = D.richardson_lucy(B, k, init=outs["qwen_spec"], iters=15)
                r = {"kernel_est": kinfo, "qwen_s": round(info["seconds"], 1), "peak_gb": info["peak_gb"]}
                for m, im in outs.items():
                    r[m] = {"psnr": M.psnr(im, S), "ssim": M.ssim(im, S), "lpips": M.lpips_dist(im, S), "sharp": D.sharpness(im)}
                    Image.fromarray(im).save(out / "img" / f"{key}_{m}.png")
                r["sharp_true"] = D.sharpness(S)
                res[key] = r
                rp.write_text(json.dumps(res, indent=1))
                _strip([("sharp (answer)", S)] + [(m, outs[m]) for m in outs]).save(out / f"grid_{key}.jpg", quality=85)
            except Exception:  # noqa: BLE001
                _err(out, key)
    methods = ["blurred", "wiener", "qwen_user", "qwen_spec", "ours", "ours_true_k"]
    L = ["# v6 motion deblur (synthetic blur, sharp answer known)", "",
         "| blur | method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | sharpness (true = %.0f) |" % np.mean([v["sharp_true"] for v in res.values()] or [0]),
         "|---|---|---|---|---|---|"]
    for kind in ("line", "shake"):
        for m in methods:
            rs = [v[m] for k, v in res.items() if k.endswith("@" + kind) and m in v]
            if rs:
                L.append(f"| {kind} | {m} | {np.mean([x['psnr'] for x in rs]):.2f} | {np.mean([x['ssim'] for x in rs]):.3f} | "
                         f"{np.mean([x['lpips'] for x in rs]):.3f} | {np.mean([x['sharp'] for x in rs]):.0f} |")
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


def main(argv=None) -> int:
    from editing import qwen as Q

    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="deblur,relight,pose")
    a = ap.parse_args(argv)
    pipe = Q.load_pipe("nf4")
    Q.enable_memory_savers(pipe)
    for part in a.parts.split(","):
        fn = {"pose": part_pose, "relight": part_relight, "deblur": part_deblur}[part]
        try:
            fn(Q, pipe)
        except Exception:  # noqa: BLE001
            _err(_out(f"v6_{part}"), f"part {part}")
            free()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
