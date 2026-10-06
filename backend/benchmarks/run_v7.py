#!/usr/bin/env python
"""v7: prompt enhancer (only where Qwen misread us) + precise pose + 3-D-aware recolour + object completion for 3-D.
Needs outputs/eval/v7_pe/pe_*.jsonl from modal_pe.py (parts that need it are skipped if it is missing).

  t2i        20 short prompts: raw vs enhancer-rewritten (with its aspect ratio); Qwen3-VL pairwise preference
  deblur     the user's own words vs enhancer rewrite (+ our physics check); LPIPS vs the sharp photo
  relight    the user's words vs enhancer rewrite for 'dim lamp' and 'light from the left'; effect, detail, judge,
             "did it add new objects?"
  pose       precise geometric words (1 round) / correction loop (3 rounds, best by pose match among same-person
             results) / enhancer-rewritten 2-image instruction; PCK + same person
  recolour3d stripes and halves on product photos, flat vs following the 3-D surface (MoGe-2 point map)
  amodal     complete partly hidden objects (cut-outs) before 3-D: Qwen draws the full object on white
  python -m benchmarks.run_v7 [--parts t2i,deblur,relight,pose,recolour3d,amodal]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from benchmarks.run_qwen_check import SEED, free  # noqa: E402
from benchmarks.run_v6 import _err, _out, _strip, yes  # noqa: E402
from benchmarks import metrics as M  # noqa: E402
from benchmarks import v7_cases as C  # noqa: E402
from core.config import project_path  # noqa: E402

def _root() -> Path:
    """Directory holding outputs/ and cache/ (the volume on Modal; the project folder locally)."""
    p = project_path("outputs").resolve().parent
    return p


def _rw(rw: dict, cid: str) -> str | None:
    r = rw.get(cid)
    if not r:
        return None
    t = (r.get("positive_prompt") or "").strip()
    return t or None


def _ratio_size(ratio: str | None, area: int = 1024 * 1024) -> tuple[int, int]:
    m = re.match(r"\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)", ratio or "")
    r = float(m.group(1)) / float(m.group(2)) if m else 1.0
    w = int((area * r) ** 0.5) // 32 * 32
    h = int((area / r) ** 0.5) // 32 * 32
    return w, h


def source_path(name: str) -> Path:
    root = _root()
    kind, rest = name.split("_", 1)
    if kind == "scene":
        return C._rord(root) / f"{rest}_gt.png"
    if kind == "person":
        return C._rord(root) / f"{rest}_input.png"
    return root / "outputs" / "eval" / "recolour" / "inputs" / f"{rest}.png"


# ----------------------------------------------------------------------------- t2i


def part_t2i(Q, pipe, rw):
    import torch

    out = _out("v7_t2i")
    rp = out / "results.json"
    res = json.loads(rp.read_text()) if rp.exists() else {}
    for case in C.t2i_cases():
        cid = case["id"]
        if cid in res:
            continue
        pe = _rw(rw, cid)
        if not pe:
            continue
        try:
            imgs = {}
            for m, prompt, (w, h) in (("raw", case["prompt"], (1024, 1024)),
                                      ("enhanced", pe, _ratio_size(rw[cid].get("wh_ratio")))):
                t0 = time.perf_counter()
                imgs[m] = pipe(prompt=prompt, height=h, width=w, num_inference_steps=40,
                               generator=torch.Generator("cpu").manual_seed(SEED)).images[0]
                imgs[m].save(out / f"{cid.split('|')[1]}_{m}.png")
            order = ["raw", "enhanced"]
            random.Random(cid).shuffle(order)
            a = Q.ask(pipe, [imgs[order[0]], imgs[order[1]]],
                      f"The request was: \"{case['prompt']}\". Which image fulfils the request better and is the better "
                      f"picture? Answer with exactly 1 or 2.", max_new_tokens=3).strip()
            win = order[0] if a.startswith("1") else order[1] if a.startswith("2") else "unclear"
            res[cid] = {"prompt": case["prompt"], "enhanced": pe[:600], "ratio": rw[cid].get("wh_ratio"), "winner": win}
            rp.write_text(json.dumps(res, indent=1))
            _strip([("raw: " + case["prompt"], imgs["raw"]), ("enhanced", imgs["enhanced"])], 300).save(
                out / f"grid_{cid.split('|')[1]}.jpg", quality=85)
        except Exception:  # noqa: BLE001
            _err(out, cid)
    w = [v["winner"] for v in res.values()]
    L = ["# v7 text-to-image: raw prompt vs Qwen prompt enhancer", "",
         f"Qwen3-VL pairwise preference (order randomised): enhanced {w.count('enhanced')}, raw {w.count('raw')}, "
         f"unclear {w.count('unclear')} of {len(w)}."]
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


# ----------------------------------------------------------------------------- deblur


def part_deblur(Q, pipe, rw):
    from editing import deblur as D

    out = _out("v7_deblur")
    (out / "img").mkdir(exist_ok=True)
    v6 = _root() / "outputs" / "eval" / "v6_deblur"
    old = json.loads((v6 / "results.json").read_text())
    rp = out / "results.json"
    res = json.loads(rp.read_text()) if rp.exists() else {}
    for key in sorted(old):
        cid = f"deblur|{key}"
        pe = _rw(rw, cid)
        if not pe or key in res:
            continue
        try:
            name, kind = key.split("@")
            S = np.asarray(Image.open(source_path(name)).convert("RGB"))
            B = np.asarray(Image.open(v6 / "img" / f"{key}_blurred.png").convert("RGB"))
            q, info = D.qwen_deblur(pipe, Q, B, pe, seed=SEED)
            ke, _ = D.estimate_kernel(B)
            chk = D.richardson_lucy(B, ke, init=q, iters=15)
            r = {"enhanced_prompt": pe[:500]}
            for m, im in (("enhanced", q), ("enhanced+check", chk)):
                Image.fromarray(im).save(out / "img" / f"{key}_{m}.png")
                r[m] = {"psnr": M.psnr(im, S), "ssim": M.ssim(im, S), "lpips": M.lpips_dist(im, S)}
            for m in ("blurred", "qwen_user", "qwen_spec", "ours"):
                r[m] = {k: old[key][m][k] for k in ("psnr", "ssim", "lpips")}
            res[key] = r
            rp.write_text(json.dumps(res, indent=1))
            _strip([("sharp", S), ("blurred", B), ("user's words (v6)", v6 / "img" / f"{key}_qwen_user.png"),
                    ("enhanced prompt", q), ("enhanced + physics", chk)], 220).save(out / f"grid_{key}.jpg", quality=85)
        except Exception:  # noqa: BLE001
            _err(out, key)
    L = ["# v7 deblur: the user's own words, rewritten by the enhancer", "", "| blur | method | LPIPS ↓ | PSNR ↑ |", "|---|---|---|---|"]
    for kind in ("line", "shake"):
        for m in ("blurred", "qwen_user", "qwen_spec", "ours", "enhanced", "enhanced+check"):
            rs = [v[m] for k, v in res.items() if k.endswith("@" + kind) and m in v]
            if rs:
                L.append(f"| {kind} | {m} | {np.mean([x['lpips'] for x in rs]):.3f} | {np.mean([x['psnr'] for x in rs]):.2f} |")
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


# ----------------------------------------------------------------------------- relight


def part_relight(Q, pipe, rw):
    from editing import relight as R

    out = _out("v7_relight")
    (out / "img").mkdir(exist_ok=True)
    rp = out / "results.json"
    res = json.loads(rp.read_text()) if rp.exists() else {}
    desc = {"dim": "an evening scene lit only by a lamp", "left": "lit by light coming from the left"}
    for name, path in C.relight_photos(_root()):
        photo = np.asarray(Image.open(path).convert("RGB"))
        for k, user in C.RELIGHT_USER.items():
            key = f"{name}@{k}"
            pe = _rw(rw, f"relight|{key}")
            if not pe or key in res:
                continue
            try:
                r = {"enhanced_prompt": pe[:500]}
                ims = {}
                for m, prompt in (("user", user), ("enhanced", pe)):
                    res_q = Q.edit(pipe, Image.fromarray(photo), prompt, seed=SEED, steps=40, resolution=1024)
                    im = np.asarray(res_q.image.convert("RGB").resize(photo.shape[1::-1], Image.LANCZOS))
                    ims[m] = im
                    Image.fromarray(im).save(out / "img" / f"{key}_{m}.png")
                    P1, P2 = Image.fromarray(photo), Image.fromarray(im)
                    r[m] = {"effect": R.light_effect(photo, im, k), "detail": R.detail_kept(photo, im),
                            "judge": yes(Q, pipe, [P1, P2], f"Is image 2 the same scene as image 1 but {desc[k]}?"),
                            "added_objects": yes(Q, pipe, [P1, P2], "Does image 2 contain any object that is not in "
                                                 "image 1, for example a new lamp, furniture or decoration?")}
                res[key] = r
                rp.write_text(json.dumps(res, indent=1))
                _strip([("photo", photo), (f"{k}: user's words", ims["user"]), (f"{k}: enhanced", ims["enhanced"])], 240).save(
                    out / f"grid_{key}.jpg", quality=85)
            except Exception:  # noqa: BLE001
                _err(out, key)
    L = ["# v7 relight: the user's words vs the enhancer", "",
         "| light | prompt | effect ↑ | detail kept ↑ | lit as asked ↑ | added new objects ↓ |", "|---|---|---|---|---|---|"]
    for k in C.RELIGHT_USER:
        for m in ("user", "enhanced"):
            rs = [v[m] for kk, v in res.items() if kk.endswith("@" + k) and m in v]
            if rs:
                L.append(f"| {k} | {m} | {np.mean([x['effect'] for x in rs]):.1f} | {np.mean([x['detail'] for x in rs]):.3f} | "
                         f"{sum(x['judge'] for x in rs)}/{len(rs)} | {sum(x['added_objects'] for x in rs)}/{len(rs)} |")
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


# ----------------------------------------------------------------------------- pose


def part_pose(Q, pipe, rw):
    """Resumable per method: a re-run only computes what is missing (gist+precise loop, enhancer once rewrites exist)."""
    import torch

    from editing import pose as P

    out = _out("v7_pose")
    (out / "img").mkdir(exist_ok=True)
    rp = out / "results.json"
    res = json.loads(rp.read_text()) if rp.exists() else {}
    old = json.loads((_root() / "outputs" / "eval" / "v6_pose" / "results.json").read_text())
    rord = C._rord(_root())
    for i, key in enumerate(sorted(old)):
        r = res.get(key, {})
        pe = _rw(rw, f"pose|{key}")
        if "loop" in r and "gist_loop" in r and (not pe or "enhanced_photo" in r):
            continue
        try:
            s, t = key.split("__to__")
            simg = Image.open(rord / f"{s}_input.png").convert("RGB")
            timg = Image.open(rord / f"{t}_input.png").convert("RGB")
            tkp = P.keypoints(timg)
            if tkp is None:
                continue
            same = lambda o: yes(Q, pipe, [simg, o], "Are image 1 and image 2 the same person wearing the same clothes?")  # noqa: E731
            r.setdefault("v6_words", old[key].get("words"))
            r.setdefault("v6_photo", old[key].get("photo"))
            r["target_words"] = P.describe_pose(tkp)
            if "loop" not in r:
                # precise words, single round = round 0 of the loop; loop = best of up to 3 rounds
                o, info = P.repose_loop(pipe, Q, simg, tkp, rounds=3, seed=SEED, same_person=same)
                o.save(out / "img" / f"{key}_loop.png")
                h0 = info["history"][0]
                r["precise_words"] = {"pck": h0["pck"], "same_person": h0["same_person"], "seconds": h0["seconds"]}
                r["loop"] = {"pck": info["pck"], "best_round": info["best_round"], "rounds": len(info["history"]),
                             "same_person": True, "history": info["history"]}
            if "gist_loop" not in r:
                # v7b: Qwen3-VL's overall description of the target pose ("sitting cross-legged on the floor holding a
                # plant") first, then the exact joint angles - the angles alone lost the gist in some pairs
                gist = Q.ask(pipe, [timg], P.DESCRIBE_Q, max_new_tokens=120).strip()
                o, info = P.repose_loop(pipe, Q, simg, tkp, rounds=3, seed=SEED, same_person=same, gist=gist)
                o.save(out / "img" / f"{key}_gist_loop.png")
                r["gist_loop"] = {"pck": info["pck"], "best_round": info["best_round"], "rounds": len(info["history"]),
                                  "same_person": True, "gist": gist, "history": info["history"]}
            if pe and "enhanced_photo" not in r:
                torch.cuda.reset_peak_memory_stats()
                po = pipe(prompt=pe, image=[simg, timg], num_inference_steps=40, output_resolution=1024,
                          generator=torch.Generator("cpu").manual_seed(SEED)).images[0].convert("RGB").resize(simg.size)
                po.save(out / "img" / f"{key}_enhanced_photo.png")
                pk = P.keypoints(po)
                r["enhanced_photo"] = {"pck": P.pck(pk, tkp) if pk is not None else 0.0, "same_person": same(po),
                                       "prompt": pe[:500]}
            res[key] = r
            rp.write_text(json.dumps(res, indent=1))
            img = out / "img"
            tiles = [("person", simg), ("target", timg),
                     ("v6 words", _root() / "outputs" / "eval" / "v6_pose" / "img" / f"{key}_words.png"),
                     (f"precise+loop {r['loop']['pck'] or 0:.2f}", img / f"{key}_loop.png"),
                     (f"gist+precise {r['gist_loop']['pck'] or 0:.2f}", img / f"{key}_gist_loop.png")]
            if "enhanced_photo" in r:
                tiles.append((f"enhancer+photo {r['enhanced_photo']['pck']:.2f}", img / f"{key}_enhanced_photo.png"))
            _strip(tiles, 240).save(out / f"grid_{i:02d}.jpg", quality=85)
        except Exception:  # noqa: BLE001
            _err(out, key)
    L = ["# v7 pose: precise words, correction loop, enhancer", "", "| method | PCK ↑ | same person ↑ |", "|---|---|---|"]
    for m in ("v6_words", "v6_photo", "precise_words", "loop", "gist_loop", "enhanced_photo"):
        rs = [v[m] for v in res.values() if v.get(m)]
        if rs:
            L.append(f"| {m} | {np.mean([x['pck'] or 0 for x in rs]):.2f} | {sum(bool(x['same_person']) for x in rs)}/{len(rs)} |")
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


# ----------------------------------------------------------------------------- recolour 3-D


def part_recolour3d(Q, pipe, rw):
    from editing import recolour as RC
    from editing import views3d as V

    out = _out("v7_recolour3d")
    rec = _root() / "outputs" / "eval" / "recolour" / "inputs"
    names = ["P02_coffee-mug", "P29_bucket", "P07_water-bottle", "P12_ceramic-vase", "P26_teapot", "P10_baseball-cap",
             "P21_kettle", "P13_scarf"]
    progs = {"h-stripes": {"op": "stripes", "axis": "horizontal", "count": 3, "width": 0.35},
             "v-stripes": {"op": "stripes", "axis": "vertical", "count": 6, "width": 0.4},
             "upper half": {"op": "half", "side": "upper", "frac": 0.5}}
    for n in names:
        p, mp = rec / f"{n}.png", rec / f"{n}_obj.png"
        if not p.exists() or not mp.exists():
            continue
        try:
            img = Image.open(p).convert("RGB")
            I = np.asarray(img)
            Mk = (np.asarray(Image.open(mp)) > 0).astype(np.uint8)
            g = V.estimate_geometry(img)
            row = [("photo", I)]
            for lab, reg in progs.items():
                for mode, pts in (("flat", None), ("3-D", g.points)):
                    r = RC.recolour(I, Mk, {"target_color": "blue", "region": reg}, points=pts)
                    row.append((f"{lab} {mode}", r.image))
            _strip(row, 230).save(out / f"grid_{n}.jpg", quality=86)
        except Exception:  # noqa: BLE001
            _err(out, n)
    V.unload_moge()
    free()
    (out / "summary.md").write_text("# v7 recolour following the 3-D surface\n\nVisual comparison: see grid_*.jpg "
                                    "(flat = picture coordinates, 3-D = MoGe-2 point map).\n")


# ----------------------------------------------------------------------------- amodal completion for view-in-3D


from editing.object3d import completion_mask  # noqa: E402,F401  (moved to the library)


def part_amodal(Q, pipe, rw):
    out = _out("v7_amodal")
    cut = _root() / "outputs" / "eval" / "v5_object3d" / "cutout"
    rp = out / "results.json"
    res = json.loads(rp.read_text()) if rp.exists() else {}
    for p in sorted(cut.glob("rord_*.png")):
        if p.stem in res:
            continue
        try:
            rgba = Image.open(p).convert("RGBA")
            white = Image.new("RGB", rgba.size, (255, 255, 255))
            white.paste(rgba, mask=rgba.split()[3])
            what = Q.ask(pipe, [white], "What is shown in this cut-out? Answer in at most four words.", max_new_tokens=10).strip(" .")
            instr = (f"This is a cut-out of {what}; parts of it are hidden or cut off. Show the complete {what} - whole "
                     f"body, every limb and any part that was hidden or cut off - alone on a plain pure white background. "
                     f"Keep exactly the same pose and posture (if it is sitting or kneeling, it stays sitting or kneeling), "
                     f"the same size and position, and identical appearance, clothing and colours. Keep everything it is "
                     f"holding or carrying. Remove only separate things that were in front of it and are not part of it, "
                     f"such as a table.")
            r = Q.edit(pipe, white, instr, seed=SEED, steps=40, resolution=1024)
            done, keep, how = completion_mask(r.image)
            res_how = how
            out_rgba = np.dstack([np.asarray(done), keep.astype(np.uint8) * 255])  # noqa: F841
            Image.fromarray(out_rgba, "RGBA").save(out / f"{p.stem}.png")
            res[p.stem] = {"what": what, "seconds": r.seconds, "mask_from": res_how,
                           "area_before": int((np.asarray(rgba)[..., 3] > 0).sum()), "area_after": int(keep.sum())}
            rp.write_text(json.dumps(res, indent=1))
            _strip([("cut-out", white), ("completed", done)], 300).save(out / f"grid_{p.stem}.jpg", quality=86)
        except Exception:  # noqa: BLE001
            _err(out, p.stem)
    (out / "summary.md").write_text("# v7 object completion before 3-D\n\n" + "\n".join(
        f"- {k}: {v['what']} (area {v['area_before']} -> {v['area_after']} px)" for k, v in res.items()) + "\n")


def main(argv=None) -> int:
    from editing import qwen as Q

    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="recolour3d,amodal,pose,deblur,relight,t2i")
    ap.add_argument("--redo", default="", help="parts whose saved results are discarded first (e.g. amodal)")
    a = ap.parse_args(argv)
    for part in filter(None, a.redo.split(",")):
        f = _out(f"v7_{part}") / "results.json"
        if f.exists():
            f.rename(f.with_name("results_previous.json"))
    rw = C.load_rewrites(_root())
    print(f"enhancer rewrites available: {len(rw)}", flush=True)
    pipe = Q.load_pipe("nf4")
    Q.enable_memory_savers(pipe)
    for part in a.parts.split(","):
        fn = {"t2i": part_t2i, "deblur": part_deblur, "relight": part_relight, "pose": part_pose,
              "recolour3d": part_recolour3d, "amodal": part_amodal}[part]
        try:
            fn(Q, pipe, rw)
        except Exception:  # noqa: BLE001
            _err(_out(f"v7_{part}"), f"part {part}")
            free()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
