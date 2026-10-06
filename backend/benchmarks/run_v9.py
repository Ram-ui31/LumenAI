#!/usr/bin/env python
"""v9: our pipeline vs the teammates' model list, on the tasks where both can run.

Main GPU run (this file, one model family at a time so the 16 GB-class budget is respected):
  prep      inputs for every task (also read by the third-party runners in modal_benchmark_extra.py)
  qwen      OURS with Qwen-Image 2.1: guard (Qwen3-VL as filter), watermark incl. "survives our edit", remove-by-name
            masks (Qwen3-VL box -> SAM), product shoot, GEdit edits (the real web-API router + self-repair),
            new camera angle (AnyAngle LoRA + text, +-45 deg), AI upscaling (Qwen refine)
  klein     FLUX.2 klein 4B: text-to-image, product shoot, GEdit edits
  zimage    Z-Image Turbo: text-to-image
  fluxdev   FLUX.1 dev (nf4): text-to-image
  ip2p      InstructPix2Pix: GEdit edits
  q2511     Qwen-Image-Edit-2511 (nf4): GEdit edits, then + fal Multiple-Angles LoRA: new camera angle
  guards    SingGuard-4b, deberta-v3-base-prompt-injection-v2
  segformer Segformer person masks (+ our LaMa on them)
Third-party runners (own images, write to outputs/eval/v9/<name>/): leffa (pose), sf3d (3-D), tadsr (upscale), mat
(MAT + LaMa via IOPaint, removal).
  score     metrics for everything (PCK, silhouette IoU + mesh holes, PSNR/SSIM/LPIPS, removal metrics, DINOv2)
  judge     independent judge = Gemini 2.5 Flash (Modal secret adobe-ps-trends); all methods of one case in one call,
            labels shuffled. Falls back to Qwen3-VL (same family as ours -> marked as biased) if Gemini is unavailable.
  report    outputs/eval/v9/summary.md

    python -m benchmarks.run_v9 --parts prep,qwen,klein,zimage,fluxdev,ip2p,q2511,guards,segformer
    python -m benchmarks.run_v9 --parts score,judge,report
Everything is resumable: finished items are skipped on a re-run.
"""
from __future__ import annotations

import argparse
import base64
import gc
import io
import json
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter  # noqa: E402

from core.config import project_path  # noqa: E402

SEED = 7
OUT = project_path("outputs") / "eval" / "v9"
IN = OUT / "inputs"
YAWS = [-45.0, 45.0]


# ----------------------------------------------------------------------------- small helpers


def _root() -> Path:
    return project_path("outputs").resolve().parent


def jload(p: Path, default=None):
    try:
        return json.loads(Path(p).read_text())
    except Exception:  # noqa: BLE001
        return {} if default is None else default


def jsave(p: Path, d) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1, default=str))
    tmp.replace(p)


def err(label: str) -> None:
    msg = traceback.format_exc()
    print(msg, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "errors.txt", "a") as f:
        f.write(f"=== {time.strftime('%H:%M:%S')} {label}\n{msg}\n")


def rgb(p) -> Image.Image:
    return Image.open(p).convert("RGB")


def free() -> None:
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def timed(fn):
    """-> (result, seconds, peak GB allocated on the GPU during fn)."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    r = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    return r, round(time.perf_counter() - t0, 2), round(peak, 2)


def fit_size(w: int, h: int, area: int = 1024 * 1024, mult: int = 16) -> tuple[int, int]:
    s = (area / (w * h)) ** 0.5
    return max(mult, int(w * s) // mult * mult), max(mult, int(h * s) // mult * mult)


def strip(tiles, h=240) -> Image.Image:
    ims = []
    for lab, im in tiles:
        try:
            im = im if isinstance(im, Image.Image) else Image.open(im)
        except Exception:  # noqa: BLE001
            im = Image.new("RGB", (h, h), (60, 60, 60))
        im = im.convert("RGB")
        im = im.resize((max(1, int(im.width * h / im.height)), h))
        ImageDraw.Draw(im).rectangle([0, 0, im.width, 13], fill="black")
        ImageDraw.Draw(im).text((3, 1), str(lab)[:44], fill="white")
        ims.append(im)
    G = Image.new("RGB", (sum(i.width for i in ims) or 1, h), "white")
    x = 0
    for i in ims:
        G.paste(i, (x, 0))
        x += i.width
    return G


class Results:
    """outputs/eval/v9/<task>/results.json: {case: {method: {...}}}"""

    def __init__(self, task: str):
        self.dir = OUT / task
        (self.dir / "img").mkdir(parents=True, exist_ok=True)
        self.p = self.dir / "results.json"
        self.d = jload(self.p)

    def has(self, case: str, method: str) -> bool:
        return method in self.d.get(case, {})

    def put(self, case: str, method: str, row: dict) -> None:
        self.d.setdefault(case, {})[method] = row
        jsave(self.p, self.d)

    def img(self, case: str, method: str) -> Path:
        return self.dir / "img" / f"{case}__{method}.png"


# ----------------------------------------------------------------------------- prep


def part_prep():
    """Inputs for every task. Third-party runners only read IN/*, so they need none of our code."""
    from benchmarks.data import load
    from benchmarks import v9_cases as C

    root = _root()
    man = jload(IN / "manifest.json")

    # pose: the same 10 pairs as v6/v7 (ours = v7 gist+precise+loop, already scored with the same PCK code)
    d = IN / "pose"
    d.mkdir(parents=True, exist_ok=True)
    pres = jload(root / "outputs" / "eval" / "v6_pose" / "results.json")
    from benchmarks.v7_cases import _rord

    rord = _rord(root)
    pairs = []
    for key in sorted(pres):
        s, t = key.split("__to__")
        try:
            rgb(rord / f"{s}_input.png").save(d / f"{key}_src.png")
            rgb(rord / f"{t}_input.png").save(d / f"{key}_tgt.png")
            pairs.append(key)
        except Exception:  # noqa: BLE001
            err(f"prep pose {key}")
    man["pose"] = pairs

    # 3-D: the cut-outs Hunyuan3D already converted (v5: as cut, v8: after our completion)
    d = IN / "3d"
    d.mkdir(parents=True, exist_ok=True)
    objs = []
    for sub in ("v5_object3d", "v8_object3d"):
        for p in sorted((root / "outputs" / "eval" / sub / "cutout").glob("*.png")):
            name = f"{sub[:2]}_{p.stem}"
            if (root / "outputs" / "eval" / sub / "glb" / f"{p.stem}.glb").exists():
                Image.open(p).convert("RGBA").save(d / f"{name}.png")
                objs.append({"name": name, "hunyuan_glb": str(root / "outputs" / "eval" / sub / "glb" / f"{p.stem}.glb"),
                             "hunyuan_results": str(root / "outputs" / "eval" / sub / "results.json"), "stem": p.stem})
    man["3d"] = objs

    # upscaling: 16 RORD frames, 512 crop = HR; LR = x4 down (bicubic) + light blur + JPEG 75 (real-world-like)
    d = IN / "sr"
    d.mkdir(parents=True, exist_ok=True)
    samples = load(50, 768)
    sr = []
    for s in samples[::3][:16]:
        g = s.gt.convert("RGB")
        w, h = g.size
        c = min(w, h, 512)
        hr = g.crop(((w - c) // 2, (h - c) // 2, (w - c) // 2 + c, (h - c) // 2 + c)).resize((512, 512), Image.LANCZOS)
        lr = hr.filter(ImageFilter.GaussianBlur(0.8)).resize((128, 128), Image.BICUBIC)
        b = io.BytesIO()
        lr.save(b, "JPEG", quality=75)
        lr = Image.open(io.BytesIO(b.getvalue())).convert("RGB")
        hr.save(d / f"{s.name}_hr.png")
        lr.save(d / f"{s.name}_lr.png")
        sr.append(s.name)
    man["sr"] = sr

    # removal: RORD-50 + the hole LaMa/Smart used in v5 (same masks for every remover)
    d = IN / "removal"
    d.mkdir(parents=True, exist_ok=True)
    v5 = root / "outputs" / "eval" / "v5_remove"
    rem = []
    for s in samples:
        hp = v5 / "LaMa" / f"{s.name}_hole.png"
        if not hp.exists():
            continue
        s.image.convert("RGB").save(d / f"{s.name}_input.png")
        Image.open(hp).convert("L").save(d / f"{s.name}_hole.png")
        rem.append(s.name)
    man["removal"] = rem

    # product shoot: 8 product photos x 3 scenes
    d = IN / "product"
    d.mkdir(parents=True, exist_ok=True)
    rec = root / "outputs" / "eval" / "recolour" / "inputs"
    prods = []
    for p in sorted(rec.glob("P*.png")):
        if p.name.endswith(("_obj.png", "_part.png")) or len(prods) >= 8:
            continue
        name = p.stem
        rgb(p).save(d / f"{name}.png")
        objp = rec / f"{name}_obj.png"
        if objp.exists():
            Image.open(objp).convert("L").save(d / f"{name}_obj.png")
        prods.append({"name": name, "product": name.split("_", 1)[1].replace("-", " ") if "_" in name else name})
    man["product"] = prods

    # GEdit-Bench (English), 3 per task type
    d = IN / "gedit"
    d.mkdir(parents=True, exist_ok=True)
    ge = man.get("gedit") or []
    if not ge:
        try:
            for r in C.gedit_subset(3):
                r["input_image"].convert("RGB").save(d / f"{r['key']}.png")
                ge.append({"key": r["key"], "task": r["task_type"], "instruction": r["instruction"]})
        except Exception:  # noqa: BLE001
            err("prep gedit")
    man["gedit"] = ge

    # new camera angle: same 16 photos as v5 views (8 RORD scenes + 8 products)
    d = IN / "angles"
    d.mkdir(parents=True, exist_ok=True)
    from benchmarks.run_v5_views import inputs as view_inputs

    ang = []
    for name, img, mask in view_inputs(8, 8):
        img.save(d / f"{name}.png")
        Image.fromarray((mask > 0).astype(np.uint8) * 255).save(d / f"{name}_mask.png")
        ang.append(name)
    man["angles"] = ang

    # text-to-image: v7 prompts, raw + Qwen-enhanced (ours already rendered both in v7)
    from benchmarks import v7_cases as V7
    from benchmarks.run_v7 import _ratio_size, _rw

    rw = V7.load_rewrites(root)
    t2i = []
    for case in V7.t2i_cases():
        cid = case["id"]
        i = cid.split("|")[1]
        pe = _rw(rw, cid)
        w, h = _ratio_size(rw.get(cid, {}).get("wh_ratio"))
        t2i.append({"id": i, "prompt": case["prompt"], "enhanced": pe or case["prompt"], "w": w, "h": h,
                    "ours_raw": str(root / "outputs" / "eval" / "v7_t2i" / f"{i}_raw.png"),
                    "ours_enhanced": str(root / "outputs" / "eval" / "v7_t2i" / f"{i}_enhanced.png")})
    man["t2i"] = t2i
    man["guard"] = C.guard_cases()
    man["prepared_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    jsave(IN / "manifest.json", man)
    print("[prep] " + ", ".join(f"{k}: {len(v)}" for k, v in man.items() if isinstance(v, list)), flush=True)


def manifest() -> dict:
    m = jload(IN / "manifest.json")
    if not m:
        raise RuntimeError("run --parts prep first")
    return m


# ----------------------------------------------------------------------------- watermark (CPU)


WM_BITS = b"LUMN"


def wm_attacks(img: np.ndarray) -> dict[str, np.ndarray]:
    """img RGB uint8 -> attacked versions (same size where the decoder needs it)."""
    H, W = img.shape[:2]

    def jpeg(q):
        b = io.BytesIO()
        Image.fromarray(img).save(b, "JPEG", quality=q)
        return np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"))
    half = cv2.resize(cv2.resize(img, (W // 2, H // 2), interpolation=cv2.INTER_AREA), (W, H), interpolation=cv2.INTER_CUBIC)
    c = img[int(0.05 * H):int(0.95 * H), int(0.05 * W):int(0.95 * W)]
    crop = cv2.resize(c, (W, H), interpolation=cv2.INTER_CUBIC)
    bright = np.clip(img.astype(np.int16) + 20, 0, 255).astype(np.uint8)
    return {"none": img, "jpeg90": jpeg(90), "jpeg75": jpeg(75), "jpeg50": jpeg(50), "resize50%": half,
            "crop10%": crop, "brighter": bright}


def wm_bit_acc(found: bytes | None) -> float:
    if not found:
        return 0.0
    a = np.unpackbits(np.frombuffer(WM_BITS, np.uint8))
    b = np.unpackbits(np.frombuffer(bytes(found)[:len(WM_BITS)].ljust(len(WM_BITS), b"\0"), np.uint8))
    return float((a == b).mean())


def wm_embed_decode(img: np.ndarray, method: str):
    from imwatermark import WatermarkDecoder, WatermarkEncoder

    enc = WatermarkEncoder()
    enc.set_watermark("bytes", WM_BITS)
    bgr = enc.encode(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), method)
    marked = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    dec = WatermarkDecoder("bytes", len(WM_BITS) * 8)

    def decode(x: np.ndarray) -> bytes | None:
        try:
            return dec.decode(cv2.cvtColor(x, cv2.COLOR_RGB2BGR), method)
        except Exception:  # noqa: BLE001
            return None
    return marked, decode


def part_watermark(E=None):
    """dwtDct (teammate's) and dwtDctSvd on 12 photos: invisibility (PSNR) and bit accuracy after attacks, incl. a
    light pass through our own Qwen edit (does the mark survive an AI edit?)."""
    from benchmarks import metrics as M

    R = Results("watermark")
    m = manifest()
    photos = [IN / "sr" / f"{n}_hr.png" for n in m["sr"][:12]]
    for p in photos:
        img = np.asarray(rgb(p))
        for method in ("dwtDct", "dwtDctSvd"):
            if R.has(p.stem, method):
                continue
            try:
                t0 = time.perf_counter()
                marked, decode = wm_embed_decode(img, method)
                emb_s = time.perf_counter() - t0
                row = {"psnr": round(M.psnr(marked, img), 2), "embed_s": round(emb_s, 3)}
                for k, x in wm_attacks(marked).items():
                    t1 = time.perf_counter()
                    row[k] = wm_bit_acc(decode(x))
                    if k == "none":
                        row["decode_s"] = round(time.perf_counter() - t1, 3)
                if E is not None:     # our AI edit: Qwen refine at strength 0.3 ("make it a bit warmer")
                    o = E.refine(Image.fromarray(marked), "a high quality photo with slightly warmer light", SEED, 0.3,
                                 resolution=512)
                    row["ai_edit"] = wm_bit_acc(decode(np.asarray(o.resize((img.shape[1], img.shape[0])))))
                R.put(p.stem, method, row)
            except Exception:  # noqa: BLE001
                err(f"watermark {p.stem} {method}")


# ----------------------------------------------------------------------------- guard parsing


def parse_singguard(text: str) -> str:
    """SingGuard answers "<answer>Safe</answer>" or the violated category ("<answer>A. Sexual Content Risk</answer>")."""
    t = text.lower()
    m = re.search(r"<answer>(.*?)</answer>", t, re.S)
    if not m:
        return "unclear"
    a = m.group(1).strip()
    return "allow" if a.startswith("safe") else "block"


def parse_allow_block(text: str) -> str:
    t = text.strip().upper()
    return "block" if t.startswith("BLOCK") or ("BLOCK" in t and "ALLOW" not in t) else \
        "allow" if "ALLOW" in t else "unclear"


# ----------------------------------------------------------------------------- stage: ours (Qwen-Image 2.1)


def stage_qwen():
    import torch

    from editing import qwen as Q
    from server import web_api as A
    from benchmarks import v9_cases as C

    m = manifest()
    pipe = Q.load_pipe("nf4")
    Q.enable_memory_savers(pipe)
    E = A.Engine()
    E.pipe, E.Q, E.loaded_at = pipe, Q, 0.0

    # guard: our Qwen3-VL (already in memory -> 0 GB extra)
    R = Results("guard")
    for c in m["guard"]:
        if R.has(c["id"], "ours_qwen3vl"):
            continue
        try:
            def ask():
                try:
                    return E.ask([], C.OUR_GUARD_Q.format(text=c["text"]), 4)
                except Exception:  # noqa: BLE001   (processor wants an image)
                    return E.ask([Image.new("RGB", (64, 64), "white")], C.OUR_GUARD_Q.format(text=c["text"]), 4)
            raw, sec, peak = timed(ask)
            R.put(c["id"], "ours_qwen3vl", {"pred": parse_allow_block(raw), "raw": raw[:40], "seconds": sec,
                                            "extra_gb": 0.0})
        except Exception:  # noqa: BLE001
            err(f"guard ours {c['id']}")

    try:
        part_watermark(E)
    except Exception:  # noqa: BLE001
        err("watermark")

    # remove-by-name masks: Qwen3-VL box -> SAM 2.1 (what "remove the person" does in the web API)
    from benchmarks.data import load

    R = Results("seg")
    for s in load(50, 768):
        if s.name not in m["removal"] or R.has(s.name, "ours_vlm_sam"):
            continue
        try:
            mk, sec, peak = timed(lambda: A._ground_mask(E, s.image.convert("RGB"), "the person"))
            mk = np.zeros(s.mask.shape, np.uint8) if mk is None else mk
            Image.fromarray(mk * 255).save(R.img(s.name, "ours_vlm_sam"))
            R.put(s.name, "ours_vlm_sam", {"iou": iou(mk, s.mask), "seconds": sec, "peak_gb": peak})
        except Exception:  # noqa: BLE001
            err(f"seg ours {s.name}")

    # AI upscaling: Lanczos x4 then a light Qwen refine (our model's best shot at super-resolution)
    R = Results("sr")
    for n in m["sr"]:
        if R.has(n, "ours_qwen"):
            continue
        try:
            lr = rgb(IN / "sr" / f"{n}_lr.png")
            up = lr.resize((512, 512), Image.LANCZOS)
            o, sec, peak = timed(lambda: E.refine(up, "a sharp, detailed, high-resolution photo", SEED, 0.3, resolution=768))
            o.resize((512, 512), Image.LANCZOS).save(R.img(n, "ours_qwen"))
            R.put(n, "ours_qwen", {"seconds": sec, "peak_gb": peak})
        except Exception:  # noqa: BLE001
            err(f"sr ours {n}")

    # text-to-image timing for ours (images themselves come from the v7 run; 3 prompts are enough for the speed)
    R = Results("t2i")
    for t in m["t2i"][:3]:
        if R.has(t["id"], "ours_enhanced"):
            continue
        try:
            _, sec, peak = timed(lambda: pipe(prompt=t["enhanced"], height=t["h"], width=t["w"], num_inference_steps=40,
                                              generator=torch.Generator("cpu").manual_seed(SEED)).images[0])
            R.put(t["id"], "ours_enhanced", {"seconds": sec, "peak_gb": peak, "timing_only": True})
        except Exception:  # noqa: BLE001
            err(f"t2i ours timing {t['id']}")

    # product shoot
    R = Results("product")
    for pr in m["product"]:
        img = rgb(IN / "product" / f"{pr['name']}.png")
        for sk in C.PRODUCT_SCENES:
            case = f"{pr['name']}@{sk}"
            if R.has(case, "ours"):
                continue
            try:
                o, sec, peak = timed(lambda: E.edit([img], C.product_instruction(pr["product"], sk)))
                o.save(R.img(case, "ours"))
                R.put(case, "ours", {"seconds": sec, "peak_gb": peak})
            except Exception:  # noqa: BLE001
                err(f"product ours {case}")

    # GEdit: the real product path (router + measured checks + one self-repair)
    R = Results("gedit")
    for g in m["gedit"]:
        if R.has(g["key"], "ours"):
            continue
        try:
            img = rgb(IN / "gedit" / f"{g['key']}.png")
            r, sec, peak = timed(lambda: A.lumen_edit(E, img, g["instruction"]))
            out = Image.open(io.BytesIO(base64.b64decode(r["image"].split(",", 1)[1]))).convert("RGB")
            out.save(R.img(g["key"], "ours"))
            R.put(g["key"], "ours", {"seconds": sec, "peak_gb": peak, "route": r["route"],
                                     "checks_ok": all(c["ok"] for c in r["checks"]),
                                     "repaired": any(c.get("repaired") for c in r["checks"])})
        except Exception:  # noqa: BLE001
            err(f"gedit ours {g['key']}")

    # new camera angle: MoGe render at +-45 deg, then AnyAngle (render as guide) and text-only
    from editing import views3d as V

    R = Results("angles")
    rd = R.dir / "render"
    rd.mkdir(exist_ok=True)
    for name in m["angles"]:
        if all((rd / f"{name}@{y:+.0f}.png").exists() for y in YAWS):
            continue
        try:
            img = rgb(IN / "angles" / f"{name}.png")
            mask = (np.asarray(Image.open(IN / "angles" / f"{name}_mask.png")) > 0).astype(np.uint8)
            g = V.estimate_geometry(img)
            for y in YAWS:
                r, valid, _ = V.render_view(g, yaw=y, subject=mask)
                Image.fromarray(r).save(rd / f"{name}@{y:+.0f}.png")
                Image.fromarray(valid.astype(np.uint8) * 255).save(rd / f"{name}@{y:+.0f}_valid.png")
                Image.fromarray(V.fill_lama(r, V.hole_mask(valid))).save(rd / f"{name}@{y:+.0f}_lama.png")
        except Exception:  # noqa: BLE001
            err(f"angles geometry {name}")
    V.unload_moge()
    free()

    def angle_cases():
        for name in m["angles"]:
            for y in YAWS:
                key = f"{name}@{y:+.0f}"
                if (rd / f"{key}.png").exists():
                    yield name, y, key

    for name, y, key in angle_cases():
        if R.has(key, "ours_text"):
            continue
        try:
            img = rgb(IN / "angles" / f"{name}.png")
            (o, info), sec, peak = timed(lambda: V.fill_text(pipe, Q, img, y, seed=SEED))
            Image.fromarray(o).save(R.img(key, "ours_text"))
            R.put(key, "ours_text", {"seconds": sec, "peak_gb": peak, **_fid(V, o, rd, key)})
        except Exception:  # noqa: BLE001
            err(f"angles text {key}")
    try:
        V.load_anyangle(pipe)
        for name, y, key in angle_cases():
            if R.has(key, "ours_anyangle"):
                continue
            try:
                img = rgb(IN / "angles" / f"{name}.png")
                lama = np.asarray(rgb(rd / f"{key}_lama.png"))
                (o, info), sec, peak = timed(lambda: V.fill_anyangle(pipe, Q, img, lama, seed=SEED))
                Image.fromarray(o).save(R.img(key, "ours_anyangle"))
                R.put(key, "ours_anyangle", {"seconds": sec, "peak_gb": peak, **_fid(V, o, rd, key)})
            except Exception:  # noqa: BLE001
                err(f"angles anyangle {key}")
        V.set_anyangle(pipe, False)
    except Exception:  # noqa: BLE001
        err("anyangle load")
    del pipe, E
    free()


def _fid(V, o: np.ndarray, rd: Path, key: str) -> dict:
    r = np.asarray(rgb(rd / f"{key}.png"))
    valid = np.asarray(Image.open(rd / f"{key}_valid.png")) > 0
    o = np.asarray(Image.fromarray(np.asarray(o)[..., :3]).resize((r.shape[1], r.shape[0])))
    return {k: v for k, v in V.warp_fidelity(o, r, valid).items() if isinstance(v, (int, float))}


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a) > 0, np.asarray(b) > 0
    return round(float((a & b).sum() / max((a | b).sum(), 1)), 4)


# ----------------------------------------------------------------------------- stage: FLUX.2 klein 4B


def stage_klein():
    import torch
    from diffusers import Flux2KleinPipeline

    from benchmarks import v9_cases as C

    m = manifest()
    pipe, sec_load, _ = timed(lambda: Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B",
                                                                          torch_dtype=torch.bfloat16).to("cuda"))
    print(f"[klein] loaded in {sec_load}s", flush=True)

    def run(prompt, image=None, w=1024, h=1024):
        kw = dict(prompt=prompt, height=h, width=w, num_inference_steps=4, guidance_scale=1.0,
                  generator=torch.Generator("cuda").manual_seed(SEED))
        if image is not None:
            kw["image"] = image
        return pipe(**kw).images[0]

    run("warm-up")                                    # first call compiles kernels; not timed
    R = Results("t2i")
    for t in m["t2i"]:
        if R.has(t["id"], "klein"):
            continue
        try:
            o, sec, peak = timed(lambda: run(t["enhanced"], w=t["w"], h=t["h"]))
            o.save(R.img(t["id"], "klein"))
            R.put(t["id"], "klein", {"seconds": sec, "peak_gb": peak})
        except Exception:  # noqa: BLE001
            err(f"t2i klein {t['id']}")
    R = Results("product")
    for pr in m["product"]:
        img = rgb(IN / "product" / f"{pr['name']}.png")
        w, h = fit_size(*img.size)
        for sk in C.PRODUCT_SCENES:
            case = f"{pr['name']}@{sk}"
            if R.has(case, "klein"):
                continue
            try:
                o, sec, peak = timed(lambda: run(C.product_instruction(pr["product"], sk), img, w, h))
                o.resize(img.size, Image.LANCZOS).save(R.img(case, "klein"))
                R.put(case, "klein", {"seconds": sec, "peak_gb": peak})
            except Exception:  # noqa: BLE001
                err(f"product klein {case}")
    R = Results("gedit")
    for g in m["gedit"]:
        if R.has(g["key"], "klein"):
            continue
        try:
            img = rgb(IN / "gedit" / f"{g['key']}.png")
            w, h = fit_size(*img.size)
            o, sec, peak = timed(lambda: run(g["instruction"], img, w, h))
            o.resize(img.size, Image.LANCZOS).save(R.img(g["key"], "klein"))
            R.put(g["key"], "klein", {"seconds": sec, "peak_gb": peak})
        except Exception:  # noqa: BLE001
            err(f"gedit klein {g['key']}")
    del pipe
    free()


def _t2i_stage(method: str, load, call):
    m = manifest()
    pipe, sec_load, _ = timed(load)
    print(f"[{method}] loaded in {sec_load}s", flush=True)
    call(pipe, "warm-up", 512, 512)
    R = Results("t2i")
    for t in m["t2i"]:
        if R.has(t["id"], method):
            continue
        try:
            o, sec, peak = timed(lambda: call(pipe, t["enhanced"], t["w"], t["h"]))
            o.save(R.img(t["id"], method))
            R.put(t["id"], method, {"seconds": sec, "peak_gb": peak})
        except Exception:  # noqa: BLE001
            err(f"t2i {method} {t['id']}")
    del pipe
    free()


def stage_zimage():
    import torch
    from diffusers import ZImagePipeline

    def load():
        p = ZImagePipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", torch_dtype=torch.bfloat16, low_cpu_mem_usage=False)
        return p.to("cuda")

    def call(p, prompt, w, h):
        return p(prompt=prompt, height=h, width=w, num_inference_steps=9, guidance_scale=0.0,
                 generator=torch.Generator("cuda").manual_seed(SEED)).images[0]
    _t2i_stage("zimage", load, call)


def _nf4(components):
    import torch
    from diffusers.quantizers import PipelineQuantizationConfig

    return PipelineQuantizationConfig(quant_backend="bitsandbytes_4bit",
                                      quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                                                    "bnb_4bit_compute_dtype": torch.bfloat16},
                                      components_to_quantize=components)


def stage_fluxdev():
    import torch
    from diffusers import FluxPipeline

    def load():
        p = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16,
                                         quantization_config=_nf4(["transformer"]))
        p.enable_model_cpu_offload()
        return p

    def call(p, prompt, w, h):
        return p(prompt=prompt, height=h, width=w, num_inference_steps=28, guidance_scale=3.5,
                 generator=torch.Generator("cpu").manual_seed(SEED)).images[0]
    _t2i_stage("fluxdev", load, call)


def stage_ip2p():
    import torch
    from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionInstructPix2PixPipeline

    m = manifest()
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=torch.float16,
                                                                  safety_checker=None).to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    R = Results("gedit")
    for g in m["gedit"]:
        if R.has(g["key"], "ip2p"):
            continue
        try:
            img = rgb(IN / "gedit" / f"{g['key']}.png")
            w, h = fit_size(*img.size, area=512 * 512, mult=8)
            o, sec, peak = timed(lambda: pipe(g["instruction"], image=img.resize((w, h)), num_inference_steps=20,
                                              image_guidance_scale=1.5, guidance_scale=7.5,
                                              generator=torch.Generator("cuda").manual_seed(SEED)).images[0])
            o.resize(img.size, Image.LANCZOS).save(R.img(g["key"], "ip2p"))
            R.put(g["key"], "ip2p", {"seconds": sec, "peak_gb": peak})
        except Exception:  # noqa: BLE001
            err(f"gedit ip2p {g['key']}")
    del pipe
    free()


ANGLE_PROMPTS = {"front-left": "<sks> front-left quarter view eye-level shot medium shot",
                 "front-right": "<sks> front-right quarter view eye-level shot medium shot"}


def stage_q2511():
    """Qwen-Image-Edit-2511 + fal Multiple-Angles LoRA, camera angles only.
    v1/v2 (nf4, with and without CPU offload) produced speckle noise on every image -> the 4-bit weights break this
    model. v3: full bf16 with model CPU offload (peak ~40 GB, far over our 16 GB budget - reported as such).
    General edits with 2511 are dropped: 107 s per image even in nf4 rules it out for the product anyway."""
    import torch
    from diffusers import QwenImageEditPlusPipeline

    m = manifest()
    pipe = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2511", torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    V3 = {"v": 3}

    def run(img, prompt):
        o = pipe(image=[img], prompt=prompt, negative_prompt=" ", true_cfg_scale=4.0,
                 num_inference_steps=40, generator=torch.Generator("cpu").manual_seed(SEED)).images[0]
        return o.convert("RGB").resize(img.size, Image.LANCZOS)

    from editing import views3d as V

    pipe.load_lora_weights("fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA",
                           weight_name="qwen-image-edit-2511-multiple-angles-lora.safetensors", adapter_name="angles")
    pipe.set_adapters(["angles"], [0.9])
    R = Results("angles")
    rd = R.dir / "render"
    # the LoRA's "front-left/right" may mean either side relative to our yaw sign: generate both once per photo,
    # the score stage picks the one mapping (for the whole set) that matches the 3-D re-projection better
    for name in m["angles"]:
        img = rgb(IN / "angles" / f"{name}.png")
        for side, prompt in ANGLE_PROMPTS.items():
            key = f"{name}#{side}"
            if R.d.get(key, {}).get("q2511_angles", {}).get("v") == 3:
                continue
            try:
                o, sec, peak = timed(lambda: run(img, prompt))
                o.save(R.img(key, "q2511_angles"))
                row = {"seconds": sec, "peak_gb": peak, **V3}
                for y in YAWS:
                    if (rd / f"{name}@{y:+.0f}.png").exists():
                        row[f"fid@{y:+.0f}"] = _fid(V, np.asarray(o), rd, f"{name}@{y:+.0f}")
                R.put(key, "q2511_angles", row)
            except Exception:  # noqa: BLE001
                err(f"angles q2511 {key}")
    del pipe
    free()


def _valid(row: dict, meth: str) -> bool:
    """Results of the broken 2511 runs (v1/v2, speckle noise) never enter judging or the report."""
    return not meth.startswith("q2511") or (row or {}).get("v") == 3


def stage_guards():
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, pipeline

    m = manifest()
    R = Results("guard")
    # deberta prompt-injection classifier
    try:
        clf, sec_load, _ = timed(lambda: pipeline("text-classification", model="protectai/deberta-v3-base-prompt-injection-v2",
                                                  truncation=True, max_length=512, device=0))
        clf("warm-up")
        for c in m["guard"]:
            if R.has(c["id"], "deberta"):
                continue
            r, sec, peak = timed(lambda: clf(c["text"])[0])
            R.put(c["id"], "deberta", {"pred": "block" if r["label"].upper().startswith("INJ") else "allow",
                                       "score": round(float(r["score"]), 3), "seconds": sec, "extra_gb": peak})
        del clf
        free()
    except Exception:  # noqa: BLE001
        err("deberta")
    # SingGuard-4b (Qwen3-VL based guard; text policy check)
    model = proc = None
    for mid in ("inclusionAI/SingGuard-4b", "inclusionAI/Sing-Guard-4b"):
        try:
            proc = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
            model = AutoModelForImageTextToText.from_pretrained(mid, torch_dtype=torch.bfloat16, device_map="cuda",
                                                                trust_remote_code=True).eval()
            print("[guard] SingGuard loaded:", mid, flush=True)
            break
        except Exception:  # noqa: BLE001
            err(f"SingGuard load {mid}")
    if model is None:
        return

    def judge(text):
        msgs = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                       return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=512, do_sample=False)
        return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    judge("hello")
    for c in m["guard"]:
        if R.has(c["id"], "singguard"):
            continue
        try:
            raw, sec, peak = timed(lambda: judge(c["text"]))
            R.put(c["id"], "singguard", {"pred": parse_singguard(raw), "raw": raw[-160:], "seconds": sec, "extra_gb": peak})
        except Exception:  # noqa: BLE001
            err(f"singguard {c['id']}")
    del model
    free()


def stage_segformer():
    """Segformer-B5 (ADE20k) person mask -> remove with our LaMa: the teammate's #16 pipeline end to end."""
    import torch
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    from editing.common import lama_inpaint
    from editing.masking import dilate
    from benchmarks.data import load

    m = manifest()
    mid = "nvidia/segformer-b5-finetuned-ade-640-640"
    proc = SegformerImageProcessor.from_pretrained(mid)
    model = SegformerForSemanticSegmentation.from_pretrained(mid).to("cuda").eval()
    person = model.config.label2id.get("person", 12)
    R = Results("seg")
    RR = Results("removal")
    for s in load(50, 768):
        if s.name not in m["removal"] or R.has(s.name, "segformer"):
            continue
        try:
            img = s.image.convert("RGB")

            def seg():
                x = proc(images=img, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    lg = model(**x).logits
                lg = torch.nn.functional.interpolate(lg, size=img.size[::-1], mode="bilinear", align_corners=False)
                return (lg.argmax(1)[0] == person).cpu().numpy().astype(np.uint8)
            mk, sec, peak = timed(seg)
            Image.fromarray(mk * 255).save(R.img(s.name, "segformer"))
            R.put(s.name, "segformer", {"iou": iou(mk, s.mask), "seconds": sec, "peak_gb": peak})
            hole = dilate(mk, 16) if mk.sum() else mk
            o, sec2, _ = timed(lambda: lama_inpaint(np.asarray(img), hole) if hole.sum() else np.asarray(img))
            Image.fromarray(o).save(RR.img(s.name, "segformer_lama"))
            RR.put(s.name, "segformer_lama", {"seconds": round(sec + sec2, 2)})
        except Exception:  # noqa: BLE001
            err(f"segformer {s.name}")
    del model
    free()


# ----------------------------------------------------------------------------- score


def part_score():
    from benchmarks import metrics as M

    m = manifest()
    root = _root()
    S = {}

    # pose: Leffa vs ours (v7 gist+precise+loop)
    try:
        from editing import pose as P
        from editing import qwen as Q

        pipe = Q.load_pipe("nf4")
        Q.enable_memory_savers(pipe)
        lres = jload(OUT / "leffa" / "results.json")
        v7 = jload(root / "outputs" / "eval" / "v7_pose" / "results.json")
        rows = {}
        for key in m["pose"]:
            src, tgt = rgb(IN / "pose" / f"{key}_src.png"), rgb(IN / "pose" / f"{key}_tgt.png")
            tkp = P.keypoints(tgt)
            r = {"ours": {k: v7.get(key, {}).get("gist_loop", {}).get(k) for k in ("pck", "same_person", "seconds")}}
            lp = OUT / "leffa" / f"{key}.png"
            if lp.exists() and tkp is not None:
                o = rgb(lp)
                okp = P.keypoints(o)
                same = Q.ask(pipe, [src, o], "Are image 1 and image 2 the same person wearing the same clothes? "
                                            "Answer yes or no.", max_new_tokens=3).strip().lower().startswith("yes")
                r["leffa"] = {"pck": P.pck(okp, tkp) if okp is not None else 0.0, "same_person": same,
                              "seconds": lres.get(key, {}).get("seconds"), "peak_gb": lres.get(key, {}).get("peak_gb")}
            rows[key] = r
        S["pose"] = rows

        # product: DINOv2 similarity of the product crop (input mask box vs Qwen3-VL box in the result)
        from editing import recolour as RC

        dino = _dino()
        R = Results("product")
        for case, ms in R.d.items():
            name, sk = case.split("@")
            pr = next(p for p in m["product"] if p["name"] == name)
            src = rgb(IN / "product" / f"{name}.png")
            objp = IN / "product" / f"{name}_obj.png"
            mk = (np.asarray(Image.open(objp)) > 0) if objp.exists() else np.ones(src.size[::-1], bool)
            ref = _crop_masked(src, mk)
            for meth in list(ms):
                p = R.img(case, meth)
                if not p.exists() or "dino" in ms[meth]:
                    continue
                o = rgb(p)
                box = RC.ground_box(lambda q: Q.ask(pipe, [o], q, max_new_tokens=60), o.size, f"the {pr['product']}")
                crop = o.crop(box) if box else o
                ms[meth]["dino"] = round(dino(ref, crop), 4)
                ms[meth]["found"] = box is not None
            jsave(R.p, R.d)
        del pipe
        free()
    except Exception:  # noqa: BLE001
        err("score pose/product")

    # 3-D: Hunyuan3D-2 (ours) vs SF3D
    try:
        S["3d"] = score_3d(m)
    except Exception:  # noqa: BLE001
        err("score 3d")

    # upscaling: bicubic, ours (Lanczos + sharpen = current web API), ours_qwen (AI refine), TADSR
    try:
        from server.web_api import upscale

        rows = {}
        tres = jload(OUT / "tadsr" / "results.json")
        qres = Results("sr").d
        for n in m["sr"]:
            hr = np.asarray(rgb(IN / "sr" / f"{n}_hr.png"))
            lr = rgb(IN / "sr" / f"{n}_lr.png")
            outs = {"bicubic": np.asarray(lr.resize((512, 512), Image.BICUBIC)),
                    "ours_lanczos": upscale(lr, 4, True)}
            for meth, p in (("ours_qwen", Results("sr").img(n, "ours_qwen")), ("tadsr", OUT / "tadsr" / f"{n}.png")):
                if Path(p).exists():
                    outs[meth] = np.asarray(rgb(p).resize((512, 512), Image.LANCZOS))
            r = {}
            for meth, o in outs.items():
                r[meth] = {"psnr": round(M.psnr(o, hr), 2), "ssim": round(M.ssim(o, hr), 4),
                           "lpips": round(M.lpips_dist(o, hr), 4),
                           "seconds": (qres.get(n, {}).get("ours_qwen", {}).get("seconds") if meth == "ours_qwen" else
                                       tres.get(n, {}).get("seconds") if meth == "tadsr" else None),
                           "peak_gb": tres.get(n, {}).get("peak_gb") if meth == "tadsr" else
                           qres.get(n, {}).get("ours_qwen", {}).get("peak_gb") if meth == "ours_qwen" else 0.0}
            rows[n] = r
        S["sr"] = rows
    except Exception:  # noqa: BLE001
        err("score sr")

    # removal: same holes, same metrics as v5 (LaMa + Smart from v5, MAT/LaMa from IOPaint, Segformer+LaMa)
    try:
        from benchmarks.data import load

        v5 = root / "outputs" / "eval" / "v5_remove"
        mres = jload(OUT / "mat" / "results.json")
        rows = {}
        for s in load(50, 768):
            if s.name not in m["removal"]:
                continue
            G = np.asarray(s.gt.convert("RGB"))
            region = np.asarray(Image.open(v5 / "LaMa" / f"{s.name}_region.png")) > 0
            r = {}
            for meth, p in (("lama_v5", v5 / "LaMa" / f"{s.name}.png"), ("ours_smart", v5 / "Smart" / f"{s.name}.png"),
                            ("mat", OUT / "mat" / f"{s.name}_mat.png"), ("lama_iopaint", OUT / "mat" / f"{s.name}_lama.png"),
                            ("segformer_lama", Results("removal").img(s.name, "segformer_lama"))):
                if not Path(p).exists():
                    continue
                R_ = np.asarray(rgb(p).resize((G.shape[1], G.shape[0])))
                r[meth] = {"lpips_in": round(M.lpips_crop(R_, G, region), 4), "psnr_in": round(M.psnr(R_, G, region), 2),
                           "ssim_in": round(M.ssim(R_, G, region), 4), "seam": round(M.seam(R_, G, region), 2)}
                if meth in ("mat", "lama_iopaint"):
                    r[meth].update({k: mres.get(s.name, {}).get(meth, {}).get(k) for k in ("seconds", "peak_gb")})
            rows[s.name] = r
        S["removal"] = rows
    except Exception:  # noqa: BLE001
        err("score removal")
    jsave(OUT / "scores.json", S)


def _dino():
    import torch
    from transformers import AutoImageProcessor, AutoModel

    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to("cuda").eval()

    def emb(im):
        with torch.no_grad():
            x = proc(images=im, return_tensors="pt").to("cuda")
            return torch.nn.functional.normalize(model(**x).last_hidden_state[:, 0], dim=-1)[0]
    return lambda a, b: float((emb(a) * emb(b)).sum())


def _crop_masked(img: Image.Image, mk: np.ndarray) -> Image.Image:
    ys, xs = np.nonzero(mk)
    if len(xs) == 0:
        return img
    a = np.asarray(img).copy()
    a[~mk] = 255
    return Image.fromarray(a).crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


def mesh_stats(path: Path) -> dict:
    """faces, watertight, holes (boundary edges after welding UV seams) - the 'gaps' the user saw."""
    import trimesh

    mesh = trimesh.load(path, force="mesh", process=False)
    m = mesh.copy()
    m.merge_vertices(merge_tex=True, merge_norm=True)
    edges = m.edges_sorted
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    return {"faces": int(len(mesh.faces)), "watertight": bool(m.is_watertight), "boundary_edges": int((counts == 1).sum()),
            "textured": bool(getattr(mesh.visual, "kind", None) == "texture")}


def score_3d(m) -> dict:
    import trimesh

    from editing.object3d import turntable
    from benchmarks.run_v5_object3d import silhouette_iou

    sres = jload(OUT / "sf3d" / "results.json")
    (OUT / "3d").mkdir(parents=True, exist_ok=True)
    rows = {}
    for o in m["3d"]:
        name = o["name"]
        cut = Image.open(IN / "3d" / f"{name}.png").convert("RGBA")
        # Hunyuan re-run with today's code (modal_v9_extra BENCH=hunyuan) if present, else the v5/v8 model
        new = OUT / "hunyuan" / f"{name}.glb"
        hres = jload(OUT / "hunyuan" / "results.json").get(name) if new.exists() else \
            jload(Path(o["hunyuan_results"])).get(o["stem"], {})
        hglb = new if new.exists() else Path(o["hunyuan_glb"])
        r = {}
        for meth, p in (("hunyuan", hglb), ("sf3d", OUT / "sf3d" / f"{name}.glb")):
            if not p.exists():
                continue
            try:
                mesh = trimesh.load(p, force="mesh")
                views = turntable(mesh, views=8, size=320)
                ious = [silhouette_iou(v, cut) for v in views]
                k = int(np.argmax(ious))
                strip([(f"{meth} {i * 45}deg", views[(k + i * 2) % 8]) for i in range(4)], 240).save(OUT / "3d" / f"{name}__{meth}.jpg")
                info = hres if meth == "hunyuan" else sres.get(name, {})
                secs = info.get("seconds") or info.get("total_s")
                r[meth] = {"sil_iou": round(max(ious), 4), **mesh_stats(p), "seconds": secs,
                           "peak_gb": max(info.get("shape_peak_gb", 0) or 0, info.get("texture_peak_gb", 0) or 0)
                           if meth == "hunyuan" else info.get("peak_gb")}
            except Exception:  # noqa: BLE001
                err(f"3d {meth} {name}")
        rows[name] = r
    return rows


# ----------------------------------------------------------------------------- judge


GEMINI = "https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"


class Judge:
    """Gemini 2.5 Flash as an independent judge (not our model family). One call per case with every method's image
    (shuffled letters). Falls back to Qwen3-VL per image if Gemini is unavailable (then marked biased)."""

    def __init__(self):
        self.key = os.environ.get("GEMINI_API_KEY", "").strip()
        # gemini-2.5-flash is closed to new keys (404, Oct 2026); the API's own error names the current model
        self.models = [m for m in (os.environ.get("GEMINI_JUDGE_MODEL"), "gemini-3.8-flash", "gemini-flash-latest",
                                   "gemini-2.5-flash") if m]
        self.model = self.models[0]
        self.last = 0.0
        self.gemini_ok = bool(self.key)
        self.qwen = None

    @staticmethod
    def _jpeg(im: Image.Image, side: int = 768) -> str:
        im = im.convert("RGB")
        s = min(1.0, side / max(im.size))
        if s < 1:
            im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        b = io.BytesIO()
        im.save(b, "JPEG", quality=88)
        return base64.b64encode(b.getvalue()).decode()

    def gemini(self, parts: list) -> dict:
        import urllib.error
        import urllib.request

        body = {"contents": [{"role": "user", "parts": [p if isinstance(p, dict) else
                                                        {"inline_data": {"mime_type": "image/jpeg", "data": self._jpeg(p)}}
                                                        for p in parts]}],
                "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}}
        for attempt in range(6):
            wait = 7.0 - (time.time() - self.last)          # <= ~9 requests a minute (free tier is 10)
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()
            req = urllib.request.Request(GEMINI.format(m=self.model), data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "x-goog-api-key": self.key})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    j = json.loads(r.read().decode())
                text = "".join(p.get("text", "") for p in j["candidates"][0]["content"]["parts"])
                return json.loads(re.search(r"\{.*\}", text, re.S).group(0))
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:300]
                if e.code in (429, 503) and "quota" not in detail.lower() and attempt < 5:   # busy: wait; quota: give up
                    time.sleep(30 * (attempt + 1))
                    continue
                print(f"[judge] Gemini {e.code} ({self.model}): {detail}", flush=True)
                if e.code == 404 and self.model != self.models[-1]:          # model retired: next name
                    self.model = self.models[self.models.index(self.model) + 1]
                    continue
                if e.code in (400, 401, 403, 404, 429):
                    self.gemini_ok = False
                raise
        self.gemini_ok = False             # stop asking for the rest of the run
        raise RuntimeError("Gemini kept refusing")

    def score(self, intro: str, ref_images: list, cands: dict[str, Image.Image], rubric: dict[str, str], seed: str) -> dict:
        """-> {method: {criterion: 0-10}, '_by': 'gemini'|'qwen3-vl'}."""
        names = list(cands)
        random.Random(seed).shuffle(names)
        letters = {chr(65 + i): n for i, n in enumerate(names)}
        crit = "; ".join(f'"{k}": {v}' for k, v in rubric.items())
        if self.gemini_ok:
            try:
                parts = [{"text": intro}] + [x for im in ref_images for x in ({"text": "Reference image:"}, im)]
                for L, n in letters.items():
                    parts += [{"text": f"Candidate {L}:"}, cands[n]]
                parts.append({"text": f"Score EVERY candidate from 0 (worst) to 10 (best) on: {crit}. Judge each "
                                      f"candidate on its own merits; ties are allowed. Reply with JSON only: "
                                      '{"A": {"<criterion>": <int>, ...}, "B": {...}, ...}'})
                j = self.gemini(parts)
                return {letters[L]: {k: _int(v.get(k)) for k in rubric} for L, v in j.items() if L in letters} | {"_by": "gemini"}
            except Exception:  # noqa: BLE001
                err("gemini judge")
        return self._qwen_score(intro, ref_images, cands, rubric) | {"_by": "qwen3-vl"}

    def _qwen_score(self, intro, refs, cands, rubric) -> dict:
        from editing import qwen as Q

        if self.qwen is None:
            pipe = Q.load_pipe("nf4")
            Q.enable_memory_savers(pipe)
            self.qwen = (pipe, Q)
        pipe, Q = self.qwen
        out = {}
        crit = ", ".join(f'"{k}" ({v})' for k, v in rubric.items())
        for n, im in cands.items():
            t = Q.ask(pipe, list(refs) + [im], f"{intro} The LAST image is the candidate. Score it 0-10 on: {crit}. "
                                              "Reply with JSON only.", max_new_tokens=80)
            try:
                j = json.loads(re.search(r"\{.*\}", t, re.S).group(0))
            except Exception:  # noqa: BLE001
                j = {}
            out[n] = {k: _int(j.get(k)) for k in rubric}
        return out


def _int(v):
    try:
        return max(0, min(10, int(round(float(v)))))
    except Exception:  # noqa: BLE001
        return None


def part_judge():
    m = manifest()
    J = Judge()
    jp = OUT / "judge.json"
    JD = jload(jp)

    def done(task, case):      # only Gemini verdicts count; Qwen3-VL fallbacks are judged again
        return JD.get(task, {}).get(case, {}).get("_by") == "gemini"

    def put(task, case, val):
        JD.setdefault(task, {})[case] = val
        jsave(jp, JD)

    # GEdit (VIEScore-style: SC = min(follow, preserve), PQ = quality, O = sqrt(SC*PQ))
    R = Results("gedit")
    for g in m["gedit"]:
        if done("gedit", g["key"]):
            continue
        c = {meth: rgb(R.img(g["key"], meth)) for meth, row in R.d.get(g["key"], {}).items()
             if R.img(g["key"], meth).exists() and _valid(row, meth)}
        if len(c) < 2:
            continue
        try:
            put("gedit", g["key"], J.score(
                f'Image-editing benchmark. The reference image was edited with the instruction: "{g["instruction"]}".',
                [rgb(IN / "gedit" / f"{g['key']}.png")], c,
                {"follow": "the instruction was carried out exactly", "preserve": "everything the instruction did not ask "
                 "to change is unchanged (no over-editing)", "quality": "natural, realistic, no artifacts or distortions"},
                g["key"]))
        except Exception:  # noqa: BLE001
            err(f"judge gedit {g['key']}")

    R = Results("t2i")
    for t in m["t2i"]:
        if done("t2i", t["id"]):
            continue
        c = {meth: rgb(R.img(t["id"], meth)) for meth in R.d.get(t["id"], {}) if R.img(t["id"], meth).exists()}
        for k in ("ours_raw", "ours_enhanced"):
            if Path(t[k]).exists():
                c[k] = rgb(t[k])
        try:
            put("t2i", t["id"], J.score(f'Text-to-image benchmark. The request was: "{t["prompt"]}".', [], c,
                                        {"prompt": "shows exactly what was requested (count, text, attributes)",
                                         "quality": "aesthetic, realistic or well-made, no artifacts"}, t["id"]))
        except Exception:  # noqa: BLE001
            err(f"judge t2i {t['id']}")

    R = Results("product")
    from benchmarks import v9_cases as C

    for case in sorted(R.d):
        if done("product", case):
            continue
        name, sk = case.split("@")
        pr = next(p for p in m["product"] if p["name"] == name)
        c = {meth: rgb(R.img(case, meth)) for meth in R.d[case] if R.img(case, meth).exists()}
        try:
            put("product", case, J.score(
                f"Product-photo benchmark. The reference shows a {pr['product']}. Request: "
                f'"{C.product_instruction(pr["product"], sk)}"', [rgb(IN / "product" / f"{name}.png")], c,
                {"product_same": f"the {pr['product']} is identical to the reference (shape, colour, logos, details)",
                 "scene": "the requested setting and lighting are there", "quality": "looks like a real professional photo"},
                case))
        except Exception:  # noqa: BLE001
            err(f"judge product {case}")

    # angles: ours (anyangle, text) vs 2511 (with the side mapping chosen in summarise_angles)
    R = Results("angles")
    side_for = angle_side_mapping(R.d)
    for name in m["angles"]:
        for y in YAWS:
            key = f"{name}@{y:+.0f}"
            if done("angles", key):
                continue
            c = {meth: rgb(R.img(key, meth)) for meth in ("ours_anyangle", "ours_text") if R.img(key, meth).exists()}
            q = R.img(f"{name}#{side_for[y]}", "q2511_angles")
            if q.exists() and _valid(R.d.get(f"{name}#{side_for[y]}", {}).get("q2511_angles"), "q2511_angles"):
                c["q2511_angles"] = rgb(q)
            if len(c) < 2:
                continue
            try:
                put("angles", key, J.score(
                    f"New-camera-angle benchmark. Each candidate should show the SAME scene as the reference, photographed "
                    f"from a camera that moved about 45 degrees to the {'right' if y > 0 else 'left'} around the main subject.",
                    [rgb(IN / "angles" / f"{name}.png")], c,
                    {"angle": "the camera angle clearly changed as requested", "same_scene": "same objects, people and "
                     "details as the reference", "quality": "realistic, no distortions"}, key))
            except Exception:  # noqa: BLE001
                err(f"judge angles {key}")

    # 3-D: four turntable views per method vs the cut-out photo
    for o in m["3d"]:
        name = o["name"]
        if done("3d", name):
            continue
        c = {meth: Image.open(OUT / "3d" / f"{name}__{meth}.jpg") for meth in ("hunyuan", "sf3d")
             if (OUT / "3d" / f"{name}__{meth}.jpg").exists()}
        if len(c) < 2:
            continue
        try:
            put("3d", name, J.score("3-D reconstruction benchmark. Each candidate shows four views (0, 90, 180, 270 "
                                    "degrees) of a 3-D model made from the reference photo.", [rgb(IN / "3d" / f"{name}.png")], c,
                                    {"shape": "the shape matches the object in the photo", "complete": "no holes, gaps or "
                                     "missing parts", "texture": "colours and details match the photo"}, name))
        except Exception:  # noqa: BLE001
            err(f"judge 3d {name}")
    print(f"[judge] done, Gemini used: {J.gemini_ok}", flush=True)


def angle_side_mapping(d: dict) -> dict:
    """Which LoRA side ('front-left'/'front-right') corresponds to yaw -45 / +45: the mapping with the lower total
    re-projection colour error over all photos."""
    tot = {"a": [], "b": []}       # a: -45 -> front-left, +45 -> front-right; b: swapped
    for key, ms in d.items():
        if "#" not in key or "q2511_angles" not in ms or ms["q2511_angles"].get("v") != 3:
            continue
        side = key.split("#")[1]
        row = ms["q2511_angles"]
        for y, k in ((-45.0, "fid@-45"), (45.0, "fid@+45")):
            e = (row.get(k) or {}).get("geo_dE")
            if e is None:
                continue
            a_match = (y < 0 and side == "front-left") or (y > 0 and side == "front-right")
            tot["a" if a_match else "b"].append(e)
    a = np.mean(tot["a"]) if tot["a"] else 1e9
    b = np.mean(tot["b"]) if tot["b"] else 1e9
    return {-45.0: "front-left", 45.0: "front-right"} if a <= b else {-45.0: "front-right", 45.0: "front-left"}


# ----------------------------------------------------------------------------- report


def _mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def _f(x, fmt="{:.2f}"):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else fmt.format(x)


def part_report():
    m = manifest()
    S = jload(OUT / "scores.json")
    JD = jload(OUT / "judge.json")
    L = ["# v9: our pipeline vs the teammates' model list", "",
         f"Run {time.strftime('%Y-%m-%d %H:%M')}. Times = seconds per item on an L40S after warm-up; peak = GPU memory "
         "allocated (models offloaded to CPU when possible). Judge = Gemini 2.5 Flash unless marked.", ""]

    def judge_table(task, methods, crits, extra=None):
        rows = JD.get(task, {})
        by = {r.get("_by") for r in rows.values()}
        L.append(f"Judge: {', '.join(sorted(b for b in by if b)) or 'none'} on {len(rows)} cases.")
        hdr = "| method | " + " | ".join(crits) + (" | " + " | ".join(extra[0]) if extra else "") + " |"
        L.extend([hdr, "|" + "---|" * (1 + len(crits) + (len(extra[0]) if extra else 0))])
        for meth in methods:
            vals = [_mean([r.get(meth, {}).get(c) for r in rows.values() if isinstance(r.get(meth), dict)]) for c in crits]
            ex = extra[1](meth) if extra else []
            L.append(f"| {meth} | " + " | ".join(_f(v, "{:.1f}") for v in vals) + (" | " + " | ".join(ex) if ex else "") + " |")
        L.append("")

    def res_stat(task, meth, key):
        d = Results(task).d
        return _mean([v.get(meth, {}).get(key) for v in d.values()])

    # 1 guardrails
    G = Results("guard").d
    L += ["## 1. Guardrails (60 requests: 30 normal edits, 20 harmful, 10 prompt injections)", "",
          "| model | normal edits wrongly blocked | harmful blocked | injections blocked | s / request | extra GPU GB |",
          "|---|---|---|---|---|---|"]
    for meth in ("ours_qwen3vl", "singguard", "deberta"):
        rs = [(c, G.get(c["id"], {}).get(meth)) for c in m["guard"]]
        rs = [(c, r) for c, r in rs if r]
        if not rs:
            continue
        if meth == "singguard":        # re-read the stored answer (v9 first run parsed category answers as "allow")
            rs = [(c, {**r, "pred": parse_singguard(r.get("raw", ""))}) for c, r in rs]
        fb = sum(r["pred"] == "block" for c, r in rs if c["kind"] == "benign")
        hb = sum(r["pred"] == "block" for c, r in rs if c["kind"] == "harmful")
        ib = sum(r["pred"] == "block" for c, r in rs if c["kind"] == "injection")
        nb = sum(c["kind"] == "benign" for c, _ in rs)
        nh = sum(c["kind"] == "harmful" for c, _ in rs)
        ni = sum(c["kind"] == "injection" for c, _ in rs)
        L.append(f"| {meth} | {fb}/{nb} | {hb}/{nh} | {ib}/{ni} | {_f(_mean([r['seconds'] for _, r in rs]))} | "
                 f"{_f(max(r.get('extra_gb', 0) or 0 for _, r in rs), '{:.1f}')} |")
    L.append("")

    # 2 pose
    P = S.get("pose", {})
    L += ["## 2. Pose change: Leffa vs ours (10 RORD pairs, PCK@0.2, Qwen3-VL same-person check)", "",
          "| method | PCK ↑ | same person | s | peak GB |", "|---|---|---|---|---|"]
    for meth in ("ours", "leffa"):
        rs = [v[meth] for v in P.values() if meth in v and v[meth].get("pck") is not None]
        if rs:
            L.append(f"| {meth} | {_mean([r['pck'] for r in rs]):.2f} | {sum(bool(r.get('same_person')) for r in rs)}/{len(rs)} | "
                     f"{_f(_mean([r.get('seconds') for r in rs]), '{:.1f}')} | {_f(_mean([r.get('peak_gb') for r in rs]), '{:.1f}')} |")
    L.append("")

    # 3 3-D
    T = S.get("3d", {})
    L += ["## 3. Object to 3-D: Hunyuan3D-2 (ours) vs SF3D", "",
          "| method | silhouette IoU ↑ | watertight | boundary edges (holes) ↓ | faces | s | peak GB |", "|---|---|---|---|---|---|---|"]
    for meth in ("hunyuan", "sf3d"):
        rs = [v[meth] for v in T.values() if meth in v]
        if rs:
            L.append(f"| {meth} | {_mean([r['sil_iou'] for r in rs]):.3f} | {sum(r['watertight'] for r in rs)}/{len(rs)} | "
                     f"{_mean([r['boundary_edges'] for r in rs]):.0f} | {_mean([r['faces'] for r in rs]):.0f} | "
                     f"{_f(_mean([r.get('seconds') for r in rs]), '{:.1f}')} | {_f(_mean([r.get('peak_gb') for r in rs]), '{:.1f}')} |")
    L.append("")
    judge_table("3d", ["hunyuan", "sf3d"], ["shape", "complete", "texture"])

    # 4/8/9 text-to-image
    L += ["## 4. Text-to-image (20 prompts; every model gets the same Qwen-enhanced prompt; ours_raw = plain prompt)", ""]
    judge_table("t2i", ["ours_enhanced", "ours_raw", "klein", "zimage", "fluxdev"], ["prompt", "quality"],
                (["s", "peak GB"], lambda meth: [_f(res_stat("t2i", meth, "seconds"), "{:.1f}"),
                                                 _f(res_stat("t2i", meth, "peak_gb"), "{:.1f}")]))

    # 4b product shoot
    L += ["## 5. Product shoot (8 products x 3 scenes): ours (Qwen-Image 2.1) vs FLUX.2 klein 4B", ""]
    judge_table("product", ["ours", "klein"], ["product_same", "scene", "quality"],
                (["DINOv2 product sim ↑", "s", "peak GB"], lambda meth: [
                    _f(res_stat("product", meth, "dino"), "{:.3f}"), _f(res_stat("product", meth, "seconds"), "{:.1f}"),
                    _f(res_stat("product", meth, "peak_gb"), "{:.1f}")]))

    # 5 upscaling
    SR = S.get("sr", {})
    L += ["## 6. Upscaling x4 (16 photos, 128 -> 512 px, real-world-like degradation)", "",
          "| method | LPIPS ↓ | PSNR | SSIM | s | peak GB |", "|---|---|---|---|---|---|"]
    for meth in ("bicubic", "ours_lanczos", "ours_qwen", "tadsr"):
        rs = [v[meth] for v in SR.values() if meth in v]
        if rs:
            sec = _mean([r.get('seconds') for r in rs])
            L.append(f"| {meth} | {_mean([r['lpips'] for r in rs]):.3f} | {_mean([r['psnr'] for r in rs]):.2f} | "
                     f"{_mean([r['ssim'] for r in rs]):.3f} | {'<0.5' if sec == sec and sec < 0.5 else _f(sec, '{:.2f}')} | "
                     f"{_f(_mean([r.get('peak_gb') for r in rs]), '{:.1f}')} |")
    L.append("")

    # 6 watermark
    W = Results("watermark").d
    atk = ["none", "jpeg90", "jpeg75", "jpeg50", "resize50%", "crop10%", "brighter", "ai_edit"]
    L += ["## 7. Invisible watermark (12 photos, 32-bit mark; bit accuracy, 0.5 = chance)", "",
          "| method | PSNR (invisible > 38) | " + " | ".join(atk) + " | embed s |", "|---|" + "---|" * (len(atk) + 2)]
    for meth in ("dwtDct", "dwtDctSvd"):
        rs = [v[meth] for v in W.values() if meth in v]
        if rs:
            L.append(f"| {meth} | {_mean([r['psnr'] for r in rs]):.1f} | " +
                     " | ".join(_f(_mean([r.get(a) for r in rs])) for a in atk) + f" | {_mean([r['embed_s'] for r in rs]):.3f} |")
    L.append("")

    # 7 angles
    A_ = Results("angles").d
    side = angle_side_mapping(A_)
    L += ["## 8. New camera angle, +-45 deg (16 photos): ours (AnyAngle LoRA / text) vs Qwen-Image-Edit-2511 + "
          "Multiple-Angles LoRA", "", f"LoRA side mapping chosen by re-projection error: {side}.", ""]

    def geo(meth):
        vals = []
        for name in m["angles"]:
            for y in YAWS:
                key = f"{name}@{y:+.0f}"
                if meth == "q2511_angles":
                    row = A_.get(f"{name}#{side[y]}", {}).get(meth, {})
                    vals.append((row.get(f"fid@{y:+.0f}") or {}).get("geo_dE") if row.get("v") == 3 else None)
                else:
                    vals.append(A_.get(key, {}).get(meth, {}).get("geo_dE"))
        return _mean(vals)
    judge_table("angles", ["ours_anyangle", "ours_text", "q2511_angles"], ["angle", "same_scene", "quality"],
                (["geo dE ↓", "s", "peak GB"], lambda meth: [_f(geo(meth)), _f(_mean(
                    [v.get(meth, {}).get("seconds") for v in A_.values()]), "{:.1f}"), _f(_mean(
                    [v.get(meth, {}).get("peak_gb") for v in A_.values()]), "{:.1f}")]))

    # 8 removal
    RM = S.get("removal", {})
    L += ["## 9. Object removal (RORD, same holes for every remover)", "",
          "| method | LPIPS in ↓ | PSNR in | SSIM in | seam ↓ | s | peak GB |", "|---|---|---|---|---|---|---|"]
    for meth in ("ours_smart", "lama_v5", "lama_iopaint", "mat", "segformer_lama"):
        rs = [v[meth] for v in RM.values() if meth in v]
        if rs:
            L.append(f"| {meth} | {_mean([r['lpips_in'] for r in rs]):.3f} | {_mean([r['psnr_in'] for r in rs]):.2f} | "
                     f"{_mean([r['ssim_in'] for r in rs]):.3f} | {_mean([r['seam'] for r in rs]):.1f} | "
                     f"{_f(_mean([r.get('seconds') for r in rs]), '{:.2f}')} | {_f(_mean([r.get('peak_gb') for r in rs]), '{:.1f}')} |")
    L.append("")
    L.append("segformer_lama uses Segformer's own person masks (end-to-end), the others the benchmark's hole.")
    SG = Results("seg").d
    L += ["", "| remove-by-name mask | IoU with the true mask ↑ | s |", "|---|---|---|"]
    for meth in ("ours_vlm_sam", "segformer"):
        rs = [v[meth] for v in SG.values() if meth in v]
        if rs:
            L.append(f"| {meth} | {_mean([r['iou'] for r in rs]):.3f} | {_mean([r['seconds'] for r in rs]):.2f} |")
    L.append("")

    # 9 general edits
    GE = JD.get("gedit", {})
    L += ["## 10. General instruction edits: GEdit-Bench English subset (3 per task type)", ""]

    def o_score(meth):
        vals = []
        for r in GE.values():
            x = r.get(meth)
            if isinstance(x, dict) and None not in (x.get("follow"), x.get("preserve"), x.get("quality")):
                vals.append((min(x["follow"], x["preserve"]) * x["quality"]) ** 0.5)
        return _mean(vals)
    judge_table("gedit", ["ours", "klein", "ip2p"], ["follow", "preserve", "quality"],
                (["overall O ↑", "s", "peak GB"], lambda meth: [_f(o_score(meth)), _f(res_stat("gedit", meth, "seconds"), "{:.1f}"),
                                                                 _f(res_stat("gedit", meth, "peak_gb"), "{:.1f}")]))
    L += ["Not benchmarked (no overlap with our pipeline or no test possible): LTX-Video (we have no video), dolly effect "
          "(we already render it from MoGe depth: views3d.render_view(dolly=...)), portrait bokeh (need the exact model "
          "name), Gradio (UI).", ""]
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


# ----------------------------------------------------------------------------- main

STAGES = {"prep": part_prep, "qwen": stage_qwen, "klein": stage_klein, "zimage": stage_zimage, "fluxdev": stage_fluxdev,
          "ip2p": stage_ip2p, "q2511": stage_q2511, "guards": stage_guards, "segformer": stage_segformer,
          "watermark": part_watermark, "score": part_score, "judge": part_judge, "report": part_report}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="prep,qwen,klein,zimage,fluxdev,ip2p,q2511,guards,segformer")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    for part in a.parts.split(","):
        t0 = time.perf_counter()
        print(f"\n########## {part}", flush=True)
        try:
            STAGES[part]()
        except Exception:  # noqa: BLE001
            err(f"stage {part}")
        free()
        print(f"########## {part} took {time.perf_counter() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
