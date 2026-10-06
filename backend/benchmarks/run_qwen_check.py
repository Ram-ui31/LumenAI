#!/usr/bin/env python
"""Qwen-Image 2.1 checkpoint: does it run, which compression to use, and does structure-guided refinement work?

For each variant (bf16 reference, int8, nf4 - all bitsandbytes, quantized on load):
  1. Text-to-image on 6 fixed prompts (1024x1024): time, peak GPU memory, CLIP prompt match, aesthetic,
     LPIPS drift vs the bf16 image for the same prompt and seed.
  2. Object removal on RORD-50 scenes (same masks as the removal benchmark), all composited with seam smoothing:
       Qwen-edit         Qwen alone: "remove the person ..." instruction on the original photo
       LaMa>Qwen         structure-guided: start from LaMa's fill noised to `strength`, positive scene prompt
       LaMa>Qwen+cond    same, with LaMa's fill also given as a reference image
     plus LaMa alone (computed once). Metrics as in the removal benchmark.
  3. Shared VLM test: the pipeline's own text encoder answers a description and a grounding question.

  python -m benchmarks.run_qwen_check --variants bf16,int8,nf4 --n 10
Outputs: outputs/eval/qwen_check/{summary.md, removal.csv, generation.csv, vlm.json, images}
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from core.config import get_config, project_path  # noqa: E402

get_config()["memory"]["enforce"] = False   # this script measures Qwen's real memory itself; no 16 GB cap here
from benchmarks import metrics as M  # noqa: E402

SEED = 7
PROMPTS = [
    "a red fox sitting in fresh snow at golden hour, pine forest behind, photo",
    "a cosy reading corner with a green armchair, a floor lamp and a bookshelf, warm evening light",
    "a busy street market in the rain at night, neon signs reflected in puddles",
    "a close-up of a ceramic coffee cup on a wooden table, steam rising, shallow depth of field",
    "a mountain lake at sunrise with mist over the water, a wooden boat at the shore",
    "a shop sign that reads \"ADOBE PS\" in bold letters above a bakery window",
]
REMOVE_INSTR = ("Remove the person and everything they are holding or touching from this photo, including their "
                "shadow. Keep everything else in the photo exactly unchanged.")
METHODS = ["Qwen-edit", "LaMa>Qwen", "LaMa>Qwen+cond"]


def free():
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def scene_prompt(labels: list[str]) -> str:
    return f"A realistic photo of {', '.join(labels)}. Natural light, sharp, detailed texture."


def stage_lama(samples, out: Path) -> dict:
    """LaMa removal + region + CLIP background labels per scene (cached on disk)."""
    from editing.bg_prompt import describe_background
    from editing.masking import shadow_extend
    from editing.remove import remove_object

    meta_p = out / "lama_meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    (out / "LaMa").mkdir(parents=True, exist_ok=True)
    rc = get_config()["remove"]
    for s in samples:
        if s.name in meta:
            continue
        t0 = time.perf_counter()
        r = remove_object(s.image, s.mask, mode="lama", seed=SEED, save=False)
        sec = time.perf_counter() - t0
        hole = shadow_extend(s.mask) if rc.get("shadow_aware") else s.mask
        _, ranked = describe_background(s.image, r.region)
        r.image.save(out / "LaMa" / f"{s.name}.png")
        Image.fromarray((r.region > 0).astype(np.uint8) * 255).save(out / "LaMa" / f"{s.name}_region.png")
        Image.fromarray((hole > 0).astype(np.uint8) * 255).save(out / "LaMa" / f"{s.name}_hole.png")
        meta[s.name] = {"labels": [t for t, _ in ranked[:2]], "seconds": round(sec, 2)}
        meta_p.write_text(json.dumps(meta, indent=1))
    return meta


from editing.blending import composite_back  # noqa: E402,F401  (moved to the library)


def run_variant(variant: str, samples, meta: dict, out: Path, steps: int, strength: float, gen_rows: list,
                vlm: dict) -> None:
    from editing import qwen as Q

    vd = out / variant
    t0 = time.perf_counter()
    pipe = Q.load_pipe(variant)
    load_s = time.perf_counter() - t0
    print(f"[{variant}] loaded in {load_s:.0f}s mode={pipe._adobeps_mode}", flush=True)

    (vd / "gen").mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(PROMPTS):
        f = vd / "gen" / f"{i}.png"
        r = Q.generate(pipe, p, seed=100 + i, size=1024, steps=steps)   # always re-timed: cheap, and timing needs it
        r.image.save(f)
        gen_rows.append({"variant": variant, "prompt_id": i, "seconds": round(r.seconds, 2), "peak_gb": round(r.peak_gb, 2),
                         "load_s": round(load_s, 1), "mode": pipe._adobeps_mode})
        print(f"[{variant}] gen {i}: {r.seconds:.1f}s peak {r.peak_gb:.1f} GB", flush=True)

    s0 = samples[0]
    try:
        vlm[variant] = {
            "describe": Q.ask(pipe, [s0.gt], "Describe the floor, walls and background surfaces in this photo in one short sentence."),
            "ground": Q.ask(pipe, [s0.image], f"The image is {s0.image.size[0]}x{s0.image.size[1]} pixels. Give the bounding "
                            "box of the main person as JSON {\"bbox\": [x0, y0, x1, y1]} in pixel coordinates, nothing else."),
            "count": Q.ask(pipe, [s0.image], "How many people are in this photo? Answer with a number."),
        }
    except Exception as e:  # noqa: BLE001
        vlm[variant] = {"error": repr(e)}
    print(f"[{variant}] vlm: {vlm[variant]}", flush=True)

    broken: dict[str, str] = {}
    for s in samples:
        O = np.asarray(s.image)
        lama = Image.open(out / "LaMa" / f"{s.name}.png").convert("RGB")
        region = np.asarray(Image.open(out / "LaMa" / f"{s.name}_region.png")) > 0
        hole = np.asarray(Image.open(out / "LaMa" / f"{s.name}_hole.png")) > 0
        prompt = scene_prompt(meta[s.name]["labels"])
        jobs = {
            "Qwen-edit": lambda: Q.edit(pipe, s.image, REMOVE_INSTR, seed=SEED, steps=steps),
            "LaMa>Qwen": lambda: Q.refine(pipe, lama, prompt, seed=SEED, strength=strength, steps=steps),
            "LaMa>Qwen+cond": lambda: Q.refine(pipe, lama, "Keep this photo exactly the same, with natural, sharp, "
                                               f"realistic detail. {prompt}", seed=SEED, strength=strength, steps=steps,
                                               condition=lama),
        }
        for m, job in jobs.items():
            md = vd / m
            md.mkdir(parents=True, exist_ok=True)
            if (md / f"{s.name}.png").exists() or m in broken:
                continue
            try:
                r = job()
            except Exception as e:  # noqa: BLE001  fail fast per method: don't burn GPU time repeating an error
                import traceback

                traceback.print_exc()
                broken[m] = repr(e)
                vlm.setdefault(variant, {}).setdefault("method_errors", {})[m] = repr(e)[:300]
                print(f"[{variant}] {m} FAILED, skipped for the remaining scenes", flush=True)
                free()
                continue
            r.image.save(md / f"{s.name}_raw.png")
            Image.fromarray(composite_back(O, r.image, hole, region)).save(md / f"{s.name}.png")
            (md / f"{s.name}.json").write_text(json.dumps({"seconds": round(r.seconds, 2), "peak_gb": round(r.peak_gb, 2),
                                                          "prompt": prompt if m != "Qwen-edit" else REMOVE_INSTR, **r.info}))
            print(f"[{variant}] {s.name} {m}: {r.seconds:.1f}s peak {r.peak_gb:.1f} GB", flush=True)
    del pipe
    free()


def phase_shift(a: np.ndarray, b: np.ndarray) -> float:
    """Global translation (pixels) between two images by phase correlation - the 'edits move the picture' problem."""
    import cv2

    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32)
    (dx, dy), _ = cv2.phaseCorrelate(ga, gb)
    return float(np.hypot(dx, dy))


def score(samples, out: Path, variants: list[str]) -> list[dict]:
    rows = []
    for s in samples:
        O, GT = np.asarray(s.image), np.asarray(s.gt)
        region = np.asarray(Image.open(out / "LaMa" / f"{s.name}_region.png")) > 0
        cands = [("LaMa", "-", out / "LaMa")] + [(m, v, out / v / m) for v in variants for m in METHODS]
        for m, v, d in cands:
            p = d / f"{s.name}.png"
            if not p.exists():
                continue
            R = np.asarray(Image.open(p).convert("RGB"))
            st = json.loads((d / f"{s.name}.json").read_text()) if (d / f"{s.name}.json").exists() else {}
            raw = d / f"{s.name}_raw.png"
            drift = shift_px = None
            if raw.exists():   # how much Qwen's raw output moved/repainted the parts it was NOT asked to change
                Rw = np.asarray(Image.open(raw).convert("RGB").resize((O.shape[1], O.shape[0]), Image.LANCZOS))
                drift = M.psnr(Rw, O, ~region)
                shift_px = phase_shift(Rw, O)
            rows.append({"scene": s.name, "variant": v, "method": m, "raw_psnr_outside": drift, "raw_shift_px": shift_px,
                         "psnr_in": M.psnr(R, GT, region), "ssim_in": M.ssim(R, GT, region),
                         "lpips_in": M.lpips_crop(R, GT, region), "seam": M.seam(R, GT, region),
                         "clip_i": M.clip_i(R, GT, region),
                         "exact_outside": bool(np.array_equal(R[~region], O[~region])),
                         "seconds": st.get("seconds"), "peak_gb": st.get("peak_gb")})
    return rows


def score_gen(out: Path, variants: list[str], gen_rows: list[dict]) -> list[dict]:
    from critics.aesthetic import score as aes

    for r in gen_rows:
        v, i = r["variant"], r["prompt_id"]
        im = Image.open(out / v / "gen" / f"{i}.png").convert("RGB")
        sc = aes([im], PROMPTS[i])[0]
        r["clip"], r["aesthetic"] = round(sc.clip, 2), round(sc.aesthetic, 2)
        ref = out / "bf16" / "gen" / f"{i}.png"
        r["lpips_vs_bf16"] = (round(M.lpips_dist(np.asarray(im), np.asarray(Image.open(ref).convert("RGB"))), 4)
                              if ref.exists() and v != "bf16" else None)
    return gen_rows


def summarize(rows, gen_rows, vlm, variants, n, strength, steps) -> str:
    L = [f"# Qwen-Image 2.1 check: {n} RORD-50 scenes, strength {strength}, {steps} steps, seed {SEED}", "",
         "## Text-to-image (6 prompts, 1024x1024)", "",
         "| variant | mode | load s | s/img | peak GB | CLIP ↑ | aesthetic ↑ | LPIPS vs bf16 ↓ |", "|---|---|---|---|---|---|---|---|"]
    for v in variants:
        g = [r for r in gen_rows if r["variant"] == v]
        if not g:
            continue
        lp = [r["lpips_vs_bf16"] for r in g if r.get("lpips_vs_bf16") is not None]
        L.append(f"| {v} | {g[0]['mode']} | {g[0]['load_s']} | {np.mean([r['seconds'] for r in g]):.1f} | "
                 f"{max(r['peak_gb'] for r in g):.1f} | {np.mean([r['clip'] for r in g]):.1f} | "
                 f"{np.mean([r['aesthetic'] for r in g]):.2f} | {np.mean(lp):.3f} |" if lp else
                 f"| {v} | {g[0]['mode']} | {g[0]['load_s']} | {np.mean([r['seconds'] for r in g]):.1f} | "
                 f"{max(r['peak_gb'] for r in g):.1f} | {np.mean([r['clip'] for r in g]):.1f} | "
                 f"{np.mean([r['aesthetic'] for r in g]):.2f} | – |")
    L += ["", "## Object removal (mean [95% CI]; all composited with seam smoothing)", "",
          "| variant | method | PSNR in ↑ | SSIM in ↑ | LPIPS ↓ | seam ↓ | CLIP-I ↑ | outside identical | s/img | peak GB |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    drift_lines = ["", "## Raw Qwen output drift (before our compositing): PSNR outside the edit vs input ↑, global shift px ↓", "",
                   "| variant | method | raw PSNR outside | shift px (median) | shift px (max) |", "|---|---|---|---|---|"]
    keys = ["psnr_in", "ssim_in", "lpips_in", "seam", "clip_i"]
    for v, m in [("-", "LaMa")] + [(v, m) for v in variants for m in METHODS]:
        rs = [r for r in rows if r["variant"] == v and r["method"] == m]
        if not rs:
            continue
        cells = []
        for k in keys:
            mean, lo, hi = M.bootstrap_ci([r[k] for r in rs])
            f = "{:.3f}" if k in ("ssim_in", "lpips_in") else "{:.1f}"
            cells.append(f"{f.format(mean)} [{f.format(lo)}–{f.format(hi)}]")
        secs = [r["seconds"] for r in rs if r["seconds"] is not None]
        peaks = [r["peak_gb"] for r in rs if r["peak_gb"] is not None]
        L.append(f"| {v} | {m} | " + " | ".join(cells) + f" | {100 * np.mean([r['exact_outside'] for r in rs]):.0f}% | "
                 f"{np.mean(secs):.1f} | " + (f"{max(peaks):.1f}" if peaks else "–") + " |" if secs else
                 f"| {v} | {m} | " + " | ".join(cells) + f" | {100 * np.mean([r['exact_outside'] for r in rs]):.0f}% | – | – |")
    for v in variants:
        for m in METHODS:
            rs = [r for r in rows if r["variant"] == v and r["method"] == m and r.get("raw_shift_px") is not None]
            if rs:
                sh = [r["raw_shift_px"] for r in rs]
                drift_lines.append(f"| {v} | {m} | {np.mean([r['raw_psnr_outside'] for r in rs]):.1f} | "
                                   f"{np.median(sh):.2f} | {max(sh):.2f} |")
    L += drift_lines
    L += ["", "## Paired wins vs LaMa (LPIPS lower / seam lower)", ""]
    lam = {r["scene"]: r for r in rows if r["method"] == "LaMa"}
    for v in variants:
        for m in METHODS:
            rs = [r for r in rows if r["variant"] == v and r["method"] == m and r["scene"] in lam]
            if rs:
                L.append(f"- {v} {m}: LPIPS {sum(r['lpips_in'] < lam[r['scene']]['lpips_in'] for r in rs)}/{len(rs)}, "
                         f"seam {sum(r['seam'] < lam[r['scene']]['seam'] for r in rs)}/{len(rs)}")
    L += ["", "## Shared VLM test (pipeline text encoder used as the assistant)", "", "```", json.dumps(vlm, indent=1), "```"]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    from benchmarks.data import load

    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="bf16,int8,nf4")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--strength", type=float, default=0.5)
    a = ap.parse_args(argv)
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    out = project_path("outputs") / "eval" / "qwen_check"
    out.mkdir(parents=True, exist_ok=True)
    samples = load(a.n, a.size)
    meta = stage_lama(samples, out)
    from core.model_manager import get_manager

    get_manager().evict_all()
    free()
    gen_rows, vlm = [], {}
    for v in variants:
        try:
            run_variant(v, samples, meta, out, a.steps, a.strength, gen_rows, vlm)
        except Exception as e:  # noqa: BLE001  keep going: one broken variant must not lose the others
            import traceback

            traceback.print_exc()
            vlm.setdefault(v, {})["FAILED"] = repr(e)
            free()
    rows = score(samples, out, variants)
    gen_rows = score_gen(out, variants, gen_rows)
    for name, data in (("removal.csv", rows), ("generation.csv", gen_rows)):
        if data:
            with open(out / name, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
                w.writeheader()
                w.writerows(data)
    (out / "vlm.json").write_text(json.dumps(vlm, indent=1))
    md = summarize(rows, gen_rows, vlm, variants, len(samples), a.strength, a.steps)
    (out / "summary.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
