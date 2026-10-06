"""Innovation D: VLM self-correction loop (Phase 7).

generate → VLM critique (JSON) → revise prompt / seed → … (max ``vlm.max_iterations``)
remove   → VLM critique of BEFORE|AFTER (JSON) → revise mode / mask / seed → …

Every iteration's image, critique (parsed + raw) and decision are logged to
``outputs/vlm/<time>_<task>/`` and summarised in the final recipe under ``extra.vlm_loop``.
The best iteration (VLM score) is returned, not necessarily the last.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw

from core.config import get_config, project_path
from core.recipe import save_with_recipe
from core.seed import resolve_seed
from editing.masking import dilate, mask_bbox, shadow_extend, to_binary
from vlm.client import ask_json, ask_short, parse_json, yes_no

log = logging.getLogger("adobeps.self_correct")

GEN_PROMPT = """You are a strict image reviewer. The image was generated from this prompt:
"{prompt}"
Check every element of the prompt (objects, counts, colours, attributes, setting).
Answer ONLY with one JSON object, no other text:
{{"matches_prompt": true or false,
 "missing": [prompt elements that are absent],
 "wrong": [elements shown incorrectly, e.g. wrong count or colour, as short phrases],
 "artifacts": true or false,
 "where": "short location of problems, or empty",
 "score": integer 1-10 for overall correctness and quality}}"""

REMOVE_PROMPT = """You are checking an object-removal edit. The picture shows BEFORE (left) and AFTER (right) of the
same photo. The red box marks the edited area. In BEFORE the box contains: {target}.
Look only inside the red box of AFTER and compare it with the surroundings.
Answer ONLY with one JSON object, no other text:
{{"object_gone": true or false,
 "new_object_added": true or false,
 "new_object": "name of any object that appeared in the box, or empty",
 "residue": true or false,
 "artifacts": true or false,
 "where": "short location of problems, or empty",
 "score": integer 1-10 for how natural and empty the AFTER box looks}}
residue = shadow, outline or fragment of the removed object still visible.
artifacts = blur, smear, seam, repeated pattern or distortion."""


OBJ_Q = ("Look at the centre of this picture. Is there a distinct object there (for example furniture, a sign, "
         "a bin, a box, an animal, a person, a vehicle or a tool), or only ground, wall, sky, water or plants? "
         "Answer with one word: object or background.")
RES_Q = ("Is there a dark shadow, a smudge, a blurry or smeared patch, or a ghost outline in the centre of this "
         "picture? Answer only yes or no.")
DECOMP_Q = """Break this image-generation prompt into at most 5 visual requirements that can each be checked
with a yes/no question (objects, counts, colours, clothing, setting).
Prompt: "{prompt}"
Reply ONLY with a JSON list like [{{"element": "four arms", "question": "Does the person have four arms?"}}]."""


def method() -> str:
    return get_config()["vlm"].get("critique", "vqa")


# ---- critics (shared by the loops and scripts/vlm_judge_eval.py) ---------------------------

def zoom(img: Image.Image, region: np.ndarray, pad: float = 0.5, size: int = 672) -> Image.Image:
    """Crop around the edited region (+pad on each side), longest side resized to ``size`` px.
    Small VLMs judge a 1024 px scene badly; a zoomed crop of the edit is much easier."""
    x0, y0, x1, y1 = mask_bbox(region)
    w, h = img.size
    px, py = int((x1 - x0) * pad), int((y1 - y0) * pad)
    box = (max(0, x0 - px), max(0, y0 - py), min(w, x1 + px), min(h, y1 + py))
    c = img.convert("RGB").crop(box)
    s = size / max(c.size)
    return c.resize((max(1, int(c.width * s)), max(1, int(c.height * s))), Image.LANCZOS)


def critique_removal(src: Image.Image, out: Image.Image, region: np.ndarray, target: str,
                     how: str | None = None, stub_flag: bool = False) -> tuple[dict, str]:
    """Critique dict with keys object_gone/new_object_added/new_object/residue/artifacts/where/score.
    ``vqa``: three short questions on a zoomed crop of AFTER (default). ``json``: one rubric on BEFORE|AFTER."""
    how = how or method()
    if how == "json":
        c, raw = ask_json([before_after(src, out, region)], REMOVE_PROMPT.format(target=target),
                          stub={"object_gone": True, "new_object_added": stub_flag, "new_object": "sign",
                                "residue": False, "artifacts": False, "where": "", "score": 4 if stub_flag else 8})
        c["method"] = "json"
        return c, raw
    crop = zoom(out, region)
    a1 = ask_short([crop], f"Is there a {target} in this picture? Answer only yes or no.", stub="no")
    a2 = ask_short([crop], OBJ_Q, stub="object" if stub_flag else "background")
    a3 = ask_short([crop], RES_Q, stub="no")
    present, residue = bool(yes_no(a1)), bool(yes_no(a3))
    new = "object" in a2.lower() and not present
    score = max(1, 10 - 4 * present - 3 * new - 3 * residue)
    c = {"object_gone": not present, "new_object_added": new, "new_object": "object" if new else "",
         "residue": residue, "artifacts": False, "where": "edited area" if (present or new or residue) else "",
         "score": score, "method": "vqa", "answers": {"target_present": a1, "object_or_background": a2, "residue": a3}}
    return c, json.dumps(c["answers"])


def decompose(prompt: str) -> list[dict]:
    """Prompt → [{element, question}] (TIFA/DSG-style). Falls back to splitting on commas/'and'."""
    from vlm.client import vlm_backend

    items: list[dict] = []
    if vlm_backend() != "stub":
        try:
            from vlm.client import ask

            raw = ask([], DECOMP_Q.format(prompt=prompt), max_tokens=300)
            t = raw[raw.find("["): raw.rfind("]") + 1]
            items = [x for x in json.loads(t) if isinstance(x, dict) and x.get("question")][:5]
        except Exception as e:  # noqa: BLE001
            log.warning("decompose failed (%s); using comma split", e)
    if not items:
        import re

        parts = [p.strip() for p in re.split(r",| and | in | with ", prompt) if len(p.strip()) > 2][:5]
        items = [{"element": p, "question": f"Does this picture show {p}?"} for p in parts]
    return items


def critique_generation(img: Image.Image, prompt: str, how: str | None = None,
                        checks: list[dict] | None = None, stub_ok: bool = True) -> tuple[dict, str]:
    how = how or method()
    if how == "json":
        c, raw = ask_json([img], GEN_PROMPT.format(prompt=prompt),
                          stub={"matches_prompt": stub_ok, "missing": [] if stub_ok else ["detail"], "wrong": [],
                                "artifacts": False, "where": "", "score": 8 if stub_ok else 5})
        c["method"] = "json"
        return c, raw
    checks = checks or decompose(prompt)
    answers, missing = {}, []
    for k, ch in enumerate(checks):
        a = ask_short([img], f"{ch['question']} Answer only yes or no.",
                      stub="yes" if (stub_ok or k) else "no")
        answers[ch["question"]] = a
        if yes_no(a) is False:
            missing.append(ch.get("element") or ch["question"])
    n = max(1, len(checks))
    score = round(10 * (n - len(missing)) / n)
    c = {"matches_prompt": not missing, "missing": missing, "wrong": [], "artifacts": False, "where": "",
         "score": score, "method": "vqa", "answers": answers, "checks": checks}
    return c, json.dumps(answers)


@dataclass
class LoopResult:
    image: Image.Image
    recipe: Any
    path: Path | None
    iterations: list[dict[str, Any]] = field(default_factory=list)
    log_dir: Path | None = None


def _log_dir(task: str) -> Path:
    d = project_path("outputs") / "vlm" / f"{datetime.now():%Y%m%d_%H%M%S}_{task}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _score(c: dict) -> float:
    try:
        return float(c.get("score", 0))
    except (TypeError, ValueError):
        return 0.0


def _finish(best: dict, iters: list[dict], d: Path, task: str) -> LoopResult:
    summary = [{k: v for k, v in it.items() if k not in ("image", "result")} for it in iters]
    (d / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    res = best["result"]
    res.recipe.extra["vlm_loop"] = {"task": task, "chosen_iteration": best["i"], "iterations": summary,
                                    "model": get_config()["vlm"]["mlx_repo"], "log_dir": str(d)}
    if res.path:
        save_with_recipe(res.image, res.recipe, Path(res.path).with_suffix(""))
    return LoopResult(res.image, res.recipe, res.path, summary, d)


# ---- generation -------------------------------------------------------------------------

def gen_ok(c: dict) -> bool:
    return bool(c.get("matches_prompt")) and not c.get("artifacts") and _score(c) >= get_config()["vlm"]["accept_score"]


def revise_generation(prompt0: str, c: dict, i: int) -> tuple[str, str]:
    """(new prompt, reason). Missing/wrong elements are restated explicitly; otherwise only the seed changes."""
    issues = [str(x) for x in (c.get("missing") or []) + (c.get("wrong") or []) if str(x).strip()]
    if issues:
        return f"{prompt0}, clearly showing {', '.join(issues[:4])}", f"emphasise: {', '.join(issues[:4])}"
    return prompt0, "new seed"


def self_correct_generate(prompt: str, seed: int | None = None, on_iter: Callable | None = None,
                          on_step: Callable | None = None, **gen_kw: Any) -> LoopResult:
    from generation.generate import build_recipe, run_recipe

    cfg = get_config()["vlm"]
    d, iters, seed0, p = _log_dir("generate"), [], resolve_seed(seed), prompt
    checks = decompose(prompt) if method() == "vqa" else None
    for i in range(int(cfg["max_iterations"])):
        t0 = time.perf_counter()
        r = run_recipe(build_recipe(p, (seed0 + 1000 * i) % 2**32, **gen_kw), on_step)
        c, raw = critique_generation(r.image, prompt, checks=checks, stub_ok=i > 0)
        r.image.save(d / f"iter{i + 1}.png")
        ok = gen_ok(c)
        new_p, why = revise_generation(prompt, c, i)
        it = {"i": i, "prompt": p, "seed": r.recipe.seed, "critique": c, "raw": raw, "accepted": ok,
              "next": None if ok else why, "path": str(r.path), "seconds": round(time.perf_counter() - t0, 1),
              "result": r}
        iters.append(it)
        (d / f"iter{i + 1}.json").write_text(json.dumps({k: v for k, v in it.items() if k != "result"}, indent=2, default=str))
        log.info("gen iter %d score=%s ok=%s next=%s", i + 1, c.get("score"), ok, it["next"])
        if on_iter:
            on_iter(it)
        if ok:
            break
        p = new_p
    best = max(iters, key=lambda x: (x["accepted"], _score(x["critique"])))
    return _finish(best, iters, d, "generate")


# ---- removal ----------------------------------------------------------------------------

def before_after(src: Image.Image, out: Image.Image, region: np.ndarray, max_w: int = 1280) -> Image.Image:
    """Side-by-side BEFORE|AFTER with the edited box marked, scaled to <= max_w wide."""
    x0, y0, x1, y1 = mask_bbox(region)
    pad = 6
    w, h = src.size
    canvas = Image.new("RGB", (2 * w + 16, h + 40), (255, 255, 255))
    for k, im in enumerate((src, out)):
        ox = k * (w + 16)
        canvas.paste(im.convert("RGB"), (ox, 40))
        dr = ImageDraw.Draw(canvas)
        dr.rectangle([ox + x0 - pad, 40 + y0 - pad, ox + x1 + pad, 40 + y1 + pad], outline=(255, 0, 0), width=4)
        dr.text((ox + 8, 10), "BEFORE" if k == 0 else "AFTER", fill=(0, 0, 0))
    s = min(1.0, max_w / canvas.width)
    return canvas.resize((int(canvas.width * s), int(canvas.height * s)), Image.LANCZOS) if s < 1 else canvas


def remove_ok(c: dict) -> bool:
    return (bool(c.get("object_gone")) and not c.get("new_object_added") and not c.get("residue")
            and not c.get("artifacts") and _score(c) >= get_config()["vlm"]["accept_score"])


def revise_removal(c: dict, kw: dict, mask: np.ndarray) -> tuple[dict, np.ndarray, str]:
    """Targeted fix from the critique. Returns (new kwargs, new mask, reason)."""
    kw = dict(kw)
    step = int(get_config()["critics"]["retry_dilate_step"])
    if c.get("new_object_added"):
        if kw["mode"] == "flux":
            kw.update(mode="lama")
            return kw, mask, f"FLUX added '{c.get('new_object') or 'an object'}': switch to lama"
        if kw["mode"] == "lama+flux" and (kw.get("strength") or 1) > 0.35:
            kw.update(strength=round(float(kw.get("strength") or 0.5) - 0.2, 2))
            return kw, mask, f"object added: lower lama+flux strength to {kw['strength']}"
        kw.update(mode="lama")
        return kw, mask, "object added: fall back to lama"
    if c.get("residue") or not c.get("object_gone", True):
        kw["dilate_px"] = int(kw["dilate_px"]) + step
        return kw, shadow_extend(mask), f"residue/shadow: shadow-aware mask, dilation {kw['dilate_px']} px"
    if c.get("artifacts") and kw["mode"] == "lama":
        kw.update(mode="lama+flux", strength=get_config()["remove"]["lama_flux_strength"])
        return kw, mask, "LaMa smear/artifacts: refine with lama+flux"
    kw["seed"] = resolve_seed(None)
    return kw, mask, "new seed"


def self_correct_remove(image: Image.Image, mask: np.ndarray, target: str | None = None, mode: str | None = None,
                        seed: int | None = None, on_iter: Callable | None = None,
                        on_step: Callable | None = None, **kw: Any) -> LoopResult:
    from editing.remove import remove_object

    cfg = get_config()
    kw = {"mode": mode or cfg["remove"]["default_mode"], "seed": resolve_seed(seed),
          "dilate_px": cfg["remove"].get("dilate_px", cfg["editing"]["mask_dilate_px"]), **kw}
    d, iters, m = _log_dir("remove"), [], to_binary(mask)
    parent = None
    for i in range(int(cfg["vlm"]["max_iterations"])):
        t0 = time.perf_counter()
        r = remove_object(image, m, parent_recipe=parent, on_step=on_step, **kw)
        tgt = target or ((r.recipe.extra.get("removal_check") or {}).get("target") or {}).get("cls") or "an object"
        view = before_after(image, r.image, r.region)
        c, raw = critique_removal(image, r.image, r.region, tgt, stub_flag=kw["mode"] != "lama")
        view.save(d / f"iter{i + 1}_before_after.png")
        zoom(r.image, r.region).save(d / f"iter{i + 1}_zoom.png")
        ok = remove_ok(c)
        nkw, nm, why = revise_removal(c, kw, m)
        it = {"i": i, "settings": {k: v for k, v in kw.items()}, "critique": c, "raw": raw, "accepted": ok,
              "next": None if ok else why, "path": str(r.path), "seconds": round(time.perf_counter() - t0, 1),
              "result": r}
        iters.append(it)
        (d / f"iter{i + 1}.json").write_text(json.dumps({k: v for k, v in it.items() if k != "result"}, indent=2, default=str))
        log.info("remove iter %d score=%s ok=%s next=%s", i + 1, c.get("score"), ok, it["next"])
        if on_iter:
            on_iter(it)
        if ok:
            break
        kw, m, parent = nkw, nm, str(Path(r.path).with_suffix(".json")) if r.path else None
    best = max(iters, key=lambda x: (x["accepted"], _score(x["critique"])))
    return _finish(best, iters, d, "remove")
