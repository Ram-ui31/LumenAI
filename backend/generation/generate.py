"""Text-to-image generation: generate, regenerate (exact), variation (new seed), tweak (same seed)."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from core.config import get_config, project_path
from core.device import MemoryTracker
from core.recipe import Recipe, load_recipe, new_output_stem, save_with_recipe
from core.seed import resolve_seed
from generation.backends import StepFn, get_gen_backend

log = logging.getLogger("adobeps.generate")


@dataclass
class Result:
    image: Image.Image
    recipe: Recipe
    path: Path


def _round16(x: int) -> int:
    return max(16, int(round(x / 16)) * 16)


def log_run(recipe: Recipe, path: Path) -> None:
    rec = {"path": str(path), "task": recipe.task, "seed": recipe.seed, "backend": recipe.backend, **recipe.stats}
    with open(project_path("logs") / "runs.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


def run_recipe(recipe: Recipe, on_step: StepFn | None = None, out_stem: Path | None = None) -> Result:
    """Execute a *generate* recipe exactly as written and save the output with its recipe."""
    if recipe.task != "generate":
        raise ValueError("run_recipe handles task=generate; use editing.* for edits")
    backend = get_gen_backend(recipe.backend)
    now = backend.describe()
    diffs = {k: (getattr(recipe, k), v) for k, v in now.items() if getattr(recipe, k, v) != v}
    if diffs:
        msg = f"Current setup differs from recipe {diffs}; output may not be pixel-identical."
        log.warning(msg)
        recipe.extra["reproducibility_warning"] = msg
    tracker = MemoryTracker()
    t0 = time.perf_counter()
    img = backend.text2img(recipe.prompt, recipe.seed, recipe.steps, recipe.guidance_scale, recipe.width,
                           recipe.height, recipe.max_sequence_length, tracker, on_step)
    recipe.stats = {"seconds": round(time.perf_counter() - t0, 2), **tracker.report()}
    path = save_with_recipe(img, recipe, out_stem or new_output_stem("generate", recipe.seed))
    log_run(recipe, path)
    log.info("generated %s in %.1fs", path, recipe.stats["seconds"])
    return Result(img, recipe, path)


def build_recipe(prompt: str, seed: int | None = None, steps: int | None = None, width: int | None = None,
                 height: int | None = None, guidance_scale: float | None = None, negative_prompt: str | None = None,
                 parent_recipe: str | None = None) -> Recipe:
    """A new generate recipe from config defaults + overrides (random seed if none)."""
    cfg = get_config()
    c = cfg["generation"]
    backend = get_gen_backend()
    return Recipe(
        task="generate",
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        backend=backend.name,
        steps=int(steps or c["steps"]),
        guidance_scale=float(c["guidance_scale"] if guidance_scale is None else guidance_scale),
        width=_round16(width or c["width"]),
        height=_round16(height or c["height"]),
        seed=resolve_seed(seed),
        max_sequence_length=int(c["max_sequence_length"]),
        parent_recipe=parent_recipe,
        **backend.describe(),
    )


def generate(prompt: str, seed: int | None = None, on_step: StepFn | None = None, **kw: Any) -> Result:
    return run_recipe(build_recipe(prompt, seed, **kw), on_step)


def regenerate(recipe_path: str | Path, on_step: StepFn | None = None) -> Result:
    """Run a saved recipe exactly (same seed, params, weights). Output goes to a new file."""
    r = load_recipe(recipe_path)
    r = r.child()  # fresh timestamp/versions; everything that affects pixels is unchanged
    return run_recipe(r, on_step)


def variation(recipe_path: str | Path, seed: int | None = None, on_step: StepFn | None = None) -> Result:
    """Same recipe with a new seed; ``parent_recipe`` points at the source."""
    r = load_recipe(recipe_path)
    new_seed = resolve_seed(seed)
    while new_seed == r.seed:
        new_seed = resolve_seed(None)
    return run_recipe(r.child(seed=new_seed, parent_recipe=str(Path(recipe_path).with_suffix(".json"))), on_step)


def tweak(recipe_path: str | Path, on_step: StepFn | None = None, **changes: Any) -> Result:
    """Same seed, edited prompt/params; ``parent_recipe`` points at the source."""
    r = load_recipe(recipe_path)
    changes = {k: v for k, v in changes.items() if v is not None}
    for k in ("width", "height"):
        if k in changes:
            changes[k] = _round16(changes[k])
    return run_recipe(r.child(parent_recipe=str(Path(recipe_path).with_suffix(".json")), **changes), on_step)
