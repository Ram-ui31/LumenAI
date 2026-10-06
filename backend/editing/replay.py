"""Re-run any saved recipe (generate / remove / outpaint pass) exactly, or with a new seed."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from core.recipe import load_recipe
from core.seed import resolve_seed


def rerun(recipe_path: str | Path, new_seed: bool = False, on_step: Any = None) -> Any:
    """Regenerate (``new_seed=False``) or make a variation (``new_seed=True``) of any recipe."""
    from generation import generate as gen

    r = load_recipe(recipe_path)
    parent = str(Path(recipe_path).with_suffix(".json"))
    if r.task == "generate":
        return gen.variation(recipe_path, on_step=on_step) if new_seed else gen.regenerate(recipe_path, on_step)

    ed = r.edit or {}
    src = Image.open(ed["source_path"]).convert("RGB")
    seed = resolve_seed(None) if new_seed else r.seed
    if r.task == "remove":
        from editing.remove import remove_object, replay_kwargs

        mask = (np.asarray(Image.open(ed["mask_path"]).convert("L")) > 127).astype(np.uint8)
        kw = replay_kwargs(r) | {"seed": seed}
        return remove_object(src, mask, parent_recipe=parent if new_seed else r.parent_recipe, on_step=on_step, **kw)
    if r.task == "outpaint":
        from editing.outpaint import outpaint_pass, replay_kwargs

        kw = replay_kwargs(r) | {"seed": seed}
        return outpaint_pass(src, parent_recipe=parent if new_seed else r.parent_recipe, save=True, on_step=on_step, **kw)
    raise ValueError(f"Cannot replay task {r.task!r}")
