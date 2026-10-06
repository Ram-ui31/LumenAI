"""Best-of-N generation (Innovation C, part 1).

Generate N candidates (seeds s, s+1, …, s+N-1), score each with CLIP prompt-match + LAION aesthetic,
keep the best. Every candidate is saved with its own (exactly reproducible) recipe; the winner's recipe
additionally records all candidate seeds, scores and the weights used.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from core.config import get_config
from core.recipe import save_with_recipe
from core.seed import resolve_seed
from critics.aesthetic import score
from generation.generate import Result, build_recipe, run_recipe

log = logging.getLogger("adobeps.best_of_n")


@dataclass
class BestOfN:
    best: Result | None
    candidates: list[dict[str, Any]] = field(default_factory=list)   # seed, clip, aesthetic, combined, path
    results: list[Result] = field(default_factory=list)


def iter_best_of_n(prompt: str, n: int | None = None, seed: int | None = None,
                   on_step: Callable[[int, int], None] | None = None, **gen_kw: Any) -> Iterator[BestOfN]:
    """Yields a growing :class:`BestOfN` after each candidate (``best`` = leader so far); the last
    yield is final and has the winner's recipe rewritten with the ``best_of_n`` block."""
    n = int(n or get_config()["critics"]["best_of_n"])
    base = resolve_seed(seed)
    t0 = time.perf_counter()
    state = BestOfN(best=None)  # type: ignore[arg-type]
    for i in range(n):
        r = run_recipe(build_recipe(prompt, (base + i) % 2**32, **gen_kw), on_step)
        s = score([r.image], prompt)[0]
        state.results.append(r)
        state.candidates.append({"seed": r.recipe.seed, "clip": s.clip, "aesthetic": s.aesthetic,
                                 "combined": s.combined, "path": str(r.path)})
        log.info("candidate %d/%d seed=%d clip=%.2f aesthetic=%.2f", i + 1, n, r.recipe.seed, s.clip, s.aesthetic)
        k = max(range(len(state.candidates)), key=lambda j: state.candidates[j]["combined"])
        state.best = state.results[k]
        if i < n - 1:
            yield state
    k = state.results.index(state.best)
    state.best.recipe.extra["best_of_n"] = {
        "n": n, "chosen_seed": state.candidates[k]["seed"], "weights": get_config()["critics"]["weights"],
        "clip_model": get_config()["critics"]["clip_repo"], "candidates": state.candidates,
        "total_seconds": round(time.perf_counter() - t0, 1),
    }
    save_with_recipe(state.best.image, state.best.recipe, Path(state.best.path).with_suffix(""))
    yield state


def best_of_n(prompt: str, n: int | None = None, seed: int | None = None,
              on_step: Callable[[int, int], None] | None = None, **gen_kw: Any) -> BestOfN:
    """Generate ``n`` candidates (seeds seed..seed+n-1) and return the one with the highest combined score."""
    final = None
    for final in iter_best_of_n(prompt, n, seed, on_step, **gen_kw):
        pass
    return final
