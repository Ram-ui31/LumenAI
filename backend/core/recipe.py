"""Recipes: everything needed to recreate an output, saved as JSON sidecar AND PNG text metadata."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from core.config import project_path

RECIPE_VERSION = 1
PNG_KEY = "adobeps_recipe"


def library_versions() -> dict[str, str]:
    """Versions of the libraries that affect pixel output."""
    out: dict[str, str] = {}
    for name in ("torch", "diffusers", "transformers", "gguf", "mflux"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            continue
    return out


def sha256_image(img: Image.Image | np.ndarray) -> str:
    """SHA-256 of the raw RGB pixel bytes (independent of PNG encoding)."""
    arr = np.asarray(img.convert("RGB") if isinstance(img, Image.Image) else img)
    h = hashlib.sha256(np.ascontiguousarray(arr).tobytes())
    h.update(str(arr.shape).encode())
    return h.hexdigest()


@dataclass
class Recipe:
    """Serializable description of one output image. Unknown keys go into ``extra``."""

    task: str = "generate"                      # generate | remove | outpaint | background
    prompt: str = ""
    negative_prompt: str | None = None
    model_id: str = ""
    transformer_weights: str | None = None     # "repo/file.gguf"
    quantization: str | None = None
    dtype: str = "bfloat16"
    device: str = "cpu"
    backend: str = "diffusers"
    scheduler: str = "FlowMatchEulerDiscreteScheduler"
    steps: int = 4
    guidance_scale: float = 0.0
    width: int = 1024
    height: int = 1024
    seed: int = 0
    max_sequence_length: int = 256
    edit: dict[str, Any] | None = None
    versions: dict[str, str] = field(default_factory=library_versions)
    parent_recipe: str | None = None
    stats: dict[str, Any] = field(default_factory=dict)   # timing / memory (not needed to reproduce)
    extra: dict[str, Any] = field(default_factory=dict)
    recipe_version: int = RECIPE_VERSION
    created_at: str = field(default_factory=lambda: datetime.now().astimezone().isoformat(timespec="seconds"))

    # ---- (de)serialization -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # put recipe_version first for readability
        return {"recipe_version": d.pop("recipe_version"), **d}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Recipe":
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in d.items() if k in known}
        extra = {k: v for k, v in d.items() if k not in known}
        r = cls(**kwargs)
        r.extra.update(extra)
        return r

    @classmethod
    def from_json(cls, s: str) -> "Recipe":
        return cls.from_dict(json.loads(s))

    def child(self, **changes: Any) -> "Recipe":
        """Copy with changes, a fresh timestamp/versions and cleared stats."""
        d = self.to_dict()
        d.update(changes)
        d["created_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        d["versions"] = library_versions()
        d["stats"] = {}
        d["extra"] = {k: v for k, v in d.get("extra", {}).items()
                      if k not in ("best_of_n", "removal_check", "output_sha256", "reproducibility_warning", "vlm_loop")}
        return Recipe.from_dict(d)


# ---- files --------------------------------------------------------------------

def new_output_stem(task: str, seed: int, out_dir: Path | None = None) -> Path:
    """``outputs/YYYY-MM-DD/<task>_<HHMMSS>_<seed>`` (unique)."""
    now = datetime.now()
    base = out_dir or project_path("outputs") / now.strftime("%Y-%m-%d")
    base.mkdir(parents=True, exist_ok=True)
    stem = base / f"{task}_{now.strftime('%H%M%S')}_{seed}"
    i = 1
    while stem.with_suffix(".png").exists():
        stem = base / f"{task}_{now.strftime('%H%M%S')}_{seed}_{i}"
        i += 1
    return stem


def save_with_recipe(img: Image.Image, recipe: Recipe, stem: Path) -> Path:
    """Write ``stem.png`` (recipe embedded as PNG text) and ``stem.json``. Returns PNG path."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    recipe.extra["output_sha256"] = sha256_image(img)
    try:
        from core.config import get_config

        mem = get_config().get("memory", {})
        recipe.extra.setdefault("memory_profile", {"quant": mem.get("quant"), "vae_tiling": bool(mem.get("vae_tiling"))})
    except Exception:  # noqa: BLE001
        pass
    js = recipe.to_json()
    meta = PngInfo()
    meta.add_text(PNG_KEY, js)
    png = stem.with_suffix(".png")
    img.convert("RGB").save(png, pnginfo=meta)
    stem.with_suffix(".json").write_text(js)
    return png


def load_recipe(path: str | Path) -> Recipe:
    """Load a recipe from a ``.json`` sidecar or from a PNG's embedded metadata."""
    path = Path(path)
    if path.suffix.lower() == ".png":
        with Image.open(path) as im:
            txt = im.text.get(PNG_KEY) if hasattr(im, "text") else None
        if txt is None:
            side = path.with_suffix(".json")
            if side.exists():
                return Recipe.from_json(side.read_text())
            raise ValueError(f"No recipe metadata in {path}")
        return Recipe.from_json(txt)
    return Recipe.from_json(path.read_text())


def recipe_png_path(path: str | Path) -> Path:
    """PNG belonging to a recipe path (json or png)."""
    return Path(path).with_suffix(".png")
