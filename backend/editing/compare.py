"""Side-by-side comparison grids for removal (modes x latent-preserve x blend)."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image

from core.config import project_path
from core.seed import resolve_seed
from editing.masking import dilate, overlay


def labeled_grid(images: Sequence[Image.Image], labels: Sequence[str], cols: int = 4, tile: int = 384) -> Image.Image:
    tiles = []
    for im, lab in zip(images, labels):
        im = im.convert("RGB")
        s = tile / max(im.size)
        t = np.asarray(im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)).copy()
        canvas = np.full((tile + 28, tile, 3), 24, np.uint8)
        canvas[28:28 + t.shape[0], :t.shape[1]] = t
        cv2.putText(canvas, lab[:48], (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
        tiles.append(canvas)
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 24))
    rows = [np.concatenate(tiles[i:i + cols], axis=1) for i in range(0, len(tiles), cols)]
    return Image.fromarray(np.concatenate(rows, axis=0))


def removal_comparison(image: Image.Image, mask: np.ndarray, name: str, seed: int | None = None,
                       modes: Sequence[str] = ("lama", "flux", "lama+flux")) -> tuple[Path, dict]:
    """Run every mode; for flux-based modes A off/on (one model run each) x blend none/feather/laplacian.
    Saves a grid to outputs/comparisons/ and returns (path, per-variant stats incl. outside-mask check)."""
    from editing.remove import remove_object

    seed = resolve_seed(seed)
    O = np.asarray(image.convert("RGB"))
    ims, labels, stats = [image, overlay(image, mask)], ["source", "mask"], {}
    from core.config import get_config
    from editing.blending import composite

    e = get_config()["editing"]
    variants: list[tuple[str, bool | None]] = []
    for m in modes:
        variants += [("lama", None)] if m == "lama" else [(m, False), (m, True)]
    for mode, lp in variants:
        a_lab = "on" if lp else "off" if lp is not None else "-"
        try:  # one model run per (mode, A); blend modes are recomposited from the same raw output
            r = remove_object(image, mask, mode=mode, seed=seed, latent_preserve=lp, blend="laplacian", save=True)
        except Exception as ex:  # noqa: BLE001
            stats[f"{mode} A={a_lab}"] = {"error": f"{type(ex).__name__}: {ex}"}
            continue
        hole = (np.asarray(mask) > 0).astype(np.uint8)
        if r.recipe.edit.get("shadow_aware"):
            from editing.masking import shadow_extend

            hole = shadow_extend(hole)
        region = r.region
        drift = float(np.abs(r.generated.astype(np.int16) - O.astype(np.int16))[region == 0].mean())
        for blend in (("none", "feather", "laplacian") if mode != "lama" else ("laplacian",)):
            out = r.image if blend == "laplacian" else Image.fromarray(composite(
                O, r.generated, hole, region, mode=blend, levels=e["pyramid_levels"],
                band_high_px=e["blend_band_high_px"], band_low_px=e["blend_band_low_px"], feather_px=e["feather_px"]))
            arr = np.asarray(out)
            label = f"{mode} A={a_lab} B={blend}"
            stats[label] = {"seconds": r.recipe.stats.get("seconds"),
                            "outside_identical": bool(np.array_equal(arr[region == 0], O[region == 0])),
                            "raw_drift_outside": round(drift, 3), "path": str(r.path)}
            ims.append(out)
            labels.append(label)
    grid = labeled_grid(ims, labels)
    out = project_path("comparisons") / f"{name}_{datetime.now():%Y%m%d_%H%M%S}_s{seed}.png"
    grid.save(out)
    return out, stats
