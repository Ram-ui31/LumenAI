"""Phase 8 test set: RORD-50 (Apache-2.0) — 50 real scenes with object masks and real object-removed ground truth.

The dataset ships as three aligned videos per scene (``videos/``, ``masks/``, ``gts/``). We take one frame per
scene (the middle frame, same index in all three) and resize so the longest side is ``size`` px.
Frames are cached under ``.cache/eval/rord50_<size>/<name>_{input,mask,gt}.png``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from core.config import project_path

log = logging.getLogger("adobeps.eval")
REPO = "HigherHu/RORD-50"


@dataclass
class Sample:
    name: str
    image: Image.Image     # input (object present)
    mask: np.ndarray       # uint8 {0,1}
    gt: Image.Image        # real object-removed frame


def _frame(path: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(index, n - 1))
    ok, fr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {index} of {path}")
    return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)


def _count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(1, n)


def _resize(arr: np.ndarray, size: int, nearest: bool = False) -> np.ndarray:
    h, w = arr.shape[:2]
    s = size / max(h, w)
    nw, nh = max(16, int(round(w * s / 16)) * 16), max(16, int(round(h * s / 16)) * 16)
    return cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)


def download() -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(REPO, repo_type="dataset", cache_dir=str(project_path("cache") / "hf_datasets")))


def load(n: int | None = None, size: int = 768) -> list[Sample]:
    """First ``n`` scenes (sorted by name), extracted and cached."""
    out_dir = project_path("cache") / "eval" / f"rord50_{size}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cached = sorted(p.name[: -len("_input.png")] for p in out_dir.glob("*_input.png"))
    names = cached
    if n is None or len(cached) < n:
        root = download()
        names = sorted(p.stem for p in (root / "videos").glob("*.mp4"))
        for name in names[: n or len(names)]:
            if (out_dir / f"{name}_gt.png").exists():
                continue
            k = min(_count(root / d / f"{name}.mp4") for d in ("videos", "masks", "gts")) // 2
            img = _resize(_frame(root / "videos" / f"{name}.mp4", k), size)
            gt = _resize(_frame(root / "gts" / f"{name}.mp4", k), size)
            m = _resize(_frame(root / "masks" / f"{name}.mp4", k).max(axis=2), size, nearest=True)
            Image.fromarray(img).save(out_dir / f"{name}_input.png")
            Image.fromarray(gt).save(out_dir / f"{name}_gt.png")
            Image.fromarray(((m > 127) * 255).astype(np.uint8)).save(out_dir / f"{name}_mask.png")
            log.info("extracted %s (frame %d)", name, k)
    samples = []
    for name in names[: n or len(names)]:
        m = (np.asarray(Image.open(out_dir / f"{name}_mask.png")) > 127).astype(np.uint8)
        if m.sum() == 0:
            log.warning("%s: empty mask at the chosen frame, skipped", name)
            continue
        samples.append(Sample(name, Image.open(out_dir / f"{name}_input.png").convert("RGB"), m,
                              Image.open(out_dir / f"{name}_gt.png").convert("RGB")))
    return samples
