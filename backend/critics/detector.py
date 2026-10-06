"""Detector critic (Ultralytics YOLO nano, COCO classes).

* ``find_object``: which COCO object a removal mask covers (used to pick the class to check, and in the
  acceptance test to place clicks automatically).
* ``verify_removal``: after a removal, is an object of that class still inside the masked region?
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image

from core.config import get_config
from core.device import get_device


@dataclass
class Detection:
    cls: str
    conf: float
    box: tuple[int, int, int, int]   # x0, y0, x1, y1


def load_detector():
    """YOLO nano; weights are downloaded into the project cache (ultralytics fetches them by name)."""
    from ultralytics import YOLO

    from core.config import project_path

    d = project_path("cache") / "yolo"
    d.mkdir(parents=True, exist_ok=True)
    return YOLO(str(d / get_config()["critics"]["yolo_weights"]))


def detect(image: Image.Image, conf: float | None = None) -> list[Detection]:
    """COCO detections above ``conf`` (config default), most confident first."""
    conf = get_config()["critics"]["yolo_conf"] if conf is None else conf
    if get_config()["backend"] == "stub":
        return []
    from core.model_manager import get_manager

    model = get_manager().get("detector")
    try:
        res = model.predict(image.convert("RGB"), conf=conf, device=get_device(), verbose=False)[0]
    except Exception:  # noqa: BLE001  (some YOLO ops are missing on MPS: fall back to CPU, it is ~50 ms)
        res = model.predict(image.convert("RGB"), conf=conf, device="cpu", verbose=False)[0]
    out = []
    for xyxy, c, k in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy()):
        out.append(Detection(res.names[int(k)], float(c), tuple(int(v) for v in xyxy)))
    return sorted(out, key=lambda d: -d.conf)


def box_mask_overlap(box: tuple[int, int, int, int], mask: np.ndarray) -> float:
    """Fraction of the box area that lies inside the mask."""
    x0, y0, x1, y1 = box
    h, w = mask.shape
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    area = max(1, (x1 - x0) * (y1 - y0))
    return float((mask[y0:y1, x0:x1] > 0).sum()) / area


def mask_coverage(box: tuple[int, int, int, int], mask: np.ndarray) -> float:
    """Fraction of the mask that lies inside the box."""
    x0, y0, x1, y1 = box
    total = max(1, int((mask > 0).sum()))
    return float((mask[max(0, y0):y1, max(0, x0):x1] > 0).sum()) / total


def find_object(dets: list[Detection], mask: np.ndarray) -> Detection | None:
    """The detection that best matches the mask (box mostly masked AND mask mostly in the box)."""
    best, best_s = None, 0.0
    for d in dets:
        s = min(box_mask_overlap(d.box, mask), mask_coverage(d.box, mask))
        if s > best_s:
            best, best_s = d, s
    return best if best_s >= 0.3 else None


def judge(after: list[Detection], region: np.ndarray, target_cls: str | None, min_overlap: float) -> list[Detection]:
    """Detections in the result that sit inside the removal region (same class if known)."""
    return [d for d in after
            if (target_cls is None or d.cls == target_cls) and box_mask_overlap(d.box, region) >= min_overlap]


def verify_removal(source: Image.Image, result: Image.Image, region: np.ndarray) -> dict:
    """{'ok', 'target', 'remaining': [...]} — ok=False means an object of the removed class (or, if the
    class is unknown, any confident object) is still inside the masked region."""
    c = get_config()["critics"]
    target = find_object(detect(source), region)
    remaining = judge(detect(result), region, target.cls if target else None, c["remain_overlap"])
    if target is None:   # unknown class: only flag confident detections
        remaining = [d for d in remaining if d.conf >= c["unknown_class_conf"]]
    return {"ok": not remaining, "target": asdict(target) if target else None,
            "remaining": [asdict(d) for d in remaining]}
