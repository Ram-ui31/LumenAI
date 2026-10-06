"""Invisible LumenAI watermark on every image the editor returns (v9 benchmark winner).

dwtDctSvd (invisible-watermark library), 32-bit payload "LUMN". Measured on 12 photos (v9): PSNR 41 dB (invisible),
bits recovered: 100 % untouched, 99 % after JPEG 90/75, 79 % after JPEG 50, 100 % after resizing to half and back,
100 % after brightening, 72 % after one of our own AI edits. The teammate's dwtDct was at chance (53 %) after any
JPEG save. Not robust to heavy cropping (60 %) - it marks, it does not protect.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
from PIL import Image

PAYLOAD = b"LUMN"
METHOD = "dwtDctSvd"
MIN_SIDE = 256            # the transform needs room for its blocks; smaller images are left unmarked
FOUND_AT = 0.85           # share of the 32 bits that must match ("found"); chance level is 0.5


def enabled() -> bool:
    if os.environ.get("ADOBEPS_WATERMARK") is not None:
        return os.environ["ADOBEPS_WATERMARK"] not in ("0", "false", "off")
    return os.environ.get("ADOBEPS_BACKEND") != "stub"      # tests compare pixels exactly; they opt in explicitly


def embed(img: Image.Image | np.ndarray) -> Image.Image:
    """Return the image with the mark in its colour channels (alpha, if any, is kept as it is)."""
    im = img if isinstance(img, Image.Image) else Image.fromarray(np.asarray(img))
    if not enabled() or min(im.size) < MIN_SIDE:
        return im
    try:
        from imwatermark import WatermarkEncoder
    except Exception:  # noqa: BLE001   (library missing: never block a result over the mark)
        return im
    alpha = im.getchannel("A") if im.mode == "RGBA" else None
    bgr = cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR)
    enc = WatermarkEncoder()
    enc.set_watermark("bytes", PAYLOAD)
    try:
        out = Image.fromarray(cv2.cvtColor(enc.encode(bgr, METHOD), cv2.COLOR_BGR2RGB))
    except Exception:  # noqa: BLE001
        return im
    if alpha is not None:
        out.putalpha(alpha)
    return out


def detect(img: Image.Image | np.ndarray) -> dict:
    """{"found": bool, "match": share of payload bits recovered (0.5 = chance)}."""
    from imwatermark import WatermarkDecoder

    im = img if isinstance(img, Image.Image) else Image.fromarray(np.asarray(img))
    if min(im.size) < MIN_SIDE:
        return {"found": False, "match": None, "note": "image too small to carry the mark"}
    bgr = cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR)
    try:
        got = WatermarkDecoder("bytes", len(PAYLOAD) * 8).decode(bgr, METHOD)
    except Exception:  # noqa: BLE001
        return {"found": False, "match": 0.0}
    a = np.unpackbits(np.frombuffer(PAYLOAD, np.uint8))
    b = np.unpackbits(np.frombuffer(bytes(got)[:len(PAYLOAD)].ljust(len(PAYLOAD), b"\0"), np.uint8))
    m = float((a == b).mean())
    return {"found": m >= FOUND_AT, "match": round(m, 3)}
