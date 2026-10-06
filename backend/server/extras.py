"""LumenAI v10 tools (added 2026-10-06 after comparing with the teammates' pipeline, team_pipeline.md).

  /api/v1/bokeh, /bokeh-adjust   portrait bokeh: Depth Anything V2 Small + subject mask -> depth-aware lens blur.
                                 The depth is cached, so the strength slider / refocus click re-render on the CPU only.
  /api/v1/layers                 dolly / parallax: subject cut-out + the background behind it (our smart removal fills
                                 it) + depth. The browser animates the camera (push-in, dolly zoom, orbit) and records
                                 the video itself - no GPU per frame.
  /api/v1/move                   reposition: the selection is lifted, the hole filled by smart removal, the object put
                                 down at the new place (optional scale)
  /api/v1/relight                light presets or your own words: Qwen relight -> our light transfer keeps every detail
  /api/v1/style                  reference photo (style described in words, v8) or a preset look
  /api/v1/product                product shoot: Qwen-Image 2.1 keeps the product exact (v9: DINO 0.90 vs klein 0.82)
  /api/v1/pose                   pose from a reference photo (words + joint angles, never shows Qwen the reference)
"""


import base64
import threading
from collections import OrderedDict

import cv2
import numpy as np
from PIL import Image

from server import web_api as A

_BOKEH: "OrderedDict[str, tuple]" = OrderedDict()
_BOKEH_LOCK = threading.Lock()


# ----------------------------------------------------------------------------- portrait bokeh


def _subject(E, img: Image.Image, mask: np.ndarray | None) -> np.ndarray | None:
    if mask is not None and mask.sum() > 16:
        return mask.astype(np.uint8)
    return A._ground_mask(E, img, "the main subject")


def _disc(r: float) -> np.ndarray:
    n = int(np.ceil(r))
    y, x = np.mgrid[-n:n + 1, -n:n + 1]
    k = np.clip(r + 0.5 - np.hypot(x, y), 0, 1).astype(np.float32)        # anti-aliased disc = lens aperture
    return k / k.sum()


def _lin(a):
    a = a.astype(np.float32) / 255
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def _srgb(a):
    a = np.clip(a, 0, 1)
    return (np.where(a <= 0.0031308, a * 12.92, 1.055 * a ** (1 / 2.4) - 0.055) * 255).round().astype(np.uint8)


def render_bokeh(rgb: np.ndarray, depth: np.ndarray, subject: np.ndarray | None, focus: float, strength: float,
                 levels: int = 7) -> np.ndarray:
    """Depth-aware lens blur. Blur radius grows with |depth - focus depth|; the subject stays sharp. Before blurring,
    the subject is painted out of the background layer so it cannot leak a halo into the blur, and bright spots are
    boosted in linear light so they turn into round bokeh discs like a real lens."""
    H, W = rgb.shape[:2]
    s = min(1.0, 1280 / max(H, W))                      # the blur is computed at <= 1280 px (it is blurry anyway)
    w, h = max(8, round(W * s)), max(8, round(H * s))
    small = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    d = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    sub = None if subject is None else cv2.resize(subject.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    rmax = 0.022 * np.hypot(w, h) * float(np.clip(strength, 0, 1))
    if rmax < 0.75:
        return rgb.copy()
    rad = np.clip(np.abs(d - focus) / 0.45, 0, 1) * rmax
    fill = small
    if sub is not None and sub.any():
        grow = cv2.dilate(sub, np.ones((9, 9), np.uint8))
        fill = cv2.inpaint(small, grow * 255, 7, cv2.INPAINT_TELEA)
        rad = np.where(sub > 0, 0, rad)
    L = _lin(fill)
    wt = 1 + 6 * np.clip((L.mean(2, keepdims=True) - 0.7) / 0.3, 0, 1)        # highlights weigh more -> bright discs
    LW = np.concatenate([L * wt, wt], 2)                                       # weighted average keeps the overall level

    def blur(r):
        if r < 0.5:
            return L
        b = cv2.filter2D(LW, -1, _disc(r), borderType=cv2.BORDER_REFLECT)
        return b[..., :3] / np.maximum(b[..., 3:], 1e-6)
    radii = np.linspace(0, rmax, levels)
    stack = [blur(r) for r in radii]
    pos = np.clip(rad / (radii[1] if levels > 1 else 1), 0, levels - 1 - 1e-6)
    i0 = np.floor(pos).astype(np.int32)
    t = (pos - i0)[..., None]
    out = np.zeros_like(L)
    for i in range(levels - 1):
        sel = (i0 == i)[..., None]
        out += sel * ((1 - t) * stack[i] + t * stack[i + 1])
    blurred = _srgb(out)
    big = cv2.resize(blurred, (W, H), interpolation=cv2.INTER_CUBIC)
    radf = cv2.resize(rad.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    a = np.clip(radf / 1.5, 0, 1)[..., None]                            # sharp where the radius is ~0
    if subject is not None:
        soft = cv2.GaussianBlur(subject.astype(np.float32), (0, 0), max(1.0, 0.0015 * np.hypot(W, H)))
        a = a * (1 - soft[..., None])
    return (rgb.astype(np.float32) * (1 - a) + big.astype(np.float32) * a).round().clip(0, 255).astype(np.uint8)


def lumen_bokeh(E, image: Image.Image, mask: np.ndarray | None, strength: float = 0.6, point=None) -> dict:
    import uuid

    Wk = A.Work(image)
    img = Wk.img
    depth = E.depth(img)
    sub = _subject(E, img, Wk.mask(mask))
    focus = _focus(depth, sub, point, Wk.s)
    out = render_bokeh(np.asarray(img), depth, sub, focus, strength)
    bid = uuid.uuid4().hex[:12]
    with _BOKEH_LOCK:
        _BOKEH[bid] = (Wk, np.asarray(img), depth, sub)
        while len(_BOKEH) > 6:
            _BOKEH.popitem(last=False)
    return {"image": A.data_url(Wk.back(out)), "route": "bokeh", "bokeh_id": bid, "focus": round(float(focus), 3),
            "subject": sub is not None, "checks": []}


def _focus(depth: np.ndarray, sub: np.ndarray | None, point, s: float) -> float:
    if point:
        x, y = int(point[0] * s), int(point[1] * s)
        y, x = np.clip(y, 0, depth.shape[0] - 1), np.clip(x, 0, depth.shape[1] - 1)
        return float(np.median(depth[max(0, y - 3):y + 4, max(0, x - 3):x + 4]))
    if sub is not None and sub.any():
        return float(np.median(depth[sub > 0]))
    return float(np.percentile(depth, 90))                 # nothing found: focus on the nearest things


def bokeh_adjust(bid: str, strength: float, point=None) -> dict:
    with _BOKEH_LOCK:
        hit = _BOKEH.get(bid)
    if hit is None:
        raise ValueError("That bokeh preview expired - press Apply again.")
    Wk, rgb, depth, sub = hit
    if point:                                               # refocus on what was clicked; no subject protection then
        focus = _focus(depth, None, point, Wk.s)
        out = render_bokeh(rgb, depth, None, focus, strength)
    else:
        focus = _focus(depth, sub, None, Wk.s)
        out = render_bokeh(rgb, depth, sub, focus, strength)
    return {"image": A.data_url(Wk.back(out)), "route": "bokeh", "bokeh_id": bid, "focus": round(float(focus), 3),
            "checks": []}


# ----------------------------------------------------------------------------- dolly / parallax layers


def _png_url(a: np.ndarray | Image.Image) -> str:
    return "data:image/png;base64," + base64.b64encode(A.png(a)).decode()


def lumen_layers(E, image: Image.Image, mask: np.ndarray | None) -> dict:
    img = A._small(image, 1280)
    m = None if mask is None else cv2.resize(mask.astype(np.uint8), img.size, interpolation=cv2.INTER_NEAREST)
    sub = _subject(E, img, m)
    if sub is None or sub.sum() < 64:
        raise ValueError("Select the subject first (click it), then press Make video.")
    plate, _ = A.remove_object_info(E, img, sub)
    plate = np.asarray(plate)
    d_img = E.depth(img)
    d_plate = E.depth(Image.fromarray(plate))
    alpha = cv2.GaussianBlur((sub > 0).astype(np.float32), (0, 0), 1.2)
    rgba = np.dstack([np.asarray(img), (alpha * 255).round().astype(np.uint8)])
    ys, xs = np.nonzero(sub)
    return {"plate": A.data_url(plate), "subject": _png_url(Image.fromarray(rgba, "RGBA")),
            "depth": _png_url(Image.fromarray((d_plate * 255).round().astype(np.uint8), "L")),
            "subject_depth": round(float(np.median(d_img[sub > 0])), 3),
            "box": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
            "size": list(img.size), "route": "dolly", "checks": []}


# ----------------------------------------------------------------------------- move an object


def lumen_move(E, image: Image.Image, mask: np.ndarray | None, dx: float, dy: float, scale: float = 1.0) -> dict:
    if mask is None or mask.sum() < 16:
        raise ValueError("Select the object to move first.")
    Wk = A.Work(image)
    img = np.asarray(Wk.img)
    m = Wk.mask(mask).astype(np.uint8)
    dx, dy, scale = float(dx) * Wk.s, float(dy) * Wk.s, float(np.clip(scale or 1.0, 0.3, 3.0))
    if abs(dx) < 2 and abs(dy) < 2 and abs(scale - 1) < 0.02:
        raise ValueError("Drag the selection to where it should go first.")
    plate, _ = A.remove_object_info(E, Wk.img, m)
    plate = np.asarray(plate)
    ys, xs = np.nonzero(m)
    cx, cy = xs.mean(), ys.mean()
    M = np.float32([[scale, 0, (1 - scale) * cx + dx], [0, scale, (1 - scale) * cy + dy]])
    H, W = m.shape
    obj = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    a = cv2.warpAffine(cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.0), M, (W, H), flags=cv2.INTER_LINEAR)
    out = (plate.astype(np.float32) * (1 - a[..., None]) + obj.astype(np.float32) * a[..., None]).round().astype(np.uint8)
    new = (a > 0.5).astype(np.uint8)
    region = A._dilate(np.maximum(m, new), 24)
    full_new = A._full_mask(new, image.size)
    nys, nxs = np.nonzero(full_new) if full_new is not None and full_new.any() else (np.array([0]), np.array([0]))
    return {"image": A.data_url(Wk.back(out, region)), "route": "move", "mask": A.mask_url(full_new),
            "box": [int(nxs.min()), int(nys.min()), int(nxs.max()) + 1, int(nys.max()) + 1], "checks": []}


# ----------------------------------------------------------------------------- relight

KEEP = "Keep every object, person, colour and position exactly the same."
LIGHTS = {
    "left": "Relight this photo so that a strong light source shines from the left side: the left side is brightly lit "
            "and the right side falls into shadow. " + KEEP,
    "right": "Relight this photo so that a strong light source shines from the right side: the right side is brightly "
             "lit and the left side falls into shadow. " + KEEP,
    "top": "Relight this photo with soft studio light from directly above, gentle shadows under objects. " + KEEP,
    "golden": "Relight this photo with warm golden-hour sunlight low from the side, long soft shadows. " + KEEP,
    "blue": "Relight this photo as cool blue hour twilight with soft ambient light. " + KEEP,
    "neon": "Relight this photo with pink and blue neon light from both sides, a night mood. " + KEEP,
    "dim": "Relight this photo as a dim evening scene lit only by a single soft lamp. " + KEEP,
}


def lumen_relight(E, image: Image.Image, light: str = "left", words: str = "", keep_detail: bool = True) -> dict:
    from editing import relight as RL

    if words.strip():
        E.guard(words)
        instr = f"Relight this photo: {words.strip().rstrip('.')}. {KEEP}"
    else:
        instr = LIGHTS.get(light) or LIGHTS["left"]
    Wk = A.Work(image)
    img = Wk.img
    pipe, Q = E.load()
    q = Q.edit(pipe, img, instr, seed=A.mix_seed(E.seed, img), steps=E.steps).image.convert("RGB")
    qa = np.asarray(q.resize(img.size, Image.LANCZOS))
    out = RL.transfer(np.asarray(img), qa) if keep_detail else qa
    return {"image": A.data_url(Wk.back(out)), "route": "relight", "checks": []}


# ----------------------------------------------------------------------------- style

STYLES = {
    "watercolour": "a loose watercolour painting with soft washes, visible paper texture and gentle bleeding edges",
    "oil": "a classical oil painting with rich impasto brushstrokes and warm glazes",
    "anime": "a clean anime illustration with crisp line art, cel shading and vivid colours",
    "pencil": "a detailed graphite pencil sketch with cross-hatching on white paper",
    "popart": "bold pop art with flat bright colours, thick outlines and halftone dots",
    "clay": "a soft 3-D clay render with rounded shapes and studio lighting",
    "film": "a 1970s film photograph with warm faded colours and fine grain",
}


def lumen_style(E, image: Image.Image, reference: Image.Image | None, preset: str = "", strength: float = 0.85) -> dict:
    if reference is not None:
        out = A.style_transfer(E, image, reference, strength)
        return {"image": A.data_url(out), "route": "style", "checks": []}
    look = STYLES.get(preset)
    if not look:
        raise ValueError("Pick a style or add a reference photo.")
    Wk = A.Work(image)
    instr = (f"Repaint this photo as {look}. Keep the content, composition, people and every object exactly where "
             f"they are; change only the rendering style.")
    o = E.flash_edit([Wk.img], instr) if E.tier == "flash" else E.edit([Wk.img], instr)
    out = np.asarray(o.resize(Wk.img.size, Image.LANCZOS))
    s = float(np.clip(strength, 0, 1))
    if s < 0.98:
        out = (np.asarray(Wk.img).astype(np.float32) * (1 - s) + out * s).round().astype(np.uint8)
    return {"image": A.data_url(Wk.back(out)), "route": "style", "checks": []}


# ----------------------------------------------------------------------------- product shoot

SCENES = {
    "marble": "on a white marble kitchen counter in soft morning window light",
    "cafe": "on a wooden table in a sunlit cafe, shallow depth of field",
    "studio": "on a pastel pink podium in a studio, soft light and a soft shadow, e-commerce product photo",
    "nature": "on a mossy rock beside a forest stream, dappled sunlight",
    "luxury": "on black velvet with dramatic rim light and soft reflections, luxury advertising photo",
    "beach": "on warm sand at the beach with turquoise water softly blurred behind",
}


def lumen_product(E, image: Image.Image, scene: str = "studio", words: str = "") -> dict:
    if words.strip():
        E.guard(words)
    where = words.strip() or SCENES.get(scene) or SCENES["studio"]
    Wk = A.Work(image)
    what = E.ask([A._small(Wk.img, 768)], "What product is shown? Answer with one to four words.", 12).strip(" .") or "product"
    instr = (f"Place this {what} {where}. Keep the {what} exactly the same: same shape, colours, logos, text and details. "
             f"Professional product photograph.")
    out = E.edit([Wk.img], instr)
    return {"image": A.data_url(Wk.back(np.asarray(out.resize(Wk.img.size, Image.LANCZOS)))), "route": "product",
            "product": what, "checks": []}


# ----------------------------------------------------------------------------- routes


def register(app, E, jrun, h):
    """h: helpers from build_lumen_routes (body, need, img_of, mask_of, quality)."""
    from fastapi import Request

    body, need, img_of, mask_of, quality = h["body"], h["need"], h["img_of"], h["mask_of"], h["quality"]

    def num(d, k, lo, hi, default):
        try:
            return float(np.clip(float(d.get(k, default)), lo, hi))
        except Exception:  # noqa: BLE001
            return default

    @app.post("/api/v1/bokeh")
    async def x_bokeh(req: Request):
        d = await body(req)
        try:
            im = need(d)
            m = mask_of(d, im.size)
        except ValueError as e:
            return jrun.bad(e)
        return await jrun("bokeh", lambda: lumen_bokeh(E, im, m, num(d, "strength", 0, 1, 0.6), d.get("point")))

    @app.post("/api/v1/bokeh-adjust")
    async def x_bokeh_adj(req: Request):
        d = await body(req)
        return await jrun("bokeh-adjust", lambda: bokeh_adjust(str(d.get("bokeh_id", "")),
                                                               num(d, "strength", 0, 1, 0.6), d.get("point")))

    @app.post("/api/v1/layers")
    async def x_layers(req: Request):
        d = await body(req)
        try:
            im = need(d)
            m = mask_of(d, im.size)
        except ValueError as e:
            return jrun.bad(e)
        return await jrun("layers", quality(d, lambda: lumen_layers(E, im, m)))

    @app.post("/api/v1/move")
    async def x_move(req: Request):
        d = await body(req)
        try:
            im = need(d)
            m = mask_of(d, im.size)
        except ValueError as e:
            return jrun.bad(e)
        return await jrun("move", quality(d, lambda: lumen_move(E, im, m, d.get("dx") or 0, d.get("dy") or 0,
                                                                  d.get("scale") or 1.0)))

    @app.post("/api/v1/relight")
    async def x_relight(req: Request):
        d = await body(req)
        try:
            im = need(d)
        except ValueError as e:
            return jrun.bad(e)
        return await jrun("relight", quality(d, lambda: lumen_relight(E, im, str(d.get("light", "left")),
                                                                        str(d.get("words", "")))))

    @app.post("/api/v1/style")
    async def x_style(req: Request):
        d = await body(req)
        try:
            im = need(d)
            ref = img_of(d, "reference")
        except ValueError as e:
            return jrun.bad(e)
        return await jrun("style", quality(d, lambda: lumen_style(E, im, ref, str(d.get("preset", "")),
                                                                    num(d, "strength", 0.2, 1, 0.85))))

    @app.post("/api/v1/product")
    async def x_product(req: Request):
        d = await body(req)
        try:
            im = need(d)
        except ValueError as e:
            return jrun.bad(e)
        return await jrun("product", quality(d, lambda: lumen_product(E, im, str(d.get("scene", "studio")),
                                                                        str(d.get("words", "")))))

    @app.post("/api/v1/pose")
    async def x_pose(req: Request):
        d = await body(req)
        try:
            im = need(d)
            ref = img_of(d, "reference")
        except ValueError as e:
            return jrun.bad(e)

        def go():
            out, info = A.pose_transfer(E, im, ref)
            return {"image": A.data_url(out), "route": "pose", "info": {k: v for k, v in info.items()
                                                                         if isinstance(v, (int, float, str, bool)) or v is None},
                    "checks": []}
        return await jrun("pose", quality(d, go))
