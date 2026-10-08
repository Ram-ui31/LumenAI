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
import io
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


IC_PROMPTS = {"left": ("bright light from the left side", "left"), "right": ("bright light from the right side", "right"),
              "top": ("soft studio light from above", "top"), "golden": ("warm golden hour sunlight", "left"),
              "blue": ("cool blue hour twilight light", "none"), "neon": ("pink and blue neon light, night", "none"),
              "dim": ("dim warm lamp light, evening", "none")}


def lumen_relight(E, image: Image.Image, light: str = "left", words: str = "", keep_detail: bool = True,
                  engine: str = "") -> dict:
    from editing import relight as RL

    if engine == "iclight":
        if words.strip():
            E.guard(words)
        prompt, direction = IC_PROMPTS.get(light, IC_PROMPTS["left"])
        if words.strip():
            prompt = words.strip()
        out = E.iclight(image, prompt, direction, 0.7)
        return {"image": A.data_url(out.resize(image.size, Image.LANCZOS)), "route": "relight", "model": "iclight",
                "checks": []}

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


def lumen_style(E, image: Image.Image, reference: Image.Image | None, preset: str = "", strength: float = 0.85,
                engine: str = "") -> dict:
    if reference is not None and engine == "magenta":
        Wk = A.Work(image)
        o = E.magenta(A._small(Wk.img, 1024), A._small(reference, 1024))
        out = np.asarray(o.resize(Wk.img.size, Image.LANCZOS)).astype(np.float32)
        s = float(np.clip(strength, 0, 1))
        if s < 0.98:
            out = np.asarray(Wk.img).astype(np.float32) * (1 - s) + out * s
        return {"image": A.data_url(Wk.back(out.round().astype(np.uint8))), "route": "style", "checks": []}
    if reference is not None and engine == "klein":
        Wk = A.Work(image)
        try:
            o = E.flash_edit([Wk.img, A._small(reference, 1024)], ART_STYLE_Q)
        except ValueError:                                 # Lumina not deployed: describe the style in words instead
            out = A.style_transfer(E, image, reference, strength)
            return {"image": A.data_url(out), "route": "style", "checks": []}
        out = np.asarray(o.resize(Wk.img.size, Image.LANCZOS)).astype(np.float32)
        s = float(np.clip(strength, 0, 1))
        if s < 0.98:
            out = np.asarray(Wk.img).astype(np.float32) * (1 - s) + out * s
        return {"image": A.data_url(Wk.back(out.round().astype(np.uint8))), "route": "style", "model": "flash", "checks": []}
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
    out = E.flash_edit([Wk.img], instr) if E.tier == "flash" else E.edit([Wk.img], instr)
    return {"image": A.data_url(Wk.back(np.asarray(out.resize(Wk.img.size, Image.LANCZOS)))), "route": "product",
            "product": what, "checks": []}


# ----------------------------------------------------------------------------- depth glow (team's depth-glow light model)

_GLOW: "OrderedDict[str, tuple]" = OrderedDict()


def _hex_rgb(c: str) -> tuple:
    c = str(c or "#ffb464").lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4)) if len(c) == 6 else (255, 180, 100)


def _glow_render(img: Image.Image, depth: np.ndarray, size: tuple, d: dict) -> np.ndarray:
    from editing.depth_glow import glow

    x, y = (d.get("point") or [size[0] / 2, size[1] / 2])[:2]
    out = glow(img, depth, position=(float(x) / size[0], float(y) / size[1]), color=_hex_rgb(d.get("colour")),
               mode=str(d.get("mode") or "spread"), intensity=float(np.clip(float(d.get("intensity", 65)), 0, 100)),
               size=float(np.clip(float(d.get("size", 0.5)), 0.05, 2.0)), direction=float(d.get("direction", 90)),
               mix=0.85)
    return np.asarray(out)


def lumen_glow(E, image: Image.Image, d: dict) -> dict:
    import uuid

    Wk = A.Work(image)
    depth = E.depth(Wk.img)
    out = _glow_render(Wk.img, depth, image.size, d)
    gid = uuid.uuid4().hex[:12]
    with _BOKEH_LOCK:
        _GLOW[gid] = (Wk, depth)
        while len(_GLOW) > 6:
            _GLOW.popitem(last=False)
    return {"image": A.data_url(Wk.back(out)), "route": "glow", "glow_id": gid, "checks": [], **_glow_live(Wk.img, depth)}


def _glow_live(img: Image.Image, depth: np.ndarray, side: int = 1024) -> dict:
    """What the editor needs to preview the light live under the cursor (WebGL): the photo, an edge-preserving
    denoised copy (as in depth_glow's reveal) and the depth map, all at most ``side`` px."""
    import cv2

    from editing.depth_glow import linear_to_srgb, srgb_to_linear

    k = min(1.0, side / max(img.size))
    w, h = max(1, round(img.width * k)), max(1, round(img.height * k))
    small = img.convert("RGB").resize((w, h), Image.LANCZOS)
    lin = srgb_to_linear(np.asarray(small, np.float32) / 255).astype(np.float32)
    den = (linear_to_srgb(cv2.bilateralFilter(lin, 9, 0.02, 4)) * 255).round().astype(np.uint8)
    d = np.asarray(Image.fromarray(np.asarray(depth, np.float32)).resize((w, h), Image.BILINEAR))
    d8 = (np.clip(d, 0, 1) * 255).round().astype(np.uint8)
    b = io.BytesIO()
    Image.fromarray(d8).save(b, "PNG")
    return {"live": {"photo": A.data_url(small), "den": A.data_url(Image.fromarray(den)),
                     "depth": "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()}}


def glow_adjust(d: dict) -> dict:
    with _BOKEH_LOCK:
        hit = _GLOW.get(str(d.get("glow_id", "")))
    if hit is None:
        raise ValueError("That light preview expired - press Add light again.")
    Wk, depth = hit
    out = _glow_render(Wk.img, depth, Wk.full.size, d)
    return {"image": A.data_url(Wk.back(out)), "route": "glow", "glow_id": d.get("glow_id"), "checks": []}


# ----------------------------------------------------------------------------- auto collage (team's collage_v0)

def auto_collage(images: list, prompt: str = "", title: str = "", size: str = "post") -> dict:
    import tempfile
    from pathlib import Path

    from editing import collage_auto as C

    if len(images) < 2:
        raise ValueError("Add at least two photos for an automatic collage.")
    with tempfile.TemporaryDirectory() as td:
        paths = []
        for i, im in enumerate(images[:10]):
            p = Path(td) / f"photo_{i:02d}.jpg"
            im.convert("RGB").save(p, quality=95)
            paths.append(p)
        import torch

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        ims, recs, model = C.analyze_photos(paths, "openai/clip-vit-base-patch32", dev, 8, False, False, prompt)
        sel = C.select_photos(recs, 0.94, max(2, min(8, len(recs))))
        out = Path(td) / "out"
        out.mkdir()
        info = C.render_all([ims[i] for i in sel], [recs[i] for i in sel], out, 2026, prompt, title, "", "auto")
        size = size if size in C.SIZES else "post"
        designs = [{"style": st, "image": A.data_url(Image.open(out / info["renders"][st][size]["file"]).convert("RGB"))}
                   for st in ("editorial", "minimal", "scrapbook")]
    return {"designs": designs, "title": info["prompt_design"].get("title", ""), "route": "collage", "checks": []}


# ----------------------------------------------------------------------------- product angles (FLUX.2 klein, Lumina)

PRODUCT_STYLE = ("photorealistic product photo of the same exact object, identical shape, colors and natural surface "
                 "texture, soft studio lighting, clean white background, sharp focus, natural shadow")
ANGLES = [("45° left", "Show the product rotated 45 degrees to the left."),
          ("Side", "Show the product rotated 90 degrees, side view."),
          ("Back", "Show the product from behind, back view."),
          ("Top", "Show the product from a high top-down angle.")]


def product_angles(E, image: Image.Image) -> dict:
    Wk = A.Work(image)
    out = []
    for k, (name, how) in enumerate(ANGLES):
        E.seed = A.SEED + k
        o = E.flash_edit([Wk.img], f"{PRODUCT_STYLE}. {how}")
        out.append({"angle": name, "image": A.data_url(o)})
    return {"angles": out, "route": "product", "model": "flash", "checks": []}


# ----------------------------------------------------------------------------- put a captured 3-D view in place of the object

def replace_with_view(E, image: Image.Image, mask: np.ndarray | None, view_rgba: Image.Image, harmonise: bool = False) -> dict:
    if mask is None or mask.sum() < 16:
        raise ValueError("The object's outline is missing - build the 3D model from a selection first.")
    Wk = A.Work(image)
    m = Wk.mask(mask).astype(np.uint8)
    plate, _ = A.remove_object_info(E, Wk.img, m)
    plate = np.asarray(plate).astype(np.float32)
    v = view_rgba.convert("RGBA")
    a = np.asarray(v)[..., 3]
    ys, xs = np.nonzero(a > 8)
    if not len(xs):
        raise ValueError("The captured view is empty.")
    v = v.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    oy, ox = np.nonzero(m)
    bx0, by0, bx1, by1 = ox.min(), oy.min(), ox.max() + 1, oy.max() + 1
    s = min((bx1 - bx0) / v.width, (by1 - by0) / v.height) * 1.0
    nw, nh = max(1, round(v.width * s)), max(1, round(v.height * s))
    v = v.resize((nw, nh), Image.LANCZOS)
    x0 = int((bx0 + bx1) / 2 - nw / 2)
    y0 = int(by1 - nh)                                   # stand on the same ground line as the old object
    layer = Image.new("RGBA", Wk.img.size, (0, 0, 0, 0))
    layer.paste(v, (x0, y0), v)
    L = np.asarray(layer).astype(np.float32)
    al = L[..., 3:4] / 255
    out = (plate * (1 - al) + L[..., :3] * al).round().clip(0, 255).astype(np.uint8)
    new = (al[..., 0] > 0.5).astype(np.uint8)
    region = A._dilate(np.maximum(m, new), 24)
    k = 1.0 / Wk.s                                       # working size -> full size
    placed = {"view": A.b64png(v), "cx": round((x0 + nw / 2) * k, 1), "cy": round((y0 + nh / 2) * k, 1),
              "w": round(nw * k, 1), "h": round(nh * k, 1)}
    plate_full = Wk.back(plate.round().clip(0, 255).astype(np.uint8), A._dilate(m, 24))
    return {"image": A.data_url(Wk.back(out, region)), "route": "replace3d", "mask": A.mask_url(A._full_mask(new, image.size)),
            "plate": A.data_url(plate_full), "placed": placed, "checks": []}


ART_STYLE_Q = ("Repaint image 1 in the artistic style of image 2: its colour palette, brushwork, texture and light. Keep "
               "the subject, composition, camera angle and every recognisable detail of image 1. Do not copy objects or "
               "the composition of image 2.")


SCENE_Q = ("List the separate objects you can see in this photo, including small and background ones (people, furniture, "
           "signs, wires, bins, cups, cars, plants, litter...). For each give a short specific name, where it is "
           "(e.g. 'left edge, on the table'), and its role: 'subject', 'supporting' or 'distraction' (something a photo "
           "editor would usually remove: clutter, a stray person, a cable, a bin, a sign, a reflection). Also give the "
           "setting in a few words and the empty surfaces where something could stand. Reply with JSON only: "
           '{"setting": "...", "objects": [{"name": "...", "where": "...", "role": "..."}], "free_surfaces": ["..."]}')
SUGGEST_Q = ("You are a professional photo retoucher. The photo shows: {scene}\n"
             "Suggest {n} different OBJECT-LEVEL edits for this exact photo, each naming a specific object above:\n"
             "- remove: take out distractions (always first when there are any), e.g. 'remove the trash bin on the left';\n"
             "- add: one object that fits the setting, on one of the free surfaces, with its place, e.g. 'add a small "
             "potted fern on the empty shelf on the right';\n"
             "- replace: swap one object for a better one of similar size;\n"
             "- move: put one object somewhere better for the composition;\n"
             "- recolour: change the colour or material of one named object.\n"
             "Use at least three different kinds, at most one edit that changes only light or colour of the whole photo, "
             "and never an object that is not in the list (except for 'add'). Each prompt must be one clear instruction "
             "under 20 words that starts with the verb.{avoid} Reply with JSON only: "
             '[{{"kind": "remove|add|replace|move|recolour|look", "title": "2 to 4 words", "prompt": "the instruction", '
             '"why": "one short sentence"}}, ...]')
SUGGEST_FALLBACK = [
    {"title": "Warm golden light", "prompt": "give the photo warm golden-hour light", "why": "Warmer light flatters most scenes.", "kind": "look"},
    {"title": "Cleaner background", "prompt": "remove small distractions from the background", "why": "Keeps the eye on the subject.", "kind": "remove"},
    {"title": "Richer colour", "prompt": "make the colours richer and the contrast a little stronger", "why": "Adds punch without changing the scene.", "kind": "look"},
    {"title": "Soft background blur", "prompt": "blur the background softly like a portrait lens", "why": "Separates the subject.", "kind": "look"},
]


def lumen_suggest(E, image: Image.Image, n: int = 6, avoid: list | None = None) -> dict:
    """Photo-specific edit suggestions from Qwen3-VL: [{title, prompt, why}]. Each one runs as a normal edit."""
    import json
    import re

    Wk = A.Work(image)
    av = [str(x)[:60] for x in (avoid or []) if str(x).strip()][:20]
    small = A._small(Wk.img, 1024)
    scene_raw = E.ask([small], SCENE_Q, 600) or ""          # pass 1: what objects are in the photo, and where
    sm = re.search(r"\{.*\}", scene_raw, re.S)
    try:
        sc = json.loads(sm.group(0)) if sm else {}
    except Exception:  # noqa: BLE001
        sc = {}
    objs = [o for o in (sc.get("objects") or []) if isinstance(o, dict) and str(o.get("name", "")).strip()][:25]
    scene = (f"setting: {str(sc.get('setting', ''))[:80]}; objects: "
             + "; ".join(f"{str(o['name'])[:40]} ({str(o.get('where', ''))[:40]}, {str(o.get('role', ''))[:12]})" for o in objs)
             + "; free surfaces: " + ", ".join(str(x)[:40] for x in (sc.get("free_surfaces") or [])[:6])) if objs \
        else "(look at the photo yourself and name the objects you see)"
    q = SUGGEST_Q.format(scene=scene, n=n, avoid=(" Do not repeat these: " + "; ".join(av) + ".") if av else "")
    raw = E.ask([small], q, 800)                              # pass 2: object-level edits that name those objects
    m = re.search(r"\[.*\]", raw or "", re.S)
    try:
        items = json.loads(m.group(0)) if m else []
    except Exception:  # noqa: BLE001
        items = []
    out, seen = [], {x.lower() for x in av}
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        pr = str(it.get("prompt", "")).strip()[:200]
        t = str(it.get("title", "")).strip()[:48] or pr[:40]
        if not pr or t.lower() in seen:
            continue
        seen.add(t.lower())
        kind = str(it.get("kind", "")).strip().lower()
        out.append({"title": t, "prompt": pr, "why": str(it.get("why", "")).strip()[:160],
                    "kind": kind if kind in ("remove", "add", "replace", "move", "recolour", "look") else "edit"})
    generic = not out
    if generic:
        out = [x for x in SUGGEST_FALLBACK if x["title"].lower() not in seen]
    return {"suggestions": out[:n], "generic": generic, "route": "suggest", "checks": []}


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

    @app.post("/api/v1/suggest")
    async def x_suggest(req: Request):
        d = await body(req)
        try:
            im = need(d)
        except ValueError as e:
            return jrun.bad(e)
        return await jrun("suggest", lambda: lumen_suggest(E, im, int(num(d, "n", 1, 8, 6)), d.get("avoid") or []))

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
                                                                        str(d.get("words", "")), True,
                                                                        str(d.get("engine", "")))))

    @app.post("/api/v1/style")
    async def x_style(req: Request):
        d = await body(req)
        try:
            im = need(d)
            ref = img_of(d, "reference")
        except ValueError as e:
            return jrun.bad(e)
        return await jrun("style", quality(d, lambda: lumen_style(E, im, ref, str(d.get("preset", "")),
                                                                    num(d, "strength", 0.2, 1, 0.85),
                                                                    str(d.get("engine", "")))))

    @app.post("/api/v1/product")
    async def x_product(req: Request):
        d = await body(req)
        try:
            im = need(d)
        except ValueError as e:
            return jrun.bad(e)
        return await jrun("product", quality(d, lambda: lumen_product(E, im, str(d.get("scene", "studio")),
                                                                        str(d.get("words", "")))))

    @app.post("/api/v1/glow")
    async def x_glow(req: Request):
        d = await body(req)
        try:
            im = need(d)
        except ValueError as e:
            return jrun.bad(e)
        return await jrun("glow", lambda: lumen_glow(E, im, d))

    @app.post("/api/v1/glow-adjust")
    async def x_glow_adj(req: Request):
        d = await body(req)
        return await jrun("glow-adjust", lambda: glow_adjust(d))

    @app.post("/api/v1/auto-collage")
    async def x_collage(req: Request):
        from server.web_api import read_data_url, read_image

        d = await body(req)
        try:
            ims = [read_image(b) for b in (read_data_url(u) for u in (d.get("images") or [])[:10]) if b]
        except ValueError as e:
            return jrun.bad(e)
        return await jrun("auto-collage", lambda: auto_collage(ims, str(d.get("prompt", "")), str(d.get("title", "")),
                                                               str(d.get("size", "post"))))

    @app.post("/api/v1/product-angles")
    async def x_angles(req: Request):
        d = await body(req)
        try:
            im = need(d)
        except ValueError as e:
            return jrun.bad(e)
        return await jrun("product-angles", quality(d, lambda: product_angles(E, im)))

    @app.post("/api/v1/replace-object")
    async def x_replace(req: Request):
        d = await body(req)
        try:
            im = need(d)
            m = mask_of(d, im.size)
            view = img_of(d, "view")
            if view is None:
                raise ValueError("No captured view was sent.")
            from server.web_api import read_data_url

            raw = Image.open(__import__("io").BytesIO(read_data_url(d["view"])))
            raw.load()
        except ValueError as e:
            return jrun.bad(e)
        return await jrun("replace-object", quality(d, lambda: replace_with_view(E, im, m, raw)))

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
