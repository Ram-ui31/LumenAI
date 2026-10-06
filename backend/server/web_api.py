"""LumenAI web API (FastAPI). Two families of endpoints share one engine (Qwen-Image 2.1 + Qwen3-VL + SAM 2.1):

Multipart endpoints (form data in, PNG out; JSON {"variations": [...]} for inpaint; errors as JSON {"detail": ...}):

  /api/v1/remove-background  Qwen3-VL finds the main subject -> SAM 2.1 outline -> soft edge (guided filter) -> RGBA PNG
  /api/v1/remove-object      smart removal (v5): Qwen3-VL reads the selection, draft@512 -> sharpen@768, self-check +
                             3 fallbacks, shadow trick, Laplacian seam blend
  /api/v1/inpaint            "ghost" fill (v5 adder, 27/32): selection greyed out, "replace the grey area with <prompt>",
                             N seeds -> N variations; optional reference image as <image2>
  /api/v1/outpaint           community Outpaint LoRA v2, 10 steps (v8 winner over our keep-outside method)
  /api/v1/edit-prompt        routes by what the words ask for:
                               pose reference from the Pose tool -> precise joint words + correction loop (v7)
                               "blurry / sharpen / while running" -> deblur with the precise prompt (v6 best)
                               "remove the X"                     -> Qwen3-VL box -> SAM -> smart removal
                               "make the X <colour>"              -> our OKLab recolour (48-task benchmark: leak 0)
                               anything else                      -> Qwen edit; with a mask, only inside the mask
  /api/v1/style-transfer     Qwen3-VL describes the reference style in words -> single-image Qwen edit
  /api/v1/upscale            Lanczos + light sharpening (NOT an AI model of ours yet - says so in a header)
  /api/v1/health             model loaded? which routes exist

LumenAI editor (frontend/editor.html) - JSON in/out, images as data URLs ("checks" kept empty for old clients):
  /api/v1/edit            prompt -> the routes above
  /api/v1/segment         click points -> SAM 2.1 mask        /api/v1/recolour, /recolour-shade   colour slider (cached)
  /api/v1/expand          aspect ratio -> Outpaint LoRA       /api/v1/ideas, /place, /add         ideas -> spot+size
  /api/v1/cutout          transparent PNG                     /api/v1/view3d                      GLB via adobe-ps-object3d
  /api/v1/trends          Gemini + Google Search -> trending looks as ready-to-run edit prompts (server/trends.py)
  /api/v1/upscale-ai      TADSR x2/x4 (modal_tadsr.py)        /api/v1/verify                      LumenAI mark present?
Exports (/api/v1/export) carry an invisible LumenAI watermark (server/watermark.py); every prompt passes the request
guard first (server/guard.py: deberta injection classifier + our Qwen3-VL policy check).

Heavy imports happen inside functions so the module imports without a GPU (tests use a fake engine).
"""

import base64
import io
import os
import re
import threading
import time
import traceback
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image

MAX_SIDE = int(os.environ.get("ADOBEPS_API_MAX_SIDE", "2048"))   # bigger uploads are processed at this size
STEPS = int(os.environ.get("ADOBEPS_API_STEPS", "40"))
SEED = 7
KEEP_REST = "Keep everything else exactly the same: same people, objects, colours, lighting and positions."
COLOUR_WORDS = None   # filled lazily from editing.recolour.COLOURS


# ----------------------------------------------------------------------------- engine (model holder)


class Engine:
    """Holds the Qwen-Image 2.1 pipeline (nf4 + text encoder offload, no VAE tiling = our tested setting)."""

    def __init__(self):
        self.pipe = None
        self.Q = None
        self.lock = threading.Lock()
        self.loaded_at = None
        self.steps = STEPS          # per request: Lumen / Lumen Flash use QUICK_STEPS for Qwen work
        self.seed = SEED            # per request: "Try another version" = a new seed
        self.tier = "lumen"         # per request: flash | lumen | pro (TIERS)
        self._depth = None
        self.remote_peak = None     # set by calls that run in another container (Lumina, TADSR, 3-D)

    def load(self):
        if self.pipe is None:
            from editing import qwen as Q

            t0 = time.perf_counter()
            self.Q = Q
            self.pipe = Q.load_pipe("nf4")
            Q.enable_memory_savers(self.pipe)
            self.loaded_at = round(time.perf_counter() - t0, 1)
        return self.pipe, self.Q

    def ask(self, images: list[Image.Image], question: str, n: int = 128) -> str:
        pipe, Q = self.load()
        return Q.ask(pipe, images, question, max_new_tokens=n)

    def yes(self, images: list[Image.Image], question: str) -> bool:
        return self.ask(images, question + " Answer yes or no.", 3).strip().lower().startswith("yes")

    def edit(self, images: list[Image.Image], instruction: str, seed: int | None = None, resolution: int = 1024) -> Image.Image:
        """Qwen edit with one or more input images; result resized to the first image's size."""
        import torch

        seed = mix_seed(self.seed if seed is None else seed, images[0])
        pipe, Q = self.load()
        if len(images) == 1:
            out = Q.edit(pipe, images[0], instruction, seed=seed, steps=self.steps, resolution=resolution).image
            return _flatten(out)
        out = pipe(prompt=instruction, image=[im.convert("RGB") for im in images], num_inference_steps=self.steps,
                   output_resolution=resolution, generator=torch.Generator("cpu").manual_seed(seed)).images[0]
        return _flatten(out).resize(images[0].size, Image.LANCZOS)

    def outpaint_lora(self, image: Image.Image, tw: int, th: int, ox: int, oy: int, scene: str) -> Image.Image:
        """Community Outpaint LoRA v2, 10 steps (v8: closest to the real borders, 0.364 vs 0.505 LPIPS, and faster)."""
        from editing import outpaint_lora as OL

        pipe, Q = self.load()
        OL.load_lora(pipe)
        out, _ = OL.outpaint(pipe, Q, image, tw, th, ox, oy, scene, steps=10, seed=mix_seed(self.seed, image))
        return out

    def make3d(self, rgba: Image.Image) -> tuple[bytes, dict]:
        """Hunyuan3D-2 runs in its own Modal app (adobe-ps-object3d, deployed function make3d): its CUDA build and
        16 GB of weights never share this container's GPU with Qwen."""
        import modal

        self.remote_peak = self.REMOTE_PEAK_GB["hunyuan3d"]
        try:
            f = modal.Function.from_name("adobe-ps-object3d", "make3d")
            f.hydrate()
        except Exception as e:  # noqa: BLE001  (modal.exception.NotFoundError when the 3-D app is not deployed)
            if "not found" in str(e).lower():
                raise ValueError("The 3D service is not deployed yet. In the project folder run: "
                                 "modal deploy modal_object3d.py") from e
            raise
        r = f.remote(png(rgba))
        return r["glb"], r.get("info", {})

    def generate(self, prompt: str, w: int, h: int) -> Image.Image:
        """Text-to-image with the same Qwen-Image 2.1 pipeline (no extra model)."""
        import torch

        pipe, Q = self.load()
        out = pipe(prompt=prompt, height=h, width=w, num_inference_steps=self.steps,
                   generator=torch.Generator("cpu").manual_seed(mix_seed(self.seed, prompt))).images[0]
        return _flatten(out)

    # peak GPU memory of the models that run in their own containers (measured: flash test 2026-10-06, v9 benchmark);
    # the editor shows it as "Peak VRAM" for requests that ran there
    REMOTE_PEAK_GB = {"flash": 13.35, "tadsr": 5.7, "hunyuan3d": 11.5}

    # --- Lumen Flash: FLUX.2 klein 4B in its own container (modal_flash.py) - it would not fit next to Qwen in 16 GB
    def _flash(self):
        return _remote_cls("adobe-ps-flash", "Flash", "modal_flash.py", "Lumen Flash")

    def flash_generate(self, prompt: str, w: int, h: int) -> Image.Image:
        self.remote_peak = self.REMOTE_PEAK_GB["flash"]
        out = self._flash().generate.remote(prompt, int(w), int(h), mix_seed(self.seed, prompt))
        return Image.open(io.BytesIO(out)).convert("RGB")

    def flash_edit(self, images: list[Image.Image], instruction: str) -> Image.Image:
        self.remote_peak = self.REMOTE_PEAK_GB["flash"]
        out = self._flash().edit.remote([png(im.convert("RGB")) for im in images], instruction,
                                        mix_seed(self.seed, images[0]))
        return Image.open(io.BytesIO(out)).convert("RGB")

    # --- Lumen Pro: Qwen3-VL turns a short request into the precise prompt the image model follows best
    def rewrite_t2i(self, prompt: str) -> str:
        return _clean_rewrite(self.ask([], T2I_REWRITE.format(p=prompt), 260), prompt)

    def rewrite_edit(self, image: Image.Image, instruction: str) -> str:
        return _clean_rewrite(self.ask([_small(image, 768)], EDIT_REWRITE.format(p=instruction), 140), instruction)

    # --- depth (bokeh, dolly): Depth Anything V2 Small (Apache-2.0, 25 M parameters) in this container
    def depth(self, image: Image.Image) -> np.ndarray:
        """Relative depth, float32 0..1, 1 = nearest."""
        if self._depth is None:
            from transformers import pipeline

            self._depth = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf",
                                   device=0 if os.environ.get("ADOBEPS_DEVICE", "cuda") == "cuda" else -1)
        im = _small(image, 1024)
        d = np.asarray(self._depth(im)["predicted_depth"].squeeze().float().cpu().numpy(), np.float32)
        d = cv2.resize(d, image.size, interpolation=cv2.INTER_CUBIC)
        lo, hi = np.percentile(d, 1), np.percentile(d, 99)
        return np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)

    def guard(self, text: str) -> None:
        """Raise ValueError (-> HTTP 400, shown in the editor) if the request must not be run (server/guard.py)."""
        from server import guard as G

        def ask(q):
            try:
                return self.ask([], q, 4)
            except Exception:  # noqa: BLE001   (processor wants an image)
                return self.ask([Image.new("RGB", (64, 64), "white")], q, 4)
        ok, kind, msg = G.check(ask, text)
        if not ok:
            raise ValueError(msg)

    def upscale_ai(self, img: Image.Image, scale: int) -> Image.Image:
        """TADSR in its own Modal app (modal_tadsr.py): old torch pins, so it cannot share this container."""
        import modal

        self.remote_peak = self.REMOTE_PEAK_GB["tadsr"]
        try:
            up = modal.Cls.from_name("adobe-ps-tadsr", "Upscaler")()
            out = up.upscale.remote(png(img.convert("RGB")), int(scale))
        except Exception as e:  # noqa: BLE001
            if "not found" in str(e).lower():
                raise ValueError("The AI upscaler is not deployed yet. In the project folder run: "
                                 "modal deploy modal_tadsr.py") from e
            raise
        return Image.open(io.BytesIO(out)).convert("RGB")

    def refine(self, structure: Image.Image, prompt: str, seed: int, strength: float, resolution: int = 1024,
               keep=None, condition=None) -> Image.Image:
        pipe, Q = self.load()
        return _flatten(Q.refine(pipe, structure, prompt, seed=seed, strength=strength, steps=self.steps,
                                 resolution=resolution, keep=keep, condition=condition).image)


def _remote_cls(app_name: str, cls: str, deploy_file: str, label: str):
    import modal

    try:
        c = modal.Cls.from_name(app_name, cls)
        c.hydrate()
        return c()
    except Exception as e:  # noqa: BLE001
        if "not found" in str(e).lower():
            raise ValueError(f"{label} is not deployed yet. In the project folder run: modal deploy {deploy_file}") from e
        raise


T2I_REWRITE = ("You write prompts for an image generator. Rewrite the request below into one detailed English paragraph "
               "(60 to 110 words) that describes the finished image as if looking at it: main subject, setting, "
               "composition and framing, lighting, colours, materials, camera and lens or art medium, mood. Keep every "
               "object, count, colour, position and piece of text (in double quotes) the request gives; decide the rest "
               "tastefully. Reply with the paragraph only.\nRequest: {p}")
EDIT_REWRITE = ("You rewrite photo-editing instructions for an image-editing model. Look at the photo and rewrite the "
                "instruction below as one precise English instruction of at most 60 words: say exactly which object or "
                "area changes and how (where, size, colour, material, pose), refer to things as they appear in this "
                "photo, keep any text to be written in double quotes, and end by saying what must stay unchanged. "
                "Reply with the instruction only.\nInstruction: {p}")


def _clean_rewrite(out: str, original: str) -> str:
    t = re.sub(r"^(rewritten( prompt| instruction)?|prompt|instruction)\s*:\s*", "", (out or "").strip(), flags=re.I)
    t = t.strip().strip('"').strip()
    return t if len(t.split()) >= max(4, len(original.split())) else original


# ----------------------------------------------------------------------------- image helpers


def _flatten(img: Image.Image) -> Image.Image:
    """Qwen-Image 2.1 decodes RGBA; where it chose transparency the RGB values are garbage (v7 completion test:
    magenta noise). Composite onto white instead of a plain convert('RGB')."""
    if img.mode != "RGBA":
        return img.convert("RGB")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, img).convert("RGB")


def read_image(data: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(data))
    im.load()
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    return im.convert("RGB")


def read_mask(data: bytes, size: tuple[int, int]) -> np.ndarray:
    """The UI's mask canvas: coloured semi-transparent brush strokes on a transparent canvas -> {0,1}."""
    im = Image.open(io.BytesIO(data))
    im.load()
    if im.mode in ("RGBA", "LA", "P"):
        a = np.asarray(im.convert("RGBA"))[..., 3]
        m = a > 20 if a.min() < 250 else np.asarray(im.convert("L")) > 127
    else:
        m = np.asarray(im.convert("L")) > 127
    m = Image.fromarray(m.astype(np.uint8) * 255).resize(size, Image.NEAREST)
    return (np.asarray(m) > 0).astype(np.uint8)


def png(img: Image.Image | np.ndarray) -> bytes:
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    b = io.BytesIO()
    img.save(b, "PNG")
    return b.getvalue()


def b64png(img) -> str:
    return "data:image/png;base64," + base64.b64encode(png(img)).decode()


def mix_seed(seed: int, key) -> int:
    """The noise seed actually used: the user's seed mixed with what is being edited (an image) or generated (a prompt).

    Without this, "generate" at seed 7 and then "edit" at seed 7 started the edit from the very noise that had just
    produced the picture: Qwen slid back to the same image (slightly softer) and ignored the request ("add a woman on
    the sofa" added nothing), while "Another version" (seed 8) worked. Chained edits had the same trap. Mixing in the
    input keeps runs reproducible (same seed + same input = same result) but never reuses the image's own noise."""
    import hashlib

    if isinstance(key, Image.Image):
        t = key.convert("RGB").resize((32, 32), Image.BILINEAR)
        data = t.tobytes()
    else:
        data = str(key).encode()
    h = hashlib.sha256(int(seed).to_bytes(8, "little", signed=True) + data).digest()
    return int.from_bytes(h[:4], "little") & 0x7FFFFFFF


class Work:
    """Process big uploads at MAX_SIDE, then put the result back at full size (only inside ``region`` when given)."""

    def __init__(self, image: Image.Image):
        self.full = image
        w, h = image.size
        self.s = min(1.0, MAX_SIDE / max(w, h))
        self.img = image if self.s == 1.0 else image.resize((round(w * self.s), round(h * self.s)), Image.LANCZOS)

    def mask(self, m: np.ndarray | None) -> np.ndarray | None:
        if m is None or self.s == 1.0:
            return m
        return (np.asarray(Image.fromarray(m * 255).resize(self.img.size, Image.NEAREST)) > 0).astype(np.uint8)

    def back(self, result, region: np.ndarray | None = None) -> np.ndarray:
        r = np.asarray(result.convert("RGB") if isinstance(result, Image.Image) else result)
        if self.s == 1.0:
            return r
        up = np.asarray(Image.fromarray(r).resize(self.full.size, Image.LANCZOS)).astype(np.float32)
        if region is None:
            return up.astype(np.uint8)
        a = np.asarray(Image.fromarray(region.astype(np.uint8) * 255).resize(self.full.size, Image.BILINEAR))
        a = cv2.GaussianBlur(a.astype(np.float32) / 255, (0, 0), 3)[..., None]
        O = np.asarray(self.full).astype(np.float32)
        return (O * (1 - a) + up * a).round().clip(0, 255).astype(np.uint8)


def _dilate(m: np.ndarray, px: int) -> np.ndarray:
    from editing.masking import dilate

    return dilate(m, px)


def _composite(O: np.ndarray, G, hole: np.ndarray, region: np.ndarray) -> np.ndarray:
    from editing.blending import composite_back

    return composite_back(O, G if isinstance(G, Image.Image) else Image.fromarray(G), hole, region)


# ----------------------------------------------------------------------------- features


def remove_background(E: Engine, image: Image.Image, refine_edges: bool = True) -> Image.Image:
    from editing import recolour as RC
    from editing.relight import _guided

    W = Work(image)
    img = W.img
    box = RC.ground_box(lambda p: E.ask([img], p, 60), img.size, "the main subject")
    if box is None:
        w, h = img.size
        box = (int(0.02 * w), int(0.02 * h), int(0.98 * w), int(0.98 * h))
    m = RC.box_to_mask(img, box).astype(np.float32)
    if m.sum() < 16:
        raise ValueError("Could not find a main subject in this photo.")
    if W.s != 1.0:
        m = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)).astype(np.float32) / 255
    rgb = np.asarray(image)
    if refine_edges:   # edge-aware soft alpha: follows hair / fur edges of the photo instead of SAM's hard outline
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
        r = max(2, int(max(image.size) / 400))
        a = np.clip(_guided(g, m, r, eps=1e-4), 0, 1)
        a = np.where(cv2.dilate((m > 0.5).astype(np.uint8), np.ones((2 * r + 1,) * 2, np.uint8)) > 0, a, 0)
    else:
        a = cv2.GaussianBlur(m, (0, 0), 0.8)
    return Image.fromarray(np.dstack([rgb, (a * 255).round().astype(np.uint8)]), "RGBA")


def remove_object(E: Engine, image: Image.Image, mask: np.ndarray) -> np.ndarray:
    return remove_object_info(E, image, mask)[0]


def remove_object_info(E: Engine, image: Image.Image, mask: np.ndarray) -> tuple[np.ndarray, dict]:
    """Smart removal; also returns what the self-check saw (attempts, chosen method, shadow pixels)."""
    from editing import smart_remove as SR
    from editing.masking import reshape, shadow_extend

    if mask.sum() < 4:
        raise ValueError("Paint over the object you want to remove first.")
    W = Work(image)
    obj = W.mask(mask)
    hole = shadow_extend(obj)                       # include the cast shadow below the object
    region = _dilate(reshape(hole, "exact"), 16)    # 16 px + shadow-aware = cleanest (2026-10-01 dilation test)
    pipe, Q = E.load()
    r = SR.remove(pipe, Q, W.img, obj, hole, region, seed=mix_seed(E.seed, W.img), steps=E.steps)
    return W.back(r.image, region), {**r.info, "attempts": r.attempts}


def _a(phrase: str) -> str:
    p = phrase.strip().rstrip(".")
    if re.match(r"^(a|an|the|some|two|three|\d+)\b", p, re.I) or p.endswith("s"):
        return p
    return ("an " if p[:1].lower() in "aeiou" else "a ") + p


def inpaint(E: Engine, image: Image.Image, mask: np.ndarray, prompt: str, style: str = "", n: int = 1,
            reference: Image.Image | None = None) -> list[np.ndarray]:
    """Ghost fill: the selection is greyed out (25 % photo + 75 % grey) inside a context crop and Qwen is asked to
    replace the grey area. Measured on 32 placements (v5): 27 natural results vs 5 with keep-outside alone."""
    if mask.sum() < 4:
        raise ValueError("Paint the area you want to fill first.")
    pipe, Q = E.load()
    Wk = Work(image)
    O = np.asarray(Wk.img)
    hole = _dilate(Wk.mask(mask), 4)
    x0, y0, x1, y1 = Q.crop_window(hole, pad_frac=0.6, min_side=384)
    crop = O[y0:y1, x0:x1].astype(np.float32)
    hc = hole[y0:y1, x0:x1] > 0
    crop[hc] = 0.25 * crop[hc] + 0.75 * 150
    g = crop.astype(np.uint8)
    cnts, _ = cv2.findContours(hc.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(g, cnts, -1, (90, 90, 90), 2)
    what = prompt.strip() or "whatever naturally belongs there"
    tail = (" It must fit the photo naturally: same lighting, perspective, colours and scale, with soft natural "
            "shadows. Remove the grey completely. Keep everything outside the grey area exactly unchanged.")
    if style and style.lower() not in ("photorealistic", "photo", "realistic", "none", ""):
        tail += f" Style: {style}."
    ims = [Image.fromarray(g)]
    if reference is not None:
        ims.append(reference)
        instr = f"Replace the grey area in <image1> with {_a(what)}, taking its look from <image2>." + tail
    else:
        instr = f"Replace the grey area with {_a(what)}." + tail
    region = _dilate(hole, 9)
    outs = []
    for v in range(max(1, min(int(n or 1), 4))):
        r = E.edit(ims, instr, seed=E.seed + 13 * v, resolution=768)
        full = O.copy()
        full[y0:y1, x0:x1] = np.asarray(r.resize((x1 - x0, y1 - y0), Image.LANCZOS))
        outs.append(Wk.back(_composite(O, full, hole, region), region))
    return outs


def outpaint(E: Engine, image: Image.Image, tw: int, th: int, ox: int, oy: int, prompt: str = "") -> np.ndarray:
    """Outpaint LoRA (v8 winner); falls back to our blurred-edge + keep-outside method if the LoRA cannot load."""
    w, h = image.size
    ox, oy = max(0, int(ox)), max(0, int(oy))
    # ml-api.js caps the canvas at 1536 px but sends offsets for the uncapped size: shrink the photo to fit
    s = min(1.0, tw / max(1, w + 2 * ox), th / max(1, h + 2 * oy))
    if s < 1.0:
        w, h, ox, oy = round(w * s), round(h * s), round(ox * s), round(oy * s)
        image = image.resize((w, h), Image.LANCZOS)
    ox, oy = min(ox, max(0, tw - w)), min(oy, max(0, th - h))
    if (tw, th) == (w, h):
        return np.asarray(image)
    try:
        return np.asarray(E.outpaint_lora(image, tw, th, ox, oy, prompt))
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return outpaint_keep(E, image, tw, th, ox, oy, prompt)


def outpaint_keep(E: Engine, image: Image.Image, tw: int, th: int, ox: int, oy: int, prompt: str = "") -> np.ndarray:
    """Fallback. New area starts as a heavily blurred edge extension (our tested outpaint fill); Qwen generates only there
    (keep-outside latent preservation) and a Laplacian blend hides the seam."""
    from editing.outpaint import prepare_canvas

    w, h = image.size
    ox, oy = max(0, int(ox)), max(0, int(oy))
    # ml-api.js caps the canvas at 1536 px but sends offsets for the uncapped size: shrink the photo to fit
    s = min(1.0, tw / max(1, w + 2 * ox), th / max(1, h + 2 * oy))
    if s < 1.0:
        w, h, ox, oy = round(w * s), round(h * s), round(ox * s), round(oy * s)
        image = image.resize((w, h), Image.LANCZOS)
    amt = {"left": ox, "top": oy, "right": max(0, tw - w - ox), "bottom": max(0, th - h - oy)}
    if not any(amt.values()):
        return np.asarray(image)
    canvas, hole = prepare_canvas(np.asarray(image), amt, "blur")
    canvas = canvas[:th, :tw]
    hole = hole[:th, :tw]
    if not prompt.strip():
        prompt = E.ask([image], "Describe the setting and background of this photo in one short sentence, without "
                                "mentioning any people or animals.", 60)
    text = (f"{prompt.strip().rstrip('.')}. A seamless, natural continuation of the same photo: same lighting, "
            f"perspective, colours and style. No new people, animals or text.")
    ci = Image.fromarray(canvas)
    G = E.refine(ci, text, seed=SEED, strength=0.95, resolution=1024, keep=(ci, hole))
    region = np.maximum(hole, _dilate(hole, 24))
    return _composite(canvas, G, hole, region)


# --- edit-prompt routes

_DEBLUR = re.compile(r"\b(blur\w*|unblur|deblur|sharpen\w*|shaky|shake|camera shake|motion blur|out of focus|"
                     r"not in focus|while running|while walking|jitter\w*)\b", re.I)
_NOT_RECOLOUR = re.compile(r"\b(grade|grading|tone|tonal|filter|lighting|light|glow|style|mood|cinematic|hdr)\b", re.I)
_REMOVE = re.compile(r"^\s*(?:please\s+|can you\s+|could you\s+)?(?:remove|erase|delete|get rid of|take out)\s+(.+)$", re.I)
_RECOLOUR_VERB = re.compile(r"\b(colou?r|recolou?r|paint|make|turn|change|dye)\b", re.I)


_POSE_WORDS = re.compile(r"\b(pose[sd]?|posing|posture|stance|body position|sit(?:s|ting)? like|stand(?:s|ing)? like|"
                         r"same position|copy (?:the |his |her |their )?(?:pose|posture|position)|body language)\b", re.I)
_STYLE_WORDS = re.compile(r"\b(style|styled|stylise|stylize|painting|painted|aesthetic|art ?work|artistic|look and feel|"
                          r"colou?r grade|colou?r palette|in the manner)\b", re.I)


def route(prompt: str, has_reference: bool, has_mask: bool) -> str:
    """Which pipeline a prompt goes to. With a reference photo the words decide: pose words -> pose transfer, style
    words -> style transfer. (Before 2026-10-06 only the exact phrase "Transfer exact body pose..." reached the pose
    engine; anything else became a two-image edit and Qwen simply returned the reference photo.)"""
    p = prompt.strip()
    if has_reference and (p.lower().startswith("transfer exact body pose") or _POSE_WORDS.search(p)):
        return "pose"
    if has_reference and _STYLE_WORDS.search(p):
        return "style"
    if _DEBLUR.search(p) and not has_mask:
        return "deblur"
    if _REMOVE.match(p):
        return "remove"
    global COLOUR_WORDS
    if COLOUR_WORDS is None:
        from editing.recolour import COLOURS

        COLOUR_WORDS = sorted(COLOURS, key=len, reverse=True)
    low = p.lower()
    if _RECOLOUR_VERB.search(low) and not _NOT_RECOLOUR.search(low) and any(re.search(rf"\b{re.escape(c)}\b", low) for c in COLOUR_WORDS) \
            and len(low.split()) <= 16 and not has_reference:
        return "recolour"
    return "general"


def _ground_mask(E: Engine, img: Image.Image, what: str) -> np.ndarray | None:
    from editing import recolour as RC

    box = RC.ground_box(lambda p: E.ask([img], p, 60), img.size, what)
    if box is None:
        return None
    m = RC.box_to_mask(img, box)
    return m if m.sum() > 16 else None


def edit_prompt(E: Engine, image: Image.Image, prompt: str, mask: np.ndarray | None = None,
                reference: Image.Image | None = None, strength: float = 1.0) -> tuple[np.ndarray, dict]:
    kind = route(prompt, reference is not None, mask is not None)
    info: dict[str, Any] = {"route": kind}
    Wk = Work(image)
    img = Wk.img
    O = np.asarray(img)

    if kind == "pose":
        return pose_transfer(E, image, reference, info)

    if kind == "style":
        extra = re.sub(r"(?i)\b(in |with )?(the )?(same )?(style|look)( of| from)? (the |this )?(reference( image| photo)?|<image2>)\b",
                       "", prompt).strip(" .,")
        out = style_transfer(E, image, reference, 0.85, extra if len(extra.split()) > 3 else "")
        return out, info

    if kind == "deblur":
        from editing.deblur import SPEC_PROMPT

        return Wk.back(E.edit([img], SPEC_PROMPT)), info

    if kind == "remove":
        what = _REMOVE.match(prompt.strip()).group(1).strip().rstrip(".!")
        what = re.sub(r"^(the|a|an)\s+", "", what, flags=re.I)
        m = Wk.mask(mask) if mask is not None else _ground_mask(E, img, f"the {what}")
        if m is not None:
            from editing.masking import shadow_extend

            info["object"] = what
            full, rinfo = remove_object_info(E, img, m)
            info["removal"] = rinfo
            info["_region"] = _full_mask(_dilate(shadow_extend(m), 64), image.size)
            return Wk.back(full, _dilate(shadow_extend(m), 24)), info
        info["route"] = kind = "general"

    if kind == "recolour":
        from editing import recolour as RC

        prog = RC.parse_instruction(prompt, ask=lambda p: E.ask([img], p, 200))
        obj = Wk.mask(mask) if mask is not None else _ground_mask(E, img, f"the {prog.get('object') or 'main object'}")
        if obj is not None:
            parts = {}
            for nm in RC.part_names(prog.get("region", {})):
                b = RC.ground_box(lambda p: E.ask([img], p, 60), img.size, f"the {nm} of the {prog.get('object', 'object')}")
                parts[nm] = RC.box_to_mask(img, b, within=obj) if b else np.zeros_like(obj)
            try:
                res = RC.recolour(O, obj, prog, parts)
                info.update(program=prog, metrics=res.metrics, recolour_id=_rc_store(Wk, O, obj, prog, parts))
                return Wk.back(res.image, _dilate(obj, 3)), info
            except ValueError as e:
                info["recolour_error"] = str(e)
        info["route"] = kind = "general"

    # general instruction edit
    ims = [img] + ([reference] if reference is not None else [])
    instr = prompt.strip()
    if reference is not None and "<image2>" not in instr:
        instr = instr.replace("the reference image", "<image2>").replace("reference image", "<image2>")
        if "<image2>" not in instr:
            instr += " Use <image2> as the reference."
    if mask is not None and mask.sum() > 4:
        pipe, Q = E.load()
        hole = _dilate(Wk.mask(mask), 4)
        x0, y0, x1, y1 = Q.crop_window(hole, pad_frac=0.8, min_side=384)
        ims[0] = Image.fromarray(O[y0:y1, x0:x1])
        r = E.edit(ims, instr + " " + KEEP_REST, resolution=768)
        full = O.copy()
        full[y0:y1, x0:x1] = np.asarray(r.resize((x1 - x0, y1 - y0), Image.LANCZOS))
        region = _dilate(hole, 12)
        out = _composite(O, full, hole, region)
        info["masked"] = True
        return Wk.back(out, region), info
    if E.tier == "pro" and reference is None:
        instr = E.rewrite_edit(img, instr)
        info["rewritten"] = instr
    if E.tier == "flash":
        out = np.asarray(E.flash_edit(ims, instr).resize(img.size, Image.LANCZOS))
        info["model"] = "flash"
    else:
        out = np.asarray(E.edit(ims, instr))
    if strength is not None and 0.0 < float(strength) < 0.98 and reference is None:
        out = (O.astype(np.float32) * (1 - strength) + out.astype(np.float32) * strength).round().astype(np.uint8)
    return Wk.back(out), info


def pose_transfer(E: Engine, image: Image.Image, reference: Image.Image, info: dict | None = None) -> tuple:
    """The person in ``image`` takes the pose of the person in ``reference``. Qwen never sees the reference photo (given
    both, it copied the reference): the pose is turned into words - an overall description from Qwen3-VL plus exact
    joint angles from OpenPose - and the photo is edited from those words, with up to two correction rounds."""
    from editing import pose as P

    info = {"route": "pose"} if info is None else info
    if reference is None:
        raise ValueError("Add a reference photo of someone in the pose you want (the + button).")
    Wk = Work(image)
    img = Wk.img
    pipe, Q = E.load()
    ref = _small(reference, 1024)
    tkp = P.keypoints(ref)
    gist = E.ask([ref], P.DESCRIBE_Q, 120).strip()
    seed = mix_seed(E.seed, img)
    if tkp is None or (tkp[:, 2] > 0).sum() < 4:            # OpenPose found no usable skeleton: words only
        if not E.yes([ref], "Is there a person in this photo?"):
            raise ValueError("I couldn't find a person in the reference photo. Use a photo where someone's body is "
                             "clearly visible in the pose you want.")
        instr = (f"Change the pose of the person in this photo to this pose: {gist.rstrip('.')}. Keep the same person, "
                 f"face, hairstyle, clothing, background and lighting.")
        o = Q.edit(pipe, img, instr, seed=seed, steps=E.steps).image
        info.update(pck=None, words_only=True)
        return Wk.back(o), info
    tkp = tkp.copy()
    tkp[:, 0] *= img.width / ref.width                        # joints in the photo's pixel frame (PCK is relative)
    tkp[:, 1] *= img.height / ref.height
    same = lambda o: E.yes([img, o], "Are image 1 and image 2 the same person wearing the same clothes?")  # noqa: E731
    o, pinfo = P.repose_loop(pipe, Q, img, tkp, rounds=3, seed=seed, same_person=same, steps=E.steps, gist=gist)
    info.update(pck=pinfo["pck"], best_round=pinfo["best_round"],
                same_person=next((h["same_person"] for h in pinfo.get("history", [])
                                  if h.get("round") == pinfo["best_round"]), None))
    return Wk.back(o), info


STYLE_Q = ("Describe the artistic style of this image so a painter could copy it, in one or two sentences: medium, "
           "colour palette, brushwork or texture, lighting and mood. Do not describe what is shown.")


def style_transfer(E: Engine, image: Image.Image, reference: Image.Image, strength: float = 0.85,
                   style_prompt: str = "") -> np.ndarray:
    """v8: giving Qwen both images returned the reference painting itself. Now Qwen3-VL first describes the
    reference's style in words, then a single-image edit repaints the photo in that style (content kept)."""
    Wk = Work(image)
    style = E.ask([reference], STYLE_Q, 120).strip().rstrip(".")
    instr = (f"Repaint this photo in this artistic style: {style}. Keep the content, composition, people and every "
             f"object exactly where they are; change only the rendering style.")
    if style_prompt.strip():
        instr += f" {style_prompt.strip()}"
    out = np.asarray(E.edit([Wk.img], instr)).astype(np.float32)
    s = float(strength if strength is not None else 1.0)
    if 0 < s < 0.98:
        out = np.asarray(Wk.img).astype(np.float32) * (1 - s) + out * s
    return Wk.back(out.round().clip(0, 255).astype(np.uint8))


def upscale(image: Image.Image, scale: float = 2, enhance: bool = True) -> np.ndarray:
    s = max(1.0, min(float(scale or 2), 4.0, 4096 / max(image.size)))
    out = image.resize((round(image.width * s), round(image.height * s)), Image.LANCZOS)
    a = np.asarray(out).astype(np.float32)
    if enhance:
        a = a + 0.5 * (a - cv2.GaussianBlur(a, (0, 0), 1.2))
    return a.round().clip(0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------- LumenAI editor (JSON endpoints)
#
# The LumenAI editor (lumen_web/editor.html) sends JSON with images as data URLs and gets back the result plus the
# checks it shows in its status card. Every check is measured, not decorative: pixels outside the edit, the recolour
# leak/hue numbers, the removal self-check, pose PCK, sharpness gain, and a Qwen3-VL "was the request carried out"
# question. When that question says no for a general edit, the edit is redone once with a new seed (live self-repair).

QUICK_STEPS = int(os.environ.get("ADOBEPS_QUICK_STEPS", "16"))

# The model picker (Lumen 1.1 Lumina / 1.2 Kroma / 1.3 Lux; internal keys flash / lumen / pro). Specialist tools (remove, recolour, pose, 3-D, upscale...) always use their own best pipeline; the
# tier decides how prompt edits and new images are made. v9 numbers: klein 1.1 s/image and GEdit 7.8 vs our 7.4;
# Qwen-rewritten prompts lifted prompt-following 8.4 -> 9.7.
TIERS = {
    "flash": {"name": "Lumen 1.1 - Lumina", "about": "FLUX.2 klein 4B, 4 steps - about 2 s, 13.4 GB"},
    "lumen": {"name": "Lumen 1.2 - Kroma", "about": "Qwen-Image 2.1, 16 steps - about 12 s"},
    "pro": {"name": "Lumen 1.3 - Lux", "about": "Qwen-Image 2.1, 40 steps, prompt rewritten by Qwen3-VL - about 40 s"},
}


def tier_of(d: dict) -> str:
    m = str(d.get("model") or "").lower()
    if m in TIERS:
        return m
    return "pro" if d.get("quality") == "full" else "lumen"      # old clients: quick / full
_RC_CACHE = None   # OrderedDict id -> (Work, image, mask, program, parts)


def _rc_store(Wk: "Work", O: np.ndarray, obj: np.ndarray, prog: dict, parts: dict) -> str:
    """Keep what a recolour needs so the editor's colour slider can re-run only the (fast, CPU) colour step."""
    global _RC_CACHE
    import uuid
    from collections import OrderedDict

    if _RC_CACHE is None:
        _RC_CACHE = OrderedDict()
    k = uuid.uuid4().hex[:12]
    _RC_CACHE[k] = (Wk, O, obj, dict(prog), parts)
    while len(_RC_CACHE) > 8:
        _RC_CACHE.popitem(last=False)
    return k


def recolour_shade(rid: str, colour: str) -> tuple[np.ndarray, dict]:
    from editing import recolour as RC

    if not _RC_CACHE or rid not in _RC_CACHE:
        raise ValueError("This recolour has expired. Recolour the object again.")
    Wk, O, obj, prog, parts = _RC_CACHE[rid]
    prog = {**prog, "target_color": colour}
    res = RC.recolour(O, obj, prog, parts)
    return Wk.back(res.image, _dilate(obj, 3)), {"route": "recolour", "metrics": res.metrics, "recolour_id": rid}


def read_data_url(s: str | None) -> bytes | None:
    if not s:
        return None
    if s.startswith("data:"):
        s = s.split(",", 1)[1]
    return base64.b64decode(s)


def data_url(img) -> str:
    """JPEG 95 for photos (about 1/6 of the PNG size, visually lossless), PNG when there is transparency."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    if img.mode == "RGBA":
        return b64png(img)
    b = io.BytesIO()
    img.convert("RGB").save(b, "JPEG", quality=95, subsampling=0)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def mask_url(m: np.ndarray) -> str:
    return b64png(Image.fromarray((m > 0).astype(np.uint8) * 255, "L"))


def chk(label: str, ok, note: str = "", repaired: bool = False) -> dict:
    d = {"label": label, "ok": bool(ok), "note": note}
    if repaired:
        d["repaired"] = True
    return d


def _small(a, side: int = 768) -> Image.Image:
    im = a if isinstance(a, Image.Image) else Image.fromarray(np.asarray(a)[..., :3])
    s = min(1.0, side / max(im.size))
    return im if s == 1.0 else im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)


def unchanged_outside(O: np.ndarray, R: np.ndarray, region: np.ndarray | None) -> float:
    """Share of pixels outside ``region`` that did not visibly change (max channel difference <= 6)."""
    O, R = np.asarray(O)[..., :3], np.asarray(R)[..., :3]
    if O.shape != R.shape:
        return 0.0
    out = np.ones(O.shape[:2], bool) if region is None else region == 0
    if not out.any():
        return 1.0
    d = np.abs(O.astype(np.int16) - R.astype(np.int16)).max(-1)
    return float((d[out] <= 6).mean())


def sharpness(a) -> float:
    g = cv2.cvtColor(np.asarray(_small(a, 1024)), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def _full_mask(m: np.ndarray | None, size: tuple[int, int]) -> np.ndarray | None:
    if m is None:
        return None
    if m.shape[::-1] == tuple(size):
        return m
    return (np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 0).astype(np.uint8)


def lumen_edit(E: Engine, image: Image.Image, prompt: str, mask: np.ndarray | None = None,
               reference: Image.Image | None = None) -> dict:
    """One prompt -> routed edit. No after-the-fact checking (removed 2026-10-05: it cost seconds per edit - a
    Qwen3-VL question or two, sometimes a full second edit - without making the result better)."""
    if not prompt.strip():
        raise ValueError("Describe the edit you want first.")
    E.guard(prompt)
    out, info = edit_prompt(E, image, prompt, mask, reference, 1.0)
    for k in ("program", "removal", "_region", "metrics"):
        info.pop(k, None)
    return {"image": data_url(out), "route": info["route"], "checks": [],
            "recolour_id": info.get("recolour_id"), "info": info}



def lumen_segment(E: Engine, image: Image.Image, points: list, box=None) -> dict:
    """Click points ([x, y, 1 = object | 0 = not]) in image pixels and/or a box -> SAM 2.1 mask."""
    from editing.masking import sam_masks

    Wk = Work(image)
    s = Wk.s
    pts = [(int(p[0] * s), int(p[1] * s), int(p[2]) if len(p) > 2 else 1) for p in points or []]
    bx = tuple(int(v * s) for v in box) if box else None
    cands = sam_masks(Wk.img, pts, box=bx)
    if not cands:
        raise ValueError("Click on the object you want to select.")
    m = _full_mask(cands[0].mask, image.size)
    if box:                                  # a drawn box: nothing outside it may be selected (2 % slack)
        W_, H_ = image.size
        x0, y0, x1, y1 = (int(v) for v in box)
        px, py = int(0.02 * W_), int(0.02 * H_)
        keep = np.zeros_like(m)
        keep[max(0, y0 - py):min(H_, y1 + py), max(0, x0 - px):min(W_, x1 + px)] = 1
        m = m * keep
    if m.sum() < 16:
        raise ValueError("Nothing found there - click right on the object.")
    ys, xs = np.nonzero(m)
    return {"mask": mask_url(m), "box": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
            "area": round(float(m.mean()), 4), "score": round(cands[0].score, 3)}


def lumen_recolour(E: Engine, image: Image.Image, colour: str, mask: np.ndarray | None, what: str = "") -> dict:
    from editing import recolour as RC

    Wk = Work(image)
    img = Wk.img
    obj = Wk.mask(mask) if mask is not None else _ground_mask(E, img, f"the {what or 'main object'}")
    if obj is None or obj.sum() < 16:
        raise ValueError("Select the object to recolour first (click it with Select).")
    prog = {"object": what or "object", "target_color": colour, "frame": "object", "region": {"op": "object"}}
    O = np.asarray(img)
    res = RC.recolour(O, obj, prog, {})
    rid = _rc_store(Wk, O, obj, prog, {})
    return {"image": data_url(Wk.back(res.image, _dilate(obj, 3))), "route": "recolour", "recolour_id": rid,
            "checks": []}


RATIOS = {"16:9": 16 / 9, "9:16": 9 / 16, "4:3": 4 / 3, "3:4": 3 / 4, "1:1": 1.0, "3:2": 3 / 2, "2:3": 2 / 3, "4:5": 4 / 5}


def expand_plan(w: int, h: int, ratio: str | None, grow: float = 0.0, cap: int = 1536) -> tuple:
    """-> (scale for the photo, target w, target h, offset x, offset y). Grows the shorter side to reach the ratio
    (or every side by ``grow``); keeps the canvas within ``cap`` px by shrinking the photo, sizes multiple of 16."""
    if ratio:
        r = RATIOS.get(ratio) or (lambda a, b: a / b)(*map(float, ratio.split(":")))
        tw, th = (w, round(w / r)) if w / h > r else (round(h * r), h)
        tw, th = max(tw, w), max(th, h)
        if (tw, th) == (w, h):          # already that ratio: grow it a little all round instead
            tw, th = round(w * 1.25), round(h * 1.25)
    else:
        g = max(0.05, min(float(grow or 0.25), 1.0))
        tw, th = round(w * (1 + g)), round(h * (1 + g))
    s = min(1.0, cap / max(tw, th))
    tw, th = int(tw * s) // 16 * 16, int(th * s) // 16 * 16
    pw, ph = min(tw, round(w * s)), min(th, round(h * s))
    return s, tw, th, (tw - pw) // 2, (th - ph) // 2, pw, ph


def lumen_expand(E: Engine, image: Image.Image, ratio: str | None, grow: float = 0.0, prompt: str = "") -> dict:
    w, h = image.size
    s, tw, th, ox, oy, pw, ph = expand_plan(w, h, ratio, grow)
    img = image.resize((pw, ph), Image.LANCZOS) if (pw, ph) != (w, h) else image
    out = outpaint(E, img, tw, th, ox, oy, prompt)
    return {"image": data_url(out), "route": "expand", "size": [tw, th], "checks": []}



IDEAS_Q = ("You are a photo stylist. Suggest {n} different objects that could be added to this exact photo to make it "
           "more interesting, and that would look natural here (right setting, era, style and scale). Be specific to "
           "this scene - for example 'a folded linen napkin' or 'a small brass key', not just 'a plant'. Every idea must "
           "be a different kind of thing.{avoid} For each, name the visible surface it would stand or lie on and where "
           "on it, e.g. 'on the stone slab, left of the bottle'. Reply with JSON only: "
           '[{{"object": "...", "reason": "one short sentence", "where": "..."}}, ...]')

SIZE_Q = ("The red box marks a reference object in this photo. Imagine a real {obj} placed {where}. How tall would it "
          "appear in this photo compared with the object in the red box, and what is its width-to-height ratio as seen "
          'from this camera? Reply with JSON only: {{"height_vs_red_box": <number>, "width_to_height": <number>}}')


GEN_SIZES = {"1:1": (1024, 1024), "4:5": (896, 1120), "3:4": (896, 1184), "3:2": (1216, 816), "2:3": (816, 1216),
             "16:9": (1344, 768), "9:16": (768, 1344)}


def lumen_generate(E: Engine, prompt: str, ratio: str = "1:1") -> dict:
    if not prompt.strip():
        raise ValueError("Describe the image you want to create.")
    E.guard(prompt)
    w, h = GEN_SIZES.get(ratio, GEN_SIZES["1:1"])
    p = prompt.strip()
    if E.tier == "pro":
        p = E.rewrite_t2i(p)
    img = E.flash_generate(p, w, h) if E.tier == "flash" else E.generate(p, w, h)
    return {"image": data_url(img), "route": "generate", "size": [w, h], "checks": [], "model": E.tier,
            "prompt_used": p if p != prompt.strip() else None}



def lumen_ideas(E: Engine, image: Image.Image, n: int = 5, avoid: list | None = None) -> dict:
    """Object ideas as words (what + where). Boxes are worked out per idea in lumen_place, when one is picked:
    asking a VLM for n boxes at once gave the same few ideas with poor positions and sizes."""
    import json

    Wk = Work(image)
    av = [str(a)[:40] for a in (avoid or []) if str(a).strip()][:20]
    q = IDEAS_Q.format(n=n, avoid=(" Do not suggest any of these again: " + ", ".join(av) + ".") if av else "")
    raw = E.ask([_small(Wk.img, 1024)], q, 500)
    m = re.search(r"\[.*\]", raw, re.S)
    try:
        items = json.loads(m.group(0)) if m else []
    except Exception:  # noqa: BLE001
        items = []
    seen, out = {a.lower() for a in av}, []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict) or not str(it.get("object", "")).strip():
            continue
        o = str(it["object"]).strip()[:48]
        if o.lower() in seen:
            continue
        seen.add(o.lower())
        out.append({"object": o, "reason": str(it.get("reason", ""))[:160], "where": str(it.get("where", ""))[:120]})
    return {"ideas": out[:n]}


def _num(d: dict, k: str, lo: float, hi: float, default: float) -> float:
    try:
        return float(min(hi, max(lo, float(d.get(k)))))
    except Exception:  # noqa: BLE001
        return default


def place_box(size, surface, ref, h_ratio: float, aspect: float) -> list[int]:
    """Box for a new object: height from the reference object x the VLM's size ratio, standing in the chosen spot
    (base in the lower part of the spot), moved sideways off the reference object if it would cover it."""
    W, H = size
    rx0, ry0, rx1, ry1 = ref
    h = min(0.8 * H, max(0.05 * H, h_ratio * (ry1 - ry0)))
    w = min(0.8 * W, max(0.04 * W, h * aspect))
    sx0, sy0, sx1, sy1 = surface
    cx = (sx0 + sx1) / 2
    base = sy0 + 0.7 * (sy1 - sy0)
    base = min(H - 2, max(base, h + 2))
    x0, y0 = cx - w / 2, base - h

    def overlap(x0):
        ix = max(0.0, min(x0 + w, rx1) - max(x0, rx0)) * max(0.0, min(base, ry1) - max(y0, ry0))
        return ix / (w * h)

    if overlap(x0) > 0.25:         # slide left or right (whichever needs less) until it no longer covers the reference
        left, right = rx0 - w - 0.04 * W, rx1 + 0.04 * W
        cands = [x for x in (left, right) if 0 <= x <= W - w] or [min(max(0, left), W - w)]
        x0 = min(cands, key=lambda x: abs(x - x0))
    x0 = min(max(0.0, x0), W - w)
    return [int(x0), int(max(0, y0)), int(x0 + w), int(base)]


def lumen_place(E: Engine, image: Image.Image, obj: str, where: str = "") -> dict:
    """Where and how big should ``obj`` go? 1) Qwen3-VL finds the free spot it described, 2) the main subject is the
    size reference, 3) Qwen3-VL estimates height relative to that reference and the width/height ratio."""
    import json

    from editing import recolour as RC

    if not obj.strip():
        raise ValueError("Say what you want to add.")
    E.guard(f"add {obj} {where}")
    Wk = Work(image)
    img = Wk.img
    W, H = img.size
    ask1 = lambda q: E.ask([img], q, 60)  # noqa: E731
    where = where.strip() or "on a visible surface where it would naturally go"
    spot = RC.ground_box(ask1, img.size, f"the free space {where}, where a {obj} could be placed") \
        or RC.ground_box(ask1, img.size, f"a free spot on a surface where a {obj} could stand")
    ref = RC.ground_box(ask1, img.size, "the main subject") or (int(.35 * W), int(.3 * H), int(.65 * W), int(.9 * H))
    if spot is None:
        spot = (int(.1 * W), int(.55 * H), int(.45 * W), int(.95 * H))
    marked = np.asarray(img).copy()
    cv2.rectangle(marked, ref[:2], ref[2:], (255, 0, 0), max(2, W // 300))
    raw = E.ask([Image.fromarray(marked)], SIZE_Q.format(obj=obj.strip(), where=where), 60)
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except Exception:  # noqa: BLE001
        d = {}
    hr, asp = _num(d, "height_vs_red_box", 0.05, 3.0, 0.4), _num(d, "width_to_height", 0.15, 4.0, 1.0)
    b = place_box((W, H), spot, ref, hr, asp)
    inv = 1 / Wk.s
    return {"box": [round(v * inv) for v in b], "spot": [round(v * inv) for v in spot],
            "size_vs_subject": round(hr, 2), "aspect": round(asp, 2)}


def lumen_add(E: Engine, image: Image.Image, obj: str, box) -> dict:
    from editing import add_object as AO

    if not obj.strip() or not box or len(box) != 4:
        raise ValueError("Pick an idea (or draw a box) to add something.")
    E.guard(f"add {obj}")
    pipe, Q = E.load()
    Wk = Work(image)
    b = tuple(int(v * Wk.s) for v in box)
    r = AO.add(pipe, Q, Wk.img, obj.strip(), b, "ghost", seed=mix_seed(E.seed, Wk.img), steps=E.steps)
    area = AO.edit_area(b, Wk.img.size)
    out = Wk.back(r.image, _dilate(area, 9))
    return {"image": data_url(out), "route": "add", "checks": []}



def lumen_cutout(E: Engine, image: Image.Image) -> dict:
    return {"image": data_url(remove_background(E, image, True)), "route": "cutout", "checks": []}



COMPLETE_Q = "What is shown in this cut-out? Answer in at most four words."


def view3d_prepare(E: Engine, image: Image.Image, mask: np.ndarray | None, complete: bool = True) -> tuple:
    """Object cut-out (selection, or the main subject) -> optionally completed by Qwen (hidden parts drawn in, v7:
    better 3-D in 4/4) -> RGBA for Hunyuan3D."""
    from editing.object3d import cutout

    Wk = Work(image)
    m = Wk.mask(mask) if mask is not None else _ground_mask(E, Wk.img, "the main subject")
    if m is None or m.sum() < 64:
        raise ValueError("Select the object to turn into 3D first.")
    cut = cutout(Wk.img, m)
    info = {"completed": False}
    if complete:
        from editing.object3d import completion_mask

        white = Image.new("RGB", cut.size, (255, 255, 255))
        white.paste(cut, mask=cut.split()[3])
        what = E.ask([white], COMPLETE_Q, 10).strip(" .") or "object"
        instr = (f"This is a cut-out of {what}; parts of it are hidden or cut off. Show the complete {what} - every part "
                 f"that was hidden or cut off - alone on a plain pure white background. Keep exactly the same pose, the same "
                 f"size and position, and identical appearance and colours. Keep everything it is holding or carrying. "
                 f"Remove only separate things that were in front of it and are not part of it.")
        done, keep, how = completion_mask(E.edit([white], instr, resolution=1024))
        if keep.sum() > 0.5 * (np.asarray(cut)[..., 3] > 127).sum():
            cut = Image.fromarray(np.dstack([np.asarray(done.resize(cut.size)),
                                             (np.asarray(Image.fromarray(keep.astype(np.uint8) * 255).resize(cut.size)))]), "RGBA")
            info.update(completed=True, what=what)
    return cut, info


PHOTO_WORDS_Q = ("Describe this photo in one short line for a trend search: the kind of photo (portrait, product, food, "
                 "street, interior, landscape...), the main subject and the mood. No names.")


def build_lumen_routes(app, E: Engine, jrun):
    from fastapi import Request

    def img_of(d: dict, key: str = "image") -> Image.Image | None:
        b = read_data_url(d.get(key))
        return None if b is None else read_image(b)

    def mask_of(d: dict, size) -> np.ndarray | None:
        b = read_data_url(d.get("mask"))
        if b is None:
            return None
        m = read_mask(b, size)
        return m if m.sum() >= 4 else None

    def need(d: dict) -> Image.Image:
        im = img_of(d)
        if im is None:
            raise ValueError("No photo was sent.")
        return im

    async def body(req: Request) -> dict:
        try:
            return await req.json()
        except Exception:  # noqa: BLE001
            return {}

    def quality(d: dict, fn):
        """The model picked in the editor (Lumen Flash / Lumen / Lumen Pro, see TIERS) and the seed (variations)."""
        def go():
            E.tier = tier_of(d)
            E.steps = STEPS if E.tier == "pro" else QUICK_STEPS
            E.seed = int(d.get("seed") or SEED)
            try:
                return fn()
            finally:
                E.steps, E.seed, E.tier = STEPS, SEED, "lumen"
        return go

    from server import extras

    extras.register(app, E, jrun, {"body": body, "need": need, "img_of": img_of, "mask_of": mask_of, "quality": quality})

    @app.post("/api/v1/edit")
    async def l_edit(req: Request):
        d = await body(req)
        try:
            im = need(d)
            m, ref = mask_of(d, im.size), img_of(d, "reference")
        except ValueError as e:
            return jrun.bad(e)
        return await jrun("edit", quality(d, lambda: lumen_edit(E, im, d.get("prompt", ""), m, ref)))

    @app.post("/api/v1/segment")
    async def l_seg(req: Request):
        d = await body(req)
        try:
            im = need(d)
        except ValueError as e:
            return jrun.bad(e)
        return await jrun("segment", lambda: lumen_segment(E, im, d.get("points") or [], d.get("box")))

    @app.post("/api/v1/recolour")
    async def l_rc(req: Request):
        d = await body(req)
        try:
            im = need(d)
            m = mask_of(d, im.size)
        except ValueError as e:
            return jrun.bad(e)
        return await jrun("recolour", lambda: lumen_recolour(E, im, d.get("colour", "#2f4f7b"), m, d.get("object", "")))

    @app.post("/api/v1/recolour-shade")
    async def l_shade(req: Request):
        d = await body(req)

        def go():
            out, info = recolour_shade(str(d.get("recolour_id", "")), d.get("colour", "#2f4f7b"))
            return {"image": data_url(out), "route": "recolour", "recolour_id": info["recolour_id"],
                    "checks": []}
        return await jrun("recolour-shade", go)

    @app.post("/api/v1/expand")
    async def l_expand(req: Request):
        d = await body(req)
        try:
            im = need(d)
        except ValueError as e:
            return jrun.bad(e)
        return await jrun("expand", quality(d, lambda: lumen_expand(E, im, d.get("ratio"), float(d.get("grow") or 0),
                                                                    d.get("prompt", ""))))

    @app.post("/api/v1/ideas")
    async def l_ideas(req: Request):
        d = await body(req)
        try:
            im = need(d)
        except ValueError as e:
            return jrun.bad(e)
        return await jrun("ideas", lambda: lumen_ideas(E, im, int(d.get("n") or 5), d.get("avoid") or []))

    @app.post("/api/v1/place")
    async def l_place(req: Request):
        d = await body(req)
        try:
            im = need(d)
        except ValueError as e:
            return jrun.bad(e)
        return await jrun("place", lambda: lumen_place(E, im, str(d.get("object", "")), str(d.get("where", ""))))

    @app.post("/api/v1/add")
    async def l_add(req: Request):
        d = await body(req)
        try:
            im = need(d)
        except ValueError as e:
            return jrun.bad(e)
        return await jrun("add", quality(d, lambda: lumen_add(E, im, d.get("object", ""), d.get("box"))))

    @app.post("/api/v1/trends")
    async def l_trends(req: Request):
        """Gemini + Google Search. With for_photo, Qwen3-VL describes the photo in words first (the photo itself
        never leaves our server); the Gemini call runs without the GPU lock."""
        from server import trends as T

        d = await body(req)
        im = img_of(d) if d.get("for_photo") else None

        def prep():
            if str(d.get("topic", "")).strip():
                E.guard(f"show trends for {d['topic']}")
            return E.ask([_small(Work(im).img, 768)], PHOTO_WORDS_Q, 60).strip() if im is not None else ""

        def after(words):
            try:
                return T.trends(str(d.get("topic", "")), words, int(d.get("n") or 6))
            except T.TrendsError as e:
                raise ValueError(str(e)) from e
        return await jrun("trends", prep, after=after)

    @app.post("/api/v1/cutout")
    async def l_cut(req: Request):
        d = await body(req)
        try:
            im = need(d)
        except ValueError as e:
            return jrun.bad(e)
        return await jrun("cutout", lambda: lumen_cutout(E, im))

    @app.post("/api/v1/generate")
    async def l_gen(req: Request):
        d = await body(req)
        return await jrun("generate", quality(d, lambda: lumen_generate(E, str(d.get("prompt", "")),
                                                                         str(d.get("ratio") or "1:1"))))

    @app.post("/api/v1/upscale-ai")
    async def l_up(req: Request):
        d = await body(req)
        try:
            im = need(d)
        except ValueError as e:
            return jrun.bad(e)
        scale = 2 if int(d.get("scale") or 4) == 2 else 4

        def after(img):          # no GPU lock: TADSR runs in its own app
            note = "TADSR (one-step diffusion)"
            try:
                out = E.upscale_ai(img, scale)
                ai = True
            except ValueError as e:
                out, ai, note = Image.fromarray(upscale(img, scale, True)), False, str(e)
            return {"image": data_url(out), "route": "upscale", "size": list(out.size),
                    "checks": [], "note": None if ai else note}
        return await jrun("upscale-ai", lambda: im, after=after)

    @app.post("/api/v1/export")
    async def l_export(req: Request):
        """The final image with the invisible LumenAI mark. Only exports are marked: marking every preview cost
        1-2 s per result (2048 px) and made the colour slider slow."""
        from server.watermark import embed

        d = await body(req)
        try:
            im = img_of(d)
            if im is None:
                raise ValueError("No photo was sent.")
            if d.get("image", "").startswith("data:image/png"):          # keep transparency (cut-outs)
                b = read_data_url(d["image"])
                raw = Image.open(io.BytesIO(b))
                raw.load()
                im = raw.convert("RGBA") if raw.mode in ("RGBA", "LA", "P") else im
        except ValueError as e:
            return jrun.bad(e)

        def go(_):                   # CPU only: no GPU lock needed
            out = embed(im)
            if out.mode == "RGBA":
                return {"image": b64png(out), "marked": True}
            b = io.BytesIO()
            out.convert("RGB").save(b, "JPEG", quality=95, subsampling=0)
            return {"image": "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode(), "marked": True}
        return await jrun("export", lambda: None, after=go)

    @app.post("/api/v1/verify")
    async def l_verify(req: Request):
        """Was this image made with LumenAI? (looks for the invisible mark)"""
        from server.watermark import detect

        d = await body(req)
        try:
            im = need(d)
        except ValueError as e:
            return jrun.bad(e)
        return await jrun("verify", lambda: {**detect(im), "route": "verify"})

    @app.post("/api/v1/view3d")
    async def l_3d(req: Request):
        d = await body(req)
        try:
            im = need(d)
            m = mask_of(d, im.size)
        except ValueError as e:
            return jrun.bad(e)
        complete = bool(d.get("complete", True))

        def after(prep):   # runs WITHOUT the GPU lock: the 3-D model is built in the separate object3d app
            cut, info = prep
            glb, minfo = E.make3d(cut)
            return {"glb": "data:model/gltf-binary;base64," + base64.b64encode(glb).decode(), "cutout": b64png(cut),
                    "route": "view3d", "info": {**info, **{k: v for k, v in minfo.items() if isinstance(v, (int, float, str))}},
                    "checks": []}
        return await jrun("view3d", lambda: view3d_prepare(E, im, m, complete), after=after)


# ----------------------------------------------------------------------------- FastAPI app


def build_app(E: Engine):
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, Response

    app = FastAPI(title="LumenAI API", version="1.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
                       expose_headers=["X-AdobePS-Route", "X-AdobePS-Seconds", "X-AdobePS-Note"])

    from starlette.concurrency import run_in_threadpool

    def locked(fn):
        with E.lock:
            return fn()

    def gpu_reset():
        E.remote_peak = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:  # noqa: BLE001
            pass

    def gpu_peak() -> float | None:
        """Peak GPU memory of this request: the other container's measured peak when the work ran there, else this
        container's own peak (torch allocator) since the request started."""
        if E.remote_peak:
            return round(float(E.remote_peak), 2)
        try:
            import torch

            if torch.cuda.is_available():
                return round(torch.cuda.max_memory_allocated() / 1e9, 2)
        except Exception:  # noqa: BLE001
            pass
        return None

    def locked_measured(fn):
        with E.lock:
            gpu_reset()
            out = fn()
            return out, gpu_peak()

    async def run(name: str, fn: Callable[[], Any], headers: dict | None = None, after: Callable | None = None):
        t0 = time.perf_counter()
        try:
            out, peak = await run_in_threadpool(locked_measured, fn)
            if after is not None:                       # second stage without the GPU lock (remote 3-D build)
                out = await run_in_threadpool(after, out)
                if E.remote_peak:
                    peak = round(float(E.remote_peak), 2)
        except ValueError as e:
            return JSONResponse({"detail": str(e)}, status_code=400)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            return JSONResponse({"detail": f"{name} failed: {type(e).__name__}: {e}"[:400]}, status_code=500)
        sec = f"{time.perf_counter() - t0:.1f}"
        print(f"[api] {name} {sec}s", flush=True)
        h = {"X-AdobePS-Seconds": sec, **(headers or {})}
        if isinstance(out, tuple):
            out, info = out
            h["X-AdobePS-Route"] = str(info.get("route", ""))
            print(f"[api] {name} info {info}", flush=True)
        if isinstance(out, dict):
            if "checks" in out or "ideas" in out or "mask" in out:
                out = {**out, "seconds": float(sec), "peak_gb": peak}
            return JSONResponse(out, headers=h)
        return Response(content=png(out), media_type="image/png", headers=h)

    async def jrun(name: str, fn: Callable[[], Any], after: Callable | None = None):
        return await run(name, fn, after=after)

    jrun.bad = lambda e: JSONResponse({"detail": str(e)}, status_code=400)
    build_lumen_routes(app, E, jrun)

    async def rd(f: UploadFile | None):
        return None if f is None else await f.read()

    @app.get("/api/v1/health")
    async def health():
        return {"status": "ok", "model_loaded": E.pipe is not None, "load_seconds": E.loaded_at,
                "routes": ["remove-background", "remove-object", "inpaint", "outpaint", "edit-prompt", "style-transfer",
                           "upscale"],
                "trends_key": bool(os.environ.get("GEMINI_API_KEY")),
                "lumen": ["edit", "segment", "recolour", "recolour-shade", "expand", "ideas", "place", "add", "trends", "upscale-ai", "export", "verify", "generate", "cutout", "view3d",
                          "bokeh", "bokeh-adjust", "layers", "move", "relight", "style", "product", "pose"],
                "models": TIERS,
                "quick_steps": QUICK_STEPS, "full_steps": STEPS}

    @app.post("/api/v1/remove-background")
    async def r_bg(image: UploadFile = File(...), refine_edges: str = Form("true")):
        img = read_image(await rd(image))
        return await run("remove-background", lambda: remove_background(E, img, refine_edges.lower() != "false"))

    @app.post("/api/v1/remove-object")
    async def r_obj(image: UploadFile = File(...), mask: UploadFile = File(...)):
        img = read_image(await rd(image))
        m = read_mask(await rd(mask), img.size)
        return await run("remove-object", lambda: remove_object(E, img, m))

    @app.post("/api/v1/inpaint")
    async def r_inpaint(image: UploadFile = File(...), mask: UploadFile = File(...), prompt: str = Form(""),
                        style: str = Form(""), variations: int = Form(1), reference_image: UploadFile | None = File(None)):
        img = read_image(await rd(image))
        m = read_mask(await rd(mask), img.size)
        ref = read_image(await rd(reference_image)) if reference_image is not None else None
        return await run("inpaint", lambda: (E.guard(prompt), {"variations": [b64png(o) for o in
                                                                             inpaint(E, img, m, prompt, style, variations, ref)]})[1])

    @app.post("/api/v1/outpaint")
    async def r_out(image: UploadFile = File(...), target_width: int = Form(...), target_height: int = Form(...),
                    offset_x: int = Form(0), offset_y: int = Form(0), prompt: str = Form("")):
        img = read_image(await rd(image))
        return await run("outpaint", lambda: outpaint(E, img, target_width, target_height, offset_x, offset_y, prompt))

    @app.post("/api/v1/edit-prompt")
    async def r_edit(image: UploadFile = File(...), prompt: str = Form(...), strength: float = Form(1.0),
                     style: str = Form(""), reference_image: UploadFile | None = File(None),
                     mask: UploadFile | None = File(None)):
        img = read_image(await rd(image))
        ref = read_image(await rd(reference_image)) if reference_image is not None else None
        m = read_mask(await rd(mask), img.size) if mask is not None else None
        if m is not None and m.sum() < 4:
            m = None
        return await run("edit-prompt", lambda: (E.guard(prompt), edit_prompt(E, img, prompt, m, ref, strength))[1])

    @app.post("/api/v1/style-transfer")
    async def r_style(image: UploadFile = File(...), reference_image: UploadFile = File(...), strength: float = Form(0.85),
                      style_prompt: str = Form("")):
        img = read_image(await rd(image))
        ref = read_image(await rd(reference_image))
        return await run("style-transfer", lambda: style_transfer(E, img, ref, strength, style_prompt))

    @app.post("/api/v1/upscale")
    async def r_up(image: UploadFile = File(...), scale: float = Form(2), enhance_details: str = Form("true"),
                   portrait_mode: str = Form("false"), prompt: str = Form("")):
        img = read_image(await rd(image))

        def go():
            try:
                return E.upscale_ai(img, 2 if float(scale or 2) <= 2 else 4)
            except Exception:  # noqa: BLE001   (not deployed / failed: the plain method, as before)
                return upscale(img, scale, enhance_details.lower() != "false")
        return await run("upscale", lambda: img, after=lambda _: go())

    return app
