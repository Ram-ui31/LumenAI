"""New camera angles from one photo: geometry decides what is visible, Qwen only fills what was hidden.

  1. depth      MoGe-2 (MIT): metric 3-D point for every pixel + camera intrinsics.
  2. re-project rotate the camera around the subject (yaw / pitch, optional dolly) and splat every point into the new
                view with a z-buffer -> `render` + `valid` (pixels seen in the original photo).
  3. fill       the disoccluded holes (things that were hidden behind the subject):
                  lama   - LaMa fills the holes (fast baseline)
                  keep   - Qwen-Image 2.1 refine with latent preservation: valid pixels are reset to the render at every
                           step, x = m*x + (1-m)((1-s) z_render + s eps), so only the holes are generated (our maths)
                  anyangle - Qwen-Image 2.1 + AnyAngle LoRA: "Change the camera angle from <image2> to <image1>."
                             with <image1> = original photo, <image2> = the rough render (Apache-2.0 LoRA)
                  text   - Qwen-Image 2.1 alone, told to rotate the camera (does it work natively?)

Maths of step 2 (OpenCV camera, x right, y down, z forward), pivot c = median 3-D point of the subject:
    p' = R(yaw, pitch) (p - c) + c + t ,   pixel = K p' / z'     (K from MoGe's normalised intrinsics x image size)
Z-buffer: nearest point wins per pixel; points are splatted as 2x2 to avoid pin holes; a hole is any pixel no point
landed on (after a 3x3 depth-aware close), i.e. surface the original camera could not see.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

MOGE_REPO = "Ruicheng/moge-2-vitl-normal"
ANYANGLE_REPO, ANYANGLE_FILE = "lilylilith/QI_2.1_AnyAngle", "QI2.1_AnyAngle.safetensors"
ANYANGLE_PROMPT = "Change the camera angle from <image2> to <image1>."


@dataclass
class Geometry:
    points: np.ndarray        # H x W x 3, metres, OpenCV camera coordinates
    mask: np.ndarray          # H x W bool, valid depth
    K: np.ndarray             # 3 x 3 pixel intrinsics
    image: np.ndarray         # H x W x 3 uint8


# ----------------------------------------------------------------------------- depth

_MOGE = None


def load_moge(device: str = "cuda"):
    global _MOGE
    if _MOGE is None:
        from moge.model.v2 import MoGeModel

        _MOGE = MoGeModel.from_pretrained(MOGE_REPO).to(device).eval()
    return _MOGE


def unload_moge():
    global _MOGE
    _MOGE = None


def estimate_geometry(image: Image.Image, device: str = "cuda") -> Geometry:
    import torch

    rgb = np.asarray(image.convert("RGB"))
    H, W = rgb.shape[:2]
    model = load_moge(device)
    t = torch.tensor(rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
    with torch.no_grad():
        out = model.infer(t)
    pts = out["points"].float().cpu().numpy()
    mask = out["mask"].cpu().numpy().astype(bool) & np.isfinite(pts).all(-1)
    Kn = out["intrinsics"].float().cpu().numpy()
    K = Kn.copy()
    K[0] *= W
    K[1] *= H
    return Geometry(pts, mask, K, rgb)


# ----------------------------------------------------------------------------- re-projection


def rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Camera orbit: yaw turns around the vertical (y) axis, pitch around the horizontal (x) axis."""
    a, b = np.radians(yaw_deg), np.radians(pitch_deg)
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    return Rx @ Ry


def pivot_point(geo: Geometry, subject: np.ndarray | None = None) -> np.ndarray:
    """Orbit centre: median 3-D point of the subject (or of the central third of the frame)."""
    H, W = geo.mask.shape
    sel = geo.mask.copy()
    if subject is not None and (subject > 0).sum() > 50:
        sel &= subject > 0
    else:
        c = np.zeros_like(sel)
        c[H // 3: 2 * H // 3, W // 3: 2 * W // 3] = True
        sel &= c
    return np.median(geo.points[sel], axis=0)


def render_view(geo: Geometry, yaw: float = 20.0, pitch: float = 0.0, dolly: float = 0.0,
                subject: np.ndarray | None = None, splat: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (render uint8 HxWx3, valid bool HxW, depth HxW) of the scene seen from the orbited camera.
    The object turns by ``yaw`` degrees as if walking around it (camera moves the opposite way)."""
    H, W = geo.mask.shape
    c = pivot_point(geo, subject)
    R = rotation(-yaw, -pitch)
    P = geo.points[geo.mask]
    C = geo.image[geo.mask]
    Pn = (P - c) @ R.T + c
    Pn[:, 2] -= dolly * c[2]
    z = Pn[:, 2]
    ok = z > 1e-3
    Pn, C, z = Pn[ok], C[ok], z[ok]
    u = geo.K[0, 0] * Pn[:, 0] / z + geo.K[0, 2]
    v = geo.K[1, 1] * Pn[:, 1] / z + geo.K[1, 2]
    order = np.argsort(-z)                       # far first, near last -> near overwrites
    u, v, z, C = u[order], v[order], z[order], C[order]
    img = np.zeros((H, W, 3), np.uint8)
    dep = np.full((H, W), np.inf, np.float32)
    u0, v0 = np.rint(u).astype(int), np.rint(v).astype(int)
    offsets = [(0, 0)] + [(dx, dy) for dy in range(-(splat // 2), splat - splat // 2) for dx in
                          range(-(splat // 2), splat - splat // 2) if (dx, dy) != (0, 0)]
    for k, (dx, dy) in enumerate(offsets):
        ui, vi = u0 + dx, v0 + dy
        inb = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        ui, vi, zz, cc = ui[inb], vi[inb], z[inb], C[inb]
        if k == 0:
            # exact pixel, z-buffer: points are sorted far -> near, so the last write per pixel is the nearest
            img[vi, ui] = cc
            np.minimum.at(dep, (vi, ui), zz)
        else:
            # neighbours only fill pixels nothing landed on (closes pin holes without bleeding over edges)
            empty = ~np.isfinite(dep0[vi, ui])
            ui, vi, zz, cc = ui[empty], vi[empty], zz[empty], cc[empty]
            img[vi, ui] = cc
            np.minimum.at(dep, (vi, ui), zz)
        if k == 0:
            dep0 = dep.copy()
    valid = np.isfinite(dep)
    # depth-aware close: fill single-pixel gaps between valid neighbours, not real disocclusions
    small = cv2.morphologyEx(valid.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    gap = small & ~valid
    if gap.any():
        blur = cv2.medianBlur(img, 3)
        img[gap] = blur[gap]
        valid = small
    return img, valid, dep


def hole_mask(valid: np.ndarray, grow: int = 4) -> np.ndarray:
    """Pixels to generate: everything the original camera did not see, slightly grown (edge splats are unreliable)."""
    return cv2.dilate((~valid).astype(np.uint8), np.ones((2 * grow + 1, 2 * grow + 1), np.uint8))


# ----------------------------------------------------------------------------- filling the holes


def fill_lama(render: np.ndarray, holes: np.ndarray) -> np.ndarray:
    from editing.common import lama_inpaint

    return np.asarray(lama_inpaint(render, holes))


def fill_keep(pipe, Q, original: Image.Image, render_filled: np.ndarray, holes: np.ndarray, seed: int = 7,
              strength: float = 0.75, steps: int = 40, resolution: int = 768) -> tuple[np.ndarray, dict]:
    """Our maths: Qwen refines the LaMa-filled render; outside the holes the latents are reset to the render at every
    step (latent preservation), so geometry-correct pixels stay put and only hidden surfaces are generated.
    The original photo is the reference image (identity, materials, lighting)."""
    prompt = ("The same scene photographed from a slightly different camera angle. Photorealistic, sharp, natural "
              "lighting, consistent with the reference photo.")
    st = Image.fromarray(render_filled)
    r = Q.refine(pipe, st, prompt, seed=seed, strength=strength, steps=steps, resolution=resolution,
                 condition=original, keep=(st, holes))
    out = np.asarray(r.image.convert("RGB").resize(st.size, Image.LANCZOS))
    return out, {"seconds": r.seconds, "peak_gb": r.peak_gb, **r.info}


def load_anyangle(pipe) -> str:
    """Load the AnyAngle LoRA. Tries diffusers' loader; if the ComfyUI key names are not recognised, renames
    `diffusion_model.` -> `transformer.` and loads the state dict directly. Returns how it was loaded."""
    try:
        pipe.load_lora_weights(ANYANGLE_REPO, weight_name=ANYANGLE_FILE, adapter_name="anyangle")
        return "diffusers"
    except Exception as e1:  # noqa: BLE001
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        sd = load_file(hf_hub_download(ANYANGLE_REPO, ANYANGLE_FILE))
        conv = {}
        for k, v in sd.items():
            k2 = k.replace("diffusion_model.", "transformer.", 1)
            k2 = k2.replace(".lora_down.weight", ".lora_A.weight").replace(".lora_up.weight", ".lora_B.weight")
            if k2.startswith("transformer."):
                conv[k2] = v
        try:
            pipe.load_lora_weights(conv, adapter_name="anyangle")
            return f"converted ({len(conv)} tensors; first error: {repr(e1)[:120]})"
        except Exception as e2:  # noqa: BLE001
            raise RuntimeError(f"AnyAngle LoRA could not be loaded: {repr(e1)[:200]} / {repr(e2)[:200]}") from e2


def set_anyangle(pipe, on: bool):
    try:
        if on:
            pipe.set_adapters(["anyangle"], adapter_weights=[1.0])
        else:
            pipe.disable_lora()
    except Exception:  # noqa: BLE001
        if on:
            pipe.enable_lora()


def fill_anyangle(pipe, Q, original: Image.Image, render_filled: np.ndarray, seed: int = 7, steps: int = 24,
                  cfg: float = 3.0, resolution: int = 768) -> tuple[np.ndarray, dict]:
    import time

    import torch

    set_anyangle(pipe, True)
    try:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        img = pipe(prompt=ANYANGLE_PROMPT, image=[original.convert("RGB"), Image.fromarray(render_filled)],
                   negative_prompt=" ", true_cfg_scale=cfg, num_inference_steps=steps, output_resolution=resolution,
                   generator=torch.Generator("cpu").manual_seed(seed)).images[0]
        sec = time.perf_counter() - t0
    finally:
        set_anyangle(pipe, False)
    out = np.asarray(img.convert("RGB").resize(original.size, Image.LANCZOS))
    return out, {"seconds": round(sec, 2), "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}


def fill_text(pipe, Q, original: Image.Image, yaw: float, seed: int = 7, steps: int = 40,
              resolution: int = 768) -> tuple[np.ndarray, dict]:
    side = "left" if yaw > 0 else "right"
    instr = (f"Show the same scene from a camera rotated {abs(yaw):.0f} degrees to the {side} around the main subject. "
             f"Keep every object, colour and the lighting the same.")
    r = Q.edit(pipe, original, instr, seed=seed, steps=steps, resolution=resolution)
    return np.asarray(r.image.convert("RGB").resize(original.size, Image.LANCZOS)), {"seconds": r.seconds, "peak_gb": r.peak_gb}


# ----------------------------------------------------------------------------- consistency score (no ground truth)


def warp_fidelity(result: np.ndarray, render: np.ndarray, valid: np.ndarray) -> dict:
    """How well a result respects the geometry: colour error on pixels the original camera saw (should be ~0 for a
    geometry-respecting method), measured in CIEDE2000 after a 2-px erosion of the valid area."""
    from editing.recolour import delta_e2000, rgb_to_cielab

    v = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if v.sum() < 100:
        return {"geo_dE": None, "valid_share": float(valid.mean())}
    de = delta_e2000(rgb_to_cielab(render[v]), rgb_to_cielab(result[v]))
    return {"geo_dE": float(np.mean(de)), "valid_share": float(valid.mean())}
