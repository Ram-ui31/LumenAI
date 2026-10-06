"""Photo of an object -> textured 3-D model (GLB) the browser can rotate, via Hunyuan3D-2 (Tencent community licence:
non-commercial/academic use allowed in India; excluded in the EU, UK and South Korea).

  cutout(image, mask)  -> RGBA object on transparent background (we already have the SAM / VLM+SAM mask)
  make_mesh(rgba)      -> trimesh (Hunyuan3D-2mini shape, ~6 GB) + texture (Hunyuan3D-Paint, ~16 GB total; optional)
  turntable(mesh)      -> N rendered views for reports / previews (CPU point-splat renderer, no OpenGL needed)
Run as a separate step: unload Qwen first (16 GB budget).
"""
from __future__ import annotations

import time

import numpy as np
from PIL import Image

SHAPE_REPO, SHAPE_SUB = "tencent/Hunyuan3D-2mini", "hunyuan3d-dit-v2-mini"
PAINT_REPO = "tencent/Hunyuan3D-2"


def cutout(image: Image.Image, mask: np.ndarray, pad: float = 0.12, size: int = 512) -> Image.Image:
    """Square RGBA crop of the masked object, centred, with transparent background (what Hunyuan3D expects)."""
    rgb = np.asarray(image.convert("RGB"))
    m = (mask > 0).astype(np.uint8) * 255
    ys, xs = np.nonzero(m)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    side = int(max(x1 - x0, y1 - y0) * (1 + 2 * pad))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    canvas = np.zeros((side, side, 4), np.uint8)
    sx0, sy0 = cx - side // 2, cy - side // 2
    for (a0, a1, b0, b1) in [(max(0, sy0), min(rgb.shape[0], sy0 + side), max(0, sx0), min(rgb.shape[1], sx0 + side))]:
        canvas[a0 - sy0:a1 - sy0, b0 - sx0:b1 - sx0, :3] = rgb[a0:a1, b0:b1]
        canvas[a0 - sy0:a1 - sy0, b0 - sx0:b1 - sx0, 3] = m[a0:a1, b0:b1]
    return Image.fromarray(canvas, "RGBA").resize((size, size), Image.LANCZOS)


_SHAPE = _PAINT = None


def make_mesh(rgba: Image.Image, texture: bool = True, seed: int = 7, steps: int = 30, octree: int = 256,
              max_faces: int = 40000) -> tuple:
    """Returns (trimesh, info). Shape first; the shape model is freed before texturing to stay within 16 GB."""
    import gc

    import torch
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    global _SHAPE, _PAINT
    info = {}
    t0 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    if _SHAPE is None:
        _SHAPE = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(SHAPE_REPO, subfolder=SHAPE_SUB)
    mesh = _SHAPE(image=rgba, num_inference_steps=steps, octree_resolution=octree,
                  generator=torch.Generator("cuda").manual_seed(seed))[0]
    info.update(shape_s=round(time.perf_counter() - t0, 1), shape_peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
                faces=int(len(mesh.faces)))
    if texture:
        # simplify before texturing: UV-unwrapping an 800k-face mesh took up to 96 min in v5 (reducer failed silently)
        try:
            from hy3dgen.shapegen import DegenerateFaceRemover, FaceReducer, FloaterRemover

            mesh = FaceReducer()(DegenerateFaceRemover()(FloaterRemover()(mesh)), max_facenum=max_faces)
            info["reducer"] = "hunyuan"
        except Exception as e:  # noqa: BLE001
            info["reducer_error"] = repr(e)[:200]
            try:
                mesh = mesh.simplify_quadric_decimation(face_count=max_faces)
                info["reducer"] = "trimesh"
            except Exception as e2:  # noqa: BLE001
                info["reducer_error2"] = repr(e2)[:200]
        info["faces_textured"] = int(len(mesh.faces))
        _SHAPE = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t1 = time.perf_counter()
        from hy3dgen.texgen import Hunyuan3DPaintPipeline

        if _PAINT is None:
            _PAINT = Hunyuan3DPaintPipeline.from_pretrained(PAINT_REPO)
        mesh = _PAINT(mesh, image=rgba)
        info.update(texture_s=round(time.perf_counter() - t1, 1),
                    texture_peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
    return mesh, info


def unload():
    global _SHAPE, _PAINT
    _SHAPE = _PAINT = None


def turntable(mesh, views: int = 8, size: int = 384, elevation: float = 15.0, n_points: int = 400_000) -> list[Image.Image]:
    """Render ``views`` images around the object with a z-buffered point splat of coloured surface samples."""
    import trimesh

    m = mesh.dump(concatenate=True) if isinstance(mesh, trimesh.Scene) else mesh
    try:
        pts, fid, cols = trimesh.sample.sample_surface(m, n_points, sample_color=True)
        cols = np.asarray(cols)[:, :3].astype(np.float32)
    except Exception:  # noqa: BLE001  (untextured mesh): plain grey
        pts, fid = trimesh.sample.sample_surface(m, n_points)
        cols = np.tile(np.array([200, 200, 205], np.float32), (len(pts), 1))
    nrm = m.face_normals[fid]
    pts = pts - pts.mean(0)
    pts = pts / np.abs(pts).max()
    out = []
    e = np.radians(elevation)
    for k in range(views):
        a = 2 * np.pi * k / views
        Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
        P = pts @ (Rx @ Ry).T
        N = nrm @ (Rx @ Ry).T
        light = np.array([0.35, 0.55, -0.76])                       # from upper left, towards the viewer side
        shade = 0.45 + 0.55 * np.clip(np.abs(N @ light), 0, 1)       # two-sided Lambert (normals may be flipped)
        col = np.clip(cols * shade[:, None], 0, 255).astype(np.uint8)
        z = P[:, 2] + 3.0
        f = size * 1.1
        u = np.rint(f * P[:, 0] / z + size / 2).astype(int)
        v = np.rint(-f * P[:, 1] / z + size / 2).astype(int)          # y up in mesh space
        ok = (u >= 0) & (u < size) & (v >= 0) & (v < size)
        order = np.argsort(-z[ok])
        uu, vv, cc = u[ok][order], v[ok][order], col[ok][order]
        img = np.full((size, size, 3), 245, np.uint8)
        img[vv, uu] = cc
        out.append(Image.fromarray(img))
    return out


def completion_mask(img: Image.Image) -> tuple[Image.Image, np.ndarray, str]:
    """Object mask of Qwen's completed cut-out -> (RGB on white, mask, how).
    Qwen-Image 2.1 decodes RGBA: asked for an object 'alone on a plain background' it may return a TRANSPARENT
    background whose RGB values are garbage (v7: a magenta noise field; converting to RGB then made the whole square
    'foreground'). If the alpha channel is used, it is the mask; otherwise the background colour is estimated from the
    border (not assumed to be pure white) and the largest connected blob that differs from it is kept."""
    import cv2

    a = np.asarray(img.convert("RGBA"))
    alpha = a[..., 3]
    if img.mode == "RGBA" and (alpha < 128).mean() > 0.05:
        fg, how = (alpha >= 128).astype(np.uint8), "alpha"
        rgb = (a[..., :3].astype(np.float32) * (alpha[..., None] / 255.0) + 255 * (1 - alpha[..., None] / 255.0))
        rgb = rgb.round().astype(np.uint8)
    else:
        rgb = a[..., :3]
        border = np.concatenate([rgb[:4].reshape(-1, 3), rgb[-4:].reshape(-1, 3), rgb[:, :4].reshape(-1, 3),
                                 rgb[:, -4:].reshape(-1, 3)])
        bg = np.median(border, 0)
        fg, how = (np.abs(rgb.astype(np.int16) - bg).max(-1) > 18).astype(np.uint8), "border colour"
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg)
    keep = (lab == (1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))) if n > 1 else fg > 0
    return Image.fromarray(rgb), keep, how
