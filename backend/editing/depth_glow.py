"""Depth glow: a coloured light placed at a point in the photo, shading the 2.5-D surface given by a depth map.

Light model from the team's depth-glow pipeline (Lambert shading on depth normals, radial / spot / directional
falloff, linear-light compositing). Here the depth comes from Depth Anything V2 Small (already in the API container)
and is cached, so moving the light re-renders on the CPU.
"""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

DEFAULTS = dict(position=(0.5, 0.5), color=(255, 180, 100), mode="spread", direction=90.0, cone_angle=80.0, z=1.0,
                intensity=65.0, size=0.5, softness=0.8, mix=0.85, light_only=False, parent_target=None,
                reveal=True, stops=6.0)


def glow(image: Image.Image, depth: np.ndarray, **params) -> Image.Image:
    """depth: float array, 1 = near (white disparity), any size; params override DEFAULTS."""
    p = {**DEFAULTS, **{k: v for k, v in params.items() if v is not None}}
    p["position"] = tuple(float(v) for v in p["position"])
    p["color"] = tuple(int(v) for v in p["color"])
    return relight(image.convert("RGB"), np.asarray(depth, np.float32), SimpleNamespace(**p))


def srgb_to_linear(x):
    return np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= .0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - .055)


def relight(image, depth, args):
    """2.5D surface lighting, linear-light compositing, bounded CPU row tiles."""
    w, h = image.size
    d = np.asarray(Image.fromarray(depth.astype(np.float32)).resize((w,h), Image.BILINEAR))
    # White disparity is near: convert to a surface receding away from the camera.
    surface = 1 - d
    dy = np.gradient(surface, axis=0) * h if h > 1 else np.zeros_like(surface)
    dx = np.gradient(surface, axis=1) * w if w > 1 else np.zeros_like(surface)
    px, py = args.position
    if args.parent_target:
        parent = json.loads(Path(args.parent_target).read_text())
        angle = math.radians(float(parent.get('rotation', 0)))
        sx, sy = parent.get('scale', [1,1])
        tx, ty = parent.get('position', [0,0])
        px, py = tx + math.cos(angle)*px*sx - math.sin(angle)*py*sy, ty + math.sin(angle)*px*sx + math.cos(angle)*py*sy
    if not all(math.isfinite(v) for v in [px,py]):
        raise ValueError('Parent transform produced a nonfinite position')
    color = srgb_to_linear(np.array(args.color, dtype=np.float32)/255)
    az = math.radians(args.direction)
    direction = np.array([math.cos(az), math.sin(az), .6], dtype=np.float32)
    direction /= np.linalg.norm(direction)
    output = np.empty((h,w,3), dtype=np.uint8)
    # LumenAI "reveal": light multiplies what is already in the photo (an exposure gain of up to 2**stops) instead of
    # painting colour on top, so a dark wall shows its real paint, writing and portrait. Near-black JPEG noise is
    # amplified by the gain, so the photo is first smoothed with an edge-preserving filter where the gain is large.
    reveal = bool(getattr(args, 'reveal', False))
    full_lin = srgb_to_linear(np.asarray(image, dtype=np.float32) / 255)
    if reveal:
        import cv2
        den = cv2.bilateralFilter(full_lin.astype(np.float32), 9, 0.02, 4)
        tint = color / max(float(color.max()), 1e-6)
        # auto exposure: Strength 65 lifts the lit area to about mid-grey, whatever the photo's darkness
        gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
        r_full = np.sqrt((px - (gx + .5) / w) ** 2 + (py - (gy + .5) / h) ** 2)
        Y = den @ np.array([.2126, .7152, .0722], np.float32)
        inside = Y[r_full < max(args.size * .6, .05)]
        m = float(np.percentile(inside, 60)) if inside.size else float(Y.mean())
        gmax = float(np.clip(.22 / max(m, 1e-4), 1.6, 256))
    xx = (np.arange(w, dtype=np.float32) + .5) / w
    for start in range(0,h,128):
        end = min(h,start+128)
        yy = (np.arange(start,end,dtype=np.float32)+.5)[:,None]/h
        normal = np.stack((-dx[start:end]*.25, -dy[start:end]*.25, -np.ones((end-start,w),dtype=np.float32)),axis=-1)
        normal /= np.maximum(np.linalg.norm(normal,axis=-1,keepdims=True),1e-6)
        vx = np.broadcast_to(px-xx,(end-start,w))
        vy = np.broadcast_to(py-yy,(end-start,w))
        vz = surface[start:end] - (1-args.z) + .05
        vector = np.stack((vx,vy,-vz),axis=-1)
        distance = np.linalg.norm(vector,axis=-1)
        light_dir = vector / np.maximum(distance[...,None],1e-6)
        radial = np.sqrt(vx*vx+vy*vy)
        edge = np.clip((args.size-radial)/max(args.size*args.softness,1e-6),0,1)
        falloff = edge*edge*(3-2*edge) / (1+(distance/args.size)**2)
        if args.mode == 'directional':
            light_dir = np.broadcast_to(-direction,vector.shape)
            falloff = np.ones_like(radial)
        elif args.mode == 'spot':
            # Cone projects into the image at the requested azimuth.
            cone = np.sum((-light_dir)*direction,axis=-1)
            cutoff = math.cos(math.radians(args.cone_angle/2))
            cone_weight = np.clip((cone-cutoff)/max((1-cutoff)*args.softness,1e-6),0,1)
            falloff *= cone_weight*cone_weight*(3-2*cone_weight)
        lambert = np.maximum(np.sum(normal*light_dir,axis=-1),0)
        contribution = (args.intensity/100)*4*falloff[...,None]*(.15+.85*lambert[...,None])*color
        linear = full_lin[start:end]
        if reveal and not args.light_only:
            shape = np.ones_like(radial) if args.mode == 'directional' else np.clip(falloff*(1+(distance/args.size)**2), 0, 1)
            F = (args.intensity/65)**.6*shape*(.35+.65*lambert)                      # 1 = mid-grey at the centre (Strength 65)
            gain = gmax ** F - 1                                                     # multiplicative exposure lift
            base = linear + (den[start:end]-linear)*np.clip(gain/24, 0, .75)[...,None]
            lit = base*(1 + gain[...,None]*tint) + .03*F[...,None]*color              # faint glow keeps pure black lit
            hi = .8
            lit = np.where(lit > hi, hi + (1-hi)*(1-np.exp(-(lit-hi)/(1-hi))), lit)  # soft shoulder, no clipping
        else:
            lit = contribution if args.light_only else linear + contribution*(.25+.75*linear)
        mixed = linear*(1-args.mix)+lit*args.mix
        output[start:end] = np.round(linear_to_srgb(mixed)*255).astype(np.uint8)
    return Image.fromarray(output)


