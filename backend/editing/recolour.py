"""Training-free part-level recolouring via mask algebra (no diffusion, CPU, ~0.1 s).

    instruction --(Qwen3-VL or rule parser)--> program --(mask algebra on the SAM mask)--> soft weight W
    image --(OKLab recolour, colour-affinity A)--> R ;   out = I + (W * A) * (R - I)

Guarantees (by construction, checked by ``verify``):
  * every pixel further than 2*tau outside the selected part is returned bit-identical (compact S-curve);
  * shading/texture survive: lightness variation inside the part is kept, only its colour moves.

Program grammar (JSON; every expression is a soft mask in [0, 1], finally intersected with the object):
  {"op": "object"}                                    whole object
  {"op": "half", "side": "upper|lower|left|right", "frac": 0.5}   part holding ``frac`` of the object's AREA
  {"op": "band", "side": ..., "from": 0.2, "to": 0.4}  area band measured from that side
  {"op": "stripes", "axis": "horizontal|vertical", "count": 4}     ``count`` coloured stripes
  {"op": "border", "width": 0.06}                      rim of the object (width as fraction of sqrt(area))
  {"op": "part", "name": "pocket"}                     named part mask supplied by the caller (VLM box -> SAM)
  {"op": "and|or", "args": [..]}, {"op": "minus", "args": [a, b]}, {"op": "not", "arg": x}
Top level: {"target_color": "green" | "#22aa44" | [r, g, b], "region": EXPR,
            "frame": "object|image", "softness": 0.006, "lightness": 0.7}
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import cv2
import numpy as np

# ----------------------------------------------------------------------------- colour spaces


def _lin(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _gam(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)


_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])
_M1i, _M2i = np.linalg.inv(_M1), np.linalg.inv(_M2)


def rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """uint8 (or 0-255 float) sRGB [..., 3] -> OKLab float [..., 3] (L in 0..1)."""
    lms = _lin(np.asarray(rgb, np.float64) / 255.0) @ _M1.T
    return np.cbrt(lms) @ _M2.T


def oklab_to_linear(lab: np.ndarray) -> np.ndarray:
    return ((lab @ _M2i.T) ** 3) @ _M1i.T


def oklab_to_rgb(lab: np.ndarray) -> np.ndarray:
    return _gam(oklab_to_linear(lab)) * 255.0


def gamut_fit(lab: np.ndarray, iters: int = 10) -> np.ndarray:
    """Reduce chroma (keep L and hue) until each pixel fits sRGB - avoids hue-shifting channel clipping."""
    lin = oklab_to_linear(lab)
    bad = ((lin < -1e-4) | (lin > 1 + 1e-4)).any(-1)
    if not bad.any():
        return lab
    sub = lab[bad].copy()
    lo, hi = np.zeros(len(sub)), np.ones(len(sub))
    for _ in range(iters):
        mid = (lo + hi) / 2
        t = sub.copy()
        t[:, 1:] *= mid[:, None]
        l2 = oklab_to_linear(t)
        ok = ((l2 >= -1e-4) & (l2 <= 1 + 1e-4)).all(-1)
        lo, hi = np.where(ok, mid, lo), np.where(ok, hi, mid)
    sub[:, 1:] *= lo[:, None]
    out = lab.copy()
    out[bad] = sub
    return out


def rgb_to_cielab(rgb: np.ndarray) -> np.ndarray:
    lin = _lin(np.asarray(rgb, np.float64) / 255.0)
    xyz = lin @ np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                          [0.0193339, 0.1191920, 0.9503041]]).T
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def delta_e2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 between CIELAB arrays [..., 3]."""
    L1, a1, b1 = np.moveaxis(lab1, -1, 0)
    L2, a2, b2 = np.moveaxis(lab2, -1, 0)
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cb = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cb ** 7 / (Cb ** 7 + 25 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p, h2p = np.degrees(np.arctan2(b1, a1p)) % 360, np.degrees(np.arctan2(b2, a2p)) % 360
    dLp, dCp = L2 - L1, C2p - C1p
    dh = h2p - h1p
    dh = np.where(C1p * C2p == 0, 0, np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh)))
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh / 2))
    Lbp, Cbp = (L1 + L2) / 2, (C1p + C2p) / 2
    hs = h1p + h2p
    hbp = np.where(C1p * C2p == 0, hs, np.where(np.abs(h1p - h2p) <= 180, hs / 2,
                                                  np.where(hs < 360, (hs + 360) / 2, (hs - 360) / 2)))
    T = (1 - 0.17 * np.cos(np.radians(hbp - 30)) + 0.24 * np.cos(np.radians(2 * hbp))
         + 0.32 * np.cos(np.radians(3 * hbp + 6)) - 0.20 * np.cos(np.radians(4 * hbp - 63)))
    dth = 30 * np.exp(-(((hbp - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25 ** 7))
    Sl = 1 + 0.015 * (Lbp - 50) ** 2 / np.sqrt(20 + (Lbp - 50) ** 2)
    Sc, Sh = 1 + 0.045 * Cbp, 1 + 0.015 * Cbp * T
    Rt = -np.sin(np.radians(2 * dth)) * Rc
    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh))


COLOURS = {
    "red": "#d02a2a", "dark red": "#8b1a1a", "maroon": "#7a1f2b", "crimson": "#c8102e", "pink": "#f08aa8",
    "hot pink": "#ff3d8b", "magenta": "#d0359e", "purple": "#7b3fa0", "violet": "#8f5fd0", "lavender": "#b9a6e0",
    "blue": "#2a5bd0", "navy": "#1d2b5c", "navy blue": "#1d2b5c", "sky blue": "#79b8e8", "light blue": "#9cc8ee",
    "royal blue": "#2f50c8", "teal": "#16847f", "turquoise": "#30c0b8", "cyan": "#22b8d0", "mint": "#9de0c0",
    "green": "#2f9e44", "dark green": "#1f5e2c", "forest green": "#22573a", "olive": "#6b7a2a", "lime": "#8fd13a",
    "emerald": "#1f9d6b", "sage": "#9caf88", "yellow": "#f2c81f", "mustard": "#c9a227", "gold": "#d4a92a",
    "orange": "#ec7a1c", "coral": "#f0705a", "peach": "#f6b28a", "brown": "#7a4b2a", "tan": "#c8a47a",
    "beige": "#dccbae", "cream": "#f2e8d0", "white": "#f4f4f2", "grey": "#8a8a8a", "gray": "#8a8a8a",
    "light grey": "#c4c4c4", "light gray": "#c4c4c4", "dark grey": "#4a4a4a", "dark gray": "#4a4a4a",
    "charcoal": "#36393d", "black": "#1c1c1e", "silver": "#b8bcc0",
}


def parse_colour(c: Any) -> np.ndarray:
    """Colour name / '#rrggbb' / [r, g, b] -> sRGB float [3]."""
    if isinstance(c, (list, tuple, np.ndarray)):
        return np.asarray(c, np.float64)[:3]
    s = str(c).strip().lower()
    s = COLOURS.get(s, s)
    m = re.fullmatch(r"#?([0-9a-f]{6})", s)
    if not m:
        raise ValueError(f"unknown colour {c!r}")
    h = m.group(1)
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float64)


# ----------------------------------------------------------------------------- mask algebra

_DIRS = {"upper": (0, -1), "top": (0, -1), "lower": (0, 1), "bottom": (0, 1), "left": (-1, 0), "right": (1, 0)}


def clean_points(points: np.ndarray, mask: np.ndarray, k: float = 6.0) -> np.ndarray:
    """Point maps (MoGe-2) put a few object pixels - thin handles, spouts, edges against a pale background - at absurd
    depths. One such point made the object 'huge', the seam softness tau = softness x size swallowed every stripe and
    recolour() failed with 'selected part is empty' (v7: mug, bucket, teapot, kettle). Points that are not finite or
    lie > k robust deviations (MAD) from the object's median are replaced by their nearest good neighbour."""
    from scipy.ndimage import distance_transform_edt

    P = points.astype(np.float64).copy()
    m = mask > 0
    good = np.isfinite(P).all(-1)
    obj = P[m & good]
    if len(obj) >= 10:
        med = np.median(obj, 0)
        mad = np.median(np.abs(obj - med), 0) * 1.4826 + 1e-9
        good &= (np.abs(np.nan_to_num(P, nan=np.inf) - med) <= k * mad).all(-1) | ~m
    if not good.any():
        return np.zeros_like(P)
    if (~good).any():
        _, (iy, ix) = distance_transform_edt(~good, return_indices=True)
        P = P[iy, ix]
    return P


@dataclass
class Frame:
    """Geometry of the object: pixel coords, PCA axes, scale (all from the binary mask)."""
    mask: np.ndarray
    frame: str = "object"
    points: np.ndarray | None = None      # optional H x W x 3 camera-space points (MoGe): patterns follow the surface
    xs: np.ndarray = field(init=False)
    ys: np.ndarray = field(init=False)
    axes: np.ndarray = field(init=False)
    scale: float = field(init=False)

    def __post_init__(self):
        ys, xs = np.nonzero(self.mask)
        if len(xs) < 10:
            raise ValueError("object mask is (nearly) empty")
        self.ys, self.xs = ys, xs
        self.scale = float(np.sqrt(len(xs)))
        if self.points is not None:
            self.points = clean_points(self.points, self.mask)
            P = self.points[ys, xs]
            ext = np.percentile(P, 99, 0) - np.percentile(P, 1, 0)      # robust size: ignores stray depth points
            self.scale3 = float(np.sqrt(max(ext[0] * ext[1], 1e-12)))
            # vertical axis of a round object: through the median X, behind the visible surface by ~half its width
            self.cx = float(np.median(P[:, 0]))
            self.radius = max(float(ext[0]) / 2, 1e-6)
            self.cz = float(np.median(P[:, 2])) + self.radius
        if self.frame == "object":
            P = np.stack([xs, ys], 1).astype(np.float64)
            P -= P.mean(0)
            _, vecs = np.linalg.eigh(P.T @ P / len(P))
            self.axes = vecs.T[::-1]          # major axis first
            # handles, arms and straps tilt the shape axis of upright objects: within 20 deg of the photo's own axes,
            # use the photo's axes (a mug's stripes stay horizontal); only clearly tilted objects keep their own frame
            ang = np.degrees(np.arctan2(self.axes[0][1], self.axes[0][0])) % 90
            if min(ang, 90 - ang) < 20:
                self.axes = np.eye(2)
        else:
            self.axes = np.eye(2)

    def direction(self, side: str) -> np.ndarray:
        """Unit vector for 'upper' etc.: the object axis closest to the image direction, sign-aligned."""
        d = np.array(_DIRS[side], np.float64)
        if self.frame != "object":
            return d
        k = int(np.argmax(np.abs(self.axes @ d)))
        u = self.axes[k]
        return u if u @ d > 0 else -u

    def coord(self, u: np.ndarray) -> np.ndarray:
        """t(p) = u . p for every pixel of the image (H x W). With a point map, p is the pixel's 3-D position
        (image right -> camera X, image down -> camera Y), so a cut at constant t is a plane through the object,
        which projects as a curve that follows its surface."""
        if self.points is not None:
            P = np.nan_to_num(self.points.astype(np.float64))
            return u[0] * P[..., 0] + u[1] * P[..., 1]
        H, W = self.mask.shape
        yy, xx = np.mgrid[0:H, 0:W]
        return u[0] * xx + u[1] * yy

    def around(self) -> np.ndarray:
        """Arc length around the object's vertical axis (3-D mode): vertical stripes wrap around a round object."""
        P = np.nan_to_num(self.points.astype(np.float64))
        return self.radius * np.arctan2(P[..., 0] - self.cx, self.cz - P[..., 2])

    def quantile_cut(self, u: np.ndarray, frac: float) -> float:
        """c such that the part {t > c} holds exactly ``frac`` of the object's area (not the bbox midline)."""
        t = self.coord(u)[self.ys, self.xs] if self.points is not None else u[0] * self.xs + u[1] * self.ys
        return float(np.quantile(t, 1 - frac))


def _sig(x: np.ndarray) -> np.ndarray:
    """Compact S-curve of a signed distance in units of tau: exactly 0 below -2, exactly 1 above +2
    (smoothstep - sigmoid-shaped, but with a hard guarantee: nothing changes beyond 2 tau from the cut)."""
    u = np.clip((np.asarray(x, np.float64) + 2) / 4, 0, 1)
    return u * u * (3 - 2 * u)


def evaluate(expr: dict, F: Frame, tau: float, parts: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """Soft mask in [0, 1] for one expression (not yet intersected with the object)."""
    op = expr.get("op", "object")
    if op == "object":
        return np.ones(F.mask.shape)
    if op in ("half", "band"):
        u = F.direction(expr.get("side", "upper"))
        t = F.coord(u)
        if op == "half":
            c = F.quantile_cut(u, float(expr.get("frac", 0.5)))
            return _sig((t - c) / tau)
        f0, f1 = float(expr.get("from", 0.0)), float(expr.get("to", 0.25))
        c0, c1 = F.quantile_cut(u, f0) if f0 > 0 else np.inf, F.quantile_cut(u, f1)
        inner = _sig((c0 - t) / tau) if np.isfinite(c0) else 1.0
        return inner * _sig((t - c1) / tau)
    if op == "stripes":
        side = "lower" if expr.get("axis", "horizontal") == "horizontal" else "right"
        u = F.direction(side)
        t = F.around() if (F.points is not None and side == "right") else F.coord(u)
        tm = t[F.ys, F.xs] if F.points is not None else u[0] * F.xs + u[1] * F.ys
        lo, hi = float(tm.min()), float(tm.max())
        n = max(1, int(expr.get("count", 3)))
        period = (hi - lo) / (2 * n)                  # n coloured + n untouched stripes
        width = float(expr.get("width", 0.5))         # coloured share of each cycle (0.5 = equal bands; 0.2 = thin stripes)
        cyc = 2 * period
        on = width * cyc
        q = np.mod(t - lo - (cyc - on) / 2, cyc)      # stripes centred in their cycle
        d = np.where(q < on, np.minimum(q, on - q), -np.minimum(q - on, cyc - q))
        floor = 0.5 if F.points is None else 1e-9
        return _sig(d / max(floor, min(tau, on / 6)))  # d = exact signed distance; edge <= 1/3 stripe wide
    if op == "border":
        d = cv2.distanceTransform(F.mask.astype(np.uint8), cv2.DIST_L2, 5)
        w = float(expr.get("width", 0.06)) * F.scale
        return _sig((w - d) / tau)
    if op == "part":
        name = expr.get("name", "")
        if not parts or name not in parts:
            raise KeyError(f"part {name!r} needs a mask (VLM box -> SAM)")
        m = parts[name].astype(np.float64)
        return cv2.GaussianBlur(m, (0, 0), max(tau, 0.5)) if tau > 0 else m
    if op == "and":
        out = np.ones(F.mask.shape)
        for a in expr["args"]:
            out = out * evaluate(a, F, tau, parts)
        return out
    if op == "or":
        out = np.zeros(F.mask.shape)
        for a in expr["args"]:
            e = evaluate(a, F, tau, parts)
            out = out + e - out * e              # probabilistic OR: smooth, = max on hard masks
        return out
    if op == "minus":
        a, b = expr["args"]
        return evaluate(a, F, tau, parts) * (1 - evaluate(b, F, tau, parts))
    if op == "not":
        return 1 - evaluate(expr["arg"], F, tau, parts)
    raise ValueError(f"unknown op {op!r}")


def region_weight(object_mask: np.ndarray, program: dict, parts: dict | None = None,
                  points: np.ndarray | None = None) -> np.ndarray:
    """Final soft weight W = M (.) expr, snapped to exact 0/1 outside the seam band (the guarantee).
    ``points``: optional camera-space point map (H x W x 3, e.g. MoGe-2) -> halves, bands and stripes follow the 3-D
    surface instead of the flat picture."""
    M = object_mask > 0
    F = Frame(M, program.get("frame", "object"), points)
    soft = float(program.get("softness", 0.006))
    tau = soft * F.scale3 if points is not None else max(0.75, soft * F.scale)
    W = evaluate(program.get("region", {"op": "object"}), F, tau, parts) * M
    eps = 1 / 510
    W = np.where(W < eps, 0.0, np.where(W > 1 - eps, 1.0, W))
    return W


# ----------------------------------------------------------------------------- recolour


@dataclass
class RecolourResult:
    image: np.ndarray            # uint8 H x W x 3
    weight: np.ndarray           # final per-pixel alpha (region x colour affinity)
    region: np.ndarray           # geometric region weight W
    info: dict
    metrics: dict


def recolour(image: np.ndarray, object_mask: np.ndarray, program: dict,
             parts: dict | None = None, points: np.ndarray | None = None) -> RecolourResult:
    I = np.asarray(image)[..., :3]
    W = region_weight(object_mask, program, parts, points)
    lab = rgb_to_oklab(I)
    core = W > 0.5
    if core.sum() < 10:
        raise ValueError("selected part is empty")
    L, ab = lab[..., 0], lab[..., 1:]
    # dominant colour of the part (median is robust to logos/zips) and its spread
    dom_L = float(np.median(L[core]))
    dom_ab = np.median(ab[core], axis=0)
    # affinity: pixels whose colour is far from the dominant one (accents, logos, zips) are left alone
    dist = np.linalg.norm(ab - dom_ab, axis=-1)
    mad = float(np.median(dist[core]))
    sig = max(0.035, 2.5 * mad)
    A = np.exp(-0.5 * (np.maximum(dist - 0.5 * sig, 0) / sig) ** 2) if program.get("affinity", True) else np.ones_like(L)
    tgt = rgb_to_oklab(parse_colour(program["target_color"])[None])[0]
    lam = float(program.get("lightness", 0.7))     # 0 keep the part's own lightness, 1 match the target's
    # chroma follows shading: in a shadow at half the lightness, the colour is about half as saturated
    r = np.clip(L / max(dom_L, 1e-3), 0, 1.6)[..., None]
    new_ab = tgt[1:] * r + (ab - dom_ab * r)
    new_L = np.clip(L + lam * (tgt[0] - dom_L), 0, 1)
    R = oklab_to_rgb(gamut_fit(np.concatenate([new_L[..., None], new_ab], -1)))
    alpha = W * A
    alpha = np.where(alpha < 1 / 510, 0.0, alpha)
    out = I.astype(np.float64) + alpha[..., None] * (R - I)
    out = np.where(alpha[..., None] > 0, np.clip(np.rint(out), 0, 255), I).astype(np.uint8)
    info = {"dominant_rgb": [int(v) for v in np.clip(oklab_to_rgb(np.r_[dom_L, dom_ab][None])[0], 0, 255)],
            "target_rgb": [int(v) for v in parse_colour(program["target_color"])],
            "affinity_sigma": round(sig, 4), "tau_px": round(max(0.75, float(program.get("softness", 0.006))
                                                                   * np.sqrt((object_mask > 0).sum())), 2)}
    res = RecolourResult(out, alpha, W, info, {})
    res.metrics = verify(I, out, W, alpha, tgt)
    return res


def verify(before: np.ndarray, after: np.ndarray, W: np.ndarray, alpha: np.ndarray, tgt_oklab: np.ndarray) -> dict:
    """Numbers, not a VLM: leakage, target accuracy, shading kept, seam width."""
    prot = W == 0
    edited = (W >= 0.99) & (alpha > 0.5)
    lab_b, lab_a = rgb_to_cielab(before), rgb_to_cielab(after)
    de = delta_e2000(lab_b[prot], lab_a[prot]) if prot.any() else np.zeros(1)
    ok_a = rgb_to_oklab(after)
    m = {"protected_identical": bool((before[prot] == after[prot]).all()),
         "leak_dE00_max": round(float(de.max()), 4),
         "seam_band_px": int(((W > 0) & (W < 1)).sum()),
         "edited_px": int(edited.sum())}
    if edited.any():
        mab = ok_a[edited][:, 1:].mean(0)
        h_out = np.degrees(np.arctan2(mab[1], mab[0])) % 360
        h_tgt = np.degrees(np.arctan2(tgt_oklab[2], tgt_oklab[1])) % 360
        m["hue_out_deg"], m["hue_target_deg"] = round(float(h_out), 1), round(float(h_tgt), 1)
        m["hue_err_deg"] = round(float(abs((h_out - h_tgt + 180) % 360 - 180)), 1)
        Lb, La = rgb_to_oklab(before)[edited][:, 0], ok_a[edited][:, 0]
        m["shading_corr"] = round(float(np.corrcoef(Lb, La)[0, 1]), 4) if Lb.std() > 1e-6 else 1.0
    return m


# ----------------------------------------------------------------------------- instruction -> program

PARSE_PROMPT = """You turn a recolouring request into a JSON program. Reply with JSON only.
Schema: {"object": <noun>, "target_color": <colour name or #rrggbb>, "frame": "object",
 "region": EXPR}
EXPR is one of:
 {"op":"object"}                                   the whole object
 {"op":"half","side":"upper|lower|left|right","frac":0.5}   the part holding that fraction of the object's area
 {"op":"band","side":"upper|lower|left|right","from":0.0,"to":0.2}   a band, as area fractions from that side
 {"op":"stripes","axis":"horizontal|vertical","count":N}
 {"op":"border","width":0.06}                      the rim/trim of the object
 {"op":"part","name":<short noun>}                 a named sub-part such as "pocket", "strap", "left sleeve"
 {"op":"and"|"or","args":[EXPR,...]}, {"op":"minus","args":[EXPR,EXPR]}, {"op":"not","arg":EXPR}
Examples:
 "make the upper half of the bag green" -> {"object":"bag","target_color":"green","frame":"object","region":{"op":"half","side":"upper","frac":0.5}}
 "bottom third red except the pocket" -> {"object":"bag","target_color":"red","frame":"object","region":{"op":"minus","args":[{"op":"half","side":"lower","frac":0.333},{"op":"part","name":"pocket"}]}}
Request: """


def _json_block(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON in {text[:120]!r}")
    return json.loads(m.group(0))


def validate(program: dict) -> dict:
    parse_colour(program["target_color"])

    def walk(e):
        op = e.get("op")
        if op not in {"object", "half", "band", "stripes", "border", "part", "and", "or", "minus", "not"}:
            raise ValueError(f"bad op {op!r}")
        if op in ("and", "or", "minus"):
            for a in e["args"]:
                walk(a)
        if op == "not":
            walk(e["arg"])
        if op in ("half", "band") and e.get("side", "upper") not in _DIRS:
            raise ValueError(f"bad side {e.get('side')!r}")
    walk(program.get("region", {"op": "object"}))
    return program


_FRAC = {"half": 0.5, "third": 1 / 3, "quarter": 0.25, "fourth": 0.25, "fifth": 0.2}
_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}


def rule_parse(text: str) -> dict:
    """Offline fallback parser for the common phrasings (also the VLM's safety net)."""
    t = text.lower()
    colour = next((c for c in sorted(COLOURS, key=len, reverse=True) if re.search(rf"\b{c}\b", t)), None)
    hexm = re.search(r"#[0-9a-f]{6}", t)
    colour = hexm.group(0) if hexm else colour
    if colour is None:
        raise ValueError("no target colour found")
    region: dict = {"op": "object"}
    m = re.search(r"\b(upper|top|lower|bottom|left|right)\s+(half|third|quarter|fourth|fifth)\b", t)
    if m:
        region = {"op": "half", "side": m.group(1), "frac": round(_FRAC[m.group(2)], 4)}
    m = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight)?\s*(horizontal|vertical)?\s*stripes?\b", t)
    if m and "stripe" in t:
        n = m.group(1)
        region = {"op": "stripes", "axis": m.group(2) or "horizontal",
                  "count": int(n) if n and n.isdigit() else _NUM.get(n or "", 3)}
    if re.search(r"\b(border|trim|rim|edge|edges|outline)\b", t):
        region = {"op": "border", "width": 0.06}
    m = re.search(r"\b(?:person|man|woman|boy|girl)'?s ([a-z][a-z-]*)", t)
    if m:
        region = {"op": "part", "name": m.group(1)}
    return {"target_color": colour, "frame": "object", "region": region, "parser": "rules"}


_SAME = [{"person", "man", "woman", "boy", "girl", "people", "child", "kid", "lady", "guy"}]


def resolve_object(program: dict, scene_object: str | None) -> dict:
    """If the parser names a different object than the one whose mask we hold (e.g. object "shirt" in a photo of a
    person), that name is a PART of the scene object: region := part(name) AND region. (v1 recoloured the whole
    person for "make the person's shirt teal" because Qwen3-VL answered object="shirt", region=object.)"""
    o = str(program.get("object") or "").lower().strip()
    sc = str(scene_object or "").lower().strip()
    if not o or not sc or o in sc or sc in o or any(o in g and sc in g for g in _SAME):
        return program
    region = program.get("region", {"op": "object"})
    part = {"op": "part", "name": o}
    program = {**program, "region": part if region.get("op") == "object" else {"op": "and", "args": [part, region]},
               "object_resolved": f"{o} -> part of {sc}"}
    return program


def parse_instruction(text: str, ask: Callable[[str], str] | None = None, scene_object: str | None = None) -> dict:
    """VLM (``ask(prompt) -> reply``) first, rules as fallback; always returns a validated program.
    ``scene_object``: what the object mask covers; a different object named in the request becomes a part of it."""
    if ask is not None:
        try:
            p = validate(_json_block(ask(PARSE_PROMPT + text)))
            p.setdefault("frame", "object")
            p["parser"] = "vlm"
            return validate(resolve_object(p, scene_object))
        except Exception:  # noqa: BLE001
            pass
    return validate(rule_parse(text))


# ----------------------------------------------------------------------------- parts: VLM box -> SAM

GROUND_PROMPT = ('Locate {what} in the image. Output JSON: [{{"bbox_2d": [x1, y1, x2, y2], "label": "{label}"}}]. '
                 'If it is not visible, output [].')


def ground_box(ask: Callable[[str], str], size: tuple[int, int], what: str) -> tuple[int, int, int, int] | None:
    """Qwen3-VL grounding -> pixel box (Qwen3-VL answers on a 0-1000 relative scale), or None."""
    raw = ask(GROUND_PROMPT.format(what=what, label=what.split()[-1]))
    m = re.search(r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]", raw)
    if not m:
        return None
    x0, y0, x1, y1 = (float(g) for g in m.groups())
    w, h = size
    box = (int(x0 * w / 1000), int(y0 * h / 1000), int(x1 * w / 1000), int(y1 * h / 1000))
    return box if box[2] > box[0] + 2 and box[3] > box[1] + 2 else None


def box_to_mask(image, box: tuple[int, int, int, int], within: np.ndarray | None = None) -> np.ndarray:
    """SAM 2.1 inside a box (best-scored candidate), optionally intersected with an outer mask."""
    from editing.masking import sam_masks

    cands = sam_masks(image, [], box=box)
    if not cands:
        return np.zeros(image.size[::-1], np.uint8)
    m = cands[0].mask.astype(np.uint8)
    return (m & (within > 0)).astype(np.uint8) if within is not None else m


def part_names(expr: dict) -> list[str]:
    """All {"op": "part"} names used by a program's region."""
    if expr.get("op") == "part":
        return [expr.get("name", "")]
    out = []
    for a in expr.get("args", []):
        out += part_names(a)
    if "arg" in expr:
        out += part_names(expr["arg"])
    return out


# ----------------------------------------------------------------------------- hybrid: Qwen decides WHERE, maths decides HOW


def region_from_edit(before: np.ndarray, edited: np.ndarray, obj: np.ndarray, thresh: float = 12.0,
                     min_px: int = 200) -> np.ndarray:
    """Where did a generative edit change the colour of the object? CIEDE2000 > ``thresh`` on the object, cleaned
    (open/close, small blobs dropped). Used as a {"op": "part"} mask so the exact recolour follows Qwen's semantics
    (real stripes, the sole as 'trim') without its side effects (shading loss, background drift, pixel shift)."""
    if edited.shape != before.shape:
        from PIL import Image as _I

        edited = np.asarray(_I.fromarray(edited).resize(before.shape[1::-1], _I.LANCZOS))
    de = delta_e2000(rgb_to_cielab(cv2.GaussianBlur(before, (0, 0), 1.0)),
                     rgb_to_cielab(cv2.GaussianBlur(edited, (0, 0), 1.0)))
    m = ((de > thresh) & (obj > 0)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_px
    return (keep[lab] & (obj > 0)).astype(np.uint8)
