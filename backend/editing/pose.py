"""Target-pose reconstruction of a person (re-pose someone from one photo into another pose).

Qwen-Image 2.1 takes several reference images, so the target pose can be given three ways:
  skeleton  <image2> = an OpenPose-style stick figure of the target pose
  photo     <image2> = a photo of someone else in the target pose
  words     Qwen3-VL describes the target pose in words; single-image instruction edit
Scored with the pose detector itself: detect the pose in the result and compare it with the target (PCK: share of
joints within 0.2 x torso length after aligning position and scale), plus identity (same person + clothes, Qwen3-VL).
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

# 18-joint OpenPose/COCO order: nose, neck, r-shoulder, r-elbow, r-wrist, l-shoulder, l-elbow, l-wrist, r-hip, r-knee,
# r-ankle, l-hip, l-knee, l-ankle, r-eye, l-eye, r-ear, l-ear
LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0),
         (0, 14), (14, 16), (0, 15), (15, 17)]
COLOURS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
           (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255),
           (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85)]

_DET = None


def detector():
    global _DET
    if _DET is None:
        from controlnet_aux import OpenposeDetector

        _DET = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
        try:
            _DET = _DET.to("cuda")
        except Exception:  # noqa: BLE001
            pass
    return _DET


def keypoints(image: Image.Image) -> np.ndarray | None:
    """Largest person's 18 joints as an (18, 3) array: x, y in pixels, visible flag; None if nobody is found."""
    det = detector()
    arr = np.asarray(image.convert("RGB"))
    H, W = arr.shape[:2]
    poses = det.detect_poses(arr)
    best, best_n = None, 0
    for p in poses or []:
        kps = getattr(p, "body", p).keypoints
        pts = np.zeros((18, 3), np.float32)
        for i, k in enumerate(kps[:18]):
            if k is not None:
                pts[i] = (k.x * W, k.y * H, 1.0)
        span = np.ptp(pts[pts[:, 2] > 0, :2], 0).prod() if (pts[:, 2] > 0).sum() > 1 else 0
        if pts[:, 2].sum() > 0 and (best is None or span > best_n):
            best, best_n = pts, span
    return best


def draw_skeleton(kp: np.ndarray, size: tuple[int, int], width: int = 6) -> Image.Image:
    """OpenPose-style coloured stick figure on black."""
    img = Image.new("RGB", size, (0, 0, 0))
    d = ImageDraw.Draw(img)
    for i, (a, b) in enumerate(LIMBS):
        if kp[a, 2] > 0 and kp[b, 2] > 0:
            d.line([tuple(kp[a, :2]), tuple(kp[b, :2])], fill=COLOURS[i % len(COLOURS)], width=width)
    for i in range(18):
        if kp[i, 2] > 0:
            x, y = kp[i, :2]
            d.ellipse([x - width, y - width, x + width, y + width], fill=COLOURS[i])
    return img


def retarget(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Move/scale the target skeleton onto the source person's position and size (so the person stays where they are)."""
    def anchor(k):
        v = k[:, 2] > 0
        c = k[v, :2].mean(0)
        s = np.ptp(k[v, 1]) if v.sum() > 1 else 1.0
        return c, max(s, 1.0)
    ct, st = anchor(target)
    cs, ss = anchor(source)
    out = target.copy()
    out[:, :2] = (target[:, :2] - ct) * (ss / st) + cs
    return out


def pck(a: np.ndarray, b: np.ndarray, thr: float = 0.2) -> float | None:
    """Share of joints (visible in both) within thr x torso length, after aligning centre and scale."""
    v = (a[:, 2] > 0) & (b[:, 2] > 0)
    if v.sum() < 4:
        return None

    def norm(k):
        p = k[:, :2] - k[v, :2].mean(0)
        torso = np.linalg.norm(k[1, :2] - (k[8, :2] + k[11, :2]) / 2) if (k[[1, 8, 11], 2] > 0).all() else np.ptp(k[v, 1]) / 3
        return p / max(torso, 1e-3)
    d = np.linalg.norm(norm(a)[v] - norm(b)[v], axis=1)
    return float((d < thr).mean())


DESCRIBE_Q = ("Describe the body pose of the main person in this photo precisely in one or two sentences: position of "
              "the arms, hands, legs and torso, and which way they face. Do not describe clothing or the background.")


def prompt_for(method: str, pose_words: str = "") -> str:
    keep = " Keep the same person, face, hairstyle, clothing and the same background and lighting."
    if method == "skeleton":
        return "Make the person in <image1> take exactly the pose of the stick figure in <image2>." + keep
    if method == "photo":
        return "Make the person in <image1> take exactly the same body pose as the person in <image2>." + keep
    return f"Change the pose of the person in this photo: {pose_words}" + keep


def repose(pipe, Q, person: Image.Image, method: str, target_img: Image.Image | None = None, pose_words: str = "",
           seed: int = 7, steps: int = 40, resolution: int = 1024):
    import time

    import torch

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    if method == "words":
        r = Q.edit(pipe, person, prompt_for("words", pose_words), seed=seed, steps=steps, resolution=resolution)
        img = r.image
    else:
        img = pipe(prompt=prompt_for(method), image=[person.convert("RGB"), target_img.convert("RGB")],
                   num_inference_steps=steps, output_resolution=resolution,
                   generator=torch.Generator("cpu").manual_seed(seed)).images[0]
    out = img.convert("RGB").resize(person.size, Image.LANCZOS)
    return out, {"seconds": round(time.perf_counter() - t0, 2), "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}


# ----------------------------------------------------------------------------- precise words from geometry

# limbs as (name, joint a, joint b); "right"/"left" are the PERSON's own sides (OpenPose convention)
SEGMENTS = [("right upper arm", 2, 3), ("right forearm", 3, 4), ("left upper arm", 5, 6), ("left forearm", 6, 7),
            ("right thigh", 8, 9), ("right lower leg", 9, 10), ("left thigh", 11, 12), ("left lower leg", 12, 13)]


def _dir(kp, a, b):
    if kp[a, 2] <= 0 or kp[b, 2] <= 0:
        return None
    v = kp[b, :2] - kp[a, :2]
    n = np.linalg.norm(v)
    return None if n < 1e-3 else v / n


def _angle_word(v) -> str:
    """Image-space direction (y down) -> words. Angle measured from straight down."""
    ang = np.degrees(np.arctan2(v[0], v[1]))          # 0 = down, +90 = image right, -90 = image left, 180 = up
    a = abs(ang)
    side = "towards the right of the picture" if ang > 0 else "towards the left of the picture"
    if a < 25:
        return "pointing straight down"
    if a < 65:
        return f"angled down and out {side}"
    if a < 115:
        return f"held out horizontally {side}"
    if a < 155:
        return f"raised up and out {side}"
    return "raised straight up"


def _bend(kp, a, b, c) -> float | None:
    u, v = _dir(kp, b, a), _dir(kp, b, c)
    if u is None or v is None:
        return None
    return float(np.degrees(np.arccos(np.clip(u @ v, -1, 1))))   # 180 = straight


def describe_pose(kp: np.ndarray) -> str:
    """Deterministic, precise pose description computed from 18 OpenPose joints."""
    parts = []
    for side, (s, e, w) in (("right", (2, 3, 4)), ("left", (5, 6, 7))):
        up, lo = _dir(kp, s, e), _dir(kp, e, w)
        bend = _bend(kp, s, e, w)
        if up is None:
            continue
        txt = f"the person's {side} upper arm is {_angle_word(up)}"
        if lo is not None:
            txt += f", forearm {_angle_word(lo)}"
        if bend is not None:
            txt += ", elbow straight" if bend > 155 else f", elbow bent to about {int(round(bend / 15) * 15)} degrees"
        parts.append(txt)
    for side, (h, k, a) in (("right", (8, 9, 10)), ("left", (11, 12, 13))):
        th, sh = _dir(kp, h, k), _dir(kp, k, a)
        bend = _bend(kp, h, k, a)
        if th is None:
            continue
        txt = f"the {side} thigh is {_angle_word(th)}"
        if sh is not None:
            txt += f", lower leg {_angle_word(sh)}"
        if bend is not None:
            txt += ", knee straight" if bend > 155 else f", knee bent to about {int(round(bend / 15) * 15)} degrees"
        parts.append(txt)
    if (kp[[1, 8, 11], 2] > 0).all():
        hip = (kp[8, :2] + kp[11, :2]) / 2
        t = kp[1, :2] - hip
        lean = np.degrees(np.arctan2(t[0], -t[1]))
        parts.append("the torso is upright" if abs(lean) < 12 else
                     f"the torso leans {'towards the right of the picture' if lean > 0 else 'towards the left of the picture'} "
                     f"by about {int(round(abs(lean) / 5) * 5)} degrees")
        hips_y, knees_y = hip[1], np.mean([kp[i, 1] for i in (9, 12) if kp[i, 2] > 0] or [hip[1] + 1])
        if knees_y - hips_y < 0.25 * np.linalg.norm(t):
            parts.append("the person is sitting or crouching (knees about level with the hips)")
    if kp[0, 2] > 0 and kp[2, 2] > 0 and kp[5, 2] > 0:
        sw = kp[5, 0] - kp[2, 0]          # person's left shoulder minus right shoulder (image x)
        parts.append("the person faces the camera" if sw > 0 else "the person has their back to the camera")
    return "; ".join(parts) + "."


def corrections(result: np.ndarray, target: np.ndarray, thr_deg: float = 30.0, max_items: int = 3) -> str:
    """Feedback for the next round: limbs whose direction differs from the target by more than thr_deg."""
    errs = []
    for name, a, b in SEGMENTS:
        r, t = _dir(result, a, b), _dir(target, a, b)
        if r is None or t is None:
            continue
        d = float(np.degrees(np.arccos(np.clip(r @ t, -1, 1))))
        if d > thr_deg:
            errs.append((d, f"the person's {name} should be {_angle_word(t)} (it is currently {_angle_word(r)})"))
    errs.sort(reverse=True)
    return "; ".join(e for _, e in errs[:max_items])


def repose_loop(pipe, Q, person: Image.Image, target_kp: np.ndarray, rounds: int = 3, seed: int = 7,
                same_person=None, steps: int = 40, resolution: int = 1024, gist: str = ""):
    """Precise words -> edit -> detect pose -> say what is wrong -> edit again. Keeps the best round (highest PCK among
    results judged to be the same person). ``same_person(img) -> bool`` (Qwen3-VL check), optional."""
    words = describe_pose(target_kp)
    best, history = None, []
    cur = person
    if gist:   # overall description first (e.g. "sitting cross-legged on the floor"), exact joint angles second
        instr = (f"Change the pose of the person in this photo to this pose: {gist.strip().rstrip('.')}. Precisely: "
                 f"{words} Keep the same person, face, hairstyle, clothing, background and lighting.")
    else:
        instr = (f"Change the pose of the person in this photo so that {words} Keep the same person, face, hairstyle, "
                 f"clothing, background and lighting.")
    for r in range(rounds):
        res = Q.edit(pipe, cur, instr, seed=seed + r, steps=steps, resolution=resolution)
        o = res.image.convert("RGB").resize(person.size, Image.LANCZOS)
        kp = keypoints(o)
        score = pck(kp, target_kp) if kp is not None else 0.0
        ok = True if same_person is None else bool(same_person(o))
        history.append({"round": r, "pck": score, "same_person": ok, "instruction": instr[:400], "seconds": res.seconds})
        if ok and (best is None or (score or 0) > (best[1] or 0)):
            best = (o, score, r)
        if kp is None or (score or 0) >= 0.9:
            break
        fix = corrections(kp, target_kp)
        if not fix:
            break
        cur = o if ok else cur              # continue from the result only if identity survived
        instr = (f"Adjust the pose of the person in this photo: {fix}. Keep everything else exactly the same: same "
                 f"person, face, clothing, background and lighting.")
    if best is None:
        best = (o, score, len(history) - 1)
    return best[0], {"best_round": best[2], "pck": best[1], "history": history, "words": words}
