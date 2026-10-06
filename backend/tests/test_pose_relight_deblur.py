"""Model-free parts of editing.pose / relight / deblur."""
import numpy as np

from editing import deblur as D, pose as P, relight as R


def skel(dx=0.0, scale=1.0, arm_up=False):
    k = np.zeros((18, 3), np.float32)
    pts = {0: (50, 10), 1: (50, 25), 2: (40, 25), 3: (35, 45), 4: (33, 65), 5: (60, 25), 6: (65, 45), 7: (67, 65),
           8: (44, 70), 9: (44, 95), 10: (44, 120), 11: (56, 70), 12: (56, 95), 13: (56, 120)}
    if arm_up:
        pts[3], pts[4] = (35, 10), (33, -5)
    for i, (x, y) in pts.items():
        k[i] = (x * scale + dx, y * scale, 1)
    return k


def test_pck_is_position_and_scale_invariant():
    a = skel()
    assert P.pck(a, skel(dx=200, scale=2.0)) == 1.0
    assert P.pck(a, skel(arm_up=True)) < 1.0


def test_retarget_moves_target_onto_source():
    src, tgt = skel(dx=300, scale=2.0), skel(arm_up=True)
    r = P.retarget(tgt, src)
    assert abs(r[r[:, 2] > 0, 0].mean() - src[src[:, 2] > 0, 0].mean()) < 40
    assert P.pck(r, tgt) == 1.0


def test_draw_skeleton():
    im = P.draw_skeleton(skel(dx=20), (160, 160))
    assert (np.asarray(im).sum(-1) > 0).mean() > 0.02


def test_kernel_estimate_and_consistency():
    rng = np.random.default_rng(0)
    S = (rng.random((128, 160, 3)) * 255).astype(np.uint8)
    S = np.asarray(__import__("cv2").GaussianBlur(S, (0, 0), 1.2))
    k = D.line_kernel(17, 35)
    B = D.blur(S, k)
    _, info = D.estimate_kernel(B)
    assert abs(info["length"] - 17) <= 2.5
    R1 = D.richardson_lucy(B, k, init=S, iters=10)
    assert np.abs(R1.astype(int) - S).mean() < 6


def test_light_transfer_keeps_detail():
    rng = np.random.default_rng(1)
    P0 = (rng.random((96, 128, 3)) * 200 + 20).astype(np.uint8)
    g = np.linspace(1.5, 0.5, 128)[None, :, None]
    Q = R._srgb(R._lin(P0) * g)
    import cv2
    Qb = cv2.GaussianBlur(Q, (0, 0), 1.5)
    T = R.transfer(P0, Qb)
    assert R.light_effect(P0, T, "left") > 0.8 * R.light_effect(P0, Qb, "left")
    assert R.detail_kept(P0, T) > R.detail_kept(P0, Qb) + 0.2


def test_describe_pose_and_corrections():
    a, b = skel(), skel(arm_up=True)
    d = P.describe_pose(b)
    assert "right upper arm is raised" in d and "left upper arm is pointing straight down" in d and "faces the camera" in d
    fix = P.corrections(a, b)
    assert "right upper arm should be raised" in fix
    assert P.corrections(b, b) == ""
