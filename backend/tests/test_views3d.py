"""Re-projection maths for editing.views3d on a synthetic scene (no models)."""
import numpy as np

from editing import views3d as V


def scene(H=120, W=160):
    f = 160.0
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    Z = np.full((H, W), 6.0)                          # back wall at 6 m
    box = (np.abs(xx - W / 2) < 25) & (np.abs(yy - H / 2) < 25)
    Z[box] = 3.0                                      # box 3 m away
    X = (xx - K[0, 2]) / f * Z
    Y = (yy - K[1, 2]) / f * Z
    img = np.zeros((H, W, 3), np.uint8)
    img[..., 0] = (xx * 1.5).astype(np.uint8)
    img[..., 1] = 120
    img[box] = (230, 40, 40)
    return V.Geometry(np.stack([X, Y, Z], -1), np.ones((H, W), bool), K, img), box


def test_identity_view():
    g, box = scene()
    r, valid, _ = V.render_view(g, yaw=0, pitch=0)
    assert valid.mean() > 0.995
    assert np.abs(r.astype(int) - g.image).max() <= 2          # splat neighbours only touch edges


def test_yaw_opens_a_gap_beside_the_box():
    g, box = scene()
    r, valid, _ = V.render_view(g, yaw=15, subject=box)
    holes = ~valid
    import cv2
    n, lab, st, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8))
    H, W = holes.shape
    inner = [i for i in range(1, n) if st[i, 0] > 0 and st[i, 0] + st[i, 2] < W]   # not touching the left/right border
    assert inner and sum(st[i, 4] for i in inner) > 30          # a gap opened beside the box (hidden wall)
    assert holes.mean() < 0.35                                  # the rest is new view area outside the old frame
    red = (r[..., 0] > 200) & (r[..., 1] < 80)
    assert abs(red.sum() - box.sum()) / box.sum() < 0.35       # the box is still there, about the same size
    # the pivot (box) stays centred while the wall slides -> box centre moves little
    ys, xs = np.nonzero(red)
    assert abs(xs.mean() - 80) < 6
    r2, v2, _ = V.render_view(g, yaw=-15, subject=box)
    hx = np.nonzero(~valid)[1].mean(); hx2 = np.nonzero(~v2)[1].mean()
    assert (hx - 80) * (hx2 - 80) < 0                          # opposite turns open gaps on opposite sides


def test_hole_mask_and_fidelity():
    g, box = scene()
    r, valid, _ = V.render_view(g, yaw=12, subject=box)
    h = V.hole_mask(valid)
    assert h.sum() >= (~valid).sum()
    f = V.warp_fidelity(r, r, valid)
    assert f["geo_dE"] < 1e-6


def test_object3d_cutout_and_turntable():
    import trimesh
    from PIL import Image
    from editing import object3d as O3
    img = Image.new("RGB", (200, 150), (90, 90, 90))
    m = np.zeros((150, 200), np.uint8); m[40:110, 60:120] = 1
    c = O3.cutout(img, m)
    a = np.asarray(c)
    assert c.size == (512, 512) and a.shape[2] == 4 and 0.2 < (a[..., 3] > 0).mean() < 0.8
    box = trimesh.creation.box((1.0, 2.0, 0.5))
    box.visual.vertex_colors = np.tile([200, 30, 30, 255], (len(box.vertices), 1))
    views = O3.turntable(box, views=4, size=96, n_points=20000)
    assert len(views) == 4
    fg = [(np.asarray(v) != 245).any(-1).mean() for v in views]
    assert all(f > 0.02 for f in fg) and fg[0] != fg[1]          # object visible; width changes as it turns
