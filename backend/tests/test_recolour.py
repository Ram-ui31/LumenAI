"""Recolour + mask algebra: exactness and geometry checks on a synthetic bag (CPU, < 2 s)."""
import numpy as np
import pytest

from editing import recolour as RC


def bag(h=240, w=200, tilt=0.0):
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.full((h, w, 3), 205, np.uint8)
    # tapered body: wider at the bottom, so area-half != bbox-midline
    cx = w / 2 + tilt * (yy - h / 2)
    half_w = 40 + 40 * (yy - 30) / (h - 60)
    M = ((yy >= 30) & (yy < h - 30) & (np.abs(xx - cx) < half_w)).astype(np.uint8)
    shade = 0.65 + 0.35 * (1 - np.abs(xx - cx) / 80).clip(0, 1)            # rounded shading
    yellow = np.array([235, 190, 40], np.float64)
    img[M > 0] = (yellow * shade[..., None])[M > 0].astype(np.uint8)
    zip_ = (np.abs(yy - 150) < 2) & (M > 0)
    img[zip_] = 25                                                          # black zip
    logo = ((yy - 80) ** 2 + (xx - cx) ** 2 < 49) & (M > 0)
    img[logo] = (200, 30, 40)                                               # red logo
    return img, M, zip_, logo


def test_colour_roundtrip_and_de():
    rgb = np.random.default_rng(0).integers(0, 256, (500, 3))
    back = RC.oklab_to_rgb(RC.rgb_to_oklab(rgb))
    assert np.abs(back - rgb).max() < 0.01
    lab = RC.rgb_to_cielab(rgb)
    assert np.abs(RC.delta_e2000(lab, lab)).max() < 1e-9
    # Sharma et al. reference pair 1: dE00 = 2.0425
    assert abs(RC.delta_e2000(np.array([50, 2.6772, -79.7751]), np.array([50, 0.0, -82.7485])) - 2.0425) < 1e-3


def test_upper_half_green_exact():
    img, M, zip_, logo = bag()
    res = RC.recolour(img, M, RC.parse_instruction("make the upper half of the bag green"))
    m = res.metrics
    assert m["protected_identical"] and m["leak_dE00_max"] == 0.0
    assert m["hue_err_deg"] < 6, m
    assert m["shading_corr"] > 0.95, m
    # area split, not bbox midline: the coloured part holds half the object's area
    frac = (res.region * M).sum() / M.sum()
    assert abs(frac - 0.5) < 0.02
    ys = np.nonzero(M)[0]
    assert np.nonzero(res.region > 0.5)[0].max() > (ys.min() + ys.max()) / 2   # cut sits below the bbox midline
    assert (res.image[zip_] == img[zip_]).all()                                # zip is in the lower half
    assert res.weight[logo].mean() < 0.2                                        # red logo kept by affinity
    assert (res.image[M == 0] == img[M == 0]).all()                            # background untouched


def test_tilted_object_frame():
    img, M, *_ = bag(tilt=0.6)
    p = {"target_color": "blue", "frame": "object", "region": {"op": "half", "side": "upper", "frac": 0.5}}
    W = RC.region_weight(M, p)
    assert abs((W * M).sum() / M.sum() - 0.5) < 0.02


def test_algebra_ops():
    img, M, *_ = bag()
    up = {"op": "half", "side": "upper", "frac": 0.5}
    lo = {"op": "half", "side": "lower", "frac": 0.5}
    W = lambda r: RC.region_weight(M, {"target_color": "red", "region": r, "softness": 0.0})  # noqa: E731
    a = M.sum()
    assert abs(W({"op": "or", "args": [up, lo]}).sum() - a) / a < 0.01
    assert W({"op": "and", "args": [up, lo]}).sum() / a < 0.01
    assert abs(W({"op": "minus", "args": [{"op": "object"}, up]}).sum() / a - 0.5) < 0.02
    band = W({"op": "band", "side": "lower", "from": 0.2, "to": 0.4}).sum() / a
    assert abs(band - 0.2) < 0.02
    rim = W({"op": "border", "width": 0.06})
    assert 0 < rim.sum() / a < 0.5
    st = W({"op": "stripes", "axis": "horizontal", "count": 4}) > 0.5
    col = st[:, 100][M[:, 100] > 0].astype(int)
    assert (np.diff(col) == 1).sum() + col[0] == 4                             # 4 separate coloured stripes
    with pytest.raises(KeyError):
        W({"op": "part", "name": "pocket"})
    parts = {"pocket": np.zeros_like(M)}
    parts["pocket"][150:180, 80:120] = 1
    r = RC.region_weight(M, {"target_color": "red", "region": {"op": "part", "name": "pocket"}}, parts)
    assert r[160, 100] > 0.9 and r[50, 100] == 0


def test_grey_object_gets_colour():
    img = np.full((100, 100, 3), 128, np.uint8)
    img[20:80, 20:80] = 200
    M = np.zeros((100, 100), np.uint8)
    M[20:80, 20:80] = 1
    res = RC.recolour(img, M, {"target_color": "red", "region": {"op": "object"}})
    assert res.metrics["hue_err_deg"] < 6 and res.metrics["protected_identical"]


def test_parsers():
    assert RC.rule_parse("Paint the bottom third navy blue")["region"] == {"op": "half", "side": "bottom", "frac": 0.3333}
    assert RC.rule_parse("add 3 vertical stripes in red")["region"] == {"op": "stripes", "axis": "vertical", "count": 3}
    assert RC.rule_parse("make the trim gold")["region"]["op"] == "border"
    vlm = lambda p: 'Sure: {"object":"bag","target_color":"#22aa44","region":{"op":"half","side":"left","frac":0.5}}'  # noqa: E731
    assert RC.parse_instruction("x", ask=vlm)["parser"] == "vlm"
    bad = lambda p: '{"target_color":"green","region":{"op":"explode"}}'  # noqa: E731
    assert RC.parse_instruction("upper half green", ask=bad)["parser"] == "rules"


def test_part_parse_and_ground():
    assert RC.rule_parse("make the person's shirt teal")["region"] == {"op": "part", "name": "shirt"}
    ask = lambda p: '```json\n[{"bbox_2d": [100, 200, 500, 900], "label": "shirt"}]\n```'  # noqa: E731
    assert RC.ground_box(ask, (800, 400), "the shirt") == (80, 80, 400, 360)
    assert RC.ground_box(lambda p: "[]", (800, 400), "x") is None


def test_resolve_object_as_part():
    p = {"object": "shirt", "target_color": "teal", "region": {"op": "object"}}
    assert RC.resolve_object(p, "person")["region"] == {"op": "part", "name": "shirt"}
    p2 = {"object": "shirt", "target_color": "teal", "region": {"op": "half", "side": "upper"}}
    assert RC.resolve_object(p2, "person")["region"]["op"] == "and"
    assert RC.resolve_object({"object": "man", "region": {"op": "object"}}, "person")["region"] == {"op": "object"}
    assert RC.resolve_object({"object": "bag", "region": {"op": "object"}}, "tote bag")["region"] == {"op": "object"}
    vlm = lambda q: '{"object":"shirt","target_color":"teal","region":{"op":"object"}}'  # noqa: E731
    assert RC.parse_instruction("make the person's shirt teal", ask=vlm, scene_object="person")["region"]["op"] == "part"



def test_upright_object_with_handle_uses_image_axes():
    M = np.zeros((200, 200), np.uint8)
    M[40:170, 50:130] = 1                     # mug body
    M[70:130, 130:160] = 1                    # handle on the right
    W = RC.region_weight(M, {"target_color": "blue", "region": {"op": "stripes", "axis": "horizontal", "count": 3}})
    rows = W[:, 60:120] > 0.5
    assert all(len(set(r[M[i, 60:120] > 0])) <= 1 for i, r in enumerate(rows))   # each row fully in or out: level stripes


def cylinder(H=240, W=240, r=0.5, f=300.0, cam_h=-0.6):
    """Synthetic upright mug body (cylinder) seen slightly from above: returns image, mask, camera-space points."""
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    cz = 3.0
    X = (xx - W / 2) / f * cz
    inside = np.abs(X) < r * 0.98
    Z = np.where(inside, cz - np.sqrt(np.clip(r * r - X * X, 0, None)), np.nan)
    X = (xx - W / 2) / f * Z                       # re-project with the true depth
    Y = (yy - H / 2) / f * Z + cam_h               # camera above the mug's middle
    M = (inside & (Y > -0.95) & (Y < -0.2)).astype(np.uint8)
    img = np.full((H, W, 3), 230, np.uint8)
    img[M > 0] = (240, 240, 235)
    return img, M, np.stack([X, Y, Z], -1)


def test_3d_vertical_stripes_wrap_and_horizontal_follow_surface():
    img, M, P = cylinder()
    prog = {"target_color": "blue", "region": {"op": "stripes", "axis": "vertical", "count": 6}, "softness": 0.001}
    W2 = RC.region_weight(M, prog) > 0.5
    W3 = RC.region_weight(M, prog, points=P) > 0.5
    row = M.shape[0] // 2

    def runs(line, m):
        v = line[m].astype(int)
        edges = np.flatnonzero(np.diff(v)) + 1
        segs = np.split(v, edges)
        return [len(s) for s in segs if s[0] == 1]
    r2, r3 = runs(W2[row], M[row] > 0), runs(W3[row], M[row] > 0)
    assert max(r2) - min(r2) <= 3                          # flat: all stripes equally wide in the picture
    assert r3[len(r3) // 2] > 1.6 * min(r3[0], r3[-1])     # wrapped: stripes near the edge look much narrower
    h = {"target_color": "blue", "region": {"op": "stripes", "axis": "horizontal", "count": 3}, "softness": 0.001}
    W3h = RC.region_weight(M, h, points=P) > 0.5
    cols = np.flatnonzero(M.any(0))
    def first_on(c):
        ys = np.flatnonzero(W3h[:, c] & (M[:, c] > 0))
        return ys.min() if len(ys) else None
    mid, edge = first_on(cols[len(cols) // 2]), first_on(cols[3])
    assert mid is not None and edge is not None and mid != edge   # a constant-height stripe projects as a curve


def test_3d_stray_depth_points_do_not_empty_the_region():
    """v7 bug: a few absurd depth values on the object (handle/edge) blew up the size estimate -> 'part is empty'."""
    img, M, P = cylinder()
    P = P.copy()
    ys, xs = np.nonzero(M)
    P[ys[::97], xs[::97], 2] = 5e3          # stray far-away points on the object
    P[ys[5], xs[5]] = np.inf
    for reg in ({"op": "stripes", "axis": "horizontal", "count": 3, "width": 0.35},
                {"op": "stripes", "axis": "vertical", "count": 6, "width": 0.4},
                {"op": "half", "side": "upper", "frac": 0.5}):
        r = RC.recolour(img, M, {"target_color": "blue", "region": reg}, points=P)
        share = (r.region > 0.5).sum() / M.sum()
        assert 0.15 < share < 0.75, (reg, share)
