"""Suggestion parsing, edit area and the success check of editing.add_object (no models)."""
import numpy as np

from editing import add_object as A


def test_parse_suggestions_filters_bad_boxes():
    t = ('```json\n[{"object": "lamp", "reason": "dark corner", "bbox_2d": [100, 200, 300, 600]},'
         '{"object": "huge", "bbox_2d": [0, 0, 1000, 1000]}, {"object": "tiny", "bbox_2d": [10, 10, 12, 12]}]\n```')
    s = A.parse_suggestions(t, (800, 400))
    assert [x.object for x in s] == ["lamp"] and s[0].box == (80, 80, 240, 240)


def test_edit_area_extends_below():
    m = A.edit_area((100, 100, 200, 200), (400, 400))
    ys, xs = np.nonzero(m)
    assert ys.max() >= 230 and ys.min() >= 90 and xs.min() >= 85


def test_check_success_and_shadow():
    H, W = 300, 400
    O = np.full((H, W, 3), 200, np.uint8)
    R = O.copy()
    box = (150, 80, 250, 200)
    R[80:200, 150:250] = (40, 120, 50)               # the added object
    R[200:230, 150:250] = 150                        # its shadow under it (darker floor)
    area = A.edit_area(box, (W, H))
    ys, xs = np.nonzero(area)
    x0, y0 = max(0, xs.min() - 40), max(0, ys.min() - 40)
    x1, y1 = min(W, xs.max() + 40), min(H, ys.max() + 40)
    cw, ch = x1 - x0, y1 - y0
    rel = [(150 - x0) / cw * 1000, (80 - y0) / ch * 1000, (250 - x0) / cw * 1000, (200 - y0) / ch * 1000]
    ask = lambda ims, q, n: '[{"bbox_2d": [%d, %d, %d, %d]}]' % tuple(rel)  # noqa: E731
    c = A.check(ask, O, R, "plant", box)
    assert c["found"] and c["inside"] and c["iou"] > 0.9 and c["success"]
    assert c["shadow_dL"] > 10
    c2 = A.check(lambda ims, q, n: "[]", O, O, "plant", box)
    assert not c2["success"]
    G = O.copy(); G[80:200, 150:250] = 150             # grey placeholder left behind
    c3 = A.check(ask, O, G, "plant", box)
    assert c3["grey_left"] > 0.5 and not c3["success"]
