"""LumenAI editor endpoints (JSON in, JSON out) with a fake engine: routing, measured checks, self-repair, recolour
slider cache, expand maths, ideas/add/cutout/view3d plumbing."""
import base64
import io

import numpy as np
from PIL import Image

from server import web_api as A
from tests.test_web_api import FakeEngine


class Fake(FakeEngine):
    """Answers like Qwen3-VL would: yes/no questions, grounding boxes, add-object suggestions."""

    def __init__(self, verdict="yes"):
        super().__init__()
        self.verdict = verdict
        self.steps_seen = []

    def ask(self, images, question, n=128):
        self.calls.append(("ask", question[:60]))
        if "Answer yes or no" in question:
            return self.verdict
        if "photo stylist" in question:
            if "vase" in question:                      # asked to avoid earlier ideas
                return '[{"object": "brass key", "reason": "adds a story", "where": "on the floor, right"}]'
            return ('[{"object": "vase", "reason": "the table is empty", "where": "on the table, left of the box"},'
                    ' {"object": "Vase", "reason": "duplicate", "where": "x"}]')
        if "red box" in question:
            return '{"height_vs_red_box": 0.5, "width_to_height": 0.6}'
        if "bbox" in question or "Locate" in question or "locate" in question:
            return '[{"bbox_2d": [375, 333, 625, 667], "label": "x"}]'
        return "a vase"

    def edit(self, images, instruction, seed=None, resolution=1024):
        self.steps_seen.append((self.steps, self.seed if seed is None else seed))
        return super().edit(images, instruction, seed, resolution)

    def make3d(self, rgba):
        self.calls.append(("make3d", rgba.size))
        return b"glTF-fake", {"faces": 1234, "texture_s": 12.5}


def scene(w=320, h=240):
    """Flat cream background with an olive square in the middle (the stub segmenter flood-fills it)."""
    a = np.full((h, w, 3), (240, 230, 216), np.uint8)
    a[80:160, 120:200] = (85, 89, 52)
    return Image.fromarray(a)


def url(img, fmt="PNG"):
    b = io.BytesIO()
    img.save(b, fmt)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def square_mask(w=320, h=240):
    m = Image.new("L", (w, h), 0)
    m.paste(255, (120, 80, 200, 160))
    return url(m)


def decode(u):
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(u.split(",", 1)[1]))))


def client(E):
    from fastapi.testclient import TestClient

    return TestClient(A.build_app(E))


def test_segment_click_selects_the_object():
    r = client(Fake()).post("/api/v1/segment", json={"image": url(scene()), "points": [[160, 120, 1]]}).json()
    m = decode(r["mask"])
    assert m.shape == (240, 320) and m[120, 160] > 0 and m[10, 10] == 0
    assert r["box"] == [120, 80, 200, 160]


def test_quick_quality_and_seed_reach_the_engine_then_reset():
    E = Fake("yes")
    client(E).post("/api/v1/edit", json={"image": url(scene()), "prompt": "make it night", "quality": "quick", "seed": 9})
    assert E.steps_seen[0] == (A.QUICK_STEPS, 9)
    assert (E.steps, E.seed) == (A.STEPS, A.SEED)


def test_recolour_then_slider_shade():
    c = client(Fake())
    img = scene()
    r = c.post("/api/v1/recolour", json={"image": url(img), "mask": square_mask(), "colour": "#2f4f7b"}).json()
    a = decode(r["image"])
    assert r["recolour_id"] and a[120, 160, 2] > a[120, 160, 0]          # bluish now
    assert np.abs(a[:40, :40].astype(int) - np.asarray(img)[:40, :40]).max() <= 3   # background untouched
    assert all(ch["ok"] for ch in r["checks"])
    s = c.post("/api/v1/recolour-shade", json={"recolour_id": r["recolour_id"], "colour": "#a8741f"}).json()
    b = decode(s["image"])
    assert b[120, 160, 0] > b[120, 160, 2]                               # amber now
    bad = c.post("/api/v1/recolour-shade", json={"recolour_id": "nope", "colour": "#000"})
    assert bad.status_code == 400 and "expired" in bad.json()["detail"]


def test_prompt_recolour_returns_slider_id():
    r = client(Fake()).post("/api/v1/edit", json={"image": url(scene()), "prompt": "make the square blue",
                                                  "mask": square_mask()}).json()
    assert r["route"] == "recolour" and r["recolour_id"]


def test_expand_plan_ratios_and_cap():
    s, tw, th, ox, oy, pw, ph = A.expand_plan(800, 600, "16:9")
    assert abs(tw / th - 16 / 9) < 0.03 and th >= 592 and ox > 0 and oy == 0 and tw % 16 == 0
    s, tw, th, ox, oy, pw, ph = A.expand_plan(3000, 2000, "16:9")
    assert max(tw, th) <= 1536 and s < 1 and pw <= tw and ph <= th
    s, tw, th, ox, oy, pw, ph = A.expand_plan(400, 400, None, 0.5)
    assert (tw, th) == (592, 592) and ox == oy == 96


def test_ideas_and_add(monkeypatch):
    from editing import add_object as AO

    c = client(Fake())
    r = c.post("/api/v1/ideas", json={"image": url(scene())}).json()
    assert [i["object"] for i in r["ideas"]] == ["vase"] and r["ideas"][0]["where"]          # duplicates dropped
    r2 = c.post("/api/v1/ideas", json={"image": url(scene()), "avoid": ["vase"]}).json()
    assert r2["ideas"][0]["object"] == "brass key"                                       # "New ideas" are new
    p = c.post("/api/v1/place", json={"image": url(scene()), "object": "vase", "where": r["ideas"][0]["where"]}).json()
    x0, y0, x1, y1 = p["box"]
    assert 0 <= x0 < x1 <= 320 and 0 <= y0 < y1 <= 240 and abs((y1 - y0) - 0.5 * 80) <= 3   # half the subject's height
    r = {"ideas": [{"box": p["box"]}]}

    def fake_add(pipe, Q, image, obj, box, method="ghost", seed=7, steps=40, **k):
        a = np.asarray(image).copy()
        a[box[1]:box[3], box[0]:box[2]] = (200, 30, 30)
        return AO.AddResult(a, a, method)

    monkeypatch.setattr(AO, "add", fake_add)
    monkeypatch.setattr(AO, "check", lambda ask, o, r, obj, box: {"found": True, "inside": True, "grey_left": 0.0})
    r = c.post("/api/v1/add", json={"image": url(scene()), "object": "vase", "box": r["ideas"][0]["box"]}).json()
    assert all(ch["ok"] for ch in r["checks"]) and r["route"] == "add"


def test_cutout_is_transparent_png():
    r = client(Fake()).post("/api/v1/cutout", json={"image": url(scene())}).json()
    assert r["image"].startswith("data:image/png") and decode(r["image"]).shape[2] == 4


def test_view3d_builds_outside_the_lock():
    E = Fake()
    r = client(E).post("/api/v1/view3d", json={"image": url(scene()), "mask": square_mask(), "complete": False}).json()
    assert r["glb"].startswith("data:model/gltf-binary;base64,") and base64.b64decode(r["glb"].split(",")[1]) == b"glTF-fake"
    assert ("make3d", (512, 512)) in E.calls and r["info"]["faces"] == 1234


def test_bad_requests_are_400():
    c = client(Fake())
    assert c.post("/api/v1/edit", json={"prompt": "x"}).status_code == 400
    assert c.post("/api/v1/edit", json={"image": url(scene()), "prompt": "  "}).status_code == 400


def test_place_box_moves_off_the_subject():
    b = A.place_box((400, 300), (150, 150, 250, 250), (160, 50, 240, 250), 0.5, 0.5)
    x0, y0, x1, y1 = b
    assert x1 <= 160 or x0 >= 240                                    # not covering the reference object
    assert y1 - y0 == 100 and x1 - x0 == 50


def test_view3d_not_deployed_message(monkeypatch):
    E = Fake()

    def nope(rgba):
        raise ValueError("The 3D service is not deployed yet. In the project folder run: modal deploy modal_object3d.py")
    E.make3d = nope
    r = client(E).post("/api/v1/view3d", json={"image": url(scene()), "mask": square_mask(), "complete": False})
    assert r.status_code == 400 and "modal deploy modal_object3d.py" in r.json()["detail"]


def test_general_edit_is_one_pass_without_checks():
    E = Fake("no")                       # even a "no" from the VLM must not trigger extra work any more
    r = client(E).post("/api/v1/edit", json={"image": url(scene()), "prompt": "make it a snowy day"}).json()
    assert r["route"] == "general" and r["checks"] == [] and len(E.steps_seen) == 1
    assert not [c for c in E.calls if c[0] == "ask" and "carried out" in c[1]]


def test_masked_edit_stays_local():
    r = client(Fake("yes")).post("/api/v1/edit", json={"image": url(scene()), "prompt": "turn this into a book",
                                                       "mask": square_mask()}).json()
    out = decode(r["image"]).astype(int)                              # JPEG 95: within 3 levels
    assert np.abs(out[:30, :30] - np.asarray(scene())[:30, :30]).max() <= 3


def test_remove_route(monkeypatch):
    def fake_remove(E, image, mask):
        a = np.asarray(image).copy()
        a[mask > 0] = (240, 230, 216)
        return a, {"shadow_px": 40, "attempts": [{"name": "draft-refine", "ok": True}]}

    monkeypatch.setattr(A, "remove_object_info", fake_remove)
    r = client(Fake()).post("/api/v1/edit", json={"image": url(scene()), "prompt": "remove the square"}).json()
    assert r["route"] == "remove" and r["checks"] == [] and decode(r["image"])[120, 160, 0] > 200


def test_expand_endpoint():
    r = client(Fake()).post("/api/v1/expand", json={"image": url(scene()), "ratio": "16:9"}).json()
    out = decode(r["image"])
    assert out.shape[:2] == (r["size"][1], r["size"][0]) and abs(out.shape[1] / out.shape[0] - 16 / 9) < 0.05


def test_edit_never_reuses_the_noise_that_made_the_image():
    img = scene()
    s_edit = A.mix_seed(7, img)
    assert s_edit != 7 and s_edit != A.mix_seed(7, "a cozy home interior")
    assert A.mix_seed(7, img) == s_edit                       # reproducible
    assert A.mix_seed(8, img) != s_edit                       # "Another version" still differs
    edited = np.asarray(img).copy(); edited[100:140, 20:60] = (200, 40, 40)
    assert A.mix_seed(7, Image.fromarray(edited)) != s_edit    # chained edits get new noise
