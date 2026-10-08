"""v10: pose/style routing, model tiers, bokeh, dolly layers, move, relight, style, product, prompt cache."""
import base64
import io
from types import SimpleNamespace

import numpy as np
from PIL import Image

from server import extras as X
from server import web_api as A
from tests.test_lumen_api import Fake, client, decode, scene, square_mask, url


class V10(Fake):
    def __init__(self, verdict="yes"):
        super().__init__(verdict)
        self.flash_calls, self.edits = [], []

    def depth(self, image):
        w, h = image.size
        d = np.tile(np.linspace(0, 0.3, h, dtype=np.float32)[:, None], (1, w))     # background: far
        a = np.asarray(image.convert("RGB")).astype(int)
        d[(np.abs(a - (85, 89, 52)).sum(2) < 30)] = 0.9                               # the olive square: near
        return d

    def flash_generate(self, prompt, w, h):
        self.flash_calls.append(("gen", prompt))
        self.remote_peak = self.REMOTE_PEAK_GB["flash"]
        return Image.new("RGB", (w, h), (10, 200, 10))

    def flash_edit(self, images, instruction):
        self.flash_calls.append(("edit", instruction))
        self.remote_peak = self.REMOTE_PEAK_GB["flash"]
        return Image.new("RGB", (1024, 768), (10, 200, 10))

    def generate(self, prompt, w, h):
        self.calls.append(("generate", prompt))
        return Image.new("RGB", (w, h), (200, 10, 10))

    def edit(self, images, instruction, seed=None, resolution=1024):
        self.edits.append((len(images), instruction))
        return super().edit(images, instruction, seed, resolution)

    def ask(self, images, question, n=128):
        if "rewrite" in question.lower() or "You write prompts" in question:
            return "A precise rewritten instruction that names the sofa and the woman sitting on it, keep the rest."
        if "What product" in question:
            return "glass bottle"
        return super().ask(images, question, n)

    def load(self):
        pipe, Q = super().load()

        class Q2(Q):
            @staticmethod
            def edit(pipe, image, instruction, seed=7, steps=40, resolution=1024):
                a = np.asarray(image.convert("RGB")).astype(np.float32)
                a[:, : a.shape[1] // 2] *= 1.3                                     # "light from the left"
                return SimpleNamespace(image=Image.fromarray(a.clip(0, 255).astype(np.uint8)), seconds=1.0)
        return pipe, Q2


def test_routes_with_reference():
    assert A.route("make her pose like this", True, False) == "pose"
    assert A.route("Transfer exact body pose from the reference image", True, False) == "pose"
    assert A.route("sitting like the person in the reference", True, False) == "pose"
    assert A.route("paint it in the style of the reference", True, False) == "style"
    assert A.route("put the hat from the reference on him", True, False) == "general"
    assert A.route("make her pose like this", False, False) == "general"          # no reference: plain edit


def test_pose_never_shows_qwen_the_reference(monkeypatch):
    from editing import pose as P

    seen = {}

    def loop(pipe, Q, person, tkp, rounds=3, seed=7, same_person=None, steps=40, gist=""):
        seen.update(size=person.size, gist=gist, seed=seed, kp=tkp)
        return person, {"pck": 0.8, "best_round": 0, "history": [{"round": 0, "same_person": True}]}

    kp = np.zeros((18, 3), np.float32)
    kp[:6] = [(10, 10, 1), (20, 20, 1), (30, 30, 1), (40, 40, 1), (50, 50, 1), (60, 60, 1)]
    monkeypatch.setattr(P, "keypoints", lambda im: kp)
    monkeypatch.setattr(P, "repose_loop", loop)
    E = V10()
    ref = Image.new("RGB", (640, 480), (1, 2, 3))
    r = client(E).post("/api/v1/edit", json={"image": url(scene()), "prompt": "make him pose like this",
                                             "reference": url(ref)}).json()
    assert r["route"] == "pose" and r["info"]["pck"] == 0.8
    assert not [e for e in E.edits if e[0] == 2]                       # no two-image edit
    assert seen["size"] == (320, 240) and seen["kp"][1, 0] == 10        # joints rescaled 640 -> 320
    assert seen["seed"] != A.SEED                                       # seed mixed with the photo


def test_pose_without_person_says_so(monkeypatch):
    from editing import pose as P

    monkeypatch.setattr(P, "keypoints", lambda im: None)
    r = client(V10("no")).post("/api/v1/pose", json={"image": url(scene()), "reference": url(scene())})
    assert r.status_code == 400 and "person" in r.json()["detail"]


def test_tiers_flash_and_pro():
    E = V10()
    c = client(E)
    r = c.post("/api/v1/generate", json={"prompt": "a cozy home interior", "model": "flash"}).json()
    assert r["model"] == "flash" and E.flash_calls[0] == ("gen", "a cozy home interior")
    assert decode(r["image"])[5, 5, 1] > 150
    r = c.post("/api/v1/generate", json={"prompt": "a cozy home interior", "model": "pro"}).json()
    assert r["model"] == "pro" and r["prompt_used"].startswith("A precise")
    r = c.post("/api/v1/edit", json={"image": url(scene()), "prompt": "add a woman on the sofa", "model": "flash"}).json()
    assert r["info"]["model"] == "flash" and decode(r["image"]).shape[:2] == (240, 320)
    r = c.post("/api/v1/edit", json={"image": url(scene()), "prompt": "add a woman on the sofa", "model": "pro"}).json()
    assert E.edits[-1][1].startswith("A precise") and E.steps_seen[-1][0] == A.STEPS
    assert (E.tier, E.steps) == ("lumen", A.STEPS)                     # reset after the request
    assert A.tier_of({"quality": "quick"}) == "lumen" and A.tier_of({"quality": "full"}) == "pro"


def test_bokeh_blurs_background_keeps_subject():
    img = scene()
    a = np.asarray(img).copy()
    a[::8, :] = 255                                                    # stripes so blur is measurable
    a[80:160, 120:200] = (85, 89, 52)
    img = Image.fromarray(a)
    c = client(V10())
    r = c.post("/api/v1/bokeh", json={"image": url(img), "mask": square_mask(), "strength": 1}).json()
    o = decode(r["image"]).astype(int)
    assert np.abs(o[100:140, 140:180] - a[100:140, 140:180]).max() <= 3          # subject sharp
    assert o[:60, :100].std() < a[:60, :100].std() * 0.8                        # background blurred
    r2 = c.post("/api/v1/bokeh-adjust", json={"bokeh_id": r["bokeh_id"], "strength": 0.0}).json()
    assert np.abs(decode(r2["image"]).astype(int) - a).max() <= 3                 # strength 0 = original
    assert c.post("/api/v1/bokeh-adjust", json={"bokeh_id": "x"}).status_code == 400


def test_layers_for_dolly(monkeypatch):
    monkeypatch.setattr(A, "remove_object_info", lambda E, im, m: (np.full((*m.shape, 3), 240, np.uint8), {}))
    r = client(V10()).post("/api/v1/layers", json={"image": url(scene()), "mask": square_mask()}).json()
    sub = np.asarray(Image.open(io.BytesIO(base64.b64decode(r["subject"].split(",")[1]))))
    assert sub.shape[2] == 4 and sub[120, 160, 3] > 200 and sub[10, 10, 3] == 0
    assert r["subject_depth"] > 0.8 and r["box"] == [120, 80, 200, 160]
    assert r["depth"].startswith("data:image/png")


def test_move_object(monkeypatch):
    def fake_remove(E, image, mask):
        a = np.asarray(image).copy()
        a[mask > 0] = (240, 230, 216)
        return a, {}
    monkeypatch.setattr(A, "remove_object_info", fake_remove)
    r = client(V10()).post("/api/v1/move", json={"image": url(scene()), "mask": square_mask(), "dx": 80, "dy": 0}).json()
    o = decode(r["image"]).astype(int)
    assert abs(o[120, 140] - (240, 230, 216)).max() < 12              # old place filled
    assert abs(o[120, 260] - (85, 89, 52)).max() < 12                 # object now 80 px to the right
    assert r["box"][0] == 200


def test_relight_style_product():
    E = V10()
    c = client(E)
    r = c.post("/api/v1/relight", json={"image": url(scene()), "light": "left"}).json()
    o = decode(r["image"]).astype(int)
    assert o[:, :100].mean() > o[:, -100:].mean()
    r = c.post("/api/v1/style", json={"image": url(scene()), "preset": "watercolour"}).json()
    assert r["route"] == "style" and "watercolour" in E.edits[-1][1]
    assert c.post("/api/v1/style", json={"image": url(scene())}).status_code == 400
    r = c.post("/api/v1/product", json={"image": url(scene()), "scene": "marble"}).json()
    assert r["product"] == "glass bottle" and "marble" in E.edits[-1][1]


def test_prompt_cache_hits_only_exact_repeats():
    from editing.qwen import PromptCache

    pc = PromptCache(2)
    im = scene()
    k1 = pc.key((), {"prompt": "add a woman", "image": [im], "device": "cuda:0"})
    pc.put(k1, "E1")
    assert pc.get(pc.key((), {"prompt": "add a woman", "image": [scene()], "device": "cuda"})) == "E1"
    other = np.asarray(im).copy()
    other[0, 0] = 0
    assert pc.get(pc.key((), {"prompt": "add a woman", "image": [Image.fromarray(other)]})) is None
    assert pc.get(pc.key((), {"prompt": "add a man", "image": [im]})) is None


def test_peak_and_time_reported():
    E = V10()
    r = client(E).post("/api/v1/generate", json={"prompt": "a cozy home interior", "model": "flash"}).json()
    assert r["peak_gb"] == A.Engine.REMOTE_PEAK_GB["flash"] and r["seconds"] >= 0
    r = client(E).post("/api/v1/edit", json={"image": url(scene()), "prompt": "make it night"}).json()
    assert "peak_gb" in r and r["peak_gb"] is None                       # no GPU in tests: nothing measured
    assert A.TIERS["flash"]["name"].startswith("Lumen 1.1") and "Lux" in A.TIERS["pro"]["name"]


def test_multi_reference_edit_sends_all_images():
    E = V10()
    refs = [url(Image.new("RGB", (64, 64), (i * 20, 0, 0))) for i in range(4)]
    r = client(E).post("/api/v1/edit", json={"image": url(scene()), "prompt": "put these four items on the table",
                                             "references": refs}).json()
    assert r["route"] == "general" and E.edits[-1][0] == 5 and "<image5>" in E.edits[-1][1]


def test_glow_and_adjust():
    c = client(V10())
    r = c.post("/api/v1/glow", json={"image": url(scene()), "point": [40, 40], "colour": "#3366ff", "intensity": 90}).json()
    a = decode(r["image"]).astype(int)
    assert a[30:50, 30:50, 2].mean() > np.asarray(scene())[30:50, 30:50, 2].mean()       # bluer near the light
    r2 = c.post("/api/v1/glow-adjust", json={"glow_id": r["glow_id"], "point": [280, 200], "colour": "#ff3300"}).json()
    assert decode(r2["image"]).shape == a.shape


def test_product_angles_and_replace(monkeypatch):
    E = V10()
    r = client(E).post("/api/v1/product-angles", json={"image": url(scene())}).json()
    assert [x["angle"] for x in r["angles"]] == ["45° left", "Side", "Back", "Top"] and len(E.flash_calls) == 4
    monkeypatch.setattr(A, "remove_object_info", lambda E, im, m: (np.full((*m.shape, 3), 240, np.uint8), {}))
    view = Image.new("RGBA", (100, 50), (0, 0, 0, 0))
    view.paste((10, 200, 10, 255), (10, 10, 90, 40))
    r = client(E).post("/api/v1/replace-object", json={"image": url(scene()), "mask": square_mask(), "view": url(view)}).json()
    o = decode(r["image"]).astype(int)
    assert o[150, 160, 1] > 150 and o[150, 160, 0] < 60 and o[100, 160, 0] > 200   # view on the old ground line, hole filled
