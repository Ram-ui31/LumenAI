"""Web API tests with a fake engine (no GPU): request parsing, routing, sizes, and that pixels outside a mask survive."""
import io

import numpy as np
from PIL import Image, ImageDraw

from server import web_api as A


class FakeEngine(A.Engine):
    def __init__(self):
        super().__init__()
        self.calls = []

    def load(self):
        class Q:
            @staticmethod
            def crop_window(region, pad_frac=0.8, min_side=256):
                from editing.qwen import crop_window
                return crop_window(region, pad_frac, min_side)
        return None, Q

    def ask(self, images, question, n=128):
        self.calls.append(("ask", question[:40]))
        return "a sunny beach"

    def edit(self, images, instruction, seed=7, resolution=1024):
        self.calls.append(("edit", instruction))
        a = np.asarray(images[0]).copy()
        a[...] = (200, 30, 30)                       # a visible "generated" result
        return Image.fromarray(a)

    def outpaint_lora(self, image, tw, th, ox, oy, scene):
        self.calls.append(("outpaint_lora", scene))
        c = Image.new("RGB", (tw, th), (128, 128, 128))
        c.paste(image, (ox, oy))
        return c

    def refine(self, structure, prompt, seed, strength, resolution=1024, keep=None, condition=None):
        self.calls.append(("refine", prompt))
        return structure


def photo(w=320, h=240):
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))


def stroke_mask(w=320, h=240, box=(100, 80, 180, 160)):
    m = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(m).ellipse(box, fill=(239, 68, 68, 166))
    b = io.BytesIO()
    m.save(b, "PNG")
    return b.getvalue()


def pngb(img):
    b = io.BytesIO()
    img.save(b, "PNG")
    return b.getvalue()


def test_route():
    assert A.route("I captured this image while running, make it correct.", False, False) == "deblur"
    assert A.route("remove the red car", False, False) == "remove"
    assert A.route("make the shirt navy blue", False, False) == "recolour"
    assert A.route("Apply teal-orange cinematic colour grade with tonal mapping", False, False) == "general"
    assert A.route("Transfer exact body pose from the reference image at 80% strength", True, False) == "pose"
    assert A.route("Artistic transfer", False, False) == "general"
    assert A.route("add a cat on the sofa", False, False) == "general"


def test_read_mask_alpha_strokes():
    m = A.read_mask(stroke_mask(), (320, 240))
    assert m.shape == (240, 320) and m[120, 140] == 1 and m[10, 10] == 0


def test_outpaint_size_and_original_kept():
    E = FakeEngine()
    img = photo()
    out = A.outpaint(E, img, 480, 240, 80, 0, "")
    assert out.shape == (240, 480, 3)
    # centre of the original is untouched
    assert np.abs(out[60:180, 140:340].astype(int) - np.asarray(img)[60:180, 60:260].astype(int)).max() <= 2
    assert any(c[0] == "outpaint_lora" for c in E.calls)   # v8: LoRA method is the default


def test_outpaint_capped_canvas_shrinks_photo():
    E = FakeEngine()
    out = A.outpaint(E, photo(1600, 800), 1536, 800, 200, 0, "beach")
    assert out.shape == (800, 1536, 3)


def test_inpaint_variations_and_outside_untouched():
    E = FakeEngine()
    img = photo()
    m = A.read_mask(stroke_mask(), img.size)
    outs = A.inpaint(E, img, m, "a red balloon", "Photorealistic", 3)
    assert len(outs) == 3 and outs[0].shape == (240, 320, 3)
    assert np.array_equal(outs[0][:30, :30], np.asarray(img)[:30, :30])
    assert outs[0][120, 140, 0] > 150                      # generated colour inside the mask
    assert "grey area" in E.calls[0][1] and "Style" not in E.calls[0][1]


def test_masked_general_edit_stays_local():
    E = FakeEngine()
    img = photo()
    m = A.read_mask(stroke_mask(), img.size)
    out, info = A.edit_prompt(E, img, "add sunglasses", m, None, 0.8)
    assert info["route"] == "general" and info.get("masked")
    assert np.array_equal(out[:30, :30], np.asarray(img)[:30, :30])


def test_big_upload_returns_full_size():
    E = FakeEngine()
    img = photo(3000, 1500)
    out, _ = A.edit_prompt(E, img, "make it snowy", None, None, 1.0)
    assert out.shape == (1500, 3000, 3)


def test_http_endpoints():
    from fastapi.testclient import TestClient

    c = TestClient(A.build_app(FakeEngine()))
    assert c.get("/api/v1/health").json()["status"] == "ok"
    img = pngb(photo())
    r = c.post("/api/v1/upscale", files={"image": ("i.png", img)}, data={"scale": "2"})
    assert r.status_code == 200 and Image.open(io.BytesIO(r.content)).size == (640, 480)
    r = c.post("/api/v1/inpaint", files={"image": ("i.png", img), "mask": ("m.png", stroke_mask())},
               data={"prompt": "a lamp", "style": "Photorealistic", "variations": "2"})
    v = r.json()["variations"]
    assert len(v) == 2 and v[0].startswith("data:image/png;base64,")
    r = c.post("/api/v1/outpaint", files={"image": ("i.png", img)},
               data={"target_width": "400", "target_height": "300", "offset_x": "40", "offset_y": "30", "prompt": ""})
    assert r.status_code == 200 and Image.open(io.BytesIO(r.content)).size == (400, 300)
    r = c.post("/api/v1/edit-prompt", files={"image": ("i.png", img)}, data={"prompt": "make it night", "strength": "0.8"})
    assert r.status_code == 200 and r.headers["X-AdobePS-Route"] == "general"
    r = c.post("/api/v1/remove-object", files={"image": ("i.png", img), "mask": ("m.png", pngb(Image.new("RGBA", (320, 240))))})
    assert r.status_code == 400 and "Paint" in r.json()["detail"]


def test_outpaint_falls_back_when_lora_fails():
    E = FakeEngine()
    E.outpaint_lora = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no LoRA"))
    out = A.outpaint(E, photo(), 480, 240, 80, 0, "")
    assert out.shape == (240, 480, 3) and any(c[0] == "refine" for c in E.calls)


def test_style_transfer_describes_style_then_single_image_edit():
    E = FakeEngine()
    out = A.style_transfer(E, photo(), photo(100, 100), 1.0, "")
    edits = [c for c in E.calls if c[0] == "edit"]
    assert out.shape == (240, 320, 3) and len(edits) == 1 and "artistic style" in edits[0][1]
    assert any(c[0] == "ask" for c in E.calls)
