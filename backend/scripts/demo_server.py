"""Demo server: the real LumenAI API routes with a stand-in engine - no GPU, no model downloads.

    cd backend
    python scripts/demo_server.py               # API at http://127.0.0.1:8765/api/v1

Every tool answers within a second or two with a placeholder result (a colour tint instead of a real edit), so the
whole editor can be tried and tested without a GPU. Open the editor with ?api=http://127.0.0.1:8765/api/v1.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["ADOBEPS_BACKEND"] = "stub"

import numpy as np  # noqa: E402
import uvicorn  # noqa: E402
from PIL import Image  # noqa: E402

from server import web_api as A  # noqa: E402
from tests.test_v10 import V10 as Fake  # noqa: E402


class Slow(Fake):
    """Stand-in engine with small delays, so the editor's progress animation can be seen."""

    def ask(self, images, question, n=128):
        if "ALLOW or BLOCK" in question:
            return "BLOCK" if "naked" in question else "ALLOW"
        return super().ask(images, question, n)

    def generate(self, prompt, w, h):
        time.sleep(1.5)
        y, x = np.mgrid[0:h, 0:w]
        return Image.fromarray(np.dstack([x * 255 // w, y * 255 // h, np.full((h, w), 140)]).astype(np.uint8))

    def upscale_ai(self, img, scale):
        time.sleep(1)
        return img.resize((img.width * scale, img.height * scale), Image.LANCZOS)

    def edit(self, images, instruction, seed=None, resolution=1024):
        time.sleep(1.5)
        a = np.asarray(images[0]).astype(np.float32)
        tint = np.array([(seed or 7) * 37 % 255, 120, 200 - (seed or 7) * 11 % 120], np.float32)
        return Image.fromarray((0.6 * a + 0.4 * tint).clip(0, 255).astype(np.uint8))

    def flash_generate(self, prompt, w, h):
        time.sleep(0.5)
        y, x = np.mgrid[0:h, 0:w]
        return Image.fromarray(np.dstack([np.full((h, w), 60), x * 255 // w, y * 255 // h]).astype(np.uint8))

    def flash_edit(self, images, instruction):
        time.sleep(0.5)
        a = np.asarray(images[0]).astype(np.float32)
        return Image.fromarray((0.5 * a + 0.5 * np.array([40, 200, 90])).clip(0, 255).astype(np.uint8))

    def depth(self, image):
        w, h = image.size
        a = np.asarray(image.convert("L")).astype(np.float32) / 255
        y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        cx = np.abs(np.linspace(-1, 1, w, dtype=np.float32))[None, :]
        return np.clip(0.25 + 0.55 * y * (1 - 0.6 * cx) + 0.1 * a, 0, 1)

    def make3d(self, rgba):
        import trimesh

        time.sleep(1)
        m = trimesh.creation.icosphere(subdivisions=3)
        m.visual.vertex_colors = [85, 89, 52, 255]
        return m.export(file_type="glb"), {"faces": len(m.faces)}


def fake_remove(E, image, mask):
    time.sleep(1.5)
    a = np.asarray(image).copy()
    a[mask > 0] = np.median(a[mask == 0], axis=0)
    return a, {"shadow_px": 120, "attempts": [{"name": "draft-refine", "ok": False}, {"name": "firm", "ok": True}]}


def fake_add(pipe, Q, image, obj, box, method="ghost", seed=7, steps=40, **k):
    from editing import add_object as AO

    time.sleep(1.5)
    a = np.asarray(image).copy()
    a[box[1]:box[3], box[0]:box[2]] = (180, 60, 40)
    return AO.AddResult(a, a, method)


if __name__ == "__main__":
    from editing import add_object as AO

    A.remove_object_info = fake_remove
    AO.add = fake_add
    AO.check = lambda ask, o, r, obj, box: {"found": True, "inside": True, "grey_left": 0.0}
    from server import trends as T
    from tests.test_trends import GOOD

    from editing import pose as P

    def _kp(im):
        kp = np.zeros((18, 3), np.float32)
        kp[:8] = [(x * im.width / 10, y * im.height / 10, 1) for x, y in [(5, 1), (5, 2), (4, 2), (3, 3), (3, 4), (6, 2), (7, 3), (7, 4)]]
        return kp

    def _loop(pipe, Q, person, tkp, rounds=3, seed=7, same_person=None, steps=40, gist=""):
        time.sleep(1.5)
        a = np.asarray(person).astype(np.float32)
        return Image.fromarray((0.7 * a + 0.3 * np.array([90, 60, 160])).clip(0, 255).astype(np.uint8)), \
            {"pck": 0.83, "best_round": 1, "history": [{"round": 1, "same_person": True}]}

    P.keypoints, P.repose_loop = _kp, _loop
    os.environ["GEMINI_API_KEY"] = "fake"
    T.call_gemini = lambda prompt, timeout=60, post=None: (time.sleep(1), GOOD)[1]
    uvicorn.run(A.build_app(Slow()), host="127.0.0.1", port=8765, log_level="warning")
