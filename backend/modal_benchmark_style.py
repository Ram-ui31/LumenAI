"""Style-transfer benchmark: our FLUX.2 klein artwork styles against Magenta (TensorFlow Hub, the team's
tensorflow_style.py), on 12 photos x 4 artworks (48 pairs per method).

    modal run modal_benchmark_style.py

Needs the Lumina app deployed (modal deploy modal_flash.py). Results are written to ./outputs/style_bench/
(summary.md, results.json, grid.jpg) and to the adobe-ps-data volume (outputs/eval/style_bench).

Metrics (all against the inputs, no ground truth exists for style transfer):
  content kept   DINOv2-base CLS cosine(output, photo)                    higher = the photo's content survives
                 CLIP ViT-L/14 image cosine(output, photo)                 higher = same scene
  style match    CLIP ViT-L/14 image cosine(output, artwork)               higher = closer to the artwork
                 VGG-19 Gram style loss (Gatys et al.), log10              lower  = textures/colours closer
  time           warm seconds per image on the same GPU type (L40S), measured inside Modal
"""
import io
import json
import os
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "frontend" / "assets"
GPU = os.environ.get("MODAL_GPU", "L40S")
CONTENT = ["hero_perfume.jpg", "hero_room.jpg", "hero_woman.jpg", "c1_mercedes.jpg", "c2_white_brown_model.jpg",
           "c3_road.jpg", "c4_wimbeldon.jpg", "w_bag_before.jpg", "w_gallery_before.jpg", "w_pingpong_before.jpg",
           "w_truck_before.jpg", "sq_side_car.jpg"]
STYLES = [("108700", "Baroque Drama"), ("324310", "Impressionist"), ("644443", "Post-Impression"),
          ("86092", "Charcoal Noir")]
# the exact instruction the editor sends to klein (server/extras.py ART_STYLE_Q)
ART_STYLE_Q = ("Repaint image 1 in the artistic style of image 2: its colour palette, brushwork, texture and light. Keep "
               "the subject, composition, camera angle and every recognisable detail of image 1. Do not copy objects or "
               "the composition of image 2.")
SHIPPED_STRENGTH = 0.85            # the editor's default strength slider: 85 % klein, 15 % original photo
HUB = "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"

vol = modal.Volume.from_name("adobe-ps-data", create_if_missing=True)
tf_image = (modal.Image.debian_slim(python_version="3.11")
            .pip_install("tensorflow[and-cuda]==2.18.0", "tensorflow-hub==0.16.1", "tf-keras==2.18.0",
                         "pillow>=10.4,<13", "numpy>=1.26,<2.1")
            .env({"TFHUB_CACHE_DIR": "/vol/cache/tfhub", "TF_CPP_MIN_LOG_LEVEL": "2", "TF_USE_LEGACY_KERAS": "1"}))
eval_image = (modal.Image.debian_slim(python_version="3.11")
              .pip_install("torch==2.5.1", "torchvision==0.20.1", "transformers==4.48.3", "pillow", "numpy<2")
              .env({"HF_HOME": "/vol/hf", "TORCH_HOME": "/vol/torch"}))
cpu_image = modal.Image.debian_slim(python_version="3.11").pip_install("pillow", "numpy<2")
app = modal.App("lumen-style-bench")


def _png(im) -> bytes:
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


@app.cls(image=tf_image, gpu=GPU, volumes={"/vol": vol}, timeout=30 * 60, max_containers=1)
class Magenta:
    @modal.enter()
    def load(self):
        import tensorflow as tf
        import tensorflow_hub as hub

        self.tf = tf
        self.model = hub.load(HUB)

    @modal.method()
    def run(self, content_png: bytes, style_png: bytes) -> dict:
        import numpy as np
        from PIL import Image

        tf = self.tf
        c = Image.open(io.BytesIO(content_png)).convert("RGB")
        s = Image.open(io.BytesIO(style_png)).convert("RGB").resize((256, 256), Image.LANCZOS)   # the model's style size
        ct = tf.constant(np.asarray(c, np.float32)[None] / 255.0)
        st = tf.nn.avg_pool(tf.constant(np.asarray(s, np.float32)[None] / 255.0), ksize=[3, 3], strides=[1, 1], padding="SAME")
        try:
            tf.config.experimental.reset_memory_stats("GPU:0")
        except Exception:  # noqa: BLE001
            pass
        t0 = time.perf_counter()
        out = self.model(ct, st)[0].numpy()[0]
        sec = time.perf_counter() - t0
        try:
            peak = tf.config.experimental.get_memory_info("GPU:0")["peak"] / 1e9
        except Exception:  # noqa: BLE001
            peak = None
        o = Image.fromarray((np.clip(out, 0, 1) * 255).round().astype("uint8")).resize(c.size, Image.LANCZOS)
        return {"png": _png(o), "seconds": sec, "peak_gb": peak}


@app.cls(image=eval_image, gpu="T4", volumes={"/vol": vol}, timeout=30 * 60, max_containers=1)
class Eval:
    @modal.enter()
    def load(self):
        import torch
        import torchvision
        from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

        self.torch = torch
        self.dino_p = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.dino = AutoModel.from_pretrained("facebook/dinov2-base").cuda().eval()
        self.clip_p = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").cuda().eval()
        self.vgg = torchvision.models.vgg19(weights="IMAGENET1K_V1").features[:30].cuda().eval()
        try:
            vol.commit()
        except Exception:  # noqa: BLE001
            pass

    def _dino(self, im):
        x = self.dino_p(images=im, return_tensors="pt").to("cuda")
        f = self.dino(**x).last_hidden_state[:, 0]
        return self.torch.nn.functional.normalize(f, dim=-1)

    def _clip(self, im):
        x = self.clip_p(images=im, return_tensors="pt").to("cuda")
        return self.torch.nn.functional.normalize(self.clip.get_image_features(**x), dim=-1)

    def _grams(self, im):
        import numpy as np

        torch = self.torch
        w, h = im.size
        k = 512 / min(w, h)
        im = im.resize((max(64, round(w * k)), max(64, round(h * k))))
        x = torch.tensor(np.asarray(im, np.float32) / 255).permute(2, 0, 1)[None].cuda()
        x = (x - torch.tensor([0.485, 0.456, 0.406], device="cuda")[:, None, None]) / torch.tensor([0.229, 0.224, 0.225], device="cuda")[:, None, None]
        out = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in (1, 6, 11, 20, 29):                    # relu1_1 .. relu5_1
                _, c, hh, ww = x.shape
                f = x.reshape(c, hh * ww)
                out.append(f @ f.T / (c * hh * ww))
        return out

    @modal.method()
    def score(self, content_png: bytes, style_png: bytes, out_png: bytes) -> dict:
        from PIL import Image

        torch = self.torch
        c, s, o = (Image.open(io.BytesIO(b)).convert("RGB") for b in (content_png, style_png, out_png))
        with torch.no_grad():
            dc = float((self._dino(o) * self._dino(c)).sum())
            co, cc, cs = self._clip(o), self._clip(c), self._clip(s)
            gs, go = self._grams(s), self._grams(o)
            loss = sum(float(((a - b) ** 2).sum()) for a, b in zip(go, gs))
        import math

        return {"dino_content": dc, "clip_content": float((co * cc).sum()), "clip_style": float((co * cs).sum()),
                "gram_log10": math.log10(max(loss, 1e-12))}


@app.function(image=cpu_image, volumes={"/vol": vol}, timeout=60 * 60)
def bench(contents: dict, styles: dict) -> dict:
    import numpy as np
    from PIL import Image, ImageDraw

    flash = modal.Cls.from_name("adobe-ps-flash", "Flash")()
    mag, ev = Magenta(), Eval()
    first_c, first_s = next(iter(contents.values())), next(iter(styles.values()))
    mag.run.remote(first_c, first_s)                       # warm-up: model load and first-call compile
    flash.edit.remote([first_c, first_s], ART_STYLE_Q, 7)
    rows, outs = [], {}
    for cn, cb in contents.items():
        cim = Image.open(io.BytesIO(cb)).convert("RGB")
        for sn, sb in styles.items():
            m = mag.run.remote(cb, sb)
            t0 = time.perf_counter()
            kb = flash.edit.remote([cb, sb], ART_STYLE_Q, 7)
            ksec = time.perf_counter() - t0
            kim = Image.open(io.BytesIO(kb)).convert("RGB").resize(cim.size, Image.LANCZOS)
            k85 = Image.fromarray((np.asarray(cim, np.float32) * (1 - SHIPPED_STRENGTH)
                                   + np.asarray(kim, np.float32) * SHIPPED_STRENGTH).round().astype("uint8"))
            res = {"photo": cn, "style": sn, "magenta_s": m["seconds"], "magenta_peak_gb": m["peak_gb"], "klein_call_s": ksec}
            for meth, ob in (("magenta", m["png"]), ("klein", _png(kim)), ("klein85", _png(k85))):
                res[meth] = ev.score.remote(cb, sb, ob)
                outs[(cn, sn, meth)] = ob
            rows.append(res)
            print(cn, sn, {k: res[k] for k in ("magenta", "klein")}, flush=True)

    def mean(meth, k):
        v = [r[meth][k] for r in rows]
        return float(np.mean(v)), float(np.std(v))

    summ = {meth: {k: mean(meth, k) for k in ("dino_content", "clip_content", "clip_style", "gram_log10")}
            for meth in ("magenta", "klein", "klein85")}
    summ["time"] = {"magenta_s": float(np.median([r["magenta_s"] for r in rows])),
                    "klein_call_s": float(np.median([r["klein_call_s"] for r in rows])),
                    "magenta_peak_gb": max((r["magenta_peak_gb"] or 0) for r in rows)}
    wins = {k: sum(r["klein85"][k] > r["magenta"][k] if k != "gram_log10" else r["klein85"][k] < r["magenta"][k]
                   for r in rows) for k in ("dino_content", "clip_content", "clip_style", "gram_log10")}
    summ["klein85_wins_of"] = {"n": len(rows), **wins}

    # figure: 4 examples, columns photo | artwork | Magenta | klein (shipped)
    pick = [(CONTENT_I, STYLE_I) for CONTENT_I, STYLE_I in zip(list(contents)[:4], list(styles)[:4])]
    W, H, pad, top = 300, 300, 8, 36
    g = Image.new("RGB", (4 * W + 5 * pad, top + len(pick) * (H + pad) + pad), (248, 241, 232))
    d = ImageDraw.Draw(g)
    for j, t in enumerate(("Photo", "Artwork", "Magenta (TensorFlow)", "FLUX.2 klein (LumenAI)")):
        d.text((pad + j * (W + pad) + 6, 12), t, fill=(20, 20, 16))

    def tile(b):
        im = Image.open(io.BytesIO(b)).convert("RGB")
        k = max(W / im.width, H / im.height)
        im = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
        l, t = (im.width - W) // 2, (im.height - H) // 2
        return im.crop((l, t, l + W, t + H))

    for i, (cn, sn) in enumerate(pick):
        y = top + i * (H + pad)
        for j, b in enumerate((contents[cn], styles[sn], outs[(cn, sn, "magenta")], outs[(cn, sn, "klein85")])):
            g.paste(tile(b), (pad + j * (W + pad), y))
    gb = io.BytesIO()
    g.save(gb, "JPEG", quality=90)

    L = ["# Style transfer: FLUX.2 klein (LumenAI) vs Magenta (TensorFlow Hub)", "",
         f"{len(contents)} photos x {len(styles)} artworks = {len(rows)} pairs. Mean (std).", "",
         "| Method | DINOv2 content ↑ | CLIP content ↑ | CLIP style ↑ | Gram loss log10 ↓ |", "|---|---|---|---|---|"]
    for meth, name in (("magenta", "Magenta arbitrary-image-stylization (TF Hub)"), ("klein", "FLUX.2 klein, raw"),
                       ("klein85", "FLUX.2 klein, shipped (85 % blend)")):
        L.append(f"| {name} | " + " | ".join(f"{summ[meth][k][0]:.3f} ({summ[meth][k][1]:.3f})"
                                              for k in ("dino_content", "clip_content", "clip_style", "gram_log10")) + " |")
    t = summ["time"]
    L += ["", f"Median warm time per image: Magenta {t['magenta_s']:.2f} s on {GPU} (peak {t['magenta_peak_gb']:.2f} GB); "
              f"klein {t['klein_call_s']:.2f} s per call (on {GPU}, including the call between containers).",
          f"Shipped klein beats Magenta per pair: {json.dumps(summ['klein85_wins_of'])}"]
    md = "\n".join(L) + "\n"
    out = Path("/vol/outputs/eval/style_bench")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(md)
    (out / "results.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=1))
    (out / "grid.jpg").write_bytes(gb.getvalue())
    vol.commit()
    return {"md": md, "json": json.dumps({"summary": summ, "rows": rows}, indent=1), "grid": gb.getvalue()}


@app.local_entrypoint()
def main():
    def small(p, side):
        from PIL import Image

        im = Image.open(p).convert("RGB")
        k = side / max(im.size)
        if k < 1:
            im = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
        return _png(im)

    contents = {n: small(ASSETS / n, 1024) for n in CONTENT if (ASSETS / n).exists()}
    styles = {n: small(ASSETS / "artworks" / f"{i}.jpg", 1024) for i, n in STYLES}
    print(f"{len(contents)} photos, {len(styles)} artworks")
    r = bench.remote(contents, styles)
    out = HERE / "outputs" / "style_bench"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(r["md"])
    (out / "results.json").write_text(r["json"])
    (out / "grid.jpg").write_bytes(r["grid"])
    print(r["md"])
    print("saved to", out)
