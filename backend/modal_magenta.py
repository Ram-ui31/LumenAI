"""Fast style transfer: Magenta arbitrary image stylisation (TensorFlow Hub), the team's tensorflow_style.py, as the
"Fast" engine of the Style tool. Its own small app so TensorFlow never shares a container with PyTorch.

    modal deploy modal_magenta.py

Measured in modal_benchmark_style.py (48 pairs): 0.04 s per image warm, 1.3 GB peak; matches the artwork's texture
more closely than FLUX.2 klein but keeps much less of the photo (DINOv2 content 0.44 against 0.81).
"""
import io
import os
import time

import modal

HUB = "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("tensorflow[and-cuda]==2.18.0", "tensorflow-hub==0.16.1", "tf-keras==2.18.0",
                      "pillow>=10.4,<13", "numpy>=1.26,<2.1")
         .env({"TFHUB_CACHE_DIR": "/vol/cache/tfhub", "TF_CPP_MIN_LOG_LEVEL": "2", "TF_USE_LEGACY_KERAS": "1"}))
vol = modal.Volume.from_name("adobe-ps-data", create_if_missing=True)
app = modal.App("lumen-magenta", image=image)


@app.cls(gpu=os.environ.get("MODAL_GPU_MAGENTA", "L4"), volumes={"/vol": vol}, timeout=5 * 60, scaledown_window=5 * 60, max_containers=1)
class Magenta:
    @modal.enter()
    def load(self):
        import numpy as np
        import tensorflow as tf
        import tensorflow_hub as hub

        self.tf, self.np = tf, np
        self.model = hub.load(HUB)
        self.model(tf.zeros([1, 256, 256, 3]), tf.zeros([1, 256, 256, 3]))       # warm-up
        try:
            vol.commit()
        except Exception as e:  # noqa: BLE001
            print("volume commit failed:", e)

    @modal.method()
    def run(self, content_png: bytes, style_png: bytes) -> dict:
        from PIL import Image

        tf, np = self.tf, self.np
        c = Image.open(io.BytesIO(content_png)).convert("RGB")
        s = Image.open(io.BytesIO(style_png)).convert("RGB").resize((256, 256), Image.LANCZOS)
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
            peak = round(tf.config.experimental.get_memory_info("GPU:0")["peak"] / 1e9, 2)
        except Exception:  # noqa: BLE001
            peak = None
        o = Image.fromarray((np.clip(out, 0, 1) * 255).round().astype("uint8")).resize(c.size, Image.LANCZOS)
        b = io.BytesIO()
        o.save(b, "PNG")
        return {"png": b.getvalue(), "seconds": round(sec, 3), "peak_gb": peak}
