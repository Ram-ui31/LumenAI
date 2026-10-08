"""LumenAI API on Modal: the backend for the editor in ../frontend. Logic lives in server/web_api.py.

    modal deploy modal_api.py        # prints the permanent URL: https://<workspace>--adobe-ps-api-studio-web.modal.run
    modal app logs adobe-ps-api      # live log of every request

Put "<that URL>/api/v1" into frontend/config.js. The GPU starts on the first request (about 1-3 min: the models load
from the volume cache) and shuts down after 10 idle minutes, so an idle API costs nothing.
"""
import os
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REMOTE = "/root/adobe-ps"
GPU = os.environ.get("MODAL_GPU", "H100")
# always-warm containers (no wake-up wait). Every warm container is billed for every second it runs:
# deploy with LUMEN_KEEP_WARM=0 to let it scale to zero again.
KEEP_WARM = int(os.environ.get("LUMEN_KEEP_WARM", "1"))

DIFFUSERS = "git+https://github.com/huggingface/diffusers@578c9b2c6636ab2424a0e56186268b83623656b2"
IGNORE = [".venv", ".venv/**", ".cache", ".cache/**", ".hf_cache", ".hf_cache/**", "outputs", "outputs/**", "**/__pycache__/**", ".env"]
ENV = {"HF_HOME": "/vol/hf", "TORCH_HOME": "/vol/torch", "ADOBEPS_DEVICE": "cuda", "PYTHONUNBUFFERED": "1"}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0", "fonts-dejavu-core")
    .pip_install_from_requirements(str(HERE / "requirements.txt"))
    .pip_install(DIFFUSERS, "transformers>=5.17", "accelerate>=1.0", "bitsandbytes>=0.45")
    .pip_install("peft>=0.17", "trimesh", "git+https://github.com/microsoft/MoGe.git")
    .pip_install("controlnet-aux>=0.0.9")
    .pip_install("fastapi[standard]>=0.115", "python-multipart")
    .pip_install("invisible-watermark", "PyWavelets")          # LumenAI watermark (dwtDctSvd)
    .env(ENV)
    .add_local_dir(str(HERE), remote_path=REMOTE, ignore=IGNORE)
)
vol = modal.Volume.from_name("adobe-ps-data", create_if_missing=True)
app = modal.App("adobe-ps-api", image=image)


def _optional_secret(name: str) -> list:
    """The secret if it exists in this workspace, else nothing (deploy/run must not fail without it)."""
    try:
        s = modal.Secret.from_name(name)
    except Exception:  # noqa: BLE001
        return []
    try:
        s.hydrate()
        return [s]
    except AttributeError:                # older modal client without hydrate(): assume it exists
        return [s]
    except Exception:  # noqa: BLE001
        print(f"note: Modal secret '{name}' not found - running without it", flush=True)
        return []


def _link_volume():
    for name, target in (("outputs", "/vol/outputs"), (".cache", "/vol/cache")):
        os.makedirs(target, exist_ok=True)
        link = Path(REMOTE) / name
        if not link.exists():
            link.symlink_to(target)


@app.cls(gpu=GPU, volumes={"/vol": vol}, timeout=60 * 60, scaledown_window=10 * 60, min_containers=KEEP_WARM, max_containers=2,
         secrets=[modal.Secret.from_name("adobe-ps-hf")] + _optional_secret("adobe-ps-trends"))   # GEMINI_API_KEY for Trends
class Studio:
    @modal.enter()
    def start(self):
        _link_volume()
        os.chdir(REMOTE)
        sys.path.insert(0, REMOTE)
        from server.web_api import Engine

        self.engine = Engine()
        self.engine.load()                 # load once per container start, before the first request is served
        import threading

        def warm():                        # first use of these was slow (watermark library ~1 min, guard model download)
            try:
                from server import guard, watermark
                from PIL import Image as _I

                watermark.embed(_I.new("RGB", (512, 512), "white"))
                guard.injection_score("make the sky blue")
                self.engine.depth(_I.new("RGB", (256, 256), "white"))    # Depth Anything V2 Small (bokeh, dolly)
                print("[api] watermark + guard + depth warmed up", flush=True)
            except Exception as e:  # noqa: BLE001
                print("[api] warm-up skipped:", e, flush=True)
        threading.Thread(target=warm, daemon=True).start()
        print(f"[api] Qwen-Image 2.1 loaded in {self.engine.loaded_at}s", flush=True)
        try:
            vol.commit()                   # keep any newly downloaded weights for the next cold start
        except Exception as e:  # noqa: BLE001
            print("volume commit failed:", e)

    @modal.exit()
    def stop(self):
        try:
            vol.commit()
        except Exception:  # noqa: BLE001
            pass

    @modal.asgi_app()
    def web(self):
        from server.web_api import build_app

        return build_app(self.engine)
