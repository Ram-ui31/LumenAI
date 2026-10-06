"""Lumen Flash: FLUX.2 klein 4B (Apache-2.0) for fast generation and prompt edits - its own container.

    modal run modal_flash.py          # memory + speed test of the three ways to load it (writes outputs/flash_test/)
    modal deploy modal_flash.py       # makes Flash callable by the web API ("Lumen Flash" in the model picker)

Why its own app: Qwen-Image 2.1 already uses ~14 GB of the 16 GB budget in the API container; klein gets its own GPU
container (the teammates' "multi-container architecture"). v9: 1.1 s per 1024 px image, GEdit 7.8 vs our 7.4, but
19 GB peak with everything on the GPU in bf16. To stay under 16 GB the 8 GB Qwen3-4B text encoder is loaded in 4-bit
(MODE "nf4te", ~2.5 GB, stays on the GPU = no transfers) - or, if that hurts quality, the pipeline is model-CPU-offloaded
(MODE "offload": one component on the GPU at a time, ~1-2 s slower). `modal run modal_flash.py` measures both against
plain bf16 and prints which to use.
"""
import io
import os
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
GPU = os.environ.get("MODAL_GPU", "L40S")
MODE = os.environ.get("FLASH_MODE", "nf4te")
REPO = "black-forest-labs/FLUX.2-klein-4B"
DIFFUSERS = "git+https://github.com/huggingface/diffusers@578c9b2c6636ab2424a0e56186268b83623656b2"
BUDGET_GB = 16.0

image = (    # the first layers equal modal_benchmark.py's image (already built and cached by Modal)
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install_from_requirements(str(HERE / "requirements.txt"))
    .pip_install(DIFFUSERS, "transformers>=5.17", "accelerate>=1.0", "bitsandbytes>=0.45")
    .env({"HF_HOME": "/vol/hf", "PYTHONUNBUFFERED": "1", "FLASH_MODE": MODE})
)
vol = modal.Volume.from_name("adobe-ps-data", create_if_missing=True)
app = modal.App("adobe-ps-flash", image=image)
SECRETS = [modal.Secret.from_name("adobe-ps-hf")]


def load(mode: str):
    import torch
    from diffusers import Flux2KleinPipeline

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dt = torch.bfloat16
    if mode == "nf4te":
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        te = AutoModelForCausalLM.from_pretrained(
            REPO, subfolder="text_encoder", torch_dtype=dt, device_map={"": 0},
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dt))
        pipe = Flux2KleinPipeline.from_pretrained(REPO, text_encoder=te, torch_dtype=dt)
        for name in ("transformer", "vae"):
            getattr(pipe, name).to("cuda")
    elif mode == "offload":
        pipe = Flux2KleinPipeline.from_pretrained(REPO, torch_dtype=dt)
        pipe.enable_model_cpu_offload()
    else:                                    # "bf16": everything resident (the v9 setting, 19 GB) - reference only
        pipe = Flux2KleinPipeline.from_pretrained(REPO, torch_dtype=dt).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def fit(w: int, h: int, area: int = 1024 * 1024) -> tuple[int, int]:
    s = (area / max(1, w * h)) ** 0.5
    return max(256, int(w * s) // 16 * 16), max(256, int(h * s) // 16 * 16)


def run(pipe, prompt: str, w: int, h: int, seed: int, images=None):
    import torch

    kw = dict(prompt=prompt, height=h, width=w, num_inference_steps=4, guidance_scale=1.0,
              generator=torch.Generator("cpu").manual_seed(int(seed)))
    if images:
        kw["image"] = images if len(images) > 1 else images[0]
    with torch.inference_mode():
        return pipe(**kw).images[0]


def _png(img) -> bytes:
    b = io.BytesIO()
    img.save(b, "PNG")
    return b.getvalue()


@app.cls(gpu=GPU, volumes={"/vol": vol}, timeout=10 * 60, scaledown_window=10 * 60, max_containers=1, secrets=SECRETS)
class Flash:
    @modal.enter()
    def start(self):
        t0 = time.perf_counter()
        self.pipe = load(MODE)
        run(self.pipe, "warm-up", 512, 512, 1)                 # first call compiles kernels
        print(f"[flash] klein loaded ({MODE}) in {time.perf_counter() - t0:.1f}s", flush=True)
        try:
            vol.commit()
        except Exception:  # noqa: BLE001
            pass

    @modal.method()
    def generate(self, prompt: str, w: int, h: int, seed: int = 7) -> bytes:
        return _png(run(self.pipe, prompt, int(w) // 16 * 16, int(h) // 16 * 16, seed))

    @modal.method()
    def edit(self, pngs: list, instruction: str, seed: int = 7) -> bytes:
        from PIL import Image

        ims = [Image.open(io.BytesIO(b)).convert("RGB") for b in pngs]
        w, h = fit(*ims[0].size)
        return _png(run(self.pipe, instruction, w, h, seed, ims))


@app.function(gpu=GPU, volumes={"/vol": vol}, timeout=40 * 60, secrets=SECRETS)
def bench() -> dict:
    """Peak GPU memory + seconds per image for each loading mode, and how close nf4te / offload stay to bf16."""
    import gc

    import numpy as np
    import torch
    from PIL import Image

    prompts = ["a cozy home interior with a green velvet sofa, warm evening light",
               "a red vintage bicycle leaning on a blue wall, street photo",
               "a bowl of ramen on a wooden table, top view, soft daylight"]
    edit_src = Image.new("RGB", (1024, 768), (230, 225, 215))
    res, imgs = {}, {}
    for mode in ("bf16", "nf4te", "offload"):
        pipe = load(mode)
        run(pipe, "warm-up", 512, 512, 1)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        secs, outs = [], []
        for i, p in enumerate(prompts):
            t0 = time.perf_counter()
            outs.append(run(pipe, p, 1024, 1024, 7 + i))
            torch.cuda.synchronize()
            secs.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        e = run(pipe, "add a green potted plant in the middle of the empty room", 1024, 768, 7, [edit_src])
        torch.cuda.synchronize()
        res[mode] = {"peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                     "s_per_image": round(float(np.mean(secs)), 2), "edit_s": round(time.perf_counter() - t0, 2)}
        imgs[mode] = outs + [e]
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
    out = Path("/vol/outputs/flash_test")
    out.mkdir(parents=True, exist_ok=True)
    for mode, ims in imgs.items():
        for i, im in enumerate(ims):
            im.save(out / f"{mode}_{i}.jpg", quality=92)
    for mode in ("nf4te", "offload"):
        diffs = [float(np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32)).mean())
                 for a, b in zip(imgs[mode], imgs["bf16"])]
        res[mode]["mean_abs_diff_vs_bf16"] = round(float(np.mean(diffs)), 2)
    ok = [m for m in ("nf4te", "offload") if res[m]["peak_gb"] <= BUDGET_GB]
    # the pixel difference only says the 4-bit encoder gives a different (equally good) picture for the same seed -
    # 2026-10-06 run: same quality by eye, 13.35 GB, 1.13 s vs offload 9.5 GB but 7.5 s -> pick the fastest that fits
    res["use"] = min(ok, key=lambda m: res[m]["s_per_image"]) if ok else None
    import json

    (out / "result.json").write_text(json.dumps(res, indent=2))
    vol.commit()
    return res


@app.local_entrypoint()
def main(wait: bool = False):
    """Default: start the test on Modal and return at once (it keeps running if the laptop sleeps); the result is saved
    to outputs/flash_test/result.json in the volume. --wait: stay and print it."""
    import json

    if not wait:
        bench.spawn()
        print("Flash test started on Modal (about 10 min). Later: "
              "modal volume get --force adobe-ps-data outputs/flash_test outputs/modal/")
        return
    r = bench.remote()
    print(json.dumps(r, indent=2))
