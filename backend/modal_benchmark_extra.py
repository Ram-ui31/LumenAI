"""v9 third-party models, each in its own image (their pinned dependencies clash with ours and with each other).
Only the image for BENCH is defined, so a broken build of one never blocks the others.

    BENCH=leffa   modal run --detach modal_benchmark_extra.py     # Leffa pose transfer (SDXL-inpainting based)
    BENCH=sf3d    modal run --detach modal_benchmark_extra.py     # Stable Fast 3D (gated: accept the licence on HF first)
    BENCH=tadsr   modal run --detach modal_benchmark_extra.py     # TADSR x4 super-resolution
    BENCH=mat     modal run --detach modal_benchmark_extra.py     # MAT + LaMa through IOPaint
    BENCH=hunyuan modal run --detach modal_benchmark_extra.py     # our Hunyuan3D-2 again with today's code (fair timing)

Each waits (up to 60 min) for outputs/eval/v9/inputs/manifest.json from `modal run modal_benchmark.py`, then writes
outputs/eval/v9/<BENCH>/ and done.json (or failed.json). Runner code: benchmarks/v9_thirdparty.py.
"""
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REMOTE = "/root/adobe-ps"
BENCH = os.environ.get("BENCH", "").strip().lower()
GPU = os.environ.get("MODAL_GPU", "L40S")
IGNORE = [".venv", ".venv/**", ".cache", ".cache/**", ".hf_cache", ".hf_cache/**", "outputs", "outputs/**", "**/__pycache__/**", ".env"]
ENV = {"HF_HOME": "/vol/hf", "TORCH_HOME": "/vol/torch", "PYTHONUNBUFFERED": "1", "V9_DIR": "/vol/outputs/eval/v9"}
CUDA_DEVEL = "nvidia/cuda:12.4.1-devel-ubuntu22.04"


def _leffa():
    # Leffa (Dec 2024) was built against the diffusers/transformers of that time; its requirements.txt is unpinned
    return (modal.Image.from_registry(CUDA_DEVEL, add_python="3.10")
            .apt_install("git", "libgl1", "libglib2.0-0", "build-essential", "ninja-build")
            .pip_install("torch==2.4.1", "torchvision==0.19.1", index_url="https://download.pytorch.org/whl/cu124")
            .pip_install("numpy<2", "diffusers==0.31.0", "transformers==4.46.3", "huggingface_hub==0.26.2", "accelerate==1.1.1",
                         "peft==0.13.2", "safetensors", "einops", "fvcore", "omegaconf", "onnxruntime", "opencv-python-headless",
                         "pycocotools", "scikit-image", "scipy", "timm", "pillow", "pyyaml", "regex", "matplotlib", "cloudpickle",
                         "imageio", "psutil", "tqdm", "av")              # av: densepose imports its video loader
            .run_commands("git clone --depth 1 https://github.com/franciszzj/Leffa /opt/leffa")
            .run_commands("python -c 'import detectron2' 2>/dev/null || ls /opt/leffa/3rdparty/detectron2 >/dev/null 2>&1 || "
                          "pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'")
            .env({**ENV, "PYTHONPATH": "/opt/leffa:/opt/leffa/3rdparty"}))


def _sf3d():
    return (modal.Image.from_registry(CUDA_DEVEL, add_python="3.10")
            .apt_install("git", "libgl1", "libglib2.0-0", "build-essential", "ninja-build")
            .pip_install("torch==2.4.1", "torchvision==0.19.1", index_url="https://download.pytorch.org/whl/cu124")
            .pip_install("setuptools==69.5.1", "wheel")
            .env({"TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9", "FORCE_CUDA": "1"})
            .run_commands("git clone --depth 1 https://github.com/Stability-AI/stable-fast-3d /opt/sf3d",
                          "cd /opt/sf3d && pip install --no-build-isolation -r requirements.txt")
            .env(ENV))


def _tadsr():
    # TADSR pins torch 2.0.1 / transformers 4.28 (SeeSR lineage); diffusers 0.21 matches that generation
    return (modal.Image.debian_slim(python_version="3.10")
            .apt_install("git", "libgl1", "libglib2.0-0")
            .pip_install("torch==2.0.1", "torchvision==0.15.2", index_url="https://download.pytorch.org/whl/cu118")
            .run_commands("git clone --depth 1 https://github.com/zty557/TADSR /opt/tadsr",
                          "cd /opt/tadsr && grep -v -i -E '^(torch|torchvision)([=<> ]|$)' requirements.txt > /tmp/req.txt && "
                          "pip install -r /tmp/req.txt")
            .pip_install("diffusers==0.21.2", "accelerate==0.24.1", "huggingface_hub==0.25.0")
            .env(ENV))


def _mat():
    return (modal.Image.debian_slim(python_version="3.10")
            .apt_install("libgl1", "libglib2.0-0")
            .pip_install("torch==2.4.1", "torchvision==0.19.1", index_url="https://download.pytorch.org/whl/cu124")
            .pip_install("iopaint", "opencv-python-headless")
            .env(ENV))


def _hunyuan():
    # = modal_object3d.py's image (Modal reuses the cached layers)
    return (modal.Image.from_registry(CUDA_DEVEL, add_python="3.11")
            .apt_install("git", "libgl1", "libglib2.0-0", "build-essential", "ninja-build", "libegl1", "libgomp1", "clang")
            .pip_install("torch==2.5.1", "torchvision==0.20.1", index_url="https://download.pytorch.org/whl/cu124")
            .pip_install("pybind11", "setuptools<70", "wheel", "numpy<2")
            .env({"TORCH_CUDA_ARCH_LIST": "8.6;8.9", "FORCE_CUDA": "1", "MAX_JOBS": "4"})
            .run_commands("git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2 /opt/h3d")
            .run_commands("cd /opt/h3d && grep -v -i -E '^(torch|torchvision|gradio|fastapi|uvicorn|bpy)' requirements.txt > /tmp/req.txt"
                          " && pip install -r /tmp/req.txt && pip install --no-deps -e .")
            .run_commands("cd /opt/h3d/hy3dgen/texgen/custom_rasterizer && pip install --no-build-isolation .")
            .run_commands("cd /opt/h3d/hy3dgen/texgen/differentiable_renderer && pip install --no-build-isolation . "
                          "|| (python setup.py build_ext --inplace && echo built-inplace)")
            .pip_install("opencv-python-headless", "pyyaml", "trimesh", "scipy", "fast-simplification", "pymeshlab")
            .pip_install("transformers==4.48.3", "diffusers==0.32.2", "huggingface_hub>=0.26,<1.0", "accelerate<1.5")
            .env({**ENV, "ADOBEPS_DEVICE": "cuda",
                  "PYTHONPATH": "/opt/h3d:/opt/h3d/hy3dgen/texgen/differentiable_renderer"}))


BUILDERS = {"leffa": _leffa, "sf3d": _sf3d, "tadsr": _tadsr, "mat": _mat, "hunyuan": _hunyuan}
if BENCH not in BUILDERS:
    raise SystemExit(f"Set BENCH to one of {', '.join(BUILDERS)}  (e.g.  BENCH=leffa modal run --detach modal_benchmark_extra.py)")

# BENCH is baked into the image env: inside the container this file is imported again without your shell's env
image = BUILDERS[BENCH]().env({"BENCH": BENCH}).add_local_dir(str(HERE), remote_path=REMOTE, ignore=IGNORE)
vol = modal.Volume.from_name("adobe-ps-data", create_if_missing=True)
app = modal.App(f"adobe-ps-v9-{BENCH}", image=image)


@app.function(gpu=GPU, volumes={"/vol": vol}, timeout=6 * 3600, secrets=[modal.Secret.from_name("adobe-ps-hf")])
def run(bench: str) -> str:
    out = Path(f"/vol/outputs/eval/v9/{bench}")
    man = Path("/vol/outputs/eval/v9/inputs/manifest.json")
    t0 = time.time()
    while not man.exists():                      # inputs come from `modal run modal_benchmark.py` (prep, ~5 min)
        if time.time() - t0 > 3600:
            return "gave up: no inputs after 60 min - start `modal run --detach modal_benchmark.py` first"
        print("waiting for v9 inputs...", flush=True)
        time.sleep(30)
        vol.reload()
    out.mkdir(parents=True, exist_ok=True)
    for f in ("done.json", "failed.json"):
        (out / f).unlink(missing_ok=True)
    if bench == "hunyuan":
        os.makedirs("/vol/outputs", exist_ok=True)
        for name, target in (("outputs", "/vol/outputs"), (".cache", "/vol/cache")):
            os.makedirs(target, exist_ok=True)
            link = Path(REMOTE) / name
            if not link.exists():
                link.symlink_to(target)
    stop = threading.Event()

    def autosave():
        while not stop.wait(120):
            try:
                vol.commit()
            except Exception as e:  # noqa: BLE001
                print("volume commit failed:", e)

    threading.Thread(target=autosave, daemon=True).start()
    t1 = time.time()
    try:
        sys.path.insert(0, REMOTE)
        from benchmarks.v9_thirdparty import RUNNERS

        RUNNERS[bench]()
        (out / "done.json").write_text(json.dumps({"minutes": round((time.time() - t1) / 60, 1)}))
        msg = f"{bench}: done in {(time.time() - t1) / 60:.1f} min"
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb, flush=True)
        (out / "failed.json").write_text(json.dumps({"error": tb[-3000:]}))
        msg = f"{bench}: FAILED - see outputs/eval/v9/{bench}/failed.json\n{tb[-1500:]}"
    finally:
        stop.set()
        vol.commit()
    return msg


@app.local_entrypoint()
def main():
    call = run.spawn(BENCH)            # keeps running in the cloud even if this laptop sleeps
    print(f"{BENCH} started in the cloud (call {call.object_id}). Watch: modal app logs adobe-ps-v9-{BENCH}")
