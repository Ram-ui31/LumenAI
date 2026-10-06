"""Object -> 3-D model (Hunyuan3D-2) on Modal - a separate app so its heavy CUDA build can never block other runs.

    modal run --detach modal_object3d.py
    modal run --detach modal_object3d.py --completed      # v8: 3-D from the completed (amodal) cut-outs
    modal volume get --force adobe-ps-data outputs/eval/v5_object3d outputs/modal/
    modal deploy modal_object3d.py                        # makes make3d callable by the LumenAI editor (View in 3D)
"""
import os
import subprocess
import threading
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REMOTE = "/root/adobe-ps"
GPU = os.environ.get("MODAL_GPU", "L40S")
IGNORE = [".venv", ".venv/**", ".cache", ".cache/**", ".hf_cache", ".hf_cache/**", "outputs", "outputs/**", "**/__pycache__/**", ".env"]

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0", "build-essential", "ninja-build", "libegl1", "libgomp1", "clang")   # Modal's python links C++ extensions with clang++
    .pip_install("torch==2.5.1", "torchvision==0.20.1", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("pybind11", "setuptools<70", "wheel", "numpy<2")
    .env({"TORCH_CUDA_ARCH_LIST": "8.6;8.9", "FORCE_CUDA": "1", "MAX_JOBS": "4"})
    .run_commands("git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2 /opt/h3d")
    # requirements minus the web-UI extras; keep torch as installed above
    .run_commands("cd /opt/h3d && grep -v -i -E '^(torch|torchvision|gradio|fastapi|uvicorn|bpy)' requirements.txt > /tmp/req.txt"
                  " && pip install -r /tmp/req.txt && pip install --no-deps -e .")
    .run_commands("cd /opt/h3d/hy3dgen/texgen/custom_rasterizer && pip install --no-build-isolation .")
    .run_commands("cd /opt/h3d/hy3dgen/texgen/differentiable_renderer && pip install --no-build-isolation . "
                  "|| (python setup.py build_ext --inplace && echo built-inplace)")
    .pip_install("opencv-python-headless", "pyyaml", "trimesh", "scipy", "fast-simplification", "pymeshlab")
    # Hunyuan3D-2's checkpoints use the Dinov2 key names of transformers 4.x (query/key/value); transformers 5 renamed
    # them (q_proj/k_proj/...), which broke loading in the first v5 run -> pin the versions it was released with
    .pip_install("transformers==4.48.3", "diffusers==0.32.2", "huggingface_hub>=0.26,<1.0", "accelerate<1.5")
    .env({"HF_HOME": "/vol/hf", "TORCH_HOME": "/vol/torch", "ADOBEPS_DEVICE": "cuda", "PYTHONUNBUFFERED": "1",
          "PYTHONPATH": "/opt/h3d:/opt/h3d/hy3dgen/texgen/differentiable_renderer"})
    .add_local_dir(str(HERE), remote_path=REMOTE, ignore=IGNORE)
)
vol = modal.Volume.from_name("adobe-ps-data", create_if_missing=True)
app = modal.App("adobe-ps-object3d", image=image)


@app.function(gpu=GPU, volumes={"/vol": vol}, timeout=4 * 3600)
def object3d(n_obj: int = 8, n_scene: int = 4, completed: bool = False) -> str:
    for name, target in (("outputs", "/vol/outputs"), (".cache", "/vol/cache")):
        os.makedirs(target, exist_ok=True)
        link = Path(REMOTE) / name
        if not link.exists():
            link.symlink_to(target)
    stop = threading.Event()

    def loop():
        while not stop.wait(120):
            try:
                vol.commit()
            except Exception as e:  # noqa: BLE001
                print("volume commit failed:", e)

    threading.Thread(target=loop, daemon=True).start()
    sub = "v8_object3d" if completed else "v5_object3d"
    logdir = Path(f"/vol/outputs/eval/{sub}")
    logdir.mkdir(parents=True, exist_ok=True)
    cmd = ["python", "-m", "benchmarks.run_v5_object3d", "--n_obj", str(n_obj), "--n_scene", str(n_scene)]
    if completed:
        cmd.append("--completed")
    try:
        with open(logdir / "run.log", "a") as lf:
            lf.write(f"\n===== {' '.join(cmd)}\n")
            proc = subprocess.Popen(cmd, cwd=REMOTE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                    env={**os.environ, "PYTHONFAULTHANDLER": "1"})
            for line in proc.stdout:
                print(line, end="", flush=True)
                lf.write(line)
                lf.flush()
            lf.write(f"===== exit code {proc.wait()}\n")
    finally:
        stop.set()
        vol.commit()
    p = logdir / "summary.md"
    return p.read_text() if p.exists() else f"no summary - see outputs/eval/{sub}/run.log"


@app.function(gpu=GPU, volumes={"/vol": vol}, timeout=30 * 60, scaledown_window=5 * 60, max_containers=1)
def make3d(rgba_png: bytes, texture: bool = True) -> dict:
    """LumenAI "View in 3D": RGBA cut-out (PNG bytes) -> textured GLB bytes. Called by the web API (modal_api.py)
    with modal.Function.from_name("adobe-ps-object3d", "make3d"). Hunyuan3D's own face reducer keeps the mesh
    watertight (our earlier holes came from decimating the UV-split mesh afterwards, which this does not do)."""
    import io
    import sys

    from PIL import Image

    for name, target in (("outputs", "/vol/outputs"), (".cache", "/vol/cache")):
        os.makedirs(target, exist_ok=True)
        link = Path(REMOTE) / name
        if not link.exists():
            link.symlink_to(target)
    os.chdir(REMOTE)
    if REMOTE not in sys.path:
        sys.path.insert(0, REMOTE)
    from editing import object3d as O3

    img = Image.open(io.BytesIO(rgba_png)).convert("RGBA")
    mesh, info = O3.make_mesh(img, texture=texture)
    glb = mesh.export(file_type="glb")
    info["glb_mb"] = round(len(glb) / 1e6, 2)
    print("[make3d]", info, flush=True)
    try:
        vol.commit()                 # keep downloaded weights for the next cold start
    except Exception as e:  # noqa: BLE001
        print("volume commit failed:", e)
    return {"glb": glb, "info": info}


@app.local_entrypoint()
def main(n_obj: int = 8, n_scene: int = 4, completed: bool = False):
    print(object3d.remote(n_obj=n_obj, n_scene=n_scene, completed=completed))
