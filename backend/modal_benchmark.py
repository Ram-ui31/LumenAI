"""v9 benchmark - our pipeline vs the teammates' model list (logic: benchmarks/run_v9.py).

    modal run --detach modal_benchmark.py                     # everything: prep -> all models -> wait for extras -> score, judge, report
    modal run modal_benchmark.py --parts score,judge,report   # only re-score / re-judge / re-write the summary

Start the third-party runners at the same time (they wait for the prep step by themselves):
    BENCH=leffa   modal run --detach modal_benchmark_extra.py
    BENCH=tadsr   modal run --detach modal_benchmark_extra.py
    BENCH=mat     modal run --detach modal_benchmark_extra.py
    BENCH=hunyuan modal run --detach modal_benchmark_extra.py

Results: modal volume get --force adobe-ps-data outputs/eval/v9 outputs/modal/
Needs: secret adobe-ps-hf. Optional: adobe-ps-trends (GEMINI_API_KEY) -> Gemini is the judge; without it Qwen3-VL judges
(marked as biased). FLUX.1 dev / SF3D are gated and left out for now (--parts ...,fluxdev / BENCH=sf3d once accepted).
"""
import os
import subprocess
import threading
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REMOTE = "/root/adobe-ps"
GPU = os.environ.get("MODAL_GPU", "L40S")
DIFFUSERS = "git+https://github.com/huggingface/diffusers@578c9b2c6636ab2424a0e56186268b83623656b2"
IGNORE = [".venv", ".venv/**", ".cache", ".cache/**", ".hf_cache", ".hf_cache/**", "outputs", "outputs/**", "**/__pycache__/**", ".env"]
ENV = {"HF_HOME": "/vol/hf", "TORCH_HOME": "/vol/torch", "ADOBEPS_DEVICE": "cuda", "PYTHONUNBUFFERED": "1"}
EXTRAS = ["leffa", "sf3d", "tadsr", "mat", "hunyuan"]   # sf3d optional (gated)
MAIN = "prep,qwen,klein,zimage,ip2p,guards,segformer,q2511"   # fluxdev left out for now (gated)
FINAL = "score,judge,report"

image = (    # = modal_qwen.py's image (reused from cache) + three small packages for v9
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install_from_requirements(str(HERE / "requirements.txt"))
    .pip_install(DIFFUSERS, "transformers>=5.17", "accelerate>=1.0", "bitsandbytes>=0.45")
    .pip_install("peft>=0.17", "trimesh", "git+https://github.com/microsoft/MoGe.git")
    .pip_install("controlnet-aux>=0.0.9")
    .pip_install("invisible-watermark", "PyWavelets", "datasets>=3.0")
    .env(ENV)
    .add_local_dir(str(HERE), remote_path=REMOTE, ignore=IGNORE)
)
vol = modal.Volume.from_name("adobe-ps-data", create_if_missing=True)
app = modal.App("adobe-ps-v9", image=image)


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


SECRETS = [modal.Secret.from_name("adobe-ps-hf")] + _optional_secret("adobe-ps-trends")   # Gemini judge; else Qwen3-VL


def _link_volume():
    for name, target in (("outputs", "/vol/outputs"), (".cache", "/vol/cache")):
        os.makedirs(target, exist_ok=True)
        link = Path(REMOTE) / name
        if not link.exists():
            link.symlink_to(target)


@app.function(gpu=GPU, volumes={"/vol": vol}, timeout=10 * 3600, secrets=SECRETS)
def bench(parts: str) -> str:
    _link_volume()
    stop = threading.Event()

    def autosave():
        while not stop.wait(120):
            try:
                vol.commit()
            except Exception as e:  # noqa: BLE001
                print("volume commit failed:", e)

    threading.Thread(target=autosave, daemon=True).start()
    log = Path("/vol/outputs/eval/v9")
    log.mkdir(parents=True, exist_ok=True)
    cmd = ["python", "-m", "benchmarks.run_v9", "--parts", parts]
    try:
        with open(log / "run.log", "a") as lf:
            lf.write(f"\n===== {time.strftime('%H:%M:%S')} {' '.join(cmd)}\n")
            p = subprocess.Popen(cmd, cwd=REMOTE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                 env={**os.environ, "PYTHONFAULTHANDLER": "1"})
            for line in p.stdout:
                print(line, end="", flush=True)
                lf.write(line)
                lf.flush()
            lf.write(f"===== exit code {p.wait()}\n")
    finally:
        stop.set()
        vol.commit()
    s = log / "summary.md"
    return s.read_text() if s.exists() and "report" in parts else f"{parts}: done (see outputs/eval/v9/run.log)"


@app.function(volumes={"/vol": vol}, timeout=120)
def extras_status() -> dict:
    vol.reload()
    out = {}
    for x in EXTRAS:
        d = Path(f"/vol/outputs/eval/v9/{x}")
        out[x] = "done" if (d / "done.json").exists() else "failed" if (d / "failed.json").exists() else \
            "running" if d.exists() else "not started"
    return out


@app.function(volumes={"/vol": vol}, timeout=14 * 3600)
def orchestrate(wait_minutes: int = 90) -> str:
    print(bench.remote(MAIN), flush=True)
    t0 = time.time()
    while time.time() - t0 < wait_minutes * 60:
        st = extras_status.remote()
        print("[extras]", st, flush=True)
        if all(v in ("done", "failed", "not started") for v in st.values()):
            break
        time.sleep(120)
    return bench.remote(FINAL)


@app.local_entrypoint()
def main(parts: str = "", wait_minutes: int = 90):
    # spawn, not remote: the work must not depend on this laptop's connection. With .remote() the laptop waits on
    # the call, and when it slept (lid closed) the call was cancelled. A spawned call keeps going in the cloud.
    call = bench.spawn(parts) if parts else orchestrate.spawn(wait_minutes)
    print(f"started in the cloud (call {call.object_id}). You can close the laptop now.")
    print("watch:  modal app logs adobe-ps-v9      results:  outputs/eval/v9/summary.md on the volume")
