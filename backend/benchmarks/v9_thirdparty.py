"""v9 runners for the teammates' models that need their own environment (see modal_benchmark_extra.py).

Each runner only reads outputs/eval/v9/inputs/ (written by `run_v9 --parts prep`) and writes outputs/eval/v9/<name>/:
images or GLBs + results.json {case: {"seconds", "peak_gb", ...}}. Scoring happens later in run_v9 (score), with the
same code for every method. Standalone on purpose: nothing from our project is imported except in `hunyuan`.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

V9 = Path(os.environ.get("V9_DIR", "/vol/outputs/eval/v9"))
IN = V9 / "inputs"


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return {}


class Out:
    def __init__(self, name: str):
        self.dir = V9 / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.p = self.dir / "results.json"
        self.d = _load(self.p)

    def put(self, case: str, row: dict):
        self.d[case] = row
        self.p.write_text(json.dumps(self.d, indent=1, default=str))

    def err(self, label: str):
        msg = traceback.format_exc()
        print(msg, flush=True)
        with open(self.dir / "errors.txt", "a") as f:
            f.write(f"=== {label}\n{msg}\n")


def timed(fn):
    import torch

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    r = fn()
    torch.cuda.synchronize()
    return r, round(time.perf_counter() - t0, 2), round(torch.cuda.max_memory_allocated() / 1e9, 2)


# ----------------------------------------------------------------------------- Leffa (pose transfer)


def pad_to(img, W=768, H=1024):
    """Fit inside W x H on white (what Leffa's resize_and_center does) and remember how to undo it."""
    from PIL import Image

    w, h = img.size
    s = min(W / w, H / h)
    nw, nh = int(w * s), int(h * s)
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    left, top = (W - nw) // 2, (H - nh) // 2
    canvas.paste(img.convert("RGB").resize((nw, nh), Image.BICUBIC), (left, top))
    return canvas, (left, top, left + nw, top + nh)


def run_leffa(repo: str = "/opt/leffa", ckpt: str = "/vol/thirdparty/leffa", steps: int = 30):
    import numpy as np
    from huggingface_hub import snapshot_download
    from PIL import Image

    sys.path[:0] = [repo, os.path.join(repo, "3rdparty")]
    os.chdir(repo)
    snapshot_download("franciszzj/Leffa", local_dir=ckpt,
                      allow_patterns=["densepose/*", "stable-diffusion-xl-1.0-inpainting-0.1/**", "pose_transfer.pth"])
    from leffa.inference import LeffaInference
    from leffa.model import LeffaModel
    from leffa.transform import LeffaTransform
    from leffa_utils.densepose_predictor import DensePosePredictor

    O = Out("leffa")
    man = _load(IN / "manifest.json")
    dp = DensePosePredictor(config_path=f"{ckpt}/densepose/densepose_rcnn_R_50_FPN_s1x.yaml",
                            weights_path=f"{ckpt}/densepose/model_final_162be9.pkl")
    model = LeffaModel(pretrained_model_name_or_path=f"{ckpt}/stable-diffusion-xl-1.0-inpainting-0.1",
                       pretrained_model=f"{ckpt}/pose_transfer.pth", dtype="float16")
    inf = LeffaInference(model=model)
    tf = LeffaTransform()

    def one(src_pose, ref_person):
        iuv = dp.predict_iuv(np.array(src_pose))[:, :, ::-1]
        data = tf({"src_image": [src_pose], "ref_image": [ref_person],
                   "mask": [Image.fromarray(np.ones_like(np.array(src_pose)) * 255)],
                   "densepose": [Image.fromarray(np.ascontiguousarray(iuv))]})
        out = inf(data, ref_acceleration=True, num_inference_steps=steps, guidance_scale=2.5, seed=42, repaint=False)
        return out["generated_image"][0]

    first = True
    for key in man.get("pose", []):
        if key in O.d:
            continue
        try:
            person = Image.open(IN / "pose" / f"{key}_src.png").convert("RGB")      # who must be kept
            target = Image.open(IN / "pose" / f"{key}_tgt.png").convert("RGB")      # whose pose to copy
            tp, box = pad_to(target)
            rp, _ = pad_to(person)
            if first:                       # warm-up (CUDA kernels), not timed
                one(tp, rp)
                first = False
            g, sec, peak = timed(lambda: one(tp, rp))
            g = g.convert("RGB").resize((768, 1024)).crop(box).resize(target.size, Image.BICUBIC)
            g.save(O.dir / f"{key}.png")
            O.put(key, {"seconds": sec, "peak_gb": peak, "steps": steps})
        except Exception:  # noqa: BLE001
            O.err(key)


# ----------------------------------------------------------------------------- SF3D (object -> 3-D)


def run_sf3d(repo: str = "/opt/sf3d"):
    import torch
    from PIL import Image

    sys.path.insert(0, repo)
    from sf3d.system import SF3D
    from sf3d.utils import resize_foreground

    O = Out("sf3d")
    man = _load(IN / "manifest.json")
    model = SF3D.from_pretrained("stabilityai/stable-fast-3d", config_name="config.yaml",
                                 weight_name="model.safetensors").to("cuda").eval()

    def one(img):
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            mesh, _ = model.run_image([img], bake_resolution=1024, remesh="none", vertex_count=-1)
        return mesh[0] if isinstance(mesh, list) else mesh

    first = True
    for o in man.get("3d", []):
        name = o["name"]
        if name in O.d:
            continue
        try:
            img = resize_foreground(Image.open(IN / "3d" / f"{name}.png").convert("RGBA"), 0.85)
            if first:
                one(img)
                first = False
            mesh, sec, peak = timed(lambda: one(img))
            mesh.export(str(O.dir / f"{name}.glb"), include_normals=True)
            O.put(name, {"seconds": sec, "peak_gb": peak, "faces": int(len(mesh.faces))})
        except Exception:  # noqa: BLE001
            O.err(name)


# ----------------------------------------------------------------------------- TADSR (x4 upscaling)


def _smi_peak(stop: threading.Event, box: list):
    while not stop.wait(0.2):
        try:
            v = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True).stdout.split()
            box[0] = max(box[0], float(v[0]) / 1024)
        except Exception:  # noqa: BLE001
            pass


def run_tadsr(repo: str = "/opt/tadsr", weights: str = "/vol/thirdparty/tadsr"):
    """TADSR's own test script, run twice (1 image, then all 16) -> per-image time = difference / 15, without the
    model-loading time. Peak memory from nvidia-smi (includes the CUDA context)."""
    import shutil

    from huggingface_hub import snapshot_download

    snapshot_download("zty557/TADSR", local_dir=weights)
    pw = Path(repo) / "preset" / "weights"
    for link in (pw, Path(repo) / "weights"):      # RAM's tokenizer loads ./weights/bert-base-uncased
        if not link.exists():
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(weights)
    O = Out("tadsr")
    man = _load(IN / "manifest.json")
    names = man.get("sr", [])
    work = Path("/tmp/tadsr")

    def go(ns, out):
        inp = work / f"in_{len(ns)}"
        shutil.rmtree(inp, ignore_errors=True)
        inp.mkdir(parents=True)
        for n in ns:
            shutil.copy(IN / "sr" / f"{n}_lr.png", inp / f"{n}.png")
        stop, box = threading.Event(), [0.0]
        threading.Thread(target=_smi_peak, args=(stop, box), daemon=True).start()
        t0 = time.perf_counter()
        cmd = [sys.executable, "test_tadsr.py", "--input_image", str(inp), "--output_dir", str(out),
               "--tadsr_path", f"{pw}/tadsr.pkl", "--pretrained_model_name_or_path", str(pw),
               "--ram_ft_path", f"{pw}/DAPE.pth", "--ram_path", f"{pw}/ram_swin_large_14m.pth"]
        print(" ".join(cmd), flush=True)
        rc = subprocess.run(cmd, cwd=repo).returncode
        stop.set()
        return rc, time.perf_counter() - t0, round(box[0], 2)

    rc1, t1, _ = go(names[:1], work / "out_1")
    rc, tn, peak = go(names, work / "out_all")
    per = (tn - t1) / max(1, len(names) - 1)
    print(f"[tadsr] rc {rc1}/{rc}, 1 image {t1:.1f}s, {len(names)} images {tn:.1f}s -> {per:.2f}s per image", flush=True)
    outs = list((work / "out_all").rglob("*.png"))
    for n in names:
        hit = [p for p in outs if p.stem.startswith(n)]
        if hit:
            shutil.copy(hit[0], O.dir / f"{n}.png")
            O.put(n, {"seconds": round(per, 2), "peak_gb": peak, "peak_from": "nvidia-smi"})


# ----------------------------------------------------------------------------- MAT + LaMa via IOPaint (removal)


def run_mat():
    import cv2
    import numpy as np
    import torch
    from iopaint.model_manager import ModelManager
    from iopaint.schema import InpaintRequest

    O = Out("mat")
    man = _load(IN / "manifest.json")
    from iopaint.download import cli_download_model

    for meth in ("mat", "lama"):
        cli_download_model(meth)                 # erase models are only "available" once downloaded
        mm = ModelManager(name=meth, device=torch.device("cuda"))
        first = True
        for n in man.get("removal", []):
            if meth in O.d.get(n, {}):
                continue
            try:
                img = cv2.cvtColor(cv2.imread(str(IN / "removal" / f"{n}_input.png")), cv2.COLOR_BGR2RGB)
                mask = (cv2.imread(str(IN / "removal" / f"{n}_hole.png"), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8) * 255   # H x W (IOPaint adds the channel)
                if first:
                    mm(img, mask, InpaintRequest())
                    first = False
                bgr, sec, peak = timed(lambda: mm(img, mask, InpaintRequest()))
                cv2.imwrite(str(O.dir / f"{n}_{meth}.png"), np.asarray(bgr).astype(np.uint8))
                row = O.d.get(n, {})
                row[meth] = {"seconds": sec, "peak_gb": peak}
                O.put(n, row)
            except Exception:  # noqa: BLE001
                O.err(f"{meth} {n}")
        del mm
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- Hunyuan3D-2 re-run (ours, today's code)


def run_hunyuan(project: str = "/root/adobe-ps"):
    from PIL import Image

    sys.path.insert(0, project)
    from editing import object3d as O3

    O = Out("hunyuan")
    man = _load(IN / "manifest.json")
    first = True
    for o in man.get("3d", []):
        name = o["name"]
        if name in O.d:
            continue
        try:
            cut = Image.open(IN / "3d" / f"{name}.png").convert("RGBA")
            if first:
                O3.make_mesh(cut, texture=False)      # loads the shape model once; not timed
                first = False
            (mesh, info), sec, peak = timed(lambda: O3.make_mesh(cut, texture=True))
            mesh.export(str(O.dir / f"{name}.glb"))
            O.put(name, {**{k: v for k, v in info.items() if isinstance(v, (int, float, str))}, "seconds": sec,
                         "peak_gb": max(info.get("shape_peak_gb", 0) or 0, info.get("texture_peak_gb", 0) or 0)})
        except Exception:  # noqa: BLE001
            O.err(name)


RUNNERS = {"leffa": run_leffa, "sf3d": run_sf3d, "tadsr": run_tadsr, "mat": run_mat, "hunyuan": run_hunyuan}

if __name__ == "__main__":
    RUNNERS[sys.argv[1]]()
