"""View in 3D: object cut-out -> textured 3-D model with SPAR3D (default) or Stable Fast 3D (SF3D), in its own Modal app.

    modal deploy modal_object3d.py      # makes make3d callable by the LumenAI editor (View in 3D)

SF3D replaced Hunyuan3D-2: one forward pass and cleaner meshes with a baked UV texture. Measured on a T4
(2026-10-07, first request incl. background removal): 34 s, 10.97 GB peak, 17,612 faces, 0.7 MB GLB. The model is gated on Hugging Face: accept its licence at
https://huggingface.co/stabilityai/stable-fast-3d with the account whose token is in the adobe-ps-hf secret.
The build follows the team's 3dreposition.py (stable-fast-3d from GitHub, CUDA 12.1, torch 2.4).
"""
import io
import os
import time

import modal

GPU = os.environ.get("MODAL_GPU_3D", "H100")
# always-warm containers (no wake-up wait). Every warm container is billed for every second it runs:
# deploy with LUMEN_KEEP_WARM=0 to let it scale to zero again.
KEEP_WARM = int(os.environ.get("LUMEN_KEEP_WARM", "1"))


image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "build-essential", "libgomp1", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.4.0", "torchvision==0.19.0", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("setuptools==69.5.1", "wheel", "ninja")
    .env({"TORCH_CUDA_ARCH_LIST": "7.5;8.0;8.6;8.9;9.0", "FORCE_CUDA": "1", "USE_CUDA": "1"})
    .run_commands("git clone https://github.com/Stability-AI/stable-fast-3d /opt/sf3d")
    .run_commands("cd /opt/sf3d && sed -i 's/-march=native/-mtune=generic/g' texture_baker/setup.py uv_unwrapper/setup.py")
    # build the C++/CUDA add-ons against the installed torch (pip's isolated build env has no torch: "No module named
    # 'torch'" when the image was rebuilt on a new account, 2026-10-08)
    .run_commands("cd /opt/sf3d && CC=gcc CXX=g++ pip install --no-cache-dir --no-build-isolation -r requirements.txt")
    .env({"HF_HOME": "/vol/hf", "U2NET_HOME": "/vol/u2net", "PYTHONUNBUFFERED": "1"})
)
spar_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "build-essential", "libgomp1", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.4.0", "torchvision==0.19.0", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("setuptools==69.5.1", "wheel", "ninja")
    .env({"TORCH_CUDA_ARCH_LIST": "7.5;8.0;8.6;8.9;9.0", "FORCE_CUDA": "1", "USE_CUDA": "1"})
    .run_commands("git clone https://github.com/Stability-AI/stable-point-aware-3d /opt/spar3d")
    .run_commands("cd /opt/spar3d && (sed -i 's/-march=native/-mtune=generic/g' texture_baker/setup.py uv_unwrapper/setup.py || true)")
    # AlphaCLIP's setup.py imports pkg_resources, which the newest setuptools (pulled in by pip's isolated build) no
    # longer ships: build without isolation so the pinned setuptools 69.5.1 (and the installed torch) are used
    .run_commands("cd /opt/spar3d && CC=gcc CXX=g++ pip install --no-cache-dir --no-build-isolation -r requirements.txt")
    # transparent-background 1.3.3 imports its (unused) GUI, written for flet < 0.25; newer flet dropped ElevatedButton
    .pip_install("flet==0.24.1")
    .env({"HF_HOME": "/vol/hf", "PYTHONUNBUFFERED": "1"})
)
SPAR_GPU = os.environ.get("MODAL_GPU_SPAR3D", "H100")
vol = modal.Volume.from_name("adobe-ps-data", create_if_missing=True)
app = modal.App("adobe-ps-object3d", image=image)


@app.cls(gpu=GPU, volumes={"/vol": vol}, secrets=[modal.Secret.from_name("adobe-ps-hf")],
         timeout=10 * 60, scaledown_window=5 * 60, min_containers=KEEP_WARM, max_containers=1)
class SF3D:
    @modal.enter()
    def load(self):
        import sys

        import torch

        sys.path.insert(0, "/opt/sf3d")
        from sf3d.system import SF3D as Model

        self.torch = torch
        self.model = Model.from_pretrained("stabilityai/stable-fast-3d", config_name="config.yaml",
                                           weight_name="model.safetensors").to("cuda").eval()
        self.rembg = None
        try:
            vol.commit()                       # keep the weights for the next cold start
        except Exception as e:  # noqa: BLE001
            print("volume commit failed:", e)

    @modal.method()
    def make3d(self, rgba_png: bytes, texture: bool = True, texture_resolution: int = 1024) -> dict:
        """RGBA cut-out (PNG bytes) -> {"glb": bytes, "info": {...}}. Our cut-out already has an alpha channel, so
        SF3D's background removal is skipped; rembg is only loaded for a photo without transparency."""
        from PIL import Image
        from sf3d.utils import remove_background, resize_foreground

        torch = self.torch
        t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        img = Image.open(io.BytesIO(rgba_png)).convert("RGBA")
        if img.getextrema()[3][0] == 255:      # no transparency: cut the object out first
            if self.rembg is None:
                import rembg

                self.rembg = rembg.new_session()
            img = remove_background(img, self.rembg)
        img = resize_foreground(img, 0.85)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
            mesh, _ = self.model.run_image([img], bake_resolution=texture_resolution if texture else 512,
                                           remesh="none", vertex_count=-1)
        buf = io.BytesIO()
        mesh.export(buf, file_type="glb", include_normals=True)
        glb = buf.getvalue()
        info = {"engine": "SF3D", "seconds": round(time.perf_counter() - t0, 2),
                "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                "faces": int(len(mesh.faces)), "glb_mb": round(len(glb) / 1e6, 2)}
        print("[make3d]", info, flush=True)
        return {"glb": glb, "info": info}


@app.cls(image=spar_image, gpu=SPAR_GPU, volumes={"/vol": vol}, secrets=[modal.Secret.from_name("adobe-ps-hf")],
         timeout=10 * 60, scaledown_window=5 * 60, min_containers=KEEP_WARM, max_containers=1)
class SPAR3D:
    """Stable Point Aware 3D: first samples a point cloud of the whole object (a small diffusion model), then builds
    the mesh from it, so depth and the unseen back come out fuller than SF3D's single pass. Gated on Hugging Face:
    accept the licence at https://huggingface.co/stabilityai/stable-point-aware-3d with the adobe-ps-hf account.
    Measured on an L4 (2026-10-08, first request incl. background removal): 28.6 s, 11.03 GB peak, 7,584 faces."""

    @modal.enter()
    def load(self):
        import sys

        import torch

        sys.path.insert(0, "/opt/spar3d")
        from spar3d.system import SPAR3D as Model

        self.torch = torch
        self.model = Model.from_pretrained("stabilityai/stable-point-aware-3d", config_name="config.yaml",
                                           weight_name="model.safetensors", low_vram_mode=False).to("cuda").eval()
        self.remover = None
        try:
            vol.commit()
        except Exception as e:  # noqa: BLE001
            print("volume commit failed:", e)

    @modal.method()
    def make3d(self, rgba_png: bytes, texture: bool = True, texture_resolution: int = 1024) -> dict:
        from PIL import Image
        from spar3d.utils import foreground_crop, remove_background

        torch = self.torch
        t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        img = Image.open(io.BytesIO(rgba_png)).convert("RGBA")
        if img.getextrema()[3][0] == 255:      # no transparency: cut the object out first
            if self.remover is None:
                from transparent_background import Remover

                self.remover = Remover()
            img = remove_background(img, self.remover)
        img = foreground_crop(img, 1.3)        # run.py's default framing
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
            mesh, _ = self.model.run_image(img, bake_resolution=texture_resolution if texture else 512,
                                           remesh="none", vertex_count=-1, return_points=False)
        if isinstance(mesh, (list, tuple)):
            mesh = mesh[0]
        buf = io.BytesIO()
        mesh.export(buf, file_type="glb", include_normals=True)
        glb = buf.getvalue()
        info = {"engine": "SPAR3D", "seconds": round(time.perf_counter() - t0, 2),
                "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                "faces": int(len(mesh.faces)), "glb_mb": round(len(glb) / 1e6, 2)}
        print("[make3d]", info, flush=True)
        return {"glb": glb, "info": info}


@app.local_entrypoint()
def main(path: str = "samples/fox.png", engine: str = "both"):
    """Smoke test: modal run modal_object3d.py --path some_cutout.png [--engine spar3d|sf3d|both]"""
    with open(path, "rb") as f:
        data = f.read()
    for name, cls in (("spar3d", SPAR3D), ("sf3d", SF3D)):
        if engine not in (name, "both"):
            continue
        r = cls().make3d.remote(data)
        with open(f"{name}_test.glb", "wb") as f:
            f.write(r["glb"])
        print(r["info"], f"-> {name}_test.glb")
