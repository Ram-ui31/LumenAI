"""AI upscaling for LumenAI: TADSR (one-step diffusion super-resolution), the v9 benchmark winner.

    modal deploy modal_tadsr.py

v9 (16 photos, x4 from a degraded 128 px image): LPIPS 0.217 vs 0.303 for our old Lanczos + sharpening and 0.290
for a light Qwen refine; under 0.5 s per image; 5.7 GB. Its own environment (torch 2.0.1 / transformers 4.28, as
the authors pin) - that is why it is a separate app, called by the web API like the 3-D app.
Weights: huggingface.co/zty557/TADSR (cached in the shared volume by the v9 run). Pi-Lab licence (academic use).
"""
import io
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import modal

GPU = os.environ.get("MODAL_GPU", "H100")
# always-warm containers (no wake-up wait). Every warm container is billed for every second it runs:
# deploy with LUMEN_KEEP_WARM=0 to let it scale to zero again.
KEEP_WARM = int(os.environ.get("LUMEN_KEEP_WARM", "1"))

REPO = "/opt/tadsr"
WEIGHTS = "/vol/thirdparty/tadsr"

image = (    # = the v9 benchmark image for TADSR (Modal reuses the cached layers)
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.0.1", "torchvision==0.15.2", index_url="https://download.pytorch.org/whl/cu118")
    .run_commands("git clone --depth 1 https://github.com/zty557/TADSR /opt/tadsr",
                  "cd /opt/tadsr && grep -v -i -E '^(torch|torchvision)([=<> ]|$)' requirements.txt > /tmp/req.txt && "
                  "pip install -r /tmp/req.txt")
    .pip_install("diffusers==0.21.2", "accelerate==0.24.1", "huggingface_hub==0.25.0")
    .env({"HF_HOME": "/vol/hf", "TORCH_HOME": "/vol/torch", "PYTHONUNBUFFERED": "1"})
)
vol = modal.Volume.from_name("adobe-ps-data", create_if_missing=True)
app = modal.App("adobe-ps-tadsr", image=image)
MAX_OUT = 4096            # longest side of the result


@app.cls(gpu=GPU, volumes={"/vol": vol}, timeout=15 * 60, scaledown_window=5 * 60, min_containers=KEEP_WARM, max_containers=1)
class Upscaler:
    @modal.enter()
    def load(self):
        import torch
        from huggingface_hub import snapshot_download

        snapshot_download("zty557/TADSR", local_dir=WEIGHTS)
        for link in (Path(REPO) / "preset" / "weights", Path(REPO) / "weights"):   # RAM loads ./weights/bert-base-uncased
            if not link.exists():
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(WEIGHTS)
        os.chdir(REPO)
        sys.path.insert(0, REPO)                       # the repo ships its own patched diffusers
        from ram.models.ram_lora import ram
        from tadsr import TADSR_test

        self.args = SimpleNamespace(pretrained_model_name_or_path=WEIGHTS, tadsr_path=f"{WEIGHTS}/tadsr.pkl",
                                    ram_path=f"{WEIGHTS}/ram_swin_large_14m.pth", ram_ft_path=f"{WEIGHTS}/DAPE.pth",
                                    mixed_precision="fp16", merge_and_unload_lora=False, vae_decoder_tiled_size=224,
                                    vae_encoder_tiled_size=1024, latent_tiled_size=96, latent_tiled_overlap=32,
                                    timesteps=500, prompt="", process_size=512, upscale=4, align_method="adain", seed=42)
        self.dtype = torch.float16
        self.model = TADSR_test(self.args)
        self.dape = ram(pretrained=self.args.ram_path, pretrained_condition=self.args.ram_ft_path, image_size=384,
                        vit="swin_l").eval().to("cuda").to(dtype=self.dtype)
        try:
            vol.commit()
        except Exception:  # noqa: BLE001
            pass

    @modal.method()
    def upscale(self, png: bytes, scale: int = 4) -> bytes:
        """PNG in -> PNG out, ``scale`` x larger (2 or 4). TADSR always works x4; x2 is its x4 result resized."""
        import torch
        from PIL import Image
        from torchvision import transforms

        from my_utils.wavelet_color_fix import adain_color_fix
        from ram import inference_ram

        scale = 2 if int(scale) == 2 else 4
        im = Image.open(io.BytesIO(png)).convert("RGB")
        w, h = im.size
        target = (w * scale, h * scale)
        s = min(1.0, MAX_OUT / max(target))
        target = (int(target[0] * s), int(target[1] * s))
        lr = im.resize((max(8, target[0] // 4), max(8, target[1] // 4)), Image.LANCZOS) if (scale == 2 or s < 1) else im
        up = lr.resize((lr.width * 4 // 8 * 8, lr.height * 4 // 8 * 8), Image.BICUBIC)   # what test_tadsr.py does
        lq = transforms.ToTensor()(up).unsqueeze(0).to("cuda")
        ram_in = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(
            transforms.Resize((384, 384))(lq)).to(dtype=self.dtype)
        prompt = f"{inference_ram(ram_in, self.dape)[0]}, ,"
        with torch.no_grad():
            out = self.model(lq * 2 - 1, prompt=prompt, timesteps=self.args.timesteps)
        res = transforms.ToPILImage()(out[0].float().cpu().clamp(-1, 1) * 0.5 + 0.5)
        res = adain_color_fix(target=res, source=up)
        if res.size != target:
            res = res.resize(target, Image.LANCZOS)
        b = io.BytesIO()
        res.save(b, "PNG")
        return b.getvalue()


@app.local_entrypoint()
def main(path: str = ""):
    """modal run modal_tadsr.py --path photo.jpg   (writes photo_x4.png)"""
    if path:
        out = Upscaler().upscale.remote(Path(path).read_bytes(), 4)
        Path(path).with_name(Path(path).stem + "_x4.png").write_bytes(out)
        print("written", Path(path).stem + "_x4.png")
