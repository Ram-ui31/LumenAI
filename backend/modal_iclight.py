"""
modal_iclight.py - AI relighting of the WHOLE image with IC-Light (text-prompt + direction).

MODEL  : IC-Light only (SD1.5 foreground-conditioned, Modal GPU).
         NO segmentation, NO depth model. The full frame is relit - there is no subject mask.
RUN    : modal run modal_iclight.py --image photo.jpg --prompt "warm sunset light from the left" --direction left
GPU    : T4 by default (fp16, fits comfortably).  STUDIO_GPU=L4 modal run modal_iclight.py ...
LICENSE: IC-Light weights come from lllyasviel/ic-light (check that repo's license); base model is
         stablediffusionapi/realistic-vision-v51.

--------------------------------------------------------------------------------------------
HOW IT WORKS
--------------------------------------------------------------------------------------------
  1. The photo is downscaled to a working size (--side) and encoded into the diffusion model as
     the "foreground" condition, so content and identity are preserved.
  2. The model generates a relit version following your text prompt. If --direction is set, the
     starting latent is a brightness gradient (bright on the chosen side), which strongly biases
     where the light comes from.
  3. Compose step: the relit result is upscaled back to the original size. Only its LOW-FREQUENCY
     lighting (a big-radius colour/brightness ratio) is transferred onto your ORIGINAL pixels, so
     fine texture and sharpness survive (--detail). --strength blends the final result with the
     original.

TUNABLE PARAMETERS
  --prompt "..."     describe the light, e.g. "warm sunset light", "soft studio light",
                     "neon blue and pink light", "cinematic dramatic light". Default
                     "soft warm light from the left".
  --direction        none | left | right | top | bottom. Where the light comes from (default none =
                     let the prompt decide). Use together with a matching prompt.
  --steps N          denoising steps, default 25 (15 = fast, 40 = refined).
  --cfg C            prompt strength, default 2.0. Higher = follows the prompt harder but can look
                     harsh/over-lit (typical range 1.5..4).
  --seed N           random seed; change it for different lighting variations (default 0).
  --side N           working resolution (long side), default 768. IC-Light was trained around
                     512-768; bigger = slower, more VRAM, not always better.
  --detail D         0..1, default 0.7. 0 = use IC-Light's own pixels (can look "AI-smoothed");
                     1 = keep your original texture and only take the lighting from IC-Light.
  --strength S       0..1, default 1. Blend of the relit result with the original photo.
  --max-side N       cap on the output size, default 1280.
  --fmt png|jpg      --debug/--no-debug (save the raw IC-Light output)   --out DIR (default out_iclight)

OUTPUT: result.<fmt>, compare.jpg (before | after), debug_iclight_raw.png (raw model output)

TIPS
  * Works best when the picture has a clear main subject (people, products).
  * Match prompt and direction, e.g. --prompt "sunlight from the right" --direction right.
  * Models are cached in the "light-studio-cache" volume, so only the first run downloads them.
"""
import os
import time

import modal
import numpy as np

GPU = os.environ.get("STUDIO_GPU", "L40S")

app = modal.App("lumen-iclight")
cache = modal.Volume.from_name("light-studio-cache", create_if_missing=True)  # model weights

studio_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch==2.5.1")
    # pinned: unpinned installs pulled diffusers / transformers / accelerate releases that call torch.accelerator,
    # which only exists from torch 2.6 ("module 'torch' has no attribute 'accelerator'" on 2026-10-08)
    .pip_install("diffusers==0.32.2", "transformers==4.48.3", "accelerate==1.2.1", "safetensors",
                 "huggingface_hub>=0.26,<1.0", "opencv-python-headless", "pillow", "numpy")
    .env({"HF_HOME": "/cache/hf"})
)

with studio_image.imports():
    import cv2
    import torch

IC_BASE = "stablediffusionapi/realistic-vision-v51"
IC_NEG = "lowres, bad anatomy, bad hands, cropped, worst quality"


# ----------------------------------------------------------------------------- helpers
def to_u8(x):
    return (np.clip(x, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def blur(x, sigma):
    if sigma < 0.3:
        return x
    return cv2.GaussianBlur(x, (0, 0), sigmaX=float(sigma), borderType=cv2.BORDER_REFLECT)


def enc(x, fmt):
    u = cv2.cvtColor(to_u8(x), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg" if fmt == "jpg" else ".png", u,
                           [cv2.IMWRITE_JPEG_QUALITY, 95] if fmt == "jpg" else [])
    return buf.tobytes()


def prep(img_bytes, max_side):
    bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not decode the image")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    h, w = rgb.shape[0] // 2 * 2, rgb.shape[1] // 2 * 2
    return np.ascontiguousarray(rgb[:h, :w]).astype(np.float32) / 255.0


def iclight_compose(img, relit_small, detail, strength):
    """Put IC-Light's lighting on the original pixels.
    detail=0: use IC-Light pixels as they are; detail=1: keep original texture and take only the
    low-frequency lighting (ratio transfer), which preserves identity and fine detail."""
    H, W = img.shape[:2]
    relit = np.clip(cv2.resize(relit_small, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32), 0, 1)
    sig = 0.02 * max(H, W)
    ratio = np.clip((blur(relit, sig) + 0.02) / (blur(img, sig) + 0.02), 0.2, 5.0)
    transfer = np.clip(img * ratio, 0, 1)
    result = relit * (1 - detail) + transfer * detail
    return np.clip(img * (1 - strength) + result * strength, 0, 1).astype(np.float32)


# ----------------------------------------------------------------------------- Modal service
@app.cls(gpu=GPU, image=studio_image, volumes={"/cache": cache}, timeout=1800,
         scaledown_window=180, cpu=4, memory=16384)
class Job:
    @modal.enter()
    def load(self):
        self.ic = None

    def _load_iclight(self):
        import safetensors.torch as sf
        from diffusers import AutoencoderKL, UNet2DConditionModel
        from huggingface_hub import hf_hub_download
        from transformers import CLIPTextModel, CLIPTokenizer

        tok = CLIPTokenizer.from_pretrained(IC_BASE, subfolder="tokenizer")
        te = CLIPTextModel.from_pretrained(IC_BASE, subfolder="text_encoder")
        vae = AutoencoderKL.from_pretrained(IC_BASE, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(IC_BASE, subfolder="unet")
        with torch.no_grad():                                    # 4 -> 8 input channels (foreground latent)
            conv = torch.nn.Conv2d(8, unet.conv_in.out_channels, unet.conv_in.kernel_size,
                                   unet.conv_in.stride, unet.conv_in.padding)
            conv.weight.zero_()
            conv.weight[:, :4].copy_(unet.conv_in.weight)
            conv.bias = unet.conv_in.bias
            unet.conv_in = conv
        offset = sf.load_file(hf_hub_download("lllyasviel/ic-light", "iclight_sd15_fc.safetensors"))
        origin = unet.state_dict()
        unet.load_state_dict({k: origin[k] + offset[k] for k in origin}, strict=True)
        unet = unet.to(device="cuda", dtype=torch.float16).eval()
        te = te.to(device="cuda", dtype=torch.float16).eval()
        vae = vae.to(device="cuda", dtype=torch.float16).eval()
        self.ic = dict(tok=tok, te=te, vae=vae, unet=unet)

    def iclight(self, fg_f32, prompt, direction, steps, cfg, seed):
        """fg_f32: the image at the working size (H,W,3 in 0..1). Returns relit (H,W,3)."""
        from diffusers import DPMSolverMultistepScheduler

        with torch.inference_mode():
            if self.ic is None:
                self._load_iclight()
            tok, te, vae, unet = (self.ic[k] for k in ("tok", "te", "vae", "unet"))
            dev, dt = "cuda", torch.float16
            H, W = fg_f32.shape[:2]

            def enc_text(text):
                ids = tok(text, padding="max_length", max_length=tok.model_max_length, truncation=True,
                          return_tensors="pt").input_ids.to(dev)
                return te(ids)[0]

            def to_latent(u8):
                t = torch.from_numpy(u8).float().div(127.0).sub(1.0).permute(2, 0, 1)[None].to(dev, dt)
                return vae.encode(t).latent_dist.mode() * vae.config.scaling_factor

            cond = enc_text(prompt + ", best quality")
            uncond = enc_text(IC_NEG)
            fg_lat = to_latent(np.clip(fg_f32 * 255.0, 0, 255).astype(np.uint8))
            gen = torch.Generator(device=dev).manual_seed(int(seed))
            sch = DPMSolverMultistepScheduler(
                num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
                algorithm_type="sde-dpmsolver++", use_karras_sigmas=True, steps_offset=1)
            sch.set_timesteps(steps, device=dev)
            timesteps = sch.timesteps

            if direction in ("left", "right", "top", "bottom"):
                if direction in ("left", "right"):
                    g = np.linspace(255 if direction == "left" else 0, 0 if direction == "left" else 255, W)
                    bg = np.tile(g[None, :], (H, 1))
                else:
                    g = np.linspace(255 if direction == "top" else 0, 0 if direction == "top" else 255, H)
                    bg = np.tile(g[:, None], (1, W))
                bg_u8 = np.stack([bg] * 3, -1).astype(np.uint8)
                strength = 0.9                                       # initial-latent lighting preference
                init_t = min(int(steps * strength), steps)
                t_start = max(steps - init_t, 0)
                timesteps = sch.timesteps[t_start * sch.order:]
                if hasattr(sch, "set_begin_index"):
                    sch.set_begin_index(t_start * sch.order)
                lat0 = to_latent(bg_u8)
                noise = torch.randn(lat0.shape, generator=gen, device=dev, dtype=dt)
                latents = sch.add_noise(lat0, noise, timesteps[:1])
            else:
                latents = torch.randn((1, 4, H // 8, W // 8), generator=gen, device=dev, dtype=dt) * sch.init_noise_sigma

            emb = torch.cat([uncond, cond])
            concat2 = torch.cat([fg_lat, fg_lat])
            for t in timesteps:
                lin = sch.scale_model_input(latents, t)
                inp = torch.cat([torch.cat([lin, lin]), concat2], dim=1)
                eps = unet(inp, t, encoder_hidden_states=emb).sample
                e_u, e_c = eps.chunk(2)
                eps = e_u + cfg * (e_c - e_u)
                latents = sch.step(eps, t, latents, generator=gen).prev_sample
            px = vae.decode(latents / vae.config.scaling_factor).sample
            out = (px * 127.5 + 127.5).clamp(0, 255)[0].permute(1, 2, 0).float().cpu().numpy() / 255.0
            return out.astype(np.float32)

    @modal.method()
    def run(self, img_bytes, p):
        t0 = time.perf_counter()
        if p["direction"] not in ("none", "left", "right", "top", "bottom"):
            raise ValueError("--direction must be none, left, right, top or bottom")
        img = prep(img_bytes, p["max_side"])
        H, W = img.shape[:2]
        s = p["side"] / max(H, W)
        Wi, Hi = max(int(round(W * s / 8)) * 8, 64), max(int(round(H * s / 8)) * 8, 64)
        small = cv2.resize(img, (Wi, Hi), interpolation=cv2.INTER_AREA)

        t_ic = time.perf_counter()
        relit = self.iclight(small, p["prompt"], p["direction"], int(p["steps"]), float(p["cfg"]), int(p["seed"]))
        t_ic = time.perf_counter() - t_ic
        out = iclight_compose(img, relit, float(p["detail"]), float(p["strength"]))

        fmt = p["fmt"]
        files = {"result." + fmt: enc(out, fmt)}
        sc = min(1.0, 900.0 / H)
        cmp = np.concatenate([cv2.resize(a, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) for a in (img, out)], 1)
        files["compare.jpg"] = enc(cmp, "jpg")
        if p["debug"]:
            files["debug_iclight_raw.png"] = enc(cv2.resize(relit, (W, H), interpolation=cv2.INTER_CUBIC), "png")
        return {"files": files, "info": {"size": (W, H), "work": (Wi, Hi), "ic_sec": t_ic,
                                         "sec": time.perf_counter() - t0}}


# ----------------------------------------------------------------------------- CLI
@app.local_entrypoint()
def main(
    image: str,
    prompt: str = "soft warm light from the left", direction: str = "none",
    steps: int = 25, cfg: float = 2.0, seed: int = 0, side: int = 768,
    detail: float = 0.7, strength: float = 1.0,
    max_side: int = 1280, fmt: str = "png", debug: bool = True,
    out: str = "out_iclight",
):
    import pathlib

    params = {k: v for k, v in locals().items() if k not in ("image", "out", "pathlib")}
    if fmt not in ("png", "jpg"):
        raise SystemExit("--fmt must be png or jpg")
    res = Job().run.remote(pathlib.Path(image).read_bytes(), params)
    out_dir = pathlib.Path(out)
    out_dir.mkdir(exist_ok=True)
    for name, blob in res["files"].items():
        (out_dir / name).write_bytes(blob)
    i = res["info"]
    print(f"image {i['size'][0]}x{i['size'][1]} (IC-Light ran at {i['work'][0]}x{i['work'][1]}) | "
          f"IC-Light {i['ic_sec']:.1f}s | total {i['sec']:.1f}s")
    print(f"saved {len(res['files'])} files to {out_dir}/")