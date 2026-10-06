"""Qwen-Image 2.1 backend: loading (bf16 / int8 / nf4), text-to-image, instruction editing, and
structure-guided refinement (start denoising from a noised structure image, e.g. LaMa's fill).

Qwen-Image 2.1 has no mask input: edits are instruction + condition image. Locality comes from our own
compositing (editing.blending.composite) after generation.

Structure-guided refinement (innovation 2): FLUX-style flow matching, x_sigma = (1 - sigma) * z0 + sigma * eps.
Diffusion settles layout in the early (high-sigma) steps and texture in the late ones, so starting at an
intermediate sigma from the structure image's latents keeps its layout and lets Qwen add only detail.
Needs diffusers with QwenImage21Pipeline (git main, Oct 2026).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

REPO = "Qwen/Qwen-Image-2.1"
VARIANTS = ("bf16", "int8", "nf4")


@dataclass
class QwenRun:
    image: Image.Image
    seconds: float
    peak_gb: float
    info: dict[str, Any] = field(default_factory=dict)


def load_pipe(variant: str = "bf16", repo: str = REPO, offload: bool | None = None):
    """Load the pipeline. int8 / nf4 quantize the transformer AND the text encoder on load (bitsandbytes, CUDA).
    ``offload`` (model CPU offload: one big component on the GPU at a time) defaults to on for quantized variants."""
    import torch
    from diffusers import QwenImage21Pipeline

    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    torch.backends.cuda.matmul.allow_tf32 = True      # TF32 tensor cores for the fp32 parts (VAE, norms)
    torch.backends.cudnn.allow_tf32 = True
    dt = torch.bfloat16
    if variant == "bf16":
        pipe = QwenImage21Pipeline.from_pretrained(repo, torch_dtype=dt)
    else:
        from diffusers import BitsAndBytesConfig as DiffBnb
        from diffusers import QwenImage21Transformer2DModel
        from transformers import BitsAndBytesConfig as TfBnb
        from transformers import Qwen3VLForConditionalGeneration

        if variant == "int8":
            dq, tq = DiffBnb(load_in_8bit=True), TfBnb(load_in_8bit=True)
        else:
            kw = dict(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dt)
            dq, tq = DiffBnb(**kw), TfBnb(**kw)
        tr = QwenImage21Transformer2DModel.from_pretrained(repo, subfolder="transformer", quantization_config=dq,
                                                           torch_dtype=dt)
        te = Qwen3VLForConditionalGeneration.from_pretrained(repo, subfolder="text_encoder", quantization_config=tq,
                                                             torch_dtype=dt)
        pipe = QwenImage21Pipeline.from_pretrained(repo, transformer=tr, text_encoder=te, torch_dtype=dt)
    # nf4 fits in ~10 GB, so keep it resident (offload was slower and broke the VLM path); int8 is broken (fp16 overflow)
    offload = False if offload is None else offload
    mode = "gpu"
    if offload:
        try:
            pipe.enable_model_cpu_offload()
            mode = "model_cpu_offload"
        except Exception as e:  # noqa: BLE001  (some bnb builds cannot move 8-bit weights)
            print(f"[qwen] offload unavailable ({e}); keeping everything on the GPU", flush=True)
            pipe.to("cuda")
    else:
        pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    pipe._adobeps_variant, pipe._adobeps_mode = variant, mode
    return pipe


def enable_memory_savers(pipe, vae_tiling: bool = False) -> dict:
    """Cut peak GPU memory: the text encoder (4.4 GB in nf4) lives on the CPU and is moved to the GPU only while
    it encodes a prompt or answers a question. Output is bit-identical (qwen_ab: diff 0.00); peak 20.7 -> 14.0 GB at 1024.
    ``vae_tiling`` is OFF by default: tiled VAE encoding changes the condition image Qwen sees and made it keep the
    person in 14/50 removals (v3); it saved only ~0.9 GB more. Returns what was enabled."""
    import torch

    info = {"vae_tiling": False}
    if vae_tiling:
        try:
            pipe.vae.enable_tiling()
            info["vae_tiling"] = True
        except Exception as e:  # noqa: BLE001
            info["vae_tiling"] = repr(e)[:120]
    te = pipe.text_encoder
    try:
        te.to("cpu")
        torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001  (e.g. 8-bit bnb); keep it on the GPU
        info["text_encoder_offload"] = repr(e)[:120]
        return info
    orig = pipe.encode_prompt
    cache = PromptCache()

    def encode_on_gpu(*a, **k):
        key = cache.key(a, k)
        hit = cache.get(key)
        if hit is not None:            # same prompt + same picture ("Another version", Pro, retries): no encoder trip
            return hit
        te.to("cuda")
        try:
            out = orig(*a, **k)
        finally:
            te.to("cpu")
            torch.cuda.empty_cache()
        cache.put(key, out)
        return out

    pipe.encode_prompt = encode_on_gpu
    pipe._adobeps_prompt_cache = cache
    pipe._adobeps_te_offload = True
    info["text_encoder_offload"] = True
    return info


class PromptCache:
    """Last few text-encoder outputs (prompt embeddings + masks), keyed by the prompt and the exact input pictures.
    The teammates' "prefix KV cache" idea for our pipeline: a repeated request skips moving the 4.4 GB encoder to the
    GPU and re-encoding (about 1-2 s). Only exact repeats hit, so results are bit-identical."""

    def __init__(self, size: int = 8):
        from collections import OrderedDict

        self.size, self.d, self.hits = size, OrderedDict(), 0

    @staticmethod
    def _h(x) -> str:
        import hashlib

        if x is None:
            return "-"
        if isinstance(x, (list, tuple)):
            return "[" + ",".join(PromptCache._h(v) for v in x) + "]"
        if isinstance(x, Image.Image):
            return hashlib.md5(x.mode.encode() + str(x.size).encode() + x.tobytes()).hexdigest()
        if isinstance(x, np.ndarray):
            return hashlib.md5(str(x.shape).encode() + x.tobytes()).hexdigest()
        if hasattr(x, "detach"):                       # torch tensor
            return hashlib.md5(str(tuple(x.shape)).encode() + x.detach().float().cpu().numpy().tobytes()).hexdigest()
        return repr(x)

    def key(self, a, k):
        if k.get("prompt_embeds") is not None:
            return None
        return self._h(list(a)) + "|" + "|".join(f"{n}={self._h(v)}" for n, v in sorted(k.items()) if n != "device")

    def get(self, key):
        if key is None or key not in self.d:
            return None
        self.d.move_to_end(key)
        self.hits += 1
        return self.d[key]

    def put(self, key, val):
        if key is None:
            return
        self.d[key] = val
        while len(self.d) > self.size:
            self.d.popitem(last=False)


def _gen(seed: int):
    import torch

    return torch.Generator("cpu").manual_seed(int(seed))


def _timed(fn):
    import torch

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0, torch.cuda.max_memory_allocated() / 1e9


def work_dims(w: int, h: int, resolution: int = 1024) -> tuple[int, int]:
    """Pipeline's own sizing rule: area resolution**2 at the input aspect ratio, multiples of 32."""
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions

    W, H, _ = calculate_dimensions(resolution * resolution, w / h)
    return int(W) // 32 * 32, int(H) // 32 * 32


def generate(pipe, prompt: str, seed: int, size: int = 1024, steps: int = 40) -> QwenRun:
    out, sec, peak = _timed(lambda: pipe(prompt=prompt, height=size, width=size, num_inference_steps=steps,
                                         generator=_gen(seed)).images[0])
    return QwenRun(out, sec, peak, {"steps": steps})


def edit(pipe, image: Image.Image, instruction: str, seed: int, steps: int = 40, resolution: int = 1024) -> QwenRun:
    """Plain instruction edit (Qwen alone). Returns the image resized back to the input size."""
    w, h = image.size
    out, sec, peak = _timed(lambda: pipe(prompt=instruction, image=image.convert("RGB"), num_inference_steps=steps,
                                         output_resolution=resolution, generator=_gen(seed)).images[0])
    return QwenRun(out.resize((w, h), Image.LANCZOS), sec, peak, {"steps": steps})


def refine(pipe, structure: Image.Image, prompt: str, seed: int, strength: float = 0.5, steps: int = 40,
           condition: Image.Image | None = None, resolution: int = 1024,
           keep: tuple[Image.Image, np.ndarray] | None = None, keep_stop_frac: float = 0.1) -> QwenRun:
    """Structure-guided refinement: run only the last ``strength`` fraction of the schedule, starting from
    ``structure`` noised to that level. ``condition`` (optional) is passed as a reference image (edit mode);
    without it the pass is pure text-to-image from the structure latents.
    ``keep`` = (original image, hole mask): latent preservation - outside the hole the latents are reset after every step
    to the ORIGINAL image noised to the next sigma, x = (1 - sigma) * z_orig + sigma * eps (same eps), so only the hole
    is generated; stops for the last ``keep_stop_frac`` of the steps so the seam can harmonise."""
    import torch
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift, retrieve_timesteps

    w, h = structure.size
    W, H = work_dims(w, h, resolution)
    dev = pipe._execution_device   # with model CPU offload, the VAE's encode() hook moves it to this device

    def run():
        sched = pipe.scheduler
        # the 2.1 VAE is RGBA (native transparency): 4 input channels, as the pipeline itself feeds it
        img_t = pipe.image_processor.preprocess(structure.convert("RGBA"), height=H, width=W).unsqueeze(2)
        with torch.no_grad():
            z0 = pipe._encode_vae_image(img_t.to(dev, pipe.vae.dtype), _gen(seed))
        C = z0.shape[1]
        z0 = pipe._pack_latents(z0, 1, C, H // pipe.vae_scale_factor, W // pipe.vae_scale_factor)
        base = np.linspace(1.0, 1 / steps, steps)
        k = min(steps - 1, max(0, int(round(steps * (1 - strength)))))
        mu = calculate_shift(z0.shape[1], sched.config.get("base_image_seq_len", 256),
                             sched.config.get("max_image_seq_len", 4096), sched.config.get("base_shift", 0.5),
                             sched.config.get("max_shift", 1.15))
        retrieve_timesteps(sched, steps, dev, sigmas=base, mu=mu)        # same shift the pipeline will apply
        s = float(sched.sigmas[k])
        eps = torch.randn(z0.shape, generator=_gen(seed + 1), dtype=torch.float32).to(dev)
        x = ((1 - s) * z0.float() + s * eps)
        kw = dict(prompt=prompt, height=H, width=W, sigmas=list(base[k:]), num_inference_steps=steps - k,
                  generator=_gen(seed))
        if keep is not None:
            orig, hole = keep
            o_t = pipe.image_processor.preprocess(orig.convert("RGBA"), height=H, width=W).unsqueeze(2)
            with torch.no_grad():
                zo = pipe._encode_vae_image(o_t.to(dev, pipe.vae.dtype), _gen(seed))
            zo = pipe._pack_latents(zo, 1, C, H // pipe.vae_scale_factor, W // pipe.vae_scale_factor).float()
            hl, wl = H // pipe.vae_scale_factor, W // pipe.vae_scale_factor
            hm = torch.from_numpy((np.asarray(Image.fromarray((hole > 0).astype(np.uint8) * 255).resize((W, H))) > 0)
                                  .astype(np.float32))[None, None]
            m = torch.nn.functional.max_pool2d(hm, pipe.vae_scale_factor).reshape(1, hl * wl, 1).to(dev)  # 1 = generate
            x = m * x + (1 - m) * ((1 - s) * zo + s * eps)
            n_run = steps - k
            stop_at = n_run - max(1, int(np.ceil(keep_stop_frac * n_run)))

            def cb(p, i, t, kwargs):
                if i < stop_at:
                    sn = float(p.scheduler.sigmas[p.scheduler.step_index])
                    lat = kwargs["latents"]
                    kwargs["latents"] = (m * lat.float() + (1 - m) * ((1 - sn) * zo + sn * eps)).to(lat.dtype)
                return kwargs

            kw.update(callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents"])
        kw["latents"] = x.to(torch.bfloat16)
        if condition is not None:
            kw.update(image=condition.convert("RGB"), output_resolution=resolution)
        return pipe(**kw).images[0], s, steps - k

    (out, s, n), sec, peak = _timed(run)
    return QwenRun(out.resize((w, h), Image.LANCZOS), sec, peak,
                   {"strength": strength, "start_sigma": round(s, 4), "steps_run": n, "condition": condition is not None,
                    "keep_outside": keep is not None, "resolution": resolution})


def ask(pipe, images: list[Image.Image], question: str, max_new_tokens: int = 128) -> str:
    """Use the pipeline's own text encoder (a full Qwen3-VL) as the vision-language assistant.
    Note: Qwen3-VL grounding answers boxes on a 0-1000 relative scale (see to_pixels)."""
    import torch

    proc, te = pipe.processor, pipe.text_encoder
    content = [{"type": "image", "image": im.convert("RGB")} for im in images] + [{"type": "text", "text": question}]
    msgs = [{"role": "user", "content": content}]
    inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                      return_tensors="pt")
    dev = pipe._execution_device   # generate() runs forward(), so offload hooks move the encoder in
    inputs = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}
    moved = getattr(pipe, "_adobeps_te_offload", False)
    if moved:
        te.to("cuda")
    try:
        with torch.no_grad():
            out = te.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    finally:
        reset_vlm_state(te)
        if moved:
            te.to("cpu")
            torch.cuda.empty_cache()
    new = out[:, inputs["input_ids"].shape[1]:]
    return proc.batch_decode(new, skip_special_tokens=True)[0].strip()


def reset_vlm_state(te) -> int:
    """generate() leaves the 3-D rotary offsets of its question in the model (``rope_deltas``); clear them so the next
    prompt encode computes fresh positions. A precaution: the "edit adds nothing after generate" bug of 2026-10-06 was
    the shared seed (fixed by web_api.mix_seed), not this. -> modules reset."""
    n = 0
    for mod in te.modules():
        if getattr(mod, "rope_deltas", None) is not None:
            mod.rope_deltas = None
            n += 1
    return n


def to_pixels(box_0_1000: list[float], size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Qwen3-VL grounding boxes are relative (0-1000 on each axis); convert to pixel coordinates."""
    w, h = size
    x0, y0, x1, y1 = box_0_1000
    return int(x0 * w / 1000), int(y0 * h / 1000), int(x1 * w / 1000), int(y1 * h / 1000)


def crop_window(region: np.ndarray, pad_frac: float = 0.8, min_side: int = 256) -> tuple[int, int, int, int]:
    """Box around the edit region with context (pad = pad_frac x the region's larger side), clipped to the image.
    Qwen only needs to redraw this window: everything outside it is thrown away by the compositing anyway."""
    ys, xs = np.where(region > 0)
    H, W = region.shape
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    pad = int(pad_frac * max(x1 - x0, y1 - y0))
    x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    if x1 - x0 < min_side:
        c = (x0 + x1) // 2; x0, x1 = max(0, min(c - min_side // 2, W - min_side)), min(W, max(c - min_side // 2, 0) + min_side)
    if y1 - y0 < min_side:
        c = (y0 + y1) // 2; y0, y1 = max(0, min(c - min_side // 2, H - min_side)), min(H, max(c - min_side // 2, 0) + min_side)
    return int(x0), int(y0), int(x1), int(y1)


def median_anchor(drafts: list[Image.Image]) -> Image.Image:
    """Consensus anchor: per-pixel median of several drafts (robust estimator - an object invented in only one
    draft is voted out)."""
    a = np.stack([np.asarray(d.convert("RGB")).astype(np.float32) for d in drafts])
    return Image.fromarray(np.median(a, axis=0).astype(np.uint8))
