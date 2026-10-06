"""Innovation A: latent-preserving editing for FLUX.1-Fill.

During denoising, the region OUTSIDE the mask is reset after every step to the original image's
latents noised to the scheduler's next sigma (flow matching: x_s = (1-s)*z0 + s*eps), using ONE fixed
noise tensor eps. Only the masked region is ever generated, so untouched content stays intact by
construction. Re-imposition stops for the final ~10 % of steps so the model can harmonise the seam.

Implemented with ``callback_on_step_end`` (diffusers >= 0.32 lets the callback replace ``latents``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from core.seed import make_generator


@dataclass
class PreservedState:
    """Packed tensors ([1, N, C*4]) used by the callback."""

    z0: torch.Tensor          # original image latents (scaled, shifted, packed)
    eps: torch.Tensor         # fixed noise
    m: torch.Tensor           # 1 = generate, 0 = keep (packed)
    sigma0: float             # first sigma of the (strength-truncated) schedule


# ---- packing helpers (mirror FluxPipeline._pack_latents exactly) ------------------

def pack(latents: torch.Tensor) -> torch.Tensor:
    """[B, C, H, W] -> [B, (H/2)*(W/2), C*4] (FLUX 2x2 patchify)."""
    b, c, h, w = latents.shape
    x = latents.view(b, c, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(b, (h // 2) * (w // 2), c * 4)


def unpack(latents: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Inverse of :func:`pack` for latent height/width ``h``, ``w``."""
    b, n, ch = latents.shape
    x = latents.view(b, h // 2, w // 2, ch // 4, 2, 2).permute(0, 3, 1, 4, 2, 5)
    return x.reshape(b, ch // 4, h, w)


def mask_to_latent(mask: np.ndarray, lat_h: int, lat_w: int, channels: int) -> torch.Tensor:
    """Binary pixel mask [H, W] -> packed latent mask. A latent cell counts as 'generate' if ANY of its
    pixels is masked (max-pool), so the known region never overlaps masked pixels."""
    t = torch.from_numpy((mask > 0).astype(np.float32))[None, None]
    t = F.adaptive_max_pool2d(t, (lat_h, lat_w))
    return pack(t.repeat(1, channels, 1, 1))


def blend_known(latents: torch.Tensor, st: PreservedState, sigma_next: float) -> torch.Tensor:
    """The core update:  latents = m*latents + (1-m)*((1-s)*z0 + s*eps)."""
    z_known = (1.0 - sigma_next) * st.z0 + sigma_next * st.eps
    return st.m * latents + (1.0 - st.m) * z_known


# ---- schedule ----------------------------------------------------------------------

def first_sigma(pipe: Any, steps: int, height: int, width: int, strength: float) -> float:
    """Replicates the pipeline's timestep set-up to find the starting sigma (needed to build
    initial latents from the same fixed eps). Deterministic; the pipeline recomputes the same."""
    from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps

    sigmas = np.linspace(1.0, 1 / steps, steps)
    seq = (height // pipe.vae_scale_factor // 2) * (width // pipe.vae_scale_factor // 2)
    sc = pipe.scheduler.config
    mu = calculate_shift(seq, sc.get("base_image_seq_len", 256), sc.get("max_image_seq_len", 4096),
                         sc.get("base_shift", 0.5), sc.get("max_shift", 1.15))
    retrieve_timesteps(pipe.scheduler, steps, "cpu", sigmas=sigmas, mu=mu)
    init_t = min(steps * strength, steps)
    t_start = int(max(steps - init_t, 0))
    return float(pipe.scheduler.sigmas[t_start * pipe.scheduler.order])


# ---- main entry ----------------------------------------------------------------------

@torch.no_grad()
def encode_image_latents(pipe: Any, image: Image.Image, height: int, width: int) -> torch.Tensor:
    """VAE-encode ``image`` with the pipeline's own preprocessing; returns unpacked, scaled latents.
    Uses the distribution mode (not a sample) so it is deterministic."""
    dev, dtype = pipe._execution_device, pipe.vae.dtype
    x = pipe.image_processor.preprocess(image, height=height, width=width).to(dev, dtype)
    z = pipe.vae.encode(x).latent_dist.mode()
    return (z - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor


def prepare_state(pipe: Any, image: Image.Image, mask: np.ndarray, seed: int, steps: int,
                  strength: float, dtype: torch.dtype) -> PreservedState:
    w, h = image.size
    dev = pipe._execution_device
    z0 = encode_image_latents(pipe, image, h, w).to(dtype)
    c, lat_h, lat_w = z0.shape[1:]
    eps = torch.randn((1, c, lat_h, lat_w), generator=make_generator(seed), dtype=torch.float32)
    return PreservedState(
        z0=pack(z0).to(dev),
        eps=pack(eps.to(dtype)).to(dev),
        m=mask_to_latent(mask, lat_h, lat_w, c).to(dev, dtype),
        sigma0=first_sigma(pipe, steps, h, w, strength),
    )


def run_fill(pipe: Any, image: Image.Image, mask: np.ndarray, prompt_embeds: torch.Tensor,
             pooled_prompt_embeds: torch.Tensor, seed: int, steps: int, guidance_scale: float,
             strength: float = 1.0, latent_preserve: bool = True, stop_frac: float = 0.1,
             max_sequence_length: int = 512, on_step: Any = None,
             output_type: str = "pil") -> Any:
    """FLUX.1-Fill on ``image`` (dims multiple of 16) with binary ``mask`` (1 = regenerate).

    Initial latents are built from the same fixed ``eps`` whether or not ``latent_preserve`` is on, so
    A-on vs A-off comparisons differ only by the preservation step. ``output_type="latent"`` returns
    ``(packed_latents, state)`` (used by tests).
    """
    w, h = image.size
    assert w % 16 == 0 and h % 16 == 0, "FLUX needs dims divisible by 16"
    dtype = pipe.transformer.dtype if pipe.transformer.dtype in (torch.float16, torch.bfloat16) else prompt_embeds.dtype
    st = prepare_state(pipe, image, mask, seed, steps, strength, dtype)
    init = (1.0 - st.sigma0) * st.z0 + st.sigma0 * st.eps

    total_holder: dict[str, int] = {}

    def callback(p: Any, i: int, t: Any, kw: dict[str, Any]) -> dict[str, Any]:
        lat = kw["latents"]
        total = total_holder.setdefault("n", p._num_timesteps)
        n_free = max(1, math.ceil(total * stop_frac)) if stop_frac > 0 else 0
        if latent_preserve and i < total - n_free:
            sigma_next = float(p.scheduler.sigmas[p.scheduler.step_index])
            lat = blend_known(lat, st, sigma_next).to(lat.dtype)
        if on_step is not None:
            on_step(i, total)
        return {"latents": lat}

    mask_img = Image.fromarray(((mask > 0) * 255).astype(np.uint8))
    out = pipe(
        image=image,
        mask_image=mask_img,
        prompt_embeds=prompt_embeds.to(pipe._execution_device, dtype),
        pooled_prompt_embeds=pooled_prompt_embeds.to(pipe._execution_device, dtype),
        latents=init.to(dtype),
        height=h,
        width=w,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=make_generator(seed),
        max_sequence_length=max_sequence_length,
        callback_on_step_end=callback,
        callback_on_step_end_tensor_inputs=["latents"],
        output_type=output_type,
    )
    if output_type == "latent":
        return out.images, st
    return out.images[0]
