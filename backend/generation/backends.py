"""Generation backends behind one interface: diffusers+GGUF (default), mflux (Mac fallback), stub (tests)."""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Callable, Protocol

import numpy as np
from PIL import Image

from core.config import get_config
from core.device import MemoryTracker, dtype_name, get_device, get_dtype
from core.seed import make_generator

log = logging.getLogger("adobeps.gen")
StepFn = Callable[[int, int], None]


class GenBackend(Protocol):
    name: str

    def describe(self) -> dict[str, Any]: ...

    def text2img(self, prompt: str, seed: int, steps: int, guidance_scale: float, width: int, height: int,
                 max_sequence_length: int, tracker: MemoryTracker, on_step: StepFn | None = None) -> Image.Image: ...


class DiffusersFluxBackend:
    name = "diffusers"

    def describe(self) -> dict[str, Any]:
        c = get_config()["generation"]
        return {
            "model_id": c["repo"],
            "transformer_weights": f"{c['gguf_repo']}/{c['gguf_file']}",
            "quantization": c["quantization"],
            "dtype": dtype_name(get_dtype(get_config().get("dtype", "bfloat16"))),
            "device": get_device(),
            "scheduler": c["scheduler"],
        }

    def text2img(self, prompt, seed, steps, guidance_scale, width, height, max_sequence_length, tracker, on_step=None):
        from core.model_manager import get_manager
        from core.text import encode

        emb, hit = encode(prompt, max_sequence_length)
        log.info("prompt embeddings %s", "cached" if hit else "encoded")
        tracker.sample()
        pipe = get_manager().get("flux_schnell")          # evicts T5 first if it is resident
        dtype = pipe.transformer.dtype

        def cb(p, i, t, kw):
            tracker.sample()
            if on_step:
                on_step(i, p._num_timesteps)
            return kw

        out = pipe(
            prompt_embeds=emb["prompt_embeds"].to(get_device(), dtype),
            pooled_prompt_embeds=emb["pooled_prompt_embeds"].to(get_device(), dtype),
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=make_generator(seed),
            max_sequence_length=max_sequence_length,
            callback_on_step_end=cb,
        )
        tracker.sample()
        return out.images[0]


class MfluxBackend:
    """MLX-native FLUX (Apple Silicon). Only used if diffusers+GGUF is too slow; see NOTES.md."""

    name = "mflux"
    _model: Any = None

    def describe(self) -> dict[str, Any]:
        c = get_config()["generation"]
        return {"model_id": c["repo"], "transformer_weights": "mflux:schnell", "quantization": f"mlx-q{c['mflux_quantize']}",
                "dtype": "mlx", "device": "mlx", "scheduler": "mflux-default"}

    def text2img(self, prompt, seed, steps, guidance_scale, width, height, max_sequence_length, tracker, on_step=None):
        from mflux.models.flux.variants.txt2img.flux import Flux1

        if MfluxBackend._model is None:
            MfluxBackend._model = Flux1.from_name("schnell", quantize=get_config()["generation"]["mflux_quantize"])
        img = MfluxBackend._model.generate_image(seed=seed, prompt=prompt, num_inference_steps=steps,
                                                 height=height, width=width, guidance=guidance_scale)
        return img.image


class StubBackend:
    """Deterministic procedural images; exercises everything except the neural nets."""

    name = "stub"

    def describe(self) -> dict[str, Any]:
        return {"model_id": "stub", "transformer_weights": None, "quantization": None, "dtype": "float32",
                "device": "cpu", "scheduler": "none"}

    def text2img(self, prompt, seed, steps, guidance_scale, width, height, max_sequence_length, tracker, on_step=None):
        return stub_image(prompt, seed, width, height)


def stub_image(prompt: str, seed: int, width: int, height: int) -> Image.Image:
    """A smooth coloured scene with a few shapes, fully determined by (prompt, seed, size)."""
    import cv2

    ph = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)
    g = make_generator(seed ^ ph)
    rnd = lambda *s: __import__("torch").rand(*s, generator=g).numpy()  # noqa: E731
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    c0, c1 = rnd(3) * 255, rnd(3) * 255
    t = (yy / height)[..., None]
    img = (c0 * (1 - t) + c1 * t).astype(np.float32)
    img += (np.sin(xx / (20 + 60 * rnd(1)[0]))[..., None] * 10)
    img = np.clip(img, 0, 255).astype(np.uint8)
    for _ in range(3):
        cx, cy = (rnd(2) * [width, height]).astype(int)
        r = int(min(width, height) * (0.06 + 0.1 * rnd(1)[0]))
        col = tuple(int(v) for v in rnd(3) * 255)
        cv2.circle(img, (int(cx), int(cy)), r, col, -1)
    return Image.fromarray(img)


def get_gen_backend(name: str | None = None) -> GenBackend:
    name = name or get_config()["backend"]
    return {"diffusers": DiffusersFluxBackend, "mflux": MfluxBackend, "stub": StubBackend}[name]()
