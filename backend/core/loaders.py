"""Model loaders registered with the ModelManager. All device/dtype choices come from core.device."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch

from core.config import get_config, project_path
from core.device import get_device, get_dtype

log = logging.getLogger("adobeps.loaders")


class ModelAccessError(RuntimeError):
    """Raised when a download fails because the model is gated or the user is not logged in."""


def _hf_error_hint(repo: str, err: Exception) -> Exception:
    msg = str(err)
    name = type(err).__name__
    if "GatedRepo" in name or "401" in msg or "403" in msg or "gated" in msg.lower() or "restricted" in msg.lower():
        return ModelAccessError(
            f"Model '{repo}' is gated or you are not logged in.\n"
            f"  1. Open https://huggingface.co/{repo} and accept its licence.\n"
            f"  2. In your own terminal run:  hf auth login\n"
            f"Then run the command again. (Original error: {name}: {msg[:300]})"
        )
    return err


def _download(repo: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(repo, filename)
    except Exception as e:  # noqa: BLE001
        raise _hf_error_hint(repo, e) from e


def _dtype() -> torch.dtype:
    return get_dtype(get_config().get("dtype", "bfloat16"))


def load_gguf_transformer(repo: str, gguf_repo: str, gguf_file: str) -> Any:
    """FluxTransformer2DModel from a GGUF file; architecture config is read from ``repo``."""
    from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig

    path = _download(gguf_repo, gguf_file)
    dtype = _dtype()
    try:
        return FluxTransformer2DModel.from_single_file(
            path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
            torch_dtype=dtype,
            config=repo,
            subfolder="transformer",
        )
    except Exception as e:  # noqa: BLE001
        raise _hf_error_hint(repo, e) from e


def load_text_encoders() -> Any:
    """A FluxPipeline holding only tokenizers + CLIP-L + T5-XXL (no transformer, no VAE)."""
    from diffusers import FluxPipeline

    repo = get_config()["text_encoders"]["repo"]
    try:
        pipe = FluxPipeline.from_pretrained(repo, transformer=None, vae=None, torch_dtype=_dtype())
    except Exception as e:  # noqa: BLE001
        raise _hf_error_hint(repo, e) from e
    return pipe.to(get_device())


def _load_vae() -> Any:
    """FLUX.1 VAE (identical across schnell/dev/fill), loaded from the ungated base repo.
    With ``memory.vae_tiling`` it encodes/decodes in 512 px tiles, which caps the decode spike."""
    from diffusers import AutoencoderKL

    repo = get_config()["text_encoders"]["repo"]
    try:
        vae = AutoencoderKL.from_pretrained(repo, subfolder="vae", torch_dtype=_dtype())
    except Exception as e:  # noqa: BLE001
        raise _hf_error_hint(repo, e) from e
    if get_config().get("memory", {}).get("vae_tiling"):
        vae.enable_tiling()
        vae.tile_sample_min_size = 512            # default 1024 would never tile a 1024² image
        vae.tile_latent_min_size = 512 // 8
    return vae


_NO_TEXT = dict(text_encoder=None, text_encoder_2=None, tokenizer=None, tokenizer_2=None)


def load_flux_schnell() -> Any:
    from diffusers import FluxPipeline

    c = get_config()["generation"]
    tr = load_gguf_transformer(c["repo"], c["gguf_repo"], c["gguf_file"])
    try:
        pipe = FluxPipeline.from_pretrained(c["repo"], transformer=tr, vae=_load_vae(), torch_dtype=_dtype(), **_NO_TEXT)
    except Exception as e:  # noqa: BLE001
        raise _hf_error_hint(c["repo"], e) from e
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(get_device())


def load_flux_fill() -> Any:
    from diffusers import FluxFillPipeline

    c = get_config()["fill"]
    tr = load_gguf_transformer(c["repo"], c["gguf_repo"], c["gguf_file"])
    try:
        pipe = FluxFillPipeline.from_pretrained(c["repo"], transformer=tr, vae=_load_vae(), torch_dtype=_dtype(), **_NO_TEXT)
    except Exception as e:  # noqa: BLE001
        raise _hf_error_hint(c["repo"], e) from e
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(get_device())


def load_sam() -> Any:
    """(model, processor) for SAM 2.1 via transformers. float32 for mask quality."""
    from transformers import Sam2Model, Sam2Processor

    repo = get_config()["sam"]["repo"]
    try:
        model = Sam2Model.from_pretrained(repo).to(get_device()).eval()
        proc = Sam2Processor.from_pretrained(repo)
    except Exception as e:  # noqa: BLE001
        raise _hf_error_hint(repo, e) from e
    return model, proc


def load_lama() -> Any:
    """Big-LaMa TorchScript (same weights simple-lama-inpainting uses). Returns (module, device)."""
    c = get_config()["lama"]
    path = Path(os.environ.get("LAMA_MODEL", project_path("cache") / "lama" / c["file"]))
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        log.info("Downloading Big-LaMa to %s", path)
        torch.hub.download_url_to_file(c["url"], str(path), progress=True)
    dev = get_device()
    for d in ([dev, "cpu"] if dev != "cpu" else ["cpu"]):
        try:
            m = torch.jit.load(str(path), map_location=d).eval()
            with torch.inference_mode():  # smoke test: some TorchScript ops are missing on MPS
                m(torch.zeros(1, 3, 64, 64, device=d), torch.zeros(1, 1, 64, 64, device=d))
            log.info("LaMa running on %s", d)
            return m, d
        except Exception as e:  # noqa: BLE001
            log.warning("LaMa on %s failed (%s); trying next device", d, e)
    raise RuntimeError("Could not load Big-LaMa on any device")


def register_all(manager: Any) -> None:
    manager.register("text_encoders", load_text_encoders, heavy=True)
    manager.register("flux_schnell", load_flux_schnell, heavy=True)
    manager.register("flux_fill", load_flux_fill, heavy=True)
    manager.register("sam", load_sam, heavy=False)
    manager.register("lama", load_lama, heavy=False)
    from critics.aesthetic import load_critic
    from critics.detector import load_detector

    manager.register("critic", load_critic, heavy=False)
    manager.register("detector", load_detector, heavy=False)
    from vlm.client import load_vlm

    manager.register("vlm", load_vlm, heavy=True)
