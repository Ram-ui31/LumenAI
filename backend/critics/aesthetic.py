"""Prompt-match and aesthetic critics.

* CLIP score: 100 x cosine(image, prompt) with OpenAI CLIP ViT-L/14 (``openai/clip-vit-large-patch14``).
* Aesthetic: LAION "improved aesthetic predictor" (sac+logos+ava1-l14-linearMSE) — an MLP on the
  L2-normalised ViT-L/14 image embedding, predicting a 1–10 rating.
Both share one CLIP forward pass. CLIP is a *light* model: it may stay resident next to FLUX.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from core.config import get_config, project_path
from core.device import get_device

log = logging.getLogger("adobeps.critics")


@dataclass
class Scores:
    clip: float        # ~15–40 for typical prompt/image pairs; higher = better match
    aesthetic: float   # ~1–10
    combined: float    # weighted, both scaled to roughly 0–1


class AestheticHead(nn.Module):
    """Same layout as the LAION predictor so its state_dict loads unchanged."""

    def __init__(self, input_size: int = 768) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024), nn.Dropout(0.2), nn.Linear(1024, 128), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.Dropout(0.1), nn.Linear(64, 16), nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def _features(out) -> torch.Tensor:
    """transformers >= 5 returns a ModelOutput from get_*_features; older versions a tensor."""
    return out.pooler_output if hasattr(out, "pooler_output") else out


def load_critic() -> tuple:
    """(clip_model, clip_processor, aesthetic_head, device) — registered as the light model 'critic'."""
    from transformers import CLIPModel, CLIPProcessor

    c = get_config()["critics"]
    dev = get_device()
    dtype = torch.float16 if dev in ("cuda", "mps") else torch.float32
    model = CLIPModel.from_pretrained(c["clip_repo"], torch_dtype=dtype).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(c["clip_repo"])
    path = project_path("cache") / "aesthetic" / Path(c["aesthetic_url"]).name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(c["aesthetic_url"], str(path), progress=True)
    head = AestheticHead(model.config.projection_dim)
    head.load_state_dict(torch.load(str(path), map_location="cpu"))
    head = head.to(dev, torch.float32).eval()
    return model, proc, head, dev


def combine(clip: float, aesthetic: float) -> float:
    """Weighted sum with both terms scaled to ~0–1 (CLIP/40, aesthetic/10); weights from config."""
    w = get_config()["critics"]["weights"]
    return round(w["clip"] * clip / 40.0 + w["aesthetic"] * aesthetic / 10.0, 4)


@torch.no_grad()
def score(images: list[Image.Image], prompt: str) -> list[Scores]:
    """Score each image against ``prompt``."""
    if get_config()["backend"] == "stub":
        return [_stub_score(im, prompt) for im in images]
    from core.model_manager import get_manager

    model, proc, head, dev = get_manager().get("critic")
    inputs = proc(text=[prompt], images=[im.convert("RGB") for im in images], return_tensors="pt",
                  padding=True, truncation=True)
    pix = inputs["pixel_values"].to(dev, model.dtype)
    img = _features(model.get_image_features(pixel_values=pix)).float()
    txt = _features(model.get_text_features(input_ids=inputs["input_ids"].to(dev),
                                            attention_mask=inputs["attention_mask"].to(dev))).float()
    img = img / img.norm(dim=-1, keepdim=True)
    txt = txt / txt.norm(dim=-1, keepdim=True)
    clip = (100.0 * img @ txt.T).squeeze(-1).cpu().tolist()
    aes = head(img).squeeze(-1).cpu().tolist()
    rounded = [(round(c, 3), round(a, 3)) for c, a in zip(clip, aes)]
    return [Scores(c, a, combine(c, a)) for c, a in rounded]


def _stub_score(im: Image.Image, prompt: str) -> Scores:
    h = hashlib.sha256(np.asarray(im.convert("RGB").resize((32, 32))).tobytes() + prompt.encode()).digest()
    clip, aes = 20 + h[0] / 255 * 15, 4 + h[1] / 255 * 3
    return Scores(round(clip, 3), round(aes, 3), combine(clip, aes))


@torch.no_grad()
def clip_rank(image: Image.Image, texts: list[str]) -> list[tuple[str, float]]:
    """Zero-shot ranking of ``texts`` for ``image`` with the same CLIP (best first, cosine x 100)."""
    if get_config()["backend"] == "stub":
        h = hashlib.sha256(np.asarray(image.convert("RGB").resize((16, 16))).tobytes()).digest()
        return sorted(((t, float(h[i % len(h)])) for i, t in enumerate(texts)), key=lambda x: -x[1])
    from core.model_manager import get_manager

    model, proc, _, dev = get_manager().get("critic")
    inputs = proc(text=texts, images=[image.convert("RGB")], return_tensors="pt", padding=True, truncation=True)
    img = _features(model.get_image_features(pixel_values=inputs["pixel_values"].to(dev, model.dtype))).float()
    txt = _features(model.get_text_features(input_ids=inputs["input_ids"].to(dev),
                                            attention_mask=inputs["attention_mask"].to(dev))).float()
    sims = (100 * (img / img.norm(dim=-1, keepdim=True)) @ (txt / txt.norm(dim=-1, keepdim=True)).T)[0]
    return sorted(zip(texts, sims.cpu().tolist()), key=lambda x: -x[1])
