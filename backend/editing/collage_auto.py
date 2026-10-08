#!/usr/bin/env python3
"""Auto collage for LumenAI (the team's collage_v0, wired to /api/v1/auto-collage).

Prompt-directed collage V0.2: rank 5-10 photos and render three Instagram designs.

The rendering path is deterministic. The optional Hugging Face CLIP model supplies
semantic embeddings, prompt relevance, and a zero-shot aesthetic signal. If model
loading fails and --strict-models is not set, the program still runs with
perceptual hashes and technical image-quality metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
SIZES = {"post": (1080, 1350), "story": (1080, 1920)}
RESAMPLE = Image.Resampling.LANCZOS


@dataclass
class PhotoAnalysis:
    index: int
    path: str
    width: int
    height: int
    aspect_ratio: float
    sharpness: float = 0.0
    exposure: float = 0.0
    contrast: float = 0.0
    resolution: float = 0.0
    clip_aesthetic: float | None = None
    prompt_relevance: float | None = None
    technical_score: float = 0.0
    total_score: float = 0.0
    embedding: list[float] | None = field(default=None, repr=False)
    perceptual_hash: list[float] | None = field(default=None, repr=False)
    faces: list[list[int]] = field(default_factory=list)
    focal_point: list[float] = field(default_factory=lambda: [0.5, 0.5])
    saliency_point: list[float] = field(default_factory=lambda: [0.5, 0.5])
    duplicate_of: str | None = None
    max_similarity: float = 0.0
    selected: bool = False
    role: str = "unused"

    def public_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("embedding", None)
        result.pop("perceptual_hash", None)
        return result


def log(message: str) -> None:
    print(message, flush=True)


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def discover_images(inputs: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        path = Path(raw).expanduser()
        if path.is_dir():
            paths.extend(p for p in sorted(path.iterdir()) if p.suffix.lower() in VALID_EXTENSIONS)
        elif path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
            paths.append(path)
        else:
            raise ValueError(f"Not a supported image or directory: {path}")
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    if not 5 <= len(unique) <= 10:
        raise ValueError(f"Expected 5-10 photos, found {len(unique)}.")
    return unique


def open_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def robust_normalize(values: Sequence[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float32)
    if len(array) == 0:
        return []
    lo, hi = np.percentile(array, [10, 90])
    if hi - lo < 1e-6:
        return [0.5] * len(array)
    return np.clip((array - lo) / (hi - lo), 0.0, 1.0).tolist()


def technical_features(image: Image.Image) -> tuple[float, float, float, float]:
    rgb = np.asarray(image.resize((min(720, image.width), min(720, image.height))), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sharpness = float(np.log1p(cv2.Laplacian(gray, cv2.CV_64F).var()))
    mean = float(gray.mean() / 255.0)
    clipped = float(np.mean((gray < 8) | (gray > 247)))
    exposure = max(0.0, 1.0 - 2.2 * abs(mean - 0.52) - 1.5 * clipped)
    contrast = float(min(1.0, gray.std() / 64.0))
    resolution = float(np.log1p(image.width * image.height))
    return sharpness, exposure, contrast, resolution


def perceptual_hash(image: Image.Image, size: int = 16) -> list[float]:
    gray = np.asarray(ImageOps.grayscale(image).resize((size, size), RESAMPLE), dtype=np.float32)
    dct = cv2.dct(gray)
    low = dct[:8, :8]
    return (low > np.median(low[1:])).astype(np.float32).reshape(-1).tolist()


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    return float(np.dot(av, bv) / denom) if denom else 0.0


def hash_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    return float(1.0 - np.mean(np.abs(np.asarray(a) - np.asarray(b))))


class ClipAnalyzer:
    """Hugging Face CLIP embeddings plus a prompt-relative aesthetic score."""

    POSITIVE_PROMPTS = [
        "a beautiful professional photograph with excellent composition",
        "a sharp well-lit memorable photograph",
        "an aesthetically pleasing high quality social media photo",
    ]
    NEGATIVE_PROMPTS = [
        "a blurry poorly composed photograph",
        "a badly exposed low quality snapshot",
        "an unappealing photograph with distracting composition",
    ]

    def __init__(self, model_id: str, device: str):
        import torch

        # Some hosted notebook images ship torch and torchvision wheels compiled
        # against different versions. Transformers treats torchvision as optional
        # for CLIP, but merely finding that broken installation makes it import
        # torchvision::nms and crash. Detect that exact situation and let
        # Transformers use its Pillow/NumPy image path instead.
        try:
            import torchvision  # noqa: F401
        except Exception as exc:
            if "torchvision::nms" not in str(exc) and "operator torchvision" not in str(exc):
                raise
            for module_name in [name for name in sys.modules if name == "torchvision" or name.startswith("torchvision.")]:
                sys.modules.pop(module_name, None)
            import transformers.utils.import_utils as transformers_imports

            transformers_imports._torchvision_available = False

        from transformers import AutoProcessor, CLIPModel

        self.torch = torch
        self.device = self._resolve_device(device)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id, torch_dtype=dtype).to(self.device).eval()
        text_inputs = self.processor(
            text=self.POSITIVE_PROMPTS + self.NEGATIVE_PROMPTS,
            return_tensors="pt",
            padding=True,
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        with torch.inference_mode():
            text = self.model.get_text_features(**text_inputs)
            if hasattr(text, "pooler_output"):
                text = text.pooler_output
            self.text_features = text / text.norm(dim=-1, keepdim=True)

    def _resolve_device(self, requested: str) -> str:
        if requested != "auto":
            return requested
        if self.torch.cuda.is_available():
            return "cuda"
        if getattr(self.torch.backends, "mps", None) and self.torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def analyze(
        self, images: Sequence[Image.Image], batch_size: int, prompt: str = ""
    ) -> tuple[np.ndarray, list[float], list[float] | None]:
        all_features: list[np.ndarray] = []
        all_scores: list[float] = []
        npos = len(self.POSITIVE_PROMPTS)
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with self.torch.inference_mode():
                features = self.model.get_image_features(**inputs)
                if hasattr(features, "pooler_output"):
                    features = features.pooler_output
                features = features / features.norm(dim=-1, keepdim=True)
                logits = features @ self.text_features.T * 10.0
                pos = logits[:, :npos].mean(dim=1)
                neg = logits[:, npos:].mean(dim=1)
                # Logistic pairwise preference, normalized to [0, 1].
                scores = self.torch.sigmoid(pos - neg)
            all_features.append(features.float().cpu().numpy())
            all_scores.extend(scores.float().cpu().tolist())
        image_features = np.concatenate(all_features, axis=0)
        relevance: list[float] | None = None
        if prompt.strip():
            query_inputs = self.processor(
                text=[prompt.strip(), f"a photograph of {prompt.strip()}"],
                return_tensors="pt",
                padding=True,
            )
            query_inputs = {k: v.to(self.device) for k, v in query_inputs.items()}
            with self.torch.inference_mode():
                query = self.model.get_text_features(**query_inputs)
                if hasattr(query, "pooler_output"):
                    query = query.pooler_output
                query = query / query.norm(dim=-1, keepdim=True)
                query = query.mean(dim=0)
                query = query / query.norm()
            relevance = (image_features @ query.float().cpu().numpy()).astype(np.float32).tolist()
        return image_features, all_scores, relevance


class SubjectLocator:
    """Lightweight face-first crop focus. Replaceable by a segmenter later."""

    def __init__(self) -> None:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.face_detector = cv2.CascadeClassifier(str(cascade_path))

    @staticmethod
    def saliency_centroid(rgb: np.ndarray) -> tuple[float, float]:
        small = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        spectrum = np.fft.fft2(gray)
        log_amp = np.log(np.abs(spectrum) + 1e-8)
        avg = cv2.blur(log_amp, (3, 3))
        residual = log_amp - avg
        saliency = np.abs(np.fft.ifft2(np.exp(residual + 1j * np.angle(spectrum)))) ** 2
        saliency = cv2.GaussianBlur(saliency.astype(np.float32), (9, 9), 2.5)
        threshold = np.percentile(saliency, 80)
        weights = np.maximum(saliency - threshold, 0)
        total = float(weights.sum())
        if total < 1e-8:
            return 0.5, 0.5
        ys, xs = np.indices(weights.shape)
        return float((xs * weights).sum() / total / 127.0), float((ys * weights).sum() / total / 127.0)

    def locate(self, image: Image.Image) -> tuple[list[list[int]], list[float], list[float]]:
        rgb = np.asarray(image)
        scale = min(1.0, 1000.0 / max(image.size))
        work = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else rgb
        gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)
        found = () if self.face_detector.empty() else self.face_detector.detectMultiScale(
            gray, scaleFactor=1.12, minNeighbors=5, minSize=(28, 28))
        faces: list[list[int]] = []
        for x, y, w, h in found:
            faces.append([round(x / scale), round(y / scale), round(w / scale), round(h / scale)])
        saliency = list(self.saliency_centroid(rgb))
        if faces:
            weights = np.asarray([w * h for _, _, w, h in faces], dtype=np.float32)
            centers = np.asarray([[(x + w / 2) / image.width, (y + h / 2) / image.height] for x, y, w, h in faces])
            face_center = np.average(centers, axis=0, weights=weights)
            focal = (0.8 * face_center + 0.2 * np.asarray(saliency)).tolist()
        else:
            focal = saliency
        return faces, [round(float(v), 5) for v in focal], [round(float(v), 5) for v in saliency]


def analyze_photos(
    paths: Sequence[Path],
    model_id: str,
    device: str,
    batch_size: int,
    no_models: bool,
    strict_models: bool,
    prompt: str = "",
) -> tuple[list[Image.Image], list[PhotoAnalysis], dict[str, Any]]:
    images = [open_image(path) for path in paths]
    records: list[PhotoAnalysis] = []
    locator = SubjectLocator()
    for index, (path, image) in enumerate(zip(paths, images)):
        sharpness, exposure, contrast, resolution = technical_features(image)
        faces, focal, saliency = locator.locate(image)
        records.append(
            PhotoAnalysis(
                index=index,
                path=str(path),
                width=image.width,
                height=image.height,
                aspect_ratio=round(image.width / image.height, 5),
                sharpness=sharpness,
                exposure=exposure,
                contrast=contrast,
                resolution=resolution,
                perceptual_hash=perceptual_hash(image),
                faces=faces,
                focal_point=focal,
                saliency_point=saliency,
            )
        )

    model_info: dict[str, Any] = {"embedding_model": None, "model_status": "disabled" if no_models else "pending"}
    if not no_models:
        try:
            log(f"Loading Hugging Face model: {model_id}")
            clip = ClipAnalyzer(model_id, device)
            embeddings, aesthetic_scores, prompt_similarities = clip.analyze(images, batch_size, prompt)
            normalized_relevance = robust_normalize(prompt_similarities or [])
            for position, (record, embedding, aesthetic) in enumerate(zip(records, embeddings, aesthetic_scores)):
                record.embedding = embedding.tolist()
                record.clip_aesthetic = round(float(aesthetic), 6)
                if normalized_relevance:
                    record.prompt_relevance = round(float(normalized_relevance[position]), 6)
            model_info = {
                "embedding_model": model_id,
                "model_status": "loaded",
                "device": clip.device,
                "aesthetic_method": "CLIP positive-vs-negative aesthetic prompt preference",
                "prompt_ranking": bool(prompt.strip()),
            }
        except Exception as exc:
            if strict_models:
                raise
            log(f"WARNING: model unavailable ({exc}). Continuing with deterministic fallback.")
            model_info = {
                "embedding_model": None,
                "model_status": "fallback",
                "model_error": str(exc),
                "aesthetic_method": "technical image metrics fallback",
            }

    sharp = robust_normalize([r.sharpness for r in records])
    res = robust_normalize([r.resolution for r in records])
    for record, sharp_norm, res_norm in zip(records, sharp, res):
        technical = 0.45 * sharp_norm + 0.25 * record.exposure + 0.20 * record.contrast + 0.10 * res_norm
        record.technical_score = round(technical, 6)
        aesthetic = record.clip_aesthetic if record.clip_aesthetic is not None else technical
        if prompt.strip() and record.prompt_relevance is not None:
            record.total_score = round(0.46 * aesthetic + 0.29 * technical + 0.25 * record.prompt_relevance, 6)
        else:
            record.total_score = round(0.62 * aesthetic + 0.38 * technical, 6)
    return images, records, model_info


def select_photos(records: list[PhotoAnalysis], duplicate_threshold: float, max_selected: int) -> list[int]:
    ranked = sorted(records, key=lambda r: (-r.total_score, r.index))
    selected: list[PhotoAnalysis] = []
    for candidate in ranked:
        nearest: PhotoAnalysis | None = None
        similarity = 0.0
        for kept in selected:
            if candidate.embedding is not None and kept.embedding is not None:
                current = cosine(candidate.embedding, kept.embedding)
            else:
                current = hash_similarity(candidate.perceptual_hash or [], kept.perceptual_hash or [])
            if current > similarity:
                nearest, similarity = kept, current
        candidate.max_similarity = round(similarity, 6)
        if nearest is not None and similarity >= duplicate_threshold:
            candidate.duplicate_of = nearest.path
            continue
        selected.append(candidate)
        if len(selected) >= max_selected:
            break

    # Collages remain useful even when many inputs are near-duplicates.
    minimum = min(5, len(records))
    if len(selected) < minimum:
        for candidate in ranked:
            if candidate not in selected:
                candidate.duplicate_of = None
                selected.append(candidate)
                if len(selected) >= minimum:
                    break

    for position, record in enumerate(selected):
        record.selected = True
        record.role = "hero" if position == 0 else "supporting"
    return [r.index for r in selected]


MOOD_ACCENTS: dict[str, tuple[int, int, int]] = {
    "warm": (224, 112, 72),
    "cool": (67, 132, 180),
    "romantic": (202, 88, 125),
    "bold": (240, 86, 54),
    "natural": (74, 139, 96),
    "mono": (100, 100, 104),
}


def infer_mood(prompt: str, requested: str) -> str:
    if requested != "auto":
        return requested
    text = prompt.lower()
    keyword_groups = [
        ("romantic", ("love", "wedding", "couple", "romantic", "anniversary", "date")),
        ("warm", ("sunset", "summer", "beach", "golden", "vacation", "travel", "sunny")),
        ("cool", ("winter", "snow", "night", "city", "blue", "ocean", "mountain")),
        ("bold", ("party", "birthday", "festival", "concert", "celebration", "dance")),
        ("natural", ("nature", "forest", "garden", "green", "hike", "wildlife", "camp")),
        ("mono", ("classic", "minimal", "black and white", "monochrome", "formal")),
    ]
    for mood, keywords in keyword_groups:
        if any(keyword in text for keyword in keywords):
            return mood
    return "natural"


def prompt_copy(prompt: str, title: str, subtitle: str, mood: str) -> dict[str, str]:
    if title.strip():
        resolved_title = title.strip()
    elif prompt.strip():
        stopwords = {"a", "an", "the", "create", "make", "collage", "photo", "photos", "of", "for", "with"}
        words = [word.strip(".,!?-_/") for word in prompt.split()]
        useful = [word for word in words if word and word.lower() not in stopwords]
        resolved_title = " ".join(useful[:4]).title() or "Moments"
    else:
        resolved_title = "Moments"
    default_subtitles = {
        "warm": "Sunlit stories worth keeping",
        "cool": "A quiet study in atmosphere",
        "romantic": "The little moments, together",
        "bold": "Loud memories. Good energy.",
        "natural": "A small story, beautifully arranged",
        "mono": "Collected with intention",
    }
    return {
        "title": resolved_title[:32],
        "subtitle": (subtitle.strip() or default_subtitles[mood])[:52],
    }


def blend_color(a: Sequence[int], b: Sequence[int], amount: float) -> tuple[int, int, int]:
    return tuple(int(round((1.0 - amount) * x + amount * y)) for x, y in zip(a, b))


def extract_palette(images: Sequence[Image.Image], mood: str = "natural") -> dict[str, tuple[int, int, int]]:
    strip = Image.new("RGB", (96 * len(images), 96))
    for i, image in enumerate(images):
        strip.paste(ImageOps.fit(image, (96, 96), method=RESAMPLE), (96 * i, 0))
    quantized = strip.quantize(colors=7, method=Image.Quantize.MEDIANCUT).convert("RGB")
    colors = sorted(quantized.getcolors(maxcolors=10000) or [], reverse=True)
    palette = [color for _, color in colors]
    colorful = max(palette, key=lambda c: max(c) - min(c), default=(192, 112, 88))
    mean = tuple(int(v) for v in np.asarray(strip).reshape(-1, 3).mean(axis=0))
    light_bg = tuple(int(0.13 * v + 0.87 * 255) for v in mean)
    dark_bg = tuple(int(0.18 * v + 0.82 * 24) for v in mean)
    accent = tuple(int(0.82 * v + 0.18 * 128) for v in colorful)
    mood_accent = MOOD_ACCENTS[mood]
    accent = blend_color(accent, mood_accent, 0.46)
    light_bg = blend_color(light_bg, mood_accent, 0.035 if mood != "mono" else 0.015)
    dark_bg = blend_color(dark_bg, mood_accent, 0.10 if mood != "mono" else 0.035)
    return {"light_bg": light_bg, "dark_bg": dark_bg, "accent": accent, "ink": (26, 26, 28), "white": (248, 247, 243)}


def crop_to_aspect(
    image: Image.Image, target_size: tuple[int, int], focal: Sequence[float], faces: Sequence[Sequence[int]]
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    tw, th = target_size
    target_ratio = tw / th
    source_ratio = image.width / image.height
    if source_ratio > target_ratio:
        crop_h = image.height
        crop_w = round(crop_h * target_ratio)
    else:
        crop_w = image.width
        crop_h = round(crop_w / target_ratio)

    cx, cy = focal[0] * image.width, focal[1] * image.height
    left = cx - crop_w / 2
    top = cy - crop_h / 2
    if faces:
        margin = 0.08 * min(image.width, image.height)
        face_left = min(f[0] for f in faces) - margin
        face_top = min(f[1] for f in faces) - margin
        face_right = max(f[0] + f[2] for f in faces) + margin
        face_bottom = max(f[1] + f[3] for f in faces) + margin
        if face_right - face_left <= crop_w:
            left = min(left, face_left)
            left = max(left, face_right - crop_w)
        if face_bottom - face_top <= crop_h:
            top = min(top, face_top)
            top = max(top, face_bottom - crop_h)

    left = int(round(np.clip(left, 0, image.width - crop_w)))
    top = int(round(np.clip(top, 0, image.height - crop_h)))
    box = (left, top, left + crop_w, top + crop_h)
    return image.crop(box).resize(target_size, RESAMPLE), box


def rounded_mask(size: tuple[int, int], radius: int) -> Image.Image:
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=radius, fill=255)
    return mask


def paste_crop(
    canvas: Image.Image,
    image: Image.Image,
    record: PhotoAnalysis,
    box: tuple[int, int, int, int],
    radius: int = 0,
    border: int = 0,
    border_color: tuple[int, int, int] = (255, 255, 255),
) -> dict[str, Any]:
    x, y, w, h = box
    inner_w, inner_h = max(1, w - 2 * border), max(1, h - 2 * border)
    cropped, source_box = crop_to_aspect(image, (inner_w, inner_h), record.focal_point, record.faces)
    if border:
        tile = Image.new("RGB", (w, h), border_color)
        tile.paste(cropped, (border, border))
    else:
        tile = cropped
    mask = rounded_mask(tile.size, radius) if radius else None
    canvas.paste(tile, (x, y), mask)
    return {"photo": record.path, "destination": [x, y, w, h], "source_crop": list(source_box)}


def add_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    size: int,
    fill: tuple[int, int, int],
    bold: bool = False,
    max_width: int | None = None,
    min_size: int = 14,
) -> None:
    font = load_font(size, bold=bold)
    while max_width and size > min_size and draw.textbbox((0, 0), text, font=font)[2] > max_width:
        size -= 1
        font = load_font(size, bold=bold)
    draw.text(xy, text, font=font, fill=fill)


def editorial_layout(size_name: str, count: int) -> list[tuple[int, int, int, int]]:
    if size_name == "post":
        base = [(62, 176, 660, 780), (750, 176, 268, 360), (750, 560, 268, 396), (62, 990, 300, 278), (386, 990, 300, 278), (710, 990, 308, 278)]
    else:
        base = [(64, 218, 952, 934), (64, 1190, 448, 336), (536, 1190, 480, 336), (64, 1550, 294, 284), (382, 1550, 294, 284), (700, 1550, 316, 284)]
    return base[:count]


def minimal_layout(size_name: str, count: int) -> list[tuple[int, int, int, int]]:
    if size_name == "post":
        hero = (48, 120, 984, 600)
        area_top, area_bottom, gap = 746, 1298, 18
        # Four supports look more balanced as 2x2; five supports use a 3-column grid.
        cols = 2 if count - 1 <= 4 else 3
    else:
        hero = (48, 142, 984, 860)
        area_top, area_bottom, gap, cols = 1028, 1868, 18, 2
    support_count = max(0, count - 1)
    boxes = [hero]
    if support_count:
        rows = math.ceil(support_count / cols)
        width = (984 - gap * (cols - 1)) // cols
        height = (area_bottom - area_top - gap * (rows - 1)) // rows
        for i in range(support_count):
            row, col = divmod(i, cols)
            boxes.append((48 + col * (width + gap), area_top + row * (height + gap), width, height))
    return boxes


def scrapbook_layout(size_name: str, count: int) -> list[tuple[int, int, int, int]]:
    if size_name == "post":
        base = [(116, 176, 610, 590), (622, 102, 360, 420), (650, 554, 360, 390), (78, 720, 390, 450), (402, 886, 430, 330), (742, 946, 280, 290)]
    else:
        base = [(102, 252, 720, 716), (654, 126, 354, 470), (626, 766, 390, 486), (72, 952, 410, 510), (344, 1278, 510, 424), (704, 1540, 310, 292)]
    return base[:count]


def render_editorial(
    images: Sequence[Image.Image],
    records: Sequence[PhotoAnalysis],
    size_name: str,
    palette: dict[str, tuple[int, int, int]],
    copy: dict[str, str],
) -> tuple[Image.Image, list[dict[str, Any]]]:
    size = SIZES[size_name]
    canvas = Image.new("RGB", size, palette["light_bg"])
    draw = ImageDraw.Draw(canvas)
    add_label(draw, (62, 54), f"{copy['title'].upper()} / 01", 28, palette["accent"], True, 956)
    add_label(draw, (62, 92), copy["subtitle"], 25, palette["ink"], max_width=956)
    boxes = editorial_layout(size_name, len(images))
    placements = [paste_crop(canvas, image, record, box, radius=8) for image, record, box in zip(images, records, boxes)]
    draw = ImageDraw.Draw(canvas)
    if size_name == "post":
        draw.rectangle((722, 176, 730, 956), fill=palette["accent"])
    else:
        draw.rectangle((64, 180, 1016, 188), fill=palette["accent"])
    return canvas, placements


def render_minimal(
    images: Sequence[Image.Image],
    records: Sequence[PhotoAnalysis],
    size_name: str,
    palette: dict[str, tuple[int, int, int]],
    copy: dict[str, str],
) -> tuple[Image.Image, list[dict[str, Any]]]:
    canvas = Image.new("RGB", SIZES[size_name], (246, 246, 244))
    draw = ImageDraw.Draw(canvas)
    add_label(draw, (48, 42), copy["title"].upper(), 36, palette["ink"], True, 750)
    add_label(draw, (850, 54), f"{len(images):02d}", 24, palette["accent"], True)
    boxes = minimal_layout(size_name, len(images))
    placements = [paste_crop(canvas, image, record, box, radius=2) for image, record, box in zip(images, records, boxes)]
    return canvas, placements


def render_scrapbook(
    images: Sequence[Image.Image],
    records: Sequence[PhotoAnalysis],
    size_name: str,
    palette: dict[str, tuple[int, int, int]],
    copy: dict[str, str],
    seed: int,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    width, height = SIZES[size_name]
    canvas = Image.new("RGB", (width, height), palette["dark_bg"])
    draw = ImageDraw.Draw(canvas)
    # Deterministic paper grain/dots.
    rng = random.Random(seed + (0 if size_name == "post" else 1000))
    for _ in range(180):
        x, y = rng.randrange(width), rng.randrange(height)
        color = tuple(min(255, c + rng.randrange(5, 18)) for c in palette["dark_bg"])
        draw.ellipse((x, y, x + 2, y + 2), fill=color)
    add_label(draw, (66, 50), copy["title"].lower(), 50, palette["white"], True, 948, 26)
    add_label(draw, (70, 108), copy["subtitle"].upper(), 19, palette["accent"], True, 940, 13)
    boxes = scrapbook_layout(size_name, len(images))
    angles = [-4.5, 4.0, -2.5, 3.2, -3.0, 4.6]
    placements: list[dict[str, Any]] = []
    for i, (image, record, box) in enumerate(zip(images, records, boxes)):
        x, y, w, h = box
        border = 14
        cropped, source_box = crop_to_aspect(image, (w - 2 * border, h - 2 * border), record.focal_point, record.faces)
        card = Image.new("RGB", (w, h), (250, 247, 238))
        card.paste(cropped, (border, border))
        shadow = Image.new("RGBA", card.size, (0, 0, 0, 115)).filter(ImageFilter.GaussianBlur(10))
        angle = angles[i]
        rotated_shadow = shadow.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
        rotated = card.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(250, 247, 238))
        px = x - (rotated.width - w) // 2
        py = y - (rotated.height - h) // 2
        canvas.paste(rotated_shadow, (px + 10, py + 13), rotated_shadow)
        canvas.paste(rotated, (px, py))
        placements.append({"photo": record.path, "destination": [x, y, w, h], "rotation": angle, "source_crop": list(source_box)})
    draw = ImageDraw.Draw(canvas)
    draw.line((64, height - 62, width - 64, height - 62), fill=palette["accent"], width=5)
    return canvas, placements


def render_all(
    images: Sequence[Image.Image],
    records: Sequence[PhotoAnalysis],
    output_dir: Path,
    seed: int,
    prompt: str,
    title: str,
    subtitle: str,
    requested_mood: str,
) -> dict[str, Any]:
    mood = infer_mood(prompt, requested_mood)
    copy = prompt_copy(prompt, title, subtitle, mood)
    palette = extract_palette(images, mood)
    renderers = {"editorial": render_editorial, "scrapbook": render_scrapbook, "minimal": render_minimal}
    manifests: dict[str, Any] = {}
    for style, renderer in renderers.items():
        manifests[style] = {}
        for size_name in SIZES:
            if style == "scrapbook":
                canvas, placements = renderer(images, records, size_name, palette, copy, seed)  # type: ignore[misc]
            else:
                canvas, placements = renderer(images, records, size_name, palette, copy)  # type: ignore[misc]
            filename = f"{style}_{size_name}.jpg"
            canvas.save(output_dir / filename, quality=94, subsampling=0, optimize=True)
            manifests[style][size_name] = {"file": filename, "size": list(canvas.size), "placements": placements}
            log(f"Saved {output_dir / filename}")
    return {
        "prompt_design": {"prompt": prompt, "mood": mood, **copy},
        "palette": {k: list(v) for k, v in palette.items()},
        "renders": manifests,
    }


def save_debug_contact_sheet(images: Sequence[Image.Image], records: Sequence[PhotoAnalysis], output: Path) -> None:
    cell_w, cell_h = 360, 320
    sheet = Image.new("RGB", (cell_w * 2, cell_h * math.ceil(len(images) / 2)), (235, 235, 235))
    for i, (image, record) in enumerate(zip(images, records)):
        thumb = ImageOps.contain(image, (330, 240), RESAMPLE)
        x = (i % 2) * cell_w + 15
        y = (i // 2) * cell_h + 10
        sheet.paste(thumb, (x + (330 - thumb.width) // 2, y))
        draw = ImageDraw.Draw(sheet)
        fx = x + (330 - thumb.width) // 2 + int(record.focal_point[0] * thumb.width)
        fy = y + int(record.focal_point[1] * thumb.height)
        draw.ellipse((fx - 8, fy - 8, fx + 8, fy + 8), outline=(255, 50, 50), width=4)
        status = record.role if record.selected else ("duplicate" if record.duplicate_of else "unused")
        draw.text((x, y + 248), f"#{record.index + 1} score={record.total_score:.3f}  {status}", fill=(20, 20, 20), font=load_font(18, True))
        draw.text((x, y + 274), Path(record.path).name[:36], fill=(50, 50, 50), font=load_font(15))
    sheet.save(output, quality=92)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank 5-10 photos and render Editorial, Scrapbook, and Minimal Instagram collages.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+", help="Image files and/or one directory containing images")
    parser.add_argument("-o", "--output", default="collage_output", help="Output directory")
    parser.add_argument(
        "--prompt",
        default="",
        help="Creative brief used for semantic photo ranking, title generation, and color mood",
    )
    parser.add_argument("--title", default="", help="Optional visible title override")
    parser.add_argument("--subtitle", default="", help="Optional visible subtitle override")
    parser.add_argument(
        "--mood",
        choices=["auto", "warm", "cool", "romantic", "bold", "natural", "mono"],
        default="auto",
        help="Color direction; auto infers it from the prompt",
    )
    parser.add_argument("--model", default="openai/clip-vit-base-patch32", help="Hugging Face CLIP model id")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--duplicate-threshold", type=float, default=0.94, help="Cosine similarity cutoff")
    parser.add_argument("--max-selected", type=int, choices=range(5, 9), default=6)
    parser.add_argument("--no-models", action="store_true", help="Offline smoke-test mode: hashes + CV metrics only")
    parser.add_argument("--strict-models", action="store_true", help="Fail instead of falling back if CLIP cannot load")
    parser.add_argument("--seed", type=int, default=2026, help="Controls deterministic scrapbook texture")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()
    try:
        paths = discover_images(args.inputs)
        output_dir = Path(args.output).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        log(f"Analyzing {len(paths)} photos...")
        if args.prompt:
            log(f"Creative prompt: {args.prompt}")
        images, records, model_info = analyze_photos(
            paths,
            args.model,
            args.device,
            args.batch_size,
            args.no_models,
            args.strict_models,
            args.prompt,
        )
        selected_indices = select_photos(records, args.duplicate_threshold, args.max_selected)
        selected_images = [images[i] for i in selected_indices]
        selected_records = [records[i] for i in selected_indices]
        render_info = render_all(
            selected_images,
            selected_records,
            output_dir,
            args.seed,
            args.prompt,
            args.title,
            args.subtitle,
            args.mood,
        )
        save_debug_contact_sheet(images, records, output_dir / "debug_ranking.jpg")
        metadata = {
            "version": "0.2.0",
            "created_unix": int(time.time()),
            "runtime_seconds": round(time.time() - started, 3),
            "configuration": vars(args),
            "model": model_info,
            "selection": {
                "hero": selected_records[0].path,
                "supporting": [record.path for record in selected_records[1:]],
                "selected_indices": selected_indices,
            },
            "photos": [record.public_dict() for record in sorted(records, key=lambda r: -r.total_score)],
            **render_info,
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        log(f"Done. Hero: {Path(selected_records[0].path).name}")
        log(f"Metadata: {output_dir / 'metadata.json'}")
        return 0
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
