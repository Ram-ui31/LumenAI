# Benchmarks

These tables are the ones in the technical report ([LumenAI_Technical_Report.pdf](LumenAI_Technical_Report.pdf)),
which describes every metric and test set. Times are warm times at 1024 px, measured after a warm-up run in October
2026. "Peak VRAM" is the peak GPU memory allocated while the model ran. Every model LumenAI ships stays inside a
**16 GB** budget.

## Speed and memory of the shipped models

| Feature | Model | Peak VRAM | Time |
|---|---|---|---|
| Create an image | Lumen 1.1 Lumina (FLUX.2 klein 4B, 4-bit encoder) | 13.4 GB | 1.1 s |
| Edit with words | Lumen 1.1 Lumina | 13.4 GB | 1.7 s |
| Edit with words | Lumen 1.3 Lux | 13.8 GB | 18 s |
| Product shot | Qwen-Image 2.1 | 14.2 GB | 23.5 s |
| AI upscale ×4 | TADSR | 5.7 GB | < 0.5 s |
| View in 3D (default) | SPAR3D \* | 11.0 GB | 25 s |
| View in 3D (fast option) | Stable Fast 3D \* | 11.0 GB | 20 s |
| View in 3D | Hunyuan3D-2 | 11.5 GB | 54 s |
| Safety check | Qwen3-VL (shared) + DeBERTa-v3 | +0.8 GB | 2.3 s |
| Invisible watermark | DWT-DCT-SVD | none | 0.08 s |

\* SPAR3D and SF3D were measured separately on the same photo (first request, including background removal):
SPAR3D 7,584 faces and a 0.4 MB GLB, SF3D 17,612 faces and 0.7 MB. IC-Light was added after this run.

## The three model tiers

| | Lumen 1.1 Lumina | Lumen 1.2 Kroma | Lumen 1.3 Lux |
|---|---|---|---|
| Diffusion model | FLUX.2 klein 4B, step-distilled rectified-flow transformer | Qwen-Image 2.1, MMDiT transformer, 4-bit | Qwen-Image 2.1, MMDiT transformer, 4-bit |
| Denoising steps | 4 | 16 | 40 |
| Prompt | as typed | as typed | rewritten by Qwen3-VL while looking at the photo |
| Runs in | its own container | the main API container | the main API container |
| Peak VRAM | 13.35 GB | 12.5 GB (image), 13.8 GB (edit) | 12.5 GB (image), 13.8 GB (edit) |
| Model time, new image | 1.1 s | about 0.4× Lux (estimated from 16 of 40 steps) | 20.6 s |
| Model time, edit | 1.7 s | about 0.4× Lux (estimated) | 18 s |
| Full wait in the editor (2026-10-07) | 40 s when its container was asleep; a few seconds when warm | 20 s | 30 s |
| Text to image (judge, 0 to 10) | prompt 9.7, quality 9.7 | not benchmarked | prompt 9.7, quality 9.8 |
| Instruction edits (GEdit, overall) | **7.80** | 7.36 (Qwen-Image) | 7.36 (Qwen-Image) |
| Keeps logos and text | weaker (DINOv2 0.815) | strong | strong (DINOv2 0.900) |
| Best for | drafts, quick ideas, many variations | everyday edits and images | final images, long or detailed requests |

## Ways to load Lumen 1.1 Lumina

3 prompts and one edit at 1024 px.

| Mode | Peak VRAM | s / image | Edit | Note |
|---|---|---|---|---|
| bf16, fully resident | 18.6 GB | 1.11 | 1.73 | over budget |
| **4-bit text encoder (shipped)** | **13.35 GB** | **1.13** | **1.72** | same quality by eye |
| model offload | 9.5 GB | 7.55 | 7.42 | 7× slower |

## Quality against the alternatives we evaluated

| Task (data) | Metric | LumenAI | Alternatives |
|---|---|---|---|
| Object removal (RORD) | LPIPS in the hole ↓ | **0.108** | LaMa 0.151, MAT 0.179, Segformer+LaMa 0.149 |
| Pose transfer (10 pairs) | PCK@0.2 ↑ / same person | **0.59 / 10 of 10** | Leffa 0.36 / 2 of 10 |
| Upscale ×4 (16 photos) | LPIPS ↓ | **0.217** (TADSR) | Lanczos 0.303, Qwen refine 0.290 |
| Product shot (24 cases) | DINOv2 product similarity ↑ | **0.900** | FLUX.2 klein 0.815 |
| Text to image (20 prompts) | judge: prompt / quality ↑ | 9.7 / 9.8 (Lux) | klein 9.7 / 9.7, Z-Image 9.6 / 9.6 |
| Instruction edits (GEdit, 33) | overall ↑ | 7.36 (Qwen), **7.80** (Lumina) | InstructPix2Pix 2.91 |
| Style transfer (48 pairs) | content kept (DINOv2) ↑ / style (Gram, log) ↓ | **0.81** / 1.25 (klein) | Magenta (TensorFlow) 0.44 / **0.87** |
| Recolour (48 tasks) | leak ΔE00 ↓ / shading ↑ | **0.00 / 0.997** | Qwen edit 3.69 / 0.735 |
| Motion deblur (16 photos) | LPIPS ↓ | **0.142** | blurred 0.248, Wiener 0.331 |
| Watermark after JPEG 75 | bit accuracy ↑ | **0.99** (DWT-DCT-SVD) | DWT-DCT 0.53 |
| Safety (60 requests) | harmful / injections blocked, false blocks | 19/20, 10/10, 0 | SingGuard-4B 19/20, 8/10, 0 (12.6 s, 9.1 GB) |

## Style transfer

48 photo and artwork pairs (mean, standard deviation in brackets). Content scores compare the result with the photo,
style scores with the artwork. Times are warm.

| Method | DINOv2 content ↑ | CLIP content ↑ | CLIP style ↑ | Gram loss (log₁₀) ↓ | Time |
|---|---|---|---|---|---|
| Magenta (TensorFlow Hub) | 0.44 (0.20) | 0.64 (0.07) | **0.62** (0.07) | **0.87** (0.44) | **0.04 s** |
| FLUX.2 klein, raw | 0.80 (0.13) | 0.70 (0.08) | 0.60 (0.09) | 1.39 (0.43) | 3.1 s |
| FLUX.2 klein, 85 % blend | **0.81** (0.12) | **0.71** (0.08) | 0.60 (0.09) | 1.25 (0.42) | 3.1 s |

Peak VRAM: Magenta 1.3 GB, klein 13.35 GB (its container). Klein's time includes the call between containers. Klein
(85 %) beats Magenta on DINOv2 content in 46 of 48 pairs and on CLIP content in 39; Magenta wins on CLIP style in 33
and on Gram loss in 39. The editor's Style tool uses Magenta.

## Recolour

48 tasks (30 generated product photos, 18 people from RORD). Leak is the CIEDE2000 change on protected pixels of the
object and of the background; coverage is the share of the requested part that changed colour.

| Method | Leak object ↓ | Leak background ↓ | Hue error ↓ | Coverage ↑ | Shading corr. ↑ | Time |
|---|---|---|---|---|---|---|
| **Ours (recolour)** | **0.00** | **0.00** | **0.6°** | **0.99** | **0.997** | 3 s |
| Hybrid (Qwen decides where) | 1.04 | 0.00 | 3.5° | 0.82 | 0.870 | 21 s |
| Qwen-Image 2.1 edit | 3.69 | 2.03 | 7.7° | 0.83 | 0.735 | 21 s |
| InstructPix2Pix | 20.10 | 12.76 | 12.0° | 0.89 | 0.822 | 1 s |

## Reproduce

Run from `backend/` after the one-time setup in the main README:

```bash
modal run --detach modal_benchmark.py
modal run --detach modal_flash.py
modal run modal_benchmark_style.py
```

The style comparison needs the Lumina app deployed (`modal deploy modal_flash.py`) and saves its results to
`./outputs/style_bench/`, so it runs without `--detach`.

`modal_benchmark.py` writes to `outputs/eval/v9` on the `adobe-ps-data` volume and `modal_flash.py` to
`outputs/flash_test`. Download them with:

```bash
modal volume get --force adobe-ps-data outputs/eval/v9 ./results
modal volume get --force adobe-ps-data outputs/flash_test ./results
```
