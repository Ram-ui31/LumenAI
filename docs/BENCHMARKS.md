# Benchmarks

All numbers were measured on one NVIDIA L40S, after a warm-up run, in October 2026. "Peak GB" is the peak GPU memory
allocated while the model ran (PyTorch allocator). Every model LumenAI ships stays inside a **16 GB** budget.

Reproduce: `cd backend && modal run --detach modal_benchmark.py` (logic in `benchmarks/run_v9.py`; third-party models in
`modal_benchmark_extra.py`). The report is written to `outputs/eval/v9/summary.md` on the Modal volume.

## Speed and memory of the shipped models

| Feature | Model | Peak VRAM | Time |
|---|---|---|---|
| Create an image (1024 px) | Lumen 1.1 – Lumina (FLUX.2 klein 4B, 4-bit text encoder) | 13.4 GB | 1.1 s |
| Create an image (1024 px) | Lumen 1.3 – Lux (Qwen-Image 2.1, 40 steps) | 12.5 GB | 20.6 s |
| Edit with words | Lumen 1.1 – Lumina | 13.4 GB | 1.7 s |
| Edit with words | Lumen 1.3 – Lux | 13.8 GB | 35 s |
| Product shot | Qwen-Image 2.1 | 14.2 GB | 23.5 s |
| AI upscale ×4 | TADSR | 5.7 GB | < 0.5 s |
| View in 3D | Hunyuan3D-2 | 11.5 GB | 54 s |
| Safety check | Qwen3-VL (shared) + DeBERTa-v3 injection classifier | +0.8 GB | 2.3 s |
| Invisible watermark | dwtDctSvd | CPU | 0.08 s |

Lumen 1.1 – Lumina loading options (`modal run modal_flash.py`):

| Mode | Peak VRAM | s / image | Edit | Note |
|---|---|---|---|---|
| bf16, all on GPU | 18.6 GB | 1.11 | 1.73 | over budget |
| **4-bit text encoder (shipped)** | **13.35 GB** | **1.13** | **1.72** | same quality by eye |
| model CPU offload | 9.5 GB | 7.55 | 7.42 | 7× slower |

## Quality against the alternatives we tested

| Task | Metric | LumenAI | Alternatives |
|---|---|---|---|
| Object removal (RORD) | LPIPS in the hole ↓ | **0.108** | LaMa 0.151 · MAT 0.179 · Segformer + LaMa 0.149 |
| Pose transfer (10 pairs) | PCK@0.2 ↑ / same person | **0.59 / 10 of 10** | Leffa 0.36 / 2 of 10 |
| Upscale ×4 (16 photos) | LPIPS ↓ | **0.217** (TADSR) | Lanczos 0.303 · Qwen refine 0.290 |
| Product shot (24 cases) | DINOv2 product similarity ↑ | **0.900** | FLUX.2 klein 0.815 |
| Text to image (20 prompts) | judge score: prompt / quality | 9.7 / 9.8 (Lux) | klein 9.7 / 9.7 · Z-Image 9.6 / 9.6 |
| Instruction edits (GEdit-Bench, 33) | overall ↑ | 7.36 (Qwen) · **7.80 (Lumina)** | InstructPix2Pix 2.91 |
| Watermark after JPEG-75 | bit accuracy ↑ | **0.99** (dwtDctSvd) | dwtDct 0.53 |
| Safety (60 requests) | harmful blocked / injections blocked / false blocks | 19 of 20 / 10 of 10 / 0 | SingGuard-4B 19/20, 8/10, 0 at 12.6 s and 9.1 GB |

Not adopted: Qwen-Image-Edit-2511 + Multiple-Angles LoRA (best camera-angle scores, but 42.5 GB and 140 s per image),
Z-Image-Turbo (slower than klein for the same quality), FLUX.1-dev and SF3D (gated).
