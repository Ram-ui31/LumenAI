# Architecture

```
 Browser (frontend/)                         Modal (backend/)
 ┌──────────────────────┐   JSON + images   ┌────────────────────────────────────────────────────────────┐
 │ index.html  landing  │   as data URLs    │ adobe-ps-api  (modal_api.py, one GPU)                      │
 │ editor.html editor   │ ───────────────▶  │   FastAPI  server/web_api.py + server/extras.py            │
 │  · tools, history    │                   │   request guard (server/guard.py) → router → tool          │
 │  · selection, brush  │ ◀───────────────  │   Engine: Qwen-Image 2.1 (nf4) + Qwen3-VL + SAM 2.1        │
 │  · dolly renderer    │   result + time   │           Depth Anything V2 Small · OpenPose · watermark   │
 │    (three.js)        │   + peak VRAM     │                                                            │
 └──────────────────────┘                   │   calls ──▶ adobe-ps-flash    FLUX.2 klein 4B (Lumina)     │
                                            │         ──▶ adobe-ps-tadsr    TADSR upscaler               │
                                            │         ──▶ adobe-ps-object3d Hunyuan3D-2                  │
                                            │   Volume adobe-ps-data: model weights + run outputs        │
                                            └────────────────────────────────────────────────────────────┘
```

## Why several containers

The 16 GB budget is per model. Qwen-Image 2.1 with its Qwen3-VL text encoder already peaks at about 14 GB, so models
with their own large weights (FLUX.2 klein, Hunyuan3D-2) or incompatible dependencies (TADSR pins torch 2.0) each run in
their own Modal app and are called with `modal.Cls.from_name(...)`. Idle containers scale to zero.

## One engine, many tools

`server/web_api.py: Engine` holds the shared pipeline. Qwen-Image 2.1 does generation, instruction edits, outpainting
(with the Outpaint LoRA) and object insertion; its text encoder is a full Qwen3-VL, which is reused as the
vision-language model for grounding ("where is the backpack?"), ideas, pose descriptions, prompt rewriting and the
safety policy check. That reuse is why the guard costs no extra GPU memory.

| Tool (endpoint) | Pipeline |
|---|---|
| Edit with words (`/edit`) | guard → router: remove / recolour / pose / style / deblur / general edit (masked edits crop around the selection and blend back with a Laplacian pyramid) |
| Generate (`/generate`) | Lumina: FLUX.2 klein · Kroma: Qwen 16 steps · Lux: Qwen3-VL rewrites the prompt, Qwen 40 steps |
| Select (`/segment`) | SAM 2.1 from clicks or a box (nothing outside the box is kept) |
| Remove | Qwen3-VL reads the selection → draft at 512 px → sharpen at 768 px → self-check with up to 3 fallbacks → shadow removal → seam blend |
| Move (`/move`) | lift the object, fill the hole with Remove, put it down at the new place and size |
| Recolour (`/recolour`, `/recolour-shade`) | OKLab recolour that keeps the shading; the slider re-renders from a cached mask |
| Expand (`/expand`) | Qwen + Outpaint LoRA, 10 steps |
| Add object (`/ideas`, `/place`, `/add`) | Qwen3-VL ideas → spot and realistic size → "ghost" fill |
| Pose (`/pose`) | OpenPose joints + Qwen3-VL description → words-only edit with up to 2 correction rounds (Qwen never sees the reference photo, so it cannot copy it) |
| Style (`/style`) | reference style described in words, or a preset |
| Relight (`/relight`) | Qwen relight, then light transfer: new low-frequency light, original high-frequency detail |
| Bokeh (`/bokeh`, `/bokeh-adjust`) | Depth Anything V2 Small + subject mask → layered disc blur in linear light; depth is cached so the slider and refocus run on the CPU |
| Dolly video (`/layers`) | subject cut-out + filled background plate + depth; the browser renders the camera move with three.js and records it |
| Product shot (`/product`) | Qwen-Image 2.1 (keeps logos; klein redraws them) |
| Upscale (`/upscale-ai`) | TADSR ×4 (×2 = ×4 then resize) |
| View in 3D (`/view3d`) | cut-out → Qwen completes hidden parts → Hunyuan3D-2 textured mesh (GLB) |
| Trends (`/trends`) | Gemini with Google Search grounding → ready-to-run edit prompts |
| Export (`/export`) | invisible dwtDctSvd watermark ("LUMN"), checked by `/verify` |

## Seeds and repeatability

A request's seed is mixed with its input (`mix_seed`: the image for edits, the prompt for generation). Same seed +
same input gives the same result, but an edit never starts from the noise that created the image it edits (that made
the model reproduce the original instead of applying the edit).

## Memory and speed

- Qwen-Image 2.1 transformer and text encoder in 4-bit nf4; the text encoder lives on the CPU and moves to the GPU only
  to encode a prompt or answer a question (peak 20.7 → 14.0 GB, output identical).
- Text-encoder outputs are cached for exact repeats ("Another version", retries).
- TF32 matmuls; VAE tiling stays off (it changed what the model saw and hurt removals).
- Klein's 8 GB text encoder is loaded in 4-bit: 18.6 → 13.35 GB at the same speed.
- Every response reports `seconds` and `peak_gb`; the editor shows both in its "Last run" card.
