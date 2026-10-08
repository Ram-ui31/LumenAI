# LumenAI

**An AI photo editor and image generator that does what you type.** Describe a change ("make the backpack navy blue",
"remove the person on the left") or a whole new picture, and LumenAI picks the right model for the job. Every model
runs inside a 16 GB GPU budget.

- **Edit with words**, or select by clicking, dragging a box or painting with a brush
- **Remove, move, recolour, expand** (preset or any ratio), add objects by prompt, change a person's **pose**, **relight**, **restyle**
- **Create images** with three models: Lumen 1.1 – Lumina (FLUX.2 klein), 1.2 – Kroma (Qwen-Image), 1.3 – Lux (Qwen-Image with prompt rewriting, best detail)
- **Portrait bokeh**, **glow light** (a depth-aware light that reveals dark scenes; a live preview follows the cursor), **dolly video**
- **Product shots** from **4 angles** (FLUX.2 klein), **AI upscale ×4**, **view any object in 3D** (SPAR3D or SF3D), put a new angle of it back into the photo and rotate or resize it
- **Styles** from 27 classic artworks or your own reference (Magenta, TensorFlow Hub); **IC-Light** relighting
- **Up to 10 reference photos** per edit, brush selection that snaps to the object (SAM), collage and **AI auto-collage**
- **Suggestions**: the AI looks at the objects in your photo and proposes what to remove, add, replace or move, one tap to try
- Request guard against harmful prompts and prompt injection; invisible watermark on every export

**Technical report:** [docs/LumenAI_Technical_Report.pdf](docs/LumenAI_Technical_Report.pdf) (23 pages: architecture,
routing, every pipeline with its mathematics, the recolour slider, benchmarks and limitations).
Measured speed, memory and quality: [docs/BENCHMARKS.md](docs/BENCHMARKS.md). How it fits together:
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Repository layout

```
LumenAI/
├── frontend/                  static web app (no build step)
│   ├── index.html             landing page
│   ├── editor.html            the editor
│   ├── config.js              ← the API URL the editor talks to
│   └── assets/                images; artworks.zip = the 27 style thumbnails (unzip in place)
├── backend/                   Python 3.11 API (FastAPI) and the Modal apps
│   ├── server/                web API: routes, tools, guard, suggestions, watermark, local runner
│   ├── editing/               editing pipelines (Qwen wrapper, removal, recolour, pose, relight, 3-D, ...)
│   ├── core/  generation/  critics/  vlm/   shared helpers, generation backends, quality critics, VLM client
│   ├── benchmarks/            benchmark runners and metrics
│   ├── tests/                 offline test suite (no GPU, no downloads)
│   ├── scripts/demo_server.py the whole API with a stand-in engine (try the UI without a GPU)
│   ├── modal_api.py           main API app             modal_flash.py    Lumen 1.1 – Lumina (FLUX.2 klein)
│   ├── modal_tadsr.py         AI upscaler              modal_object3d.py SPAR3D + SF3D (View in 3D)
│   ├── modal_iclight.py       IC-Light relighting      modal_magenta.py  Magenta styles (TensorFlow)
│   ├── modal_benchmark*.py    benchmark runs on Modal
│   ├── config.yaml            defaults
│   └── requirements*.txt
├── docs/                      ARCHITECTURE.md, BENCHMARKS.md, LumenAI_Technical_Report.pdf
└── vercel.json                publishes frontend/ as a static site
```

---

## Quick start

### 1. Try the editor without a GPU (2 minutes)

The demo server runs the real API routes with a stand-in engine, so every tool answers instantly with a placeholder
result. Good for UI work and a first look.

The 27 artwork thumbnails used by the Style tool ship as a zip; unpack them once after cloning:

```bash
git clone https://github.com/Ram-ui31/LumenAI.git
cd LumenAI/frontend/assets && unzip artworks.zip && cd ../..
cd backend
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python scripts/demo_server.py
```

In a second terminal:

```bash
cd LumenAI/frontend
python3 -m http.server 3001
```

Open http://localhost:3001/editor.html?api=http://127.0.0.1:8765/api/v1

### 2. Run it for real on Modal (recommended)

You need a [Modal](https://modal.com) account (GPU containers that scale to zero) and a
[Hugging Face](https://huggingface.co) account with an access token (Settings → Access Tokens, "read").

**One-time setup**

```bash
cd LumenAI/backend
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
modal setup
```

Store your keys as Modal secrets. `read -s` keeps them off the screen and out of your shell history:

```bash
read -s HF_TOKEN
modal secret create adobe-ps-hf HF_TOKEN="$HF_TOKEN"
```

**Deploy**

```bash
modal deploy modal_api.py
modal deploy modal_flash.py
modal deploy modal_tadsr.py
modal deploy modal_object3d.py
modal deploy modal_iclight.py
modal deploy modal_magenta.py
```

`modal_api.py` is required; the others add Lumen 1.1 – Lumina (also product angles), the AI upscaler, View in 3D, IC-Light relighting and
Magenta styles (the editor tells you if one is missing). The first deploy builds the container images (5–20 minutes); later deploys take seconds.

**Point the editor at your API.** `modal deploy modal_api.py` prints a URL like
`https://<workspace>--adobe-ps-api-studio-web.modal.run`. Put it, followed by `/api/v1`, in `frontend/config.js`:

```js
window.LUMEN_API = "https://<workspace>--adobe-ps-api-studio-web.modal.run/api/v1";
```

**Open it**

```bash
cd ../frontend
python3 -m http.server 3001
```

Open http://localhost:3001. The first request wakes the GPU and loads the models (1–3 minutes; the editor shows
"Waking the AI"). After 10 idle minutes the GPU shuts down again, so an idle deployment costs nothing.

**Put the website online (Vercel).** Only `frontend/` is hosted; all Python and AI stay on Modal. In Vercel, import
this repository and keep the defaults: `vercel.json` publishes `frontend/`, unzips the artwork thumbnails during the
build and skips `backend/`. `frontend/config.js` must hold your Modal API URL before you deploy.

### 3. Run the API on your own GPU

On a Linux machine with an NVIDIA GPU (16 GB or more):

```bash
cd LumenAI/backend
source .venv/bin/activate
hf auth login
python -m server.serve --port 8000
```

Set `window.LUMEN_API = "http://127.0.0.1:8000/api/v1"` in `frontend/config.js`. Models download on the first start
(about 30 GB). Lumina, the upscaler and View in 3D still run on Modal; without those apps the three tools say so and
everything else works.

---

## Tests

```bash
cd LumenAI/backend
python -m pytest -q
```

The suite runs offline with stand-in models (135 tests, under a minute).

## Benchmarks

```bash
cd LumenAI/backend
modal run --detach modal_benchmark.py
modal run --detach modal_flash.py
```

Results land on the Modal volume (`modal volume get adobe-ps-data outputs/eval/v9 ./results`). The latest numbers are
in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Configuration

| Setting | Where | Default |
|---|---|---|
| API URL used by the editor | `frontend/config.js` (or `?api=` in the address) | — |
| Diffusion steps (Lux / Kroma) | `ADOBEPS_API_STEPS` / `ADOBEPS_QUICK_STEPS` | 40 / 16 |
| Largest side processed | `ADOBEPS_API_MAX_SIDE` | 2048 |
| Lumina loading mode | `FLASH_MODE` when deploying `modal_flash.py`: `nf4te` or `offload` | `nf4te` |
| GPU type on Modal | `MODAL_GPU` (API, Lumina, upscaler), `MODAL_GPU_3D`, `MODAL_GPU_SPAR3D`, `STUDIO_GPU` (IC-Light), `MODAL_GPU_MAGENTA` | H100 for the API and Lumina, L40S for 3D, IC-Light and the upscaler, L4 for Magenta |
| Watermark on/off | `ADOBEPS_WATERMARK` | on |
| No models (tests, demo) | `ADOBEPS_BACKEND=stub` | — |

## Models and licences

| Used for | Model | Licence |
|---|---|---|
| Edits, generation, outpainting | [Qwen-Image 2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) (+ its Qwen3-VL text encoder) | Apache-2.0 |
| Lumen 1.1 – Lumina | [FLUX.2 klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | Apache-2.0 |
| Selection | SAM 2.1 | Apache-2.0 |
| Depth (bokeh, dolly) | Depth Anything V2 Small | Apache-2.0 |
| Pose joints | OpenPose (controlnet-aux annotators) | see model card |
| Upscaling | [TADSR](https://huggingface.co/zty557/TADSR) | academic use only |
| 3-D | [SPAR3D](https://huggingface.co/stabilityai/stable-point-aware-3d) (default) and [Stable Fast 3D](https://huggingface.co/stabilityai/stable-fast-3d) (both gated: accept the licences with your HF account) | Stability AI Community licence |
| Fast style | Magenta arbitrary image stylisation (TensorFlow Hub) | Apache-2.0 |
| Prompt-injection guard | deberta-v3-base-prompt-injection-v2 | Apache-2.0 |
| Expand | Qwen-Image Outpaint LoRA (community) | see model card |

Check each model card before any commercial use.
