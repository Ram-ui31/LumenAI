# LumenAI

**An AI photo editor and image generator that does what you type.** Describe a change ("make the backpack navy blue",
"remove the person on the left") or a whole new picture, and LumenAI picks the right model for the job. Every model
runs inside a 16 GB GPU budget.

- **Edit with words**, or select by clicking, dragging a box or painting with a brush
- **Remove, move, recolour, expand, add objects**, change a person's **pose**, **relight**, **restyle**
- **Create images** with three models: Lumen 1.1 – Lumina (≈2 s), 1.2 – Kroma (≈12 s), 1.3 – Lux (best detail)
- **Portrait bokeh**, **dolly video**, **product shots**, **AI upscale ×4**, **view any object in 3D**, collage
- **Trends**: today's popular looks (Gemini + Google Search) as one-click edits
- Request guard against harmful prompts and prompt injection; invisible watermark on every export

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
│   └── assets/
├── backend/                   Python 3.11 API (FastAPI) and the Modal apps
│   ├── server/                web API: routes, tools, guard, trends, watermark, local runner
│   ├── editing/               editing pipelines (Qwen wrapper, removal, recolour, pose, relight, 3-D, ...)
│   ├── core/  generation/  critics/  vlm/   shared helpers, generation backends, quality critics, VLM client
│   ├── benchmarks/            benchmark runners and metrics
│   ├── tests/                 offline test suite (no GPU, no downloads)
│   ├── scripts/demo_server.py the whole API with a stand-in engine (try the UI without a GPU)
│   ├── modal_api.py           main API app             modal_flash.py    Lumen 1.1 – Lumina (FLUX.2 klein)
│   ├── modal_tadsr.py         AI upscaler              modal_object3d.py Hunyuan3D-2 (View in 3D)
│   ├── modal_benchmark*.py    benchmark runs on Modal
│   ├── config.yaml            defaults
│   └── requirements*.txt
└── docs/                      architecture and benchmarks
```

---

## Quick start

### 1. Try the editor without a GPU (2 minutes)

The demo server runs the real API routes with a stand-in engine, so every tool answers instantly with a placeholder
result. Good for UI work and a first look.

```bash
git clone https://github.com/Ram-ui31/LumenAI.git
cd LumenAI/backend
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

Optional, for the Trends tool (a key from Google AI Studio):

```bash
read -s GEMINI_API_KEY
modal secret create adobe-ps-trends GEMINI_API_KEY="$GEMINI_API_KEY"
```

**Deploy**

```bash
modal deploy modal_api.py
modal deploy modal_flash.py
modal deploy modal_tadsr.py
modal deploy modal_object3d.py
```

`modal_api.py` is required; the other three add Lumen 1.1 – Lumina, the AI upscaler and View in 3D (the editor tells
you if one is missing). The first deploy builds the container images (5–20 minutes, `modal_object3d.py` is the
longest); later deploys take seconds.

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

The suite runs offline with stand-in models (about 130 tests, a minute or two).

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
| GPU type on Modal | `MODAL_GPU` when deploying | `L40S` |
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
| 3-D | [Hunyuan3D-2](https://huggingface.co/tencent/Hunyuan3D-2) | Tencent Hunyuan community licence |
| Prompt-injection guard | deberta-v3-base-prompt-injection-v2 | Apache-2.0 |
| Expand | Qwen-Image Outpaint LoRA (community) | see model card |
| Trends | Google Gemini API | Google API terms |

Check each model card before any commercial use.
