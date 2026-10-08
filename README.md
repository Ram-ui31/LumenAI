# LumenAI

**An AI photo editor and image generator that does what you type.** Describe a change ("make the backpack navy blue",
"remove the person on the left") or a whole new picture, and LumenAI picks the right model for the job. Every model
runs inside a 16 GB GPU memory budget.

- **Edit with words**, or select by clicking, dragging a box or painting with a brush
- **Remove, move, recolour, expand** (preset or any ratio), change a person's **pose**, **relight**, **restyle**
- **Create images** with three models: Lumen 1.1 Lumina (FLUX.2 klein), 1.2 Kroma (Qwen-Image), 1.3 Lux (Qwen-Image with prompt rewriting, best detail)
- **Portrait bokeh**, **glow light** (a depth-aware light that reveals dark scenes; a live preview follows the cursor), **dolly video**
- **Product shots** from **4 angles** (FLUX.2 klein), **AI upscale ×4**, **view any object in 3D** (SPAR3D or SF3D), put a new angle of it back into the photo and rotate or resize it
- **Styles** from 27 classic artworks or your own reference (Magenta, TensorFlow Hub); **IC-Light** relighting
- **Up to 10 reference photos** per edit, brush selection that snaps to the object (SAM), collage and **AI auto-collage**
- **Suggestions**: the AI looks at the objects in your photo and proposes what to remove, add, replace or move, one tap to try
- Request guard against harmful prompts and prompt injection; invisible watermark on every export

**Technical report:** [docs/LumenAI_Technical_Report.pdf](docs/LumenAI_Technical_Report.pdf) (architecture,
routing, every pipeline with its mathematics, the recolour slider, benchmarks and limitations).
Speed, memory and quality tables: [docs/BENCHMARKS.md](docs/BENCHMARKS.md). How it fits together:
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Repository layout

```
LumenAI/
├── frontend/                  static web app (no build step)
│   ├── index.html             landing page
│   ├── editor.html            the editor
│   ├── config.js              the API URL the editor talks to
│   └── assets/                images; artworks.zip = the 27 style thumbnails (unzip in place)
├── backend/                   Python API (FastAPI) and the Modal apps
│   ├── server/                web API: routes, tools, guard, suggestions, watermark, local runner
│   ├── editing/               editing pipelines (Qwen wrapper, removal, recolour, pose, relight, 3-D, ...)
│   ├── core/  generation/  critics/  vlm/   shared helpers, generation backends, quality critics, VLM client
│   ├── benchmarks/            benchmark runners and metrics
│   ├── tests/                 offline test suite (no GPU, no downloads)
│   ├── scripts/demo_server.py the whole API with a stand-in engine (try the UI without a GPU)
│   ├── modal_api.py           main API app             modal_flash.py    Lumen 1.1 Lumina (FLUX.2 klein)
│   ├── modal_tadsr.py         AI upscaler              modal_object3d.py SPAR3D + SF3D (View in 3D)
│   ├── modal_iclight.py       IC-Light relighting      modal_magenta.py  Magenta styles (TensorFlow)
│   ├── modal_benchmark*.py    benchmark runs on Modal
│   ├── config.yaml            defaults
│   └── requirements*.txt
├── tools/unzip-artworks.mjs   unpacks artworks.zip (used by the Vercel build; works anywhere with Node)
├── docs/                      ARCHITECTURE.md, BENCHMARKS.md, LumenAI_Technical_Report.pdf
└── vercel.json                publishes frontend/ as a static site
```

---

## Quick start

You need Python 3.10 or newer and Git. All commands start from the folder you cloned into.

### 1. Get the code and the artwork thumbnails

```bash
git clone https://github.com/Ram-ui31/LumenAI.git
cd LumenAI
node tools/unzip-artworks.mjs
```

The last line unpacks the 27 Style thumbnails to `frontend/assets/artworks/`. Without Node, run
`cd frontend/assets && unzip artworks.zip && cd ../..` instead.

### 2. Try the editor without a GPU

The demo server runs the real API routes with a stand-in engine, so every tool answers at once with a placeholder
result. Good for UI work and a first look. The install pulls in PyTorch, so it takes a few minutes the first time.

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python scripts/demo_server.py
```

In a second terminal, from the `LumenAI` folder:

```bash
cd frontend
python3 -m http.server 3001
```

Open http://localhost:3001/editor.html?api=http://127.0.0.1:8765/api/v1

On Windows, use `.venv\Scripts\activate` instead of `source .venv/bin/activate`.

### 3. Run it for real on Modal (recommended)

You need a [Modal](https://modal.com) account and a [Hugging Face](https://huggingface.co) account with a "read"
access token (Settings, Access Tokens). With the same Hugging Face account, accept the licences of the two gated 3-D
models: [SPAR3D](https://huggingface.co/stabilityai/stable-point-aware-3d) and
[Stable Fast 3D](https://huggingface.co/stabilityai/stable-fast-3d).

**One-time setup** (skip the first three lines if you did step 2):

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
modal setup
```

Store the token as a Modal secret. `read -s` keeps it off the screen and out of your shell history; paste the token
and press Enter:

```bash
read -s HF_TOKEN
modal secret create adobe-ps-hf HF_TOKEN="$HF_TOKEN"
```

**Deploy** (from `backend/`). This version lets every app scale to zero, so you pay only while it is in use:

```bash
LUMEN_KEEP_WARM=0 modal deploy modal_api.py
LUMEN_KEEP_WARM=0 modal deploy modal_flash.py
LUMEN_KEEP_WARM=0 modal deploy modal_object3d.py
LUMEN_KEEP_WARM=0 modal deploy modal_iclight.py
LUMEN_KEEP_WARM=0 modal deploy modal_magenta.py
LUMEN_KEEP_WARM=0 modal deploy modal_tadsr.py
```

`modal_api.py` is required; the others add Lumen 1.1 Lumina (also product angles), View in 3D, IC-Light relighting,
Magenta styles and the AI upscaler (the editor tells you if one is missing). The first deploy builds the container
images (5 to 20 minutes); later deploys take seconds.

To remove the wake-up wait, deploy without `LUMEN_KEEP_WARM=0` (for example `modal deploy modal_api.py`). That app
then keeps one GPU container running at all times (two can run for the API), and every running container is billed
by the second, even when nobody uses it. To stop everything:

```bash
for a in adobe-ps-api adobe-ps-flash adobe-ps-object3d lumen-iclight lumen-magenta adobe-ps-tadsr; do modal app stop "$a"; done
```

**Point the editor at your API.** `modal deploy modal_api.py` prints a URL like
`https://<workspace>--adobe-ps-api-studio-web.modal.run`. Put it, followed by `/api/v1`, in `frontend/config.js`:

```js
window.LUMEN_API = "https://<workspace>--adobe-ps-api-studio-web.modal.run/api/v1";
```

**Open it** (from the `LumenAI` folder):

```bash
cd frontend
python3 -m http.server 3001
```

Open http://localhost:3001. With scale-to-zero, the first request after a quiet spell wakes the GPU and loads the
models (1 to 3 minutes; the editor shows "Waking the AI"). After about 10 idle minutes the containers shut down again.

**Put the website online (Vercel).** Only `frontend/` is hosted; all Python and AI stay on Modal. In Vercel, import
this repository and keep the defaults: `vercel.json` publishes `frontend/`, unzips the artwork thumbnails during the
build and skips `backend/`. `frontend/config.js` must hold your Modal API URL before you deploy.

### 4. Run the API on your own GPU

On a Linux machine with an NVIDIA GPU (16 GB or more), after the setup in step 2:

```bash
cd backend
source .venv/bin/activate
hf auth login
python -m server.serve --port 8000
```

Set `window.LUMEN_API = "http://127.0.0.1:8000/api/v1";` in `frontend/config.js`. Models download on the first start
(about 30 GB). Lumina, the upscaler, View in 3D, IC-Light and Magenta styles still run on Modal; without those apps
the tools say so and everything else works.

---

## Tests

```bash
cd backend
source .venv/bin/activate
python -m pytest -q
```

The suite runs offline with stand-in models (no GPU, no downloads).

## Benchmarks

```bash
cd backend
modal run --detach modal_benchmark.py
modal run --detach modal_flash.py
modal volume get --force adobe-ps-data outputs/eval/v9 ./results
```

The last line downloads the results once the run has finished. All tables, and the style comparison, are in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Configuration

Environment variables are read when you deploy (`NAME=value modal deploy ...`) or start the local server.

| Setting | Where | Default |
|---|---|---|
| API URL used by the editor | `frontend/config.js` (or `?api=` in the address) | none |
| Diffusion steps (Lux / Kroma) | `ADOBEPS_API_STEPS` / `ADOBEPS_QUICK_STEPS` | 40 / 16 |
| Largest side processed | `ADOBEPS_API_MAX_SIDE` | 2048 |
| Lumina loading mode | `FLASH_MODE` when deploying `modal_flash.py`: `nf4te` (4-bit text encoder) or `offload` | `nf4te` |
| GPU type on Modal | `MODAL_GPU` (API, Lumina, upscaler), `MODAL_GPU_3D` (SF3D), `MODAL_GPU_SPAR3D`, `STUDIO_GPU` (IC-Light), `MODAL_GPU_MAGENTA` | `H100` |
| Always-warm containers | `LUMEN_KEEP_WARM` when deploying (1 = one container always running, no wake-up wait; 0 = scale to zero when idle) | 1 |
| Watermark on/off | `ADOBEPS_WATERMARK` | on |
| No models (tests, demo) | `ADOBEPS_BACKEND=stub` | off |

## Models and licences

| Used for | Model | Licence |
|---|---|---|
| Generation, instruction edits, removal, outpainting, relight, product shots | [Qwen-Image 2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) | Apache-2.0 |
| Lumen 1.1 Lumina | [FLUX.2 klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | Apache-2.0 |
| Selection | SAM 2.1 | Apache-2.0 |
| Depth (bokeh, dolly, glow) | Depth Anything V2 Small | Apache-2.0 |
| Pose joints | OpenPose (controlnet-aux annotators) | see model card |
| Prompt-injection guard | deberta-v3-base-prompt-injection-v2 | Apache-2.0 |
| Upscaling | [TADSR](https://huggingface.co/zty557/TADSR) | academic use only |
| View in 3D | [SPAR3D](https://huggingface.co/stabilityai/stable-point-aware-3d) (default) and [Stable Fast 3D](https://huggingface.co/stabilityai/stable-fast-3d) (both gated) | Stability AI Community licence |
| Style | Magenta arbitrary image stylisation (TensorFlow Hub) | Apache-2.0 |
| Relighting | IC-Light (SD 1.5, fc) | Apache-2.0 |
| Auto collage | CLIP ViT-B/32 | MIT |
| Expand | Qwen-Image Outpaint LoRA (community) | see model card |

Check each model card before any commercial use.
