"""Run the LumenAI API on your own CUDA machine (instead of Modal).

    cd backend
    python -m server.serve --port 8000          # then set frontend/config.js to http://127.0.0.1:8000/api/v1

Needs an NVIDIA GPU with 16 GB or more and the Hugging Face models (downloaded on first start, ~30 GB). Lumen 1.1 -
Lumina, the AI upscaler and View in 3D run in their own Modal apps; without them those tools answer with a short
"not deployed" message and everything else works.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("ADOBEPS_DEVICE", "cuda")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()

    import uvicorn

    from server.web_api import Engine, build_app

    engine = Engine()
    engine.load()
    print(f"[serve] models loaded in {engine.loaded_at}s - API at http://{a.host}:{a.port}/api/v1", flush=True)
    uvicorn.run(build_app(engine), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
