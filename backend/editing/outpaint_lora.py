"""Outpainting with the community Outpaint LoRA for Qwen-Image 2.1 (ausboss/Qwen-Image-2.1-Outpaint-LoRA, v2) - the
method in a teammate's backend (m3.py), reproduced on our nf4 pipeline so it can be compared fairly with ours
(server.web_api.outpaint: blurred-edge canvas + keep-outside latent preservation).

  grey canvas (128,128,128) with the photo pasted at its offset -> "Outpaint the image: replace the solid gray areas
  ..." + a Qwen3-VL scene caption -> Qwen edit with the LoRA, true_cfg 1.0, 10 steps (his default) -> paste the
  original back with a 32 px feathered edge.
"""
import numpy as np
from PIL import Image

LORA_REPO = "ausboss/Qwen-Image-2.1-Outpaint-LoRA"
LORA_FILE = "qwen-image-2.1-outpaint-v2.safetensors"
ADAPTER = "outpaint_v2"
INSTRUCTION = ("Outpaint the image: replace the solid gray areas with a seamless continuation of the scene, "
               "keeping the existing picture unchanged.")
CAPTION_Q = ("Describe this picture in one or two plain sentences: the main subject, setting, lighting and visual "
             "style. Do not mention borders or gray areas.")
GRAY = (128, 128, 128)


def comfy_to_diffusers(sd: dict) -> dict:
    """ComfyUI key names -> diffusers (as in m3.py): fused img_mlp.gate_up is split into gate_layer + proj."""
    out = {}
    for key, t in sd.items():
        if not key.startswith("diffusion_model."):
            continue
        name = "transformer." + key[len("diffusion_model."):]
        name = name.replace(".lora_down.weight", ".lora_A.weight").replace(".lora_up.weight", ".lora_B.weight")
        if ".img_mlp.gate_up." not in name:
            out[name] = t
        elif name.endswith("lora_A.weight"):
            out[name.replace(".gate_up.", ".gate_layer.")] = t
            out[name.replace(".gate_up.", ".proj.")] = t.clone()
        else:
            half = t.shape[0] // 2
            out[name.replace(".gate_up.", ".gate_layer.")] = t[:half].contiguous()
            out[name.replace(".gate_up.", ".proj.")] = t[half:].contiguous()
    return out


def load_lora(pipe) -> str:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    if ADAPTER in getattr(pipe, "_adobeps_loras", set()):
        return "already loaded"
    sd = comfy_to_diffusers(load_file(hf_hub_download(LORA_REPO, LORA_FILE)))
    pipe.load_lora_weights(sd, adapter_name=ADAPTER)
    pipe._adobeps_loras = getattr(pipe, "_adobeps_loras", set()) | {ADAPTER}
    return f"loaded {len(sd)} tensors"


def set_lora(pipe, on: bool) -> None:
    if on:
        pipe.set_adapters([ADAPTER], adapter_weights=[1.0])
    else:
        pipe.disable_lora()


def feather_paste(result: Image.Image, pic: Image.Image, box, feather: float = 32.0) -> Image.Image:
    """Paste the original back; alpha ramps up from the edges that border new area (m3.py paste_original)."""
    bx, by, bw, bh = box
    gw, gh = result.size
    yy, xx = np.mgrid[0:bh, 0:bw].astype(np.float32)
    d = np.full((bh, bw), np.inf, np.float32)
    if bx > 0:
        d = np.minimum(d, xx)
    if bx + bw < gw:
        d = np.minimum(d, bw - 1 - xx)
    if by > 0:
        d = np.minimum(d, yy)
    if by + bh < gh:
        d = np.minimum(d, bh - 1 - yy)
    a = np.clip(d / max(1.0, feather), 0, 1)[..., None]
    reg = np.asarray(result.crop((bx, by, bx + bw, by + bh)), np.float32)
    blend = reg * (1 - a) + np.asarray(pic, np.float32) * a
    out = result.copy()
    out.paste(Image.fromarray(blend.round().clip(0, 255).astype(np.uint8)), (bx, by))
    return out


def outpaint(pipe, Q, image: Image.Image, tw: int, th: int, ox: int, oy: int, scene: str = "", steps: int = 10,
             seed: int = 42, resolution: int = 1024) -> tuple[Image.Image, dict]:
    import torch

    w, h = image.size
    canvas = Image.new("RGB", (tw, th), GRAY)
    canvas.paste(image.convert("RGB"), (ox, oy))
    if not scene.strip():
        scene = Q.ask(pipe, [image], CAPTION_Q, max_new_tokens=80)
    prompt = INSTRUCTION + f" Scene: {scene.strip()}"
    set_lora(pipe, True)
    try:
        out = pipe(prompt=prompt, image=[canvas], num_inference_steps=steps, true_cfg_scale=1.0,
                   output_resolution=resolution, generator=torch.Generator("cpu").manual_seed(seed)).images[0]
    finally:
        set_lora(pipe, False)
    if out.mode == "RGBA":   # Qwen-Image 2.1 decodes RGBA; flatten instead of a plain convert (garbage RGB under alpha)
        bg = Image.new("RGBA", out.size, (255, 255, 255, 255))
        out = Image.alpha_composite(bg, out)
    out = out.convert("RGB").resize((tw, th), Image.LANCZOS)
    return feather_paste(out, image.convert("RGB"), (ox, oy, w, h)), {"scene": scene, "steps": steps}
