#!/usr/bin/env python
"""
all_image_gen.py  – unified generator (alias‑fixed)
=================================================
• Generates **10 JPEGs** (0‑9.jpg) for every *prompt × model* pair.
• Supports:  gpt-image-1 · dall-e-2/3 · minDALL‑E · FLUX.1‑dev ·
             stable‑diffusion‑1.4/1.5/2.1 · SD‑3‑medium · SD‑3.5‑large(+turbo) · SD‑XL.
• Stores to:
    /local1/bryanzhou008/Dialect/multimodal-dialectal-bias/plotting/{prompt_raw}/{model}/

Usage (globals or CLI)
----------------------
```python
MODELS  = ["dalle3", "stable-diffusion1.5", "gpt-image-1"]
PROMPTS = [
    "a woman buying brinjal in a market",
    "veranda bathed in sunset light"
]

# or override:
python all_image_gen.py \
    --models dalle3 gpt-image-1 \
    --prompts "a woman buying brinjal in a market"
```
The script tries **very hard not to alter your prompt text**.  It only replaces
`/` with the Unicode *full‑width* slash `／` so directories are legal on Linux.

Changes in this version
-----------------------
* Added an **ALIAS_MAP** so any of these spellings work:
    * `dalle3`, `dall-e3`, `dalle-3`  → *dall-e-3*
    * `dalle2`, `dall-e2`, `dalle-2`  → *dall-e-2*
* OpenAI branch now feeds `model=ALIAS_MAP[…]` ⇒ no more “invalid model” 400s.
* Still no `slugify`, no prompt truncation.
"""
from __future__ import annotations
import argparse, base64, hashlib, os, random, sys, time
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ---------------------- GLOBALS ----------------------
BASE_OUT = Path("/local1/bryanzhou008/Dialect/multimodal-dialectal-bias/plotting")
DEFAULT_MODELS  = [
    "stable-diffusion-3.5-large-turbo"
]
DEFAULT_PROMPTS = ["a woman buying brinjal in a market"]
IMAGES_PER_COMBO = 10
SEED = 42

def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
fix_seed(SEED)

# -------- alias map so multiple spellings are accepted ------------------
ALIAS_MAP: Dict[str,str] = {
    # OpenAI
    "dalle3": "dall-e-3", "dall-e3": "dall-e-3", "dalle-3": "dall-e-3",
    "dalle2": "dall-e-2", "dall-e2": "dall-e-2", "dalle-2": "dall-e-2",
    # keep canonical ones unchanged
    "dall-e-3": "dall-e-3", "dall-e-2": "dall-e-2",
}

# ---------------- OPENAI back‑end ---------------------
try:
    from openai import OpenAI
    _openai_client = OpenAI()
except Exception:
    _openai_client = None  # will error gracefully later

def _openai_generate(model: str, prompt: str) -> Image.Image:
    """Handles gpt-image-1 (b64_json) and DALL‑E (url). Returns RGB PIL image."""
    if _openai_client is None:
        raise RuntimeError("OpenAI package not available")
    model = ALIAS_MAP.get(model, model)
    try:
        resp = _openai_client.images.generate(
            model=model,
            prompt=prompt,
            size="1024x1024",
            n=1,
            quality="standard" if model.startswith("dall-e-") else "low",
        )
        data = resp.data[0]
        if hasattr(data, "b64_json") and data.b64_json:
            img_bytes = base64.b64decode(data.b64_json)
            img = Image.open(io.BytesIO(img_bytes))
        else:
            import requests, io
            img_data = requests.get(data.url).content
            img = Image.open(io.BytesIO(img_data))
        if img.mode == "RGBA":
            img = img.convert("RGB")
        return img
    except Exception as e:
        print(f"[error] {model} – '{prompt}': {e}")
        return Image.new("RGB", (1024,1024), "black")

# --------------- Diffusers / local back‑ends -------------------
_BACKEND_CACHE = {}

def _load_diffusers(name: str):
    if name in _BACKEND_CACHE:
        return _BACKEND_CACHE[name]
    from diffusers import (
        StableDiffusionPipeline, StableDiffusion3Pipeline, DiffusionPipeline,
        StableDiffusion3Pipeline as SD3,
    )
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    if name == "stable-diffusion-xl-base-1.0":
        pipe = DiffusionPipeline.from_pretrained(
            name, torch_dtype=torch_dtype, use_safetensors=True, variant="fp16")
    elif name in {"stable-diffusion3-medium"}:
        pipe = SD3.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch_dtype)
    elif name in {"stable-diffusion-3.5-large", "stable-diffusion-3.5-large-turbo"}:
        pipe = SD3.from_pretrained(name.replace("_", "-"), torch_dtype=torch_dtype)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(name.replace("_", "-"), torch_dtype=torch_dtype)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    _BACKEND_CACHE[name] = pipe
    return pipe

def _diffusers_generate(model: str, prompt: str) -> Image.Image:
    pipe = _load_diffusers(model)
    if model.endswith("turbo"):
        imgs = pipe([prompt], num_inference_steps=4, guidance_scale=0.0).images
    elif model.endswith("3-medium"):
        imgs = pipe([prompt], num_inference_steps=28, guidance_scale=7.0).images
    elif model.endswith("3.5-large"):
        imgs = pipe([prompt], num_inference_steps=28, guidance_scale=3.5).images
    else:
        imgs = pipe([prompt]).images
    return imgs[0]

# --------------- Replicate back‑ends (Flux, SD‑XL via API) --------------
try:
    import replicate
    _replicate_client = replicate
except Exception:
    _replicate_client = None

def _replicate_gen(model: str, prompt: str) -> Image.Image:
    if _replicate_client is None:
        raise RuntimeError("replicate pkg not installed")
    # Map to replicate tags
    replicate_map = {
        "flux.1-dev": "black-forest-labs/flux-1-dev",
        "minDALL-E": "kuprel/min-dalle",
    }
    version = replicate_map[model]
    output = _replicate_client.run(version, input={"prompt": prompt})
    url = output[0] if isinstance(output, list) else output
    import requests, io
    img = Image.open(io.BytesIO(requests.get(url).content)).convert("RGB")
    return img

# --------------- dispatch table ---------------------
_BACKENDS = {
    # OpenAI
    "gpt-image-1": _openai_generate,
    "dall-e-2": _openai_generate,
    "dall-e-3": _openai_generate,
    # Diffusers local
    "stable-diffusion-xl-base-1.0": _diffusers_generate,
    "stable-diffusion1.4": _diffusers_generate,
    "stable-diffusion1.5": _diffusers_generate,
    "stable-diffusion2.1": _diffusers_generate,
    "stable-diffusion3-medium": _diffusers_generate,
    "stable-diffusion-3.5-large": _diffusers_generate,
    "stable-diffusion-3.5-large-turbo": _diffusers_generate,
    # Replicate
    "flux.1-dev": _replicate_gen,
    "minDALL-E": _replicate_gen,
}

# Lower‑case key lookup convenience
for k in list(_BACKENDS):
    _BACKENDS[k.lower()] = _BACKENDS[k]

# ---------------- Utils -----------------------
INVALID_FS_CHARS = {"/": "／"}

def safe_path(text: str) -> str:
    for bad, good in INVALID_FS_CHARS.items():
        text = text.replace(bad, good)
    return text.strip()

# ---------------- Main logic ------------------

def parse_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    p.add_argument("--prompts", nargs="*", default=DEFAULT_PROMPTS)
    return p.parse_args()

def main():
    args = parse_cli()
    models  = args.models
    prompts = args.prompts

    for raw_model in models:
        canonical = ALIAS_MAP.get(raw_model, raw_model)
        if canonical not in _BACKENDS:
            print(f"[skip] Unknown model '{raw_model}'.")
            continue
        gen_func = _BACKENDS[canonical]

        for prompt in prompts:
            out_dir = BASE_OUT / safe_path(prompt) / canonical
            out_dir.mkdir(parents=True, exist_ok=True)
            for idx in range(IMAGES_PER_COMBO):
                img_path = out_dir / f"{idx}.jpg"
                if img_path.exists():
                    continue
                img = gen_func(canonical, prompt) if gen_func is _openai_generate else gen_func(canonical, prompt)
                img.save(img_path, "JPEG")
                print(f"saved {img_path.relative_to(BASE_OUT)}")

if __name__ == "__main__":
    main()
