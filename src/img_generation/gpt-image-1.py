#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate images with gpt-image-1 for one or more dialect prompt sets.

Example
-------
python generate_images.py --dialects aae che --mode concise --replace
"""

import shutil
import argparse
import base64
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from io import BytesIO
from PIL import Image
from openai import OpenAI
import random
import torch
import numpy as np
from transformers import set_seed

def fix_seed(seed: int):
    """Sets the seed for reproducibility across various libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)

# Fix the seed for reproducibility
fix_seed(42)

# ---------------------------
# Global Configuration
# ---------------------------

BASE_DIR   = Path(__file__).resolve().parents[2]
MODEL_ID   = "gpt-image-1"        # target model

client = OpenAI()

# ---------------------------
# Helper Functions
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Generate images with gpt-image-1 for specified dialect(s) and prompt type. "
            "If --replace is provided, existing images are discarded."
        )
    )
    p.add_argument("--dialects", nargs="+", required=True, choices=["aae", "bre", "che", "ine", "sge"])
    p.add_argument("--mode", required=True, choices=["concise", "detailed", "entigen", "polysemy", "explained"])
    p.add_argument("--replace", action="store_true", help="Replace existing output directories entirely.")
    return p.parse_args()

def generate_gpt_image(prompt: str) -> Image.Image:
    """
    Returns a PIL.Image produced by gpt-image-1 for the given prompt.
    On failure, returns a 1024×1024 black image.
    """
    try:
        resp = client.images.generate(
            model=MODEL_ID,
            prompt=prompt,
            size="1024x1024",
            quality="low",
            n=1,
        )
        img_bytes = base64.b64decode(resp.data[0].b64_json)
        return Image.open(BytesIO(img_bytes))
    except Exception as e:
        print(f"[ERROR] prompt='{prompt}': {e} → black image returned.")
        return Image.new("RGB", (1024, 1024), "black")

def ensure_and_generate(out_dir: Path, prompt: str, replace: bool) -> None:
    """
    Guarantees that out_dir/0.jpg exists (regenerated if --replace).
    """
    if replace:
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_path = out_dir / "0.jpg"
    if img_path.exists():
        return

    print(f"⏳  {out_dir.relative_to(BASE_DIR)}")
    image = generate_gpt_image(prompt)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image.save(img_path)

# ---------------------------
# Main Workflow
# ---------------------------

def main():
    args    = parse_args()
    mode    = args.mode
    replace = args.replace

    entigen_prefix = {
        "aae": "In African American English, ",
        "bre": "In British English, ",
        "che": "In Chicano English, ",
        "ine": "In Indian English, ",
        "sge": "In Singlish, "
    }
    sae_prefix = "In Standard American English, "

    for dialect in args.dialects:
        csv_file = BASE_DIR / "data" / "text" / mode / f"{dialect}.csv"
        img_root = BASE_DIR / "data" / "image" / mode / dialect / MODEL_ID
        lr_dir   = img_root / "dialect_imgs"
        hr_dir   = img_root / "sae_imgs"

        if replace:
            shutil.rmtree(img_root, ignore_errors=True)
        lr_dir.mkdir(parents=True, exist_ok=True)
        hr_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(csv_file, encoding="unicode_escape")
        for dp, sp in tqdm(
            zip(df["Dialect_Prompt"], df["SAE_Prompt"]),
            total=len(df),
            desc=f"[{dialect}]"
        ):
            if mode == "entigen":
                dp = entigen_prefix[dialect] + dp
                sp = sae_prefix + sp

            ensure_and_generate(lr_dir / dp, dp, replace)
            ensure_and_generate(hr_dir / sp, sp, replace)

if __name__ == "__main__":
    main()
