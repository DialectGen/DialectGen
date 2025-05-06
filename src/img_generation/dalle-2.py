import os
import sys
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import requests
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
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_ID = "dall-e-2"
client = OpenAI()

# ---------------------------
# Helper Functions
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate images using DALL-E 2 for specified dialect(s) and prompt type. "
            "If --replace is provided, the output image directory is replaced; "
            "otherwise, missing images are generated to complete one per prompt."
        )
    )
    parser.add_argument(
        "--dialects",
        type=str,
        nargs="+",
        required=True,
        choices=["aae", "bre", "che", "ine", "sge"],
        help="One or more dialect codes (aae, bre, che, ine, sge)."
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["concise", "detailed", "entigen", "polysemy", "explained"],
        help="Mode to use (concise, detailed, entigen, polysemy, or explained)."
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help=(
            "If provided, the output image directory for each dialect is replaced; "
            "otherwise, the script resumes and generates only missing or black placeholder images."
        )
    )
    return parser.parse_args()

def generate_dalle2_image(prompt: str) -> Image:
    try:
        response = client.images.generate(
            model=MODEL_ID,
            prompt=prompt,
            size="1024x1024",
            n=1,
        )
        image_url = response.data[0].url
        img_data = requests.get(image_url).content
        image = Image.open(BytesIO(img_data))
    except Exception as e:
        print(f"Error generating image for prompt '{prompt}': {e}. Returning black image.")
        image = Image.new("RGB", (1024, 1024), color="black")
    return image

def is_black_image(image_path: Path) -> bool:
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            extrema = img.getextrema()
            return extrema == ((0, 0), (0, 0), (0, 0))
    except Exception as e:
        print(f"Error checking if image is black at '{image_path}': {e}")
        return False

def ensure_and_generate(prompt_folder: Path, prompt: str, replace: bool) -> None:
    image_path = prompt_folder / "0.jpg"
    if replace:
        if prompt_folder.exists():
            shutil.rmtree(prompt_folder)
        prompt_folder.mkdir(parents=True, exist_ok=True)
        regenerate = True
    else:
        if not prompt_folder.exists():
            prompt_folder.mkdir(parents=True, exist_ok=True)
            regenerate = True
        elif not image_path.is_file() or is_black_image(image_path):
            regenerate = True
        else:
            regenerate = False

    if not regenerate:
        print(f"Skipping generation for folder '{prompt_folder}' (already contains valid image).")
        return

    print(f"Generating image for prompt '{prompt}' in folder '{prompt_folder}'.")
    new_image = generate_dalle2_image(prompt)
    new_image.save(str(image_path))

# ---------------------------
# Main Workflow
# ---------------------------
def main():
    args = parse_args()
    mode = args.mode
    replace_flag = args.replace

    entigen_prefixes = {
        "aae": "In African American English, ",
        "bre": "In British English, ",
        "che": "In Chicano English, ",
        "ine": "In Indian English, ",
        "sge": "In Singlish, "
    }
    sae_prefix = "In Standard American English, "

    for dialect in args.dialects:
        data_file = BASE_DIR / "data" / "text" / mode / f"{dialect}.csv"
        img_dir = BASE_DIR / "data" / "image" / mode / f"{dialect}" / "dalle2"
        lr_subdir = img_dir / "dialect_imgs"
        hr_subdir = img_dir / "sae_imgs"

        if replace_flag:
            if img_dir.exists():
                shutil.rmtree(img_dir)
            lr_subdir.mkdir(parents=True, exist_ok=True)
            hr_subdir.mkdir(parents=True, exist_ok=True)
        else:
            lr_subdir.mkdir(parents=True, exist_ok=True)
            hr_subdir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(data_file, encoding='unicode_escape')
        dialect_prompts = df["Dialect_Prompt"].tolist()
        sae_prompts = df["SAE_Prompt"].tolist()

        for dp, sp in tqdm(zip(dialect_prompts, sae_prompts), total=len(dialect_prompts), desc=f"Processing dialect {dialect}"):
            if mode == "entigen":
                dp = entigen_prefixes[dialect] + dp
                sp = sae_prefix + sp

            dp_dir = lr_subdir / dp
            sp_dir = hr_subdir / sp

            ensure_and_generate(dp_dir, dp, replace_flag)
            ensure_and_generate(sp_dir, sp, replace_flag)

if __name__ == "__main__":
    main()
