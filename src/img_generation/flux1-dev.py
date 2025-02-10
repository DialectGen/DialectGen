import os
import sys
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from PIL import Image
from diffusers import FluxPipeline

# ---------------------------
# Global Configuration
# ---------------------------
# Since this script is in project_home/src/image_generation, we go 2 levels up.
BASE_DIR = Path(__file__).resolve().parents[2]

# Define model and initialize the FluxPipeline
model_id = "black-forest-labs/FLUX.1-dev"
pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()  # Offload to CPU to save VRAM

# ---------------------------
# Helper Functions
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate images using FluxPipeline (black-forest-labs/FLUX.1-dev) for a specified dialect and prompt type."
    )
    parser.add_argument(
        "--dialect",
        type=str,
        required=True,
        choices=["aae", "bre", "che", "ine", "sge"],
        help="Dialect code (aae, bre, che, ine, sge)."
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        required=True,
        choices=["basic", "entigen", "polysemy"],
        help="Type of prompt to use (basic, entigen, or polysemy)."
    )
    return parser.parse_args()

def prepare_directory(path: Path) -> None:
    """
    Recursively creates a directory. If it exists and is non-empty,
    prompts the user whether to replace its contents.
    """
    if path.exists():
        if any(path.iterdir()):
            response = input(
                f"The directory '{path}' is not empty. Do you want to replace its contents? (y/n): "
            ).strip().lower()
            if response == 'y':
                shutil.rmtree(path)
                path.mkdir(parents=True, exist_ok=True)
            else:
                print("Operation aborted by the user.")
                sys.exit(1)
    else:
        path.mkdir(parents=True, exist_ok=True)

def generate_flux(prompt: str, save_dir: Path) -> None:
    """
    Generates images using FluxPipeline and saves them to the specified directory.
    Generates a 3×3 grid of images.
    """
    num_cols = 3
    num_rows = 3
    prompt_list = [prompt] * num_cols

    all_images = []
    for _ in range(num_rows):
        result = pipe(
            prompt_list,
            height=1024,
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=50,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(0)
        )
        images = result.images
        all_images.extend(images)

    for i, image in enumerate(all_images):
        image.save(str(save_dir / f"{i}.jpg"))

# ---------------------------
# Main Workflow
# ---------------------------
def main():
    args = parse_args()
    dialect = args.dialect
    prompt_type = args.prompt_type

    # Define paths based on prompt_type and dialect
    data_file = BASE_DIR / "data" / "text" / prompt_type / f"{dialect}.csv"
    img_dir = BASE_DIR / "data" / "image" / prompt_type / f"{dialect}" / "flux.1-dev"

    # ENTIGEN prompt prefixes for dialect and SAE prompts
    entigen_prefixes = {
        "aae": "In African American English, ",
        "bre": "In British English, ",
        "che": "In Chicano English, ",
        "ine": "In Indian English, ",
        "sge": "In Singlish, "
    }
    sae_prefix = "In Standard American English, "

    # Prepare the image output directory and subdirectories
    prepare_directory(img_dir)
    lr_subdir = img_dir / "dialect_imgs"
    hr_subdir = img_dir / "sae_imgs"
    prepare_directory(lr_subdir)
    prepare_directory(hr_subdir)

    # Read CSV data containing prompts
    df = pd.read_csv(data_file, encoding="unicode_escape")
    dialect_prompts = df["Dialect_Prompt"].tolist()
    sae_prompts = df["SAE_Prompt"].tolist()

    # Iterate over each prompt pair and generate images
    for dp, sp in tqdm(zip(dialect_prompts, sae_prompts), total=len(dialect_prompts)):
        if prompt_type == "entigen":
            dp = entigen_prefixes[dialect] + dp
            sp = sae_prefix + sp

        dp_dir = lr_subdir / dp
        sp_dir = hr_subdir / sp

        if not dp_dir.exists():
            dp_dir.mkdir(parents=True, exist_ok=True)
            generate_flux(dp, dp_dir)
        if not sp_dir.exists():
            sp_dir.mkdir(parents=True, exist_ok=True)
            generate_flux(sp, sp_dir)

if __name__ == "__main__":
    main()