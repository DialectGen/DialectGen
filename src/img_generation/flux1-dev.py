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
        description=(
            "Generate images using FluxPipeline (black-forest-labs/FLUX.1-dev) "
            "for one or more specified dialects and a given prompt mode. "
            "If --replace is given, the output image directory is replaced; "
            "otherwise, missing images are generated to complete a set of 9 per prompt."
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
        choices=["basic", "complex", "entigen", "polysemy"],
        help="Mode to use (basic, complex, entigen, or polysemy)."
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help=(
            "If provided, replace the entire output image directory for each dialect. "
            "Otherwise, the script will resume and generate missing images."
        )
    )
    return parser.parse_args()

def generate_flux(prompt: str, num_images: int) -> list:
    """
    Generates a specified number of images using FluxPipeline.
    Generates in batches of up to 3 images.
    """
    images = []
    # Generate in batches (the pipeline expects a list of prompts)
    while len(images) < num_images:
        batch_size = min(3, num_images - len(images))
        result = pipe(
            [prompt] * batch_size,
            height=1024,
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=50,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(0)
        )
        images.extend(result.images)
    return images[:num_images]

def ensure_and_generate(prompt_folder: Path, prompt: str, replace: bool) -> None:
    """
    For a given prompt folder:
      - If replace==True, the folder is removed and recreated.
      - If the folder does not exist, it is created.
      - Then, if the folder already contains images "0.jpg" to "8.jpg", skip generation.
      - Otherwise, generate only the missing images so that the folder eventually contains 9 images.
    """
    # If replacement is requested, remove any existing folder and start over.
    if replace:
        if prompt_folder.exists():
            shutil.rmtree(prompt_folder)
        prompt_folder.mkdir(parents=True, exist_ok=True)
        missing_indices = list(range(9))
    else:
        if not prompt_folder.exists():
            prompt_folder.mkdir(parents=True, exist_ok=True)
            missing_indices = list(range(9))
        else:
            # Check for missing images named "0.jpg", "1.jpg", ..., "8.jpg"
            missing_indices = [i for i in range(9) if not (prompt_folder / f"{i}.jpg").is_file()]
    
    if len(missing_indices) == 0:
        print(f"Skipping generation for folder '{prompt_folder}' (already complete).")
        return

    num_missing = len(missing_indices)
    print(f"Generating {num_missing} image(s) for prompt '{prompt}' in folder '{prompt_folder}'.")
    new_images = generate_flux(prompt, num_missing)
    for idx, image in zip(missing_indices, new_images):
        image.save(str(prompt_folder / f"{idx}.jpg"))

# ---------------------------
# Main Workflow
# ---------------------------
def main():
    args = parse_args()
    mode = args.mode
    replace_flag = args.replace

    # ENTIGEN prompt prefixes for dialect and SAE prompts
    entigen_prefixes = {
        "aae": "In African American English, ",
        "bre": "In British English, ",
        "che": "In Chicano English, ",
        "ine": "In Indian English, ",
        "sge": "In Singlish, "
    }
    sae_prefix = "In Standard American English, "

    # Process each dialect in the order given
    for dialect in args.dialects:
        # Define paths based on mode and dialect
        data_file = BASE_DIR / "data" / "text" / mode / f"{dialect}.csv"
        img_dir = BASE_DIR / "data" / "image" / mode / f"{dialect}" / "flux.1-dev"
        lr_subdir = img_dir / "dialect_imgs"
        hr_subdir = img_dir / "sae_imgs"

        # For each dialect, if --replace is set, remove the whole image directory
        if replace_flag:
            if img_dir.exists():
                shutil.rmtree(img_dir)
            lr_subdir.mkdir(parents=True, exist_ok=True)
            hr_subdir.mkdir(parents=True, exist_ok=True)
        else:
            lr_subdir.mkdir(parents=True, exist_ok=True)
            hr_subdir.mkdir(parents=True, exist_ok=True)

        # Read CSV data containing prompts
        df = pd.read_csv(data_file, encoding="unicode_escape")
        dialect_prompts = df["Dialect_Prompt"].tolist()
        sae_prompts = df["SAE_Prompt"].tolist()

        # Iterate over each prompt pair and generate missing images
        for dp, sp in tqdm(
            zip(dialect_prompts, sae_prompts),
            total=len(dialect_prompts),
            desc=f"Processing dialect {dialect}"
        ):
            # In entigen mode, add the appropriate prefixes.
            if mode == "entigen":
                dp = entigen_prefixes[dialect] + dp
                sp = sae_prefix + sp

            # Use the prompt text as the folder name.
            # (In production, you may wish to sanitize these names to remove special characters.)
            dp_dir = lr_subdir / dp
            sp_dir = hr_subdir / sp

            # For each prompt folder, check existing images and generate missing ones.
            ensure_and_generate(dp_dir, dp, replace_flag)
            ensure_and_generate(sp_dir, sp, replace_flag)

if __name__ == "__main__":
    main()