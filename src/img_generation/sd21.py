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
from diffusers import StableDiffusionPipeline

# ---------------------------
# Global Configuration
# ---------------------------
# BASE_DIR is set to the repository root.
# Since this script is in project_home/src/image_generation, we go 2 levels up.
BASE_DIR = Path(__file__).resolve().parents[2]

# Choose Stable Diffusion version:
model_id = "stabilityai/stable-diffusion-2-1"

# Initialize the Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# ---------------------------
# Helper Functions
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate images using Stable Diffusion for specified dialect(s) and prompt type. "
            "If --replace is provided, the output image directory is replaced; "
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
            "If provided, the output image directory for each dialect is replaced; "
            "otherwise, the script resumes and generates only missing images."
        )
    )
    return parser.parse_args()

def generate_stable_diffusion_batch(prompt: str, num_images: int) -> list:
    """
    Generates a specified number of images using Stable Diffusion.
    Images are generated in batches (up to 3 at a time).
    """
    images = []
    while len(images) < num_images:
        batch_size = min(3, num_images - len(images))
        # Generate a row of images; note that the pipeline call returns a list of images.
        result = pipe([prompt] * batch_size).images
        images.extend(result)
    return images[:num_images]

def ensure_and_generate(prompt_folder: Path, prompt: str, replace: bool) -> None:
    """
    For a given prompt folder:
      - If replace==True, the folder is removed and recreated.
      - If the folder does not exist, it is created.
      - Then, if the folder already contains images "0.jpg" to "8.jpg", generation is skipped.
      - Otherwise, only the missing images are generated so that the folder eventually contains 9 images.
    """
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
            missing_indices = [i for i in range(9) if not (prompt_folder / f"{i}.jpg").is_file()]

    if not missing_indices:
        print(f"Skipping generation for folder '{prompt_folder}' (already complete).")
        return

    num_missing = len(missing_indices)
    print(f"Generating {num_missing} image(s) for prompt '{prompt}' in folder '{prompt_folder}'.")
    new_images = generate_stable_diffusion_batch(prompt, num_missing)
    for idx, image in zip(missing_indices, new_images):
        image.save(str(prompt_folder / f"{idx}.jpg"))

# ---------------------------
# Main Workflow
# ---------------------------
def main():
    args = parse_args()
    mode = args.mode
    replace_flag = args.replace

    # ENTIGEN prompt prefixes mapping for dialect prompts and SAE prompt prefix.
    entigen_prefixes = {
        "aae": "In African American English, ",
        "bre": "In British English, ",
        "che": "In Chicano English, ",
        "ine": "In Indian English, ",
        "sge": "In Singlish, "
    }
    sae_prefix = "In Standard American English, "

    # Process each dialect in the order provided.
    for dialect in args.dialects:
        # Define paths based on the specified mode and dialect.
        data_file = BASE_DIR / "data" / "text" / mode / f"{dialect}.csv"
        img_dir = BASE_DIR / "data" / "image" / mode / f"{dialect}" / "stable-diffusion2.1"
        lr_subdir = img_dir / "dialect_imgs"
        hr_subdir = img_dir / "sae_imgs"

        # If --replace is set, remove the entire image directory for the dialect.
        if replace_flag:
            if img_dir.exists():
                shutil.rmtree(img_dir)
            lr_subdir.mkdir(parents=True, exist_ok=True)
            hr_subdir.mkdir(parents=True, exist_ok=True)
        else:
            lr_subdir.mkdir(parents=True, exist_ok=True)
            hr_subdir.mkdir(parents=True, exist_ok=True)

        # Read CSV data containing prompts.
        df = pd.read_csv(data_file, encoding='unicode_escape')
        dialect_prompts = df["Dialect_Prompt"].tolist()
        sae_prompts = df["SAE_Prompt"].tolist()

        # Iterate over each prompt pair and generate missing images.
        for dp, sp in tqdm(zip(dialect_prompts, sae_prompts), total=len(dialect_prompts), desc=f"Processing dialect {dialect}"):
            if mode == "entigen":
                dp = entigen_prefixes[dialect] + dp
                sp = sae_prefix + sp

            # Use the prompt text as the folder name (in production, consider sanitizing these names).
            dp_dir = lr_subdir / dp
            sp_dir = hr_subdir / sp

            ensure_and_generate(dp_dir, dp, replace_flag)
            ensure_and_generate(sp_dir, sp, replace_flag)

if __name__ == "__main__":
    main()
