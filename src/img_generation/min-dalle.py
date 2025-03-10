import os
import sys
import shutil
import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
from min_dalle import MinDalle
from IPython.display import display, update_display

# ---------------------------
# Global Configuration
# ---------------------------
# BASE_DIR is set to the repository root.
# Since this script is in project_home/src/image_generation, we go two levels up.
BASE_DIR = Path(__file__).resolve().parents[2]

# Hyperparameters
TEMPERATURE = 1            # Parameter: 0.01 to 16
GRID_SIZE = 3              # Default grid size (for full generation, 3x3=9 images)
SUPERCONDITION_FACTOR = 16 # Numeric parameter
TOP_K = 128                # Integer parameter
SEAMLESS = False
DTYPE = "float32"

# Initialize the MinDalle model
model = MinDalle(
    models_root=BASE_DIR / "pretrained",
    dtype=getattr(torch, DTYPE),
    device='cuda',
    is_mega=True,
    is_reusable=True
)

# ---------------------------
# Helper Functions
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate images using MinDalle for specified dialect(s) and prompt type. "
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
        help="Type of prompt to use (basic, entigen, or polysemy)."
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

def generate_min_dalle_batch(prompt: str, num_images: int, seamless: bool,
                             temperature: float, top_k: int,
                             supercondition_factor: float) -> list:
    """
    Generates a specified number of images using MinDalle.
    To generate at least num_images, a temporary grid size is computed
    as the ceiling of sqrt(num_images). The model then generates grid**2 images,
    and only the first num_images are returned.
    """
    grid = math.ceil(math.sqrt(num_images))
    images_tensor = model.generate_images(
        text=prompt,
        seed=-1,
        grid_size=grid,
        is_seamless=seamless,
        temperature=temperature,
        top_k=top_k,
        supercondition_factor=supercondition_factor
    )
    images_array = images_tensor.to('cpu').numpy()
    pil_images = []
    for i, img in enumerate(images_array):
        image = Image.fromarray((img * 1).astype(np.uint8)).convert('RGB')
        pil_images.append(image)
    return pil_images[:num_images]

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
    new_images = generate_min_dalle_batch(prompt, num_missing, SEAMLESS, TEMPERATURE, TOP_K, SUPERCONDITION_FACTOR)
    for idx, image in zip(missing_indices, new_images):
        image.save(str(prompt_folder / f"{idx}.jpg"))

# ---------------------------
# Main Workflow
# ---------------------------
def main():
    args = parse_args()
    mode = args.mode
    replace_flag = args.replace

    # ENTIGEN prefixes mapping for dialect prompts and standard American English prefix.
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
        # Define paths based on the specified prompt type and dialect.
        data_file = BASE_DIR / "data" / "text" / mode / f"{dialect}.csv"
        img_dir = BASE_DIR / "data" / "image" / mode / f"{dialect}" / "minDALL-E"
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
