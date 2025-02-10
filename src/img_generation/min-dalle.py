import os
import sys
import shutil
import argparse
from pathlib import Path
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
GRID_SIZE = 3              # Integer parameter
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
        description="Generate images using MinDalle for a specified dialect and prompt type."
    )
    parser.add_argument(
        "--dialect",
        type=str,
        required=True,
        choices=["aae", "bre", "che", "ine", "sge"],
        help="Dialect code (aae, bre, che, ine, sge)."
    )
    parser.add_argument(
        "--mode",
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

def generate_min_dalle(prompt: str, save_dir: Path, seamless: bool,
                        temperature: float, grid_size: int,
                        supercondition_factor: float) -> None:
    """
    Generates images using MinDalle and saves them to the specified directory.
    """
    images = model.generate_images(
        text=prompt,
        seed=-1,
        grid_size=grid_size,
        is_seamless=seamless,
        temperature=temperature,
        top_k=int(TOP_K),
        supercondition_factor=float(supercondition_factor)
    )
    images = images.to('cpu').numpy()
    for i, img in enumerate(images):
        image = Image.fromarray((img * 1).astype(np.uint8)).convert('RGB')
        image.save(str(save_dir / f"{i}.jpg"))

# ---------------------------
# Main Workflow
# ---------------------------
def main():
    args = parse_args()
    dialect = args.dialect
    mode = args.mode

    # Define paths based on the specified prompt type and dialect
    data_file = BASE_DIR / "data" / "text" / mode / f"{dialect}.csv"
    img_dir = BASE_DIR / "data" / "image" / mode / f"{dialect}" / "minDALL-E"

    # ENTIGEN prefixes mapping and standard American English prefix
    entigen_prefixes = {
        "aae": "In African American English, ",
        "bre": "In British English, ",
        "che": "In Chicano English, ",
        "ine": "In Indian English, ",
        "sge": "In Singlish, "
    }
    sae_prefix = "In Standard American English, "

    # Prepare output directories
    prepare_directory(img_dir)
    lr_subdir = img_dir / "dialect_imgs"
    hr_subdir = img_dir / "sae_imgs"
    prepare_directory(lr_subdir)
    prepare_directory(hr_subdir)

    # Read data from CSV
    df = pd.read_csv(data_file, encoding='unicode_escape')
    dialect_prompts = df["Dialect_Prompt"].tolist()
    sae_prompts = df["SAE_Prompt"].tolist()

    # Iterate over prompt pairs and generate images
    for dp, sp in tqdm(zip(dialect_prompts, sae_prompts), total=len(dialect_prompts)):
        if mode == "entigen":
            dp = entigen_prefixes[dialect] + dp
            sp = sae_prefix + sp

        dp_dir = lr_subdir / dp
        sp_dir = hr_subdir / sp

        if not dp_dir.exists():
            dp_dir.mkdir(parents=True, exist_ok=True)
            generate_min_dalle(dp, dp_dir, SEAMLESS, TEMPERATURE, GRID_SIZE, SUPERCONDITION_FACTOR)
        if not sp_dir.exists():
            sp_dir.mkdir(parents=True, exist_ok=True)
            generate_min_dalle(sp, sp_dir, SEAMLESS, TEMPERATURE, GRID_SIZE, SUPERCONDITION_FACTOR)

if __name__ == "__main__":
    main()
