import os
import sys
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
from IPython.display import display, update_display
import torch
import numpy as np
import pandas as pd
from PIL import Image
from diffusers import StableDiffusionPipeline

# ---------------------------
# Global Configuration
# ---------------------------
# BASE_DIR is set to the repository root.
# Since this script is in project_home/src/image_generation, we go two levels up.
BASE_DIR = Path(__file__).resolve().parents[2]

# Choose the Stable Diffusion model version.
model_id = "stabilityai/stable-diffusion-2-1"

# Initialize the Stable Diffusion pipeline.
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# ---------------------------
# Helper Functions
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate images using Stable Diffusion for polysemy prompts with optional ENTIGEN modifications."
    )
    parser.add_argument(
        "--dialect",
        type=str,
        required=True,
        choices=["aae", "bre", "che", "ine", "sge"],
        help="Dialect code (aae, bre, che, ine, sge)."
    )
    parser.add_argument(
        "--entigen",
        action="store_true",
        help="If set, prepends a fixed ENTIGEN prompt (In Standard American English) to each prompt."
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

def generate_stable_diffusion(prompt: str, save_dir: Path) -> None:
    """
    Generates images using Stable Diffusion and saves them to the specified directory.
    """
    num_cols = 3
    num_rows = 3
    prompt_list = [prompt] * num_cols

    all_images = []
    for _ in range(num_rows):
        images = pipe(prompt_list).images
        all_images.extend(images)

    for i, image in enumerate(all_images):
        image.save(str(save_dir / f"{i}.jpg"))

# ---------------------------
# Main Workflow
# ---------------------------
def main():
    args = parse_args()
    dialect = args.dialect
    use_entigen = args.entigen

    # Mapping for full dialect names (for CSV naming conventions).
    dialect_names = {
        "aae": "AfricanAmericanEnglish",
        "bre": "BritishEnglish",
        "che": "ChicanoEnglish",
        "ine": "IndianEnglish",
        "sge": "Singlish"
    }

    # Define paths based on the specified dialect.
    data_file = BASE_DIR / "data" / "text" / "polysemy_csvs" / f"{dialect_names[dialect]}_Polysemy.csv"
    base_img_dir = BASE_DIR / "data" / "images" / "oct_5_entigen" / dialect
    img_dir = base_img_dir / "stable-diffusion2.1"
    polysemy_subdir = img_dir / "polysemy"

    # Prepare the output directory.
    prepare_directory(polysemy_subdir)

    # Read data from CSV.
    df = pd.read_csv(data_file)
    prompts = list(df["Prompt"])

    # Iterate over each prompt and generate images.
    for prompt in tqdm(prompts, total=len(prompts)):
        # Apply ENTIGEN modification if enabled.
        if use_entigen:
            prompt = "In Standard American English " + prompt

        prompt_dir = polysemy_subdir / prompt
        if not prompt_dir.exists():
            prompt_dir.mkdir(parents=True, exist_ok=True)

        generate_stable_diffusion(prompt, prompt_dir)

if __name__ == "__main__":
    main()