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

# ---------------------------
# Global Configuration
# ---------------------------
# BASE_DIR is set to the repository root.
BASE_DIR = Path(__file__).resolve().parents[2]

# Model identifier for DALL-E 2
MODEL_ID = "dall-e-2"

# Initialize the OpenAI client for DALL-E 2
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
            "otherwise, the script resumes and generates only missing images."
        )
    )
    return parser.parse_args()

def generate_dalle2_image(prompt: str) -> Image:
    """
    Generates a single image using DALL-E 2.
    The DALL-E 2 API returns a response containing a URL pointing to the generated image.
    If any error occurs, this function returns a completely black image (1024x1024).
    """
    try:
        response = client.images.generate(
            model=MODEL_ID,
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url
        # Download the image content and open it with PIL
        img_data = requests.get(image_url).content
        image = Image.open(BytesIO(img_data))
    except Exception as e:
        print(f"Error generating image for prompt '{prompt}': {e}. Returning black image.")
        image = Image.new("RGB", (1024, 1024), color="black")
    return image

def ensure_and_generate(prompt_folder: Path, prompt: str, replace: bool) -> None:
    """
    For a given prompt folder:
      - If replace==True, the folder is removed and recreated.
      - If the folder does not exist, it is created.
      - Then, if the folder does not contain "0.jpg", the missing image is generated.
    """
    if replace:
        if prompt_folder.exists():
            shutil.rmtree(prompt_folder)
        prompt_folder.mkdir(parents=True, exist_ok=True)
        missing_indices = [0]
    else:
        if not prompt_folder.exists():
            prompt_folder.mkdir(parents=True, exist_ok=True)
            missing_indices = [0]
        else:
            missing_indices = [0] if not (prompt_folder / "0.jpg").is_file() else []

    if not missing_indices:
        print(f"Skipping generation for folder '{prompt_folder}' (already complete).")
        return

    print(f"Generating image for prompt '{prompt}' in folder '{prompt_folder}'.")
    new_image = generate_dalle2_image(prompt)
    # Save the generated image as "0.jpg"
    new_image.save(str(prompt_folder / "0.jpg"))

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
        # Change folder name to "dalle2" instead of "dalle3"
        img_dir = BASE_DIR / "data" / "image" / mode / f"{dialect}" / "dalle2"
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
