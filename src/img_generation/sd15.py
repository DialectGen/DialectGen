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
# BASE_DIR is set to the repository root.
# Since this script is in project_home/src/image_generation, we go 2 levels up.
BASE_DIR = Path(__file__).resolve().parents[2]

# Choose Stable Diffusion version:
model_id = "sd-legacy/stable-diffusion-v1-5"

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
        choices=["concise", "detailed", "entigen", "polysemy", "rewrite_concise", "rewrite_detailed"],
        help="Mode to use (concise, detailed, entigen, polysemy, rewrite_concise, or rewrite_detailed)."
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

def encode_long_prompt(pipe, prompt: str):
    """
    Turn an (arbitrarily long) text prompt into `prompt_embeds`
    and `negative_prompt_embeds` that the pipeline can digest.
    """
    tokenizer      = pipe.tokenizer
    text_encoder   = pipe.text_encoder
    device         = pipe.device
    max_len        = tokenizer.model_max_length        # 77 for SD‑1.5/2.1

    # 1) tokenise without truncating
    input_ids = tokenizer(prompt,
                          return_tensors="pt",
                          truncation=False).input_ids.to(device)

    # 2) empty negative prompt, padded to the same length as `input_ids`
    neg_ids = tokenizer("",
                        padding="max_length",
                        max_length=input_ids.shape[-1],
                        return_tensors="pt").input_ids.to(device)

    # 3) slide over the sequence in 77‑token chunks
    pos_chunks, neg_chunks = [], []
    for i in range(0, input_ids.shape[-1], max_len):
        pos_chunks.append(text_encoder(input_ids[:, i:i+max_len])[0])
        neg_chunks.append(text_encoder(neg_ids  [:, i:i+max_len])[0])

    # 4) stitch the chunks back together
    prompt_embeds          = torch.cat(pos_chunks, dim=1)
    negative_prompt_embeds = torch.cat(neg_chunks, dim=1)
    return prompt_embeds, negative_prompt_embeds

# ---------------------------
# MODIFIED: image generator that auto‑handles long prompts
# ---------------------------
def generate_stable_diffusion_batch(prompt: str, num_images: int) -> list:
    """
    Generates `num_images` images from **any‑length** prompt.
    Images come back in batches of ≤3 for GPU efficiency.
    """
    images = []
    while len(images) < num_images:
        batch_size = min(3, num_images - len(images))

        # --- encode prompt once, then replicate along batch dimension
        p_emb, n_emb = encode_long_prompt(pipe, prompt)
        p_emb = p_emb.repeat(batch_size, 1, 1)
        n_emb = n_emb.repeat(batch_size, 1, 1)

        # --- run the diffusion model
        batch_imgs = pipe(
            prompt_embeds=p_emb,
            negative_prompt_embeds=n_emb
        ).images

        images.extend(batch_imgs)

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

    # ENTIGEN prompt prefixes mapping for dialect prompts and standard American English prefix.
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
        img_dir = BASE_DIR / "data" / "image" / mode / f"{dialect}" / "stable-diffusion1.5"
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
            import hashlib

            # Hash dialect prompts if using rewrite modes
            if mode in {"rewrite_concise", "rewrite_detailed"}:
                dp_hash = hashlib.md5(dp.encode()).hexdigest()
                dp_dir = lr_subdir / dp_hash
            else:
                dp_dir = lr_subdir / dp

            # Always use full SAE prompt (assumed shorter and safer)
            sp_dir = hr_subdir / sp

            ensure_and_generate(dp_dir, dp, replace_flag)
            ensure_and_generate(sp_dir, sp, replace_flag)

if __name__ == "__main__":
    main()
