import os
import sys
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
from IPython.display import display, update_display
import random
import hashlib                         # <-- moved to top
import torch
import numpy as np
import pandas as pd
from PIL import Image
from diffusers import StableDiffusionPipeline
from transformers import set_seed
from typing import List      


# ------------------------------------------------------------------
# Reproducibility helpers
# ------------------------------------------------------------------
def fix_seed(seed: int) -> None:
    """Sets the seed for reproducibility across random / numpy / torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)

fix_seed(42)  # global seed
# ------------------------------------------------------------------
# Paths & model
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

model_id = "sd-legacy/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
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
        choices=[
            "concise", "detailed",
            "entigen", "polysemy",
            "rewrite_concise", "rewrite_detailed",
            "translate_concise", "translate_detailed",
            "translate_concise_gpt41", "translate_detailed_gpt41"
        ],
        help="Generation mode."
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

# ------------------------------------------------------------------
# Long-prompt helpers
# ------------------------------------------------------------------
def encode_long_prompt(pipe, prompt: str):
    tokenizer    = pipe.tokenizer
    text_encoder = pipe.text_encoder
    device       = pipe.device
    max_len      = tokenizer.model_max_length  # 77 for SD-1.5

    input_ids = tokenizer(prompt, return_tensors="pt", truncation=False).input_ids.to(device)
    neg_ids   = tokenizer("", padding="max_length", max_length=input_ids.shape[-1],
                          return_tensors="pt").input_ids.to(device)

    pos_chunks, neg_chunks = [], []
    for i in range(0, input_ids.shape[-1], max_len):
        pos_chunks.append(text_encoder(input_ids[:, i:i+max_len])[0])
        neg_chunks.append(text_encoder(neg_ids  [:, i:i+max_len])[0])

    prompt_embeds          = torch.cat(pos_chunks, dim=1)
    negative_prompt_embeds = torch.cat(neg_chunks, dim=1)
    return prompt_embeds, negative_prompt_embeds

def generate_stable_diffusion_batch(prompt: str, num_images: int) -> List[Image.Image]:
    images = []
    while len(images) < num_images:
        batch_size = min(3, num_images - len(images))
        p_emb, n_emb = encode_long_prompt(pipe, prompt)
        p_emb = p_emb.repeat(batch_size, 1, 1)
        n_emb = n_emb.repeat(batch_size, 1, 1)

        batch_imgs = pipe(
            prompt_embeds=p_emb,
            negative_prompt_embeds=n_emb
        ).images
        images.extend(batch_imgs)

    return images[:num_images]

# ------------------------------------------------------------------
# File-system helpers
# ------------------------------------------------------------------
def ensure_and_generate(prompt_folder: Path, prompt: str, replace: bool) -> None:
    """
    Creates `prompt_folder` (or refreshes it when --replace) and guarantees
    exactly 9 images named 0-8.jpg inside, generating only what’s missing.
    """
    if replace and prompt_folder.exists():
        shutil.rmtree(prompt_folder)
    prompt_folder.mkdir(parents=True, exist_ok=True)

    missing = [i for i in range(9) if not (prompt_folder / f"{i}.jpg").exists()]
    if not missing:
        print(f"✔ Prompt '{prompt}' already complete.")
        return

    print(f"→ Generating {len(missing)} image(s) for prompt '{prompt}'.")
    new_imgs = generate_stable_diffusion_batch(prompt, len(missing))
    for idx, img in zip(missing, new_imgs):
        img.save(prompt_folder / f"{idx}.jpg")

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    args = parse_args()
    mode         = args.mode
    replace_flag = args.replace

    entigen_prefixes = {
        "aae": "In African American English, ",
        "bre": "In British English, ",
        "che": "In Chicano English, ",
        "ine": "In Indian English, ",
        "sge": "In Singlish, ",
    }
    sae_prefix = "In Standard American English, "

    # ------------------------------------------------------------------
    for dialect in args.dialects:
        data_file = BASE_DIR / "data" / "text" / mode / f"{dialect}.csv"
        img_dir   = BASE_DIR / "data" / "image" / mode / dialect / "stable-diffusion1.5"
        lr_dir    = img_dir / "dialect_imgs"
        hr_dir    = img_dir / "sae_imgs"

        # fresh start if --replace
        if replace_flag and img_dir.exists():
            shutil.rmtree(img_dir)
        lr_dir.mkdir(parents=True, exist_ok=True)
        hr_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(data_file, encoding="unicode_escape")
        dialect_prompts = df["Dialect_Prompt"].tolist()
        sae_prompts     = df["SAE_Prompt"].tolist()

        for dp, sp in tqdm(zip(dialect_prompts, sae_prompts),
                           total=len(df), desc=f"[{dialect}]"):
            if mode == "entigen":
                dp = entigen_prefixes[dialect] + dp
                sp = sae_prefix + sp

            # ------------------------------------------------------------------
            # Folder naming logic
            # ------------------------------------------------------------------
            if mode in {"rewrite_concise", "rewrite_detailed"}:
                dp_dir = lr_dir / hashlib.md5(dp.encode()).hexdigest()
                sp_dir = hr_dir / hashlib.md5(sp.encode()).hexdigest()
            else:
                dp_dir = lr_dir / dp
                sp_dir = hr_dir / sp

            ensure_and_generate(dp_dir, dp, replace_flag)
            ensure_and_generate(sp_dir, sp, replace_flag)

# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
