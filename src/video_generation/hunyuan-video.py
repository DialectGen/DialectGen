import os
import sys
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import pandas as pd
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video

# ---------------------------
# Global Configuration
# ---------------------------
# Since this script is in project_home/src/image_generation, we go 2 levels up.
BASE_DIR = Path(__file__).resolve().parents[2]

# Define model ID and initialize the transformer and pipeline
model_id = "hunyuanvideo-community/HunyuanVideo"
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id, subfolder="transformer", torch_dtype=torch.bfloat16
)
pipe = HunyuanVideoPipeline.from_pretrained(
    model_id, transformer=transformer, torch_dtype=torch.float16
)
pipe.vae.enable_tiling()
pipe.to("cuda")

# ---------------------------
# Helper Functions
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate videos using HunyuanVideoPipeline for a specified dialect and mode."
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
        help="Mode to use (basic, entigen, or polysemy)."
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

def generate_hunyuan_video(prompt: str, save_dir: Path) -> None:
    """
    Generates a video using HunyuanVideoPipeline and saves it to the specified directory.
    The video is generated with the following parameters:
      - height=320
      - width=512
      - num_frames=61
      - num_inference_steps=30
    The resulting video is exported as "video.mp4" at 15 fps.
    """
    result = pipe(
        prompt=prompt,
        height=320,
        width=512,
        num_frames=61,
        num_inference_steps=30,
    )
    video_frames = result.frames[0]
    output_file = save_dir / "video.mp4"
    export_to_video(video_frames, str(output_file), fps=15)

# ---------------------------
# Main Workflow
# ---------------------------
def main():
    args = parse_args()
    dialect = args.dialect
    mode = args.mode

    # Define input CSV file and output video directory based on mode and dialect.
    data_file = BASE_DIR / "data" / "text" / mode / f"{dialect}.csv"
    video_dir = BASE_DIR / "data" / "video" / mode / f"{dialect}" / "hunyuanvideo"

    # ENTIGEN prompt prefixes mapping for dialect and SAE prompts.
    entigen_prefixes = {
        "aae": "In African American English, ",
        "bre": "In British English, ",
        "che": "In Chicano English, ",
        "ine": "In Indian English, ",
        "sge": "In Singlish, "
    }
    sae_prefix = "In Standard American English, "

    # Prepare output directories for videos.
    prepare_directory(video_dir)
    dialect_vid_dir = video_dir / "dialect_videos"
    sae_vid_dir = video_dir / "sae_videos"
    prepare_directory(dialect_vid_dir)
    prepare_directory(sae_vid_dir)

    # Read CSV data containing prompts.
    df = pd.read_csv(data_file, encoding="unicode_escape")
    dialect_prompts = df["Dialect_Prompt"].tolist()
    sae_prompts = df["SAE_Prompt"].tolist()

    # Iterate over each prompt pair and generate videos.
    for dp, sp in tqdm(zip(dialect_prompts, sae_prompts), total=len(dialect_prompts)):
        if mode == "entigen":
            dp = entigen_prefixes[dialect] + dp
            sp = sae_prefix + sp

        dp_dir = dialect_vid_dir / dp
        sp_dir = sae_vid_dir / sp

        if not dp_dir.exists():
            dp_dir.mkdir(parents=True, exist_ok=True)
            generate_hunyuan_video(dp, dp_dir)
        if not sp_dir.exists():
            sp_dir.mkdir(parents=True, exist_ok=True)
            generate_hunyuan_video(sp, sp_dir)

if __name__ == "__main__":
    main()
