import os
import sys
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import pandas as pd
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from PIL import Image  # (Optional: in case you need additional image processing)

# ---------------------------
# Global Configuration
# ---------------------------
# Since this script is in project_home/src/image_generation, we go 2 levels up.
BASE_DIR = Path(__file__).resolve().parents[2]

# Define and initialize the CogVideoX pipeline
model_id = "THUDM/CogVideoX1.5-5B"
pipe = CogVideoXPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload()
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

# pipe.to("cuda")

# Enable any offloading or VAE options as required (if available)
# (The sample does not include additional VAE configuration)

# ---------------------------
# Helper Functions
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate videos using CogVideoXPipeline for a specified dialect and prompt type."
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
        choices=["basic", "complex", "entigen", "polysemy"],
        help="Type of prompt to use (basic, complex, entigen, or polysemy)."
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

def generate_video(prompt: str, save_dir: Path) -> None:
    """
    Generates a video using the CogVideoX pipeline and saves it to the specified directory.
    The video is generated with:
      - num_videos_per_prompt=1
      - num_inference_steps=50
      - num_frames=81
      - guidance_scale=6
      - A fixed random seed for reproducibility.
    The resulting frames are then exported to a video file.
    """
    result = pipe(
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=10,
        num_frames=10,
        guidance_scale=6,
        generator=torch.Generator(device="cuda").manual_seed(42)
    )
    # Retrieve the video frames (assuming a list of videos is returned)
    video_frames = result.frames[0]
    # Define the output file path (e.g., "video.mp4")
    output_file = save_dir / "video.mp4"
    export_to_video(video_frames, str(output_file), fps=8)

# ---------------------------
# Main Workflow
# ---------------------------
def main():
    args = parse_args()
    dialect = args.dialect
    mode = args.mode

    # Define the input CSV file path and output video directory path based on mode and dialect.
    data_file = BASE_DIR / "data" / "text" / mode / f"{dialect}.csv"
    video_dir = BASE_DIR / "data" / "video" / mode / f"{dialect}" / "cogvideox1.5"
    
    # ENTIGEN prompt prefixes mapping for dialect and SAE prompts.
    entigen_prefixes = {
        "aae": "In African American English, ",
        "bre": "In British English, ",
        "che": "In Chicano English, ",
        "ine": "In Indian English, ",
        "sge": "In Singlish, "
    }
    sae_prefix = "In Standard American English, "

    # Prepare the output directory and subdirectories for dialect and SAE videos.
    prepare_directory(video_dir)
    dialect_vid_dir = video_dir / "dialect_videos"
    sae_vid_dir = video_dir / "sae_videos"
    prepare_directory(dialect_vid_dir)
    prepare_directory(sae_vid_dir)

    # Read the CSV file containing prompts.
    df = pd.read_csv(data_file, encoding="unicode_escape")
    dialect_prompts = df["Dialect_Prompt"].tolist()
    sae_prompts = df["SAE_Prompt"].tolist()

    # Iterate over each pair of prompts and generate videos.
    for dp, sp in tqdm(zip(dialect_prompts, sae_prompts), total=len(dialect_prompts)):
        # If mode is "entigen", prepend the corresponding prefixes.
        if mode == "entigen":
            dp = entigen_prefixes[dialect] + dp
            sp = sae_prefix + sp

        # Create subdirectories for each prompt (using the prompt text as directory names)
        dp_dir = dialect_vid_dir / dp
        sp_dir = sae_vid_dir / sp

        if not dp_dir.exists():
            dp_dir.mkdir(parents=True, exist_ok=True)
            generate_video(dp, dp_dir)
        if not sp_dir.exists():
            sp_dir.mkdir(parents=True, exist_ok=True)
            generate_video(sp, sp_dir)

if __name__ == "__main__":
    main()
