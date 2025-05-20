import argparse
import os
import shutil
import sys
from pathlib import Path
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from tqdm import tqdm
from diffusers import CogVideoXPipeline, CogVideoXDPMScheduler
from diffusers.utils import export_to_video

# ---------------------------
# Configuration
# ---------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
ENTIGEN_PREFIXES = {
    "aae": "In African American English, ",
    "bre": "In British English, ",
    "che": "In Chicano English, ",
    "ine": "In Indian English, ",
    "sge": "In Singlish, "
}
SAE_PREFIX = "In Standard American English, "

# ---------------------------
# Dataset Class
# ---------------------------
class PromptDataset(Dataset):
    def __init__(self, data_file, mode, dialect):
        self.df = pd.read_csv(data_file, encoding="unicode_escape")
        self.mode = mode
        self.dialect = dialect
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dialect_prompt = row["Dialect_Prompt"]
        sae_prompt = row["SAE_Prompt"]
        
        if self.mode == "entigen":
            dialect_prompt = ENTIGEN_PREFIXES[self.dialect] + dialect_prompt
            sae_prompt = SAE_PREFIX + sae_prompt
            
        return self.process_prompt(dialect_prompt), self.process_prompt(sae_prompt)
    
    def process_prompt(self, prompt):
        clean_prompt = prompt#.strip()#.replace("/", "_").replace(" ", "_")
        return {
            "text": prompt,
            "dir_name": clean_prompt[:100]  # Truncate to avoid long paths
        }

# ---------------------------
# Main Functions
# ---------------------------
def prepare_directory(args, path: Path):
    path.mkdir(parents=True, exist_ok=True)

def generate_video(pipe, prompt, output_dir, seed):
    
    for i in range(5):
        output_path = output_dir / prompt['dir_name'][0] / f"{i}.mp4"
        (output_dir / prompt['dir_name'][0]).mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            continue
        result = pipe(
            prompt=prompt["text"][0],
            num_videos_per_prompt=1,
            num_inference_steps=10,
            num_frames=10,
            guidance_scale=6,
            generator=torch.Generator(device="cuda").manual_seed(seed+i)
        )
        export_to_video(result.frames[0], str(output_path), fps=8)

def main():
    # ---------------------------
    # Initialization
    # ---------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--dialect", required=True, choices=["aae", "bre", "che", "ine", "sge"])
    parser.add_argument("--mode", required=True, choices=["concise", "detailed", "entigen", "polysemy"],)
    parser.add_argument("--overwrite", action='store_true')
    parser.add_argument("--model_path", default="THUDM/CogVideoX-5b")
    args = parser.parse_args()

    dist.init_process_group()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # ---------------------------
    # Path Configuration
    # ---------------------------
    data_file = BASE_DIR / "data" / "text" / args.mode / f"{args.dialect}.csv"
    video_dir = BASE_DIR / "data" / "video" / args.mode / args.dialect / "cogvideox"
    
    if local_rank == 0:
        prepare_directory(args,video_dir / "dialect_videos")
        prepare_directory(args,video_dir / "sae_videos")
    # ---------------------------
    # Pipeline Setup
    # ---------------------------
    pipe = CogVideoXPipeline.from_pretrained(
        args.model_path, 
        torch_dtype=torch.bfloat16
    ).to(f"cuda:{local_rank}")
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload(device=f"cuda:{local_rank}")

    # ---------------------------
    # Data Processing
    # ---------------------------
    dataset = PromptDataset(data_file, args.mode, args.dialect)
    sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=1)

    for batch in tqdm(dataloader, disable=local_rank != 0):
        dialect_prompt, sae_prompt = batch
        
        # Generate dialect video
        d_path = generate_video(
            pipe, dialect_prompt,
            video_dir / "dialect_videos",
            seed=42 + dist.get_rank()
        )
        
        # Generate SAE video
        s_path = generate_video(
            pipe, sae_prompt,
            video_dir / "sae_videos",
            seed=42 + dist.get_rank() + 1000
        )

if __name__ == "__main__":
    main()
    
"""
export CUDA_VISIBLE_DEVICES=1,6,7
torchrun \
    --nproc-per-node=3 \
    --master-port=29507 \
    --rdzv-endpoint=localhost:30507\
    cogvideox5.py --dialect aae --mode concise --overwrite

export CUDA_VISIBLE_DEVICES=1,6,7
torchrun \
    --nproc-per-node=3 \
    --master-port=29507 \
    --rdzv-endpoint=localhost:30507\
    cogvideox5.py --dialect aae --mode detailed --overwrite
"""