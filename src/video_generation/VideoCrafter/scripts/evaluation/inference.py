import argparse
import os
import warnings
from pathlib import Path
from tqdm import tqdm
import shutil

import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler

import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from funcs import load_model_checkpoint, load_prompts, save_videos, batch_ddim_sampling
from utils.utils import instantiate_from_config

ENTIGEN_PREFIXES = {
    "aae": "In African American English, ",
    "bre": "In British English, ",
    "che": "In Chicano English, ",
    "ine": "In Indian English, ",
    "sge": "In Singlish, "
}
SAE_PREFIX = "In Standard American English, "

BASE_DIR = Path("/home/multimodal-dialectal-bias")

class PromptDataset(Dataset):
    def __init__(self, csv_path, mode, dialect):
        self.df = pd.read_csv(csv_path, encoding="unicode_escape")
        self.mode = mode
        self.dialect = dialect

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dialect_prompt = row["Dialect_Prompt"]
        sae_prompt = row["SAE_Prompt"]
        print(dialect_prompt, sae_prompt)
        if self.mode == "entigen":
            dialect_prompt = ENTIGEN_PREFIXES[self.dialect] + dialect_prompt
            sae_prompt = SAE_PREFIX + sae_prompt

        return dialect_prompt, sae_prompt

def generate(model, prompt, output_path, fps, args):
    torch.manual_seed(args.seed)

    assert (args.height % 16 == 0) and (args.width % 16 == 0), "Image size must be divisible by 16"
    h, w = args.height // 8, args.width // 8
    frames = model.temporal_length if args.frames < 0 else args.frames
    channels = model.channels

    text_emb = model.get_learned_conditioning([prompt])
    cond = {"c_crossattn": [text_emb], "fps": torch.tensor([fps]).to(model.device).long()}

    noise_shape = [1, channels, frames, h, w]
    batch_samples = batch_ddim_sampling(
        model, cond, noise_shape, args.num_samples,
        args.ddim_steps, args.ddim_eta, args.guidance_scale
    )
    # batch_samples = batch_ddim_sampling(
    #         model, cond, noise_shape, args.n_samples,
    #         args.ddim_steps, args.ddim_eta,
    #         args.unconditional_guidance_scale
    #     )
    save_videos(batch_samples, output_path.parent, [output_path.stem], fps=fps)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dialect", required=True, choices=list(ENTIGEN_PREFIXES.keys()))
    parser.add_argument("--mode", required=True, choices=["concise", "detailed", "entigen", "polysemy"])
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--frames", type=int, default=-1)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--ddim_eta", type=float, default=1.0)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    device = torch.device("cuda", local_rank)
    seed_everything(args.seed + rank)

    csv_path = BASE_DIR / "data" / "text" / args.mode / f"{args.dialect}.csv"
    output_dir = BASE_DIR / "data" / "video" / args.mode / args.dialect / "videocrafter"
    dialect_dir = output_dir / "dialect_videos"
    sae_dir = output_dir / "sae_videos"
    
    if rank == 0:
        if os.path.exists(dialect_dir) and args.overwrite:
            shutil.rmtree(dialect_dir)
            shutil.rmtree(sae_dir)
        dialect_dir.mkdir(parents=True, exist_ok=True)
        sae_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    dataset = PromptDataset(csv_path, args.mode, args.dialect)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=1)

    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())
    model = instantiate_from_config(model_config)
    model = model.to(device)
    model = load_model_checkpoint(model, args.ckpt)
    model.eval()

    for batch in tqdm(dataloader, disable=rank != 0):
        dialect_prompt, sae_prompt = batch[0][0], batch[1][0]

        for j in range(args.num_samples):
            d_path = dialect_dir / dialect_prompt.replace("/", "_").replace(" ", "_") / f"{j}.mp4"
            s_path = sae_dir / sae_prompt.replace("/", "_").replace(" ", "_") / f"{j}.mp4"

            d_path.parent.mkdir(parents=True, exist_ok=True)
            s_path.parent.mkdir(parents=True, exist_ok=True)

            if not d_path.exists():
                generate(model, dialect_prompt, d_path, args.fps, args)
            if not s_path.exists():
                generate(model, sae_prompt, s_path, args.fps, args)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()