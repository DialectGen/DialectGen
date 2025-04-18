import argparse
import logging
import os
import sys
import warnings
import random
from pathlib import Path
from tqdm import tqdm

import pandas as pd
import torch
import torch.distributed as dist

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES
from wan.utils.utils import cache_video
from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment

ENTIGEN_PREFIXES = {
    "aae": "In African American English, ",
    "bre": "In British English, ",
    "che": "In Chicano English, ",
    "ine": "In Indian English, ",
    "sge": "In Singlish, "
}
SAE_PREFIX = "In Standard American English, "

BASE_DIR = Path(__file__).resolve().parents[3]

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dialect", required=True, choices=list(ENTIGEN_PREFIXES.keys()))
    parser.add_argument("--mode", required=True, choices=["concise", "detailed", "entigen", "polysemy"])
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--task", type=str, default="t2v-14B", choices=list(WAN_CONFIGS.keys()))
    parser.add_argument("--size", type=str, default="1280*720", choices=list(SIZE_CONFIGS.keys()))
    parser.add_argument("--frame_num", type=int, default=81)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--sample_shift", type=float, default=5.0)
    parser.add_argument("--sample_guide_scale", type=float, default=5.0)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--offload_model", type=str2bool, default=True)
    parser.add_argument("--ulysses_size", type=int, default=1)
    parser.add_argument("--ring_size", type=int, default=1)
    parser.add_argument("--t5_fsdp", action="store_true", default=False)
    parser.add_argument("--t5_cpu", action="store_true", default=False)
    parser.add_argument("--dit_fsdp", action="store_true", default=False)
    return parser.parse_args()

def generate(wan_t2v, prompt, video_path, args):
    cfg = WAN_CONFIGS[args.task]
    video = wan_t2v.generate(
        prompt,
        size=SIZE_CONFIGS[args.size],
        frame_num=args.frame_num,
        shift=args.sample_shift,
        sample_solver='unipc',
        sampling_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale,
        seed=args.base_seed,
        offload_model=args.offload_model,
    )
    if video is not None and dist.get_rank() == 0:
        cache_video(
            tensor=video[None],
            save_file=str(video_path),
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
    return str(video_path)

def main():
    args = _parse_args()

    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    
    # cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    # if cuda_visible_devices:
    #     visible_devices = [int(x) for x in cuda_visible_devices.split(",")]
    #     torch.cuda.set_device(visible_devices[local_rank])
    # else:
    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend="nccl", init_method="env://")

    if args.ulysses_size > 1 or args.ring_size > 1:
        init_distributed_environment(rank=rank, world_size=world_size)
        initialize_model_parallel(
            sequence_parallel_degree=world_size,
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    csv_file = BASE_DIR / "data" / "text" / args.mode / f"{args.dialect}.csv"
    output_base = BASE_DIR / "data" / "video" / args.mode / args.dialect / "wan"
    dialect_dir = output_base / "dialect_videos"
    sae_dir = output_base / "sae_videos"

    if rank == 0:
        dialect_dir.mkdir(parents=True, exist_ok=True)
        sae_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    df = pd.read_csv(csv_file)

    cfg = WAN_CONFIGS[args.task]
    wan_t2v = wan.WanT2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu,
    )

    dialect_paths = []
    sae_paths = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        dialect_prompt = row["Dialect_Prompt"]
        sae_prompt = row["SAE_Prompt"]

        if args.mode == "entigen":
            dialect_prompt = ENTIGEN_PREFIXES[args.dialect] + dialect_prompt
            sae_prompt = SAE_PREFIX + sae_prompt


        for j in range(1):
            d_name = f"{j}.mp4"
            s_name = f"{j}.mp4"

            (dialect_dir / dialect_prompt).mkdir(parents=True, exist_ok=True)
            (sae_dir / sae_prompt).mkdir(parents=True, exist_ok=True)
            d_path = dialect_dir / dialect_prompt / d_name
            s_path = sae_dir / sae_prompt / s_name

            if not d_path.exists():
                generate(wan_t2v, dialect_prompt, d_path, args)
            if not s_path.exists():
                generate(wan_t2v, sae_prompt, s_path, args)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
