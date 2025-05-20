import argparse
import os, sys
import warnings
from pathlib import Path
from tqdm import tqdm
import shutil
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

sys.path.append(os.path.abspath("./Open-Sora"))
sys.path.append(os.path.abspath("../Open-Sora"))

from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.datasets import save_sample
from opensora.datasets.aspect import get_image_size, get_num_frames
from opensora.models.text_encoder.t5 import text_preprocessing
from opensora.utils.inference_utils import prepare_multi_resolution_info, apply_mask_strategy
from opensora.utils.misc import to_torch_dtype
from opensora.utils.inference_utils import (
    add_watermark,
    append_generated,
    append_score_to_prompts,
    apply_mask_strategy,
    collect_references_batch,
    dframe_to_frame,
    extract_json_from_prompts,
    extract_prompts_loop,
    get_save_path_name,
    load_prompts,
    merge_prompt,
    prepare_multi_resolution_info,
    refine_prompts_by_openai,
    split_prompt,
)

ENTIGEN_PREFIXES = {
    "aae": "In African American English, ",
    "bre": "In British English, ",
    "che": "In Chicano English, ",
    "ine": "In Indian English, ",
    "sge": "In Singlish, "
}
SAE_PREFIX = "In Standard American English, "

BASE_DIR = Path("home/multimodal-dialectal-bias")

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
        if self.mode == "entigen":
            dialect_prompt = ENTIGEN_PREFIXES[self.dialect] + dialect_prompt
            sae_prompt = SAE_PREFIX + sae_prompt
        return dialect_prompt, sae_prompt


def generate(
    model,
    text_encoder,
    vae,
    scheduler,
    prompt: str,
    output_path: Path,
    cfg,
    device,
    dtype,
):
    # Disable gradients & speed up
    with torch.inference_mode():
        # 1) build a 1‑element batch
        batch_prompts = [prompt]

        # 2) parse out JSON refs & mask strategies
        batch_prompts, refs, ms = extract_json_from_prompts(
            batch_prompts,
            [""] * 1,
            [""] * 1,
        )

        # 3) text preprocessing
        batch_prompts = [text_preprocessing(p) for p in batch_prompts]

        # 4) compute model_args exactly as in your batch code
        image_size = get_image_size(cfg.resolution, cfg.aspect_ratio)
        num_frames = get_num_frames(cfg.num_frames)
        model_args = prepare_multi_resolution_info(
            cfg.multi_resolution,
            batch_size=1,
            image_size=image_size,
            num_frames=num_frames,
            fps=cfg.fps,
            device=device,
            dtype=dtype,
        )

        # 5) sample a batch of latents z
        latent_size = vae.get_latent_size((num_frames, *image_size))
        z = torch.randn(
            1,
            vae.out_channels,
            *latent_size,
            device=device,
            dtype=dtype,
        )

        # 6) build masks
        masks = apply_mask_strategy(z, refs, ms, loop_i=0)

        # 7) call scheduler.sample
        samples = scheduler.sample(
            model,
            text_encoder,
            z=z,
            prompts=batch_prompts,
            device=device,
            additional_args=model_args,
            progress=False,
            mask=masks,
        )

        # free z & masks before decode
        del z, masks
        torch.cuda.empty_cache()

        # 8) decode to video frames
        videos = vae.decode(samples.to(dtype), num_frames=num_frames)

        # free samples before saving
        del samples
        torch.cuda.empty_cache()

        # 9) save the single video (move to CPU to avoid holding GPU mem)
        video = videos[0].cpu()
        save_stem = str(output_path.with_suffix(""))  # yields ".../0" with no ".mp4"
        save_sample(
            video,
            save_path=save_stem,                    # no extension here
            fps=cfg.get("save_fps", cfg.fps),
        )

        # final cleanup
        del videos, video
        torch.cuda.empty_cache()
def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--dialect", required=True, choices=list(ENTIGEN_PREFIXES.keys()))
    # parser.add_argument("--mode", required=True, choices=["concise", "detailed", "entigen", "polysemy"])
    # parser.add_argument("--overwrite", action="store_true", default=False)
    # parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--num_samples", type=int, default=1)
    # cfg = parser.parse_cfg()

    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    
    cfg = parse_configs(training=False)
    print(cfg.num_frames)

    device = torch.device("cuda", local_rank)
    seed_everything(cfg.seed + rank)

    csv_path = BASE_DIR / "data" / "text" / cfg.mode / f"{cfg.dialect}.csv"
    output_dir = BASE_DIR / "data" / "video" / cfg.mode / cfg.dialect / "opensora"
    dialect_dir = output_dir / "dialect_videos"
    sae_dir = output_dir / "sae_videos"

    if rank == 0:
        if os.path.exists(dialect_dir) and cfg.overwrite:
            shutil.rmtree(dialect_dir)
            shutil.rmtree(sae_dir)
        dialect_dir.mkdir(parents=True, exist_ok=True)
        sae_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    dataset = PromptDataset(csv_path, cfg.mode, cfg.dialect)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=1)

    dtype = to_torch_dtype("bf16")

    text_encoder = build_module(cfg.text_encoder, MODELS, device=device)
    vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()
    model = build_module(
        cfg.model, MODELS,
        input_size=vae.get_latent_size((int(cfg.num_frames), *get_image_size(cfg.resolution, cfg.aspect_ratio))),
        in_channels=vae.out_channels,
        caption_channels=text_encoder.output_dim,
        model_max_length=text_encoder.model_max_length
    ).to(device, dtype).eval()
    text_encoder.y_embedder = model.y_embedder
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    for batch in tqdm(dataloader, disable=rank != 0):
        dialect_prompt, sae_prompt = batch[0][0], batch[1][0]

        for j in range(1):
            d_path = dialect_dir / dialect_prompt.replace("/", "_").replace(" ", "_") / f"{j}.mp4"
            s_path = sae_dir / sae_prompt.replace("/", "_").replace(" ", "_") / f"{j}.mp4"

            # print(d_path.parent)
            d_path.parent.mkdir(parents=True, exist_ok=True)
            s_path.parent.mkdir(parents=True, exist_ok=True)

            if not d_path.exists():
                generate(model, text_encoder, vae, scheduler, dialect_prompt, d_path, cfg, device, dtype)
            if not s_path.exists():
                generate(model, text_encoder, vae, scheduler, sae_prompt, s_path, cfg, device, dtype)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
