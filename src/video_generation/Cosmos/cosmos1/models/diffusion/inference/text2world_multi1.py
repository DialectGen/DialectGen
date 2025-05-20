# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

import torch

from cosmos1.models.diffusion.inference.inference_utils import add_common_arguments, validate_args
from cosmos1.models.diffusion.inference.world_generation_pipeline import DiffusionText2WorldGenerationPipeline
from cosmos1.utils import log, misc
from cosmos1.utils.io import save_video

from torch.utils.data import Dataset, DataLoader, DistributedSampler

import pandas as pd
import torch.distributed as dist

from tqdm import tqdm

ENTIGEN_PREFIXES = {
    "aae": "In African American English, ",
    "bre": "In British English, ",
    "che": "In Chicano English, ",
    "ine": "In Indian English, ",
    "sge": "In Singlish, "
}
SAE_PREFIX = "In Standard American English, "

BASE_DIR = 'home/multimodal-dialectal-bias/'

torch.set_grad_enabled(False)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cosmos1 Text2World generation script")
    add_common_arguments(parser)
    parser.add_argument("--dialect", required=True, choices=list(ENTIGEN_PREFIXES.keys()))
    parser.add_argument("--mode", required=True, choices=["concise", "detailed", "entigen", "polysemy"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--diffusion_transformer_dir", type=str, default="Cosmos-1.0-Diffusion-7B-Text2World")
    parser.add_argument("--prompt_upsampler_dir", type=str, default="Cosmos-1.0-Prompt-Upsampler-12B-Text2World")
    parser.add_argument("--word_limit_to_skip_upsampler", type=int, default=250)
    return parser.parse_args()

class PromptDataset(Dataset):
    def __init__(self, csv_path, mode, dialect):
        self.df = pd.read_csv(csv_path)
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

def generate_text2video(pipeline, prompt, output_path, args):
    result = pipeline.generate(prompt, args.negative_prompt, args.word_limit_to_skip_upsampler)
    if result is None:
        log.critical("Guardrail blocked text2world generation.")
        return
    video, _ = result
    save_video(
        video=video,
        fps=args.fps,
        H=args.height,
        W=args.width,
        video_save_quality=5,
        video_save_path=str(output_path),
    )
    log.info(f"Saved video to {output_path}")

def main():
    args = parse_arguments()
    misc.set_random_seed(args.seed)
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(local_rank)

    text_path = os.path.join(BASE_DIR, "data", "text", args.mode, f"{args.dialect}.csv")
    video_dir = os.path.join(BASE_DIR, "data", "video", args.mode, args.dialect, "cosmos1")
    dialect_dir = os.path.join(video_dir, "dialect_videos")
    sae_dir = os.path.join(video_dir, "sae_videos")

    if rank == 0:
        os.makedirs(dialect_dir, exist_ok=True)
        os.makedirs(sae_dir, exist_ok=True)
    dist.barrier()

    dataset = PromptDataset(text_path, args.mode, args.dialect)
    sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=1)

    pipeline = DiffusionText2WorldGenerationPipeline(
        inference_type="text2world",
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.diffusion_transformer_dir,
        prompt_upsampler_dir=args.prompt_upsampler_dir,
        enable_prompt_upsampler=not args.disable_prompt_upsampler,
        offload_network=args.offload_diffusion_transformer,
        offload_tokenizer=args.offload_tokenizer,
        offload_text_encoder_model=args.offload_text_encoder_model,
        offload_prompt_upsampler=args.offload_prompt_upsampler,
        offload_guardrail_models=args.offload_guardrail_models,
        guidance=args.guidance,
        num_steps=args.num_steps,
        height=args.height,
        width=args.width,
        fps=args.fps,
        num_video_frames=args.num_video_frames,
        seed=args.seed,
        enable_text_guardrail = False,
        enable_video_guardrail = False,
    )

    for batch in tqdm(dataloader, disable=rank != 0):
        dialect_prompt, sae_prompt = batch[0][0], batch[1][0]
        for j in range(5):
            d_name = f"{j}.mp4"
            s_name = f"{j}.mp4"
            d_path = os.path.join(dialect_dir, dialect_prompt.replace("/", "_").replace(" ", "_"), d_name)
            s_path = os.path.join(sae_dir, sae_prompt.replace("/", "_").replace(" ", "_"), s_name)

            os.makedirs(os.path.dirname(d_path), exist_ok=True)
            os.makedirs(os.path.dirname(s_path), exist_ok=True)

            if not os.path.exists(d_path) or args.overwrite:
                generate_text2video(pipeline, dialect_prompt, d_path, args)
            if not os.path.exists(s_path) or args.overwrite:
                generate_text2video(pipeline, sae_prompt, s_path, args)

if __name__ == "__main__":
    main()
