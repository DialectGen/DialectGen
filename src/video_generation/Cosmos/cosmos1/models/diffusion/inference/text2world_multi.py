# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
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
from cosmos1.utils.io import read_prompts_from_file, save_video

from torch.utils.data import Dataset, DataLoader, DistributedSampler

import pandas as pd
import torch.distributed as dist

from tqdm import tqdm

torch.enable_grad(False)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text to world generation demo script")
    # Add common arguments
    add_common_arguments(parser)

    # Add text2world specific arguments
    parser.add_argument(
        "--diffusion_transformer_dir",
        type=str,
        default="Cosmos-1.0-Diffusion-7B-Text2World",
        help="DiT model weights directory name relative to checkpoint_dir",
        choices=[
            "Cosmos-1.0-Diffusion-7B-Text2World",
            "Cosmos-1.0-Diffusion-14B-Text2World",
        ],
    )
    parser.add_argument(
        "--prompt_upsampler_dir",
        type=str,
        default="Cosmos-1.0-Prompt-Upsampler-12B-Text2World",
        help="Prompt upsampler weights directory relative to checkpoint_dir",
    )

    parser.add_argument(
        "--word_limit_to_skip_upsampler",
        type=int,
        default=250,
        help="Skip prompt upsampler for better robustness if the number of words in the prompt is greater than this value",
    )

    return parser.parse_args()


class CaptionDataset(Dataset):

    def __init__(self, input_file):
        df = pd.read_csv(input_file)
        self.captions = df['caption'].tolist()
        self.upsampled_captions = df['upsampled_caption'].tolist()
    
    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        return {'caption': self.captions[index], 'upsampled_caption': self.upsampled_captions[index]}

def demo(cfg):
    """Run text-to-world generation demo.

    This function handles the main text-to-world generation pipeline, including:
    - Setting up the random seed for reproducibility
    - Initializing the generation pipeline with the provided configuration
    - Processing single or multiple prompts from input
    - Generating videos from text prompts
    - Saving the generated videos and corresponding prompts to disk

    Args:
        cfg (argparse.Namespace): Configuration namespace containing:
            - Model configuration (checkpoint paths, model settings)
            - Generation parameters (guidance, steps, dimensions)
            - Input/output settings (prompts, save paths)
            - Performance options (model offloading settings)

    The function will save:
        - Generated MP4 video files
        - Text files containing the processed prompts

    If guardrails block the generation, a critical log message is displayed
    and the function continues to the next prompt if available.
    """
    misc.set_random_seed(cfg.seed)
    inference_type = "text2world"
    validate_args(cfg, inference_type)

    local_rank = int(os.environ['LOCAL_RANK'])
    device = f'cuda:{local_rank}'
    torch.cuda.set_device(device)

    # Initialize text2world generation model pipeline
    pipeline = DiffusionText2WorldGenerationPipeline(
        inference_type=inference_type,
        checkpoint_dir=cfg.checkpoint_dir,
        checkpoint_name=cfg.diffusion_transformer_dir,
        prompt_upsampler_dir=cfg.prompt_upsampler_dir,
        enable_prompt_upsampler=not cfg.disable_prompt_upsampler,
        offload_network=cfg.offload_diffusion_transformer,
        offload_tokenizer=cfg.offload_tokenizer,
        offload_text_encoder_model=cfg.offload_text_encoder_model,
        offload_prompt_upsampler=cfg.offload_prompt_upsampler,
        offload_guardrail_models=cfg.offload_guardrail_models,
        guidance=cfg.guidance,
        num_steps=cfg.num_steps,
        height=cfg.height,
        width=cfg.width,
        fps=cfg.fps,
        num_video_frames=cfg.num_video_frames,
        seed=cfg.seed,
        enable_text_guardrail = False,
        enable_video_guardrail = False,
    )

    dist.init_process_group()
    dataset = CaptionDataset(cfg.batch_input_path)
    sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=1)

    os.makedirs(cfg.video_save_folder, exist_ok=True)

    for batch in tqdm(dataloader):
        caption = batch['caption'][0]
        video_name = caption.replace(" ", "_")
        if video_name.endswith("."):
            video_name = video_name + "mp4"
        else:
            video_name = video_name + ".mp4"

        video_save_path = os.path.join(cfg.video_save_folder, video_name)
        if not os.path.exists(video_save_path):
            upsampled_caption = batch['upsampled_caption'][0]
            generated_output = pipeline.generate(upsampled_caption, cfg.negative_prompt, cfg.word_limit_to_skip_upsampler)
            if generated_output is None:
                log.critical("Guardrail blocked text2world generation.")
                continue
            video, prompt = generated_output
            # Save video
            save_video(
                video=video,
                fps=cfg.fps,
                H=cfg.height,
                W=cfg.width,
                video_save_quality=5,
                video_save_path=video_save_path,
            )

            log.info(f"Saved video to {video_save_path}")

if __name__ == "__main__":
    args = parse_arguments()
    demo(args)
