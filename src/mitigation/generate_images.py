import os
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel
import pandas as pd
from argparse import ArgumentParser
from utils.misc import fix_seed
from const import *

NUM_SAMPLES = 9
fix_seed(42)


def main(args):
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model,
    ).to("cuda")
    # pipe.safety_checker = None  # disable safety checker if desired
    if args.swap:
        base_dir = BASE_SWAP_DIR
        text_encoder = CLIPTextModel.from_pretrained(args.encoder, use_safetensors=True, device_map="auto")
        pipe.text_encoder = text_encoder
    else:
        base_dir = BASE_ORIG_DIR

    data_path = os.path.join(args.data_dir, args.dialect, "test.csv")
    df = pd.read_csv(data_path, encoding="unicode_escape")
    dialect_prompts = df["Dialect_Prompt"].tolist()
    sae_prompts = df["SAE_Prompt"].tolist()

    print(">>> encoder name: " + args.encoder.split("/")[-1])
    for i in range(len(dialect_prompts)):
        dialect_prompt = dialect_prompts[i]
        sae_prompt = sae_prompts[i]
        
        model_base_name = args.model.split("/")[-1] + "/"
        if "best" in args.encoder or "last" in args.encoder:
            model_base_name += "/".join(args.encoder.split("/")[-2:])
        else:
            model_base_name += args.encoder.split("/")[-1]
        if not args.swap:
            model_base_name = args.model.split("/")[-1]
        
        dialect_dir = os.path.join(base_dir, model_base_name, args.mode, args.dialect, dialect_prompt)
        sae_dir = os.path.join(base_dir, model_base_name, args.mode, f"{args.dialect}_sae", sae_prompt)
        if os.path.isdir(dialect_dir) and os.path.isdir(sae_dir):
            dialect_imgs_exist = all(os.path.isfile(os.path.join(dialect_dir, f"{k}.jpg")) for k in range(NUM_SAMPLES))
            sae_imgs_exist = all(os.path.isfile(os.path.join(sae_dir, f"{k}.jpg")) for k in range(NUM_SAMPLES))
            if dialect_imgs_exist and sae_imgs_exist:
                continue
        os.makedirs(dialect_dir, exist_ok=True)
        os.makedirs(sae_dir, exist_ok=True)
        
        for k in range(NUM_SAMPLES):
            ## DIALECT
            image = pipe(dialect_prompt).images[0]
            image_path = os.path.join(dialect_dir, f"{k}.jpg")
            image.save(image_path)
            
            ## SAE
            image = pipe(sae_prompt).images[0]
            image_path = os.path.join(sae_dir, f"{k}.jpg")
            image.save(image_path)


def parse_arguments():
    parser = ArgumentParser(description="Generate images using a stable diffusion model.")
    parser.add_argument("--model", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5", 
                        choices=["stable-diffusion-v1-5/stable-diffusion-v1-5"])
    parser.add_argument("--encoder", type=str, default="models/sge/singlish_kl_iac_20ep")
    parser.add_argument("--swap", type=int, default=0, help="Swap in the trained text encoder.")
    parser.add_argument("--data_dir", type=str, default="../../data/text/train_val_test/4-1-1/")
    parser.add_argument("--mode", type=str, default="concise")
    parser.add_argument("--dialect", type=str, default="sge")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    args.data_dir = os.path.join(args.data_dir, args.mode)
    main(args)
