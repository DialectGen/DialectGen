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
    data_path = os.path.join(args.data_dir, args.dialect, "test.csv")
    df = pd.read_csv(data_path, encoding="unicode_escape")
    polysemic = df["polysemic"].tolist()
    polysemy_prompts = [item for i, item in enumerate(df["Polysemy_Prompt"].tolist()) if polysemic[i]]
    
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

    print(">>> encoder name: " + args.encoder.split("/")[-1])
    for i in range(len(polysemy_prompts)):
        polysemy_prompt = polysemy_prompts[i]
        
        model_base_name = args.model.split("/")[-1] + "/"
        if "best" in args.encoder or "last" in args.encoder:
            model_base_name += "/".join(args.encoder.split("/")[-2:])
        else:
            model_base_name += args.encoder.split("/")[-1]
        if not args.swap:
            model_base_name = args.model.split("/")[-1]
        
        polysemy_dir = os.path.join(base_dir, model_base_name, args.mode, f"{args.dialect}_polysemy", polysemy_prompt)
        if os.path.isdir(polysemy_dir):
            polysemy_imgs_exist = all(os.path.isfile(os.path.join(polysemy_dir, f"{k}.jpg")) for k in range(NUM_SAMPLES))
            if polysemy_imgs_exist:
                continue
        os.makedirs(polysemy_dir, exist_ok=True)
        
        for k in range(NUM_SAMPLES):
            image = pipe(polysemy_prompt).images[0]
            image_path = os.path.join(polysemy_dir, f"{k}.jpg")
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
