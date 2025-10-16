import os
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel
from argparse import ArgumentParser
from utils.hf_captions import create_hf_coco_dataset
from utils.misc import fix_seed
from const import *

NUM_SAMPLES = 9
fix_seed(42)


def main(args):
    mscoco = create_hf_coco_dataset(CAPTION_FILE_PATH, IMAGE_FOLDER_PATH).select(range(4950, 5000))
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model,
    ).to("cuda")
    
    if args.swap:
        base_dir = MSCOCO_BASE_SWAP_DIR
        text_encoder = CLIPTextModel.from_pretrained(args.encoder, use_safetensors=True, device_map="auto")
        pipe.text_encoder = text_encoder
    else:
        base_dir = MSCOCO_BASE_ORIG_DIR

    prompts = [ct[0] for ct in mscoco["captions"]]

    for prompt in prompts:
        model_base_name = args.model.split("/")[-1] + "/"
        if "best" in args.encoder or "last" in args.encoder:
            model_base_name += "/".join(args.encoder.split("/")[-2:])
        else:
            model_base_name += args.encoder.split("/")[-1]
        if not args.swap:
            model_base_name = args.model.split("/")[-1]
        
        prompt_dir = os.path.join(base_dir, model_base_name, prompt)
        if os.path.isdir(prompt_dir):
            prompt_exist = all(os.path.isfile(os.path.join(prompt_dir, f"{k}.jpg")) for k in range(NUM_SAMPLES))
            if prompt_exist:
                continue
        os.makedirs(prompt_dir, exist_ok=False)
        
        for k in range(NUM_SAMPLES):
            image = pipe(prompt).images[0]
            image_path = os.path.join(prompt_dir, f"{k}.jpg")
            image.save(image_path)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5", 
                        choices=["stable-diffusion-v1-5/stable-diffusion-v1-5"])
    parser.add_argument("--encoder", type=str, default="models/...", help="encoder path")
    parser.add_argument("--swap", type=int, default=0, help="Swap in the trained text encoder.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
