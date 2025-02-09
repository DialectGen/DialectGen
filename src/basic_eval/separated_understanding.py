# Create file structure
import os
import sys
from transformers.models.poolformer.image_processing_poolformer import ImageInput
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Plot similarity matrix
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt



# Load both text-to-image models
dtype = "float32"
import torch
from IPython.display import display, update_display
from min_dalle import MinDalle
from diffusers import StableDiffusionPipeline


# Load the clip model & compute cosine simlarity for MIN-DALLE
import clip
device = "cuda" if torch.cuda.is_available() else "cpu"
CLIP, preprocess = clip.load("ViT-B/32", device=device, jit=False)


import torchmetrics
_ = torch.manual_seed(42)
from torchmetrics.multimodal import CLIPScore
metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")


# ---------------------------------------- Basic Paths ----------------------------------------

# img_dir = "/local1/bryanzhou008/Dialect/data/images/oct_5_basic/aae"
# data_file = "/local1/bryanzhou008/Dialect/data/text/oct_5_csvs/african_american_english_prompts.csv"

# img_dir = "/local1/bryanzhou008/Dialect/data/images/oct_5_basic/bre"
# data_file = "/local1/bryanzhou008/Dialect/data/text/oct_5_csvs/british_english_prompts.csv"

# img_dir = "/local1/bryanzhou008/Dialect/data/images/oct_5_basic/che"
# data_file = "/local1/bryanzhou008/Dialect/data/text/oct_5_csvs/chicano_english_prompts.csv"

# img_dir = "/local1/bryanzhou008/Dialect/data/images/oct_5_basic/ine"
# data_file = "/local1/bryanzhou008/Dialect/data/text/oct_5_csvs/indian_english_prompts.csv"

# img_dir = "/local1/bryanzhou008/Dialect/data/images/oct_5_basic/sge"
# data_file = "/local1/bryanzhou008/Dialect/data/text/oct_5_csvs/singlish_prompts.csv"

# ---------------------------------------------------------------------------------------------


# ---------------------------------------- Simplified ----------------------------------------

# img_dir = "/local1/bryanzhou008/Dialect/data/images/oct_23_simplified/aae"
# data_file = "/local1/bryanzhou008/Dialect/data/text/simplified/aae.csv"

img_dir = "/local1/bryanzhou008/Dialect/data/images/oct_23_simplified/bre"
data_file = "/local1/bryanzhou008/Dialect/data/text/simplified/bre.csv"

# img_dir = "/local1/bryanzhou008/Dialect/data/images/oct_23_simplified/che"
# data_file = "/local1/bryanzhou008/Dialect/data/text/simplified/che.csv"

# img_dir = "/local1/bryanzhou008/Dialect/data/images/oct_23_simplified/ine_v2"
# data_file = "/local1/bryanzhou008/Dialect/data/text/simplified/ine_v2.csv"

# img_dir = "/local1/bryanzhou008/Dialect/data/images/oct_23_simplified/sge"
# data_file = "/local1/bryanzhou008/Dialect/data/text/simplified/sge.csv"

# ---------------------------------------------------------------------------------------------




# Global Variables
# library = "torchmetrics"
library = "openai"


# Read Data
df = pd.read_csv(data_file, encoding='unicode_escape')
dialect_prompts = list(df["Dialect_Prompt"])
sae_prompts = list(df["SAE_Prompt"])
polysemic = list(df["polysemic"])
# dialect_prompts = list(df["DialectPrompt"])[:-2]
# sae_prompts = list(df["SAEPrompt"])[:-2]

# dialect_prompts = [i[28:] for i  in dialect_prompts]
# sae_prompts = [i[29:] for i  in sae_prompts]



def get_average_scores(img_dir, model, dialect, gen_prompt, ref_prompt):
    prompt_dir = os.path.join(img_dir, model, dialect, gen_prompt)
    scores = []
    for i in range(0, 9):
        
        # this is to handle DALLE-mini tending to change "'" to "_" 
        try:
            image = preprocess(Image.open(prompt_dir + f'/{i}.jpg')).unsqueeze(0).to(device)
        except:
            processed_gen_prompt = gen_prompt.replace("'", "_")
            prompt_dir = os.path.join(img_dir, model, dialect, processed_gen_prompt)
            image = preprocess(Image.open(prompt_dir + f'/{i}.jpg')).unsqueeze(0).to(device)
            
            
        text = clip.tokenize([ref_prompt]).to(device)
        with torch.no_grad():
            image_features = CLIP.encode_image(image)
            text_features = CLIP.encode_text(text)
            if library == "openai":
                score = cosine_similarity(image_features.cpu().numpy(), text_features.cpu().numpy())[0][0]
            elif library == "torchmetrics":
                print("this method still needs debugging")
                raise
                score = metric(image, prompt)
            else:
                print("library not defined!")
                raise
            scores.append(score)
    avg_score = sum(scores)/len(scores)
    return avg_score




dialect_total_score_dalle_mini = []
dialect_total_score_dalle = []
dialect_total_score_stable = []
dialect_total_score_stable_2 = []

sae_total_score_dalle_mini = []
sae_total_score_dalle = []
sae_total_score_stable = []
sae_total_score_stable_2 = []


for i in tqdm(range(len(dialect_prompts))):
    
    if polysemic[i] == 0:
        pass
    else:
        # dialect
        dialect_prompt = dialect_prompts[i]
        print("dialectal prompt is:", dialect_prompt)

        # test
        sae_prompt = sae_prompts[i]
        print("sae prompt is:", sae_prompt)
        
        dialect_dalle_mini_score = get_average_scores(img_dir, "dalle-mini", "dialect_imgs", dialect_prompt, sae_prompt)
        dialect_total_score_dalle_mini.append(dialect_dalle_mini_score)
        print("dialect_dalle_score:", dialect_dalle_mini_score)

        dialect_dalle_score = get_average_scores(img_dir, "minDALL-E", "dialect_imgs", dialect_prompt, sae_prompt)
        dialect_total_score_dalle.append(dialect_dalle_score)
        print("dialect_dalle_score:", dialect_dalle_score)

        dialect_stable_score = get_average_scores(img_dir, "stable-diffusion1.4", "dialect_imgs", dialect_prompt, sae_prompt)
        dialect_total_score_stable.append(dialect_stable_score)
        print("dialect_stable_score:", dialect_stable_score)
        
        dialect_stable_score_2 = get_average_scores(img_dir, "stable-diffusion2.1", "dialect_imgs", dialect_prompt, sae_prompt)
        dialect_total_score_stable_2.append(dialect_stable_score_2)
        print("dialect_stable_score:", dialect_stable_score_2)


        print("\n")

        # sae
        sae_prompt = sae_prompts[i]
        print("sae prompt is:", sae_prompt)
        
        
        sae_dalle_mini_score = get_average_scores(img_dir, "dalle-mini", "sae_imgs", sae_prompt, sae_prompt)
        sae_total_score_dalle_mini.append(sae_dalle_mini_score)
        print("dialect_dalle_score:", sae_dalle_mini_score)


        sae_dalle_score = get_average_scores(img_dir, "minDALL-E", "sae_imgs", sae_prompt, sae_prompt)
        sae_total_score_dalle.append(sae_dalle_score)
        print("sae_dalle_score:", sae_dalle_score)


        sae_stable_score = get_average_scores(img_dir, "stable-diffusion1.4", "sae_imgs", sae_prompt, sae_prompt)
        sae_total_score_stable.append(sae_stable_score)
        print("sae_stable_score:", sae_stable_score)
        
        sae_stable_score_2 = get_average_scores(img_dir, "stable-diffusion2.1", "sae_imgs", sae_prompt, sae_prompt)
        sae_total_score_stable_2.append(sae_stable_score_2)
        print("sae_stable_score:", sae_stable_score_2)


        print("\n")


print("-------------------final results-------------------")
print("dialect_total_score_dalle_mini:", sum(dialect_total_score_dalle_mini)/len(dialect_total_score_dalle_mini))
print("dialect_total_score_dalle:", sum(dialect_total_score_dalle)/len(dialect_total_score_dalle)) 
print("dialect_total_score_stable1.4:", sum(dialect_total_score_stable)/len(dialect_total_score_stable))
print("dialect_total_score_stable2.1:", sum(dialect_total_score_stable_2)/len(dialect_total_score_stable_2))

print("sae_total_score_dalle_mini:", sum(sae_total_score_dalle_mini)/len(sae_total_score_dalle_mini))
print("sae_total_score_dalle:", sum(sae_total_score_dalle)/len(sae_total_score_dalle))
print("sae_total_score_stable1.4:", sum(sae_total_score_stable)/len(sae_total_score_stable))
print("sae_total_score_stable2.1:", sum(sae_total_score_stable_2)/len(sae_total_score_stable_2))
