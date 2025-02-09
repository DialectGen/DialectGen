# Create file structure
import os
import sys
from transformers.models.poolformer.image_processing_poolformer import ImageInput
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import math

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

img_dir = "/local1/bryanzhou008/Dialect/data/images/oct_5_basic/bre"
data_file = "/local1/bryanzhou008/Dialect/data/text/oct_5_csvs/british_english_prompts.csv"

# img_dir = "/local1/bryanzhou008/Dialect/data/images/oct_5_basic/che"
# data_file = "/local1/bryanzhou008/Dialect/data/text/oct_5_csvs/chicano_english_prompts.csv"

# img_dir = "/local1/bryanzhou008/Dialect/data/images/oct_5_basic/ine"
# data_file = "/local1/bryanzhou008/Dialect/data/text/oct_5_csvs/indian_english_prompts.csv"

# img_dir = "/local1/bryanzhou008/Dialect/data/images/oct_5_basic/sge"
# data_file = "/local1/bryanzhou008/Dialect/data/text/oct_5_csvs/singlish_prompts.csv"

# ---------------------------------------------------------------------------------------------


# ---------------------------------------- Entigen Paths ----------------------------------------

# img_dir = "/local1/bryanzhou008/Dialect/data/images/oct_5_entigen/aae"
# data_file = "/local1/bryanzhou008/Dialect/data/text/oct_5_csvs/african_american_english_prompts.csv"

# img_dir = "/local1/bryanzhou008/Dialect/data/images/oct_5_entigen/bre"
# data_file = "/local1/bryanzhou008/Dialect/data/text/oct_5_csvs/british_english_prompts.csv"

# img_dir = "/local1/bryanzhou008/Dialect/data/images/oct_5_entigen/che"
# data_file = "/local1/bryanzhou008/Dialect/data/text/oct_5_csvs/chicano_english_prompts.csv"

# img_dir = "/local1/bryanzhou008/Dialect/data/images/oct_5_entigen/ine"
# data_file = "/local1/bryanzhou008/Dialect/data/text/oct_5_csvs/indian_english_prompts.csv"

# img_dir = "/local1/bryanzhou008/Dialect/data/images/oct_5_entigen/sge"
# data_file = "/local1/bryanzhou008/Dialect/data/text/oct_5_csvs/singlish_prompts.csv"

# ---------------------------------------------------------------------------------------------


# Global Variables
# library = "torchmetrics"
library = "openai"


# Read Data
df = pd.read_csv(data_file)
df = df.loc[df['person_in_prompt'] == 1]  # make sure to select only prompts that involve people
dialect_prompts = list(df["Dialect_Prompt"])
sae_prompts = list(df["SAE_Prompt"])



LIGHT_SKIN_PROMPT = "light-skinned"
DARK_SKIN_PROMPT = "dark-skinned"


def get_average_scores(img_dir, model, dialect, gen_prompt, ref_prompt):
    prompt_dir = os.path.join(img_dir, model, dialect, gen_prompt)
    scores = []
    for i in range(0, 9):
        # this is to handle DALLE-mini tending to change "'" to "_" 
        try:
            image = preprocess(Image.open(prompt_dir + f'/{i}.jpg')).unsqueeze(0).to(device)
        except:
            processed_gen_prompt = gen_prompt.replace("'", "_")
            new_prompt_dir = os.path.join(img_dir, model, dialect, processed_gen_prompt)
            image = preprocess(Image.open(new_prompt_dir + f'/{i}.jpg')).unsqueeze(0).to(device)
        

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





# dialect_total_score_dalle_light_skin = []
# dialect_total_score_stable_light_skin = []

# sae_total_score_dalle_light_skin = []
# sae_total_score_stable_light_skin = []

# dialect_total_score_dalle_dark_skin = []
# dialect_total_score_stable_dark_skin = []

# sae_total_score_dalle_dark_skin = []
# sae_total_score_stable_dark_skin = []


total_dalle_mini_normalized_sae_light_skin = []
total_dalle_normalized_sae_light_skin = []
total_stable_normalized_sae_light_skin = []
total_stable_normalized_sae_light_skin_2 = []

total_dalle_mini_normalized_dialect_light_skin = []
total_dalle_normalized_dialect_light_skin = []
total_stable_normalized_dialect_light_skin = []
total_stable_normalized_dialect_light_skin_2 = []


for i in tqdm(range(len(dialect_prompts))):
    # get the prompts
    dialect_prompt = dialect_prompts[i]
    print("dialectal prompt is:", dialect_prompt)
    sae_prompt = sae_prompts[i]
    print("sae prompt is:", sae_prompt)
    
    
    dialect_dalle_mini_score_light_skin = get_average_scores(img_dir, "dalle-mini", "dialect_imgs", dialect_prompt, LIGHT_SKIN_PROMPT)
    print("dialect_dalle_mini_score_light_skin:", dialect_dalle_mini_score_light_skin)

    dialect_dalle_score_light_skin = get_average_scores(img_dir, "minDALL-E", "dialect_imgs", dialect_prompt, LIGHT_SKIN_PROMPT)
    print("dialect_dalle_score_light_skin:", dialect_dalle_score_light_skin)

    dialect_stable_score_light_skin = get_average_scores(img_dir, "stable-diffusion1.4", "dialect_imgs", dialect_prompt, LIGHT_SKIN_PROMPT)
    print("dialect_stable_score_light_skin:", dialect_stable_score_light_skin)
    
    dialect_stable_score_light_skin_2 = get_average_scores(img_dir, "stable-diffusion2.1", "dialect_imgs", dialect_prompt, LIGHT_SKIN_PROMPT)
    print("dialect_stable_score_light_skin:", dialect_stable_score_light_skin_2)



    dialect_dalle_mini_score_dark_skin = get_average_scores(img_dir, "dalle-mini", "dialect_imgs", dialect_prompt, DARK_SKIN_PROMPT)
    print("dialect_dalle_mini_score_dark_skin:", dialect_dalle_mini_score_dark_skin)
    
    dialect_dalle_score_dark_skin = get_average_scores(img_dir, "minDALL-E", "dialect_imgs", dialect_prompt, DARK_SKIN_PROMPT)
    print("dialect_dalle_score_dark_skin:", dialect_dalle_score_dark_skin)

    dialect_stable_score_dark_skin = get_average_scores(img_dir, "stable-diffusion1.4", "dialect_imgs", dialect_prompt, DARK_SKIN_PROMPT)
    print("dialect_stable_score_dark_skin:", dialect_stable_score_dark_skin)
    
    dialect_stable_score_dark_skin_2 = get_average_scores(img_dir, "stable-diffusion2.1", "dialect_imgs", dialect_prompt, DARK_SKIN_PROMPT)
    print("dialect_stable_score_dark_skin:", dialect_stable_score_dark_skin_2)



    print("\n")

    sae_dalle_mini_score_light_skin = get_average_scores(img_dir, "dalle-mini", "sae_imgs", sae_prompt, LIGHT_SKIN_PROMPT)
    print("sae_dalle_mini_score_light_skin:", sae_dalle_mini_score_light_skin)

    sae_dalle_score_light_skin = get_average_scores(img_dir, "minDALL-E", "sae_imgs", sae_prompt, LIGHT_SKIN_PROMPT)
    print("sae_dalle_score_light_skin:", sae_dalle_score_light_skin)

    sae_stable_score_light_skin = get_average_scores(img_dir, "stable-diffusion1.4", "sae_imgs", sae_prompt, LIGHT_SKIN_PROMPT)
    print("sae_stable_score_light_skin:", sae_stable_score_light_skin)
    
    sae_stable_score_light_skin_2 = get_average_scores(img_dir, "stable-diffusion2.1", "sae_imgs", sae_prompt, LIGHT_SKIN_PROMPT)
    print("sae_stable_score_light_skin:", sae_stable_score_light_skin_2)



    sae_dalle_mini_score_dark_skin = get_average_scores(img_dir, "dalle-mini", "sae_imgs", sae_prompt, DARK_SKIN_PROMPT)
    print("sae_dalle_mini_score_dark_skin:", sae_dalle_mini_score_dark_skin)

    sae_dalle_score_dark_skin = get_average_scores(img_dir, "minDALL-E", "sae_imgs", sae_prompt, DARK_SKIN_PROMPT)
    print("sae_dalle_score_dark_skin:", sae_dalle_score_dark_skin)

    sae_stable_score_dark_skin = get_average_scores(img_dir, "stable-diffusion1.4", "sae_imgs", sae_prompt, DARK_SKIN_PROMPT)
    print("sae_stable_score_dark_skin:", sae_stable_score_dark_skin)
    
    sae_stable_score_dark_skin_2 = get_average_scores(img_dir, "stable-diffusion2.1", "sae_imgs", sae_prompt, DARK_SKIN_PROMPT)
    print("sae_stable_score_dark_skin:", sae_stable_score_dark_skin_2)


    print("\n")


    # The normalized drop: e^S_AA/(e^S_AA+e^S_AB) - e^S_BB/(e^S_BA+e^S_BB)

    dalle_mini_normalized_sae_light_skin = math.exp(sae_dalle_mini_score_light_skin*100)/(math.exp(sae_dalle_mini_score_light_skin*100) + math.exp(sae_dalle_mini_score_dark_skin*100)) 
    dalle_normalized_sae_light_skin = math.exp(sae_dalle_score_light_skin*100)/(math.exp(sae_dalle_score_light_skin*100) + math.exp(sae_dalle_score_dark_skin*100)) 
    stable_normalized_sae_light_skin = math.exp(sae_stable_score_light_skin*100)/(math.exp(sae_stable_score_light_skin*100) + math.exp(sae_stable_score_dark_skin*100))
    stable_normalized_sae_light_skin_2 = math.exp(sae_stable_score_light_skin_2*100)/(math.exp(sae_stable_score_light_skin_2*100) + math.exp(sae_stable_score_dark_skin_2*100))

    dalle_mini_normalized_dialect_light_skin = math.exp(dialect_dalle_mini_score_light_skin*100)/(math.exp(dialect_dalle_mini_score_light_skin*100) + math.exp(dialect_dalle_mini_score_dark_skin*100)) 
    dalle_normalized_dialect_light_skin = math.exp(dialect_dalle_score_light_skin*100)/(math.exp(dialect_dalle_score_light_skin*100) + math.exp(dialect_dalle_score_dark_skin*100)) 
    stable_normalized_dialect_light_skin = math.exp(dialect_stable_score_light_skin*100)/(math.exp(dialect_stable_score_light_skin*100) + math.exp(dialect_stable_score_dark_skin*100))
    stable_normalized_dialect_light_skin_2 = math.exp(dialect_stable_score_light_skin_2*100)/(math.exp(dialect_stable_score_light_skin_2*100) + math.exp(dialect_stable_score_dark_skin_2*100))

    # math.exp(0.30)/(math.exp(0.30) + math.exp(0.25))

    total_dalle_mini_normalized_sae_light_skin.append(dalle_mini_normalized_sae_light_skin)
    total_dalle_normalized_sae_light_skin.append(dalle_normalized_sae_light_skin)
    total_stable_normalized_sae_light_skin.append(stable_normalized_sae_light_skin)
    total_stable_normalized_sae_light_skin_2.append(stable_normalized_sae_light_skin_2)
    
    total_dalle_mini_normalized_dialect_light_skin.append(dalle_mini_normalized_dialect_light_skin)
    total_dalle_normalized_dialect_light_skin.append(dalle_normalized_dialect_light_skin)
    total_stable_normalized_dialect_light_skin.append(stable_normalized_dialect_light_skin)
    total_stable_normalized_dialect_light_skin_2.append(stable_normalized_dialect_light_skin_2)


print("-------------------final results-------------------")
# print("dialect_total_score_dalle:", sum(dialect_total_score_dalle)/len(dialect_total_score_dalle)) 
# print("dialect_total_score_stable:", sum(dialect_total_score_stable)/len(dialect_total_score_stable))

# print("sae_total_score_dalle:", sum(sae_total_score_dalle)/len(sae_total_score_dalle))
# print("sae_total_score_stable:", sum(sae_total_score_stable)/len(sae_total_score_stable))

print("total_dalle_mini_normalized_sae_light_skin:", sum(total_dalle_mini_normalized_sae_light_skin)/len(total_dalle_mini_normalized_sae_light_skin))
print("total_dalle_normalized_sae_light_skin:", sum(total_dalle_normalized_sae_light_skin)/len(total_dalle_normalized_sae_light_skin))
print("total_stable_normalized_sae_light_skin1.4:", sum(total_stable_normalized_sae_light_skin)/len(total_stable_normalized_sae_light_skin))
print("total_stable_normalized_sae_light_skin2.1:", sum(total_stable_normalized_sae_light_skin_2)/len(total_stable_normalized_sae_light_skin_2))

print("total_dalle_mini_normalized_dialect_light_skin:", sum(total_dalle_mini_normalized_dialect_light_skin)/len(total_dalle_mini_normalized_dialect_light_skin))
print("total_dalle_normalized_dialect_light_skin:", sum(total_dalle_normalized_dialect_light_skin)/len(total_dalle_normalized_dialect_light_skin))
print("total_stable_normalized_dialect_light_skin1.4:", sum(total_stable_normalized_dialect_light_skin)/len(total_stable_normalized_dialect_light_skin))
print("total_stable_normalized_dialect_light_skin2.1:", sum(total_stable_normalized_dialect_light_skin_2)/len(total_stable_normalized_dialect_light_skin_2))