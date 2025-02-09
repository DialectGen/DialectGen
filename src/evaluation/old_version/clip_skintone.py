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
img_dir = "/local1/bryanzhou008/Dialect/data/expanded_Singlish_v1/"
data_file = "/local1/bryanzhou008/Dialect/data/prompt_csvs/singlish_prompts.csv"

# img_dir = "/local1/bryanzhou008/Dialect/data/expanded_ChicanoEnglish_v1/"
# data_file = "/local1/bryanzhou008/Dialect/data/prompt_csvs/chicano_english_prompts.csv"

# img_dir = "/local1/bryanzhou008/Dialect/data/expanded_BritishEnglish_v1/"
# data_file = "/local1/bryanzhou008/Dialect/data/prompt_csvs/british_english_prompts.csv"

# img_dir = "/local1/bryanzhou008/Dialect/data/expanded_aae_v1/"
# data_file = "/local1/bryanzhou008/Dialect/data/prompt_csvs/african_american_english_prompts_with_people.csv"

# img_dir = "/local1/bryanzhou008/Dialect/data/test_269_aae_entigen"
# data_file = "/local1/bryanzhou008/Dialect/data/prompt_csvs/aae269_entigen.csv"

# img_dir = "/local1/bryanzhou008/Dialect/data/test_269_aae"
# data_file = "/local1/bryanzhou008/Dialect/data/prompt_csvs/aae269.csv"

# ---------------------------------------------------------------------------------------------




# Global Variables
# library = "torchmetrics"
library = "openai"


# Read Data
df = pd.read_csv(data_file)
dialect_prompts = list(df["DialectPrompt"])
sae_prompts = list(df["SAEPrompt"])
# dialect_prompts = list(df["DialectPrompt"])[:-2]
# sae_prompts = list(df["SAEPrompt"])[:-2]

# dialect_prompts = [i[28:] for i  in dialect_prompts]
# sae_prompts = [i[29:] for i  in sae_prompts]

LIGHT_SKIN_PROMPT = "light-skinned"
DARK_SKIN_PROMPT = "dark-skinned"


def get_average_scores(img_dir, model, dialect, gen_prompt, ref_prompt):
    prompt_dir = os.path.join(img_dir, model, dialect, gen_prompt)
    scores = []
    for i in range(0, 9):
        image = preprocess(Image.open(prompt_dir + f'/{i}.jpg')).unsqueeze(0).to(device)

        # if dialect == "sae_imgs":
        #     prompt = prompt[29:]
        # elif dialect == "dialect_imgs":
        #     prompt = prompt[28:]
        # else:
        #     print(dialect)
        #     raise

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



total_dalle_normalized_sae_light_skin = []
total_stable_normalized_sae_light_skin = []
total_dalle_normalized_dialect_light_skin = []
total_stable_normalized_dialect_light_skin = []


for i in tqdm(range(len(dialect_prompts))):
    # get the prompts
    dialect_prompt = dialect_prompts[i]
    print("dialectal prompt is:", dialect_prompt)

    sae_prompt = sae_prompts[i]
    print("sae prompt is:", sae_prompt)

    dialect_dalle_score_light_skin = get_average_scores(img_dir, "minDALL-E", "dialect_imgs", dialect_prompt, LIGHT_SKIN_PROMPT)
    print("dialect_dalle_score_light_skin:", dialect_dalle_score_light_skin)

    dialect_stable_score_light_skin = get_average_scores(img_dir, "stable-diffusion", "dialect_imgs", dialect_prompt, LIGHT_SKIN_PROMPT)
    print("dialect_stable_score_light_skin:", dialect_stable_score_light_skin)

    dialect_dalle_score_dark_skin = get_average_scores(img_dir, "minDALL-E", "dialect_imgs", dialect_prompt, DARK_SKIN_PROMPT)
    print("dialect_dalle_score_dark_skin:", dialect_dalle_score_dark_skin)

    dialect_stable_score_dark_skin = get_average_scores(img_dir, "stable-diffusion", "dialect_imgs", dialect_prompt, DARK_SKIN_PROMPT)
    print("dialect_stable_score_dark_skin:", dialect_stable_score_dark_skin)

    print("\n")

    sae_dalle_score_light_skin = get_average_scores(img_dir, "minDALL-E", "sae_imgs", sae_prompt, LIGHT_SKIN_PROMPT)
    print("sae_dalle_score_light_skin:", sae_dalle_score_light_skin)


    sae_stable_score_light_skin = get_average_scores(img_dir, "stable-diffusion", "sae_imgs", sae_prompt, LIGHT_SKIN_PROMPT)
    print("sae_stable_score_light_skin:", sae_stable_score_light_skin)


    sae_dalle_score_dark_skin = get_average_scores(img_dir, "minDALL-E", "sae_imgs", sae_prompt, DARK_SKIN_PROMPT)
    print("sae_dalle_score_dark_skin:", sae_dalle_score_dark_skin)


    sae_stable_score_dark_skin = get_average_scores(img_dir, "stable-diffusion", "sae_imgs", sae_prompt, DARK_SKIN_PROMPT)
    print("sae_stable_score_dark_skin:", sae_stable_score_dark_skin)


    print("\n")


    # The normalized drop: e^S_AA/(e^S_AA+e^S_AB) - e^S_BB/(e^S_BA+e^S_BB)

    dalle_normalized_sae_light_skin = math.exp(sae_dalle_score_light_skin*100)/(math.exp(sae_dalle_score_light_skin*100) + math.exp(sae_dalle_score_dark_skin*100)) 
    stable_normalized_sae_light_skin = math.exp(sae_stable_score_light_skin*100)/(math.exp(sae_stable_score_light_skin*100) + math.exp(sae_stable_score_dark_skin*100))

    dalle_normalized_dialect_light_skin = math.exp(dialect_dalle_score_light_skin*100)/(math.exp(dialect_dalle_score_light_skin*100) + math.exp(dialect_dalle_score_dark_skin*100)) 
    stable_normalized_dialect_light_skin = math.exp(dialect_stable_score_light_skin*100)/(math.exp(dialect_stable_score_light_skin*100) + math.exp(dialect_stable_score_dark_skin*100))

    # math.exp(0.30)/(math.exp(0.30) + math.exp(0.25))

    total_dalle_normalized_sae_light_skin.append(dalle_normalized_sae_light_skin)
    total_stable_normalized_sae_light_skin.append(stable_normalized_sae_light_skin)
    total_dalle_normalized_dialect_light_skin.append(dalle_normalized_dialect_light_skin)
    total_stable_normalized_dialect_light_skin.append(stable_normalized_dialect_light_skin)


print("-------------------final results-------------------")
# print("dialect_total_score_dalle:", sum(dialect_total_score_dalle)/len(dialect_total_score_dalle)) 
# print("dialect_total_score_stable:", sum(dialect_total_score_stable)/len(dialect_total_score_stable))

# print("sae_total_score_dalle:", sum(sae_total_score_dalle)/len(sae_total_score_dalle))
# print("sae_total_score_stable:", sum(sae_total_score_stable)/len(sae_total_score_stable))


print("total_dalle_normalized_sae_light_skin:", sum(total_dalle_normalized_sae_light_skin)/len(total_dalle_normalized_sae_light_skin))
print("total_stable_normalized_sae_light_skin:", sum(total_stable_normalized_sae_light_skin)/len(total_stable_normalized_sae_light_skin))
print("total_dalle_normalized_dialect_light_skin:", sum(total_dalle_normalized_dialect_light_skin)/len(total_dalle_normalized_dialect_light_skin))
print("total_stable_normalized_dialect_light_skin:", sum(total_stable_normalized_dialect_light_skin)/len(total_stable_normalized_dialect_light_skin))