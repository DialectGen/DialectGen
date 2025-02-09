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


# ---------------------------------------- Basic Paths ----------------------------------------

# img_dir = "/local1/bryanzhou008/Dialect/data/expanded_ChicanoEnglish_v1"
# data_file = "/local1/bryanzhou008/Dialect/data/prompt_csvs/chicano_english_prompts.csv"

# img_dir = "/local1/bryanzhou008/Dialect/data/expanded_Singlish_v1/"
# data_file = "/local1/bryanzhou008/Dialect/data/prompt_csvs/singlish_prompts.csv"


# img_dir = "/local1/bryanzhou008/Dialect/data/expanded_BritishEnglish_v1/"
# data_file = "/local1/bryanzhou008/Dialect/data/prompt_csvs/british_english_prompts.csv"


# img_dir = "/local1/bryanzhou008/Dialect/data/expanded_BritishEnglish_v1/"
# data_file = "/local1/bryanzhou008/Dialect/data/prompt_csvs/british_english_prompts.csv"

# img_dir = "/local1/bryanzhou008/Dialect/data/expanded_aae_v1/"
# data_file = "/local1/bryanzhou008/Dialect/data/prompt_csvs/african_american_english_prompts.csv"

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


def get_all_black_num(img_dir, model, dialect, gen_prompt, ref_prompt):
    prompt_dir = os.path.join(img_dir, model, dialect, gen_prompt)
    black_num = 0
    for i in range(0, 9):
        img = Image.open(prompt_dir + f'/{i}.jpg')
        if sum(img.convert("L").getextrema()) in (0, 2):
            black_num += 1

    return black_num





dialect_total_score_dalle = []
dialect_total_score_stable = []

sae_total_score_dalle = []
sae_total_score_stable = []

for i in tqdm(range(len(dialect_prompts))):
    # dialect
    dialect_prompt = dialect_prompts[i]
    print("dialectal prompt is:", dialect_prompt)

    # test
    sae_prompt = sae_prompts[i]
    print("sae prompt is:", sae_prompt)

    dialect_dalle_score = get_all_black_num(img_dir, "minDALL-E", "dialect_imgs", dialect_prompt, sae_prompt)
    dialect_total_score_dalle.append(dialect_dalle_score)
    print("dialect_dalle_score:", dialect_dalle_score)

    dialect_stable_score = get_all_black_num(img_dir, "stable-diffusion", "dialect_imgs", dialect_prompt, sae_prompt)
    dialect_total_score_stable.append(dialect_stable_score)
    print("dialect_stable_score:", dialect_stable_score)

    print("\n")

    # sae
    sae_prompt = sae_prompts[i]
    print("sae prompt is:", sae_prompt)


    sae_dalle_score = get_all_black_num(img_dir, "minDALL-E", "sae_imgs", sae_prompt, sae_prompt)
    sae_total_score_dalle.append(sae_dalle_score)
    print("sae_dalle_score:", sae_dalle_score)


    sae_stable_score = get_all_black_num(img_dir, "stable-diffusion", "sae_imgs", sae_prompt, sae_prompt)
    sae_total_score_stable.append(sae_stable_score)
    print("sae_stable_score:", sae_stable_score)


    print("\n")


print("-------------------final results-------------------")
print("dialect_avg_nsfw_dalle:", sum(dialect_total_score_dalle)/len(dialect_total_score_dalle)) 
print("dialect_avg_nsfw_stable:", sum(dialect_total_score_stable)/len(dialect_total_score_stable))

print("sae_avg_nsfw_dalle:", sum(sae_total_score_dalle)/len(sae_total_score_dalle))
print("sae_avg_nsfw_stable:", sum(sae_total_score_stable)/len(sae_total_score_stable))

print("dialect_total_nsfw_dalle:", sum(dialect_total_score_dalle))
print("dialect_total_nsfw_stable:", sum(dialect_total_score_stable))
      
print("sae_total_nsfw_dalle:", sum(sae_total_score_dalle))
print("sae_total_nsfw_stable:", sum(sae_total_score_stable))