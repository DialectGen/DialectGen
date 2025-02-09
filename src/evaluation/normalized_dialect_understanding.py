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
import string



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

# img_dir = "/local1/bryanzhou008/Dialect/data/images/oct_5_basic/aae"
# data_file = "/local1/bryanzhou008/Dialect/data/text/oct_5_csvs/african_american_english_prompts.csv"

# img_dir = "/local1/bryanzhou008/Dialect/data/images/oct_5_basic/bre"
# data_file = "/local1/bryanzhou008/Dialect/data/text/oct_5_csvs/british_english_prompts.csv"

# img_dir = "/local1/bryanzhou008/Dialect/data/images/oct_5_basic/che"
# data_file = "/local1/bryanzhou008/Dialect/data/text/oct_5_csvs/chicano_english_prompts.csv"

img_dir = "/local1/bryanzhou008/Dialect/data/images/oct_23_simplified/ine"
data_file = "/local1/bryanzhou008/Dialect/data/text/simplified/ine.csv"

# img_dir = "/local1/bryanzhou008/Dialect/data/images/oct_5_basic/sge"
# data_file = "/local1/bryanzhou008/Dialect/data/text/oct_5_csvs/singlish_prompts.csv"

# ---------------------------------------------------------------------------------------------











# ---------------------------------------- Entigen Paths ----------------------------------------
USE_ENTIGEN = False
# USE_ENTIGEN = True
# sae_prefix = "In Standard American English, "

# img_dir = "/local1/bryanzhou008/Dialect/data/images/oct_5_entigen/aae"
# data_file = "/local1/bryanzhou008/Dialect/data/text/oct_5_csvs/african_american_english_prompts.csv"
# dialect_prefix = "In African American English, "

# img_dir = "/local1/bryanzhou008/Dialect/data/images/oct_5_entigen/bre"
# data_file = "/local1/bryanzhou008/Dialect/data/text/oct_5_csvs/british_english_prompts.csv"
# dialect_prefix = "In British English, "

# img_dir = "/local1/bryanzhou008/Dialect/data/images/oct_5_entigen/che"
# data_file = "/local1/bryanzhou008/Dialect/data/text/oct_5_csvs/chicano_english_prompts.csv"
# dialect_prefix = "In Chicano English, "

# img_dir = "/local1/bryanzhou008/Dialect/data/images/oct_5_entigen/ine"
# data_file = "/local1/bryanzhou008/Dialect/data/text/oct_5_csvs/indian_english_prompts.csv"
# dialect_prefix = "In Indian English, "

# img_dir = "/local1/bryanzhou008/Dialect/data/images/oct_5_entigen/sge"
# data_file = "/local1/bryanzhou008/Dialect/data/text/oct_5_csvs/singlish_prompts.csv"
# dialect_prefix = "In Singlish "

# ---------------------------------------------------------------------------------------------







# Global Variables
# library = "torchmetrics"
library = "openai"


# Read Data
df = pd.read_csv(data_file)
dialect_prompts = list(df["Dialect_Prompt"])
sae_prompts = list(df["SAE_Prompt"])
# dialect_prompts = list(df["DialectPrompt"])[:-2]
# sae_prompts = list(df["SAEPrompt"])[:-2]

# dialect_prompts = [i[28:] for i  in dialect_prompts]
# sae_prompts = [i[29:] for i  in sae_prompts]


def get_average_scores(img_dir, model, img_type, gen_prompt, ref_prompt):
    if USE_ENTIGEN == True:
        print("USE_ENTIGEN == True")
        if img_type == "sae_imgs":
            gen_prompt = sae_prefix + gen_prompt
        elif img_type == "dialect_imgs":
            gen_prompt = dialect_prefix + gen_prompt
        else:
            print("img_type must be either sae_imgs or dialect_imgs")
            raise
    else:
        print("USE_ENTIGEN != True")
        
    prompt_dir = os.path.join(img_dir, model, img_type, gen_prompt)
    scores = []
    for i in range(0, 9):
        try:
            image = preprocess(Image.open(prompt_dir + f'/{i}.jpg')).unsqueeze(0).to(device)
        except:
            # for punctuation in string.punctuation:
            #     gen_prompt = gen_prompt.replace(punctuation, '')
            gen_prompt = gen_prompt.replace(",", '')
            prompt_dir = os.path.join(img_dir, model, img_type, gen_prompt)
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





dialect_total_score_dalle = []
dialect_total_score_stable = []
dialect_total_score_stable_2 = []

sae_total_score_dalle = []
sae_total_score_stable = []
sae_total_score_stable_2 = []

dalle_total_normalized_drop = []
stable_total_normalized_drop = []
stable_total_normalized_drop_2 = []

dalle_total_normalized_drop_v2 = []
stable_total_normalized_drop_v2 = []
stable_total_normalized_drop_v2_2 = []

for i in tqdm(range(len(dialect_prompts))):
    # get the prompts
    dialect_prompt = dialect_prompts[i]
    print("dialectal prompt is:", dialect_prompt)
    
    sae_prompt = sae_prompts[i]
    print("sae prompt is:", sae_prompt)

    print("\n")

    # dialect_dalle_score_ref_sae = get_average_scores(img_dir, "minDALL-E", "dialect_imgs", dialect_prompt, sae_prompt)
    # dialect_total_score_dalle.append(dialect_dalle_score_ref_sae)
    # print("dialect_dalle_score_ref_sae:", dialect_dalle_score_ref_sae)

    # dialect_stable_score_ref_sae = get_average_scores(img_dir, "stable-diffusion1.4", "dialect_imgs", dialect_prompt, sae_prompt)
    # dialect_total_score_stable.append(dialect_stable_score_ref_sae)
    # print("dialect_stable_score_ref_sae:", dialect_stable_score_ref_sae)
    
    dialect_stable_score_ref_sae_2 = get_average_scores(img_dir, "stable-diffusion2.1", "dialect_imgs", dialect_prompt, sae_prompt)
    dialect_total_score_stable_2.append(dialect_stable_score_ref_sae_2)
    print("dialect_stable_score_ref_sae:", dialect_stable_score_ref_sae_2)

    print("\n")

    # dialect_dalle_score_ref_dialect = get_average_scores(img_dir, "minDALL-E", "dialect_imgs", dialect_prompt, dialect_prompt)
    # print("dialect_dalle_score_ref_dialect:", dialect_dalle_score_ref_dialect)

    # dialect_stable_score_ref_dialect = get_average_scores(img_dir, "stable-diffusion1.4", "dialect_imgs", dialect_prompt, dialect_prompt)
    # print("dialect_stable_score_ref_dialect:", dialect_stable_score_ref_dialect)
    
    # dialect_stable_score_ref_dialect_2 = get_average_scores(img_dir, "stable-diffusion2.1", "dialect_imgs", dialect_prompt, dialect_prompt)
    # print("dialect_stable_score_ref_dialect:", dialect_stable_score_ref_dialect_2)

    print("\n")
    

    # sae_dalle_score_ref_sae = get_average_scores(img_dir, "minDALL-E", "sae_imgs", sae_prompt, sae_prompt)
    # sae_total_score_dalle.append(sae_dalle_score_ref_sae)
    # print("sae_dalle_score_ref_sae:", sae_dalle_score_ref_sae)


    # sae_stable_score_ref_sae = get_average_scores(img_dir, "stable-diffusion1.4", "sae_imgs", sae_prompt, sae_prompt)
    # sae_total_score_stable.append(sae_stable_score_ref_sae)
    # print("sae_stable_score_ref_sae:", sae_stable_score_ref_sae)
    
    sae_stable_score_ref_sae_2 = get_average_scores(img_dir, "stable-diffusion2.1", "sae_imgs", sae_prompt, sae_prompt)
    sae_total_score_stable_2.append(sae_stable_score_ref_sae_2)
    print("sae_stable_score_ref_sae:", sae_stable_score_ref_sae_2)
    


    # sae_dalle_score_ref_dialect = get_average_scores(img_dir, "minDALL-E", "sae_imgs", sae_prompt, dialect_prompt)
    # print("sae_dalle_score_ref_dialect:", sae_dalle_score_ref_dialect)


    # sae_stable_score_ref_dialect = get_average_scores(img_dir, "stable-diffusion1.4", "sae_imgs", sae_prompt, dialect_prompt)
    # print("sae_stable_score_ref_dialect:", sae_stable_score_ref_dialect)
    
    # sae_stable_score_ref_dialect_2 = get_average_scores(img_dir, "stable-diffusion2.1", "sae_imgs", sae_prompt, dialect_prompt)
    # print("sae_stable_score_ref_dialect:", sae_stable_score_ref_dialect_2)


    print("\n")


    # The normalized drop: e^S_AA/(e^S_AA+e^S_AB) - e^S_BB/(e^S_BA+e^S_BB)

    # dalle_normalized_drop = math.exp(sae_dalle_score_ref_sae)/(math.exp(sae_dalle_score_ref_sae) + math.exp(sae_dalle_score_ref_dialect)) - math.exp(dialect_dalle_score_ref_dialect)/(math.exp(dialect_dalle_score_ref_dialect) + math.exp(dialect_dalle_score_ref_sae))
    # stable_normalized_drop = math.exp(sae_stable_score_ref_sae)/(math.exp(sae_stable_score_ref_sae) + math.exp(sae_stable_score_ref_dialect)) - math.exp(dialect_stable_score_ref_dialect)/(math.exp(dialect_stable_score_ref_dialect) + math.exp(dialect_stable_score_ref_sae))
    # stable_normalized_drop_2 = math.exp(sae_stable_score_ref_sae_2)/(math.exp(sae_stable_score_ref_sae_2) + math.exp(sae_stable_score_ref_dialect_2)) - math.exp(dialect_stable_score_ref_dialect_2)/(math.exp(dialect_stable_score_ref_dialect_2) + math.exp(dialect_stable_score_ref_sae_2))
    
    
    # dalle_total_normalized_drop.append(dalle_normalized_drop)
    # stable_total_normalized_drop.append(stable_normalized_drop)
    # stable_total_normalized_drop_2.append(stable_normalized_drop_2)


    # dalle_normalized_drop_v2 = math.exp(sae_dalle_score_ref_sae)/(math.exp(sae_dalle_score_ref_sae) + math.exp(sae_dalle_score_ref_sae)) - math.exp(dialect_dalle_score_ref_dialect)/(math.exp(dialect_dalle_score_ref_dialect) + math.exp(dialect_dalle_score_ref_sae))
    # stable_normalized_drop_v2 = math.exp(sae_stable_score_ref_sae)/(math.exp(sae_stable_score_ref_sae) + math.exp(sae_stable_score_ref_sae)) - math.exp(dialect_stable_score_ref_dialect)/(math.exp(dialect_stable_score_ref_dialect) + math.exp(dialect_stable_score_ref_sae))
    # stable_normalized_drop_v2_2 = math.exp(sae_stable_score_ref_sae_2)/(math.exp(sae_stable_score_ref_sae_2) + math.exp(sae_stable_score_ref_sae_2)) - math.exp(dialect_stable_score_ref_dialect_2)/(math.exp(dialect_stable_score_ref_dialect_2) + math.exp(dialect_stable_score_ref_sae_2))
    

    # dalle_total_normalized_drop_v2.append(dalle_normalized_drop_v2)
    # stable_total_normalized_drop_v2.append(stable_normalized_drop_v2)
    # stable_total_normalized_drop_v2_2.append(stable_normalized_drop_v2_2)



print("-------------------final results-------------------")
# print("dialect_total_score_dalle:", sum(dialect_total_score_dalle)/len(dialect_total_score_dalle)) 
# print("dialect_total_score_stable1.4:", sum(dialect_total_score_stable)/len(dialect_total_score_stable))
print("dialect_total_score_stable2.1:", sum(dialect_total_score_stable_2)/len(dialect_total_score_stable_2))

# print("sae_total_score_dalle:", sum(sae_total_score_dalle)/len(sae_total_score_dalle))
# print("sae_total_score_stable1.4:", sum(sae_total_score_stable)/len(sae_total_score_stable))
print("sae_total_score_stable2.1:", sum(sae_total_score_stable_2)/len(sae_total_score_stable_2))

# print("dalle_total_normalized_drop:", sum(dalle_total_normalized_drop)/len(dalle_total_normalized_drop))
# print("stable_total_normalized_drop1.4:", sum(stable_total_normalized_drop)/len(stable_total_normalized_drop))
# print("stable_total_normalized_drop2.1:", sum(stable_total_normalized_drop_2)/len(stable_total_normalized_drop_2))

# print("dalle_total_normalized_drop_v2:", sum(dalle_total_normalized_drop_v2)/len(dalle_total_normalized_drop_v2))
# print("stable_total_normalized_drop_v2_1.4:", sum(stable_total_normalized_drop_v2)/len(stable_total_normalized_drop_v2))
# print("stable_total_normalized_drop_v2_2.1:", sum(stable_total_normalized_drop_v2_2)/len(stable_total_normalized_drop_v2_2))
