import os
import sys
from tqdm import tqdm


from IPython.display import display, update_display
import torch
# from min_dalle import MinDalle
import numpy as np
import pandas as pd
from PIL import Image
from diffusers import StableDiffusionPipeline



def generate_stable_diffusion(prompt, save_dir):
    num_cols = 3
    num_rows = 3

    prompt = [prompt] * num_cols

    all_images = []
    for i in range(num_rows):
        images = pipe(prompt).images
        all_images.extend(images)

    for i in range(len(all_images)):
        image = all_images[i]
        image.save(save_dir + "/" + str(i) + ".jpg")



# choose stable diffusion version:
# model_id = "CompVis/stable-diffusion-v1-1"
# model_id = "CompVis/stable-diffusion-v1-2"
# model_id = "CompVis/stable-diffusion-v1-3"
# model_id = "CompVis/stable-diffusion-v1-4"
# model_id = "runwayml/stable-diffusion-v1-5"
# model_id = "stabilityai/stable-diffusion-2"
model_id = "stabilityai/stable-diffusion-2-1"



# Set Hyperparameters
# img_dir = "/local1/bryanzhou008/Dialect/data/test_269_aae/stable-diffusion/"
# data_file = "/local1/bryanzhou008/Dialect/data/test_269_aae/aae269.csv"
# temperature = 1 #@param {type:"slider", min:0.01, max:16, step:0.01}
# grid_size = 3 #@param {type:"integer"}
# supercondition_factor = 16 #@param {type:"number"}
# top_k = 128 #@param {type:"integer"}
# seamless = False
# dtype = "float32"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")


# entigen
# run_entigen = False
run_entigen = True
# entigen_dialect = "In Chicano English, "
# entigen_dialect = "In Singlish, "
# entigen_dialect = "In African American English, "
# entigen_dialect = "In British English, "
# entigen_dialect = "In Indian English, "
# entigen_sae = "In Standard American English, "


# Specify the data files and image prompt csv

# base_img_dir = "/local1/bryanzhou008/Dialect/data/expanded_BritishEnglish_v1_entigen/"
# data_file = "/local1/bryanzhou008/Dialect/data/prompt_csvs/british_english_prompts.csv"

# base_img_dir = "/local1/bryanzhou008/Dialect/data/expanded_aae_v1_entigen/"
# data_file = "/local1/bryanzhou008/Dialect/data/prompt_csvs/african_american_english_prompts.csv"

# base_img_dir = "/local1/bryanzhou008/Dialect/data/expanded_Singlish_v1_entigen/"
# data_file = "/local1/bryanzhou008/Dialect/data/prompt_csvs/singlish_prompts.csv"

# base_img_dir = "/local1/bryanzhou008/Dialect/data/expanded_ChicanoEnglish_v1_entigen/"
# data_file = "/local1/bryanzhou008/Dialect/data/prompt_csvs/chicano_english_prompts.csv"

# base_img_dir = "/local1/bryanzhou008/Dialect/data/expanded_Singlish_v1/"
# data_file = "/local1/bryanzhou008/Dialect/data/prompt_csvs/singlish_prompts.csv"

# base_img_dir = "/local1/bryanzhou008/Dialect/data/expanded_ChicanoEnglish_v1/"
# data_file = "/local1/bryanzhou008/Dialect/data/prompt_csvs/chicano_english_prompts.csv"

# base_img_dir = "/local1/bryanzhou008/Dialect/data/expanded_BritishEnglish_v1/"
# data_file = "/local1/bryanzhou008/Dialect/data/prompt_csvs/british_english_prompts.csv"

# base_img_dir = "/local1/bryanzhou008/Dialect/data/expanded_aae_v1/"
# data_file = "/local1/bryanzhou008/Dialect/data/prompt_csvs/african_american_english_prompts.csv"

# img_dir = "/local1/bryanzhou008/Dialect/data/test_269_aae_entigen/stable-diffusion/"
# data_file = "/local1/bryanzhou008/Dialect/data/test_269_aae/aae269_entigen.csv"


base_img_dir = "/local1/bryanzhou008/Dialect/data/images/oct_5_entigen/ine/"
img_dir = base_img_dir + "stable-diffusion2.1/"
data_file = "/local1/bryanzhou008/Dialect/data/text/polysemy_csvs/IndianEnglish_Polysemy.csv"

# Read Data
df = pd.read_csv(data_file)
Prompts = list(df["Prompt"])

# Create Output Directory and two subdirectories for low-resource and high-resource dialects
# if os.path.exists(base_img_dir):
#   pass
# else:
#   os.mkdir(base_img_dir)

# if os.path.exists(img_dir):
#   os.system(f'rm -rf {img_dir}')
# os.mkdir(img_dir)

# lr_subdir = img_dir + "dialect_imgs/"
# if os.path.exists(lr_subdir):
#   os.system(f'rm -rf {lr_subdir}')
# os.mkdir(lr_subdir)

polysemy_subdir = img_dir + "polysemy/"
if os.path.exists(polysemy_subdir):
  os.system(f'rm -rf {polysemy_subdir}')
os.mkdir(polysemy_subdir)


for i in tqdm(range(len(Prompts))):
    prompt = Prompts[i]


    # entigen
    if run_entigen == True:
       prompt = "In Standard American English " + prompt


    prompt_dir = polysemy_subdir + prompt

    if not os.path.exists(prompt_dir):
        os.mkdir(prompt_dir)


    generate_stable_diffusion(prompt, prompt_dir)
