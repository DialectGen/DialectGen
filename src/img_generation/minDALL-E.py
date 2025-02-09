import os
import sys
from tqdm import tqdm


from IPython.display import display, update_display
import torch
from min_dalle import MinDalle
import numpy as np
import pandas as pd
from PIL import Image


def generate_min_dalle(prompt, save_dir, seamless, temperature, grid_size, supercondition_factor):
    images = model.generate_images(
        text=prompt,
        seed=-1,
        grid_size=grid_size,
        is_seamless=seamless,
        temperature=temperature,
        top_k=int(top_k),
        supercondition_factor=float(supercondition_factor)
    )

    images = images.to('cpu').numpy()
    for i, img in enumerate(images):
        image = Image.fromarray((img * 1).astype(np.uint8)).convert('RGB')
        image.save(save_dir + "/" + str(i) + ".jpg")
        
        
# entigen
run_entigen = False
# run_entigen = True
# entigen_dialect = "In Chicano English, "
# entigen_dialect = "In Singlish, "
# entigen_dialect = "In African American English, "
# entigen_dialect = "In British English, "
# entigen_dialect = "In Indian English, "
# entigen_sae = "In Standard American English, "


# Set Hyperparameters
img_dir = "/local1/bryanzhou008/Dialect/data/images/oct_23_simplified/aae/minDALL-E/"
data_file = "/local1/bryanzhou008/Dialect/data/text/simplified/aae.csv"
temperature = 1 #@param {type:"slider", min:0.01, max:16, step:0.01}
grid_size = 3 #@param {type:"integer"}
supercondition_factor = 16 #@param {type:"number"}
top_k = 128 #@param {type:"integer"}
seamless = False
dtype = "float32"
model = MinDalle(
    dtype=getattr(torch, dtype),
    device='cuda',
    is_mega=True,
    is_reusable=True
)

# Read Data
df = pd.read_csv(data_file, encoding='unicode_escape')
dialect_prompts = list(df["Dialect_Prompt"])
sae_prompts = list(df["SAE_Prompt"])

# Create Output Directory and two subdirectories for low-resource and high-resource dialects

if os.path.exists(img_dir):
  pass
else:
  os.mkdir(img_dir)

lr_subdir = img_dir + "dialect_imgs/"
if os.path.exists(lr_subdir):
  pass
else:
  os.mkdir(lr_subdir)

hr_subdir = img_dir + "sae_imgs/"
if os.path.exists(hr_subdir):
  pass
else:
  os.mkdir(hr_subdir)



# if os.path.exists(img_dir):
#   os.system(f'rm -rf {img_dir}')
# os.mkdir(img_dir)

# lr_subdir = img_dir + "dialect_imgs/"
# if os.path.exists(lr_subdir):
#   os.system(f'rm -rf {lr_subdir}')
# os.mkdir(lr_subdir)

# hr_subdir = img_dir + "sae_imgs/"
# if os.path.exists(hr_subdir):
#   os.system(f'rm -rf {hr_subdir}')
# os.mkdir(hr_subdir)


for i in tqdm(range(len(dialect_prompts))):
    dp = dialect_prompts[i]
    sp = sae_prompts[i]
    
    # entigen
    if run_entigen == True:
       dp = entigen_dialect + dp
       sp = entigen_sae + sp

    dp_dir = lr_subdir + dp
    sp_dir = hr_subdir + sp

    if not os.path.exists(dp_dir):
        os.mkdir(dp_dir)
        generate_min_dalle(dp, dp_dir, seamless, temperature, grid_size, supercondition_factor)


    if not os.path.exists(sp_dir):
        os.mkdir(sp_dir)
        generate_min_dalle(sp, sp_dir, seamless, temperature, grid_size, supercondition_factor)


    # generate_min_dalle(dp, dp_dir, seamless, temperature, grid_size, supercondition_factor)
    # generate_min_dalle(sp, sp_dir, seamless, temperature, grid_size, supercondition_factor)
