import os
import math
import pandas as pd
from tqdm import tqdm
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import torch
import clip

# ------------------------- Configuration -------------------------
MODELS_TO_EVALUATE = ["stable-diffusion2.1"]
MODES = ["concise", "detailed"]
DIALECTS = ["aae", "che", "sge", "ine", "bre"]
FOLDER_MAPPING = {"dialect": "dialect_imgs", "sae": "sae_imgs"}
# ------------------------------------------------------------------

# Path settings for the skintone evaluation task.
TASK = "skintone"
MODEL = "clip"
BASE_DIR = "/local1/bryanzhou008/Dialect/multimodal-dialectal-bias"
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, f"out/{TASK}/base_models_{MODEL}")

# Skin-tone reference prompts.
LIGHT_SKIN_PROMPT = "light-skinned"
DARK_SKIN_PROMPT = "dark-skinned"

# CLIP model setup.
device = "cuda" if torch.cuda.is_available() else "cpu"
LIBRARY = "openai"  # Only 'openai' is implemented.
CLIP_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
torch.manual_seed(42)

def get_average_score(img_dir, model_name, folder, gen_prompt, ref_prompt, num_images):
    """
    Compute the average CLIP similarity score for a set of generated images.
    """
    prompt_dir = os.path.join(img_dir, model_name, folder, gen_prompt)
    scores = []
    
    for i in range(num_images):
        image_path = os.path.join(prompt_dir, f"{i}.jpg")
        try:
            image = Image.open(image_path)
        except Exception:
            # Handle filename inconsistencies by replacing problematic characters.
            processed_prompt = gen_prompt.replace("'", "_")
            new_prompt_dir = os.path.join(img_dir, model_name, folder, processed_prompt)
            image_path = os.path.join(new_prompt_dir, f"{i}.jpg")
            image = Image.open(image_path)
        
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        text_tensor = clip.tokenize([ref_prompt]).to(device)
        with torch.no_grad():
            image_features = CLIP_model.encode_image(image_tensor)
            text_features = CLIP_model.encode_text(text_tensor)
            if LIBRARY == "openai":
                score = cosine_similarity(
                    image_features.cpu().numpy(), text_features.cpu().numpy()
                )[0][0]
            elif LIBRARY == "torchmetrics":
                raise NotImplementedError("Torchmetrics scoring is not implemented.")
            else:
                raise ValueError("Undefined library specified.")
        scores.append(score)
    
    return sum(scores) / len(scores)

def compute_normalized_score(score_light, score_dark, factor=100):
    """
    Compute the normalized score using the drop formula.
    """
    exp_light = math.exp(score_light * factor)
    exp_dark = math.exp(score_dark * factor)
    return exp_light / (exp_light + exp_dark)

def main():
    # Loop over each combination of mode and dialect.
    for mode in MODES:
        for dialect in DIALECTS:
            # Construct paths for the CSV data and image directory.
            data_file = os.path.join(DATA_DIR, "text", mode, f"{dialect}.csv")
            img_dir = os.path.join(DATA_DIR, "image", mode, dialect)
            
            print(f"\n{'='*60}\nEvaluating Skintone for MODE: '{mode}', DIALECT: '{dialect}'\n{'='*60}\n")
            
            # Load CSV data and filter to only those prompts with people.
            try:
                df = pd.read_csv(data_file, encoding="unicode_escape")
            except Exception as e:
                print(f"Failed to load file {data_file}: {e}")
                continue
            
            # Only consider prompts where "person_in_prompt" equals 1.
            df = df.loc[df['person_in_prompt'] == 1]
            dialect_prompts = df["Dialect_Prompt"].tolist()
            sae_prompts = df["SAE_Prompt"].tolist()
            
            # For each model, process all prompts.
            for model in MODELS_TO_EVALUATE:
                # for dalle2 and dalle3, we only have 1 image, otherwise we have 9
                if model in ["dalle2", "dalle3"]:
                    NUM_IMAGES = 1
                else:
                    NUM_IMAGES = 9
                
                # Prepare output directory for this configuration.
                output_model_dir = os.path.join(OUTPUT_DIR, mode, dialect, model)
                os.makedirs(output_model_dir, exist_ok=True)
                
                # Define output file paths.
                breakdown_dialect_path = os.path.join(output_model_dir, "breakdown_dialect.csv")
                breakdown_sae_path = os.path.join(output_model_dir, "breakdown_sae.csv")
                summary_path = os.path.join(output_model_dir, "summary.csv")
                
                # Check if outputs already exist; if so, skip evaluation.
                if os.path.exists(breakdown_dialect_path) and os.path.exists(breakdown_sae_path) and os.path.exists(summary_path):
                    print(f"Results already exist for MODE: '{mode}', DIALECT: '{dialect}', MODEL: '{model}'. Skipping evaluation.\n")
                    continue
                
                results_dialect = []  # Breakdown for dialect images evaluation.
                results_sae = []      # Breakdown for SAE images evaluation.
                
                print(f"\n>> Evaluating MODEL: '{model}'")
                for i in tqdm(range(len(dialect_prompts)), desc="Processing prompts"):
                    dialect_prompt = dialect_prompts[i]
                    sae_prompt = sae_prompts[i]
                    
                    # Evaluate only prompts with people.
                    # For dialect images.
                    folder = FOLDER_MAPPING["dialect"]
                    score_light = get_average_score(img_dir, model, folder, dialect_prompt, LIGHT_SKIN_PROMPT, NUM_IMAGES)
                    score_dark = get_average_score(img_dir, model, folder, dialect_prompt, DARK_SKIN_PROMPT, NUM_IMAGES)
                    norm_dialect = round(compute_normalized_score(score_light, score_dark), 4)
                    results_dialect.append({
                        "Prompt_Index": i,
                        "Dialect_Prompt": dialect_prompt,
                        "SAE_Prompt": sae_prompt,
                        "Normalized_Score": norm_dialect
                    })
                    print(f"MODE: {mode} | DIALECT: {dialect} | Prompt {i} (dialect) | '{dialect_prompt}': {norm_dialect:.4f}")
                    
                    # For SAE images.
                    folder = FOLDER_MAPPING["sae"]
                    score_light = get_average_score(img_dir, model, folder, sae_prompt, LIGHT_SKIN_PROMPT, NUM_IMAGES)
                    score_dark = get_average_score(img_dir, model, folder, sae_prompt, DARK_SKIN_PROMPT, NUM_IMAGES)
                    norm_sae = round(compute_normalized_score(score_light, score_dark), 4)
                    results_sae.append({
                        "Prompt_Index": i,
                        "SAE_Prompt": sae_prompt,
                        "Normalized_Score": norm_sae
                    })
                    print(f"MODE: {mode} | DIALECT: {dialect} | Prompt {i} (sae) | '{sae_prompt}': {norm_sae:.4f}")
                
                # Calculate overall average normalized scores.
                avg_dialect = round(sum(r["Normalized_Score"] for r in results_dialect) / len(results_dialect) if results_dialect else 0, 4)
                avg_sae = round(sum(r["Normalized_Score"] for r in results_sae) / len(results_sae) if results_sae else 0, 4)
                
                print(f"\n--- Final Results for MODE: '{mode}', DIALECT: '{dialect}', MODEL: '{model}' ---")
                print(f"Overall Dialect Normalized Score: {avg_dialect:.4f}")
                print(f"Overall SAE Normalized Score: {avg_sae:.4f}\n")
                
                # Save breakdown results.
                df_dialect = pd.DataFrame(results_dialect)
                df_sae = pd.DataFrame(results_sae)
                df_dialect.to_csv(breakdown_dialect_path, index=False)
                df_sae.to_csv(breakdown_sae_path, index=False)
                
                # Calculate additional metrics for the summary.
                absolute_drop = round(avg_sae - avg_dialect, 4)
                drop_ratio = round((absolute_drop / avg_sae) if avg_sae != 0 else 0, 4)
                
                # Save summary results.
                summary_df = pd.DataFrame({
                    "Evaluation_Type": ["Dialect", "SAE", "Absolute Drop", "Drop Ratio"],
                    "Overall_Average_Score": [avg_dialect, avg_sae, absolute_drop, drop_ratio]
                })
                summary_df.to_csv(summary_path, index=False)
                
                print(f"Results saved to: {output_model_dir}\n")

if __name__ == "__main__":
    main()
