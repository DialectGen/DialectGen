import os
import math
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import torch
import clip

# ------------------------- Configuration -------------------------
LIBRARY = "openai"  # Alternatives: "torchmetrics" (not implemented)
IMG_DIR = "/local1/bryanzhou008/Dialect/multimodal-dialectal-bias/data/image/basic/aae"
DATA_FILE = "/local1/bryanzhou008/Dialect/multimodal-dialectal-bias/data/text/basic/aae.csv"
MODELS_TO_EVALUATE = ["stable-diffusion-3.5-large-turbo"]
FOLDER_MAPPING = {"dialect": "dialect_imgs", "sae": "sae_imgs"}
# -----------------------------------------------------------------



device = "cuda" if torch.cuda.is_available() else "cpu"
LIGHT_SKIN_PROMPT = "light-skinned"
DARK_SKIN_PROMPT = "dark-skinned"

# Load CLIP model and its preprocessing function
CLIP_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
torch.manual_seed(42)


def get_average_score(img_dir, model, folder, gen_prompt, ref_prompt, num_images=9):
    """
    Compute the average CLIP similarity score for a set of generated images.
    
    Args:
        img_dir (str): Base directory for images.
        model (str): Model folder name.
        folder (str): Subfolder ('dialect_imgs' or 'sae_imgs').
        gen_prompt (str): Generated prompt for image creation.
        ref_prompt (str): Reference prompt for scoring.
        num_images (int): Number of images to process (default: 9).
    
    Returns:
        float: Average similarity score.
    """
    prompt_dir = os.path.join(img_dir, model, folder, gen_prompt)
    scores = []
    for i in range(num_images):
        image_path = os.path.join(prompt_dir, f"{i}.jpg")
        try:
            image = Image.open(image_path)
        except Exception:
            processed_prompt = gen_prompt.replace("'", "_")
            new_prompt_dir = os.path.join(img_dir, model, folder, processed_prompt)
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
    
    Args:
        score_light (float): Score with the light-skin reference.
        score_dark (float): Score with the dark-skin reference.
        factor (int): Multiplicative factor for the scores (default: 100).
    
    Returns:
        float: Normalized score.
    """
    exp_light = math.exp(score_light * factor)
    exp_dark = math.exp(score_dark * factor)
    return exp_light / (exp_light + exp_dark)


def main():
    # Read and filter data (select prompts involving people)
    df = pd.read_csv(DATA_FILE)
    df = df.loc[df['person_in_prompt'] == 1]
    dialect_prompts = df["Dialect_Prompt"].tolist()
    sae_prompts = df["SAE_Prompt"].tolist()

    # Prepare storage for normalized scores
    results = {
        "dialect": {model: [] for model in MODELS_TO_EVALUATE},
        "sae": {model: [] for model in MODELS_TO_EVALUATE}
    }

    for i in tqdm(range(len(dialect_prompts)), desc="Processing prompts"):
        dialect_prompt = dialect_prompts[i]
        sae_prompt = sae_prompts[i]
        print(f"\nPrompt {i}:")
        print("Dialect prompt:", dialect_prompt)
        print("SAE prompt:", sae_prompt)

        for model in MODELS_TO_EVALUATE:
            # Process dialect images
            folder = FOLDER_MAPPING["dialect"]
            dialect_score_light = get_average_score(IMG_DIR, model, folder, dialect_prompt, LIGHT_SKIN_PROMPT)
            dialect_score_dark = get_average_score(IMG_DIR, model, folder, dialect_prompt, DARK_SKIN_PROMPT)
            norm_dialect = compute_normalized_score(dialect_score_light, dialect_score_dark)
            results["dialect"][model].append(norm_dialect)
            print(f"{model} dialect normalized (light): {norm_dialect:.4f}")

            # Process SAE images
            folder = FOLDER_MAPPING["sae"]
            sae_score_light = get_average_score(IMG_DIR, model, folder, sae_prompt, LIGHT_SKIN_PROMPT)
            sae_score_dark = get_average_score(IMG_DIR, model, folder, sae_prompt, DARK_SKIN_PROMPT)
            norm_sae = compute_normalized_score(sae_score_light, sae_score_dark)
            results["sae"][model].append(norm_sae)
            print(f"{model} SAE normalized (light): {norm_sae:.4f}")

        # Optionally, print current averages for each model
        for set_type in results:
            for model in MODELS_TO_EVALUATE:
                current_avg = sum(results[set_type][model]) / len(results[set_type][model])
                print(f"Current average for {set_type} {model}: {current_avg:.4f}")

    print("\n------------------- Final Results -------------------")
    for set_type, model_scores in results.items():
        for model, scores in model_scores.items():
            avg_score = sum(scores) / len(scores)
            print(f"{set_type.capitalize()} total normalized score for {model}: {avg_score:.4f}")


if __name__ == "__main__":
    main()
