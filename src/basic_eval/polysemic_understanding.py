import os
import math
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import torch
import clip
import torchmetrics

# ------------------------- Configuration -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
LIBRARY = "openai"  # Alternatives: "torchmetrics"
IMG_DIR = "/local1/bryanzhou008/Dialect/multimodal-dialectal-bias/data/image/basic/aae"
DATA_FILE = "/local1/bryanzhou008/Dialect/multimodal-dialectal-bias/data/text/basic/aae.csv"
MODELS_TO_EVALUATE = ["stable-diffusion-3.5-large-turbo"]
FOLDER_MAPPING = {"dialect": "dialect_imgs", "sae": "sae_imgs"}
ENCODING = "unicode_escape"
# -----------------------------------------------------------------

# Load the CLIP model and its preprocessing function.
clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
torch.manual_seed(42)
# Note: torchmetrics is not used if LIBRARY=="openai"
# metric = torchmetrics.multimodal.CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")

  
def get_average_score(img_dir, model, folder, gen_prompt, ref_prompt, num_images=9):
    """
    Compute the average CLIP similarity score for a set of generated images.
    
    Args:
        img_dir (str): Base directory containing images.
        model (str): Name of the model folder.
        folder (str): Subfolder (e.g., 'dialect_imgs' or 'sae_imgs').
        gen_prompt (str): Generated prompt used for image creation.
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
            # Handle filename inconsistencies (e.g., replacing "'" with "_")
            processed_prompt = gen_prompt.replace("'", "_")
            new_prompt_dir = os.path.join(img_dir, model, folder, processed_prompt)
            image_path = os.path.join(new_prompt_dir, f"{i}.jpg")
            image = Image.open(image_path)
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        text_tensor = clip.tokenize([ref_prompt]).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image_tensor)
            text_features = clip_model.encode_text(text_tensor)
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


def main():
    """
    Main processing routine:
      - Reads the CSV data and filters rows based on the 'polysemic' column.
      - For each prompt (only if polysemic is nonzero), computes average scores
        for both dialect and SAE images across a set of models.
      - Prints both per-prompt scores and overall average scores.
    """
    # Read CSV data and convert relevant columns to lists.
    df = pd.read_csv(DATA_FILE, encoding=ENCODING)
    dialect_prompts = df["Dialect_Prompt"].tolist()
    sae_prompts = df["SAE_Prompt"].tolist()
    polysemic = df["polysemic"].tolist()

    # Storage for results: structure {set_type: {model: [scores, ...]}}
    results = {
        "dialect": {model: [] for model in MODELS_TO_EVALUATE},
        "sae": {model: [] for model in MODELS_TO_EVALUATE}
    }

    for i in tqdm(range(len(dialect_prompts)), desc="Processing prompts"):
        # Process only if polysemic value is nonzero.
        if polysemic[i] == 0:
            continue

        dialect_prompt = dialect_prompts[i]
        sae_prompt = sae_prompts[i]
        print(f"\nPrompt {i}:")
        print("Dialect prompt:", dialect_prompt)
        print("SAE prompt:", sae_prompt)

        # Evaluate for each model.
        for model in MODELS_TO_EVALUATE:
            # Process dialect images: use dialect prompt to generate images and SAE prompt as reference.
            folder = FOLDER_MAPPING["dialect"]
            score = get_average_score(IMG_DIR, model, folder, dialect_prompt, sae_prompt)
            results["dialect"][model].append(score)
            print(f"{model} dialect score: {score:.4f}")

            # Process SAE images: use SAE prompt for both generation and reference.
            folder = FOLDER_MAPPING["sae"]
            score = get_average_score(IMG_DIR, model, folder, sae_prompt, sae_prompt)
            results["sae"][model].append(score)
            print(f"{model} SAE score: {score:.4f}")

        # Optionally, display current average scores for each model.
        for set_type in results:
            for model in MODELS_TO_EVALUATE:
                if results[set_type][model]:
                    current_avg = sum(results[set_type][model]) / len(results[set_type][model])
                    print(f"Current average for {set_type} {model}: {current_avg:.4f}")

    print("\n------------------- Final Results -------------------")
    for set_type, model_scores in results.items():
        for model, scores in model_scores.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                print(f"{set_type.capitalize()} total score for {model}: {avg_score:.4f}")


if __name__ == "__main__":
    main()
