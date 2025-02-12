import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import torch
import clip

# ------------------------- Configuration -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
LIBRARY = "openai"  # Alternatives: "torchmetrics"
IMG_DIR = "/local1/bryanzhou008/Dialect/multimodal-dialectal-bias/data/image/basic/aae"
DATA_FILE = "/local1/bryanzhou008/Dialect/multimodal-dialectal-bias/data/text/basic/aae.csv"
MODELS_TO_EVALUATE = ["stable-diffusion-3.5-large-turbo"]
# ------------------------------------------------------------------

# Load the CLIP model and its preprocessing function.
CLIP_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
torch.manual_seed(42)


def get_average_score(img_dir, model_name, folder, gen_prompt, ref_prompt, num_images=9):
    """
    Compute the average CLIP similarity score for a set of generated images.

    Args:
        img_dir (str): Base directory containing images.
        model_name (str): Name of the model folder.
        folder (str): Subfolder (e.g., 'dialect_imgs' or 'sae_imgs').
        gen_prompt (str): Generated prompt used to create the images.
        ref_prompt (str): Reference prompt for scoring.
        num_images (int): Number of images to process (default is 9).

    Returns:
        float: Average similarity score.
    """
    prompt_dir = os.path.join(img_dir, model_name, folder, gen_prompt)
    scores = []

    for i in range(num_images):
        image_path = os.path.join(prompt_dir, f"{i}.jpg")
        try:
            image = Image.open(image_path)
        except Exception:
            # Handle filename inconsistencies (e.g., replacing "'" with "_").
            processed_prompt = gen_prompt.replace("'", "_")
            prompt_dir = os.path.join(img_dir, model_name, folder, processed_prompt)
            image_path = os.path.join(prompt_dir, f"{i}.jpg")
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


def main():
    # Read CSV data.
    df = pd.read_csv(DATA_FILE, encoding="unicode_escape")
    dialect_prompts = df["Dialect_Prompt"].tolist()
    sae_prompts = df["SAE_Prompt"].tolist()

    # Initialize dictionaries to store scores.
    results = {
        "dialect": {model: [] for model in MODELS_TO_EVALUATE},
        "sae": {model: [] for model in MODELS_TO_EVALUATE},
    }

    for i in tqdm(range(len(dialect_prompts)), desc="Processing prompts"):
        dialect_prompt = dialect_prompts[i]
        sae_prompt = sae_prompts[i]

        # Evaluate dialect images (using SAE prompt as reference).
        for model in MODELS_TO_EVALUATE:
            score = get_average_score(IMG_DIR, model, "dialect_imgs", dialect_prompt, sae_prompt)
            results["dialect"][model].append(score)
            print(f"Prompt {i} - {model} (dialect): {score:.4f}")

        # Evaluate SAE images (using SAE prompt for both generated and reference).
        for model in MODELS_TO_EVALUATE:
            score = get_average_score(IMG_DIR, model, "sae_imgs", sae_prompt, sae_prompt)
            results["sae"][model].append(score)
            print(f"Prompt {i} - {model} (sae): {score:.4f}")

    print("\n------------------- Final Results -------------------")
    for set_type, model_scores in results.items():
        for model, scores in model_scores.items():
            avg_score = sum(scores) / len(scores)
            print(f"{set_type.capitalize()} total score for {model}: {avg_score:.4f}")


if __name__ == "__main__":
    main()
