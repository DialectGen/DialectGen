#!/usr/bin/env python3
"""
VQA-understanding evaluation with 4-digit rounding and drop metrics.
"""
import os
import hashlib
import pandas as pd
from tqdm import tqdm
import t2v_metrics
import torch  # type checking
import random
import numpy as np
from transformers import set_seed

def fix_seed(seed: int):
    """Sets the seed for reproducibility across various libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)

# Fix the seed for reproducibility
fix_seed(42)


# ------------------------- Configuration -------------------------
MODELS_TO_EVALUATE = ["stable-diffusion2.1"]
MODES = ["concise", "detailed", "rewrite_concise", "rewrite_detailed"]
DIALECTS = ["aae", "che", "sge", "ine", "bre"]
# ------------------------------------------------------------------

TASK = "understanding"
BASE_DIR = "/local1/bryanzhou008/Dialect/multimodal-dialectal-bias"
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, f"out/{TASK}/base_models_vqa")

scorer = t2v_metrics.VQAScore(model="clip-flant5-xxl")


def md5_hash(text: str) -> str:
    """Generate MD5 hash of the input string."""
    return hashlib.md5(text.encode()).hexdigest()


def get_average_score(img_dir, model_name, folder, prompt_key, ref_prompt, num_images, use_hash=False):
    """
    Compute the average similarity score for a set of generated images.
    If `use_hash` is True, hashes the prompt_key to find the directory.
    """
    subdir = md5_hash(prompt_key) if use_hash else prompt_key
    prompt_dir = os.path.join(img_dir, model_name, folder, subdir)
    scores = []

    for i in range(num_images):
        image_path = os.path.join(prompt_dir, f"{i}.jpg")
        if not os.path.exists(image_path):
            continue  # Skip missing images

        score_output = scorer(images=[image_path], texts=[ref_prompt])
        try:
            score_tensor = score_output[0][0]
        except TypeError:
            score_tensor = score_output

        score = (
            score_tensor.detach().cpu().item()
            if isinstance(score_tensor, torch.Tensor)
            else float(score_tensor)
        )
        scores.append(score)

    return sum(scores) / len(scores) if scores else 0.0


def main():
    for mode in MODES:
        use_hash = "rewrite" in mode

        for dialect in DIALECTS:
            data_file = os.path.join(DATA_DIR, "text", mode, f"{dialect}.csv")
            img_dir = os.path.join(DATA_DIR, "image", mode, dialect)

            print(f"\n{'='*60}\nEvaluating for MODE: '{mode}', DIALECT: '{dialect}'\n{'='*60}\n")

            try:
                df = pd.read_csv(data_file, encoding="unicode_escape")
            except Exception as e:
                print(f"Failed to load file {data_file}: {e}")
                continue

            dialect_prompts = df["Dialect_Prompt"].tolist()
            sae_prompts = df["SAE_Prompt"].tolist()

            for model in MODELS_TO_EVALUATE:
                NUM_IMAGES = 1 if model in ["dalle2", "dalle3"] else 9

                output_model_dir = os.path.join(OUTPUT_DIR, mode, dialect, model)
                os.makedirs(output_model_dir, exist_ok=True)

                breakdown_dialect_path = os.path.join(output_model_dir, "breakdown_dialect.csv")
                breakdown_sae_path = os.path.join(output_model_dir, "breakdown_sae.csv")
                summary_path = os.path.join(output_model_dir, "summary.csv")

                if (
                    os.path.exists(breakdown_dialect_path)
                    and os.path.exists(breakdown_sae_path)
                    and os.path.exists(summary_path)
                ):
                    print(f"Results already exist for MODE: '{mode}', DIALECT: '{dialect}', MODEL: '{model}'. Skipping.\n")
                    continue

                results_dialect, results_sae = [], []

                print(f"\n>> Evaluating MODEL: '{model}'")
                for i in tqdm(range(len(dialect_prompts)), desc="Processing prompts"):
                    dialect_prompt = dialect_prompts[i]
                    sae_prompt = sae_prompts[i]

                    score_dialect = get_average_score(
                        img_dir, model, "dialect_imgs", dialect_prompt, sae_prompt, NUM_IMAGES, use_hash=use_hash
                    )
                    score_dialect = round(score_dialect, 4)
                    results_dialect.append(
                        {
                            "Prompt_Index": i,
                            "Dialect_Prompt": dialect_prompt,
                            "SAE_Prompt": sae_prompt,
                            "Score": score_dialect,
                        }
                    )
                    print(f"Mode: {mode} | Dialect: {dialect} | Prompt {i} (dialect) | '{dialect_prompt}': {score_dialect:.4f}")

                    score_sae = get_average_score(
                        img_dir, model, "sae_imgs", sae_prompt, sae_prompt, NUM_IMAGES, use_hash=False
                    )
                    score_sae = round(score_sae, 4)
                    results_sae.append(
                        {
                            "Prompt_Index": i,
                            "SAE_Prompt": sae_prompt,
                            "Score": score_sae,
                        }
                    )
                    print(f"Mode: {mode} | Dialect: {dialect} | Prompt {i} (sae) | '{sae_prompt}': {score_sae:.4f}")

                avg_dialect = round(
                    sum(r["Score"] for r in results_dialect) / len(results_dialect) if results_dialect else 0,
                    4,
                )
                avg_sae = round(
                    sum(r["Score"] for r in results_sae) / len(results_sae) if results_sae else 0,
                    4,
                )

                print(f"\n--- Final Results for MODE: '{mode}', DIALECT: '{dialect}', MODEL: '{model}' ---")
                print(f"Overall Dialect Evaluation Average Score: {avg_dialect:.4f}\nOverall SAE Evaluation Average Score: {avg_sae:.4f}\n")

                pd.DataFrame(results_dialect).to_csv(breakdown_dialect_path, index=False)
                pd.DataFrame(results_sae).to_csv(breakdown_sae_path, index=False)

                absolute_drop = round(avg_sae - avg_dialect, 4)
                drop_ratio = round((absolute_drop / avg_sae) if avg_sae != 0 else 0, 4)

                summary_df = pd.DataFrame(
                    {
                        "Evaluation_Type": ["Dialect", "SAE", "Absolute Drop", "Drop Ratio"],
                        "Overall_Average_Score": [avg_dialect, avg_sae, absolute_drop, drop_ratio],
                    }
                )
                summary_df.to_csv(summary_path, index=False)

                print(f"Results saved to: {output_model_dir}\n")


if __name__ == "__main__":
    main()
