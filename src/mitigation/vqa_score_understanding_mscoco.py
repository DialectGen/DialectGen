import os
from tqdm import tqdm
import t2v_metrics
import json
from argparse import ArgumentParser
from utils.hf_captions import create_hf_coco_dataset
from utils.misc import fix_seed
from const import *

fix_seed(42)

# Initialize the new scoring metric.
scorer = t2v_metrics.VQAScore(model='clip-flant5-xxl')


def get_average_score(res_dir, folder, gen_prompt, num_images=9):
    """
    Compute the average similarity score for a set of generated images using the new metric.
    """
    prompt_dir = os.path.join(res_dir, folder, gen_prompt)
    scores = []

    for i in range(num_images):
        image_path = os.path.join(prompt_dir, f"{i}.jpg")
        if not os.path.exists(image_path):
            # Handle filename inconsistencies.
            processed_prompt = gen_prompt.replace("'", "_")
            prompt_dir = os.path.join(res_dir, folder, processed_prompt)
            image_path = os.path.join(prompt_dir, f"{i}.jpg")

        # Compute the score for the (image, text) pair.
        score_output = scorer(images=[image_path], texts=[gen_prompt])
        try:
            score = score_output[0][0]
        except TypeError:
            score = score_output
        scores.append(score)
    return float(sum(scores)/len(scores))


def main(args):
    mscoco = create_hf_coco_dataset(CAPTION_FILE_PATH, IMAGE_FOLDER_PATH).select(range(4950, 5000))
    prompts = [ct[0] for ct in mscoco["captions"]]

    results = {"mscoco": []}
    for i in tqdm(range(len(prompts)), desc="Processing prompts"):
        prompt = prompts[i]

        # Evaluate dialect images (using SAE prompt as reference).
        score = get_average_score(args.res_dir, '', prompt)
        results["mscoco"].append(score)
        print(f"Prompt {i} (mscoco): {score:.4f}")

    print("\n------------------- Final Results -------------------")
    scores = results["mscoco"]
    avg_score = sum(scores) / len(scores)
    print(f"mscoco total score: {avg_score:.4f}")
    results["mscoco_avg"] = avg_score
    
    output_file = os.path.join(args.res_dir, "vqa_score_understanding_mscoco.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--res_dir", type=str, default="", help="the parent results directory with prompt subfolders")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
