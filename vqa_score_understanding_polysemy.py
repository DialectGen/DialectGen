# pip install t2v_metrics==1.1
import os
import pandas as pd
from tqdm import tqdm
import t2v_metrics
import json
from argparse import ArgumentParser
from collections import defaultdict, OrderedDict
from utils.misc import fix_seed
from const import *

fix_seed(42)

# Initialize the new scoring metric.
scorer = t2v_metrics.VQAScore(model='clip-flant5-xxl')


def get_average_score(res_dir, folder, gen_prompt, ref_prompt, num_images=9):
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
        score_output = scorer(images=[image_path], texts=[ref_prompt])
        try:
            score = score_output[0][0]
        except TypeError:
            score = score_output
        scores.append(score)

    return float(sum(scores)/len(scores))


def main(args):
    data_path = os.path.join(args.data_dir, args.dialect, "test.csv")
    df = pd.read_csv(data_path, encoding="unicode_escape")
    polysemic = df["polysemic"].tolist()
    polysemy_prompts = [item for i, item in enumerate(df["Polysemy_Prompt"].tolist()) if polysemic[i]]
    
    results = defaultdict(list)
    for i in tqdm(range(len(polysemy_prompts)), desc="Processing prompts"):
        polysemy_prompt = polysemy_prompts[i]

        # Evaluate polysemy images
        score = get_average_score(args.res_dir, f"{args.dialect}_polysemy", polysemy_prompt, polysemy_prompt)
        results[f"{args.dialect}_polysemy"].append(score)
        print(f"Prompt {i} (polysemy): {score:.4f}")

    print("\n------------------- Final Results -------------------")
    scores = results[f"{args.dialect}_polysemy"]
    avg_score = sum(scores) / len(scores)
    print(f"{args.dialect} polysemy total score: {avg_score:.4f}")
    results[f"{args.dialect}_polysemy_avg"] = avg_score
    
    output_file = os.path.join(args.res_dir, "vqa_score_understanding_polysemy.json")
    # Load existing results if file exists
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            existing_results = json.load(f)
    else:
        existing_results = {}

    # Merge new results
    existing_results.update(results)

    # Save back with sorted keys
    sorted_results = OrderedDict()
    for key in sorted(key for key in existing_results if not key.endswith('_avg')):
        sorted_results[key] = existing_results[key]
    for key in sorted(key for key in existing_results if key.endswith('_avg')):
        sorted_results[key] = existing_results[key]
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sorted_results, f, indent=4)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--res_dir", type=str, default="", help="the parent results directory with subfolders like sae and sge")
    parser.add_argument("--data_dir", type=str, default="./data/text/train_val_test/4-1-1/")
    parser.add_argument("--mode", type=str, default="concise")
    parser.add_argument("--dialect", type=str, default="sge")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    args.data_dir = os.path.join(args.data_dir, args.mode)
    dialect_list = args.dialect.split(",")
    for args.dialect in dialect_list:
        main(args)
