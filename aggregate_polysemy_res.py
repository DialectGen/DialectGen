import json
import os
from argparse import ArgumentParser
import numpy as np

def main(args):
    polysemy_data_path = os.path.join(args.res_path)
    with open(polysemy_data_path, "r") as f:
        polysemy_data = json.load(f)
    
    results = []
    for dialect in ["aae", "bre", "che", "ine", "sge"]:
        results.extend(polysemy_data[f"{dialect}_polysemy"])
    avg_results = np.mean(results)
    print(f">>> Polysemy avg results: {avg_results:.4f}")

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--res_path", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)