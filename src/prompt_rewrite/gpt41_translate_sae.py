#!/usr/bin/env python3
"""
translate_to_SAE_gpt41.py  –  Translates prompts in the SAE_Prompt column
                              to concise Standard American English (if needed)
                              using OpenAI gpt-4.1 Responses API.
"""

from __future__ import annotations
import os
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm
from openai import OpenAI

# ------------------------------------------------------------------#
# Model & client setup                                              #
# ------------------------------------------------------------------#
MODEL_ID = "gpt-4.1"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SYSTEM_MSG = (
    "You are an expert linguist who translates English dialect prompts into "
    "concise Standard American English. If the input is already SAE, return it "
    "unchanged."
)

PROMPT_TEMPLATE = (
    "If the following prompt is in Standard American English, please do not "
    "change it and reply with the exact same prompt, otherwise please translate "
    "it into Standard American English and reply with your final translated "
    "prompt. Output **only** the final prompt:\n\n{dialect_prompt}"
)

# ------------------------------------------------------------------#
# File iteration setup                                              #
# ------------------------------------------------------------------#
ROOTS = [
    Path("/local1/bryanzhou008/Dialect/multimodal-dialectal-bias/data/text/translate_concise_gpt41"),
    Path("/local1/bryanzhou008/Dialect/multimodal-dialectal-bias/data/text/translate_detailed_gpt41"),
]
FILENAMES = ["aae.csv", "bre.csv", "che.csv", "ine.csv", "sge.csv"]

# ------------------------------------------------------------------#
# Translation helper                                                #
# ------------------------------------------------------------------#
def translate(text: str) -> str:
    """Call GPT-4.1 via the Responses API and return the model’s reply."""
    response = client.responses.create(
        model=MODEL_ID,
        instructions=SYSTEM_MSG,
        input=PROMPT_TEMPLATE.format(dialect_prompt=text),
        temperature=0.0,   # deterministic
    )
    return response.output_text.strip()

# ------------------------------------------------------------------#
# CSV processing                                                    #
# ------------------------------------------------------------------#
def process_csv(path: Path) -> None:
    print(f"📄 Processing {path.relative_to(path.parent.parent)}")
    df = pd.read_csv(path)

    if "SAE_Prompt" not in df.columns:
        raise ValueError(f"'SAE_Prompt' column missing in {path}")

    for idx, sae_prompt in tqdm(
        list(df["SAE_Prompt"].items()),
        desc=path.name,
        leave=False,
    ):
        try:
            df.at[idx, "SAE_Prompt"] = translate(str(sae_prompt))
        except Exception as err:
            print(f"⚠️  Row {idx} in {path.name}: {err}")

    df.to_csv(path, index=False)
    print(f"💾 Saved {path}")

def main() -> None:
    for root in ROOTS:
        for fname in FILENAMES:
            csv_path = root / fname
            if csv_path.exists():
                process_csv(csv_path)
            else:
                print(f"⚠️  Missing file: {csv_path}")
    print("🎉 Done!")

if __name__ == "__main__":
    main()
