#!/usr/bin/env python3
"""
translate_to_SAE_gpt41.py  –  Dialect_Prompt ➜ Standard American English
Uses OpenAI gpt-4.1 via the Responses API.

Prereqs:
    pip install "openai>=1.25" pandas tqdm
    export OPENAI_API_KEY=sk-...
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
client = OpenAI(  # picks up OPENAI_API_KEY from your env
    api_key=os.environ.get("OPENAI_API_KEY"),
)

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
        instructions=SYSTEM_MSG,                    # system-level guidance
        input=PROMPT_TEMPLATE.format(dialect_prompt=text),  # user message
        temperature=0.0,                            # deterministic like before
        # max_tokens=128,
    )
    return response.output_text.strip()            # main text field:contentReference[oaicite:0]{index=0}

# ------------------------------------------------------------------#
# CSV processing                                                    #
# ------------------------------------------------------------------#
def process_csv(path: Path) -> None:
    print(f"📄 Processing {path.relative_to(path.parent.parent)}")
    df = pd.read_csv(path)

    if "Dialect_Prompt" not in df.columns:
        raise ValueError(f"'Dialect_Prompt' column missing in {path}")

    for idx, dialect_prompt in tqdm(
        list(df["Dialect_Prompt"].items()),
        desc=path.name,
        leave=False,
    ):
        try:
            df.at[idx, "Dialect_Prompt"] = translate(str(dialect_prompt))
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
