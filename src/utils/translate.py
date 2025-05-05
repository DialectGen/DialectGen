#!/usr/bin/env python3
"""
translate_to_SAE_instruct.py  –  Dialect_Prompt ➜ Standard American English
Uses meta‑llama/Meta‑Llama‑3‑8B‑Instruct (chat model).

Prereqs:
    pip install "transformers>=4.40" "accelerate" "torch" pandas tqdm
    export HF_TOKEN=hf_your_token_here
"""

from __future__ import annotations
import os
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm
import transformers
import torch

# ------------------------------------------------------------------#
# Model setup                                                       #
# ------------------------------------------------------------------#

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

print("🔄 Loading Meta‑Llama‑3‑8B‑Instruct…")
pipeline = transformers.pipeline(
    task="text-generation",
    model=MODEL_ID,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    token=os.environ.get("HF_TOKEN"),  # needs a valid token
    return_full_text=False,
)
tokenizer = pipeline.tokenizer

# Terminators for chat models
TERMINATORS = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]

# ------------------------------------------------------------------#
# File iteration setup                                              #
# ------------------------------------------------------------------#

ROOTS = [
    Path("/local1/bryanzhou008/Dialect/multimodal-dialectal-bias/data/text/translate_concise"),
    Path("/local1/bryanzhou008/Dialect/multimodal-dialectal-bias/data/text/translate_detailed"),
]
FILENAMES = ["aae.csv", "bre.csv", "che.csv", "ine.csv", "sge.csv"]

PROMPT_TEMPLATE = (
    "If the following prompt is in Standard American English, please do not "
    "change it and reply with the exact same prompt, otherwise please translate "
    "it into Standard American English and reply with your final translated "
    "prompt. Output **only** the final prompt:\n\n{dialect_prompt}"
)

SYSTEM_MSG = (
    "You are an expert linguist who translates English dialect prompts into "
    "concise Standard American English. If the input is already SAE, return it "
    "unchanged."
)

# ------------------------------------------------------------------#
# Translation helper                                                #
# ------------------------------------------------------------------#
def translate(text: str) -> str:
    """Chat with LLAMA‑Instruct and return the assistant’s reply."""
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": PROMPT_TEMPLATE.format(dialect_prompt=text)},
    ]

    out = pipeline(
        messages,
        max_new_tokens=128,
        do_sample=False,          # greedy, deterministic
        eos_token_id=TERMINATORS,
        pad_token_id=tokenizer.eos_token_id,
    )[0]["generated_text"]

    # `generated_text` is a list of chat messages; get the assistant message
    assistant_msg = out[-1]["content"] if isinstance(out, list) else out
    return assistant_msg.strip()


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
