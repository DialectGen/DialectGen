#!/usr/bin/env python3
"""
translate_SAE_prompt_to_SAE.py
———————————————
Overwrite the SAE_Prompt column in every CSV under
    translate_concise / translate_detailed
with its Standard-American-English translation produced by
meta-llama/Meta-Llama-3-8B-Instruct (chat model).

Requirements
------------
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

# ──────────────────────────────────────────────────────────────────────────────
# Model setup
# ──────────────────────────────────────────────────────────────────────────────
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

print("🔄 Loading Meta-Llama-3-8B-Instruct…")
pipeline = transformers.pipeline(
    task="text-generation",
    model=MODEL_ID,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    token=os.environ.get("HF_TOKEN"),       # Hugging Face token
    return_full_text=False,
)
tokenizer = pipeline.tokenizer
TERMINATORS = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# ──────────────────────────────────────────────────────────────────────────────
# Prompts (UNCHANGED)
# ──────────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────────
# Data locations
# ──────────────────────────────────────────────────────────────────────────────
ROOTS = [
    Path("/local1/bryanzhou008/Dialect/multimodal-dialectal-bias/data/text/translate_concise"),
    Path("/local1/bryanzhou008/Dialect/multimodal-dialectal-bias/data/text/translate_detailed"),
]
FILENAMES = ["aae.csv", "bre.csv", "che.csv", "ine.csv", "sge.csv"]

# ──────────────────────────────────────────────────────────────────────────────
# Translation helper
# ──────────────────────────────────────────────────────────────────────────────
def translate(text: str) -> str:
    """Return the chat model’s answer for a single prompt string."""
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": PROMPT_TEMPLATE.format(dialect_prompt=text)},
    ]
    out = pipeline(
        messages,
        max_new_tokens=128,
        do_sample=False,                 # deterministic / greedy
        eos_token_id=TERMINATORS,
        pad_token_id=tokenizer.eos_token_id,
    )[0]["generated_text"]

    # If transformers>=4.40 return list-of-dicts; otherwise plain string
    return out[-1]["content"].strip() if isinstance(out, list) else out.strip()

# ──────────────────────────────────────────────────────────────────────────────
# CSV processing
# ──────────────────────────────────────────────────────────────────────────────
TARGET_COLUMN = "SAE_Prompt"          # ← changed from Dialect_Prompt

def process_csv(path: Path) -> None:
    print(f"📄 Processing {path.relative_to(path.parent.parent)}")
    df = pd.read_csv(path)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"'{TARGET_COLUMN}' column missing in {path}")

    for idx, orig in tqdm(list(df[TARGET_COLUMN].items()), desc=path.name, leave=False):
        try:
            df.at[idx, TARGET_COLUMN] = translate(str(orig))
        except Exception as err:
            print(f"⚠️  Row {idx} in {path.name}: {err}")

    df.to_csv(path, index=False)
    print(f"💾 Saved {path}")

# ──────────────────────────────────────────────────────────────────────────────
# Main driver
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    for root in ROOTS:
        for fname in FILENAMES:
            csv_path = root / fname
            if csv_path.exists():
                process_csv(csv_path)
            else:
                print(f"⚠️  Missing file: {csv_path}")
    print("🎉 All done!")

if __name__ == "__main__":
    main()
