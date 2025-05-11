#!/usr/bin/env python3
"""
extract_original_prompts.py  –  ADD‐columns version (Python 3.6+)

• Reads every dialect CSV in “concise” and “detailed”.
• Extracts Dialect_Prompt / SAE_Prompt and renames them to
  Original_Dialect_Prompt / Original_SAE_Prompt.
• For every target rewrite_* / translate_* CSV:
      – If the file exists, the two columns are ADDED (or updated) in-place
        **while preserving all existing columns**.
      – If it does not exist, the script creates it with just those two columns.
• Verifies row counts match before writing.
"""

import sys
from pathlib import Path
from typing import List
import pandas as pd

# --------------------------------------------------------------------
# Adjust BASE if the root moves
# --------------------------------------------------------------------
BASE = Path("/local1/bryanzhou008/Dialect/multimodal-dialectal-bias/data/text")

SOURCE_MODES = {
    "concise": [
        "rewrite_concise",
        "translate_concise",
        "translate_concise_gpt41",
    ],
    "detailed": [
        "rewrite_detailed",
        "translate_detailed",
        "translate_detailed_gpt41",
    ],
}


def add_columns_to_file(
    src_subset: pd.DataFrame, target_path: Path
) -> None:
    """
    Add (or update) two “Original_*” columns in target_path.
    Create the file if it doesn't exist.
    """

    if target_path.exists():
        tgt_df = pd.read_csv(target_path)
        # --- sanity check ---------------------------------------------------
        if len(tgt_df) != len(src_subset):
            raise ValueError(
                f"Row-count mismatch for {target_path}: "
                f"{len(tgt_df)} rows vs {len(src_subset)} in source"
            )
    else:
        tgt_df = pd.DataFrame(index=range(len(src_subset)))

    # Ensure indices align, then add / replace the two columns
    tgt_df = tgt_df.reset_index(drop=True)
    tgt_df["Original_Dialect_Prompt"] = src_subset[
        "Original_Dialect_Prompt"
    ].values
    tgt_df["Original_SAE_Prompt"] = src_subset[
        "Original_SAE_Prompt"
    ].values

    target_path.parent.mkdir(parents=True, exist_ok=True)
    tgt_df.to_csv(target_path, index=False)


def process_source_file(src_csv: Path, target_dirs: List[Path]) -> None:
    """Extract the two columns from one source CSV and update every target file."""
    df = pd.read_csv(src_csv)
    subset = df[["Dialect_Prompt", "SAE_Prompt"]].rename(
        columns={
            "Dialect_Prompt": "Original_Dialect_Prompt",
            "SAE_Prompt": "Original_SAE_Prompt",
        }
    )

    for tdir in target_dirs:
        add_columns_to_file(subset, tdir / src_csv.name)


def main() -> None:
    for mode, targets in SOURCE_MODES.items():
        src_dir = BASE / mode
        if not src_dir.is_dir():
            print(f"⚠️  Missing source directory: {src_dir}", file=sys.stderr)
            continue

        target_dirs = [BASE / t for t in targets]

        for csv_path in src_dir.glob("*.csv"):
            try:
                process_source_file(csv_path, target_dirs)
                print(f"✅  Updated {csv_path.relative_to(BASE)} → {', '.join(t.name for t in target_dirs)}")
            except Exception as exc:
                print(f"❌  {csv_path}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
