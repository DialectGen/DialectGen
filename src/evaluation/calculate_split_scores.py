#!/usr/bin/env python3
"""
build_split_summaries.py  •  2025‑05‑04

Generate train_summary.csv, val_summary.csv and test_summary.csv for every
model directory under:

/local1/bryanzhou008/Dialect/multimodal-dialectal-bias/out/

– In the “understanding” branch, breakdown CSVs keep scores in a column named
  “Score”.
– In the “skintone” branch, the column is “Normalized_Score”.
The script now handles either name automatically (a file will never contain
both).

Author: (your name)
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
OUT_ROOT = Path(
    "/local1/bryanzhou008/Dialect/multimodal-dialectal-bias/out"
)
SPLIT_ROOT = Path(
    "/local1/bryanzhou008/Dialect/multimodal-dialectal-bias/data/text/train_val_test/4-1-1"
)

DIALECTS = ["aae", "bre", "che", "ine", "sge"]
MODES = ["concise", "detailed", "rewrite_concise", "rewrite_detailed", "translate_concise", "translate_detailed"]
SPLITS = ["train", "val", "test"]

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def load_split_prompts() -> dict[str, dict[str, dict[str, dict[str, set[str]]]]]:
    """Cache 4‑1‑1 prompt lists for every (mode, dialect, split)."""
    store: dict = {m: {d: {} for d in DIALECTS} for m in MODES}

    for mode in MODES:
        for dialect in DIALECTS:
            base = SPLIT_ROOT / mode / dialect
            for split in SPLITS:
                df = pd.read_csv(base / f"{split}.csv")
                store[mode][dialect][split] = {
                    "dialect": set(df["Dialect_Prompt"].astype(str).str.strip()),
                    "sae": set(df["SAE_Prompt"].astype(str).str.strip()),
                }
    return store


def score_column(df: pd.DataFrame) -> str:
    """Return the name of the score column in *df*."""
    if "Score" in df.columns:
        return "Score"
    if "Normalized_Score" in df.columns:
        return "Normalized_Score"
    raise ValueError("Neither 'Score' nor 'Normalized_Score' found in CSV.")


def summarize_scores(
    df_dialect: pd.DataFrame,
    df_sae: pd.DataFrame,
    prompts: dict[str, set[str]],
) -> tuple[float, float, float, float]:
    """Compute Dialect/SAE means, absolute drop, and drop ratio for one split."""
    mask_d = df_dialect["Dialect_Prompt"].astype(str).str.strip().isin(
        prompts["dialect"]
    )
    mask_s = df_sae["SAE_Prompt"].astype(str).str.strip().isin(prompts["sae"])

    col_d = score_column(df_dialect)
    col_s = score_column(df_sae)

    dialect_mean = df_dialect.loc[mask_d, col_d].mean()
    sae_mean = df_sae.loc[mask_s, col_s].mean()

    # Gracefully handle empty splits (shouldn’t normally occur)
    dialect_mean = 0.0 if pd.isna(dialect_mean) else dialect_mean
    sae_mean = 0.0 if pd.isna(sae_mean) else sae_mean

    abs_drop = sae_mean - dialect_mean
    drop_ratio = abs_drop / sae_mean if sae_mean else 0.0
    return dialect_mean, sae_mean, abs_drop, drop_ratio


def write_summary(
    dest: Path,
    dialect_mean: float,
    sae_mean: float,
    abs_drop: float,
    drop_ratio: float,
):
    """Save the four metrics to *dest*."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Evaluation_Type", "Overall_Average_Score"])
        w.writerow(["Dialect", f"{dialect_mean:.4f}"])
        w.writerow(["SAE", f"{sae_mean:.4f}"])
        w.writerow(["Absolute Drop", f"{abs_drop:.4f}"])
        w.writerow(["Drop Ratio", f"{drop_ratio:.4f}"])


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    split_prompts = load_split_prompts()
    log.info("Loaded 4‑1‑1 train/val/test prompt lists.")

    for task_dir in OUT_ROOT.iterdir():  # understanding / skintone
        if not task_dir.is_dir():
            continue
        for base_dir in task_dir.iterdir():  # base_models_clip / base_models_vqa
            if not base_dir.is_dir():
                continue
            for mode in MODES:
                mode_dir = base_dir / mode
                if not mode_dir.exists():
                    continue
                for dialect in DIALECTS:
                    dialect_dir = mode_dir / dialect
                    if not dialect_dir.exists():
                        continue
                    for model_dir in dialect_dir.iterdir():  # each model
                        if not model_dir.is_dir():
                            continue

                        bd_d = model_dir / "breakdown_dialect.csv"
                        bd_s = model_dir / "breakdown_sae.csv"
                        if not (bd_d.exists() and bd_s.exists()):
                            log.warning("Missing breakdown files in %s", model_dir)
                            continue

                        df_d = pd.read_csv(bd_d)
                        df_s = pd.read_csv(bd_s)

                        for split in SPLITS:
                            prompts = split_prompts[mode][dialect][split]
                            d_mean, s_mean, drop, ratio = summarize_scores(
                                df_d, df_s, prompts
                            )
                            write_summary(
                                model_dir / f"{split}_summary.csv",
                                d_mean,
                                s_mean,
                                drop,
                                ratio,
                            )

                        log.info("Created summaries for %s", model_dir.relative_to(OUT_ROOT))


if __name__ == "__main__":
    main()
