#!/usr/bin/env python3
"""
aggregate_results.py   •   2025‑05‑05

Create an aggregated_results.csv in *every*   .../concise/   and   .../detailed/
folder under the four outer results trees:

    out/understanding/base_models_vqa
    out/understanding/base_models_clip
    out/skintone/base_models_vqa
    out/skintone/base_models_clip

Each aggregated CSV has columns:

  Model Name,
  <dialect>_SAE, <dialect>_Dialect, <dialect>_Drop  (for aae, bre, che, ine, sge),
  average_SAE, average_Dialect, average_Drop

The script pulls numbers from each model’s *summary.csv*:

  Evaluation_Type,Overall_Average_Score
  Dialect,0.5207
  SAE,0.8119
  Absolute Drop,0.2912
  Drop Ratio,0.3587

Only SAE, Dialect, and Drop Ratio are used.

Author: (your name)
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
OUT_ROOT = Path("/Users/bryan/Desktop/wkdir/Dialect/multimodal-dialectal-bias/out")

OUTER_DIRS = [
    OUT_ROOT / "understanding" / "base_models_vqa",
    OUT_ROOT / "understanding" / "base_models_clip",
    OUT_ROOT / "skintone" / "base_models_vqa",
    OUT_ROOT / "skintone" / "base_models_clip",
]

MODES = ["concise", "detailed"]
DIALECTS = ["aae", "bre", "che", "ine", "sge"]

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def parse_summary(path: Path) -> tuple[float, float, float]:
    """Return (SAE, Dialect, Drop) from a summary.csv file."""
    df = pd.read_csv(path)
    sae = float(df.loc[df["Evaluation_Type"] == "SAE", "Overall_Average_Score"].iloc[0])
    dia = float(
        df.loc[df["Evaluation_Type"] == "Dialect", "Overall_Average_Score"].iloc[0]
    )
    drop = float(
        df.loc[df["Evaluation_Type"] == "Drop Ratio", "Overall_Average_Score"].iloc[0]
    )
    return sae, dia, drop


def collect_for_mode(mode_dir: Path) -> pd.DataFrame:
    """
    Build a DataFrame with one row per model aggregating across all five dialects
    for the given *mode* directory (either …/concise or …/detailed).
    """
    # model_stats[model_name][dialect] -> (sae, dialect, drop)
    model_stats: dict[str, dict[str, tuple[float, float, float]]] = {}

    for dialect in DIALECTS:
        dialect_dir = mode_dir / dialect
        if not dialect_dir.exists():
            log.warning("Missing dialect folder %s", dialect_dir)
            continue
        for model_dir in dialect_dir.iterdir():
            if not model_dir.is_dir():
                continue
            summary_path = model_dir / "summary.csv"
            if not summary_path.exists():
                log.warning("No summary.csv in %s", model_dir)
                continue

            sae, dia, drop = parse_summary(summary_path)
            stats = model_stats.setdefault(model_dir.name, {})
            stats[dialect] = (sae, dia, drop)

    # Build rows
    rows = []
    for model, dialect_dict in sorted(model_stats.items()):
        row: list[str | float] = [model]
        sae_vals, dia_vals, drop_vals = [], [], []

        for dialect in DIALECTS:
            sae, dia, drop = dialect_dict.get(dialect, (None, None, None))
            row.extend(
                [
                    f"{sae:.4f}" if sae is not None else "",
                    f"{dia:.4f}" if dia is not None else "",
                    f"{drop:.4f}" if drop is not None else "",
                ]
            )
            if sae is not None:
                sae_vals.append(sae)
                dia_vals.append(dia)
                drop_vals.append(drop)

        # Averages across available dialects
        avg_sae = sum(sae_vals) / len(sae_vals) if sae_vals else 0.0
        avg_dia = sum(dia_vals) / len(dia_vals) if dia_vals else 0.0
        avg_drop = sum(drop_vals) / len(drop_vals) if drop_vals else 0.0
        row.extend([f"{avg_sae:.4f}", f"{avg_dia:.4f}", f"{avg_drop:.4f}"])
        rows.append(row)

    # Column headers
    header = ["Model Name"]
    for dialect in DIALECTS:
        header.extend([f"{dialect}_SAE", f"{dialect}_Dialect", f"{dialect}_Drop"])
    header.extend(["average_SAE", "average_Dialect", "average_Drop"])

    return pd.DataFrame(rows, columns=header)


def process_outer_dir(outer_dir: Path):
    for mode in MODES:
        mode_dir = outer_dir / mode
        if not mode_dir.exists():
            log.warning("Missing mode directory %s", mode_dir)
            continue

        df = collect_for_mode(mode_dir)
        out_csv = mode_dir / "aggregated_results.csv"
        df.to_csv(out_csv, index=False)
        log.info("Wrote %s", out_csv.relative_to(OUT_ROOT))


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    for outer in OUTER_DIRS:
        process_outer_dir(outer)


if __name__ == "__main__":
    main()
