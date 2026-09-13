"""
Build reports/tables/summary.csv from baselines.csv (Person 3) and
improvements.csv (Persons 4 & 5).

Owner: Person 1.

Input schemas (as agreed in the team rules):
    baselines.csv     : model | [source] | val_f1 | test_f1 | accuracy | train_time_s
    improvements.csv  : version | change | val_f1 | test_f1 | delta

`source` in baselines.csv is optional (present only if baselines are
reproduced Kaggle notebooks rather than plain sklearn models); the
script fills it with "own model" for the improvement rows.

Output schema (summary.csv, 5 rows: 3 baselines + MLP_v1 + MLP_v2):
    model | kind | val_f1 | test_f1 | accuracy | train_time_s | notes

Usage:
    python src/analysis/build_summary.py \
        --baselines reports/tables/baselines.csv \
        --improvements reports/tables/improvements.csv \
        --out reports/tables/summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _load_baselines(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"model", "val_f1", "test_f1", "accuracy", "train_time_s"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"baselines.csv is missing columns: {sorted(missing)}")

    notes = df["source"].fillna("") if "source" in df.columns else ""
    out = pd.DataFrame(
        {
            "model": df["model"],
            "kind": "baseline",
            "val_f1": df["val_f1"],
            "test_f1": df["test_f1"],
            "accuracy": df["accuracy"],
            "train_time_s": df["train_time_s"],
            "notes": notes,
        }
    )
    return out


def _load_improvements(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"version", "change", "val_f1", "test_f1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"improvements.csv is missing columns: {sorted(missing)}")

    notes = df["change"].astype(str)
    if "delta" in df.columns:
        notes = notes + " (delta test_f1 = " + df["delta"].astype(str) + ")"

    out = pd.DataFrame(
        {
            "model": "MLP_" + df["version"].astype(str),
            "kind": "own model",
            "val_f1": df["val_f1"],
            "test_f1": df["test_f1"],
            # improvements.csv has no accuracy / train_time_s in the agreed
            # schema; keep the columns so summary.csv stays uniform, but
            # leave them blank rather than inventing numbers.
            "accuracy": pd.NA,
            "train_time_s": pd.NA,
            "notes": notes,
        }
    )
    return out


def build_summary(baselines_path: str | Path, improvements_path: str | Path) -> pd.DataFrame:
    baselines = _load_baselines(baselines_path)
    improvements = _load_improvements(improvements_path)
    summary = pd.concat([baselines, improvements], ignore_index=True)
    summary = summary.sort_values("test_f1", ascending=False).reset_index(drop=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", default="reports/tables/baselines.csv")
    parser.add_argument("--improvements", default="reports/tables/improvements.csv")
    parser.add_argument("--out", default="reports/tables/summary.csv")
    args = parser.parse_args()

    summary = build_summary(args.baselines, args.improvements)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(summary)} rows:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
