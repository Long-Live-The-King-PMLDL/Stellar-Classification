"""
Shared data loader for all solutions.

Feature/target contract (owned by the data engineer):
    features: alpha, delta, u, g, r, i, z, redshift
    target:   label (int) — GALAXY=0, QSO=1, STAR=2
"""
import pandas as pd

from src.utils.paths import PROCESSED_DATA_DIR

TARGET_COLUMN = "label"
FEATURES = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
CLASS_NAMES = ["GALAXY", "QSO", "STAR"]


def load_split(name: str):
    df = pd.read_csv(PROCESSED_DATA_DIR / f"{name}.csv")
    df.columns = [c.strip() for c in df.columns]

    if TARGET_COLUMN not in df.columns:
        raise KeyError(
            f"Target column '{TARGET_COLUMN}' not found in {name}.csv. "
            f"Available columns: {list(df.columns)}."
        )
    return df, df[TARGET_COLUMN].values