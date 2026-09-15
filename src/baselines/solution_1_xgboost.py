"""
Baseline 1: XGBoost with astronomical feature engineering
Adapted from: @emanuellcs, "Predicting Stellar Class | XGBoost", https://www.kaggle.com/code/emanuellcs/predicting-stellar-class-xgboost

Deviations from the original (to fit our protocol):
1. Data source: the original notebook was built for the synthetic Playground
   S6E6 dataset and used SDSS17 only as a lookup table for target encoding.
   We have SDSS17 only, so target encoding is removed; the
   spectral_type / galaxy_population / combo bins are kept as categorical
   features.
2. Multi-seed (3 seeds) x 5-fold CV -> a single fit on train with early
   stopping on val (our protocol: decisions on val, test used once).
3. The author's balanced accuracy -> our evaluate() (macro-F1 etc.);
   the class-weight calibration via differential evolution is removed —
   it optimized the author's metric, not macro-F1.
4. LabelEncoder removed — classes are already encoded in prepare.py
   (the mapping matches the author's: GALAXY=0, QSO=1, STAR=2).
5. -9999 artifacts: absent in the author's (playground) data; ours are
   already cleaned in prepare.py.

XGBoost hyperparameters, feature engineering and sample_weight='balanced' are kept as in the original.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.evaluation.metrics import evaluate
from src.data.loading import load_split, TARGET_COLUMN, FEATURES, CLASS_NAMES
from src.utils.paths import PREDS_DIR, ensure_dirs
from src.utils.seed import seed_everything

warnings.filterwarnings("ignore")
SEED = 42

NEEDS_SCALING = False


BANDS = ["u", "g", "r", "i", "z"]


def create_astronomical_bins(df: pd.DataFrame) -> pd.DataFrame:
    df["spectral_type"] = pd.cut(
        df["r"] - df["g"], [-np.inf, -1, -0.5, 0, np.inf],
        labels=["M", "G/K", "A/F", "O/B"],
    ).astype(str)
    df["galaxy_population"] = pd.cut(
        df["u"] - df["r"], [-np.inf, 2.2, np.inf],
        labels=["Blue_Cloud", "Red_Sequence"],
    ).astype(str)
    df["combo"] = df["spectral_type"] + "_" + df["galaxy_population"]
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = create_astronomical_bins(df)

    df["mag_mean"] = df[BANDS].mean(axis=1)
    df["mag_std"] = df[BANDS].std(axis=1)
    df["mag_max"] = df[BANDS].max(axis=1)
    df["mag_min"] = df[BANDS].min(axis=1)
    df["mag_range"] = df["mag_max"] - df["mag_min"]

    df["u_g"] = df["u"] - df["g"]
    df["g_r"] = df["g"] - df["r"]
    df["r_i"] = df["r"] - df["i"]
    df["i_z"] = df["i"] - df["z"]
    df["u_r"] = df["u"] - df["r"]
    df["u_i"] = df["u"] - df["i"]
    df["u_z"] = df["u"] - df["z"]
    df["g_i"] = df["g"] - df["i"]
    df["g_z"] = df["g"] - df["z"]
    df["r_z"] = df["r"] - df["z"]

    for b in BANDS:
        clipped = df[b].clip(lower=0, upper=40)
        df[f"flux_{b}"] = 10 ** (-0.4 * clipped)
    flux_cols = [f"flux_{b}" for b in BANDS]
    df["flux_mean"] = df[flux_cols].mean(axis=1)
    df["flux_std"] = df[flux_cols].std(axis=1)
    df["flux_max"] = df[flux_cols].max(axis=1)
    df["flux_min"] = df[flux_cols].min(axis=1)
    df["flux_range"] = df["flux_max"] - df["flux_min"]

    alpha_rad = np.radians(df["alpha"])
    delta_rad = np.radians(df["delta"])
    df["alpha_sin"], df["alpha_cos"] = np.sin(alpha_rad), np.cos(alpha_rad)
    df["delta_sin"], df["delta_cos"] = np.sin(delta_rad), np.cos(delta_rad)

    abs_redshift = df["redshift"].abs() + 1e-5
    for col in ["u_g", "g_r", "r_i", "i_z"]:
        df[f"{col}_per_redshift"] = df[col] / abs_redshift
    for b in BANDS:
        df[f"redshift_{b}"] = df["redshift"] * df[b]

    for c in ["spectral_type", "galaxy_population", "combo"]:
        df[c] = df[c].astype("category")
    return df


def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    df = engineer_features(df.copy())
    return df.drop(columns=[TARGET_COLUMN])


def get_model() -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        learning_rate=0.015,
        n_estimators=6000,
        early_stopping_rounds=150,
        max_depth=8,
        min_child_weight=5,
        max_delta_step=1,
        gamma=0.2,
        reg_alpha=0.5,
        reg_lambda=2.5,
        subsample=0.75,
        colsample_bytree=0.7,
        colsample_bylevel=0.8,
        enable_categorical=True,
        random_state=SEED,
        n_jobs=-1,
    )


def run(split: str = "val"):
    ensure_dirs()
    seed_everything(SEED)

    train_df, y_train = load_split("train")
    eval_df, y_eval = load_split(split)

    X_train = transform_features(train_df)
    X_eval = transform_features(eval_df)

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    t0 = time.time()
    model = get_model()
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_eval, y_eval)],
        verbose=500,
    )
    train_time = time.time() - t0

    y_proba = model.predict_proba(X_eval)
    m = evaluate(y_eval, y_proba)

    print(f"\n=== XGBoost baseline ({split}) ===")
    print(f"macro_f1={m['macro_f1']:.5f}  accuracy={m['accuracy']:.5f}  "
          f"roc_auc={m['roc_auc_ovr']:.5f}  train={train_time:.1f}s  "
          f"best_iteration={model.best_iteration}")

    if split == "test":
        np.savez(PREDS_DIR / "xgboost_test.npz",
                 y_true=y_eval, y_proba=y_proba)
        print(f"Saved: {'reports/preds/xgboost_test.npz'}")
    return m, train_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val",)
    args = parser.parse_args()
    run(args.split)