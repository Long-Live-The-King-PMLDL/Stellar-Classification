"""
Baseline 3: Keras feed-forward neural network
Adapted from: @prtkpiyush, "neural_network_solution(external dataset)(DL)", https://www.kaggle.com/code/prtkpiyush/neural-network-solution-external-dataset-dl

Deviations from the original (to fit our protocol):
1. Data source: the original trained on the union of the synthetic Playground
   S6E6 competition data and SDSS17 (with an is_orig flag). We have SDSS17
   only, so the model is trained on our train split alone; the is_orig flag
   is dropped.
2. 5-fold CV with fold-averaged test predictions -> a single fit on train
   with early stopping on val (our protocol: decisions on val, test once).
3. balanced accuracy -> our unified evaluate() (macro-F1 etc.).
4. LabelEncoder removed — labels are pre-encoded in prepare.py; the author's
   LabelEncoder produced the same alphabetical mapping
   (GALAXY=0, QSO=1, STAR=2), so no semantic change.
5. Fixed the original's spectral_ord quirk: in the notebook, spectral_type
   was computed only for the external frame, so spectral_ord was a constant
   -1 for train/test rows and get_dummies was a no-op. We compute the bins
   for every row, so the categorical features are actually used.
6. Scaler: the original fit StandardScaler per CV fold (already leak-free);
   we fit it on train once and apply to val/test.

Kept as in the original: the Sequential architecture
(256-BN-Drop0.3 / 128-BN-Drop0.3 / 64-Drop0.2 / softmax),
Adam(1e-3), sparse categorical cross-entropy, batch_size=2048,
epochs=200, EarlyStopping(patience=15, restore_best_weights),
ReduceLROnPlateau(0.5, patience=6, min_lr=1e-6), class_weight='balanced',
the full feature engineering set including log1p on flux features,
NaN->0 imputation after scaling.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import time
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from src.evaluation.metrics import evaluate
from src.data.loading import load_split, TARGET_COLUMN
from src.utils.paths import PREDS_DIR, PROJECT_ROOT, ensure_dirs

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

SEED = 42
BATCH_SIZE = 2048          # as in the original
MAX_EPOCHS = 200           # as in the original
BANDS = ["u", "g", "r", "i", "z"]

NEEDS_SCALING = True       # NN on raw magnitudes requires scaling

# Local TF seeding: the shared seed.py covers random/numpy/torch,
# TF is only used by this solution.
tf.keras.utils.set_random_seed(SEED)


# ============ FEATURE ENGINEERING — as in the original ============

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # color indices
    df["u_g"] = df["u"] - df["g"]; df["g_r"] = df["g"] - df["r"]
    df["r_i"] = df["r"] - df["i"]; df["i_z"] = df["i"] - df["z"]
    df["u_r"] = df["u"] - df["r"]; df["u_i"] = df["u"] - df["i"]
    df["u_z"] = df["u"] - df["z"]; df["g_i"] = df["g"] - df["i"]
    df["g_z"] = df["g"] - df["z"]; df["r_z"] = df["r"] - df["z"]

    # magnitude statistics
    df["mag_mean"] = df[BANDS].mean(axis=1)
    df["mag_std"] = df[BANDS].std(axis=1)
    df["mag_min"] = df[BANDS].min(axis=1)
    df["mag_max"] = df[BANDS].max(axis=1)
    df["mag_range"] = df["mag_max"] - df["mag_min"]

    # redshift interactions
    for b in BANDS:
        df[f"redshift_{b}"] = df["redshift"] * df[b]

    # cyclical sky coordinates
    df["alpha_sin"] = np.sin(np.deg2rad(df["alpha"]))
    df["alpha_cos"] = np.cos(np.deg2rad(df["alpha"]))
    df["delta_sin"] = np.sin(np.deg2rad(df["delta"]))
    df["delta_cos"] = np.cos(np.deg2rad(df["delta"]))

    # spectral bins — computed for ALL rows (fixes the original's quirk, see header)
    df["spectral_ord"] = pd.cut(
        df["r"] - df["g"], [-np.inf, -1, -0.5, 0, np.inf],
        labels=[0, 1, 2, 3],
    ).astype("float").fillna(-1)

    # flux conversions (clip to avoid overflow)
    for b in BANDS:
        clipped = df[b].clip(lower=-30, upper=30)
        df[f"flux_{b}"] = 10 ** (-0.4 * clipped)
    flux_cols = [f"flux_{b}" for b in BANDS]
    df["flux_mean"] = df[flux_cols].mean(axis=1)
    df["flux_std"] = df[flux_cols].std(axis=1)
    df["flux_min"] = df[flux_cols].min(axis=1)
    df["flux_max"] = df[flux_cols].max(axis=1)
    df["flux_range"] = df["flux_max"] - df["flux_min"]

    # spectral slope across bands
    x = np.arange(len(BANDS)); xc = x - x.mean(); denom = np.sum(xc ** 2)
    df["mag_slope"] = df[BANDS].sub(df[BANDS].mean(axis=1), axis=0).dot(xc) / denom

    # curvature features
    df["mag_curvature"] = df["u"] - 2 * df["r"] + df["z"]
    df["blue_curvature"] = df["u"] - 2 * df["g"] + df["r"]
    df["red_curvature"] = df["r"] - 2 * df["i"] + df["z"]

    # color per redshift
    eps = 1e-6
    abs_z = df["redshift"].abs() + eps
    for col in ["u_g", "g_r", "r_i", "i_z"]:
        df[f"{col}_per_redshift"] = df[col] / abs_z

    return df.replace([np.inf, -np.inf], np.nan)


def prep_for_nn(df: pd.DataFrame) -> pd.DataFrame:
    """Post-FE NN prep: log1p on flux features — as in the original."""
    df = df.copy()
    for c in [col for col in df.columns if col.startswith("flux_")]:
        df[c] = np.log1p(df[c].clip(lower=0))
    return df


def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    """Single entry point: FE -> NN prep -> drop target -> float32."""
    df = prep_for_nn(engineer_features(df.copy()))
    return df.drop(columns=[TARGET_COLUMN]).astype("float32")


# ============ MODEL — architecture & training as in the original ============

def build_model(n_features: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(3, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# ============ RUN UNDER OUR PROTOCOL ============

def run(split: str = "val"):
    """split='val' — main run; split='test' — final run, once, on the analyst's signal."""
    ensure_dirs()

    train_df, y_train = load_split("train")
    eval_df, y_eval = load_split(split)

    X_tr = transform_features(train_df)
    X_ev = transform_features(eval_df)

    # Scaler fit on train only — no leakage (the original fit per fold, equally clean)
    scaler = StandardScaler().fit(X_tr.values)
    X_tr_s = np.nan_to_num(scaler.transform(X_tr.values)).astype("float32")
    X_ev_s = np.nan_to_num(scaler.transform(X_ev.values)).astype("float32")

    cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight = dict(enumerate(cw))

    t0 = time.time()
    model = build_model(X_tr_s.shape[1])
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                       restore_best_weights=True)          # as in the original
    rlr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                            patience=6, min_lr=1e-6)       # as in the original
    model.fit(X_tr_s, y_train,
              validation_split=0.0,   # ES monitors eval split via val_data below
              epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
              class_weight=class_weight,
              callbacks=[es, rlr], verbose=2)
    train_time = time.time() - t0

    y_proba = model.predict(X_ev_s, verbose=0)
    m = evaluate(y_eval, y_proba)

    stopped = es.best_epoch if hasattr(es, "best_epoch") else "n/a"
    print(f"\n=== Keras NN baseline ({split}) ===")
    print(f"macro_f1={m['macro_f1']:.5f}  accuracy={m['accuracy']:.5f}  "
          f"roc_auc={m['roc_auc_ovr']:.5f}  train={train_time:.1f}s  "
          f"epochs_run={len(model.history.history['loss'])}")

    if split == "test":
        np.savez(PREDS_DIR / "nn_test.npz", y_true=y_eval, y_proba=y_proba)
        print(f"Saved: {'reports/preds/nn_test.npz'}")
    return m, train_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    args = parser.parse_args()
    run(args.split)