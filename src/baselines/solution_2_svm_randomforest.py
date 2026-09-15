"""
Baseline 2: Classical models — RBF SVM + Random Forest
(one Kaggle notebook, two classifiers)
Adapted from: @beyzanks, "Stellar Classification - 98.4% Acc 100% AUC", https://www.kaggle.com/code/beyzanks/stellar-classification-98-4-acc-100-auc

Deviations from the original (to fit our protocol / fix leaks):
1. Leaks removed. The original pipeline, in order, applied to the FULL dataset
   before any split: (a) LocalOutlierFactor row removal, (b) SMOTE
   oversampling, (c) StandardScaler fit. All three leak future information
   into training. We: skip (a) (our -9999 artifacts are already cleaned in
   prepare.py); replace (b) with class_weight='balanced'; fit the scaler on
   TRAIN only.
2. The original kept spec_obj_ID as a feature — a hard leak (same ID implies
   the same class per dataset docs). ID columns do not exist in our processed
   data, so this leak cannot be reproduced here.
3. Feature selection kept as in the original: alpha/delta dropped,
   photometric bands + redshift kept (plate/MJD/spec_obj_ID unavailable
   in our data).
4. The original encoded classes as GALAXY=0, STAR=1, QSO=2 (non-alphabetical).
   We train on pre-encoded labels from prepare.py (GALAXY=0, QSO=1, STAR=2);
   the encoding choice does not affect the learned decision function.
5. SVM: trained on a stratified train subsample (RBF SVM is ~O(n^2));
   probability=True added — the original model had no predict_proba,
   and it runs an internal 5-fold Platt calibration (slower but needed
   for our evaluate()).
6. The author's 33% train/test split -> our train/val/test protocol,
   metrics via the unified evaluate() (macro-F1 etc.).

Kept as in the original: SVC(kernel='rbf', C=1); default RandomForest
hyperparameters; the author's feature selection.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.evaluation.metrics import evaluate
from src.data.loading import load_split
from src.utils.paths import PREDS_DIR, PROJECT_ROOT, ensure_dirs
from src.utils.seed import seed_everything

warnings.filterwarnings("ignore")

SEED = 42
SVM_SAMPLE_SIZE = 20_000

# Feature selection as in the original: alpha/delta dropped by the author;
AUTHOR_FEATURES = ["u", "g", "r", "i", "z", "redshift"]


def run_svm(train_df, y_train, eval_df, y_eval, split):
    idx = np.arange(len(train_df))
    sub_idx, _ = train_test_split(
        idx, train_size=SVM_SAMPLE_SIZE, stratify=y_train, random_state=SEED
    )
    X_sub = train_df.iloc[sub_idx][AUTHOR_FEATURES].values
    y_sub = y_train[sub_idx]

    scaler = StandardScaler().fit(X_sub)
    X_tr = scaler.transform(X_sub)
    X_ev = scaler.transform(eval_df[AUTHOR_FEATURES].values)

    t0 = time.time()
    model = SVC(kernel="rbf", C=1, random_state=SEED,
                probability=True,
                class_weight="balanced")
    model.fit(X_tr, y_sub)
    train_time = time.time() - t0

    y_proba = model.predict_proba(X_ev)
    m = evaluate(y_eval, y_proba)

    print(f"\n=== SVM (RBF, C=1) baseline ({split}) ===")
    print(f"macro_f1={m['macro_f1']:.5f}  accuracy={m['accuracy']:.5f}  "
          f"roc_auc={m['roc_auc_ovr']:.5f}  train={train_time:.1f}s  "
          f"(trained on {len(y_sub)} subsampled rows)")

    if split == "test":
        np.savez(PREDS_DIR / "svm_test.npz", y_true=y_eval, y_proba=y_proba)
        print(f"Saved: {'reports/preds/svm_test.npz'}")
    return m, train_time


def run_rf(train_df, y_train, eval_df, y_eval, split):
    X_tr = train_df[AUTHOR_FEATURES].values
    X_ev = eval_df[AUTHOR_FEATURES].values

    t0 = time.time()
    model = RandomForestClassifier(
        random_state=SEED,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_tr, y_train)
    train_time = time.time() - t0

    y_proba = model.predict_proba(X_ev)
    m = evaluate(y_eval, y_proba)

    print(f"\n=== Random Forest (default) baseline ({split}) ===")
    print(f"macro_f1={m['macro_f1']:.5f}  accuracy={m['accuracy']:.5f}  "
          f"roc_auc={m['roc_auc_ovr']:.5f}  train={train_time:.1f}s")

    if split == "test":
        np.savez(PREDS_DIR / "randomforest_test.npz", y_true=y_eval, y_proba=y_proba)
        print(f"Saved: {'reports/preds/randomforest_test.npz'}")
    return m, train_time


def run(split: str = "val", which: str = "both"):
    ensure_dirs()
    seed_everything(SEED)

    train_df, y_train = load_split("train")
    eval_df, y_eval = load_split(split)

    results = {}
    if which in ("svm", "both"):
        results["svm_rbf"] = run_svm(train_df, y_train, eval_df, y_eval, split)
    if which in ("rf", "both"):
        results["random_forest"] = run_rf(train_df, y_train, eval_df, y_eval, split)

    print("\n=== Summary ===")
    for name, (m, t) in results.items():
        print(f"{name:15s} macro_f1={m['macro_f1']:.5f}  "
              f"accuracy={m['accuracy']:.5f}  train={t:.1f}s")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--model", choices=["both", "svm", "rf"], default="both")
    args = parser.parse_args()
    run(args.split, args.model)