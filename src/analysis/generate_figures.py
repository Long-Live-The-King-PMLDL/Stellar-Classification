"""
Regenerate fig_1, fig_2, fig_4 from the files actually in the repo, so
anyone can verify the figures independently instead of trusting them
blindly.

Run from the repo root, after summary.csv exists (run build_summary.py
first if it doesn't):

    python src/analysis/generate_figures.py

Requires (already in the repo once Person 2/3/4/5 have delivered):
    data/processed/{train,val,test}.csv
    reports/tables/summary.csv
    reports/preds/*.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from viz.plots import plot_class_distribution, plot_confusion_matrix, plot_model_comparison

# Class order as announced by Person 2 after the split (see split_report.txt).
# If this ever changes, update it here — every figure depends on it.
CLASS_NAMES = ["GALAXY", "QSO", "STAR"]

# Maps each summary.csv "model" value to its prediction file in reports/preds/.
# Update this if a model is renamed.
MODEL_TO_NPZ = {
    "xgboost_fe": "xgboost_test.npz",
    "svm_rbf": "svm_test.npz",
    "random_forest": "randomforest_test.npz",
    "keras_nn": "nn_test.npz",
    "MLP_mlp_v1": "mlp_v1_test.npz",
    "MLP_mlp_v2": "mlp_v2_test.npz",
}

DISPLAY_NAME = {
    "xgboost_fe": "XGBoost",
    "svm_rbf": "SVM (RBF)",
    "random_forest": "RandomForest",
    "keras_nn": "Keras NN",
    "MLP_mlp_v1": "MLP v1",
    "MLP_mlp_v2": "MLP v2",
}


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    # --- fig_1: class distribution, straight from the real data -----------
    train = pd.read_csv(repo_root / "data/processed/train.csv")
    val = pd.read_csv(repo_root / "data/processed/val.csv")
    test = pd.read_csv(repo_root / "data/processed/test.csv")
    full = pd.concat([train, val, test], ignore_index=True)
    plot_class_distribution(
        full["label"].values, CLASS_NAMES,
        repo_root / "reports/figures/fig_1_class_distribution.png",
        title=f"Class Distribution (full processed dataset, n={len(full):,})",
    )
    print(f"fig_1 written ({len(full)} rows)")

    # --- fig_2 & fig_4: need summary.csv to know the current best model ---
    summary_path = repo_root / "reports/tables/summary.csv"
    summary = pd.read_csv(summary_path)
    best_row = summary.loc[summary["test_f1"].idxmax()]
    best_model, best_f1 = best_row["model"], best_row["test_f1"]
    print(f"Best model per summary.csv: {best_model} (test_f1={best_f1:.5f})")

    npz_path = repo_root / "reports/preds" / MODEL_TO_NPZ[best_model]
    d = np.load(npz_path)
    y_true, y_proba = d["y_true"], d["y_proba"]
    y_pred = y_proba.argmax(axis=1)
    plot_confusion_matrix(
        y_true, y_pred, CLASS_NAMES,
        repo_root / "reports/figures/fig_2_confusion_matrix.png",
        title=f"Confusion Matrix — {DISPLAY_NAME.get(best_model, best_model)} "
              f"(best of {len(summary)}, test macro-F1={best_f1:.3f})",
    )
    print(f"fig_2 written (from {npz_path.name})")

    labels = [DISPLAY_NAME.get(m, m) for m in summary["model"]]
    plot_model_comparison(
        labels, summary["test_f1"].tolist(),
        repo_root / "reports/figures/fig_4_model_comparison.png",
        title=f"Model Comparison — all {len(summary)} models (final)",
    )
    print(f"fig_4 written ({len(summary)} models)")


if __name__ == "__main__":
    main()
