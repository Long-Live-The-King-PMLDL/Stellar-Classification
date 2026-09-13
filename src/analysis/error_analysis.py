"""
Error analysis for the best model.

Owner: Person 1.

Reads a single reports/preds/<model>_test.npz (format fixed by Person 1:
  y_true  : shape (n,)   int labels
  y_proba : shape (n, 3) predicted class probabilities

and produces:
  - overall macro-F1 / accuracy
  - per-class precision / recall / F1 (sklearn classification_report)
  - the most confused class pair (largest off-diagonal confusion count)
  - a handful of concrete misclassified examples for that pair, optionally
    joined against the processed test split so we can show real feature
    values instead of bare row indices.

Usage:
    python src/analysis/error_analysis.py \
        --npz reports/preds/lightgbm_test.npz \
        --class-names GALAXY QSO STAR \
        --test-csv data/processed/test.csv \
        --n-examples 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def load_predictions(npz_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    y_true = data["y_true"]
    y_proba = data["y_proba"]
    if y_true.ndim != 1:
        raise ValueError(f"y_true must be 1-D, got shape {y_true.shape}")
    if y_proba.ndim != 2 or y_proba.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"y_proba must be (n, n_classes) matching y_true; got {y_proba.shape} "
            f"vs y_true {y_true.shape}"
        )
    return y_true, y_proba


def most_confused_pair(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]
) -> tuple[int, int, int]:
    """Returns (true_idx, pred_idx, count) for the largest off-diagonal cell."""
    n = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    off_diag = cm.copy()
    np.fill_diagonal(off_diag, -1)
    true_idx, pred_idx = np.unravel_index(np.argmax(off_diag), off_diag.shape)
    return int(true_idx), int(pred_idx), int(cm[true_idx, pred_idx])


def top_confused_examples(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    true_idx: int,
    pred_idx: int,
    n_examples: int,
    test_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Rows where true==true_idx and pred==pred_idx, ranked by how
    confidently the model got it wrong (highest predicted probability
    for the wrong class first — the most informative failures)."""
    mask = (y_true == true_idx) & (y_pred == pred_idx)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return pd.DataFrame()
    confidence = y_proba[idx, pred_idx]
    order = idx[np.argsort(-confidence)][:n_examples]

    out = pd.DataFrame(
        {
            "row_index": order,
            "true_proba": y_proba[order, true_idx],
            "pred_proba": y_proba[order, pred_idx],
        }
    )
    if test_df is not None:
        feature_cols = test_df.reset_index(drop=True).loc[order]
        out = pd.concat([out.reset_index(drop=True), feature_cols.reset_index(drop=True)], axis=1)
    return out


def run(
    npz_path: str | Path,
    class_names: list[str],
    n_examples: int = 3,
    test_csv: str | Path | None = None,
) -> None:
    y_true, y_proba = load_predictions(npz_path)
    y_pred = y_proba.argmax(axis=1)

    print("=== Classification report ===")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

    true_idx, pred_idx, count = most_confused_pair(y_true, y_pred, class_names)
    print(
        f"=== Most confused pair: true={class_names[true_idx]} "
        f"predicted={class_names[pred_idx]} (count={count}) ==="
    )

    test_df = pd.read_csv(test_csv) if test_csv else None
    examples = top_confused_examples(
        y_true, y_pred, y_proba, true_idx, pred_idx, n_examples, test_df
    )
    if examples.empty:
        print("(no examples for this pair)")
    else:
        print(examples.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--class-names", nargs="+", required=True)
    parser.add_argument("--n-examples", type=int, default=3)
    parser.add_argument("--test-csv", default=None)
    args = parser.parse_args()
    run(args.npz, args.class_names, args.n_examples, args.test_csv)


if __name__ == "__main__":
    main()
