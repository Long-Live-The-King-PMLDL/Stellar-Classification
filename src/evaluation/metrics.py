"""
TEMPORARY LOCAL STUB.

Owned by role 3.
Do not commit this implementation.
Replace with the shared project evaluate() when available.
"""

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def evaluate(
    y_true,
    y_proba,
) -> dict:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    y_pred = np.argmax(
        y_proba,
        axis=1,
    )

    precision, recall, _, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average=None,
            zero_division=0,
        )
    )

    metrics = {
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
        ),
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "precision_per_class": precision,
        "recall_per_class": recall,
    }

    try:
        metrics["roc_auc"] = roc_auc_score(
            y_true,
            y_proba,
            multi_class="ovr",
            average="macro",
        )
    except ValueError:
        metrics["roc_auc"] = float("nan")

    return metrics