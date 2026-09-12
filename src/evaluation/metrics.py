"""
Единая функция метрик проекта.
Все модели (бейзлайны, MLP v1, MLP v2) считают метрики ТОЛЬКО здесь.

Контракт (заморожен, менять только через объявление в общий чат):
    evaluate(y_true, y_proba) -> dict
    y_true:  (n,) int — метки классов
    y_proba: (n, 3) float — вероятности классов
    Порядок классов: GALAXY=0, QSO=1, STAR=2  (подтвердит Чел. 2 после сплита)
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

CLASS_NAMES = ["GALAXY", "QSO", "STAR"]  # TODO: подтвердить порядок у Чел. 2
N_CLASSES = len(CLASS_NAMES)


def evaluate(y_true, y_proba) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)

    if y_proba.ndim != 2 or y_proba.shape[1] != N_CLASSES:
        raise ValueError(
            f"y_proba должен быть формы (n, {N_CLASSES}), пришло {y_proba.shape}"
        )
    if y_true.shape[0] != y_proba.shape[0]:
        raise ValueError(
            f"Длины не совпадают: y_true={y_true.shape[0]}, y_proba={y_proba.shape[0]}"
        )

    y_pred = y_proba.argmax(axis=1)

    metrics = {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_per_class": precision_score(
            y_true, y_pred, average=None,
            labels=list(range(N_CLASSES)), zero_division=0,
        ).tolist(),
        "recall_per_class": recall_score(
            y_true, y_pred, average=None,
            labels=list(range(N_CLASSES)), zero_division=0,
        ).tolist(),
    }

    try:
        metrics["roc_auc_ovr"] = float(
            roc_auc_score(
                y_true, y_proba,
                multi_class="ovr", average="macro",
                labels=list(range(N_CLASSES)),
            )
        )
    except ValueError:
        # в выборке нет какого-то класса — метрика не определена
        metrics["roc_auc_ovr"] = float("nan")

    return metrics