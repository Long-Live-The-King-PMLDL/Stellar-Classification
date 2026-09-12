"""Запуск из корня проекта: python -m src.evaluation.test_metrics"""
import numpy as np
from src.evaluation.metrics import evaluate, N_CLASSES


def test_perfect():
    # идеальный классификатор -> все метрики = 1.0
    y_true = np.array([0, 0, 1, 1, 2, 2])
    m = evaluate(y_true, np.eye(N_CLASSES)[y_true])
    assert abs(m["macro_f1"] - 1.0) < 1e-9
    assert abs(m["accuracy"] - 1.0) < 1e-9
    assert abs(m["roc_auc_ovr"] - 1.0) < 1e-9


def test_hand_computed():
    # истину и предсказания считаем руками:
    # класс 0: P=0.5, R=0.5, F1=0.5
    # класс 1: P=2/3, R=1.0, F1=0.8
    # класс 2: P=1.0, R=2/3, F1=0.8
    # macro_f1 = 0.70, accuracy = 5/7
    y_true = np.array([0, 0, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0, 2])
    y_proba = np.full((7, N_CLASSES), 0.1)
    y_proba[np.arange(7), y_pred] = 0.8

    m = evaluate(y_true, y_proba)
    assert abs(m["macro_f1"] - 0.70) < 1e-9
    assert abs(m["accuracy"] - 5 / 7) < 1e-9
    assert np.allclose(m["precision_per_class"], [0.5, 2 / 3, 1.0])
    assert np.allclose(m["recall_per_class"], [0.5, 1.0, 2 / 3])
    assert 0.0 <= m["roc_auc_ovr"] <= 1.0


def test_missing_class():
    # класса 2 нет в y_true -> roc_auc не определён -> NaN, без падения
    y_true = np.array([0, 0, 1, 1])
    m = evaluate(y_true, np.eye(N_CLASSES)[y_true])
    assert np.isnan(m["roc_auc_ovr"])


if __name__ == "__main__":
    test_perfect();        print("perfect:        OK")
    test_hand_computed();  print("hand-computed:  OK")
    test_missing_class();  print("missing class:  OK")
    print("Все тесты пройдены")