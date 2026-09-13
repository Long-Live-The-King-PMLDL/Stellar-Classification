from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
CLASS_ORDER = {"GALAXY": 0, "QSO": 1, "STAR": 2}

RAW = Path("data/raw/star_classification.csv")
PROCESSED = Path("data/processed")
MODELS = Path("models")

ID_COLS = ["obj_ID", "spec_obj_ID", "run_ID", "rerun_ID", "cam_col",
           "field_ID", "plate", "MJD", "fiber_ID"]
TARGET = "class"


def main():
    np.random.seed(SEED)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW)
    n_start = len(df)
    print(f"Загружено: {n_start} строк x {df.shape[1]} колонок")

    bad = (df.drop(columns=[TARGET]) == -9999).any(axis=1)
    print(f"Строк с -9999: {bad.sum()} (индексы: {df.index[bad].tolist()})")
    df = df[~bad].reset_index(drop=True)
    n_clean = len(df)

    present = [c for c in ID_COLS if c in df.columns]
    df = df.drop(columns=present)
    FEATURES = [c for c in df.columns if c != TARGET]
    assert len(FEATURES) == 8, f"Ожидалось 8 фичей, получено: {FEATURES}"

    df["label"] = df[TARGET].map(CLASS_ORDER)
    assert df["label"].notna().all(), "В данных неизвестный класс!"

    X, y = df[FEATURES], df["label"]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)

    scaler = StandardScaler()
    X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), columns=FEATURES, index=X_tr.index)
    X_val_s = pd.DataFrame(scaler.transform(X_val), columns=FEATURES, index=X_val.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=FEATURES, index=X_test.index)
    joblib.dump(scaler, MODELS / "scaler.pkl")

    lines = [
        f"seed={SEED}",
        f"rows_start={n_start}",
        f"rows_dropped={n_start - n_clean}",
        f"rows_clean={n_clean}",
        f"features={FEATURES}",
        f"class_order={CLASS_ORDER}",
    ]
    for name, Xs, ys in [("train", X_tr_s, y_tr), ("val", X_val_s, y_val), ("test", X_test_s, y_test)]:
        out = Xs.copy()
        out["label"] = ys
        out.to_csv(PROCESSED / f"{name}.csv", index=False)
        dist = ys.value_counts().sort_index()
        line = f"{name}: n={len(ys)}, " + ", ".join(
            f"{cls}={cnt} ({cnt/len(ys)*100:.1f}%)" for cls, cnt in dist.items())
        lines.append(line)
        print(line)
    (PROCESSED / "split_report.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\n=== СКОПИРУЙ В ЧАТ ===")
    print("Порядок классов: GALAXY=0, QSO=1, STAR=2")
    print(f"Колонки X ({len(FEATURES)}): {FEATURES}")
    print(f"Строк: {n_clean} (удалено {n_start - n_clean} с -9999)")


if __name__ == "__main__":
    main()