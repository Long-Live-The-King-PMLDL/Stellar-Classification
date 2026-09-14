"""One-time final-test entry point for the selected MLP v2 model.

Run this only after MLP v1 has produced its row in improvements.csv and the
team has approved final test evaluation.
"""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import torch
import yaml

from src.evaluation.metrics import evaluate
from src.models.train import load_config, test
from src.utils.paths import PROJECT_ROOT


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def require_v1_result(table_path: Path) -> tuple[pd.DataFrame, float]:
    if not table_path.exists():
        raise FileNotFoundError(
            f"{table_path} is missing. Role 4 must run the approved MLP v1 "
            "final test before MLP v2."
        )

    table = pd.read_csv(table_path)
    required_columns = {"version", "change", "val_f1", "test_f1", "delta"}
    missing = required_columns - set(table.columns)
    if missing:
        raise ValueError(f"Missing improvements.csv columns: {sorted(missing)}")

    v1 = table[table["version"] == "mlp_v1"]
    if len(v1) != 1 or not np.isfinite(float(v1.iloc[0]["test_f1"])):
        raise ValueError("Expected exactly one completed mlp_v1 row before finalizing v2.")
    return table, float(v1.iloc[0]["test_f1"])


def validate_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as predictions:
        if set(predictions.files) != {"y_true", "y_proba"}:
            raise ValueError("Prediction NPZ must contain only y_true and y_proba.")
        y_true = predictions["y_true"]
        y_proba = predictions["y_proba"]

    if y_true.ndim != 1 or y_proba.shape != (len(y_true), 3):
        raise ValueError(
            f"Invalid prediction shapes: y_true={y_true.shape}, y_proba={y_proba.shape}"
        )
    if not np.isfinite(y_proba).all():
        raise ValueError("Predicted probabilities contain non-finite values.")
    if not np.allclose(y_proba.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Predicted probabilities do not sum to one.")
    return y_true, y_proba


def finalize(config_path: Path) -> None:
    config = load_config(str(config_path))
    output = config["output"]
    checkpoint_path = repo_path(output["checkpoint_path"])
    predictions_path = repo_path(output["predictions_path"])
    table_path = repo_path(output["improvements_table_path"])

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Selected MLP v2 checkpoint is missing: {checkpoint_path}")
    if predictions_path.exists():
        raise FileExistsError(
            f"{predictions_path} already exists; refusing to repeat the one-time test."
        )

    table, v1_test_f1 = require_v1_result(table_path)

    temporary_config = copy.deepcopy(config)
    temporary_table = PROJECT_ROOT / ".dev" / "mlp_v2_final" / "unused.csv"
    temporary_config["output"]["improvements_table_path"] = str(temporary_table)
    temporary_config_path = temporary_table.with_suffix(".yaml")
    temporary_config_path.parent.mkdir(parents=True, exist_ok=True)
    with temporary_config_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(temporary_config, stream, sort_keys=False)

    os.chdir(PROJECT_ROOT)
    test(str(temporary_config_path))

    y_true, y_proba = validate_predictions(predictions_path)
    metrics = evaluate(y_true, y_proba)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    row = {
        "version": "mlp_v2",
        "change": "dropout 0.20->0.05; weight_decay 1e-4->1e-5",
        "val_f1": float(checkpoint["best_val_macro_f1"]),
        "test_f1": float(metrics["macro_f1"]),
        "delta": float(metrics["macro_f1"]) - v1_test_f1,
    }
    table = table[table["version"] != "mlp_v2"]
    table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    table.to_csv(table_path, index=False)

    print(f"MLP v2 row saved to: {table_path}")
    print(f"Test macro-F1: {row['test_f1']:.6f}")
    print(f"Delta vs MLP v1 test: {row['delta']:+.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize MLP v2 on the test split once.")
    parser.add_argument("--config", default="configs/mlp_v2.yaml")
    args = parser.parse_args()
    finalize(repo_path(args.config))


if __name__ == "__main__":
    main()
