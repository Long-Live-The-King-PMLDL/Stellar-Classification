"""Small validation-only hyperparameter search for MLP v2.

The script deliberately reuses ``src.models.train.train`` without changing the
MLP v1 implementation. It never calls the final-test path.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import time
from pathlib import Path
from typing import Any

# Role 4 saves a training figure after every run. Force a headless backend so
# tuning also works on CI and machines without Tcl/Tk.
os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd
import torch
import yaml

from src.models.train import train
from src.utils.paths import PROJECT_ROOT


ALLOWED_OVERRIDE_SECTIONS = {"model", "training", "optimizer"}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(value, stream, sort_keys=False)


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)


def validate_trials(trials: list[dict[str, Any]]) -> None:
    if not 4 <= len(trials) <= 6:
        raise ValueError("The search must contain between 4 and 6 trials.")

    names: set[str] = set()
    for trial in trials:
        name = trial.get("name")
        overrides = trial.get("overrides")
        if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9_]+", name):
            raise ValueError(f"Invalid trial name: {name!r}")
        if name in names:
            raise ValueError(f"Duplicate trial name: {name}")
        names.add(name)

        if not isinstance(overrides, dict):
            raise ValueError(f"Trial {name} has no overrides mapping.")
        unexpected = set(overrides) - ALLOWED_OVERRIDE_SECTIONS
        if unexpected:
            raise ValueError(
                f"Trial {name} cannot override {sorted(unexpected)}. "
                "Data, seed, and outputs are fixed during tuning."
            )


def trial_config(
    base: dict[str, Any],
    trial: dict[str, Any],
    work_dir: Path,
    epochs_override: int | None = None,
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    deep_update(config, trial["overrides"])

    if epochs_override is not None:
        config["training"]["epochs"] = epochs_override
        config["training"]["patience"] = epochs_override

    name = trial["name"]
    config["output"] = {
        "checkpoint_path": str(work_dir / f"{name}.pt"),
        "predictions_path": str(work_dir / f"{name}_test.npz"),
        "training_figure_path": str(work_dir / f"{name}_curves.png"),
        "improvements_table_path": str(work_dir / "unused_improvements.csv"),
    }
    return config


def result_row(
    name: str,
    config: dict[str, Any],
    baseline_val_f1: float,
    checkpoint: dict[str, Any],
    elapsed: float,
) -> dict[str, Any]:
    val_f1 = float(checkpoint["best_val_macro_f1"])
    return {
        "trial": name,
        "config_signature": config_signature(config),
        "hidden_dims": json.dumps(config["model"]["hidden_dims"]),
        "dropout": float(config["model"]["dropout"]),
        "learning_rate": float(config["training"]["learning_rate"]),
        "weight_decay": float(config["training"]["weight_decay"]),
        "batch_size": int(config["training"]["batch_size"]),
        "optimizer": config["optimizer"]["name"],
        "epochs": int(config["training"]["epochs"]),
        "best_epoch": int(checkpoint["best_epoch"]),
        "val_f1": val_f1,
        "delta_vs_v1": val_f1 - baseline_val_f1,
        "train_time_s": elapsed,
        "status": "ok",
        "error": "",
    }


def save_results(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def config_signature(config: dict[str, Any]) -> str:
    tuned_sections = {
        key: config[key]
        for key in sorted(ALLOWED_OVERRIDE_SECTIONS)
    }
    return json.dumps(tuned_sections, sort_keys=True, separators=(",", ":"))


def run_search(search_config_path: Path, smoke: bool = False) -> pd.DataFrame:
    search = load_yaml(search_config_path)
    trials = search.get("trials")
    if not isinstance(trials, list):
        raise ValueError("Search configuration must contain a trials list.")
    validate_trials(trials)

    base = load_yaml(repo_path(search["base_config"]))
    baseline_val_f1 = float(search["baseline_val_f1"])

    if smoke:
        trials = trials[:1]
        work_dir = PROJECT_ROOT / ".dev" / "mlp_v2_smoke"
        results_path = work_dir / "results.csv"
        epochs_override = 2
    else:
        work_dir = repo_path(search["work_dir"])
        results_path = repo_path(search["results_path"])
        epochs_override = None

    work_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    existing_by_trial: dict[str, dict[str, Any]] = {}
    if not smoke and results_path.exists():
        existing = pd.read_csv(results_path).fillna("")
        existing_by_trial = {
            str(row["trial"]): row.to_dict()
            for _, row in existing.iterrows()
        }

    # Role 4's loader uses repository-relative paths.
    os.chdir(PROJECT_ROOT)

    for index, trial in enumerate(trials, start=1):
        name = trial["name"]
        config = trial_config(base, trial, work_dir, epochs_override)
        generated_config_path = work_dir / f"{name}.yaml"
        write_yaml(generated_config_path, config)

        previous = existing_by_trial.get(name)
        checkpoint_path = repo_path(config["output"]["checkpoint_path"])
        if (
            previous
            and previous.get("status") == "ok"
            and previous.get("config_signature") == config_signature(config)
            and checkpoint_path.exists()
        ):
            print(f"\n=== Trial {index}/{len(trials)}: {name} (resumed) ===", flush=True)
            rows.append(previous)
            save_results(results_path, rows)
            continue

        print(f"\n=== Trial {index}/{len(trials)}: {name} ===", flush=True)
        started = time.perf_counter()
        try:
            train(str(generated_config_path))
            checkpoint = torch.load(
                repo_path(config["output"]["checkpoint_path"]),
                map_location="cpu",
                weights_only=False,
            )
            row = result_row(
                name,
                config,
                baseline_val_f1,
                checkpoint,
                time.perf_counter() - started,
            )
        except Exception as exc:
            row = {
                "trial": name,
                "config_signature": config_signature(config),
                "hidden_dims": json.dumps(config["model"]["hidden_dims"]),
                "dropout": float(config["model"]["dropout"]),
                "learning_rate": float(config["training"]["learning_rate"]),
                "weight_decay": float(config["training"]["weight_decay"]),
                "batch_size": int(config["training"]["batch_size"]),
                "optimizer": config["optimizer"]["name"],
                "epochs": int(config["training"]["epochs"]),
                "best_epoch": pd.NA,
                "val_f1": pd.NA,
                "delta_vs_v1": pd.NA,
                "train_time_s": time.perf_counter() - started,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(row["error"], flush=True)

        rows.append(row)
        save_results(results_path, rows)

    results = pd.DataFrame(rows)
    successful = results[results["status"] == "ok"]
    if successful.empty:
        raise RuntimeError(f"All tuning trials failed. See {results_path}")

    if not smoke:
        best_row = successful.loc[successful["val_f1"].astype(float).idxmax()]
        best_trial = next(t for t in trials if t["name"] == best_row["trial"])
        selected = trial_config(base, best_trial, work_dir)
        selected["output"] = copy.deepcopy(search["final_output"])
        selected_path = repo_path(search["selected_config_path"])
        write_yaml(selected_path, selected)

        print("\n=== Selection ===", flush=True)
        print(f"Best trial: {best_row['trial']}", flush=True)
        print(f"Validation macro-F1: {float(best_row['val_f1']):.6f}", flush=True)
        print(f"Delta vs MLP v1: {float(best_row['delta_vs_v1']):+.6f}", flush=True)
        print(f"Selected config: {selected_path}", flush=True)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune MLP v2 on validation macro-F1.")
    parser.add_argument(
        "--search-config",
        default="configs/mlp_v2_search.yaml",
        help="Path to the search YAML.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the first trial for two epochs without selecting a final config.",
    )
    args = parser.parse_args()
    run_search(repo_path(args.search_config), smoke=args.smoke)


if __name__ == "__main__":
    main()
