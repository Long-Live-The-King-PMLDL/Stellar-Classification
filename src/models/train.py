import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import evaluate
from src.models.mlp import StellarMLP
from src.utils.seed import seed_everything


# ============================================================
# Configuration
# ============================================================

def load_config(config_path: str) -> dict:
    """Load YAML configuration."""

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


# ============================================================
# Data
# ============================================================

def encode_target(
    y_raw: pd.Series,
    class_order: Optional[list[str]],
) -> np.ndarray:
    """
    Convert target labels to integer indices.

    The official processed dataset currently contains numeric
    labels (0, 1, 2), so they are normally returned directly.

    String labels are also supported if class_order is provided.
    """

    if pd.api.types.is_numeric_dtype(y_raw):
        return y_raw.to_numpy(dtype=np.int64)

    if class_order is None:
        raise ValueError(
            "Target labels are strings, but data.class_order is null. "
            "Set class_order in the YAML configuration."
        )

    class_to_idx = {
        class_name: idx
        for idx, class_name in enumerate(class_order)
    }

    unknown_classes = set(y_raw.unique()) - set(class_to_idx)

    if unknown_classes:
        raise ValueError(
            f"Unknown target classes found: {sorted(unknown_classes)}"
        )

    return (
        y_raw
        .map(class_to_idx)
        .to_numpy(dtype=np.int64)
    )


def load_split(
    path: str,
    target_column: str,
    class_order: Optional[list[str]],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load one processed train/val/test split.

    The processed CSV files produced by the Data Engineer are
    already scaled, therefore no scaler is applied here.
    """

    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(
            f"Dataset split was not found: {path}"
        )

    df = pd.read_csv(path_obj)

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found in '{path}'. "
            f"Available columns: {df.columns.tolist()}"
        )

    feature_columns = [
        column
        for column in df.columns
        if column != target_column
    ]

    X_df = df[feature_columns]
    y_raw = df[target_column]

    # Processed files from Чел. 2 are already standardized.
    X = X_df.to_numpy(dtype=np.float32)

    y = encode_target(
        y_raw=y_raw,
        class_order=class_order,
    )

    if len(X) != len(y):
        raise ValueError(
            f"Feature/target length mismatch in '{path}': "
            f"{len(X)} != {len(y)}"
        )

    if not np.isfinite(X).all():
        raise ValueError(
            f"Non-finite feature values found in '{path}'."
        )

    return X, y, feature_columns


def validate_labels(
    y: np.ndarray,
    num_classes: int,
    split_name: str,
) -> None:
    """Validate that numeric labels fit the configured class count."""

    unique_labels = np.unique(y)

    if len(unique_labels) == 0:
        raise ValueError(
            f"{split_name} contains no labels."
        )

    if unique_labels.min() < 0 or unique_labels.max() >= num_classes:
        raise ValueError(
            f"{split_name} contains labels {unique_labels.tolist()}, "
            f"but model.num_classes={num_classes}."
        )


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Create a deterministic PyTorch DataLoader."""

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
    )

    y_tensor = torch.tensor(
        y,
        dtype=torch.long,
    )

    dataset = TensorDataset(
        X_tensor,
        y_tensor,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
    )


# ============================================================
# Training
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train the model for one epoch."""

    model.train()

    total_loss = 0.0
    total_samples = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(X_batch)

        loss = criterion(
            logits,
            y_batch,
        )

        loss.backward()
        optimizer.step()

        batch_size = X_batch.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Training DataLoader is empty.")

    return total_loss / total_samples


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    """
    Evaluate the model using the project's shared evaluate()
    function supplied by Чел. 3.
    """

    model.eval()

    total_loss = 0.0
    total_samples = 0

    y_true_parts = []
    y_proba_parts = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)

        loss = criterion(
            logits,
            y_batch,
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        batch_size = X_batch.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        y_true_parts.append(
            y_batch.cpu().numpy()
        )

        y_proba_parts.append(
            probabilities.cpu().numpy()
        )

    if total_samples == 0:
        raise ValueError("Evaluation DataLoader is empty.")

    y_true = np.concatenate(
        y_true_parts,
        axis=0,
    )

    y_proba = np.concatenate(
        y_proba_parts,
        axis=0,
    )

    average_loss = total_loss / total_samples

    metrics = evaluate(
        y_true,
        y_proba,
    )

    return (
        average_loss,
        metrics,
        y_true,
        y_proba,
    )


# ============================================================
# Optimizer
# ============================================================

def create_optimizer(
    model: nn.Module,
    config: dict,
) -> torch.optim.Optimizer:
    """Build optimizer from YAML configuration."""

    optimizer_name = (
        config["optimizer"]["name"]
        .lower()
    )

    training_cfg = config["training"]

    learning_rate = float(
        training_cfg["learning_rate"]
    )

    weight_decay = float(
        training_cfg["weight_decay"]
    )

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    raise ValueError(
        f"Unsupported optimizer: {optimizer_name}"
    )


# ============================================================
# Training curves
# ============================================================

def save_training_curve(
    history: list[dict],
    output_path: str,
) -> None:
    """Save fig_3_mlp_training_curves.png."""

    if not history:
        raise ValueError(
            "Training history is empty."
        )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_df = pd.DataFrame(history)

    plt.figure(figsize=(8, 5))

    plt.plot(
        history_df["epoch"],
        history_df["train_loss"],
        label="Train loss",
    )

    plt.plot(
        history_df["epoch"],
        history_df["val_loss"],
        label="Validation loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("MLP v1 Training Curves")

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()

    print(
        f"Training figure saved to: {output_path}"
    )


# ============================================================
# Improvements table
# ============================================================

def save_improvements_row(
    table_path: str,
    val_f1: float,
    test_f1: float,
) -> None:
    """
    Create/update the MLP v1 row in improvements.csv.

    Чел. 5 can later append/update the MLP v2 row.
    """

    table_path = Path(table_path)

    table_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    row = {
        "version": "mlp_v1",
        "change": "Base MLP architecture",
        "val_f1": val_f1,
        "test_f1": test_f1,
        "delta": 0.0,
    }

    if table_path.exists():
        df = pd.read_csv(table_path)

        if "version" in df.columns:
            df = df[
                df["version"] != "mlp_v1"
            ]

        df = pd.concat(
            [
                df,
                pd.DataFrame([row]),
            ],
            ignore_index=True,
        )

    else:
        df = pd.DataFrame([row])

    df.to_csv(
        table_path,
        index=False,
    )

    print(
        f"Updated table: {table_path}"
    )


# ============================================================
# Model builder
# ============================================================

def build_model(
    config: dict,
    actual_input_dim: int,
) -> StellarMLP:
    """Create StellarMLP from YAML configuration."""

    model_cfg = config["model"]

    configured_input_dim = model_cfg.get(
        "input_dim"
    )

    if configured_input_dim is None:
        input_dim = actual_input_dim

    else:
        input_dim = int(
            configured_input_dim
        )

        if input_dim != actual_input_dim:
            raise ValueError(
                "Configured model.input_dim does not match "
                f"the processed dataset: "
                f"{input_dim} != {actual_input_dim}"
            )

    return StellarMLP(
        input_dim=input_dim,
        hidden_dims=model_cfg["hidden_dims"],
        num_classes=int(
            model_cfg["num_classes"]
        ),
        dropout=float(
            model_cfg["dropout"]
        ),
    )


# ============================================================
# Main training procedure
# ============================================================

def train(config_path: str) -> None:
    """
    Train MLP v1 using only the official train and validation
    splits.

    The test split is NOT touched here.
    """

    config = load_config(
        config_path
    )

    seed = int(
        config["seed"]
    )

    seed_everything(seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    data_cfg = config["data"]
    training_cfg = config["training"]
    model_cfg = config["model"]
    output_cfg = config["output"]

    class_order = data_cfg.get(
        "class_order"
    )

    num_classes = int(
        model_cfg["num_classes"]
    )

    # --------------------------------------------------------
    # Load official train / validation splits
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        train_features,
    ) = load_split(
        path=data_cfg["train_path"],
        target_column=data_cfg["target_column"],
        class_order=class_order,
    )

    (
        X_val,
        y_val,
        val_features,
    ) = load_split(
        path=data_cfg["val_path"],
        target_column=data_cfg["target_column"],
        class_order=class_order,
    )

    if train_features != val_features:
        raise ValueError(
            "Train and validation feature columns/order differ.\n"
            f"Train: {train_features}\n"
            f"Val:   {val_features}"
        )

    validate_labels(
        y_train,
        num_classes=num_classes,
        split_name="Train split",
    )

    validate_labels(
        y_val,
        num_classes=num_classes,
        split_name="Validation split",
    )

    print(
        f"Train shape:      {X_train.shape}"
    )

    print(
        f"Validation shape: {X_val.shape}"
    )

    print(
        f"Features ({len(train_features)}): "
        f"{train_features}"
    )

    print(
        f"Train labels: {np.unique(y_train).tolist()}"
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    batch_size = int(
        training_cfg["batch_size"]
    )

    train_loader = make_loader(
        X=X_train,
        y=y_train,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    val_loader = make_loader(
        X=X_val,
        y=y_val,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = build_model(
        config=config,
        actual_input_dim=X_train.shape[1],
    )

    model = model.to(device)

    print()
    print(model)
    print()

    # --------------------------------------------------------
    # Loss + optimizer
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = create_optimizer(
        model=model,
        config=config,
    )

    # --------------------------------------------------------
    # Output path
    # --------------------------------------------------------

    checkpoint_path = Path(
        output_cfg["checkpoint_path"]
    )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    epochs = int(
        training_cfg["epochs"]
    )

    patience = int(
        training_cfg["patience"]
    )

    best_val_f1 = float("-inf")
    best_epoch = 0

    epochs_without_improvement = 0

    history = []

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(
        1,
        epochs + 1,
    ):

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        (
            val_loss,
            val_metrics,
            _,
            _,
        ) = evaluate_model(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        if "macro_f1" not in val_metrics:
            raise KeyError(
                "evaluate() must return a dictionary "
                "containing the key 'macro_f1'."
            )

        val_f1 = float(
            val_metrics["macro_f1"]
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_f1": val_f1,
            }
        )

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_macro_f1={val_f1:.4f}"
        )

        # ----------------------------------------------------
        # Save best model according to validation macro-F1
        # ----------------------------------------------------

        if val_f1 > best_val_f1:

            best_val_f1 = val_f1
            best_epoch = epoch

            epochs_without_improvement = 0

            checkpoint = {
                "model_state_dict":
                    model.state_dict(),

                "input_dim":
                    X_train.shape[1],

                "feature_columns":
                    train_features,

                "hidden_dims":
                    model_cfg["hidden_dims"],

                "num_classes":
                    num_classes,

                "dropout":
                    float(
                        model_cfg["dropout"]
                    ),

                "class_order":
                    class_order,

                "best_epoch":
                    best_epoch,

                "best_val_macro_f1":
                    best_val_f1,
            }

            torch.save(
                checkpoint,
                checkpoint_path,
            )

            print(
                "  -> Saved new best checkpoint "
                f"(macro-F1={best_val_f1:.4f})"
            )

        else:
            epochs_without_improvement += 1

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if epochs_without_improvement >= patience:

            print()
            print(
                f"Early stopping at epoch {epoch}. "
                f"No validation improvement for "
                f"{patience} epochs."
            )

            break

    # --------------------------------------------------------
    # Training figure
    # --------------------------------------------------------

    save_training_curve(
        history=history,
        output_path=output_cfg[
            "training_figure_path"
        ],
    )

    print()
    print("Training completed.")

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best validation macro-F1: "
        f"{best_val_f1:.4f}"
    )

    print(
        f"Checkpoint: {checkpoint_path}"
    )

    print()
    print(
        "Official test split was NOT evaluated."
    )

    print(
        "Run --mode test only after model selection "
        "and tuning are finished."
    )


# ============================================================
# Final test
# ============================================================

def test(config_path: str) -> None:
    """
    Perform the final one-time evaluation on the official
    test split.

    Do not run during model selection or hyperparameter tuning.
    """

    config = load_config(
        config_path
    )

    seed = int(
        config["seed"]
    )

    seed_everything(seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    data_cfg = config["data"]
    output_cfg = config["output"]

    class_order = data_cfg.get(
        "class_order"
    )

    # --------------------------------------------------------
    # Load best checkpoint first
    # --------------------------------------------------------

    checkpoint_path = Path(
        output_cfg["checkpoint_path"]
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint was not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    # --------------------------------------------------------
    # Load official test split
    # --------------------------------------------------------

    (
        X_test,
        y_test,
        test_features,
    ) = load_split(
        path=data_cfg["test_path"],
        target_column=data_cfg["target_column"],
        class_order=class_order,
    )

    expected_features = checkpoint.get(
        "feature_columns"
    )

    if (
        expected_features is not None
        and test_features != expected_features
    ):
        raise ValueError(
            "Test feature columns/order do not match "
            "the features used during training.\n"
            f"Training: {expected_features}\n"
            f"Test:     {test_features}"
        )

    if X_test.shape[1] != int(
        checkpoint["input_dim"]
    ):
        raise ValueError(
            "Test feature dimension does not match "
            f"training: {X_test.shape[1]} != "
            f"{checkpoint['input_dim']}"
        )

    validate_labels(
        y_test,
        num_classes=int(
            checkpoint["num_classes"]
        ),
        split_name="Test split",
    )

    print(
        f"Test shape: {X_test.shape}"
    )

    # --------------------------------------------------------
    # Test DataLoader
    # --------------------------------------------------------

    test_loader = make_loader(
        X=X_test,
        y=y_test,
        batch_size=int(
            config["training"]["batch_size"]
        ),
        shuffle=False,
        seed=seed,
    )

    # --------------------------------------------------------
    # Restore best model
    # --------------------------------------------------------

    model = StellarMLP(
        input_dim=int(
            checkpoint["input_dim"]
        ),
        hidden_dims=checkpoint[
            "hidden_dims"
        ],
        num_classes=int(
            checkpoint["num_classes"]
        ),
        dropout=float(
            checkpoint["dropout"]
        ),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Final test evaluation
    # --------------------------------------------------------

    (
        test_loss,
        test_metrics,
        y_true,
        y_proba,
    ) = evaluate_model(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print()
    print("Final test results")
    print("------------------")

    print(
        f"Loss: {test_loss:.4f}"
    )

    for metric_name, value in test_metrics.items():

        if np.isscalar(value):
            print(
                f"{metric_name}: "
                f"{float(value):.4f}"
            )

        else:
            print(
                f"{metric_name}: {value}"
            )

    # --------------------------------------------------------
    # Required NPZ artifact
    # --------------------------------------------------------

    predictions_path = Path(
        output_cfg["predictions_path"]
    )

    predictions_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        predictions_path,
        y_true=y_true,
        y_proba=y_proba,
    )

    print()
    print(
        f"Predictions saved to: {predictions_path}"
    )

    print(
        f"y_true shape:  {y_true.shape}"
    )

    print(
        f"y_proba shape: {y_proba.shape}"
    )

    # --------------------------------------------------------
    # improvements.csv
    # --------------------------------------------------------

    if "macro_f1" not in test_metrics:
        raise KeyError(
            "evaluate() must return 'macro_f1'."
        )

    save_improvements_row(
        table_path=output_cfg[
            "improvements_table_path"
        ],
        val_f1=float(
            checkpoint[
                "best_val_macro_f1"
            ]
        ),
        test_f1=float(
            test_metrics["macro_f1"]
        ),
    )


# ============================================================
# CLI
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Train or evaluate StellarMLP."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/mlp_v1.yaml",
        help="Path to YAML configuration.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "train",
            "test",
        ],
        default="train",
        help=(
            "'train' uses train/validation only. "
            "'test' performs the final test evaluation."
        ),
    )

    args = parser.parse_args()

    if args.mode == "train":
        train(
            args.config
        )

    elif args.mode == "test":
        test(
            args.config
        )


if __name__ == "__main__":
    main()