"""
Unified plotting style for the project.

Everyone imports from this module instead of calling matplotlib directly,
so all figures in the report share the same look (fonts, colors, grid).

Public functions:
    plot_class_distribution   -> fig_1
    plot_confusion_matrix     -> fig_2
    plot_learning_curves      -> fig_3 (used by Person 4 for the MLP)
    plot_model_comparison     -> fig_4

Each function takes a `save_path` and writes a PNG there (dpi=150),
and also returns the matplotlib Figure in case the caller wants it
(e.g. for a notebook).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# Shared style: clean, minimal, no decorative elements.
# ---------------------------------------------------------------------------

_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]

_RC = {
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
}


def _apply_style() -> None:
    plt.rcParams.update(_RC)


def _save(fig: Figure, save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")


# ---------------------------------------------------------------------------
# fig_1: class distribution
# ---------------------------------------------------------------------------

def plot_class_distribution(
    y: Sequence[int],
    class_names: Sequence[str],
    save_path: str | Path,
    title: str = "Class Distribution",
) -> Figure:
    """Bar chart of class counts + percentage labels. `y` is the label
    array (e.g. the full dataset or the train split) using the class
    order announced by Person 2 (index -> class_names[index])."""
    _apply_style()
    y = np.asarray(y)
    counts = np.array([(y == i).sum() for i in range(len(class_names))])
    pct = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(class_names, counts, color=_PALETTE[: len(class_names)])
    for bar, c, p in zip(bars, counts, pct):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{c:,}\n({p:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.margins(y=0.15)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# fig_2: confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    save_path: str | Path,
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> Figure:
    """Confusion matrix heatmap for the best model (fed from that
    model's npz predictions: y_pred = y_proba.argmax(axis=1))."""
    from sklearn.metrics import confusion_matrix

    _apply_style()
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    cm_display = cm.astype(float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_display = np.divide(
            cm, row_sums, out=np.zeros_like(cm_display), where=row_sums != 0
        )

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm_display, cmap="Blues", vmin=0, vmax=cm_display.max() or 1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    ax.grid(False)

    thresh = cm_display.max() / 2 if cm_display.max() else 0.5
    for i in range(n):
        for j in range(n):
            label = f"{cm_display[i, j]:.2f}" if normalize else f"{cm[i, j]:d}"
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                color="white" if cm_display[i, j] > thresh else "black",
                fontsize=10,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# fig_3: training curves (owned by Person 4, kept here for a shared style)
# ---------------------------------------------------------------------------

def plot_learning_curves(
    train_loss: Sequence[float],
    val_loss: Sequence[float],
    save_path: str | Path,
    train_metric: Sequence[float] | None = None,
    val_metric: Sequence[float] | None = None,
    metric_name: str = "macro-F1",
    title: str = "MLP Training Curves",
) -> Figure:
    """Loss curves, plus optional metric curves as a second panel."""
    _apply_style()
    epochs = np.arange(1, len(train_loss) + 1)
    has_metric = train_metric is not None and val_metric is not None

    fig, axes = plt.subplots(1, 2 if has_metric else 1, figsize=(11 if has_metric else 6, 4))
    axes = np.atleast_1d(axes)

    ax = axes[0]
    ax.plot(epochs, train_loss, color=_PALETTE[0], label="train")
    ax.plot(epochs, val_loss, color=_PALETTE[1], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()

    if has_metric:
        ax2 = axes[1]
        ax2.plot(epochs, train_metric, color=_PALETTE[0], label="train")
        ax2.plot(epochs, val_metric, color=_PALETTE[1], label="val")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel(metric_name)
        ax2.set_title(metric_name)
        ax2.legend()

    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# fig_4: model comparison across all 5 models
# ---------------------------------------------------------------------------

def plot_model_comparison(
    model_names: Sequence[str],
    scores: Sequence[float],
    save_path: str | Path,
    metric_name: str = "macro-F1 (test)",
    title: str = "Model Comparison",
) -> Figure:
    """Horizontal bar chart ranking all models by a single metric
    (usually test macro-F1 from summary.csv), best on top."""
    _apply_style()
    order = np.argsort(scores)  # ascending -> best ends up at the top when plotted
    names_sorted = [model_names[i] for i in order]
    scores_sorted = [scores[i] for i in order]

    fig, ax = plt.subplots(figsize=(6.5, 0.6 * len(model_names) + 1.5))
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(model_names))]
    bars = ax.barh(names_sorted, scores_sorted, color=colors)
    for bar, s in zip(bars, scores_sorted):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {s:.3f}",
                va="center", fontsize=10)
    ax.set_xlabel(metric_name)
    ax.set_title(title)
    ax.set_xlim(0, max(scores_sorted) * 1.15 if scores_sorted else 1)
    fig.tight_layout()
    _save(fig, save_path)
    return fig
