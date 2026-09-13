"""
Centralized project paths.
All file I/O must go through these constants so that every script
works regardless of the current working directory.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"

REPORTS_DIR = PROJECT_ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
PREDS_DIR = REPORTS_DIR / "preds"
FIGURES_DIR = REPORTS_DIR / "figures"


def ensure_dirs() -> None:
    """Create output directories if missing (safe to call repeatedly)."""
    for d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR,
              TABLES_DIR, PREDS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)