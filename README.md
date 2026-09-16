# Stellar Classification (Stars, Galaxies and Quasars)

Three-class classification of SDSS17 astronomical objects (GALAXY / QSO / STAR)
from photometric and positional features. Baselines (XGBoost, SVM, Random
Forest, Keras NN) are reproduced from public Kaggle notebooks under a unified,
leak-free protocol; the proposed model is a PyTorch MLP (v1) with a
hyperparameter-search variant (v2). Full write-up: `reports/project.pdf`.

## 1. Setup

Python 3.10+. CPU-only, no GPU required.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Data

The raw CSV is not included in this repository (Kaggle terms). Download it
manually:

1. <https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17/data>
   (requires a free Kaggle account).
2. Save it as `data/raw/star_classification.csv`.

## 3. Run the full pipeline

All commands run from the repository root.

```bash
# Clean + stratified 80/10/10 split (seed=42) + StandardScaler
python src/data/prepare.py

# Baselines: validation, then one final test run each
python -m src.baselines.solution_1_xgboost --split val
python -m src.baselines.solution_1_xgboost --split test
python -m src.baselines.solution_2_svm_randomforest --split val
python -m src.baselines.solution_2_svm_randomforest --split test
python -m src.baselines.solution_3_NN --split val
python -m src.baselines.solution_3_NN --split test

# Proposed model: MLP v1
python -m src.models.train --config configs/mlp_v1.yaml --mode train
python -m src.models.train --config configs/mlp_v1.yaml --mode test

# Model improvement: hyperparameter search -> MLP v2
python -m src.models.tune --smoke      # optional 2-epoch dry run
python -m src.models.tune              # full search (configs/mlp_v2_search.yaml)
python -m src.models.finalize_v2 --config configs/mlp_v2.yaml

# Aggregate tables and figures
python -m src.analysis.build_summary \
    --baselines reports/tables/baselines.csv \
    --improvements reports/tables/improvements.csv \
    --out reports/tables/summary.csv
python src/analysis/generate_figures.py

# Optional: error analysis for the best model
python src/analysis/error_analysis.py \
    --npz reports/preds/randomforest_test.npz \
    --class-names GALAXY QSO STAR \
    --test-csv data/processed/test.csv \
    --n-examples 3
```

## 4. Expected runtime (CPU reference environment)

| Step | Time |
|---|---|
| `prepare.py` | a few seconds |
| Random Forest | ~2 s |
| SVM (20k-row subsample) | ~19 s |
| XGBoost | ~51 s |
| Keras NN | ~81 s |
| MLP v1 training | a few minutes |
| MLP v2 search (6 trials) | ~13.5 min total |

Seeds (`random`/`numpy`/`torch` = 42, split seed = 42) are fixed throughout
via `src/utils/seed.py`; minor (<0.001) macro-F1 differences across CPU
environments are possible due to BLAS/threading non-determinism and are not
a sign of a broken pipeline.

## 5. Project layout

See `docs/` for per-baseline adaptation notes (`baseline_sources.md`) and
per-role write-ups; see `reports/project.pdf` for the full 7-section report.
