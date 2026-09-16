# Section 2. Baselines — Reproducing Public Solutions

## 2.1. Stage Objectives

Per the project requirements (requirement 1), the project must test several
existing solutions and compute their metrics. As "existing solutions" we used
public Kaggle notebooks published for this dataset (Stellar Classification
Dataset — SDSS17) and its synthetic version (Playground S6E6).

Stage objectives:
1. Select public solutions representative of different approaches
   (boosting / classical models / neural network).
2. Reproduce each solution under the project's unified protocol (a single
   split, a shared metrics function, one-time use of the test set).
3. Identify and fix methodological issues in the originals (data leakage).
4. Obtain honest metrics for comparison with the proposed model.

## 2.2. Unified Metrics Function

The first artifact of this role was the project-wide `evaluate()` function
(`src/evaluation/metrics.py`), used by every model in the project, including
the proposed MLP:

- input: `y_true (n,)`, `y_proba (n, 3)`; predictions = `argmax` over probabilities;
- metrics: **macro-F1 (primary)**, accuracy, per-class precision/recall,
  ROC-AUC (OvR, macro);
- class order fixed by the data owner: GALAXY=0, QSO=1, STAR=2;
- the function is covered by tests: a perfect classifier (all metrics = 1.0),
  a hand-computed reference case, and the case of a missing class in the
  sample (ROC-AUC → NaN without raising).

A single metrics function eliminates discrepancies caused by different metric
implementations and is a prerequisite for correct comparison of all models in
Section 5.

## 2.3. Selecting Public Solutions

Selection criteria:
- public availability and reproducibility (visible cell outputs);
- diversity of approaches for a meaningful comparison;
- training on dataset features (not ensembling other models' predictions);
- readable code and available hyperparameters.

Considered and rejected:
- the @cdeotte stacking ensemble (S6E6): a meta-logistic regression on top of
  ~19 OOF-prediction files of other models — the files are unavailable and the
  dataset is synthetic, so reproduction is impossible in principle. The
  multinomial-LR training recipe inside the stack was noted but superseded by
  the standalone NN solution (S3).

The final portfolio — three sources, four models:

| Solution | Method | Source |
|---|---|---|
| S1 | XGBoost + astronomical feature engineering | @emanuellcs, S6E6 |
| S2a | SVM (RBF, C=1) | @beyzanks, SDSS17 |
| S2b | Random Forest (default) | same notebook |
| S3 | Keras feed-forward NN (256/128/64) | @prtkpiyush, S6E6 |

## 2.4. Reproduction Protocol

Each solution was ported into the project code with a mandatory set of
replacements (uniform across all baselines):

1. Data — our `train/val/test` from `data/processed/` (stratified 80/10/10
   split, seed=42); the authors' own splits were removed;
2. Class encoding — the pre-encoded `label` column (all authors' encodings
   matched ours; S2b's non-alphabetical GALAXY=0/STAR=1/QSO=2 encoding is
   documented as a deviation and does not affect the learned decision function);
3. Metrics — computed only via `evaluate()`; the authors' reported metrics are
   quoted separately and are not directly comparable (different splits and
   metrics);
4. seed=42; the test set was used once per solution;
5. Attribution — a link to the source in the header of each solution file.

## 2.5. Solution 1 — XGBoost with Feature Engineering

**Original:** an S6E6 solution: XGBoost (6,000 trees, lr=0.015, depth=8,
subsample=0.75, colsample 0.7/0.8, etc.), sample_weight='balanced', multi-seed
(3) × 5-fold CV, OOF blending, and class-weight calibration via differential
evolution; SDSS17 was used as a lookup table for target encoding. Feature
engineering: 10 color indices, magnitude statistics, flux conversion
(10^(−0.4m), clip [0, 40]) with statistics, sin/cos of coordinates, redshift
interactions, and categorical astronomical bins (spectral_type,
galaxy_population, combo).

**Adaptations:** target encoding removed (a lookup built from the same dataset
would leak; the competition data is unavailable to us — the bins are kept as
categorical features); multi-seed CV → a single fit with early stopping on
val; balanced accuracy → macro-F1; the class-weight calibration removed (it
optimized the author's metric, not macro-F1).

**Leaks in the original:** none explicit (CV on train, no scaler).

**Result:** early stopping halted training at iteration 1,379 of 6,000
(val) / 1,509 (test run).

## 2.6. Solution 2 — SVM (RBF) and Random Forest

**Original:** an SDSS17 notebook with the sequence: LocalOutlierFactor →
correlations → feature selection (dropping alpha, delta, all ID columns and
cam_col; keeping u, g, r, i, z, redshift) → SMOTE → StandardScaler → 33%
holdout → SVC(rbf, C=1) and RandomForest (default) with visualizations
(confusion matrix, ROC, class prediction error).

**Leaks found (all applied BEFORE the split):**
1. LOF outlier filtering on the full dataset;
2. SMOTE oversampling on the full dataset (synthetic rows contain test
   information);
3. StandardScaler.fit on the full dataset;
4. `spec_obj_ID` kept as a feature — a hard leak: per the dataset
   documentation, identical IDs imply identical classes; likely the main
   source of the author's inflated metrics.

**Adaptations:** all four leaks removed — LOF dropped (the −9999 artifacts are
already cleaned during data preparation), SMOTE replaced with
`class_weight='balanced'`, the scaler fit on train only, and the ID feature
cannot be reproduced at all (ID columns are absent from our data). The SVM was
trained on a stratified subsample of 20,000 rows (RBF SVC complexity is
~O(n²)) with `probability=True` (internal 5-fold Platt calibration) to support
`evaluate()`. The hyperparameters of both models and the author's feature
selection are preserved.

## 2.7. Solution 3 — Keras Neural Network

**Original:** S6E6, trained on the union of competition and SDSS17 data (the
is_orig flag), 5-fold CV, balanced accuracy. Architecture:
Dense(256)-BN-Dropout(0.3) → Dense(128)-BN-Dropout(0.3) →
Dense(64)-Dropout(0.2) → softmax(3); Adam(1e-3), sparse categorical CE,
batch 2048, up to 200 epochs, EarlyStopping (patience=15,
restore_best_weights), ReduceLROnPlateau (factor 0.5, patience 6). Feature
engineering: color indices, magnitude statistics, log1p on fluxes, spectral
slope and curvatures, redshift interactions, cyclical coordinates, ordinal
spectral bins.

**Adaptations:** training on our train split only (no union); CV → a single
fit with ES on val; metric → macro-F1. A quirk of the original was fixed: the
spectral bins were computed only for the external frame, so `spectral_ord` was
a constant −1 for training rows and the categorical features were effectively
unused — in our version the bins are computed for every row.

**Leaks in the original:** none found (the scaler was fit per fold — correct).

## 2.8. Results

Table 1. Metrics of the reproduced solutions (unified protocol; macro-F1 is
the primary metric).

| Solution | val macro-F1 | test macro-F1 | test accuracy | test ROC-AUC | Training time |
|---|---|---|---|---|---|
| S1 XGBoost + FE | 0.97647 | 0.97467 | 0.97780 | 0.99675 | 50.7 s |
| S2a SVM (RBF) | 0.96074 | 0.95650 | 0.96130 | 0.99040 | 18.5 s (20k subsample) |
| S2b Random Forest | 0.97486 | 0.97569 | 0.97890 | 0.99593 | 2.3 s |
| S3 Keras NN | 0.96805 | 0.96988 | 0.97320 | 0.99515 | 80.6 s |

Authors' reported metrics are omitted: all original notebooks used different splits and/or metrics, making direct comparison invalid.

## 2.9. Analysis

1. **Task saturation.** All four solutions fall within a ~2 pp range
   (0.956–0.976 test macro-F1): the redshift feature alone separates the
   classes almost perfectly, so solution complexity brings no advantage.
2. **Default RF ≥ tuned XGBoost** (0.97569 vs 0.97467): a default forest on
   six author-selected features matches boosting on an extended set of ~40
   engineered features — further evidence of saturation.
3. **The neural network (0.96988) trails the tree ensembles** but is well
   above the SVM — an expected outcome for low-dimensional tabular data.
4. **Protocol stability.** All val→test deviations are ≤ 0.005 (XGBoost
   −0.0018; SVM −0.0042; RF +0.0008; NN +0.0018) — no overfitting; the
   positive deltas of RF and NN lie within sampling noise.
5. **Leaks are a typical problem of public solutions:** in 2 of 3 sources,
   preprocessing was performed before the split (SMOTE, scaler, LOF), and one
   used an ID feature that nearly determines the class. Our numbers are honest
   and in some cases lower than the authors' reported ones — the expected
   consequence of a correct protocol.

## 2.10. Contributions to Project Infrastructure

Beyond the baselines, this role delivered shared components used by every
model in the project:
- `src/evaluation/metrics.py` — the unified metrics function with tests;
- `src/utils/paths.py` — centralized paths (works from any working directory);
- `src/data/loading.py` — a shared loader with the feature/target contract;
- the `solution_*.py` template (a uniform structure: attribution → FE → model
  → run → npz), adopted by the other roles for the proposed model;
- the prediction format `reports/preds/<model>_test.npz` (y_true + y_proba)
  used for error analysis and confusion matrices in Section 5.

## 2.11. Limitations

- The SVM was trained on a subsample of 20,000 of ~80,000 rows (RBF SVC
  complexity), so its metrics are conservative; full training could yield
  ~+0.005.
- Two sources are solutions for the synthetic version of the dataset (S6E6);
  the port to SDSS17 preserves the method but is not fully identical to the
  originals.
- `redshift` is retained among the features (as in all sources): this keeps
  the numbers comparable with the community's, but makes the task close to
  trivial — see the Limitations discussion (Section 7).
