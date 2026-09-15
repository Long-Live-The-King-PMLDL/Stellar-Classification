## Solution 1 — XGBoost + astronomical feature engineering

- **Source:** @emanuellcs, "Predicting Stellar Class | XGBoost", https://www.kaggle.com/code/emanuellcs/predicting-stellar-class-xgboost
- **Votes:** 35 | **Library:** xgboost
- **Original context:** built for Playground S6E6 (synthetic version of this
  dataset); SDSS17 used as a lookup table for target encoding.

**Method (kept as-is):**
XGBClassifier: 6000 trees, lr=0.015, max_depth=8, min_child_weight=5,
max_delta_step=1, gamma=0.2, reg_alpha=0.5, reg_lambda=2.5, subsample=0.75,
colsample_bytree=0.7, colsample_bylevel=0.8, tree_method=hist, ES=150;
sample_weight=class-balanced.

**Features (kept as-is):** 10 color indices (u_g ... r_z), magnitude stats
(mean/std/min/max/range), flux conversion 10^(-0.4m) with clip [0, 40] + flux
stats, sin/cos of alpha/delta, redshift x band interactions, color-per-redshift,
categorical bins spectral_type / galaxy_population / combo.

**Author's validation:** multi-seed (3) x 5-fold StratifiedKFold, balanced
accuracy, OOF blending + differential-evolution calibration of class weights.

**Leaks in the original:** none explicit (CV on train, no scaler). redshift-
based features are intentionally strong — discussed in report Limitations.

**Adaptations:**
1. Target encoding removed (lookup = same dataset would leak; no comp data).
2. Multi-seed CV -> single fit, early stopping on our val (protocol).
3. balanced accuracy -> unified evaluate(); DE class-weight calibration removed.
4. LabelEncoder dropped (author's mapping matched ours: GALAXY=0, QSO=1, STAR=2).

---

## Solution 2 — SVM (RBF) + Random Forest (one notebook, two classifiers)

- **Source:** @beyzanks, "Stellar Classification - 98.4% Acc 100% AUC", https://www.kaggle.com/code/beyzanks/stellar-classification-98-4-acc-100-auc
- **Votes:** 149 | **Libraries:** sklearn, imblearn (SMOTE), yellowbrick
- **Original pipeline:** LOF outlier removal -> correlation heatmap -> feature
  selection (drop obj_ID, alpha, delta, run_ID, rerun_ID, cam_col, field_ID,
  fiber_ID) -> SMOTE on the full dataset -> StandardScaler on the full dataset
  -> 33% holdout -> SVC(rbf, C=1) and RandomForest(default) -> accuracy,
  confusion matrix, ROC, class-prediction-error visuals.

**Method (kept as-is):** SVC(kernel='rbf', C=1); RandomForestClassifier()
defaults; author's feature selection (bands + redshift kept; alpha/delta and
IDs dropped).

**Author's stated metrics:** SVM accuracy 0.971, RF accuracy 0.984
(inflated by the leaks below; not directly comparable — different split).

**Leaks in the original (all applied BEFORE the split):**
1. LocalOutlierFactor row removal on the full dataset.
2. SMOTE oversampling on the full dataset (synthetic rows contain test info).
3. StandardScaler fit on the full dataset.
4. spec_obj_ID kept as a feature — hard leak (same ID implies same class per
   dataset docs); likely the main driver of the author's high scores.

**Class encoding in the original:** GALAXY=0, STAR=1, QSO=2 (non-alphabetical,
differs from ours; irrelevant to reproduction since we use pre-encoded labels).

**Adaptations:**
1. LOF removed (our -9999 artifacts already cleaned in prepare.py).
2. SMOTE -> class_weight='balanced' (leak-free imbalance handling).
3. Scaler fit on train only.
4. spec_obj_ID not reproducible (ID columns absent from our processed data).
5. SVM trained on a stratified 20k subsample of train (RBF SVC ~O(n^2));
   probability=True added (internal 5-fold Platt calibration) for our
   evaluate().
6. 33% holdout -> our train/val/test protocol, unified evaluate().

## Solution 3 — Keras feed-forward NN

- **Source:** @prtkpiyush, "neural_network_solution(external dataset)(DL)", https://www.kaggle.com/code/prtkpiyush/neural-network-solution-external-dataset-dl
- **Votes:** 9 | **Library:** tensorflow/keras
- **Original context:** S6E6; trained on the union of competition + SDSS17
  data (is_orig flag); 5-fold CV, ES on fold-validation; balanced accuracy.

**Method (kept as-is):**
Sequential: Dense(256, relu) - BN - Dropout(0.3) - Dense(128, relu) - BN -
Dropout(0.3) - Dense(64, relu) - Dropout(0.2) - Dense(3, softmax).
Adam(1e-3), sparse categorical CE, batch 2048, epochs 200,
EarlyStopping(patience=15, restore_best_weights),
ReduceLROnPlateau(0.5, patience=6, min_lr=1e-6), class_weight balanced.
Full FE: color indices, magnitude stats, log1p(flux), spectral slope,
curvatures (blue/mag/red), redshift interactions, cyclical coordinates,
spectral_ord bins. NaN->0 after scaling.

**Leaks in the original:** none found (scaler fit per fold — clean).

**Quirk fixed:** in the notebook, spectral_type bins were computed only for
the external frame, so spectral_ord was a constant -1 for train/test rows and
the categorical dummies were effectively unused. Our version computes the
bins for every row.

**Adaptations:**
1. Trained on our train split only (no comp+orig union, no is_orig flag).
2. 5-fold CV -> single fit with ES on our val (protocol).
3. balanced accuracy -> unified evaluate().
4. LabelEncoder dropped (alphabetical = our mapping, verified).
