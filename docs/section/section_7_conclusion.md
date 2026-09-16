<!--
Owner: Person 1. Final version — all six models have test results as of
this version (verified against role4_report.md / role5_report.md and
computed independently from the npz files).
-->

## 7. Conclusion

### 7.1 Findings

1. **RandomForest is the best-performing model** of the six evaluated,
   with test macro-F1 0.97569 — narrowly ahead of XGBoost (0.97467,
   −0.001) despite using only 6 hand-picked features against XGBoost's
   ~40 engineered ones. Neither MLP version nor the Keras NN baseline
   beat either tree ensemble.
2. **The v1→v2 hyperparameter search did not improve the MLP** — it
   produced a small regression instead (test macro-F1 0.96977 vs
   0.97029 for v1, Δ = −0.00052; validation Δ = −0.00010 during model
   selection). Per the project's contingency plan, this is a valid,
   reportable outcome: the search covered 6 configurations (width,
   dropout, learning rate, weight decay) and found nothing that beat
   the v1 reference — evidence the MLP is close to saturating on this
   feature set, not a failed search.
3. **Gradient boosting beats neural networks on this task.** Both tree
   ensembles (XGBoost 0.97467, RandomForest 0.97569) outperform both
   neural-network solutions (MLP v1 0.97029, Keras NN 0.96988, MLP v2
   0.96977) by a consistent margin, and the two MLP versions land
   essentially on top of the Keras NN baseline rather than closing the
   gap to the trees — consistent with the well-known pattern that
   boosted trees have an edge over neural networks on tabular data at
   this scale, and confirms the pattern already visible among the
   baselines alone extends to the project's own model.

### 7.2 Limitations

- **Feature leakage risk.** `plate` and `MJD` were dropped along with the
  other ID-like columns during preprocessing (Section 1), but `redshift`
  was kept as one of the 8 modeling features. Redshift is strongly
  diagnostic of class (it is part of how SDSS itself assigns labels),
  so reported scores may be optimistic relative to a model that only
  sees photometric features. Re-running without it would give a more
  conservative estimate of generalization.
  (Confirmed empirically: the error-analysis case studies — GALAXY
  rows misclassified as QSO — were all galaxies with atypically high
  redshift, i.e. redshift-driven confusion is not hypothetical.)
- **Apparent performance ceiling.** All six models land within a
  0.0192 band (0.95650–0.97569 test macro-F1), and five of the six
  (everything except SVM) sit within 0.006 of each other — despite
  wildly different model complexity, from a default RandomForest on 6
  features to a 6000-tree XGBoost on ~40 engineered features to two
  MLP architectures. This is strong evidence the task is close to
  saturating on this feature set; further tuning is unlikely to move
  the needle much without new features or more data. The MLP tuning
  result (Finding 2) is itself direct evidence of this ceiling for the
  neural-network family specifically.
- **Model family trade-off.** Confirmed across all six models, not
  just the baselines: gradient boosting (~0.975) clearly ahead of both
  neural-network solutions (~0.970) — this is the actual point of
  Requirement 2/3 in the ТЗ, and the project's own MLP follows the
  same family pattern as the Keras NN baseline rather than closing the
  gap to the trees.

### 7.3 Scope not covered

Feature engineering (e.g. photometric color indices), leakage
ablations, and a deeper EDA were deliberately left out of scope for
this iteration to fit the project timeline; they are natural next
steps if the project continues.


