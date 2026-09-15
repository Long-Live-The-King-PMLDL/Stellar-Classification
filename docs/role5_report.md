# MLP v2: Hyperparameter Search and Result

## Method

MLP v2 reused the MLP v1 architecture and training implementation without
changing its source code. Model selection used only macro-F1 on the fixed
validation split. The official test split was not accessed during tuning.

The search evaluated six configurations around the MLP v1 reference. It varied
hidden-layer width, dropout, learning rate, weight decay, and the maximum number
of epochs. Every trial used AdamW, batch size 256, early stopping, and seed 42.
The complete machine-readable results are stored in
`reports/tables/mlp_v2_tuning.csv`.

## Validation results

| Trial | Hidden layers | Dropout | Learning rate | Weight decay | Best epoch | Validation macro-F1 |
|---|---|---:|---:|---:|---:|---:|
| MLP v1 reference | 128, 64 | 0.20 | 0.0010 | 0.00010 | 49 | **0.970681** |
| Base, extended training | 128, 64 | 0.20 | 0.0010 | 0.00010 | 32 | 0.969456 |
| Lower dropout | 128, 64 | 0.10 | 0.0010 | 0.00010 | 46 | 0.970367 |
| Wider | 256, 128 | 0.20 | 0.0010 | 0.00010 | 32 | 0.969869 |
| Wider, lower dropout | 256, 128 | 0.10 | 0.0010 | 0.00010 | 29 | 0.969785 |
| Lower learning rate | 128, 64 | 0.10 | 0.0005 | 0.00010 | 49 | 0.969795 |
| Lighter regularization | 128, 64 | 0.05 | 0.0010 | 0.00001 | 29 | **0.970580** |

The best modified configuration retained hidden layers `[128, 64]`, reduced
dropout from 0.20 to 0.05, and reduced weight decay from `1e-4` to `1e-5`.
Its validation macro-F1 was 0.970580, a difference of -0.000101 from MLP v1.

## Interpretation

The search did not produce a genuine validation improvement. The difference
between MLP v1 and the best modified model is negligible relative to the score
variation observed when retraining the reference configuration and is too small
to support a meaningful performance claim. Increasing model width did not help,
while lighter regularization came closest to the reference score. These results
suggest that the current MLP and feature set are close to saturation and that
further gains would more likely require feature engineering or a different
tabular-model family than a small change in MLP capacity.

The stored MLP v1 checkpoint reports 0.970681, while retraining its reference
architecture in the current environment reached 0.969456. Because the project
dependencies are currently unpinned, small environment-dependent differences
in the training trace are plausible. The model comparison therefore retains the
stored v1 result as the official reference instead of replacing it with the
lower rerun score.

This is still a valid model modification experiment: the search protocol was
fixed, all decisions used validation macro-F1, and the negative result is
reported rather than selecting a model using the test set.

## Final test results

After model selection was frozen, MLP v2 was evaluated on the official test
split exactly once using the shared `evaluate()` function.

| Version | Validation macro-F1 | Test macro-F1 | Test delta vs v1 |
|---|---:|---:|---:|
| MLP v1 | 0.970681 | **0.970295** | 0.000000 |
| MLP v2 | 0.970580 | 0.969771 | -0.000523 |

MLP v2 achieved test accuracy 0.9734 and macro ROC-AUC OVR 0.9947. The test
result confirms the validation conclusion: lighter regularization did not
produce a meaningful improvement over MLP v1. The small negative delta is
consistent with a task that is already close to saturation for this MLP and
feature representation.

## Reproduction

```powershell
python -m src.models.tune --smoke
python -m src.models.tune
$env:MPLBACKEND = "Agg"
python -m src.models.train --config configs/mlp_v2.yaml --mode train
```

The following command was used exactly once after final testing was authorized
and the completed MLP v1 result became available:

```powershell
python -m src.models.finalize_v2 --config configs/mlp_v2.yaml
```

It produced `reports/preds/mlp_v2_test.npz` and added the `mlp_v2` row to
`reports/tables/improvements.csv`. The finalizer refuses to run again while the
prediction artifact exists, protecting the one-time test protocol.
