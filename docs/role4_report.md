# Role 4 Report — ML Engineer (MLP v1)

## 1. Role

My role in the project was **ML Engineer — MLP v1 (Role 4)**.

The main responsibilities were to implement the custom PyTorch MLP, make the training pipeline configurable through YAML, integrate the shared project utilities, train and validate MLP v1, save the best checkpoint, generate the training curve, run the final one-time test evaluation, and produce the required prediction and result artifacts.

## 2. Main Files

```text
configs/mlp_v1.yaml
src/models/mlp.py
src/models/train.py
models/mlp_v1.pt
reports/figures/fig_3_mlp_training_curves.png
reports/preds/mlp_v1_test.npz
reports/tables/improvements.csv
docs/role4_report.md
```

## 3. Input Data

The processed MLP input contains 8 numerical features:

```text
alpha
delta
u
g
r
i
z
redshift
```

Target column:

```text
label
```

The target is encoded as three integer classes: `0`, `1`, and `2`.

| Split | Samples |
|---|---:|
| Train | 79,999 |
| Validation | 10,000 |
| Test | 10,000 |

The processed CSV files were already standardized by the shared preprocessing pipeline, so no additional scaling was performed inside the MLP training code.

## 4. MLP Architecture

The model is implemented in PyTorch as `StellarMLP`.

```text
Input: 8 features
      |
      v
Linear(8 -> 128)
BatchNorm1d(128)
ReLU
Dropout(0.2)
      |
      v
Linear(128 -> 64)
BatchNorm1d(64)
ReLU
Dropout(0.2)
      |
      v
Linear(64 -> 3)
      |
      v
3 output logits
```

| Layer | Configuration |
|---|---|
| Input | 8 numerical features |
| Hidden layer 1 | Linear(8, 128) |
| Normalization | BatchNorm1d(128) |
| Activation | ReLU |
| Regularization | Dropout(0.2) |
| Hidden layer 2 | Linear(128, 64) |
| Normalization | BatchNorm1d(64) |
| Activation | ReLU |
| Regularization | Dropout(0.2) |
| Output | Linear(64, 3) |

The network returns raw logits. Softmax is applied only during evaluation, while training uses `CrossEntropyLoss`.

## 5. Training Configuration

The model is controlled by `configs/mlp_v1.yaml`.

| Hyperparameter | Value |
|---|---:|
| Hidden dimensions | [128, 64] |
| Dropout | 0.2 |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Batch size | 256 |
| Maximum epochs | 50 |
| Early stopping patience | 7 |
| Random seed | 42 |

The YAML-driven design allows MLP v2 to reuse the same training implementation and modify only configuration values.

## 6. Training Procedure

The training pipeline:

1. Loads the YAML configuration.
2. Applies the shared seed utility.
3. Loads the official train and validation splits.
4. Builds `StellarMLP`.
5. Trains with `CrossEntropyLoss`.
6. Optimizes using AdamW.
7. Evaluates each epoch using the shared `evaluate(y_true, y_proba)` function.
8. Uses validation macro-F1 for checkpoint selection.
9. Saves the best model checkpoint.
10. Generates the MLP training curve.
11. Keeps the test split untouched until final evaluation.

Training command:

```powershell
python -m src.models.train --config configs/mlp_v1.yaml --mode train
```

## 7. Validation Results

The best checkpoint was obtained at:

```text
Best epoch: 49
Best validation macro-F1: 0.9707
```

Checkpoint:

```text
models/mlp_v1.pt
```

Training figure:

```text
reports/figures/fig_3_mlp_training_curves.png
```

## 8. Baseline Sanity Check

| Model | Validation macro-F1 |
|---|---:|
| SVM (RBF) | 0.96074 |
| **MLP v1** | **0.97070** |
| Random Forest | 0.97486 |

The MLP v1 outperformed SVM by `+0.00996` macro-F1 and was only `0.00416` below Random Forest.

This confirmed that MLP v1 was competitive with the classical baselines and that the implementation was behaving correctly.

## 9. Final Test Evaluation

The official test split was evaluated once after model selection was completed.

Command:

```powershell
python -m src.models.train --config configs/mlp_v1.yaml --mode test
```

Final results:

| Metric | Value |
|---|---:|
| Test loss | 0.0979 |
| **Macro-F1** | **0.9703** |
| Accuracy | 0.9739 |
| ROC-AUC OVR | 0.9947 |

Per-class precision:

```text
[0.97684565375838926,
 0.9635671560630777,
 0.9745570195365743]
```

Per-class recall:

```text
[0.9793103448275862,
 0.9345991561181435,
 0.9935155164427976]
```

Validation/test comparison:

```text
Validation macro-F1 = 0.9707
Test macro-F1       = 0.9703
Difference          = -0.0004
```

The validation and test results are very close, showing that the selected checkpoint generalized consistently to unseen data.

## 10. Prediction Artifact

The required prediction file was generated:

```text
reports/preds/mlp_v1_test.npz
```

Required arrays:

```text
y_true  -> (10000,)
y_proba -> (10000, 3)
```

## 11. Improvements Table

The MLP v1 result was written to:

```text
reports/tables/improvements.csv
```

Required values for the MLP v1 row:

```text
version = mlp_v1
val_f1  = 0.9707
test_f1 = 0.9703
delta   = 0
```

`delta = 0` because MLP v1 is the base model for comparison with MLP v2.

## 12. Final Status

- [x] Implemented `StellarMLP`
- [x] Implemented YAML-driven training
- [x] Integrated shared `evaluate()`
- [x] Integrated shared `seed_everything()`
- [x] Integrated the official processed dataset
- [x] Completed the synthetic smoke test
- [x] Trained MLP v1 on the official training split
- [x] Selected the best checkpoint with validation macro-F1
- [x] Achieved validation macro-F1 = **0.9707**
- [x] Saved `models/mlp_v1.pt`
- [x] Generated `fig_3_mlp_training_curves.png`
- [x] Compared against validation baselines
- [x] Performed the final one-time test evaluation
- [x] Achieved test macro-F1 = **0.9703**
- [x] Generated `reports/preds/mlp_v1_test.npz`
- [x] Updated `reports/tables/improvements.csv`
- [x] Completed the Role 4 report

## 13. Conclusion

The MLP v1 implementation was completed successfully.

The model achieved a validation macro-F1 of **0.9707** and a final test macro-F1 of **0.9703**, with test accuracy of **0.9739** and ROC-AUC OVR of **0.9947**.

The small difference between validation and test macro-F1 indicates stable generalization. The MLP outperformed the SVM baseline and remained close to Random Forest. The implementation is configuration-driven and reusable by the MLP v2 role.


