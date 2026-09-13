# ML Engineer (MLP v1) — Complete Role Report

## 1. Role and Responsibility

My role in the project is **ML Engineer — MLP v1 (Role 4)**.

The responsibility of this role is to implement and train the first custom neural-network solution for the stellar classification task. The implementation must be reusable by the MLP v2 engineer through YAML configuration, use the shared project evaluation function, follow the common preprocessing pipeline, save the best model checkpoint, generate the MLP training curves, and provide the MLP v1 result for the final comparison.

The main owned files and artifacts are:

```text
configs/mlp_v1.yaml
src/models/mlp.py
src/models/train.py
models/mlp_v1.pt
reports/figures/fig_3_mlp_training_curves.png
reports/preds/mlp_v1_test.npz        # final test stage
reports/tables/improvements.csv      # MLP v1 row after final test
docs/sections/section_3_model.md
```

---

## 2. Integration with Other Team Members

The MLP v1 pipeline depends on outputs from three other roles:

- **Role 2 — Data Engineer**
  - `data/processed/train.csv`
  - `data/processed/val.csv`
  - `data/processed/test.csv`
  - preprocessing and train-only scaling

- **Role 3 — Baseline Engineer**
  - `src/evaluation/metrics.py`
  - shared `evaluate(y_true, y_proba)` function
  - validation baseline results for sanity checking

- **Role 6 — MLOps**
  - `src/utils/seed.py`
  - shared `seed_everything(42)` function

The MLP implementation does not duplicate these responsibilities. It directly consumes the processed data and uses the shared metric and reproducibility utilities.

---

## 3. Input Data Used by MLP v1

The processed training data contains **8 numerical features**:

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

The target column is:

```text
label
```

The target is already integer encoded using three classes:

```text
0
1
2
```

The processed splits used for training were:

| Split | Samples |
|---|---:|
| Train | 79,999 |
| Validation | 10,000 |
| Test | Reserved for final one-time evaluation |

The processed CSV files are already standardized by the shared preprocessing pipeline, so the MLP training code does not fit or apply another scaler. This avoids accidental double scaling and keeps preprocessing centralized in the Data Engineer's pipeline.

---

## 4. MLP v1 Architecture

The proposed model is implemented in PyTorch as `StellarMLP`.

The architecture is:

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
3 class logits
```

Architecture table:

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

The model returns **raw logits**. Softmax is not included in the network because training uses `CrossEntropyLoss`, which expects logits directly. Softmax is applied only during evaluation to produce class probabilities.

---

## 5. Configuration Interface

The MLP v1 pipeline is fully controlled through:

```text
configs/mlp_v1.yaml
```

The base configuration used for MLP v1 is:

```yaml
seed: 42

model:
  input_dim: null
  hidden_dims:
    - 128
    - 64
  num_classes: 3
  dropout: 0.2

training:
  epochs: 50
  batch_size: 256
  learning_rate: 0.001
  weight_decay: 0.0001
  patience: 7

optimizer:
  name: adamw

data:
  train_path: data/processed/train.csv
  val_path: data/processed/val.csv
  test_path: data/processed/test.csv
  target_column: label
  class_order: null

output:
  checkpoint_path: models/mlp_v1.pt
  predictions_path: reports/preds/mlp_v1_test.npz
  training_figure_path: reports/figures/fig_3_mlp_training_curves.png
  improvements_table_path: reports/tables/improvements.csv
```

The important design decision is that MLP v2 can reuse the same training implementation and change only YAML parameters such as:

```text
hidden_dims
dropout
learning_rate
weight_decay
batch_size
optimizer
```

This satisfies the team requirement that the improved model should be built on top of the same training code rather than by modifying the MLP v1 implementation.

---

## 6. Training Pipeline

The complete training pipeline is implemented in:

```text
src/models/train.py
```

The pipeline performs the following steps:

1. Load the YAML configuration.
2. Set the shared random seed using `seed_everything(42)`.
3. Load the official train and validation CSV files.
4. Validate target and feature columns.
5. Convert data to PyTorch tensors.
6. Create deterministic `DataLoader` objects.
7. Build `StellarMLP`.
8. Train with `CrossEntropyLoss`.
9. Optimize with AdamW.
10. Evaluate on validation data after each epoch.
11. Calculate metrics through the shared `evaluate()` function.
12. Use **validation macro-F1** as the model-selection metric.
13. Save a checkpoint whenever validation macro-F1 improves.
14. Apply early stopping according to the configured patience.
15. Save the MLP training curve figure.
16. Keep the official test split untouched during model selection.

The main training command is:

```powershell
python -m src.models.train --config configs/mlp_v1.yaml --mode train
```

The final test path is deliberately separated:

```powershell
python -m src.models.train --config configs/mlp_v1.yaml --mode test
```

The second command must only be used after model selection and the MLP v2 stage are finished.

---

## 7. Smoke Test Before Real Training

Before using the official data, the complete pipeline was tested on a synthetic three-class dataset.

The smoke test verified that the following components work together correctly:

```text
YAML configuration
-> DataLoader
-> StellarMLP
-> CrossEntropyLoss
-> AdamW
-> validation evaluation
-> macro-F1
-> early stopping
-> model checkpoint
-> training figure
```

The synthetic test completed successfully, confirming that the pipeline was technically functional before integration with the official project data.

---

## 8. Real Training Results

The MLP v1 model was trained on the official processed training split.

Final training summary:

| Item | Result |
|---|---:|
| Training samples | 79,999 |
| Validation samples | 10,000 |
| Input features | 8 |
| Maximum epochs | 50 |
| Best epoch | **49** |
| Best validation macro-F1 | **0.9707** |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Batch size | 256 |
| Dropout | 0.2 |

The validation macro-F1 improved throughout training from approximately **0.9440** after the first epoch to a best value of **0.9707** at epoch 49.

The best checkpoint was saved to:

```text
models/mlp_v1.pt
```

The training curve was saved to:

```text
reports/figures/fig_3_mlp_training_curves.png
```

The final epoch itself was not automatically selected. The checkpoint corresponds to the epoch with the highest validation macro-F1, which protects the final model from later validation fluctuations.

---

## 9. Baseline Sanity Check

MLP v1 was compared against available validation results from the baseline implementation.

| Model | Validation macro-F1 |
|---|---:|
| SVM (RBF) | 0.96074 |
| **MLP v1** | **0.97070** |
| Random Forest | **0.97486** |

Differences relative to MLP v1:

```text
MLP v1 - SVM           = +0.00996
MLP v1 - RandomForest  = -0.00416
```

The MLP therefore performs clearly better than the SVM baseline and remains very close to Random Forest.

This is a successful sanity check. MLP v1 is competitive with the classical baselines and does not show signs of an implementation or training failure.

The result also supports an expected observation for this project: strong tree-based methods can be highly competitive on relatively low-dimensional tabular data, while the MLP still achieves similar performance.

---

## 10. Training Curve

The required training figure is:

```text
reports/figures/fig_3_mlp_training_curves.png
```

It shows training and validation cross-entropy loss as a function of epoch.

The training loss generally decreases over time, while validation loss also improves but contains some fluctuations in later epochs. Because checkpoint selection is based on validation macro-F1 rather than the final epoch, the best observed validation model is retained.

Markdown reference for the report:

```markdown
![MLP v1 training curves](../../reports/figures/fig_3_mlp_training_curves.png)
```

---

## 11. Reproducibility

The implementation follows the shared project reproducibility rules:

- random seed: **42**
- model parameters controlled through YAML
- common processed dataset
- common train/validation split
- common project evaluation function
- deterministic DataLoader generator
- checkpoint selected only using validation data
- official test split isolated from model development

The input feature dimensionality is inferred from the processed training data instead of being permanently hardcoded. The code also checks that train, validation, and final test feature structure is consistent.

---

## 12. Produced Artifacts

### Source files

```text
configs/mlp_v1.yaml
src/models/mlp.py
src/models/train.py
```

### Generated artifacts already produced

```text
models/mlp_v1.pt
reports/figures/fig_3_mlp_training_curves.png
```

### Artifacts produced during the final test stage

```text
reports/preds/mlp_v1_test.npz
reports/tables/improvements.csv
```

The prediction artifact uses the required format:

```text
y_true   -> shape (n,)
y_proba  -> shape (n, 3)
```

This allows Role 1 to construct the final confusion matrix and error analysis from a common prediction format.

---

## 13. Handoff to MLP v2

The MLP v1 result provides the reference point for Role 5.

Reference configuration:

```text
hidden_dims = [128, 64]
dropout = 0.2
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
optimizer = AdamW
```

Reference result:

```text
validation macro-F1 = 0.9707
```

MLP v2 should attempt to improve this score using configuration changes while reusing the same training implementation.

Possible tunable parameters include:

```text
hidden_dims
dropout
learning_rate
weight_decay
batch_size
optimizer
```

The official test set must not be used to choose among these configurations.

---

## 14. Current Status

Completed:

- [x] PyTorch MLP architecture
- [x] YAML-driven model configuration
- [x] YAML-driven training configuration
- [x] shared metric integration
- [x] shared seed integration
- [x] official processed-data integration
- [x] synthetic end-to-end smoke test
- [x] real train/validation training
- [x] validation macro-F1 model selection
- [x] best checkpoint generation
- [x] training-curve generation
- [x] baseline sanity comparison
- [x] handoff result for MLP v2

Pending:

- [ ] wait for MLP v2 tuning/model-selection stage to finish
- [ ] run MLP v1 on the official test split once
- [ ] save `reports/preds/mlp_v1_test.npz`
- [ ] add MLP v1 row to `reports/tables/improvements.csv`
- [ ] insert final test macro-F1 into the report

---

## 15. Conclusion

The MLP v1 pipeline was successfully implemented and integrated with the shared project structure.

The final MLP v1 architecture uses two hidden layers with batch normalization, ReLU activation, and dropout. The model achieved a **validation macro-F1 of 0.9707**, outperforming the SVM baseline (`0.96074`) and remaining close to the Random Forest baseline (`0.97486`).

The training pipeline is reproducible, configuration-driven, compatible with the shared preprocessing and evaluation modules, and reusable by the MLP v2 engineer.

The only remaining MLP v1 evaluation step is the final one-time test evaluation, which is intentionally postponed until the model-selection and MLP v2 stages are complete.
