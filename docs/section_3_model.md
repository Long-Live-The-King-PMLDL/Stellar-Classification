# 3. Proposed Model

## 3.1 Architecture

The proposed model is a multilayer perceptron (MLP) implemented in PyTorch for three-class stellar object classification. The model takes eight processed numerical features as input:

`alpha`, `delta`, `u`, `g`, `r`, `i`, `z`, and `redshift`.

The architecture consists of two hidden fully connected blocks followed by a three-unit output layer:

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

The network returns raw logits. Softmax is applied only during evaluation, while training uses `CrossEntropyLoss`, which operates directly on logits.

## 3.2 Hyperparameters and Training Procedure

All model and training parameters are defined in `configs/mlp_v1.yaml`, allowing the same implementation to be reused for the modified MLP version with a different configuration.

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

The model was trained on 79,999 training samples and evaluated after every epoch on 10,000 validation samples. The shared `evaluate(y_true, y_proba)` function was used for metric computation, and validation macro-F1 was used as the model-selection criterion.

Whenever validation macro-F1 improved, the corresponding checkpoint was saved to:

`models/mlp_v1.pt`

The best checkpoint was obtained at **epoch 49**, with a validation macro-F1 of **0.9707**.

## 3.3 Training Curves

The training and validation loss curves are shown in Figure 3.

![MLP v1 training curves](/reports/figures/fig_3_mlp_training_curves.png)

The training loss decreased from approximately 0.355 at the first epoch to approximately 0.104 by the end of training. Validation loss also decreased overall, with some fluctuations in later epochs. Checkpoint selection based on validation macro-F1 ensured that the best-performing model was retained rather than simply using the final epoch.

## 3.4 MLP v1 Results

After model selection was complete, the best MLP v1 checkpoint was evaluated once on the official test split.

| Metric | Validation | Test |
|---|---:|---:|
| Macro-F1 | **0.9707** | **0.9703** |
| Accuracy | — | 0.9739 |
| ROC-AUC OVR | — | 0.9947 |
| Cross-entropy loss | — | 0.0979 |

The difference between validation and test macro-F1 was only **0.0004**, indicating stable generalization from validation to unseen test data.

For a validation sanity check, MLP v1 achieved a macro-F1 of **0.9707**, compared with **0.96074** for the RBF SVM baseline and **0.97486** for Random Forest. Thus, the MLP outperformed the SVM baseline and remained close to the Random Forest baseline on the same validation split.

The final prediction artifact was saved to:

`reports/preds/mlp_v1_test.npz`

with `y_true` of shape `(10000,)` and `y_proba` of shape `(10000, 3)`.
