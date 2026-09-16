# 4. Improvements

The MLP v2 experiment investigated whether configuration-level hyperparameter
changes could improve the MLP v1 classifier. The model and training source code
were reused without modification; only YAML configuration values were changed.
Both versions used the same processed data, train/validation/test split,
train-fitted scaling, random seed of 42, and shared `evaluate()` function.

## Hyperparameter search

A bounded search evaluated six configurations. The initial four configurations
compared hidden-layer dimensions of `[128, 64]` and `[256, 128]` with dropout
rates of 0.20 and 0.10. Two additional configurations tested a lower learning
rate of 0.0005 and lighter regularization with dropout 0.05 and weight decay
`1e-5`. The learning rate and weight decay were otherwise 0.001 and `1e-4`,
respectively. AdamW and a batch size of 256 were retained throughout the search.

The trials allowed up to 70 epochs with early-stopping patience of 10, except
for the lower-learning-rate trial, which allowed 90 epochs and patience of 15.
Each checkpoint was selected by its highest validation macro-F1, and the final
configuration was selected by the same metric. The test split was excluded
from tuning and evaluated once per final model after selection was complete.

## Candidate configurations and measurements

The six evaluated configurations are listed below. C1 retains the v1 model
and optimizer settings but uses the extended training budget; its rerun result
is distinct from the recorded MLP v1 reference.

| ID | Configuration | Hidden layers | Dropout | Learning rate | Weight decay |
|---|---|---|---:|---:|---:|
| C1 | Extended training | 128, 64 | 0.20 | 0.0010 | 0.00010 |
| C2 | Lower dropout | 128, 64 | 0.10 | 0.0010 | 0.00010 |
| C3 | Wider network | 256, 128 | 0.20 | 0.0010 | 0.00010 |
| C4 | Wider, lower dropout | 256, 128 | 0.10 | 0.0010 | 0.00010 |
| C5 | Lower learning rate | 128, 64 | 0.10 | 0.0005 | 0.00010 |
| C6 | Lighter regularization | 128, 64 | 0.05 | 0.0010 | 0.00001 |

| Model / candidate | Best epoch | Validation macro-F1 | Delta vs v1 | Runtime (s) |
|---|---:|---:|---:|---:|
| MLP v1 reference | 49 | 0.970681 | 0.000000 | Not recorded |
| C1 | 32 | 0.969456 | -0.001225 | 83.3 |
| C2 | 46 | 0.970367 | -0.000314 | 169.4 |
| C3 | 32 | 0.969869 | -0.000812 | 211.9 |
| C4 | 29 | 0.969785 | -0.000896 | 197.4 |
| C5 | 49 | 0.969795 | -0.000886 | 98.8 |
| C6 (selected as v2) | 29 | 0.970580 | -0.000101 | 50.2 |

Candidate measurements are taken from `reports/tables/mlp_v2_tuning.csv`.
Runtime is the observed CPU wall-clock duration of the trial, including
validation and artifact generation. Different stopping epochs and machine load
make these timings descriptive rather than a controlled speed comparison.
Intermediate candidates were not evaluated on the test set, so test metrics
are reported only for the two final models below.

The best modified configuration retained hidden layers `[128, 64]`, reduced
dropout from 0.20 to 0.05, and reduced weight decay from `1e-4` to `1e-5`.
Relative to v1, the maximum training budget increased from 50 to 70 epochs and
patience from 7 to 10. Early stopping retained the checkpoint at epoch 29.
Retraining the selected configuration reproduced its validation macro-F1.

## Effect of the modification

| Model | Validation macro-F1 | Test macro-F1 | Validation delta | Test delta |
|---|---:|---:|---:|---:|
| MLP v1 | 0.970681 | 0.970295 | 0.000000 | 0.000000 |
| MLP v2 | 0.970580 | 0.969771 | -0.000101 | -0.000523 |

Deltas are absolute macro-F1 differences relative to MLP v1; negative values
indicate lower performance. The version metrics and test delta are taken from
`reports/tables/improvements.csv`.

None of the six configurations surpassed the recorded MLP v1 validation
score. Increasing network width did not help, while lighter regularization
came closest to the reference result. The selected v2 model also had a slightly
lower test macro-F1, so the modification did not improve generalization in this
experiment. The results suggest diminishing returns from this limited search
space, rather than demonstrating that further improvements are impossible.
Because the comparison uses a single seed and split, the small differences
should not be interpreted as a statistically established performance gap.
