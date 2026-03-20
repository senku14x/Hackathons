# Methodology

## Problem Statement

Binary classification task: predict the probability that a credit card account is flagged as `bad_flag = 1` (high credit risk). The primary metric is AUC; log loss is tracked as a secondary calibration metric.

---

## Data

| Split | File | Samples | Features | Label |
|---|---|---|---|---|
| Training | `Dev_data_to_be_shared.csv` | ~96,806 | 500+ | `bad_flag` |
| Validation | `validation_data_to_be_shared.csv` | ~41,792 | 500+ | — |

The target variable `bad_flag` is heavily imbalanced at ~1.4% positive rate.

> **Note:** Throughout the code, the validation set is referred to as `test_data` — a naming artefact from the original notebook.

---

## Preprocessing

### Feature / Target Separation
`account_number` (primary key) and `bad_flag` (target) are dropped from the feature matrix before training to prevent data leakage.

### Missing Values
LightGBM handles missing values natively, so no imputation is strictly required. A median-fill step was explored during development but ultimately skipped.

### Normalisation
`StandardScaler` is fit on the training set and applied to both train and validation features. The scaler is fit once before the CV loop — the same scaled matrix is used for all folds and for final inference.

### Class Imbalance
`scale_pos_weight` was trialled to up-weight the minority class but was reverted: on this large dataset it degraded both AUC and log loss, suggesting the imbalance ratio (~1:70) is handled adequately by the model without explicit reweighting. Future work could explore SMOTE.

---

## Model: LightGBM

LightGBM was selected for its strong track record on tabular data, fast training on high-dimensional feature spaces, and native support for missing values.

### Key design choices vs. standard GBDT

| Property | LightGBM behaviour |
|---|---|
| Tree growth | Leaf-wise (splits the highest-loss leaf) rather than level-wise — more accurate but needs `num_leaves` / `max_depth` constraints to avoid overfitting on small data |
| Splitting | Histogram-based bucketing — faster and more memory-efficient |
| Missing values | Handled natively; no imputation needed |
| Categorical features | Can be passed directly without one-hot encoding |

### Final Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| `objective` | `binary` | Binary classification task |
| `metric` | `auc` | Primary evaluation metric |
| `boosting_type` | `gbdt` | Standard gradient boosting |
| `learning_rate` | `0.02` | Low rate for stable convergence; compensated by early stopping |
| `num_leaves` | `31` | Moderate complexity |
| `max_depth` | `6` | Reduced from 8 — prevented overfitting without AUC loss |
| `min_child_samples` | `50` | Minimum samples per leaf split |
| `min_data_in_leaf` | `20` | Minimum samples in a leaf node |
| `feature_fraction` | `0.7` | 70% of features per iteration — adds randomness, reduces overfitting |
| `bagging_fraction` | `0.9` | 90% of data per iteration |
| `bagging_freq` | `5` | Bagging applied every 5 iterations |
| `lambda_l1` | `0.1` | L1 regularisation |
| `lambda_l2` | `0.1` | L2 regularisation |
| `verbose` | `-1` | Suppress training output |

Hyperparameters were tuned manually by monitoring AUC and log loss on held-out folds. The two most impactful changes were:
1. Lowering `learning_rate` from `0.05` → `0.02` improved log loss at the cost of longer training time.
2. Reducing `max_depth` from `8` → `6` eliminated signs of overfitting without hurting AUC.

---

## Validation Strategy: 10-Fold Stratified Cross-Validation

Stratified K-Fold (k=10) is used so that each fold preserves the ~1.4% minority class ratio. Early stopping (`stopping_rounds=50`) is applied within each fold against the fold's validation set.

The model from the final fold is retained for inference on the held-out validation set.

### Per-Fold Results

| Fold | Validation AUC | Validation Log Loss |
|---|---|---|
| 1 | 0.8218 | 0.0636 |
| 2 | 0.8374 | 0.0628 |
| 3 | 0.8416 | 0.0634 |
| 4 | 0.8290 | 0.0631 |
| 5 | 0.8466 | 0.0617 |
| 6 | 0.8255 | 0.0637 |
| 7 | 0.8301 | 0.0632 |
| 8 | 0.8596 | 0.0615 |
| 9 | 0.8324 | 0.0630 |
| 10 | 0.8359 | 0.0633 |
| **Mean** | **0.8360** | **0.0629** |

Low variance across folds confirms the model generalises well and is not sensitive to a particular train/validation split.

---

## Feature Importance

LightGBM's built-in gain-based importance shows that `onus_attribute_2` (column 1117) dominates the feature ranking. Despite this concentration, fold-level AUC is stable, which indicates the feature carries genuine predictive signal rather than representing leakage or noise.

---

## Inference on Validation Set

Predictions are generated using the last-fold model applied to `X_val_scaled` (the scaler-transformed validation features). Probabilities are rounded to 6 decimal places and written to `Credit_Card_Behaviour_Score_Submission.csv`.

### Predicted Probability Distribution (validation set)

| Threshold | % of accounts |
|---|---|
| > 0.1 | 1.26% |
| > 0.3 | 0.19% |
| > 0.5 | 0.02% |

The majority of predictions cluster near 0.01, consistent with a ~1.4% true positive rate in the training data. The model is conservative — low false positive rate, but may under-detect borderline high-risk cases.

---

## Limitations and Future Work

- **Single-fold inference:** Final predictions come from only the last fold's model. Averaging predictions across all 10 fold models (ensemble) would likely improve calibration.
- **Low-risk bias:** Adjusting the decision threshold or post-hoc probability calibration (Platt scaling / isotonic regression) could improve recall for high-risk accounts.
- **Class imbalance:** SMOTE or other oversampling strategies are worth exploring to improve minority-class recall without degrading AUC.
- **Feature engineering:** Domain-specific aggregations (e.g. rolling transaction statistics) were not explored.
- **Hyperparameter search:** Only manual search was performed. Bayesian optimisation (e.g. Optuna) could yield further gains.
