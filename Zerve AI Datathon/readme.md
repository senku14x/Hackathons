# Health Insurance Claim Risk Prediction

**Zerve AI Datathon 2025 — Techfest IIT Bombay**  
**National Finalist**  
**Team:** The Error Guy | **Presented by:** Vishesh Gupta

---

## Problem Statement

Health insurers need to identify high-risk customers — those likely to file significant claims — to improve pricing accuracy and reduce fraud. The task is to predict the **probability** that a customer files a health insurance claim (target = 1), evaluated on the **Normalized Gini Coefficient**.

The dataset contains 50 anonymized features (numeric, categorical, and binary) with significant class imbalance: claim events are rare, making naive classifiers biased toward the majority class.

---

## Repository Structure

```
zerve/
    step_01_lgbm_cv_oof.py          # LightGBM 5-fold stratified CV + OOF predictions
    step_02_lgbm_final_train.py     # LightGBM final model trained on full data
    step_03_xgboost_cv_oof.py       # XGBoost 5-fold stratified CV + OOF predictions
    step_04_xgboost_final_train.py  # XGBoost final model trained on full data
    step_05_ensemble_analysis.py    # Ensemble diagnostics, stacking, submission generation
    run_pipeline.py                 # End-to-end runner (executes steps 01-05 in order)
    README.md
```

---

## Methodology

### Evaluation Metric

The Normalized Gini Coefficient measures how well the **ranking** of predicted probabilities separates claim from non-claim customers. It is equivalent to `2 * AUC - 1` and is the industry standard for insurance risk models because it is calibration-agnostic and handles imbalanced datasets well.

---

### Validation Strategy

All architectural and hyperparameter decisions were driven exclusively by **Out-of-Fold (OOF)** performance, not held-out test scores.

**Phase 1 — Baseline sanity checks:** Single-split validation with vanilla models (no feature engineering) to verify metric correctness and confirm predictive signal.

| Model | Gini (single split) |
|-------|-------------------|
| Logistic Regression | 0.256 |
| LightGBM (vanilla) | 0.263 |
| XGBoost (vanilla) | 0.265 |

**Phase 2 — Rigorous evaluation:** 5-Fold Stratified Cross-Validation. All feature engineering is performed strictly inside the CV loop — encoders and statistics are fit only on training folds, and validation folds remain completely unseen during transformation. This prevents target leakage.

---

### Feature Engineering

After vanilla tuning exposed a parametric ceiling, feature engineering was introduced to improve how information is presented to the models. All transformations are applied within each CV fold.

| Transformation | Applied to | Rationale |
|----------------|-----------|-----------|
| Rank normalization | Numeric features | Ranking-aware; aligned with Gini metric |
| Explicit missingness flags | `feature_45`, `feature_36` | Treats NaN patterns as predictive signal |
| Frequency encoding | `feature_12` (categorical) | Stable, leak-free count-based encoding |
| Interaction features | `feature_45 x feature_36`, `feature_24 x feature_38`, `feature_45 x feature_22` | Captures cross-feature risk signals |
| One-hot encoding | Remaining categoricals (XGBoost pipeline) | Sparse-friendly for tree splits |
| Native categoricals | Remaining categoricals (LightGBM pipeline) | LightGBM handles these natively |

XGBoost and LightGBM use **model-specific feature representations** to match their inductive biases. Feature importance analysis confirmed the two models prioritize different feature subsets, indicating complementary risk signals.

---

### Models

**XGBoost** — optimized for depth-wise tree splits. Uses rank-normalized numerics, OHE categoricals, and missingness indicators.
- OOF Normalized Gini: **~0.282**

**LightGBM** — optimized for leaf-wise splits. Uses a hybrid representation (rank features + raw values) with native categorical handling and missingness-aware features.
- OOF Normalized Gini: **~0.275**

Both models were tuned via Optuna with hyperparameter search driven exclusively by OOF Gini.

---

### Ensemble Strategy

The Pearson correlation between XGBoost and LightGBM OOF predictions was **~0.93**, indicating strong convergence on a shared risk ranking structure. This rules out aggressive stacking but motivates conservative ensemble averaging.

**Method: Rank-based averaging.** Both model predictions are rank-normalized before averaging. Since Normalized Gini depends purely on ordering, rank averaging is robust to calibration differences between the two models.

**Stacking was evaluated** (Logistic Regression and Elastic Net meta-learners on OOF predictions) but produced marginal improvement (<0.003 Gini) that did not justify the added overfitting risk.

---

## Results

| Model / Strategy | OOF Normalized Gini |
|------------------|-------------------|
| XGBoost vanilla (baseline) | 0.265 |
| LightGBM vanilla (baseline) | 0.263 |
| XGBoost (feature-engineered) | ~0.282 |
| LightGBM (feature-engineered) | ~0.275 |
| Final Ensemble (rank-based) | **~0.283** |

---

## Outputs

| File | Description |
|------|-------------|
| `xgboost_oof_predictions.csv` | XGBoost OOF predictions (used for ensemble diagnostics) |
| `lightgbm_oof_predictions.csv` | LightGBM OOF predictions |
| `xgboost_test_predictions.csv` | XGBoost predictions on test set |
| `lightgbm_test_predictions.csv` | LightGBM predictions on test set |
| `final_submission.csv` | Final submission file (`id`, `target` probability) |

---

## Installation

```bash
pip install pandas numpy scikit-learn lightgbm xgboost scipy
```

GPU acceleration is used by default (`device: gpu` for LightGBM, `tree_method: gpu_hist` for XGBoost). Change to CPU if no GPU is available.

---

## Usage

Place `training_data.csv` and `test_data.csv` in the working directory, then run:

```bash
python run_pipeline.py
```

Or execute steps individually in order (01 through 05). The final submission file is `final_submission.csv`.

---

## Tech Stack

- **Models:** XGBoost, LightGBM
- **Validation:** Scikit-learn StratifiedKFold
- **Feature engineering:** Pandas, NumPy (rank normalization, frequency encoding, OHE)
- **Ensemble diagnostics:** SciPy (rank averaging), Scikit-learn (Logistic Regression, Elastic Net)
- **Hyperparameter tuning:** Optuna (OOF-driven)

---

## Acknowledgements

Dataset and problem statement provided by Zerve in partnership with Techfest IIT Bombay for the Zerve AI Datathon 2025.
