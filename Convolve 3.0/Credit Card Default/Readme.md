# Credit Card Behaviour Score

Predicts the probability of a credit card account being flagged as `bad_flag` (i.e. high credit risk) using LightGBM with 10-fold stratified cross-validation.

Built for the IIT credit risk challenge by **Vishesh Gupta**.

---

## Results

| Metric | Score |
|---|---|
| Average Validation AUC | **0.8360** |
| Average Validation Log Loss | **0.0629** |

Consistent across all 10 folds — see [METHODOLOGY.md](METHODOLOGY.md) for the full fold-by-fold breakdown.

---

## Repository Structure

```
.
├── credit_card_behaviour_score.py   # Main training + inference script
├── METHODOLOGY.md                   # Full methodology and design decisions
├── README.md
└── Credit_Card_Behaviour_Score_Submission.csv   # Generated on run
```

---

## Setup

**Requirements**
```
lightgbm
scikit-learn
pandas
numpy
matplotlib
```

Install with:
```bash
pip install lightgbm scikit-learn pandas numpy matplotlib
```

---

## Usage

Place the two data files in the same directory as the script:
- `Dev_data_to_be_shared.csv` — labelled training data (100k samples, 500+ features)
- `validation_data_to_be_shared.csv` — unlabelled validation data

Then run:
```bash
python credit_card_behaviour_score.py
```

**Outputs:**
- Console: per-fold AUC and log loss, plus averages
- `feature_importance.png` — top-20 features by gain
- `Credit_Card_Behaviour_Score_Submission.csv` — predicted probabilities per account

---

## Approach Summary

- **Model:** LightGBM (GBDT), binary classification
- **Validation:** 10-fold stratified cross-validation
- **Preprocessing:** StandardScaler normalisation; LightGBM handles missing values natively
- **Key finding:** `onus_attribute_2` (column 1117) is the dominant feature by gain, yet fold performance remains stable, confirming it is a genuine signal rather than leakage

See [METHODOLOGY.md](METHODOLOGY.md) for the full write-up.
