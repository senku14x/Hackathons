"""
Credit Card Behaviour Score Prediction
---------------------------------------
Uses LightGBM with 10-fold stratified cross-validation to predict
the probability of a customer being a 'bad' flag (credit risk).

Inputs:
    - Dev_data_to_be_shared.csv    : Training data with 'bad_flag' labels
    - validation_data_to_be_shared.csv : Held-out validation data (no labels)

Output:
    - Credit_Card_Behaviour_Score_Submission.csv : Predicted probabilities per account
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler


# ──────────────────────────────────────────────
# 1. Load Data
# ──────────────────────────────────────────────

dev_data = pd.read_csv("Dev_data_to_be_shared.csv")
test_data = pd.read_csv("validation_data_to_be_shared.csv")

print("Development Data:")
print(dev_data.head())
print("\nValidation Data:")
print(test_data.head())


# ──────────────────────────────────────────────
# 2. Feature / Target Preparation
# ──────────────────────────────────────────────

X = dev_data.drop(columns=["account_number", "bad_flag"]).reset_index(drop=True).values
y = dev_data["bad_flag"].reset_index(drop=True).values

val_account_numbers = test_data["account_number"]
X_validation = test_data.drop(columns=["account_number"])

# Normalise features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_val_scaled = scaler.transform(X_validation)


# ──────────────────────────────────────────────
# 3. LightGBM Hyperparameters
# ──────────────────────────────────────────────

params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.02,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 50,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
}


# ──────────────────────────────────────────────
# 4. Stratified K-Fold Cross-Validation
# ──────────────────────────────────────────────

N_SPLITS = 10
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

fold_aucs = []
fold_log_losses = []
model = None  # will hold the last-fold model for feature importance / inference

print("Starting K-Fold Cross-Validation...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
    print(f"  Processing Fold {fold + 1}...")

    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50),
        ],
    )

    y_pred = model.predict(X_val)
    fold_auc = roc_auc_score(y_val, y_pred)
    fold_ll = log_loss(y_val, y_pred)

    fold_aucs.append(fold_auc)
    fold_log_losses.append(fold_ll)
    print(f"  Fold {fold + 1}: AUC = {fold_auc:.4f}, Log Loss = {fold_ll:.4f}")

avg_auc = np.mean(fold_aucs)
avg_log_loss = np.mean(fold_log_losses)
print(f"\nAverage Validation AUC  ({N_SPLITS} folds): {avg_auc:.4f}")
print(f"Average Validation Log Loss ({N_SPLITS} folds): {avg_log_loss:.4f}")


# ──────────────────────────────────────────────
# 5. Feature Importance (last fold model)
# ──────────────────────────────────────────────

lgb.plot_importance(model, max_num_features=20, importance_type="gain")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()
print("Feature importance plot saved to feature_importance.png")


# ──────────────────────────────────────────────
# 6. Generate Predictions on Validation Set
# ──────────────────────────────────────────────

val_predictions = model.predict(X_val_scaled)
val_predictions = np.round(val_predictions, 6)

submission = pd.DataFrame(
    {
        "account_number": val_account_numbers,
        "predicted_probability": val_predictions,
    }
)

submission.to_csv("Credit_Card_Behaviour_Score_Submission.csv", index=False)
print("Submission file saved: Credit_Card_Behaviour_Score_Submission.csv")
