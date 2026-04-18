"""
File: step_04_xgboost_final_train.py

Zerve AI Datathon 2025 — Health Insurance Claim Risk Prediction
Team: The Error Guy | Presented by: Vishesh Gupta
Hosted by: Zerve x Techfest IIT Bombay

Objective:
    Predict the probability of a customer filing a health insurance claim.
    Evaluated on the Normalized Gini Coefficient.

Pipeline:
    step_01_lgbm_cv_oof.py        ->  LightGBM 5-fold OOF predictions
    step_02_lgbm_final_train.py   ->  LightGBM final model + test predictions
    step_03_xgboost_cv_oof.py     ->  XGBoost 5-fold OOF predictions  [see note]
    step_04_xgboost_final_train.py->  XGBoost final model + test predictions [see note]
    step_05_ensemble_analysis.py  ->  Stacking / rank-averaging ensemble diagnostics
    run_pipeline.py               ->  End-to-end runner

Note:
    XGBoost scripts follow the same structure as the LightGBM scripts above.
    The LeakageFreeEngineer class is shared across both (sync comments inline).

Final Results:
    XGBoost OOF Gini  : ~0.282
    LightGBM OOF Gini : ~0.275
    Ensemble Gini     : ~0.283  (rank-based averaging)
"""

"""
XGBoost Final Training — generates test set predictions.
Mirrors step_02_lgbm_final_train.py using XGBoost + LeakageFreeEngineerXGB.
Import the engineer class from step_03_xgboost_cv_oof.py or copy it here.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from step_03_xgboost_cv_oof import FROZEN_PARAMS, LeakageFreeEngineerXGB

df_train = pd.read_csv("training_data.csv")
df_test  = pd.read_csv("test_data.csv")

X_train = df_train.drop(columns=["id", "target"])
y_train = df_train["target"]
X_test  = df_test.drop(columns=["id"])

missing_pct = X_train.isnull().sum() / len(X_train)
to_drop = missing_pct[missing_pct > 0.5].index.tolist()
X_train = X_train.drop(columns=to_drop)
X_test  = X_test.drop(columns=to_drop)

ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
FROZEN_PARAMS["scale_pos_weight"] = ratio

engineer = LeakageFreeEngineerXGB()
X_train_eng = engineer.fit_transform(X_train)
X_test_eng  = engineer.transform(X_test)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_eng, y_train, test_size=0.2, stratify=y_train, random_state=42
)

model = xgb.XGBClassifier(**FROZEN_PARAMS)
model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=100,
    verbose=False,
)

test_predictions = model.predict_proba(X_test_eng)[:, 1]
pd.DataFrame({"id": df_test["id"], "xgb_pred": test_predictions}).to_csv(
    "xgboost_test_predictions.csv", index=False
)
print("Saved: xgboost_test_predictions.csv")
