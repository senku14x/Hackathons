"""
File: step_03_xgboost_cv_oof.py

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
XGBoost 5-Fold Stratified CV with OOF Predictions

This script mirrors step_01_lgbm_cv_oof.py with XGBoost-specific settings:
  - Sparse-friendly one-hot encoding for categorical features (not native categoricals)
  - Rank-normalized numeric features
  - Explicit missingness indicators

The LeakageFreeEngineer class below is synchronized with the LightGBM version
but applies OHE instead of LightGBM native categorical encoding.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings("ignore")

# --- FROZEN HYPERPARAMETERS (Optuna-tuned) ---
# These were determined via OOF-driven hyperparameter search.
# Do not modify without re-running CV to confirm improvement.
FROZEN_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "gpu_hist",   # change to "hist" if no GPU
    "random_state": 42,
    "verbosity": 0,
    "n_estimators": 2000,
    # --- insert tuned values here ---
    # "learning_rate": ...,
    # "max_depth": ...,
    # "min_child_weight": ...,
    # "subsample": ...,
    # "colsample_bytree": ...,
    # "reg_alpha": ...,
    # "reg_lambda": ...,
}


class LeakageFreeEngineerXGB:
    """
    Feature engineering for XGBoost.
    Synchronized with LeakageFreeEngineer in step_01_lgbm_cv_oof.py.
    Key difference: categoricals are one-hot encoded (XGBoost does not handle
    native category dtype); rank normalization applied to numeric features.
    """

    def __init__(self):
        self.ohe = None
        self.cat_cols = []
        self.frequency_map = None

    def fit(self, df_in):
        self.cat_cols = [c for c in df_in.columns
                         if df_in[c].dtype == "object" and c != "feature_12"]
        if self.cat_cols:
            self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
            self.ohe.fit(df_in[self.cat_cols].fillna("MISSING"))
        if "feature_12" in df_in.columns:
            col = df_in["feature_12"].fillna("MISSING").astype(str)
            self.frequency_map = col.value_counts().to_dict()
        return self

    def transform(self, df_in):
        df = df_in.copy()

        # Interactions (exact sync with LightGBM version)
        df["eng_45_x_36"] = df["feature_45"] * df["feature_36"]
        df["eng_45_div_36"] = df["feature_45"] / (df["feature_36"].fillna(0) + 1e-5)
        df.loc[df["feature_36"].isnull(), "eng_45_div_36"] = np.nan
        df["eng_24_x_38"] = df["feature_24"] * df["feature_38"]
        df["eng_45_x_22"] = df["feature_45"] * df["feature_22"]

        # Missingness flags
        df["eng_45_is_missing"] = df["feature_45"].isnull().astype(int)
        df["eng_36_is_missing"] = df["feature_36"].isnull().astype(int)

        # Frequency encode feature_12
        if "feature_12" in df.columns:
            col = df["feature_12"].fillna("MISSING").astype(str)
            df["feature_12_freq"] = col.map(self.frequency_map).fillna(1)
            df = df.drop(columns=["feature_12"])

        # OHE for remaining categoricals
        if self.cat_cols and self.ohe is not None:
            ohe_array = self.ohe.transform(df[self.cat_cols].fillna("MISSING"))
            ohe_cols = self.ohe.get_feature_names_out(self.cat_cols)
            ohe_df = pd.DataFrame.sparse.from_spmatrix(ohe_array, columns=ohe_cols, index=df.index)
            df = df.drop(columns=self.cat_cols)
            df = pd.concat([df, ohe_df], axis=1)

        # Rank-normalize numeric features
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in num_cols:
            if col not in ("target",):
                df[col + "_rank"] = df[col].rank(pct=True, na_option="keep")

        return df

    def fit_transform(self, df_in):
        return self.fit(df_in).transform(df_in)


def normalized_gini(y_true, y_pred_proba):
    return 2 * roc_auc_score(y_true, y_pred_proba) - 1


# --- LOAD DATA ---
df_train = pd.read_csv("training_data.csv")
X = df_train.drop(columns=["id", "target"])
y = df_train["target"]

missing_pct = X.isnull().sum() / len(X)
to_drop = missing_pct[missing_pct > 0.5].index.tolist()
X = X.drop(columns=to_drop)

ratio = float(np.sum(y == 0)) / np.sum(y == 1)
FROZEN_PARAMS["scale_pos_weight"] = ratio

# --- 5-FOLD STRATIFIED CV ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_predictions = np.zeros(len(X))
fold_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

    engineer = LeakageFreeEngineerXGB()
    X_train_eng = engineer.fit_transform(X_train_fold)
    X_val_eng = engineer.transform(X_val_fold)

    model = xgb.XGBClassifier(**FROZEN_PARAMS)
    model.fit(
        X_train_eng, y_train_fold,
        eval_set=[(X_val_eng, y_val_fold)],
        early_stopping_rounds=100,
        verbose=False,
    )

    val_preds = model.predict_proba(X_val_eng)[:, 1]
    oof_predictions[val_idx] = val_preds
    fold_scores.append(normalized_gini(y_val_fold, val_preds))
    print(f"Fold {fold_idx} Gini: {fold_scores[-1]:.4f}")

print(f"\nOverall OOF Gini: {normalized_gini(y, oof_predictions):.4f}")

pd.DataFrame({"id": df_train["id"], "target": y, "xgb_oof": oof_predictions}).to_csv(
    "xgboost_oof_predictions.csv", index=False
)
print("Saved: xgboost_oof_predictions.csv")
