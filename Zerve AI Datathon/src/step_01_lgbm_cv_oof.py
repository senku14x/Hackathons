"""
File: step_01_lgbm_cv_oof.py

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
LightGBM 5-Fold Stratified CV with OOF Predictions
Updated: Synchronized Feature Engineering with XGBoost
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("LIGHTGBM 5-FOLD CV WITH OOF PREDICTIONS (SYNCED FE)")
print("="*80)

# --- FROZEN HYPERPARAMETERS ---
FROZEN_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'device': 'gpu',
    'random_state': 42,
    'verbose': -1,
    'n_estimators': 2000,
    'learning_rate': 0.011142820996144547,
    'num_leaves': 33,
    'max_depth': 7,
    'min_child_samples': 496,
    'feature_fraction': 0.5043728591108573,
    'bagging_fraction': 0.4814175864415237,
    'bagging_freq': 4,
    'lambda_l1': 0.025055464387457583,
    'lambda_l2': 2.047941242904544
}

# --- SYNCED FEATURE ENGINEERING CLASS ---
class LeakageFreeEngineer:
    """Synchronized with XGBoost engineering_pipeline"""
    def __init__(self):
        self.frequency_map = None
        self.cat_cols = []

    def fit(self, df_in):
        # Identify categorical columns (excluding feature_12 which is frequency encoded)
        self.cat_cols = [c for c in df_in.columns
                         if df_in[c].dtype == 'object' and c != 'feature_12']

        # Fit Frequency Map for feature_12
        if 'feature_12' in df_in.columns:
            col = df_in['feature_12'].fillna('MISSING').astype(str)
            self.frequency_map = col.value_counts().to_dict()
        return self

    def transform(self, df_in):
        df = df_in.copy()

        # 1. Interactions (Exact sync with XGBoost)
        df['eng_45_x_36'] = df['feature_45'] * df['feature_36']
        df['eng_45_div_36'] = df['feature_45'] / (df['feature_36'].fillna(0) + 1e-5)
        df.loc[df['feature_36'].isnull(), 'eng_45_div_36'] = np.nan
        df['eng_24_x_38'] = df['feature_24'] * df['feature_38']
        df['eng_45_x_22'] = df['feature_45'] * df['feature_22']

        # 2. Missingness flags
        df['eng_45_is_missing'] = df['feature_45'].isnull().astype(int)
        df['eng_36_is_missing'] = df['feature_36'].isnull().astype(int)

        # 3. Frequency encoding for feature_12
        if 'feature_12' in df.columns:
            col = df['feature_12'].fillna('MISSING').astype(str)
            df['feature_12_freq'] = col.map(self.frequency_map).fillna(1)
            # LightGBM handles the original feature_12 if we drop it or keep as category
            # To match XGBoost logic (which drops it in get_column_groups), we drop it:
            df = df.drop(columns=['feature_12'])

        # 4. Handle Categoricals for LightGBM
        for col in self.cat_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')

        return df, self.cat_cols

    def fit_transform(self, df_in):
        return self.fit(df_in).transform(df_in)

def normalized_gini(y_true, y_pred_proba):
    return 2 * roc_auc_score(y_true, y_pred_proba) - 1

# --- LOAD DATA ---
df_train = pd.read_csv('training_data.csv')
X = df_train.drop(columns=['id', 'target'])
y = df_train['target']

# Cleanup (Drop > 50% missing)
missing_pct = X.isnull().sum() / len(X)
to_drop = missing_pct[missing_pct > 0.5].index.tolist()
X = X.drop(columns=to_drop)

ratio = float(np.sum(y == 0)) / np.sum(y == 1)
FROZEN_PARAMS['scale_pos_weight'] = ratio

# --- 5-FOLD STRATIFIED CV ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_predictions = np.zeros(len(X))
fold_scores = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

    engineer = LeakageFreeEngineer()
    X_train_eng, cat_cols = engineer.fit_transform(X_train_fold)
    X_val_eng, _ = engineer.transform(X_val_fold)

    model = lgb.LGBMClassifier(**FROZEN_PARAMS)
    model.fit(
        X_train_eng, y_train_fold,
        eval_set=[(X_val_eng, y_val_fold)],
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )

    val_preds = model.predict_proba(X_val_eng)[:, 1]
    oof_predictions[val_idx] = val_preds
    fold_scores.append(normalized_gini(y_val_fold, val_preds))
    print(f"Fold {fold_idx} Gini: {fold_scores[-1]:.4f}")

print(f"\nOverall OOF Gini: {normalized_gini(y, oof_predictions):.4f}")

# Save OOF
pd.DataFrame({'id': df_train['id'], 'target': y, 'lgb_oof': oof_predictions}).to_csv('lightgbm_oof_predictions.csv', index=False)