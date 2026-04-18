"""
File: step_05_ensemble_analysis.py

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
Stacking Ensemble: Logistic Regression Meta-Learner
Train on OOF predictions, evaluate vs simple ensemble strategies
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("STACKING ENSEMBLE WITH LOGISTIC REGRESSION")
print("="*80)


def normalized_gini(y_true, y_pred_proba):
    """Normalized Gini coefficient"""
    return 2 * roc_auc_score(y_true, y_pred_proba) - 1


# --- LOAD OOF PREDICTIONS ---
print("\n[1/4] Loading OOF predictions...")
xgb_oof = pd.read_csv('xgboost_oof_predictions.csv')
lgb_oof = pd.read_csv('lightgbm_oof_predictions.csv')

# Verify alignment
assert (xgb_oof['id'] == lgb_oof['id']).all(), "ID mismatch!"
assert (xgb_oof['target'] == lgb_oof['target']).all(), "Target mismatch!"

y_true = xgb_oof['target'].values
xgb_preds = xgb_oof['xgb_oof'].values
lgb_preds = lgb_oof['lgb_oof'].values

print(f"  ✓ Loaded {len(y_true)} OOF predictions")
print(f"  ✓ XGBoost OOF Gini: {normalized_gini(y_true, xgb_preds):.4f}")
print(f"  ✓ LightGBM OOF Gini: {normalized_gini(y_true, lgb_preds):.4f}")


# --- PREPARE FEATURES FOR META-LEARNER ---
print("\n[2/4] Preparing meta-learner features...")

# Strategy 1: Raw probabilities
X_meta_raw = np.column_stack([xgb_preds, lgb_preds])

# Strategy 2: Rank-normalized probabilities (often more stable)
xgb_ranks = rankdata(xgb_preds, method='average') / len(xgb_preds)
lgb_ranks = rankdata(lgb_preds, method='average') / len(lgb_preds)
X_meta_ranks = np.column_stack([xgb_ranks, lgb_ranks])

# Strategy 3: Both raw + ranks (4 features)
X_meta_combined = np.column_stack([xgb_preds, lgb_preds, xgb_ranks, lgb_ranks])

print(f"  ✓ Meta-features prepared:")
print(f"    - Raw probabilities: {X_meta_raw.shape}")
print(f"    - Rank probabilities: {X_meta_ranks.shape}")
print(f"    - Combined: {X_meta_combined.shape}")


# --- TRAIN STACKING MODELS ---
print("\n[3/4] Training stacking meta-learners...")

# Logistic Regression with different regularization strengths
meta_models = {
    'LR_raw_C1': (LogisticRegression(C=1.0, random_state=42, max_iter=1000), X_meta_raw),
    'LR_raw_C0.1': (LogisticRegression(C=0.1, random_state=42, max_iter=1000), X_meta_raw),
    'LR_raw_C10': (LogisticRegression(C=10.0, random_state=42, max_iter=1000), X_meta_raw),
    'LR_ranks_C1': (LogisticRegression(C=1.0, random_state=42, max_iter=1000), X_meta_ranks),
    'LR_combined_C1': (LogisticRegression(C=1.0, random_state=42, max_iter=1000), X_meta_combined),
}

stacking_results = {}

for name, (model, X_meta) in meta_models.items():
    # Train meta-learner on OOF predictions
    model.fit(X_meta, y_true)

    # Predict (these are still on training data, but fair since meta-learner trained on OOF)
    meta_preds = model.predict_proba(X_meta)[:, 1]
    gini = normalized_gini(y_true, meta_preds)

    # Get learned weights
    if X_meta.shape[1] == 2:
        weights = model.coef_[0]
        intercept = model.intercept_[0]
        stacking_results[name] = {
            'gini': gini,
            'xgb_weight': weights[0],
            'lgb_weight': weights[1],
            'intercept': intercept,
            'model': model,
            'X_meta': X_meta
        }
    else:
        stacking_results[name] = {
            'gini': gini,
            'weights': model.coef_[0],
            'intercept': model.intercept_[0],
            'model': model,
            'X_meta': X_meta
        }

print("  ✓ Meta-learners trained")


# --- COMPARE WITH BASELINE ENSEMBLES ---
print("\n[4/4] Comparing stacking vs baseline ensembles...")

# Baseline strategies
rank_avg = (xgb_ranks + lgb_ranks) / 2
rank_avg_gini = normalized_gini(y_true, rank_avg)

simple_avg = (xgb_preds + lgb_preds) / 2
simple_avg_gini = normalized_gini(y_true, simple_avg)

# Weighted averages
weighted_results = []
for w_xgb in [0.3, 0.4, 0.5, 0.6, 0.7]:
    w_lgb = 1.0 - w_xgb
    weighted = w_xgb * xgb_preds + w_lgb * lgb_preds
    gini = normalized_gini(y_true, weighted)
    weighted_results.append((w_xgb, w_lgb, gini))


# --- RESULTS SUMMARY ---
print("\n" + "="*80)
print("ENSEMBLE STRATEGY COMPARISON")
print("="*80)

print(f"\n📊 BASELINE STRATEGIES:")
print(f"  XGBoost only:         {normalized_gini(y_true, xgb_preds):.4f}")
print(f"  LightGBM only:        {normalized_gini(y_true, lgb_preds):.4f}")
print(f"  Rank Average:         {rank_avg_gini:.4f}")
print(f"  Simple Average:       {simple_avg_gini:.4f}")

print(f"\n📊 WEIGHTED AVERAGES:")
for w_xgb, w_lgb, gini in weighted_results:
    print(f"  ({w_xgb:.1f}, {w_lgb:.1f}):           {gini:.4f}")

print(f"\n🧠 STACKING (LOGISTIC REGRESSION):")
for name, results in stacking_results.items():
    print(f"\n  {name}:")
    print(f"    OOF Gini:          {results['gini']:.4f}")
    if 'xgb_weight' in results:
        print(f"    XGBoost coef:      {results['xgb_weight']:.4f}")
        print(f"    LightGBM coef:     {results['lgb_weight']:.4f}")
        print(f"    Intercept:         {results['intercept']:.4f}")
    else:
        print(f"    Coefficients:      {results['weights']}")
        print(f"    Intercept:         {results['intercept']:.4f}")


# --- FIND BEST STRATEGY ---
print("\n" + "="*80)
print("BEST STRATEGY SELECTION")
print("="*80)

all_strategies = {
    'rank_average': rank_avg_gini,
    'simple_average': simple_avg_gini,
}

for name, results in stacking_results.items():
    all_strategies[name] = results['gini']

for w_xgb, w_lgb, gini in weighted_results:
    all_strategies[f'weighted_{w_xgb:.1f}_{w_lgb:.1f}'] = gini

best_strategy = max(all_strategies.items(), key=lambda x: x[1])
print(f"\n🏆 BEST STRATEGY: {best_strategy[0]}")
print(f"   OOF Gini: {best_strategy[1]:.4f}")

# Calculate improvement over best single model
best_single = max(normalized_gini(y_true, xgb_preds), normalized_gini(y_true, lgb_preds))
improvement = best_strategy[1] - best_single
print(f"   Improvement over best single: +{improvement:.4f} ({improvement/best_single*100:.2f}%)")


# --- RECOMMENDATION ---
print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

if 'LR_' in best_strategy[0]:
    print(f"\n✅ Stacking outperforms simple ensembles!")
    print(f"   Best meta-learner: {best_strategy[0]}")
    print(f"   OOF Gini: {best_strategy[1]:.4f}")
    print("\n⚠️  CAUTION:")
    print("   - Stacking can overfit on OOF predictions")
    print("   - Test set performance may be lower than OOF suggests")
    print("   - Consider using if improvement > 0.003")

    if improvement < 0.003:
        print("\n💡 SUGGESTION: Improvement is marginal (<0.003)")
        print("   → Stick with rank_average for safety (less overfitting risk)")
    else:
        print("\n💡 SUGGESTION: Significant improvement detected!")
        print("   → Proceed with stacking if you're comfortable with the risk")

else:
    print(f"\n❌ Stacking does NOT outperform simple ensembles")
    print(f"   Best strategy: {best_strategy[0]}")
    print(f"   → Stick with simpler approach (less overfitting risk)")


# --- SAVE STACKING MODEL (OPTIONAL) ---
print("\n" + "="*80)
print("SAVING BEST STACKING MODEL")
print("="*80)

if 'LR_' in best_strategy[0] and improvement >= 0.003:
    import pickle

    best_meta_learner = stacking_results[best_strategy[0]]['model']

    # Save model
    with open('stacking_meta_learner.pkl', 'wb') as f:
        pickle.dump(best_meta_learner, f)

    # Save metadata
    import json
    stacking_config = {
        'strategy': best_strategy[0],
        'oof_gini': float(best_strategy[1]),
        'improvement': float(improvement),
        'xgb_coef': float(stacking_results[best_strategy[0]].get('xgb_weight', 0)),
        'lgb_coef': float(stacking_results[best_strategy[0]].get('lgb_weight', 0)),
        'intercept': float(stacking_results[best_strategy[0]]['intercept']),
        'feature_type': 'ranks' if 'ranks' in best_strategy[0] else 'raw',
    }

    with open('stacking_config.json', 'w') as f:
        json.dump(stacking_config, f, indent=2)

    print(f"✅ Stacking model saved:")
    print(f"   - stacking_meta_learner.pkl")
    print(f"   - stacking_config.json")
    print(f"\n⚠️  To use stacking for submission:")
    print(f"   1. Modify generate_submission.py to load this model")
    print(f"   2. Apply meta-learner to test predictions")
else:
    print("ℹ️  Stacking not recommended - no model saved")
    print("   → Use frozen_ensemble_strategy.json from ensemble_diagnostics.py")

print("\n" + "="*80)
print("✅ STACKING ANALYSIS COMPLETE")
print("="*80)

# ==============================================================================
# PART 2: ELASTIC NET META-LEARNER
# ==============================================================================

"""
Stacking Ensemble: Elastic Net Meta-Learner
Combines L1 and L2 regularization to handle high model correlation.
"""

# --- (imports and data loading shared from Part 1 above) ---
from sklearn.linear_model import ElasticNet, ElasticNetCV

print("="*80)
print("PART 2: ELASTIC NET META-LEARNER")
print("="*80)

# --- 2. PREPARE META-FEATURES ---
# We use Rank-Normalized features as they are more stable for linear models [cite: 16]
xgb_ranks = rankdata(xgb_preds) / len(xgb_preds)
lgb_ranks = rankdata(lgb_preds) / len(lgb_preds)
X_meta = np.column_stack([xgb_ranks, lgb_ranks])

# --- 3. TRAIN ELASTIC NET META-LEARNER ---
print("\n[2/4] Training Elastic Net meta-learners...")

# We test different L1/L2 ratios:
# l1_ratio = 1 is Lasso, l1_ratio = 0 is Ridge (using ElasticNet logic)
configs = [
    {'l1_ratio': 0.1, 'alpha': 0.001, 'name': 'EN_RidgeHeavy'},
    {'l1_ratio': 0.5, 'alpha': 0.01, 'name': 'EN_Balanced'},
    {'l1_ratio': 0.9, 'alpha': 0.01, 'name': 'EN_LassoHeavy'}
]

stacking_results = {}

for cfg in configs:
    model = ElasticNet(alpha=cfg['alpha'], l1_ratio=cfg['l1_ratio'], random_state=42)
    model.fit(X_meta, y_true)

    # Predict continuous scores (order matters for Gini)
    preds = model.predict(X_meta)
    gini = normalized_gini(y_true, preds)

    stacking_results[cfg['name']] = {
        'gini': gini,
        'weights': model.coef_,
        'intercept': model.intercept_,
        'model': model
    }

# Bonus: Auto-tuned ElasticNet
print("Running ElasticNetCV for automated tuning...")
encv = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=5, random_state=42)
encv.fit(X_meta, y_true)
encv_preds = encv.predict(X_meta)
stacking_results['EN_CV_Optimized'] = {
    'gini': normalized_gini(y_true, encv_preds),
    'weights': encv.coef_,
    'intercept': encv.intercept_,
    'model': encv
}

# --- 4. COMPARE AND FINAL RECOMMENDATION ---
print("\n" + "="*80)
print("ENSEMBLE STRATEGY COMPARISON")
print("="*80)

best_single = max(normalized_gini(y_true, xgb_preds), normalized_gini(y_true, lgb_preds))
print(f"Best Single Model (XGBoost): {best_single:.4f}")

for name, res in stacking_results.items():
    improvement = res['gini'] - best_single
    print(f"\n{name}:")
    print(f"  OOF Gini:    {res['gini']:.4f}")
    print(f"  Improvement: +{improvement:.4f}")
    print(f"  Weights:     XGB={res['weights'][0]:.4f}, LGB={res['weights'][1]:.4f}")

# Strategy Decision
best_name = max(stacking_results, key=lambda x: stacking_results[x]['gini'])
best_gini = stacking_results[best_name]['gini']

print("\n" + "="*80)
print("FINAL DECISION")
print("="*80)

if best_gini > best_single + 0.001:
    print(f"✅ USE ELASTIC NET STACKING ({best_name})")
    print(f"   Reason: Meaningful gain over single model.")
else:
    print("❌ STICK TO SIMPLE WEIGHTED AVERAGE OR SINGLE MODEL")
    print("   Reason: Stacking complexity not justified by marginal Gini gain.")