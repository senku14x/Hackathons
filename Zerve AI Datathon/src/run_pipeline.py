"""
File: run_pipeline.py

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
End-to-end pipeline runner.
Run this file to execute all steps in sequence.
Ensure training_data.csv and test_data.csv are in the working directory.
"""

import subprocess
import sys

steps = [
    ("step_01_lgbm_cv_oof.py",        "LightGBM 5-fold CV + OOF predictions"),
    ("step_02_lgbm_final_train.py",    "LightGBM final training + test predictions"),
    ("step_03_xgboost_cv_oof.py",      "XGBoost 5-fold CV + OOF predictions"),
    ("step_04_xgboost_final_train.py", "XGBoost final training + test predictions"),
    ("step_05_ensemble_analysis.py",   "Ensemble diagnostics + final submission"),
]

for script, description in steps:
    print(f"\n{'='*80}")
    print(f"Running: {script}")
    print(f"Description: {description}")
    print("="*80)
    result = subprocess.run([sys.executable, script], check=True)

print("\nPipeline complete. Submit: final_submission.csv")
