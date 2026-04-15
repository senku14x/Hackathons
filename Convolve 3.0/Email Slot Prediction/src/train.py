"""
Step 5 – Model training with LightGBM LambdaRank + Group K-Fold CV.

The dataset is streamed in chunks; for each chunk a fresh 10-fold GroupKFold
cross-validation is run.  MAP scores from every fold/chunk are collected and
averaged at the end to give the overall performance estimate.

Why LambdaRank?
---------------
The task is to *rank* the 28 weekly slots by engagement probability for each
customer, not to classify a single binary outcome.  LambdaRank optimises a
listwise ranking objective (NDCG) which directly aligns with this goal.
MAP@K is used as the evaluation metric.

Data-leakage guard
------------------
`same_slot_engagement` is computed from open_time_slot and send_time_slot.
Including it as a feature would leak the target; any column correlated
above LEAKAGE_CORR_THRESHOLD with `relevance` is dropped automatically.

Noise injection
---------------
5 % of target labels are randomly flipped during training.  This acts as
a weak regulariser and prevents the model from memorising rare label patterns.
"""

import random

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

from configs.config import (
    FEATURED_TRAIN_FILE, MODEL_PATH,
    TRAIN_MODEL_CHUNK_SIZE, CATEGORICAL_COLUMNS,
    LGBM_PARAMS, N_SPLITS, NUM_BOOST_ROUND, EARLY_STOPPING, LOG_PERIOD,
    LEAKAGE_CORR_THRESHOLD,
)


def _encode_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Label-encode categorical and datetime columns in-place."""
    for col in CATEGORICAL_COLUMNS:
        if col in chunk.columns:
            chunk[col] = chunk[col].astype("category").cat.codes

    for date_col in ("batch_date", "acc_date"):
        if date_col in chunk.columns:
            chunk[date_col] = (
                pd.to_datetime(chunk[date_col], errors="coerce")
                .astype("category")
                .cat.codes
            )
    return chunk


def _drop_leakage(X: pd.DataFrame, y: pd.Series, threshold: float) -> pd.DataFrame:
    """Remove features whose absolute correlation with y exceeds threshold."""
    correlation = X.corrwith(y)
    leakage_cols = correlation[abs(correlation) > threshold].index.tolist()
    if leakage_cols:
        print(f"  Dropping leakage features (|corr| > {threshold}): {leakage_cols}")
        X = X.drop(columns=leakage_cols, errors="ignore")
    return X


def train(
    input_file: str = FEATURED_TRAIN_FILE,
    model_path: str = MODEL_PATH,
    chunksize: int = TRAIN_MODEL_CHUNK_SIZE,
) -> lgb.Booster:
    """
    Train LightGBM LambdaRank model, report per-fold MAP, and save the model.

    Parameters
    ----------
    input_file  : Feature-enriched training CSV.
    model_path  : Where to save the trained model (.txt).
    chunksize   : Rows per chunk.

    Returns
    -------
    The last trained lgb.Booster instance (from the final fold of the final chunk).
    """
    map_scores: list[float] = []
    model: lgb.Booster | None = None

    for chunk in pd.read_csv(input_file, chunksize=chunksize, engine="python"):
        print(f"\nProcessing chunk ({len(chunk):,} rows)…")

        chunk = _encode_chunk(chunk)

        # Build relevance target (ranking label)
        chunk["relevance"] = np.where(
            chunk["open_time_slot"] == chunk["send_time_slot"], 1, 0
        )

        # Features / target split
        X = chunk.drop(
            columns=["relevance", "send_timestamp", "open_timestamp", "customer_code"],
            errors="ignore",
        )
        y = chunk["relevance"]

        X = _drop_leakage(X, y, LEAKAGE_CORR_THRESHOLD)

        # Inject 5 % label noise as a light regulariser
        y = y.apply(lambda v: random.choice([0, 1]) if random.random() < 0.05 else v)

        groups = chunk["customer_code"].astype("category").cat.codes
        gkf = GroupKFold(n_splits=N_SPLITS)

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            print(f"  Fold {fold + 1}/{N_SPLITS}…", end=" ")

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            train_group = X_train.groupby(groups.iloc[train_idx]).size().tolist()
            val_group   = X_val.groupby(groups.iloc[val_idx]).size().tolist()

            train_data = lgb.Dataset(X_train, label=y_train, group=train_group)
            val_data   = lgb.Dataset(X_val,   label=y_val,   group=val_group)

            model = lgb.train(
                LGBM_PARAMS,
                train_data,
                valid_sets=[train_data, val_data],
                num_boost_round=NUM_BOOST_ROUND,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=EARLY_STOPPING, verbose=False),
                    lgb.log_evaluation(period=LOG_PERIOD),
                ],
            )

            y_pred = model.predict(X_val)
            map_score = average_precision_score(y_val, y_pred)
            map_scores.append(map_score)
            print(f"MAP = {map_score:.4f}")

    avg_map = float(np.mean(map_scores))
    print(f"\nAverage MAP across all folds/chunks: {avg_map:.6f}")

    if model is not None:
        model.save_model(model_path)
        print(f"Model saved to: {model_path}")
    else:
        raise RuntimeError("No data was processed – check input_file path.")

    return model


if __name__ == "__main__":
    train()
