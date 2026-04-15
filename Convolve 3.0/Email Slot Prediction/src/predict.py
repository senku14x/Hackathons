"""
Step 6 – Test-set preprocessing and inference.

Mirrors the training preprocessing pipeline exactly:
  1. Merge test communication history with CDNA (asof merge).
  2. Rename v11 → acc_date for consistency.
  3. Generate time-slot features and encode categoricals.
  4. Expand each customer to all 28 slots.
  5. Run the saved LightGBM model to score every (customer, slot) pair.
  6. Rank slots by predicted probability and write the submission file.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb

from configs.config import (
    TEST_COMM_FILE, TEST_CDNA_FILE, SUBMISSION_FILE,
    MERGED_TEST_FILE, MODEL_PATH, SUBMISSION_OUTPUT_FILE,
    TEST_PRED_CHUNK_SIZE, N_SLOTS, FINAL_FEATURES,
    STRING_COLUMNS, CATEGORICAL_COLUMNS,
)
from src.data_merging import merge_communication_cdna
from src.feature_engineering import get_weekly_slot


def prepare_test_data(
    comm_file: str = TEST_COMM_FILE,
    cdna_file: str = TEST_CDNA_FILE,
    output_file: str = MERGED_TEST_FILE,
) -> None:
    """
    Merge and lightly clean the test communication and CDNA files.

    Uses the same asof-merge logic as the training pipeline so that the
    resulting dataset has an identical schema.

    Parameters
    ----------
    comm_file   : Raw test communication history CSV.
    cdna_file   : Raw test CDNA CSV.
    output_file : Path for the merged test CSV.
    """
    # Re-use the training merge logic; it works for test data too.
    merge_communication_cdna(
        comm_file=comm_file,
        cdna_file=cdna_file,
        output_file=output_file,
    )

    # Rename v11 → acc_date in the merged output
    data = pd.read_csv(output_file)
    if "v11" in data.columns:
        data.rename(columns={"v11": "acc_date"}, inplace=True)
    data.to_csv(output_file, index=False)
    print(f"Test data prepared and saved to: {output_file}")


def predict(
    test_file: str = MERGED_TEST_FILE,
    model_path: str = MODEL_PATH,
    submission_file: str = SUBMISSION_FILE,
    output_file: str = SUBMISSION_OUTPUT_FILE,
    chunksize: int = TEST_PRED_CHUNK_SIZE,
) -> None:
    """
    Score all (customer, slot) pairs and write a ranked submission CSV.

    Each customer in the test set is assigned a predicted engagement
    probability for all 28 weekly slots.  Slots are ranked in descending
    order of probability and written as ["slot_1", "slot_7", ...].

    Parameters
    ----------
    test_file        : Merged test CSV from prepare_test_data.
    model_path       : Saved LightGBM model (.txt).
    submission_file  : Original submission template (customer_code column).
    output_file      : Where to write the final ranked predictions.
    chunksize        : Rows per chunk.
    """
    print(f"Loading model from: {model_path}")
    model = lgb.Booster(model_file=model_path)

    all_predictions: list[pd.DataFrame] = []

    for chunk in pd.read_csv(test_file, chunksize=chunksize, engine="python"):
        print(f"  Processing chunk ({len(chunk):,} rows)…")

        # ── String conversion (must match training order) ──────────────────
        for col in STRING_COLUMNS:
            if col in chunk.columns:
                chunk[col] = chunk[col].astype(str)

        # ── Time slot features ─────────────────────────────────────────────
        chunk["send_timestamp"] = pd.to_datetime(
            chunk["send_timestamp"], errors="coerce"
        )
        chunk["open_timestamp"] = pd.to_datetime(
            chunk["open_timestamp"], errors="coerce"
        )
        chunk["send_time_slot"] = chunk["send_timestamp"].apply(get_weekly_slot)
        chunk["open_time_slot"] = chunk["open_timestamp"].apply(get_weekly_slot)

        for col in ("send_time_slot", "open_time_slot"):
            chunk[col] = chunk[col].replace("No Engagement", np.nan)
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        # ── Numeric / datetime encoding ────────────────────────────────────
        if "batch_id" in chunk.columns:
            chunk["batch_id"] = pd.to_numeric(chunk["batch_id"], errors="coerce")

        for date_col in ("batch_date", "acc_date"):
            if date_col in chunk.columns:
                chunk[date_col] = (
                    pd.to_datetime(chunk[date_col], errors="coerce")
                    .astype("category")
                    .cat.codes
                )

        # ── Categorical encoding ───────────────────────────────────────────
        for col in CATEGORICAL_COLUMNS:
            if col in chunk.columns:
                chunk[col] = chunk[col].astype("category").cat.codes

        # ── Expand to all 28 slots ─────────────────────────────────────────
        unique_customers = chunk["customer_code"].unique()
        all_slots = pd.DataFrame({
            "customer_code": np.repeat(unique_customers, N_SLOTS),
            "slot": np.tile(range(1, N_SLOTS + 1), len(unique_customers)),
        })
        expanded_chunk = all_slots.merge(chunk, on="customer_code", how="left")

        # Fill missing slot engagement history with 0
        for slot in range(1, N_SLOTS + 1):
            col = f"slot_{slot}_engagements"
            if col not in expanded_chunk.columns:
                expanded_chunk[col] = 0

        # ── Predict ────────────────────────────────────────────────────────
        X_chunk = expanded_chunk[FINAL_FEATURES].copy()
        expanded_chunk["engagement_probability"] = model.predict(X_chunk)

        all_predictions.append(
            expanded_chunk[["customer_code", "slot", "engagement_probability"]]
        )

    # ── Rank slots per customer ────────────────────────────────────────────
    print("Ranking slots by engagement probability…")
    all_preds = pd.concat(all_predictions, ignore_index=True)

    ranked = (
        all_preds.groupby("customer_code")
        .apply(
            lambda df: df.sort_values("engagement_probability", ascending=False)[
                "slot"
            ].unique().tolist()
        )
        .reset_index()
        .rename(columns={0: "predicted_slots_order"})
    )
    ranked["predicted_slots_order"] = ranked["predicted_slots_order"].apply(
        lambda slots: [f"slot_{s}" for s in slots]
    )

    # ── Merge into submission template and save ────────────────────────────
    print("Writing submission file…")
    submission = pd.read_csv(submission_file)
    submission.columns = [c.lower() for c in submission.columns]
    submission = submission.merge(ranked, on="customer_code", how="left")
    submission.to_csv(output_file, index=False)
    print(f"Submission saved to: {output_file}")


if __name__ == "__main__":
    prepare_test_data()
    predict()
