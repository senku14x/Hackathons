"""
Step 4 – Feature engineering and target variable creation.

Converts raw timestamps into 28 weekly 3-hour time-slots, builds the
binary relevance target, and aggregates per-customer historical slot
engagement counts (slot_X_engagements).

Slot numbering
--------------
Each day has 4 business slots (09-12, 12-15, 15-18, 18-21).
Emails sent outside these windows are labelled "No Engagement".

    Monday  slots  1 –  4
    Tuesday slots  5 –  8
    ...
    Sunday  slots 25 – 28
"""

import numpy as np
import pandas as pd

from configs.config import (
    PROCESSED_TRAIN_FILE, FEATURED_TRAIN_FILE, N_SLOTS,
)


def get_weekly_slot(timestamp) -> int | str:
    """
    Map a timestamp to one of 28 weekly slots (int) or "No Engagement".

    Parameters
    ----------
    timestamp : datetime-like or NaT

    Returns
    -------
    int in [1, 28] or the string "No Engagement".
    """
    if pd.isna(timestamp):
        return "No Engagement"

    day_of_week = timestamp.dayofweek  # 0 = Monday, 6 = Sunday
    hour = timestamp.hour

    if 9 <= hour < 12:
        intra_day_slot = 1
    elif 12 <= hour < 15:
        intra_day_slot = 2
    elif 15 <= hour < 18:
        intra_day_slot = 3
    elif 18 <= hour < 21:
        intra_day_slot = 4
    else:
        return "No Engagement"

    return day_of_week * 4 + intra_day_slot  # 1-28


def build_features(
    input_file: str = PROCESSED_TRAIN_FILE,
    output_file: str = FEATURED_TRAIN_FILE,
) -> None:
    """
    Add time-slot columns, the relevance target, and historical engagement
    counts, then save the enriched dataset.

    Parameters
    ----------
    input_file  : Dtype-normalised CSV from preprocessing.py.
    output_file : Feature-enriched CSV ready for model training.
    """
    data = pd.read_csv(input_file)

    # ── Timestamps ────────────────────────────────────────────────────────────
    data["send_timestamp"] = pd.to_datetime(data["send_timestamp"], errors="coerce")
    data["open_timestamp"] = pd.to_datetime(data["open_timestamp"], errors="coerce")

    # ── Weekly slots ──────────────────────────────────────────────────────────
    data["send_time_slot"] = data["send_timestamp"].apply(get_weekly_slot)
    data["open_time_slot"] = data["open_timestamp"].apply(get_weekly_slot)

    data["send_time_slot"] = (
        data["send_time_slot"].replace("No Engagement", np.nan)
    )
    data["open_time_slot"] = (
        data["open_time_slot"].replace("No Engagement", np.nan)
    )
    data["send_time_slot"] = pd.to_numeric(data["send_time_slot"], errors="coerce")
    data["open_time_slot"] = pd.to_numeric(data["open_time_slot"], errors="coerce")

    # ── Target variable ───────────────────────────────────────────────────────
    # 1 if the email was opened in the same slot it was sent, else 0.
    data["same_slot_engagement"] = (
        data["send_time_slot"] == data["open_time_slot"]
    ).astype(int)

    # ── Historical engagement counts per slot ─────────────────────────────────
    historical = (
        data.groupby(["customer_code", "open_time_slot"])
        .size()
        .unstack(fill_value=0)
    )
    historical.columns = [
        f"slot_{int(col)}_engagements" for col in historical.columns
    ]
    # Ensure all 28 slot columns exist even if some slots had zero engagement.
    for slot in range(1, N_SLOTS + 1):
        col = f"slot_{slot}_engagements"
        if col not in historical.columns:
            historical[col] = 0

    data = data.merge(historical, how="left", left_on="customer_code", right_index=True)

    data.to_csv(output_file, index=False)
    print(f"Feature-enriched dataset saved to: {output_file}")


if __name__ == "__main__":
    build_features()
