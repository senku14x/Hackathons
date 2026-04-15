"""
Step 2 & 3 – Column selection, dtype normalisation.

Two passes over the merged CSV:
  - Pass A (drop_columns)  : keep only the columns needed downstream.
  - Pass B (fix_dtypes)    : cast each column to its correct dtype so that
                             LightGBM and downstream code receive consistent input.
"""

import pandas as pd

from configs.config import (
    MERGED_TRAIN_FILE, CLEANED_TRAIN_FILE, PROCESSED_TRAIN_FILE,
    CLEAN_CHUNK_SIZE,
    COLUMNS_TO_KEEP, STRING_COLUMNS, NUMERIC_COLUMNS, DATETIME_COLUMNS,
)


# ── Pass A: column selection ──────────────────────────────────────────────────

def drop_columns(
    input_file: str = MERGED_TRAIN_FILE,
    output_file: str = CLEANED_TRAIN_FILE,
    chunksize: int = CLEAN_CHUNK_SIZE,
) -> None:
    """
    Keep only COLUMNS_TO_KEEP (renaming v11 → acc_date in the process).

    Parameters
    ----------
    input_file  : Merged CSV produced by data_merging.py.
    output_file : Cleaned CSV with only the required columns.
    chunksize   : Rows per chunk (tune to available RAM).
    """
    header_written = False

    for chunk in pd.read_csv(input_file, chunksize=chunksize):
        chunk.rename(columns={"v11": "acc_date"}, inplace=True)
        # Keep only columns that actually exist in this chunk
        cols = [c for c in COLUMNS_TO_KEEP if c in chunk.columns]
        chunk = chunk[cols]

        chunk.to_csv(output_file, mode="a", index=False, header=not header_written)
        header_written = True

    print(f"Cleaned dataset saved to: {output_file}")


# ── Pass B: dtype normalisation ───────────────────────────────────────────────

def fix_dtypes(
    input_file: str = CLEANED_TRAIN_FILE,
    output_file: str = PROCESSED_TRAIN_FILE,
) -> None:
    """
    Load the cleaned CSV once, cast each column to its declared dtype, and save.

    Note: this step loads the entire cleaned file.  If memory is tight, you can
    incorporate these casts inside drop_columns above (per chunk).

    Parameters
    ----------
    input_file  : Cleaned CSV from drop_columns.
    output_file : Dtype-normalised CSV ready for feature engineering.
    """
    data = pd.read_csv(input_file)

    for col in STRING_COLUMNS:
        if col in data.columns:
            data[col] = data[col].astype(str)

    for col in NUMERIC_COLUMNS:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    for col in DATETIME_COLUMNS:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors="coerce")

    data.to_csv(output_file, index=False)
    print(f"Dtype-normalised dataset saved to: {output_file}")


if __name__ == "__main__":
    drop_columns()
    fix_dtypes()
