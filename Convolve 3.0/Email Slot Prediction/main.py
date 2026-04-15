"""
End-to-end pipeline entry point.

Run the full training + inference pipeline sequentially:

    python main.py

Or run individual stages by importing from src/.

Before running:
  - Set DATA_DIR environment variable (or edit configs/config.py directly)
    to point to the folder containing your IIT_ROUND2/ data.
  - Adjust TRAIN_CHUNK_SIZE / CDNA_CHUNK_SIZE in configs/config.py to
    match your available RAM (500_000 for Colab Pro; ~25_000 for free tier).
"""

from src.data_merging import merge_communication_cdna
from src.preprocessing import drop_columns, fix_dtypes
from src.feature_engineering import build_features
from src.train import train
from src.predict import prepare_test_data, predict


def run_pipeline() -> None:
    print("=" * 60)
    print("STAGE 1/6  Merging training communication + CDNA data")
    print("=" * 60)
    merge_communication_cdna()

    print("\n" + "=" * 60)
    print("STAGE 2/6  Dropping unnecessary columns")
    print("=" * 60)
    drop_columns()

    print("\n" + "=" * 60)
    print("STAGE 3/6  Normalising dtypes")
    print("=" * 60)
    fix_dtypes()

    print("\n" + "=" * 60)
    print("STAGE 4/6  Feature engineering")
    print("=" * 60)
    build_features()

    print("\n" + "=" * 60)
    print("STAGE 5/6  Training LightGBM LambdaRank model")
    print("=" * 60)
    train()

    print("\n" + "=" * 60)
    print("STAGE 6/6  Preparing test data & generating predictions")
    print("=" * 60)
    prepare_test_data()
    predict()

    print("\nPipeline complete.")


if __name__ == "__main__":
    run_pipeline()
