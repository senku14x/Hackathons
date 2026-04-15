"""
Centralized configuration for all file paths, hyperparameters, and constants.
Adjust TRAIN_COMM_FILE, TRAIN_CDNA_FILE, etc. to point to your local/Drive paths.
"""

import os

# ── Data paths ────────────────────────────────────────────────────────────────
BASE_DIR = os.getenv("DATA_DIR", "/content/drive/My Drive")

TRAIN_COMM_FILE   = f"{BASE_DIR}/IIT_ROUND2/Train_r2/train_action_history.csv"
TRAIN_CDNA_FILE   = f"{BASE_DIR}/IIT_ROUND2/Train_r2/train_cdna_data.csv"
TEST_COMM_FILE    = f"{BASE_DIR}/IIT_ROUND2/Test_r2/test_action_history.csv"
TEST_CDNA_FILE    = f"{BASE_DIR}/IIT_ROUND2/Test_r2/test_cdna_data.csv"
SUBMISSION_FILE   = f"{BASE_DIR}/IIT_ROUND2/Test_r2/test_customers.csv"

MERGED_TRAIN_FILE       = f"{BASE_DIR}/merged_data.csv"
CLEANED_TRAIN_FILE      = f"{BASE_DIR}/cleaned_merged_data.csv"
PROCESSED_TRAIN_FILE    = f"{BASE_DIR}/processed_train_data.csv"
FEATURED_TRAIN_FILE     = f"{BASE_DIR}/featured_train_data.csv"
MERGED_TEST_FILE        = f"{BASE_DIR}/test_merged_data.csv"
MODEL_PATH              = f"{BASE_DIR}/final_lightgbm_model.txt"
SUBMISSION_OUTPUT_FILE  = f"{BASE_DIR}/submission_final.csv"

# ── Chunk sizes ───────────────────────────────────────────────────────────────
# Reduce to ~25_000 on free Colab; 500_000 is safe on Colab Pro.
TRAIN_CHUNK_SIZE = 500_000
CDNA_CHUNK_SIZE  = 500_000
CLEAN_CHUNK_SIZE = 1_500_000
TRAIN_MODEL_CHUNK_SIZE = 500_000
TEST_PRED_CHUNK_SIZE   = 100_000

# ── Feature schema ────────────────────────────────────────────────────────────
COLUMNS_TO_KEEP = [
    "customer_code", "offer_id", "offer_subid",
    "product_category", "product_sub_category",
    "batch_id", "send_timestamp", "open_timestamp",
    "batch_date", "v6", "acc_date",
]

STRING_COLUMNS   = ["customer_code", "offer_id", "offer_subid",
                    "product_category", "product_sub_category", "v6"]
DATETIME_COLUMNS = ["send_timestamp", "open_timestamp", "batch_date", "acc_date"]
NUMERIC_COLUMNS  = ["batch_id"]

CATEGORICAL_COLUMNS = ["offer_id", "offer_subid", "product_category",
                       "product_sub_category", "v6"]

N_SLOTS = 28
SLOT_ENGAGEMENT_COLS = [f"slot_{i}_engagements" for i in range(1, N_SLOTS + 1)]

FINAL_FEATURES = (
    ["offer_id", "offer_subid", "product_category", "product_sub_category",
     "batch_id", "batch_date", "send_time_slot", "open_time_slot", "acc_date", "v6"]
    + SLOT_ENGAGEMENT_COLS
)

# ── LightGBM hyperparameters ──────────────────────────────────────────────────
LGBM_PARAMS = {
    "objective":        "lambdarank",
    "metric":           "map",
    "boosting_type":    "gbdt",
    "device":           "cpu",
    "learning_rate":    0.03,
    "num_leaves":       10,
    "max_depth":        5,
    "lambda_l1":        2.0,
    "lambda_l2":        2.0,
    "min_child_samples": 150,
    "seed":             42,
}

N_SPLITS         = 10
NUM_BOOST_ROUND  = 1000
EARLY_STOPPING   = 50
LOG_PERIOD       = 50

# ── Leakage detection ─────────────────────────────────────────────────────────
LEAKAGE_CORR_THRESHOLD = 0.9
