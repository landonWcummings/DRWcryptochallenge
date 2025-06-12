import pandas as pd
import numpy as np
import logging
import os

from autogluon.tabular import TabularPredictor
from base_utils import (
    clean_and_scale,
    load_and_split_random  # still available if needed elsewhere
)

# Setup logging
typ = logging.getLogger('autogluon')
typ.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# --- Configuration ---
CSV_PATH      = r'C:\Users\lando\Desktop\AI\DRWcryptochallenge\train.csv'
TEST_CSV_PATH = r'C:\Users\lando\Desktop\AI\DRWcryptochallenge\test.csv'
TS_COL        = 'timestamp'
TARGET_COL    = 'label'
SAVE_DIR      = 'ag_models'
VALID_FRAC    = 0.20  # last 20% for validation
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Main Pipeline ---
def main():
    # --- Load and sort data by time ---
    df = pd.read_csv(CSV_PATH)
    df = df.sort_values(TS_COL).reset_index(drop=True)

    # --- Train/Validation Split (last 20%) ---
    split_idx = int(len(df) * (1 - VALID_FRAC))
    df_train = df.iloc[:split_idx].copy().reset_index(drop=True)
    df_val   = df.iloc[split_idx:].copy().reset_index(drop=True)

    # --- Feature Columns ---
    feature_cols = [c for c in df.columns if c not in [TS_COL, TARGET_COL]]

    # --- Clean and Scale Features ---
    X_tr, X_val, scaler = clean_and_scale(df_train, df_val, feature_cols)
    df_tr = pd.DataFrame(X_tr, columns=feature_cols)
    df_tr[TARGET_COL] = df_train[TARGET_COL].values
    df_val_scaled = pd.DataFrame(X_val, columns=feature_cols)
    df_val_scaled[TARGET_COL] = df_val[TARGET_COL].values

    # --- AutoGluon Training ---
    predictor = TabularPredictor(
        label=TARGET_COL,
        eval_metric='pearsonr',
        path=SAVE_DIR
    ).fit(
        train_data=df_tr,
        presets='best_quality',
        time_limit=3600*8
    )

    # --- Validation Leaderboard ---
    lb = predictor.leaderboard(df_val_scaled, silent=True)
    logging.info("Validation Leaderboard:\n%s", lb)

    # --- Inference on Full Test CSV ---
    test_df = pd.read_csv(TEST_CSV_PATH)
    # apply same scaling
    X_test = scaler.transform(
        test_df[feature_cols]
               .replace([np.inf, -np.inf], np.nan)
               .fillna(0)
    )
    df_test_scaled = pd.DataFrame(X_test, columns=feature_cols)

    # generate predictions
    preds = predictor.predict(df_test_scaled)

    # write submission
    submission = pd.DataFrame({
        'ID': np.arange(1, len(preds) + 1),
        'prediction': preds
    })
    submission.to_csv('AG_submission.csv', index=False)
    logging.info("Saved AG_submission.csv with AutoGluon predictions")

if __name__ == '__main__':
    main()
