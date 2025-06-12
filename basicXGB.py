import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
CSV_PATH      = Path("C:/Users/lndnc/OneDrive/Desktop/AI/DRWkaggle/train.csv")
TEST_CSV_PATH = Path("C:/Users/lndnc/OneDrive/Desktop/AI/DRWkaggle/test.csv")
TARGET_COL    = "label"
TS_COL        = "timestamp"

RNG_SEED  = 42
CV_FOLDS  = 5       # number of sliding CV folds

# Hyperparameters tuned for noisy crypto: focus on correlation
XGB_PARAMS = {
    "tree_method": "hist",
    "device": "gpu",
    "colsample_bylevel": 0.4778,
    "colsample_bynode": 0.3628,
    "colsample_bytree": 0.7107,
    "gamma": 1.7095,
    "learning_rate": 0.02213,
    "max_depth": 20,
    "max_leaves": 12,
    "min_child_weight": 16,
    "n_estimators": 1667,
    "subsample": 0.06567,
    "reg_alpha": 39.3524,
    "reg_lambda": 75.4484,
    "verbosity": 0,
    "random_state": RNG_SEED,
    "n_jobs": -1
}
NUM_ROUNDS_CV   = 100
EARLY_STOP_CV   = 10
NUM_ROUNDS_FULL = 500

# ─── LOGGER SETUP ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

# ─── FEATURE ENGINEERING ────────────────────────────────────────────────────────
import json

FEATURE_JSON_PATH = Path("feature_defs.json")  # JSON file defining feature formulas

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dynamically create features based on JSON definitions.
    JSON maps new_column -> [operand1, operator, operand2].
    operand can be a column name (string) or a numeric constant.
    Supported operators: +, -, *, /.
    """
    df = df.copy()
    # Load feature definitions
    with open(FEATURE_JSON_PATH, 'r') as f:
        feature_defs = json.load(f)

    # Apply each definition
    for new_col, definition in feature_defs.items():
        op1, operator, op2 = definition
        # Determine operand values
        if isinstance(op1, str):
            val1 = df[op1]
        else:
            val1 = op1
        if isinstance(op2, str):
            val2 = df[op2]
        else:
            val2 = op2
        # Compute
        if operator == '+':
            df[new_col] = val1 + val2
        elif operator == '-':
            df[new_col] = val1 - val2
        elif operator == '*':
            df[new_col] = val1 * val2
        elif operator == '/':
            df[new_col] = val1 / (val2 + 1e-8)
        else:
            raise ValueError(f"Unsupported operator: {operator}")
    return df

# ─── UTILITIES ─────────────────────────────────────────────────────────────────
def clean_data(X: pd.DataFrame) -> pd.DataFrame:
    """Replace infinities and NaNs with zeros."""
    return X.replace([np.inf, -np.inf], np.nan).fillna(0)

# ─── SLIDING WINDOW CV ───────────────────────────────────────────────────────────
def sliding_cv(df: pd.DataFrame, feature_cols, target_col):
    n = len(df)
    fold_size = n // (CV_FOLDS + 1)
    cv_rmses = []
    cv_rs = []

    for i in range(1, CV_FOLDS + 1):
        train_end = i * fold_size
        test_start = train_end
        test_end = (i + 1) * fold_size

        train_df = df.iloc[:train_end]
        test_df  = df.iloc[test_start:test_end]

        X_tr = clean_data(train_df[feature_cols])
        y_tr = train_df[target_col]
        X_te = clean_data(test_df[feature_cols])
        y_te = test_df[target_col]

        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        deval  = xgb.DMatrix(X_te, label=y_te)

        bst = xgb.train(
            XGB_PARAMS,
            dtrain,
            num_boost_round=NUM_ROUNDS_CV,
            evals=[(deval, 'eval')],
            early_stopping_rounds=EARLY_STOP_CV,
            verbose_eval=False
        )

        preds = bst.predict(deval)
        rmse = mean_squared_error(y_te, preds, squared=False)
        r_val = pearsonr(y_te, preds)[0] if len(y_te) > 1 else 0.0
        logging.info(f"Fold {i}/{CV_FOLDS}: rows {test_start}-{test_end} | RMSE={rmse:.4f} | Pearson R={r_val:.4f}")

        cv_rmses.append(rmse)
        cv_rs.append(r_val)

    logging.info(f"Sliding CV mean RMSE: {np.mean(cv_rmses):.4f} +- {np.std(cv_rmses):.4f}")
    logging.info(f"Sliding CV mean Pearson R: {np.mean(cv_rs):.4f} +- {np.std(cv_rs):.4f}")

# ─── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    df = pd.read_csv(CSV_PATH)
    df = make_features(df)
    feature_cols = [c for c in df.columns if c not in (TS_COL, TARGET_COL)]

    logging.info("Starting sliding-window cross-validation...")
    sliding_cv(df, feature_cols, TARGET_COL)

    # Final training on entire dataset
    X_full = clean_data(df[feature_cols])
    y_full = df[TARGET_COL]
    dtrain_full = xgb.DMatrix(X_full, label=y_full)

    bst_full = xgb.train(
        XGB_PARAMS,
        dtrain_full,
        num_boost_round=NUM_ROUNDS_FULL,
        verbose_eval=50
    )

    # Inference on hold-out test.csv
    df_sub = pd.read_csv(TEST_CSV_PATH)
    df_sub = make_features(df_sub)
    X_sub  = clean_data(df_sub[feature_cols])
    dsub   = xgb.DMatrix(X_sub)
    sub_preds = bst_full.predict(dsub)

    submission = pd.DataFrame({
        'ID': np.arange(1, len(sub_preds) + 1),
        'prediction': sub_preds
    })
    submission.to_csv('xgb_submission.csv', index=False)
    logging.info("Saved xgb_submission.csv ")

if __name__ == '__main__':
    main()
