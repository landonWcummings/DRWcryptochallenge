import pandas as pd
import numpy as np
import optuna
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

# =========================
# Config
# =========================
class Config:
    TRAIN_CSV = "new_train.csv"
    TARGET_COLS = [c for c in pd.read_csv(TRAIN_CSV, nrows=0).columns if c.startswith(("1X","2X"))]
    FEATURE_COLS = [c for c in pd.read_csv(TRAIN_CSV, nrows=0).columns if c not in TARGET_COLS]
    RANDOM_STATE = 42
    N_SPLITS = 2
    RECENT_FRACTION = 0.65
    TEST_SIZE = 0.1
    N_EPOCHS = 20
    PATIENCE = 3

    # base params to perturb ±50%
    XGB_BASE = {
        "colsample_bylevel": 0.4778,
        "colsample_bynode":  0.3628,
        "colsample_bytree":  0.7107,
        "gamma":             1.7095,
        "learning_rate":     0.02213,
        "max_depth":         20,
        "max_leaves":        12,
        "min_child_weight":  16,
        "n_estimators":      1667,
        "subsample":         0.06567,
        "reg_alpha":         39.3524,
        "reg_lambda":        75.4484,
    }

    LGB_BASE = {
        "colsample_bytree": 0.5039,
        "learning_rate":    0.01260,
        "min_child_samples": 20,
        "min_child_weight": 0.1146,
        "n_estimators":     915,
        "num_leaves":       145,
        "reg_alpha":        19.2447,
        "reg_lambda":       55.5046,
        "subsample":        0.9709,
        "max_depth":        9,
    }

# =========================
# Data prep
# =========================
df = pd.read_csv(Config.TRAIN_CSV)
X_full = df[Config.FEATURE_COLS].values
y_full = df[Config.TARGET_COLS].values
n = len(df)
cut = int((1 - Config.RECENT_FRACTION) * n)
X_recent, y_recent = X_full[cut:], y_full[cut:]

def make_splits(X, y, n_splits, seed_offset=0):
    """Yield (X_tr, X_val, y_tr, y_val) for n_splits random 90/10 splits."""
    for i in range(n_splits):
        rs = Config.RANDOM_STATE + seed_offset + i
        yield train_test_split(X, y,
                               test_size=Config.TEST_SIZE,
                               random_state=rs)

# =========================
# XGB objective
# =========================
def objective_xgb(trial):
    # sample each param ±50%, clamped if [0,1]
    p = {}
    for k, base in Config.XGB_BASE.items():
        low, high = 0.5 * base, 1.5 * base
        if 0.0 <= base <= 1.0:
            low, high = max(0., low), min(1., high)
        if isinstance(base, int):
            p[k] = trial.suggest_int(k, int(low), int(high))
        else:
            p[k] = trial.suggest_float(k, low, high)

    p.update({
        "tree_method": "gpu_hist",
        "predictor":   "gpu_predictor",
        "verbosity":    0,
        "random_state": Config.RANDOM_STATE,
        "n_jobs":      -1,
        "early_stopping_rounds": Config.PATIENCE
    })

    total_loss = 0.0

    # full data splits
    for X_tr, X_val, y_tr, y_val in make_splits(X_full, y_full, Config.N_SPLITS, seed_offset=0):
        model = XGBRegressor(**p)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  eval_metric="rmse",
                  verbose=False)
        total_loss += model.best_score

    # recent data splits
    for X_tr, X_val, y_tr, y_val in make_splits(X_recent, y_recent, Config.N_SPLITS, seed_offset=100):
        model = XGBRegressor(**p)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  eval_metric="rmse",
                  verbose=False)
        total_loss += model.best_score

    return total_loss

# =========================
# LGB objective
# =========================
def objective_lgb(trial):
    # same ±50% logic
    p = {}
    for k, base in Config.LGB_BASE.items():
        low, high = 0.5 * base, 1.5 * base
        if 0.0 <= base <= 1.0:
            low, high = max(0., low), min(1., high)
        if isinstance(base, int):
            p[k] = trial.suggest_int(k, int(low), int(high))
        else:
            p[k] = trial.suggest_float(k, low, high)

    p.update({
        "boosting_type": "gbdt",
        "device":        "gpu",
        "verbosity":    -1,
        "random_state": Config.RANDOM_STATE,
        "n_jobs":       -1,
        "early_stopping_rounds": Config.PATIENCE
    })

    total_loss = 0.0

    for X_tr, X_val, y_tr, y_val in make_splits(X_full, y_full, Config.N_SPLITS, seed_offset=0):
        model = LGBMRegressor(**p)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  eval_metric="rmse",
                  verbose=False)
        total_loss += model.best_score_["valid_0"]["rmse"]

    for X_tr, X_val, y_tr, y_val in make_splits(X_recent, y_recent, Config.N_SPLITS, seed_offset=100):
        model = LGBMRegressor(**p)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  eval_metric="rmse",
                  verbose=False)
        total_loss += model.best_score_["valid_0"]["rmse"]

    return total_loss

# =========================
# Run studies
# =========================
if __name__ == "__main__":
    # XGB tuning
    study_xgb = optuna.create_study(direction="minimize")
    study_xgb.optimize(objective_xgb, n_trials=50)
    print("Best XGB params:", study_xgb.best_params)

    # LGB tuning
    study_lgb = optuna.create_study(direction="minimize")
    study_lgb.optimize(objective_lgb, n_trials=50)
    print("Best LGB params:", study_lgb.best_params)
