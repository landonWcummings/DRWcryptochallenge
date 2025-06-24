import pandas as pd
import numpy as np
import logging
import optuna
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

# =========================
# Configuration
# =========================
class Config:
    TRAIN_PATH      = "new_train.csv"
    TEST_PATH       = "new_test.csv"
    # read columns
    sample = pd.read_csv(TRAIN_PATH, nrows=0)
    TARGET_COL      = 'label'
    FEATURE_COLS    = [c for c in sample.columns if c != 'label']

    # optuna settings
    N_TRIALS        = 4
    SPLIT_FRACTIONS = [1.0, 0.8, 0.6, 0.45]
    TEST_SIZE       = 0.1
    RANDOM_STATE    = 42

    # base hyperparams ±50%
    XGB_BASE = {
        "colsample_bylevel":   0.4778,
        "colsample_bynode":    0.3628,
        "colsample_bytree":    0.7107,
        "gamma":               1.7095,
        "learning_rate":       0.02213,
        "max_depth":           20,
        "max_leaves":          194,
        "min_child_weight":    0.0734122046534916,
        "n_estimators":        1006,
        "subsample":           0.6891921395606442,
        "reg_alpha":           39.3524,
        "reg_lambda":          75.4484,
    }
    LGB_BASE = {
        "colsample_bytree":    0.8859795017649658,
        "learning_rate":       0.028397302540593607,
        "min_child_samples":   20,
        "min_child_weight":    20,
        "n_estimators":        2484,
        "num_leaves":          16,
        "reg_alpha":           19.2447,
        "reg_lambda":          55.5046,
        "subsample":           0.08914129080676321,
        "max_depth":           20,
    }

    # fixed params
    XGB_FIXED = {
        "tree_method":         "gpu_hist",
        "predictor":           "gpu_predictor",
        "verbosity":           0,
        "random_state":        RANDOM_STATE,
        "n_jobs":              -1,
    }
    LGB_FIXED = {
        "boosting_type":       "gbdt",
        "device":              "gpu",
        "verbosity":           -1,
        "random_state":        RANDOM_STATE,
        "n_jobs":              -1,
    }

    EARLY_STOP_ROUNDS = 3

# =========================
# Logger
# =========================
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# =========================
# Data Loading
# =========================
df = pd.read_csv(Config.TRAIN_PATH)
X_full = df[Config.FEATURE_COLS].values
y_full = df[Config.TARGET_COL].values

# =========================
# Utility: yield splits
# =========================
def make_splits(X, y, frac, n_splits, seed_offset=0):
    start = int((1-frac) * len(X))
    X_slice, y_slice = X[start:], y[start:]
    for i in range(n_splits):
        rs = Config.RANDOM_STATE + seed_offset + i
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_slice, y_slice,
            test_size=Config.TEST_SIZE,
            random_state=rs
        )
        yield X_tr, X_val, y_tr, y_val

# =========================
# Optuna objectives
# =========================
def objective_xgb(trial):
    # 1) sample around your base values
    params = {}
    for k, base in Config.XGB_BASE.items():
        low, high = 0.5 * base, 1.5 * base
        if isinstance(base, (int, np.integer)):
            params[k] = trial.suggest_int(k, int(low), int(high))
        else:
            params[k] = trial.suggest_float(k, low, high)
    # clamp any 0-1 params
    for k, base in Config.XGB_BASE.items():
        if isinstance(base, float) and 0 <= base <= 1:
            params[k] = max(0.0, min(1.0, params[k]))

    # add fixed params & early stopping
    params.update(Config.XGB_FIXED)
    params["early_stopping_rounds"] = Config.EARLY_STOP_ROUNDS
    params["eval_metric"] = "rmse"

    total_loss = 0.0
    for fi, frac in enumerate(Config.SPLIT_FRACTIONS):
        for X_tr, X_val, y_tr, y_val in make_splits(X_full, y_full, frac, 1, seed_offset=fi*10):
            model = XGBRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            total_loss += model.best_score
    return total_loss


def objective_lgb(trial):
    params = {}
    for k, base in Config.LGB_BASE.items():
        low, high = 0.5 * base, 1.5 * base
        if isinstance(base, (int, np.integer)):
            params[k] = trial.suggest_int(k, int(low), int(high))
        else:
            params[k] = trial.suggest_float(k, low, high)
    # clamp any 0-1 parameters to [0,1]
    for k, base in Config.LGB_BASE.items():
        if isinstance(base, float) and 0 <= base <= 1:
            params[k] = max(0.0, min(1.0, params[k]))
    params.update(Config.LGB_FIXED)
    params["early_stopping_rounds"] = Config.EARLY_STOP_ROUNDS

    total_loss = 0.0
    for fi, frac in enumerate(Config.SPLIT_FRACTIONS):
        for X_tr, X_val, y_tr, y_val in make_splits(X_full, y_full, frac, 1, seed_offset=fi*10):
            model = LGBMRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="rmse")
            total_loss += model.best_score_["valid_0"]["rmse"]
    return total_loss

# =========================
# Main
# =========================
if __name__ == "__main__":
    # Tune XGB
    study_xgb = optuna.create_study(direction="minimize")
    study_xgb.enqueue_trial({
        "colsample_bytree":    0.6523896204219245,
        "learning_rate":        0.01825851493012502,
        "min_child_weight":     0.0734122046534916,
        "n_estimators":         1006,
        "max_leaves":           194,
        "reg_alpha":            21.38050031937534,
        "reg_lambda":           55.32830737909552,
        "subsample":            0.6891921395606442,
        "max_depth":            12,
    })
    study_xgb.optimize(objective_xgb, n_trials=Config.N_TRIALS)
    top2_xgb = sorted(study_xgb.trials, key=lambda t: t.value)[:2]

    # Tune LGBM
    study_lgb = optuna.create_study(direction="minimize")
    study_lgb.enqueue_trial({
        "colsample_bytree":     0.8859795017649658,
        "learning_rate":        0.028397302540593607,
        "min_child_weight":     20,
        "n_estimators":         2484,
        "num_leaves":           16,
        "subsample":            0.08914129080676321,
        "max_depth":            20,
    })
    study_lgb.optimize(objective_lgb, n_trials=Config.N_TRIALS)
    top2_lgb = sorted(study_lgb.trials, key=lambda t: t.value)[:2]

    # Log top trials
    logging.info("Top 2 XGB trials:")
    for t in top2_xgb:
        logging.info(f"Value {t.value:.4f} Params {t.params}")
    logging.info("Top 2 LGBM trials:")
    for t in top2_lgb:
        logging.info(f"Value {t.value:.4f} Params {t.params}")

    # Retrain best models on full data
    xgb_models, lgb_models = [], []
    for t in top2_xgb:
        p = t.params.copy(); p.update(Config.XGB_FIXED)
        xgb_models.append(XGBRegressor(**p).fit(X_full, y_full))
    for t in top2_lgb:
        p = t.params.copy(); p.update(Config.LGB_FIXED)
        lgb_models.append(LGBMRegressor(**p).fit(X_full, y_full))

    # Ensemble & predict
    df_test = pd.read_csv(Config.TEST_PATH)
    X_test = df_test[Config.FEATURE_COLS].values
    preds = [m.predict(X_test) for m in (xgb_models + lgb_models)]
    ensemble = np.mean(preds, axis=0)

    df_sub = pd.read_csv('transformer_submission.csv')
    df_sub['prediction'] = ensemble
    df_sub.to_csv('submission.csv', index=False)
    logging.info("Wrote submission.csv with updated prediction column")
