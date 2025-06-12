import logging
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import xgboost as xgb

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
MAX_SEARCH_DURATION = 3600       # seconds to run feature search (1 hour)
BASELINE_RUNS = 2               # number of times to run baseline sliding CV
# each call to sample_and_test runs one sliding CV (CV_FOLDS folds)
# so SAMPLE_CV_CALLS = number of candidate tests


CSV_PATH       = Path("C:/Users/lndnc/OneDrive/Desktop/AI/DRWkaggle/train.csv")
TEST_CSV_PATH  = Path("C:/Users/lndnc/OneDrive/Desktop/AI/DRWkaggle/test.csv")
TARGET_COL     = "label"
TS_COL         = "timestamp"

RNG_SEED        = 42
CV_FOLDS        = 5
NUM_ROUNDS_CV   = 100
EARLY_STOP_CV   = 10
NUM_ROUNDS_FULL = 500
FEATURE_JSON    = Path("feature_defs.json")

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

FEATURES = []  # will be set to top correlated features

# ─── LOGGER SETUP ───────────────────────────────────────────────────────────────
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

# ─── FEATURE ENGINEERING ────────────────────────────────────────────────────────

def load_feature_defs():
    if FEATURE_JSON.exists():
        return json.loads(FEATURE_JSON.read_text())
    return {}


def save_feature_defs(feature_defs):
    FEATURE_JSON.write_text(json.dumps(feature_defs, indent=2))


def apply_definition(df, name, definition):
    # definition is list: for pair [f1, op, f2]
    # or for triple [f1, op1, f2, op2, f3]
    vals = []
    # build values list
    if len(definition) == 3:
        op1, operator, op2 = definition
        v1 = df[op1] if isinstance(op1, str) else op1
        v2 = df[op2] if isinstance(op2, str) else op2
        if operator == '*': df[name] = v1 * v2
        elif operator == '/': df[name] = v1 / (v2 + 1e-8)
        elif operator == '+': df[name] = v1 + v2
        elif operator == '-': df[name] = v1 - v2
    elif len(definition) == 5:
        # f1 op1 f2 op2 f3
        f1, op1, f2, op2, f3 = definition
        v1 = df[f1] if isinstance(f1, str) else f1
        v2 = df[f2] if isinstance(f2, str) else f2
        v3 = df[f3] if isinstance(f3, str) else f3
        # first combine
        if op1 == '*': temp = v1 * v2
        elif op1 == '/': temp = v1 / (v2 + 1e-8)
        elif op1 == '+': temp = v1 + v2
        elif op1 == '-': temp = v1 - v2
        # then op2
        if op2 == '*': df[name] = temp * v3
        elif op2 == '/': df[name] = temp / (v3 + 1e-8)
        elif op2 == '+': df[name] = temp + v3
        elif op2 == '-': df[name] = temp - v3
    else:
        raise ValueError(f"Invalid definition length for {name}")


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    defs = load_feature_defs()
    for name, definition in defs.items():
        apply_definition(df, name, definition)
    return df

# ─── UTILITIES ─────────────────────────────────────────────────────────────────

def clean_data(X: pd.DataFrame) -> pd.DataFrame:
    return X.replace([np.inf, -np.inf], np.nan).fillna(0)

# ─── SLIDING WINDOW CV ───────────────────────────────────────────────────────────

def sliding_cv(df: pd.DataFrame, feature_cols):
    rmses, maes = [], []
    n = len(df)
    fold = n // (CV_FOLDS + 1)
    for i in range(1, CV_FOLDS + 1):
        tr = df.iloc[:i*fold]; te = df.iloc[i*fold:(i+1)*fold]
        Xtr, ytr = clean_data(tr[feature_cols]), tr[TARGET_COL]
        Xte, yte = clean_data(te[feature_cols]), te[TARGET_COL]
        dtr = xgb.DMatrix(Xtr, label=ytr); dve = xgb.DMatrix(Xte, label=yte)
        bst = xgb.train(
            XGB_PARAMS, dtr,
            num_boost_round=NUM_ROUNDS_CV,
            evals=[(dve, 'eval')],
            early_stopping_rounds=EARLY_STOP_CV,
            verbose_eval=False
        )
        preds = bst.predict(dve)
        rmses.append(root_mean_squared_error(yte, preds))
        maes.append(mean_absolute_error(yte, preds))
    return np.mean(rmses), np.std(rmses), np.mean(maes), np.std(maes)

# ─── FEATURE SEARCH ─────────────────────────────────────────────────────────────

def sample_and_test_pair(df: pd.DataFrame, baseline_mae: float) -> float:
    """
    Sample two new feature definitions at once, evaluate with two seeds, and add both if combined MAE improves.
    """
    # pick two distinct new features to propose
    def propose_feature():
        count = random.random()
        if count < 0.5:
            cols = random.sample(FEATURES, 2)
            op = random.choice(['*','/'])
            return (cols[0], op, cols[1])
        else:
            cols = random.sample(FEATURES, 3)
            ops = [random.choice(['*','/']) for _ in range(2)]
            return (cols[0], ops[0], cols[1], ops[1], cols[2])

    def evaluate_defs(defs):
        maes = []
        for seed in [RNG_SEED, RNG_SEED+1]:
            # reseed for reproducibility
            random.seed(seed)
            np.random.seed(seed)
            df2 = df.copy()
            for nm, defn in defs.items():
                apply_definition(df2, nm, defn)
            cols2 = [c for c in df2.columns if c not in (TS_COL, TARGET_COL)]
            _, _, mean_mae, _ = sliding_cv(df2, cols2)
            maes.append(mean_mae)
        return float(np.mean(maes))

    # propose two new features
    f1 = propose_feature()
    name1 = "_".join(map(str, f1)).replace(' ', '')
    f2 = propose_feature()
    name2 = "_".join(map(str, f2)).replace(' ', '')

    # temp defs include both
    current_defs = load_feature_defs()
    temp_defs = {**current_defs, name1: list(f1), name2: list(f2)}

    mean_mae = evaluate_defs(temp_defs)
    logging.info(f"Tested {name1} and {name2}: combined MAE={mean_mae:.4f}")

    # accept both if improvement
    if mean_mae < baseline_mae :
        logging.info(f"Accepting features {name1}, {name2}")
        current_defs[name1] = list(f1)
        current_defs[name2] = list(f2)
        save_feature_defs(current_defs)
        return mean_mae

    logging.info(f"Rejecting features {name1}, {name2}")
    return baseline_mae

# ─── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    df = pd.read_csv(CSV_PATH)
    raw_feats = [c for c in df.columns if c not in (TS_COL, TARGET_COL)]
    corrs = df[raw_feats].corrwith(df[TARGET_COL]).abs()
    top_feats = corrs.nlargest(70).index.tolist()
    global FEATURES; FEATURES = top_feats
    total_pairs = len(top_feats)*(len(top_feats)-1)//2*2
    total_triples = (len(top_feats)*(len(top_feats)-1)*(len(top_feats)-2)//6)*2
    logging.info(f"Search space: {total_pairs} pairs + {total_triples} triples = {total_pairs+total_triples}")

    df = make_features(df)
    feat_cols = [c for c in df.columns if c not in (TS_COL, TARGET_COL)]

    # Compute baseline MAE over two seeds and multiple runs
    baseline_maes = []
    for run in range(BASELINE_RUNS):
        maes = []
        for seed in [RNG_SEED, RNG_SEED+1]:
            random.seed(seed)
            np.random.seed(seed)
            _, _, mae, _ = sliding_cv(df, feat_cols)
            maes.append(mae)
        avg_mae = float(np.mean(maes))
        baseline_maes.append(avg_mae)
        logging.info(f"Baseline run {run+1}/{BASELINE_RUNS} with seeds {RNG_SEED},{RNG_SEED+1}: MAE={avg_mae:.4f}")
    baseline_mae = float(np.mean(baseline_maes))
    baseline_mae_std = float(np.std(baseline_maes))
    logging.info(f"Baseline MAE over {BASELINE_RUNS} runs: {baseline_mae:.4f} ± {baseline_mae_std:.4f}")

    # Run feature search for a fixed duration
    import time
    sample_count = 0
    start_time = time.time()
    while time.time() - start_time < MAX_SEARCH_DURATION:
        baseline_mae = sample_and_test_pair(df, baseline_mae)
        sample_count += 1
        elapsed = time.time() - start_time
        logging.info(f"Elapsed time: {elapsed/60:.1f} min | Samples tried: {sample_count}")
    logging.info(f"Feature search completed: baseline run {BASELINE_RUNS} time(s), {sample_count} sample tests.")

    # final model training and inference omitted

if __name__ == '__main__': main()