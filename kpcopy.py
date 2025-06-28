import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

EARLY_PERCENTAGE = 0.35  # Change this to 0.20, 0.25, 0.30, 0.35, 0.40, or 0.45
# ======================================================

# Imports
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr

# Feature Engineering
def feature_engineering(df):
    df['volume_weighted_sell'] = df['sell_qty'] * df['volume']
    df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-8)
    df['selling_pressure'] = df['sell_qty'] / (df['volume'] + 1e-8)
    df['effective_spread_proxy'] = np.abs(df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-8)
    
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    return df 

# Configuration
class Config:
    TRAIN_PATH = r"C:\Users\lando\Desktop\AI\DRWcryptochallenge\train.csv"
    TEST_PATH = r"C:\Users\lando\Desktop\AI\DRWcryptochallenge\test.csv"
    SUBMISSION_PATH = r"C:\Users\lando\Desktop\AI\DRWcryptochallenge\submission.csv"

    FEATURES = [
        "X863", "X856", "X598", "X862", "X385", "X852", "X603", "X860", "X674",
        "X415", "X345", "X855", "X174", "X302", "X178", "X168", "X612",
        "buy_qty", "sell_qty", "volume", "X888", "X421", "X333", "X292", "X532", 'X344'
    ]

    LABEL_COLUMN = "label"
    N_FOLDS = 9
    RANDOM_STATE = 42

XGB_PARAMS = {
    'tree_method': 'hist', 
    'device': 'gpu',
    'n_jobs': -1,
    'colsample_bytree': 0.4111224922845363, 
    'colsample_bynode': 0.28869302181383194,
    'gamma': 1.4665430311056709, 
    'learning_rate': 0.014053505540364681, 
    'max_depth': 7, 
    'max_leaves': 40, 
    'n_estimators': 500,
    'reg_alpha': 27.791606770656145, 
    'reg_lambda': 84.90603428439086,
    'subsample': 0.06567,
    'verbosity': 0,
    'random_state': Config.RANDOM_STATE
}

LEARNERS = [
    {"name": "xgb", "Estimator": XGBRegressor, "params": XGB_PARAMS},
]

# Loading Data
def create_time_decay_weights(n: int, decay: float = 0.9, reverse: bool = False) -> np.ndarray:
    """Create time decay weights. If reverse=True, older data gets higher weight."""
    positions = np.arange(n)
    if reverse:
        normalized = 1.0 - (positions / (n - 1))
    else:
        normalized = positions / (n - 1)
    weights = decay ** (1.0 - normalized)
    return weights * n / weights.sum()

def load_data():
    train_df = pd.read_csv(
        Config.TRAIN_PATH,
        usecols=Config.FEATURES + [Config.LABEL_COLUMN]
    )
    test_df = pd.read_csv(
        Config.TEST_PATH,
        usecols=Config.FEATURES
    )

    submission_df = pd.read_csv(Config.SUBMISSION_PATH)

    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)
    
    print(f"Loaded data - Train: {train_df.shape}, Test: {test_df.shape}, Submission: {submission_df.shape}")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), submission_df

Config.FEATURES += ["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"]
Config.FEATURES = list(set(Config.FEATURES))  # remove duplicates

# Training and Evaluation
def get_model_slices(n_samples: int):
    return [
        {"name": "full_data", "type": "full", "cutoff": 0},
        {"name": "last_75pct", "type": "recent", "cutoff": int(0.25 * n_samples)},
        {"name": "last_50pct", "type": "recent", "cutoff": int(0.50 * n_samples)},
        {"name": f"first_{int(EARLY_PERCENTAGE*100)}pct", "type": "early", "cutoff": int(EARLY_PERCENTAGE * n_samples)},
    ]

def train_and_evaluate(train_df, test_df):
    n_samples = len(train_df)
    model_slices = get_model_slices(n_samples)

    oof_preds = {
        learner["name"]: {s["name"]: np.zeros(n_samples) for s in model_slices}
        for learner in LEARNERS
    }
    test_preds = {
        learner["name"]: {s["name"]: np.zeros(len(test_df)) for s in model_slices}
        for learner in LEARNERS
    }

    full_weights = create_time_decay_weights(n_samples)
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=False)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df), start=1):
        print(f"\n--- Fold {fold}/{Config.N_FOLDS} ---")
        X_valid = train_df.iloc[valid_idx][Config.FEATURES]
        y_valid = train_df.iloc[valid_idx][Config.LABEL_COLUMN]

        for s in model_slices:
            cutoff = s["cutoff"]
            slice_name = s["name"]
            slice_type = s["type"]
            
            if slice_type == "full":
                # Use all data
                subset = train_df.reset_index(drop=True)
                rel_idx = train_idx
                sw = full_weights[train_idx]
                
            elif slice_type == "recent":
                # Use data from cutoff to end (recent data)
                subset = train_df.iloc[cutoff:].reset_index(drop=True)
                rel_idx = train_idx[train_idx >= cutoff] - cutoff
                if cutoff > 0:
                    sw = create_time_decay_weights(len(subset))[rel_idx]
                else:
                    sw = full_weights[train_idx]
                    
            elif slice_type == "early":
                # Use data from start to cutoff (early data)
                subset = train_df.iloc[:cutoff].reset_index(drop=True)
                rel_idx = train_idx[train_idx < cutoff]
                if len(rel_idx) > 0:
                    # For early data, we might want to give more weight to later samples within the subset
                    sw = create_time_decay_weights(len(subset))[rel_idx]
                else:
                    sw = np.array([])

            # Skip if no training data available for this slice
            if len(rel_idx) == 0:
                print(f"  Skipping slice: {slice_name} (no training data in fold)")
                continue

            X_train = subset.iloc[rel_idx][Config.FEATURES]
            y_train = subset.iloc[rel_idx][Config.LABEL_COLUMN]
            
            X_train_np = X_train.values
            y_train_np = y_train.values
            X_valid_np = X_valid.values
            y_valid_np = y_valid.values
            
            print(f"  Training slice: {slice_name}, samples: {len(X_train)}")

            for learner in LEARNERS:
                model = learner["Estimator"](**learner["params"])
                model.fit(X_train_np, y_train_np, sample_weight=sw, 
                          eval_set=[(X_valid_np, y_valid_np)], verbose=False)
                
                # Handle predictions based on slice type
                if slice_type == "early":
                    # For early slice, only predict on validation samples that were in the training range
                    mask = valid_idx < cutoff
                    if mask.any():
                        idxs = valid_idx[mask]
                        oof_preds[learner["name"]][slice_name][idxs] = model.predict(train_df.iloc[idxs][Config.FEATURES])
                    # For validation samples outside the early training range, use full_data predictions
                    if (~mask).any():
                        oof_preds[learner["name"]][slice_name][valid_idx[~mask]] = oof_preds[learner["name"]]["full_data"][valid_idx[~mask]]
                else:
                    # For recent slices and full data
                    mask = valid_idx >= cutoff if slice_type == "recent" else np.ones(len(valid_idx), dtype=bool)
                    if mask.any():
                        idxs = valid_idx[mask]
                        oof_preds[learner["name"]][slice_name][idxs] = model.predict(train_df.iloc[idxs][Config.FEATURES])
                    if slice_type == "recent" and cutoff > 0 and (~mask).any():
                        oof_preds[learner["name"]][slice_name][valid_idx[~mask]] = oof_preds[learner["name"]]["full_data"][valid_idx[~mask]]

                # Test predictions (always use the model regardless of slice type)
                test_preds[learner["name"]][slice_name] += model.predict(test_df[Config.FEATURES])

    # Normalize test predictions
    for learner_name in test_preds:
        for slice_name in test_preds[learner_name]:
            test_preds[learner_name][slice_name] /= Config.N_FOLDS

    return oof_preds, test_preds, model_slices

# Submission
def ensemble_and_submit(train_df, oof_preds, test_preds, submission_df):
    learner_ensembles = {}
    
    print("\nIndividual Slice Scores:")
    for learner_name in oof_preds:
        scores = {}
        for s in oof_preds[learner_name]:
            # Calculate score only on samples where the model made actual predictions
            # (not filled from other models)
            score = pearsonr(train_df[Config.LABEL_COLUMN], oof_preds[learner_name][s])[0]
            scores[s] = score
            print(f"  {learner_name} - {s}: {score:.4f}")
        
        total_score = sum(scores.values())

        # Simple average ensemble
        oof_simple = np.mean(list(oof_preds[learner_name].values()), axis=0)
        test_simple = np.mean(list(test_preds[learner_name].values()), axis=0)
        score_simple = pearsonr(train_df[Config.LABEL_COLUMN], oof_simple)[0]

        # Weighted ensemble based on OOF scores
        oof_weighted = sum(scores[s] / total_score * oof_preds[learner_name][s] for s in scores)
        test_weighted = sum(scores[s] / total_score * test_preds[learner_name][s] for s in scores)
        score_weighted = pearsonr(train_df[Config.LABEL_COLUMN], oof_weighted)[0]

        print(f"\n{learner_name.upper()} Simple Ensemble Pearson:   {score_simple:.4f}")
        print(f"{learner_name.upper()} Weighted Ensemble Pearson: {score_weighted:.4f}")

        # Store the better performing ensemble
        if score_weighted > score_simple:
            learner_ensembles[learner_name] = {
                "oof": oof_weighted,
                "test": test_weighted,
                "type": "weighted"
            }
        else:
            learner_ensembles[learner_name] = {
                "oof": oof_simple,
                "test": test_simple,
                "type": "simple"
            }

    # Final ensemble across all learners
    final_oof = np.mean([le["oof"] for le in learner_ensembles.values()], axis=0)
    final_test = np.mean([le["test"] for le in learner_ensembles.values()], axis=0)
    final_score = pearsonr(train_df[Config.LABEL_COLUMN], final_oof)[0]

    print(f"\nFINAL ensemble across learners Pearson: {final_score:.4f}")
    print(f"Ensemble types used: {[le['type'] for le in learner_ensembles.values()]}")

    # Save with percentage in filename
    filename = f"submission_early_{int(EARLY_PERCENTAGE*100)}pct.csv"
    submission_df["prediction"] = final_test
    submission_df.to_csv(filename, index=False)
    print(f"\nSaved: {filename}")

# Main
if __name__ == "__main__":
    print(f"\nRunning with EARLY_PERCENTAGE = {EARLY_PERCENTAGE} ({int(EARLY_PERCENTAGE*100)}%)")
    train_df, test_df, submission_df = load_data()
    oof_preds, test_preds, model_slices = train_and_evaluate(train_df, test_df)
    ensemble_and_submit(train_df, oof_preds, test_preds, submission_df)