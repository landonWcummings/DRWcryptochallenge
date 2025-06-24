import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import (
    HuberRegressor, RANSACRegressor, TheilSenRegressor,
    Lasso, ElasticNet, Ridge
)
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
import warnings
import numpy as np # linear algebra
import pandas as pd
warnings.filterwarnings('ignore')

# ===== Feature Engineering =====
def feature_engineering(df):
    """Original features plus new robust features"""
    # Original features
    df['volume_weighted_sell'] = df['sell_qty'] * df['volume']
    df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-8)
    df['selling_pressure'] = df['sell_qty'] / (df['volume'] + 1e-8)
    df['effective_spread_proxy'] = np.abs(df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-8)
    
    # New robust features
    df['log_volume'] = np.log1p(df['volume'])
    df['bid_ask_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'] + 1e-8)
    df['order_flow_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + 1e-8)
    df['liquidity_ratio'] = (df['bid_qty'] + df['ask_qty']) / (df['volume'] + 1e-8)
    
    # Handle infinities and NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # For each column, replace NaN with median for robustness
    for col in df.columns:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0)
    
    return df

# ===== Configuration =====
class Config:
    TRAIN_PATH = r"C:\Users\lndnc\OneDrive\Desktop\AI\DRWkaggle\train.csv"
    TEST_PATH = r"C:\Users\lndnc\OneDrive\Desktop\AI\DRWkaggle\test.csv"
    SUBMISSION_PATH = r"C:\Users\lndnc\OneDrive\Desktop\AI\DRWkaggle\submission11.99.csv"
    
    # Original features plus additional market features
    FEATURES = [
        "X863", "X856", "X598", "X862", "X385", "X852", "X603", "X860", "X674",
        "X415", "X345", "X855", "X174", "X302", "X178", "X168", "X612",
        "buy_qty", "sell_qty", "volume", "X888", "X421", "X333",
        "bid_qty", "ask_qty"
    ]
    
    LABEL_COLUMN = "label"
    N_FOLDS = 3
    RANDOM_STATE = 42

# ===== Model Parameters =====
# Original XGBoost parameters
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
    "random_state": Config.RANDOM_STATE,
    "n_jobs": -1
}

# LightGBM parameters (simpler model for noisy data)
LGBM_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 10,
    "reg_lambda": 10,
    "random_state": Config.RANDOM_STATE,
    "device": "gpu",
    "verbosity": -1,
    "n_jobs": -1
}

# Define all learners
LEARNERS = [
    {"name": "xgb_baseline", "Estimator": XGBRegressor, "params": XGB_PARAMS, "need_scale": False},
    {"name": "lgbm", "Estimator": LGBMRegressor, "params": LGBM_PARAMS, "need_scale": False},
    {"name": "huber", "Estimator": HuberRegressor, "params": {"epsilon": 1.5, "alpha": 0.01, "max_iter": 500}, "need_scale": True},
    {"name": "ransac", "Estimator": RANSACRegressor, "params": {"min_samples": 0.7, "max_trials": 100, "random_state": Config.RANDOM_STATE}, "need_scale": True},
    {"name": "theilsen", "Estimator": TheilSenRegressor, "params": {"max_subpopulation": 10000, "random_state": Config.RANDOM_STATE}, "need_scale": True},
    {"name": "lasso", "Estimator": Lasso, "params": {"alpha": 0.001, "max_iter": 1000}, "need_scale": True},
    {"name": "elasticnet", "Estimator": ElasticNet, "params": {"alpha": 0.001, "l1_ratio": 0.5, "max_iter": 1000}, "need_scale": True},
    {"name": "pls", "Estimator": PLSRegression, "params": {"n_components": 33}, "need_scale": True},
]

# ===== Data Loading =====
def create_time_decay_weights(n: int, decay: float = 0.9) -> np.ndarray:
    """Create time decay weights for more recent data importance"""
    positions = np.arange(n)
    normalized = positions / (n - 1)
    weights = decay ** (1.0 - normalized)
    return weights * n / weights.sum()

def load_data():
    """Load and preprocess data"""
    train_df = pd.read_csv(
        Config.TRAIN_PATH,
        usecols=Config.FEATURES + [Config.LABEL_COLUMN]
    )
    test_df = pd.read_csv(
        Config.TEST_PATH,
        usecols=Config.FEATURES
    )

    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    
    # Apply feature engineering
    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)
    
    # Update features list with engineered features
    engineered_features = [
        "volume_weighted_sell", "buy_sell_ratio", "selling_pressure", 
        "effective_spread_proxy", "log_volume", "bid_ask_imbalance",
        "order_flow_imbalance", "liquidity_ratio"
    ]
    Config.FEATURES = list(set(Config.FEATURES + engineered_features))
    
    print(f"Loaded data - Train: {train_df.shape}, Test: {test_df.shape}, Submission: {submission_df.shape}")
    print(f"Total features: {len(Config.FEATURES)}")
    
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), submission_df

# ===== Model Training =====
def get_model_slices(n_samples: int):
    """Define different data slices for training"""
    return [
        {"name": "full_data", "cutoff": 0},
        {"name": "last_75pct", "cutoff": int(0.25 * n_samples)},
        {"name": "last_50pct", "cutoff": int(0.50 * n_samples)},
    ]

def train_single_model(X_train, y_train, X_valid, y_valid, X_test, learner, sample_weights=None):
    """Train a single model with appropriate scaling if needed"""
    if learner["need_scale"]:
        scaler = RobustScaler()  # More robust to outliers than StandardScaler
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_valid_scaled = X_valid
        X_test_scaled = X_test
    
    model = learner["Estimator"](**learner["params"])
    
    # Handle different model training approaches
    if learner["name"] in ["xgb_baseline", "lgbm"]:
        if learner["name"] == "xgb_baseline":
            model.fit(X_train_scaled, y_train, sample_weight=sample_weights, 
                     eval_set=[(X_valid_scaled, y_valid)], verbose=False)
        else:  # LightGBM
            model.fit(X_train_scaled, y_train, sample_weight=sample_weights,
                     eval_set=[(X_valid_scaled, y_valid)], callbacks=[])
    elif learner["name"] in ["huber", "lasso", "elasticnet"]:
        model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    else:
        # RANSAC, TheilSen, PLS don't support sample weights
        model.fit(X_train_scaled, y_train)
    
    valid_pred = model.predict(X_valid_scaled)
    test_pred = model.predict(X_test_scaled)
    
    return valid_pred, test_pred

def train_and_evaluate(train_df, test_df):
    """Train all models with cross-validation"""
    n_samples = len(train_df)
    model_slices = get_model_slices(n_samples)
    
    # Initialize prediction dictionaries
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
        X_test = test_df[Config.FEATURES]
        
        for s in model_slices:
            cutoff = s["cutoff"]
            slice_name = s["name"]
            subset = train_df.iloc[cutoff:].reset_index(drop=True)
            rel_idx = train_idx[train_idx >= cutoff] - cutoff
            
            if len(rel_idx) == 0:
                continue
                
            X_train = subset.iloc[rel_idx][Config.FEATURES]
            y_train = subset.iloc[rel_idx][Config.LABEL_COLUMN]
            sw = create_time_decay_weights(len(subset))[rel_idx] if cutoff > 0 else full_weights[train_idx]
            
            print(f"  Training slice: {slice_name}, samples: {len(X_train)}")
            
            for learner in LEARNERS:
                try:
                    valid_pred, test_pred = train_single_model(
                        X_train, y_train, X_valid, y_valid, X_test, learner, sw
                    )
                    
                    # Store OOF predictions
                    mask = valid_idx >= cutoff
                    if mask.any():
                        idxs = valid_idx[mask]
                        X_valid_subset = train_df.iloc[idxs][Config.FEATURES]
                        if learner["need_scale"]:
                            scaler = RobustScaler()
                            scaler.fit(X_train)
                            valid_pred_subset = learner["Estimator"](**learner["params"]).fit(
                                scaler.transform(X_train), y_train
                            ).predict(scaler.transform(X_valid_subset))
                            oof_preds[learner["name"]][slice_name][idxs] = valid_pred_subset
                        else:
                            oof_preds[learner["name"]][slice_name][idxs] = valid_pred[mask]
                    
                    if cutoff > 0 and (~mask).any():
                        oof_preds[learner["name"]][slice_name][valid_idx[~mask]] = \
                            oof_preds[learner["name"]]["full_data"][valid_idx[~mask]]
                    
                    test_preds[learner["name"]][slice_name] += test_pred
                    
                except Exception as e:
                    print(f"    Error training {learner['name']}: {str(e)}")
                    continue
    
    # Normalize test predictions
    for learner_name in test_preds:
        for slice_name in test_preds[learner_name]:
            test_preds[learner_name][slice_name] /= Config.N_FOLDS
    
    return oof_preds, test_preds, model_slices

# ===== Ensemble and Submission =====
def create_submissions(train_df, oof_preds, test_preds, submission_df):
    """Create multiple submission files for different strategies"""
    all_submissions = {}
    
    # 1. Original baseline (XGBoost only)
    if "xgb_baseline" in oof_preds:
        xgb_oof = np.mean(list(oof_preds["xgb_baseline"].values()), axis=0)
        xgb_test = np.mean(list(test_preds["xgb_baseline"].values()), axis=0)
        xgb_score = pearsonr(train_df[Config.LABEL_COLUMN], xgb_oof)[0]
        print(f"\nXGBoost Baseline Score: {xgb_score:.4f}")
        
        submission_xgb = submission_df.copy()
        submission_xgb["prediction"] = xgb_test
        submission_xgb.to_csv("submission_xgb_baseline.csv", index=False)
        all_submissions["xgb_baseline"] = xgb_score
    
    # 2. Robust methods ensemble
    robust_methods = ["huber", "ransac", "theilsen"]
    robust_oof_list = []
    robust_test_list = []
    
    for method in robust_methods:
        if method in oof_preds:
            method_oof = np.mean(list(oof_preds[method].values()), axis=0)
            method_test = np.mean(list(test_preds[method].values()), axis=0)
            method_score = pearsonr(train_df[Config.LABEL_COLUMN], method_oof)[0]
            print(f"{method.upper()} Score: {method_score:.4f}")
            
            if not np.isnan(method_score):
                robust_oof_list.append(method_oof)
                robust_test_list.append(method_test)
    
    if robust_oof_list:
        robust_oof = np.mean(robust_oof_list, axis=0)
        robust_test = np.mean(robust_test_list, axis=0)
        robust_score = pearsonr(train_df[Config.LABEL_COLUMN], robust_oof)[0]
        print(f"\nRobust Ensemble Score: {robust_score:.4f}")
        
        submission_robust = submission_df.copy()
        submission_robust["prediction"] = robust_test
        submission_robust.to_csv("submission_robust_ensemble.csv", index=False)
        all_submissions["robust_ensemble"] = robust_score
    
    # 3. Regularized methods ensemble
    regularized_methods = ["lasso", "elasticnet"]
    reg_oof_list = []
    reg_test_list = []
    
    for method in regularized_methods:
        if method in oof_preds:
            method_oof = np.mean(list(oof_preds[method].values()), axis=0)
            method_test = np.mean(list(test_preds[method].values()), axis=0)
            method_score = pearsonr(train_df[Config.LABEL_COLUMN], method_oof)[0]
            print(f"{method.upper()} Score: {method_score:.4f}")
            
            if not np.isnan(method_score):
                reg_oof_list.append(method_oof)
                reg_test_list.append(method_test)
    
    if reg_oof_list:
        reg_oof = np.mean(reg_oof_list, axis=0)
        reg_test = np.mean(reg_test_list, axis=0)
        reg_score = pearsonr(train_df[Config.LABEL_COLUMN], reg_oof)[0]
        print(f"\nRegularized Ensemble Score: {reg_score:.4f}")
        
        submission_reg = submission_df.copy()
        submission_reg["prediction"] = reg_test
        submission_reg.to_csv("submission_regularized_ensemble.csv", index=False)
        all_submissions["regularized_ensemble"] = reg_score
    
    # 4. Tree-based ensemble (XGB + LightGBM)
    tree_methods = ["xgb_baseline", "lgbm"]
    tree_oof_list = []
    tree_test_list = []
    
    for method in tree_methods:
        if method in oof_preds:
            method_oof = np.mean(list(oof_preds[method].values()), axis=0)
            method_test = np.mean(list(test_preds[method].values()), axis=0)
            tree_oof_list.append(method_oof)
            tree_test_list.append(method_test)
    
    if tree_oof_list:
        tree_oof = np.mean(tree_oof_list, axis=0)
        tree_test = np.mean(tree_test_list, axis=0)
        tree_score = pearsonr(train_df[Config.LABEL_COLUMN], tree_oof)[0]
        print(f"\nTree Ensemble Score: {tree_score:.4f}")
        
        submission_tree = submission_df.copy()
        submission_tree["prediction"] = tree_test
        submission_tree.to_csv("submission_tree_ensemble.csv", index=False)
        all_submissions["tree_ensemble"] = tree_score
    
    # 5. Full ensemble (weighted by performance)
    all_oof_scores = {}
    all_oof_preds = {}
    all_test_preds = {}
    
    for learner_name in oof_preds:
        learner_oof = np.mean(list(oof_preds[learner_name].values()), axis=0)
        learner_test = np.mean(list(test_preds[learner_name].values()), axis=0)
        score = pearsonr(train_df[Config.LABEL_COLUMN], learner_oof)[0]
        
        if not np.isnan(score) and score > 0:  # Only include positive correlations
            all_oof_scores[learner_name] = score
            all_oof_preds[learner_name] = learner_oof
            all_test_preds[learner_name] = learner_test
    
    # Weighted ensemble
    if all_oof_scores:
        total_score = sum(all_oof_scores.values())
        weights = {k: v/total_score for k, v in all_oof_scores.items()}
        
        weighted_oof = sum(weights[k] * all_oof_preds[k] for k in weights)
        weighted_test = sum(weights[k] * all_test_preds[k] for k in weights)
        weighted_score = pearsonr(train_df[Config.LABEL_COLUMN], weighted_oof)[0]
        
        print(f"\nWeighted Full Ensemble Score: {weighted_score:.4f}")
        print("Weights:", {k: f"{v:.3f}" for k, v in weights.items()})
        
        submission_weighted = submission_df.copy()
        submission_weighted["prediction"] = weighted_test
        submission_weighted.to_csv("submission_weighted_ensemble.csv", index=False)
        all_submissions["weighted_ensemble"] = weighted_score
    
    # 6. Simple average of all valid models
    simple_oof = np.mean(list(all_oof_preds.values()), axis=0)
    simple_test = np.mean(list(all_test_preds.values()), axis=0)
    simple_score = pearsonr(train_df[Config.LABEL_COLUMN], simple_oof)[0]
    
    print(f"\nSimple Full Ensemble Score: {simple_score:.4f}")
    
    submission_simple = submission_df.copy()
    submission_simple["prediction"] = simple_test
    submission_simple.to_csv("submission_simple_ensemble.csv", index=False)
    all_submissions["simple_ensemble"] = simple_score
    
    # Print summary
    print("\n" + "="*50)
    print("SUBMISSION SUMMARY:")
    print("="*50)
    for name, score in sorted(all_submissions.items(), key=lambda x: x[1], reverse=True):
        print(f"{name:25s}: {score:.4f}")
    
    return all_submissions

# ===== Main Execution =====
if __name__ == "__main__":
    print("Loading data...")
    train_df, test_df, submission_df = load_data()
    
    print("\nTraining models...")
    oof_preds, test_preds, model_slices = train_and_evaluate(train_df, test_df)
    
    print("\nCreating submissions...")
    submission_scores = create_submissions(train_df, oof_preds, test_preds, submission_df)
    
    print("\nAll submissions created successfully!")
    print("Files created:")
    print("- submission_xgb_baseline.csv (original baseline)")
    print("- submission_robust_ensemble.csv (Huber + RANSAC + TheilSen)")
    print("- submission_regularized_ensemble.csv (Lasso + ElasticNet)")
    print("- submission_tree_ensemble.csv (XGBoost + LightGBM)")
    print("- submission_weighted_ensemble.csv (weighted by performance)")
    print("- submission_simple_ensemble.csv (simple average)")