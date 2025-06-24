import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from scipy.stats import pearsonr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging

# =========================
# Configuration
# =========================
class Config:
    TRAIN_PATH = r"C:\Users\lndnc\OneDrive\Desktop\AI\DRWkaggle\train.csv"
    TEST_PATH = r"C:\Users\lndnc\OneDrive\Desktop\AI\DRWkaggle\test.csv"
    SUBMISSION_PATH = r"C:\Users\lndnc\OneDrive\Desktop\AI\DRWkaggle\xgb_submission.csv"

    FEATURES = [
        "X863", "X856", "X344", "X598", "X862", "X385", "X852", "X603", "X860", "X674",
        "X415", "X345", "X137", "X855", "X174", "X302", "X178", "X532", "X168", "X612",
        "bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume", "X888", "X421", "X333"
    ]
    NN_TARGETS = ['X345', 'X888', 'X862', 'X302', 'X532', 'X344', 'X385', 'X856', 'X178']
    LABEL_COLUMN = "label"
    N_FOLDS = 3
    RANDOM_STATE = 42
    NN_BATCH = 512
    NN_EPOCHS = 5  # fixed epochs, no early stopping
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# XGBoost parameters
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

LEARNERS = [
    {"name": "xgb", "Estimator": XGBRegressor, "params": XGB_PARAMS}
]

# =========================
# Logger Setup
# =========================
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# =========================
# Define three NN architectures
# =========================
class PredictorA(nn.Module):  # shallow wide
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, out_dim)
        )
    def forward(self, x):
        return self.model(x)

class PredictorB(nn.Module):  # deeper
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, out_dim)
        )
    def forward(self, x):
        return self.model(x)

class PredictorC(nn.Module):  # with dropout
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, out_dim)
        )
    def forward(self, x):
        return self.model(x)

# =========================
# Utility Functions
# =========================
def load_data():
    train_df = pd.read_csv(Config.TRAIN_PATH, usecols=Config.FEATURES + [Config.LABEL_COLUMN])
    test_df = pd.read_csv(Config.TEST_PATH, usecols=Config.FEATURES)
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    logging.info(f"Loaded data - Train: {train_df.shape}, Test: {test_df.shape}, Submission: {submission_df.shape}")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), submission_df

# =========================
# Train one NN and produce predictions
# =========================
def train_single_nn(model_cls, train_df, test_df):
    df = train_df.copy()
    df_shift = df[Config.NN_TARGETS].shift(1).dropna().reset_index(drop=True)
    X = df.loc[1:, Config.FEATURES].values
    y = df_shift.values

    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=Config.NN_BATCH, shuffle=True)

    model = model_cls(len(Config.FEATURES), len(Config.NN_TARGETS)).to(Config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.MSELoss()

    for epoch in range(1, Config.NN_EPOCHS + 1):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(Config.DEVICE), yb.to(Config.DEVICE)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        logging.info(f"{model_cls.__name__} Epoch {epoch}: MSE={avg_loss:.4f}")

    model.eval()
    with torch.no_grad():
        X_train = torch.tensor(train_df[Config.FEATURES].values, dtype=torch.float32).to(Config.DEVICE)
        pred_tr = model(X_train).cpu().numpy()
        X_test = torch.tensor(test_df[Config.FEATURES].values, dtype=torch.float32).to(Config.DEVICE)
        pred_te = model(X_test).cpu().numpy()

    return pred_tr, pred_te

# =========================
# Time-decay & Slices (unchanged)
# =========================
def create_time_decay_weights(n: int, decay: float = 0.95) -> np.ndarray:
    positions = np.arange(n)
    normalized = positions / (n-1)
    weights = decay**(1.0 - normalized)
    return weights * n / weights.sum()

def get_model_slices(n_samples: int):
    return [
        {"name": "full_data", "cutoff": 0},
        {"name": "last_75pct", "cutoff": int(0.25 * n_samples)},
        {"name": "last_50pct", "cutoff": int(0.50 * n_samples)}
    ]

# =========================
# Training & Evaluation (unchanged)
# =========================
def train_and_evaluate(train_df, test_df):
    n = len(train_df)
    slices = get_model_slices(n)
    oof = {ln['name']:{s['name']:np.zeros(n) for s in slices} for ln in LEARNERS}
    test_p = {ln['name']:{s['name']:np.zeros(len(test_df)) for s in slices} for ln in LEARNERS}
    full_w = create_time_decay_weights(n)
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=False)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train_df), 1):
        logging.info(f"Fold {fold}/{Config.N_FOLDS}")
        Xv, yv = train_df.iloc[va_idx][Config.FEATURES], train_df.iloc[va_idx][Config.LABEL_COLUMN]
        for s in slices:
            c, name = s['cutoff'], s['name']
            sub = train_df.iloc[c:].reset_index(drop=True)
            rel = tr_idx[tr_idx >= c] - c
            Xt, yt = sub.iloc[rel][Config.FEATURES], sub.iloc[rel][Config.LABEL_COLUMN]
            sw = create_time_decay_weights(len(sub))[rel] if c > 0 else full_w[tr_idx]
            logging.info(f" Slice {name}: train={len(Xt)}")
            for ln in LEARNERS:
                model = ln['Estimator'](**ln['params'])
                model.fit(Xt, yt, sample_weight=sw, eval_set=[(Xv, yv)], verbose=False)
                preds = model.predict(train_df.iloc[va_idx][Config.FEATURES])
                logging.info(f"  [{ln['name']}] slice={name} RMSE={np.sqrt(((preds - train_df.iloc[va_idx][Config.LABEL_COLUMN])**2).mean()):.4f} Pearson={pearsonr(train_df.iloc[va_idx][Config.LABEL_COLUMN], preds)[0]:.4f}")
                mask = va_idx >= c
                if mask.any():
                    oof[ln['name']][name][va_idx[mask]] = preds[mask]
                if c > 0 and (~mask).any():
                    oof[ln['name']]['full_data'][va_idx[~mask]] = oof[ln['name']]['full_data'][va_idx[~mask]]
                test_p[ln['name']][name] += model.predict(test_df[Config.FEATURES])

    for ln in test_p:
        for name in test_p[ln]:
            test_p[ln][name] /= Config.N_FOLDS

    return oof, test_p, slices

# =========================
# Ensemble & Submission (unchanged)
# =========================
def ensemble_and_submit(train_df, oof, test_p, submission_df):
    for ln in oof:
        scores = {s: pearsonr(train_df[Config.LABEL_COLUMN], oof[ln][s])[0] for s in oof[ln]}
        logging.info(f"{ln.upper()} Simple Pearson={pearsonr(train_df[Config.LABEL_COLUMN], np.mean(list(oof[ln].values()), axis=0))[0]:.4f}")
    final_oof = np.mean([np.mean(list(oof[ln].values()), axis=0) for ln in oof], axis=0)
    final_test = np.mean([np.mean(list(test_p[ln].values()), axis=0) for ln in test_p], axis=0)
    submission_df['prediction'] = final_test
    submission_df.to_csv('submission.csv', index=False)
    logging.info("Saved submission.csv")

# =========================
# Main
# =========================
if __name__ == '__main__':
    train_df, test_df, submission_df = load_data()

    # run each NN and add their predictions
    for model_cls, tag in [(PredictorA, 'A')]:
        pred_tr, pred_te = train_single_nn(model_cls, train_df, test_df)
        for i, f in enumerate(Config.NN_TARGETS):
            train_df[f'pred_{tag}_{f}'] = pred_tr[:, i]
            test_df[f'pred_{tag}_{f}'] = pred_te[:, i]
        Config.FEATURES += [f'pred_{tag}_{f}' for f in Config.NN_TARGETS]

    # now train and ensemble
    oof, test_p, _ = train_and_evaluate(train_df, test_df)
    ensemble_and_submit(train_df, oof, test_p, submission_df)
