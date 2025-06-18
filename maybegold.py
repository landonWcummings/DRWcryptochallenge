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
    EMBED_DIM = 256  # embedding dimension for past-feature predictor
    LABEL_COLUMN = "label"
    N_FOLDS = 3
    RANDOM_STATE = 42
    NN_BATCH = 512
    NN_EPOCHS = 5
    NN_PATIENCE = 1
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
# NN Predictor for Past-Day Features
# =========================
class PastFeaturePredictor(nn.Module):
    def __init__(self, in_dim, embed_dim):
        super().__init__()
        # embed to 256
        self.embed = nn.Linear(in_dim, embed_dim)
        # deep MLP: 256 -> 512 -> 256 -> 128
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 128),
            nn.ReLU()
        )
        # output head to reconstruct all features
        self.head = nn.Linear(128, in_dim)

    def forward(self, x):
        z = self.embed(x)
        h = self.layers(z)
        out = self.head(h)
        return out

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
# Train NN to predict past-day features
# =========================
def train_past_feature_predictor(train_df, test_df):
    df = train_df.copy()
    # shift all FEATURES by 1 day to use as y
    y_shift = df[Config.FEATURES].shift(1).dropna().reset_index(drop=True)
    X_all = df.loc[1:, Config.FEATURES].values
    y_all = y_shift.values

    # train/val split
    idx = np.arange(len(X_all))
    np.random.seed(Config.RANDOM_STATE)
    np.random.shuffle(idx)
    split = int(0.9 * len(idx))
    tr_idx, va_idx = idx[:split], idx[split:]

    X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
    X_va, y_va = X_all[va_idx], y_all[va_idx]

    train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                              torch.tensor(y_tr, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_va, dtype=torch.float32),
                            torch.tensor(y_va, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=Config.NN_BATCH, shuffle=True)

    model = PastFeaturePredictor(in_dim=len(Config.FEATURES), embed_dim=Config.EMBED_DIM)
    model.to(Config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    best_loss = float('inf')
    patience = 0

    for epoch in range(1, Config.NN_EPOCHS+1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(Config.DEVICE), yb.to(Config.DEVICE)
            pred = model(xb)
            rmse = torch.sqrt(((pred - yb)**2).mean(dim=0))
            loss = rmse.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train = total_loss / len(train_loader)

        # val
        model.eval()
        with torch.no_grad():
            Xv = torch.tensor(X_va, dtype=torch.float32).to(Config.DEVICE)
            yv = torch.tensor(y_va, dtype=torch.float32).to(Config.DEVICE)
            pred_v = model(Xv)
            rmse_v = torch.sqrt(((pred_v - yv)**2).mean(dim=0))
            val_loss = rmse_v.sum().item()
        logging.info(f"PF Epoch {epoch}: train_loss={avg_train:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            patience = 0
            torch.save(model.state_dict(), 'best_past_feat.pt')
        else:
            patience += 1
            if patience > Config.NN_PATIENCE:
                logging.info("PF Early stopping")
                break

    model.load_state_dict(torch.load('best_past_feat.pt'))
    model.eval()
    with torch.no_grad():
        X_full = torch.tensor(train_df[Config.FEATURES].values, dtype=torch.float32).to(Config.DEVICE)
        pred_tr = model(X_full).cpu().numpy()
        X_test = torch.tensor(test_df[Config.FEATURES].values, dtype=torch.float32).to(Config.DEVICE)
        pred_te = model(X_test).cpu().numpy()

    # return reconstructed past-day features
    return pred_tr, pred_te

# =========================
# Time-decay & Slices (unchanged)
# =========================
def create_time_decay_weights(n: int, decay: float = 0.95) -> np.ndarray:
    positions = np.arange(n)
    normalized = positions/(n-1)
    weights = decay**(1.0-normalized)
    return weights * n/weights.sum()

def get_model_slices(n_samples: int):
    return [
        {"name": "full_data", "cutoff": 0},
        {"name": "last_75pct", "cutoff": int(0.25*n_samples)},
        {"name": "last_50pct", "cutoff": int(0.50*n_samples)}
    ]

# =========================
# Training & Evaluation for regressors
# =========================
def train_and_evaluate(train_df, test_df):
    n = len(train_df)
    slices = get_model_slices(n)
    oof = {ln['name']:{s['name']:np.zeros(n) for s in slices} for ln in LEARNERS}
    test_p = {ln['name']:{s['name']:np.zeros(len(test_df)) for s in slices} for ln in LEARNERS}
    full_w = create_time_decay_weights(n)
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=False)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train_df),1):
        logging.info(f"Fold {fold}/{Config.N_FOLDS}")
        Xv, yv = train_df.iloc[va_idx][Config.FEATURES], train_df.iloc[va_idx][Config.LABEL_COLUMN]
        for s in slices:
            c, name = s['cutoff'], s['name']
            sub = train_df.iloc[c:].reset_index(drop=True)
            rel = tr_idx[tr_idx>=c] - c
            Xt, yt = sub.iloc[rel][Config.FEATURES], sub.iloc[rel][Config.LABEL_COLUMN]
            sw = create_time_decay_weights(len(sub))[rel] if c>0 else full_w[tr_idx]
            logging.info(f" Slice {name}: train={len(Xt)}")
            for ln in LEARNERS:
                model = ln['Estimator'](**ln['params'])
                model.fit(Xt, yt, sample_weight=sw, eval_set=[(Xv,yv)], verbose=False)
                preds = model.predict(train_df.iloc[va_idx][Config.FEATURES])
                rmse = np.sqrt(((preds-train_df.iloc[va_idx][Config.LABEL_COLUMN])**2).mean())
                p = pearsonr(train_df.iloc[va_idx][Config.LABEL_COLUMN], preds)[0]
                logging.info(f"  [{ln['name']}] slice={name} RMSE={rmse:.4f} Pearson={p:.4f}")
                mask = va_idx>=c
                if mask.any():
                    oof[ln['name']][name][va_idx[mask]] = model.predict(train_df.iloc[va_idx[mask]][Config.FEATURES])
                if c>0 and (~mask).any():
                    oof[ln['name']][name][va_idx[~mask]] = oof[ln['name']]['full_data'][va_idx[~mask]]
                test_p[ln['name']][name] += model.predict(test_df[Config.FEATURES])

    for ln in test_p:
        for name in test_p[ln]:
            test_p[ln][name] /= Config.N_FOLDS

    return oof, test_p, slices

# =========================
# Ensemble & Submission
# =========================
def ensemble_and_submit(train_df, oof, test_p, submission_df):
    for ln in oof:
        scores = {s: pearsonr(train_df[Config.LABEL_COLUMN], oof[ln][s])[0] for s in oof[ln]}
        total = sum(scores.values())
        oof_s = np.mean(list(oof[ln].values()), axis=0)
        test_s = np.mean(list(test_p[ln].values()), axis=0)
        wt_oof = sum(scores[s]/total * oof[ln][s] for s in scores)
        wt_test = sum(scores[s]/total * test_p[ln][s] for s in scores)
        logging.info(f"[{ln}] Simple Pearson={pearsonr(train_df[Config.LABEL_COLUMN], oof_s)[0]:.4f}")
        logging.info(f"[{ln}] Weighted Pearson={pearsonr(train_df[Config.LABEL_COLUMN], wt_oof)[0]:.4f}")
    final_oof = np.mean([np.mean(list(oof[ln].values()), axis=0) for ln in oof], axis=0)
    final_test = np.mean([np.mean(list(test_p[ln].values()), axis=0) for ln in test_p], axis=0)
    logging.info(f"FINAL Pearson={pearsonr(train_df[Config.LABEL_COLUMN], final_oof)[0]:.4f}")
    submission_df['prediction'] = final_test
    submission_df.to_csv('submission.csv', index=False)
    logging.info("Saved submission.csv")

# =========================
# Main
# =========================
if __name__ == '__main__':
    train_df, test_df, submission_df = load_data()
    # predict past-day features
    past_tr, past_te = train_past_feature_predictor(train_df, test_df)
    # append reconstructed past-day features
    for i, feat in enumerate(Config.FEATURES):
        train_df[f'past_{feat}'] = past_tr[:, i]
        test_df[f'past_{feat}'] = past_te[:, i]
    # extend FEATURES list
    Config.FEATURES += [f'past_{feat}' for feat in Config.FEATURES]
    # train regressors
    oof, test_p, _ = train_and_evaluate(train_df, test_df)
    # ensemble & submit
    ensemble_and_submit(train_df, oof, test_p, submission_df)
