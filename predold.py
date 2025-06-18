import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging

# =========================
# Configuration
# =========================
class Config:
    TRAIN_PATH = r"C:\Users\lndnc\OneDrive\Desktop\AI\DRWkaggle\train.csv"
    FEATURES = [
        "X863", "X856", "X344", "X598", "X862", "X385", "X852", "X603", "X860", "X674",
        "X415", "X345", "X137", "X855", "X174", "X302", "X178", "X532", "X168", "X612",
        "bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume", "X888", "X421", "X333"
    ]
    TARGETS = FEATURES  # predict all features
    BATCH_SIZE = 256
    EPOCHS = 20
    LEARNING_RATE = 1e-3
    PATIENCE = 2
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    RANDOM_SEEDS = [42, 43, 44]

# =========================
# Model Definitions (unchanged)
# =========================
class BASELINE(nn.Module):
    def __init__(self, in_dim, embed_dim, out_dim):
        super().__init__()
        self.embed = nn.Linear(in_dim, embed_dim)
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.head = nn.Linear(128, out_dim)
    def forward(self, x):
        z = self.embed(x)
        h = self.layers(z)
        return self.head(h)

class ModelSmall(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, out_dim)
        )
    def forward(self, x): return self.net(x)

class ModelMedium(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, out_dim)
        )
    def forward(self, x): return self.net(x)

class ModelLarge(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, out_dim)
        )
    def forward(self, x): return self.net(x)

class ModelSuper(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, out_dim)
        )
    def forward(self, x): return self.net(x)

# =========================
# Data Utilities
# =========================
def load_data(path):
    df = pd.read_csv(path, usecols=Config.FEATURES + ['timestamp'])
    return df

def prepare_xy(df):
    y = df[Config.FEATURES].shift(1).iloc[1:].reset_index(drop=True)
    X = df[Config.FEATURES].iloc[1:].reset_index(drop=True)
    return X.values, y.values

# =========================
# Training & Evaluation
# =========================
def train_eval_predict(model, X_train, y_train, X_val, y_val):
    # standardize features on train fold
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-6
    Xtr = (X_train - mean) / std
    Xva = (X_val - mean) / std

    model = model.to(Config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=Config.PATIENCE, verbose=True
    )
    criterion = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(
            torch.tensor(Xtr, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        ), batch_size=Config.BATCH_SIZE, shuffle=True
    )

    best_rmse = float('inf')
    no_imp = 0
    best_preds = None
    for epoch in range(1, Config.EPOCHS+1):
        model.train()
        preds_list, targets_list = [], []
        for xb, yb in loader:
            xb, yb = xb.to(Config.DEVICE), yb.to(Config.DEVICE)
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            preds_list.append(preds.cpu().numpy())
            targets_list.append(yb.cpu().numpy())
        train_preds = np.vstack(preds_list)
        train_targets = np.vstack(targets_list)
        train_rmse = np.sqrt(((train_preds-train_targets)**2).mean(axis=0)).sum()

        model.eval()
        with torch.no_grad():
            val_tensor = torch.tensor(Xva, dtype=torch.float32).to(Config.DEVICE)
            val_preds = model(val_tensor).cpu().numpy()
        val_rmse = np.sqrt(((val_preds-y_val)**2).mean(axis=0)).sum()

        logging.info(f"{model.__class__.__name__} Epoch {epoch}: Train RMSE-sum={train_rmse:.4f}, Val RMSE-sum={val_rmse:.4f}")
        scheduler.step(val_rmse)
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_preds = val_preds
            no_imp = 0
        else:
            no_imp += 1
            if no_imp > Config.PATIENCE:
                logging.info(f"Early stopping at epoch {epoch}")
                break
    return best_rmse, best_preds

# =========================
# Main
# =========================
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    df = load_data(Config.TRAIN_PATH)
    X_all, y_all = prepare_xy(df)
    n = len(X_all)

    # prepare folds
    splits = []
    for seed in Config.RANDOM_SEEDS:
        idx = np.random.RandomState(seed).permutation(n)
        cut = int(0.8*n)
        splits.append((idx[:cut], idx[cut:]))
    # final split
    splits.append((np.arange(int(0.8*n)), np.arange(int(0.8*n), n)))

    constructors = {
        'Baseline': lambda: BASELINE(len(Config.FEATURES), 512, len(Config.TARGETS)),
        'Small':     lambda: ModelSmall(len(Config.FEATURES), len(Config.TARGETS)),
        'Medium':    lambda: ModelMedium(len(Config.FEATURES), len(Config.TARGETS)),
        'Large':     lambda: ModelLarge(len(Config.FEATURES), len(Config.TARGETS)),
        'Super':     lambda: ModelSuper(len(Config.FEATURES), len(Config.TARGETS))
    }

    fold_scores = {name: [] for name in constructors}
    fold_scores['Ensemble'] = []
    ensemble_preds_all = []
    y_val_all = []

    for tr_idx, va_idx in splits:
        X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
        X_va, y_va = X_all[va_idx], y_all[va_idx]
        preds_per_model = {}
        for name, ctor in constructors.items():
            rmse, preds = train_eval_predict(ctor(), X_tr, y_tr, X_va, y_va)
            fold_scores[name].append(rmse)
            preds_per_model[name] = preds
        # ensemble
        ens = np.mean(np.stack(list(preds_per_model.values())), axis=0)
        ensemble_preds_all.append(ens)
        y_val_all.append(y_va)
        fold_scores['Ensemble'].append(np.sqrt(((ens-y_va)**2).mean(axis=0)).sum())

    # overall results
    avg_scores = {name: np.mean(scores) for name, scores in fold_scores.items()}
    best = min(avg_scores, key=avg_scores.get)
    print("Average RMSE-sum across splits:")
    for name, sc in avg_scores.items(): print(f"  {name}: {sc:.4f}")
    print(f"Top performer: {best} @ {avg_scores[best]:.4f}")

    # Feature-wise difficulty:
    # compute RMSE per feature averaged across folds
    rmse_per_fold = []
    for preds, actual in zip(ensemble_preds_all, y_val_all):
        per_feat_rmse = np.sqrt(((preds - actual)**2).mean(axis=0))
        rmse_per_fold.append(per_feat_rmse)
    mean_rmse = np.mean(np.stack(rmse_per_fold), axis=0)
    print("\nFeature-wise avg RMSE:")
    for feat, err in zip(Config.FEATURES, mean_rmse):
        print(f"  {feat}: {err:.4f}")
