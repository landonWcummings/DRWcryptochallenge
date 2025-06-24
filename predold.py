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
    TARGETS = ['X345', 'X888', 'X862', 'X302', 'X532', 'X344', 'X385', 'X856', 'X178']
    BATCH_SIZE = 256
    EPOCHS = 20
    LEARNING_RATE = 1e-3
    PATIENCE = 2
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    RANDOM_SEEDS = [42, 43, 44]

# =========================
# Model Definitions
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

# New SUPER size model (approximately 2x larger than Large)
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
# Utility Functions
# =========================

def load_data(path):
    df = pd.read_csv(path, usecols=Config.FEATURES + Config.TARGETS + ['timestamp'])
    return df


def prepare_xy(df):
    y = df[Config.TARGETS].shift(1).iloc[1:].reset_index(drop=True)
    X = df[Config.FEATURES].iloc[1:].reset_index(drop=True)
    return X.values, y.values

# train model with early stopping & ReduceLROnPlateau scheduler
def train_eval_predict(model, X_train, y_train, X_val, y_val):
    model = model.to(Config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=Config.PATIENCE,
        verbose=True
    )
    criterion = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        ),
        batch_size=Config.BATCH_SIZE,
        shuffle=True
    )
    best_rmse = float('inf')
    no_imp = 0
    best_preds = None
    for epoch in range(1, Config.EPOCHS + 1):
        model.train()
        train_preds_list, train_targets_list = [], []
        for xb, yb in loader:
            xb, yb = xb.to(Config.DEVICE), yb.to(Config.DEVICE)
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_preds_list.append(preds.detach().cpu().numpy())
            train_targets_list.append(yb.detach().cpu().numpy())
        train_preds = np.vstack(train_preds_list)
        train_targets = np.vstack(train_targets_list)
        train_rmse = np.sqrt(((train_preds - train_targets) ** 2).mean(axis=0)).sum()
        model.eval()
        with torch.no_grad():
            xv = torch.tensor(X_val, dtype=torch.float32).to(Config.DEVICE)
            yv = torch.tensor(y_val, dtype=torch.float32).to(Config.DEVICE)
            val_preds = model(xv).cpu().numpy()
        val_rmse = np.sqrt(((val_preds - y_val) ** 2).mean(axis=0)).sum()
        logging.info(f"{model.__class__.__name__} Epoch {epoch}: "
                     f"Train RMSE-sum={train_rmse:.4f}, Val RMSE-sum={val_rmse:.4f}")
        scheduler.step(val_rmse)
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            no_imp = 0
            best_preds = val_preds
        else:
            no_imp += 1
            if no_imp > Config.PATIENCE:
                logging.info(f"Early stopping {model.__class__.__name__} at epoch {epoch} ")
                break
    return best_rmse, best_preds

# =========================
# Main Execution
# =========================
def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    df = load_data(Config.TRAIN_PATH)
    X_all, y_all = prepare_xy(df)
    n = len(X_all)
    splits = []
    for seed in Config.RANDOM_SEEDS:
        idx = np.random.RandomState(seed).permutation(n)
        cut = int(0.8 * n)
        splits.append((idx[:cut], idx[cut:]))
    splits.append((np.arange(int(0.8 * n)), np.arange(int(0.8 * n), n)))
    model_constructors = {
        'Baseline': lambda: BASELINE(len(Config.FEATURES), 512, len(Config.TARGETS)),
        'Small': lambda: ModelSmall(len(Config.FEATURES), len(Config.TARGETS)),
        'Medium': lambda: ModelMedium(len(Config.FEATURES), len(Config.TARGETS)),
        'Large': lambda: ModelLarge(len(Config.FEATURES), len(Config.TARGETS)),
        'Super': lambda: ModelSuper(len(Config.FEATURES), len(Config.TARGETS))
    }
    fold_scores = {m: [] for m in model_constructors}
    fold_scores['Ensemble'] = []
    for tr_idx, va_idx in splits:
        X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
        X_va, y_va = X_all[va_idx], y_all[va_idx]
        preds_per_model = {}
        for name, ctor in model_constructors.items():
            model = ctor()
            rmse, preds = train_eval_predict(model, X_tr, y_tr, X_va, y_va)
            fold_scores[name].append(rmse)
            preds_per_model[name] = preds
        ensemble_preds = np.mean(np.stack(list(preds_per_model.values())), axis=0)
        ens_rmse = np.sqrt(((ensemble_preds - y_va) ** 2).mean(axis=0)).sum()
        fold_scores['Ensemble'].append(ens_rmse)
    avg_scores = {m: np.mean(v) for m, v in fold_scores.items()}
    best = min(avg_scores, key=avg_scores.get)
    print("Average RMSE-sum across splits:")
    for name, sc in avg_scores.items(): print(f"  {name}: {sc:.4f}")
    print(f"Top performer: {best} @ {avg_scores[best]:.4f}")

if __name__ == '__main__':
    main()
