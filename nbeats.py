import os
import logging

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from scipy.stats import pearsonr

from DRWkaggle.transformer_files.base_utils import (
    load_and_split_random,
    clean_and_scale,
    apply_pca
)

# --- Configuration & Tunables ---
CSV_PATH          = 'C:/Users/lndnc/OneDrive/Desktop/AI/DRWkaggle/train.csv'
TEST_SIZE         = 0.2              # fraction for held-out test set
WINDOW_SIZE       = 10               # sliding window size
PCA_COMPONENTS    = 50               # PCA dims
BATCH_SIZE        = 128              # training batch size
LEARNING_RATE     = 1e-3
WEIGHT_DECAY      = 1e-3
NUM_EPOCHS        = 50               # training epochs
HIDDEN_UNITS      = 128              # N-BEATS hidden layer size
STACKS            = 3                # number of stacks
BLOCKS_PER_STACK  = 2                # blocks per stack
THETAS_DIM        = 1                # forecasting horizon (1 step)

TARGET_COL        = 'label'
TS_COL            = 'timestamp'

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# --- Dataset using sliding window ---
class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, window: int):
        self.X, self.y, self.window = X, y, window

    def __len__(self):
        return len(self.X) - self.window + 1

    def __getitem__(self, idx):
        x = self.X[idx:idx+self.window]
        y = self.y[idx+self.window-1]
        return torch.from_numpy(x).float(), torch.tensor(y).float()

# --- N-BEATS Block ---
class NBeatsBlock(nn.Module):
    def __init__(self, input_dim, hidden_units, thetas_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, hidden_units)
        # output: backcast (input_dim) + forecast (thetas_dim)
        self.theta_layer = nn.Linear(hidden_units, input_dim + thetas_dim)
        self.relu = nn.ReLU()
        self.input_dim = input_dim
        self.thetas_dim = thetas_dim

    def forward(self, x):
        # x: (batch, input_dim)
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        theta = self.theta_layer(h)
        backcast = theta[:, :self.input_dim]
        forecast = theta[:, self.input_dim:]
        return backcast, forecast

# --- N-BEATS Model ---
class NBeats(nn.Module):
    def __init__(self, window_size, feature_dim, stacks, blocks_per_stack, hidden_units, thetas_dim):
        super().__init__()
        self.input_dim = window_size * feature_dim
        self.stacks = nn.ModuleList()
        for _ in range(stacks):
            blocks = nn.ModuleList([
                NBeatsBlock(self.input_dim, hidden_units, thetas_dim)
                for __ in range(blocks_per_stack)
            ])
            self.stacks.append(blocks)

    def forward(self, x):
        # x: (batch, window, features)
        b, w, f = x.size()
        residual = x.view(b, -1)
        forecast_sum = torch.zeros(b, THETAS_DIM, device=x.device)
        for blocks in self.stacks:
            for block in blocks:
                backcast, forecast = block(residual)
                residual = residual - backcast
                forecast_sum = forecast_sum + forecast
        return forecast_sum.squeeze(-1)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load & split data
    df = pd.read_csv(CSV_PATH)
    df_tr, df_test = load_and_split_random(df, 1, TEST_SIZE)
    logging.info(f"Train points: {len(df_tr)}, Test points: {len(df_test)}")

    features = [c for c in df.columns if c not in [TS_COL, TARGET_COL]]
    X_tr_raw, X_test_raw, scaler = clean_and_scale(df_tr, df_test, features)
    X_tr, X_test, pca = apply_pca(X_tr_raw, X_test_raw, n_components=PCA_COMPONENTS)
    y_tr, y_test = df_tr[TARGET_COL].values, df_test[TARGET_COL].values

    train_ds = WindowDataset(X_tr, y_tr, WINDOW_SIZE)
    test_ds  = WindowDataset(X_test, y_test, WINDOW_SIZE)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # Model, loss, optimizer
    model = NBeats(WINDOW_SIZE, PCA_COMPONENTS, STACKS, BLOCKS_PER_STACK, HIDDEN_UNITS, THETAS_DIM).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Training loop with per-epoch Pearson evaluation
    for epoch in range(1, NUM_EPOCHS+1):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(Xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        # Evaluate on test set each epoch
        model.eval()
        epoch_rs = []
        with torch.no_grad():
            all_preds, all_trues = [], []
            for Xb, yb in test_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                out = model(Xb)
                all_preds.append(out.cpu().numpy())
                all_trues.append(yb.cpu().numpy())
            preds_np = np.concatenate(all_preds)
            trues_np = np.concatenate(all_trues)
            # full-epoch Pearson
            if len(preds_np)>1 and np.std(preds_np)>0:
                epoch_r = pearsonr(trues_np, preds_np)[0]
            else:
                epoch_r = 0.0
            epoch_rs.append(epoch_r)
        mean_r = float(np.mean(epoch_rs))
        std_r = float(np.std(epoch_rs))
        logging.info(f"Epoch {epoch}/{NUM_EPOCHS} - Pearson R: mean={mean_r:.4f}, std={std_r:.4f}")

    # Final evaluation
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            out = model(Xb).cpu().numpy()
            all_preds.append(out)
            all_trues.append(yb.cpu().numpy())
    preds = np.concatenate(all_preds)
    trues = np.concatenate(all_trues)
    r = pearsonr(trues, preds)[0] if len(preds)>1 and np.std(preds)>0 else 0.0
    logging.info(f"Test Pearson R = {r:.4f}")

    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/nbeats_best.pt')

if __name__ == '__main__':
    main()