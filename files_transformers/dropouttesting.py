import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
import logging
import math
import matplotlib.pyplot as plt

from DRWkaggle.transformer_files.base_utils import (
    RollingWindowDataset,
    load_and_split_random,
    clean_and_scale,
    apply_pca
)

# --- Configuration ---
CSV_PATH      = 'C:/Users/lndnc/OneDrive/Desktop/AI/DRWkaggle/train.csv'
TEST_SEGMENTS = 20
SEGMENT_FRAC  = 0.01
WINDOW_SIZE    = 10
PCA_COMPONENTS = 50
BATCH_SIZE     = 128
LEARNING_RATE  = 1e-3
TARGET_COL = 'label'
TS_COL     = 'timestamp'

# sweep parameters
dropout_values = np.arange(0.0, 1.0001, 0.05)
runs_per_setting = 18

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# PositionalEncoding
def positional_encoding(d_model: int, max_len: int = 5000):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)

# Factory for Transformer with variable dropout
def make_transformer(dropout_rate):
    class TransformerRegressor(nn.Module):
        def __init__(self, in_features: int, noise_std: float = 0.0):
            super().__init__()
            self.noise_std = noise_std
            self.embed = nn.Linear(in_features, EMBED_DIM)
            self.pos_enc = positional_encoding(EMBED_DIM, max_len=WINDOW_SIZE)
            self.register_buffer('pe', self.pos_enc)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=EMBED_DIM,
                nhead=NHEADS,
                dim_feedforward=FFN_DIM,
                dropout=dropout_rate,
                activation='relu'
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=NLAYERS
            )
            self.head = nn.Sequential(
                nn.LayerNorm(EMBED_DIM),
                nn.Dropout(dropout_rate),
                nn.Linear(EMBED_DIM, 1)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.embed(x)
            x = x + self.pe[:, :x.size(1)]
            if self.training and self.noise_std > 0:
                x = x + torch.randn_like(x) * self.noise_std
            x = x.transpose(0, 1)
            x = self.transformer(x)
            x = x[-1]
            return self.head(x).squeeze(-1)
    return TransformerRegressor

# Fixed hyperparameters
EMBED_DIM   = 128
NHEADS      = 8
FFN_DIM     = 256
NLAYERS     = 4

# Run one epoch (train + eval)
def run_one_epoch(model, loader_train, loader_valid, device):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    # Train
    for x, y in loader_train:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
    # Eval
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader_valid:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds.append(out.cpu().numpy())
            trues.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    r = pearsonr(trues, preds)[0] if len(trues)>1 and np.std(preds)>0 else 0.0
    return r

# Prepare data once
df = pd.read_csv(CSV_PATH)
feature_cols = [c for c in df.columns if c not in [TS_COL, TARGET_COL]]
df_tr, df_val = load_and_split_random(df, TEST_SEGMENTS, SEGMENT_FRAC)
X_tr, X_val, scaler = clean_and_scale(df_tr, df_val, feature_cols)
X_tr_pca, X_val_pca, pca = apply_pca(X_tr, X_val, n_components=PCA_COMPONENTS)
cols = [f'pca{i}' for i in range(PCA_COMPONENTS)]
df_train = pd.DataFrame(X_tr_pca, columns=cols); df_train[TARGET_COL] = df_tr[TARGET_COL].values
df_valid = pd.DataFrame(X_val_pca, columns=cols); df_valid[TARGET_COL] = df_val[TARGET_COL].values
train_loader = DataLoader(RollingWindowDataset(df_train, WINDOW_SIZE, cols, TARGET_COL), batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(RollingWindowDataset(df_valid, WINDOW_SIZE, cols, TARGET_COL), batch_size=BATCH_SIZE)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Sweep recording
avg_r1_list, avg_r2_list = [], []
for dr in dropout_values:
    r1_runs, r2_runs = [], []
    for _ in range(runs_per_setting):
        ModelClass = make_transformer(dr)
        model = ModelClass(in_features=PCA_COMPONENTS).to(device)
        # Epoch 1
        r1 = run_one_epoch(model, train_loader, valid_loader, device)
        # Epoch 2 (continuing training)
        r2 = run_one_epoch(model, train_loader, valid_loader, device)
        r1_runs.append(r1)
        r2_runs.append(r2)
    avg_r1, avg_r2 = np.mean(r1_runs), np.mean(r2_runs)
    logging.info(f"Dropout {dr:.2f}: E1 runs {r1_runs}, avg={avg_r1:.4f}; E2 runs {r2_runs}, avg={avg_r2:.4f}")
    avg_r1_list.append(avg_r1)
    avg_r2_list.append(avg_r2)

# Plot results
plt.figure()
plt.plot(dropout_values, avg_r1_list, 'o-', label='Epoch 1')
plt.plot(dropout_values, avg_r2_list, 's-', label='Epoch 2')
plt.xlabel('Dropout rate')
plt.ylabel('Avg Val Pearson R')
plt.title('Dropout Sweep: Epoch 1 vs 2')
plt.legend()
plt.grid(True)
plt.savefig('dropout_sweep_epochs1_2.png')
plt.show()