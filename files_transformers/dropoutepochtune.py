import os
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

from base_utils import (
    RollingWindowDataset,
    load_and_split_random,
    clean_and_scale,
    apply_pca
)

# --- Configuration ---
CSV_PATH            = 'C:/Users/lndnc/OneDrive/Desktop/AI/DRWkaggle/train.csv'
TEST_SEGMENTS       = 20
SEGMENT_FRAC        = 0.01
WINDOW_SIZE         = 10
PCA_COMPONENTS      = 50
BATCH_SIZE          = 128
LEARNING_RATE       = 1e-3    # lowered initial LR
NUM_EPOCHS          = 15
NUM_RUNS            = 6

# Dropout settings to test
dropout_values      = np.arange(0.2, 0.7001, 0.1)

# Regularization & scheduling
INITIAL_NOISE_STD   = 0.01
WEIGHT_DECAY        = 1e-3

TARGET_COL = 'label'
TS_COL     = 'timestamp'

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

os.makedirs('plots', exist_ok=True)

# Positional encoding
def positional_encoding(d_model: int, max_len: int=5000):
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)

class TransformerRegressor(nn.Module):
    def __init__(self, dropout_rate: float):
        super().__init__()
        self.noise_std = INITIAL_NOISE_STD
        self.initial_dropout = dropout_rate
        self.embed = nn.Linear(PCA_COMPONENTS, 256)
        self.register_buffer('pe', positional_encoding(256, WINDOW_SIZE))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8,
            dim_feedforward=512,
            dropout=dropout_rate,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.head = nn.Sequential(
            nn.LayerNorm(256),
            self.dropout_layer,
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.embed(x)
        x = x + self.pe[:, :x.size(1)]
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        x = x.transpose(0,1)
        x = self.transformer(x)
        x = x[-1]
        return self.head(x).squeeze(-1)

# Load & preprocess once
df = pd.read_csv(CSV_PATH)
features = [c for c in df.columns if c not in [TS_COL, TARGET_COL]]
df_tr, df_val = load_and_split_random(df, TEST_SEGMENTS, SEGMENT_FRAC)
X_tr, X_val, scaler = clean_and_scale(df_tr, df_val, features)
X_tr_pca, X_val_pca, pca = apply_pca(X_tr, X_val, n_components=PCA_COMPONENTS)
cols = [f'pca{i}' for i in range(PCA_COMPONENTS)]
train_df = pd.DataFrame(X_tr_pca, columns=cols);  train_df[TARGET_COL] = df_tr[TARGET_COL].values
val_df   = pd.DataFrame(X_val_pca, columns=cols);   val_df[TARGET_COL]   = df_val[TARGET_COL].values

train_loader = DataLoader(RollingWindowDataset(train_df, WINDOW_SIZE, cols, TARGET_COL),
                          batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(RollingWindowDataset(val_df,   WINDOW_SIZE, cols, TARGET_COL),
                          batch_size=BATCH_SIZE)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_single(dropout_rate):
    model = TransformerRegressor(dropout_rate).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max',
        factor=0.5, patience=1, min_lr=1e-6, verbose=True
    )
    best_r, best_state = -np.inf, None

    all_pearsons = []
    for epoch in range(1, NUM_EPOCHS+1):
        # ---- Anneal noise & dropout linearly down to 0 over first 5 epochs ----
        frac = max(0, 1 - (epoch-1)/5)
        model.noise_std = INITIAL_NOISE_STD * frac
        model.dropout_layer.p = model.initial_dropout * frac

        # train
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = nn.MSELoss()(out, y)
            loss.backward()
            optimizer.step()

        # eval
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds.append(model(x).cpu().numpy())
                trues.append(y.cpu().numpy())
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        r = pearsonr(trues, preds)[0] if len(trues)>1 and np.std(preds)>0 else 0.0
        all_pearsons.append(r)

        # checkpoint best
        if r > best_r:
            best_r, best_state = r, model.state_dict()

        # step LR scheduler on validation metric
        scaler.step(r)
        logging.info(f"Run dr={dropout_rate:.2f} epoch {epoch} — val R={r:.4f}")

    # restore best
    model.load_state_dict(best_state)
    logging.info(f" Restored best epoch R={best_r:.4f}")
    return all_pearsons

# Main sweep
enum_epochs = list(range(1, NUM_EPOCHS+1))
results = {}
for dr in dropout_values:
    runs = [run_single(dr) for _ in range(NUM_RUNS)]
    results[dr] = np.mean(runs, axis=0)

# Save & plot
df_res = pd.DataFrame(results, index=enum_epochs)
df_res.index.name = 'epoch'
df_res.to_csv('plots/dropout_epoch_results.csv')

plt.figure(figsize=(10,6))
for dr, vals in results.items():
    plt.plot(enum_epochs, vals, label=f"dropout={dr:.2f}")
plt.xlabel('Epoch'); plt.ylabel('Avg Val Pearson R')
plt.title('Validation Pearson vs Epoch (Annealed Reg & Lower LR)')
plt.legend(); plt.grid(True)
plt.savefig('plots/dropout_epoch_sweep.png')
plt.show()
