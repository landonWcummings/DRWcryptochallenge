import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
import logging
import math

from base_utils import (
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
NUM_EPOCHS     = 20   # focus on early stopping
PATIENCE       = 5    # stop once R starts to plateau

EMBED_DIM   = 128
NHEADS      = 8
FFN_DIM     = 256
NLAYERS     = 4
DROPOUT     = 0.4     # moderate dropout for generalization

NOISE_STD         = 0.01
WEIGHT_DECAY      = 1e-3
WARMUP_FRAC       = 0.1
COSINE_ANNEAL_FRAC= 0.9

TARGET_COL = 'label'
TS_COL     = 'timestamp'

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# Positional encoding
def positional_encoding(d_model: int, max_len: int=5000):
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000)/d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)

class TransformerRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.noise_std = NOISE_STD
        self.embed = nn.Linear(PCA_COMPONENTS, EMBED_DIM)
        self.register_buffer('pe', positional_encoding(EMBED_DIM, WINDOW_SIZE))
        encoder = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM, nhead=NHEADS,
            dim_feedforward=FFN_DIM, dropout=DROPOUT,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder, num_layers=NLAYERS)
        self.head = nn.Sequential(
            nn.LayerNorm(EMBED_DIM),
            nn.Dropout(DROPOUT),
            nn.Linear(EMBED_DIM, 1)
        )

    def forward(self, x):
        # x: (B, T, F)
        x = self.embed(x)
        x = x + self.pe[:, :x.size(1)]
        if self.training:
            x = x + torch.randn_like(x) * self.noise_std
        x = x.transpose(0,1)
        x = self.transformer(x)
        x = x[-1]
        return self.head(x).squeeze(-1)

# Data preparation
df = pd.read_csv(CSV_PATH)
feats = [c for c in df.columns if c not in [TS_COL, TARGET_COL]]
df_tr, df_val = load_and_split_random(df, TEST_SEGMENTS, SEGMENT_FRAC)
X_tr, X_val, scaler = clean_and_scale(df_tr, df_val, feats)
X_tr_pca, X_val_pca, pca = apply_pca(X_tr, X_val, n_components=PCA_COMPONENTS)
cols = [f'pca{i}' for i in range(PCA_COMPONENTS)]
train_df = pd.DataFrame(X_tr_pca, columns=cols)
train_df[TARGET_COL] = df_tr[TARGET_COL].values
val_df   = pd.DataFrame(X_val_pca, columns=cols)
val_df[TARGET_COL]   = df_val[TARGET_COL].values
train_loader = DataLoader(RollingWindowDataset(train_df, WINDOW_SIZE, cols, TARGET_COL), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(RollingWindowDataset(val_df, WINDOW_SIZE, cols, TARGET_COL),   batch_size=BATCH_SIZE)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerRegressor().to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# LR scheduler: warmup + cosine annealing
total_steps = NUM_EPOCHS * len(train_loader)
warmup_steps = int(WARMUP_FRAC * total_steps)
def lr_lambda(step):
    if step < warmup_steps:
        return float(step) / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Training loop with early stopping
best_r, patience = -np.inf, 0
for epoch in range(1, NUM_EPOCHS+1):
    # train
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

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
    r = pearsonr(trues, preds)[0] if len(trues)>1 else 0.0
    logging.info(f"Epoch {epoch}  Val Pearson R: {r:.4f}")

    # early stopping
    if r > best_r:
        best_r, patience = r, 0
    else:
        patience += 1
        if patience >= PATIENCE:
            logging.info(f"Stopping at epoch {epoch}")
            break

logging.info(f"Final best Val R: {best_r:.4f}")
