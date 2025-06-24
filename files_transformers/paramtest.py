import os
import math
import logging

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

from DRWkaggle.transformer_files.base_utils import (
    RollingWindowDataset,
    load_and_split_random,
    clean_and_scale,
    apply_pca
)

# --- Configuration & Tunables ---
# --- Configuration & Tunables ---
CSV_PATH                = 'C:/Users/lndnc/OneDrive/Desktop/AI/DRWkaggle/train.csv'
TEST_SEGMENTS           = 20
SEGMENT_FRAC            = 0.01
WINDOW_SIZE             = 14       # from target params 'window_size'
NUM_FOLDS               = 5
PCA_COMPONENTS          = 39       # from target params 'pca_components'
BATCH_SIZE              = 128      # from target params 'batch_size'

# Transformer architecture
D_MODEL                 = 512      # from target params 'd_model'
NUM_HEADS               = 16       # from target params 'num_heads'
DIM_FEEDFORWARD         = 256      # from target params 'dim_ff'
NUM_LAYERS              = 4        # from target params 'num_layers'

# Base optimizer & regularization
LEARNING_RATE           = 0.005945003080660254   # from target params 'lr'
WEIGHT_DECAY            = 4.897809997721506e-05  # from target params 'weight_decay'

# 1) Huber Loss δ (robust to outliers vs MSE)
LOSS_FN                 = 'huber'  # from target params 'loss_fn'
HUBER_DELTA             = 1.0      # keep default or adjust as needed

# 2) Gradient clipping (avoid exploding gradients)
CLIP_GRAD_NORM          = 0.7533296876992163    # from target params 'clip_grad_norm'

# 3) LR warm-up epochs (linearly ramp from 0 → LR)
WARMUP_EPOCHS           = 4        # from target params 'warmup_epochs'

# 4) Cosine-anneal down to ηₘᵢₙ
COSINE_ETA_MIN          = 9.361787998583915e-07 # from target params 'cosine_eta_min'

# 5) Separate dropouts (attention vs feed-forward)
DROPOUT_ATTENTION       = 0.42695900549025884   # from target params 'dropout_attn'
DROPOUT_FF              = 0.17856387615492594   # from target params 'dropout_ff'

# 6) Noise & annealing
USE_NOISE               = True    # from target params 'use_noise'
INITIAL_NOISE_STD       = 0.06539539828952132   # from target params 'noise_std'

# Overall training
NUM_EPOCHS              = 1        # from target params 'num_epochs'

TARGET_COL              = 'label'
TS_COL                  = 'timestamp'


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

os.makedirs('checkpoints', exist_ok=True)

# Positional encoding
def positional_encoding(d_model: int, max_len: int=5000):
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)  # shape (1, max_len, d_model)

class TransformerRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.noise_std = INITIAL_NOISE_STD
        self.attn_dropout_rate = DROPOUT_ATTENTION
        self.ff_dropout_rate = DROPOUT_FF

        self.embed = nn.Linear(PCA_COMPONENTS, D_MODEL)
        self.register_buffer('pe', positional_encoding(D_MODEL, WINDOW_SIZE))

        # build encoder layer with bulk dropout, then override specifics
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NUM_HEADS,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=self.ff_dropout_rate,
            activation='relu',
            batch_first=True
        )
        # override attention-dropout probability
        layer.self_attn.dropout = self.attn_dropout_rate
        # separate residual-dropout modules
        layer.dropout1.p = self.attn_dropout_rate  # after attn
        layer.dropout2.p = self.ff_dropout_rate    # after feed-forward

        self.transformer = nn.TransformerEncoder(layer, num_layers=NUM_LAYERS)

        # head uses feed-forward dropout
        self.head_dropout = nn.Dropout(self.ff_dropout_rate)
        self.head = nn.Sequential(
            nn.LayerNorm(D_MODEL),
            self.head_dropout,
            nn.Linear(D_MODEL, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = self.embed(x)
        x = x + self.pe[:, :x.size(1), :]
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        # transformer with batch_first=True
        x = self.transformer(x)
        # take last time-step
        x = x[:, -1, :]
        return self.head(x).squeeze(-1)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load full dataset ---
    df = pd.read_csv(CSV_PATH)
    logging.info(f"Total points: {len(df)}")
    features = [c for c in df.columns if c not in [TS_COL, TARGET_COL]]

    # --- Preprocess full data once ---
    X_all, _, scaler = clean_and_scale(df, df, features)
    X_all_pca, _, pca = apply_pca(X_all, X_all, n_components=PCA_COMPONENTS)
    cols = [f'pca{i}' for i in range(PCA_COMPONENTS)]
    all_df = pd.DataFrame(X_all_pca, columns=cols)
    all_df[TARGET_COL] = df[TARGET_COL].values

    # --- Time-series cross-validation folds ---

    # reuse the same LR schedule function defined earlier
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return float(epoch + 1) / WARMUP_EPOCHS
        t = (epoch - WARMUP_EPOCHS) / (NUM_EPOCHS - WARMUP_EPOCHS)
        return COSINE_ETA_MIN/LEARNING_RATE + (1 - COSINE_ETA_MIN/LEARNING_RATE) * 0.5 * (1 + math.cos(math.pi * t))

    total_len = len(all_df)
    fold_size = total_len // (NUM_FOLDS + 1)

    fold_rs = []
    for fold in range(NUM_FOLDS):
        # define train/val split indices
        train_end = fold_size * (fold + 1)
        val_start = train_end
        val_end = val_start + fold_size

        train_df = all_df.iloc[:train_end].reset_index(drop=True)
        val_df = all_df.iloc[val_start:val_end + WINDOW_SIZE - 1].reset_index(drop=True)

        logging.info(f"Fold {fold+1}/{NUM_FOLDS}: train up to idx {train_end}, val idx [{val_start}:{val_end}]")

        train_loader = DataLoader(
            RollingWindowDataset(train_df, WINDOW_SIZE, cols, TARGET_COL),
            batch_size=BATCH_SIZE, shuffle=True
        )
        val_loader = DataLoader(
            RollingWindowDataset(val_df, WINDOW_SIZE, cols, TARGET_COL),
            batch_size=BATCH_SIZE
        )

        # --- Initialize model per-fold ---
        model = TransformerRegressor().to(device)
        criterion = nn.HuberLoss(delta=HUBER_DELTA)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        best_r = -np.inf
        best_state = None

        # --- Training loop ---
        for epoch in range(NUM_EPOCHS):
            model.train()
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                preds = model(x_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
                optimizer.step()
            scheduler.step()

        # --- Validation ---
        model.eval()
        all_preds, all_trues = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                out = model(x_batch).cpu().numpy()
                all_preds.append(out)
                all_trues.append(y_batch.cpu().numpy())

        preds = np.concatenate(all_preds)
        trues = np.concatenate(all_trues)
        r = pearsonr(trues, preds)[0] if len(preds) > 1 and np.std(preds) > 0 else 0.0
        logging.info(f"Fold {fold+1} Pearson R = {r:.4f}")
        fold_rs.append(r)

    # --- Summary across folds ---
    mean_r = float(np.mean(fold_rs))
    std_r = float(np.std(fold_rs))
    logging.info(f"CV mean Pearson R = {mean_r:.4f} ± {std_r:.4f} across {NUM_FOLDS} folds")

if __name__ == '__main__':
    main()