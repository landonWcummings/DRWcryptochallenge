import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import pearsonr
import logging

from base_utils import (
    RollingWindowDataset,
    clean_and_scale,
    apply_pca,
    load_and_split_random,
)

# Transformer‐specific configuration
BATCH_SIZE      = 128
LEARNING_RATE   = 1e-3
NUM_EPOCHS      = 5
PATIENCE        = 10
PCA_COMPONENTS  = 30
PROJ_DIM        = 256
D_MODEL         = 128
NOISE_STD       = 0.01
WINDOW_SIZE     = 60
TARGET_COL      = 'label'
TS_COL = 'timestamp'
CSV_PATH        = 'C:/Users/lndnc/OneDrive/Desktop/AI/DRWkaggle/train.csv'
TEST_CSV_PATH   = 'C:/Users/lndnc/OneDrive/Desktop/AI/DRWkaggle/test.csv'

# --- Train/Eval Helpers --------------------------------
def train_epoch(model, loader, opt, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds.append(out.cpu())
            trues.append(y.cpu())
            total_loss += criterion(out, y).item()
    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()
    return total_loss / len(loader), pearsonr(trues, preds)[0]

# --- Positional Encoding --------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, T, D)
        seq_len = x.size(1)
        return x + self.pe[:seq_len].unsqueeze(0)

# --- Pearson Loss --------------------------------------
class PearsonLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, preds, trues):
        preds = preds.view(-1)
        trues = trues.view(-1)
        pm = preds - preds.mean()
        tm = trues - trues.mean()
        num = (pm * tm).sum()
        den = torch.sqrt((pm**2).sum() * (tm**2).sum())
        if den.item() < self.eps:
            return torch.tensor(1.0, device=preds.device)
        r = num / den.clamp(min=self.eps)
        r = torch.clamp(r, -1.0, 1.0)
        return 1 - r

# --- Transformer Model ---------------------------------
class StockTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj1 = nn.Linear(PCA_COMPONENTS, PROJ_DIM)
        self.proj2 = nn.Linear(PROJ_DIM, D_MODEL)
        self.pos_enc = PositionalEncoding(D_MODEL)
        encoder_layer = nn.TransformerEncoderLayer(
            D_MODEL, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.pool = lambda x: x[:, -1, :]
        self.head = nn.Sequential(
            nn.Linear(D_MODEL, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        self.noise_std = NOISE_STD

    def forward(self, x):
        # x: (B, T, 30)
        x = torch.relu(self.proj1(x))
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        x = torch.relu(self.proj2(x))
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = self.pool(x)
        return self.head(x).squeeze(-1)

# --- Main Training + Inference ------------------------
def train_transformer():
    df = pd.read_csv(CSV_PATH)
    # 1) Randomly split into train/test segments
    df_train, df_test = load_and_split_random(df,
                                              test_segments=20,
                                              segment_frac=0.01,
                                              seed=42)

    # 2) Clean + scale
    feature_cols = [c for c in df.columns if c not in [TS_COL, TARGET_COL]]
    X_tr, X_te, scaler = clean_and_scale(df_train, df_test, feature_cols)

    # 3) PCA → 30 dims
    X_tr_pca, X_te_pca, pca = apply_pca(X_tr, X_te, n_components=PCA_COMPONENTS)
    cols30 = [f'pca{i}' for i in range(PCA_COMPONENTS)]
    df_tr = pd.DataFrame(X_tr_pca, columns=cols30)
    df_tr[TARGET_COL] = df_train[TARGET_COL].values
    df_te = pd.DataFrame(X_te_pca, columns=cols30)
    df_te[TARGET_COL] = df_test[TARGET_COL].values

    # 4) Build loaders
    train_loader = DataLoader(
        RollingWindowDataset(df_tr, WINDOW_SIZE, cols30, TARGET_COL),
        batch_size=BATCH_SIZE, shuffle=True
    )
    valid_loader = DataLoader(
        RollingWindowDataset(df_te, WINDOW_SIZE, cols30, TARGET_COL),
        batch_size=BATCH_SIZE
    )

    # 5) Model, optimizer, loss
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model    = StockTransformer().to(device)
    optimizer= optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion= PearsonLoss()

    # 6) Train with early stopping
    best_val, patience = float('inf'), 0
    for epoch in range(1, NUM_EPOCHS+1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_r = eval_epoch(model, valid_loader, criterion, device)
        logging.info(f"Epoch {epoch} TrainL:{train_loss:.4f} ValL:{val_loss:.4f} ValR:{val_r:.4f}")
        if val_loss < best_val:
            best_val, patience = val_loss, 0
        else:
            patience += 1
            if patience >= PATIENCE:
                logging.info(f"Early stopping at epoch {epoch}")
                break

    # Final test correlation
    _, final_r = eval_epoch(model, valid_loader, criterion, device)
    logging.info(f"Final Test Pearson: {final_r:.4f}")

    # 7) Inference on test.csv
    test_df = pd.read_csv(TEST_CSV_PATH)
    test_df[feature_cols] = (
        scaler
        .fit_transform(test_df[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0))
    )
    test_pca = pca.transform(test_df[feature_cols])
    df_test_pca = pd.DataFrame(test_pca, columns=cols30)
    df_test_pca[TARGET_COL] = 0.0

    test_loader = DataLoader(
        RollingWindowDataset(df_test_pca, WINDOW_SIZE, cols30, TARGET_COL),
        batch_size=BATCH_SIZE
    )

    preds = [0.0] * (WINDOW_SIZE - 1)
    model.eval()
    with torch.no_grad():
        for x, _ in test_loader:
            preds.extend(model(x.to(device)).cpu().tolist())
    preds = preds[:len(test_df)]

    submission = pd.DataFrame({'ID': np.arange(1, len(preds)+1),
                               'prediction': preds})
    submission.to_csv('transformer_submission.csv', index=False)
    logging.info("Saved transformer_submission.csv")

if __name__ == '__main__':
    train_transformer()
