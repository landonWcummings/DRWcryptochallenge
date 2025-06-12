import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
import logging
import os

from base_utils import (
    RollingWindowDataset,
    load_and_split_random,
    clean_and_scale,
    apply_pca
)

# Setup logging
torch.manual_seed(42)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# --- Configuration ---
CSV_PATH       = r'C:\Users\lando\Desktop\AI\DRWcryptochallenge\train.csv'
TEST_CSV_PATH  = r'C:\Users\lando\Desktop\AI\DRWcryptochallenge\test.csv'
TEST_SEGMENTS  = 20
SEGMENT_FRAC   = 0.01

WINDOW_SIZE_NN = 1       # only current day
PCA_COMPONENTS = 55      # reduced to 40 PCA features
BATCH_SIZE     = 128
LEARNING_RATE  = 1e-3
NUM_EPOCHS     = 54      # transformer training
TARGET_COL     = 'label'
TS_COL         = 'timestamp'
SAVE_DIR       = 'saved_models'

dropout_rate = 0.3
os.makedirs(SAVE_DIR, exist_ok=True)

# --- TabTransformer for numeric features ---
class TabTransformer(nn.Module):
    def __init__(self,
                 num_features,
                 d_model=256,
                 num_layers=2,
                 num_heads=4,
                 dim_ff=256,
                 dropout=dropout_rate):
        super().__init__()
        # embed each numeric feature into d_model dims via linear projection
        self.feature_embed = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, num_heads, dim_ff, dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        # head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * num_features),
            nn.Linear(d_model * num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        # x: (batch, window=1, num_features)
        x = x.squeeze(1)  # -> (batch, num_features)
        # embed features
        tokens = self.feature_embed(x.unsqueeze(-1))  # (batch, num_features, d_model)
        # self-attention across features
        enc = self.transformer(tokens)                # (batch, num_features, d_model)
        flat = enc.flatten(1)                         # (batch, num_features * d_model)
        return self.head(flat).squeeze(-1)

# --- Train/Eval Functions ---
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
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
            preds.append(out.cpu().numpy())
            trues.append(y.cpu().numpy())
            total_loss += criterion(out, y).item()
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    r = pearsonr(trues, preds)[0] if np.std(preds) > 0 else 0.0
    return total_loss / len(loader), r

# --- Main Pipeline ---
def main():
    # Load and split
    df = pd.read_csv(CSV_PATH)
    feature_cols = [c for c in df.columns if c not in [TS_COL, TARGET_COL]]
    df_train, df_test = load_and_split_random(df, TEST_SEGMENTS, SEGMENT_FRAC)

    # Scaling and PCA
    X_tr, X_te, scaler = clean_and_scale(df_train, df_test, feature_cols)
    X_tr_pca, X_te_pca, pca = apply_pca(X_tr, X_te, n_components=PCA_COMPONENTS)

    # --- Simple denoising: clip extreme percentiles ---
    low = np.percentile(X_tr_pca, 2, axis=0)
    high = np.percentile(X_tr_pca, 98, axis=0)
    X_tr_pca = np.clip(X_tr_pca, low, high)
    X_te_pca = np.clip(X_te_pca, low, high)

    # Build DataFrames
    pca_cols = [f'pca{i}' for i in range(PCA_COMPONENTS)]
    df_tr = pd.DataFrame(X_tr_pca, columns=pca_cols)
    df_tr[TARGET_COL] = df_train[TARGET_COL].values
    df_te = pd.DataFrame(X_te_pca, columns=pca_cols)
    df_te[TARGET_COL] = df_test[TARGET_COL].values

    train_loader = DataLoader(
        RollingWindowDataset(df_tr, WINDOW_SIZE_NN, pca_cols, TARGET_COL),
        batch_size=BATCH_SIZE, shuffle=True
    )
    valid_loader = DataLoader(
        RollingWindowDataset(df_te, WINDOW_SIZE_NN, pca_cols, TARGET_COL),
        batch_size=BATCH_SIZE
    )

    # Model instantiation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TabTransformer(num_features=PCA_COMPONENTS).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_r = eval_epoch(model, valid_loader, criterion, device)
        logging.info(f"Epoch {epoch:2d} | TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f} | ValR={val_r:.4f}")

    # Save model
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'tabtransformer_pca40_denoised.pt'))
    logging.info("Saved PCA-40 TabTransformer with clip-denoising model")

if __name__ == '__main__':
    main()
