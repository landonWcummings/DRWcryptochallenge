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
TEST_CSV_PATH = 'C:/Users/lndnc/OneDrive/Desktop/AI/DRWkaggle/test.csv'
TEST_SEGMENTS = 20
SEGMENT_FRAC  = 0.01

WINDOW_SIZE    = 10
PCA_COMPONENTS = 50
BATCH_SIZE     = 128
LEARNING_RATE  = 1e-3
NUM_EPOCHS     = 250
PATIENCE       = 249
MOVING_AVG_WIN = 10

EMBED_DIM   = 128
NHEADS      = 8
FFN_DIM     = 512
NLAYERS     = 8
DROPOUT     = 0.22

# noise scheduling
max_noise = 0.05
noise_decay_epochs = 50

# weight decay
WEIGHT_DECAY = 1e-4

# LR warmup + cosine schedule
warmup_frac = 0.1

TARGET_COL = 'label'
TS_COL     = 'timestamp'

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TransformerRegressor(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.noise_std = max_noise

        self.embed = nn.Linear(in_features, EMBED_DIM)
        self.pos_enc = PositionalEncoding(EMBED_DIM, max_len=WINDOW_SIZE)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=NHEADS,
            dim_feedforward=FFN_DIM,
            dropout=DROPOUT,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=NLAYERS
        )

        # add dropout before the final head
        self.head = nn.Sequential(
            nn.LayerNorm(EMBED_DIM),
            nn.Dropout(0.2),
            nn.Linear(EMBED_DIM, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = self.pos_enc(x)
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x[-1]
        return self.head(x).squeeze(-1)


def train_epoch(model, loader, criterion, optimizer, scheduler, device, epoch):
    model.train()
    # decay noise
    if epoch <= noise_decay_epochs:
        model.noise_std = max_noise * (1 - epoch / noise_decay_epochs)
    else:
        model.noise_std = 0.0

    total = 0.0
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total += loss.item()
    return total / len(loader)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total, preds, trues = 0.0, [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds.append(out.cpu().numpy())
            trues.append(y.cpu().numpy())
            total += criterion(out, y).item()
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    r = pearsonr(trues, preds)[0] if len(trues)>1 and np.std(preds)>0 else 0.0
    return total / len(loader), r


def main():
    # data prep
    df = pd.read_csv(CSV_PATH)
    feature_cols = [c for c in df.columns if c not in [TS_COL, TARGET_COL]]
    df_tr, df_te = load_and_split_random(df, TEST_SEGMENTS, SEGMENT_FRAC)

    X_tr, X_te, scaler = clean_and_scale(df_tr, df_te, feature_cols)
    X_tr_pca, X_te_pca, pca = apply_pca(X_tr, X_te, n_components=PCA_COMPONENTS)
    cols = [f'pca{i}' for i in range(PCA_COMPONENTS)]
    df_train = pd.DataFrame(X_tr_pca, columns=cols)
    df_train[TARGET_COL] = df_tr[TARGET_COL].values
    df_valid = pd.DataFrame(X_te_pca, columns=cols)
    df_valid[TARGET_COL] = df_te[TARGET_COL].values

    train_loader = DataLoader(
        RollingWindowDataset(df_train, WINDOW_SIZE, cols, TARGET_COL),
        batch_size=BATCH_SIZE, shuffle=True
    )
    valid_loader = DataLoader(
        RollingWindowDataset(df_valid, WINDOW_SIZE, cols, TARGET_COL),
        batch_size=BATCH_SIZE
    )

    # model, optimizer, scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerRegressor(in_features=PCA_COMPONENTS).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    total_steps = NUM_EPOCHS * len(train_loader)
    warmup_steps = int(warmup_frac * total_steps)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # training loop
    best_r, patience = -np.inf, 0
    hist_r, hist_loss = [], []

    for epoch in range(1, NUM_EPOCHS+1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch)
        val_loss, val_r = eval_epoch(model, valid_loader, criterion, device)
        hist_r.append(val_r)
        hist_loss.append(val_loss)

        logging.info(f"Epoch {epoch}  TrainLoss:{train_loss:.4f}  ValLoss:{val_loss:.4f}  ValR:{val_r:.4f}")

        if val_r > best_r:
            best_r, patience = val_r, 0
        else:
            patience += 1
            if patience >= PATIENCE:
                logging.info(f"Early stopping at epoch {epoch}")
                break

    logging.info(f"Best Validation Pearson: {best_r:.4f}")

    # plot metrics
    import matplotlib.pyplot as plt
    epochs = np.arange(1, len(hist_r)+1)
    plt.figure()
    plt.plot(epochs, hist_r, label='Val Pearson R')
    mav = np.convolve(hist_loss, np.ones(MOVING_AVG_WIN)/MOVING_AVG_WIN, mode='valid')
    plt.plot(epochs[MOVING_AVG_WIN-1:], mav, label=f'{MOVING_AVG_WIN}-epoch MA Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()
    plt.title('Validation Pearson & MA Loss')
    plt.savefig('training.png')
    plt.close()
    logging.info("Saved training.png")

    # inference & submission
    test_df = pd.read_csv(TEST_CSV_PATH)
    X_test = scaler.transform(test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0))
    X_test_pca = pca.transform(X_test)
    df_test_pca = pd.DataFrame(X_test_pca, columns=cols); df_test_pca[TARGET_COL]=0
    test_loader = DataLoader(
        RollingWindowDataset(df_test_pca, WINDOW_SIZE, cols, TARGET_COL),
        batch_size=BATCH_SIZE
    )

    preds = [0.0]*(WINDOW_SIZE-1)
    model.eval()
    with torch.no_grad():
        for x, _ in test_loader:
            preds.extend(model(x.to(device)).cpu().tolist())
    preds = preds[:len(test_df)]

    pd.DataFrame({'ID': np.arange(1, len(preds)+1), 'prediction': preds}) \
      .to_csv('transformer_submission.csv', index=False)
    logging.info("Saved transformer_submission.csv")


if __name__ == '__main__':
    main()
