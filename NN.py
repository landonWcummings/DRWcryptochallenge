import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
import logging

from base_utils import (
    RollingWindowDataset,
    load_and_split_random,
    clean_and_scale,
    apply_pca
)

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# --- Configuration ---
CSV_PATH      = 'C:/Users/lndnc/OneDrive/Desktop/AI/DRWkaggle/train.csv'
TEST_CSV_PATH = 'C:/Users/lndnc/OneDrive/Desktop/AI/DRWkaggle/test.csv'
TEST_SEGMENTS = 20
SEGMENT_FRAC  = 0.01

WINDOW_SIZE_NN = 2       # use 2 days: previous + current
PCA_COMPONENTS = 30
BATCH_SIZE     = 128
LEARNING_RATE  = 1e-3
NUM_EPOCHS     = 250
PATIENCE       = 249
MOVING_AVG_WIN = 10

TARGET_COL = 'label'
TS_COL     = 'timestamp'

# --- Baseline Feedforward Model ---
class BaselineNN(nn.Module):
    def __init__(self, input_dim, noise_std=0.01, dropout=0.1):
        super().__init__()
        self.noise_std = noise_std
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        return self.net(x).squeeze(-1)

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
    if len(trues) > 1 and np.std(trues) > 0 and np.std(preds) > 0:
        r = pearsonr(trues, preds)[0]
    else:
        r = 0.0
    return total_loss / len(loader), r

# --- Main Pipeline ---
def main():
    df = pd.read_csv(CSV_PATH)
    feature_cols = [c for c in df.columns if c not in [TS_COL, TARGET_COL]]
    df_train, df_test = load_and_split_random(df, TEST_SEGMENTS, SEGMENT_FRAC)

    X_tr, X_te, scaler = clean_and_scale(df_train, df_test, feature_cols)
    X_tr_pca, X_te_pca, pca = apply_pca(X_tr, X_te, n_components=PCA_COMPONENTS)
    cols30 = [f'pca{i}' for i in range(PCA_COMPONENTS)]
    df_tr = pd.DataFrame(X_tr_pca, columns=cols30)
    df_tr[TARGET_COL] = df_train[TARGET_COL].values
    df_te = pd.DataFrame(X_te_pca, columns=cols30)
    df_te[TARGET_COL] = df_test[TARGET_COL].values

    train_loader = DataLoader(
        RollingWindowDataset(df_tr, WINDOW_SIZE_NN, cols30, TARGET_COL),
        batch_size=BATCH_SIZE, shuffle=True
    )
    valid_loader = DataLoader(
        RollingWindowDataset(df_te, WINDOW_SIZE_NN, cols30, TARGET_COL),
        batch_size=BATCH_SIZE
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = WINDOW_SIZE_NN * PCA_COMPONENTS
    model = BaselineNN(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_r, patience = -np.inf, 0
    val_r_history, val_loss_history = [], []
    for epoch in range(1, NUM_EPOCHS+1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_r = eval_epoch(model, valid_loader, criterion, device)
        val_r_history.append(val_r)
        val_loss_history.append(val_loss)
        logging.info(f"Epoch {epoch} TrainLoss:{train_loss:.4f} ValLoss:{val_loss:.4f} ValR:{val_r:.4f}")
        if val_r > best_r:
            best_r, patience = val_r, 0
        else:
            patience += 1
            if patience >= PATIENCE:
                logging.info(f"Early stopping at epoch {epoch}")
                break

    logging.info(f"Best Validation Pearson: {best_r:.4f}")

    # Plot Pearson and moving average of validation loss
    import matplotlib.pyplot as plt
    plt.figure()
    epochs = np.arange(1, len(val_r_history)+1)
    plt.plot(epochs, val_r_history, label='Val Pearson R')
    # moving average
    mav = np.convolve(val_loss_history, np.ones(MOVING_AVG_WIN)/MOVING_AVG_WIN, mode='valid')
    plt.plot(epochs[MOVING_AVG_WIN-1:], mav, label=f'{MOVING_AVG_WIN}-epoch MA Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()
    plt.title('Validation Pearson & MA Loss')
    plt.savefig('training.png')
    plt.close()
    logging.info("Saved training.png")

    test_df = pd.read_csv(TEST_CSV_PATH)
    test_X = scaler.transform(test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0))
    test_pca = pca.transform(test_X)
    df_test_pca = pd.DataFrame(test_pca, columns=cols30)
    df_test_pca[TARGET_COL] = 0.0
    test_loader = DataLoader(
        RollingWindowDataset(df_test_pca, WINDOW_SIZE_NN, cols30, TARGET_COL),
        batch_size=BATCH_SIZE
    )
    preds = [0.0] * (WINDOW_SIZE_NN - 1)
    model.eval()
    with torch.no_grad():
        for x, _ in test_loader:
            preds.extend(model(x.to(device)).cpu().tolist())
    preds = preds[:len(test_df)]

    submission = pd.DataFrame({'ID': np.arange(1, len(preds)+1), 'prediction': preds})
    submission.to_csv('baseline_nn_submission.csv', index=False)
    logging.info("Saved baseline_nn_submission.csv")

if __name__ == '__main__':
    main()