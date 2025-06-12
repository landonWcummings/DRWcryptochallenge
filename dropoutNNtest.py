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
CSV_PATH      = r'C:\Users\lando\Desktop\AI\DRWcryptochallenge\train.csv'
TEST_CSV_PATH = r'C:\Users\lando\Desktop\AI\DRWcryptochallenge\test.csv'
TEST_SEGMENTS = 20
SEGMENT_FRAC  = 0.01

WINDOW_SIZE_NN = 3       # use 2 days: previous + current
PCA_COMPONENTS = 32
BATCH_SIZE     = 128
LEARNING_RATE  = 1e-3
NUM_EPOCHS     = 300     # fixed number of epochs
MOVING_AVG_WIN = 10      # window for moving average of Pearson R
DROPOUT_START  = 0.6
DROPOUT_END    = 0.5
DECAY_EPOCHS   = 300     # epochs over which to anneal dropout

NOISE_STD      = 0.06    # Gaussian noise std for input
NUM_RUNS       = 5       # number of training runs to average

TARGET_COL = 'label'
TS_COL     = 'timestamp'

# --- Baseline Feedforward Model ---
class BaselineNN(nn.Module):
    def __init__(self, input_dim, noise_std=0.01, dropout=0.6):
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
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds.append(out.cpu().numpy())
            trues.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    if len(trues) > 1 and np.std(trues) > 0 and np.std(preds) > 0:
        r = pearsonr(trues, preds)[0]
    else:
        r = 0.0
    return r

# --- Single Run Pipeline ---
def run_training():
    # data prep
    df = pd.read_csv(CSV_PATH)
    feature_cols = [c for c in df.columns if c not in [TS_COL, TARGET_COL]]
    df_train, df_test = load_and_split_random(df, TEST_SEGMENTS, SEGMENT_FRAC)

    X_tr, X_te, scaler = clean_and_scale(df_train, df_test, feature_cols)
    X_tr_pca, X_te_pca, pca = apply_pca(X_tr, X_te, n_components=PCA_COMPONENTS)
    cols_pca = [f'pca{i}' for i in range(PCA_COMPONENTS)]
    df_tr = pd.DataFrame(X_tr_pca, columns=cols_pca)
    df_tr[TARGET_COL] = df_train[TARGET_COL].values
    df_te = pd.DataFrame(X_te_pca, columns=cols_pca)
    df_te[TARGET_COL] = df_test[TARGET_COL].values

    train_loader = DataLoader(
        RollingWindowDataset(df_tr, WINDOW_SIZE_NN, cols_pca, TARGET_COL),
        batch_size=BATCH_SIZE, shuffle=True
    )
    valid_loader = DataLoader(
        RollingWindowDataset(df_te, WINDOW_SIZE_NN, cols_pca, TARGET_COL),
        batch_size=BATCH_SIZE
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = WINDOW_SIZE_NN * PCA_COMPONENTS
    model = BaselineNN(input_dim, noise_std=NOISE_STD, dropout=DROPOUT_START).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    val_r_history = []
    # training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        # dropout annealing
        if epoch <= DECAY_EPOCHS:
            p = DROPOUT_START - (DROPOUT_START - DROPOUT_END) * ((epoch - 1) / (DECAY_EPOCHS - 1))
        else:
            p = DROPOUT_END
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = p

        _ = train_epoch(model, train_loader, criterion, optimizer, device)
        val_r = eval_epoch(model, valid_loader, criterion, device)
        val_r_history.append(val_r)
        logging.info(f"Run Epoch {epoch}/{NUM_EPOCHS} - Val Pearson R: {val_r:.4f}")

    return val_r_history, scaler, pca, cols_pca

# --- Main Pipeline ---
def main():
    all_runs = []
    # repeat training
    for run in range(NUM_RUNS):
        logging.info(f"Starting run {run+1}/{NUM_RUNS}")
        val_r_history, scaler, pca, cols_pca = run_training()
        all_runs.append(val_r_history)

    # compute average Pearson R per epoch
    avg_r = np.mean(np.vstack(all_runs), axis=0)

    # compute moving average
    mav = np.convolve(avg_r, np.ones(MOVING_AVG_WIN)/MOVING_AVG_WIN, mode='valid')

    # --- Plot results ---
    import matplotlib.pyplot as plt
    epochs = np.arange(1, len(avg_r)+1)
    plt.figure()
    plt.plot(epochs, avg_r, label='Avg Val Pearson R')
    plt.plot(epochs[MOVING_AVG_WIN-1:], mav, label=f'{MOVING_AVG_WIN}-Epoch MA')
    plt.xlabel('Epoch')
    plt.ylabel('Pearson R')
    plt.legend()
    plt.title('Average Validation Pearson R over Runs')
    plt.savefig('avg_val_pearson_r4.png')
    plt.close()
    logging.info("Saved avg_val_pearson_r.png")

    # --- Inference on held-out test set using last run's scaler and pca ---
    test_df = pd.read_csv(TEST_CSV_PATH)
    feature_cols = [c for c in test_df.columns if c not in [TS_COL, TARGET_COL]]
    test_X = scaler.transform(test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0))
    test_pca = pca.transform(test_X)
    df_test_pca = pd.DataFrame(test_pca, columns=cols_pca)
    df_test_pca[TARGET_COL] = 0.0
    test_loader = DataLoader(
        RollingWindowDataset(df_test_pca, WINDOW_SIZE_NN, cols_pca, TARGET_COL),
        batch_size=BATCH_SIZE
    )

    preds = [0.0] * (WINDOW_SIZE_NN - 1)
    model = None  # reuse last model if needed
    with torch.no_grad():
        for x, _ in test_loader:
            preds.extend(model(x.to(device)).cpu().tolist())
    preds = preds[:len(test_df)]

    submission = pd.DataFrame({'ID': np.arange(1, len(preds) + 1), 'prediction': preds})
    submission.to_csv('nn_submission.csv', index=False)
    logging.info("Saved nn_submission.csv")

if __name__ == '__main__':
    main()
