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
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# --- Configuration ---
CSV_PATH       = r'C:\Users\lando\Desktop\AI\DRWcryptochallenge\train.csv'
TEST_CSV_PATH  = r'C:\Users\lando\Desktop\AI\DRWcryptochallenge\test.csv'
TEST_SEGMENTS  = 20
SEGMENT_FRAC   = 0.01

WINDOW_SIZE_NN = 1       # use 2 days: previous + current
PCA_COMPONENTS = 50
BATCH_SIZE     = 128
LEARNING_RATE  = 1e-3
NUM_EPOCHS     = 54      # train exactly 54 epochs
MOVING_AVG_WIN = 10      # window for moving average of Pearson r

DROPOUT_START  = 0.6
DROPOUT_END    = 0.5
DECAY_EPOCHS   = 150     # over how many epochs to decay (we only run 54)

NOISE_STD      = 0.05    # Gaussian noise std for input

TARGET_COL = 'label'
TS_COL     = 'timestamp'
SAVE_DIR   = 'saved_models'

os.makedirs(SAVE_DIR, exist_ok=True)

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
    # --- Data Preparation ---
    df = pd.read_csv(CSV_PATH)
    feature_cols = [c for c in df.columns if c not in [TS_COL, TARGET_COL]]
    df_train, df_test = load_and_split_random(df, TEST_SEGMENTS, SEGMENT_FRAC)

    X_tr, X_te, scaler = clean_and_scale(df_train, df_test, feature_cols)
    X_tr_pca, X_te_pca, pca = apply_pca(X_tr, X_te, n_components=PCA_COMPONENTS)
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

    # --- Model, Loss, Optimizer ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = WINDOW_SIZE_NN * PCA_COMPONENTS
    model = BaselineNN(input_dim, noise_std=NOISE_STD, dropout=DROPOUT_START).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training Loop with Dropout Decay & Model Saving ---
    val_r_history = []
    for epoch in range(1, NUM_EPOCHS + 1):
        # linear decay of dropout p
        if epoch <= DECAY_EPOCHS:
            p = DROPOUT_START - (DROPOUT_START - DROPOUT_END) * ((epoch - 1) / (DECAY_EPOCHS - 1))
        else:
            p = DROPOUT_END
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = p

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_r = eval_epoch(model, valid_loader, criterion, device)
        val_r_history.append(val_r)

        logging.info(f"Epoch {epoch:2d} | p={p:.3f} | TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f} | ValR={val_r:.4f}")

        # save model for epochs 48 through 54
        if 48 <= epoch <= 54:
            path = os.path.join(SAVE_DIR, f'model_epoch{epoch}.pt')
            torch.save(model.state_dict(), path)
            logging.info(f"Saved {path}")

    # --- Select Best Model from 48–54 by Val Pearson ---
    # val_r_history is length 54; index 47→epoch48 ... index 53→epoch54
    recent_rs = val_r_history[47:54]
    best_rel_idx = int(np.argmax(recent_rs))  # 0..6
    best_epoch = 48 + best_rel_idx
    best_path = os.path.join(SAVE_DIR, f'model_epoch{best_epoch}.pt')
    logging.info(f"Loading best model from epoch {best_epoch} with ValR={recent_rs[best_rel_idx]:.4f}")

    # reload model weights
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    # --- Inference on Full Test CSV ---
    test_df = pd.read_csv(TEST_CSV_PATH)
    X_all = scaler.transform(
        test_df[feature_cols]
               .replace([np.inf, -np.inf], np.nan)
               .fillna(0)
    )
    pca_all = pca.transform(X_all)
    df_all_pca = pd.DataFrame(pca_all, columns=pca_cols)
    df_all_pca[TARGET_COL] = 0.0
    test_loader = DataLoader(
        RollingWindowDataset(df_all_pca, WINDOW_SIZE_NN, pca_cols, TARGET_COL),
        batch_size=BATCH_SIZE
    )

    # collect predictions from each saved model
    all_preds = []
    for epoch in range(48, 55):  # 48,49,…,54
        model_path = os.path.join(SAVE_DIR, f'model_epoch{epoch}.pt')
        if not os.path.isfile(model_path):
            logging.warning(f"Missing checkpoint: {model_path}")
            continue

        # load weights and run inference
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        preds = []
        with torch.no_grad():
            # pad first WINDOW_SIZE_NN-1 entries with zeros (or np.nan)
            preds = [0.0] * (WINDOW_SIZE_NN - 1)
            for x, _ in test_loader:
                x = x.to(device)
                out = model(x).cpu().tolist()
                preds.extend(out)
            preds = preds[:len(test_df)]

        all_preds.append(preds)
        logging.info(f"Collected predictions from epoch {epoch}")

    # average across models
    if not all_preds:
        raise RuntimeError("No model checkpoints found for ensembling!")
    ensemble_preds = np.mean(np.stack(all_preds, axis=0), axis=0)

    # write submission
    submission = pd.DataFrame({
        'ID': np.arange(1, len(ensemble_preds) + 1),
        'prediction': ensemble_preds
    })
    submission.to_csv('NN_submission.csv', index=False)
    logging.info("Saved NN_submission.csv with ensemble of epochs 48–54")


if __name__ == '__main__':
    main()
