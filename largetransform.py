import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.stats import pearsonr
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# --- Configuration ---
CSV_PATH = 'C:/Users/lndnc/OneDrive/Desktop/AI/DRWkaggle/train.csv'
TEST_CSV_PATH = 'C:/Users/lndnc/OneDrive/Desktop/AI/DRWkaggle/test.csv'
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WINDOW_SIZE = 60  # Use 60 timesteps of OHLCV features only
FEATURE_COLS = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
TARGET_COL = 'label'
INITIAL_TRAIN_ROWS = 368000
MIN_TEST_ROWS = 128
MAX_ROLLS = 30
NUM_EPOCHS = 6
PATIENCE = 3

# --- Dataset ---
class RollingWindowDataset(Dataset):
    def __init__(self, df, window_size, feature_cols, target_col):
        self.window_size = window_size
        self.features = df[feature_cols].values.astype(np.float32)
        self.targets = df[target_col].values.astype(np.float32)

    def __len__(self):
        return len(self.features) - self.window_size + 1

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.window_size]      # (60,5)
        y = self.targets[idx + self.window_size - 1]         # scalar
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)

# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x):  # x: (T, B, D)
        return x + self.pe[:x.size(0)].unsqueeze(1)

# --- Pearson Loss ---
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
        den = torch.sqrt((pm**2).sum() * (tm**2).sum()).clamp(min=self.eps)
        r = num / den
        return 1 - r

# --- Transformer Model ---
class StockTransformer(nn.Module):
    def __init__(self, feature_dim=5, d_model=128, nhead=4, num_layers=6, dim_ff=256, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(feature_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers)
        self.pool = lambda x: x[-1]                 # causal pooling
        self.head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64,1)
        )

    def forward(self, x):
        # x: (B, T, F)
        x = x.transpose(0,1)               # (T, B, F)
        x = self.embed(x)                 # (T, B, D)
        x = self.pos_enc(x)
        x = self.transformer(x)           # (T, B, D)
        x = self.pool(x)                  # (B, D)
        return self.head(x).squeeze(-1)   # (B,)

# --- Train & Validate Functions ---
def train_epoch(model, loader, opt, criterion, device):
    model.train()
    total_loss = 0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    preds, trues = [], []
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            preds.append(out.cpu())
            trues.append(y.cpu())
            total_loss += criterion(out, y).item()
    preds = torch.cat(preds)
    trues = torch.cat(trues)
    return total_loss/len(loader), pearsonr(trues.numpy(), preds.numpy())[0]

# --- Main ---
def main():
    # Load & preprocess
    df = pd.read_csv(CSV_PATH)
    feature_cols = FEATURE_COLS
    df[TARGET_COL] = df[TARGET_COL].astype(float)

    # Rolling config
    total = len(df) - WINDOW_SIZE + 1
    samples_after = total - INITIAL_TRAIN_ROWS
    folds = min(MAX_ROLLS, samples_after // max(1, MIN_TEST_ROWS - WINDOW_SIZE + 1))
    step = max(samples_after//folds, BATCH_SIZE)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StockTransformer().to(device)
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = PearsonLoss()

    all_preds, all_trues = [], []
    for fold in range(folds):
        start = INITIAL_TRAIN_ROWS + fold*step
        end = start + step
        df_tr = df.iloc[:start+WINDOW_SIZE-1].copy()
        df_te = df.iloc[start:end+WINDOW_SIZE-1].copy()

        # scale
        scaler = StandardScaler().fit(df_tr[feature_cols])
        df_tr[feature_cols] = scaler.transform(df_tr[feature_cols])
        df_te[feature_cols] = scaler.transform(df_te[feature_cols])

        tr_loader = DataLoader(RollingWindowDataset(df_tr, WINDOW_SIZE, feature_cols, TARGET_COL), batch_size=BATCH_SIZE, shuffle=True)
        te_loader = DataLoader(RollingWindowDataset(df_te, WINDOW_SIZE, feature_cols, TARGET_COL), batch_size=BATCH_SIZE)

        best_val=1e9; patience=0
        for epoch in range(1, NUM_EPOCHS+1):
            tr_loss = train_epoch(model, tr_loader, opt, criterion, device)
            val_loss, val_r = eval_epoch(model, te_loader, criterion, device)
            logging.info(f"Fold{fold+1} Ep{epoch} TrainL:{tr_loss:.4f} ValL:{val_loss:.4f} ValR:{val_r:.4f}")
            if val_loss<best_val: best_val=val_loss; patience=0
            else: patience+=1
            if patience>=PATIENCE: logging.info(f"Early stop fold{fold+1} at ep{epoch}"); break

        # final fold preds
        _, fold_r = eval_epoch(model, te_loader, criterion, device)
        all_trues.extend(RollingWindowDataset(df_te,WINDOW_SIZE,feature_cols,TARGET_COL).__getitem__(i)[1].item() for i in range(len(RollingWindowDataset(df_te,WINDOW_SIZE,feature_cols,TARGET_COL))))
        all_preds.extend(model(torch.stack([RollingWindowDataset(df_te,WINDOW_SIZE,feature_cols,TARGET_COL).__getitem__(i)[0] for i in range(len(RollingWindowDataset(df_te,WINDOW_SIZE,feature_cols,TARGET_COL)))]).to(device)).cpu().tolist())
        logging.info(f"Fold{fold+1} Pearson:{fold_r:.4f}")

    overall_r = pearsonr(all_trues, all_preds)[0]
    logging.info(f"OverallPearson:{overall_r:.4f}")

    # Inference
    test_df = pd.read_csv(TEST_CSV_PATH)
    preds = [0.0]*(WINDOW_SIZE-1)
    test_df_scaled = test_df.copy()
    test_df_scaled[feature_cols] = scaler.transform(test_df[feature_cols])
    test_ds = RollingWindowDataset(test_df_scaled, WINDOW_SIZE, feature_cols, TARGET_COL)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    model.eval()
    with torch.no_grad():
        for x,_ in test_loader:
            preds.extend(model(x.to(device)).cpu().tolist())
    preds = preds[:len(test_df)]
    sub = pd.DataFrame({'ID':np.arange(1,len(preds)+1),'prediction':preds})
    sub.to_csv('transformer_submission.csv', index=False)
    logging.info("Saved transformer_submission.csv")

if __name__=='__main__':
    main()
