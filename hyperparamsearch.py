import os
import math
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

import optuna
from optuna.pruners import MedianPruner

# import user utilities
from base_utils import RollingWindowDataset, clean_and_scale, apply_pca

# --------------------------------------------------------------------------
#  Configuration (no command-line inputs)
# --------------------------------------------------------------------------
N_TRIALS = 100
STUDY_NAME = 'crypto_transformer_optuna'
CSV_PATH = r'C:/Users/lndnc/OneDrive/Desktop/AI/DRWkaggle/train.csv'
TARGET_COL = 'label'
TS_COL = 'timestamp'
MAX_SEQ_LEN = 20
LOG_PATH = Path('study_log.csv')

# Define columns for CSV log: metrics + hyperparameters
CSV_COLUMNS = [
    'trial_number', 'mean_pearson', 'std_pearson',
    'window_size','pca_components','batch_size',
    'd_model','dim_ff','num_layers',
    'dropout_attn','dropout_ff','initial_noise_std',
    'lr','weight_decay','clip_grad_norm',
    'warmup_epochs','cosine_eta_min','num_epochs','loss_fn','num_heads'
]

# Create log file header if missing
if not LOG_PATH.exists():
    LOG_PATH.write_text(','.join(CSV_COLUMNS) + '\n')

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# --------------------------------------------------------------------------
# Positional encoding
# --------------------------------------------------------------------------
def positional_encoding(d_model: int, max_len: int = MAX_SEQ_LEN):
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)

# --------------------------------------------------------------------------
# Differentiable Pearson loss
# --------------------------------------------------------------------------
class PearsonLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    def forward(self, pred, target):
        pred = pred - pred.mean()
        target = target - target.mean()
        corr = (pred * target).sum() / (
            torch.sqrt((pred ** 2).sum() + self.eps) *
            torch.sqrt((target ** 2).sum() + self.eps)
        )
        return 1.0 - corr

# --------------------------------------------------------------------------
# Transformer Regressor
# --------------------------------------------------------------------------
class TransformerRegressor(nn.Module):
    def __init__(self,
                 pca_components: int,
                 d_model: int,
                 num_heads: int,
                 dim_ff: int,
                 num_layers: int,
                 dropout_attn: float,
                 dropout_ff: float,
                 initial_noise_std: float,
                 window_size: int):
        super().__init__()
        self.noise_std = initial_noise_std
        self.attn_dropout_rate = dropout_attn
        self.ff_dropout_rate = dropout_ff
        self.embed = nn.Linear(pca_components, d_model)
        self.register_buffer('pe', positional_encoding(d_model, MAX_SEQ_LEN))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_ff,
            dropout=self.ff_dropout_rate,
            activation='relu',
            batch_first=True
        )
        layer.self_attn.dropout = self.attn_dropout_rate
        layer.dropout1.p = self.attn_dropout_rate
        layer.dropout2.p = self.ff_dropout_rate
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(self.ff_dropout_rate),
            nn.Linear(d_model, 1)
        )
    def forward(self, x):
        x = self.embed(x)
        x = x + self.pe[:, :x.size(1), :]
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        x = self.transformer(x)
        x = x[:, -1, :]
        return self.head(x).squeeze(-1)

# --------------------------------------------------------------------------
# Single training run
# --------------------------------------------------------------------------
def train_one_run(params, train_loader, val_loader, device, run_seed):
    torch.manual_seed(run_seed)
    np.random.seed(run_seed)
    random.seed(run_seed)

    model = TransformerRegressor(
        pca_components=params['pca_components'],
        d_model=params['d_model'],
        num_heads=params['num_heads'],
        dim_ff=params['dim_ff'],
        num_layers=params['num_layers'],
        dropout_attn=params['dropout_attn'],
        dropout_ff=params['dropout_ff'],
        initial_noise_std=params['initial_noise_std'],
        window_size=params['window_size']
    ).to(device)

    # choose loss
    if params['loss_fn'] == 'huber':
        criterion = nn.HuberLoss(delta=1.0)
    elif params['loss_fn'] == 'mse':
        criterion = nn.MSELoss()
    elif params['loss_fn'] == 'rmse':
        criterion = lambda p, t: torch.sqrt(nn.functional.mse_loss(p, t) + 1e-8)
    else:
        criterion = PearsonLoss()

    optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    def lr_lambda(epoch):
        if epoch < params['warmup_epochs']:
            return float(epoch + 1) / params['warmup_epochs']
        t = (epoch - params['warmup_epochs']) / max(1, params['num_epochs'] - params['warmup_epochs'])
        return params['cosine_eta_min']/params['lr'] + (1 - params['cosine_eta_min']/params['lr']) * 0.5 * (1 + math.cos(math.pi * t))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    for epoch in range(params['num_epochs']):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), params['clip_grad_norm'])
            optimizer.step()
        scheduler.step()

        # quick check for negative
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb).cpu().numpy()
                preds.append(out); trues.append(yb.cpu().numpy())
        preds = np.concatenate(preds); trues = np.concatenate(trues)
        r = pearsonr(trues, preds)[0] if preds.std() > 0 else 0.0
        if r < 0:
            return -0.2

    # final Pearson
    preds, trues = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb).cpu().numpy()
            preds.append(out); trues.append(yb.cpu().numpy())
    return pearsonr(np.concatenate(trues), np.concatenate(preds))[0]

# --------------------------------------------------------------------------
# Optuna objective
# --------------------------------------------------------------------------
def objective(trial):
    params = {
        'window_size': trial.suggest_int('window_size', 2, 20),
        'pca_components': trial.suggest_int('pca_components', 25, 50),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128]),
        'd_model': trial.suggest_categorical('d_model', [128, 256, 512]),
        'dim_ff': trial.suggest_categorical('dim_ff', [256, 512, 1024]),
        'num_layers': trial.suggest_int('num_layers', 2, 8),
        'dropout_attn': trial.suggest_float('dropout_attn', 0.0, 0.7),
        'dropout_ff': trial.suggest_float('dropout_ff', 0.0, 0.7),
        'initial_noise_std': 0.0 if trial.suggest_categorical('use_noise', [0, 1]) == 0 else trial.suggest_float('noise_std', 0.0, 0.1),
        'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
        'clip_grad_norm': trial.suggest_float('clip_grad_norm', 0.1, 2.0),
        'warmup_epochs': trial.suggest_int('warmup_epochs', 1, 4),
        'cosine_eta_min': trial.suggest_float('cosine_eta_min', 1e-8, 1e-4, log=True),
        # 50% chance num_epochs = 1, else sample 2-7
        'num_epochs': 1 if trial.suggest_categorical('one_epoch', [0, 1]) == 1 else trial.suggest_int('num_epochs', 2, 7),
        'loss_fn': trial.suggest_categorical('loss_fn', ['huber', 'mse', 'rmse', 'pearson'])
    }
    possible_heads = [h for h in [2,4,8,16] if params['d_model'] % h == 0]
    params['num_heads'] = trial.suggest_categorical('num_heads', possible_heads)

    # load and preprocess data
    df = pd.read_csv(CSV_PATH)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx - params['window_size'] + 1:].reset_index(drop=True)
    X_train, X_val, _ = clean_and_scale(train_df, val_df, [c for c in df.columns if c not in [TS_COL, TARGET_COL]])
    X_train, X_val, _ = apply_pca(X_train, X_val, n_components=params['pca_components'])
    cols = [f'pca{i}' for i in range(params['pca_components'])]
    train_proc = pd.DataFrame(X_train, columns=cols)
    train_proc[TARGET_COL] = train_df[TARGET_COL].values
    val_proc = pd.DataFrame(X_val, columns=cols)
    val_proc[TARGET_COL] = val_df[TARGET_COL].values

    train_loader = DataLoader(RollingWindowDataset(train_proc, params['window_size'], cols, TARGET_COL),
                              batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(RollingWindowDataset(val_proc, params['window_size'], cols, TARGET_COL),
                            batch_size=params['batch_size'], shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    run_rs = []
    for k in range(4):
        r = train_one_run(params, train_loader, val_loader, device, run_seed=42+k)
        run_rs.append(r)
        trial.report(np.mean(run_rs), k)
        if trial.should_prune():
            raise optuna.TrialPruned()

    mean_r = float(np.mean(run_rs))
    std_r = float(np.std(run_rs))

    # log metrics + hyperparams
    row = [
        trial.number, mean_r, std_r,
        params['window_size'], params['pca_components'], params['batch_size'],
        params['d_model'], params['dim_ff'], params['num_layers'],
        params['dropout_attn'], params['dropout_ff'], params['initial_noise_std'],
        params['lr'], params['weight_decay'], params['clip_grad_norm'],
        params['warmup_epochs'], params['cosine_eta_min'], params['num_epochs'],
        params['loss_fn'], params['num_heads']
    ]
    LOG_PATH.write_text(LOG_PATH.read_text() + ','.join(map(str, row)) + '\n')
    trial.set_user_attr('std_r', std_r)
    return mean_r

# --------------------------------------------------------------------------
# Run study
# --------------------------------------------------------------------------
if __name__ == '__main__':
    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction='maximize',
        pruner=MedianPruner(n_warmup_steps=3)
    )
    study.optimize(objective, n_trials=N_TRIALS)
    best = study.best_trial
    logging.info(f"Best mean Pearson R = {best.value:.4f} ± {best.user_attrs['std_r']:.4f}")
    logging.info(f"Best params: {best.params}")
