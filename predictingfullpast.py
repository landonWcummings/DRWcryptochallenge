import os
import logging

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from scipy.stats import pearsonr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# =========================
# Configuration
# =========================
class Config:
    TRAIN_PATH = r"C:\Users\lndnc\OneDrive\Desktop\AI\DRWkaggle\train.csv"
    TEST_PATH = r"C:\Users\lndnc\OneDrive\Desktop\AI\DRWkaggle\test.csv"
    SUBMISSION_PATH = r"C:\Users\lndnc\OneDrive\Desktop\AI\DRWkaggle\submission.csv"

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 512
    MAX_EPOCHS = 20
    PATIENCE = 2
    VALID_SIZE = 0.1
    RANDOM_STATES = [42, 7]
    WEIGHT_DECAY = 1e-5
    N_FOLDS = 3
    EMBED_DIM = 512
    GRAD_CLIP = 1.0
    XGB_PARAMS = {
        'tree_method': 'hist', 'device': 'gpu',
        'colsample_bytree': 0.71, 'learning_rate': 0.022,
        'max_depth': 20, 'n_estimators': 1667,
        'reg_alpha': 39.35, 'reg_lambda': 75.45,
        'subsample': 0.066, 'verbosity': 0,
        'random_state': 42,
        'early_stopping_rounds': 50,
    }

    @staticmethod
    def get_X_cols(df):
        return [c for c in df.columns if c.startswith('X')]

# =========================
# NN architectures
# =========================
class FeatureNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, out_dim)
        )
    def forward(self, x): return self.net(x)

# =========================
# Helpers
# =========================
def train_epoch(model, optimizer, loader, scaler_y):
    model.train()
    total_loss = 0.0
    criterion = nn.SmoothL1Loss()
    for xb, yb in loader:
        xb, yb = xb.to(Config.DEVICE), yb.to(Config.DEVICE)
        # normalize target
        yb_norm = scaler_y.transform(yb.cpu().numpy()).astype(np.float32)
        yb_norm = torch.from_numpy(yb_norm).to(Config.DEVICE)

        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb_norm)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def eval_epoch(model, loader, scaler_y):
    model.eval()
    total_loss = 0.0
    criterion = nn.SmoothL1Loss()
    for xb, yb in loader:
        xb, yb = xb.to(Config.DEVICE), yb.to(Config.DEVICE)
        yb_norm = scaler_y.transform(yb.cpu().numpy()).astype(np.float32)
        yb_norm = torch.from_numpy(yb_norm).to(Config.DEVICE)
        preds = model(xb)
        loss = criterion(preds, yb_norm)
        total_loss += loss.item()
    return total_loss / len(loader)

# =========================
# Main pipeline
# =========================
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    df = pd.read_csv(Config.TRAIN_PATH)
    df_test = pd.read_csv(Config.TEST_PATH)
    X_cols = Config.get_X_cols(df)

    # Prepare features and sequential targets
    X = df[X_cols].values.astype(np.float32)
    Y_raw = np.vstack([np.zeros((1, len(X_cols))), X[:-1]])
    Y = Y_raw.astype(np.float32)

    # scaler for Y normalization
    scaler_y = StandardScaler().fit(Y)

    # OOF containers
    oof_feat = np.zeros_like(Y)
    test_feats = []
    val_scores = []

    for seed in Config.RANDOM_STATES:
        logging.info(f"NN split seed={seed}")
        X_tr, X_val, Y_tr, Y_val = train_test_split(X, Y, test_size=Config.VALID_SIZE, random_state=seed)
        scaler = StandardScaler().fit(X_tr)
        X_tr_s, X_val_s = scaler.transform(X_tr), scaler.transform(X_val)
        X_test_s = scaler.transform(df_test[X_cols].values.astype(np.float32))

        tr_loader = DataLoader(TensorDataset(
            torch.from_numpy(X_tr_s).float(), torch.from_numpy(Y_tr).float()
        ), batch_size=Config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TensorDataset(
            torch.from_numpy(X_val_s).float(), torch.from_numpy(Y_val).float()
        ), batch_size=Config.BATCH_SIZE)

        # train ensemble of 6 models per split
        preds_val_accum = np.zeros_like(Y_val)
        for model_idx in range(len(NN_LEARNERS)):
            ModelClass = NN_LEARNERS[model_idx]
            model = ModelClass(len(X_cols), len(X_cols)) if ModelClass is not ModelBaseline else ModelClass(len(X_cols), Config.EMBED_DIM, len(X_cols))
            model.to(Config.DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=Config.WEIGHT_DECAY)

            best_loss, patience = float('inf'), 0
            best_state = model.state_dict()
            for ep in range(1, Config.MAX_EPOCHS+1):
                tr_loss = train_epoch(model, optimizer, tr_loader, scaler_y)
                val_loss = eval_epoch(model, val_loader, scaler_y)
                logging.info(f"{ModelClass.__name__}[{model_idx}] seed{seed} ep{ep} tr={tr_loss:.4f} val={val_loss:.4f}")
                if val_loss < best_loss:
                    best_loss, patience = val_loss, 0
                    best_state = {k:v.cpu() for k,v in model.state_dict().items()}
                else:
                    patience += 1
                if patience > Config.PATIENCE:
                    logging.info(f"Early stop {ModelClass.__name__}[{model_idx}] at ep{ep}")
                    break

            model.load_state_dict(best_state)
            # validation predictions (denormalize)
            val_pred_norm = model(torch.from_numpy(X_val_s).to(Config.DEVICE)).detach().cpu().numpy()
            val_pred = scaler_y.inverse_transform(val_pred_norm)
            preds_val_accum += val_pred

            # test preds
            test_pred_norm = model(torch.from_numpy(X_test_s).to(Config.DEVICE)).detach().cpu().numpy()
            test_pred = scaler_y.inverse_transform(test_pred_norm)
            test_feats.append(test_pred)

        # average over 6 models
        avg_val_pred = preds_val_accum / len(NN_LEARNERS)
        corr = pearsonr(Y_val.flatten(), avg_val_pred.flatten())[0]
        val_scores.append(corr)
        logging.info(f"Ensembled NN seed{seed} pearson: {corr:.4f}")

        # fill OOF
        oof_feat[np.isin(range(len(X)), df.index[X_val_s.shape[0]:]) == False] = avg_val_pred

    logging.info(f"Avg NN ensemble pearson: {np.mean(val_scores):.4f}")

    # stack features
    test_feat_mean = np.mean(np.stack(test_feats), axis=0)
    stack_X = np.hstack([oof_feat, X])
    stack_test = np.hstack([test_feat_mean, df_test[X_cols].values.astype(np.float32)])

    # XGB stacking
    oof_pred = np.zeros(len(df))
    test_pred = np.zeros(len(df_test))
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=42)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(stack_X),1):
        logging.info(f"XGB fold {fold}")
        Xtr, ytr = stack_X[tr_idx], df['label'].iloc[tr_idx]
        Xva, yva = stack_X[va_idx], df['label'].iloc[va_idx]
        xgb = XGBRegressor(**Config.XGB_PARAMS)
        xgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=Config.XGB_PARAMS['verbosity'])
        oof_pred[va_idx] = xgb.predict(Xva)
        test_pred += xgb.predict(stack_test)
        corr = pearsonr(yva, oof_pred[va_idx])[0]
        logging.info(f"XGB fold{fold} pearson: {corr:.4f}")

    test_pred /= Config.N_FOLDS
    total_corr = pearsonr(df['label'], oof_pred)[0]
    logging.info(f"XGB OOF pearson: {total_corr:.4f}")

    # save submission
    sub = pd.DataFrame({'label': test_pred})
    sub.to_csv(Config.SUBMISSION_PATH, index=False)
    logging.info("Done.")
