import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging

# =========================
# Configuration
# =========================
class Config:
    TRAIN_PATH = r"C:\Users\lndnc\OneDrive\Desktop\AI\DRWkaggle\train.csv"
    TEST_PATH = r"C:\Users\lndnc\OneDrive\Desktop\AI\DRWkaggle\test.csv"
    SUBMISSION_PATH = r"C:\Users\lndnc\OneDrive\Desktop\AI\DRWkaggle\xgb_submission.csv"

    FEATURES = [
        "X863", "X856", "X344", "X598", "X862", "X385", "X852", "X603", "X860", "X674",
        "X415", "X345", "X137", "X855", "X174", "X302", "X178", "X532", "X168", "X612",
        "bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume", "X888", "X333"
    ]
    NN_TARGETS = ['X345', 'X888', 'X862', 'X302', 'X532', 'X344', 'X385', 'X856', 'X178', "X860", "X598"]

    LABEL_COLUMN = "label"
    N_FOLDS = 3
    RANDOM_STATE = 42
    NN_BATCH = 512
    NN_EPOCHS = 25
    NN_PATIENCE = 4
    NN_MODELS = 3
    EARLY_STOPPING_ROUNDS = 100
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# XGBoost parameters (verbosity=0 to suppress logs)
XGB_PARAMS = {
    "tree_method": "gpu_hist",
    "device": "gpu",
    "colsample_bylevel": 0.4778,
    "colsample_bynode": 0.3628,
    "colsample_bytree": 0.7107,
    "gamma": 1.7095,
    "learning_rate": 0.02213,
    "max_depth": 20,
    "max_leaves": 12,
    "min_child_weight": 16,
    "n_estimators": 1667,
    "subsample": 0.06567,
    "reg_alpha": 39.3524,
    "reg_lambda": 75.4484,
    "verbosity": 0,
    "random_state": Config.RANDOM_STATE,
    "n_jobs": -1,
    "early_stopping_rounds":Config.EARLY_STOPPING_ROUNDS
}
LGBM_PARAMS = {
    "boosting_type": "gbdt",
    "device": "gpu",
    "n_jobs": -1,
    "verbose": -1,
    "random_state": Config.RANDOM_STATE,
    "colsample_bytree": 0.5039,
    "learning_rate": 0.01260,
    "min_child_samples": 20,
    "min_child_weight": 0.1146,
    "n_estimators": 915,
    "num_leaves": 145,
    "reg_alpha": 19.2447,
    "reg_lambda": 55.5046,
    "subsample": 0.9709,
    "max_depth": 9,
    "early_stopping_rounds":Config.EARLY_STOPPING_ROUNDS

}

LEARNERS = [
    {"name": "xgb",  "Estimator": XGBRegressor,  "params": XGB_PARAMS},
    {"name": "lgbm", "Estimator": LGBMRegressor, "params": LGBM_PARAMS}
]

# =========================
# Logger Setup
# =========================
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logging.getLogger('lightgbm').setLevel(logging.ERROR)

# =========================
# Predictor NN
# =========================
class Predictor(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        return self.net(x)

# =========================
# Data I/O
# =========================
def load_data():
    train_df = pd.read_csv(Config.TRAIN_PATH, usecols=Config.FEATURES + [Config.LABEL_COLUMN])
    test_df  = pd.read_csv(Config.TEST_PATH,  usecols=Config.FEATURES)
    sub_df   = pd.read_csv(Config.SUBMISSION_PATH)
    logging.info(f"Loaded: train {train_df.shape}, test {test_df.shape}")
    return train_df, test_df, sub_df

# =========================
# Chunk Predict for XGB
# =========================
def predict_in_chunks(model, df, features, chunk_size=20000):
    preds = []
    model.set_params(tree_method='gpu_hist', predictor='gpu_predictor')
    for i in range(0, len(df), chunk_size):
        preds.append(model.predict(df.iloc[i:i+chunk_size][features]))
    return np.concatenate(preds)

# =========================
# NN Training -> preds only (average across models)
# =========================
def train_nn_predictor(train_df, test_df):
    df_shift = train_df[Config.NN_TARGETS].shift(1).dropna().reset_index(drop=True)
    X_all = train_df.loc[1:, Config.FEATURES].values
    y_all = df_shift.values
    all_preds_tr, all_preds_te = [], []

    for m in range(Config.NN_MODELS):
        idx = np.arange(len(X_all))
        np.random.seed(Config.RANDOM_STATE + m)
        np.random.shuffle(idx)
        split = int(0.9 * len(idx))
        tr_i, val_i = idx[:split], idx[split:]
        X_tr, y_tr = X_all[tr_i], y_all[tr_i]
        X_val, y_val = X_all[val_i], y_all[val_i]
        tr_ld = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32)), batch_size=Config.NN_BATCH, shuffle=True)
        val_ld = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)), batch_size=Config.NN_BATCH, shuffle=False)
        mdl = Predictor(len(Config.FEATURES), len(Config.NN_TARGETS)).to(Config.DEVICE)
        opt = torch.optim.Adam(mdl.parameters(), lr=5e-4)
        best_loss, patience = float('inf'), 0
        for ep in range(1, Config.NN_EPOCHS+1):
            mdl.train()
            for xb, yb in tr_ld:
                xb, yb = xb.to(Config.DEVICE), yb.to(Config.DEVICE)
                loss = ((mdl(xb) - yb)**2).mean().sqrt()
                opt.zero_grad(); loss.backward(); opt.step()
            mdl.eval()
            val_loss = np.mean([((mdl(xb.to(Config.DEVICE)) - yb.to(Config.DEVICE))**2).mean().sqrt().item() for xb, yb in val_ld])
            logging.info(f"NN#{m} Ep{ep} val_loss={val_loss:.4f}")
            if val_loss < best_loss:
                best_loss, patience = val_loss, 0
                torch.save(mdl.state_dict(), f'best_nn_{m}.pt')
            else:
                patience += 1
                if patience > Config.NN_PATIENCE:
                    logging.info(f"NN#{m} early stop")
                    break
        mdl.load_state_dict(torch.load(f'best_nn_{m}.pt'))
        def run_np(X_np):
            ld = DataLoader(TensorDataset(torch.tensor(X_np, dtype=torch.float32)), batch_size=Config.NN_BATCH)
            ps = []
            mdl.eval()
            with torch.no_grad():
                for (xb,) in ld:
                    xb = xb.to(Config.DEVICE)
                    p = mdl(xb)
                    ps.append(p.cpu().numpy())
            return np.vstack(ps)
        all_preds_tr.append(run_np(train_df[Config.FEATURES].values))
        all_preds_te.append(run_np(test_df[Config.FEATURES].values))

    p_tr_avg = np.mean(all_preds_tr, axis=0)
    p_te_avg = np.mean(all_preds_te, axis=0)
    return p_tr_avg, p_te_avg

# =========================
# Training & Eval with early stopping
# =========================
def train_and_evaluate(train_df, test_df):
    # keep a local, deduped feature list for slicing
    features = list(dict.fromkeys(Config.FEATURES))
    X_test_all = test_df[features].values  # numpy once, for all folds

    n = len(train_df)
    slices = [
        {'name': 'full',   'cutoff': 0},
        {'name': 'last75', 'cutoff': int(0.25 * n)},
        {'name': 'last50', 'cutoff': int(0.5  * n)}
    ]
    oof = {ln['name']: {s['name']: np.zeros(n) for s in slices} for ln in LEARNERS}
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=False)
    fold_scores = []
    test_preds = []

    for f, (tr_i, va_i) in enumerate(kf.split(train_df), 1):
        logging.info(f"===== Fold {f} =====")

        # validation set as numpy
        Xv_np = train_df.iloc[va_i][features].values
        yv_np = train_df.iloc[va_i][Config.LABEL_COLUMN].values

        # storage for this fold’s test preds
        test_fold_preds = {ln['name']: np.zeros(len(test_df)) for ln in LEARNERS}

        for s in slices:
            cut, nm = s['cutoff'], s['name']
            sub = train_df.iloc[cut:].reset_index(drop=True)

            # get the train‐on slice indices, then pull numpy
            rel = tr_i[tr_i >= cut] - cut
            Xt_np = sub.iloc[rel][features].values
            yt_np = sub.iloc[rel][Config.LABEL_COLUMN].values

            for ln in LEARNERS:
                mdl = ln['Estimator'](**ln['params'])
                # fit on numpy arrays
                mdl.fit(
                    Xt_np, yt_np,
                    eval_set=[(Xv_np, yv_np)]
                )

                # predict on numpy
                pv = mdl.predict(Xv_np)
                oof[ln['name']][nm][va_i] = pv

                # chunked test pred also on numpy slices
                if ln['name'] == 'xgb':
                    # reuse your chunk loop but pull .values
                    preds = []
                    for i in range(0, len(test_df), 20000):
                        block = test_df.iloc[i:i+20000][features].values
                        preds.append(mdl.predict(block))
                    tp = np.concatenate(preds)
                else:
                    tp = mdl.predict(X_test_all)

                test_fold_preds[ln['name']] += tp

        # average across slices
        for ln in LEARNERS:
            test_fold_preds[ln['name']] /= len(slices)
        fold_pred = 0.5 * (test_fold_preds['xgb'] + test_fold_preds['lgbm'])
        test_preds.append(fold_pred)

        # log the fold’s Pearson
        full_mean = np.mean([oof[ln['name']]['full'] for ln in LEARNERS], axis=0)
        p = pearsonr(train_df[Config.LABEL_COLUMN], full_mean)[0]
        logging.info(f"Fold {f} overall Pearson = {p:.4f}")
        fold_scores.append(p)

    return oof, test_preds, fold_scores



# =========================
# Ensemble & Submission
# =========================
def ensemble_and_submit(train_df, oof, test_preds, fold_scores, sub_df):
    for ln in oof:
        scores = {s: pearsonr(train_df[Config.LABEL_COLUMN], oof[ln][s])[0] for s in oof[ln]}
        mean_oof = np.mean(list(oof[ln].values()), axis=0)
        wt_oof = sum(scores[s]/sum(scores.values()) * oof[ln][s] for s in scores)
        logging.info(f"{ln.upper()} Simple={pearsonr(train_df[Config.LABEL_COLUMN],mean_oof)[0]:.4f} Wtd={pearsonr(train_df[Config.LABEL_COLUMN],wt_oof)[0]:.4f}")
    top3 = np.argsort(fold_scores)[-3:]
    final = np.mean([test_preds[i] for i in top3], axis=0)
    sub_df['prediction'] = final
    sub_df.to_csv('submission_top3of5.csv', index=False)

# =========================
# Main
# =========================
if __name__=='__main__':
    train_df, test_df, sub_df = load_data()
    p_tr, p_te = train_nn_predictor(train_df, test_df)
    pred_tr_df = pd.DataFrame(p_tr, columns=Config.NN_TARGETS)
    pred_te_df = pd.DataFrame(p_te, columns=Config.NN_TARGETS)
    train_df = pd.concat([train_df, pred_tr_df], axis=1)
    test_df = pd.concat([test_df, pred_te_df], axis=1)
    Config.FEATURES += Config.NN_TARGETS

    oof, test_preds, fold_scores = train_and_evaluate(train_df, test_df)
    ensemble_and_submit(train_df, oof, test_preds, fold_scores, sub_df)
