import numpy as np
import pandas as pd
import os
import sys
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# This NN uses a single shared MLP body with three heads (t-1, t-2, t-3), all trained jointly.
# Paths & Config
def create_config():
    class C:
        TRAIN_PATH = r"C:\Users\lando\Desktop\AI\DRWcryptochallenge\train.csv"
        TEST_PATH = r"C:\Users\lando\Desktop\AI\DRWcryptochallenge\test.csv"
        SUBMISSION_PATH = r"C:\Users\lando\Desktop\AI\DRWcryptochallenge\submission.csv"
        LABEL_COLUMN = "label"
        RANDOM_STATE = 42
        N_FOLDS = 3
        EARLY_PERCENTAGE = 0.35
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return C()
Config = create_config()

# Base feature list
BASE_FEATURES = [
    "X863","X856","X598","X862","X385","X852","X603","X860","X674",
    "X415","X345","X855","X174","X302","X178","X168","X612",
    "buy_qty","sell_qty","volume","X888","X421","X333","X292","X532","X344",
    "bid_qty","ask_qty"
]

# XGBoost params (unchanged)
XGB_PARAMS = {
    'tree_method':'hist','device':'gpu','n_jobs':-1,
    'colsample_bytree':0.4111,'colsample_bynode':0.2887,'gamma':1.4665,
    'learning_rate':0.01405,'max_depth':7,'max_leaves':40,'n_estimators':500,
    'reg_alpha':27.79,'reg_lambda':84.91,'subsample':0.0657,'verbosity':0,
    'random_state':Config.RANDOM_STATE
}

# NN settings: predict ALL base features, only small arch, fixed 4 epochs
NN_TARGETS = BASE_FEATURES.copy()
ARCHITECTURES = [{"name":"arch_small","layers":[256,128]}]
EPOCH_LIST = [4]
PATIENCE_HIGH = 3

# Feature engineering & load (unchanged)
def feature_engineering(df):
    df['volume_weighted_sell'] = df['sell_qty'] * df['volume']
    df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-8)
    df['selling_pressure'] = df['sell_qty'] / (df['volume'] + 1e-8)
    df['effective_spread_proxy'] = np.abs(df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-8)
    return df.replace([np.inf, -np.inf], 0).fillna(0)

def load_data():
    tr = pd.read_csv(Config.TRAIN_PATH, usecols=BASE_FEATURES + [Config.LABEL_COLUMN])
    te = pd.read_csv(Config.TEST_PATH, usecols=BASE_FEATURES)
    sub = pd.read_csv(Config.SUBMISSION_PATH)
    return feature_engineering(tr), feature_engineering(te), sub

# NN builder with three heads
def make_predictor(in_dim, arch):
    class Predictor(nn.Module):
        def __init__(self):
            super().__init__()
            layers=[]; prev=in_dim
            for h in arch['layers']:
                layers += [nn.Linear(prev, h), nn.ReLU()]
                prev=h
            self.mlp = nn.Sequential(*layers)
            # three heads: t-1, t-2, t-3
            self.head1 = nn.Linear(prev, len(NN_TARGETS))
            self.head2 = nn.Linear(prev, len(NN_TARGETS))
            self.head3 = nn.Linear(prev, len(NN_TARGETS))
        def forward(self, x):
            h = self.mlp(x)
            return self.head1(h), self.head2(h), self.head3(h)
    return Predictor().to(Config.DEVICE)

# Train NN and augment with three heads
def train_and_integrate(train_df, test_df, arch, epochs):
    feats = BASE_FEATURES.copy()
    df = train_df.copy()
    # prepare shifted targets
    y1 = df[NN_TARGETS].shift(1)
    y2 = df[NN_TARGETS].shift(2)
    y3 = df[NN_TARGETS].shift(3)
    mask = ~y3.isna().any(axis=1)
    X = df.loc[mask, feats].values
    data1 = y1.loc[mask, NN_TARGETS].values
    data2 = y2.loc[mask, NN_TARGETS].values
    data3 = y3.loc[mask, NN_TARGETS].values

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(data1, dtype=torch.float32),
        torch.tensor(data2, dtype=torch.float32),
        torch.tensor(data3, dtype=torch.float32)
    )
    loader = DataLoader(dataset, batch_size=512, shuffle=True)

    model = make_predictor(len(feats), arch)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    best_state=None; best_loss=float('inf'); stall=0
    for ep in range(1, epochs+1):
        model.train(); total=0
        for xb, yb1, yb2, yb3 in loader:
            xb, yb1, yb2, yb3 = xb.to(Config.DEVICE), yb1.to(Config.DEVICE), yb2.to(Config.DEVICE), yb3.to(Config.DEVICE)
            p1, p2, p3 = model(xb)
            l1 = ((p1-yb1)**2).mean().sqrt().sum()
            l2 = ((p2-yb2)**2).mean().sqrt().sum()
            l3 = ((p3-yb3)**2).mean().sqrt().sum()
            loss = l1 + l2 + l3
            opt.zero_grad(); loss.backward(); opt.step(); total += loss.item()
        avg = total/len(loader)
        logging.info(f"Epoch {ep}, Loss {avg:.4f}")
        if avg < best_loss:
            best_loss, stall, best_state = avg, 0, model.state_dict()
        else:
            stall += 1
        if stall > PATIENCE_HIGH:
            break
    model.load_state_dict(best_state)

    # generate predictions
    with torch.no_grad():
        Xtr = torch.tensor(train_df[feats].values, dtype=torch.float32).to(Config.DEVICE)
        o1_tr, o2_tr, o3_tr = model(Xtr)
        Xte = torch.tensor(test_df[feats].values, dtype=torch.float32).to(Config.DEVICE)
        o1_te, o2_te, o3_te = model(Xte)

    out_tr = train_df.copy(); out_te = test_df.copy()
    p1_tr = o1_tr.cpu().numpy(); p2_tr = o2_tr.cpu().numpy(); p3_tr = o3_tr.cpu().numpy()
    # head1 shift-1
    for i, t in enumerate(NN_TARGETS):
        out_tr[f"nnp1_{t}"] = np.concatenate(([np.nan], p1_tr[:-1, i]))
    # head2 shift-2
    for i, t in enumerate(NN_TARGETS):
        out_tr[f"nnp2_{t}"] = np.concatenate(([np.nan, np.nan], p2_tr[:-2, i]))
    # head3 shift-3
    for i, t in enumerate(NN_TARGETS):
        out_tr[f"nnp3_{t}"] = np.concatenate(([np.nan, np.nan, np.nan], p3_tr[:-3, i]))

    p1_te = o1_te.cpu().numpy(); p2_te = o2_te.cpu().numpy(); p3_te = o3_te.cpu().numpy()
    for i, t in enumerate(NN_TARGETS): out_te[f"nnp1_{t}"] = p1_te[:, i]
    for i, t in enumerate(NN_TARGETS): out_te[f"nnp2_{t}"] = p2_te[:, i]
    for i, t in enumerate(NN_TARGETS): out_te[f"nnp3_{t}"] = p3_te[:, i]

    new_feats = feats + [f"nnp{j}_{t}" for j in [1,2,3] for t in NN_TARGETS]
    return out_tr, out_te, new_feats

# XGB train/eval (unchanged)
def train_and_evaluate(train_df, test_df, features):
    n = len(train_df); wk = Config.EARLY_PERCENTAGE
    slices = [('full',0), ('75pct',int(0.25*n)), ('50pct',int(0.5*n)), ('early',int(wk*n))]
    oof = {'xgb':{s[0]:np.zeros(n) for s in slices}}
    test_preds = {'xgb':{s[0]:np.zeros(len(test_df)) for s in slices}}
    weights = np.arange(n)/(n-1)
    sw = (0.9**(1-weights))*n/np.sum(0.9**(1-weights))
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=False)
    for fold, (tr, val) in enumerate(kf.split(train_df),1):
        Xv = train_df.iloc[val][features].values; yv = train_df.iloc[val][Config.LABEL_COLUMN].values
        for slice_name, cut in slices:
            if slice_name=='full': tr_idx=tr; swt=sw[tr]
            elif slice_name=='early': mask=tr<cut; tr_idx=tr[mask]; swt=sw[tr_idx]
            else: mask=tr>=cut; tr_idx=tr[mask]; swt=sw[tr_idx] if cut>0 else sw[tr]
            if len(tr_idx)==0: continue
            Xt = train_df.iloc[tr_idx][features].values; yt = train_df.iloc[tr_idx][Config.LABEL_COLUMN].values
            model = XGBRegressor(**XGB_PARAMS)
            model.fit(Xt, yt, sample_weight=swt, eval_set=[(Xv,yv)], verbose=False)
            oof['xgb'][slice_name][val] = model.predict(Xv)
            test_preds['xgb'][slice_name] += model.predict(test_df[features].values)
    for sl in test_preds['xgb']: test_preds['xgb'][sl] /= Config.N_FOLDS
    return oof, test_preds

# Search & select best (only small arch, 4 epochs)
def search_best():
    train_df, test_df, sub = load_data()
    best_score = -np.inf; best_cfg=None
    for arch in ARCHITECTURES:
        for ep in EPOCH_LIST:
            logging.info(f"Testing {arch['name']} epochs={ep}")
            df_tr, df_te, feats = train_and_integrate(train_df, test_df, arch, ep)
            oof, test_preds = train_and_evaluate(df_tr, df_te, feats)
            blended = np.mean(list(oof['xgb'].values()), axis=0)
            score = pearsonr(df_tr[Config.LABEL_COLUMN], blended)[0]
            logging.info(f" Score={score:.4f}")
            if score > best_score:
                best_score, best_cfg = score, (arch, ep, feats)
    logging.info(f"Best score={best_score:.4f} with {best_cfg[0]['name']} @ epochs={best_cfg[1]}")
    return best_cfg

# Final full train & predict
def final_predict():
    train_df, test_df, sub = load_data()
    arch, ep, feats = search_best()
    df_tr, df_te, feats = train_and_integrate(train_df, test_df, arch, ep)
    oof, test_preds = train_and_evaluate(df_tr, df_te, feats)
    blended_test = np.mean(list(test_preds['xgb'].values()), axis=0)
    sub['prediction'] = blended_test
    sub.to_csv(Config.SUBMISSION_PATH, index=False)
    logging.info(f"Saved predictions to {Config.SUBMISSION_PATH}")

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    final_predict()
