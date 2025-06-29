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

# XGBoost params
XGB_PARAMS = {
    'tree_method':'hist','device':'gpu','n_jobs':-1,
    'colsample_bytree':0.4111,'colsample_bynode':0.2887,'gamma':1.4665,
    'learning_rate':0.01405,'max_depth':7,'max_leaves':40,'n_estimators':500,
    'reg_alpha':27.79,'reg_lambda':84.91,'subsample':0.0657,'verbosity':0,
    'random_state':Config.RANDOM_STATE
}

# NN settings
NN_TARGETS = ['X345','X888','X862','X302','X532','X344','X385','X856','X178']
ARCHITECTURES = [
    {"name":"arch_small","embed_dim":256,"layers":[256,128]},
    {"name":"arch_large","embed_dim":512,"layers":[512,512,256]}
]
EPOCH_LIST = list(range(3,32,1))
PATIENCE_HIGH = 3

# Feature engineering & load
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

# NN builder
def make_predictor(in_dim, arch):
    class Predictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(in_dim, arch['embed_dim'])
            layers=[]; prev=arch['embed_dim']
            for h in arch['layers']:
                layers += [nn.ReLU(), nn.Linear(prev, h)]; prev=h
            layers += [nn.ReLU()]
            self.mlp = nn.Sequential(*layers)
            self.head = nn.Linear(prev, len(NN_TARGETS))
        def forward(self, x):
            z = self.embed(x)
            h = self.mlp(z)
            return self.head(h)
    return Predictor().to(Config.DEVICE)

# Train NN and augment (only add NN_TARGETS preds)
def train_and_integrate(train_df, test_df, arch, epochs):
    feats = BASE_FEATURES.copy()
    df = train_df.copy()
    df_shift = df[NN_TARGETS].shift(1).dropna().reset_index(drop=True)
    X = df.loc[1:, feats].values; y = df_shift.values
    loader = DataLoader(
        TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        ), batch_size=512, shuffle=True
    )
    model = make_predictor(len(feats), arch)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    best_state=None; best_loss=float('inf'); stall=0
    for ep in range(1, epochs+1):
        model.train(); total=0
        for xb,yb in loader:
            xb,yb=xb.to(Config.DEVICE), yb.to(Config.DEVICE)
            pred = model(xb)
            loss = ((pred-yb)**2).mean().sqrt().sum()
            opt.zero_grad(); loss.backward(); opt.step(); total+=loss.item()
        avg = total/len(loader)
        if avg < best_loss:
            best_loss, stall, best_state = avg, 0, model.state_dict()
        else:
            stall += 1
        if stall > 1: break
    model.load_state_dict(best_state)

    with torch.no_grad():
        Xtr = torch.tensor(train_df[feats].values, dtype=torch.float32).to(Config.DEVICE)
        pred_tr = model(Xtr)
        Xte = torch.tensor(test_df[feats].values, dtype=torch.float32).to(Config.DEVICE)
        pred_te = model(Xte)

    out_tr = train_df.copy(); out_te = test_df.copy()
    for i, t in enumerate(NN_TARGETS):
        arr = pred_tr.cpu().numpy()[:, i]
        shifted = np.concatenate(([np.nan], arr[:-1]))
        out_tr[f"nnp_{t}"] = pd.Series(shifted, index=out_tr.index)
        out_te[f"nnp_{t}"] = pd.Series(pred_te.cpu().numpy()[:, i], index=out_te.index)

    new_feats = feats + [f"nnp_{t}" for t in NN_TARGETS]
    return out_tr, out_te, new_feats

# XGB train/eval
def train_and_evaluate(train_df,test_df,features):
    n=len(train_df); wk=Config.EARLY_PERCENTAGE
    slices=[('full',0),('75pct',int(0.25*n)),('50pct',int(0.5*n)),('early',int(wk*n))]
    oof={}; test_preds={}
    for name,_ in [('xgb',None)]:
        oof[name]={s[0]:np.zeros(n) for s in slices}
        test_preds[name]={s[0]:np.zeros(len(test_df)) for s in slices}
    weights=np.arange(n)/(n-1); sw=(0.9**(1-weights))*n/np.sum(0.9**(1-weights))
    kf=KFold(n_splits=Config.N_FOLDS,shuffle=False)
    for fold,(tr,val) in enumerate(kf.split(train_df),1):
        Xv=train_df.iloc[val][features].values; yv=train_df.iloc[val][Config.LABEL_COLUMN].values
        for slice_name,cut in slices:
            if slice_name=='full': tr_idx=tr; swt=sw[tr]
            elif slice_name=='early': mask=tr<cut; tr_idx=tr[mask]; swt=sw[tr_idx]
            else: mask=tr>=cut; tr_idx=tr[mask]; swt=sw[tr_idx] if cut>0 else sw[tr]
            if len(tr_idx)==0: continue
            Xt=train_df.iloc[tr_idx][features].values; yt=train_df.iloc[tr_idx][Config.LABEL_COLUMN].values
            model=XGBRegressor(**XGB_PARAMS)
            model.fit(Xt, yt, sample_weight=swt, eval_set=[(Xv, yv)], verbose=False)
            preds = model.predict(Xv)
            oof['xgb'][slice_name][val]=preds
            test_preds['xgb'][slice_name]+=model.predict(test_df[features].values)
    for sl in test_preds['xgb']: test_preds['xgb'][sl]/=Config.N_FOLDS
    return oof, test_preds

# Search & select best per architecture
def search_best():
    train_df,test_df,sub=load_data()
    best_per_arch = {}
    for arch in ARCHITECTURES:
        arch_best=-np.inf; stall=0; best_epoch=None; best_feats=None
        for ep in EPOCH_LIST:
            print(f"Testing {arch['name']} epochs={ep}")
            df_tr,df_te,feats = train_and_integrate(train_df,test_df,arch,ep)
            oof,_ = train_and_evaluate(df_tr,df_te,feats)
            blended = np.mean(list(oof['xgb'].values()),axis=0)
            score = pearsonr(df_tr[Config.LABEL_COLUMN], blended)[0]
            print(f" Score for {arch['name']} epoch {ep}: {score:.4f}")
            if score>arch_best:
                arch_best, stall = score, 0
                best_epoch, best_feats = ep, feats
            else:
                stall += 1
            if stall>=PATIENCE_HIGH: break
        best_per_arch[arch['name']] = (arch, best_epoch, best_feats)
        print(f"Best for {arch['name']} @ epochs={best_epoch} score={arch_best:.4f}")
    return best_per_arch

# Final full train

def final_predict():
    train_df,test_df,sub = load_data()
    bests = search_best()
    preds = {}
    for name,(arch,ep,feats) in bests.items():
        df_tr,df_te,_ = train_and_integrate(train_df,test_df,arch,ep)
        _, test_preds = train_and_evaluate(df_tr,df_te,feats)
        preds[name] = np.mean(list(test_preds['xgb'].values()),axis=0)
        print(f"Collected test predictions for {name} (epoch {ep})")

    # average small & large
    blended_test = (preds['arch_small'] + preds['arch_large']) / 2
    sub['prediction'] = blended_test

    # final XGB importances on concatenated df_tr of large
    arch,ep,feats = bests['arch_large']
    df_tr,_ ,_ = train_and_integrate(train_df,test_df,arch,ep)
    final_xgb = XGBRegressor(**XGB_PARAMS)
    final_xgb.fit(df_tr[feats].values, df_tr[Config.LABEL_COLUMN].values)
    feat_imp = sorted(zip(feats, final_xgb.feature_importances_), key=lambda x: x[1], reverse=True)
    print("XGB Feature Importances:")
    for feat,imp in feat_imp:
        print(f"{feat}: {imp:.4f}")

    sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Saved predictions to {Config.SUBMISSION_PATH}")

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    final_predict()
