import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
import optuna

# =========================
# Configuration
# =========================
class Config:
    TRAIN_PATH = r"C:\Users\lndnc\OneDrive\Desktop\AI\DRWkaggle\train.csv"
    TEST_PATH = r"C:\Users\lndnc\OneDrive\Desktop\AI\DRWkaggle\test.csv"
    SUBMISSION_PATH = r"C:\Users\lndnc\OneDrive\Desktop\AI\DRWkaggle\xgb_submission.csv"

    BASE_FEATURES = [
        "X863", "X856", "X344", "X598", "X862", "X385", "X852", "X603", "X860", "X674",
        "X415", "X345", "X137", "X855", "X174", "X302", "X178", "X532", "X168", "X612",
        "X888", "X421", "X333"
    ]
    NN_TARGETS = BASE_FEATURES.copy()
    EMBED_DIM = 512

    LABEL_COLUMN = "label"
    RANDOM_STATE = 42
    NN_BATCH = 512
    NN_EPOCHS = 25
    NN_PATIENCE = 3
    NN_MODELS = 3
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# XGBoost parameters
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
    "n_jobs": -1
}
LEARNERS = [{"name": "xgb", "Estimator": XGBRegressor, "params": XGB_PARAMS.copy()}]

# Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# NN Predictor
class Predictor(nn.Module):
    def __init__(self, in_dim, embed_dim, out_dim):
        super().__init__()
        self.embed = nn.Linear(in_dim, embed_dim)
        self.layers = nn.Sequential(
            nn.ReLU(), nn.Linear(embed_dim, embed_dim),
            nn.ReLU(), nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU()
        )
        self.head = nn.Linear(embed_dim//2, out_dim)

    def forward(self, x):
        z = self.embed(x)
        h = self.layers(z)
        return z, self.head(h)

# Load data
def load_data():
    train_df = pd.read_csv(Config.TRAIN_PATH, usecols=Config.BASE_FEATURES + [Config.LABEL_COLUMN])
    test_df  = pd.read_csv(Config.TEST_PATH,  usecols=Config.BASE_FEATURES)
    sub_df   = pd.read_csv(Config.SUBMISSION_PATH)
    logging.info(f"Loaded: train {train_df.shape}, test {test_df.shape}")
    return train_df, test_df, sub_df

# Predict in chunks
def predict_in_chunks(model, df, features, chunk_size=20000):
    preds=[]
    model.set_params(tree_method='gpu_hist', predictor='gpu_predictor')
    for i in range(0, len(df), chunk_size):
        preds.append(model.predict(df.iloc[i:i+chunk_size][features]))
    return np.concatenate(preds)

# Train NN and get predictions
def train_nn_predictor(train_df, test_df):
    df_shift = train_df[Config.NN_TARGETS].shift(1).dropna().reset_index(drop=True)
    X_all = train_df.loc[1:, Config.BASE_FEATURES].values
    y_all = df_shift.values
    all_preds_tr, all_preds_te = [], []
    for m in range(Config.NN_MODELS):
        idx = np.arange(len(X_all)); np.random.seed(Config.RANDOM_STATE + m); np.random.shuffle(idx)
        split = int(0.9 * len(idx)); tr_i, _ = idx[:split], idx[split:]
        tr_ld = DataLoader(TensorDataset(
            torch.tensor(X_all[tr_i], dtype=torch.float32),
            torch.tensor(y_all[tr_i], dtype=torch.float32)
        ), batch_size=Config.NN_BATCH, shuffle=True)
        mdl = Predictor(len(Config.BASE_FEATURES), Config.EMBED_DIM, len(Config.NN_TARGETS)).to(Config.DEVICE)
        opt = torch.optim.Adam(mdl.parameters(), lr=5e-4)
        for _ in range(Config.NN_EPOCHS):
            mdl.train()
            for xb,yb in tr_ld:
                xb,yb=xb.to(Config.DEVICE),yb.to(Config.DEVICE)
                loss=torch.sqrt(((mdl(xb)[1]-yb)**2).mean(dim=0)).sum()
                opt.zero_grad(); loss.backward(); opt.step()
        mdl.eval()
        def run_preds(X):
            ld=DataLoader(TensorDataset(torch.tensor(X,dtype=torch.float32)), batch_size=Config.NN_BATCH)
            ps=[]
            with torch.no_grad():
                for (xb,) in ld:
                    xb=xb.to(Config.DEVICE)
                    _,p=mdl(xb)
                    ps.append(p.cpu().numpy())
            return np.vstack(ps)
        all_preds_tr.append(run_preds(train_df[Config.BASE_FEATURES].values))
        all_preds_te.append(run_preds(test_df[Config.BASE_FEATURES].values))
    return np.concatenate(all_preds_tr,axis=1), np.concatenate(all_preds_te,axis=1)

# Time-decay weights
def create_time_decay_weights(n, decay=0.9):
    pos = np.arange(n); norm = pos/(n-1)
    w = decay**(1.0-norm)
    return w*n/w.sum()

# Optuna search with slice-size guards
def optimize_xgb(train_df, mult_lower=0.7, mult_upper=1.5):
    orig = XGB_PARAMS.copy(); n=len(train_df)
    def objective(trial):
        # multipliers
        params={"tree_method":"gpu_hist","predictor":"gpu_predictor","device":"gpu",
                "verbosity":0,"random_state":Config.RANDOM_STATE,"n_jobs":-1}
                # floats; clamp subset ratios to [0,1]
        for k in ["colsample_bylevel","colsample_bynode","colsample_bytree","subsample"]:
            low = max(orig[k] * mult_lower, 0.0)
            high = min(orig[k] * mult_upper, 1.0)
            params[k] = trial.suggest_float(k, low, high)
        # other floats (can exceed 1)
        for k in ["gamma","learning_rate","reg_alpha","reg_lambda"]:
            params[k] = trial.suggest_float(
                k,
                orig[k] * mult_lower,
                orig[k] * mult_upper
            )
        # ints

        for k in ["max_depth","max_leaves","min_child_weight","n_estimators"]:
            low=int(orig[k]*mult_lower); high=int(orig[k]*mult_upper)
            params[k] = trial.suggest_int(k, max(1,low), max(low+1,high))

        n = len(train_df)
        # three slices: full, most recent 75%, and most recent 50%
        slices = [
            
            {"name": "full_data", "cutoff": 0},
            {"name": "last_75pct", "cutoff": int(0.25 * n)},
            {"name": "last_50pct", "cutoff": int(0.50 * n)}
        ]
        kf=KFold(n_splits=3,shuffle=False); corrs=[]
        for tr_idx,va_idx in kf.split(train_df):
            y_va = train_df.iloc[va_idx][Config.LABEL_COLUMN].values
            trial_corrs=[]
            for s in slices:
                cut=s["cutoff"]
                mask = va_idx>=cut
                if mask.sum()<2:
                    continue  # skip small validation
                sub = train_df.iloc[cut:].reset_index(drop=True)
                rel = tr_idx[tr_idx>=cut]-cut
                if len(rel)<2:
                    continue
                X_tr=sub.iloc[rel][Config.FEATURES].values; y_tr=sub.iloc[rel][Config.LABEL_COLUMN].values
                sw = create_time_decay_weights(len(sub))[rel] if cut>0 else create_time_decay_weights(n)[tr_idx]
                mdl=XGBRegressor(**params); mdl.fit(X_tr,y_tr,sample_weight=sw,verbose=False)
                X_val=train_df.iloc[va_idx[mask]][Config.FEATURES].values
                y_val=y_va[mask]; p=mdl.predict(X_val)
                trial_corrs.append(pearsonr(y_val,p)[0])
            if trial_corrs:
                corrs.append(np.mean(trial_corrs))
        return np.mean(corrs) if corrs else 0.0
    study=optuna.create_study(direction="maximize")
    study.optimize(objective,n_trials=62)
    logging.info(f"Optuna best value: {study.best_value:.4f}")
    return study.best_params

# =========================
# Training & Eval XGB
# =========================
def train_and_evaluate(train_df, test_df):
    n = len(train_df)
    # three slices: full, most recent 75%, and most recent 50%
    slices = [
        
        {"name": "full_data", "cutoff": 0},
        {"name": "last_75pct", "cutoff": int(0.25 * n)},
        {"name": "last_50pct", "cutoff": int(0.50 * n)}
    ]

    oof = {ln["name"]: {s["name"]: np.zeros(n) for s in slices} for ln in LEARNERS}
    kf = KFold(n_splits=3, shuffle=False)

    fold_scores = []
    test_preds = []

    for f, (tr_i, va_i) in enumerate(kf.split(train_df), 1):
        Xv = train_df.iloc[va_i][Config.FEATURES]
        yv = train_df.iloc[va_i][Config.LABEL_COLUMN]
        fold_pred = np.zeros(len(test_df))

        for s in slices:
            cut, nm = s["cutoff"], s["name"]
            sub = train_df.iloc[cut:].reset_index(drop=True)
            rel = tr_i[tr_i >= cut] - cut

            Xt = sub.iloc[rel][Config.FEATURES]
            yt = sub.iloc[rel][Config.LABEL_COLUMN]
            sw = create_time_decay_weights(len(sub))[rel] if cut > 0 else create_time_decay_weights(n)[tr_i]

            logging.info(f"Fold{f} Slice {nm}: train {len(Xt)}; eval on full val set")
            for ln in LEARNERS:
                mdl = ln["Estimator"](**ln["params"])
                mdl.fit(Xt, yt, sample_weight=sw, eval_set=[(Xv, yv)], verbose=False)

                pv = mdl.predict(Xv)
                rmse = np.sqrt(mean_squared_error(yv, pv))
                corr = pearsonr(yv, pv)[0]
                logging.info(f" Fold{f} Slice{nm} {ln['name']} RMSE={rmse:.4f} Pearson={corr:.4f}")

                mask = va_i >= cut
                oof[ln["name"]][nm][va_i[mask]] = pv[mask]
                if (~mask).any() and cut > 0:
                    oof[ln["name"]][nm][va_i[~mask]] = oof[ln["name"]]["full_data"][va_i[~mask]]

                fold_pred += predict_in_chunks(mdl, test_df, Config.FEATURES)

        fold_pred /= len(slices)
        test_preds.append(fold_pred)

        mean_oof = np.mean([oof["xgb"][s["name"]] for s in slices], axis=0)
        p = pearsonr(train_df[Config.LABEL_COLUMN], mean_oof)[0]
        logging.info(f"Fold {f} overall Pearson={p:.4f}")
        fold_scores.append((f, p))

    return oof, test_preds, fold_scores, slices

# =========================
# Ensemble & Final Logging
# =========================
def ensemble_and_submit(train_df, oof, test_preds, fold_scores, sub_df):
    for ln in oof:
        scores = {s: pearsonr(train_df[Config.LABEL_COLUMN], oof[ln][s])[0] for s in oof[ln]}
        mean_oof = np.mean(list(oof[ln].values()), axis=0)
        wt_oof = sum(scores[s]/sum(scores.values()) * oof[ln][s] for s in scores)
        logging.info(f"{ln.upper()} Simple={pearsonr(train_df[Config.LABEL_COLUMN], mean_oof)[0]:.4f} "
                     f"Wtd={pearsonr(train_df[Config.LABEL_COLUMN], wt_oof)[0]:.4f}")

    top3 = sorted(fold_scores, key=lambda x: x[1], reverse=True)[:3]
    idxs = [f-1 for f, _ in top3]
    logging.info(f"Top3 folds: {idxs}")

    final = np.stack([test_preds[i] for i in idxs], axis=1).mean(axis=1)
    sub_df['prediction'] = final
    sub_df.to_csv('submission_top3of3.csv', index=False)

    full_model = XGBRegressor(**XGB_PARAMS)
    full_model.fit(train_df[Config.FEATURES], train_df[Config.LABEL_COLUMN], verbose=False)
    imps = pd.Series(full_model.feature_importances_, index=Config.FEATURES).sort_values(ascending=False)
    logging.info("Top 10 features:\n" + imps.head(10).to_string())
    logging.info("Bottom 10 features:\n" + imps.tail(10).to_string())

# =========================
# Main
# =========================
if __name__=='__main__':
    train_df,test_df,sub_df=load_data()
    p_tr,p_te=train_nn_predictor(train_df,test_df)
    avg_tr=p_tr.reshape(-1,Config.NN_MODELS,len(Config.NN_TARGETS)).mean(1)
    avg_te=p_te.reshape(-1,Config.NN_MODELS,len(Config.NN_TARGETS)).mean(1)
    pred_cols=[f'pred_avg_{f}' for f in Config.NN_TARGETS]
    train_df=pd.concat([train_df.reset_index(drop=True),pd.DataFrame(avg_tr,columns=pred_cols)],axis=1)
    test_df=pd.concat([test_df.reset_index(drop=True),pd.DataFrame(avg_te,columns=pred_cols)],axis=1)

    # Move this line up
    Config.FEATURES=Config.BASE_FEATURES+pred_cols

    best=optimize_xgb(train_df,0.6,1.5)
    LEARNERS[0]['params'].update(best)
    oof,test_preds,scores,_=train_and_evaluate(train_df,test_df)
    ensemble_and_submit(train_df,oof,test_preds,scores,sub_df)
