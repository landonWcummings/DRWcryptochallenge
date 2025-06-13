import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from scipy.stats import pearsonr
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Configuration ---
CSV_PATH      = r'C:\Users\lndnc\OneDrive\Desktop\AI\DRWkaggle\train.csv'
TEST_CSV_PATH = r'C:\Users\lndnc\OneDrive\Desktop\AI\DRWkaggle\test.csv'
TIME_COL      = 'timestamp'
TARGET_COL    = 'label'
BATCH_SIZE    = 128
NUM_EPOCHS    = 10
LR            = 1e-3
DROPOUT       = 0.3
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_RUNS      = 7
MODEL_DIR     = 'model_runs'
BEST_MODEL    = 'best_model.pt'
PLOT_PATH     = 'validation_runs.png'
tr_loader = None
vl_loader = None


# Create directories
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Selected features ---
selected_features = [
    "X863", "X856", "X344", "X598", "X862", "X385", "X852", "X603", "X860", "X674",
    "X415", "X345", "X137", "X855", "X174", "X302", "X178", "X532", "X168", "X612",
    "bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume", "X888", "X421", "X333",
    'bid_ask_interaction', 'bid_buy_interaction', 'bid_sell_interaction', 'ask_buy_interaction',
    'ask_sell_interaction', 'buy_sell_interaction', 'spread_indicator', 'volume_weighted_buy',
    'volume_weighted_sell', 'volume_weighted_bid', 'volume_weighted_ask', 'buy_sell_ratio',
    'bid_ask_ratio', 'order_flow_imbalance', 'buying_pressure', 'selling_pressure',
    'total_liquidity', 'liquidity_imbalance', 'relative_spread', 'trade_intensity',
    'avg_trade_size', 'net_trade_flow', 'depth_ratio', 'volume_participation', 'market_activity',
    'effective_spread_proxy', 'realized_volatility_proxy', 'normalized_buy_volume',
    'normalized_sell_volume', 'liquidity_adjusted_imbalance', 'pressure_spread_interaction',
    'trade_direction_ratio', 'net_buy_volume', 'bid_skew', 'ask_skew'
]

# --- Feature engineering ---
def add_features(df):
    df['bid_ask_interaction'] = df['bid_qty'] * df['ask_qty']
    df['bid_buy_interaction'] = df['bid_qty'] * df['buy_qty']
    df['bid_sell_interaction'] = df['bid_qty'] * df['sell_qty']
    df['ask_buy_interaction'] = df['ask_qty'] * df['buy_qty']
    df['ask_sell_interaction'] = df['ask_qty'] * df['sell_qty']
    df['buy_sell_interaction'] = df['buy_qty'] * df['sell_qty']
    df['spread_indicator'] = (df['ask_qty'] - df['bid_qty']) / (df['ask_qty'] + df['bid_qty'] + 1e-8)
    df['volume_weighted_buy'] = df['buy_qty'] * df['volume']
    df['volume_weighted_sell'] = df['sell_qty'] * df['volume']
    df['volume_weighted_bid'] = df['bid_qty'] * df['volume']
    df['volume_weighted_ask'] = df['ask_qty'] * df['volume']
    df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-8)
    df['bid_ask_ratio'] = df['bid_qty'] / (df['ask_qty'] + 1e-8)
    df['order_flow_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-8)
    df['buying_pressure'] = df['buy_qty'] / (df['volume'] + 1e-8)
    df['selling_pressure'] = df['sell_qty'] / (df['volume'] + 1e-8)
    df['total_liquidity'] = df['bid_qty'] + df['ask_qty']
    df['liquidity_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['total_liquidity'] + 1e-8)
    df['relative_spread'] = (df['ask_qty'] - df['bid_qty']) / (df['volume'] + 1e-8)
    df['trade_intensity'] = (df['buy_qty'] + df['sell_qty']) / (df['volume'] + 1e-8)
    df['avg_trade_size'] = df['volume'] / (df['buy_qty'] + df['sell_qty'] + 1e-8)
    df['net_trade_flow'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + 1e-8)
    df['depth_ratio'] = df['total_liquidity'] / (df['volume'] + 1e-8)
    df['volume_participation'] = (df['buy_qty'] + df['sell_qty']) / (df['total_liquidity'] + 1e-8)
    df['market_activity'] = df['volume'] * df['total_liquidity']
    df['effective_spread_proxy'] = np.abs(df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-8)
    df['realized_volatility_proxy'] = np.abs(df['order_flow_imbalance']) * df['volume']
    df['normalized_buy_volume'] = df['buy_qty'] / (df['bid_qty'] + 1e-8)
    df['normalized_sell_volume'] = df['sell_qty'] / (df['ask_qty'] + 1e-8)
    df['liquidity_adjusted_imbalance'] = df['order_flow_imbalance'] * df['depth_ratio']
    df['pressure_spread_interaction'] = df['buying_pressure'] * df['spread_indicator']
    df['trade_direction_ratio'] = df['buy_qty'] / (df['buy_qty'] + df['sell_qty'] + 1e-8)
    df['net_buy_volume'] = df['buy_qty'] - df['sell_qty']
    df['bid_skew'] = df['bid_qty'] / (df['bid_qty'] + df['ask_qty'] + 1e-8)
    df['ask_skew'] = df['ask_qty'] / (df['bid_qty'] + df['ask_qty'] + 1e-8)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df

# --- Dataset ---
class TimeSeriesDataset(Dataset):
    def __init__(self, df):
        df = df.sort_values(TIME_COL).reset_index(drop=True)
        prev = df[selected_features + [TARGET_COL]].shift(1).dropna()
        curr = df[selected_features].iloc[1:].reset_index(drop=True)
        prev_feat = prev[selected_features].values
        curr_label= df[TARGET_COL].iloc[1:].values
        self.X_curr = torch.tensor(curr.values, dtype=torch.float32)
        self.X_prev  = torch.tensor(prev_feat, dtype=torch.float32)
        self.y_curr_label= torch.tensor(curr_label, dtype=torch.float32)
    def __len__(self): return len(self.X_curr)
    def __getitem__(self, idx):
        return self.X_curr[idx], self.X_prev[idx], self.y_curr_label[idx]
    
class LinearDecoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
    def forward(self, x):
        return self.lin(x)

class ResidualDecoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
    def forward(self, x):
        return x + self.lin(x)

# --- Models ---
class TransformerPredictor(nn.Module):
    def __init__(self, feat_dim, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim*2, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dropout=DROPOUT, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers)
        self.fc = nn.Linear(d_model, 1)
    def forward(self, x_curr, x_prev_input):
        x = torch.cat([x_curr, x_prev_input], dim=-1)
        z = self.transformer(self.input_proj(x).unsqueeze(1)).squeeze(1)
        return self.fc(z).squeeze(1)

class PrevRefiner(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(DROPOUT)
        )
        self.out = nn.Linear(hidden, dim)
    def forward(self, x):
        h = self.net(x)
        return self.out(h)

# --- Load pre-trained decoders ---
def load_decoders():
    # Instantiate decoders with same architecture as training
    linear_decoder = LinearDecoder(len(selected_features)).to(DEVICE)
    residual_decoder = ResidualDecoder(len(selected_features)).to(DEVICE)
    # Load state dicts
    lin_sd = torch.load('linear_decoder.pth', map_location=DEVICE)
    res_sd = torch.load('residual_decoder.pth', map_location=DEVICE)
    linear_decoder.load_state_dict(lin_sd)
    residual_decoder.load_state_dict(res_sd)
    linear_decoder.eval(); residual_decoder.eval()
    return linear_decoder, residual_decoder

# --- Train ---
def train():
    global tr_loader, vl_loader
    # Data prep
    df = pd.read_csv(CSV_PATH); df = add_features(df)
    df = df[[TIME_COL] + selected_features + [TARGET_COL]]
    cut = int(0.8 * len(df))
    train_df, val_df = df.iloc[:cut].copy(), df.iloc[cut:].copy()
    mean = train_df[selected_features].mean()
    std  = train_df[selected_features].std().replace(0,1)
    for d in (train_df, val_df): d[selected_features] = (d[selected_features]-mean)/std
    tr_loader = DataLoader(TimeSeriesDataset(train_df), BATCH_SIZE, shuffle=False)
    vl_loader = DataLoader(TimeSeriesDataset(val_df), BATCH_SIZE)

    # Models
    lin_dec, res_dec = load_decoders()
    refiner = PrevRefiner(len(selected_features)).to(DEVICE)
    transformer = TransformerPredictor(len(selected_features)).to(DEVICE)

    mse = nn.MSELoss()
    opt_ref = optim.Adam(refiner.parameters(), lr=LR)
    opt_tr  = optim.Adam(transformer.parameters(), lr=LR)

    for epoch in range(1, NUM_EPOCHS+1):
        alpha = (epoch-1)/(NUM_EPOCHS-1)
        refiner.train(); transformer.train()
        t_metrics = {'mse_ref':0, 'mae_ref':0, 'mse_tr':0, 'mae_tr':0, 'cnt':0}
        for Xc, Xp, y in tr_loader:
            Xc, Xp, y = Xc.to(DEVICE), Xp.to(DEVICE), y.to(DEVICE)
            # base decoders
            pf_lin = lin_dec(Xc); pf_res = res_dec(Xc)
            pf_base = (pf_lin + pf_res)/2
            # refine
            pf_ref, pl_ref = refiner(pf_base)
            # refine metrics
            t_metrics['mse_ref'] += mse(pf_ref, Xp).item()*Xc.size(0)
            t_metrics['mae_ref'] += torch.abs(pf_ref-Xp).sum().item()
            # mixing
            prev_input = (1-alpha)*Xp + alpha*pf_ref.detach()
            # predict
            pred = transformer(Xc, prev_input)
            # tr metrics
            t_metrics['mse_tr'] += mse(pred, y).item()*Xc.size(0)
            t_metrics['mae_tr'] += torch.abs(pred-y).sum().item()
            t_metrics['cnt'] += Xc.size(0)
            # optimize
            loss = mse(pf_ref, Xp) + mse(pred, y)
            opt_ref.zero_grad(); opt_tr.zero_grad(); loss.backward()
            opt_ref.step(); opt_tr.step()
        # compute train stats
        cnt = t_metrics['cnt']
        tr_mse_ref = t_metrics['mse_ref']/cnt; tr_mae_ref = t_metrics['mae_ref']/cnt
        tr_mse_tr  = t_metrics['mse_tr']/cnt; tr_mae_tr  = t_metrics['mae_tr']/cnt

        # validation
        refiner.eval(); transformer.eval()
        v_metrics = {'mse_ref':0,'mae_ref':0,'mse_tr':0,'mae_tr':0,'cnt':0}
        ps, ts = [], []
        with torch.no_grad():
            for Xc, Xp, y in vl_loader:
                Xc, Xp, y = Xc.to(DEVICE), Xp.to(DEVICE), y.to(DEVICE)
                pf_base = (lin_dec(Xc)+res_dec(Xc))/2
                pf_ref, _ = refiner(pf_base)
                v_metrics['mse_ref'] += mse(pf_ref, Xp).item()*Xc.size(0)
                v_metrics['mae_ref'] += torch.abs(pf_ref-Xp).sum().item()
                prev_input = (1-alpha)*Xp + alpha*pf_ref
                pred = transformer(Xc, prev_input)
                v_metrics['mse_tr'] += mse(pred, y).item()*Xc.size(0)
                v_metrics['mae_tr'] += torch.abs(pred-y).sum().item()
                v_metrics['cnt'] += Xc.size(0)
                ps.extend(pred.cpu().numpy()); ts.extend(y.cpu().numpy())
        cnt = v_metrics['cnt']
        v_mse_ref = v_metrics['mse_ref']/cnt; v_mae_ref = v_metrics['mae_ref']/cnt
        v_mse_tr  = v_metrics['mse_tr']/cnt; v_mae_tr  = v_metrics['mae_tr']/cnt
        v_r = pearsonr(ts, ps)[0]

        logging.info(
            f"Ep{epoch}/{NUM_EPOCHS} alpha={alpha:.2f} "
            f"Train(ref MSE={tr_mse_ref:.4f}, MAE={tr_mae_ref:.4f}; tr MSE={tr_mse_tr:.4f}, MAE={tr_mae_tr:.4f}) "
            f"Val(ref MSE={v_mse_ref:.4f}, MAE={v_mae_ref:.4f}; tr MSE={v_mse_tr:.4f}, MAE={v_mae_tr:.4f}, R={v_r:.4f})"
        )
    return lin_dec, res_dec, refiner, transformer, mean, std

# --- Inference ---
def train_one_epoch(lin_dec, res_dec, refiner, transformer, opt, mse, mae, alpha):
    lin_dec.eval(); res_dec.eval(); refiner.train(); transformer.train()
    t_mse = t_mae = cnt = 0
    for Xc, Xp, y in tr_loader:
        Xc, Xp, y = Xc.to(DEVICE), Xp.to(DEVICE), y.to(DEVICE)
        pf_base = (lin_dec(Xc) + res_dec(Xc)) / 2
        pf_ref, _ = refiner(pf_base)
        prev_input = (1-alpha)*Xp + alpha*pf_ref.detach()
        pred = transformer(Xc, prev_input)
        loss = mse(pf_ref, Xp) + mse(pred, y)
        opt.zero_grad(); loss.backward(); opt.step()
        t_mse += mse(pred, y).item()*Xc.size(0)
        t_mae += mae(pred, y).item()*Xc.size(0)
        cnt += Xc.size(0)
    return t_mse/cnt, t_mae/cnt


def validate(lin_dec, res_dec, refiner, transformer, mse, mae, alpha):
    lin_dec.eval(); res_dec.eval(); refiner.eval(); transformer.eval()
    v_mse = v_mae = cnt = 0; preds, trues = [], []
    with torch.no_grad():
        for Xc, Xp, y in vl_loader:
            Xc, Xp, y = Xc.to(DEVICE), Xp.to(DEVICE), y.to(DEVICE)
            pf_base = (lin_dec(Xc) + res_dec(Xc)) / 2
            pf_ref, _ = refiner(pf_base)
            prev_input = (1-alpha)*Xp + alpha*pf_ref
            pred = transformer(Xc, prev_input)
            v_mse += mse(pred, y).item()*Xc.size(0)
            v_mae += mae(pred, y).item()*Xc.size(0)
            preds.extend(pred.cpu().numpy()); trues.extend(y.cpu().numpy())
            cnt += Xc.size(0)
    return v_mse/cnt, v_mae/cnt, pearsonr(trues, preds)[0]

# --- Training loop for one model ---
def run_seed(seed):
    torch.manual_seed(seed)
    df = pd.read_csv(CSV_PATH)
    df = add_features(df)
    df = df[[TIME_COL] + selected_features + [TARGET_COL]]
    cut = int(0.8 * len(df))
    train_df, val_df = df.iloc[:cut], df.iloc[cut:]
    mean = train_df[selected_features].mean(); std = train_df[selected_features].std().replace(0,1)
    train_df.loc[:, selected_features] = (train_df[selected_features] - mean) / std
    val_df.loc[:, selected_features]   = (val_df[selected_features]   - mean) / std
    tr_loader = DataLoader(TimeSeriesDataset(train_df), BATCH_SIZE, shuffle=True)
    vl_loader = DataLoader(TimeSeriesDataset(val_df), BATCH_SIZE)

    lin_dec, res_dec = load_decoders()
    refiner = PrevRefiner(len(selected_features)).to(DEVICE)
    transformer = TransformerPredictor(len(selected_features)).to(DEVICE)
    opt = optim.Adam(list(refiner.parameters()) + list(transformer.parameters()), lr=LR)
    mse = nn.MSELoss()

    metrics = []
    for epoch in range(1, NUM_EPOCHS+1):
        alpha = (epoch-1)/(NUM_EPOCHS-1)
        refiner.train(); transformer.train()
        train_loss = val_loss = cnt = 0
        tr_preds, tr_trues = [], []
        for Xc, Xp, y in tr_loader:
            Xc, Xp, y = Xc.to(DEVICE), Xp.to(DEVICE), y.to(DEVICE)
            pf = (lin_dec(Xc) + res_dec(Xc)) / 2
            pf_ref = refiner(pf)
            pred = transformer(Xc, pf_ref)
            loss = mse(pf_ref, Xp) + mse(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_preds.append(pred.detach().cpu().numpy()); tr_trues.append(y.cpu().numpy())
            cnt += Xc.size(0)
            train_loss += mse(pred, y).item() * Xc.size(0)
        tr_r = pearsonr(np.concatenate(tr_trues), np.concatenate(tr_preds))[0]
        tr_mse = train_loss/cnt

        refiner.eval(); transformer.eval()
        val_preds, val_trues = [], []
        cnt = val_loss = 0
        with torch.no_grad():
            for Xc, Xp, y in vl_loader:
                Xc, Xp, y = Xc.to(DEVICE), Xp.to(DEVICE), y.to(DEVICE)
                pf = (lin_dec(Xc) + res_dec(Xc)) / 2
                pf_ref = refiner(pf)
                pred = transformer(Xc, pf_ref)
                val_preds.append(pred.cpu().numpy()); val_trues.append(y.cpu().numpy())
                cnt += Xc.size(0)
                val_loss += mse(pred, y).item() * Xc.size(0)
        v_r = pearsonr(np.concatenate(val_trues), np.concatenate(val_preds))[0]
        v_mse = val_loss/cnt
        metrics.append((tr_mse, tr_r, v_mse, v_r))
        logging.info(f"Seed {seed} Ep{epoch}: TrainMSE={tr_mse:.4f}, R={tr_r:.4f} | ValMSE={v_mse:.4f}, R={v_r:.4f}")

    # Save run
    path = os.path.join(MODEL_DIR, f'run_{seed}.pt')
    torch.save({'state_ref': refiner.state_dict(), 'state_tr': transformer.state_dict(), 'metrics': metrics}, path)
    return metrics

# --- Main ---
def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    all_metrics = {}
    for seed in range(NUM_RUNS):
        logging.info(f"Starting run seed={seed}")
        all_metrics[seed] = run_seed(seed)

    # Plot validation R
    plt.figure(figsize=(10,6))
    for seed, met in all_metrics.items():
        plt.plot(range(1, NUM_EPOCHS+1), [m[3] for m in met], alpha=0.3)
    plt.xlabel('Epoch'); plt.ylabel('Val Pearson R'); plt.title('Validation R per run')
    plt.savefig(PLOT_PATH)

    # Pick best
    best_seed, best_score = max(
        ((s, max(m[3] for m in met)) for s,met in all_metrics.items()),
        key=lambda x: x[1]
    )
    best_path = os.path.join(MODEL_DIR, f'run_{best_seed}.pt')
    os.replace(best_path, BEST_MODEL)
    logging.info(f"Best run={best_seed} with R={best_score:.4f}. Saved to {BEST_MODEL}")

if __name__=='__main__':
    
    main()
