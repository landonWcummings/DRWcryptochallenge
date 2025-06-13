import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from scipy.stats import pearsonr
import logging
from tqdm import tqdm

# --- Configuration ---
CSV_PATH      = r'C:\Users\lndnc\OneDrive\Desktop\AI\DRWkaggle\train.csv'
TEST_CSV_PATH = r'C:\Users\lndnc\OneDrive\Desktop\AI\DRWkaggle\test.csv'
TIME_COL      = 'timestamp'
TARGET_COL    = 'label'
BATCH_SIZE    = 128
PHASE1_EPOCHS = 6    # decoder pretrain max epochs
PATIENCE      = 2    # early stopping patience
PHASE2_EPOCHS = 10    # transformer training epochs (reduced)
LR            = 1e-3
DROPOUT       = 0.3
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    def __init__(self, df, shuffle_prev=False):
        df = df.sort_values(TIME_COL).reset_index(drop=True)
        prev = df[selected_features + [TARGET_COL]].shift(1).dropna().reset_index(drop=True)
        curr = df[selected_features].iloc[1:].reset_index(drop=True)
        labels = df[TARGET_COL].iloc[1:].reset_index(drop=True)
        self.X_curr = torch.tensor(curr.values, dtype=torch.float32)
        self.X_prev_true = torch.tensor(prev[selected_features].values, dtype=torch.float32)
        self.y = torch.tensor(labels.values, dtype=torch.float32)
        self.shuffle_prev = shuffle_prev
    def __len__(self): return len(self.X_curr)
    def __getitem__(self, idx):
        x_curr = self.X_curr[idx]
        x_prev = (self.X_curr[torch.randint(0, len(self), (1,)).item()] if self.shuffle_prev else self.X_prev_true[idx])
        return x_curr, x_prev, self.y[idx]

# --- Models ---
class LinearDecoder(nn.Module):
    def __init__(self, dim): super().__init__(); self.lin = nn.Linear(dim, dim)
    def forward(self, x): return self.lin(x)

class ResidualDecoder(nn.Module):
    def __init__(self, dim): super().__init__(); self.lin = nn.Linear(dim, dim)
    def forward(self, x): return x + self.lin(x)

class TransformerPredictor(nn.Module):
    def __init__(self, feat_dim, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim*2, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dropout=DROPOUT, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers)
        self.fc = nn.Linear(d_model, 1)
    def forward(self, x_curr, x_prev_pred):
        x = torch.cat([x_curr, x_prev_pred], dim=-1)
        z = self.transformer(self.input_proj(x).unsqueeze(1)).squeeze(1)
        return self.fc(z).squeeze(1)

# --- Phase 1: Train or load decoders ---
def train_decoder():
    lin_path, res_path = 'linear_decoder.pth', 'residual_decoder.pth'
    lin_dec = LinearDecoder(len(selected_features)).to(DEVICE)
    res_dec = ResidualDecoder(len(selected_features)).to(DEVICE)
    if os.path.exists(lin_path) and os.path.exists(res_path):
        lin_dec.load_state_dict(torch.load(lin_path, map_location=DEVICE))
        res_dec.load_state_dict(torch.load(res_path, map_location=DEVICE))
        logging.info('Loaded saved decoders')
        df = pd.read_csv(CSV_PATH); df = add_features(df)
        train_df = df.iloc[:int(0.8*len(df))]
        mean = train_df[selected_features].mean()
        std  = train_df[selected_features].std().replace(0,1)
        return lin_dec, res_dec, mean, std

    df = pd.read_csv(CSV_PATH); df = add_features(df)
    df = df[[TIME_COL] + selected_features + [TARGET_COL]].copy()
    cut = int(0.8*len(df)); train_df, val_df = df.iloc[:cut].copy(), df.iloc[cut:].copy()
    mean, std = train_df[selected_features].mean(), train_df[selected_features].std().replace(0,1)
    train_df.loc[:, selected_features] = (train_df[selected_features] - mean) / std
    val_df.loc[:, selected_features]   = (val_df[selected_features]   - mean) / std
    tl = DataLoader(TimeSeriesDataset(train_df, shuffle_prev=False), BATCH_SIZE, shuffle=True)
    vl = DataLoader(TimeSeriesDataset(val_df, shuffle_prev=False),   BATCH_SIZE)

    mse, mae = nn.MSELoss(), nn.L1Loss()
    opt_lin = optim.Adam(lin_dec.parameters(), lr=LR, weight_decay=1e-4)
    opt_res = optim.Adam(res_dec.parameters(), lr=LR, weight_decay=1e-4)

    best_val, wait = float('inf'), 0
    for ep in range(1, PHASE1_EPOCHS+1):
        lin_dec.train(); res_dec.train(); t_mse=t_mae=cnt=0
        for Xc, Xp, _ in tl:
            Xc, Xp = Xc.to(DEVICE), Xp.to(DEVICE)
            out_lin = lin_dec(Xc); out_res = res_dec(Xc)
            pf = (out_lin + out_res) / 2
            loss = mse(pf, Xp)
            opt_lin.zero_grad(); opt_res.zero_grad(); loss.backward()
            opt_lin.step(); opt_res.step()
            t_mse += loss.item()*Xc.size(0); t_mae += mae(pf,Xp).item()*Xc.size(0); cnt += Xc.size(0)
        lin_dec.eval(); res_dec.eval(); v_mse=v_mae=v_cnt=0
        with torch.no_grad():
            for Xc, Xp, _ in vl:
                Xc, Xp = Xc.to(DEVICE), Xp.to(DEVICE)
                pf = (lin_dec(Xc)+res_dec(Xc))/2
                v_mse += mse(pf,Xp).item()*Xc.size(0)
                v_mae += mae(pf,Xp).item()*Xc.size(0)
                v_cnt += Xc.size(0)
        logging.info(f"Ph1 Ep{ep} | TrMSE={t_mse/cnt:.4f} MAE={t_mae/cnt:.4f} ValMSE={v_mse/v_cnt:.4f} MAE={v_mae/v_cnt:.4f}")
        if (v_mse/v_cnt) < best_val:
            best_val=v_mse/v_cnt; wait=0
        else:
            wait+=1
            if wait>=PATIENCE:
                logging.info('Early stop decoders')
                break
    torch.save(lin_dec.state_dict(), lin_path)
    torch.save(res_dec.state_dict(), res_path)
    logging.info('Saved decoders')
    return lin_dec, res_dec, mean, std

# --- Phase 2: Transformer training with scheduled mixing ---
def train_transformer(lin_dec, res_dec, mean, std):
    # Data prep
    df = pd.read_csv(CSV_PATH); df = add_features(df)
    df = df[[TIME_COL] + selected_features + [TARGET_COL]].copy()
    cut = int(0.8 * len(df))
    train_df, val_df = df.iloc[:cut], df.iloc[cut:]
    for d in (train_df, val_df): d.loc[:, selected_features] = (d[selected_features] - mean) / std
    train_loader = DataLoader(TimeSeriesDataset(train_df, shuffle_prev=False), BATCH_SIZE, shuffle=True)
    val_ds = TimeSeriesDataset(val_df, shuffle_prev=True)
    n = len(val_ds)
    tv_ds, rv_ds = random_split(val_ds, [n//2, n - n//2])
    tv_loader, rv_loader = DataLoader(tv_ds, BATCH_SIZE), DataLoader(rv_ds, BATCH_SIZE)

    # Restart loop until first-epoch rand-val R > 0.1
    attempt = 0
    while True:
        attempt += 1
        transformer = TransformerPredictor(len(selected_features)).to(DEVICE)
        opt = optim.Adam(transformer.parameters(), lr=LR)
        mse, mae_loss = nn.MSELoss(), nn.L1Loss()
        # Run first epoch only
        transformer.train()
        # scheduled alpha=0 for first epoch (use true prev)
        alpha = 0.0
        for Xc, Xp_true, y in train_loader:
            Xc, Xp_true, y = Xc.to(DEVICE), Xp_true.to(DEVICE), y.to(DEVICE)
            with torch.no_grad(): Xp_dec = (lin_dec(Xc) + res_dec(Xc)) / 2
            Xp_mix = Xp_true  # alpha=0
            pred = transformer(Xc, Xp_mix)
            loss = mse(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
        # Evaluate rand-val on first epoch
        transformer.eval(); preds, trues = [], []
        with torch.no_grad():
            for Xc, Xp_true, y in rv_loader:
                Xc, y = Xc.to(DEVICE), y.to(DEVICE)
                Xp_dec = (lin_dec(Xc) + res_dec(Xc)) / 2
                out = transformer(Xc, Xp_dec)
                preds.append(out.cpu().numpy()); trues.append(y.cpu().numpy())
        r_rand = pearsonr(np.concatenate(trues), np.concatenate(preds))[0]
        logging.info(f"Init attempt {attempt}: first-epoch Rand-val R={r_rand:.4f}")
        if r_rand > 0.15:
            logging.info("Good init found, proceeding with full training.")
            break
        if attempt >= 12:
            logging.warning("Max attempts reached, proceeding anyway.")
            break

    # Full training over PHASE2_EPOCHS
    for ep in range(1, PHASE2_EPOCHS+1):
        alpha = (ep - 1) / (PHASE2_EPOCHS - 1)
        transformer.train(); tr_mse=tr_mae=cnt=0
        for Xc, Xp_true, y in train_loader:
            Xc, Xp_true, y = Xc.to(DEVICE), Xp_true.to(DEVICE), y.to(DEVICE)
            with torch.no_grad(): Xp_dec = (lin_dec(Xc) + res_dec(Xc)) / 2
            Xp_mix = (1 - alpha)*Xp_true + alpha*Xp_dec
            pred = transformer(Xc, Xp_mix)
            loss = mse(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_mse += loss.item()*Xc.size(0)
            tr_mae += mae_loss(pred, y).item()*Xc.size(0)
            cnt += Xc.size(0)
        # Validation logging ... (same as before)
        # [omitted for brevity]
        logging.info(f"Ph2 Ep{ep}: alpha={alpha:.2f}, TrMSE={tr_mse/cnt:.4f}, MAE={tr_mae/cnt:.4f}")

    return transformer

# --- Phase 3: Inference ---
def inference(lin_dec, res_dec, transformer, mean, std):
    df = pd.read_csv(TEST_CSV_PATH); df = add_features(df)
    feats = (df[selected_features] - mean) / std
    Xc = torch.tensor(feats.values, dtype=torch.float32).to(DEVICE)
    lin_dec.eval(); res_dec.eval(); transformer.eval()
    preds = []
    for x in tqdm(Xc, desc='Inf'):
        Xp_dec = (lin_dec(x.unsqueeze(0)) + res_dec(x.unsqueeze(0))) / 2
        pred = transformer(x.unsqueeze(0), Xp_dec)
        preds.append(pred.item())
    pd.DataFrame({'ID': np.arange(1, len(preds)+1), 'prediction': preds}).to_csv(
        'transformer_submission.csv', index=False
    )

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    lin_dec, res_dec, mean, std = train_decoder()
    transformer = train_transformer(lin_dec, res_dec, mean, std)
    inference(lin_dec, res_dec, transformer, mean, std)
