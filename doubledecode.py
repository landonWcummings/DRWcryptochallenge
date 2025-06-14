import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Configuration ---
CSV_PATH      = r'C:\Users\lando\Desktop\AI\DRWcryptochallenge\train.csv'
TIME_COL      = 'timestamp'
TARGET_COL    = 'label'
BATCH_SIZE    = 128
NUM_EPOCHS    = 6  # progressive mix
LR            = 1e-3
DROPOUT       = 0.3
NUM_RUNS      = 3   # train 7 models
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
    df['bid_ask_interaction']   = df['bid_qty'] * df['ask_qty']
    df['bid_buy_interaction']   = df['bid_qty'] * df['buy_qty']
    df['bid_sell_interaction']  = df['bid_qty'] * df['sell_qty']
    df['ask_buy_interaction']   = df['ask_qty'] * df['buy_qty']
    df['ask_sell_interaction']  = df['ask_qty'] * df['sell_qty']
    df['buy_sell_interaction']  = df['buy_qty'] * df['sell_qty']
    df['spread_indicator']      = (df['ask_qty'] - df['bid_qty']) / (df['ask_qty'] + df['bid_qty'] + 1e-8)
    df['volume_weighted_buy']   = df['buy_qty'] * df['volume']
    df['volume_weighted_sell']  = df['sell_qty'] * df['volume']
    df['volume_weighted_bid']   = df['bid_qty'] * df['volume']
    df['volume_weighted_ask']   = df['ask_qty'] * df['volume']
    df['buy_sell_ratio']        = df['buy_qty'] / (df['sell_qty'] + 1e-8)
    df['bid_ask_ratio']         = df['bid_qty'] / (df['ask_qty'] + 1e-8)
    df['order_flow_imbalance']  = (df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-8)
    df['buying_pressure']       = df['buy_qty'] / (df['volume'] + 1e-8)
    df['selling_pressure']      = df['sell_qty'] / (df['volume'] + 1e-8)
    df['total_liquidity']       = df['bid_qty'] + df['ask_qty']
    df['liquidity_imbalance']   = (df['bid_qty'] - df['ask_qty']) / (df['total_liquidity'] + 1e-8)
    df['relative_spread']       = (df['ask_qty'] - df['bid_qty']) / (df['volume'] + 1e-8)
    df['trade_intensity']       = (df['buy_qty'] + df['sell_qty']) / (df['volume'] + 1e-8)
    df['avg_trade_size']        = df['volume'] / (df['buy_qty'] + df['sell_qty'] + 1e-8)
    df['net_trade_flow']        = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + 1e-8)
    df['depth_ratio']           = df['total_liquidity'] / (df['volume'] + 1e-8)
    df['volume_participation']  = (df['buy_qty'] + df['sell_qty']) / (df['total_liquidity'] + 1e-8)
    df['market_activity']       = df['volume'] * df['total_liquidity']
    df['effective_spread_proxy']= np.abs(df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-8)
    df['realized_volatility_proxy'] = np.abs(df['order_flow_imbalance']) * df['volume']
    df['normalized_buy_volume'] = df['buy_qty'] / (df['bid_qty'] + 1e-8)
    df['normalized_sell_volume']= df['sell_qty'] / (df['ask_qty'] + 1e-8)
    df['liquidity_adjusted_imbalance'] = df['order_flow_imbalance'] * df['depth_ratio']
    df['pressure_spread_interaction']  = df['buying_pressure'] * df['spread_indicator']
    df['trade_direction_ratio'] = df['buy_qty'] / (df['buy_qty'] + df['sell_qty'] + 1e-8)
    df['net_buy_volume']        = df['buy_qty'] - df['sell_qty']
    df['bid_skew']              = df['bid_qty'] / (df['bid_qty'] + df['ask_qty'] + 1e-8)
    df['ask_skew']              = df['ask_qty'] / (df['bid_qty'] + df['ask_qty'] + 1e-8)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df

# --- Spoofers ---
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

# --- Dataset ---
class TimeSeriesDataset(Dataset):
    def __init__(self, df):
        df = df.sort_values(TIME_COL).reset_index(drop=True)
        prev = df[selected_features + [TARGET_COL]].shift(1).dropna()
        curr = df[selected_features].iloc[1:].reset_index(drop=True)
        self.X_curr       = torch.tensor(curr.values, dtype=torch.float32)
        self.X_prev       = torch.tensor(prev[selected_features].values, dtype=torch.float32)
        self.y_prev_feat  = self.X_prev.clone()
        self.y_prev_label = torch.tensor(prev[TARGET_COL].values, dtype=torch.float32)
        self.y_curr_label = torch.tensor(df[TARGET_COL].iloc[1:].values, dtype=torch.float32)
    def __len__(self):
        return len(self.X_curr)
    def __getitem__(self, idx):
        return (
            self.X_curr[idx],
            self.X_prev[idx],
            self.y_prev_feat[idx],
            self.y_prev_label[idx],
            self.y_curr_label[idx]
        )

# --- Models ---
class DecoderNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.out_feat  = nn.Linear(hidden_dim, input_dim)
        self.out_label = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        h = self.net(x)
        return self.out_feat(h), self.out_label(h).squeeze(1)

class TransformerPredictor(nn.Module):
    def __init__(self, feat_dim, d_model=128, nhead=4, num_layers=2, dropout=DROPOUT):
        super().__init__()
        # input = X_curr (F) + X_prev_input (F) + pl_prev (1) + pl_curr (1) = 2F+2
        self.input_proj = nn.Linear(feat_dim * 2 + 2, d_model)
        encoder_layer  = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x_curr, x_prev_input, pl_prev, pl_curr):
        # x_curr, x_prev_input: [B×F], pl_prev/pl_curr: [B]
        lbls = torch.stack([pl_prev, pl_curr], dim=1)  # [B×2]
        x = torch.cat([x_curr, x_prev_input, lbls], dim=1)  # [B×(2F+2)]
        x = self.input_proj(x).unsqueeze(1)                # [B×1×d_model]
        z = self.transformer(x).squeeze(1)                 # [B×d_model]
        return self.fc(z).squeeze(1)                       # [B]

# --- Single run ---
def train_one(run_id, mean, std, train_loader, val_loader, spoofers):
    linear_spoofer, resid_spoofer = spoofers
    torch.manual_seed(42 + run_id)
    dec = DecoderNN(len(selected_features)).to(DEVICE)
    trf = TransformerPredictor(len(selected_features)).to(DEVICE)
    opt_dec = optim.Adam(dec.parameters(), lr=LR)
    opt_tr  = optim.Adam(trf.parameters(), lr=LR)
    mse = nn.MSELoss()

    epoch_rs = []
    for epoch in range(1, NUM_EPOCHS + 1):
        alpha = (epoch - 1) / (NUM_EPOCHS - 1)
        dec.train(); trf.train()
        for Xc, Xp, ypf, ypl, ycl in train_loader:
            Xc, Xp, ypf, ypl, ycl = [t.to(DEVICE) for t in (Xc, Xp, ypf, ypl, ycl)]
            # generate spoofed yesterday features
            s1 = linear_spoofer(Xc)
            s2 = resid_spoofer(Xc)
            spoof = 0.5 * (s1 + s2)
            # decoder reconstructs yesterday features & label
            dec_in    = (1 - alpha) * Xp + alpha * spoof.detach()
            pf_prev, pl_prev = dec(dec_in)
            # decoder predicts today's label from X_curr
            _, pl_curr = dec(Xc)
            # mix reconstructed previous features
            prev_input = (1 - alpha) * Xp + alpha * pf_prev.detach()
            # transformer prediction
            tr_pred = trf(Xc, prev_input, pl_prev.detach(), pl_curr.detach())
            loss = mse(pf_prev, ypf) + mse(pl_prev, ypl) + mse(tr_pred, ycl)

            opt_dec.zero_grad(); opt_tr.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
            nn.utils.clip_grad_norm_(trf.parameters(), 1.0)
            opt_dec.step(); opt_tr.step()

        # validation
        dec.eval(); trf.eval()
        preds, trues = [], []
        with torch.no_grad():
            for Xc, Xp, _, _, ycl in val_loader:
                Xc, Xp = Xc.to(DEVICE), Xp.to(DEVICE)
                s1 = linear_spoofer(Xc)
                s2 = resid_spoofer(Xc)
                spoof = 0.5 * (s1 + s2)
                dec_in, _ = ((1 - alpha) * Xp + alpha * spoof, )
                pf_prev, pl_prev = dec((1 - alpha) * Xp + alpha * spoof)
                _, pl_curr = dec(Xc)
                prev_input = (1 - alpha) * Xp + alpha * pf_prev
                pc = trf(Xc, prev_input, pl_prev, pl_curr)
                preds.extend(pc.cpu().numpy())
                trues.extend(ycl.numpy())

        r = pearsonr(trues, preds)[0]
        logging.info(f"Run {run_id} Epoch {epoch}/{NUM_EPOCHS} alpha={alpha:.2f} Pearson={r:.4f}")
        epoch_rs.append(r)

    return dec, trf, epoch_rs

# --- Main ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    # load & preprocess
    df = pd.read_csv(CSV_PATH)
    df = add_features(df)
    df = df[[TIME_COL] + selected_features + [TARGET_COL]]
    cut = int(0.8 * len(df))
    train_df, val_df = df.iloc[:cut].copy(), df.iloc[cut:].copy()

    # standardize
    mean = train_df[selected_features].mean()
    std  = train_df[selected_features].std().replace(0, 1)
    for d in (train_df, val_df):
        d[selected_features] = (d[selected_features] - mean) / std

    # load spoofers
    linear_spoofer = LinearDecoder(len(selected_features)).to(DEVICE)
    resid_spoofer  = ResidualDecoder(len(selected_features)).to(DEVICE)
    linear_spoofer.load_state_dict(torch.load('linear_decoder.pth'))
    resid_spoofer.load_state_dict(torch.load('residual_decoder.pth'))
    spoofers = (linear_spoofer.eval(), resid_spoofer.eval())

    # print spoofer metrics
    Xc_all    = torch.tensor(val_df[selected_features].values, dtype=torch.float32).to(DEVICE)
    Xc_val    = Xc_all[1:]
    true_prev = val_df[selected_features].iloc[:-1].values
    with torch.no_grad():
        s1     = spoofers[0](Xc_val)
        s2     = spoofers[1](Xc_val)
        sp_avg = 0.5 * (s1 + s2).cpu().numpy()
    logging.info(f"Linear MAE={mean_absolute_error(true_prev, s1.cpu().numpy()):.4f} "
                 f"MSE={mean_squared_error(true_prev, s1.cpu().numpy()):.4f}")
    logging.info(f"Residual MAE={mean_absolute_error(true_prev, s2.cpu().numpy()):.4f} "
                 f"MSE={mean_squared_error(true_prev, s2.cpu().numpy()):.4f}")
    logging.info(f"Average MAE={mean_absolute_error(true_prev, sp_avg):.4f} "
                 f"MSE={mean_squared_error(true_prev, sp_avg):.4f}")

    # data loaders
    train_loader = DataLoader(TimeSeriesDataset(train_df), batch_size=BATCH_SIZE, shuffle=False)
    val_loader   = DataLoader(TimeSeriesDataset(val_df),   batch_size=BATCH_SIZE)

    # run training
    all_runs   = {}
    best_r     = -np.inf
    best_models = None
    for run in range(1, NUM_RUNS + 1):
        dec, trf, rs = train_one(run, mean, std, train_loader, val_loader, spoofers)
        all_runs[run] = rs
        final_r = rs[-1]
        if final_r > best_r:
            best_r     = final_r
            best_models = (dec.state_dict(), trf.state_dict())

    # save best
    os.makedirs('models', exist_ok=True)
    torch.save(best_models[0], 'models/best_decoder.pt')
    torch.save(best_models[1], 'models/best_transformer.pt')
    logging.info(f"Saved best model with Pearson={best_r:.4f}")

    # plot
    plt.figure(figsize=(10, 6))
    for run, rs in all_runs.items():
        plt.plot(range(1, NUM_EPOCHS + 1), rs, label=f'Run {run}')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Pearson R')
    plt.title('Pearson R per Epoch for Each Run')
    plt.legend()
    plt.grid(True)
    plt.savefig('run_comparison.png')
    plt.show()

    # inference
    test_df = pd.read_csv(CSV_PATH.replace('train.csv', 'test.csv'))
    test_df = add_features(test_df)
    tf = (test_df[selected_features] - mean) / std
    test_dataset = torch.tensor(tf.values, dtype=torch.float32)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

    best_dec = DecoderNN(len(selected_features)).to(DEVICE)
    best_trf = TransformerPredictor(len(selected_features)).to(DEVICE)
    best_dec.load_state_dict(torch.load('models/best_decoder.pt'))
    best_trf.load_state_dict(torch.load('models/best_transformer.pt'))
    best_dec.eval(); best_trf.eval()

    all_preds = []
    with torch.no_grad():
        for Xc in tqdm(test_loader, desc='Inference', unit='row'):
            Xc = Xc.to(DEVICE)
            s1 = spoofers[0](Xc)
            s2 = spoofers[1](Xc)
            sp_avg = 0.5 * (s1 + s2)
            pf_prev, pl_prev = best_dec(sp_avg)
            _, pl_curr     = best_dec(Xc)
            prev_input     = pf_prev
            pred = best_trf(Xc, prev_input, pl_prev, pl_curr)
            all_preds.append(pred.item())

    submission = pd.DataFrame({
        'ID': np.arange(1, len(all_preds) + 1),
        'prediction': all_preds
    })
    submission.to_csv('doubledecode_submission.csv', index=False)
    logging.info('Saved doubledecode_submission.csv')
