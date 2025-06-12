import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

# --- Configuration ---
CSV_PATH      = r'C:\Users\lndnc\OneDrive\Desktop\AI\DRWkaggle\train.csv'
TEST_CSV_PATH = r'C:\Users\lndnc\OneDrive\Desktop\AI\DRWkaggle\test.csv'
TIME_COL      = 'timestamp'
TARGET_COL    = 'label'
BATCH_SIZE    = 128
NUM_EPOCHS    = 5
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
    # engineered features only
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

# --- Dataset ---
class TimeSeriesDataset(Dataset):
    def __init__(self, df):
        df = df.sort_values(TIME_COL).reset_index(drop=True)
        df_prev = df[selected_features + [TARGET_COL]].shift(1)
        df_comb = pd.concat([df[selected_features], df_prev.add_suffix('_prev')], axis=1)
        df_comb[TARGET_COL] = df[TARGET_COL]
        df_comb = df_comb.dropna()
        self.X_curr = torch.tensor(df_comb[selected_features].values, dtype=torch.float32)
        self.X_prev = torch.tensor(df_comb[[f + '_prev' for f in selected_features]].values, dtype=torch.float32)
        self.y_prev_feat = self.X_prev.clone()
        self.y_prev_label= torch.tensor(df_comb[TARGET_COL + '_prev'].values, dtype=torch.float32)
        self.y_curr_label= torch.tensor(df_comb[TARGET_COL].values, dtype=torch.float32)
    def __len__(self): return len(self.X_curr)
    def __getitem__(self, idx):
        return (self.X_curr[idx], self.X_prev[idx], self.y_prev_feat[idx], self.y_prev_label[idx], self.y_curr_label[idx])

# --- Models ---
class DecoderNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(DROPOUT)
        )
        self.out_feat  = nn.Linear(hidden_dim, input_dim)
        self.out_label = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        h = self.net(x)
        return self.out_feat(h), self.out_label(h).squeeze(1)

class TransformerPredictor(nn.Module):
    def __init__(self, feat_dim, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim * 2, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dropout=DROPOUT)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 1)
    def forward(self, x_curr, x_prev_pred):
        x = torch.cat([x_curr, x_prev_pred], dim=-1)
        x = self.input_proj(x).unsqueeze(1)
        z = self.transformer(x).squeeze(1)
        return self.fc(z).squeeze(1)

# --- Main ---
def main():
    df = pd.read_csv(CSV_PATH)
    df = add_features(df)
    df = df[[TIME_COL] + selected_features + [TARGET_COL]]
    cut = int(0.8 * len(df))
    train_df = df.iloc[:cut].copy()
    val_df   = df.iloc[cut:].copy()

    mean = train_df[selected_features].mean()
    std  = train_df[selected_features].std().replace(0,1)
    train_df[selected_features] = (train_df[selected_features] - mean) / std
    val_df[selected_features]   = (val_df[selected_features] - mean) / std

    train_ds = TimeSeriesDataset(train_df)
    val_ds   = TimeSeriesDataset(val_df)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

    decoder = DecoderNN(len(selected_features)).to(DEVICE)
    transformer = TransformerPredictor(len(selected_features)).to(DEVICE)
    mse = nn.MSELoss()
    opt_dec = optim.Adam(decoder.parameters(), lr=LR)
    opt_tr  = optim.Adam(transformer.parameters(), lr=LR)

    for epoch in range(1, NUM_EPOCHS+1):
        decoder.train(); transformer.train()
        total_loss = 0.0
        for Xc, Xp, ypf, ypl, ycl in train_loader:
            Xc, Xp, ypf, ypl, ycl = [t.to(DEVICE) for t in (Xc, Xp, ypf, ypl, ycl)]
            pf, pl = decoder(Xp)
            pf = torch.nan_to_num(pf)
            pl = torch.nan_to_num(pl)
            loss_dec = mse(pf, ypf) + mse(pl, ypl)
            pc = transformer(Xc, pf.detach())
            pc = torch.nan_to_num(pc)
            loss_tr = mse(pc, ycl)
            loss = loss_dec + loss_tr
            if torch.isnan(loss): continue
            opt_dec.zero_grad(); opt_tr.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(),1.0)
            torch.nn.utils.clip_grad_norm_(transformer.parameters(),1.0)
            opt_dec.step(); opt_tr.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        decoder.eval(); transformer.eval()
        val_losses, preds, trues = [], [], []
        with torch.no_grad():
            for Xc, Xp, ypf, ypl, ycl in val_loader:
                Xc, Xp, ycl = Xc.to(DEVICE), Xp.to(DEVICE), ycl.to(DEVICE)
                pf, pl = decoder(Xp); pf = torch.nan_to_num(pf)
                pc = transformer(Xc, pf)
                pc = torch.nan_to_num(pc)
                val_losses.append(mse(pc, ycl).item())
                preds.extend(pc.cpu().numpy()); trues.extend(ycl.cpu().numpy())
        val_mse = np.mean(val_losses); val_r = pearsonr(trues, preds)[0]
        logging.info(f"Epoch {epoch} TrainLoss={avg_loss:.4f} ValMSE={val_mse:.6f} ValR={val_r:.4f}")

    # Final hold-out evaluation
    decoder.eval(); transformer.eval()
    test_losses, test_preds, test_trues = [], [], []
    with torch.no_grad():
        for Xc, Xp, ypf, ypl, ycl in val_loader:
            Xc, Xp, ycl = Xc.to(DEVICE), Xp.to(DEVICE), ycl.to(DEVICE)
            pf, pl = decoder(Xp); pf = torch.nan_to_num(pf)
            pc = transformer(Xc, pf)
            pc = torch.nan_to_num(pc)
            test_losses.append(mse(pc, ycl).item())
            test_preds.extend(pc.cpu().numpy()); test_trues.extend(ycl.cpu().numpy())
    final_mse = np.mean(test_losses); final_r   = pearsonr(test_trues, test_preds)[0]
    logging.info(f"FINAL EVAL — MSE={final_mse:.6f}, Pearson R={final_r:.4f}")

                # --- Inference on new test data with progress tracking ---
    from tqdm import tqdm
    logging.info("Starting inference on test data...")
    test_df = pd.read_csv(TEST_CSV_PATH)
    logging.debug(f"Loaded test data with shape {test_df.shape}")

    test_df = add_features(test_df)
    logging.debug("Applied feature engineering to test data")

    # Keep only selected features
    test_features = test_df[selected_features].copy()
    logging.debug(f"Isolated selected features, resulting shape {test_features.shape}")

    # Normalize
    test_features = (test_features - mean) / std
    logging.debug("Normalized test features using train mean/std")

    # Shift for previous-day features
    df_prev_test = test_features.shift(1)
    df_inf = pd.concat([test_features, df_prev_test.add_suffix('_prev')], axis=1).dropna()
    logging.info(f"Prepared inference DataFrame with prev-day features, shape {df_inf.shape}")

    Xc_test = torch.tensor(df_inf[selected_features].values, dtype=torch.float32).to(DEVICE)
    Xp_test = torch.tensor(df_inf[[f + '_prev' for f in selected_features]].values, dtype=torch.float32).to(DEVICE)
    n_samples = Xc_test.size(0)
    logging.info(f"Prepared tensors: Xc_test {Xc_test.shape}, Xp_test {Xp_test.shape}")

    decoder.eval(); transformer.eval()
    preds_test = []
    # Use tqdm to track progress
    for i in tqdm(range(n_samples), desc="Inference", unit="row"):
        with torch.no_grad():
            x_curr = Xc_test[i].unsqueeze(0)
            x_prev = Xp_test[i].unsqueeze(0)
            pf, _ = decoder(x_prev)
            pf = torch.nan_to_num(pf)
            pc = transformer(x_curr, pf)
            pc = torch.nan_to_num(pc)
            preds_test.append(pc.item())
    logging.info(f"Completed inference for {n_samples} samples")

    # Pad first day prediction with zero
    preds_full = [0.0] + preds_test
    preds_full = preds_full[:len(test_df)]
    logging.info(f"Padded predictions to full length {len(preds_full)}")

    # Save submission
    sub_df = pd.DataFrame({'ID': np.arange(1, len(preds_full) + 1), 'prediction': preds_full})
    sub_path = 'transformer_submission.csv'
    sub_df.to_csv(sub_path, index=False)
    logging.info(f"Saved inference results to {sub_path}")

if __name__ == '__main__':
    main()
