import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --- Configuration ---
CSV_PATH = r'C:\Users\lndnc\OneDrive\Desktop\AI\DRWkaggle\train.csv'
TEST_PATH = CSV_PATH.replace('train.csv', 'test.csv')
TIME_COL = 'timestamp'
TARGET_COL = 'label'
NUM_BOOST_ROUND = 915
EARLY_STOP = 50

# NN settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 256
epochs = 50
lr = 1e-3

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Raw and engineered features
selected_features = [
    "X863", "X856", "X344", "X598", "X862", "X385", "X852", "X603", "X860", "X674",
    "X415", "X345", "X137", "X855", "X174", "X302", "X178", "X532", "X168", "X612",
    "bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume","X888", "X421", "X333"
]
# Targets to reconstruct
top10 = ['label', 'X345', 'X888', 'X862', 'X302', 'X532', 'X344', 'X385', 'X856', 'X178']
LGBM_PARAMS = {
    "boosting_type": "gbdt",
    "device_type": "gpu",
    "n_jobs": -1,
    "verbose": -1,
    "random_state": 42,
    "colsample_bytree": 0.5039,
    "learning_rate": 0.01260,
    "min_child_samples": 20,
    "min_child_weight": 0.1146,
    "num_leaves": 145,
    "reg_alpha": 19.2447,
    "reg_lambda": 55.5046,
    "subsample": 0.9709,
    "max_depth": 9
}
def add_features(df):
    """Compute all engineered features as before."""
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


class ReconNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, out_dim)
        )
    def forward(self, x):
        return self.net(x)
      
if __name__ == '__main__' == '__main__':
    # Load & feature-engineer
    df = pd.read_csv(CSV_PATH)
    df = add_features(df)
    # Align past features
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    X_all = df[selected_features].iloc[1:].reset_index(drop=True)
    y_recon_all = df[top10].shift(1).iloc[1:].reset_index(drop=True)
    y_final = df[TARGET_COL].iloc[1:].reset_index(drop=True)

    # Split
    X_train, X_val, y_recon_train, y_recon_val, y_train, y_val = train_test_split(
        X_all, y_recon_all, y_final, test_size=0.2, random_state=42, shuffle=False
    )

    # Scale
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled   = scaler.transform(X_val)

    # Create datasets
    train_ds = TensorDataset(torch.from_numpy(X_train_scaled).float(),
                             torch.from_numpy(y_recon_train.values).float())
    val_ds   = TensorDataset(torch.from_numpy(X_val_scaled).float(),
                             torch.from_numpy(y_recon_val.values).float())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    # Init ReconNN
    recon_model = ReconNN(len(selected_features), len(top10)).to(device)
    optimizer = torch.optim.Adam(recon_model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # NN training with early stopping
    best_loss = float('inf')
    wait = 0
    for ep in range(1, epochs+1):
        recon_model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = recon_model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        recon_model.eval()
        with torch.no_grad():
            out_val = recon_model(torch.from_numpy(X_val_scaled).float().to(device)).cpu().numpy()
        rmses = np.sqrt(((out_val - y_recon_val.values)**2).mean(axis=0))
        mean_rmse = rmses.mean()
        logging.info(f"Epoch {ep}/{epochs} recon RMSEs: {dict(zip(top10, np.round(rmses,4)))} Mean: {mean_rmse:.4f}")
        if mean_rmse < best_loss:
            best_loss = mean_rmse; wait=0
        else:
            wait+=1
            if wait>=2:
                logging.info("Early stopping NN training")
                break

    # Build LGBM training sets
    recon_train = recon_model(torch.from_numpy(X_train_scaled).float().to(device)).cpu().detach().numpy()
    recon_val   = recon_model(torch.from_numpy(X_val_scaled).float().to(device)).cpu().detach().numpy()
    X_train_final = np.hstack([X_train_scaled, recon_train])
    X_val_final   = np.hstack([X_val_scaled, recon_val])

    # Final LGBM
    dtrain = lgb.Dataset(X_train_final, label=y_train)
    dval   = lgb.Dataset(X_val_final,   label=y_val, reference=dtrain)
    final_model = lgb.train(
        params=LGBM_PARAMS,
        train_set=dtrain,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOP, verbose=True), lgb.log_evaluation(period=50)],
        num_boost_round=NUM_BOOST_ROUND
    )
    val_preds = final_model.predict(X_val_final, num_iteration=final_model.best_iteration)
    logging.info(f"Final LGBM RMSE: {root_mean_squared_error(y_val, val_preds):.6f}")
    # Compute Pearson R on validation hold-out
    from scipy.stats import pearsonr
    pearson_val = pearsonr(y_val, val_preds)[0]
    logging.info(f"Final LGBM Pearson R on validation: {pearson_val:.4f}")

    # Inference
    logging.info("Running inference...")
    df_test = pd.read_csv(TEST_PATH)
    df_test = add_features(df_test)
    X_test = scaler.transform(df_test[selected_features])
    recon_test = recon_model(torch.from_numpy(X_test).float().to(device)).cpu().detach().numpy()
    X_test_final = np.hstack([X_test, recon_test])
    final_preds = []
    for i in tqdm(range(len(X_test_final)), desc='Inference'):
        final_preds.append(final_model.predict(X_test_final[i:i+1])[0])
    pd.DataFrame({'ID': np.arange(1,len(final_preds)+1), 'prediction': final_preds}).to_csv('NN_LGBM_submission.csv', index=False)
    logging.info("Saved NN_LGBM_submission.csv")
