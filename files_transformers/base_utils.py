import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# Configuration (common)
WINDOW_SIZE = 60  # default window size, can be set to 1 for other models
TS_COL = 'timestamp'
TARGET_COL = 'label'

class RollingWindowDataset(Dataset):
    """
    Generic rolling window dataset for time series data.
    df: DataFrame with feature columns + TARGET_COL.
    window_size: number of timesteps per sample.
    feature_cols: list of columns to use as features.
    """
    def __init__(self, df, window_size, feature_cols, target_col):
        self.window_size = window_size
        self.features = df[feature_cols].values.astype(np.float32)
        self.targets = df[target_col].values.astype(np.float32)

    def __len__(self):
        return len(self.features) - self.window_size + 1

    def __getitem__(self, idx):
        x = self.features[idx: idx + self.window_size]
        y = self.targets[idx + self.window_size - 1]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


def load_and_split_random(df, test_segments=20, segment_frac=0.01, seed=42):
    """
    Splits df into train/test by randomly selecting 'test_segments' segments each of
    length segment_frac * len(df). Ensures no windows start before WINDOW_SIZE.
    Returns train_df, test_df.
    """
    N = len(df)
    seg_len = int(N * segment_frac)
    rng = np.random.default_rng(seed=seed)

    valid_starts = np.arange(WINDOW_SIZE - 1, N - seg_len)
    test_starts = rng.choice(valid_starts, size=test_segments, replace=False)
    test_idxs = []
    for s in test_starts:
        test_idxs.extend(range(s, s + seg_len))
    mask = np.zeros(N, dtype=bool)
    mask[test_idxs] = True
    mask[:WINDOW_SIZE - 1] = False
    train_df = df.loc[~mask].reset_index(drop=True)
    test_df = df.loc[mask].reset_index(drop=True)
    logging.info(f"Train points: {len(train_df)}, Test points: {len(test_df)}")
    return train_df, test_df


def clean_and_scale(df_train, df_test, feature_cols):
    """
    Replace inf/nan in features with zero and apply StandardScaler.
    Returns scaled numpy arrays for train and test.
    """
    for df_ in [df_train, df_test]:
        df_[feature_cols] = df_[feature_cols] \
            .replace([np.inf, -np.inf], np.nan) \
            .fillna(0)
    scaler = StandardScaler().fit(df_train[feature_cols])
    X_tr = scaler.transform(df_train[feature_cols])
    X_te = scaler.transform(df_test[feature_cols])
    return X_tr, X_te, scaler


def apply_pca(X_tr, X_te, n_components=30):
    """
    Fit PCA on X_tr and transform both X_tr and X_te.
    Returns X_tr_pca, X_te_pca, pca_model.
    """
    pca = PCA(n_components=n_components)
    X_tr_pca = pca.fit_transform(X_tr)
    X_te_pca = pca.transform(X_te)
    return X_tr_pca, X_te_pca, pca
