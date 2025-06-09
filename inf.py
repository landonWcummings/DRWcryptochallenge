import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import math # Needed for PositionalEncoding

# --- Configuration (MUST match training script) ---
# These parameters are directly taken from your provided training script.
# Do NOT change these unless your 'stock_transformer_20250608_122113.pt'
# file was trained with different values.
WINDOW_SIZE = 60 # 59 history + current
FEATURE_COLS = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
TARGET_COLUMN = 'label' # Although not used for inference features, it's part of the original RollingWindowDataset structure.

# Model Hyperparameters (MUST match training script's StockTransformer defaults)
FEATURE_DIM = 6 # This is feature_cols (5) + the padded 'label' dimension (1)
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 6
DIM_FEEDFORWARD = 256
DROPOUT = 0.1

# Training-related configuration for scaler (MUST match training script)
INITIAL_TRAIN_ROWS = 368000 # <--- THIS WAS MISSING AND NOW ADDED
BATCH_SIZE = 128            # <--- THIS WAS ALSO MISSING AND NOW ADDED

# Paths
MODEL_PATH = r'C:\Users\lndnc\OneDrive\Desktop\AI\DRWkaggle\stock_transformer_20250608_122113.pt'
TEST_DATA_PATH = 'test.csv'
SUBMISSION_PATH = 'sample_submission.csv'
TRAIN_CSV_PATH = 'C:/Users/lndnc/OneDrive/Desktop/AI/DRWkaggle/train.csv' # Added for clarity for scaler loading

# --- Positional Encoding (Copied directly from your training script) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (seq_len, batch, d_model)
        return x + self.pe[:x.size(0)].unsqueeze(1)

# --- Transformer Model (Copied directly from your training script) ---
class StockTransformer(nn.Module):
    def __init__(self, feature_dim=6, d_model=128, nhead=4, num_layers=6, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                 dim_feedforward=dim_feedforward, dropout=dropout,
                                                 batch_first=False)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = lambda x: x[-1] # This pools the last element of the sequence
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, feature_dim)
        x = x.transpose(0, 1)          # (seq_len, batch, feature_dim)
        x = self.input_proj(x)         # (seq_len, batch, d_model)
        x = self.pos_encoder(x)        # add pos encoding
        x = self.transformer(x)        # (seq_len, batch, d_model)
        x = self.pool(x)               # (batch, d_model) - pools the last element
        return self.head(x).squeeze(-1) # (batch,)

# --- Dataset for Inference (Adapted from your training script's RollingWindowDataset) ---
class InferenceRollingWindowDataset(Dataset):
    def __init__(self, df, window_size, feature_cols):
        self.window_size = window_size
        self.feature_cols = feature_cols
        
        if len(df) < window_size:
            raise ValueError(f"Input DataFrame has {len(df)} rows, which is less than WINDOW_SIZE ({window_size}). Cannot form sequences.")

        self.arr = df[feature_cols].values.astype(np.float32)

    def __len__(self):
        return len(self.arr) - self.window_size + 1

    def __getitem__(self, idx):
        hist_feats = self.arr[idx : idx + self.window_size - 1]
        hist_padded = np.pad(hist_feats, ((0, 0), (0, 1)), 'constant', constant_values=0.0)

        current_feats = self.arr[idx + self.window_size - 1, :]
        current_padded = np.concatenate([current_feats, np.array([0.0], dtype=np.float32)])

        seq = np.vstack([hist_padded, current_padded]).astype(np.float32)
        x = torch.from_numpy(seq)
        
        return x

# --- Main Inference Logic ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize the model with the exact architecture from training
    model = StockTransformer(
        feature_dim=FEATURE_DIM,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    ).to(device)

    # 2. Load the trained model state
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        print("Please ensure MODEL_PATH is correct and the StockTransformer architecture matches the saved model exactly.")
        return

    model.eval() # Set the model to evaluation mode

    # 3. Load and preprocess test data
    try:
        test_df = pd.read_csv(TEST_DATA_PATH)
        
        # Ensure test_df has all FEATURE_COLS
        missing_cols = [col for col in FEATURE_COLS if col not in test_df.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns in test.csv: {missing_cols}")
        
        # Load the original training data to fit the scaler
        # This emulates the scaler's state from the training process's initial fitting.
        train_df_for_scaler = pd.read_csv(TRAIN_CSV_PATH)
        
        # Fit scaler on the initial training rows, as done in your training script's first fold.
        scaler = StandardScaler().fit(train_df_for_scaler.iloc[:INITIAL_TRAIN_ROWS][FEATURE_COLS])
        
        # Apply scaler to test features
        test_df_scaled = test_df.copy()
        test_df_scaled[FEATURE_COLS] = scaler.transform(test_df_scaled[FEATURE_COLS])

        # Create the inference dataset
        inference_dataset = InferenceRollingWindowDataset(test_df_scaled, WINDOW_SIZE, FEATURE_COLS)
        inference_loader = DataLoader(
            inference_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False
        )
        print(f"Test data loaded and preprocessed. Will generate {len(inference_dataset)} predictions.")

    except Exception as e:
        print(f"Error loading or preprocessing test data: {e}")
        print("Please check your 'test.csv' file, feature columns, and ensure 'train.csv' is accessible for scaler fitting.")
        return

    # 4. Run inference
    predictions = []
    with torch.no_grad(): # Disable gradient calculation for inference
        for batch_x in inference_loader:
            batch_x = batch_x.to(device)
            output = model(batch_x)
            predictions.extend(output.cpu().numpy().flatten())

    print(f"Inference complete. Generated {len(predictions)} predictions.")

    # 5. Create submission file
    submission_df = pd.DataFrame({
        'ID': np.arange(1, len(predictions) + 1),
        'prediction': predictions
    })

    submission_df.to_csv(SUBMISSION_PATH, index=False)

    print(f"Submission file created at {SUBMISSION_PATH}")

if __name__ == "__main__":
    main()