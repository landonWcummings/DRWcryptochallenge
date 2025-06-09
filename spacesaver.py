import pandas as pd
import numpy as np

# ─── USER CONFIG ───────────────────────────────────────────────────────────────
INPUT_FILE  = r"c:\Users\lndnc\Downloads\drw-crypto-market-prediction\train.parquet"
OUTPUT_FILE = r"C:\Users\lndnc\OneDrive\Desktop\AI\DRWkaggle\train.csv"
# ────────────────────────────────────────────────────────────────────────────────

def reduce_mem_usage(df):
    """
    Optimizes the memory usage of a DataFrame by downcasting numeric columns to smaller data types.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage of dataframe is {start_mem:.2f} MB")

    for col in df.columns:
        col_type = df[col].dtype
        if str(col_type) == "category":
            continue
        if col_type != object:
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type).startswith("int"):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage after optimization is: {end_mem:.2f} MB")
    print(f"Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%")
    return df

def main():
    # Load
    df = pd.read_parquet(INPUT_FILE)
    print(f"Loaded DataFrame with shape: {df.shape}")

    # Optimize
    df_opt = reduce_mem_usage(df)

    # Save
    df_opt.to_csv(OUTPUT_FILE, index=True)
    print(f"Saved optimized DataFrame to CSV at: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
