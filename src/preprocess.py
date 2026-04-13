import pandas as pd

def load_and_preprocess(filepath="energy.csv"):
    df = pd.read_csv(filepath, parse_dates=["Datetime"], index_col="Datetime")
    df = df.resample("h").mean()
    df = df.ffill().bfill()

    df["hour"]       = df.index.hour
    df["day"]        = df.index.dayofweek
    df["month"]      = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    df["quarter"]    = df.index.quarter
    df["lag_1h"]     = df["Energy"].shift(1)
    df["lag_24h"]    = df["Energy"].shift(24)
    df["lag_168h"]   = df["Energy"].shift(168)

    df.dropna(inplace=True)
    print(f"✅ Preprocessed: {len(df):,} rows, {df.shape[1]} columns")
    return df

def get_features_target(df):
    features = ["hour","day","month","is_weekend","quarter",
                "lag_1h","lag_24h","lag_168h"]
    return df[features], df["Energy"]