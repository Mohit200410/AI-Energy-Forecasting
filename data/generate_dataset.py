import pandas as pd
import numpy as np

np.random.seed(42)

date_range = pd.date_range(start="2022-01-01", end="2023-12-31 23:00", freq="h")
n = len(date_range)

hour = date_range.hour
base = (
    200
    + 80 * np.sin(2 * np.pi * hour / 24)
    + 30 * (hour >= 8) * (hour <= 20)
    - 20 * (hour >= 22)
)

dow = date_range.dayofweek
base -= 40 * (dow >= 5).astype(float)

month = date_range.month
base += 30 * np.cos(2 * np.pi * (month - 1) / 12)

noise = np.random.normal(0, 15, n)
energy = np.clip(base + noise, 50, 500)

df = pd.DataFrame({"Datetime": date_range, "Energy": np.round(energy, 2)})
df.to_csv("energy.csv", index=False)

print(f"✅ Dataset created: {len(df):,} rows")
print(df.head())