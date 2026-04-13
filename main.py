import os
from src.train_model import train

print("="*50)
print("  ⚡ AI Energy Forecasting — Full Pipeline")
print("="*50)

# Step 1: Generate dataset
if not os.path.exists("energy.csv"):
    print("\n📦 Generating dataset...")
    exec(open("data/generate_dataset.py").read())
else:
    print("\n✅ energy.csv already exists — skipping")

# Step 2: Train model
print("\n🤖 Training model...")
model, scaler, mae, rmse, r2 = train()

print("\n" + "="*50)
print("  ✅ Pipeline Complete!")
print(f"     MAE  : {mae:.2f} kWh")
print(f"     RMSE : {rmse:.2f} kWh")
print(f"     R²   : {r2:.4f}")
print("\n  Run dashboard:  streamlit run dashboard.py")
print("="*50)