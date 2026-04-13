import os, sys, joblib, numpy as np, matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocess import load_and_preprocess, get_features_target

def train():
    df = load_and_preprocess("energy.csv")
    X, y = get_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu", solver="adam",
        max_iter=500, early_stopping=True,
        validation_fraction=0.1, random_state=42
    )
    print("🔄 Training MLP...")
    model.fit(X_train_s, y_train)

    preds = model.predict(X_test_s)
    mae   = mean_absolute_error(y_test, preds)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    r2    = r2_score(y_test, preds)

    print(f"MAE: {mae:.2f}  RMSE: {rmse:.2f}  R²: {r2:.4f}")

    os.makedirs("models",  exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    joblib.dump(model,  "models/energy_forecast_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    with open("outputs/metrics.txt","w") as f:
        f.write(f"MAE  : {mae:.2f} kWh\nRMSE : {rmse:.2f} kWh\nR²   : {r2:.4f}\n")

    # Plot
    test_dates = df.index[-len(y_test):]
    fig, axes = plt.subplots(2, 1, figsize=(15, 9))
    axes[0].plot(test_dates, y_test.values, label="Actual",    color="#2196F3", lw=1.2)
    axes[0].plot(test_dates, preds,         label="Predicted", color="#FF5722", lw=1.2, alpha=0.8)
    axes[0].set_title("Actual vs Predicted – Full Test Period")
    axes[0].legend()

    zoom = 7 * 24
    axes[1].plot(test_dates[:zoom], y_test.values[:zoom], color="#2196F3", lw=1.5, label="Actual")
    axes[1].plot(test_dates[:zoom], preds[:zoom],         color="#FF5722", lw=1.5, alpha=0.85, label="Predicted")
    axes[1].set_title("Zoom – First 7 Days")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("outputs/actual_vs_predicted.png", dpi=150)
    plt.close()

    errors = y_test.values - preds
    plt.figure(figsize=(8, 4))
    plt.hist(errors, bins=60, color="#4CAF50", edgecolor="white", alpha=0.85)
    plt.axvline(0, color="red", linestyle="--")
    plt.title("Prediction Error Distribution")
    plt.tight_layout()
    plt.savefig("outputs/error_distribution.png", dpi=150)
    plt.close()

    print("✅ Model, scaler, charts saved!")
    return model, scaler, mae, rmse, r2

if __name__ == "__main__":
    train()