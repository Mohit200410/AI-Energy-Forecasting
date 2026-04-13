"""
dashboard.py  –  AI Energy Forecasting · Streamlit Dashboard
=============================================================
Run:  streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import joblib
import os

# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be the very first st call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="⚡ Energy Forecasting AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS  – dark industrial theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- Font import ---------- */
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=DM+Sans:wght@300;400;600;700&display=swap');

/* ---------- Root palette ---------- */
:root {
    --bg:        #0d1117;
    --surface:   #161b22;
    --border:    #30363d;
    --accent:    #f0a500;
    --accent2:   #00d4aa;
    --text:      #e6edf3;
    --muted:     #8b949e;
    --danger:    #ff6b6b;
}

/* ---------- App background ---------- */
.stApp { background: var(--bg); color: var(--text);
         font-family: 'DM Sans', sans-serif; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

/* ---------- Metric cards ---------- */
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px !important;
}
[data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.8rem !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; }

/* ---------- Buttons ---------- */
.stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    font-weight: 700;
    border: none;
    border-radius: 8px;
    padding: 10px 28px;
    font-family: 'DM Sans', sans-serif;
    letter-spacing: 0.5px;
    transition: transform .15s, box-shadow .15s;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(240,165,0,.35);
}

/* ---------- Sliders ---------- */
[data-testid="stSlider"] > div > div > div {
    background: var(--accent) !important;
}

/* ---------- DataFrames ---------- */
.dataframe { background: var(--surface) !important; color: var(--text) !important; }

/* ---------- Section header helper ---------- */
.section-title {
    font-family: 'Share Tech Mono', monospace;
    color: var(--accent);
    font-size: 1.1rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin-bottom: 20px;
}

/* ---------- Prediction result box ---------- */
.pred-box {
    background: linear-gradient(135deg, #1a2a0a 0%, #0d1a1a 100%);
    border: 1px solid var(--accent2);
    border-radius: 16px;
    padding: 28px 36px;
    text-align: center;
    margin: 20px 0;
}
.pred-value {
    font-family: 'Share Tech Mono', monospace;
    font-size: 3rem;
    color: var(--accent2);
    line-height: 1;
}
.pred-label {
    color: var(--muted);
    font-size: 0.9rem;
    margin-top: 6px;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ---------- Info / warning boxes ---------- */
.stInfo, .stSuccess, .stWarning {
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  CACHED DATA & MODEL LOADERS
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("energy.csv", parse_dates=["Datetime"], index_col="Datetime")
    df = df.resample("h").mean().ffill().bfill()
    df["hour"]       = df.index.hour
    df["day"]        = df.index.dayofweek
    df["month"]      = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    df["quarter"]    = df.index.quarter
    df["lag_1h"]     = df["Energy"].shift(1)
    df["lag_24h"]    = df["Energy"].shift(24)
    df["lag_168h"]   = df["Energy"].shift(168)
    df.dropna(inplace=True)
    return df


@st.cache_resource
def load_model():
    model  = joblib.load("models/energy_forecast_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler


# ─────────────────────────────────────────────
#  SIDEBAR  –  Navigation + Info
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px 0;'>
        <span style='font-size:2.4rem;'>⚡</span><br>
        <span style='font-family:"Share Tech Mono",monospace;
                     color:#f0a500; font-size:1rem; letter-spacing:3px;'>
            ENERGY AI
        </span><br>
        <span style='color:#8b949e; font-size:0.75rem; letter-spacing:1px;'>
            FORECASTING DASHBOARD
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Navigate",
        ["📊 Overview",
         "🔮 Live Forecast",
         "📈 Model Performance",
         "💡 Energy Insights"],
        label_visibility="collapsed",
    )

    st.divider()

    # Quick stats in sidebar
    try:
        df_side = load_data()
        st.markdown("<p style='color:#8b949e;font-size:0.78rem;letter-spacing:1px;text-transform:uppercase;'>Dataset</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#e6edf3;font-family:\"Share Tech Mono\",monospace;margin:0;'>📅 {df_side.index.min().strftime('%b %Y')} → {df_side.index.max().strftime('%b %Y')}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#e6edf3;font-family:\"Share Tech Mono\",monospace;margin:0;'>🔢 {len(df_side):,} hourly readings</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#e6edf3;font-family:\"Share Tech Mono\",monospace;margin:0;'>⚡ Avg {df_side['Energy'].mean():.1f} kWh</p>", unsafe_allow_html=True)
    except:
        st.warning("Run main.py first to generate data & model.")

    st.divider()
    st.markdown("<p style='color:#8b949e;font-size:0.72rem;text-align:center;'>Model · MLP Regressor<br>Features · 8 engineered<br>Built with Python 3.11</p>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD DATA & MODEL
# ─────────────────────────────────────────────
try:
    df    = load_data()
    model, scaler = load_model()
    data_ok = True
except Exception as e:
    st.error(f"⚠️ Could not load data/model: {e}\n\nPlease run `python main.py` first.")
    data_ok = False
    st.stop()

FEATURES = ["hour", "day", "month", "is_weekend", "quarter",
            "lag_1h", "lag_24h", "lag_168h"]

# ─────────────────────────────────────────────
#  PAGE 1 — OVERVIEW
# ─────────────────────────────────────────────
if page == "📊 Overview":
    st.markdown("## 📊 Dataset Overview")
    st.markdown("Explore the historical smart-grid energy consumption data used to train the AI.")

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records",   f"{len(df):,}")
    col2.metric("Avg Consumption", f"{df['Energy'].mean():.1f} kWh")
    col3.metric("Peak Demand",     f"{df['Energy'].max():.1f} kWh")
    col4.metric("Min Demand",      f"{df['Energy'].min():.1f} kWh")

    st.divider()

    # Full time-series chart
    st.markdown('<p class="section-title">⚡ Full Energy Consumption Timeline</p>', unsafe_allow_html=True)

    date_range = st.slider(
        "Select date range to view",
        min_value=df.index.min().to_pydatetime(),
        max_value=df.index.max().to_pydatetime(),
        value=(df.index.min().to_pydatetime(), df.index.max().to_pydatetime()),
        format="MMM YYYY",
    )
    df_view = df.loc[date_range[0]:date_range[1]]

    fig, ax = plt.subplots(figsize=(14, 4))
    fig.patch.set_facecolor("#161b22")
    ax.set_facecolor("#0d1117")
    ax.plot(df_view.index, df_view["Energy"], color="#f0a500", linewidth=0.8, alpha=0.9)
    ax.fill_between(df_view.index, df_view["Energy"], alpha=0.12, color="#f0a500")
    ax.set_ylabel("Energy (kWh)", color="#8b949e")
    ax.set_xlabel("")
    ax.tick_params(colors="#8b949e")
    ax.spines[["top","right","left","bottom"]].set_color("#30363d")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.divider()

    col_a, col_b = st.columns(2)

    # Monthly average bar chart
    with col_a:
        st.markdown('<p class="section-title">📅 Monthly Average</p>', unsafe_allow_html=True)
        monthly = df.groupby("month")["Energy"].mean()
        month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        fig2.patch.set_facecolor("#161b22")
        ax2.set_facecolor("#0d1117")
        bars = ax2.bar(month_names, monthly.values, color="#f0a500", alpha=0.85, edgecolor="#30363d")
        ax2.set_ylabel("Avg Energy (kWh)", color="#8b949e")
        ax2.tick_params(colors="#8b949e")
        ax2.spines[["top","right","left","bottom"]].set_color("#30363d")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # Hourly distribution box
    with col_b:
        st.markdown('<p class="section-title">🕐 Hourly Distribution</p>', unsafe_allow_html=True)
        hourly = df.groupby("hour")["Energy"].mean()
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        fig3.patch.set_facecolor("#161b22")
        ax3.set_facecolor("#0d1117")
        ax3.fill_between(hourly.index, hourly.values, color="#00d4aa", alpha=0.3)
        ax3.plot(hourly.index, hourly.values, color="#00d4aa", linewidth=2)
        ax3.set_xlabel("Hour of Day", color="#8b949e")
        ax3.set_ylabel("Avg Energy (kWh)", color="#8b949e")
        ax3.tick_params(colors="#8b949e")
        ax3.spines[["top","right","left","bottom"]].set_color("#30363d")
        ax3.set_xticks(range(0, 24, 2))
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    st.divider()
    st.markdown('<p class="section-title">🗂️ Raw Data Preview</p>', unsafe_allow_html=True)
    st.dataframe(
        df[["Energy","hour","day","month","is_weekend","lag_1h","lag_24h"]].head(50),
        use_container_width=True,
        height=280,
    )


# ─────────────────────────────────────────────
#  PAGE 2 — LIVE FORECAST
# ─────────────────────────────────────────────
elif page == "🔮 Live Forecast":
    st.markdown("## 🔮 Live Energy Forecast")
    st.markdown("Adjust the parameters below and get an instant AI prediction.")

    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.markdown('<p class="section-title">⚙️ Input Parameters</p>', unsafe_allow_html=True)

        hour       = st.slider("🕐 Hour of Day",        0, 23, 14)
        day        = st.slider("📅 Day of Week (0=Mon)", 0, 6,  2)
        month      = st.slider("🗓️ Month",              1, 12,  6)
        is_weekend = 1 if day >= 5 else 0
        quarter    = (month - 1) // 3 + 1

        st.markdown("---")
        st.markdown("**Lag Features** *(recent consumption)*")
        lag_1h   = st.number_input("⚡ 1 hour ago (kWh)",       value=220.0, step=5.0)
        lag_24h  = st.number_input("⚡ 24 hours ago (kWh)",     value=215.0, step=5.0)
        lag_168h = st.number_input("⚡ 1 week ago (kWh)",       value=210.0, step=5.0)

        st.info(f"📌 Detected: {'Weekend' if is_weekend else 'Weekday'}  |  Quarter {quarter}")

        predict_btn = st.button("⚡ Predict Energy Consumption", use_container_width=True)

    with col_out:
        st.markdown('<p class="section-title">🎯 Prediction Result</p>', unsafe_allow_html=True)

        if predict_btn:
            features = np.array([[hour, day, month, is_weekend, quarter,
                                   lag_1h, lag_24h, lag_168h]])
            scaled = scaler.transform(features)
            pred   = model.predict(scaled)[0]

            # Category
            if pred < 150:
                category, cat_color, icon = "Low",    "#00d4aa", "🟢"
            elif pred < 280:
                category, cat_color, icon = "Medium", "#f0a500", "🟡"
            else:
                category, cat_color, icon = "High",   "#ff6b6b", "🔴"

            st.markdown(f"""
            <div class="pred-box">
                <div class="pred-value">{pred:.1f}</div>
                <div class="pred-label">kWh predicted</div>
                <div style="margin-top:14px; font-size:1.1rem; color:{cat_color}; font-weight:600;">
                    {icon} {category} Demand
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Gauge bar
            pct = min(pred / 400, 1.0)
            fig_g, ax_g = plt.subplots(figsize=(7, 1.2))
            fig_g.patch.set_facecolor("#161b22")
            ax_g.set_facecolor("#161b22")
            ax_g.barh(0, 1,   height=0.5, color="#30363d")
            ax_g.barh(0, pct, height=0.5, color=cat_color, alpha=0.9)
            ax_g.set_xlim(0, 1); ax_g.axis("off")
            ax_g.text(0.01, 0, "0", color="#8b949e", va="center", fontsize=8)
            ax_g.text(0.99, 0, "400 kWh", color="#8b949e", va="center",
                      ha="right", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig_g)
            plt.close()

            # Context breakdown
            st.markdown("**Input summary:**")
            summary = pd.DataFrame({
                "Parameter": ["Hour","Day","Month","Weekend","Quarter","Lag 1h","Lag 24h","Lag 168h"],
                "Value":     [hour, day, month, is_weekend, quarter, lag_1h, lag_24h, lag_168h],
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)

        else:
            st.markdown("""
            <div style='background:#161b22; border:1px dashed #30363d; border-radius:16px;
                        padding:60px 20px; text-align:center; color:#8b949e;'>
                <div style='font-size:2.5rem;'>⚡</div>
                <div style='margin-top:10px;font-size:0.95rem;'>
                    Set parameters on the left<br>and click <b>Predict</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Batch prediction: 24-hour forecast
    st.divider()
    st.markdown('<p class="section-title">📆 24-Hour Forecast Preview</p>', unsafe_allow_html=True)
    st.markdown("Simulates predictions for every hour of a typical day with your current lag settings.")

    hours_24 = np.arange(24)
    rows = []
    for h in hours_24:
        is_wk = 1 if day >= 5 else 0
        q     = (month - 1) // 3 + 1
        f     = np.array([[h, day, month, is_wk, q, lag_1h, lag_24h, lag_168h]])
        p     = model.predict(scaler.transform(f))[0]
        rows.append(p)

    fig24, ax24 = plt.subplots(figsize=(14, 3.5))
    fig24.patch.set_facecolor("#161b22")
    ax24.set_facecolor("#0d1117")
    colors_24 = ["#ff6b6b" if v > 280 else "#f0a500" if v > 150 else "#00d4aa" for v in rows]
    ax24.bar(hours_24, rows, color=colors_24, alpha=0.85, edgecolor="#161b22", width=0.8)
    ax24.axhline(np.mean(rows), color="#8b949e", linestyle="--", linewidth=1, label=f"Mean: {np.mean(rows):.1f} kWh")
    ax24.set_xlabel("Hour of Day", color="#8b949e")
    ax24.set_ylabel("Predicted kWh", color="#8b949e")
    ax24.tick_params(colors="#8b949e")
    ax24.spines[["top","right","left","bottom"]].set_color("#30363d")
    ax24.set_xticks(range(24))
    ax24.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#8b949e")
    plt.tight_layout()
    st.pyplot(fig24)
    plt.close()


# ─────────────────────────────────────────────
#  PAGE 3 — MODEL PERFORMANCE
# ─────────────────────────────────────────────
elif page == "📈 Model Performance":
    st.markdown("## 📈 Model Performance")
    st.markdown("Evaluate how accurately the MLP model forecasts energy consumption on held-out test data.")

    # Read saved metrics
    metrics_path = "outputs/metrics.txt"
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            lines = f.readlines()
        mae_val  = float([l for l in lines if "MAE" in l][0].split(":")[1].strip().split()[0])
        rmse_val = float([l for l in lines if "RMSE" in l][0].split(":")[1].strip().split()[0])
        r2_val   = float([l for l in lines if "R²" in l][0].split(":")[1].strip())
    else:
        mae_val, rmse_val, r2_val = 0, 0, 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE",       f"{mae_val:.2f} kWh",  help="Mean Absolute Error")
    col2.metric("RMSE",      f"{rmse_val:.2f} kWh", help="Root Mean Squared Error")
    col3.metric("R² Score",  f"{r2_val:.4f}",       help="Coefficient of Determination")
    col4.metric("Model",     "MLP Regressor",        help="Multi-Layer Perceptron")

    st.divider()

    # Actual vs Predicted
    st.markdown('<p class="section-title">📉 Actual vs Predicted</p>', unsafe_allow_html=True)
    img_path = "outputs/actual_vs_predicted.png"
    if os.path.exists(img_path):
        st.image(img_path, use_container_width=True)
    else:
        st.warning("Run `python main.py` to generate performance charts.")

    st.divider()

    col_err, col_scatter = st.columns(2)

    with col_err:
        st.markdown('<p class="section-title">📊 Error Distribution</p>', unsafe_allow_html=True)
        err_path = "outputs/error_distribution.png"
        if os.path.exists(err_path):
            st.image(err_path, use_container_width=True)

    with col_scatter:
        st.markdown('<p class="section-title">🎯 Scatter: Actual vs Predicted</p>', unsafe_allow_html=True)
        # Re-run predictions on test slice for scatter
        X = df[FEATURES]
        y = df["Energy"]
        split = int(len(X) * 0.8)
        X_test = scaler.transform(X.iloc[split:])
        y_test = y.iloc[split:]
        preds  = model.predict(X_test)

        fig_s, ax_s = plt.subplots(figsize=(6, 5))
        fig_s.patch.set_facecolor("#161b22")
        ax_s.set_facecolor("#0d1117")
        ax_s.scatter(y_test.values, preds, alpha=0.2, s=6, color="#f0a500")
        mn = min(y_test.min(), preds.min())
        mx = max(y_test.max(), preds.max())
        ax_s.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect fit")
        ax_s.set_xlabel("Actual kWh", color="#8b949e")
        ax_s.set_ylabel("Predicted kWh", color="#8b949e")
        ax_s.tick_params(colors="#8b949e")
        ax_s.spines[["top","right","left","bottom"]].set_color("#30363d")
        ax_s.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#8b949e")
        plt.tight_layout()
        st.pyplot(fig_s)
        plt.close()

    # Metric explanation
    st.divider()
    st.markdown('<p class="section-title">📖 Metric Explanations</p>', unsafe_allow_html=True)
    exp1, exp2, exp3 = st.columns(3)
    with exp1:
        st.info("**MAE** (Mean Absolute Error)\n\nAverage absolute difference between actual and predicted values. Lower = better.")
    with exp2:
        st.info("**RMSE** (Root Mean Squared Error)\n\nPenalizes large errors more than MAE. Good for spotting outlier predictions.")
    with exp3:
        st.success(f"**R² Score = {r2_val:.4f}**\n\nModel explains **{r2_val*100:.1f}%** of variance in energy data. Anything above 0.90 is excellent.")


# ─────────────────────────────────────────────
#  PAGE 4 — ENERGY INSIGHTS
# ─────────────────────────────────────────────
elif page == "💡 Energy Insights":
    st.markdown("## 💡 Energy Usage Insights")
    st.markdown("Discover patterns in energy consumption across hours, days, and months.")

    st.markdown('<p class="section-title">🔥 Consumption Heatmap — Hour × Day of Week</p>', unsafe_allow_html=True)

    pivot = df.pivot_table(values="Energy", index="hour", columns="day", aggfunc="mean")
    pivot.columns = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

    fig_h, ax_h = plt.subplots(figsize=(12, 7))
    fig_h.patch.set_facecolor("#161b22")
    ax_h.set_facecolor("#0d1117")
    sns.heatmap(
        pivot, ax=ax_h,
        cmap="YlOrRd",
        linewidths=0.3, linecolor="#0d1117",
        annot=True, fmt=".0f",
        annot_kws={"size": 7, "color": "#0d1117"},
        cbar_kws={"label": "Avg kWh"},
    )
    ax_h.set_xlabel("Day of Week", color="#8b949e")
    ax_h.set_ylabel("Hour of Day", color="#8b949e")
    ax_h.tick_params(colors="#8b949e")
    ax_h.set_title("Average Energy Consumption by Hour & Day", color="#e6edf3", fontsize=12, pad=12)
    plt.tight_layout()
    st.pyplot(fig_h)
    plt.close()

    st.divider()

    col_wk, col_mo = st.columns(2)

    with col_wk:
        st.markdown('<p class="section-title">📅 Weekday vs Weekend</p>', unsafe_allow_html=True)
        wk_avg = df.groupby("is_weekend")["Energy"].mean()
        labels = ["Weekday", "Weekend"]
        colors = ["#f0a500", "#00d4aa"]
        fig_wk, ax_wk = plt.subplots(figsize=(6, 4))
        fig_wk.patch.set_facecolor("#161b22")
        ax_wk.set_facecolor("#0d1117")
        bars = ax_wk.bar(labels, wk_avg.values, color=colors, width=0.45, edgecolor="#30363d")
        for bar, val in zip(bars, wk_avg.values):
            ax_wk.text(bar.get_x() + bar.get_width()/2, val + 1,
                       f"{val:.1f}", ha="center", va="bottom", color="#e6edf3", fontsize=10)
        ax_wk.set_ylabel("Avg Energy (kWh)", color="#8b949e")
        ax_wk.tick_params(colors="#8b949e")
        ax_wk.spines[["top","right","left","bottom"]].set_color("#30363d")
        plt.tight_layout()
        st.pyplot(fig_wk)
        plt.close()

    with col_mo:
        st.markdown('<p class="section-title">📈 Monthly Trend with Std Dev</p>', unsafe_allow_html=True)
        mo_mean = df.groupby("month")["Energy"].mean()
        mo_std  = df.groupby("month")["Energy"].std()
        mn_labels = ["J","F","M","A","M","J","J","A","S","O","N","D"]
        fig_mo, ax_mo = plt.subplots(figsize=(6, 4))
        fig_mo.patch.set_facecolor("#161b22")
        ax_mo.set_facecolor("#0d1117")
        ax_mo.fill_between(range(1,13), mo_mean-mo_std, mo_mean+mo_std, alpha=0.2, color="#f0a500")
        ax_mo.plot(range(1,13), mo_mean.values, color="#f0a500", marker="o", markersize=5, linewidth=2)
        ax_mo.set_xticks(range(1,13)); ax_mo.set_xticklabels(mn_labels)
        ax_mo.set_ylabel("Avg Energy (kWh)", color="#8b949e")
        ax_mo.tick_params(colors="#8b949e")
        ax_mo.spines[["top","right","left","bottom"]].set_color("#30363d")
        plt.tight_layout()
        st.pyplot(fig_mo)
        plt.close()

    st.divider()

    # Peak hour finder
    st.markdown('<p class="section-title">⚡ Peak & Off-Peak Hours</p>', unsafe_allow_html=True)
    hourly_avg = df.groupby("hour")["Energy"].mean().sort_values(ascending=False)
    peak_hours    = hourly_avg.head(3).index.tolist()
    offpeak_hours = hourly_avg.tail(3).index.tolist()

    c1, c2 = st.columns(2)
    with c1:
        st.success(f"**🔴 Top 3 Peak Hours:** {', '.join([f'{h}:00' for h in peak_hours])}\n\n"
                   f"Average: {hourly_avg.head(3).mean():.1f} kWh")
    with c2:
        st.info(f"**🟢 Top 3 Off-Peak Hours:** {', '.join([f'{h}:00' for h in offpeak_hours])}\n\n"
                f"Average: {hourly_avg.tail(3).mean():.1f} kWh")

    # Hourly box plot
    st.markdown('<p class="section-title">📦 Hourly Distribution Boxplot</p>', unsafe_allow_html=True)
    fig_box, ax_box = plt.subplots(figsize=(14, 4))
    fig_box.patch.set_facecolor("#161b22")
    ax_box.set_facecolor("#0d1117")
    hourly_data = [df[df["hour"] == h]["Energy"].values for h in range(24)]
    bp = ax_box.boxplot(hourly_data, patch_artist=True, medianprops=dict(color="#ff6b6b", linewidth=2))
    for patch in bp["boxes"]:
        patch.set_facecolor("#f0a500")
        patch.set_alpha(0.4)
    for whisker in bp["whiskers"]:
        whisker.set_color("#30363d")
    for flier in bp["fliers"]:
        flier.set(marker=".", color="#8b949e", alpha=0.3, markersize=3)
    ax_box.set_xlabel("Hour of Day", color="#8b949e")
    ax_box.set_ylabel("Energy (kWh)", color="#8b949e")
    ax_box.tick_params(colors="#8b949e")
    ax_box.spines[["top","right","left","bottom"]].set_color("#30363d")
    plt.tight_layout()
    st.pyplot(fig_box)
    plt.close()