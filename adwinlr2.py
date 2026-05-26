import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from river import drift, metrics, linear_model, preprocessing, compose, optim
import warnings

warnings.filterwarnings('ignore')

# =====================================================
# 1. DATA PREPARATION
# =====================================================
df = pd.read_csv('master.csv')
df['Tanggal'] = pd.to_datetime(df['Tanggal'])
df = df.sort_values('Tanggal').reset_index(drop=True)

features = [
    'Month', 'twi_final', 'elevation_mean', 'slope_mean', 'soil_type',
    'drainage_pct', 'precip_t0', 'precip_t1', 'precip_t2', 'precip_3d',
    'precip_7d', 'ndvi', 'mndwi', 'ndbi',
    'nb_precip_t0', 'nb_precip_t1', 'nb_precip_t2', 'nb_twi',
    'nb_elevation', 'nb_slope', 'nb_drainage', 'nb_soil',
    'nb_ndvi', 'nb_mndwi', 'nb_ndbi',
    '1daylag', '3daylag', '7daylag', 'ibi', 'nb_ibi'
]
target_col = 'Target'

# Cleaning data
for col in features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=features + [target_col]).reset_index(drop=True)

# =====================================================
# 2. INITIAL TRAINING (10%)
# =====================================================
split_idx = int(len(df) * 0.10)
print(f"📦 Initial training on 10% data ({split_idx} samples)")

# ---- Batch Linear Regression ----
scaler_batch = StandardScaler()
X_train_init = scaler_batch.fit_transform(df[features].iloc[:split_idx])
y_train_init = df[target_col].iloc[:split_idx]

linreg_static = LinearRegression()
linreg_static.fit(X_train_init, y_train_init)

# ---- Incremental Linear Regression (River) ----
inc_linreg = compose.Pipeline(
    preprocessing.StandardScaler(),
    linear_model.LinearRegression(
        optimizer=optim.SGD(lr=0.01)
    )
)

# Warm-up incremental model on same 10%
for xi, yi in zip(
    df[features].iloc[:split_idx].to_dict(orient='records'),
    df[target_col].iloc[:split_idx]
):
    inc_linreg.learn_one(xi, yi)

# =====================================================
# 3. STREAMING + DRIFT MONITORING
# =====================================================
adwin = drift.ADWIN(delta=0.002)

batch_acc = metrics.Accuracy()
inc_acc = metrics.Accuracy()

batch_acc_history = []
inc_acc_history = []
dates_history = []
drift_points = []

print("🌊 Streaming evaluation on remaining 90% data...")

for i in range(split_idx, len(df)):
    xi_raw = df[features].iloc[i:i+1]
    xi_scaled = scaler_batch.transform(xi_raw)
    xi_dict = df[features].iloc[i].to_dict()
    yi = int(df[target_col].iloc[i])

    # ---- Batch Linear Regression (STATIC) ----
    y_score_batch = linreg_static.predict(xi_scaled)[0]
    y_pred_batch = 1 if y_score_batch >= 0.5 else 0

    # ---- Incremental Linear Regression ----
    y_score_inc = inc_linreg.predict_one(xi_dict)
    y_pred_inc = 1 if y_score_inc is not None and y_score_inc >= 0.5 else 0

    # ---- ADWIN monitors batch error ----
    error = 0 if y_pred_batch == yi else 1
    adwin.update(error)

    if adwin.drift_detected:
        drift_date = df['Tanggal'].iloc[i]
        drift_points.append(drift_date)
        print(f"⚠️ Drift detected on: {drift_date.date()}")

    # ---- Update metrics ----
    batch_acc.update(yi, y_pred_batch)
    inc_acc.update(yi, y_pred_inc)

    batch_acc_history.append(batch_acc.get())
    inc_acc_history.append(inc_acc.get())
    dates_history.append(df['Tanggal'].iloc[i])

    # ---- Incremental learning step ----
    inc_linreg.learn_one(xi_dict, yi)

# =====================================================
# 4. VISUALIZATION
# =====================================================
plt.figure(figsize=(15, 8), dpi=100)
date_fmt = mdates.DateFormatter('%Y-%m')

# Batch LR curve
plt.plot(
    dates_history,
    batch_acc_history,
    label='Batch Linear Regression (Static)',
    color='#d62728',
    lw=2
)

# Incremental LR curve
plt.plot(
    dates_history,
    inc_acc_history,
    label='Incremental Linear Regression',
    color='#1f77b4',
    lw=2.5
)

# Drift markers
for d in drift_points:
    plt.axvline(
        x=d,
        color='black',
        linestyle='--',
        alpha=0.4,
        label='Detected Drift' if d == drift_points[0] else ""
    )

# Training boundary
plt.axvline(
    x=df['Tanggal'].iloc[split_idx],
    color='gray',
    linestyle=':',
    lw=2,
    label='End of Training (10%)'
)

# Final annotations
plt.text(
    dates_history[-1], batch_acc_history[-1],
    f'  {batch_acc_history[-1]:.4f}',
    color='#d62728', fontweight='bold', va='center'
)

plt.text(
    dates_history[-1], inc_acc_history[-1],
    f'  {inc_acc_history[-1]:.4f}',
    color='#1f77b4', fontweight='bold', va='center'
)

plt.title(
    "Batch vs Incremental Linear Regression under Concept Drift",
    fontsize=14, fontweight='bold'
)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Cumulative Accuracy Score", fontsize=12)
plt.gca().xaxis.set_major_formatter(date_fmt)
plt.xticks(rotation=45)

plt.legend(loc='lower left')
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()

# =====================================================
# 5. SUMMARY
# =====================================================
print("\n✅ Analysis completed.")
print(f"Total drift events detected (batch): {len(drift_points)}")
print(f"Final Batch Accuracy: {batch_acc_history[-1]:.4f}")
print(f"Final Incremental Accuracy: {inc_acc_history[-1]:.4f}")
