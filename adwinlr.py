import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from river import drift, metrics
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
# 2. BATCH TRAINING (10% DATA AWAL)
# =====================================================
split_idx = int(len(df) * 0.10)
print(f"📦 Training Static Linear Regression Model pada 10% data awal ({split_idx} baris)...")

# Scaling is important for linear models
scaler = StandardScaler()
X_train_init = scaler.fit_transform(df[features].iloc[:split_idx])
y_train_init = df[target_col].iloc[:split_idx]

# Batch Linear Regression (STATIC)
linreg_static = LinearRegression()
linreg_static.fit(X_train_init, y_train_init)

# =====================================================
# 3. STREAMING MONITORING DENGAN ADWIN
# =====================================================
adwin = drift.ADWIN(delta=0.002)
acc_metric = metrics.Accuracy()

acc_history = []
dates_history = []
drift_points = []

print("🌊 Memulai evaluasi streaming pada 90% data sisa...")

for i in range(split_idx, len(df)):
    xi_raw = df[features].iloc[i:i+1]
    xi_scaled = scaler.transform(xi_raw)
    yi = int(df[target_col].iloc[i])

    # ---- Linear Regression Prediction ----
    y_score = linreg_static.predict(xi_scaled)[0]

    # Thresholding for classification
    y_pred = 1 if y_score >= 0.5 else 0

    # Error stream for ADWIN
    error = 0 if y_pred == yi else 1
    adwin.update(error)

    if adwin.drift_detected:
        drift_date = df['Tanggal'].iloc[i]
        drift_points.append(drift_date)
        print(f"⚠️ Concept Drift terdeteksi pada: {drift_date.date()} (Model Linear mulai gagal)")

    # Update cumulative accuracy
    acc_metric.update(yi, y_pred)
    acc_history.append(acc_metric.get())
    dates_history.append(df['Tanggal'].iloc[i])

# =====================================================
# 4. VISUALISASI HASIL
# =====================================================
plt.figure(figsize=(15, 8), dpi=100)
date_fmt = mdates.DateFormatter('%Y-%m')

plt.plot(
    dates_history,
    acc_history,
    label='Batch Learning (Linear Regression) Cumulative Accuracy',
    color='#d62728',
    lw=2
)

# Vertical drift markers
for d_date in drift_points:
    plt.axvline(
        x=d_date,
        color='blue',
        linestyle='--',
        alpha=0.4,
        label='Drift' if d_date == drift_points[0] else ""
    )

# Training–testing boundary
plt.axvline(
    x=df['Tanggal'].iloc[split_idx],
    color='black',
    linestyle=':',
    lw=2,
    label='End of Training (10%)'
)

# Final annotation
plt.text(
    dates_history[-1],
    acc_history[-1],
    f' Final Acc: {acc_history[-1]:.4f}',
    color='#d62728',
    fontweight='bold',
    va='center'
)

plt.title(
    "Detected Drifts in Batch Learning (Linear Regression)",
    fontsize=14,
    fontweight='bold'
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
print("\n✅ Analisis Selesai.")
print(f"Total Drift terdeteksi: {len(drift_points)}")
