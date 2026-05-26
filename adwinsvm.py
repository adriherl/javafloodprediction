import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.svm import SVC
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
print(f"📦 Training Static SVM Model pada 10% data awal ({split_idx} baris)...")

# --- Scaling is mandatory for SVM ---
scaler = StandardScaler()
X_train_init = scaler.fit_transform(df[features].iloc[:split_idx])
y_train_init = df[target_col].iloc[:split_idx]

# Default batch SVM (RBF)
svm_static = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_static.fit(X_train_init, y_train_init)

# =====================================================
# 3. STREAMING MONITORING DENGAN ADWIN
# =====================================================
adwin = drift.ADWIN(delta=0.002)  # Drift detector
acc_metric = metrics.Accuracy()

acc_history = []
dates_history = []
drift_points = []

print("🌊 Memulai evaluasi streaming pada 90% data sisa...")

for i in range(split_idx, len(df)):
    xi_raw = df[features].iloc[i:i+1]
    xi_scaled = scaler.transform(xi_raw)
    yi = int(df[target_col].iloc[i])

    # ---- Static SVM prediction ----
    y_pred = svm_static.predict(xi_scaled)[0]

    # Error stream for ADWIN
    error = 0 if y_pred == yi else 1
    adwin.update(error)

    if adwin.drift_detected:
        drift_date = df['Tanggal'].iloc[i]
        drift_points.append(drift_date)
        print(f"⚠️ Concept Drift terdeteksi pada: {drift_date.date()} (Akurasi mulai tidak stabil)")

    # Update cumulative accuracy
    acc_metric.update(yi, y_pred)
    acc_history.append(acc_metric.get())
    dates_history.append(df['Tanggal'].iloc[i])

# =====================================================
# 4. VISUALISASI HASIL
# =====================================================
plt.figure(figsize=(15, 8), dpi=100)
date_fmt = mdates.DateFormatter('%Y-%m')

# Plot cumulative accuracy
plt.plot(
    dates_history,
    acc_history,
    label='Batch Learning (SVM RBF) Cumulative Accuracy',
    color='#d62728',
    lw=2
)

# Vertical lines for detected drifts
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

# Annotation of final accuracy
plt.text(
    dates_history[-1],
    acc_history[-1],
    f' Final Acc: {acc_history[-1]:.4f}',
    color='#d62728',
    fontweight='bold',
    va='center'
)

plt.title(
    "Detected Drifts in Batch Learning (SVM RBF)",
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
