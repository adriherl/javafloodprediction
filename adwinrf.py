import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestClassifier
from river import drift, metrics
import warnings

warnings.filterwarnings('ignore')

# --- 1. DATA PREPARATION ---
# Memastikan data urut berdasarkan Tanggal untuk simulasi streaming yang jujur
df = pd.read_csv('master.csv')
df['Tanggal'] = pd.to_datetime(df['Tanggal'])
df = df.sort_values('Tanggal').reset_index(drop=True)

features = [
    'Month', 'twi_final', 'elevation_mean', 'slope_mean', 'soil_type', 
    'drainage_pct', 'precip_t0', 'precip_t1', 'precip_t2', 'precip_3d', 
    'precip_7d', 'ndvi', 'mndwi', 'ndbi',
    'nb_precip_t0', 'nb_precip_t1', 'nb_precip_t2', 'nb_twi', 'nb_elevation', 
    'nb_slope', 'nb_drainage', 'nb_soil', 'nb_ndvi', 'nb_mndwi', 'nb_ndbi',
    '1daylag', '3daylag', '7daylag', 'ibi', 'nb_ibi'
]
target_col = 'Target'

# Cleaning data
for col in features:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=features + [target_col]).reset_index(drop=True)

# --- 2. BATCH TRAINING (10% DATA AWAL) ---
split_idx = int(len(df) * 0.10)
print(f"📦 Training Static Model pada 10% data awal ({split_idx} baris)...")

rf_static = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
rf_static.fit(df[features].iloc[:split_idx], df[target_col].iloc[:split_idx])

# --- 3. STREAMING MONITORING DENGAN ADWIN ---
adwin = drift.ADWIN(delta=0.002) # Watchman untuk mendeteksi drift
acc_metric = metrics.Accuracy()
acc_history = []
dates_history = []
drift_points = [] # Menyimpan tanggal terjadinya drift

print("🌊 Memulai evaluasi streaming pada 90% data sisa...")

for i in range(split_idx, len(df)):
    xi = df[features].iloc[i:i+1]
    yi = int(df[target_col].iloc[i])
    
    # Predict (Model Statis tidak belajar lagi)
    y_pred = rf_static.predict(xi)[0]
    
    # Hitung Error (0 jika benar, 1 jika salah)
    error = 0 if y_pred == yi else 1
    
    # Update ADWIN dengan aliran error
    adwin.update(error)
    
    # Cek apakah terjadi drift (perubahan statistik pada error rate)
    if adwin.drift_detected:
        drift_date = df['Tanggal'].iloc[i]
        drift_points.append(drift_date)
        print(f"⚠️ Concept Drift terdeteksi pada: {drift_date.date()} (Akurasi mulai tidak stabil)")
    
    # Update metrik akurasi kumulatif
    acc_metric.update(yi, y_pred)
    acc_history.append(acc_metric.get())
    dates_history.append(df['Tanggal'].iloc[i])

# --- 4. VISUALISASI HASIL ---
plt.figure(figsize=(15, 8), dpi=100)
date_fmt = mdates.DateFormatter('%Y-%m')

# Plot Akurasi Kumulatif
plt.plot(dates_history, acc_history, label='Batch Learning (RF Classifier) Cumulative Accuracy', color='#d62728', lw=2)

# Tambahkan garis vertikal untuk setiap Drift yang dideteksi ADWIN
for d_date in drift_points:
    plt.axvline(x=d_date, color='blue', linestyle='--', alpha=0.4, 
                label='Drift' if d_date == drift_points[0] else "")

# Garis pemisah antara fase training (10%) dan testing
plt.axvline(x=df['Tanggal'].iloc[split_idx], color='black', linestyle=':', lw=2, label='End of Training (10%)')

# Annotasi Nilai Terakhir
plt.text(dates_history[-1], acc_history[-1], f' Final Acc: {acc_history[-1]:.4f}', 
         color='#d62728', fontweight='bold', va='center')

plt.title("Detected Drifts in Batch Learning (RF Classifier)", fontsize=14, fontweight='bold')
plt.xlabel("Date", fontsize=12)
plt.ylabel("Cumulative Accuracy Score", fontsize=12)
plt.gca().xaxis.set_major_formatter(date_fmt)
plt.xticks(rotation=45)

plt.legend(loc='lower left')
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()

print(f"\n✅ Analisis Selesai.")
print(f"Total Drift terdeteksi: {len(drift_points)}")