import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.svm import SVC
from river import linear_model, metrics
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

# Ensure numeric
for col in features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=features + [target_col])

# =====================================================
# 2. INITIALIZATION
# =====================================================
split_idx = 1000  # initial training window

# ---- Tuned parameters ----
svm_params = {
    'C': 0.6164741003514174, 
    'gamma': 0.017086793693622697,
    'kernel': 'rbf',
    'probability': True
}

pa_params = {
    'C': 0.014600840528904077,
    'mode': 1
}

# ---- Batch SVM (Static) ----
svm_static = SVC(**svm_params)
svm_static.fit(
    df[features].iloc[:split_idx],
    df[target_col].iloc[:split_idx]
)

# ---- Incremental PA-SVM ----
pa_model = linear_model.PAClassifier(**pa_params)

# ---- Metrics ----
static_acc_metric = metrics.Accuracy()
adaptive_acc_metric = metrics.Accuracy()

svm_acc_history = []
pa_acc_history = []
dates_stream = []

# =====================================================
# 3. STREAMING ANALYSIS
# =====================================================
print("🚀 Analyzing Data Stream (SVM vs PA-SVM)...")

for i in range(len(df)):
    xi_raw = df[features].iloc[i:i+1]
    xi_dict = df[features].iloc[i].to_dict()
    yi = int(df[target_col].iloc[i])

    if i >= split_idx:
        # ----- Static SVM Prediction -----
        y_pred_svm = svm_static.predict(xi_raw)[0]
        static_acc_metric.update(yi, y_pred_svm)
        svm_acc_history.append(static_acc_metric.get())

        # ----- Incremental PA Prediction -----
        y_pred_pa = pa_model.predict_one(xi_dict)
        adaptive_acc_metric.update(yi, y_pred_pa)
        pa_acc_history.append(adaptive_acc_metric.get())

        dates_stream.append(df['Tanggal'].iloc[i])

    # ----- Continuous Training for PA -----
    pa_model.learn_one(xi_dict, yi)

# =====================================================
# 4. VISUALIZATION
# =====================================================
plt.figure(figsize=(14, 7), dpi=100)

date_fmt = mdates.DateFormatter('%Y-%m')

plt.plot(
    dates_stream, pa_acc_history,
    label='Incremental Learning (PA Classifier)',
    color='#1f77b4', lw=2.5
)

plt.plot(
    dates_stream, svm_acc_history,
    label='Batch Learning (SVM RBF)',
    color='#d62728', ls='--', alpha=0.7
)

# ---- Annotations ----
plt.text(
    dates_stream[-1], pa_acc_history[-1],
    f'  {pa_acc_history[-1]:.4f}',
    color='#1f77b4', fontweight='bold', va='center'
)

plt.text(
    dates_stream[-1], svm_acc_history[-1],
    f'  {svm_acc_history[-1]:.4f}',
    color='#d62728', fontweight='bold', va='center'
)

plt.title(
    "Cumulative Accuracy Comparison of Batch vs Incremental Learning (SVM)",
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
# 5. FINAL REPORT
# =====================================================
print(f"Final Data Date: {dates_stream[-1]}")
print(f"Final Incremental PA Classifier: {pa_acc_history[-1]:.4f}")
print(f"Final Static SVM Accuracy: {svm_acc_history[-1]:.4f}")
