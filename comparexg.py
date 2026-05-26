import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from river import linear_model, preprocessing, metrics, compose, optim
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
split_idx = 1000  # initial batch training window

# ---- Tuned parameters ----
batch_lr_params = {
    'C': 0.003090614746573318,
    'penalty': 'l2',
    'solver': 'lbfgs',
    'max_iter': 1000
}

incremental_lr_params = {
    'lr': 0.007612430320839992,
    'l2': 0.0008365205576328523
}

# ---- Batch Logistic Regression (Static) ----
scaler_batch = StandardScaler()
X_init = scaler_batch.fit_transform(df[features].iloc[:split_idx])

lr_static = LogisticRegression(**batch_lr_params)
lr_static.fit(
    X_init,
    df[target_col].iloc[:split_idx]
)

# ---- Incremental Logistic Regression ----
inc_lr_model = compose.Pipeline(
    preprocessing.StandardScaler(),
    linear_model.LogisticRegression(
        optimizer=optim.SGD(lr=incremental_lr_params['lr']),
        l2=incremental_lr_params['l2']
    )
)

# ---- Metrics ----
static_acc_metric = metrics.Accuracy()
adaptive_acc_metric = metrics.Accuracy()

lr_acc_history = []
inc_lr_acc_history = []
dates_stream = []

# =====================================================
# 3. STREAMING ANALYSIS
# =====================================================
print("🚀 Analyzing Data Stream (Batch LR vs Incremental LR)...")

for i in range(len(df)):
    xi_raw = df[features].iloc[i:i+1]
    xi_scaled = scaler_batch.transform(xi_raw)
    xi_dict = df[features].iloc[i].to_dict()
    yi = int(df[target_col].iloc[i])

    if i >= split_idx:
        # ----- Static Logistic Regression Prediction -----
        y_pred_lr = lr_static.predict(xi_scaled)[0]
        static_acc_metric.update(yi, y_pred_lr)
        lr_acc_history.append(static_acc_metric.get())

        # ----- Incremental Logistic Regression Prediction -----
        y_pred_inc = inc_lr_model.predict_one(xi_dict)
        adaptive_acc_metric.update(yi, y_pred_inc)
        inc_lr_acc_history.append(adaptive_acc_metric.get())

        dates_stream.append(df['Tanggal'].iloc[i])

    # ----- Continuous Training for Incremental LR -----
    inc_lr_model.learn_one(xi_dict, yi)

# =====================================================
# 4. VISUALIZATION
# =====================================================
plt.figure(figsize=(14, 7), dpi=100)

date_fmt = mdates.DateFormatter('%Y-%m')

plt.plot(
    dates_stream, inc_lr_acc_history,
    label='Incremental Learning',
    color='#1f77b4', lw=2.5
)

plt.plot(
    dates_stream, lr_acc_history,
    label='Batch Learning',
    color='#d62728', ls='--', alpha=0.7
)

# ---- Annotations for last values ----
plt.text(
    dates_stream[-1], inc_lr_acc_history[-1],
    f'  {inc_lr_acc_history[-1]:.4f}',
    color='#1f77b4', fontweight='bold', va='center'
)

plt.text(
    dates_stream[-1], lr_acc_history[-1],
    f'  {lr_acc_history[-1]:.4f}',
    color='#d62728', fontweight='bold', va='center'
)

plt.title(
    "Cumulative Accuracy Comparison of Batch vs Incremental Learning (Logistic Regression)",
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
print(f"Final Incremental Logistic Regression Accuracy: {inc_lr_acc_history[-1]:.4f}")
print(f"Final Batch Logistic Regression Accuracy: {lr_acc_history[-1]:.4f}")
