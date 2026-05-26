import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from river import forest, metrics
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. PARAMETERS & SETUP
# ==========================================
DRIFT_INDEX = 4799
LEARNING_RATE = 0.05
features = ['Month', 'twi_final', 'elevation_mean', 'slope_mean', 'soil_type', 
            'drainage_pct', 'precip_t0', 'precip_t1', 'precip_t2', 'precip_3d', 
            'precip_7d', 'ndvi', 'mndwi', 'ndbi', 'nb_precip_t0', 'nb_precip_t1', 
            'nb_precip_t2', 'nb_twi', 'nb_elevation', 'nb_slope', 'nb_drainage', 
            'nb_soil', 'nb_ndvi', 'nb_mndwi', 'nb_ndbi', '1daylag', '3daylag', 
            '7daylag', 'ibi', 'nb_ibi']
target_col = 'Target'

# 2. LOAD DATA
df = pd.read_csv('master.csv').sort_values('Tanggal')
for col in features: df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=features + [target_col])

# 3. INITIALIZE MODELS
rf_batch = RandomForestClassifier(n_estimators=567, max_depth=15, random_state=42)
rf_batch.fit(df[features].iloc[:DRIFT_INDEX], df[target_col].iloc[:DRIFT_INDEX])

arf_inc = forest.ARFClassifier(n_models=10, max_depth=10, seed=42)

# 4. INITIALIZE METRIC TRACKERS (F1 as the primary driver)
report = {
    'acc': metrics.Accuracy(),
    'prec': metrics.Precision(),
    'rec': metrics.Recall(),
    'f1': metrics.F1()
}

history = {k: [] for k in report.keys()}
history['w_inc'] = []
w_batch, w_inc = 0.5, 0.5

# ==========================================
# 5. HYBRID STREAMING EXECUTION
# ==========================================
print("🌊 Analyzing Hybrid Performance (Metrics: Acc, Prec, Rec, F1)...")

for i in range(DRIFT_INDEX, len(df)):
    xi_raw = df[features].iloc[i:i+1]
    xi_dict = df[features].iloc[i].to_dict()
    yi = int(df[target_col].iloc[i])
    
    p_batch = rf_batch.predict_proba(xi_raw)[0][1]
    p_inc = arf_inc.predict_proba_one(xi_dict).get(1, 0.0)
    
    y_prob = (p_batch * w_batch) + (p_inc * w_inc)
    y_pred = 1 if y_prob > 0.5 else 0
    
    # Weight Update
    err_batch, err_inc = abs(yi - p_batch), abs(yi - p_inc)
    if err_batch < err_inc:
        w_batch += LEARNING_RATE * (1 - w_batch)
        w_inc -= LEARNING_RATE * w_inc
    else:
        w_inc += LEARNING_RATE * (1 - w_inc)
        w_batch -= LEARNING_RATE * w_batch
    
    total = w_batch + w_inc
    w_batch, w_inc = w_batch/total, w_inc/total
    
    arf_inc.learn_one(xi_dict, yi)
    for m in report.values():
        m.update(yi, y_pred)
    
    for k in report.keys():
        history[k].append(report[k].get())
    history['w_inc'].append(w_inc)

# ==========================================
# 6. VISUALIZATION
# ==========================================
plt.figure(figsize=(15, 10))

plt.subplot(2, 1, 1)
plt.plot(history['acc'], label='Accuracy', color='gray', alpha=0.5)
plt.plot(history['f1'], label='F1-Score', color='blue', lw=2)
plt.title(f"Hybrid Performance Post-Drift (ADWIN Index: {DRIFT_INDEX})")
plt.ylabel("Score")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(history['prec'], label='Precision', color='green')
plt.plot(history['rec'], label='Recall', color='orange')
plt.plot(history['w_inc'], label='Modern Learner Weight', color='blue', ls='--', alpha=0.3)
plt.xlabel("Samples Post-Drift")
plt.ylabel("Value")
plt.legend()

plt.tight_layout()
plt.show()

print("\nFINAL HYBRID REPORT")
for name, m in report.items():
    print(f"{name.upper():<10}: {m.get():.4f}")