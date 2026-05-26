# =====================================================
# 0. IMPORTS
# =====================================================
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")

# =====================================================
# 1. LOAD & PREPARE DATA (DYNAMIC LULC)
# =====================================================
df = pd.read_csv("master.csv")
df["Tanggal"] = pd.to_datetime(df["Tanggal"])
df = df.sort_values("Tanggal").reset_index(drop=True)

features = [
    'Month', 'twi_final', 'elevation_mean', 'slope_mean', 'soil_type',
    'drainage_pct', 'precip_t0', 'precip_t1', 'precip_t2',
    'precip_3d', 'precip_7d',
    'ndvi', 'mndwi', 'ndbi',
    'nb_precip_t0', 'nb_precip_t1', 'nb_precip_t2',
    'nb_twi', 'nb_elevation', 'nb_slope',
    'nb_drainage', 'nb_soil',
    'nb_ndvi', 'nb_mndwi', 'nb_ndbi',
    '1daylag', '3daylag', '7daylag',
    'ibi', 'nb_ibi'
]
target_col = "Target"

df = df.dropna(subset=features + [target_col])

X = df[features]
y = df[target_col]

# =====================================================
# 2. TRAIN–TEST SPLIT (CHRONOLOGICAL)
# =====================================================
split_idx = int(len(df) * 0.8)

X_train = X.iloc[:split_idx]
X_test  = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test  = y.iloc[split_idx:]

# =====================================================
# 3. FAST RANDOM FOREST WITH TRAINING PROGRESS BAR
# =====================================================
N_TREES = 250

rf_model = RandomForestClassifier(
    n_estimators=1,          # start with 1 tree
    max_depth=16,
    min_samples_split=3,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
    warm_start=True          # 🔑 enables incremental tree growth
)

print("🚀 Training Random Forest (with progress bar)...")

for i in tqdm(range(1, N_TREES + 1), desc="Training trees"):
    rf_model.n_estimators = i
    rf_model.fit(X_train, y_train)

print(f"✅ Training finished with {N_TREES} trees")
print(f"✅ Test Accuracy: {accuracy_score(y_test, rf_model.predict(X_test)):.4f}")

# =====================================================
# 4. SHAP COMPUTATION WITH PROGRESS BAR
# =====================================================
N_SHAP = 1000
X_shap = X_test.iloc[:N_SHAP]

explainer = shap.TreeExplainer(rf_model)

shap_chunks = []
chunk_size = 50

print("⏳ Computing SHAP values...")

for start in tqdm(range(0, len(X_shap), chunk_size), desc="SHAP chunks"):
    end = start + chunk_size
    X_chunk = X_shap.iloc[start:end]

    shap_chunk = explainer(X_chunk)
    shap_chunks.append(shap_chunk.values[:, :, 1])  # positive class

shap_values_pos = np.vstack(shap_chunks)

# =====================================================
# 5. BUILD SHAP EXPLANATION
# =====================================================
shap_exp = shap.Explanation(
    values=shap_values_pos,
    base_values=explainer.expected_value[1],
    data=X_shap.values,
    feature_names=features
)

# =====================================================
# 6. SHAP BEESWARM
# =====================================================
plt.figure(figsize=(12, 8))
shap.plots.beeswarm(
    shap_exp,
    max_display=20
)
plt.title(
    "SHAP Beeswarm Plot\nRandom Forest – Dynamic LULC (Fast)",
    fontsize=14,
    fontweight="bold"
)
plt.tight_layout()
plt.show()

# =====================================================
# 7. SHAP BAR PLOT
# =====================================================
plt.figure(figsize=(8, 6))
shap.plots.bar(
    shap_exp,
    max_display=15
)
plt.title(
    "Mean |SHAP| Values\nRandom Forest – Dynamic LULC (Fast)",
    fontsize=14,
    fontweight="bold"
)
plt.tight_layout()
plt.show()
