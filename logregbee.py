# =====================================================
# 0. IMPORTS
# =====================================================
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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

for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=features + [target_col]).reset_index(drop=True)

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
# 3. TRAIN LOGISTIC REGRESSION (BATCH)
# =====================================================
logreg_model = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs"
    ))
])

print("🚀 Training Logistic Regression (Dynamic LULC)...")
logreg_model.fit(X_train, y_train)

print(f"✅ Test Accuracy: {accuracy_score(y_test, logreg_model.predict(X_test)):.4f}")

# =====================================================
# 4. DEFINE SHAP SAMPLE
# =====================================================
N_SHAP = 1000
X_shap = X_test.iloc[:N_SHAP]

# =====================================================
# 5. PREPARE DATA FOR SHAP (SCALE FIRST)
# =====================================================
scaler = logreg_model.named_steps["scaler"]
lr = logreg_model.named_steps["lr"]

X_train_scaled = scaler.transform(X_train)
X_shap_scaled  = scaler.transform(X_shap)

# =====================================================
# 6. SHAP LINEAR EXPLAINER (FIXED)
# =====================================================
masker = shap.maskers.Independent(X_train_scaled)

explainer = shap.LinearExplainer(
    model=lr,
    masker=masker,
    feature_names=features
)

shap_values = explainer(X_shap_scaled)

# =====================================================
# 7. SHAP BEESWARM PLOT
# =====================================================
plt.figure(figsize=(12, 8))
shap.plots.beeswarm(
    shap_values,
    max_display=20
)
plt.title(
    "SHAP Beeswarm Plot\nLogistic Regression – Dynamic LULC",
    fontsize=14,
    fontweight="bold"
)
plt.tight_layout()
plt.show()

# =====================================================
# 8. OPTIONAL: SHAP BAR PLOT
# =====================================================
plt.figure(figsize=(8, 6))
shap.plots.bar(
    shap_values,
    max_display=15
)
plt.title(
    "Mean |SHAP| Values\nLogistic Regression – Dynamic LULC",
    fontsize=14,
    fontweight="bold"
)
plt.tight_layout()
plt.show()
