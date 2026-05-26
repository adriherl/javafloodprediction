# =====================================================
# 0. IMPORTS
# =====================================================
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")

# =====================================================
# 1. LOAD & PREPARE DATA (STATIC LULC)
# =====================================================
df = pd.read_csv("masterstatic.csv")
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
# 3. TRAIN SVM (BATCH, STATIC LULC)
# =====================================================
svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf",
        probability=True,
        C=1.0,
        gamma="scale",
        random_state=42
    ))
])

print("🚀 Training SVM (Static LULC)...")
svm_model.fit(X_train, y_train)

print(f"✅ Test Accuracy: {accuracy_score(y_test, svm_model.predict(X_test)):.4f}")

# =====================================================
# 4. DEFINE SHAP SAMPLE (VERY IMPORTANT)
# =====================================================
N_SHAP = 200           # DO NOT increase (Kernel SHAP is expensive)
N_BACKGROUND = 50

X_shap = X_test.iloc[:N_SHAP]
X_background = X_train.sample(
    n=N_BACKGROUND,
    random_state=42
)

print(f"🔎 SHAP sample shape: {X_shap.shape}")
print(f"🔎 Background shape: {X_background.shape}")

# =====================================================
# 5. SHAP KERNEL EXPLAINER WITH PROGRESS BAR
# =====================================================
def svm_predict_proba(X_np):
    """
    Returns probability of positive class (flood = 1)
    """
    return svm_model.predict_proba(X_np)[:, 1]

explainer = shap.KernelExplainer(
    model=svm_predict_proba,
    data=X_background.values
)

print("⏳ Computing SHAP values (KernelExplainer)...")

shap_values = explainer.shap_values(
    X_shap.values,
    nsamples=200,
    silent=False   # <-- shows progress
)

# =====================================================
# 6. BUILD SHAP EXPLANATION OBJECT
# =====================================================
explanation = shap.Explanation(
    values=shap_values,
    base_values=np.repeat(explainer.expected_value, len(X_shap)),
    data=X_shap.values,
    feature_names=features
)

# =====================================================
# 7. SHAP BEESWARM PLOT
# =====================================================
plt.figure(figsize=(12, 8))
shap.plots.beeswarm(
    explanation,
    max_display=20
)
plt.title(
    "SHAP Beeswarm Plot\nSVM – Static LULC",
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
    explanation,
    max_display=15
)
plt.title(
    "Mean |SHAP| Values\nSVM – Static LULC",
    fontsize=14,
    fontweight="bold"
)
plt.tight_layout()
plt.show()
