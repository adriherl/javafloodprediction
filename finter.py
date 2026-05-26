# =====================================================
# 0. IMPORTS
# =====================================================
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

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
    'precip_t0',
    'precip_3d',
    'precip_7d',
    'ndvi',
    'ndbi',
    'nb_mndwi',
    'nb_soil',
    'slope_mean',
    'twi_final',
    '1daylag'
]
target_col = "Target"

# Ensure numeric
for col in features + [target_col]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna().reset_index(drop=True)

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
# 3. TRAIN RANDOM FOREST (BATCH)
# =====================================================
rf_model = RandomForestClassifier(
    n_estimators=250,
    max_depth=16,
    min_samples_split=3,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

print("🚀 Training Random Forest (Dynamic LULC)...")
rf_model.fit(X_train, y_train)

print(f"✅ Test Accuracy: {accuracy_score(y_test, rf_model.predict(X_test)):.4f}")

# =====================================================
# 4. SHAP EXPLAINER (NEW API)
# =====================================================
explainer = shap.TreeExplainer(rf_model)

N_SHAP = 800
X_shap = X_test.iloc[:N_SHAP]

print("⏳ Computing SHAP explanations...")
shap_exp = explainer(X_shap)  # <- THIS is the key fix

# Use positive class for binary classification
shap_exp_pos = shap_exp[:, :, 1]

# =====================================================
# 5. FEATURE INTERACTION EXPLANATION (5 PAIRS)
# =====================================================
interaction_pairs = [
    ("precip_3d", "ndvi"),
    ("precip_t0", "nb_soil"),
    ("nb_mndwi", "1daylag"),
    ("precip_7d", "precip_t0"),
    ("ndbi", "slope_mean")
]

for main_feat, interaction_feat in interaction_pairs:
    print(f"🔍 SHAP interaction: {main_feat} × {interaction_feat}")

    plt.figure(figsize=(7, 5))
    shap.plots.scatter(
        shap_exp_pos[:, main_feat],
        color=shap_exp_pos[:, interaction_feat],
        show=False
    )
    plt.title(
        f"SHAP Interaction: {main_feat} × {interaction_feat}",
        fontsize=12,
        fontweight="bold"
    )
    plt.tight_layout()
    plt.show()
