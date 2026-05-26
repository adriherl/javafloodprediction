# =====================================================
# 0. IMPORTS
# =====================================================
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings("ignore")

# =====================================================
# 1. LOAD & PREPARE DATA
# =====================================================
df = pd.read_csv("masters.csv")
df["Tanggal"] = pd.to_datetime(df["Tanggal"])
df["Year"] = df["Tanggal"].dt.year

years = [2019, 2020, 2021, 2022, 2023, 2024]

features = [
    'Month', 'twi', 'elevation_mean', 'slope_mean', 'soil_type',
    'drainage_pct', 'precip_t0', 'precip_t1', 'precip_t2', 'precip_3d',
    'precip_7d', 'ndvi', 'mndwi', 'ndbi',
    'nb_precip_t0', 'nb_precip_t1', 'nb_precip_t2', 'nb_twi',
    'nb_elevation', 'nb_slope', 'nb_drainage', 'nb_soil',
    'nb_ndvi', 'nb_mndwi', 'nb_ndbi',
    '1daylag', '3daylag', '7daylag', 'ibi', 'nb_ibi'
]

target = "Target"

df = df[df["Year"].isin(years)]
df = df[["Year"] + features + [target]].dropna().reset_index(drop=True)

print(f"📦 Data ready: {len(df)} rows")

# =====================================================
# 2. TRAIN BASELINE RANDOM FOREST
# =====================================================
print("🚀 Training baseline Random Forest...")
X = df[features]
y = df[target]

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
rf.fit(X, y)
print("✅ Random Forest trained.")

# =====================================================
# 3. SHAP EXPLAINER
# =====================================================
explainer = shap.TreeExplainer(rf)

# =====================================================
# 4. SHAP BAR PER YEAR (FAST + SAFE)
# =====================================================
N_SAMPLE = 500
TOP_K = 15

for yr in tqdm(years, desc="Computing SHAP per year"):
    df_y = df[df["Year"] == yr]

    if len(df_y) < 50:
        print(f"⚠️ Skipping {yr} (not enough samples)")
        continue

    X_year = df_y[features].sample(
        min(N_SAMPLE, len(df_y)),
        random_state=42
    )

    # ---- NEW SHAP API (CORRECT)
    shap_exp = explainer(X_year)

    # class = 1 (flood)
    shap_vals_pos = shap_exp.values[:, :, 1]

    # ---- HARD CHECK
    assert shap_vals_pos.shape[1] == X_year.shape[1], \
        f"Mismatch: SHAP={shap_vals_pos.shape[1]}, X={X_year.shape[1]}"

    mean_abs_shap = np.abs(shap_vals_pos).mean(axis=0)

    shap_df = (
        pd.DataFrame({
            "Feature": X_year.columns.tolist(),
            "MeanAbsSHAP": mean_abs_shap
        })
        .sort_values("MeanAbsSHAP", ascending=True)
        .tail(TOP_K)
    )

    # =================================================
    # 5. BAR PLOT
    # =================================================
    plt.figure(figsize=(8, 6))
    plt.barh(
        shap_df["Feature"],
        shap_df["MeanAbsSHAP"],
        color="#ff69b4"
    )

    plt.title(
        f"SHAP Feature Importance (Mean |SHAP|)\nRandom Forest – {yr}",
        fontsize=13,
        fontweight="bold"
    )
    plt.xlabel("Mean |SHAP value|")
    plt.tight_layout()
    plt.show()
