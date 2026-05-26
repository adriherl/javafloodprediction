# =====================================================
# 0. IMPORTS
# =====================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

import warnings
warnings.filterwarnings("ignore")

# =====================================================
# 1. LOAD & PREPARE DATA
# =====================================================
df = pd.read_csv("masters.csv")
df["Tanggal"] = pd.to_datetime(df["Tanggal"])
df["Year"] = df["Tanggal"].dt.year

# Select years of interest
years = [2019, 2020, 2021, 2022, 2023,2024]
df = df[df["Year"].isin(years)].reset_index(drop=True)

features = [
    'Month', 'twi', 'elevation_mean', 'slope_mean', 'soil_type',
    'drainage_pct', 'precip_t0', 'precip_t1', 'precip_t2', 'precip_3d',
    'precip_7d', 'ndvi', 'mndwi', 'ndbi',
    'nb_precip_t0', 'nb_precip_t1', 'nb_precip_t2',
    'nb_twi', 'nb_elevation', 'nb_slope',
    'nb_drainage', 'nb_soil', 'nb_ndvi', 'nb_mndwi', 'nb_ndbi',
    '1daylag', '3daylag', '7daylag', 'ibi', 'nb_ibi'
]

df = df[["Year"] + features].dropna().reset_index(drop=True)

# =====================================================
# 2. PSI FUNCTION
# =====================================================
def calculate_psi(expected, actual, bins=10):
    """
    expected: baseline distribution (year t)
    actual: comparison distribution (year t+1)
    """
    expected = np.array(expected)
    actual = np.array(actual)

    quantiles = np.linspace(0, 100, bins + 1)
    breakpoints = np.percentile(expected, quantiles)

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    expected_perc = expected_counts / len(expected)
    actual_perc = actual_counts / len(actual)

    psi = 0
    for e, a in zip(expected_perc, actual_perc):
        if e == 0 or a == 0:
            continue
        psi += (a - e) * np.log(a / e)

    return psi

# =====================================================
# 3. COMPUTE PSI + KS PER FEATURE, PER YEAR
# =====================================================
results = []

for i in range(len(years) - 1):
    year_t = years[i]
    year_t1 = years[i + 1]

    df_t = df[df["Year"] == year_t]
    df_t1 = df[df["Year"] == year_t1]

    for feature in features:
        psi_value = calculate_psi(
            df_t[feature],
            df_t1[feature]
        )

        ks_stat, ks_p = ks_2samp(
            df_t[feature],
            df_t1[feature]
        )

        results.append({
            "Feature": feature,
            "Year_Comparison": f"{year_t}-{year_t1}",
            "PSI": psi_value,
            "KS_pvalue": ks_p
        })

drift_df = pd.DataFrame(results)

# =====================================================
# 4. SAVE RESULTS
# =====================================================
drift_df.to_csv("feature_drift_psi_ks_yearly.csv", index=False)
print("✅ Drift table saved: feature_drift_psi_ks_yearly.csv")

# =====================================================
# 5. PLOT PSI OVER TIME (TOP DRIFTING FEATURES)
# =====================================================
# Select top drifting features (max PSI)
top_features = (
    drift_df.groupby("Feature")["PSI"]
    .max()
    .sort_values(ascending=False)
    .head(5)
    .index
)

plt.figure(figsize=(12, 7))

for feature in top_features:
    subset = drift_df[drift_df["Feature"] == feature]
    plt.plot(
        subset["Year_Comparison"],
        subset["PSI"],
        marker="o",
        label=feature
    )

plt.axhline(0.1, color="gray", linestyle="--", alpha=0.6, label="PSI = 0.1")
plt.axhline(0.25, color="red", linestyle="--", alpha=0.6, label="PSI = 0.25")

plt.title("Feature-Level Population Stability Index (PSI) Over Time", fontsize=14, fontweight="bold")
plt.xlabel("Year Comparison")
plt.ylabel("PSI Value")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# =====================================================
# 6. QUICK SUMMARY PRINT
# =====================================================
print("\n📌 Features with PSI ≥ 0.25 (Significant Drift):")
print(
    drift_df[drift_df["PSI"] >= 0.25]
    .sort_values("PSI", ascending=False)
    .head(10)
)
