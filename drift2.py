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

# Years of interest
years = [2019, 2020, 2021, 2022, 2023,2024]
df = df[df["Year"].isin(years)].reset_index(drop=True)

# Selected rainfall features ONLY
features = [
    'Month', 'twi', 'elevation_mean', 'slope_mean', 'soil_type',
    'drainage_pct', 'precip_t0', 'precip_t1', 'precip_t2', 'precip_3d',
    'precip_7d', 'ndvi', 'mndwi', 'ndbi',
    'nb_precip_t0', 'nb_precip_t1', 'nb_precip_t2', 'nb_twi',
    'nb_elevation', 'nb_slope', 'nb_drainage', 'nb_soil',
    'nb_ndvi', 'nb_mndwi', 'nb_ndbi',
    '1daylag', '3daylag', '7daylag', 'ibi', 'nb_ibi'
]

df = df[["Year"] + features].dropna().reset_index(drop=True)

print(f"📦 Data ready: {len(df)} rows")

# =====================================================
# 2. PSI FUNCTION
# =====================================================
def calculate_psi(expected, actual, bins=10):
    expected = np.asarray(expected)
    actual = np.asarray(actual)

    # Use quantile-based bins from baseline year
    breakpoints = np.percentile(
        expected,
        np.linspace(0, 100, bins + 1)
    )

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_perc = expected_counts / len(expected)
    actual_perc = actual_counts / len(actual)

    psi = 0.0
    for e, a in zip(expected_perc, actual_perc):
        if e > 0 and a > 0:
            psi += (a - e) * np.log(a / e)

    return psi

# =====================================================
# 3. COMPUTE PSI + KS (YEAR-to-YEAR)
# =====================================================
results = []

for i in range(len(years) - 1):
    year_t = years[i]
    year_t1 = years[i + 1]

    df_t = df[df["Year"] == year_t]
    df_t1 = df[df["Year"] == year_t1]

    print(f"🔍 Comparing {year_t} vs {year_t1}")

    for feature in features:
        psi_val = calculate_psi(df_t[feature], df_t1[feature])
        ks_stat, ks_p = ks_2samp(df_t[feature], df_t1[feature])

        results.append({
            "Feature": feature,
            "Year_Comparison": f"{year_t}-{year_t1}",
            "PSI": psi_val,
            "KS_pvalue": ks_p
        })

drift_df = pd.DataFrame(results)

# =====================================================
# 4. SAVE DRIFT TABLE
# =====================================================
drift_df.to_csv("rainfall_feature_drift_psi_ks.csv", index=False)
print("✅ Saved: rainfall_feature_drift_psi_ks.csv")

# =====================================================
# 5. PLOT PSI OVER TIME (ORDERED LEGEND, HORIZONTAL)
# =====================================================

# ---- Rank features by maximum PSI (descending)
feature_order = (
    drift_df.groupby("Feature")["PSI"]
    .max()
    .sort_values(ascending=False)
    .index
)

plt.figure(figsize=(14, 7))

# ---- Plot in ranked order
for feature in feature_order:
    subset = drift_df[drift_df["Feature"] == feature]
    plt.plot(
        subset["Year_Comparison"],
        subset["PSI"],
        marker="o",
        linewidth=2,
        label=feature
    )

# ---- PSI thresholds
plt.axhline(0.10, color="gray", linestyle="--", alpha=0.6, label="PSI = 0.10")
plt.axhline(0.25, color="red", linestyle="--", alpha=0.7, label="PSI = 0.25")

plt.title(
    "Population Stability Index (PSI) Over Time",
    fontsize=14,
    fontweight="bold"
)
plt.xlabel("Year Comparison")
plt.ylabel("PSI Value")

# ---- Vertical legend (compact & readable)
plt.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),   # place legend outside right
    frameon=False,
    fontsize=8,
    title="Features (↓ max PSI)",
    title_fontsize=9
)


plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# =====================================================
# 6. SUMMARY OUTPUT
# =====================================================
print("\n📌 Significant Drift (PSI ≥ 0.25):")
print(
    drift_df[drift_df["PSI"] >= 0.25]
    .sort_values("PSI", ascending=False)
)
