# =====================================================
# CORRELATION HEATMAP (EDA)
# Dynamic LULC
# =====================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -----------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------
df = pd.read_csv("masters.csv")
df["Tanggal"] = pd.to_datetime(df["Tanggal"])
df = df.sort_values("Tanggal").reset_index(drop=True)

features = [
    'Month', 'twi', 'elevation_mean', 'slope_mean',
    'drainage_pct',
    'precip_t0', 'precip_t1', 'precip_t2',
    'precip_3d', 'precip_7d',
    'ndvi', 'mndwi', 'ndbi','ibi',
    '1daylag', '3daylag', '7daylag'
]

df = df[features].apply(pd.to_numeric, errors="coerce")
df = df.dropna()

# -----------------------------------------------------
# 2. COMPUTE CORRELATION MATRIX
# -----------------------------------------------------
corr_matrix = df.corr(method="pearson")

# -----------------------------------------------------
# 3. PLOT HEATMAP
# -----------------------------------------------------
plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    annot=False,
    linewidths=0.5,
    square=True,
    cbar_kws={"label": "Pearson Correlation"}
)

plt.title(
    "Feature Correlation Heatmap (Dynamic LULC)",
    fontsize=14,
    fontweight="bold"
)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
