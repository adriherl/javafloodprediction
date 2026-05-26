# =====================================================
# 0. IMPORTS & SETTINGS
# =====================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import calendar
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# =====================================================
# 1. LOAD & PREPARE DATA
# =====================================================
df = pd.read_csv("masters.csv")

df["Tanggal"] = pd.to_datetime(df["Tanggal"])
df = df.sort_values("Tanggal").reset_index(drop=True)

# Selected key features
dist_features = [
    "precip_t0",
    "precip_3d",
    "ndvi",
    "ibi",
    "mndwi",
    "slope_mean",
    "twi"
]

temporal_features = [
    "precip_3d",
    "ndvi",
    "ibi",
    "mndwi",
    "Target"
]

# Ensure numeric
for col in set(dist_features + temporal_features):
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna().reset_index(drop=True)

# =====================================================
# 2️⃣ DISTRIBUTION ANALYSIS
# =====================================================

# -----------------------------------------------------
# 2.1 Histogram + KDE
# -----------------------------------------------------
plt.figure(figsize=(14, 12))

for i, col in enumerate(dist_features, 1):

    plt.subplot(4, 2, i)

    sns.histplot(
        df[col],
        bins=50,
        kde=True,
        color="#1f77b4"
    )

    plt.title(f"Distribution of {col}", fontsize=11)
    plt.xlabel(col)
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# -----------------------------------------------------
# 2.2 Boxplots
# -----------------------------------------------------
plt.figure(figsize=(12, 6))

sns.boxplot(
    data=df[dist_features],
    orient="h",
    palette="pastel"
)

plt.title(
    "Boxplot of Selected Hydrometeorological and LULC Features",
    fontsize=13,
    fontweight="bold"
)

plt.xlabel("Value")

plt.tight_layout()
plt.show()

# =====================================================
# 3️⃣ TEMPORAL DISTRIBUTION ANALYSIS
# =====================================================

# Function to format month ticks every 4 months
def format_month_axis(ax):

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    plt.xticks(rotation=45)

# -----------------------------------------------------
# 3.1 Temporal Evolution of Rainfall
# -----------------------------------------------------
plt.figure(figsize=(14,6))

ax = df.set_index("Tanggal")["precip_3d"].rolling(30).mean().plot(
    color="#d62728",
    lw=2
)

format_month_axis(ax)

plt.title(
    "Temporal Evolution of 3-Day Cumulative Precipitation\n(30-Day Rolling Mean)",
    fontsize=13,
    fontweight="bold"
)

plt.ylabel("Precipitation")
plt.xlabel("Date")

plt.tight_layout()
plt.show()

# -----------------------------------------------------
# 3.2 Temporal Evolution of NDVI
# -----------------------------------------------------
plt.figure(figsize=(14,6))

ax = df.set_index("Tanggal")["ndvi"].rolling(30).mean().plot(
    color="#2ca02c",
    lw=2
)

format_month_axis(ax)

plt.title(
    "Temporal Evolution of NDVI\n(30-Day Rolling Mean)",
    fontsize=13,
    fontweight="bold"
)

plt.ylabel("NDVI")
plt.xlabel("Date")

plt.tight_layout()
plt.show()

# -----------------------------------------------------
# 3.3 Temporal Evolution of IBI
# -----------------------------------------------------
plt.figure(figsize=(14,6))

ax = df.set_index("Tanggal")["ibi"].rolling(30).mean().plot(
    color="#ff7f0e",
    lw=2
)

format_month_axis(ax)

plt.title(
    "Temporal Evolution of IBI\n(30-Day Rolling Mean)",
    fontsize=13,
    fontweight="bold"
)

plt.ylabel("IBI")
plt.xlabel("Date")

plt.tight_layout()
plt.show()

# -----------------------------------------------------
# 3.4 Temporal Evolution of MNDWI
# -----------------------------------------------------
plt.figure(figsize=(14,6))

ax = df.set_index("Tanggal")["mndwi"].rolling(30).mean().plot(
    color="#17becf",
    lw=2
)

format_month_axis(ax)

plt.title(
    "Temporal Evolution of MNDWI\n(30-Day Rolling Mean)",
    fontsize=13,
    fontweight="bold"
)

plt.ylabel("MNDWI")
plt.xlabel("Date")

plt.tight_layout()
plt.show()

# -----------------------------------------------------
# 3.5 Temporal Distribution of Flood Occurrence
# -----------------------------------------------------
plt.figure(figsize=(14,6))

ax = df.set_index("Tanggal")["Target"].rolling(30).mean().plot(
    color="#1f77b4",
    lw=2
)

format_month_axis(ax)

plt.title(
    "Temporal Distribution of Flood Occurrence\n(30-Day Rolling Mean)",
    fontsize=13,
    fontweight="bold"
)

plt.ylabel("Flood Probability")
plt.xlabel("Date")

plt.tight_layout()
plt.show()

# -----------------------------------------------------
# 3.6 Monthly Seasonality
# -----------------------------------------------------
df["Month"] = df["Tanggal"].dt.month

monthly_precip = df.groupby("Month")["precip_3d"].mean()
monthly_flood = df.groupby("Month")["Target"].mean()

month_labels = [calendar.month_abbr[i] for i in monthly_precip.index]

plt.figure(figsize=(12,5))

plt.plot(
    month_labels,
    monthly_precip,
    marker="o",
    label="Mean Precip_3d"
)

plt.plot(
    month_labels,
    monthly_flood,
    marker="s",
    label="Flood Frequency"
)

plt.title(
    "Monthly Seasonality of Rainfall and Flood Occurrence",
    fontsize=13,
    fontweight="bold"
)

plt.xlabel("Month")
plt.ylabel("Value")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()