# =====================================================
# FULL FEATURE-LEVEL STATISTICS (APPENDIX TABLE)
# =====================================================
import pandas as pd

# -----------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------
df = pd.read_csv("masters.csv")
df["Tanggal"] = pd.to_datetime(df["Tanggal"])
df = df.sort_values("Tanggal").reset_index(drop=True)

features = [
    'Month', 'twi', 'elevation_mean', 'slope_mean', 'soil_type',
    'drainage_pct',
    'precip_t0', 'precip_t1', 'precip_t2',
    'precip_3d', 'precip_7d',
    'ndvi', 'mndwi', 'ndbi',
    'nb_precip_t0', 'nb_precip_t1', 'nb_precip_t2',
    'nb_twi', 'nb_elevation', 'nb_slope',
    'nb_drainage', 'nb_soil',
    'nb_ndvi', 'nb_mndwi', 'nb_ndbi',
    '1daylag', '3daylag', '7daylag',
    'ibi', 'nb_ibi'
]

# -----------------------------------------------------
# 2. CLEAN & SELECT FEATURES
# -----------------------------------------------------
df = df[features].apply(pd.to_numeric, errors="coerce")
df = df.dropna()

# -----------------------------------------------------
# 3. COMPUTE FULL DESCRIPTIVE STATISTICS
# -----------------------------------------------------
stats_table = df.describe().T

# Rename columns for thesis clarity
stats_table = stats_table.rename(columns={
    "count": "Count",
    "mean": "Mean",
    "std": "Std",
    "min": "Min",
    "25%": "Q1",
    "50%": "Median",
    "75%": "Q3",
    "max": "Max"
})

# Optional: round values for readability
stats_table = stats_table.round(4)

# -----------------------------------------------------
# 4. SAVE FOR APPENDIX
# -----------------------------------------------------
output_file = "appendix_feature_statistics_dynamic_lulc.csv"
stats_table.to_csv(output_file)

print("✅ Full feature-level statistics saved to:")
print(output_file)

# Optional preview
print("\nPreview:")
print(stats_table.head(10))
