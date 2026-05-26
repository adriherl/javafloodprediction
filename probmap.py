import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ============================================
# LOAD DATA
# ============================================

df2020 = pd.read_csv("prediction_dataset_2020.csv")
df2024 = pd.read_csv("prediction_dataset_2024.csv")

# training dataset (must contain flood label)
train_df = pd.read_csv("master.csv")

# ============================================
# FEATURES USED BY MODEL
# ============================================

features = [
    'Month', 'twi_final', 'elevation_mean', 'slope_mean', 'soil_type', 
    'drainage_pct', 'precip_t0', 'precip_t1', 'precip_t2', 'precip_3d', 
    'precip_7d', 'ndvi', 'mndwi', 'ndbi',
    'nb_precip_t0', 'nb_precip_t1', 'nb_precip_t2', 'nb_twi', 'nb_elevation', 
    'nb_slope', 'nb_drainage', 'nb_soil', 'nb_ndvi', 'nb_mndwi', 'nb_ndbi', 
    '1daylag', '3daylag', '7daylag', 'ibi', 'nb_ibi'
]

target = "Target"

# ============================================
# ENSURE SOIL TYPE NUMERIC
# ============================================

train_df["soil_type"] = pd.to_numeric(train_df["soil_type"], errors="coerce")
df2020["soil_type"] = pd.to_numeric(df2020["soil_type"], errors="coerce")
df2024["soil_type"] = pd.to_numeric(df2024["soil_type"], errors="coerce")

# ============================================
# PREPARE DATE
# ============================================

train_df["Tanggal"] = pd.to_datetime(train_df["Tanggal"], errors="coerce")

print(train_df["Tanggal"].head())

# ============================================
# TRAIN MODEL FOR 2020 (ONLY 2019)
# ============================================

train_2020 = train_df[train_df["Tanggal"].dt.year == 2019]

print("Training samples 2020 (2019 only):", len(train_2020))
print("Flood distribution:\n", train_2020[target].value_counts())

X_2020 = train_2020[features]
y_2020 = train_2020[target]

model_2020 = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)

model_2020.fit(X_2020, y_2020)

# ============================================
# TRAIN MODEL FOR 2024 (UP TO 2023)
# ============================================

train_2024 = train_df[train_df["Tanggal"].dt.year <= 2023]

print("Training samples 2024 (<=2023):", len(train_2024))
print("Flood distribution:\n", train_2024[target].value_counts())

X_2024 = train_2024[features]
y_2024 = train_2024[target]

model_2024 = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)

model_2024.fit(X_2024, y_2024)

# ============================================
# PREDICT PROBABILITY
# ============================================

print("Predicting 2020...")
df2020["probability"] = model_2020.predict_proba(df2020[features])[:,1]

print("Predicting 2024...")
df2024["probability"] = model_2024.predict_proba(df2024[features])[:,1]

# ============================================
# AGGREGATE BY KECAMATAN
# ============================================

prob2020 = (
    df2020
    .groupby(["WADMKK","WADMKC"])["probability"]
    .mean()
    .reset_index()
)

prob2024 = (
    df2024
    .groupby(["WADMKK","WADMKC"])["probability"]
    .mean()
    .reset_index()
)

# ============================================
# ADD JOIN FIELD
# ============================================

prob2020["JOIN_ID"] = prob2020["WADMKK"] + "_" + prob2020["WADMKC"]
prob2024["JOIN_ID"] = prob2024["WADMKK"] + "_" + prob2024["WADMKC"]

# ============================================
# SAVE OUTPUT
# ============================================

prob2020.to_csv("arf_prob_2020.csv", index=False)
prob2024.to_csv("arf_prob_2024.csv", index=False)

print("Finished.")