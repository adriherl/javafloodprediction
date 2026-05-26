import pandas as pd
import numpy as np
from river import linear_model, preprocessing

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
# ONLINE LOGISTIC REGRESSION MODEL
# ============================================

print("Training Online Logistic Regression...")

model = preprocessing.StandardScaler() | linear_model.LogisticRegression()

# ============================================
# TRAIN MODEL (ONLINE LEARNING)
# ============================================

for _, row in train_df.iterrows():

    x = {f: row[f] for f in features}
    y = row[target]

    model.learn_one(x, y)

# ============================================
# PREDICT PROBABILITY
# ============================================

print("Predicting 2020...")

prob_list = []

for _, row in df2020.iterrows():

    x = {f: row[f] for f in features}
    prob = model.predict_proba_one(x).get(True, 0)

    prob_list.append(prob)

df2020["probability"] = prob_list

print("Predicting 2024...")

prob_list = []

for _, row in df2024.iterrows():

    x = {f: row[f] for f in features}
    prob = model.predict_proba_one(x).get(True, 0)

    prob_list.append(prob)

df2024["probability"] = prob_list

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

prob2020.to_csv("logreg_prob_2020.csv", index=False)
prob2024.to_csv("logreg_prob_2024.csv", index=False)

print("Finished.")