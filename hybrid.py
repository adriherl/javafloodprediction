# =====================================================
# 0. IMPORTS & WARNING HANDLING
# =====================================================
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*sklearn.utils.parallel.delayed.*"
)

from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from river import forest, metrics

# =====================================================
# 1. LOAD & PREPARE DATA
# =====================================================
df = pd.read_csv("master.csv")
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

for col in features + [target_col]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna().reset_index(drop=True)

X = df[features]
y = df[target_col]

# =====================================================
# 2. HYBRID CONFIGURATION
# =====================================================
HYBRID_START = 4793  # incremental activated here

# =====================================================
# 3. TRAIN BATCH RANDOM FOREST (HISTORICAL ONLY)
# =====================================================
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=16,
    min_samples_split=3,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

print(f"\n🚀 Training batch RF on rows 0–{HYBRID_START-1} ...")
rf.fit(X.iloc[:HYBRID_START], y.iloc[:HYBRID_START])
print("✅ Batch RF training completed.")

# =====================================================
# 4. INITIALIZE INCREMENTAL ARF
# =====================================================
arf = forest.ARFClassifier(
    n_models=30,
    max_depth=6,
    seed=42
)

# =====================================================
# 5. METRICS (EVALUATED ONLY AFTER HYBRID START)
# =====================================================
acc = metrics.Accuracy()
prec = metrics.Precision()
rec = metrics.Recall()
f1 = metrics.F1()
kappa = metrics.CohenKappa()
auc = metrics.ROCAUC()

# =====================================================
# 6. STREAMING HYBRID EVALUATION (PREQUENTIAL)
# =====================================================
print("\n🌊 Running hybrid RF–ARF evaluation (fair mode)...")

for i in tqdm(range(len(df)), desc="Hybrid streaming", ncols=100):

    xi = X.iloc[i].to_dict()
    yi = int(y.iloc[i])

    # -------------------------------
    # Prediction
    # -------------------------------
    if i < HYBRID_START:
        y_pred = rf.predict(X.iloc[i:i+1])[0]
        y_proba = rf.predict_proba(X.iloc[i:i+1])[0, 1]
    else:
        y_pred = arf.predict_one(xi)
        y_proba_dict = arf.predict_proba_one(xi)
        y_proba = y_proba_dict.get(1, 0.0) if y_proba_dict else 0.0

    # -------------------------------
    # FAIR METRIC UPDATE
    # (ONLY AFTER HYBRID START)
    # -------------------------------
    if i >= HYBRID_START:
        acc.update(yi, y_pred)
        prec.update(yi, y_pred)
        rec.update(yi, y_pred)
        f1.update(yi, y_pred)
        kappa.update(yi, y_pred)
        auc.update(yi, y_proba)

    # -------------------------------
    # Incremental learning
    # -------------------------------
    if i >= HYBRID_START:
        arf.learn_one(xi, yi)

# =====================================================
# 7. FINAL RESULTS (SCIENTIFICALLY VALID)
# =====================================================
print("\n" + "=" * 65)
print("✅ HYBRID RF–ARF PERFORMANCE (POST-ACTIVATION)")
print("=" * 65)
print(f"Accuracy : {acc.get():.4f}")
print(f"Precision: {prec.get():.4f}")
print(f"Recall   : {rec.get():.4f}")
print(f"F1-score : {f1.get():.4f}")
print(f"Kappa    : {kappa.get():.4f}")
print(f"ROC-AUC  : {auc.get():.4f}")
print("=" * 65)
