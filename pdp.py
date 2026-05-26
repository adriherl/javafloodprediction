# =====================================================
# 0. IMPORTS
# =====================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")

# =====================================================
# 1. LOAD & PREPARE DATA
# =====================================================
df = pd.read_csv("master.csv")

df["Tanggal"] = pd.to_datetime(df["Tanggal"])
df = df.sort_values("Tanggal").reset_index(drop=True)

features = [
    "nb_precip_t0",
    "precip_3d",
    "precip_t0",
    "ndvi"
]

target_col = "Target"

for col in features + [target_col]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna().reset_index(drop=True)

X = df[features]
y = df[target_col]

# =====================================================
# 2. TRAIN–TEST SPLIT
# =====================================================
split_idx = int(len(df) * 0.8)

X_train = X.iloc[:split_idx]
X_test  = X.iloc[split_idx:]

y_train = y.iloc[:split_idx]
y_test  = y.iloc[split_idx:]

# =====================================================
# 3. TRAIN RANDOM FOREST
# =====================================================
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=16,
    min_samples_split=3,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

print("🚀 Training Random Forest...")
rf.fit(X_train, y_train)

print(f"✅ Test Accuracy: {accuracy_score(y_test, rf.predict(X_test)):.4f}")

# =====================================================
# 4. PDP + MOVING AVERAGE SMOOTHING
# =====================================================
threshold_prob = 0.6
window_size = 5  # smoothing strength

fig, axes = plt.subplots(2,2, figsize=(14,10))
axes = axes.flatten()

PartialDependenceDisplay.from_estimator(
    estimator=rf,
    X=X_train,
    features=features,
    grid_resolution=60,
    response_method="predict_proba",
    percentiles=(0.01,0.99),
    ax=axes
)

# -----------------------------------------------------
# Apply Moving Average smoothing
# -----------------------------------------------------
for i, feat in enumerate(features):

    line = axes[i].lines[0]

    x_vals = line.get_xdata()
    y_vals = line.get_ydata()

    # Moving average smoothing
    y_smooth = (
        pd.Series(y_vals)
        .rolling(window_size, center=True)
        .mean()
        .interpolate()
    )

    # Remove original line
    axes[i].lines[0].remove()

    # Plot smoothed curve
    axes[i].plot(
        x_vals,
        y_smooth,
        color="#1f77b4",
        linewidth=2.5
    )

    # Find threshold intersection
    idx = np.argmin(np.abs(y_smooth - threshold_prob))
    x_thresh = x_vals[idx]

    # Draw threshold lines
    axes[i].axhline(
        threshold_prob,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="P = 0.6"
    )

    axes[i].axvline(
        x_thresh,
        color="black",
        linestyle=":",
        linewidth=1.5,
        label=f"Threshold ≈ {x_thresh:.2f}"
    )

    axes[i].set_title(feat)
    axes[i].set_ylabel("Predicted Flood Probability")

    axes[i].legend()

# -----------------------------------------------------
# Figure Title
# -----------------------------------------------------
fig.suptitle(
    "Flood Warning Thresholds Based on Partial Dependence Analysis",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout()
plt.show()