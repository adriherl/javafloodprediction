# =====================================================
# 0. IMPORTS
# =====================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# =====================================================
# 1. LOAD RESULTS (REAL CSV)
# =====================================================
df = pd.read_csv("result.csv")

# -----------------------------------------------------
# FIX STRUCTURAL ISSUES IN CSV
# -----------------------------------------------------

# Forward-fill Category (because it is merged visually)
df["Category"] = df["Category"].ffill()

# Strip whitespace
df["Scenario"] = df["Scenario"].str.strip()
df["Model"] = df["Model"].str.strip()
df["Category"] = df["Category"].str.strip()

# =====================================================
# 2. DEFINE LEARNING PARADIGM
# =====================================================
batch_models = [
    "RF Classifier",
    "SVM",
    "Logistic Regression"
]

incremental_models = [
    "ARF",
    "PA Classifier",
    "Online Logistic Regression"
]

df["Learning Paradigm"] = df["Model"].apply(
    lambda x: "Batch" if x in batch_models else "Incremental"
)

# =====================================================
# 3. PLOT 1 — LULC EFFECT (F1-SCORE)
# =====================================================
plt.figure(figsize=(14, 6))

sns.barplot(
    data=df,
    x="Model",
    y="F1-Score",
    hue="Scenario",
    palette="Set2"
)

plt.title(
    "Effect of LULC Representation on Model Performance (F1-Score)",
    fontsize=14,
    fontweight="bold"
)
plt.ylabel("F1-Score")
plt.xlabel("Model")
plt.xticks(rotation=30, ha="right")
plt.legend(title="Scenario")
plt.tight_layout()
plt.show()

# =====================================================
# 4. PLOT 2 — BATCH vs INCREMENTAL (DYNAMIC LULC)
# =====================================================
df_dynamic = df[df["Scenario"] == "Dynamic LULC"]

plt.figure(figsize=(10, 6))

sns.barplot(
    data=df_dynamic,
    x="Model",
    y="F1-Score",
    hue="Learning Paradigm",
    palette=["#4C72B0", "#DD8452"]
)

plt.title(
    "Batch vs Incremental Learning Performance\n(Dynamic LULC, F1-Score)",
    fontsize=14,
    fontweight="bold"
)
plt.ylabel("F1-Score")
plt.xlabel("Model")
plt.xticks(rotation=30, ha="right")
plt.legend(title="Learning Paradigm")
plt.tight_layout()
plt.show()

# =====================================================
# 5. PLOT 3 — CATEGORY-LEVEL COMPARISON (KAPPA)
# =====================================================
plt.figure(figsize=(12, 6))

sns.barplot(
    data=df,
    x="Category",
    y="Kappa",
    hue="Scenario",
    palette="muted"
)

plt.title(
    "Model Agreement Across Categories and LULC Scenarios (Kappa)",
    fontsize=14,
    fontweight="bold"
)
plt.ylabel("Cohen's Kappa")
plt.xlabel("Model Category")
plt.legend(title="Scenario")
plt.tight_layout()
plt.show()
