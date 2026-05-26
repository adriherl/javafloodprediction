from river import forest, preprocessing, compose
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# 1. Load data
df = pd.read_csv('master.csv')


features = [
    'Month', 'twi_final', 'elevation_mean', 'slope_mean', 'soil_type', 
    'drainage_pct', 'precip_t0', 'precip_t1', 'precip_t2', 'precip_3d', 
    'precip_7d', 'ndvi', 'mndwi', 'ndbi',
    'nb_precip_t0', 'nb_precip_t1', 'nb_precip_t2', 'nb_twi', 'nb_elevation', 
    'nb_slope', 'nb_drainage', 'nb_soil', 'nb_ndvi', 'nb_mndwi', 'nb_ndbi', 
    '1daylag', '3daylag', '7daylag', 'ibi', 'nb_ibi'
]

# 2. Define Model Pipeline
model = compose.Pipeline(
    preprocessing.StandardScaler(),
    forest.ARFClassifier(n_models=30, seed=42)
)

# 3. Full Training Loop
print("Training model on the entire stream...")
for xi, yi in zip(df[features].to_dict(orient='records'), df['Target']):
    model.learn_one(xi, yi)
print("Training complete.")

# --- 4. SHAP ANALYSIS (At the End) ---

# Wrapper to handle the river pipeline
def predict_proba_pipeline(X_np):
    return np.array([model.predict_proba_one(dict(zip(features, row))).get(1, 0.0) for row in X_np])

print("\nCalculating final SHAP values (this may take a moment)...")

# Configuration for a clean, stable plot
# Use 100 recent samples to show the final state of the model
test_set = df[features].tail(100) 
# Use the median of the whole dataset as a stable background reference
background = df[features].median().values.reshape(1, -1)

explainer = shap.KernelExplainer(predict_proba_pipeline, background)

# nsamples limits the time; 200-500 is a good balance for accuracy vs speed
shap_values = explainer.shap_values(test_set, nsamples=200, silent=False)

# 5. Create the Final Beeswarm Plot
explanation = shap.Explanation(
    values=shap_values,
    base_values=explainer.expected_value,
    data=test_set.values,
    feature_names=features
)

plt.figure(figsize=(12, 8))
shap.plots.beeswarm(explanation)
plt.title("Final Model Logic: Feature Importance for Flood Prediction")
plt.show()