import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score

# 1. KONFIGURASI FITUR
features = [
    'Month', 'twi_final', 'slope_mean', 'soil_type', 
    'drainage_pct', 'precip_t0',
    'precip_7d', 'ndvi', 'mndwi', 'ndbi',
    'nb_precip_t0', 'nb_precip_t1', 'nb_precip_t2', 'nb_twi', 'nb_elevation', 
    'nb_slope', 'nb_drainage', 'nb_soil', 'nb_ndvi', 'nb_mndwi', 'nb_ndbi', 
    '1daylag', 'ibi', 'nb_ibi'
]

def run_rf_batch(file_path, scenario_name):
    print(f"Processing {scenario_name}...")
    
    # Load and Sort Chronologically
    df = pd.read_csv(file_path)
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df = df.sort_values(by='Tanggal')
    
    # Filter available features
    available_features = [f for f in features if f in df.columns]
    
    X = df[available_features]
    y = df['Target']

    # Chronological Split (80% Train, 20% Test)
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # Define and Train Batch Random Forest
    # n_estimators=100 adalah standar industri untuk baseline yang kuat
    model = RandomForestClassifier(n_estimators=705, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]

    # Calculate Metrics
    return {
        'Scenario': scenario_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'Kappa': cohen_kappa_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_probs)
    }

# 2. RUN BOTH SCENARIOS
results = []
results.append(run_rf_batch('master.csv', 'Dynamic LULC (Real)'))
results.append(run_rf_batch('masterstatic.csv', 'Static LULC (2019)'))

# 3. SAVE RESULTS TO CSV
df_results = pd.DataFrame(results)
output_filename = 'rf_comparison_results.csv'
df_results.to_csv(output_filename, index=False)

# 4. PRINT SUMMARY TABLE
print("\n" + "="*105)
print(f"{'SCENARIO':<25} | {'ACC':<8} | {'PREC':<8} | {'REC':<8} | {'F1':<8} | {'KAPPA':<8} | {'AUC':<8}")
print("-" * 105)

for _, res in df_results.iterrows():
    print(f"{res['Scenario']:<25} | {res['Accuracy']:8.4f} | {res['Precision']:8.4f} | {res['Recall']:8.4f} | {res['F1-Score']:8.4f} | {res['Kappa']:8.4f} | {res['ROC-AUC']:8.4f}")

print("-" * 105)
print(f"✅ Results successfully saved to: {output_filename}")
print("="*105)