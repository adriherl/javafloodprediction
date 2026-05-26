import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score

features = [
    'Month', 'twi_final', 'elevation_mean', 'slope_mean', 'soil_type', 
    'drainage_pct', 'precip_t0', 'precip_t1', 'precip_t2', 'precip_3d', 
    'precip_7d', 'ndvi', 'mndwi', 'ndbi',
    'nb_precip_t0', 'nb_precip_t1', 'nb_precip_t2', 'nb_twi', 'nb_elevation', 
    'nb_slope', 'nb_drainage', 'nb_soil', 'nb_ndvi', 'nb_mndwi', 'nb_ndbi', 
    '1daylag', '3daylag', '7daylag', 'ibi', 'nb_ibi'
]

def run_mlp_batch(file_path, scenario_name):
    print(f"Memproses Batch MLP (Optimized): {scenario_name}...")
    df = pd.read_csv(file_path).sort_values('Tanggal')
    
    X = df[features]
    y = df['Target']
    split_index = int(len(df) * 0.8)
    
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # PERBAIKAN:
    # 1. max_iter ditingkatkan ke 1000
    # 2. early_stopping diaktifkan untuk mencegah overfitting
    # 3. learning_rate_init disetel sedikit lebih tinggi
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64), 
        max_iter=1000, 
        learning_rate_init=0.001,
        early_stopping=True, 
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_probs = model.predict_proba(X_test_scaled)[:, 1]

    return {
        'Scenario': scenario_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'Kappa': cohen_kappa_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_probs)
    }

results_batch = [run_mlp_batch('master.csv', 'Dynamic MLP (Batch)'), 
                 run_mlp_batch('masterstatic.csv', 'Static MLP (Batch)')]
pd.DataFrame(results_batch).to_csv('batch_mlp_results.csv', index=False)