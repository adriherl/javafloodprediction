import pandas as pd
import numpy as np
import optuna
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, cohen_kappa_score, roc_auc_score)
import warnings

warnings.filterwarnings('ignore')

# 1. KONFIGURASI FITUR (30 Variabel Skripsi UGM)
features = [
    'Month', 'twi_final', 'elevation_mean', 'slope_mean', 'soil_type', 
    'drainage_pct', 'precip_t0', 'precip_t1', 'precip_t2', 'precip_3d', 
    'precip_7d', 'ndvi', 'mndwi', 'ndbi',
    'nb_precip_t0', 'nb_precip_t1', 'nb_precip_t2', 'nb_twi', 'nb_elevation', 
    'nb_slope', 'nb_drainage', 'nb_soil', 'nb_ndvi', 'nb_mndwi', 'nb_ndbi', 
    '1daylag', '3daylag', '7daylag', 'ibi', 'nb_ibi'
]

# 2. FUNGSI OBJEKTIF UNTUK OPTUNA (Mencari Window & Alpha)
def objective(trial, df):
    # Batasan window size maksimal 10% dari data
    limit_window = len(df) // 5
    window_size = trial.suggest_int('window_size', 50, limit_window)
    alpha = trial.suggest_float('alpha', 1e-4, 1e-1, log=True)
    
    X = df[features]
    y = df['Target']
    classes = np.unique(y)
    
    model = SGDClassifier(loss='hinge', penalty='l2', alpha=alpha, random_state=42)
    scaler = StandardScaler()
    
    all_y_true, all_y_pred = [], []

    # Proses Batch Incremental
    for i in range(0, len(df), window_size):
        X_batch = X.iloc[i : i + window_size]
        y_batch = y.iloc[i : i + window_size]
        if len(y_batch) < 1: continue
        
        scaler.partial_fit(X_batch)
        X_batch_scaled = scaler.transform(X_batch)
        
        if i > 0: # Test-then-Train
            all_y_true.extend(y_batch)
            all_y_pred.extend(model.predict(X_batch_scaled))
        
        model.partial_fit(X_batch_scaled, y_batch, classes=classes)
    
    return cohen_kappa_score(all_y_true, all_y_pred)

# 3. FUNGSI EKSEKUSI MODEL FINAL
def run_optimized_incremental_svm(file_path, scenario_name):
    print(f"\n" + "="*60)
    print(f"🔎 MENGOPTIMASI: {scenario_name}")
    print("="*60)
    
    df = pd.read_csv(file_path).sort_values('Tanggal')
    
    # A. Jalankan Optimizer TPE
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, df), n_trials=50)
    
    best_w = study.best_params['window_size']
    best_a = study.best_params['alpha']
    
    print(f"\n✅ Parameter Terbaik Ditemukan:")
    print(f"   - Window Size: {best_w}")
    print(f"   - Alpha: {best_a:.6f}")

    # B. Jalankan Model Final dengan Parameter Terbaik
    X, y = df[features], df['Target']
    classes = np.unique(y)
    model = SGDClassifier(loss='hinge', penalty='l2', alpha=best_a, random_state=42)
    scaler = StandardScaler()
    
    final_y_true, final_y_pred, final_y_scores = [], [], []

    for i in range(0, len(df), best_w):
        X_batch = X.iloc[i : i + best_w]
        y_batch = y.iloc[i : i + best_w]
        if len(y_batch) < 1: continue
        
        scaler.partial_fit(X_batch)
        X_batch_scaled = scaler.transform(X_batch)
        
        if i > 0:
            final_y_true.extend(y_batch)
            final_y_pred.extend(model.predict(X_batch_scaled))
            final_y_scores.extend(model.decision_function(X_batch_scaled))
        
        model.partial_fit(X_batch_scaled, y_batch, classes=classes)

    return {
        'Scenario': scenario_name,
        'Window_Size': best_w,
        'Accuracy': accuracy_score(final_y_true, final_y_pred),
        'Precision': precision_score(final_y_true, final_y_pred),
        'Recall': recall_score(final_y_true, final_y_pred),
        'F1-Score': f1_score(final_y_true, final_y_pred),
        'Kappa': cohen_kappa_score(final_y_true, final_y_pred),
        'ROC-AUC': roc_auc_score(final_y_true, final_y_scores)
    }

# 4. RUN SEMUA SKENARIO
results = []
try:
    results.append(run_optimized_incremental_svm('master.csv', 'Dynamic LULC'))
    results.append(run_optimized_incremental_svm('masterstatic.csv', 'Static LULC'))

    # Tampilkan Tabel Hasil Akhir
    df_res = pd.DataFrame(results)
    print("\n" + "#"*115)
    print(df_res.to_string(index=False, float_format=lambda x: "{:,.4f}".format(x)))
    print("#"*115)
    
    df_res.to_csv('optimized_inc_svm_results.csv', index=False)
    print("\n✅ Hasil disimpan ke 'optimized_inc_svm_results.csv'")

except Exception as e:
    print(f"❌ Error: {e}")