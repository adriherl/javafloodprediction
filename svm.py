import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score

# 1. KONFIGURASI FITUR
features = [
    'Month', 'twi_final', 'elevation_mean', 'slope_mean', 'soil_type', 
    'drainage_pct', 'precip_t0', 'precip_t1', 'precip_t2', 'precip_3d', 
    'precip_7d', 'ndvi', 'mndwi', 'ndbi',
    'nb_precip_t0', 'nb_precip_t1', 'nb_precip_t2', 'nb_twi', 'nb_elevation', 
    'nb_slope', 'nb_drainage', 'nb_soil', 'nb_ndvi', 'nb_mndwi', 'nb_ndbi', 
    '1daylag', '3daylag', '7daylag', 'ibi', 'nb_ibi'
]

def run_svm_batch(file_path, scenario_name):
    print(f"Sedang memproses Batch SVM (RBF) untuk: {scenario_name}...")
    
    # Load dan Sort berdasarkan waktu
    df = pd.read_csv(file_path)
    
    # Pastikan fitur tersedia
    available_features = [f for f in features if f in df.columns]
    X = df[available_features]
    y = df['Target']

    # Chronological Split (80% Train, 20% Test)
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # WAJIB: SVM sangat sensitif terhadap skala data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Inisialisasi Model SVC dengan RBF Kernel
    # probability=True agar kita bisa menghitung ROC-AUC
    model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    
    # Training
    model.fit(X_train_scaled, y_train)

    # Prediksi
    y_pred = model.predict(X_test_scaled)
    y_probs = model.predict_proba(X_test_scaled)[:, 1]

    # Hitung Metrik
    return {
        'Scenario': scenario_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'Kappa': cohen_kappa_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_probs)
    }

# 2. EKSEKUSI
results = []
results.append(run_svm_batch('master.csv', 'Dynamic LULC (SVM-Batch)'))
results.append(run_svm_batch('masterstatic.csv', 'Static LULC (SVM-Batch)'))

# 3. SIMPAN KE CSV
df_results = pd.DataFrame(results)
df_results.to_csv('batch_svm_results.csv', index=False)

# 4. TAMPILKAN TABEL
print("\n" + "="*105)
print(f"{'SCENARIO':<30} | {'ACC':<8} | {'PREC':<8} | {'REC':<8} | {'F1':<8} | {'KAPPA':<8}")
print("-" * 105)
for _, r in df_results.iterrows():
    print(f"{r['Scenario']:<30} | {r['Accuracy']:8.4f} | {r['Precision']:8.4f} | {r['Recall']:8.4f} | {r['F1-Score']:8.4f} | {r['Kappa']:8.4f}")
print("="*105)
print("✅ Hasil Batch SVM berhasil disimpan ke batch_svm_results.csv")