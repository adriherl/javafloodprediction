import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, cohen_kappa_score, roc_auc_score)
import warnings

# Mengabaikan peringatan agar output bersih
warnings.filterwarnings('ignore')

# 1. KONFIGURASI FITUR (30 Parameter Hidrologi & LULC UGM)
features = [
    'Month', 'twi_final', 'elevation_mean', 'slope_mean', 'soil_type', 
    'drainage_pct', 'precip_t0', 'precip_t1', 'precip_t2', 'precip_3d', 
    'precip_7d', 'ndvi', 'mndwi', 'ndbi',
    'nb_precip_t0', 'nb_precip_t1', 'nb_precip_t2', 'nb_twi', 'nb_elevation', 
    'nb_slope', 'nb_drainage', 'nb_soil', 'nb_ndvi', 'nb_mndwi', 'nb_ndbi', 
    '1daylag', '3daylag', '7daylag', 'ibi', 'nb_ibi'
]

def run_online_svm_comparison(file_path, scenario_name):
    print(f"🚀 Memproses Online SVM: {scenario_name}...")
    
    # Load data dan urutkan secara kronologis berdasarkan waktu
    df = pd.read_csv(file_path).sort_values('Tanggal')
    X = df[features]
    y = df['Target']
    classes = np.unique(y)

    # Inisialisasi Model & Scaler Inkremental
    # loss='hinge' mendefinisikan model sebagai Linear SVM
    model = SGDClassifier(loss='hinge', penalty='l2', alpha=0.01, random_state=42)
    scaler = StandardScaler()

    all_y_true = []
    all_y_pred = []
    all_y_probs = []

    # Simulasi Data Stream (Proses per Baris)
    for i in range(len(df)):
        # 1. Ambil satu sampel data (Reshape ke 2D array)
        X_sample = X.iloc[i:i+1].values
        y_sample = y.iloc[i:i+1].values

        # 2. Inkremental Scaling (Penting: Update mean/std setiap ada data baru)
        scaler.partial_fit(X_sample)
        X_sample_scaled = scaler.transform(X_sample)

        # 3. Predict-then-Train (Prequential Evaluation)
        if i > 0:
            # Prediksi kelas (0 atau 1)
            y_pred = model.predict(X_sample_scaled)
            
            # Estimasi skor keputusan (sebagai pengganti probabilitas untuk AUC)
            y_score = model.decision_function(X_sample_scaled)
            
            all_y_true.append(y_sample[0])
            all_y_pred.append(y_pred[0])
            all_y_probs.append(y_score[0])

        # 4. Model Belajar dari Sampel Ini (Update Bobot Inkremental)
        model.partial_fit(X_sample_scaled, y_sample, classes=classes)

    # Menghitung Metrik Evaluasi Akhir
    return {
        'Scenario': scenario_name,
        'Accuracy': accuracy_score(all_y_true, all_y_pred),
        'Precision': precision_score(all_y_true, all_y_pred),
        'Recall': recall_score(all_y_true, all_y_pred),
        'F1-Score': f1_score(all_y_true, all_y_pred),
        'Kappa': cohen_kappa_score(all_y_true, all_y_pred),
        'ROC-AUC': roc_auc_score(all_y_true, all_y_probs)
    }

# 2. EKSEKUSI PERBANDINGAN
results = []
try:
    results.append(run_online_svm_comparison('master.csv', 'Dynamic LULC (Online)'))
    results.append(run_online_svm_comparison('masterstatic.csv', 'Static LULC (Online)'))

    # 3. TAMPILKAN HASIL DALAM TABEL
    df_results = pd.DataFrame(results)
    print("\n" + "="*105)
    print(df_results.to_string(index=False, float_format=lambda x: "{:,.4f}".format(x)))
    print("="*105)
    
    # Simpan hasil untuk lampiran Bab 4
    df_results.to_csv('online_svm_final_results.csv', index=False)
    print("\n✅ Hasil berhasil disimpan ke 'online_svm_final_results.csv'")

except Exception as e:
    print(f"❌ Terjadi kesalahan: {e}")