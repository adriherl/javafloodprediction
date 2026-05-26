import pandas as pd
from river import linear_model, preprocessing, metrics, compose
import warnings

warnings.filterwarnings('ignore')

# 1. KONFIGURASI FITUR
features = [
    'Month', 'twi_final', 'elevation_mean', 'slope_mean', 'soil_type',
     'precip_t0', 'precip_t1', 'precip_t2', 'precip_3d',
    'precip_7d', 
    'nb_precip_t0', 'nb_precip_t1', 'nb_precip_t2', 'nb_twi',
    'nb_elevation', 'nb_slope', 'nb_soil',
    
    '1daylag', '3daylag', '7daylag', 
]

def run_svm_incremental(file_path, scenario_name):
    print(f"Sedang memproses Streaming SVM (PA) untuk: {scenario_name}...")
    
    # Load dan Sort
    df = pd.read_csv(file_path).sort_values('Tanggal')
    
    # Pipeline: Scaling adalah WAJIB untuk model linear seperti SVM
    model = compose.Pipeline(
        preprocessing.StandardScaler(),
        linear_model.PAClassifier(
            C=1.0,           # Parameter agresivitas
            mode=1,          # Varian PA-I
            learn_intercept=True
        )
    )

    # Inisialisasi Metrik
    report = {
        'Accuracy': metrics.Accuracy(),
        'Precision': metrics.Precision(),
        'Recall': metrics.Recall(),
        'F1-Score': metrics.F1(),
        'Kappa': metrics.kappa.CohenKappa(),
        'ROC-AUC': metrics.ROCAUC() 
    }

    # Prequential Evaluation (Test-then-Train)
    for xi, yi in zip(df[features].to_dict(orient='records'), df['Target']):
        y_pred = model.predict_one(xi)
        y_proba = model.predict_proba_one(xi)
        
        if y_pred is not None:
            for m in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Kappa']:
                report[m].update(yi, y_pred)
            if y_proba:
                report['ROC-AUC'].update(yi, y_proba)
        
        # Update model secara real-time
        model.learn_one(xi, yi)

    return {
        'Scenario': scenario_name,
        'Accuracy': report['Accuracy'].get(),
        'Precision': report['Precision'].get(),
        'Recall': report['Recall'].get(),
        'F1-Score': report['F1-Score'].get(),
        'Kappa': report['Kappa'].get(),
        'ROC-AUC': report['ROC-AUC'].get()
    }

# 2. EKSEKUSI
results = []
results.append(run_svm_incremental('master.csv', 'Dynamic LULC (SVM-Inc)'))
results.append(run_svm_incremental('masterstatic.csv', 'Static LULC (SVM-Inc)'))

# 3. SIMPAN KE CSV
df_results = pd.DataFrame(results)
df_results.to_csv('incremental_svm_results.csv', index=False)

# 4. TAMPILKAN TABEL
print("\n" + "="*105)
print(f"{'SCENARIO':<30} | {'ACC':<8} | {'PREC':<8} | {'REC':<8} | {'F1':<8} | {'KAPPA':<8}")
print("-" * 105)
for _, r in df_results.iterrows():
    print(f"{r['Scenario']:<30} | {r['Accuracy']:8.4f} | {r['Precision']:8.4f} | {r['Recall']:8.4f} | {r['F1-Score']:8.4f} | {r['Kappa']:8.4f}")
print("="*105)
print("✅ Hasil Incremental SVM berhasil disimpan ke incremental_svm_results.csv")