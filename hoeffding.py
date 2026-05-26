from river import tree, preprocessing, metrics, compose
import pandas as pd

# Fitur utama skripsi
features = [
    'Month', 'twi_final', 'elevation_mean', 'slope_mean', 'soil_type', 
    'drainage_pct', 'precip_t0', 'precip_t1', 'precip_t2', 'precip_3d', 
    'precip_7d', 'ndvi', 'mndwi', 'ndbi',
    'nb_precip_t0', 'nb_precip_t1', 'nb_precip_t2', 'nb_twi', 'nb_elevation', 
    'nb_slope', 'nb_drainage', 'nb_soil', 'nb_ndvi', 'nb_mndwi', 'nb_ndbi', 
    '1daylag', '3daylag', '7daylag', 'ibi', 'nb_ibi'
]

def run_ht_incremental(file_path, scenario_name):
    print(f"Memproses Hoeffding Tree: {scenario_name}...")
    df = pd.read_csv(file_path).sort_values('Tanggal')
    
    # PERBAIKAN: split_confidence diganti menjadi delta
    model = compose.Pipeline(
        preprocessing.StandardScaler(),
        tree.HoeffdingTreeClassifier(
            grace_period=100,
            delta=1e-7,       # Ini adalah parameter split confidence Anda
            tau=0.05,         # Tie-breaking threshold
            leaf_prediction='mc'
        )
    )

    # Metrik lengkap untuk Bab 4
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
            for m_name in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Kappa']:
                report[m_name].update(yi, y_pred)
            
            if y_proba:
                report['ROC-AUC'].update(yi, y_proba)
                
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

# Eksekusi
results_inc = [
    run_ht_incremental('master.csv', 'Dynamic HT (Incremental)'), 
    run_ht_incremental('masterstatic.csv', 'Static HT (Incremental)')
]

# Simpan ke CSV
df_final = pd.DataFrame(results_inc)
df_final.to_csv('incremental_ht_results_fixed.csv', index=False)
print("\n✅ Hasil berhasil disimpan di incremental_ht_results_fixed.csv")