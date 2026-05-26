import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, cohen_kappa_score, roc_auc_score)
import warnings

warnings.filterwarnings('ignore')

# 1. KONFIGURASI FITUR (30 Parameter Hidrologi & LULC)
features = [
    'Month', 'twi_final', 'elevation_mean', 'slope_mean', 'soil_type', 
    'drainage_pct', 'precip_t0', 'precip_t1', 'precip_t2', 'precip_3d', 
    'precip_7d', 'ndvi', 'mndwi', 'ndbi',
    'nb_precip_t0', 'nb_precip_t1', 'nb_precip_t2', 'nb_twi', 'nb_elevation', 
    'nb_slope', 'nb_drainage', 'nb_soil', 'nb_ndvi', 'nb_mndwi', 'nb_ndbi', 
    '1daylag', '3daylag', '7daylag', 'ibi', 'nb_ibi'
]

def run_1d_cnn_full_metrics(file_path, scenario_name):
    print(f"\n🌀 Memproses 1D-CNN: {scenario_name}...")
    
    # Load data & Sorting Kronologis
    df = pd.read_csv(file_path).sort_values('Tanggal')
    
    X = df[features].values
    y = df['Target'].values
    
    # Split Data (80% Train, 20% Test)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Preprocessing: Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Reshape untuk Conv1D: [samples, features, channels]
    X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    
    # 2. ARSITEKTUR 1D-CNN
    model = models.Sequential([
        # Layer 1: Konvolusi (Mendeteksi pola spasial/temporal lokal)
        layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(len(features), 1)),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),
        
        # Layer 2: Global Pooling untuk mengekstraksi fitur dominan
        layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        layers.GlobalAveragePooling1D(),
        
        # Layer 3: Fully Connected (MLP Part)
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid') # Output Probabilitas Biner
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Early Stopping untuk mencegah overfitting
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # 3. TRAINING
    model.fit(X_train_cnn, y_train, epochs=100, batch_size=32, verbose=0, 
              validation_split=0.1, callbacks=[early_stop])
    
    # 4. PREDIKSI & EVALUASI
    y_proba = model.predict(X_test_cnn).flatten()
    y_pred = (y_proba > 0.5).astype(int)
    
    return {
        'Scenario': scenario_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'Kappa': cohen_kappa_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba)
    }

# 5. EKSEKUSI EXPERIMEN
final_results = []
try:
    final_results.append(run_1d_cnn_full_metrics('master.csv', 'Dynamic LULC (CNN)'))
    final_results.append(run_1d_cnn_full_metrics('masterstatic.csv', 'Static LULC (CNN)'))
    
    # Tampilkan Hasil dalam Tabel
    df_comparison = pd.DataFrame(final_results)
    
    print("\n" + "="*115)
    print(df_comparison.to_string(index=False, float_format=lambda x: "{:,.4f}".format(x)))
    print("="*115)
    
    # Simpan ke CSV untuk Lampiran Skripsi
    df_comparison.to_csv('1d_cnn_full_comparison.csv', index=False)
    print("\n✅ Hasil perbandingan lengkap disimpan ke '1d_cnn_full_comparison.csv'")

except Exception as e:
    print(f"❌ Error: {e}")