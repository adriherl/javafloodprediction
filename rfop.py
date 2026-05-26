import pandas as pd
import numpy as np
import optuna

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, cohen_kappa_score, roc_auc_score
)

# =====================================================
# 1. FEATURE CONFIGURATION
# =====================================================
features = [
    'Month', 'twi_final', 'elevation_mean', 'slope_mean', 'soil_type',
     'precip_t0', 'precip_t1', 'precip_t2', 'precip_3d',
    'precip_7d', 
    'nb_precip_t0', 'nb_precip_t1', 'nb_precip_t2', 'nb_twi',
    'nb_elevation', 'nb_slope', 'nb_soil',
    
    '1daylag', '3daylag', '7daylag', 
]

# =====================================================
# 2. OPTUNA OBJECTIVE FUNCTION
#    (BATCH + TimeSeriesSplit)
# =====================================================
def objective(trial, X, y):

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical(
            'max_features', ['sqrt', 'log2', None]
        ),
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }

    model = RandomForestClassifier(**params)

    tscv = TimeSeriesSplit(n_splits=3)
    kappa_scores = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        kappa_scores.append(
            cohen_kappa_score(y_val, y_pred)
        )

    return np.mean(kappa_scores)

# =====================================================
# 3. MAIN BATCH RF FUNCTION
# =====================================================
def run_rf_batch_optimized(file_path, scenario_name):

    print("\n" + "=" * 90)
    print(f"🚀 RUNNING BATCH RANDOM FOREST: {scenario_name}")
    print("=" * 90)

    # ----------------------------------
    # Load & prepare data
    # ----------------------------------
    df = pd.read_csv(file_path)
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df = df.sort_values('Tanggal').reset_index(drop=True)
    df = df[features + ['Target']].dropna()

    X = df[features]
    y = df['Target']

    # ----------------------------------
    # Chronological 80–20 split
    # ----------------------------------
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # ----------------------------------
    # OPTUNA OPTIMIZATION
    # ----------------------------------
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=30,
        show_progress_bar=True
    )

    print("\n✅ Best Hyperparameters:")
    print(study.best_params)

    # ----------------------------------
    # FINAL MODEL TRAINING
    # ----------------------------------
    final_model = RandomForestClassifier(
        **study.best_params,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    final_model.fit(X_train, y_train)

    # ----------------------------------
    # FINAL EVALUATION
    # ----------------------------------
    y_pred = final_model.predict(X_test)
    y_prob = final_model.predict_proba(X_test)[:, 1]

    return {
        'Scenario': scenario_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'Kappa': cohen_kappa_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    }

# =====================================================
# 4. RUN BOTH SCENARIOS
# =====================================================
results = []

results.append(
    run_rf_batch_optimized(
        'master.csv',
        'Batch Random Forest (Dynamic LULC)'
    )
)

results.append(
    run_rf_batch_optimized(
        'masterstatic.csv',
        'Batch Random Forest (Static LULC)'
    )
)

# =====================================================
# 5. SAVE & DISPLAY RESULTS
# =====================================================
df_results = pd.DataFrame(results)

print("\n" + "=" * 120)
print(f"{'SCENARIO':<50} | {'ACC':<8} | {'PREC':<8} | {'REC':<8} | {'F1':<8} | {'KAPPA':<8} | {'AUC':<8}")
print("-" * 120)

for _, r in df_results.iterrows():
    print(
        f"{r['Scenario']:<50} | "
        f"{r['Accuracy']:8.4f} | "
        f"{r['Precision']:8.4f} | "
        f"{r['Recall']:8.4f} | "
        f"{r['F1-Score']:8.4f} | "
        f"{r['Kappa']:8.4f} | "
        f"{r['ROC-AUC']:8.4f}"
    )

print("=" * 120)

df_results.to_csv('batch_random_forest_results.csv', index=False)
print("✅ Results saved to: batch_random_forest_results.csv")
