import pandas as pd
import numpy as np
import optuna

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, cohen_kappa_score, roc_auc_score
)
from sklearn.model_selection import TimeSeriesSplit

# ===============================
# 1. FEATURE CONFIGURATION
# ===============================
features = [
    'Month', 'twi_final', 'elevation_mean', 'slope_mean', 'soil_type',
    'drainage_pct', 'precip_t0', 'precip_t1', 'precip_t2', 'precip_3d',
    'precip_7d', 'ndvi', 'mndwi', 'ndbi',
    'nb_precip_t0', 'nb_precip_t1', 'nb_precip_t2', 'nb_twi',
    'nb_elevation', 'nb_slope', 'nb_drainage', 'nb_soil',
    'nb_ndvi', 'nb_mndwi', 'nb_ndbi',
    '1daylag', '3daylag', '7daylag', 'ibi', 'nb_ibi'
]

# ===============================
# 2. OPTUNA OBJECTIVE FUNCTION
# ===============================
def objective(trial, X, y):

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1200),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'n_jobs': -1,
        'random_state': 42
    }

    tscv = TimeSeriesSplit(n_splits=3)
    kappa_scores = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)

        y_pred = model.predict(X_val)
        kappa_scores.append(cohen_kappa_score(y_val, y_pred))

    return np.mean(kappa_scores)

# ===============================
# 3. MAIN EXPERIMENT FUNCTION
# ===============================
def run_xgb_optimized(file_path, scenario_name):
    print("\n" + "=" * 60)
    print(f"🚀 RUNNING XGBOOST SCENARIO: {scenario_name}")
    print("=" * 60)

    # Load & sort data
    df = pd.read_csv(file_path)
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df = df.sort_values('Tanggal').reset_index(drop=True)

    available_features = [f for f in features if f in df.columns]
    X = df[available_features]
    y = df['Target']

    # ===============================
    # 80–20 CHRONOLOGICAL SPLIT
    # ===============================
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # ===============================
    # 4. OPTUNA OPTIMIZATION
    # ===============================
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=25,
        show_progress_bar=True
    )

    print("\n✅ Best Parameters:")
    print(study.best_params)

    # ===============================
    # 5. FINAL MODEL TRAINING
    # ===============================
    final_model = XGBClassifier(
        **study.best_params,
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='hist',
        n_jobs=-1,
        random_state=42
    )

    final_model.fit(X_train, y_train, verbose=False)

    # ===============================
    # 6. FINAL EVALUATION
    # ===============================
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

# ===============================
# 7. RUN ALL EXPERIMENTS
# ===============================
results = []
results.append(run_xgb_optimized('master.csv', 'Dynamic LULC (XGBoost)'))
results.append(run_xgb_optimized('masterstatic.csv', 'Static LULC (XGBoost)'))

df_results = pd.DataFrame(results)

print("\n" + "#" * 110)
print(df_results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("#" * 110)

df_results.to_csv('xgboost_timeseries_optimized_results.csv', index=False)
