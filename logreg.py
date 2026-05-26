import pandas as pd
import numpy as np
import optuna

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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

    C = trial.suggest_float('C', 1e-3, 10.0, log=True)
    penalty = trial.suggest_categorical('penalty', ['l2'])
    solver = 'lbfgs'

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            max_iter=1000,
            class_weight='balanced'
        ))
    ])

    tscv = TimeSeriesSplit(n_splits=3)
    kappa_scores = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        kappa_scores.append(cohen_kappa_score(y_val, y_pred))

    return np.mean(kappa_scores)

# =====================================================
# 3. MAIN EXPERIMENT FUNCTION
# =====================================================
def run_logreg_optimized(file_path, scenario_name):

    print("\n" + "=" * 80)
    print(f"🚀 RUNNING BATCH LOGISTIC REGRESSION: {scenario_name}")
    print("=" * 80)

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
        n_trials=20,
        show_progress_bar=True
    )

    print("\n✅ Best Hyperparameters:")
    print(study.best_params)

    # ----------------------------------
    # FINAL MODEL TRAINING
    # ----------------------------------
    final_model = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(
            **study.best_params,
            solver='lbfgs',
            max_iter=1000,
            class_weight='balanced'
        ))
    ])

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
    run_logreg_optimized(
        'master.csv',
        'Batch Logistic Regression (Dynamic LULC)'
    )
)

results.append(
    run_logreg_optimized(
        'masterstatic.csv',
        'Batch Logistic Regression (Static LULC)'
    )
)

# =====================================================
# 5. SAVE & DISPLAY RESULTS
# =====================================================
df_results = pd.DataFrame(results)

print("\n" + "=" * 110)
print(f"{'SCENARIO':<45} | {'ACC':<8} | {'PREC':<8} | {'REC':<8} | {'F1':<8} | {'KAPPA':<8} | {'AUC':<8}")
print("-" * 110)

for _, r in df_results.iterrows():
    print(
        f"{r['Scenario']:<45} | "
        f"{r['Accuracy']:8.4f} | "
        f"{r['Precision']:8.4f} | "
        f"{r['Recall']:8.4f} | "
        f"{r['F1-Score']:8.4f} | "
        f"{r['Kappa']:8.4f} | "
        f"{r['ROC-AUC']:8.4f}"
    )

print("=" * 110)

df_results.to_csv('batch_logistic_regression_results.csv', index=False)
print("✅ Results saved to: batch_logistic_regression_results.csv")
