import pandas as pd
import numpy as np
import optuna
from river import linear_model, preprocessing, metrics, compose, optim
import warnings

warnings.filterwarnings('ignore')

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
# 2. OPTUNA OBJECTIVE FUNCTION (INCREMENTAL)
# =====================================================
def objective(trial, df_stream):

    lr = trial.suggest_float('lr', 0.001, 0.1, log=True)
    l2 = trial.suggest_float('l2', 1e-6, 1e-2, log=True)

    model = compose.Pipeline(
        preprocessing.StandardScaler(),
        linear_model.LogisticRegression(
            optimizer=optim.SGD(lr=lr),
            l2=l2
        )
    )

    kappa = metrics.kappa.CohenKappa()

    for xi, yi in zip(
        df_stream[features].to_dict(orient='records'),
        df_stream['Target']
    ):
        y_pred = model.predict_one(xi)
        if y_pred is not None:
            kappa.update(yi, y_pred)

        model.learn_one(xi, yi)

    return kappa.get()

# =====================================================
# 3. MAIN INCREMENTAL LOGISTIC REGRESSION FUNCTION
# =====================================================
def run_incremental_logreg_optimized(file_path, scenario_name):

    print("\n" + "=" * 80)
    print(f"🚀 RUNNING INCREMENTAL LOGISTIC REGRESSION: {scenario_name}")
    print("=" * 80)

    # ----------------------------------
    # Load & prepare data
    # ----------------------------------
    df = pd.read_csv(file_path)
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df = df.sort_values('Tanggal').reset_index(drop=True)
    df = df[features + ['Target']].dropna()

    # ----------------------------------
    # Chronological 80–20 split
    # ----------------------------------
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]

    # ----------------------------------
    # OPTIMIZATION ON SMALL SUBSET (FAST)
    # ----------------------------------
    tuning_fraction = 0.25
    tuning_size = int(len(df_train) * tuning_fraction)
    df_tune = df_train.iloc[:tuning_size]

    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, df_tune),
        n_trials=20,
        show_progress_bar=True
    )

    print("\n✅ Best Hyperparameters:")
    print(study.best_params)

    # ----------------------------------
    # FINAL INCREMENTAL MODEL
    # ----------------------------------
    model = compose.Pipeline(
        preprocessing.StandardScaler(),
        linear_model.LogisticRegression(
            optimizer=optim.SGD(lr=study.best_params['lr']),
            l2=study.best_params['l2']
        )
    )

    # ----------------------------------
    # PREQUENTIAL EVALUATION (TEST)
    # ----------------------------------
    report = {
        'Accuracy': metrics.Accuracy(),
        'Precision': metrics.Precision(),
        'Recall': metrics.Recall(),
        'F1-Score': metrics.F1(),
        'Kappa': metrics.kappa.CohenKappa(),
        'ROC-AUC': metrics.ROCAUC()
    }

    for xi, yi in zip(
        df_test[features].to_dict(orient='records'),
        df_test['Target']
    ):
        y_pred = model.predict_one(xi)
        y_proba = model.predict_proba_one(xi)

        if y_pred is not None:
            report['Accuracy'].update(yi, y_pred)
            report['Precision'].update(yi, y_pred)
            report['Recall'].update(yi, y_pred)
            report['F1-Score'].update(yi, y_pred)
            report['Kappa'].update(yi, y_pred)

            if y_proba and 1 in y_proba:
                report['ROC-AUC'].update(yi, y_proba[1])

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

# =====================================================
# 4. RUN BOTH SCENARIOS
# =====================================================
results = []

results.append(
    run_incremental_logreg_optimized(
        'master.csv',
        'Incremental Logistic Regression (Dynamic LULC)'
    )
)

results.append(
    run_incremental_logreg_optimized(
        'masterstatic.csv',
        'Incremental Logistic Regression (Static LULC)'
    )
)

# =====================================================
# 5. SAVE & DISPLAY RESULTS
# =====================================================
df_results = pd.DataFrame(results)

print("\n" + "=" * 110)
print(f"{'SCENARIO':<50} | {'ACC':<8} | {'PREC':<8} | {'REC':<8} | {'F1':<8} | {'KAPPA':<8} | {'AUC':<8}")
print("-" * 110)

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

print("=" * 110)

df_results.to_csv('incremental_logistic_regression_results.csv', index=False)
print("✅ Results saved to: incremental_logistic_regression_results.csv")
