import os
import sys
import yaml
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import cross_validate, GroupKFold
from sklearn.metrics import f1_score, roc_auc_score

# --- Path Resolution ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.ml_utils import prepare_ml_dataset

def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def main():
    print("\n--- Starting Final Model Evaluation ---")
    
    # 1. Load config and define paths
    config = load_config(PROJECT_ROOT / "config.yml")
    processed_dir = PROJECT_ROOT / config["paths"]["processed_data"]
    models_dir = PROJECT_ROOT / config["paths"]["models_output"]
    reports_dir = PROJECT_ROOT / config["paths"]["reports"]
    
    # Crea la directory reports se non esiste
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = processed_dir / "features_target.csv"
    best_model_path = models_dir / "best_rf_model.pkl"
    features_test_path = processed_dir / "features_test.csv"
    target_test_path = processed_dir / "target_test.csv"
    
    # 2. Carica il Best Model
    print(f"Loading best model from: {best_model_path}")
    best_rf_model = joblib.load(best_model_path)
    
    # 3. Preparazione e split dei Dataset (come nello script 04)
    print("Reconstructing Train and Test datasets...")
    features_all, target_all, groups_all = prepare_ml_dataset(csv_path)
    
    # Carica il Test set salvato al termine dello script 04
    features_test = pd.read_csv(features_test_path, index_col=0)
    target_test = pd.read_csv(target_test_path, index_col=0).squeeze()
    
    # Ricava il Training set per differenza (assicura consistenza assoluta)
    train_indices = features_all.index.difference(features_test.index)
    features_train = features_all.loc[train_indices]
    target_train = target_all.loc[train_indices]
    groups_train = groups_all.loc[train_indices]
    
    # 4. Calcolo Metriche: Training e Validation Set (tramite Cross-Validation)
    print("\nCalculating metrics on Training and Validation sets via CV...")
    cv_folds = config["ml_pipeline"]["cv_folds"]
    gkf = GroupKFold(n_splits=cv_folds)
    
    scoring = {'f1': 'f1', 'roc_auc': 'roc_auc'}
    
    cv_results = cross_validate(
        estimator=best_rf_model,
        X=features_train,
        y=target_train,
        groups=groups_train,
        cv=gkf,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )
    
    train_f1 = cv_results['train_f1'].mean()
    val_f1 = cv_results['test_f1'].mean()
    train_auc = cv_results['train_roc_auc'].mean()
    val_auc = cv_results['test_roc_auc'].mean()
    
    # 5. Calcolo Metriche: Test Set
    print("Calculating metrics on Test set...")
    y_pred_test = best_rf_model.predict(features_test)
    y_pred_proba_test = best_rf_model.predict_proba(features_test)[:, 1]
    
    test_f1 = f1_score(target_test, y_pred_test)
    test_auc = roc_auc_score(target_test, y_pred_proba_test)
    
    # 6. Salvataggio dei risultati in CSV
    metrics_dict = {
        "Set_Valutazione": ["Training (CV Mean)", "Validation (CV Mean)", "Test Set"],
        "F1_Score": [round(train_f1, 4), round(val_f1, 4), round(test_f1, 4)],
        "AUC_ROC": [round(train_auc, 4), round(val_auc, 4), round(test_auc, 4)]
    }
    
    df_metrics = pd.DataFrame(metrics_dict)
    
    output_metrics_path = reports_dir / "final_model_metrics.csv"
    df_metrics.to_csv(output_metrics_path, index=False)
    
    print("\n--- Risultati Finali ---")
    print(df_metrics.to_string(index=False))
    print(f"\nTabella delle metriche salvata con successo in: {output_metrics_path}")
    print("--- Evaluation Complete ---")

if __name__ == "__main__":
    main()