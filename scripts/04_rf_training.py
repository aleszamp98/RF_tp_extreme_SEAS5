import os
import sys
import yaml
import joblib
from pathlib import Path

# --- Path Resolution ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.ml_utils import prepare_ml_dataset, train_rf_gridsearch

def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def main():
    print("\n--- Starting Random Forest Training & Grid Search ---")
    
    # 1. Load config and define paths
    config = load_config(PROJECT_ROOT / "config.yml")
    processed_dir = PROJECT_ROOT / config["paths"]["processed_data"]
    models_dir = PROJECT_ROOT / config["paths"]["models_output"]
    
    csv_path = processed_dir / "features_target.csv"
    
    # 2. Prepare Data
    features_all, target_all, groups_all = prepare_ml_dataset(csv_path)

    # 3. Define Hyperparameter Grid from config
    param_grid = config["ml_pipeline"]["random_forest_grid"]

    # 4. Train with GridSearchCV
    grid_search_results, features_test, target_test = train_rf_gridsearch(
        features=features_all,
        target=target_all,
        groups=groups_all,
        param_grid=param_grid,
        output_dir=models_dir,
        test_size=0.2 
    )

    features_test_path = processed_dir / "features_test.csv"
    target_test_path = processed_dir / "target_test.csv"
    
    features_test.to_csv(features_test_path)
    target_test.to_csv(target_test_path)
    print(f"\nTest sets saved in: {processed_dir}")

    full_grid_search_path = models_dir / "full_grid_search.pkl"
    joblib.dump(grid_search_results, full_grid_search_path)
    print(f"Complete GridSearch object saved in: {full_grid_search_path}")

    best_rf_model = grid_search_results.best_estimator_
    best_model_path = models_dir / "best_rf_model.pkl"
    joblib.dump(best_rf_model, best_model_path)
    print(f"Best Random Forest model saved in: {best_model_path}")

    print("\n--- Training Pipeline Complete ---")

if __name__ == "__main__":
    main()