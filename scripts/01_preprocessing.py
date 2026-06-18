import os
import sys
import yaml
from pathlib import Path

# --- Path Resolution ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.data_helpers import process_climate_variable

CONFIG_PATH = PROJECT_ROOT / "config.yml"


def load_config(config_path: Path) -> dict:
    """Loads the YAML configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def setup_directories(paths_config: dict, project_root: Path):
    """
    Iterates over the directories defined in the config and creates them if they don't exist.
    """
    print("--- Checking Directories ---")
    for dir_name, relative_path in paths_config.items():
        full_path = project_root / relative_path
        
        # parents=True: creates intermediate folders if missing (e.g., notebooks/reports)
        # exist_ok=True: prevents crashing if the folder already exists
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"Ready: {full_path}")


def main():
    """
    Main execution block for 01_preprocessing.py.
    """
    # 1. Load settings
    config = load_config(CONFIG_PATH)
    paths = config["paths"]
    
    # 2. Setup output folders automatically
    setup_directories(paths, PROJECT_ROOT)

    preproc_params = config["preprocessing"]
    variables_dict = config["variables"]
    raw_dir = PROJECT_ROOT / paths["raw_data"]
    processed_dir = PROJECT_ROOT / paths["processed_data"]

    # 3. Iterate through variables defined in config
    for var_name, suffixes in variables_dict.items():
        print(f"\n--- Starting processing for {var_name.upper()} ---")
        
        processed_ds = process_climate_variable(
            variable_name=var_name,
            suffixes=suffixes,
            raw_data_folder=str(raw_dir),
            target_month=preproc_params["target_month"],
            start_year=preproc_params["start_year"],
            max_ensemble_members=preproc_params["max_ensemble_members"]
        )
        
        # 4. Save to disk
        output_path = processed_dir / f"{var_name}_processed.nc"
        processed_ds.to_netcdf(output_path)
        print(f"Successfully saved {var_name.upper()} to {output_path}")


if __name__ == "__main__":
    main()