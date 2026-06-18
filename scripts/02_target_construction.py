import os
import sys
import yaml
import xarray as xr
from pathlib import Path

# --- Path Resolution ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data_helpers import extract_pot_target


def load_config(config_path: Path) -> dict:
    """Loads the YAML configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def main():
    """
    Main execution block for 02_target_construction.py.
    Loads the processed precipitation data, applies POT at the configured 
    location, and saves the binary target as a CSV file.
    """
    print("\n--- Starting Target Construction (POT) ---")
    
    # 1. Load config and paths
    config = load_config(PROJECT_ROOT / "config.yml")
    paths = config["paths"]
    target_config = config["target"]
    
    processed_dir = PROJECT_ROOT / paths["processed_data"]
    precip_file_path = processed_dir / f"{target_config['variable']}_processed.nc"
    
    # 2. Get active location coordinates dynamically
    active_loc_name = target_config["active_location"]
    target_coords = target_config["locations"][active_loc_name]
    
    print(f"Active location set to: {active_loc_name.upper()}")
    
    # 3. Load processed dataset
    if not precip_file_path.exists():
        raise FileNotFoundError(
            f"Processed file {precip_file_path} missing. Run 01_preprocessing.py first."
        )
    
    print(f"Loading processed dataset from {precip_file_path}...")
    precip_final = xr.open_dataset(precip_file_path)
    
    # 4. Extract POT target
    df_target, threshold = extract_pot_target(
        precip_ds=precip_final,
        var_name=target_config['variable'],
        lon_target=target_coords['lon'],
        lat_target=target_coords['lat'],
        percentile_threshold=target_config['pot_threshold_percentile']
    )
    
    # 5. Save the output as CSV
    output_csv_path = processed_dir / "target.csv"
    df_target.to_csv(output_csv_path, index=False)
    
    print(f"Successfully saved POT target to: {output_csv_path}")
    print("--- Target Construction Complete ---\n")


if __name__ == "__main__":
    main()