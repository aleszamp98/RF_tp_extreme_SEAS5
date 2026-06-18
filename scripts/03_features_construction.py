import sys
import yaml
import xarray as xr
import pandas as pd
from pathlib import Path

# --- Path Resolution ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data_helpers import (
    preprocess_pressure_levels, 
    compute_daily_anomalies, 
    perform_pca_workflow
)

def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def main():
    print("\n--- Starting Features Construction (PCA) ---")
    
    # 1. Load configuration
    config = load_config(PROJECT_ROOT / "config.yml")
    paths = config["paths"]
    processed_dir = PROJECT_ROOT / paths["processed_data"]
    
    # Create a subfolder for PCA analysis outputs to keep things clean
    pca_output_dir = processed_dir / "pca_results"
    pca_output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- UPDATED LOADING LOGIC HERE ---
    target_var = config["target"]["variable"]
    pca_params = config["feature_extraction"]["pca"]
    pca_seed = pca_params["random_state"]
    pca_variance = pca_params["explained_variance_ratio"]
    
    # Map external variable names (keys in config) to internal NetCDF variable names
    internal_var_mapping = {
        "msl": "msl",
        "z500": "z",
        "q850": "q"
    }
    
    # Get all variables to process (excluding the target 'tp')
    feature_vars = [v for v in config["variables"].keys() if v != target_var]
    
    # List to hold the PC dataframes for final merging
    all_pcs_dataframes = []
    
    # 2. Iterate and process each variable
    for var in feature_vars:
        print(f"\n>> Processing: {var.upper()}")
        internal_var = internal_var_mapping.get(var, var)
        
        # Load processed dataset
        file_path = processed_dir / f"{var}_processed.nc"
        ds = xr.open_dataset(file_path)
        
        # Squeeze pressure levels if they exist
        ds_squeezed = preprocess_pressure_levels(ds)
        
        # Compute Anomalies
        ds_anomalies = compute_daily_anomalies(ds_squeezed, internal_var)
        
        # Perform PCA
        pc_df, eofs_da, explained_var = perform_pca_workflow(
            ds_anom=ds_anomalies,
            var_name=internal_var,
            n_components=pca_variance,
            random_state=pca_seed
        )
        
        # Collect PC dataframe for the final ML dataset
        all_pcs_dataframes.append(pc_df)
        
        # --- 3. Save Analytical Outputs (Properly) ---
        # Anomalies to NetCDF
        ds_anomalies.to_netcdf(pca_output_dir / f"{var}_anomalies.nc")
        
        # EOFs to NetCDF (for mapping in Notebooks)
        eofs_dataset = eofs_da.to_dataset(name=f"eof_{internal_var}")
        eofs_dataset.to_netcdf(pca_output_dir / f"{var}_eofs.nc")
        
        # Explained Variance to CSV (for scree plots)
        explained_var.to_csv(pca_output_dir / f"{var}_explained_variance.csv", header=True)
        
        print(f"Saved analytical outputs for {var.upper()} in {pca_output_dir.name}/")

    # --- 4. Assemble the Final Machine Learning Dataset ---
    print("\n>> Assembling Final Feature Matrix...")
    
    # Merge all PC dataframes on their shared 'id' index
    # We drop 'forecast_time' from the subsequent dataframes to avoid duplicates
    final_features_df = all_pcs_dataframes[0]
    for df in all_pcs_dataframes[1:]:
        final_features_df = final_features_df.join(df.drop(columns=['forecast_time']))

    # Load the target.csv (containing 'id' and 'target')
    target_path = processed_dir / "target.csv"
    if target_path.exists():
        target_df = pd.read_csv(target_path)
        target_df.set_index('id', inplace=True)
        
        # Inner join to ensure alignment between Features and Target
        final_dataset = final_features_df.join(target_df, how='inner')
        
        final_output_path = processed_dir / "features_target.csv"
        final_dataset.to_csv(final_output_path)
        print(f"SUCCESS: Final dataset created with shape {final_dataset.shape}")
        print(f"Saved to: {final_output_path}")
    else:
        print(f"WARNING: {target_path} not found. Cannot attach target column.")
        
    print("--- Features Construction Complete ---\n")

if __name__ == "__main__":
    main()