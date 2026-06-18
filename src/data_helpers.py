import warnings
import gc
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.decomposition import PCA
from typing import List, Optional, Tuple


def standardize_SEAS5(ds: xr.Dataset, variable_name: Optional[str] = None) -> xr.Dataset:
    """
    Standardize coordinate names for SEAS5 NetCDF files and compute validity times.
    Optionally, it can standardize units for a specific variable, managing both
    monthly rates and daily accumulated variables.

    Parameters
    ----------
    ds : xr.Dataset
        Input xarray Dataset containing SEAS5 NetCDF data.
    variable_name : str, optional
        The name of the variable to standardize units for ("tprate", "tp", etc.).

    Returns
    -------
    xr.Dataset
        Dataset with standardized coordinate names and assigned 'valid_time' 
        and 'target_month'.
    """
    ds = ds.rename({
        "latitude": "lat",
        "longitude": "lon",
        "number": "ensemble_member",
        "forecast_reference_time": "init_time",
    })

    is_daily = "forecast_period" in ds.coords

    if is_daily:
        ds = ds.rename({"forecast_period": "lead_time"})
        
        if "valid_time" in ds.data_vars:
            ds = ds.set_coords("valid_time")
        elif "valid_time" not in ds.coords:
            ds.coords['valid_time'] = ds.init_time + ds.lead_time

        ds.coords['target_month'] = ds.valid_time.dt.month

    else:
        ds = ds.rename({"forecastMonth": "lead_time"})
        
        inits = pd.to_datetime(ds.init_time.values)
        leads = ds.lead_time.values
        valid_times = [
            [(init + pd.DateOffset(months=int(lead))) for lead in leads]
            for init in inits
        ]
            
        valid_times_np = np.array(valid_times, dtype='datetime64[ns]')
        ds = ds.assign_coords(valid_time=(('init_time', 'lead_time'), valid_times_np))
        ds.coords['target_month'] = ds.valid_time.dt.month

    if variable_name is not None and variable_name in ds:
        if not is_daily and variable_name == "tprate":
            ds[variable_name] = ds[variable_name] * 1000 * 86400
            if 'GRIB_units' in ds[variable_name].attrs:
                ds[variable_name].attrs['GRIB_units'] = 'mm/day'
                
        elif is_daily and variable_name == "tp":
            first_day = ds[variable_name].isel(lead_time=slice(0, 1))
            following_days = ds[variable_name].diff(dim='lead_time')
            ds[variable_name] = xr.concat([first_day, following_days], dim='lead_time') * 1000
            ds[variable_name].attrs['units'] = 'mm'
            if 'GRIB_units' in ds[variable_name].attrs:
                ds[variable_name].attrs['GRIB_units'] = 'mm'

    return ds


def add_coords_to_daily_SEAS5(ds: xr.Dataset) -> xr.Dataset:
    """
    Transforms a SEAS5 dataset by stacking temporal dimensions and calculating 
    monthly lead times.

    Parameters
    ----------
    ds : xr.Dataset
        The input SEAS5 dataset with `init_time` and `lead_time`.

    Returns
    -------
    xr.Dataset
        The transformed dataset with a single `forecast_time` dimension.
    """
    ds = ds.stack(forecast_time=("init_time", "lead_time"))
    ds = ds.set_index(forecast_time="valid_time")
    ds["forecast_time"].attrs = {
        "standard_name": "valid_time",
        "long_name": "validity time of the forecast",
    }

    if "target_month" in ds.coords:
        ds["target_month"].attrs = {
            "long_name": "month of the forecast",
            "units": "month"
        }

    init_year = ds["init_time"].dt.year
    init_month = ds["init_time"].dt.month
    valid_year = ds["forecast_time"].dt.year
    valid_month = ds["forecast_time"].dt.month
    monthly_lead_time = (valid_year - init_year) * 12 + (valid_month - init_month)
    
    ds = ds.assign_coords(monthly_lead_time=monthly_lead_time)
    ds["monthly_lead_time"].attrs = {
        "standard_name": "forecast_period",
        "long_name": "lead time in months",
        "units": "months",
    }
    
    return ds


def process_climate_variable(
    variable_name: str, 
    suffixes: List[str], 
    raw_data_folder: str, 
    target_month: int = 6, 
    start_year: int = 1992, 
    max_ensemble_members: int = 25
) -> xr.Dataset:
    """
    Orchestrate the loading, standardization, and flattening of SEAS5 datasets 
    for a specific variable across multiple temporal blocks.

    This function automatically stacks dimensions and renames the final temporal 
    coordinate to 'id' to resolve indexing conflicts during Machine Learning modeling.

    Parameters
    ----------
    variable_name : str
        The short name of the variable to process (e.g., 'mslp', 'tp', 'z500', 'q850').
        Note: For MSL, the file might be named 'mslp' while the internal variable is 'msl'.
    suffixes : List[str]
        A list of string suffixes representing the file chunks (e.g., ['1992', '1993'] 
        or ['1992_1996', '1997_2001']).
    raw_data_folder : str
        Path to the directory containing the raw NetCDF files.
    target_month : int, optional
        The specific month integer to filter the forecast time. Default is 6 (June).
    start_year : int, optional
        The year threshold to filter the forecast time (exclusive). Default is 1992.
    max_ensemble_members : int, optional
        The maximum number of ensemble members to slice. Default is 25.

    Returns
    -------
    xr.Dataset
        A fully processed, flattened, and sorted Dataset ready for feature extraction.
    """
    datasets_list = []
    coords_to_remove = ['target_month', 'lead_time', 'monthly_lead_time']

    for suffix in suffixes:
        print(f"Processing {variable_name} for suffix {suffix}...")
        
        # Accommodate naming differences (e.g., file is SEAS5_mslp_... but variable is msl)
        file_prefix = "mslp" if variable_name == "msl" else variable_name
        file_path = f"{raw_data_folder}/SEAS5_{file_prefix}_{suffix}.nc"
        
        try:
            ds_block = xr.open_dataset(file_path)
            ds_block = standardize_SEAS5(ds_block, variable_name=variable_name)
            ds_block = add_coords_to_daily_SEAS5(ds_block)
            
            # Subsetting
            ds_block = ds_block.isel(ensemble_member=slice(0, max_ensemble_members))
            ds_block = ds_block.sel(forecast_time=(ds_block['forecast_time'].dt.month == target_month))
            ds_block = ds_block.sel(forecast_time=(ds_block['forecast_time'].dt.year > start_year))
            
            # Cleaning
            ds_block = ds_block.drop_vars(coords_to_remove, errors='ignore')
            datasets_list.append(ds_block)
            
            # Free memory
            del ds_block
            gc.collect()
            
        except FileNotFoundError:
            warnings.warn(f"File {file_path} not found. Skipping this block.")
            continue

    if not datasets_list:
        raise ValueError(f"No valid data blocks loaded for variable {variable_name}.")

    # 1. Concatenate all loaded blocks
    ds_concat = xr.concat(datasets_list, dim='forecast_time')

    # 2. Stack dimensions 'forecast_time' and 'ensemble_member'
    ds_stacked = ds_concat.stack(sample=['forecast_time', 'ensemble_member'])

    # 3. Hierarchical sorting
    ds_sorted = ds_stacked.sortby(['init_time', 'forecast_time', 'ensemble_member'])

    # 4. Convert MultiIndex to flat coordinates
    ds_flat = ds_sorted.reset_index('sample')

    # 5. Rename to 'id' and assign numerical sequence
    ds_final = ds_flat.rename({'sample': 'id'})
    ds_final['id'] = np.arange(len(ds_final['id']))

    # 6. Final cleanup: drop obsolete coordinates post-sorting
    ds_final = ds_final.drop_vars(
        ['init_time', 'ensemble_member'], 
        errors='ignore'
    )

    return ds_final



def extract_pot_target(
    precip_ds: xr.Dataset, 
    var_name: str, 
    lon_target: float, 
    lat_target: float, 
    percentile_threshold: float = 95.0
) -> Tuple[pd.DataFrame, float]:
    """
    Extracts the time series for a single grid point and applies the Peaks Over 
    Threshold (POT) method to generate a binary target for classification.

    Parameters
    ----------
    precip_ds : xr.Dataset
        The xarray Dataset containing the precipitation data. It is expected to be
        flattened along an 'id' dimension.
    var_name : str
        The name of the precipitation variable within the dataset (e.g., 'tp').
    lon_target : float
        The longitude coordinate of the target grid point.
    lat_target : float
        The latitude coordinate of the target grid point.
    percentile_threshold : float, optional
        The percentile (0-100) to use as the threshold for determining extremes.
        Default is 95.0.

    Returns
    -------
    Tuple[pd.DataFrame, float]
        A tuple containing:
        - df_target (pd.DataFrame): A dataframe containing the 'id' and the binary 
          'target' (1 for extreme, 0 otherwise).
        - threshold_value (float): The physical precipitation value corresponding 
          to the computed threshold.
    """
    print(f"Extracting data for coordinates: Lon {lon_target}, Lat {lat_target}...")
    
    # 1. Select the grid point using the nearest neighbor method
    tp_point: xr.DataArray = precip_ds[var_name].sel(lon=lon_target, lat=lat_target, method='nearest')
    
    # Ensure data is loaded into memory as a numpy array for faster computation
    tp_values: np.ndarray = tp_point.values
    
    # 2. Calculate the threshold based on the percentile over the entire distribution
    threshold_value: float = float(np.nanpercentile(tp_values, percentile_threshold))
    print(f"Extreme threshold calculated ({percentile_threshold}th percentile): {threshold_value:.4f}")
    
    # 3. Apply POT: 1 if extreme, 0 otherwise (np.where handles NaNs safely)
    is_extreme: np.ndarray = np.where(tp_values > threshold_value, 1, 0)
    
    # 4. Insert into a clean DataFrame ready for merging with PCs
    df_target = pd.DataFrame({
        'id': precip_ds['id'].values,
        'target': is_extreme
    })
    
    return df_target, threshold_value


def preprocess_pressure_levels(ds: xr.Dataset) -> xr.Dataset:
    """
    Checks for the presence of a 'pressure_level' dimension/coordinate 
    and removes it by squeezing.
    
    Parameters
    ----------
    ds : xr.Dataset
        Input dataset, potentially containing a pressure level dimension.
        
    Returns
    -------
    xr.Dataset
        Dataset squeezed along the pressure dimension (if it existed).
    """
    if 'pressure_level' in ds.dims or 'pressure_level' in ds.coords:
        return ds.squeeze('pressure_level').drop_vars('pressure_level', errors='ignore')
    return ds


def compute_daily_anomalies(ds: xr.Dataset, var_name: str) -> xr.Dataset:
    """
    Calculates anomalies relative to the daily climatology without altering 
    the dimensional structure of the dataset.
    """
    print(f"Calculating daily anomalies for variable: {var_name}")
    
    ds = ds.load()
    
    day_groups = ds['forecast_time'].dt.day
    
    daily_clim = ds[var_name].groupby(day_groups).mean(dim='id')
    
    da_anomalies = ds[var_name].groupby(day_groups) - daily_clim
    
    ds_anomalies = ds.copy()
    
    ds_anomalies[var_name] = da_anomalies.transpose(*ds[var_name].dims).drop_vars('day', errors='ignore')
    
    return ds_anomalies


def perform_pca_workflow(
    ds_anom: xr.Dataset, 
    var_name: str, 
    n_components: float = 0.90,
    random_state: int = 42
) -> Tuple[pd.DataFrame, xr.DataArray, pd.Series]:
    """
    Executes spatially weighted PCA and structures the outputs for Machine Learning.

    Parameters
    ----------
    ds_anom : xr.Dataset
        Dataset containing the anomalies.
    var_name : str
        The internal variable name to process.
    n_components : float, optional
        The percentage of variance to explain (if < 1.0) or the exact number of 
        components (if >= 1). Default is 0.90.
    random_state : int, optional
        Seed for reproducibility. Default is 42.

    Returns
    -------
    Tuple[pd.DataFrame, xr.DataArray, pd.Series]
        - pc_df: DataFrame containing normalized PCs, indexed by 'id'.
        - eofs_da: DataArray containing un-weighted EOFs in physical space.
        - explained_var: Series containing the explained variance ratio per component.
    """
    print(f"Starting PCA workflow for {var_name}...")
    da = ds_anom[var_name]
    
    # 1. Spatial weights (Square root of cosine of latitude)
    lat_rad = np.deg2rad(da['lat'])
    weights = np.sqrt(np.cos(lat_rad))
    da_weighted = da * weights
    
    # 2. Spatial stacking (lat, lon) -> features
    da_flat = da_weighted.stack(features=('lat', 'lon')).transpose('id', 'features')
    
    # 3. PCA Fitting
    print("Fitting PCA model...")
    pca = PCA(n_components=n_components, random_state=random_state)
    PC_raw = pca.fit_transform(da_flat.values) 
    EOF_raw = pca.components_ 
    
    # 4. PC Normalization (Unit variance for Random Forest)
    pc_std = np.sqrt(pca.explained_variance_)
    pcs_norm = PC_raw / pc_std 
    
    # 5. Data Structure 1: Principal Components DataFrame
    # Note: Prefixing with var_name to avoid column clashes during the final merge
    pc_cols = [f'{var_name}_PC{i+1}' for i in range(PC_raw.shape[1])]
    pc_df = pd.DataFrame(pcs_norm, columns=pc_cols, index=da['id'].values)
    pc_df.index.name = 'id'
    
    # Retain temporal coordinate for train/test splitting
    pc_df['forecast_time'] = da['forecast_time'].values 
    
    # 6. Reconstructing EOFs to physical units
    eofs_physical = EOF_raw * pc_std[:, np.newaxis]
    
    eofs_da_weighted = xr.DataArray(
        eofs_physical,
        dims=['mode', 'features'],
        coords={
            'mode': np.arange(1, pca.n_components_ + 1),
            'features': da_flat.coords['features']
        }
    ).unstack('features')
    
    # 7. Data Structure 2: "Un-weighted" EOFs Xarray
    eofs_da = eofs_da_weighted / weights
    
    # 8. Data Structure 3: Explained Variance
    explained_var = pd.Series(
        pca.explained_variance_ratio_, 
        index=pc_cols, 
        name="Explained Variance Ratio"
    )
    
    print(f"PCA completed. Retained components: {pca.n_components_}")
    return pc_df, eofs_da, explained_var