import math
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_scree(explained_variance: pd.Series, var_name: str, max_eofs: int = None):
    """
    Generate a scree plot showing individual and cumulative explained variance.

    Parameters
    ----------
    explained_variance : pd.Series
        Series containing the explained variance ratio (output from the PCA workflow).
    var_name : str
        Name of the target climate variable (e.g., 'msl', 'z500').
    max_eofs : int, optional
        Maximum number of Empirical Orthogonal Functions (EOFs) to display on the plot.
    """
    # Extract values and convert to percentages
    var_exp = explained_variance.values * 100
    
    # Limit the number of EOFs to display if specified
    if max_eofs is not None:
        var_exp = var_exp[:max_eofs]
        
    var_cum = np.cumsum(var_exp)
    n_components = len(var_exp)
    x_axis = np.arange(1, n_components + 1)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Individual variance bars
    ax1.bar(x_axis, var_exp, color='skyblue', edgecolor='black', alpha=0.7, label='Individual Variance (%)')
    ax1.set_xlabel('Principal Component Number (PC)')
    ax1.set_ylabel('Explained Variance (%)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Twin axis for cumulative variance line
    ax2 = ax1.twinx()
    ax2.plot(x_axis, var_cum, color='red', marker='o', linestyle='-', linewidth=2, label='Cumulative Variance (%)')
    ax2.set_ylabel('Cumulative Explained Variance (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title(f'Scree Plot: Explained Variance by EOFs ({var_name.upper()})', fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Merge legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')
    
    plt.show()


def plot_eof_maps(eofs_da: xr.DataArray, explained_variance: pd.Series, var_name: str, max_eofs: int = None):
    """
    Plot the spatial distribution maps of Empirical Orthogonal Functions (EOFs).
    
    Optimized to restrict coordinate labels to the outer boundaries (left column 
    for latitude, bottom row for longitude) to avoid cluttering and overlapping.

    Parameters
    ----------
    eofs_da : xr.DataArray
        DataArray containing the un-stacked spatial maps of the EOFs over the lat/lon grid.
    explained_variance : pd.Series
        Series containing the explained variance ratio per component.
    var_name : str
        Name of the climate variable used for titles and colorbar units.
    max_eofs : int, optional
        Maximum number of spatial maps to visualize.
    """
    total_modes = len(eofs_da['mode'])
    n_plots = min(max_eofs, total_modes) if max_eofs else total_modes
    
    var_exp = explained_variance.values * 100
    
    # Dynamic grid layout setup (fixed at 3 columns)
    cols = 3
    rows = math.ceil(n_plots / cols)
    
    fig, axes = plt.subplots(
        nrows=rows, ncols=cols, 
        figsize=(14, 4 * rows), 
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    
    # Standardize axes layout into a flat list
    if n_plots == 1: 
        axes = [axes]
    else: 
        axes = axes.flatten()
    
    # Determine symmetrical color limits across the plotted modes
    vmax = float(np.abs(eofs_da.isel(mode=slice(0, n_plots))).max().values)
    vmin = -vmax

    for i in range(n_plots):
        ax = axes[i]
        eof_data = eofs_da.sel(mode=i+1)
        
        # Calculate current grid row and column
        current_row = i // cols
        current_col = i % cols
        
        mesh = ax.pcolormesh(
            eof_data.lon, eof_data.lat, eof_data,
            transform=ccrs.PlateCarree(),
            cmap='RdBu_r', 
            vmin=vmin, vmax=vmax,
            shading='auto'
        )
        
        ax.coastlines(resolution='50m', linewidth=1)
        ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)
        
        # Clean gridline label configuration to handle overlapping
        gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.4)
        gl.top_labels = False
        gl.right_labels = False
        
        # Only show latitude labels on the leftmost column
        gl.left_labels = (current_col == 0)
        
        # Only show longitude labels on the bottom row, or on the last visible plots of a column
        gl.bottom_labels = (current_row == rows - 1) or (i + cols >= n_plots)
        
        ax.set_title(f'EOF {i+1} ({var_exp[i]:.1f}%)', fontweight='bold', fontsize=12)

    # Clean up empty subplots if the grid is not completely filled
    for j in range(n_plots, len(axes)):
        fig.delaxes(axes[j])

    # Global continuous horizontal colorbar at the bottom
    cbar_ax = fig.add_axes([0.15, 0.06 / rows, 0.7, 0.02 / rows if rows > 1 else 0.02]) 
    cbar = fig.colorbar(mesh, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(f'{var_name.upper()} Spatial Anomaly Weight [Physical Unit / Std Dev]', fontsize=12)

    plt.subplots_adjust(bottom=0.18 / rows, hspace=0.1, wspace=0.1)
    #fig.suptitle(f'Spatial Patterns of the Top {n_plots} EOFs - {var_name.upper()}', fontsize=16, y=0.98)
    plt.show()


def plot_pc_timeseries(pc_df: pd.DataFrame, var_name: str, max_eofs: int = None):
    """
    Plot amplitude time series for individual Principal Components (PCs)
    with a common Y-axis scale across all subplots.

    Parameters
    ----------
    pc_df : pd.DataFrame
        DataFrame containing the PC columns (prefixed with the variable name) and indices.
    var_name : str
        Name of the climate variable for identifying specific columns and setting titles.
    max_eofs : int, optional
        Maximum number of PC timelines to visualize.
    """
    # Isolate relevant PC columns containing the variable prefix
    pc_cols = [col for col in pc_df.columns if 'PC' in col and var_name.lower() in col.lower()]
    if not pc_cols:
        # Fallback if names are not prefixed
        pc_cols = [col for col in pc_df.columns if col.startswith('PC')]
        
    total_pcs = len(pc_cols)
    n_plots = min(max_eofs, total_pcs) if max_eofs else total_pcs
    
    cols_to_plot = pc_cols[:n_plots]
    y_min = pc_df[cols_to_plot].min().min()
    y_max = pc_df[cols_to_plot].max().max()
    

    margin = (y_max - y_min) * 0.05
    y_lim_min = y_min - margin
    y_lim_max = y_max + margin
    
    x_data = pc_df.index
    
    # 2-column layout for horizontal timeseries comparison
    cols = 2 if n_plots > 1 else 1
    rows = math.ceil(n_plots / cols)
    
    # Aggiunto 'sharey=True' in modo da condividere i tick dell'asse Y
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 3 * rows), sharex=True, sharey=True)
    
    if n_plots == 1: 
        axes = [axes]
    else: 
        axes = axes.flatten()
    
    for i in range(n_plots):
        ax = axes[i]
        col_name = pc_cols[i]
        
        # Clean label for subplot titles (extracting just "PC X")
        display_name = col_name.split('_')[-1] if '_' in col_name else col_name
        
        ax.plot(x_data, pc_df[col_name], color='tab:blue', linewidth=0.8, alpha=0.9)
        
        # Reference baseline line for anomaly mean
        ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        
        # --- NOVITÀ: Impostazione dei limiti asse Y comuni ---
        ax.set_ylim(y_lim_min, y_lim_max)
        
        ax.set_title(f'{display_name}', fontweight='bold', fontsize=12)
        
        # Mostro l'etichetta dell'asse Y solo per i grafici nella prima colonna (se sharey=True)
        if i % cols == 0:
            ax.set_ylabel('Amplitude (Std)')
        if (i == n_plots - 1) or (i == n_plots - 2): 
            ax.set_xlabel('Time Index')
            
        ax.grid(True, linestyle='--', alpha=0.4)
        
    # Clean up empty subplots
    for j in range(n_plots, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    #fig.suptitle(f'Principal Component Time Series - {var_name.upper()}', fontsize=16, y=1.02)
    plt.show()