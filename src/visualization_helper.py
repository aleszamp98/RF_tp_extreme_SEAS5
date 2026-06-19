import math
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import joblib
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from typing import List, Optional, Union

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


def plot_grid_search_heatmap(grid_search):
    """
    Extracts results from a fitted GridSearchCV and plots a 2D heatmap 
    evaluating 'n_estimators' vs 'max_depth' while freezing other parameters 
    (like 'class_weight' and 'min_samples_leaf') at their best-found values.

    Parameters
    ----------
    grid_search : sklearn.model_selection.GridSearchCV
        The fitted GridSearchCV object containing cv_results_ and best_params_.
        
    Returns
    -------
    None
        Displays the matplotlib figure.
    """
    print("\n--- GENERATING GRID SEARCH HEATMAP ---")
    
    results = pd.DataFrame(grid_search.cv_results_)
    best_params = grid_search.best_params_
    
    # Filter results to fix all parameters except max_depth and n_estimators
    mask = pd.Series([True] * len(results))
    
    for param, value in best_params.items():
        if param not in ['max_depth', 'n_estimators']:
            param_col = f'param_{param}'
            mask &= (results[param_col].astype(str) == str(value))
            
    subset = results[mask].copy()
    
    if subset.empty:
        print("Warning: Could not isolate a 2D grid for the heatmap. Check your param_grid structure.")
        return

    # Handle None values in max_depth for plotting purposes
    subset['param_max_depth'] = subset['param_max_depth'].fillna('None').astype(str)
    
    # Create Pivot Table for the heatmap
    pivot_table = subset.pivot(
        index='param_max_depth', 
        columns='param_n_estimators', 
        values='mean_test_score'
    )
    
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(pivot_table.values, cmap='viridis', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(pivot_table.columns)))
    ax.set_yticks(np.arange(len(pivot_table.index)))
    ax.set_xticklabels(pivot_table.columns)
    ax.set_yticklabels(pivot_table.index)
    
    ax.set_xlabel('n_estimators', fontweight='bold')
    ax.set_ylabel('max_depth', fontweight='bold')
    
    # Add title dynamically reporting the fixed class_weight and min_samples_leaf
    fixed_cw = best_params.get('class_weight', 'Default')
    fixed_leaf = best_params.get('min_samples_leaf', 'Default')
    ax.set_title(f"GridSearch F1-Score Heatmap\n(Fixed class_weight: {fixed_cw} | min_samples_leaf: {fixed_leaf})", pad=15)
    
    # Add text annotations inside the heatmap cells
    for i in range(len(pivot_table.index)):
        for j in range(len(pivot_table.columns)):
            val = pivot_table.values[i, j]
            if not np.isnan(val):
                color = "black" if val > pivot_table.values.max() * 0.8 else "white"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontweight='bold')
                
    # Add colorbar
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label('Mean Test F1-Score')
    
    plt.tight_layout()
    plt.show()


def plot_grid_search_heatmap_from_csv(
    results_df: pd.DataFrame, 
    best_params: dict, 
    fixed_params: Optional[List[str]] = None
) -> None:
    """
    Plots a 2D heatmap of GridSearchCV F1-scores for 'max_depth' vs 'n_estimators', 
    freezing all other hyperparameters at their best values. The colormap is strictly 
    bounded between 0.0 and 1.0.

    Parameters
    ----------
    results_df : pandas.DataFrame
        The full cross-validation results dataframe (e.g., loaded from grid_search_all_results.csv).
    best_params : dict
        Dictionary of the best parameters found during the grid search.
    fixed_params : list of str, optional
        Specific parameters to freeze. If None, it freezes all parameters in best_params 
        except 'max_depth' and 'n_estimators'.

    Returns
    -------
    None
        Displays the matplotlib figure.
    """
    print("\n--- Generating Grid Search Heatmap ---")
    
    if fixed_params is None:
        fixed_params = [p for p in best_params.keys() if p not in ['max_depth', 'n_estimators']]
        
    mask = pd.Series([True] * len(results_df))
    
    # Freeze the secondary parameters to isolate the 2D grid
    for param in fixed_params:
        param_col = f'param_{param}'
        if param_col in results_df.columns:
            target_value = str(best_params[param])
            mask &= (results_df[param_col].astype(str) == target_value)
            
    subset = results_df[mask].copy()
    
    if subset.empty:
        print("Warning: Could not isolate a 2D grid. Check your parameter names and values.")
        return

    # Handle None values in max_depth for categorical plotting
    subset['param_max_depth'] = subset['param_max_depth'].fillna('None').astype(str)
    
    # Pivot the data for the heatmap
    pivot_table = subset.pivot(
        index='param_max_depth', 
        columns='param_n_estimators', 
        values='mean_test_score'
    )
    
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Enforce colormap limits from 0.0 to 1.0 using vmin and vmax
    cax = ax.imshow(pivot_table.values, cmap='viridis', aspect='auto', vmin=0.1, vmax=0.25)
    
    # Axis configuration
    ax.set_xticks(np.arange(len(pivot_table.columns)))
    ax.set_yticks(np.arange(len(pivot_table.index)))
    ax.set_xticklabels(pivot_table.columns)
    ax.set_yticklabels(pivot_table.index)
    
    ax.set_xlabel('n_estimators', fontweight='bold')
    ax.set_ylabel('max_depth', fontweight='bold')
    
    # Annotate cells with their F1-score
    for i in range(len(pivot_table.index)):
        for j in range(len(pivot_table.columns)):
            val = pivot_table.values[i, j]
            if not np.isnan(val):
                # Adjust text color for contrast against the heatmap background
                # Threshold set at 0.8 since max scale is strictly 1.0
                color = "black" if val > 0.8 else "white"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontweight='bold')
                
    # Add colorbar with explicit bounds
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label('Mean Test F1-Score')
    
    plt.tight_layout()
    plt.show()

def plot_feature_importances(
    model: RandomForestClassifier, 
    feature_names: Union[List[str], pd.Index], 
    top_n: int = 15
) -> None:
    """
    Extracts and plots the top N most important features from a fitted Random Forest.

    Parameters
    ----------
    model : sklearn.ensemble.RandomForestClassifier
        The fitted tree-based model.
    feature_names : list of str or pandas.Index
        The names of the features corresponding to the model's input columns.
    top_n : int, optional
        The number of top features to display. Default is 15.

    Returns
    -------
    None
        Displays the matplotlib figure.
    """
    print(f"\n--- Generating Top {top_n} Feature Importances ---")
    
    importances = model.feature_importances_
    
    # Sort indices in descending order of importance
    indices = np.argsort(importances)[::-1]
    
    # Select the top N features
    top_indices = indices[:top_n]
    top_importances = importances[top_indices]
    top_features = [feature_names[i] for i in top_indices]
    
    # Reverse order so the highest importance is at the top of the horizontal bar chart
    top_importances = top_importances[::-1]
    top_features = top_features[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.barh(np.arange(len(top_features)), top_importances, color='steelblue', edgecolor='black')
    
    ax.set_yticks(np.arange(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.set_xlabel('Gini Importance (Mean Decrease Impurity)')
    # ax.set_title('Feature Importances', fontweight='bold')
    
    # Add gridlines for readability
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()


def plot_custom_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    classes: List[str]
) -> None:
    """
    Computes and plots a stylized confusion matrix using only Matplotlib.
    Configured to display the X-axis on top, rotate Y-axis labels by 90 degrees,
    and include binary classification terminology (TP, FP, TN, FN) in the cells.

    Parameters
    ----------
    y_true : numpy.ndarray
        1D array of true target values.
    y_pred : numpy.ndarray
        1D array of predicted target values.
    classes : list of str
        Ordered list of class labels (e.g., ['Normal', 'Extreme']). 
        Assumes class 0 is Negative/Normal and class 1 is Positive/Extreme.

    Returns
    -------
    None
        Displays the matplotlib figure.
    """
    print("\n--- Generating Confusion Matrix ---")
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(7, 7))
    cax = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    
    # Move X-axis ticks and label to the top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    
    # Set X labels
    ax.set_xticklabels(classes, fontsize=11)
    
    # Set Y labels, rotated 90 degrees and centered vertically
    ax.set_yticklabels(classes, rotation=90, va='center', fontsize=11)
    
    ax.set_ylabel('True Label', fontweight='bold', labelpad=15)
    ax.set_xlabel('Predicted Label', fontweight='bold', labelpad=15)
    
    # Define terminology mapping for a standard 2x2 binary classification
    if cm.shape == (2, 2):
        term_labels = [
            ['True Negative', 'False Positive'],
            ['False Negative', 'True Positive']
        ]
    else:
        term_labels = None

    # Annotate cells with terminology and absolute numbers
    thresh = cm.max() / 2.
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            cell_value = cm[i, j]
            
            # Construct the display string dynamically
            if term_labels is not None:
                display_str = f"{term_labels[i][j]}\n\n{cell_value}"
            else:
                display_str = str(cell_value)
                
            ax.text(
                j, i, display_str,
                ha="center", va="center",
                color="white" if cell_value > thresh else "black",
                fontweight='bold', fontsize=11
            )
            
    # Add colorbar
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

def plot_custom_roc_curve(y_true: np.ndarray, y_prob: np.ndarray) -> None:
    """
    Computes and plots the Receiver Operating Characteristic (ROC) curve 
    and calculates the Area Under the Curve (AUC) using only Matplotlib.

    Parameters
    ----------
    y_true : numpy.ndarray
        1D array of true binary target values (0 or 1).
    y_prob : numpy.ndarray
        1D array of predicted probabilities for the positive class (class 1).

    Returns
    -------
    None
        Displays the matplotlib figure.
    """
    print("\n--- Generating ROC Curve ---")
    
    # Calculate False Positive Rate (FPR) and True Positive Rate (TPR)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    
    # Calculate Area Under the Curve (AUC)
    roc_auc = auc(fpr, tpr)
    
    # Initialize the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot the actual ROC curve
    ax.plot(
        fpr, tpr, 
        color='darkorange', 
        linewidth=2, 
        label=f'ROC curve (AUC = {roc_auc:.3f})'
    )
    
    # Plot the random chance baseline (diagonal line)
    ax.plot([0, 1], [0, 1], color='navy', linewidth=2, linestyle='--')
    
    # Enforce standard ROC boundaries
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    # Set labels and title
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate (Recall)')
    # ax.set_title('ROC Curve - Extreme Event Prediction', pad=10)
    
    # Add subtle grid matching the requested aesthetic
    ax.grid(True, linestyle='-', alpha=0.3)
    
    # Add legend to the lower right corner
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.show()