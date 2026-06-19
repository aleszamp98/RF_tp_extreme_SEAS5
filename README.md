# Extreme Precipitation Prediction in West Africa using SEAS5 and Random Forests

## Introduction
This repository contains the codebase for a Machine Learning project that investigates extreme precipitation events in the tropical context of West Africa (specifically Kumasi, Ghana). 

Following the **UNSEEN (UNprecedented Simulated Extremes using ENsembles)** methodology, this study leverages the vast dataset provided by the ECMWF SEAS5 ensemble seasonal forecasts to explore the climatic phase space. The main goal is to understand how much relevant information is contained within large-scale instantaneous synoptic fields (Mean Sea Level Pressure, 500 hPa Geopotential Height, and 850 hPa Specific Humidity) to trigger extreme precipitation at a specific grid point. 

To achieve this, we train a **Random Forest (RF) classifier** using a Peaks Over Threshold (POT) approach at the 95th percentile to identify extreme events. Daily anomalies of large-scale predictors are reduced using Principal Component Analysis (PCA) and fed into the model. The Random Forest algorithm was chosen for its ability to maintain physical interpretability and extract the relative feature importance of different synoptic patterns.

## Repository Structure

The project is organized as follows:

```text
├── config.yml                 # Master configuration file for parameters and paths
├── environment.yml            # Conda environment dependencies
├── README.md                  # Project documentation
├── scripts/                   # Core execution pipeline
│   ├── downloads/             # Scripts to fetch SEAS5 data from the CDS API
│   ├── 01_preprocessing.py    # Data cleaning and anomaly calculation
│   ├── 02_target_construction.py # POT extreme event labeling
│   ├── 03_features_construction.py # PCA application on synoptic fields
│   └── 04_rf_training.py      # Random Forest grid-search training and evaluation
├── src/                       # Custom Python modules
│   ├── data_helpers.py        # Utilities for handling NetCDF/xarray data
│   ├── ml_utils.py            # Machine Learning helper functions
│   └── visualization_helper.py# Plotting and mapping utilities
├── notebooks/                 # Jupyter Notebooks for visual analysis
│   ├── PCA.ipynb              # Visualization of EOFs and explained variance
│   ├── RF_results.ipynb       # F1-scores, Confusion Matrix, and Feature Importance
│   └── domain_visualization.ipynb # Maps of the study area and grid points
└── data/                      # (Generated locally) Raw and Processed datasets
└── models/                    # (Generated locally) Pickled trained models

```

## Installation & Setup

1. **Clone the repository:**
```bash
git clone <your-github-repo-url>
cd <repository-folder>

```


2. **Create the Conda Environment:**
The project requires specific scientific libraries (`xarray`, `cartopy`, `scikit-learn`, `cdsapi`, etc.). Install them automatically using the provided `environment.yml` file:
```bash
conda env create -f environment.yml

```


3. **Activate the Environment:**
```bash
conda activate extreme-precip-RF-env

```


4. **CDS API Configuration:**
To download the raw SEAS5 data, you need an active Climate Data Store (CDS) account. Ensure you have your `~/.cdsapirc` file correctly set up in your home directory with your personal API URL and Key.

## Execution Workflow

The pipeline is designed to be run sequentially using the scripts provided in the `scripts/` folder. All steps read parameters directly from `config.yml`.

1. **Download Data:** Run the scripts inside `scripts/downloads/` to retrieve the necessary SEAS5 variables (tp, msl, z500, q850) via the CDS API.
2. **Preprocessing (`01_preprocessing.py`):** Calculates the daily climatology across all ensemble members and years, and extracts the daily anomalies to isolate synoptic variability.
3. **Target Construction (`02_target_construction.py`):** Extracts precipitation data for the target location (e.g., Kumasi) and applies the Peaks Over Threshold (POT) mask to create binary extreme (1) vs. non-extreme (0) labels.
4. **Features Construction (`03_features_construction.py`):** Applies latitude-weighted PCA to the anomalous large-scale fields to reduce spatial dimensionality while preserving dominant synoptic modes.
5. **Model Training (`04_rf_training.py`):** Executes a Group K-Fold Cross-Validation, training a Random Forest Classifier on the PCA features. It optimizes for the $F_1$-score due to the severe class imbalance.

## Configuration Parameters (`config.yml`)

The `config.yml` file allows you to control the entire pipeline without modifying the code. Key adjustable parameters include:

* **`paths`**: Customize where raw data, processed outputs, and saved models are stored.
* **`preprocessing`**: Define the temporal scope of the analysis. For example, `target_month: 6` focuses the study on June, the core of the major rainy season in Northern Ghana. `max_ensemble_members: 25` defines the ensemble size.
* **`target`**:
* `active_location`: Switch between different coordinate pairs (currently set to `kumasi`).
* `pot_threshold_percentile`: Defines what constitutes an extreme event (default is `95`, meaning the 95th percentile of precipitation).


* **`feature_extraction`**:
* `explained_variance_ratio`: Controls the number of Principal Components retained (default is `0.99`, keeping 99% of the variance).


* **`ml_pipeline`**: Contains the hyperparameter space for the Grid Search optimization of the Random Forest:
* `n_estimators`: Number of trees in the forest.
* `max_depth`: Limits the depth of the trees to prevent overfitting.
* `min_samples_leaf`: Minimum samples required to be at a leaf node.
* `class_weight`: Strategy to penalize the misclassification of the minority class (e.g., `'balanced'`).



## Result Visualization (Notebooks)

Once the data has been processed and the model trained, you can explore the results interactively using the provided Jupyter Notebooks in the `notebooks/` directory:

* **`domain_visualization.ipynb`**: Provides spatial context, plotting the geographical domain, the target city (Kumasi), and the surrounding grid points.
* **`PCA.ipynb`**: Visualizes the un-weighted Empirical Orthogonal Functions (EOFs) reconstructed in the physical space to give a physical interpretation of the dominant atmospheric modes extracted during feature construction.
* **`RF_results.ipynb`**: Loads the trained model to display performance metrics (Precision, Recall, $F_1$-score), Confusion Matrices, and the relative importance of the PCA features used by the Random Forest, linking the machine learning classification back to physical atmospheric drivers.