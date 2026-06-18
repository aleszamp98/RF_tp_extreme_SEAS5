# RF_tp_extreme_SEAS5
Random Forest algorithm applied to the detection of total precipitation extremes based on SEAS5 model output.


├── .gitignore                   <- File essenziale per non caricare i dati su GitHub
├── environment.yml              <- (o requirements.txt) Per la riproducibilità dell'ambiente
├── README.md                    <- Guida all'uso e spiegazione del flusso
├── config.yml                   <- Parametri centralizzati (es. grid search, path)
├── data/                        <- Cartella locale per i dati (NON pushare su Git)
│   ├── raw/                     <- File NetCDF grezzi scaricati dal CDS
│   └── processed/               <- File pronti per il training (CSV o Parquet)
├── src/                         <- La tua libreria di "funzioni di base"
│   ├── __init__.py
│   ├── data_helpers.py          <- Funzioni per il preprocessing e calcolo anomalie
│   └── ml_utils.py              <- Funzioni per PCA, metriche POT e Random Forest
├── scripts/                     <- Gli script eseguibili in sequenza
│   ├── 00_download_cds.py
│   ├── 01_preprocessing.py
│   ├── 02_target_construction.py
│   ├── 03_features_construction.py
│   └── 04_rf_training.py
└── notebooks/                   <- Notebook per l'esplorazione e la reportistica
    └── 01_results_analysis.ipynb