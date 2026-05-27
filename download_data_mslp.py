import cdsapi
import logging
import concurrent.futures
from pathlib import Path

# 1. Configurazione del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 2. Credenziali CDS
CDS_URL = "https://cds.climate.copernicus.eu/api"
CDS_KEY = "f6df6ab4-1533-4366-b110-042b9c50aa5c"

# 3. Parametri statici della richiesta
DATASET = "seasonal-original-single-levels"
MONTHS = ["01", "02", "03", "04", "05", "06", "11", "12"]
DAYS = ["01"]
AREA = [30, -30, -10, 20]

# Generazione sintetica della lista dei lead time
LEADTIME_HOURS = [str(i) for i in range(24, 5161, 24)]

# 4. Definizione e creazione della cartella di output
OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def download_year(year):
    """Funzione worker per scaricare un singolo anno."""
    
    # Nome del file aggiornato per riflettere l'anno singolo e la nuova variabile (mslp)
    filepath = OUTPUT_DIR / f"SEAS5_mslp_{year}.nc"
    
    request = {
        "originating_centre": "ecmwf",
        "system": "5",
        "variable": ["mean_sea_level_pressure"],
        "year": [year],  # Le API del CDS richiedono comunque una lista, anche per un solo elemento
        "month": MONTHS,
        "day": DAYS,
        "leadtime_hour": LEADTIME_HOURS,
        "data_format": "netcdf",
        "area": AREA
    }
    
    logging.info(f"Inizio download: anno {year} -> {filepath}")
    
    try:
        client = cdsapi.Client(url=CDS_URL, key=CDS_KEY)
        client.retrieve(DATASET, request).download(str(filepath))
        logging.info(f"Download completato con successo: {filepath}")
    except Exception as e:
        logging.error(f"Errore durante il download dell'anno {year}: {e}")

def main():
    # Generazione lista di singoli anni dal 1992 al 2022
    years = [str(year) for year in range(1992, 2023)]
    
    logging.info(f"Avvio del processo: {len(years)} anni da scaricare nella cartella '{OUTPUT_DIR}'.")
    
    # Esecuzione in parallelo (il limite di 5 worker è ideale per non saturare la coda del CDS)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        executor.map(download_year, years)
        
    logging.info("Tutte le operazioni di download sono concluse.")

if __name__ == "__main__":
    main()