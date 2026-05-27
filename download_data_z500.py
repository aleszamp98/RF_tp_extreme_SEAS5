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

# 2. Credenziali CDS Esplicite
CDS_URL = "https://cds.climate.copernicus.eu/api"
CDS_KEY = "f6df6ab4-1533-4366-b110-042b9c50aa5c"

# 3. Parametri statici della richiesta
DATASET = "seasonal-original-pressure-levels"
MONTHS = ["01", "02", "03", "04", "05", "06", "11", "12"]
DAYS = ["01"]
AREA = [30, -30, -10, 20]

# Generazione sintetica della lista dei lead time (da 24 a 5160 con passo di 24)
LEADTIME_HOURS = [str(i) for i in range(24, 5161, 24)]

# 4. Definizione e creazione della cartella di output
OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def download_chunk(years_chunk):
    """Funzione worker per scaricare un blocco di anni specifico (biennio)."""
    start_year = years_chunk[0]
    # L'ultimo elemento sarà uguale al primo se c'è un anno dispari residuo (es. solo 2022)
    end_year = years_chunk[-1]
    
    # Nome del file: SEAS5_z500_{start_year}_{end_year}.nc
    filepath = OUTPUT_DIR / f"SEAS5_z500_{start_year}_{end_year}.nc"
    
    request = {
        "originating_centre": "ecmwf",
        "system": "5",
        "variable": ["geopotential"],
        "pressure_level": ["500"],
        "year": years_chunk,
        "month": MONTHS,
        "day": DAYS,
        "leadtime_hour": LEADTIME_HOURS,
        "data_format": "netcdf",
        "area": AREA
    }
    
    logging.info(f"Inizio download z500: periodo {start_year}-{end_year} -> {filepath}")
    
    try:
        client = cdsapi.Client(url=CDS_URL, key=CDS_KEY)
        client.retrieve(DATASET, request).download(str(filepath))
        logging.info(f"Download completato con successo: {filepath}")
    except Exception as e:
        logging.error(f"Errore durante il download del periodo {start_year}-{end_year}: {e}")

def main():
    # Generazione anni dal 1992 al 2022
    all_years = [str(year) for year in range(1992, 2023)]
    
    # Creazione dei bienni (liste di 2 anni)
    chunk_size = 2
    year_chunks = [all_years[i:i + chunk_size] for i in range(0, len(all_years), chunk_size)]
    
    logging.info(f"Avvio processo: {len(year_chunks)} blocchi da scaricare in '{OUTPUT_DIR}'.")
    
    # Esecuzione in parallelo (5 worker in contemporanea)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        executor.map(download_chunk, year_chunks)
        
    logging.info("Tutte le operazioni di download sono concluse.")

if __name__ == "__main__":
    main()