import time
import cdsapi
import logging
import concurrent.futures
from pathlib import Path

# 1. Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 2. Explicit CDS API Credentials 
CDS_URL = "https://cds.climate.copernicus.eu/api"
CDS_KEY = "YOUR_CDS_KEY_HERE"

# 3. Download configurations
YEARS = range(2010, 2023)
MAX_RETRIES = 3
MAX_WORKERS = 3  # Simultaneous requests

# 4. Static request parameters
DATASET = "seasonal-original-pressure-levels"
REQUEST_TEMPLATE = {
    "originating_centre": "ecmwf",
    "system": "5",
    "variable": [
        "u_component_of_wind",
        "v_component_of_wind"
    ],
    "pressure_level": ["850"],
    "month": [
        "01", "02", "03", "04", "05", "06",
        "07", "08", "09", "10", "11", "12"
    ],
    "day": ["01"],
    # Generates hours from 24 to 5160 inclusive with a step of 24
    "leadtime_hour": [str(h) for h in range(24, 5161, 24)],
    "data_format": "netcdf",
    "area": [30, -30, -10, 20]
}

# 5. Define and create output directory
OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def download_year(year):
    """Worker function to download combined wind data for a single year with retry logic."""
    # Copy template to avoid race conditions between threads
    request = REQUEST_TEMPLATE.copy()
    request["year"] = [str(year)]
    
    filepath = OUTPUT_DIR / f"SEAS5_WIND850_{year}.nc"
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logging.info(f"Starting wind850 download: year {year} (Attempt {attempt}/{MAX_RETRIES}) -> {filepath}")
            
            client = cdsapi.Client(url=CDS_URL, key=CDS_KEY)
            client.retrieve(DATASET, request).download(str(filepath))
            
            logging.info(f"Download successfully completed: {filepath}")
            return f"Year {year} successfully downloaded."
            
        except Exception as e:
            logging.error(f"Failed download for year {year} on attempt {attempt}: {e}")
            if attempt < MAX_RETRIES:
                logging.info(f"Waiting 5 seconds before retrying year {year}...")
                time.sleep(5)
            else:
                logging.critical(f"Aborted: Unable to download year {year} after {MAX_RETRIES} attempts.")
                return f"Year {year} FAILED."

def main():
    # Check if the user forgot to replace the placeholder key
    if CDS_KEY == "YOUR_CDS_KEY_HERE":
        logging.error("Please replace 'YOUR_CDS_KEY_HERE' with your actual CDS API key at the top of the script.")
        return

    logging.info(f"Starting process: {len(YEARS)} years to download into directory '{OUTPUT_DIR}'.")
    
    # Execute downloads in parallel using a thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all download tasks to the executor
        futures = {executor.submit(download_year, year): year for year in YEARS}
        
        # Monitor completion status as they finish
        for future in concurrent.futures.as_completed(futures):
            # Internal logging inside download_year already handles success/error statuses
            _ = future.result()
        
    logging.info("All download operations are completed.")

if __name__ == "__main__":
    main()