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

# 3. Static request parameters
DATASET = "seasonal-original-single-levels"
MONTHS = ["01", "02", "03", "04", "05", "06", "11", "12"]
DAYS = ["01"]
AREA = [30, -30, -10, 20]

# Synthetically generate the list of lead times (from 24 to 5160 with a step of 24)
LEADTIME_HOURS = [str(i) for i in range(24, 5161, 24)]

# 4. Define and create output directory
OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def download_chunk(years_chunk):
    """Worker function to download a specific block of years (5-year chunks)."""
    start_year = years_chunk[0]
    end_year = years_chunk[-1]
    
    # Full file path inside the output directory
    filepath = OUTPUT_DIR / f"SEAS5_tp_{start_year}_{end_year}.nc"
    
    request = {
        "originating_centre": "ecmwf",
        "system": "5",
        "variable": ["total_precipitation"],
        "year": years_chunk,
        "month": MONTHS,
        "day": DAYS,
        "leadtime_hour": LEADTIME_HOURS,
        "data_format": "netcdf",
        "area": AREA
    }
    
    logging.info(f"Starting tp download: period {start_year}-{end_year} -> {filepath}")
    
    try:
        # Initialize the client with explicit URL and KEY for each thread
        client = cdsapi.Client(url=CDS_URL, key=CDS_KEY)
        # Convert Path to string for compatibility with the download() method
        client.retrieve(DATASET, request).download(str(filepath))
        logging.info(f"Download successfully completed: {filepath}")
    except Exception as e:
        logging.error(f"Error during download for period {start_year}-{end_year}: {e}")

def main():
    # Check if the user forgot to replace the placeholder key
    if CDS_KEY == "YOUR_CDS_KEY_HERE":
        logging.error("Please replace 'YOUR_CDS_KEY_HERE' with your actual CDS API key at the top of the script.")
        return

    # Generate years from 1992 to 2022
    all_years = [str(year) for year in range(1992, 2023)]
    
    # Create 5-year chunks
    chunk_size = 5
    year_chunks = [all_years[i:i + chunk_size] for i in range(0, len(all_years), chunk_size)]
    
    logging.info(f"Starting process: {len(year_chunks)} time chunks to download into directory '{OUTPUT_DIR}'.")
    
    # Parallel execution (1 worker is safest for standard CDS queue limits to avoid simultaneous request rejections)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        executor.map(download_chunk, year_chunks)
        
    logging.info("All download operations are completed.")

if __name__ == "__main__":
    main()