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

# 2. CDS API Credentials 
CDS_URL = "https://cds.climate.copernicus.eu/api"
CDS_KEY = "YOUR_CDS_KEY_HERE"

# 3. Static request parameters
DATASET = "seasonal-original-single-levels"
MONTHS = ["01", "02", "03", "04", "05", "06", "11", "12"]
DAYS = ["01"]
AREA = [30, -30, -10, 20]

# Synthetically generate the list of lead times
LEADTIME_HOURS = [str(i) for i in range(24, 5161, 24)]

# 4. Define and create output directory
OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def download_year(year):
    """Worker function to download data for a single year."""
    
    # Filename updated to reflect the specific year and the variable (mslp)
    filepath = OUTPUT_DIR / f"SEAS5_mslp_{year}.nc"
    
    request = {
        "originating_centre": "ecmwf",
        "system": "5",
        "variable": ["mean_sea_level_pressure"],
        "year": [year],  # CDS API requires a list even for a single element
        "month": MONTHS,
        "day": DAYS,
        "leadtime_hour": LEADTIME_HOURS,
        "data_format": "netcdf",
        "area": AREA
    }
    
    logging.info(f"Starting download: year {year} -> {filepath}")
    
    try:
        # Pass the explicit credentials defined at the top of the script
        client = cdsapi.Client(url=CDS_URL, key=CDS_KEY)
        client.retrieve(DATASET, request).download(str(filepath))
        logging.info(f"Download successfully completed: {filepath}")
    except Exception as e:
        logging.error(f"Error during download for year {year}: {e}")

def main():
    # Check if the user forgot to replace the placeholder key
    if CDS_KEY == "YOUR_CDS_KEY_HERE":
        logging.error("Please replace 'YOUR_CDS_KEY_HERE' with your actual CDS API key at the top of the script.")
        return

    # Generate list of individual years from 1992 to 2022
    years = [str(year) for year in range(1992, 2023)]
    
    logging.info(f"Starting process: {len(years)} years to download into directory '{OUTPUT_DIR}'.")
    
    # Parallel execution (1 worker is safest for standard CDS queue limits)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        executor.map(download_year, years)
        
    logging.info("All download operations are completed.")

if __name__ == "__main__":
    main()