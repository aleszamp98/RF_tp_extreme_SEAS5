import os
import time
import cdsapi
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------- CONFIGURAZIONE ----------------
# Inserisci qui le tue credenziali API di Copernicus
URL_CDS = "https://cds.climate.copernicus.eu/api"
KEY_CDS = "bd254960-6dad-4263-aebc-18da7c37c806"

# Anni da scaricare e cartella di destinazione
ANNI = range(2010, 2023)
CARTELLA_DESTINAZIONE = "WIND_850"
MAX_TENTATIVI = 3
MAX_WORKERS = 3  # Numero di richieste in contemporanea

# Creazione della sottocartella se non esiste
os.makedirs(CARTELLA_DESTINAZIONE, exist_ok=True)

# Dataset e template di richiesta
dataset = "seasonal-original-pressure-levels"
request_template = {
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
    # Genera automaticamente la stringa da 24 a 5160 con step di 24 ore
    "leadtime_hour": [str(h) for h in range(24, 5161, 24)],
    "data_format": "netcdf",
    "area": [30, -30, -10, 20]
}

# Inizializza il client esplicitando URL e Key
client = cdsapi.Client(url=URL_CDS, key=KEY_CDS)

# ---------------- FUNZIONE DI DOWNLOAD ----------------
def scarica_anno(anno):
    """
    Funzione che esegue il download per un singolo anno gestendo retries ed errori.
    """
    request = request_template.copy()
    request["year"] = [str(anno)]
    
    file_path = os.path.join(CARTELLA_DESTINAZIONE, f"SEAS5_WIND850_{anno}.nc")
    
    for tentativo in range(1, MAX_TENTATIVI + 1):
        try:
            print(f"[INIZIO] Download in corso per l'anno {anno} (Tentativo {tentativo}/{MAX_TENTATIVI})...")
            # Richiesta e download
            client.retrieve(dataset, request, file_path)
            print(f"[SUCCESSO] Download completato: {file_path}")
            return f"Anno {anno} completato con successo."
            
        except Exception as e:
            print(f"[ERRORE] Fallito anno {anno} al tentativo {tentativo}: {e}")
            if tentativo < MAX_TENTATIVI:
                print(f"[RETRY] Attendo 5 secondi prima di riprovare per l'anno {anno}...")
                time.sleep(5)
            else:
                print(f"[ABORTITO] Impossibile scaricare l'anno {anno} dopo {MAX_TENTATIVI} tentativi. Passo avanti.")
                return f"Anno {anno} FALLITO."

# ---------------- ESECUZIONE PARALLELA ----------------
def main():
    print(f"--- Inizio processo di download per {len(ANNI)} anni ---")
    print(f"I file verranno salvati nella cartella: ./{CARTELLA_DESTINAZIONE}/")
    print(f"Step leadtime impostato a 24h (24, 48, 72, ..., 5160).")
    
    # Esegue 3 richieste in contemporanea
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(scarica_anno, anno): anno for anno in ANNI}
        
        for future in as_completed(futures):
            # Il log specifico (successo o fallimento) è gestito nei print della funzione
            future.result()
            
    print("--- Processo di download concluso! ---")

if __name__ == "__main__":
    main()
