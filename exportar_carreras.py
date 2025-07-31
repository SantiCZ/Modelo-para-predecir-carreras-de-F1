import fastf1
#import pandas as pd . Importa pandas para manejar DataFrames, si es necesario
import os

cache_dir = 'f1_cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

fastf1.Cache.enable_cache(cache_dir)

def exportar_temporada_completa_csv(año, carpeta_destino="datos_csv"):
    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino)

    for ronda in range(1, 25):  # Ajusta si queres menos rondas
        try:
            session = fastf1.get_session(año, ronda, 'R')
            session.load()

            nombre_gran_premio = session.event['EventName'].replace(" ", "_").replace("/", "_")
            nombre_archivo = f"{carpeta_destino}/{año}_R{ronda:02d}_{nombre_gran_premio}.csv"

            laps = session.laps[[
                "Driver", "Team", "LapNumber", "LapTime", "Position",
                "Stint", "Compound", "TyreLife"
            ]]
            laps = laps.dropna(subset=["LapTime"])
            laps["LapTime"] = laps["LapTime"].astype(str)

            laps.to_csv(nombre_archivo, index=False)
            print(f"✅ Exportado: {nombre_archivo}")

        except Exception as e:
            print(f"⚠️ Error en {año} R{ronda}: {e}")

# Elegí el año de la temporada que querés exportar
exportar_temporada_completa_csv(2023)
