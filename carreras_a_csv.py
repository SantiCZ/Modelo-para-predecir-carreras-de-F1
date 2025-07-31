import fastf1

fastf1.Cache.enable_cache('f1_cache')

def exportar_datos_vueltas(year, round_number, nombre_archivo):
    session = fastf1.get_session(year, round_number, 'R')
    session.load()

    laps = session.laps
    laps = laps[["Driver", "Team", "LapNumber", "LapTime", "Position", "Stint", "Compound", "TyreLife"]]
    laps = laps.dropna(subset=["LapTime"])

    # Convertimos LapTime a string para guardar
    laps["LapTime"] = laps["LapTime"].astype(str)

    laps.to_csv(nombre_archivo, index=False)
    print(f"Archivo exportado: {nombre_archivo}")

# Podes cambiar año y ronda segun quieras, solo asegurate que existan en la misma carpeta que este archivo
exportar_datos_vueltas(2023, 1, "carrera_2023_r1.csv")