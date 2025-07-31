import fastf1
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense, Masking
from keras.preprocessing.sequence import pad_sequences
from collections import defaultdict
import pickle
from keras.models import load_model
fastf1.Cache.enable_cache('f1_cache')

# Codificador de neumáticos (soft, medium, hard, etc.)
compound_encoder = LabelEncoder()

def cargar_datos_vueltas_avanzado(year, round_number):
    session = fastf1.get_session(year, round_number, 'R')
    session.load()

    laps = session.laps
    best_lap_time = laps['LapTime'].dt.total_seconds().min()
    drivers = laps['Driver'].unique()

    datos_pilotos = []

    # Entrenamos el codificador con todos los compuestos posibles de esta carrera
    compounds = laps['Compound'].dropna().unique()
    compound_encoder.fit(compounds)

    for driver in drivers:
        piloto_laps = laps.pick_driver(driver).pick_quicklaps()
        if len(piloto_laps) < 5:
            continue

        piloto_laps = piloto_laps.sort_values(by='LapNumber')

        tiempos = piloto_laps['LapTime'].dt.total_seconds().values
        posiciones = piloto_laps['Position'].values
        stints = piloto_laps['Stint'].values
        compuestos = piloto_laps['Compound'].fillna('UNKNOWN').values
        delta_time = tiempos - best_lap_time

        stint_encoded = (stints != np.roll(stints, 1)).astype(int).cumsum()
        compound_encoded = compound_encoder.transform(compuestos)

        # Feature por vuelta: [tiempo, posición, stint, delta, compound]
        secuencia = np.stack([tiempos, posiciones, stint_encoded, delta_time, compound_encoded], axis=1)
        label = piloto_laps.iloc[-1]['Position'] - 1

        # al final del bucle por piloto
        datos_pilotos.append((secuencia, label, {
    'Driver': piloto_laps.iloc[-1]['Driver'],
    'Team': piloto_laps.iloc[-1]['Team']
}))
    return datos_pilotos

# Recolectamos datos
carreras = [(2023, 1), (2023, 2), (2023, 3), (2023, 4)]
X_data, y_data = [], []

for year, rnd in carreras:
    try:
        datos = cargar_datos_vueltas_avanzado(year, rnd)
        for seq, label, _ in datos:
            X_data.append(seq)
            y_data.append(label)

    except Exception as e:
        print(f"Error en {year} R{rnd}: {e}")

# Padding de secuencias
X_padded = pad_sequences(X_data, padding='post', dtype='float32', value=-1.0)
y_array = np.array(y_data)

def predecir_probabilidades_ganar(modelo, year, round_number, por_equipo=False):
    datos = cargar_datos_vueltas_avanzado(year, round_number)
    
    if not datos:
        print("No hay datos suficientes para esa carrera.")
        return

    secuencias, labels, pilotos, equipos = [], [], [], []

    for secuencia, label, info in datos:  # ahora info = {'Driver': ..., 'Team': ...}
        secuencias.append(secuencia)
        labels.append(label)
        pilotos.append(info['Driver'])
        equipos.append(info['Team'])

    X_pad = pad_sequences(secuencias, padding='post', dtype='float32', value=-1.0)
    predicciones = modelo.predict(X_pad)

    # Probabilidad de que el piloto termine en posicion 1
    probabilidades = predicciones[:, 0]  # indice 0 = posicion 1 (ganador)

    if por_equipo:
        equipo_prob = defaultdict(list)
        for prob, team in zip(probabilidades, equipos):
            equipo_prob[team].append(prob)

        # Promediamos por equipo
        equipo_final = {k: np.mean(v) for k, v in equipo_prob.items()}
        resultado = sorted(equipo_final.items(), key=lambda x: x[1], reverse=True)

        print(f"Probabilidad de victoria por equipo en {year} R{round_number}:\n")
        for team, prob in resultado:
            print(f"🏎️ {team}: {prob*100:.2f}%")

    else:
        resultado = sorted(zip(pilotos, probabilidades), key=lambda x: x[1], reverse=True)
        print(f" Probabilidad de victoria por piloto en {year} R{round_number}:\n")
        for piloto, prob in resultado:
            print(f"👤 {piloto}: {prob*100:.2f}%")

# Modelo LSTM
model = Sequential()
model.add(Masking(mask_value=-1.0, input_shape=(X_padded.shape[1], X_padded.shape[2])))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(64, activation='relu'))
model.add(Dense(20, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Entrenamiento
model.fit(X_padded, y_array, epochs=50, batch_size=8, validation_split=0.2)

model.save("modelo_entrenado.keras")# Guardar el modelo entrenado

modelo = load_model("modelo_entrenado.h5")


def predecir_carrera(year, round):
    try:
        datos = cargar_datos_vueltas_avanzado(year, round)  # [(input_seq, piloto)]
        predicciones = []
        for secuencia, piloto, _ in datos:
            sec_input = np.expand_dims(secuencia, axis=0)
            pred = modelo.predict(sec_input, verbose=0)
            posicion_predicha = np.argmin(pred) + 1  # porque indice 0 = P1
            predicciones.append((piloto, posicion_predicha))

        texto = f"Predicciones para {year} - Ronda {round}:\n"
        for piloto, compuesto in predicciones:
            texto += f"{piloto}: {compuesto}\n"
        return texto
    except Exception as e:
        return f"Error: {e}"