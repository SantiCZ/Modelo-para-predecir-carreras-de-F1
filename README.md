# Modelo-para-predecir-carreras-de-F1
Un modelo que en base a archivos .csv de la temporada anterior, predice el ganador de la carrera que se le diga al modelo (va en orden numerico, ej: ronda 3= GP Australia)
# 🏁 Predicción de Resultados en Fórmula 1 con LSTM 🧠🏎️

Este proyecto utiliza datos reales de carreras de Fórmula 1 y redes neuronales LSTM para predecir las probabilidades de victoria de los pilotos. Incluye herramientas para extraer datos, entrenar modelos y mostrar predicciones mediante una interfaz gráfica amigable.

## 📁 Estructura del Proyecto

```
├── exportar_carreras.py            # Exporta toda una temporada de carreras a CSV
├── carreras_a_csv.py               # Exporta una carrera específica a CSV
├── modelo_de_prediccion_f1.py      # LSTM + procesamiento y predicción
├── interfaz.py                     # Interfaz Tkinter para predicción
├── modelo_entrenado.keras          # Modelo entrenado en formato Keras
├── modelo_entrenado.h5             # Versión HDF5 del modelo entrenado
```

## 🚀 Requisitos

- Python 3.8+
- [FastF1](https://theoehrly.github.io/Fast-F1/)
- TensorFlow / Keras
- scikit-learn
- numpy
- pandas
- tkinter (incluido con Python en la mayoría de instalaciones)

Instalar dependencias:

```
pip install fastf1 tensorflow scikit-learn numpy pandas
```

## 🛠️ ¿Como usarlo?

### 1. Descargar datos

Se puede descargar datos de toda una temporada:

python exportar_carreras.py

O solo una carrera específica:

python carreras_a_csv.py

### 2. Entrenar el modelo

El archivo `modelo_de_prediccion_f1.py` ya contiene el código para procesar datos y entrenar un modelo LSTM. Ejecutarlo entrenará el modelo y lo guardará como `modelo_entrenado.keras` y `modelo_entrenado.h5`.

python modelo_de_prediccion_f1.py


### 3. Ejecutar interfaz gráfica

Lanza la GUI para predicción con:
python interfaz.py


Podrás ingresar el año y ronda de carrera para ver las probabilidades de victoria por piloto.

## 🔍 ¿Qué hace el modelo?

- Toma como entrada secuencias por piloto: tiempo de vuelta, posición, stint, compuesto de neumático y delta respecto al mejor tiempo.
- Utiliza una red LSTM para predecir la posición final esperada de cada piloto.
- La salida puede ser vista por piloto o por equipo (en la consola o GUI).

## 📦 Cache y Datos

Los datos se almacenan en:
- `f1_cache/` → Cache de FastF1
- `datos_csv/` → Archivos CSV exportados

## 📌 Notas

- Algunas carreras pueden fallar al cargar por falta de datos en FastF1.
- Recomendado usar el formato `.keras` para guardar modelos (el `.h5` está incluido por compatibilidad).

## 📤 Créditos

- Proyecto realizado con [FastF1](https://github.com/theOehrly/Fast-F1)
- Modelo basado en TensorFlow/Keras
- Interfaz construida con Tkinter
