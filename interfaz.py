import tkinter as tk
from tkinter import ttk, messagebox
from keras.models import load_model
from modelo_de_prediccion_f1 import predecir_probabilidades_ganar, cargar_datos_vueltas_avanzado  # Asegúrate que esté en el mismo directorio o PYTHONPATH
import sys
import io

# Cargar modelo
modelo = load_model("modelo_entrenado.h5")

# --- Ventana principal ---
ventana = tk.Tk()
ventana.title("Predicción de Fórmula 1")
ventana.geometry("500x500")

# Función de predicción
def predecir():
    año = entrada_anio.get()
    ronda = entrada_ronda.get()

    if not año or not ronda:
        messagebox.showerror("Error", "Por favor, completa todos los campos.")
        return

    try:
        año = int(año)
        ronda = int(ronda)
    except ValueError:
        messagebox.showerror("Error", "Año y Ronda deben ser numeros.")
        return

    try:
        resultado.config(text="Cargando datos y prediciendo...")

        # Capturamos el print que genera la función
        buffer = io.StringIO()
        sys.stdout = buffer

        predecir_probabilidades_ganar(modelo, año, ronda, por_equipo=False)

        sys.stdout = sys.__stdout__  # Restaurar salida estándar
        salida = buffer.getvalue()

        resultado.config(text=salida)
    except Exception as e:
        sys.stdout = sys.__stdout__  # Restaurar en caso de error
        messagebox.showerror("Error", f"Error en la prediccion: {e}")

# --- Campos de entrada ---
tk.Label(ventana, text="Año de la carrera:").pack()
entrada_anio = ttk.Entry(ventana)
entrada_anio.pack()

tk.Label(ventana, text="Numero de ronda (ej. 1, 2, 3...):").pack()
entrada_ronda = ttk.Entry(ventana)
entrada_ronda.pack()

# --- Botón de predicción ---
ttk.Button(ventana, text="Predecir ganadores", command=predecir).pack(pady=10)

# --- Resultado ---
resultado = tk.Label(ventana, text="", justify="left", wraplength=480, font=("Arial", 10))
resultado.pack(pady=10)

# --- Ejecutar interfaz ---
ventana.mainloop()
