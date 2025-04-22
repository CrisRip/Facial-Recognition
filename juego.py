import tkinter as tk
from tkinter import messagebox
import subprocess
import random
import time
from PIL import Image, ImageTk

# Lista de emociones con imágenes asociadas
emociones = {

    "Happy": "archive/archive(6)/test/happy/32305.png",
}

def mostrar_emocion():
    emocion_actual = random.choice(list(emociones.keys()))
    ruta_imagen = emociones[emocion_actual]

    # Mostrar la emoción seleccionada
    imagen = Image.open(ruta_imagen)
    imagen = imagen.resize((150, 150))
    foto = ImageTk.PhotoImage(imagen)
    label_imagen.config(image=foto)
    label_imagen.image = foto
    label_emocion.config(text=f"¡Imita esta emoción!: {emocion_actual}")

    ventana.after(3000, lambda: iniciar_deteccion(emocion_actual))

def iniciar_deteccion(emocion_objetivo):
    messagebox.showinfo("¡Vamos!", f"¡Ahora imita la emoción: {emocion_objetivo}!")
    subprocess.run(["python", "facial-recognition.py"])

# Crear la ventana principal
ventana = tk.Tk()
ventana.title("Fase 2 - Juego de Emociones")
ventana.geometry("600x500")
ventana.configure(bg="#222831")

# Título
titulo = tk.Label(ventana, text="🎯 ¡Imita la Emoción!", font=("Helvetica", 20, "bold"), fg="#eeeeee", bg="#222831")
titulo.pack(pady=20)

label_emocion = tk.Label(ventana, text="", font=("Helvetica", 16), fg="#eeeeee", bg="#222831")
label_emocion.pack(pady=10)

# Imagen de la emoción
label_imagen = tk.Label(ventana, bg="#222831")
label_imagen.pack(pady=10)

# Botón para mostrar emoción
boton_jugar = tk.Button(ventana, text="Mostrar emoción", font=("Helvetica", 14, "bold"),
                        bg="#00adb5", fg="white", padx=20, pady=10, command=mostrar_emocion)
boton_jugar.pack(pady=30)

ventana.mainloop()
