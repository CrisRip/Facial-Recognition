import cv2
import os
import numpy as np
from tkinter import *
from PIL import Image, ImageTk
from keras.models import load_model
from keras.utils import img_to_array, load_img

# --- Configuración del modelo ---
modelo = load_model('archive/model/modelo_facial_recognition.h5')
modelo.load_weights('archive/model/modelo_facial_recognition.weights.h5')

clases = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
altura, anchura = 50, 50

# --- Emociones del juego ---
emociones_objetivo = ["Happy", "Sad", "Surprise", "Angry"]
indice_actual = 0

def clasificar_emocion(img_path):
    imagen = load_img(img_path, target_size=(altura, anchura))
    imagen = img_to_array(imagen)
    imagen = np.expand_dims(imagen, axis=0)
    imagen = imagen / 255.0
    prediccion = modelo.predict(imagen)
    emocion_index = np.argmax(prediccion)
    return clases[emocion_index]

def capturar_y_validar_emocion():
    global indice_actual
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    cam.release()

    if ret:
        if not os.path.exists("capturas"):
            os.makedirs("capturas")

        path_foto = "capturas/emocion_actual.png"
        cv2.imwrite(path_foto, frame)

        emocion_detectada = clasificar_emocion(path_foto)
        emocion_esperada = emociones_objetivo[indice_actual]

        print(f"Esperado: {emocion_esperada} | Detectado: {emocion_detectada}")

        if emocion_detectada == emocion_esperada:
            resultado.set("¡Correcto! 😀")
            indice_actual += 1
            if indice_actual >= len(emociones_objetivo):
                emocion_label.config(text="¡Juego completado!")
                boton_captura.config(state=DISABLED)
                imagen_emocion_label.config(image="")
            else:
                actualizar_emocion()
        else:
            resultado.set(f"Incorrecto. Intenta mostrar: {emocion_esperada} 😅")
    else:
        resultado.set("Error al acceder a la cámara.")

def actualizar_emocion():
    emocion_esperada = emociones_objetivo[indice_actual]
    emocion_label.config(text=f"Imita esta emoción: {emocion_esperada}")

    ruta_imagen = f"emociones/{emocion_esperada}.jpg"
    if os.path.exists(ruta_imagen):
        img = Image.open(ruta_imagen)
        img = img.resize((150, 150))
        img = ImageTk.PhotoImage(img)
        imagen_emocion_label.config(image=img)
        imagen_emocion_label.image = img
    else:
        imagen_emocion_label.config(image="", text="Imagen no encontrada")

# --- Interfaz gráfica ---
ventana = Tk()
ventana.title("Juego de emociones")
ventana.geometry("500x500")
ventana.configure(bg="#202020")

# --- Estilo de etiquetas ---
fuente_titulo = ("Helvetica", 18, "bold")
fuente_normal = ("Helvetica", 14)

emocion_label = Label(ventana, text="", font=fuente_titulo, fg="white", bg="#202020")
emocion_label.pack(pady=10)

imagen_emocion_label = Label(ventana, bg="#202020")
imagen_emocion_label.pack(pady=10)

boton_captura = Button(ventana, text="📸 Capturar emoción", command=capturar_y_validar_emocion, font=fuente_normal, bg="#FF5722", fg="white", padx=10, pady=5)
boton_captura.pack(pady=10)

resultado = StringVar()
resultado_label = Label(ventana, textvariable=resultado, font=fuente_normal, fg="#00E676", bg="#202020")
resultado_label.pack(pady=10)

actualizar_emocion()
ventana.mainloop()
