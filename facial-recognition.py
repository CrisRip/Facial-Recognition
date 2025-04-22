import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from keras.preprocessing.image import ImageDataGenerator #Para generar datos sintéticos
from keras import optimizers #Para usar los optimizadores(cálculo de gradiente descendente)
from keras.models import Sequential
from keras.layers import Dense , Dropout, Flatten, Activation
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import to_categorical, load_img, img_to_array #Para convertir etiquetas a one-hot
from keras.models import load_model #Para cargar el modelo
import numpy as np
import cv2 #Para cargar y mostrar imágenes

'''
entrenamiento = "EI_XD/archive/archive(6)/train" #Ruta del archivo de entrenamiento
validacion = "EI_XD/archive/archive(6)/validation" #Ruta del archivo de validación

#Deinir los hiperparametros de la arquitectura CNN
epocas = 15 #Número de épocas
altura, anchura = 50, 50 #Tamaño de la imagen

batch_size = 350 #Tamaño del batch (número de imágenes por iteración)
pasos = 100

#Defirnir hiperparametros de la red neuronal convolucional
kernel1 = 32
kernelsize = (3, 3) #Tamaño del kernel
kernel2 = 64
kernel2size = (3, 3) #Tamaño del kernel
size_pooling = (3, 3) #Tamaño del pooling

clases = 7 #Número de clases (emociones)

#Generar datos sintéticos para el entrenamiento y validación
entrenar = ImageDataGenerator(rescale=1./255, #Normalizar los datos
                              zoom_range=0.2, #Zoom
                              horizontal_flip=True, #Voltear horizontalmente
                            )
validar = ImageDataGenerator(rescale=1./255) #Normalizar los datos

#Leemos las imagenes de entrenamiento y validación
imagenes_entrenamiento = entrenar.flow_from_directory(entrenamiento, #Ruta del archivo de entrenamiento
                                                      target_size=(altura, anchura), #Tamaño de la imagen
                                                      batch_size=batch_size, #Tamaño del batch (número de imágenes por iteración)
                                                      class_mode='categorical') #Tipo de clase (categoría)

imagenes_validacion = validar.flow_from_directory(validacion, #Ruta del archivo de validación
                                                  target_size=(altura, anchura), #Tamaño de la imagen
                                                  batch_size=batch_size, #Tamaño del batch (número de imágenes por iteración)
                                                    class_mode='categorical') #Tipo de clase (categoría)

#Contruir el arquitectura de la red neuronal convolucional

ModelCNN = Sequential() #Modelo secuencial

#Determinar las capas convolucionales
ModelCNN.add(Convolution2D(kernel1, kernelsize, input_shape=(altura, anchura, 3), activation='relu')) #Capa convolucional
#Agregar el submuestreo
ModelCNN.add(MaxPooling2D(pool_size=size_pooling)) #Capa de max pooling

#Segunda capa convolucional
ModelCNN.add(Convolution2D(kernel2, kernel2size, activation='relu')) #Capa convolucional
ModelCNN.add(MaxPooling2D(pool_size=size_pooling)) #Capa de max pooling
ModelCNN.add(Flatten()) #Aplanar la matriz 3D a 1D

#Conectar la MPL (Red Neuronal Multicapa)
#Primera capa oculta
ModelCNN.add(Dense(100, activation='relu')) #Capa oculta
#Segunda capa oculta  
ModelCNN.add(Dense(100, activation='relu')) #Capa oculta
#Capa de salida
ModelCNN.add(Dense(clases, activation='softmax')) #Capa de salida (8 clases)

#Establecer los parametros de entrenamiento
ModelCNN.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(), metrics=['accuracy','mse']) #Compilar el modelo

#Entrenar el modelo
ModelCNN.fit(imagenes_entrenamiento, validation_data=imagenes_validacion, epochs=epocas, validation_steps=pasos, verbose = 1)

#Guardar el modelo entrenado
ModelCNN.save('EI_XD/archive/model/modelo_facial_recognition.h5') #Guardar el modelo
ModelCNN.save_weights('EI_XD/archive/model/modelo_facial_recognition.weights.h5') #Guardar los pesos del modelo
'''
#Imagen a clasificar
img = "archive/archive(6)/test/happy/35856.png" #Ruta de la imagen a clasificar


altura, anchura = 50, 50 #Tamaño de la imagen
#Cargar el modelo, arquitectura y pesos

modelo = load_model('archive/model/modelo_facial_recognition.h5') #Cargar el modelo
modelo.load_weights('archive/model/modelo_facial_recognition.weights.h5') #Cargar los pesos del modelo

#Transformar la imagen a clasificar
imagen = load_img(img, target_size=(altura,anchura)) #Cargar la imagen
imagen = img_to_array(imagen) #Convertir la imagen a array
imagen = np.expand_dims(imagen, axis=0) #Expandir las dimensiones de la imagen

#Clasificar la imagen
prediccion = modelo.predict(imagen) #Predecir la clase de la imagen
print(prediccion) #Imprimir la predicción

max = np.argmax(prediccion) # Obtener la clase con mayor probabilidad
clases = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
if 0 <= max < len(clases):
    print(f"La imagen es de la clase: {clases[max]}")
else:
    print("La imagen no pertenece a ninguna clase")

# Cargar el modelo y pesos entrenados
modelo = load_model('archive/model/modelo_facial_recognition.h5')
modelo.load_weights('archive/model/modelo_facial_recognition.weights.h5')

# Definir las clases
clases = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


# Cargar el clasificador de rostro de OpenCV
detector_rostro = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Capturar video desde la webcam
video = cv2.VideoCapture(1)

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Convertir a escala de grises para detectar rostro
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostros = detector_rostro.detectMultiScale(gris, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in rostros:
        rostro = frame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (50, 50))
        rostro = img_to_array(rostro)
        rostro = np.expand_dims(rostro, axis=0)
        rostro = rostro / 255.0  # Normalización

        # Predecir la emoción
        prediccion = modelo.predict(rostro)
        emocion_index = np.argmax(prediccion)
        emocion = clases[emocion_index]

        # Dibujar el recuadro y etiqueta en la imagen
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emocion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Mostrar la imagen en tiempo real
    cv2.imshow('Detector de Emociones', frame)

    # Presionar 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()


