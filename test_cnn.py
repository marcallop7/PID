import os
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf

# Directorio de las imágenes a predecir
folder_path = 'images/breast_malignant'  # Cambia esto por la ruta de tu carpeta de imágenes

# Cargar el modelo
model = tf.keras.models.load_model('model_improved.keras')

# Obtener las clases desde el generador de imágenes
# Asumimos que tienes acceso a la variable 'test_generator' o las clases de alguna forma
class_names = {0: "bening", 1: "malingn"}

# Recorrer todas las imágenes de la carpeta
for filename in os.listdir(folder_path):
    img_path = os.path.join(folder_path, filename)
    # Comprobar si es una imagen (aquí puedes añadir más extensiones si es necesario)
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        try:
            # Cargar la imagen y preprocesarla
            img = image.load_img(img_path, target_size=(64, 64))  # Redimensionar a 150x150
            img_array = image.img_to_array(img)  # Convertir la imagen a un array numpy
            img_array = np.expand_dims(img_array, axis=0)  # Añadir una dimensión extra para representar el batch

            # Normalizar la imagen (igual que durante el entrenamiento)
            img_array /= 255.0

            # Realizar la predicción
            predictions = model.predict(img_array)

            # Obtener la clase con la probabilidad más alta
            predicted_class = np.argmax(predictions, axis=1)

            # Obtener el nombre de la clase
            predicted_label = class_names[predicted_class[0]]

            # Imprimir el resultado
            print(f"Imagen: {filename} - Clase predicha: {predicted_label}")

        except Exception as e:
            print(f"Error al procesar la imagen {filename}: {e}")
