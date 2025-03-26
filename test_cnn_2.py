from collections import Counter
import os
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image

# Modifica el archivo .h5 para eliminar 'groups' (si es necesario)
with h5py.File('training_1.h5', 'r+') as f:
    model_config = f.attrs['model_config']  # Ya está decodificado, no hace falta .decode()
    model_config = model_config.replace('"groups": 1,', '')  # Eliminar 'groups': 1
    f.attrs.modify('model_config', model_config)  # Modificar el archivo

print("/" * 100)

# Cargar el modelo modificado
model = tf.keras.models.load_model('training_1.h5', compile=False)

# Guardar la arquitectura en JSON
model_json = model.to_json()

# Reconstruir el modelo desde la arquitectura
new_model = model_from_json(model_json)
new_model.load_weights('training_1.h5')

print("Modelo cargado y reconstruido correctamente")

# Función para preprocesar las imágenes
def preprocess_image(img_path, target_size=(256, 256)):  # Cambié el tamaño aquí
    try:
        img = Image.open(img_path)  # Usamos PIL para cargar la imagen
        img = img.resize(target_size)  # Redimensionar la imagen a (256, 256)
        img_array = np.array(img, dtype=np.float32)  # Convertir la imagen a array numpy de tipo float32
        img_array = np.expand_dims(img_array, axis=0)  # Añadir la dimensión del batch
        img_array /= 255.0  # Normalizar la imagen (ahora debería funcionar sin error)
        return img_array
    except Exception as e:
        print(f"Error al procesar {img_path}: {e}")
        return None

# Ruta de la carpeta con las imágenes
# folder_path = r"images\binary_scenario\test\400X\benign"
folder_path = r"images\binary_scenario\test\40X\malignant"


contador_total = {0: 0, 1: 0, 2: 0}

mapping = {0: "maligno", 1: 'otro', 2: "benigno"}


# Iterar sobre todos los archivos en la carpeta
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    
    if os.path.isfile(file_path):
        print(f"Procesando archivo: {filename}")
        
        # Preprocesar la imagen
        img = preprocess_image(file_path)

        if img is not None:
            # Realizar la predicción
            prediction = model.predict(img)

            # Aquí puedes procesar la predicción según el tipo de tarea
            predicted_class_map = np.argmax(prediction, axis=-1)  # Solo si es clasificación
            contador = Counter(predicted_class_map.flatten())
            print(contador)
            choice = max(contador, key=contador.get)
            contador_total[choice] += 1
            # plt.imshow(predicted_class_map[0], cmap='jet')  # Puedes cambiar el cmap según lo que desees
            # plt.colorbar()
            # plt.show()
            print(f"Predicción para {filename}: {mapping[choice]}")
        else:
            print(f"Error al procesar {filename}. No se realizó la predicción.")

print(contador_total)

mayormente = max(contador_total, key=contador_total.get)

print(mapping[mayormente])


