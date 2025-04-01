from tensorflow.keras.models import load_model, model_from_json # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from PIL import Image
import numpy as np
import h5py
from collections import Counter

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
    
ruta = 'models/training_1.h5'

contador_total = {0: 0, 1: 0, 2: 0}

mapping = {0: "maligno", 1: "otro", 2: "benigno"}
color = {0: "red", 1: "yellow", 2: "green"}

def analisis_ia_modelo_1(image_path):
    with h5py.File(ruta, 'r+') as f:
        model_config = f.attrs['model_config']  # Ya está decodificado, no hace falta .decode()
        model_config = model_config.replace('"groups": 1,', '')  # Eliminar 'groups': 1
        f.attrs.modify('model_config', model_config)  # Modificar el archivo
    
    model = load_model(ruta, compile=False)
    model_json = model.to_json()

    new_model = model_from_json(model_json)
    new_model.load_weights(ruta)

    img = preprocess_image(image_path)
    if img is not None:
        prediction = model.predict(img)
        predicted_class_map = np.argmax(prediction, axis=-1) 
        contador = Counter(predicted_class_map.flatten())
        choice = max(contador, key=contador.get)
        print(f"Predicción: {mapping[choice]}")
        return mapping[choice], color[choice]
    else:
        raise Exception("Imagen no cargada")
