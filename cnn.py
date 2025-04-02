from tensorflow.keras.models import Model, load_model, model_from_json # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from PIL import Image
import numpy as np
import h5py
import os
from collections import Counter
import matplotlib.pyplot as plt


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
    
ruta = 'models/training_1_40_binary_model.h5'

contador_total = {0: 0, 1: 0, 2: 0}

mapping = {0: "maligno", 1: "otro", 2: "benigno"}
color = {0: "red", 1: "yellow", 2: "green"}

def analisis_ia_modelo_1(image_path):
    ruta = 'models/training_1.h5'
    mapping = {0: "maligno", 1: "otro", 2: "benigno"}
    color = {0: "red", 1: "yellow", 2: "green"}
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
        print(img.shape)
        _, height, width, _ = img.shape
        size = height * width
        return f"{mapping[choice]} ({100*float(contador[choice])/(size):.2f}%)", color[choice]
    else:
        raise Exception("Imagen no cargada")


def analisis_ia_modelo_2(image_paths):
    ruta = 'models/training_1_40_binary_model.h5'
    mapping = {0: "maligno", 1: "benigno", 2: "otro"}
    color = {0: "red", 1: "green", 2: "yellow"}
    # mapping = {0: "maligno", 1: "otro", 2: "benigno"}
    # color = {0: "red", 1: "yellow", 2: "green"}
    
    auto_encoder = load_model(ruta, compile=False)
    # model_json = model.to_json()



    # new_model = model_from_json(model_json)
    auto_encoder.load_weights(ruta)

    encoder = Model(inputs=auto_encoder.input, outputs=auto_encoder.get_layer("encoded").output)


    res = dict()
    images = [(image_path, preprocess_image(image_path)) for image_path in image_paths]
    images = filter(lambda x: x[1] is not None, images)
    for img in images:
        features  = encoder.predict(img[1])
        print("-"*100)
        print(features)
        break
        # predicted_class_map = np.argmax(prediction, axis=-1) 
        # contador = Counter(predicted_class_map.flatten())
        # plt.imshow(predicted_class_map[0], cmap='jet')  # Puedes cambiar el cmap según lo que desees
        # plt.colorbar()
        # plt.show()        
        # choice = max(contador, key=contador.get)
        # _, height, width, _ = img[1].shape
        # size = height * width
        # print("-"*100)
        # print(f"{mapping[choice]} ({100*float(contador[choice])/(size):.2f}%)")
        # res[img[0]] = f"{mapping[choice]} ({100*float(contador[choice])/(size):.2f}%)", color[choice]

    
# Directorio de las imágenes a predecir
folder_path = r'other\histology_slides\breast\benign\SOB\adenosis\SOB_B_A_14-22549AB\40X'  # Cambia esto por la ruta de tu carpeta de imágenes

# Recorrer todas las imágenes de la carpeta
img_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]
analisis_ia_modelo_2(img_paths)

# for filename in os.listdir(folder_path):
#     img_path = os.path.join(folder_path, filename)
#     # Comprobar si es una imagen (aquí puedes añadir más extensiones si es necesario)
#     if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#         try:
#             analisis_ia_modelo_2(img_path)
#         except Exception as e:
#             print(f"Error al procesar la imagen {filename}: {e}")