from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np

def analisis_ia(image_path):
    model = load_model('model.keras')
    # Cargar la imagen
    img = image.load_img(image_path, target_size=(32, 32))

    # Convertir la imagen a un array numpy
    img_array = image.img_to_array(img)

    # Normalizar la imagen (escalarla de 0 a 1)
    img_array = img_array / 255.0

    # Añadir una dimensión extra para que sea compatible con la entrada del modelo (1, 32, 32, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # Realizar la predicción
    predictions = model.predict(img_array)
    clases = ["Maligno", "Benigno"]
    index_clase_predicha = np.argmax(predictions)
    clase_predicha = clases[index_clase_predicha]

    print(f'Clase predicha: {clase_predicha}')
    return clase_predicha, "red" if index_clase_predicha == 0 else "green"