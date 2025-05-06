import numpy as np
from tensorflow.keras.preprocessing import image  # type: ignore # Utilizado para cargar y preprocesar imágenes
from tensorflow.keras.models import load_model     # type: ignore # Carga modelos guardados (aunque en este caso se usan pesos)
import os
from train_cnn import build_cnn_model              # Función personalizada que construye la arquitectura del modelo

# Diccionario que asocia las clases con etiquetas textuales y colores
LABEL_MAP = {
    0: ["benign", "green"],   # Clase 0 → etiqueta "benign" y color "verde"
    1: ["malign", "red"]      # Clase 1 → etiqueta "malign" y color "rojo"
}

# Función para cargar el modelo CNN con sus pesos correspondientes según la ampliación (magnificient)
def load_trained_model_by_weights(magnificient, weights_dir="./saved_weights"):
    """
    Carga un modelo CNN y le aplica los pesos entrenados correspondientes según el parámetro 'magnificient'.
    
    Parámetros:
    - magnificient: una etiqueta que indica la ampliación (por ejemplo, 40x, 100x, etc.)
    - weights_dir: carpeta donde se encuentran guardados los archivos de pesos del modelo

    Retorna:
    - El modelo con los pesos cargados.
    """
    if magnificient is None:
        base_name = "modelo_cnn_all"  # Nombre base si no se especifica ampliación
    else:
        base_name = f"modelo_cnn_{magnificient}"  # Se construye el nombre base con la ampliación

    # Buscar en la carpeta de pesos un archivo que coincida con el nombre base
    for fname in os.listdir(weights_dir):
        if fname.startswith(base_name) and fname.endswith(".weights.h5"):
            print(f"Cargando pesos del modelo: {fname}")
            weight_path = os.path.join(weights_dir, fname)
            model = build_cnn_model()           # Se construye la arquitectura del modelo
            model.load_weights(weight_path)     # Se cargan los pesos
            return model                         # Se retorna el modelo listo

    # Si no se encuentra el archivo de pesos, lanzar error
    raise FileNotFoundError(f"No se encontró ningún modelo para: {base_name}")


# Función para preprocesar una imagen antes de pasarla al modelo
def preprocess_image(img_path, target_size=(128, 128)):
    """
    Carga una imagen, la redimensiona, normaliza y la transforma en un batch para el modelo.

    Parámetros:
    - img_path: ruta a la imagen.
    - target_size: tamaño al que se desea redimensionar (por defecto 128x128)

    Retorna:
    - Un array NumPy con la imagen preprocesada y lista para el modelo.
    """
    img = image.load_img(img_path, target_size=target_size)  # Cargar imagen y redimensionar
    img_array = image.img_to_array(img)                      # Convertir a array NumPy
    img_array = img_array / 255.0                            # Normalizar a rango [0, 1]
    img_array = np.expand_dims(img_array, axis=0)            # Expandir dimensiones para simular un batch
    return img_array


# Función que realiza la predicción de clase sobre una imagen
def predict_image_class(model, img_path):
    """
    Preprocesa una imagen, realiza la predicción con el modelo y retorna la clase y su color.

    Parámetros:
    - model: modelo CNN con los pesos cargados.
    - img_path: ruta de la imagen que se desea clasificar.

    Retorna:
    - Una lista con la etiqueta ("benign"/"malign") y su color ("green"/"red").
    """
    img_array = [preprocess_image(img_path)]  # Preprocesar la imagen y ponerla en una lista
    prediction = model.predict(img_array)     # Predecir con el modelo
    predicted_class = np.argmax(prediction, axis=1)[0]  # Obtener clase con mayor probabilidad
    return LABEL_MAP[predicted_class]         # Retornar la etiqueta y el color asociado


# Función principal que une carga del modelo y predicción
def predict_path_cnn(MAGNIFICIENT, image_path):
    """
    Ejecuta el pipeline completo: carga del modelo + predicción sobre imagen.

    Parámetros:
    - MAGNIFICIENT: tipo de ampliación usada para escoger los pesos correctos del modelo.
    - image_path: ruta de la imagen a analizar.

    Retorna:
    - Una lista con la clase y su color asociado (o imprime error si ocurre una excepción).
    """
    try:
        model = load_trained_model_by_weights(MAGNIFICIENT)  # Cargar el modelo correspondiente
        result = predict_image_class(model, image_path)       # Realizar predicción sobre la imagen
        return result
    except Exception as e:
        print(f"Error: {e}")  # Manejo básico de errores
