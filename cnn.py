import numpy as np
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import os

# Diccionario para mapear clases a etiquetas y colores
LABEL_MAP = {
    0: ["benign", "green"],
    1: ["malign", "red"]
}

def load_trained_model(magnificient, models_dir="./"):
    magnificient = str(magnificient).lower()
    if magnificient == "all":
        base_name = "modelo_cnn_all"
    else:
        base_name = f"modelo_cnn_{magnificient}"

    # Buscar el primer archivo .h5 que coincida con el patrón
    for fname in os.listdir(models_dir):
        if fname.startswith(base_name) and fname.endswith(".h5"):
            print(f"Cargando modelo: {fname}")
            model_path = os.path.join(models_dir, fname)
            return load_model(model_path)

    raise FileNotFoundError(f"No se encontró ningún modelo para: {base_name}")


def preprocess_image(img_path, target_size=(128, 128)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_image_class(model, img_path):
    img_array = [preprocess_image(img_path)]
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return LABEL_MAP[predicted_class]

def predict_path_cnn(MAGNIFICIENT, image_path):
    try:
        model = load_trained_model(MAGNIFICIENT)
        result = predict_image_class(model, image_path)
        return result
    except Exception as e:
        print(f"Error: {e}")
