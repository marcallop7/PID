import os
import json
import numpy as np
import cv2
from tqdm import tqdm
from tensorflow.keras.models import Model, load_model  # type: ignore

def get_all_features_knn(MAGNIFICIENT=None):
    # Cargar el modelo y configurar el encoder
    model_path = "models\\training_1.h5"
    print("[INFO] Cargando el modelo...")
    auto_encoder = load_model(model_path, compile=False)
    encoder = Model(inputs=auto_encoder.input, outputs=auto_encoder.get_layer("encoded").output)

    if MAGNIFICIENT is not None:
        features1, labels1 = get_features_knn(encoder, "images\\binary_scenario\\test", MAGNIFICIENT)
        features2, labels2 = get_features_knn(encoder, "images\\binary_scenario\\train", MAGNIFICIENT)

        features = features1 + features2
        labels = labels1 + labels2
    else:
        features1, labels1 = get_features_knn(encoder, "images\\binary_scenario_merged\\test")
        features2, labels2 = get_features_knn(encoder, "images\\binary_scenario_merged\\train")

        features = features1 + features2
        labels = labels1 + labels2

    # Guardar las características en un archivo JSON
    output_data = {
        "features": features,
        "labels": labels
    }

    if MAGNIFICIENT is None:
        output_json_path = "./models/features/binary/training_1_all_binary_feature.json"
    else:
        output_json_path = f"./training_1_{MAGNIFICIENT}_binary_feature.json"
    with open(output_json_path, "w") as json_file:
        json.dump(output_data, json_file)

    print(f"[INFO] Características guardadas en: {output_json_path}")

def get_features_knn(encoder, base_path, magnification=None):
    image_dir = base_path if magnification is None else os.path.join(base_path, f"{magnification}X")

    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"[ERROR] La ruta {image_dir} no existe.")

    IMAGE_SIZE = (256, 256)
    features = []
    labels = []

    print(f"[INFO] Procesando imágenes en: {image_dir}")

    # Contar archivos antes de empezar
    all_image_paths = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_image_paths.append(os.path.join(root, file))

    if not all_image_paths:
        raise RuntimeError(f"[ERROR] No se encontraron imágenes en {image_dir}")

    for image_path in tqdm(all_image_paths, desc="Extrayendo características"):
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"[WARNING] No se pudo leer la imagen: {image_path}")
                continue
            image = cv2.resize(image, IMAGE_SIZE)
            image = np.array(image).astype("float32") / 255.0
            image = np.expand_dims(image, axis=0)

            encoded_features = encoder.predict(image, verbose=0)[0]
            features.append(encoded_features.tolist())

            label = os.path.basename(os.path.dirname(image_path))
            labels.append(label)

        except Exception as e:
            print(f"[ERROR] Fallo procesando {image_path}: {e}")

    return features, labels


if __name__ == "__main__":
    get_all_features_knn(40)
    get_all_features_knn(100)
    get_all_features_knn(200)
    get_all_features_knn(400)
    get_all_features_knn()