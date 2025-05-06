import numpy as np
import json
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow.keras.models import Model, load_model  # type: ignore
from collections import Counter
from sklearn.manifold import TSNE

# Tamaño estándar al que se redimensionarán las imágenes
IMAGE_SIZE = (256, 256)

# Límite de imágenes a usar (si se desea usar solo un subconjunto)
LIMIT_IMAGES = None

# Número máximo de resultados similares a devolver por búsqueda
MAX_RESULTS = 5

# Ruta al modelo entrenado de autoencoder
model_path = "models\\training_1.h5"

# Diccionario que clasifica los subtipos de tumores en benignos y malignos
classifications = {
    'benign': ['adenosis', 'fibrodenoma', 'phyllodes_tumor', 'tubular_adenoma'],
    'malignant': ['ductal_carcinoma', 'lobular_carcinoma', 'mucinous_carcinoma', 'papillary_carcinoma']
}

# Etiquetas principales (benign, malignant)
labels_binary = classifications.keys()

# Lista plana con todas las subclases de tumores
labels_subclasses = [item for sublist in classifications.values() for item in sublist]

# Carga del modelo autoencoder entrenado
print("[INFO] loading autoencoder model...")
auto_encoder = load_model(model_path, compile=False)
auto_encoder.load_weights(model_path)

# Se extrae la parte del modelo encargada de codificar las imágenes (el encoder)
encoder = Model(inputs=auto_encoder.input, outputs=auto_encoder.get_layer("encoded").output)

# Función que calcula la distancia euclidiana entre dos vectores
def euclidean(a, b):
    return np.linalg.norm(a - b)

# Realiza la búsqueda por similitud usando la distancia euclidiana
def perform_search(query_features, indexed_train):
    retrieved = []
    for idx, feat in enumerate(indexed_train["features"]):
        distance = euclidean(query_features, feat)
        retrieved.append((distance, idx))
    # Devuelve los MAX_RESULTS con menor distancia
    return sorted(retrieved)[:MAX_RESULTS]

# Devuelve el elemento más común en una lista
def most_common(arr):
    return Counter(arr).most_common(1)[0][0]

# Predicción individual de una imagen por su ruta
def predict_file_by_path(path, magnificient=None, augmentation_function=None):
    if magnificient is None:
        feature_path = "models\\features\\binary\\training_1_all_binary_feature.json"
    else:    
        feature_path = f"models\\features\\binary\\training_1_{magnificient}_binary_feature.json"

    print("-"*100)
    print(feature_path)

    print("[INFO] load test image...")
    # Carga y preprocesamiento de la imagen
    image = cv2.imread(path)
    image = cv2.resize(image, IMAGE_SIZE)
    if augmentation_function is not None:
        image = augmentation_function(image)
    test_x = np.array([image]).astype("float32") / 255.0  # Normalización

    print("[INFO] encoding image...")
    # Se codifican las características de la imagen con el encoder
    features_retrieved = encoder.predict(test_x)

    # Carga del conjunto de características ya indexadas (entrenamiento)
    with open(feature_path) as f:
        training_indexed = json.load(f)

    # Se realiza la búsqueda de imágenes similares
    results = perform_search(features_retrieved[0], training_indexed)
    labels_ret = [training_indexed["labels"][r[1]] for r in results]
    label = most_common(labels_ret)

    # Asignación de color según clase: verde (benigno) o rojo (maligno)
    color = "green" if label in classifications["benign"] or label == "benign" else "red"
    return label, color

# Predicción para todas las imágenes de una carpeta
def predict_folder(folder_path, magnificient=None, augmentation_function=None):
    if magnificient is None:
        feature_path = "models\\features\\binary\\training_1_all_binary_feature.json"
    else:    
        feature_path = f"models\\features\\binary\\training_1_{magnificient}_binary_feature.json"

    dataset = []
    # Se recorre la carpeta y se agregan las rutas de las imágenes al dataset
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            dataset.append(os.path.join(root, f))
    
    if LIMIT_IMAGES is not None:
        dataset = dataset[:LIMIT_IMAGES]
    
    # Carga y preprocesamiento de las imágenes
    images = [cv2.resize(cv2.imread(img), IMAGE_SIZE) for img in dataset]
    test_x = np.array(images).astype("float32") / 255.0

    # Aplicar aumento de datos si se proporciona una función
    if augmentation_function is not None:
        augmented = []
        for image in test_x:
            augmented += augmentation_function(image)
        test_x = np.stack(augmented, axis=0)

    # Codificación de características
    features_retrieved = encoder.predict(test_x)

    # Carga del conjunto de características del entrenamiento
    with open(feature_path) as f:
        training_indexed = json.load(f)

    # Diccionario para contar predicciones
    res = dict(benign=0, malignant=0)

    # Se realiza búsqueda y clasificación para cada imagen
    for i in range(len(test_x)):
        queryFeatures = features_retrieved[i]
        results = perform_search(queryFeatures, training_indexed)
        labels_ret = [training_indexed["labels"][r[1]] for r in results]
        label = most_common(labels_ret)
        res[label] += 1
    return res

# Visualización de características con PCA
def visualize_features_pca(json_path, save=False, output_folder="outputs\\features\\pca"):
    if not os.path.exists(json_path):
        print("El archivo no existe.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)
        filename = os.path.basename(f.name)
        name_without_ext = os.path.splitext(filename)[0]

    features = np.array(data["features"])
    labels = np.array([str(lbl).strip() for lbl in data["labels"]])

    # Se aplica PCA si hay más de 2 características
    if len(features) > 2:
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(features)

        # Se crea el gráfico disperso por etiqueta
        for lbl in np.unique(labels):
            idxs = labels == lbl
            plt.scatter(reduced[idxs, 0], reduced[idxs, 1], label=lbl, alpha=0.6)

    # Guardar o mostrar la imagen
    if save:
        path_img = os.path.join(output_folder, name_without_ext + "_visualization.png")
        plt.legend(title="Etiquetas")
        plt.savefig(path_img)
        print(f"[INFO] Imagen guardada en: {path_img}")
    else:
        plt.show()
    plt.close()

# Visualización de características con t-SNE
def visualize_features_tsne(json_path, save=False, output_folder="outputs\\features\\tsne"):
    if not os.path.exists(json_path):
        print("El archivo no existe.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)
        filename = os.path.basename(f.name)
        name_without_ext = os.path.splitext(filename)[0]

    features = np.array(data["features"])
    labels = np.array([str(lbl).strip() for lbl in data["labels"]])

    print("[INFO] Reducción de dimensionalidad con t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    reduced = tsne.fit_transform(features)

    # Crear carpeta si no existe
    if save and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Graficar características por clase
    plt.figure(figsize=(7, 7))
    for lbl in np.unique(labels):
        idxs = labels == lbl
        plt.scatter(reduced[idxs, 0], reduced[idxs, 1], label=lbl, alpha=0.6)

    # Guardar o mostrar visualización
    if save:
        path_img = os.path.join(output_folder, name_without_ext + "_tsne_visualization.png")
        plt.legend(title="Etiquetas")
        plt.savefig(path_img)
        print(f"[INFO] Imagen guardada en: {path_img}")
    else:
        plt.show()
    plt.close()
