import numpy as np
import json
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow.keras.models import Model, load_model # type: ignore
from collections import Counter
from sklearn.manifold import TSNE

IMAGE_SIZE = (256, 256)
LIMIT_IMAGES = None


model_path = "models\\training_1.h5"

# Corrección de nombre
classifications = {
    'benign': ['adenosis', 'fibrodenoma', 'phyllodes_tumor', 'tubular_adenoma'],
    'malignant': ['ductal_carcinoma', 'lobular_carcinoma', 'mucinous_carcinoma', 'papillary_carcinoma']
}

labels_binary = classifications.keys()
labels_subclasses = [item for sublist in classifications.values() for item in sublist]

print("[INFO] loading autoencoder model...")
auto_encoder = load_model(model_path, compile=False)
auto_encoder.load_weights(model_path)
encoder = Model(inputs=auto_encoder.input, outputs=auto_encoder.get_layer("encoded").output)

def euclidean(a, b):
    return np.linalg.norm(a - b)

def perform_search(query_features, indexed_train, max_results=5):
    retrieved = []
    for idx, feat in enumerate(indexed_train["features"]):
        distance = euclidean(query_features, feat)
        retrieved.append((distance, idx))
    return sorted(retrieved)[:max_results]

def most_common(arr):
    return Counter(arr).most_common(1)[0][0]

def predict_file_by_path(path, magnificient = None):
    if magnificient is None:
        feature_path = "models\\features\\binary\\training_1_all_binary_feature.json"
    else:    
        feature_path = f"models\\features\\binary\\training_1_{magnificient}_binary_feature.json"

    print("[INFO] load test image...")
    image = cv2.imread(path)
    image = cv2.resize(image, IMAGE_SIZE)
    test_x = np.array([image]).astype("float32") / 255.0

    print("[INFO] encoding image...")
    features_retrieved = encoder.predict(test_x)

    with open(feature_path) as f:
        training_indexed = json.load(f)

    results = perform_search(features_retrieved[0], training_indexed)
    labels_ret = [training_indexed["labels"][r[1]] for r in results]
    label = most_common(labels_ret)

    color = "green" if label in classifications["benign"] or label == "benign" else "red"
    return label, color

def predict_folder(folder_path, magnificient = None):
    if magnificient is None:
        feature_path = "models\\features\\binary\\training_1_all_binary_feature.json"
    else:    
        feature_path = f"models\\features\\binary\\training_1_{magnificient}_binary_feature.json"

    dataset = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
    if LIMIT_IMAGES is not None:
        dataset = dataset[:LIMIT_IMAGES]
    images = [cv2.resize(cv2.imread(img), IMAGE_SIZE) for img in dataset]
    test_x = np.array(images).astype("float32") / 255.0
    features_retrieved = encoder.predict(test_x)

    with open(feature_path) as f:
        training_indexed = json.load(f)

    res = dict(benign=0, malignant=0)
    for i in range(len(test_x)):
        queryFeatures = features_retrieved[i]
        results = perform_search(queryFeatures, training_indexed)
        labels_ret = [training_indexed["labels"][r[1]] for r in results]
        label = most_common(labels_ret)
        res[label] += 1
    return res

def matriz_dispersion(res_benign, res_malignant):
    TP = res_malignant.get('malignant', 0)
    TN = res_benign.get('benign', 0)
    FP = res_benign.get('malignant', 0)
    FN = res_malignant.get('benign', 0)

    print(f"\n{'':12}|{'benign':>10}   {'malignant':>10}")
    print("-" * 34)
    print(f"{'benign':12}|{TN:>10}   {FP:>10}")
    print(f"{'malignant':12}|{FN:>10}   {TP:>10}")
    print("\nMedidas de evaluación:")

    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total else 0
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    specificity = TN / (TN + FP) if (TN + FP) else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

    print(f"{'Accuracy':20}: {accuracy:.2f}")
    print(f"{'Precision (malignant)':20}: {precision:.2f}")
    print(f"{'Recall (malignant)':20}: {recall:.2f}")
    print(f"{'Specificity (benign)':20}: {specificity:.2f}")
    print(f"{'F1 Score':20}: {f1_score:.2f}")

def visualize_features_from_json(json_path, save=False, output_folder="outputs"):
    if not os.path.exists(json_path):
        print("El archivo no existe.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)
        filename = os.path.basename(f.name)
        name_without_ext = os.path.splitext(filename)[0]

    features = np.array(data["features"])
    labels = np.array(data["labels"])

    # Si hay muchas, hacer PCA
    if len(features) > 2:
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(features)

        # plt.figure(figsize=(6, 6))
        for lbl in np.unique(labels):
            idxs = labels == lbl
            plt.scatter(reduced[idxs, 0], reduced[idxs, 1], label=lbl, alpha=0.6)

    if save:
        path_img = os.path.join(output_folder, name_without_ext + "_visualization.png")
        plt.savefig(path_img)
        print(f"[INFO] Imagen guardada en: {path_img}")
    else:
        plt.show()

def visualize_features_tsne(json_path, save=False, output_folder="outputs"):
    if not os.path.exists(json_path):
        print("El archivo no existe.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)
        filename = os.path.basename(f.name)
        name_without_ext = os.path.splitext(filename)[0]

    features = np.array(data["features"])
    labels = np.array(data["labels"])

    print("[INFO] Reducción de dimensionalidad con t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    reduced = tsne.fit_transform(features)

    # Crear la carpeta si no existe
    if save and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    plt.figure(figsize=(7, 7))
    for lbl in np.unique(labels):
        idxs = labels == lbl
        plt.scatter(reduced[idxs, 0], reduced[idxs, 1], label=lbl, alpha=0.6)

    if save:
        path_img = os.path.join(output_folder, name_without_ext + "_tsne_visualization.png")
        plt.savefig(path_img)
        print(f"[INFO] Imagen guardada en: {path_img}")
    else:
        plt.show()
