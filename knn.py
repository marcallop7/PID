import numpy as np
import json
import os
import cv2

from tensorflow.keras.models import Model, load_model # type: ignore
from collections import Counter

IMAGE_SIZE = (256, 256)
LIMIT_IMAGES = 500
# feature_path = "models\\training_1_40_binary_feature.json"
feature_path = "models\\features\\binary\\training_1_indexed_400.json"
folder_path_benign = "images\\binary_scenario\\test\\400X\\benign"
folder_path_malignant = "images\\binary_scenario\\test\\400X\\malignant"
model_path = "models\\training_1.h5"

clasiffications = {
    'benign': ['adenosis', 'fibrodenoma', 'phyllodes_tumor', 'tubular_adenoma'],
    'malignant': ['ductal_carcinoma', 'lobular_carcinoma', 'mucinous_carcinoma', 'papillary_carcinoma']
}

labels_binary = clasiffications.keys()
labels_subclasses = [item for sublist in clasiffications.values() for item in sublist]

print("[INFO] loading auto encoder model...")
auto_encoder = load_model(model_path, compile=False)
auto_encoder.load_weights(model_path)

encoder = Model(inputs=auto_encoder.input,
	outputs=auto_encoder.get_layer("encoded").output)


# compute and return the euclidean distance between two vectors
def euclidean(a, b):
	return np.linalg.norm(a - b)

def perform_search(query_features, indexed_train, max_results=5):
    retrieved = []
    for idx in range(0, len(indexed_train["features"])):
        distance = euclidean(query_features, indexed_train["features"][idx])
        retrieved.append((distance, idx))
    retrieved = sorted(retrieved)[:max_results]
    return retrieved

def most_common(arr):
    return Counter(arr).most_common(1)[0][0]

def predict_file_by_path(path):
    print("[INFO] load test images BreaKHis dataset...")
    image = cv2.imread(path)
    image = cv2.resize(image, IMAGE_SIZE)

    print("[INFO] normalization...")
    test_x = np.array([image]).astype("float32") / 255.0

    # quantify the contents of our input images using the encoder
    print("[INFO] encoding images...")
    features_retrieved = encoder.predict(test_x)

    with open(feature_path) as f:
        training_indexed = json.load(f)

    results = perform_search(features_retrieved[0], training_indexed, max_results=5)
    labels_ret = [training_indexed["labels"][r[1]] for r in results]

    return most_common(labels_ret)

r = predict_file_by_path(r"C:\Users\lmher\Desktop\Informatica_4\PID\Trabajo\PID\images\binary_scenario\test\40X\malignant\SOB_M_DC-14-2523-40-014.png")
print(r)

def predict_folder(folder_path):
    print("[INFO] indexing file images BreaKHis dataset...")
    dataset = []
    for file in os.listdir(folder_path):
        dataset.append(os.path.join(folder_path, file))

    print("test len to retrieving:", len(dataset))
    print("[INFO] load test images BreaKHis dataset...")
    #  load images
    images = []
    for image_path in dataset[:LIMIT_IMAGES]:
        image = cv2.imread(image_path)
        image = cv2.resize(image, IMAGE_SIZE)
        images.append(image)

    print("[INFO] normalization...")
    test_x = np.array(images).astype("float32") / 255.0

    # quantify the contents of our input images using the encoder
    print("[INFO] encoding images...")
    features_retrieved = encoder.predict(test_x)

    with open(feature_path) as f:
        training_indexed = json.load(f)

    query_indexes = list(range(0, test_x.shape[0]))

    res = dict(benign=0, malignant=0)
    for i in query_indexes:
        queryFeatures = features_retrieved[i]
        results = perform_search(queryFeatures, training_indexed, max_results=5)
        # Las etiquetas de los más cercanos
        labels_ret = [training_indexed["labels"][r[1]] for r in results]
        label = most_common(labels_ret)
        res[label] += 1

    return res



# predict_folder_benign = predict_folder(folder_path_benign)
# predict_folder_malignant = predict_folder(folder_path_malignant)

# def matriz_dispersion(res_benign, res_malignant):
#     # Extraemos los valores
#     TP = res_malignant['malignant']  # verdaderos positivos
#     TN = res_benign['benign']        # verdaderos negativos
#     FP = res_benign['malignant']     # falsos positivos
#     FN = res_malignant['benign']     # falsos negativos

#     # Mostrar matriz de dispersión
#     print(f"{'':12}|{'benign':>10}   {'malignant':>10}")
#     print("-" * 34)
#     print(f"{'benign':12}|{TN:>10}   {FP:>10}")
#     print(f"{'malignant':12}|{FN:>10}   {TP:>10}")
#     print("\nMedidas de evaluación:")

#     # Cálculos
#     total = TP + TN + FP + FN
#     accuracy = (TP + TN) / total if total else 0
#     precision = TP / (TP + FP) if (TP + FP) else 0
#     recall = TP / (TP + FN) if (TP + FN) else 0
#     specificity = TN / (TN + FP) if (TN + FP) else 0
#     f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

#     # Mostrar métricas
#     print(f"{'Accuracy':20}: {accuracy:.2f}")
#     print(f"{'Precision (malignant)':20}: {precision:.2f}")
#     print(f"{'Recall (malignant)':20}: {recall:.2f}")
#     print(f"{'Specificity (benign)':20}: {specificity:.2f}")
#     print(f"{'F1 Score':20}: {f1_score:.2f}")

# matriz_dispersion(predict_folder_benign, predict_folder_malignant)