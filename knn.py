import numpy as np
import json
import os
import cv2
import copy

from tensorflow.keras.models import Model, load_model # type: ignore
from collections import Counter

import matplotlib.pyplot as plt

IMAGE_SIZE = (256, 256)
feature_path = "models/training_1_40_binary_feature.json"
model_path = "models/training_1.h5"
history_path = "models/training_1_40_binary_history.json"
magnification = "40X"
base_dataset = "images/binary_scenario"
class_dir = ['benign', 'malignant']
folder_path = "images\\binary_scenario\\test\\40X\\" + str(class_dir[0])
folder_path = r"other\histology_slides\breast\benign\SOB\adenosis\SOB_B_A_14-22549AB\40X"

print("[INFO] indexing file images BreaKHis dataset...")
type_dataset = ['val', 'train']
dataset_train = []
dataset_val = []
for type_set in type_dataset:
    for class_item in class_dir:
        cur_dir = os.path.join(base_dataset, type_set, magnification ,class_item)
        for file in os.listdir(cur_dir):
            if type_set == 'train':
                dataset_train.append(os.path.join(cur_dir, file))
            else:
                dataset_val.append(os.path.join(cur_dir, file))

print("[INFO] load images BreaKHis dataset...")
#  load images
train_images = []
val_images = []
for type_set in type_dataset:
    cur_dataset = dataset_train if type_set == 'train' else dataset_val
    for image_path in cur_dataset:
        if ".png" in image_path:
            image = cv2.imread(image_path)
            image = cv2.resize(image, IMAGE_SIZE)
            if type_set == 'train':
                train_images.append(image)
            else:
                val_images.append(image)

# normalization
print("[INFO] normalization...")
train_x = np.array(train_images).astype("float32") / 255.0
val_x = np.array(val_images).astype("float32") / 255.0

auto_encoder = load_model(model_path, compile=False)
# load our auto_encoder from disk
print("[INFO] loading auto encoder model...")
auto_encoder.load_weights(model_path)

# create the encoder model which consists of *just* the encoder
# portion of the auto encoder
encoder = Model(inputs=auto_encoder.input,
	outputs=auto_encoder.get_layer("encoded").output)

# quantify the contents of our input images using the encoder
print("[INFO] encoding images...")
features = encoder.predict(train_x)


indexes = list(range(0, train_x.shape[0]))
features_array = [[float(x) for x in y] for y in features]
labels = [path.split("\\")[3] for path in dataset_train]
data = {"indexes": indexes, "features": features_array, "locations": dataset_train, "labels":labels}

with open(feature_path, 'w') as f:
    json.dump(data, f)

def euclidean(a, b):
	# compute and return the euclidean distance between two vectors
	return np.linalg.norm(a - b)

def perform_search(query_features, indexed_train, max_results=5):
    retrieved = []
    for idx in range(0, len(indexed_train["features"])):
        distance = euclidean(query_features, indexed_train["features"][idx])
        retrieved.append((distance, idx))
    retrieved = sorted(retrieved)[:max_results]
    return retrieved

print("[INFO] indexing file images BreaKHis dataset...")
# indexing file images
dataset = []
for file in os.listdir(folder_path):
    dataset.append(os.path.join(folder_path, file))

print("test len to retrieving:", len(dataset))
print("[INFO] load test images BreaKHis dataset...")
#  load images
images = []
for image_path in dataset[:100]:
    image = cv2.imread(image_path)
    image = cv2.resize(image, IMAGE_SIZE)
    images.append(image)


print("[INFO] normalization...")
test_x = np.array(images).astype("float32") / 255.0

# quantify the contents of our input images using the encoder
print("[INFO] encoding images...")
features_retrieved = encoder.predict(test_x)

def most_common(arr):
    return Counter(arr).most_common(1)[0][0]


with open(feature_path) as f:
  training_indexed = json.load(f)

query_indexes = list(range(0, test_x.shape[0]))
label_builder = list(np.unique(training_indexed["labels"]))
class_builder = {label_unique:[] for label_unique in label_builder}
recalls = copy.deepcopy(class_builder)
precisions = copy.deepcopy(class_builder)
# loop over the testing indexes

res = dict(benign=0, malignant=0)
for i in query_indexes:
    queryFeatures = features_retrieved[i]
    results = perform_search(queryFeatures, training_indexed, max_results=5)
    # Las etiquetas de los más cercanos
    labels_ret = [training_indexed["labels"][r[1]] for r in results]
    label = most_common(labels_ret)
    res[label] += 1

print(res)
