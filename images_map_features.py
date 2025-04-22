from knn_2 import visualize_features_from_json, visualize_features_tsne
import os

folder_path = "models\\features\\binary"
for filename in os.listdir(folder_path):
    path = f"{folder_path}\\{filename}"
    visualize_features_from_json(path, save=True)
    visualize_features_tsne(path, save=True)