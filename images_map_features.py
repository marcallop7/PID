from knn import visualize_features_pca, visualize_features_tsne
import os

os.makedirs('./outputs/features/pca', exist_ok=True)
os.makedirs('./outputs/features/tsne', exist_ok=True)

folder_path = "models\\features\\binary"
for filename in os.listdir(folder_path):
    path = f"{folder_path}\\{filename}"
    visualize_features_pca(path, save=True)
    visualize_features_tsne(path, save=True)