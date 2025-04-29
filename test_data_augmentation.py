import os
import cv2
import numpy as np
import random
from tensorflow.keras.models import load_model  # type: ignore
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import seaborn as sns

def augment_image(image, num_variations=5):
    augmented_images = []
    for _ in range(num_variations):
        angle = random.uniform(-30, 30)
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
        intensity_factor = random.uniform(0.7, 1.3)
        augmented_image = np.clip(rotated_image * intensity_factor, 0, 255).astype(np.uint8)
        augmented_images.append(augmented_image)
    return augmented_images

def load_trained_model(model_path):
    return load_model(model_path)

def load_images_with_labels(data_dir, target_size=(128, 128), num_variations=5):
    images = []
    labels = []
    class_names = sorted(os.listdir(data_dir))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            image = cv2.imread(img_path)
            if image is None:
                continue
            image = cv2.resize(image, target_size)
            augmented = augment_image(image, num_variations)
            images.extend(augmented)
            labels.extend([class_to_idx[class_name]] * num_variations)
    return np.array(images), np.array(labels), class_to_idx

def evaluate_model_on_augmented_data(model, data_dir, target_size=(128, 128), num_variations=5):
    images, labels, class_to_idx = load_images_with_labels(data_dir, target_size, num_variations)
    images = images.astype("float32") / 255.0

    predictions = model.predict(images, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)

    show_confusion_matrix(labels, predicted_classes, class_to_idx)

def show_confusion_matrix(true_classes, predicted_classes, class_labels):
    cm = confusion_matrix(true_classes, predicted_classes)
    fig, (ax_matrix, ax_text) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [3, 1]})

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels.keys(),
                yticklabels=class_labels.keys(),
                ax=ax_matrix)

    ax_matrix.set_xlabel('Predicciones')
    ax_matrix.set_ylabel('Valores Reales')
    ax_matrix.set_title('Matriz de Confusión')

    acc = accuracy_score(true_classes, predicted_classes)
    f1 = f1_score(true_classes, predicted_classes, average='binary')
    prec = precision_score(true_classes, predicted_classes, average='binary')
    recall = recall_score(true_classes, predicted_classes, average='binary')

    metrics_text = (f"Accuracy: {acc:.2f}\n"
                    f"F1 Score: {f1:.2f}\n"
                    f"Precision: {prec:.2f}\n"
                    f"Recall: {recall:.2f}")

    ax_text.axis('off')
    ax_text.text(0, 0.8, metrics_text, fontsize=12, va='top', ha='left',
                 bbox=dict(facecolor='white', edgecolor='black'))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    for MAGNIFICIENT_MODEL in [40, 100, 200, 400, None]:
        if MAGNIFICIENT_MODEL is not None:
            model_path = f"modelo_cnn_{MAGNIFICIENT_MODEL}x.h5"
        else:
            model_path = "modelo_cnn_all.h5"

        MAGNIFICIENT_TEST = 40
        if MAGNIFICIENT_TEST is not None:
            test_data_dir = f"./images/binary_scenario/test/{MAGNIFICIENT_TEST}X"
        else:
            test_data_dir = "./images/binary_scenario_merged/test"

        model = load_trained_model(model_path)
        evaluate_model_on_augmented_data(model, test_data_dir)
