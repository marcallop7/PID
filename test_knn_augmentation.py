from knn import predict_folder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from save_metrics import save_metricas_csv, format_metrics
import random
import cv2
import numpy as np

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


def show_confusion_matrix_from_dicts(pred_benign_dict, pred_malign_dict, class_labels, model_name, show_metrics=True):
    y_true = []
    y_pred = []

    y_true += ["benign"] * sum(pred_benign_dict.values())
    y_pred += (["benign"] * pred_benign_dict["benign"]) + (["malignant"] * pred_benign_dict["malignant"])

    y_true += ["malignant"] * sum(pred_malign_dict.values())
    y_pred += (["benign"] * pred_malign_dict["benign"]) + (["malignant"] * pred_malign_dict["malignant"])

    cm = confusion_matrix(y_true, y_pred, labels=list(class_labels.keys()))

    fig, (ax_matrix, ax_text) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [3, 1]})

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels.values(),
                yticklabels=class_labels.values(),
                ax=ax_matrix)

    ax_matrix.set_xlabel('Predicciones')
    ax_matrix.set_ylabel('Valores Reales')
    ax_matrix.set_title('Matriz de Confusión')

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label="malignant", average='binary')
    prec = precision_score(y_true, y_pred, pos_label="malignant", average='binary')
    recall = recall_score(y_true, y_pred, pos_label="malignant", average='binary')

    metrics = {
        "Accuracy": acc,
        "F1 Score": f1,
        "Precision": prec,
        "Recall": recall
    }

    metrics_text = format_metrics(metrics)
    
    ax_text.axis('off')
    ax_text.text(0, 0.8, metrics_text, fontsize=12, va='top', ha='left',
                 bbox=dict(facecolor='white', edgecolor='black'))
    
    if(show_metrics):
        plt.tight_layout()
        plt.show()

    save_metricas_csv(model_name, metrics)


if __name__ == "__main__":
    show_metrics=False
    for magnificient in [40, 100, 200, 400, None]:
        if magnificient is not None:
            folder_path_benign = f"images\\binary_scenario\\test\\{magnificient}x\\benign"
            folder_path_malignant = f"images\\binary_scenario\\test\\{magnificient}x\\malignant"

            predict_folder_benign = predict_folder(folder_path_benign, f"{magnificient}x", augment_image)
            predict_folder_malignant = predict_folder(folder_path_malignant, f"{magnificient}x", augment_image)
            model_name = f"modelo_knn_aug_{magnificient}x"
        else: 
            folder_path_benign = f"images\\binary_scenario_merged\\test\\benign"
            folder_path_malignant = f"images\\binary_scenario_merged\\test\\malignant"
        
            predict_folder_benign = predict_folder(folder_path_benign, augmentation_function=augment_image)
            predict_folder_malignant = predict_folder(folder_path_malignant, augmentation_function=augment_image)
            model_name = "modelo_knn_aug_all"

        class_labels = {"benign": "Benigno", "malignant": "Maligno"}

        show_confusion_matrix_from_dicts(predict_folder_benign, predict_folder_malignant, class_labels, model_name, show_metrics=show_metrics)