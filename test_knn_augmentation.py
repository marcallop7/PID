from knn import predict_folder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from save_metrics import save_metrics_csv, format_metrics
import random
import cv2
import numpy as np
import os

# Función para aplicar aumentos a las imágenes (rotaciones e intensidades)
def augment_image(image, num_variations=5):
    augmented_images = []
    for _ in range(num_variations):
        # Rotación aleatoria de la imagen entre -30 y 30 grados
        angle = random.uniform(-30, 30)
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
        
        # Ajuste de la intensidad de la imagen (variación de brillo)
        intensity_factor = random.uniform(0.7, 1.3)
        augmented_image = np.clip(rotated_image * intensity_factor, 0, 255).astype(np.uint8)
        
        # Almacenamos la imagen aumentada
        augmented_images.append(augmented_image)
    return augmented_images

# Función para mostrar la matriz de confusión y las métricas de evaluación
def show_confusion_matrix_from_dicts(pred_benign_dict, pred_malign_dict, class_labels, model_name, show_metrics=True):
    # Preparamos las listas de clases verdaderas y predicciones
    y_true = []
    y_pred = []

    # Añadimos las predicciones de las imágenes benignas
    y_true += ["benign"] * sum(pred_benign_dict.values())
    y_pred += (["benign"] * pred_benign_dict["benign"]) + (["malignant"] * pred_benign_dict["malignant"])

    # Añadimos las predicciones de las imágenes malignas
    y_true += ["malignant"] * sum(pred_malign_dict.values())
    y_pred += (["benign"] * pred_malign_dict["benign"]) + (["malignant"] * pred_malign_dict["malignant"])

    # Calculamos la matriz de confusión
    cm = confusion_matrix(y_true, y_pred, labels=list(class_labels.keys()))

    # Creamos una figura con dos subgráficas: una para la matriz y otra para el texto
    fig, (ax_matrix, ax_text) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [3, 1]})

    # Mostramos la matriz de confusión en el primer gráfico
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels.values(),
                yticklabels=class_labels.values(),
                ax=ax_matrix)

    ax_matrix.set_xlabel('Predicciones')
    ax_matrix.set_ylabel('Valores Reales')
    ax_matrix.set_title('Matriz de Confusión')

    # Calculamos las métricas de evaluación
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label="malignant", average='binary')
    prec = precision_score(y_true, y_pred, pos_label="malignant", average='binary')
    recall = recall_score(y_true, y_pred, pos_label="malignant", average='binary')

    # Guardamos las métricas en un diccionario
    metrics = {
        "Accuracy": acc,
        "F1 Score": f1,
        "Precision": prec,
        "Recall": recall
    }

    # Formateamos las métricas para mostrarlas como texto
    metrics_text = format_metrics(metrics)
    
    # Mostramos las métricas en el segundo gráfico
    ax_text.axis('off')
    ax_text.text(0, 0.8, metrics_text, fontsize=12, va='top', ha='left',
                 bbox=dict(facecolor='white', edgecolor='black'))
    
    # Si se desea, mostramos la gráfica
    if(show_metrics):
        plt.tight_layout()
        plt.show()

    # Guardamos las métricas en un archivo CSV
    save_metrics_csv(model_name, metrics)

# Bloque principal para ejecutar la evaluación
if __name__ == "__main__":
    # Configuramos si queremos mostrar las métricas
    show_metrics = False  # Cambia a True para ver las métricas

    # Iteramos sobre las diferentes configuraciones de magnificación
    for magnificient in [40, 100, 200, 400, None]:
        # Si se especifica un valor de magnificación, usamos un directorio específico
        if magnificient is not None:
            # Directorios para imágenes benignas y malignas en función de la magnificación
            folder_path_benign = os.path.join("images", "binary_scenario", "test", f"{magnificient}x", "benign")
            folder_path_malignant = os.path.join("images", "binary_scenario", "test", f"{magnificient}x", "malignant")

            # Realizamos la predicción en las carpetas de imágenes benignas y malignas con aumentos
            predict_folder_benign = predict_folder(folder_path_benign, f"{magnificient}x", augment_image)
            predict_folder_malignant = predict_folder(folder_path_malignant, f"{magnificient}x", augment_image)
            
            # Establecemos el nombre del modelo con el tamaño de magnificación
            model_name = f"modelo_knn_aug_{magnificient}x"
        else:
            # Si no hay magnificación, usamos el conjunto de datos combinado
            folder_path_benign = os.path.join("images", "binary_scenario_merged", "test", "benign")
            folder_path_malignant = os.path.join("images", "binary_scenario_merged", "test", "malignant")
            
            # Realizamos la predicción en las carpetas de imágenes benignas y malignas sin magnificación
            predict_folder_benign = predict_folder(folder_path_benign, augmentation_function=augment_image)
            predict_folder_malignant = predict_folder(folder_path_malignant, augmentation_function=augment_image)
            
            # Establecemos el nombre del modelo para el caso sin magnificación
            model_name = "modelo_knn_aug_all"

        # Definimos las etiquetas de las clases para la visualización
        class_labels = {"benign": "Benigno", "malignant": "Maligno"}

        # Mostramos la matriz de confusión y las métricas para las predicciones
        show_confusion_matrix_from_dicts(predict_folder_benign, predict_folder_malignant, class_labels, model_name, show_metrics=show_metrics)
