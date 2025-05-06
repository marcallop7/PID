import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import seaborn as sns
from cnn import load_trained_model_by_weights
from save_metrics import save_metrics_csv, format_metrics

def augment_image(image, num_variations=5):
    """
    Genera variaciones de una imagen mediante rotaciones aleatorias y cambios de intensidad.
    
    Parámetros:
    - image: Imagen original que se quiere aumentar.
    - num_variations: Número de variaciones que se quieren generar.
    
    Retorna:
    - augmented_images: Lista de imágenes aumentadas.
    """
    augmented_images = []
    for _ in range(num_variations):
        # Generar un ángulo de rotación aleatorio
        angle = random.uniform(-30, 30)
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))

        # Aplicar un cambio aleatorio de intensidad
        intensity_factor = random.uniform(0.7, 1.3)
        augmented_image = np.clip(rotated_image * intensity_factor, 0, 255).astype(np.uint8)
        augmented_images.append(augmented_image)

    return augmented_images

def load_images_with_labels(data_dir, target_size=(128, 128), num_variations=5):
    """
    Carga imágenes desde el directorio y genera variaciones aumentadas.
    
    Parámetros:
    - data_dir: Directorio de las imágenes, organizado por clases.
    - target_size: Tamaño al que se redimensionan las imágenes.
    - num_variations: Número de variaciones que se quieren generar por imagen.
    
    Retorna:
    - images: Lista de imágenes aumentadas.
    - labels: Lista de etiquetas correspondientes a las imágenes.
    - class_to_idx: Diccionario que mapea nombres de clase a índices.
    """
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

def evaluate_model_on_augmented_data(model, data_dir, target_size=(128, 128), num_variations=5, model_name="", show_metrics=True):
    """
    Evalúa el modelo sobre los datos aumentados y calcula métricas.
    
    Parámetros:
    - model: El modelo previamente entrenado.
    - data_dir: Directorio con las imágenes de prueba.
    - target_size: Tamaño de las imágenes.
    - num_variations: Número de variaciones por imagen.
    - model_name: Nombre del modelo, para guardar las métricas.
    - show_metrics: Si es True, muestra las métricas y la matriz de confusión.
    """
    images, labels, class_to_idx = load_images_with_labels(data_dir, target_size, num_variations)
    images = images.astype("float32") / 255.0

    # Predicción del modelo
    predictions = model.predict(images, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)

    # Calcular métricas
    acc = accuracy_score(labels, predicted_classes)
    f1 = f1_score(labels, predicted_classes, average='weighted')
    prec = precision_score(labels, predicted_classes, average='weighted')
    recall = recall_score(labels, predicted_classes, average='weighted')

    metrics = {
        "Accuracy": acc,
        "F1 Score": f1,
        "Precision": prec,
        "Recall": recall
    }

    # Mostrar la matriz de confusión
    show_confusion_matrix(labels, predicted_classes, class_to_idx, metrics, show_metrics)

    # Guardar métricas en el archivo CSV
    save_metrics_csv(model_name, metrics)

def show_confusion_matrix(true_classes, predicted_classes, class_labels, metrics, show_metrics=True):
    """
    Muestra la matriz de confusión y las métricas del modelo.
    
    Parámetros:
    - true_classes: Clases reales.
    - predicted_classes: Clases predichas por el modelo.
    - class_labels: Etiquetas de clase.
    - metrics: Métricas a mostrar junto a la matriz de confusión.
    - show_metrics: Si es True, muestra la gráfica y las métricas.
    """
    cm = confusion_matrix(true_classes, predicted_classes)
    fig, (ax_matrix, ax_text) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [3, 1]})

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels.keys(),
                yticklabels=class_labels.keys(),
                ax=ax_matrix)

    ax_matrix.set_xlabel('Predicciones')
    ax_matrix.set_ylabel('Valores Reales')
    ax_matrix.set_title('Matriz de Confusión')

    # Mostrar métricas en el segundo subplot
    metrics_text = format_metrics(metrics)
    ax_text.axis('off')
    ax_text.text(0, 0.8, metrics_text, fontsize=12, va='top', ha='left',
                 bbox=dict(facecolor='white', edgecolor='black'))

    if(show_metrics):
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    show_metrics=False
    # Ejecutar la evaluación para diferentes magnificaciones (tamaños de dataset)
    for MAGNIFICIENT in [40, 100, 200, 400, None]:
        if MAGNIFICIENT is not None:
            model_name = f"modelo_cnn_aug_{MAGNIFICIENT}x"
        else:
            model_name = "modelo_cnn_aug_all"

        # Configuración del directorio de datos de prueba según la magnificación
        if MAGNIFICIENT is not None:
            test_data_dir = f"./images/binary_scenario/test/{MAGNIFICIENT}X"
        else:
            test_data_dir = "./images/binary_scenario_merged/test"

        # Cargar el modelo entrenado con pesos correspondientes
        model = load_trained_model_by_weights(MAGNIFICIENT)
        
        # Evaluar el modelo sobre los datos aumentados
        evaluate_model_on_augmented_data(model, test_data_dir, model_name=model_name, show_metrics=show_metrics)
