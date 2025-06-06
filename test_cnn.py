import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from cnn import load_trained_model_by_weights
from save_metrics import save_metrics_csv

def evaluate_model_on_directory(model, data_dir, target_size=(128, 128), batch_size=32, model_name="", show_metrics=True):
    """
    Evalúa el modelo sobre un conjunto de datos almacenado en un directorio, calculando y mostrando métricas.
    
    Parámetros:
    - model: El modelo previamente entrenado.
    - data_dir: Ruta al directorio que contiene las imágenes de prueba.
    - target_size: Tamaño al que se redimensionan las imágenes.
    - batch_size: Tamaño del batch para el generador de imágenes.
    - model_name: Nombre del modelo, utilizado para guardar las métricas.
    - show_metrics: Si es True, muestra la matriz de confusión y las métricas.
    """
    test_datagen = ImageDataGenerator(rescale=1./255)  # Normaliza las imágenes
    test_generator = test_datagen.flow_from_directory(
        data_dir,  # Directorio de imágenes de prueba
        target_size=target_size,  # Redimensiona las imágenes
        batch_size=batch_size,  # Número de imágenes por batch
        class_mode='categorical',  # Tipo de etiquetas, en este caso categorías múltiples
        shuffle=False  # No se barajan las imágenes para asegurar el orden de las clases
    )
    
    # Realiza las predicciones
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)  # Las clases predichas (índice de la clase con la probabilidad más alta)
    true_classes = test_generator.classes  # Clases reales

    # Calcular métricas de evaluación
    acc = accuracy_score(true_classes, predicted_classes)
    f1 = f1_score(true_classes, predicted_classes, average='weighted')  # F1 para múltiples clases
    prec = precision_score(true_classes, predicted_classes, average='weighted')  # Precisión para múltiples clases
    recall = recall_score(true_classes, predicted_classes, average='weighted')  # Recall para múltiples clases

    # Mostrar la matriz de confusión
    class_labels = test_generator.class_indices  # Diccionario con las etiquetas de clase
    class_labels = {v: k for k, v in class_labels.items()}  # Invertir el diccionario para usar las etiquetas como nombres
    show_confusion_matrix(true_classes, predicted_classes, class_labels, show_metrics)

    # Guardar métricas en un archivo CSV
    metrics = {
        "Accuracy": acc,
        "F1 Score": f1,
        "Precision": prec,
        "Recall": recall
    }
    save_metrics_csv(model_name, metrics)

def show_confusion_matrix(true_classes, predicted_classes, class_labels, show_metrics=True):
    """
    Muestra la matriz de confusión junto con las métricas de evaluación.
    
    Parámetros:
    - true_classes: Clases reales (verdaderas).
    - predicted_classes: Clases predichas por el modelo.
    - class_labels: Diccionario de etiquetas de clase.
    - show_metrics: Si es True, muestra la matriz y las métricas.
    """
    cm = confusion_matrix(true_classes, predicted_classes)  # Calcular la matriz de confusión

    # Crear una figura con dos columnas: una para la matriz de confusión y otra para las métricas
    fig, (ax_matrix, ax_text) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [3, 1]})

    # Mostrar la matriz de confusión
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",  # Usar un mapa de colores azul
                xticklabels=class_labels.values(),
                yticklabels=class_labels.values(),
                ax=ax_matrix)

    ax_matrix.set_xlabel('Predicciones')
    ax_matrix.set_ylabel('Valores Reales')
    ax_matrix.set_title('Matriz de Confusión')

    # Calcular métricas de evaluación
    acc = accuracy_score(true_classes, predicted_classes)
    f1 = f1_score(true_classes, predicted_classes, average='binary')  # Para problemas binarios, cambiar si es multiclas
    prec = precision_score(true_classes, predicted_classes, average='binary')
    recall = recall_score(true_classes, predicted_classes, average='binary')

    metrics_text = (f"Accuracy: {acc}\n"
                    f"F1 Score: {f1}\n"
                    f"Precision: {prec}\n"
                    f"Recall: {recall}")

    # Mostrar las métricas en el panel derecho
    ax_text.axis('off')
    ax_text.text(0, 0.8, metrics_text, fontsize=12, va='top', ha='left',
                 bbox=dict(facecolor='white', edgecolor='black'))

    if show_metrics:
        plt.tight_layout()  # Ajustar el layout para que todo quepa
        plt.show()  # Mostrar la gráfica

if __name__ == "__main__":
    show_metrics = False
    # Evaluar el modelo para distintas configuraciones de magnificación
    for MAGNIFICIENT_MODEL in [40, 100, 200, 400, None]:
        # Establecer el nombre del modelo según la magnificación
        if MAGNIFICIENT_MODEL is not None:
            model_name = f"modelo_cnn_{MAGNIFICIENT_MODEL}x"
        else:
            model_name = "modelo_cnn_all"

        # Directorio de prueba según el tamaño del conjunto de datos
        MAGNIFICIENT_TEST = 40
        if MAGNIFICIENT_TEST is not None:
            test_data_dir = f"./images/binary_scenario/test/{MAGNIFICIENT_TEST}X"
        else:
            test_data_dir = "./images/binary_scenario_merged/test"

        # Cargar el modelo entrenado con los pesos correspondientes
        model = load_trained_model_by_weights(MAGNIFICIENT_MODEL)
        
        # Evaluar el modelo con los datos de prueba
        evaluate_model_on_directory(model, test_data_dir, model_name=model_name, show_metrics=show_metrics)
