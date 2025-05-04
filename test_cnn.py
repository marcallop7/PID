import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from cnn import load_trained_model_by_weights
from save_metrics import save_metrics_csv

def evaluate_model_on_directory(model, data_dir, target_size=(128, 128), batch_size=32, model_name="", show_metrics=True):

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes

    # Calcular métricas
    acc = accuracy_score(true_classes, predicted_classes)
    f1 = f1_score(true_classes, predicted_classes, average='weighted')  # Cambiado a 'weighted' para múltiples clases
    prec = precision_score(true_classes, predicted_classes, average='weighted')
    recall = recall_score(true_classes, predicted_classes, average='weighted')

    # Mostrar la matriz de confusión
    class_labels = test_generator.class_indices
    class_labels = {v: k for k, v in class_labels.items()}  # Invertir el diccionario
    show_confusion_matrix(true_classes, predicted_classes, class_labels, show_metrics)

    metrics = {
        "Accuracy": acc,
        "F1 Score": f1,
        "Precision": prec,
        "Recall": recall
    }

    # Guardar métricas en el archivo
    save_metrics_csv(model_name,metrics)

def show_confusion_matrix(true_classes, predicted_classes, class_labels, show_metrics=True):
    """
    Muestra la matriz de confusión y las métricas de evaluación.
    """
    cm = confusion_matrix(true_classes, predicted_classes)

    # Crear figura con dos columnas: matriz y texto
    fig, (ax_matrix, ax_text) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [3, 1]})

    # Mostrar matriz de confusión
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels.values(),
                yticklabels=class_labels.values(),
                ax=ax_matrix)

    ax_matrix.set_xlabel('Predicciones')
    ax_matrix.set_ylabel('Valores Reales')
    ax_matrix.set_title('Matriz de Confusión')

    acc = accuracy_score(true_classes, predicted_classes)
    f1 = f1_score(true_classes, predicted_classes, average='binary')
    prec = precision_score(true_classes, predicted_classes, average='binary')
    recall = recall_score(true_classes, predicted_classes, average='binary')

    metrics_text = (f"Accuracy: {acc}\n"
                    f"F1 Score: {f1}\n"
                    f"Precision: {prec}\n"
                    f"Recall: {recall}")
    
    ax_text.axis('off')
    ax_text.text(0, 0.8, metrics_text, fontsize=12, va='top', ha='left',
                 bbox=dict(facecolor='white', edgecolor='black'))

    if(show_metrics):  
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    show_metrics = False
    for MAGNIFICIENT_MODEL in [40, 100, 200, 400, None]:
        # Ruta al modelo entrenado
        if MAGNIFICIENT_MODEL is not None:
            model_name = f"modelo_cnn_{MAGNIFICIENT_MODEL}x"
        else:
            model_name = "modelo_cnn_all"

        MAGNIFICIENT_TEST = 40
        if MAGNIFICIENT_TEST is not None:
            test_data_dir = f"./images/binary_scenario/test/{MAGNIFICIENT_TEST}X"
        else:
            test_data_dir = "./images/binary_scenario_merged/test"

        # Cargar el modelo entrenado
        model = load_trained_model_by_weights(MAGNIFICIENT_MODEL)
        
        # Evaluar el modelo en el directorio de prueba
        evaluate_model_on_directory(model, test_data_dir, model_name=model_name, show_metrics=show_metrics)