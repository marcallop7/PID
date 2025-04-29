import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def load_trained_model(model_path):
    return load_model(model_path)

def evaluate_model_on_directory(model, data_dir, target_size=(128, 128), batch_size=32):
    # Configuración de aumento de datos para predicción
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Generador de datos
    test_generator = test_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False  # No mezclar las imágenes para mantener el orden
    )
    
    # Realizamos la predicción sobre el conjunto de prueba
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Obtener las clases reales
    true_classes = test_generator.classes

    # Mostrar las predicciones con los nombres de las imágenes
    class_labels = test_generator.class_indices
    class_labels = {v: k for k, v in class_labels.items()}  # Invertir el diccionario

    # Mostrar la matriz de confusión
    show_confusion_matrix(true_classes, predicted_classes, class_labels)

def show_confusion_matrix(true_classes, predicted_classes, class_labels):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

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

    # Calcular métricas
    acc = accuracy_score(true_classes, predicted_classes)
    f1 = f1_score(true_classes, predicted_classes, average='binary')
    prec = precision_score(true_classes, predicted_classes, average='binary')
    recall = recall_score(true_classes, predicted_classes, average='binary')

    # Mostrar métricas
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
        # Ruta al modelo entrenado
        if MAGNIFICIENT_MODEL is not None:
            model_path = f"modelo_cnn_{MAGNIFICIENT_MODEL}x.h5"  # Asumimos que el modelo está guardado aquí
        else:
            model_path = "modelo_cnn_all.h5"
        
        MAGNIFICIENT_TEST = 40
        # TODO: Para experimentar se puede: Modificar arquitectura, modificar hiperparametros,
        #       otras imagenes, utilizar Data Augmentation. Otra opción es comparar con el KNN
        if MAGNIFICIENT_TEST is not None:
            test_data_dir = f"./images/binary_scenario/test/{MAGNIFICIENT_TEST}X"  # Cambia esta ruta a tu directorio de prueba
        else:
            test_data_dir = "./images/binary_scenario_merged/test"

        # Cargar el modelo entrenado
        model = load_trained_model(model_path)
        
        # Evaluar el modelo en el directorio de prueba
        evaluate_model_on_directory(model, test_data_dir)