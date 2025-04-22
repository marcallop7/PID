import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def load_trained_model(model_path):
    """
    Carga el modelo previamente entrenado desde un archivo .h5.
    """
    return load_model(model_path)

def evaluate_model_on_directory(model, data_dir, target_size=(128, 128), batch_size=32):
    """
    Evaluar el modelo en un conjunto de imágenes dentro de un directorio.
    """
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
    filenames = test_generator.filenames
    class_labels = test_generator.class_indices
    class_labels = {v: k for k, v in class_labels.items()}  # Invertir el diccionario
    
    print("\nPredicciones:")
    for filename, predicted_class in zip(filenames, predicted_classes):
        print(f"{filename} -> {class_labels[predicted_class]}")

    # Mostrar imágenes con sus predicciones
    plt.figure(figsize=(10, 10))
    for i, (filename, predicted_class) in enumerate(zip(filenames[:9], predicted_classes[:9])):  # Mostrar solo las primeras 9
        img_path = os.path.join(data_dir, filename)
        img = image.load_img(img_path, target_size=target_size)
        plt.subplot(3, 3, i+1)
        plt.imshow(img)
        plt.title(f"Predicción: {class_labels[predicted_class]}")
        plt.axis('off')
    
    plt.show()

    # Mostrar la matriz de confusión
    show_confusion_matrix(true_classes, predicted_classes, class_labels)

def show_confusion_matrix(true_classes, predicted_classes, class_labels):
    """
    Muestra la matriz de confusión utilizando seaborn.
    """
    cm = confusion_matrix(true_classes, predicted_classes)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels.values(), yticklabels=class_labels.values())
    plt.xlabel('Predicciones')
    plt.ylabel('Valores Reales')
    plt.title('Matriz de Confusión')
    plt.show()

if __name__ == "__main__":
    MAGNIFICIENT_MODEL=None
    # Ruta al modelo entrenado
    if MAGNIFICIENT_MODEL is not None:
        model_path = f"modelo_cnn_{MAGNIFICIENT_MODEL}x.h5"  # Asumimos que el modelo está guardado aquí
    else:
        model_path = "modelo_cnn_all.h5"
    
    MAGNIFICIENT_TEST = None
    # Ruta al directorio de imágenes para probar
    if MAGNIFICIENT_TEST is not None:
        test_data_dir = f"./images/binary_scenario/test/{MAGNIFICIENT_TEST}X"  # Cambia esta ruta a tu directorio de prueba
    else:
        test_data_dir = "./images/binary_scenario_merged/test"

    # Cargar el modelo entrenado
    model = load_trained_model(model_path)
    
    # Evaluar el modelo en el directorio de prueba
    evaluate_model_on_directory(model, test_data_dir)
