from tensorflow.keras import layers, models, optimizers, regularizers # type: ignore # Importar las librerías necesarias de Keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore # Para la generación de datos aumentados
import os


# Función para construir el modelo CNN
def build_cnn_model(input_shape=(128, 128, 3), num_classes=2):
    model = models.Sequential()

    # 1ª capa convolucional (64 filtros, tamaño 3x3)
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                            input_shape=input_shape, kernel_regularizer=regularizers.l2(1e-3)))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2))  # Capa de MaxPooling (reducción de tamaño)

    # 2ª capa convolucional (96 filtros, tamaño 3x3)
    model.add(layers.Conv2D(96, (3, 3), activation='relu', padding='same',
                            kernel_regularizer=regularizers.l2(1e-3)))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2))  # Capa de MaxPooling

    # 3ª capa convolucional (128 filtros, tamaño 3x3)
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                            kernel_regularizer=regularizers.l2(1e-3)))

    # 4ª capa convolucional (256 filtros, tamaño 3x3)
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                            kernel_regularizer=regularizers.l2(1e-3)))

    # 5ª capa convolucional (256 filtros, tamaño 3x3)
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                            kernel_regularizer=regularizers.l2(1e-3)))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2))  # Capa de MaxPooling

    # Aplanar la salida de las capas convolucionales
    model.add(layers.Flatten())
    
    # Capa densa (2000 unidades, activación ReLU)
    model.add(layers.Dense(2000, activation='relu'))
    
    # Capa de Dropout para regularización (50% de probabilidad de "apagar" unidades)
    model.add(layers.Dropout(0.5))
    
    # Capa de salida con softmax para clasificación multiclase
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model


# Función para entrenar el modelo CNN
def train_model(data_dir, num_classes=2, input_shape=(128, 128, 3), batch_size=32, epochs=20):

    train_datagen = ImageDataGenerator(
        rescale=1./255,  # Normalización de imágenes a valores entre 0 y 1
        rotation_range=270,  # Rotaciones aleatorias de 270 grados
        horizontal_flip=True,  # Volteo horizontal aleatorio
        validation_split=0.2  # 20% para validación
    )

    # Configuración de los generadores de datos para entrenamiento y validación
    target_size = input_shape[:2]  # Tamaño de las imágenes
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',  # Multiclase
        subset='training'  # Conjunto de datos de entrenamiento
    )

    val_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',  # Multiclase
        subset='validation'  # Conjunto de datos de validación
    )

    # Construir el modelo CNN
    model = build_cnn_model(input_shape=input_shape, num_classes=num_classes)

    # Compilación del modelo con optimizador SGD, función de pérdida y métricas
    model.compile(
        optimizer=optimizers.SGD(learning_rate=0.001, momentum=0.9, decay=0.0),
        loss='categorical_crossentropy',  # Pérdida para clasificación multiclase
        metrics=['accuracy']  # Métrica de exactitud
    )

    # Entrenamiento del modelo
    model.fit(
        train_generator,  # Generador de datos para entrenamiento
        epochs=epochs,  # Número de épocas
        validation_data=val_generator  # Generador de datos para validación
    )

    return model


# Función para generar un nombre de archivo único
def get_available_filename(base_name, extension=".h5", directory="."):
    filename = f"{base_name}{extension}"
    return os.path.join(directory, filename)


# Bloque principal para ejecutar el script
if __name__ == "__main__":
    MAGNIFICIENT = None  # Si no hay magnificación, usar el conjunto de datos combinado
    SAVE_MODEL = False  # Cambia a True para guardar el modelo completo
    SAVE_WEIGHTS = True  # Cambia a True para guardar solo los pesos del modelo
    NUM_CLASSES = 2  # Número de clases (benigno/maligno)
    
    if MAGNIFICIENT is not None:
        # Ruta a los datos con magnificación específica
        DATASET_DIR = f"./images/binary_scenario/train/{MAGNIFICIENT}X"
        
        model = train_model(DATASET_DIR, num_classes=NUM_CLASSES)
        model_name = get_available_filename(f"modelo_cnn_{MAGNIFICIENT}x")
        
        # Guardar el modelo o los pesos según corresponda
        if SAVE_MODEL:
            model.save(model_name)  # Guardar el modelo completo
        if SAVE_WEIGHTS:
            model.save_weights(f"./saved_weights/modelo_cnn_{MAGNIFICIENT}x.weights.h5")  # Guardar solo los pesos
    else:
        print("IN PROCESS")  # Mensaje si no se especifica magnificación
        DATASET_DIR = f"./images/binary_scenario_merged/train"  # Ruta a los datos combinados

        model = train_model(DATASET_DIR, num_classes=NUM_CLASSES)
        model_name = get_available_filename(f"modelo_cnn_all")
        
        # Guardar el modelo o los pesos según corresponda
        if SAVE_MODEL:
            model.save(model_name)
        if SAVE_WEIGHTS:
            model.save_weights("./saved_weights/modelo_cnn_all.weights.h5")
