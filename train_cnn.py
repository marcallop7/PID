from tensorflow.keras import layers, models, optimizers, regularizers # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import os


def build_cnn_model(input_shape=(128, 128, 3), num_classes=2):
    model = models.Sequential()

    # 1ª capa convolucional
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                            input_shape=input_shape, kernel_regularizer=regularizers.l2(1e-3)))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2))

    # 2ª capa
    model.add(layers.Conv2D(96, (3, 3), activation='relu', padding='same',
                            kernel_regularizer=regularizers.l2(1e-3)))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2))

    # 3ª capa
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                            kernel_regularizer=regularizers.l2(1e-3)))

    # 4ª capa
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                            kernel_regularizer=regularizers.l2(1e-3)))

    # 5ª capa
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                            kernel_regularizer=regularizers.l2(1e-3)))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2))

    # Flatten y capas densas
    model.add(layers.Flatten())
    model.add(layers.Dense(2000, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model


def train_model(data_dir, num_classes=2, input_shape=(128, 128, 3), batch_size=32, epochs=20):
    # Aumentos de datos
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=270,
        horizontal_flip=True,
        validation_split=0.2
    )

    # Generadores
    target_size = input_shape[:2]
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    # Modelo
    model = build_cnn_model(input_shape=input_shape, num_classes=num_classes)

    # Compilación
    model.compile(
        optimizer=optimizers.SGD(learning_rate=0.001, momentum=0.9, decay=0.0),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Entrenamiento
    model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator
    )

    return model


def get_available_filename(base_name, extension=".h5", directory="."):
    filename = f"{base_name}{extension}"
    return os.path.join(directory, filename)



if __name__ == "__main__":
    MAGNIFICIENT = None
    SAVE_MODEL = False
    SAVE_WEIGHTS = True
    if MAGNIFICIENT is not None:
        DATASET_DIR = f"./images/binary_scenario/train/{MAGNIFICIENT}X"
        NUM_CLASSES = 2
        model = train_model(DATASET_DIR, num_classes=NUM_CLASSES)
        model_name = get_available_filename(f"modelo_cnn_{MAGNIFICIENT}x")
        if SAVE_MODEL:
            model.save(model_name)
        if SAVE_WEIGHTS:
            model.save_weights(f"./saved_weights/modelo_cnn_{MAGNIFICIENT}x.weights.h5")
    else:
        print("IN PROCESS")
        DATASET_DIR = f"./images/binary_scenario_merged/train"
        NUM_CLASSES = 2
        model = train_model(DATASET_DIR, num_classes=NUM_CLASSES)
        model_name = get_available_filename(f"modelo_cnn_all")
        if SAVE_MODEL:
            model.save(model_name)
        if SAVE_WEIGHTS:
            model.save_weights("./saved_weights/modelo_cnn_all.weights.h5")