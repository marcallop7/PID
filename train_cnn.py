import tensorflow as tf
from tensorflow import keras
from keras import layers, preprocessing
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# Directorio de las imágenes (ruta absoluta)
data_dir = "images/"
img_size = (32, 32)  # Tamaño de las imágenes que quieres redimensionar

num_imagenes = 1000  # Número total de imágenes a usar (1000 en total, 500 por clase)
num_iteraciones_por_epoca = 4
dense_unit = 10
num_classes = 2

batch_size = num_imagenes // num_iteraciones_por_epoca  # Tamaño de lote para la carga de datos

print("-"*100)
print("Carga de dataset")

# Cargar los datos desde las carpetas
dataset = preprocessing.image_dataset_from_directory(
    data_dir,
    labels="inferred",          
    label_mode="categorical",   
    image_size=img_size,        
    batch_size=batch_size,
    validation_split=0.2,   
    subset="training",   
    seed=1
)

# Limitar la cantidad de imágenes en el dataset de entrenamiento (500 por clase)
dataset = dataset.take(num_imagenes // 2)

print("-"*100)
print("Carga de dataset de validación")

# Cargar los datos de validación
val_dataset = preprocessing.image_dataset_from_directory(
    data_dir,
    labels="inferred",
    label_mode="categorical",
    image_size=img_size,
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=42
)

# Limitar la cantidad de imágenes en el dataset de validación (500 por clase)
val_dataset = val_dataset.take(num_imagenes // 2)  # Usar X imágenes en la validación
print(f'Número de batches de validación: {len(val_dataset)}')

print("-"*100)
print("Capas")
# Construcción del modelo CNN con Data Augmentation y MaxPooling2D
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomTranslation(0.2, 0.2),  # Añadir traslación aleatoria
])

model = keras.Sequential([
    layers.Rescaling(1./255),
    data_augmentation,  # Añadir data augmentation
    layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(96, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.Flatten(),
    layers.Dense(dense_unit, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  # Usar softmax para clasificación multiclase
])

print("-"*100)
print("Modelar")

# Compilación del modelo
model.compile(optimizer="adam",
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("-"*100)
print("Entrenar")
# Entrenamiento del modelo
history = model.fit(dataset, epochs=20, validation_data=val_dataset)

# Graficar precisión
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Graficar pérdida
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

print("-"*100)
print("Resultados")
# Evaluación del modelo
test_loss, test_acc = model.evaluate(val_dataset)
print(f'Test loss: {test_loss:.4f}')
print(f'Test accuracy: {test_acc:.4f}')

# Guardar el modelo
model.save("model.keras")
