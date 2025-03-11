import tensorflow as tf
from tensorflow import keras
from keras import layers, preprocessing
import os
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# Directorio de las imágenes (ruta absoluta)
data_dir = "C://Users//lmher//Desktop//Informatica_4//PID//Trabajo//PID//images"
img_size = (32, 32)  # Tamaño de las imágenes que quieres redimensionar

num_imagenes = 100  # Número de imágenes que quieres usar (ajústalo según lo que necesites)
num_iteraciones_por_epoca = 4
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

# Limitar la cantidad de imágenes en el dataset de entrenamiento
dataset = dataset.take(num_imagenes)  # Usar X imágenes en el entrenamiento

print("-"*100)
print("Carga de dataset de validacion")
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

# Limitar la cantidad de imágenes en el dataset de validación
val_dataset = val_dataset.take(num_imagenes)  # Usar X imágenes en la validación
print(len(val_dataset))


print("-"*100)
print("Capas")
# Construcción del modelo CNN
model = keras.Sequential([
    layers.Rescaling(1./255),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    layers.Conv2D(96, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.Flatten(),
    layers.Dense(2000, activation='relu'),
    layers.Dense(num_classes, activation='relu'),
    layers.Softmax()
])
print("-"*100)
print("Modelar")

# Compilación del modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# print("-"*100)
# print("Entrenar")
# # Entrenamiento del modelo
# model.fit(dataset, epochs=2, validation_data=val_dataset)

# print("-"*100)
# print("Resultados")
# # Evaluación del modelo
# test_loss, test_acc = model.evaluate(val_dataset)
# print(f'Test accuracy: {test_acc:.4f}')