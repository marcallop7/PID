import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Directorios de las imágenes
base_dir = 'images'
test_dir = os.path.join(base_dir, 'validation')

# Crear directorios de prueba
test_beign_dir = os.path.join(test_dir, 'breast_beign')
test_malignant_dir = os.path.join(test_dir, 'breast_malignant')

# Parámetros
img_width, img_height = 150, 150
batch_size = 32

# Generador de datos para prueba
test_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Cargar el modelo guardado
model = load_model('breast_cancer_cnn.h5')

# Evaluar el modelo
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print(f'Test accuracy: {test_acc:.4f}')
print(f'Test loss: {test_loss:.4f}')