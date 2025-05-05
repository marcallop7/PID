# Clasificación de Células en Cáncer de Mama 🧬🩺

Este proyecto permite clasificar imágenes de células como **benignas** o **malignas** mediante redes neuronales convolucionales (**CNN**) y algoritmos de clasificación como **KNN**. Además, se incluye visualización de características extraídas y evaluación del rendimiento del modelo.

---

## 📁 Descarga de Datasets e Instalación Inicial

### 1. Descargar imágenes para entrenamiento

Descarga el conjunto de datos desde el siguiente enlace:

🔗 [Dataset de imágenes](https://drive.google.com/file/d/12xOLw9wdq6hc6ya0_eLlovHN9fxICGYp/view?usp=drive_link)

Una vez descargado y descomprimido:

- Crea la siguiente estructura de carpetas en la raíz del proyecto:

```
images/
└── binary_scenario/
    ├── train/
    └── test/
```

- Ejecuta el siguiente script para unificar las imágenes:

```bash
python merge_images_folder.py
```

Esto generará una carpeta `images/binary_scenario_merged/` con las imágenes mezcladas y organizadas por clase.

---

### 2. Descargar modelo preentrenado para extracción de características

Descarga el modelo KNN desde:

🔗 [Modelo `training_1.h5`](https://drive.google.com/file/d/12P4IJARHL_6GtfDRnZBTdKd45x2awm5_/view?usp=drive_link)

Ubícalo en:

```
models/
└── training_1.h5
```

Las características de las imágenes existentes pueden encontrar aquí:

🔗 [Características para KNN](https://drive.google.com/drive/folders/16bp6vsNZt_4Db3iaqz9Adn1j0Fxet9n7)

Este modelo fue tomado del repositorio original:  
📦 https://github.com/forderation/breast-cancer-retrieval

### 3. Descarga de diferentes pesos de la CNN

Los diferentes pesos según la amplitud de la imagen para el modelo CNN:

🔗 [Pesos entranimento CNN](https://drive.google.com/drive/folders/1jemlQFu66oN8CJAz7ka0WGnAwgU-GPaq)

---

## ⚙️ Requisitos Previos

Asegúrate de tener instalado:

- Python 3.8 o superior
- Las dependencias listadas en `requirements.txt`:

```bash
pip install -r requirements.txt
```

> Se recomienda usar un entorno virtual:

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno (Windows)
.\env\Scripts\activate

# Desactivar entorno
.\env\Scripts\deactivate
```

---

## 🚀 Ejecución del Proyecto

Para iniciar la aplicación principal, ejecuta:

```bash
python app.py
```

Asegúrate de haber instalado previamente todos los requisitos.

---

## 🧠 Entrenamiento del Modelo

Para entrenar una red neuronal convolucional desde cero, descarga las imágenes como se indica anteriormente y ejecuta:

```bash
python train_cnn.py
```

Este script entrenará un modelo utilizando las imágenes de la carpeta:

```
images/binary_scenario/train/
```

### Configuración del Entrenamiento del Modelo

El entrenamiento del modelo se ajusta dinámicamente según los valores de los atributos `MAGNIFICIENT`, `SAVE_MODEL` y `SAVE_WEIGHTS`.

#### Parámetros

##### `MAGNIFICIENT`
- Define el nivel de ampliación aplicado a las imágenes.
- Si se asigna un valor numérico (por ejemplo, `10`, `20`, etc.), el modelo se entrenará exclusivamente con imágenes correspondientes a esa ampliación.
- Si se establece como `None`, se utilizará un enfoque experimental basado en la **combinación de imágenes de distintas ampliaciones** dentro de un conjunto de entrenamiento unificado.

##### `SAVE_MODEL`
- Valor booleano (`True` o `False`).
- Si es `True`, se guarda el modelo completo tras el entrenamiento.

##### `SAVE_WEIGHTS`
- Valor booleano (`True` o `False`).
- Si es `True`, se guardan únicamente los pesos del modelo, sin incluir la arquitectura.

#### Flexibilidad
Esta configuración permite:
- Controlar qué tipo de datos se utilizan durante el entrenamiento.
- Elegir la forma en la que se almacena el resultado final del modelo, ya sea completo o solo sus pesos.


---

## 📊 Evaluación del Modelo

Las métricas de evaluación incluyen:

- **Accuracy (Precisión Global)**
- **F1 Score**
- **Recall (Sensibilidad)**
- **Precision (Precisión por Clase)**

### Escenarios de evaluación:
1. **Con datos originales**
2. **Con datos aumentados** (rotaciones, escalados, etc.)

---

### Visualización de Resultados

1. Ejecuta el análisis de características:

```bash
python analice.py
```

2. Luego genera los gráficos con:

```bash
python metrics.py
```

Esto mostrará visualmente cómo afecta el aumento de datos al rendimiento del modelo guardandolo en la ruta `outputs/metrics`.

---

## 🧬 Extracción y Visualización de Características

Se extraen características a partir de un encoder entrenado (modelo autoencoder).

### Métodos de visualización:

- **PCA (Análisis de Componentes Principales):** Reduce las dimensiones a 2 para una representación clara.
- **t-SNE:** Método no lineal que preserva la estructura de datos para mostrar agrupamientos y patrones.

Los resultados se almacenan en:

```
outputs/features/
├── pca_results.png
└── tsne_results.png
```

---

## 📝 Créditos

Este proyecto se basa en modelos y datasets públicos con fines académicos y de investigación. Parte del trabajo de extracción de características está basado en el repositorio:

🔗 https://github.com/forderation/breast-cancer-retrieval
