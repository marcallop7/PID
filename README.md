# Clasificación de Células en Cáncer de Mama 🧬🩺

Este proyecto permite clasificar imágenes de células como **benignas** o **malignas** mediante redes neuronales convolucionales (**CNN**) y algoritmos de clasificación como **KNN**. Además, se incluye visualización de características extraídas y evaluación del rendimiento del modelo.

---

## Video-tutoriales

- Instalación: https://youtu.be/GDUpCo7fy58?si=KvYZZkAGXpMQy0pE
- Demo: https://youtu.be/3sMBg-RV74E?si=9ToRXA6f4icOgDZv
- Resultados experimentación: https://youtu.be/L5rsEp35fn8?si=Tvaoy4SYjc6D4sXo

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

Además de encontrarse por defecto en la ruta `models/features`

Este modelo fue tomado del repositorio original:  
📦 https://github.com/forderation/breast-cancer-retrieval

### 3. Descarga de diferentes pesos de la CNN

Los diferentes pesos según la amplitud de la imagen para el modelo CNN:

🔗 [Pesos entranimento CNN](https://drive.google.com/drive/folders/1jemlQFu66oN8CJAz7ka0WGnAwgU-GPaq)

Estos pesos deben de almacenarse en la carpeta saved_weights de la forma

```
saved_weights/
├─── modelo_cnn_40x.weights.h5
├─── modelo_cnn_100x.weights.h5
├─── modelo_cnn_200x.weights.h5
├─── modelo_cnn_400x.weights.h5
└─── modelo_cnn_all.weights.h5
```

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
.\venv\Scripts\activate

# Desactivar entorno (Windows)
.\venv\Scripts\deactivate
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

Para entrenar una red neuronal convolucional desde cero, descarga las imágenes como se indica anteriormente, modifica los parámetros como se indicará en el siguiente punto y ejecuta:

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
- Controlar qué tipo de datos se utilizan durante el entrenamiento. Esto debido a que el procesamiento de cada uno de ellos lleva un tiempo prolongado.
- Elegir la forma en la que se almacena el resultado final del modelo, ya sea completo o solo sus pesos. El programa inicialmente guardaba el modelo completo, pero para mejor uso de memoria se decidió una migración a que únicamente se guardasen los pesos. Sin embargo, en caso de que algún usuario vea de utilidad la obtención del modelo completo guardado, podrá obtenerlo cambiando las variables.


---

## 📊 Evaluación del Modelo
### Métricas de Evaluación

Las métricas siguiente son mostradas tomando como caso positivo a que el resultado sea maligno.

- **Accuracy:**  
    Proporción de predicciones correctas sobre el total de predicciones realizadas. Mide qué tan bien el modelo clasifica en general.  
    **Fórmula:**  
    `Accuracy = (TP + TN) / (TP + TN + FP + FN)`

- **Recall:**  
    Proporción de verdaderos positivos detectados sobre el total de positivos reales. Indica la capacidad del modelo para identificar correctamente los casos positivos.  
    **Fórmula:**  
    `Recall = TP / (TP + FN)`

- **Precision:**  
    Proporción de verdaderos positivos sobre el total de predicciones positivas. Evalúa la exactitud de las predicciones positivas del modelo.  
    **Fórmula:**  
    `Precision = TP / (TP + FP)`

- **F1 Score:**  
    Media armónica entre la precisión y el recall. Es útil cuando hay un desequilibrio entre las clases.  
    **Fórmula:**  
    `F1 = 2 * (Precision * Recall) / (Precision + Recall)`

Donde:  
- **TP:** Verdaderos Positivos  
- **TN:** Verdaderos Negativos  
- **FP:** Falsos Positivos  
- **FN:** Falsos Negativos

### Escenarios de evaluación:
1. **Con datos originales**  
### Estructura de la Carpeta `images`

La carpeta `images` contiene las imágenes utilizadas para el entrenamiento y prueba del modelo. Su estructura es la siguiente:

```
images/
├── binary_scenario/
│   ├── train/
│   │   ├── 40X/
│   │   │   ├── benign/
│   │   │   └── malignant/
│   │   ├── 100X/
│   │   │   ├── benign/
│   │   │   └── malignant/
│   │   ├── 200X/
│   │   │   ├── benign/
│   │   │   └── malignant/
│   │   └── 400X/
│   │       ├── benign/
│   │       └── malignant/
│   └── test/
│       ├── 40X/
│       │   ├── benign/
│       │   └── malignant/
│       ├── 100X/
│       │   ├── benign/
│       │   └── malignant/
│       ├── 200X/
│       │   ├── benign/
│       │   └── malignant/
│       └── 400X/
│           ├── benign/
│           └── malignant/
└── binary_scenario_merged/
    ├── train/
    │   ├── benign/
    │   └── malignant/
    └── test/
        ├── benign/
        └── malignant/
```

#### Descripción:
- **`binary_scenario/`:**  
    Contiene las imágenes originales organizadas en carpetas separadas para entrenamiento (`train/`) y prueba (`test/`), cada una dividida por niveles de ampliación (`40X/`, `100X/`, `200X/`, `400X/`), siendo estas a su vez dividias en malignas o benignas (`/malignant`, `/benign`).

- **`binary_scenario_merged/`:**  
    Carpeta generada tras ejecutar el script `merge_images_folder.py`. Contiene las imágenes de entrenamiento, y prueba unificadas y organizadas por malignas o benignas (`/malignant`, `/benign`).

Esta estructura facilita la manipulación y el acceso a las imágenes durante las etapas de entrenamiento, validación, prueba y análisis.

El conjunto de datos contiene casi 8,000 imágenes, distribuidas entre las diferentes carpetas y clases, lo que proporciona una base sólida para entrenar y evaluar los modelos.

2. **Con datos aumentados** (rotaciones, escalados, etc.)  
Para aumentar los datos se ha aplicado la técnica de **data augmentation**, es una técnica utilizada para incrementar la cantidad y diversidad de datos de entrenamiento mediante transformaciones como rotaciones, escalados, volteos, y cambios de brillo o contraste en las imágenes originales. Esto ayuda a mejorar la generalización del modelo y su capacidad para manejar variaciones en los datos reales.

### Resultados de Evaluación

#### Modelos CNN

| Modelo                 | Métrica 1 | Métrica 2 | Métrica 3 | Métrica 4 |
| ---------------------- | --------- | --------- | --------- | --------- |
| modelo\_cnn\_100x      | 0.7638    | 0.7670    | 0.7722    | 0.7638    |
| modelo\_cnn\_200x      | 0.8141    | 0.8074    | 0.8090    | 0.8141    |
| modelo\_cnn\_400x      | 0.8141    | 0.8001    | 0.8150    | 0.8141    |
| modelo\_cnn\_40x       | 0.8141    | 0.8001    | 0.8150    | 0.8141    |
| modelo\_cnn\_all       | 0.8241    | 0.8093    | 0.8304    | 0.8241    |
| modelo\_cnn\_aug\_100x | 0.8280    | 0.8285    | 0.8290    | 0.8280    |
| modelo\_cnn\_aug\_200x | 0.8239    | 0.8236    | 0.8234    | 0.8239    |
| modelo\_cnn\_aug\_400x | 0.8044    | 0.7817    | 0.8219    | 0.8044    |
| modelo\_cnn\_aug\_40x  | 0.7809    | 0.7626    | 0.7758    | 0.7809    |
| modelo\_cnn\_aug\_all  | 0.7911    | 0.7747    | 0.7877    | 0.7911    |

#### Modelos KNN

| Modelo                 | Métrica 1 | Métrica 2 | Métrica 3 | Métrica 4 |
| ---------------------- | --------- | --------- | --------- | --------- |
| modelo\_knn\_100x      | 0.8986    | 0.9293    | 0.8961    | 0.9650    |
| modelo\_knn\_200x      | 0.8806    | 0.9124    | 0.9259    | 0.8993    |
| modelo\_knn\_400x      | 0.8950    | 0.9237    | 0.9127    | 0.9350    |
| modelo\_knn\_40x       | 0.9045    | 0.9304    | 0.9338    | 0.9270    |
| modelo\_knn\_all       | 0.9010    | 0.9293    | 0.9128    | 0.9465    |
| modelo\_knn\_aug\_100x | 0.4966    | 0.5126    | 0.7740    | 0.3832    |
| modelo\_knn\_aug\_200x | 0.7194    | 0.8291    | 0.7162    | 0.9842    |
| modelo\_knn\_aug\_400x | 0.4177    | 0.2869    | 0.8548    | 0.1724    |
| modelo\_knn\_aug\_40x  | 0.6814    | 0.8061    | 0.6937    | 0.9620    |
| modelo\_knn\_aug\_all  | 0.4211    | 0.3144    | 0.8476    | 0.1930    |

---

### Visualización de Resultados

1. Ejecuta el análisis de características para recorrer todos los test:

**Nota:** Si prefieres ver la matríz de confusión resultante de la ejecución de alguno de los test. Puedes dirigirte a sus archivos y cambiar en el fragmento de ejecución la variable `show_metrics` a `True`. Por defecto en el proyecto se encuentra en `False`.

```bash
python analice.py
```

2. Luego genera los gráficos con:

```bash
python metrics.py
```

Esto guardará graficas de cómo afecta el aumento de datos al rendimiento del modelo guardandolo en la ruta `outputs/metrics`.

---

## 🧬 Extracción y Visualización de Características

Se extraen características a partir de un encoder entrenado (modelo autoencoder).

### Métodos de visualización:

- **PCA (Análisis de Componentes Principales):** Reduce las dimensiones a 2 para una representación clara.
- **t-SNE:** Método no lineal que preserva la estructura de datos para mostrar agrupamientos y patrones.

Los resultados se almacenan en:

```
outputs/features/
├── pca/
└── tsne/
```

---

## 📝 Créditos

Este proyecto se basa en modelos y datasets públicos con fines académicos y de investigación. Parte del trabajo de extracción de características está basado en el repositorio:

🔗 https://github.com/forderation/breast-cancer-retrieval
