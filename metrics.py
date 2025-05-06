import pandas as pd              # Para manejar archivos CSV y manipular datos en DataFrames
import re                        # Para trabajar con expresiones regulares (extraer categorías)
import matplotlib.pyplot as plt  # Para generar gráficos de comparación
import os                        # Para manejar rutas de archivos y directorios

# Crear las carpetas de salida si no existen
os.makedirs('./outputs/metrics/comparation_metrics', exist_ok=True)
os.makedirs('./outputs/metrics/comparation_magnification', exist_ok=True)

# Ruta al archivo CSV con las métricas de los modelos
file_path = 'metrics.csv'  # Cambia el nombre si es necesario

# Lista de métricas a comparar
metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
# Lista de diferentes ampliaciones del conjunto de datos (40x, 100x, etc.)
augmentations = ["40x", "100x", "200x", "400x", "all"]

# Función para extraer las métricas de comparación por modelo
def extract_comparation_metrics():
    # Extraer categoría (40x, 100x, etc.) desde el nombre del archivo
    def extraer_categoria(nombre):
        # Se usa una expresión regular para capturar la categoría como "40x", "100x", etc.
        match = re.search(r'(\d+x|all)', nombre)
        return match.group(1) if match else "otro"  # Retorna la categoría o "otro" si no se encuentra
    
    # Cargar el archivo CSV con las métricas
    df = pd.read_csv(file_path)
    
    # Agregar una nueva columna 'grupo' con las categorías extraídas del nombre del archivo
    df["grupo"] = df["archivo"].apply(extraer_categoria)

    # Función para identificar el modelo a partir del nombre del archivo
    def identificar_modelo(nombre):
        if "cnn_aug" in nombre:
            return "CNN_AUG"  # Red neuronal convolucional con aumento
        elif "cnn" in nombre:
            return "CNN"      # Red neuronal convolucional sin aumento
        elif "knn_aug" in nombre:
            return "KNN_AUG"  # K-vecinos más cercanos con aumento
        elif "knn" in nombre:
            return "KNN"      # K-vecinos más cercanos sin aumento
        else:
            return "Otro"    # Para otros casos

    # Agregar una nueva columna 'modelo' con el tipo de modelo identificado
    df["modelo"] = df["archivo"].apply(identificar_modelo)

    # Asegurar que la columna 'grupo' esté ordenada de manera lógica según la lista de augmentations
    df["grupo"] = pd.Categorical(df["grupo"], categories=augmentations, ordered=True)
    df = df.sort_values("grupo")

    # Definir colores y marcadores por tipo de modelo
    colores = {
        "CNN": "#1f77b4",        # Azul intenso para CNN
        "CNN_AUG": "#ff7f0e",    # Naranja para CNN con aumento
        "KNN": "#2ca02c",        # Verde para KNN
        "KNN_AUG": "#d62728"     # Rojo para KNN con aumento
    }
    marcadores = {
        "CNN": "o",       # Círculo para CNN
        "CNN_AUG": "^",   # Triángulo hacia arriba para CNN con aumento
        "KNN": "s",       # Cuadrado para KNN
        "KNN_AUG": "D"    # Diamante para KNN con aumento
    }

    # Crear una gráfica para cada métrica
    for metric in metrics:
        plt.figure(figsize=(10, 6))  # Definir el tamaño de la figura

        # Recorre cada tipo de modelo para crear una gráfica por modelo
        for modelo in df["modelo"].unique():
            subset = df[df["modelo"] == modelo]  # Filtrar por el modelo actual
            plt.scatter(
                subset["grupo"],             # Eje x: grupo de tamaño de dataset
                subset[metric],              # Eje y: valor de la métrica
                label=modelo,               # Etiqueta del modelo
                color=colores[modelo],      # Color según el modelo
                marker=marcadores[modelo],  # Marcador según el modelo
                s=100                        # Tamaño de los puntos
            )

        # Añadir título, etiquetas y ajustar los límites
        plt.title(f'Comparación de {metric} por tamaño de conjunto')
        plt.ylabel(metric)
        plt.xlabel('Grupo (Tamaño de dataset)')
        plt.ylim(0, 1)  # Limitar el eje y entre 0 y 1
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Guardar la gráfica como un archivo PNG
        plt.savefig(f'./outputs/metrics/comparation_metrics/{metric.lower().replace(" ", "_")}_comparativa_puntos.png')
        plt.close()  # Cerrar la figura para evitar sobrecargar la memoria

    print("Gráficas de puntos generadas por grupo.")

# Función para generar gráficas comparativas por aumento (magnificación)
def extract_comparation_magnification():
    # Cargar los datos del archivo CSV
    data = pd.read_csv(file_path)

    # Crear una gráfica para cada tipo de aumento
    for aug in augmentations:
        plt.figure(figsize=(10, 6))  # Definir el tamaño de la figura
        # Filtrar los datos por el tipo de aumento (40x, 100x, etc.)
        filtered_data = data[data['archivo'].str.contains(aug, case=False, na=False)]
        
        bar_width = 0.2  # Ancho de las barras
        x = range(len(filtered_data['archivo']))  # Posiciones de las barras en el eje x
        
        # Para cada métrica, dibujar una barra
        for i, metric in enumerate(metrics):
            bars = plt.bar(
                [pos + i * bar_width for pos in x],  # Posicionar las barras
                filtered_data[metric],               # Altura de las barras: valor de la métrica
                bar_width,                           # Ancho de la barra
                label=metric                         # Etiqueta de la métrica
            )
            
            # Agregar los valores exactos sobre las barras
            for bar in bars:
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f'{bar.get_height():.2f}', ha='center', va='bottom')

        # Añadir título, etiquetas y ajustar los límites
        plt.title(f'Comparación de métricas para modelos con {aug}')
        plt.xlabel('Modelos')
        plt.ylabel('Valor de la métrica')
        plt.ylim(0, 1)  # Limitar el eje y entre 0 y 1
        # Configurar las posiciones de las etiquetas en el eje x
        plt.xticks([pos + (len(metrics) - 1) * bar_width / 2 for pos in x], filtered_data['archivo'], rotation=45)
        plt.legend()
        plt.tight_layout()

        # Guardar la gráfica como un archivo PNG
        plt.savefig(f'./outputs/metrics/comparation_magnification/comparison_{aug}.png')

        # Imprimir los valores de las métricas para los modelos filtrados
        print(f"Valores de métricas para modelos con {aug}:")
        print(filtered_data[['archivo'] + metrics])  # Mostrar las métricas de los modelos filtrados

    print("Gráficas generadas y guardadas como archivos PNG.")

# Llamar a las funciones para generar las gráficas
extract_comparation_metrics()
extract_comparation_magnification()
