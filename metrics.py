import pandas as pd
import re
import matplotlib.pyplot as plt
import os

# Crear carpetas si no existen
os.makedirs('./outputs/metrics/comparation_metrics', exist_ok=True)
os.makedirs('./outputs/metrics/comparation_magnification', exist_ok=True)

# Cargar el archivo CSV
file_path = 'metrics.csv'  # Cambia el nombre si es necesario

# Crear gráficas comparativas
metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
augmentations = ["40x", "100x", "200x", "400x", "all"]

def extract_comparation_metrics():
    # Extraer categoría (40x, 100x, etc.) desde el nombre del archivo
    def extraer_categoria(nombre):
        match = re.search(r'(\d+x|all)', nombre)
        return match.group(1) if match else "otro"

    df = pd.read_csv(file_path)
    df["grupo"] = df["archivo"].apply(extraer_categoria)

    # Clasificar modelo más detalladamente
    def identificar_modelo(nombre):
        if "cnn_aug" in nombre:
            return "CNN_AUG"
        elif "cnn" in nombre:
            return "CNN"
        elif "knn_aug" in nombre:
            return "KNN_AUG"
        elif "knn" in nombre:
            return "KNN"
        else:
            return "Otro"
        
    # Identificar tipo de modelo
    df["modelo"] = df["archivo"].apply(identificar_modelo)

    # Asegurar orden lógico
    df["grupo"] = pd.Categorical(df["grupo"], categories=augmentations, ordered=True)
    df = df.sort_values("grupo")

    # Colores y marcadores por modelo
    colores = {
        "CNN": "#1f77b4",        # Azul intenso
        "CNN_AUG": "#ff7f0e",    # Naranja
        "KNN": "#2ca02c",        # Verde
        "KNN_AUG": "#d62728"     # Rojo
    }
    marcadores = {
        "CNN": "o",       # Círculo
        "CNN_AUG": "^",   # Triángulo hacia arriba
        "KNN": "s",       # Cuadrado
        "KNN_AUG": "D"    # Diamante
    }

    # Crear una gráfica para cada métrica
    for metric in metrics:
        plt.figure(figsize=(10, 6))

        for modelo in df["modelo"].unique():
            subset = df[df["modelo"] == modelo]
            plt.scatter(
                subset["grupo"], 
                subset[metric], 
                label=modelo, 
                color=colores[modelo], 
                marker=marcadores[modelo], 
                s=100  # tamaño de los puntos
            )

        plt.title(f'Comparación de {metric} por tamaño de conjunto')
        plt.ylabel(metric)
        plt.xlabel('Grupo (Tamaño de dataset)')
        plt.ylim(0, 1)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'./outputs/metrics/comparation_metrics/{metric.lower().replace(" ", "_")}_comparativa_puntos.png')
        plt.close()

    print("Gráficas de puntos generadas por grupo.")

def extract_comparation_magnification():    
    # Filtrar los modelos por tipo de augmentation
    data = pd.read_csv(file_path)

    for aug in augmentations:
        plt.figure(figsize=(10, 6))
        filtered_data = data[data['archivo'].str.contains(aug, case=False, na=False)]
        
        bar_width = 0.2
        x = range(len(filtered_data['archivo']))
        
        for i, metric in enumerate(metrics):
            bars = plt.bar(
                [pos + i * bar_width for pos in x],
                filtered_data[metric],
                bar_width,
                label=metric
            )
            
            # Agregar los valores exactos sobre las barras
            for bar in bars:
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{bar.get_height():.2f}', ha='center', va='bottom')
        
        plt.title(f'Comparación de métricas para modelos con {aug}')
        plt.xlabel('Modelos')
        plt.ylabel('Valor de la métrica')
        plt.ylim(0, 1)
        plt.xticks([pos + (len(metrics) - 1) * bar_width / 2 for pos in x], filtered_data['archivo'], rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'./outputs/metrics/comparation_magnification/comparison_{aug}.png')  # Guardar la gráfica como archivo
        # plt.show()
        # Imprimir los valores de las métricas para los modelos filtrados
        print(f"Valores de métricas para modelos con {aug}:")
        print(filtered_data[['archivo'] + metrics])

    print("Gráficas generadas y guardadas como archivos PNG.")

extract_comparation_metrics()
extract_comparation_magnification()